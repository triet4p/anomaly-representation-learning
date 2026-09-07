"""Restored-reference-by-default V2 inference (Sprint 11 Task 16).

The pipeline loads a coherent V2 checkpoint and scores files through
the frozen hierarchical references exactly as saved. Replacing those
references requires the explicit :meth:`refit_references` opt-in —
inference never silently refits from scored files. Every stage output
(patch/file/trajectory/confidence/risk) is validated before return,
and missing checkpoints or reference payloads fail fast.

Energy identity (Deep-Review Finding 1, Batch B2): the boundary-trained
encoder ``context_energy`` is preserved unchanged as the acute monitoring
score, and the hierarchical ``population_energy`` is returned separately
under its own name. The file/trajectory/confidence/risk chain consumes
the context signal; the population view is aggregated separately as
``file_population`` with its own explicit ``energy_source`` label, so the
two scores can never be silently conflated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from representation.v2_aggregation import aggregate_file_state
from representation.v2_checkpoint import load_v2_checkpoint
from representation.v2_config import V2Config
from representation.v2_contracts import (
    CONTEXT_ENERGY_FIELD,
    POPULATION_ENERGY_FIELD,
    validate_confidence,
    validate_context_patch_output,
    validate_file_state,
    validate_population_patch_output,
    validate_risk,
    validate_trajectory,
)
from representation.v2_geometry import FrozenReference, HierarchicalMahalanobisGeometry
from representation.v2_patch import ContextConditionedPatchEncoder
from representation.v2_risk import (
    CensoredSurvivalRisk,
    HealthyTailCalibrator,
    expected_feature_width,
    trajectory_feature_matrix,
)
from representation.v2_trajectory import TrajectoryTracker
from synth.schema import FileSample, RegimeType

#: Canonical regime-id order: index of each RegimeType in declaration order.
REGIME_ORDER: tuple[str, ...] = tuple(r.value for r in RegimeType)


def patch_regime_ids(
    samples: Sequence[FileSample],
    starts: torch.Tensor | np.ndarray,
    n_patches: int,
) -> torch.Tensor:
    """Map each patch to its operating-regime id from the patch start time."""
    order = {value: idx for idx, value in enumerate(REGIME_ORDER)}
    rows = torch.as_tensor(np.asarray(starts)).to(dtype=torch.long)
    if rows.ndim != 2 or rows.shape[1] != n_patches or rows.shape[0] != len(samples):
        raise ValueError("starts must have shape [B, N]")
    out = torch.zeros_like(rows)
    for i, sample in enumerate(samples):
        bounds = [(r.start, r.end, order[r.regime.value]) for r in sample.regime_sequence]
        if not bounds:
            raise ValueError(f"sample {sample.file_id} has an empty regime_sequence")
        for j in range(n_patches):
            start = int(rows[i, j].item())
            if start < 0:
                continue
            regime = bounds[-1][2]
            for s, e, rid in bounds:
                if s <= start < e:
                    regime = rid
                    break
            out[i, j] = regime
    return out


class V2InferencePipeline:
    """Score full files with restored V2 references (eval mode, no grad)."""

    def __init__(
        self,
        config: V2Config,
        model: ContextConditionedPatchEncoder,
        geometry: FrozenReference,
        *,
        calibrator: HealthyTailCalibrator | None = None,
        risk: CensoredSurvivalRisk | None = None,
        elevated_threshold: float | None = None,
        elevated_calibration: Mapping[str, object] | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.model = model.eval()
        self.geometry = geometry
        self.calibrator = calibrator
        self.risk = risk
        if elevated_calibration is not None:
            record = dict(elevated_calibration)
            value = record.get("value")
            if not isinstance(value, (int, float)):
                raise ValueError("elevated_calibration must carry a numeric 'value'")
            self.elevated_threshold = float(value)
            cohort = record.get("fit_cohort")
            self.elevated_threshold_source = (
                f"calibration-record ({cohort})" if isinstance(cohort, str) else "calibration-record"
            )
            self.elevated_calibration: dict[str, object] | None = record
        elif elevated_threshold is not None:
            self.elevated_threshold = float(elevated_threshold)
            self.elevated_threshold_source = "explicit-override"
            self.elevated_calibration = None
        else:
            self.elevated_threshold = float(config.elevated_threshold)
            self.elevated_threshold_source = "config-default (uncalibrated)"
            self.elevated_calibration = None
        self.device = torch.device(device)
        self.model.to(self.device)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        elevated_threshold: float | None = None,
    ) -> "V2InferencePipeline":
        """Restore the pipeline exactly as saved; fail fast on any gap.

        The dev-val-calibrated operating threshold persisted in the
        checkpoint is restored by default so production aggregation uses
        the calibrated cutoff, never the fixed config default. Pass an
        explicit ``elevated_threshold`` only to override it deliberately
        (recorded as an explicit override, not calibration).
        """
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"V2 checkpoint not found: {target}")
        probe: dict[str, object] = torch.load(str(target), map_location="cpu", weights_only=False)
        stored = probe.get("config")
        if not isinstance(stored, dict):
            raise ValueError("V2 checkpoint is missing its configuration")
        config = V2Config(**stored)
        model = ContextConditionedPatchEncoder(
            n_channels=config.n_channels,
            d_model=config.d_model,
            n_robots=config.n_robots,
            n_programs=config.n_programs,
            n_regimes=config.n_regimes,
            n_prototypes=config.n_prototypes,
            sequence_layers=config.sequence_layers,
            attention_heads=config.attention_heads,
            dropout=0.0,
        )
        restored = load_v2_checkpoint(path, model, expected_config=config)
        geometry_snap = restored.get("geometry")
        if geometry_snap is None or not isinstance(geometry_snap, dict):
            raise ValueError(
                "V2 checkpoint carries no fitted geometry references; "
                "inference refuses to run on unfitted references"
            )
        live = HierarchicalMahalanobisGeometry(
            config.d_model,
            shrinkage=config.shrinkage,
            covariance_eps=config.covariance_eps,
            min_group_samples=config.min_group_samples,
            diag_min_samples=config.diag_min_samples,
        )
        live.restore_snapshot(geometry_snap)
        geometry = live.frozen()
        calibrator = None
        if isinstance(restored.get("calibrator"), dict):
            calibrator = HealthyTailCalibrator(
                min_samples=config.calibration_min_samples
            )
            calibrator.load_state_dict(restored["calibrator"])  # type: ignore[arg-type]
        risk = None
        if isinstance(restored.get("risk"), dict):
            risk = CensoredSurvivalRisk(
                expected_feature_width(), horizons_days=tuple(config.risk_horizons_days)  # type: ignore[arg-type]
            )
            risk.load_state_dict(restored["risk"])  # type: ignore[arg-type]
        operating = restored.get("operating_threshold")
        calibration_record = operating if isinstance(operating, dict) else None
        if elevated_threshold is not None:
            calibration_record = None
        return cls(
            config, model, geometry, calibrator=calibrator, risk=risk,
            elevated_threshold=elevated_threshold,
            elevated_calibration=calibration_record,
            device=device,
        )

    def refit_references(
        self,
        latents: torch.Tensor,
        robot_ids: torch.Tensor,
        program_ids: torch.Tensor,
        regime_ids: torch.Tensor,
        healthy_mask: torch.Tensor,
    ) -> FrozenReference:
        """Explicit opt-in replacement of the restored references."""
        live = HierarchicalMahalanobisGeometry(
            self.config.d_model,
            shrinkage=self.config.shrinkage,
            covariance_eps=self.config.covariance_eps,
            min_group_samples=self.config.min_group_samples,
            diag_min_samples=self.config.diag_min_samples,
        )
        live.fit(latents, robot_ids, program_ids, regime_ids, healthy_mask)
        self.geometry = live.frozen()
        return self.geometry

    @torch.no_grad()
    def score_patches(
        self,
        patches: torch.Tensor,
        patch_pad_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
        *,
        tracker: TrajectoryTracker | None = None,
        suspect_flags: torch.Tensor | None = None,
        maintenance_resets: torch.Tensor | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return validated patch/file/trajectory/confidence/risk outputs.

        ``patch`` preserves the encoder ``context_energy`` bit-for-bit: the
        exact boundary-trained monitoring score. ``population`` carries the
        separately fitted hierarchical ``population_energy`` under its own
        name. ``file`` aggregates the context signal (and feeds the
        trajectory/confidence/risk chain); ``file_population`` aggregates
        the population signal as an explicitly labeled independent view.
        Both file states record their ``energy_source`` and share the
        restored operating elevated threshold.
        """
        b = patch_valid_mask.shape[0]
        encoded = self.model(
            patches.to(self.device),
            patch_pad_mask.to(self.device),
            patch_valid_mask.to(self.device),
            robot_idx.to(self.device),
            program_idx.to(self.device),
            regime_ids.to(self.device),
        )
        latents = encoded["patch_latents"].cpu()
        context_energy = encoded["context_energy"].cpu()
        valid = patch_valid_mask.cpu()
        robot = robot_idx.cpu()
        program = program_idx.cpu()
        regimes = regime_ids.cpu()
        scored = self.geometry.mixture_energy(latents, valid, robot, program, regimes)
        patch: dict[str, torch.Tensor] = {
            "patch_latents": latents,
            "context_energy": context_energy,
            "patch_valid_mask": valid,
        }
        validate_context_patch_output(patch)
        population: dict[str, torch.Tensor] = {
            "population_energy": scored["population_energy"],
            "patch_valid_mask": valid,
        }
        validate_population_patch_output(population)
        file_state = aggregate_file_state(
            latents,
            patch["context_energy"],
            valid,
            regimes,
            energy_source=CONTEXT_ENERGY_FIELD,
            top_q_fraction=self.config.top_q_fraction,
            elevated_threshold=self.elevated_threshold,
            n_regimes=self.config.n_regimes,
        )
        validate_file_state(file_state)
        file_population = aggregate_file_state(
            latents,
            population["population_energy"],
            valid,
            regimes,
            energy_source=POPULATION_ENERGY_FIELD,
            top_q_fraction=self.config.top_q_fraction,
            elevated_threshold=self.elevated_threshold,
            n_regimes=self.config.n_regimes,
        )
        validate_file_state(file_population)

        trajectory: dict[str, torch.Tensor] | None = None
        if tracker is not None:
            if suspect_flags is not None and tuple(suspect_flags.shape) != (b,):
                raise ValueError("suspect_flags must have shape [B]")
            if maintenance_resets is not None and tuple(maintenance_resets.shape) != (b,):
                raise ValueError("maintenance_resets must have shape [B]")
            rows: dict[str, list[torch.Tensor]] = {}
            for i in range(b):
                feats = tracker.update(
                    file_state["file_state"][i],
                    is_suspect=bool(suspect_flags[i].item()) if suspect_flags is not None else False,
                    maintenance_reset=bool(maintenance_resets[i].item())
                    if maintenance_resets is not None
                    else False,
                )
                for k, v in feats.items():
                    rows.setdefault(k, []).append(v.reshape(-1))
            trajectory = {k: torch.cat(v) for k, v in rows.items()}
            validate_trajectory(trajectory)

        confidence: dict[str, torch.Tensor] | None = None
        if self.calibrator is not None and trajectory is not None:
            confidence = self.calibrator.confidence(trajectory["displacement"])
            validate_confidence(confidence)

        failure_risk: dict[str, torch.Tensor] | None = None
        if self.risk is not None and trajectory is not None:
            features = trajectory_feature_matrix(
                trajectory,
                file_state["tail_energy"],
                file_state["elevated_fraction"],
            )
            expected = expected_feature_width()
            if features.shape[1] != expected:
                raise ValueError("risk feature width mismatch")
            failure_risk = self.risk.predict_proba(features)
            validate_risk(failure_risk)

        return {
            "patch": patch,
            "population": population,
            "file": file_state,
            "file_population": file_population,
            "trajectory": trajectory or {},
            "confidence": confidence or {},
            "risk": failure_risk or {},
        }
