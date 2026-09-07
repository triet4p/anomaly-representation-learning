"""V2 trainer with stationary-objective selection (Sprint 11 Task 16).

Integrates the context-conditioned patch encoder with the localized
counterfactual objectives and ramped boundary schedule. Encoder inputs
are built from an explicit allowlist
(:data:`V2_ENCODER_CONTEXT_FIELDS`); corruption masks and severities
shape the loss only and never enter the encoder.

Model selection minimizes the full declared signed-likelihood objective
(clean conditional-density NLL plus configured variance/covariance,
background, and final-coefficient boundary terms) on validation batches —
never squared distance to zero and never test outcomes. Severity enters
through the configured corrupted view; cross-severity ordering stays a
diagnostic, not a selection term.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from representation.v2_checkpoint import load_v2_checkpoint, save_v2_checkpoint
from representation.v2_config import V2Config, assert_encoder_inputs_clean
from representation.v2_geometry import FrozenReference, HierarchicalMahalanobisGeometry
from representation.v2_objectives import CounterfactualCriterion, synthesize_corrupted_patches
from representation.v2_patch import ContextConditionedPatchEncoder

#: Batch keys the V2 trainer consumes. ``corruption_mask``/``severity``
#: are loss-only; every other key is encoder-visible context or signal.
V2_TRAIN_BATCH_KEYS: frozenset[str] = frozenset(
    {
        "patches",
        "patch_pad_mask",
        "patch_valid_mask",
        "robot_idx",
        "program_idx",
        "regime_ids",
        "corruption_mask",
        "severity",
    }
)


class V2Trainer:
    """Train the V2 patch encoder with coherent best-state selection."""

    def __init__(
        self,
        config: V2Config,
        model: ContextConditionedPatchEncoder,
        criterion: CounterfactualCriterion,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str = "cpu",
        seed: int = 0,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm
        self.step = 0
        self.history: list[dict[str, float]] = []
        self.best_stationary: float | None = None
        self.best_state: dict[str, object] | None = None
        self.seed = int(seed)
        self.model.to(self.device)
        gen = torch.Generator().manual_seed(int(seed))
        self._generator = gen

    def _encoder_inputs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [k for k in V2_TRAIN_BATCH_KEYS if k not in batch]
        if missing:
            raise ValueError(f"V2 batch is missing keys: {missing}")
        inputs = {
            "patches": batch["patches"].to(self.device),
            "patch_pad_mask": batch["patch_pad_mask"].to(self.device),
            "patch_valid_mask": batch["patch_valid_mask"].to(self.device),
            "robot_idx": batch["robot_idx"].to(self.device),
            "program_idx": batch["program_idx"].to(self.device),
            "regime_ids": batch["regime_ids"].to(self.device),
        }
        assert_encoder_inputs_clean({k: None for k in inputs})
        return inputs

    def train_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Run one paired clean/corrupt optimization step."""
        inputs = self._encoder_inputs(batch)
        corruption_mask = batch["corruption_mask"].to(self.device)
        severity = batch["severity"]
        severity_value = float(severity.item()) if isinstance(severity, torch.Tensor) else float(severity)
        if corruption_mask.dtype is not torch.bool:
            raise ValueError("corruption_mask must have torch.bool dtype")
        self.model.train()
        clean = self.model(**inputs)
        corrupted_patches = synthesize_corrupted_patches(
            inputs["patches"],
            inputs["patch_pad_mask"],
            inputs["patch_valid_mask"],
            corruption_mask,
            severity_value,
            generator=self._generator,
        )
        corrupt = self.model(
            patches=corrupted_patches,
            patch_pad_mask=inputs["patch_pad_mask"],
            patch_valid_mask=inputs["patch_valid_mask"],
            robot_idx=inputs["robot_idx"],
            program_idx=inputs["program_idx"],
            regime_ids=inputs["regime_ids"],
        )
        terms = self.criterion(
            clean["patch_latents"],
            corrupt["patch_latents"],
            clean["context_energy"],
            corrupt["context_energy"],
            inputs["patch_valid_mask"],
            corruption_mask,
            step=self.step,
        )
        loss: torch.Tensor = terms["loss"]
        if not torch.isfinite(loss).all():
            raise ValueError("V2 training loss is non-finite")
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        return {k: float(v.detach().cpu().item()) for k, v in terms.items()}

    @torch.no_grad()
    def stationary_loss(
        self, batches: Iterable[Mapping[str, torch.Tensor]]
    ) -> float:
        """Evaluate the full declared signed-likelihood objective sans mutation.

        Each validation batch is scored exactly as trained: the clean
        encoder view plus a deterministically re-synthesized corrupted view
        at the batch's configured severity, combined through
        :meth:`CounterfactualCriterion.forward` with the boundary
        coefficient pinned to its final (fully ramped) value. The returned
        mean preserves the signed NLL: a more-negative valid likelihood
        scores lower (better) than a near-zero one. Corruption synthesis
        uses a per-batch deterministic generator derived from the trainer
        seed, so repeated evaluations compare model states rather than
        noise draws, and the training generator is never consumed here.
        """
        was_training = self.model.training
        self.model.eval()
        eval_step = int(self.criterion.boundary_schedule.full_ramp_step)
        total, count = 0.0, 0
        for index, batch in enumerate(batches):
            inputs = self._encoder_inputs(batch)
            if "corruption_mask" not in batch or "severity" not in batch:
                raise ValueError(
                    "stationary evaluation needs 'corruption_mask'/'severity' "
                    "alongside the encoder batch"
                )
            corruption_mask = batch["corruption_mask"].to(self.device)
            if corruption_mask.dtype is not torch.bool:
                raise ValueError("corruption_mask must have torch.bool dtype")
            severity = batch["severity"]
            severity_value = (
                float(severity.item())
                if isinstance(severity, torch.Tensor)
                else float(severity)
            )
            clean = self.model(**inputs)
            probe_generator = torch.Generator().manual_seed(
                (self.seed + 0x9E3779B9 * (index + 1)) % 2**32
            )
            corrupted_patches = synthesize_corrupted_patches(
                inputs["patches"],
                inputs["patch_pad_mask"],
                inputs["patch_valid_mask"],
                corruption_mask,
                severity_value,
                generator=probe_generator,
            )
            corrupt = self.model(
                patches=corrupted_patches,
                patch_pad_mask=inputs["patch_pad_mask"],
                patch_valid_mask=inputs["patch_valid_mask"],
                robot_idx=inputs["robot_idx"],
                program_idx=inputs["program_idx"],
                regime_ids=inputs["regime_ids"],
            )
            valid = inputs["patch_valid_mask"]
            if not bool(valid.any()):
                continue
            terms = self.criterion(
                clean["patch_latents"],
                corrupt["patch_latents"],
                clean["context_energy"],
                corrupt["context_energy"],
                valid,
                corruption_mask,
                step=eval_step,
            )
            value = float(terms["loss"].detach().cpu().item())
            if value != value or abs(value) == float("inf"):
                raise ValueError("stationary evaluation produced a non-finite objective")
            total += value
            count += 1
        if was_training:
            self.model.train()
        if count == 0:
            raise ValueError("stationary evaluation saw no valid patches")
        return total / count

    def record_eval(self, stationary: float) -> bool:
        """Record a validation point; keep the best stationary state."""
        if not (isinstance(stationary, float) and abs(stationary) != float("inf") and stationary == stationary):
            raise ValueError("stationary must be a finite float")
        self.history.append({"stationary": stationary, "step": float(self.step)})
        if self.best_stationary is None or stationary < self.best_stationary:
            self.best_stationary = stationary
            self.best_state = {
                "model_state": deepcopy(self.model.state_dict()),
                "optimizer_state": deepcopy(self.optimizer.state_dict()),
                "scheduler_state": (
                    None if self.scheduler is None
                    else deepcopy(self.scheduler.state_dict())
                ),
                "step": self.step,
            }
            return True
        return False

    def restore_best_state(self) -> None:
        """Restore model/optimizer/scheduler/step to the best snapshot."""
        if self.best_state is None:
            raise ValueError("no best state recorded")
        state = self.best_state
        self.model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
        if self.scheduler is not None and state["scheduler_state"] is not None:
            self.scheduler.load_state_dict(state["scheduler_state"])  # type: ignore[arg-type]
        self.step = int(state["step"])  # type: ignore[arg-type]

    def fit_geometry(
        self,
        latents: torch.Tensor,
        robot_ids: torch.Tensor,
        program_ids: torch.Tensor,
        regime_ids: torch.Tensor,
        healthy_mask: torch.Tensor,
    ) -> FrozenReference:
        """Fit hierarchical references from verified-healthy latents only."""
        geometry = HierarchicalMahalanobisGeometry(
            self.config.d_model,
            shrinkage=self.config.shrinkage,
            covariance_eps=self.config.covariance_eps,
            min_group_samples=self.config.min_group_samples,
            diag_min_samples=self.config.diag_min_samples,
        )
        geometry.fit(latents, robot_ids, program_ids, regime_ids, healthy_mask)
        return geometry.frozen()

    def save(
        self,
        path: str | Path,
        *,
        geometry: HierarchicalMahalanobisGeometry | FrozenReference | None = None,
        calibrator: object | None = None,
        risk: object | None = None,
        tracker: object | None = None,
        operating_threshold: Mapping[str, object] | None = None,
        confidence_calibrator_cohort: Mapping[str, object] | None = None,
    ) -> None:
        """Save coherent training plus monitoring state (fail fast on path).

        ``operating_threshold`` carries the dev-val-calibrated elevated-patch
        provenance record; ``confidence_calibrator_cohort`` optionally
        overrides the cohort record otherwise derived from the calibrator.
        """
        live = geometry
        if isinstance(geometry, FrozenReference):
            live = geometry._geometry  # noqa: SLF001 - snapshot needs live refs
        save_v2_checkpoint(
            path, self.model, config=self.config, step=self.step,
            optimizer=self.optimizer, scheduler=self.scheduler,
            geometry=live, calibrator=calibrator, risk=risk, tracker=tracker,
            operating_threshold=operating_threshold,
            confidence_calibrator_cohort=confidence_calibrator_cohort,
        )

    def load(
        self,
        path: str | Path,
        *,
        restore_optimizer: bool = True,
        restore_scheduler: bool = True,
    ) -> dict[str, Any]:
        """Restore coherent state saved by :meth:`save`."""
        return load_v2_checkpoint(
            path, self.model,
            optimizer=self.optimizer if restore_optimizer else None,
            scheduler=self.scheduler if (restore_scheduler and self.scheduler is not None) else None,
            expected_config=self.config,
        )


def build_v2_training_stack(
    config: V2Config,
    *,
    device: str = "cpu",
    seed: int = 0,
    lr: float = 1e-3,
) -> tuple[ContextConditionedPatchEncoder, CounterfactualCriterion, V2Trainer]:
    """Construct the encoder, criterion, optimizer, and trainer coherently."""
    from representation.criterion import ProgressiveLambda

    torch.manual_seed(int(seed))
    model = ContextConditionedPatchEncoder(
        n_channels=config.n_channels,
        d_model=config.d_model,
        n_robots=config.n_robots,
        n_programs=config.n_programs,
        n_regimes=config.n_regimes,
        n_prototypes=config.n_prototypes,
        sequence_layers=config.sequence_layers,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    )
    criterion = CounterfactualCriterion(
        boundary_margin=config.boundary_margin,
        background_weight=config.background_weight,
        variance_weight=config.variance_weight,
        covariance_weight=config.covariance_weight,
        boundary_schedule=ProgressiveLambda(
            lambda_max=config.boundary_alpha_max,
            ramp_steps=config.boundary_ramp_steps,
            warmup_steps=config.boundary_warmup_steps,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer = V2Trainer(
        config, model, criterion, optimizer, device=device, seed=seed
    )
    return model, criterion, trainer
