"""Staged V2 experiment runner and provenance contracts (Sprint 11 Task 22).

Committed-source experiment entry point for the 2-epoch contract stage,
the 5-epoch coefficient-balance stage, and the 50-epoch frozen full stage,
each with a matched normal-only geometry control and a hybrid boundary
model. Every run reuses the public training stack
(:func:`representation.v2_trainer.build_v2_training_stack`), fits
references and calibration on verified-healthy development views only,
never touches the sealed static/temporal test views, and records a
machine-readable provenance manifest covering commit, data/checkpoint/
reference checksums, device, seeds, wall time, raw/weighted losses,
gradient norms, geometry health, and exact output roots.

Variant contract (matched pair):

- ``control-normal-only`` trains the normal-manifold core with the
  boundary coefficient forced to zero (``boundary_alpha_max=0.0``).
- ``hybrid-boundary`` trains the same stack with localized synthetic
  boundary learning enabled (``boundary_alpha_max>0``).

All other coefficients, data, seeds, device, and epoch budgets match
within a stage. The 5-epoch balance stage runs the predeclared
:data:`BALANCE_MATRIX`; the 50-epoch full stage runs one frozen config
per variant (final hybrid selection itself belongs to Task 29).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from representation.data import collate_variable_files
from representation.v2_aggregation import aggregate_file_state
from representation.v2_config import V2Config
from representation.v2_inference import patch_regime_ids
from representation.v2_risk import HealthyTailCalibrator
from representation.v2_trainer import build_v2_training_stack
from representation.v2_trajectory import TrajectoryTracker
from synth.chronicle import load_chronological
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import SampleLabel

#: Stage name to frozen epoch budget. Configs MUST use these budgets.
STAGE_EPOCHS: dict[str, int] = {"contract": 2, "balance": 5, "full": 50}

#: Canonical variant names.
CONTROL_VARIANT = "control-normal-only"
HYBRID_VARIANT = "hybrid-boundary"

#: Commit placeholder recorded in configs and resolved at runtime.
COMMIT_PLACEHOLDER = "{COMMIT}"

#: Provenance manifest schema version.
PROVENANCE_SCHEMA_VERSION = 1

#: Sealed views the runner MUST NEVER train on, calibrate on, or select with.
SEALED_VIEWS: tuple[str, ...] = ("test_static", "test_temporal")

#: Views the runner is allowed to consume (verified-healthy development only).
TRAINING_VIEWS: tuple[str, ...] = ("dev_train", "dev_val")


def control_coefficients() -> dict[str, float]:
    """Return the frozen normal-only control coefficients."""
    return {
        "boundary_alpha_max": 0.0,
        "boundary_margin": 1.0,
        "background_weight": 1.0,
        "variance_weight": 1.0,
        "covariance_weight": 1.0,
    }


def hybrid_coefficients() -> dict[str, float]:
    """Return the default hybrid boundary coefficients."""
    return {
        "boundary_alpha_max": 1.0,
        "boundary_margin": 1.0,
        "background_weight": 1.0,
        "variance_weight": 1.0,
        "covariance_weight": 1.0,
    }


def balance_matrix() -> list[dict[str, object]]:
    """Return the predeclared 5-epoch coefficient-balance matrix.

    Cell ``balance-b`` equals the default hybrid; the full-stage frozen
    hybrid reuses it unless Task 29 selects otherwise on validation-only
    diagnostics. Never edited per-run: rejected cells and rationale
    belong to the Task 29 selection record.
    """
    base = hybrid_coefficients()
    return [
        {"cell": "balance-a", "coefficients": {**base, "boundary_alpha_max": 0.5}},
        {"cell": "balance-b", "coefficients": {**base}},
        {
            "cell": "balance-c",
            "coefficients": {**base, "boundary_alpha_max": 1.0, "background_weight": 0.5},
        },
        {"cell": "balance-control", "coefficients": control_coefficients()},
    ]
#: Immutable predeclared 5-epoch coefficient-balance matrix. Derived once
#: from :func:`balance_matrix` so the matrix keeps a single definition.
BALANCE_MATRIX: tuple[dict[str, object], ...] = tuple(balance_matrix())


@dataclass(frozen=True)
class StageSpec:
    """One executable staged experiment cell."""

    stage: str
    variant: str
    epochs: int
    seed: int = 0
    d_model: int = 32
    batch_size: int = 8
    lr: float = 1e-3
    corruption_rate: float = 0.1
    severity: float = 2.0
    cell: str = "default"
    coefficients: Mapping[str, float] = field(default_factory=hybrid_coefficients)
    commit: str = COMMIT_PLACEHOLDER

    def validate(self) -> "StageSpec":
        """Fail fast when the spec breaks the staged contract."""
        if self.stage not in STAGE_EPOCHS:
            raise ValueError(
                f"unknown stage {self.stage!r}; expected one of {sorted(STAGE_EPOCHS)}"
            )
        if self.epochs != STAGE_EPOCHS[self.stage]:
            raise ValueError(
                f"stage {self.stage!r} requires {STAGE_EPOCHS[self.stage]} epochs, "
                f"got {self.epochs}"
            )
        if self.variant not in (CONTROL_VARIANT, HYBRID_VARIANT):
            raise ValueError(
                f"unknown variant {self.variant!r}; "
                f"expected {CONTROL_VARIANT!r} or {HYBRID_VARIANT!r}"
            )
        if self.variant == CONTROL_VARIANT and (
            float(self.coefficients.get("boundary_alpha_max", 0.0)) != 0.0
        ):
            raise ValueError("control variant requires boundary_alpha_max == 0.0")
        if self.variant == HYBRID_VARIANT and (
            float(self.coefficients.get("boundary_alpha_max", 0.0)) <= 0.0
        ):
            raise ValueError("hybrid variant requires boundary_alpha_max > 0.0")
        if not 0.0 < self.corruption_rate < 1.0:
            raise ValueError("corruption_rate must be in (0, 1)")
        if self.batch_size <= 0 or self.d_model <= 0:
            raise ValueError("batch_size and d_model must be positive")
        return self


def stage_epochs(stage: str) -> int:
    """Return the frozen epoch budget for one stage name."""
    if stage not in STAGE_EPOCHS:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {sorted(STAGE_EPOCHS)}"
        )
    return STAGE_EPOCHS[stage]


def resolve_commit(value: str | None) -> dict[str, object]:
    """Resolve the commit placeholder to the exact runtime commit.

    Configs record ``"{COMMIT}"``; at runtime this resolves to the
    current ``git rev-parse HEAD`` plus a dirty-tree flag. An explicit
    40-hex SHA passes through with ``commit_source="explicit"``. When
    git is unavailable the run still records an honest
    ``unknown-no-git`` marker instead of guessing.
    """
    if value not in (None, "", COMMIT_PLACEHOLDER, "${COMMIT}", "placeholder"):
        sha = str(value).strip()
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
            raise ValueError(
                f"explicit commit must be a 40-hex SHA, got {value!r}"
            )
        return {"commit": sha.lower(), "commit_source": "explicit", "dirty": False}
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            .stdout.strip()
            .lower()
        )
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        return {"commit": sha, "commit_source": "git-head", "dirty": bool(dirty_out)}
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return {
            "commit": "unknown-no-git",
            "commit_source": "unresolved",
            "dirty": None,
        }


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one staged experiment YAML config."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"staged experiment config not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"staged config {target} must be a mapping")
    stage = config.get("stage")
    if stage not in STAGE_EPOCHS:
        raise ValueError(
            f"staged config {target} has unknown stage {stage!r}; "
            f"expected one of {sorted(STAGE_EPOCHS)}"
        )
    cells = config.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"staged config {target} must declare a non-empty 'cells' list")
    for entry in cells:
        if not isinstance(entry, dict) or "variant" not in entry:
            raise ValueError(f"staged config {target} has a malformed cell: {entry!r}")
        StageSpec(
            stage=stage,
            variant=str(entry["variant"]),
            epochs=int(config.get("epochs", STAGE_EPOCHS[stage])),
            seed=int(config.get("seed", 0)),
            d_model=int(config.get("d_model", 32)),
            batch_size=int(config.get("batch_size", 8)),
            cell=str(entry.get("cell", "default")),
            coefficients=dict(entry.get("coefficients", {})) or {},
            commit=str(config.get("commit", COMMIT_PLACEHOLDER)),
        ).validate()
    return config


def iter_cells(config: Mapping[str, Any]) -> list[StageSpec]:
    """Expand one validated staged config into executable cell specs."""
    stage = str(config["stage"])
    return [
        StageSpec(
            stage=stage,
            variant=str(entry["variant"]),
            epochs=int(config.get("epochs", STAGE_EPOCHS[stage])),  # type: ignore[arg-type]
            seed=int(config.get("seed", 0)),  # type: ignore[arg-type]
            d_model=int(config.get("d_model", 32)),  # type: ignore[arg-type]
            batch_size=int(config.get("batch_size", 8)),  # type: ignore[arg-type]
            lr=float(config.get("lr", 1e-3)),  # type: ignore[arg-type]
            corruption_rate=float(config.get("corruption_rate", 0.1)),  # type: ignore[arg-type]
            severity=float(config.get("severity", 2.0)),  # type: ignore[arg-type]
            cell=str(entry.get("cell", "default")),
            coefficients=dict(entry.get("coefficients", {}) or {}),
            commit=str(config.get("commit", COMMIT_PLACEHOLDER)),
        ).validate()
        for entry in config["cells"]  # type: ignore[union-attr]
    ]


def assert_no_test_files(
    used_ids: list[str], sealed_ids: set[str], *, context: str
) -> None:
    """Fail fast when any sealed test id reaches training/calibration."""
    leaked = [fid for fid in used_ids if fid in sealed_ids]
    if leaked:
        raise RuntimeError(
            f"sealed-test violation in {context}: {len(leaked)} test files "
            f"(e.g. {leaked[:3]!r}) reached a training-only path"
        )


def geometry_health(geometry: object) -> dict[str, object]:
    """Summarize fitted-reference conditioning for the provenance record."""
    inner = getattr(geometry, "_geometry", geometry)
    groups: list[object] = (
        list(getattr(inner, "_groups", {}).values())
        + list(getattr(inner, "_pair", {}).values())
        + list(getattr(inner, "_robot", {}).values())
    )
    fleet = getattr(inner, "_fleet", None)
    if fleet is not None:
        groups.append(fleet)
    if not groups:
        raise ValueError("geometry health requires at least one fitted reference")
    conds: list[float] = []
    counts: list[int] = []
    for stats in groups:
        cov = getattr(stats, "cov")
        counts.append(int(getattr(stats, "n")))
        try:
            cond = float(torch.linalg.cond(cov).item())
        except RuntimeError:
            cond = float("inf")
        conds.append(cond if cond == cond else float("inf"))
    finite = [c for c in conds if c != float("inf")]
    return {
        "n_references": len(groups),
        "min_group_samples": min(counts),
        "worst_condition_number": max(conds),
        "median_condition_number": float(
            torch.tensor(sorted(conds)).median().item()
        ),
        "all_finite_condition": len(finite) == len(conds),
    }


def _require_data_root(data_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"chronological manifest not found at {manifest_path}. "
            "Set data_root to the materialized chronicle root."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return root, manifest


def run_stage_cell(
    spec: StageSpec,
    *,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute one staged cell and write checkpoint plus provenance.

    Training, geometry commissioning, and tail calibration consume only
    ``dev_train``/``dev_val`` verified-healthy files. The sealed
    ``test_static``/``test_temporal`` views are loaded solely as id sets
    for the leakage assertion and recorded as untouched.
    """
    spec.validate()
    started_wall = time.time()
    started_perf = time.perf_counter()
    root, manifest = _require_data_root(data_root)
    out_dir = Path(output_root) / spec.stage / spec.variant / spec.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "v2_checkpoint.pt"

    splits = manifest["splits"]
    sealed_ids = {fid for view in SEALED_VIEWS for fid in splits[view]}
    for view in TRAINING_VIEWS:
        if not splits[view]:
            raise RuntimeError(f"chronicle view {view!r} is empty; regenerate the dataset")

    samples, _ = load_chronological(root)
    by_id = {s.file_id: s for s in samples}
    train_files = [by_id[i] for i in splits["dev_train"]]
    val_files = [by_id[i] for i in splits["dev_val"]]
    assert_no_test_files(
        [s.file_id for s in train_files + val_files], sealed_ids, context="stage inputs"
    )
    for sample in train_files + val_files:
        if sample.file_label is not SampleLabel.NORMAL:
            raise RuntimeError(
                f"staged training fits verified-healthy files only; got {sample.file_id}"
            )
        if sample.split_provenance is not None and sample.split_provenance.is_quarantined:
            raise RuntimeError(
                f"quarantined file {sample.file_id} must never train staged geometry"
            )

    torch.manual_seed(spec.seed)
    n_robots = max(s.robot_idx for s in samples) + 1
    n_programs = max(s.program_idx for s in samples) + 1
    config = V2Config(
        n_channels=int(samples[0].x.shape[0]),
        d_model=spec.d_model,
        n_robots=n_robots,
        n_programs=n_programs,
        n_regimes=7,
        boundary_alpha_max=float(spec.coefficients["boundary_alpha_max"]),
        boundary_margin=float(spec.coefficients["boundary_margin"]),
        background_weight=float(spec.coefficients["background_weight"]),
        variance_weight=float(spec.coefficients["variance_weight"]),
        covariance_weight=float(spec.coefficients["covariance_weight"]),
        seed=spec.seed,
    )
    patchifier = Patchifier(PatchConfig(patch_size=config.patch_size, stride=config.stride))
    model, _criterion, trainer = build_v2_training_stack(
        config, device=device, seed=spec.seed, lr=spec.lr
    )

    def make_batch(files: list[object], seed: int) -> dict[str, torch.Tensor]:
        base = collate_variable_files(files, patchifier)  # type: ignore[arg-type]
        count = base["patches"].shape[1]
        regimes = patch_regime_ids(files, base["starts"], count)  # type: ignore[arg-type]
        gen = torch.Generator().manual_seed(int(seed))
        corruption = (torch.rand(base["patch_valid_mask"].shape, generator=gen) < spec.corruption_rate)
        corruption = corruption & base["patch_valid_mask"]
        return {
            "patches": base["patches"],
            "patch_pad_mask": base["patch_pad_mask"],
            "patch_valid_mask": base["patch_valid_mask"],
            "robot_idx": base["robot_idx"],
            "program_idx": base["program_idx"],
            "regime_ids": regimes,
            "corruption_mask": corruption,
            "severity": float(spec.severity),
        }

    history: list[dict[str, float]] = []
    val_probe = [
        make_batch(val_files[start : start + spec.batch_size], spec.seed + 1)
        for start in range(0, len(val_files), spec.batch_size)
    ]
    for epoch in range(1, spec.epochs + 1):
        order_gen = torch.Generator().manual_seed(spec.seed + epoch)
        order = torch.randperm(len(train_files), generator=order_gen).tolist()
        ordered = [train_files[i] for i in order]
        agg: dict[str, list[float]] = {}
        grad_norms: list[float] = []
        for start in range(0, len(ordered), spec.batch_size):
            batch = make_batch(
                ordered[start : start + spec.batch_size],
                spec.seed + epoch * 1000 + start,
            )
            terms = trainer.train_step(batch)
            for key, value in terms.items():
                agg.setdefault(key, []).append(value)
            total_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_sq += float(param.grad.detach().pow(2).sum().item())
            grad_norms.append(total_sq**0.5)
        train_probe = [
            make_batch(train_files[start : start + spec.batch_size], spec.seed + 10000 + epoch)
            for start in range(0, len(train_files), spec.batch_size)
        ]
        train_stat = trainer.stationary_loss(train_probe)
        val_stat = trainer.stationary_loss(val_probe)
        improved = trainer.record_eval(float(val_stat))
        summary = {key: sum(values) / len(values) for key, values in agg.items()}
        summary.update(
            {
                "train_stationary": float(train_stat),
                "val_stationary": float(val_stat),
                "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                "improved": bool(improved),
                "step": float(trainer.step),
            }
        )
        if not all(v == v for v in summary.values()):
            raise RuntimeError(
                f"non-finite staged metrics at epoch {epoch}; refusing to checkpoint"
            )
        history.append(summary)
    if trainer.best_state is None:
        raise RuntimeError("stationary selection recorded no finite point; no checkpoint")
    trainer.restore_best_state()

    model.eval()
    # Commission boundary: training runs on the selected device, while
    # geometry references are commissioned on CPU (fit() normalizes there
    # and restored inference scores on CPU). Transfer explicitly here so a
    # CUDA run never mixes device placement under boolean-mask selection.
    latent_rows, robot_rows, program_rows, regime_rows, train_states = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(train_files), spec.batch_size):
            batch = make_batch(
                train_files[start : start + spec.batch_size], spec.seed + 777
            )
            inputs = trainer._encoder_inputs(batch)
            encoded = model(**inputs)
            latents_cpu = encoded["patch_latents"].cpu()
            energy_cpu = encoded["patch_energy"].cpu()
            valid = batch["patch_valid_mask"]
            latent_rows.append(latents_cpu[valid])
            robot_rows.append(batch["robot_idx"].unsqueeze(1).expand_as(valid)[valid])
            program_rows.append(batch["program_idx"].unsqueeze(1).expand_as(valid)[valid])
            regime_rows.append(batch["regime_ids"][valid])
            state = aggregate_file_state(
                latents_cpu,
                energy_cpu,
                valid,
                batch["regime_ids"],
                top_q_fraction=config.top_q_fraction,
                elevated_threshold=config.elevated_threshold,
                n_regimes=config.n_regimes,
            )
            train_states.append(state["file_state"])
    geometry = trainer.fit_geometry(
        torch.cat(latent_rows),
        torch.cat(robot_rows),
        torch.cat(program_rows),
        torch.cat(regime_rows),
        torch.ones(sum(r.shape[0] for r in latent_rows), dtype=torch.bool),
    )
    train_states = torch.cat(train_states)
    tracker = TrajectoryTracker(config.d_model)
    tracker.fit_commissioning(train_states, torch.ones(train_states.shape[0], dtype=torch.bool))

    val_states = []
    with torch.no_grad():
        for start in range(0, len(val_files), spec.batch_size):
            batch = make_batch(val_files[start : start + spec.batch_size], spec.seed + 778)
            inputs = trainer._encoder_inputs(batch)
            encoded = model(**inputs)
            state = aggregate_file_state(
                encoded["patch_latents"].cpu(),
                encoded["patch_energy"].cpu(),
                batch["patch_valid_mask"],
                batch["regime_ids"],
                top_q_fraction=config.top_q_fraction,
                elevated_threshold=config.elevated_threshold,
                n_regimes=config.n_regimes,
            )
            val_states.append(state["file_state"])
    val_states = torch.cat(val_states)
    if val_states.shape[0] >= config.calibration_min_samples:
        calibration_states, calibration_source = val_states, "dev-val only"
    else:
        calibration_states = torch.cat([train_states, val_states])
        calibration_source = (
            f"pooled dev-train+dev-val ({val_states.shape[0]} < floor "
            f"{config.calibration_min_samples}; server runs use dev-val only)"
        )
    probe_tracker = TrajectoryTracker(config.d_model)
    probe_tracker.fit_commissioning(train_states, torch.ones(train_states.shape[0], dtype=torch.bool))
    displacements = torch.cat(
        [probe_tracker.update(row)["displacement"] for row in calibration_states]
    )
    calibrator = HealthyTailCalibrator(min_samples=config.calibration_min_samples).fit(
        displacements
    )

    trainer.save(checkpoint_path, geometry=geometry, calibrator=calibrator, tracker=tracker)
    health = geometry_health(geometry)
    commit_info = resolve_commit(spec.commit)
    wall_s = time.perf_counter() - started_perf
    shard_digests = [
        {"path": shard["path"], "sha256": shard["sha256"], "count": shard["count"]}
        for shard in manifest["shards"]
    ]
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "stage": spec.stage,
        "variant": spec.variant,
        "cell": spec.cell,
        "epochs": spec.epochs,
        "commit": commit_info["commit"],
        "commit_source": commit_info["commit_source"],
        "working_tree_dirty": commit_info["dirty"],
        "data_root": str(root),
        "data_manifest_path": str(root / "manifest.json"),
        "config_hash": manifest["config_hash"],
        "generator_version": manifest.get("generator_version"),
        "shard_digests": shard_digests,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reference_source": "dev-train commissioning (frozen hierarchical references)",
        "calibration_source": calibration_source,
        "device": device,
        "torch_cuda_available": torch.cuda.is_available(),
        "seeds": {
            "training": spec.seed,
            "corruption": f"seed+epoch*1000+offset (base {spec.seed})",
            "calibration": spec.seed + 777,
        },
        "coefficients": dict(spec.coefficients),
        "d_model": spec.d_model,
        "batch_size": spec.batch_size,
        "lr": spec.lr,
        "corruption_rate": spec.corruption_rate,
        "severity": spec.severity,
        "wall_time_s": wall_s,
        "started_at_unix": started_wall,
        "output_root": str(out_dir),
        "best_stationary": trainer.best_stationary,
        "final_step": trainer.step,
        "history": history,
        "geometry_health": health,
        "sealed_test": {
            "training_views": list(TRAINING_VIEWS),
            "forbidden_views": list(SEALED_VIEWS),
            "forbidden_file_count": len(sealed_ids),
            "training_file_count": len(train_files) + len(val_files),
            "attestation": (
                "test_static/test_temporal ids asserted disjoint from every "
                "training/calibration/selection input; selection used the "
                "stationary validation objective only"
            ),
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return provenance


def run_staged_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    output_root: str | Path,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Execute every cell of one staged YAML config file."""
    config = load_stage_config(config_path)
    results = [
        run_stage_cell(spec, data_root=data_root, output_root=output_root, device=device)
        for spec in iter_cells(config)
    ]
    out_dir = Path(output_root) / str(config["stage"])
    (out_dir / "matrix_summary.json").write_text(
        json.dumps(
            {
                "stage": config["stage"],
                "commit": results[0]["commit"],
                "cells": [
                    {
                        "cell": r["cell"],
                        "variant": r["variant"],
                        "checkpoint_sha256": r["checkpoint_sha256"],
                        "best_stationary": r["best_stationary"],
                        "wall_time_s": r["wall_time_s"],
                        "worst_condition_number": r["geometry_health"][
                            "worst_condition_number"
                        ],
                    }
                    for r in results
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return results


__all__ = [
    "BALANCE_MATRIX",
    "COMMIT_PLACEHOLDER",
    "CONTROL_VARIANT",
    "HYBRID_VARIANT",
    "PROVENANCE_SCHEMA_VERSION",
    "SEALED_VIEWS",
    "STAGE_EPOCHS",
    "TRAINING_VIEWS",
    "StageSpec",
    "assert_no_test_files",
    "balance_matrix",
    "control_coefficients",
    "geometry_health",
    "hybrid_coefficients",
    "iter_cells",
    "load_stage_config",
    "resolve_commit",
    "run_stage_cell",
    "run_staged_config",
    "sha256_file",
    "stage_epochs",
]
