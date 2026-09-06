"""Coherent V2 checkpoints (Sprint 11 Task 16).

One atomic payload carries model, optimizer, scheduler, global step,
V2 configuration, hierarchical geometry references, anomaly-confidence
calibration, censored survival risk, and the longitudinal tracker fixed
baseline. Loading validates schema and configuration compatibility and
fails fast — there is no missing-checkpoint fallback and no silent
partial restore.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from representation.v2_config import V2Config

SCHEMA_VERSION = 2


def _require_nonneg_step(step: int) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")


def save_v2_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    config: V2Config,
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    geometry: object | None = None,
    calibrator: object | None = None,
    risk: object | None = None,
    tracker: object | None = None,
) -> None:
    """Atomically save coherent V2 training/monitoring state."""
    _require_nonneg_step(step)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "step": int(step),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if geometry is not None:
        snapshot = geometry.snapshot()  # type: ignore[attr-defined]
        payload["geometry"] = snapshot
    if calibrator is not None:
        payload["calibrator"] = calibrator.state_dict()  # type: ignore[attr-defined]
    if risk is not None:
        payload["risk"] = risk.state_dict()  # type: ignore[attr-defined]
    if tracker is not None:
        payload["tracker"] = tracker.state_dict()  # type: ignore[attr-defined]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(target.parent), delete=False, suffix=".tmp"
        ) as handle:
            temporary = handle.name
            torch.save(payload, handle.name)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def load_v2_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_config: V2Config | None = None,
) -> dict[str, Any]:
    """Validate and restore a V2 checkpoint into live objects.

    Returns the stored ``config`` dict, ``step``, and any reference
    payloads (``geometry``/``calibrator``/``risk``/``tracker`` snapshots)
    for the caller to restore into monitoring objects. Structural
    config mismatches raise; incompatible optimizer/scheduler state
    raises rather than partially restoring.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"V2 checkpoint not found: {target}")
    payload: Mapping[str, Any] = torch.load(str(target), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported V2 checkpoint schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise ValueError("V2 checkpoint is missing its configuration")
    if expected_config is not None:
        current = expected_config.to_dict()
        for key in (
            "n_channels", "d_model", "n_robots", "n_programs", "n_regimes",
            "n_prototypes", "patch_size", "stride",
        ):
            if stored.get(key) != current.get(key):
                raise ValueError(
                    f"V2 checkpoint config mismatch on {key!r}: "
                    f"checkpoint {stored.get(key)!r} != expected {current.get(key)!r}"
                )
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("V2 checkpoint is missing model_state")
    missing, unexpected = model.load_state_dict(model_state, strict=False), None
    if missing.missing_keys or missing.unexpected_keys:
        raise ValueError(
            "V2 checkpoint model_state is incompatible: "
            f"missing={missing.missing_keys} unexpected={missing.unexpected_keys}"
        )
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("V2 checkpoint has no optimizer_state to restore")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if "scheduler_state" not in payload:
            raise ValueError("V2 checkpoint has no scheduler_state to restore")
        scheduler.load_state_dict(payload["scheduler_state"])
    return {
        "config": stored,
        "step": int(payload.get("step", 0)),
        "geometry": payload.get("geometry"),
        "calibrator": payload.get("calibrator"),
        "risk": payload.get("risk"),
        "tracker": payload.get("tracker"),
    }
