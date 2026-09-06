"""Typed observable outputs for the V2 geometry path (Sprint 11 Task 9).

These are pure contracts: shape/dtype/finiteness checks over the tensors the
later V2 stages produce. Aggregation algorithms (Task 13), trajectory tracking
(Task 14), and the risk layer (Task 15) consume these; they do not live here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict
import torch


class V2PatchOutput(TypedDict):
    """Per-patch conditional geometry for one batch."""

    patch_latents: torch.Tensor  # [B, N, D], zero on invalid patches
    patch_energy: torch.Tensor  # [B, N] finite signed NLL, zero on invalid
    patch_valid_mask: torch.Tensor  # bool [B, N]
    group_confidence: NotRequired[torch.Tensor]  # float [B, N] in [0, 1]
    fallback_level: NotRequired[object]  # per-patch resolved hierarchy level


class V2FileState(TypedDict):
    """Distributional file summary; plain mean pooling is never sufficient."""

    file_state: torch.Tensor  # [B, D]
    energy_quantiles: torch.Tensor  # [B, Q] non-decreasing along Q
    tail_energy: torch.Tensor  # [B] upper-tail mean energy
    elevated_fraction: torch.Tensor  # [B] in [0, 1]
    patch_valid_mask: torch.Tensor  # bool [B, N]


class V2TrajectoryFeatures(TypedDict):
    """Longitudinal displacement signals for one file in its episode."""

    displacement: torch.Tensor  # [B] absolute Mahalanobis displacement
    velocity: torch.Tensor  # [B] step-to-step movement, non-negative
    trend: torch.Tensor  # [B] recent-window slope
    persistence: torch.Tensor  # [B] one-sided CUSUM, non-negative


class V2AnomalyConfidence(TypedDict):
    """Unusualness vs verified-healthy behavior; NOT failure probability."""

    confidence: torch.Tensor  # [B] in [0, 1]
    p_normal: torch.Tensor  # [B] empirical healthy-tail value in [0, 1]


class V2FailureRisk(TypedDict):
    """Discrete-time failure probabilities for the fixed 1-day/7-day horizons."""

    risk_1d: torch.Tensor  # [B] in [0, 1]
    risk_7d: torch.Tensor  # [B] in [0, 1]


def _tensor(mapping: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
    value = mapping.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got {value.ndim}")
    return value


def _bool(value: torch.Tensor, name: str) -> None:
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must have torch.bool dtype")


def validate_patch_output(output: Mapping[str, object]) -> None:
    """Validate shapes, masks, and finiteness of a patch-energy output."""
    latents = _tensor(output, "patch_latents", 3)
    energy = _tensor(output, "patch_energy", 2)
    valid = _tensor(output, "patch_valid_mask", 2)
    _bool(valid, "patch_valid_mask")
    b, n, _ = latents.shape
    if tuple(energy.shape) != (b, n) or tuple(valid.shape) != (b, n):
        raise ValueError("patch_energy/patch_valid_mask must match [B, N] of patch_latents")
    if not torch.isfinite(latents).all() or not torch.isfinite(energy).all():
        raise ValueError("patch outputs must be finite")
    # Energy is a signed continuous NLL: tight normal densities score below
    # zero, so only finiteness and invalid-patch silence are required here.
    if bool((energy[~valid] != 0).any()):
        raise ValueError("patch_energy must be zero on invalid patches")
    if bool((latents[~valid].abs().sum() != 0)):
        raise ValueError("patch_latents must be zero on invalid patches")
    if "group_confidence" in output and output["group_confidence"] is not None:
        conf = _tensor(output, "group_confidence", 2)
        if tuple(conf.shape) != (b, n):
            raise ValueError("group_confidence must have shape [B, N]")
        if bool(((conf < 0) | (conf > 1)).any()) or not torch.isfinite(conf).all():
            raise ValueError("group_confidence must be finite values in [0, 1]")


def validate_file_state(state: Mapping[str, object]) -> None:
    """Validate the distributional file-state boundary contract."""
    file_state = _tensor(state, "file_state", 2)
    quantiles = _tensor(state, "energy_quantiles", 2)
    tail = _tensor(state, "tail_energy", 1)
    elevated = _tensor(state, "elevated_fraction", 1)
    valid = _tensor(state, "patch_valid_mask", 2)
    _bool(valid, "patch_valid_mask")
    b = file_state.shape[0]
    if quantiles.shape[0] != b or tail.shape != (b,) or elevated.shape != (b,):
        raise ValueError("file-state tensors must agree on batch size")
    if valid.shape[0] != b:
        raise ValueError("patch_valid_mask must agree on batch size")
    for name, tensor in (("file_state", file_state), ("energy_quantiles", quantiles)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
    if bool((quantiles.diff(dim=1) < 0).any()):
        raise ValueError("energy_quantiles must be non-decreasing along Q")
    if bool(((elevated < 0) | (elevated > 1)).any()):
        raise ValueError("elevated_fraction must be in [0, 1]")


def validate_trajectory(features: Mapping[str, object]) -> None:
    """Validate longitudinal trajectory fields."""
    disp = _tensor(features, "displacement", 1)
    vel = _tensor(features, "velocity", 1)
    trend = _tensor(features, "trend", 1)
    persist = _tensor(features, "persistence", 1)
    if not (disp.shape == vel.shape == trend.shape == persist.shape):
        raise ValueError("trajectory fields must share [B] shape")
    for name, tensor in (("displacement", disp), ("velocity", vel), ("persistence", persist)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
    if bool((disp < 0).any()) or bool((vel < 0).any()) or bool((persist < 0).any()):
        raise ValueError("displacement/velocity/persistence must be non-negative")
    if not torch.isfinite(trend).all():
        raise ValueError("trend must be finite")


def validate_confidence(output: Mapping[str, object]) -> None:
    """Validate anomaly-confidence outputs (distinct from failure risk)."""
    for name in ("confidence", "p_normal"):
        tensor = _tensor(output, name, 1)
        if not torch.isfinite(tensor).all() or bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError(f"{name} must be finite values in [0, 1]")


def validate_risk(output: Mapping[str, object]) -> None:
    """Validate the fixed one-day/seven-day risk outputs and their ordering."""
    one = _tensor(output, "risk_1d", 1)
    seven = _tensor(output, "risk_7d", 1)
    if one.shape != seven.shape:
        raise ValueError("risk_1d and risk_7d must share [B] shape")
    for name, tensor in (("risk_1d", one), ("risk_7d", seven)):
        if not torch.isfinite(tensor).all() or bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError(f"{name} must be finite values in [0, 1]")
    if bool((seven + 1e-6 < one).any()):
        raise ValueError("risk_7d must be at least risk_1d per horizon")
