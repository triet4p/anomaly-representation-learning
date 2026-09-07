"""Typed observable outputs for the V2 geometry path (Sprint 11 Task 9).

These are pure contracts: shape/dtype/finiteness checks over the tensors the
later V2 stages produce. Aggregation algorithms (Task 13), trajectory tracking
(Task 14), and the risk layer (Task 15) consume these; they do not live here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict
import torch

#: Canonical patch-energy field identities (Deep-Review Finding 1, Batch B1).
#: ``context_energy`` is the encoder conditional NLL (signed Gaussian NLL,
#: higher-is-more-anomalous): the exact score optimized by the localized
#: boundary loss and the monitoring score Batch B2 must expose unchanged at
#: inference. ``population_energy`` is the separately fitted hierarchical
#: Mahalanobis/mixture energy over stored references: a distinct
#: population-energy signal that MUST NOT silently replace the context
#: energy. No shared legacy field exists: producers expose exactly one
#: canonical field each, and cross-field presence is rejected.
CONTEXT_ENERGY_FIELD: str = "context_energy"
POPULATION_ENERGY_FIELD: str = "population_energy"
#: Alias naming the training-to-monitoring contract: the boundary-trained
#: context energy is the monitoring energy Batch B2 exposes unchanged.
MONITORING_ENERGY_FIELD: str = CONTEXT_ENERGY_FIELD


class V2ContextPatchOutput(TypedDict):
    """Per-patch encoder conditional geometry (boundary-trained score)."""

    patch_latents: torch.Tensor  # [B, N, D], zero on invalid patches
    context_energy: torch.Tensor  # [B, N] finite signed NLL, zero on invalid
    patch_valid_mask: torch.Tensor  # bool [B, N]
    group_confidence: NotRequired[torch.Tensor]  # float [B, N] in [0, 1]
    fallback_level: NotRequired[object]  # per-patch resolved hierarchy level


class V2PopulationPatchOutput(TypedDict):
    """Per-patch hierarchical population energy (distinct signal)."""

    population_energy: torch.Tensor  # [B, N] finite signed NLL, zero on invalid
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


def _check_energy_field(
    output: Mapping[str, object],
    energy_name: str,
    forbidden_name: str,
    *,
    require_latents: bool,
) -> None:
    """Shared signed-NLL checks for one canonical energy identity."""
    if forbidden_name in output:
        raise ValueError(
            f"{energy_name} output must not carry {forbidden_name}: "
            "context and population energies are distinct signals"
        )
    energy = _tensor(output, energy_name, 2)
    valid = _tensor(output, "patch_valid_mask", 2)
    _bool(valid, "patch_valid_mask")
    b, n = energy.shape
    if tuple(valid.shape) != (b, n):
        raise ValueError(f"{energy_name}/patch_valid_mask must share [B, N]")
    if not torch.isfinite(energy).all():
        raise ValueError(f"{energy_name} must be finite")
    if bool((energy[~valid] != 0).any()):
        raise ValueError(f"{energy_name} must be zero on invalid patches")
    if require_latents:
        latents = _tensor(output, "patch_latents", 3)
        if tuple(latents.shape[:2]) != (b, n):
            raise ValueError(f"{energy_name}/patch_latents must share [B, N]")
        if not torch.isfinite(latents).all():
            raise ValueError("patch outputs must be finite")
        if bool((latents[~valid].abs().sum() != 0)):
            raise ValueError("patch_latents must be zero on invalid patches")


def validate_context_patch_output(output: Mapping[str, object]) -> None:
    """Validate the encoder context-energy output (boundary-trained score)."""
    _check_energy_field(
        output, CONTEXT_ENERGY_FIELD, POPULATION_ENERGY_FIELD, require_latents=True
    )


def validate_population_patch_output(output: Mapping[str, object]) -> None:
    """Validate a hierarchical population-energy output (distinct signal)."""
    _check_energy_field(
        output, POPULATION_ENERGY_FIELD, CONTEXT_ENERGY_FIELD, require_latents=False
    )


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
