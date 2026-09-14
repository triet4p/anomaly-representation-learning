"""Sprint 17 Task 15 — C8 scorer alternatives (C8-A / C8-B).

Frozen-registry implementation for the C8 (prediction-residual scorer)
suspect only. Both arms replace the B0 per-patch MSE energy for ``S_pred``
with a non-query-conditioned alternative, keeping the valid-masked patch
set, the file-mean aggregation, ``S_pop``, and all metric code unchanged:

- ``C8-A`` robust-residual: per-patch residuals standardized by a
  Fit-fitted per-dimension median/MAD (floor 1e-6; fitted exclusively from
  Fit healthy prediction residuals, never per-query), then Huber energy
  (delta=1.0) averaged over valid masked patches.
- ``C8-B`` cosine-residual: per-patch cosine distance
  (``1 - dot/(|a||b|)``, norms clamped to >= 1e-12, distance clamped to
  [0, 2]) averaged over valid masked patches. Scale-free; no fitting.

Non-query-conditioned means the energy is a pure function of
``(predicted, target, prediction_mask)`` plus Fit-frozen statistics: no
query-latent normalization, attention, or per-file adaptive scaling.
Labels, anomaly masks, severity, categories, and simulator hidden state
never enter fitting or scoring. Scores are finite for finite inputs;
larger means more anomalous. An empty valid-masked set scores 0.0 (the B0
``clamp_min(1)`` convention).
"""

from __future__ import annotations

import torch

#: Registered C8 arm IDs (protocol sprint17-ablation-v4 §2).
C8_ARMS = ("C8-A", "C8-B")

#: Frozen Huber knee for C8-A.
HUBER_DELTA = 1.0

#: Frozen floors for degenerate scales/norms.
MAD_FLOOR = 1e-6
NORM_FLOOR = 1e-12

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C8-A": ("non-query-conditioned standardized Huber prediction-residual "
             "energy (Fit-fitted per-dimension median/MAD, floor 1e-6, Huber "
             "delta 1.0, file-mean over valid masked patches); fitted "
             "statistics only (256 floats), no trainable parameters; "
             "no other adapter"),
    "C8-B": ("non-query-conditioned cosine distance between predicted and "
             "target latents (norm floor 1e-12, file-mean over valid masked "
             "patches); scale-free, no fitting and no parameters; "
             "no other adapter"),
}


def _check_latents(predicted: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("predicted/target latents must be [B, N, D]")
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError("predicted and target latents must share shape")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(predicted.shape[:2]):
        raise ValueError("prediction mask must be [B, N] matching latents")
    if mask.dtype is not torch.bool:
        raise ValueError("prediction mask must have torch.bool dtype")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("latents must be finite")
    return predicted, target, mask


def _check_labels(labels) -> None:
    if labels is None:
        return
    for label in labels:
        value = getattr(label, "value", label)
        if str(value).lower() == "abnormal":
            raise ValueError("C8 standardization cannot include abnormal rows")


def b0_mse_energy(predicted: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """B0 per-patch MSE energy (the replaced baseline; fidelity oracle)."""
    predicted, target, mask = _check_latents(predicted, target, mask)
    errors = ((predicted - target) ** 2).mean(dim=-1)
    return errors.masked_fill(~mask, 0.0)


def file_mean(energies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """B0 file aggregation: mean energy over valid masked patches."""
    if energies.ndim != 2 or mask.ndim != 2 or tuple(energies.shape) != tuple(mask.shape):
        raise ValueError("energies and mask must share [B, N] shape")
    counts = mask.sum(dim=1).to(energies.dtype)
    return (energies * mask.to(energies.dtype)).sum(dim=1) / counts.clamp_min(1.0)


class HuberStandardizer:
    """C8-A Fit-only per-dimension median/MAD standardization."""

    def __init__(self) -> None:
        self.center: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, residuals: torch.Tensor, labels=None) -> "HuberStandardizer":
        """Fit center/scale on Fit healthy prediction residuals [M, D]."""
        _check_labels(labels)
        if not isinstance(residuals, torch.Tensor):
            raise ValueError("residuals must be a torch.Tensor")
        if residuals.ndim != 2 or residuals.shape[0] == 0:
            raise ValueError("residuals must be a non-empty [M, D] tensor")
        if not torch.isfinite(residuals).all():
            raise ValueError("residuals must be finite")
        r = residuals.detach().to(device="cpu", dtype=torch.float64)
        self.center = r.median(dim=0).values
        self.scale = (r - self.center.unsqueeze(0)).abs().median(dim=0).values
        self.scale = self.scale.clamp_min(MAD_FLOOR)
        return self

    def standardize(self, residuals: torch.Tensor) -> torch.Tensor:
        """Standardize residuals with the frozen Fit statistics."""
        if self.center is None or self.scale is None:
            raise ValueError("HuberStandardizer has not been fitted")
        r = residuals.detach().to(device="cpu", dtype=torch.float64)
        if r.shape[-1] != self.center.shape[0]:
            raise ValueError("residual dimension must match fitted dimension")
        return (r - self.center.unsqueeze(0)) / self.scale.unsqueeze(0)

    def provenance(self) -> dict[str, object]:
        """Fitting accounting for the standardizer."""
        if self.center is None or self.scale is None:
            raise ValueError("HuberStandardizer has not been fitted")
        return {
            "source": "Fit-only healthy prediction residuals",
            "n_floats": int(2 * self.center.numel()),
        }


def huber_energy(standardized: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Elementwise Huber energy, averaged over D by the caller pattern."""
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    z = standardized.detach().to(dtype=torch.float64)
    abs_z = z.abs()
    quad = 0.5 * z * z
    lin = delta * (abs_z - 0.5 * delta)
    return torch.where(abs_z <= delta, quad, lin)


def c8a_patch_energies(predicted: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor,
                       standardizer: HuberStandardizer) -> torch.Tensor:
    """C8-A per-patch energies (zero outside the valid masked set)."""
    predicted, target, mask = _check_latents(predicted, target, mask)
    residuals = (predicted - target).detach().to(device="cpu", dtype=torch.float64)
    z = standardizer.standardize(residuals.reshape(-1, residuals.shape[-1]))
    energies = huber_energy(z).mean(dim=-1).reshape(predicted.shape[:2])
    return energies.masked_fill(~mask, 0.0)


def c8b_patch_energies(predicted: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """C8-B per-patch cosine distances (zero outside the valid masked set)."""
    predicted, target, mask = _check_latents(predicted, target, mask)
    a = predicted.detach().to(device="cpu", dtype=torch.float64)
    b = target.detach().to(device="cpu", dtype=torch.float64)
    denom = (a.norm(dim=-1).clamp_min(NORM_FLOOR)
             * b.norm(dim=-1).clamp_min(NORM_FLOOR))
    sim = (a * b).sum(dim=-1) / denom
    dist = (1.0 - sim).clamp(0.0, 2.0)
    return dist.masked_fill(~mask, 0.0)


def arm_patch_energies(arm_id: str, predicted: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor,
                       standardizer: HuberStandardizer | None = None) -> torch.Tensor:
    """Per-patch energies for a registered C8 arm."""
    if arm_id == "C8-A":
        if standardizer is None:
            raise ValueError("C8-A requires a fitted HuberStandardizer")
        return c8a_patch_energies(predicted, target, mask, standardizer)
    if arm_id == "C8-B":
        return c8b_patch_energies(predicted, target, mask)
    raise ValueError(f"unknown Sprint 17 C8 arm: {arm_id!r}")
