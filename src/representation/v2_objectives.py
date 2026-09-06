"""Localized synthetic counterfactual objectives (Sprint 11 Task 12).

Deterministic in-memory clean/corrupt pairing preserves unit, robot,
program, regime, and nuisance context: the corrupted view differs only
inside the synthetic mask. Losses combine normal density control
(variance/covariance), unaffected-background consistency, relative
localized boundary energy, progressive severity ordering, and a ramped
boundary coefficient. Synthetic masks shape the loss only and never enter
any encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from representation.criterion import ProgressiveLambda


def synthesize_corrupted_patches(
    patches: torch.Tensor,
    patch_pad_mask: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    corruption_mask: torch.Tensor,
    severity: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a corrupted copy differing only on masked, valid, real steps."""
    if patches.ndim != 4:
        raise ValueError(f"patches must be [B, N, C, W], got {patches.shape}")
    b, n, _, w = patches.shape
    if tuple(patch_valid_mask.shape) != (b, n) or tuple(corruption_mask.shape) != (b, n):
        raise ValueError("patch_valid_mask and corruption_mask must have shape [B, N]")
    if tuple(patch_pad_mask.shape) != (b, n, w):
        raise ValueError("patch_pad_mask must have shape [B, N, W]")
    if corruption_mask.dtype is not torch.bool or patch_valid_mask.dtype is not torch.bool:
        raise ValueError("corruption_mask and patch_valid_mask must be bool")
    if not math_is_finite_nonneg(severity):
        raise ValueError("severity must be a finite non-negative float")
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    direction = torch.randn(
        patches.shape, generator=gen, dtype=patches.dtype, device=patches.device
    )
    real = (~patch_pad_mask).unsqueeze(2).to(patches.dtype)
    active = (corruption_mask & patch_valid_mask).unsqueeze(-1).unsqueeze(-1).to(patches.dtype)
    perturbation = float(severity) * direction * real * active
    corrupted = patches + perturbation
    return corrupted.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)


def math_is_finite_nonneg(value: float) -> bool:
    """Return whether a severity-like scalar is finite and non-negative."""
    try:
        return bool(value >= 0.0) and bool(torch.isfinite(torch.as_tensor(float(value))))
    except (TypeError, ValueError):
        return False


def variance_loss(latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Hinge loss keeping per-dimension std near one on valid patches."""
    rows = latents[valid]
    if rows.numel() == 0:
        return latents.sum() * 0.0
    std = rows.std(dim=0, unbiased=False)
    return torch.relu(1.0 - std).mean()


def covariance_loss(latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal covariance on valid patches."""
    rows = latents[valid]
    if rows.shape[0] < 2:
        return latents.sum() * 0.0
    centered = rows - rows.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(1, rows.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off.pow(2)).mean()


class CounterfactualCriterion(nn.Module):
    """Clean/corrupt geometry losses with a ramped boundary coefficient."""

    def __init__(
        self,
        boundary_margin: float = 1.0,
        background_weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 1.0,
        boundary_schedule: ProgressiveLambda | None = None,
    ) -> None:
        super().__init__()
        if boundary_margin < 0.0 or background_weight < 0.0:
            raise ValueError("margin and background weight must be non-negative")
        if variance_weight < 0.0 or covariance_weight < 0.0:
            raise ValueError("variance/covariance weights must be non-negative")
        self.boundary_margin = float(boundary_margin)
        self.background_weight = float(background_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.boundary_schedule = boundary_schedule or ProgressiveLambda(
            lambda_max=1.0, ramp_steps=2_000, warmup_steps=500
        )

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def background(
        self,
        clean: torch.Tensor,
        corrupt: torch.Tensor,
        valid: torch.Tensor,
        corruption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between views outside the mask; zero (with grad) when empty."""
        background = valid & ~corruption_mask
        if not bool(background.any()):
            return self._zero_like(clean)
        diff = (clean - corrupt)[background]
        return diff.pow(2).mean()

    def boundary(
        self,
        clean_energy: torch.Tensor,
        corrupt_energy: torch.Tensor,
        valid: torch.Tensor,
        corruption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Relative margin: corrupted energy must exceed paired clean energy."""
        region = valid & corruption_mask
        if not bool(region.any()):
            return self._zero_like(clean_energy)
        margin = self.boundary_margin + clean_energy[region] - corrupt_energy[region]
        return torch.relu(margin).mean()

    @staticmethod
    def ordering(energies: list[torch.Tensor], deltas: list[float]) -> torch.Tensor:
        """Enforce monotone energy growth across increasing severities."""
        if len(energies) < 2 or len(deltas) != len(energies) - 1:
            raise ValueError("need N energies and N-1 deltas")
        terms = [torch.relu(float(d) + a.mean() - b.mean()) for a, b, d in zip(energies, energies[1:], deltas)]
        return torch.stack(terms).mean()

    def forward(
        self,
        clean_latents: torch.Tensor,
        corrupt_latents: torch.Tensor,
        clean_energy: torch.Tensor,
        corrupt_energy: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        corruption_mask: torch.Tensor,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Combine normal, background, and ramped boundary terms."""
        if clean_latents.shape != corrupt_latents.shape or clean_latents.ndim != 3:
            raise ValueError("clean/corrupt latents must share [B, N, D]")
        if clean_energy.shape != corrupt_energy.shape or clean_energy.ndim != 2:
            raise ValueError("clean/corrupt energies must share [B, N]")
        normal = self.variance_weight * variance_loss(
            clean_latents, patch_valid_mask
        ) + self.covariance_weight * covariance_loss(clean_latents, patch_valid_mask)
        background = self.background(clean_latents, corrupt_latents, patch_valid_mask, corruption_mask)
        boundary = self.boundary(clean_energy, corrupt_energy, patch_valid_mask, corruption_mask)
        alpha = float(self.boundary_schedule.lambda_at(step))
        loss = normal + alpha * boundary + self.background_weight * background
        return {
            "loss": loss,
            "normal_loss": normal,
            "background_loss": background,
            "boundary_loss": boundary,
            "alpha": clean_latents.new_zeros(()).fill_(alpha),
        }
