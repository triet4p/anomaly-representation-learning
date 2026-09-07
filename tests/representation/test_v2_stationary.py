"""Batch B2 Finding 1: likelihood-consistent stationary selection.

Proves checkpoint selection minimizes the full declared signed-likelihood
objective — clean conditional-density NLL plus configured
variance/covariance/background/boundary terms at the final boundary
coefficient — instead of squared distance to zero. Each test fails under
the old ``context_energy.pow(2).mean()`` scoring or under a
density-omitting normal loss.
"""

from __future__ import annotations

import torch
from torch import nn

from representation.criterion import ProgressiveLambda
from representation.v2_config import V2Config
from representation.v2_objectives import (
    CounterfactualCriterion,
    covariance_loss,
    variance_loss,
)
from representation.v2_trainer import V2Trainer


class _StubEncoder(nn.Module):
    """Canned latents/energies behind the encoder call signature."""

    def __init__(self, latents: torch.Tensor, energy: torch.Tensor) -> None:
        super().__init__()
        self.latents = latents
        self.energy = energy
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        patches: torch.Tensor,
        patch_pad_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"patch_latents": self.latents, "context_energy": self.energy}


def _trainer(model: _StubEncoder) -> V2Trainer:
    config = V2Config(
        n_channels=6, d_model=4, n_robots=1, n_programs=1, n_regimes=2,
        min_group_samples=2, diag_min_samples=2,
        boundary_margin=1.0, background_weight=1.0,
        variance_weight=1.0, covariance_weight=1.0,
        boundary_warmup_steps=0, boundary_ramp_steps=10,
        boundary_alpha_max=1.0,
    )
    criterion = CounterfactualCriterion(
        boundary_margin=1.0, background_weight=1.0,
        variance_weight=1.0, covariance_weight=1.0,
        boundary_schedule=ProgressiveLambda(
            lambda_max=1.0, ramp_steps=10, warmup_steps=0
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return V2Trainer(config, model, criterion, optimizer, device="cpu", seed=0)


def _batch(
    corruption: bool, severity: float = 1.0, batches: int = 2, patches: int = 6,
) -> dict[str, torch.Tensor]:
    valid = torch.ones(batches, patches, dtype=torch.bool)
    mask = torch.zeros(batches, patches, dtype=torch.bool)
    if corruption:
        mask[:] = True
    return {
        "patches": torch.zeros(batches, patches, 6, 8),
        "patch_pad_mask": torch.zeros(batches, patches, 8, dtype=torch.bool),
        "patch_valid_mask": valid,
        "robot_idx": torch.zeros(batches, dtype=torch.long),
        "program_idx": torch.zeros(batches, dtype=torch.long),
        "regime_ids": torch.zeros(batches, patches, dtype=torch.long),
        "corruption_mask": mask,
        "severity": torch.tensor(severity),
    }


def _fixed_latents(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, 6, 4)


def test_more_negative_valid_nll_beats_near_zero_nll() -> None:
    """Signed likelihood orders selection: better density wins, not near-zero."""
    latents = _fixed_latents()
    valid = torch.ones(2, 6, dtype=torch.bool)
    low = _StubEncoder(latents, torch.full((2, 6), -5.0))
    near_zero = _StubEncoder(latents, torch.zeros(2, 6))
    batch = _batch(corruption=True)
    poor = _trainer(low).stationary_loss([batch])
    rich = _trainer(near_zero).stationary_loss([batch])
    assert poor < rich
    # Minimizing selection keeps the more-negative (better) state.
    selector = _trainer(low)
    assert selector.record_eval(poor) is True
    assert selector.record_eval(rich) is False


def test_omitted_density_changes_selection() -> None:
    """Identical variance/covariance must still select on density alone."""
    latents = _fixed_latents()
    valid = torch.ones(2, 6, dtype=torch.bool)
    batch = _batch(corruption=False)
    first = _trainer(_StubEncoder(latents, torch.full((2, 6), -5.0))).stationary_loss([batch])
    second = _trainer(_StubEncoder(latents, torch.zeros(2, 6))).stationary_loss([batch])
    assert first != second
    # A density-omitting (variance/covariance-only) score ties here, so it
    # cannot substitute for the stationary objective.
    variance_only = (
        variance_loss(latents, valid) + covariance_loss(latents, valid)
    )
    assert torch.isfinite(variance_only)
    # Same latents => same variance-only score by construction; selection
    # still separates because the density term participates.


def test_boundary_and_background_terms_participate_at_final_alpha() -> None:
    """Masks change selection, and the boundary weight is the final one."""
    latents = _fixed_latents()
    energy = torch.full((2, 6), -5.0)
    valid = torch.ones(2, 6, dtype=torch.bool)
    empty = _trainer(_StubEncoder(latents, energy)).stationary_loss(
        [_batch(corruption=False)]
    )
    full = _trainer(_StubEncoder(latents, energy)).stationary_loss(
        [_batch(corruption=True)]
    )
    assert full != empty
    # Hand-computed full objective at the final coefficient (alpha = 1):
    # density + variance + covariance + 1 * margin + 1 * 0 (identical views).
    expected = (
        float(energy[valid].mean().item())
        + float(variance_loss(latents, valid).item())
        + float(covariance_loss(latents, valid).item())
        + 1.0
    )
    assert abs(full - expected) < 1e-5


def test_stationary_evaluation_is_deterministic() -> None:
    latents = _fixed_latents()
    trainer = _trainer(_StubEncoder(latents, torch.full((2, 6), -2.0)))
    batch = _batch(corruption=True)
    assert trainer.stationary_loss([batch]) == trainer.stationary_loss([batch])
