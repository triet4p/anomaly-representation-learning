"""Task 12: localized synthetic counterfactual objectives."""

from __future__ import annotations

import pytest
import torch

from representation.criterion import ProgressiveLambda
from representation.v2_objectives import (
    CounterfactualCriterion,
    synthesize_corrupted_patches,
)


def _tensors():
    torch.manual_seed(0)
    patches = torch.randn(2, 4, 3, 8)
    pad = torch.zeros(2, 4, 8, dtype=torch.bool)
    valid = torch.ones(2, 4, dtype=torch.bool)
    valid[0, 3] = False
    return patches, pad, valid


def test_corruption_preserves_context_and_background() -> None:
    patches, pad, valid = _tensors()
    mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
    corrupted = synthesize_corrupted_patches(
        patches, pad, valid, mask, severity=2.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert tuple(corrupted.shape) == tuple(patches.shape)
    # Unmasked patches are bit-identical: unit/robot/program/regime context kept.
    assert torch.equal(corrupted[0, 1], patches[0, 1])
    assert torch.equal(corrupted[1, 0], patches[1, 0])
    assert not torch.equal(corrupted[0, 0], patches[0, 0])
    # Deterministic under a fixed generator seed.
    repeat = synthesize_corrupted_patches(
        patches, pad, valid, mask, severity=2.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(corrupted, repeat)


def test_background_boundary_empty_mask_and_held_out_region() -> None:
    crit = CounterfactualCriterion(
        boundary_schedule=ProgressiveLambda(lambda_max=1.0, ramp_steps=0, warmup_steps=0)
    )
    clean = torch.randn(1, 4, 4, requires_grad=True)
    corrupt = (clean.detach() + 0.0).requires_grad_(True)
    corrupt_clone = corrupt.clone()
    corrupt_clone[0, 1] += 5.0
    corrupt = corrupt_clone
    valid = torch.ones(1, 4, dtype=torch.bool)
    seen = torch.tensor([[False, True, False, False]])
    clean_e = torch.tensor([[0.1, 0.1, 0.1, 0.1]], requires_grad=True)
    corrupt_e = (torch.tensor([[0.1, 4.0, 0.1, 0.1]]) + clean_e * 0.0).requires_grad_(True)
    assert float(crit.background(clean, corrupt, valid, seen).detach()) == pytest.approx(0.0)
    assert float(crit.boundary(clean_e, corrupt_e, valid, seen).detach()) == pytest.approx(0.0)
    # Held-out region (never trained on position 2) still separates relatively.
    held_out = torch.tensor([[False, False, True, False]])
    corrupt_e_held = torch.tensor([[0.1, 0.1, 3.0, 0.1]])
    assert float(crit.boundary(clean_e, corrupt_e_held, valid, held_out).detach()) == pytest.approx(0.0)
    assert float(crit.boundary(corrupt_e_held, clean_e, valid, held_out).detach()) > 0.0
    # Empty corruption mask leaves boundary empty (differentiable zero);
    # full corruption mask leaves background empty (differentiable zero).
    empty = torch.zeros(1, 4, dtype=torch.bool)
    full = torch.ones(1, 4, dtype=torch.bool)
    boundary_empty = crit.boundary(clean_e, corrupt_e, valid, empty)
    background_empty = crit.background(clean, corrupt, valid, full)
    for value in (boundary_empty, background_empty):
        assert torch.isfinite(value).all() and float(value.detach()) == pytest.approx(0.0)
        value.backward(retain_graph=True)
    sched = ProgressiveLambda(lambda_max=2.0, ramp_steps=10, warmup_steps=2)
    assert sched.lambda_at(0) == pytest.approx(0.0)
    assert sched.lambda_at(12) == pytest.approx(2.0)
    crit = CounterfactualCriterion(boundary_schedule=sched)
    ordered = [torch.full((2,), fill_value=v) for v in (0.5, 1.5, 3.0)]
    assert float(CounterfactualCriterion.ordering(ordered, [0.5, 0.5])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="deltas"):
        CounterfactualCriterion.ordering(ordered, [0.5])
    reversed_e = list(reversed(ordered))
    assert float(CounterfactualCriterion.ordering(reversed_e, [0.1, 0.1])) > 0.0

    torch.manual_seed(0)
    clean = torch.randn(2, 3, 4, requires_grad=True)
    corrupt = (clean + 1.0).detach().requires_grad_(True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    clean_e = (clean.pow(2).sum(dim=-1).detach())
    corrupt_e = clean_e + 2.0
    out = crit(clean, corrupt, clean_e, corrupt_e, valid, mask, step=12)
    assert float(out["alpha"].detach()) == pytest.approx(2.0)
    out["loss"].backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
    # Early ramp keeps the boundary term out while normal control remains.
    early = crit(clean.detach(), corrupt.detach(), clean_e, corrupt_e, valid, mask, step=0)
    assert float(early["alpha"].detach()) == pytest.approx(0.0)
    assert float(early["loss"].detach()) >= 0.0


def test_corruption_follows_input_device() -> None:
    """Synthetic direction must live on the input device (CUDA contract)."""
    patches, pad, valid = _tensors()
    mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
    device = torch.device("meta")
    corrupted = synthesize_corrupted_patches(
        patches.to(device),
        pad.to(device),
        valid.to(device),
        mask.to(device),
        severity=2.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert corrupted.device == device
    assert tuple(corrupted.shape) == tuple(patches.shape)
    assert corrupted.dtype == patches.dtype


def test_forward_reports_raw_weighted_decomposition() -> None:
    """Task 28 + Finding 1: L_normal includes clean conditional density."""
    torch.manual_seed(3)
    crit = CounterfactualCriterion(
        background_weight=0.5,
        variance_weight=2.0,
        covariance_weight=0.25,
        boundary_schedule=ProgressiveLambda(lambda_max=1.0, ramp_steps=0, warmup_steps=0),
    )
    clean = torch.randn(2, 3, 4, requires_grad=True)
    corrupt = (clean.detach() + 0.5).requires_grad_(True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    clean_e = clean.detach().pow(2).sum(dim=-1)
    corrupt_e = corrupt.detach().pow(2).sum(dim=-1)
    out = crit(clean, corrupt, clean_e, corrupt_e, valid, mask, step=7)
    for key in ("loss", "normal_loss", "density_raw", "density_weighted",
                "variance_raw", "covariance_raw",
                "background_loss", "boundary_loss", "alpha"):
        assert torch.isfinite(out[key]).all(), f"non-finite {key}"
    # Methodology §4.1: L_normal = L_conditional-density + λv·L_var + λc·L_cov.
    assert float(out["density_raw"].detach()) == pytest.approx(float(clean_e.mean().detach()))
    assert float(out["density_weighted"].detach()) == pytest.approx(float(out["density_raw"].detach()))
    expected = (
        out["density_weighted"]
        + 2.0 * out["variance_raw"]
        + 0.25 * out["covariance_raw"]
        + float(out["alpha"].detach()) * out["boundary_loss"]
        + 0.5 * out["background_loss"]
    )
    assert float(out["loss"].detach()) == pytest.approx(float(expected.detach()))
    assert float(out["normal_loss"].detach()) == pytest.approx(
        float((out["density_weighted"] + 2.0 * out["variance_raw"] + 0.25 * out["covariance_raw"]).detach())
    )
    # Omitting the density term would understate the normal loss whenever the
    # clean conditional NLL is nonzero.
    assert float(out["normal_loss"].detach()) != pytest.approx(
        float((2.0 * out["variance_raw"] + 0.25 * out["covariance_raw"]).detach())
    )
    out["loss"].backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
