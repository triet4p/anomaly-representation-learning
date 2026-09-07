"""Batch B1 Finding 1: clean-density objective + context/population energy contract.

Proves the declared methodology §4.1 objective
``L_normal = L_conditional-density + λv·L_variance + λc·L_covariance`` is the
optimized loss, that signed/invalid NLL semantics hold, and that the
boundary-trained context energy is a stable field distinct from hierarchical
population energy. Selection wiring and inference exposure belong to Batch B2.
"""

from __future__ import annotations

import pytest
import torch

from representation.criterion import ProgressiveLambda
from representation.v2_contracts import (
    CONTEXT_ENERGY_FIELD,
    MONITORING_ENERGY_FIELD,
    POPULATION_ENERGY_FIELD,
    validate_context_patch_output,
    validate_population_patch_output,
)
from representation.v2_geometry import HierarchicalMahalanobisGeometry
from representation.v2_objectives import CounterfactualCriterion
from representation.v2_patch import ContextConditionedPatchEncoder


def _criterion(**kwargs) -> CounterfactualCriterion:
    params = {
        "boundary_schedule": ProgressiveLambda(lambda_max=1.0, ramp_steps=0, warmup_steps=0),
    }
    params.update(kwargs)
    return CounterfactualCriterion(**params)


def test_clean_density_term_contributes_gradients_to_context_energy() -> None:
    """Clean signed NLL must enter L_normal with a live gradient path."""
    crit = _criterion()
    clean_e = torch.tensor([[0.5, -1.5, 2.0]], requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    density = CounterfactualCriterion.clean_density(clean_e, valid)
    assert float(density.detach()) == pytest.approx(float(clean_e.detach().mean()))
    density.backward()
    assert clean_e.grad is not None
    torch.testing.assert_close(
        clean_e.grad, torch.full_like(clean_e, 1.0 / 3.0)
    )

    # End-to-end: shifting every valid clean context energy shifts normal/loss.
    torch.manual_seed(0)
    clean = torch.randn(1, 3, 4)
    corrupt = clean + 0.25
    mask = torch.tensor([[True, False, False]])
    base = crit(clean, corrupt, clean_e.detach(), clean_e.detach() + 2.0, valid, mask)
    shifted = crit(
        clean, corrupt, clean_e.detach() + 1.0, clean_e.detach() + 3.0, valid, mask
    )
    assert float(shifted["normal_loss"].detach()) == pytest.approx(
        float(base["normal_loss"].detach()) + 1.0
    )
    assert float(shifted["density_raw"].detach()) == pytest.approx(
        float(base["density_raw"].detach()) + 1.0
    )


def test_density_excludes_invalid_and_preserves_signed_negative_nll() -> None:
    """Invalid patches never enter the density term; negative NLL lowers it."""
    crit = _criterion()
    valid = torch.tensor([[True, True, False]])
    clean_e = torch.tensor([[-2.0, -1.0, 1e6]])
    corrupt_e = torch.tensor([[1.0, 1.0, 1e6]])
    assert float(CounterfactualCriterion.clean_density(clean_e, valid)) == pytest.approx(-1.5)
    # Poisoning the invalid slot changes nothing.
    poisoned = clean_e.clone()
    poisoned[0, 2] = -1e6
    assert float(CounterfactualCriterion.clean_density(poisoned, valid)) == pytest.approx(-1.5)
    # Negative valid NLL strictly lowers the normal loss versus zeros.
    torch.manual_seed(1)
    clean = torch.randn(1, 3, 4)
    corrupt = clean + 0.5
    mask = torch.tensor([[True, False, False]])
    neg = crit(clean, corrupt, clean_e, corrupt_e, valid, mask)
    zero = crit(
        clean, corrupt, torch.zeros_like(clean_e), corrupt_e, valid, mask
    )
    assert float(neg["normal_loss"].detach()) < float(zero["normal_loss"].detach())
    # Empty valid set returns a differentiable zero instead of NaN.
    empty_valid = torch.zeros(1, 3, dtype=torch.bool)
    empty = CounterfactualCriterion.clean_density(
        torch.zeros(1, 3, requires_grad=True), empty_valid
    )
    assert float(empty.detach()) == pytest.approx(0.0)
    empty.backward()
    # Nonfinite valid density fails fast rather than silently training.
    with pytest.raises(ValueError, match="finite"):
        CounterfactualCriterion.clean_density(
            torch.tensor([[0.0, float("inf"), 0.0]]), torch.ones(1, 3, dtype=torch.bool)
        )


def test_encoder_context_energy_has_stable_monitoring_identity() -> None:
    """The encoder exposes the boundary-trained score under a fixed name."""
    assert MONITORING_ENERGY_FIELD == CONTEXT_ENERGY_FIELD
    enc = ContextConditionedPatchEncoder(
        n_channels=2, d_model=8, n_robots=2, n_programs=2, n_regimes=2,
        n_prototypes=2, sequence_layers=1, attention_heads=2, dropout=0.0,
    ).eval()
    torch.manual_seed(0)
    patches = torch.randn(1, 3, 2, 8)
    pad = torch.zeros(1, 3, 8, dtype=torch.bool)
    valid = torch.tensor([[True, True, False]])
    pad[0, 2, :] = True
    with torch.no_grad():
        out = enc(
            patches, pad, valid, torch.tensor([0]), torch.tensor([1]),
            torch.tensor([[0, 1, 0]]),
        )
    assert CONTEXT_ENERGY_FIELD in out
    assert POPULATION_ENERGY_FIELD not in out
    assert set(out) == {
        "patch_latents", CONTEXT_ENERGY_FIELD, "patch_valid_mask",
        "cond_mean", "cond_logvar", "prototype_logits",
    }
    validate_context_patch_output(out)
    # Conflation guard: a context output carrying population energy is rejected.
    with pytest.raises(ValueError, match="distinct signals"):
        validate_context_patch_output({**out, POPULATION_ENERGY_FIELD: out["context_energy"]})


def test_geometry_population_energy_is_separately_named_and_frozen() -> None:
    """Population energy keeps its own name; refs stay detached, queries flow."""
    torch.manual_seed(0)
    dim = 4
    geo = HierarchicalMahalanobisGeometry(
        d_model=dim, diag_min_samples=16, min_group_samples=4
    )
    block = torch.randn(40, dim) + 3.0
    ids = torch.zeros(40, dtype=torch.long)
    geo.fit(block, ids, ids, ids, torch.ones(40, dtype=torch.bool))
    frozen = geo.frozen()
    query = torch.randn(1, 2, dim, requires_grad=True)
    valid = torch.tensor([[True, False]])
    for scoring in (frozen.population_energy, frozen.mixture_energy):
        out = scoring(
            query, valid, torch.tensor([0]), torch.tensor([0]), torch.tensor([[0, 0]])
        )
        assert POPULATION_ENERGY_FIELD in out
        assert CONTEXT_ENERGY_FIELD not in out
        assert set(out) <= {
            POPULATION_ENERGY_FIELD, "group_confidence", "fallback_level",
        }
        assert float(out["population_energy"][0, 1].detach()) == pytest.approx(0.0)
        validate_population_patch_output(
            {"population_energy": out["population_energy"], "patch_valid_mask": valid}
        )
    scored = frozen.mixture_energy(
        query, valid, torch.tensor([0]), torch.tensor([0]), torch.tensor([[0, 0]])
    )
    scored["population_energy"].sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert bool((query.grad[valid] != 0).any())
    assert bool((query.grad[~valid] == 0).all())
    with pytest.raises(ValueError, match="distinct signals"):
        validate_population_patch_output(
            {
                "population_energy": scored["population_energy"],
                "context_energy": scored["population_energy"],
                "patch_valid_mask": valid,
            }
        )


def test_boundary_ordering_consumes_exact_context_energy_field() -> None:
    """Clean/corrupt ordering follows the passed context energies — no other field."""
    crit = _criterion()
    torch.manual_seed(2)
    clean = torch.randn(1, 3, 4)
    corrupt = clean + 1.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    mask = torch.tensor([[True, True, False]])
    clean_ctx = torch.tensor([[0.2, 0.3, 9.0]])
    corrupt_ctx = torch.tensor([[5.0, 6.0, 9.0]])
    separated = crit(clean, corrupt, clean_ctx, corrupt_ctx, valid, mask)
    assert float(separated["boundary_loss"].detach()) == pytest.approx(0.0)
    inverted = crit(clean, corrupt, corrupt_ctx, clean_ctx, valid, mask)
    assert float(inverted["boundary_loss"].detach()) > 0.0
    # Population energy is a different signal on the same latents: it must not
    # equal the context field, and boundary ordering provably follows
    # whichever field is passed — transfer is not assumed. Batch B2 must
    # thread the exact context-energy field through training and inference.
    torch.manual_seed(3)
    dim = 4
    geo = HierarchicalMahalanobisGeometry(
        d_model=dim, diag_min_samples=8, min_group_samples=2
    )
    fit_rows = torch.randn(24, dim)
    ids = torch.zeros(24, dtype=torch.long)
    geo.fit(fit_rows, ids, ids, ids, torch.ones(24, dtype=torch.bool))
    latents = torch.randn(1, 3, dim)
    pop_clean = geo.mixture_energy(
        latents, valid, torch.tensor([0]), torch.tensor([0]),
        torch.tensor([[0, 0, 0]]),
    )["population_energy"]
    pop_corrupt = geo.mixture_energy(
        latents + 2.0, valid, torch.tensor([0]), torch.tensor([0]),
        torch.tensor([[0, 0, 0]]),
    )["population_energy"]
    assert not torch.equal(pop_clean.detach(), clean_ctx)
    pop_forward = float(crit.boundary(pop_clean, pop_corrupt, valid, mask).detach())
    pop_swapped = float(crit.boundary(pop_corrupt, pop_clean, valid, mask).detach())
    assert pop_forward != pytest.approx(pop_swapped)
    via_context = float(crit.boundary(clean_ctx, corrupt_ctx, valid, mask).detach())
    assert via_context == pytest.approx(float(separated["boundary_loss"].detach()))
    # No alternative energy naming exists: every forward parameter is
    # mandatory and the energy slots carry only canonical names.
    import inspect as _inspect

    params = _inspect.signature(CounterfactualCriterion.forward).parameters
    assert list(params) == [
        "self", "clean_latents", "corrupt_latents",
        "clean_context_energy", "corrupt_context_energy",
        "patch_valid_mask", "corruption_mask", "step",
    ]
    assert all(
        p.default is _inspect.Parameter.empty
        for name, p in params.items() if name != "step"
    )
