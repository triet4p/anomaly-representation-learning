"""Focused Task 15 guards: C8 scorer definitions, mask semantics,
non-query-conditioned behavior, Fit-only standardization, and registry parity."""

import inspect
import sys

import pytest
import torch
from pathlib import Path

from representation.sprint17_c8 import (
    ARM_ADAPTER_DESCRIPTION,
    C8_ARMS,
    HUBER_DELTA,
    MAD_FLOOR,
    NORM_FLOOR,
    HuberStandardizer,
    arm_patch_energies,
    b0_mse_energy,
    c8a_patch_energies,
    c8b_patch_energies,
    file_mean,
    huber_energy,
)

B0_PARAMS_TOTAL = 1821698
B0_FLOPS_REFERENCE = 182016709


def _latents(seed=7):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(2, 4, 6, generator=g)
    targ = torch.randn(2, 4, 6, generator=g)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1, 2:] = False
    return pred, targ, mask


def test_b0_mse_formula_is_fixed_reference():
    """Pin the B0 S_pred energy both C8 arms replace (score_batch formula)."""
    import torch.nn.functional as F
    pred, targ, mask = _latents()
    want = F.mse_loss(pred, targ, reduction="none").mean(dim=-1)
    want = want.masked_fill(~mask, 0.0)
    assert torch.equal(b0_mse_energy(pred, targ, mask), want)
    counts = mask.sum(dim=1).to(want.dtype)
    assert torch.equal(file_mean(want, mask), want.sum(dim=1) / counts.clamp_min(1.0))


def test_file_mean_empty_mask_scores_zero():
    energies = torch.ones(2, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    assert torch.equal(file_mean(energies, mask), torch.zeros(2))
    with pytest.raises(ValueError):
        file_mean(energies, torch.ones(2, 4, dtype=torch.bool))


def test_huber_energy_matches_hand_values():
    z = torch.tensor([0.0, 0.5, -0.5, 1.0, 2.0, -3.0], dtype=torch.float64)
    got = huber_energy(z)
    assert torch.allclose(got, torch.tensor([0.0, 0.125, 0.125, 0.5, 1.5, 2.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        huber_energy(z, delta=0.0)
    assert HUBER_DELTA == 1.0


def test_standardizer_fits_median_mad_and_guards():
    std = HuberStandardizer()
    with pytest.raises(ValueError):
        std.standardize(torch.zeros(2, 3))
    r = torch.tensor([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]])
    std.fit(r)
    assert torch.allclose(std.center.double(), torch.tensor([2.0, 12.0], dtype=torch.float64))
    assert torch.allclose(std.scale.double(), torch.tensor([2.0, 2.0], dtype=torch.float64))
    out = std.standardize(torch.tensor([[2.0, 14.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    with pytest.raises(ValueError):
        std.fit(r, labels=["normal", "normal", "abnormal"])
    with pytest.raises(ValueError):
        std.fit(torch.full((4, 2), float("nan")))
    with pytest.raises(ValueError):
        std.standardize(torch.zeros(1, 5))
    assert MAD_FLOOR == 1e-6
    assert std.provenance()["n_floats"] == 4


def test_c8a_identical_latents_score_zero_and_masks_hold():
    pred, targ, mask = _latents()
    std = HuberStandardizer().fit((pred - targ).reshape(-1, 6))
    energies = c8a_patch_energies(targ.clone(), targ, mask, std)
    assert torch.equal(energies[~mask], torch.zeros_like(energies[~mask]))
    assert bool(torch.isfinite(energies).all())
    assert bool((energies[mask] >= 0).all())
    far = c8a_patch_energies(targ + 50.0, targ, mask, std)
    assert bool((far[mask] > energies[mask]).all())
    with pytest.raises(ValueError):
        arm_patch_energies("C8-A", pred, targ, mask, standardizer=None)


def test_c8a_depends_only_on_residuals_not_query_level():
    """Joint translation of predicted+target leaves C8-A unchanged (no query conditioning)."""
    pred, targ, mask = _latents()
    std = HuberStandardizer().fit((pred - targ).reshape(-1, 6))
    base = c8a_patch_energies(pred, targ, mask, std)
    shifted = c8a_patch_energies(pred + 5.0, targ + 5.0, mask, std)
    assert torch.allclose(base, shifted, atol=1e-12)


def test_c8b_cosine_hand_values_and_guards():
    a = torch.tensor([[[1.0, 0.0]]])
    assert torch.allclose(c8b_patch_energies(a, a, torch.ones(1, 1, dtype=torch.bool)),
                          torch.zeros(1, 1, dtype=torch.float64))
    assert torch.allclose(c8b_patch_energies(a, -a, torch.ones(1, 1, dtype=torch.bool)),
                          torch.tensor([[2.0]], dtype=torch.float64))
    b = torch.tensor([[[0.0, 1.0]]])
    assert torch.allclose(c8b_patch_energies(a, b, torch.ones(1, 1, dtype=torch.bool)),
                          torch.tensor([[1.0]], dtype=torch.float64))
    zero = torch.zeros(1, 1, 2)
    out = c8b_patch_energies(zero, zero, torch.ones(1, 1, dtype=torch.bool))
    assert bool(torch.isfinite(out).all()) and 0.0 <= float(out[0, 0]) <= 2.0
    assert NORM_FLOOR == 1e-12


def test_c8b_scale_invariant_per_patch():
    """Joint scaling of predicted+target leaves C8-B unchanged (no query conditioning)."""
    pred, targ, mask = _latents()
    base = c8b_patch_energies(pred, targ, mask)
    assert torch.allclose(base, c8b_patch_energies(3.0 * pred, 3.0 * targ, mask),
                          atol=1e-12)


def test_latents_validation_rejects_malformed_inputs():
    pred, targ, mask = _latents()
    with pytest.raises(ValueError):
        b0_mse_energy(pred.reshape(2, 24), targ, mask)
    with pytest.raises(ValueError):
        c8b_patch_energies(pred, targ[:, :, :5], mask)
    with pytest.raises(ValueError):
        c8b_patch_energies(pred, targ, mask.to(torch.float32))
    with pytest.raises(ValueError):
        c8a_patch_energies(pred, targ, mask, HuberStandardizer())
    with pytest.raises(ValueError):
        arm_patch_energies("C8-C", pred, targ, mask)
    with pytest.raises(ValueError):
        c8b_patch_energies(torch.full((1, 1, 2), float("inf")), torch.zeros(1, 1, 2),
                           torch.ones(1, 1, dtype=torch.bool))


def test_c8_consumes_no_label_anomaly_or_query_inputs():
    """Structural isolation: scorers see latents + masks (+Fit stats for C8-A)."""
    import representation.sprint17_c8 as C8
    for name in ("c8a_patch_energies", "c8b_patch_energies", "b0_mse_energy"):
        params = set(inspect.signature(getattr(C8, name)).parameters)
        assert "labels" not in params and "query" not in params, (name, params)
    bodies = "\n".join(inspect.getsource(getattr(C8, name)) for name in
                      ("c8a_patch_energies", "c8b_patch_energies", "b0_mse_energy",
                       "file_mean", "huber_energy"))
    for word in ("anomaly", "severity", "query", "label"):
        assert word not in bodies, word


def test_registry_and_parity_declarations():
    from representation import sprint17_ablation as A
    from representation.sprint17_ablation import TrainingParity
    for arm in C8_ARMS:
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C8",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300


def test_driver_builds_unchanged_b0_graph_with_exact_counts():
    from representation.config import V1Config
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments"))
    import sprint17_task15_c8 as T15
    for arm in C8_ARMS:
        model, _, _ = T15.build_model(T15.arm_config_dict(arm, 171701),
                                      torch.device("cpu"))
        assert T15.count_parameters(model) == T15.B0_PARAMS == 1821698
        assert T15.count_flops_reference(model) == T15.B0_FLOPS_REFERENCE == 182016709
