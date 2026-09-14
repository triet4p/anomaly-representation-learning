"""Focused Task 13 guards: C6 pooling definitions, valid-mask semantics,
parameter identity, model isolation, and registry parity."""

import sys

import numpy as np
import pytest
import torch
from pathlib import Path

from representation.sprint17_c6 import (
    ARM_ADAPTER_DESCRIPTION,
    ARM_POOLING_PARAMS,
    C6_ARMS,
    C6B_GATE_DIM,
    C6ArmModel,
    GatedAttentionPooling,
    MeanStdProjectionPooling,
    arm_pooling,
    count_pooling_params,
)

B0_PARAMS_TOTAL = 1821698
B0_FLOPS_REFERENCE = 182016709


def _latents(b=2, n=4, d=8, seed=11):
    rng = np.random.default_rng(seed)
    lat = torch.from_numpy(rng.standard_normal((b, n, d)).astype(np.float32))
    valid = torch.ones((b, n), dtype=torch.bool)
    valid[1, 2:] = False
    return lat, valid


def test_b0_mean_pooling_parity_is_fixed_reference():
    """Pin the B0 valid-patch mean contract both C6 arms replace."""
    from representation.model import V1RepresentationModel
    lat, valid = _latents()
    got = V1RepresentationModel._pool_file(lat, valid)
    v = valid.to(lat.dtype).unsqueeze(-1)
    want = (lat * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
    assert torch.equal(got, want)
    assert torch.equal(got[1], lat[1, :2].mean(dim=0))


def test_arm_param_identity_and_envelope():
    for arm, want in ARM_POOLING_PARAMS.items():
        got = count_pooling_params(arm_pooling(arm, 128))
        assert got == want, arm
        assert abs(got) / B0_PARAMS_TOTAL <= 0.05, arm
    assert ARM_POOLING_PARAMS == {"C6-A": 32896, "C6-B": 8289}
    assert C6B_GATE_DIM == 32


def test_c6a_matches_hand_mean_and_std():
    pool = MeanStdProjectionPooling(4).eval()
    with torch.no_grad():
        pool.project.weight.copy_(torch.cat(
            [torch.eye(4), torch.zeros(4, 4)], dim=1))
        pool.project.bias.zero_()
        lat = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                             [3.0, 4.0, 5.0, 6.0],
                             [99.0, 99.0, 99.0, 99.0]]])
        valid = torch.tensor([[True, True, False]])
        out = pool(lat, valid)
    assert torch.allclose(out[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    with torch.no_grad():
        pool.project.weight.copy_(torch.cat(
            [torch.zeros(4, 4), torch.eye(4)], dim=1))
        out = pool(lat, valid)
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_c6a_single_valid_patch_has_zero_std():
    pool = MeanStdProjectionPooling(4).eval()
    with torch.no_grad():
        pool.project.weight.copy_(torch.cat(
            [torch.eye(4), torch.eye(4)], dim=1))
        pool.project.bias.zero_()
        lat = torch.tensor([[[5.0, 6.0, 7.0, 8.0],
                             [9.0, 9.0, 9.0, 9.0]]])
        out = pool(lat, torch.tensor([[True, False]]))
    assert torch.allclose(out[0], torch.tensor([5.0, 6.0, 7.0, 8.0]))


def test_c6b_weights_sum_to_one_over_valid_only():
    pool = GatedAttentionPooling(8, 4).eval()
    lat, valid = _latents(d=8)
    with torch.no_grad():
        weights = pool.attention_weights(lat, valid)
        out = pool(lat, valid)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.equal(weights[1, 2:], torch.zeros(2))
    assert bool((weights[0] > 0).all())
    assert bool(torch.isfinite(out).all())


def test_c6b_uniform_input_gives_uniform_weights():
    pool = GatedAttentionPooling(6, 4).eval()
    lat = torch.ones(1, 3, 6)
    valid = torch.tensor([[True, True, False]])
    with torch.no_grad():
        weights = pool.attention_weights(lat, valid)
        out = pool(lat, valid)
    assert torch.allclose(weights[0, :2], torch.tensor([0.5, 0.5]), atol=1e-5)
    assert weights[0, 2].item() == 0.0
    assert torch.allclose(out[0], torch.ones(6), atol=1e-5)


def test_invalid_values_cannot_change_either_arm():
    lat, valid = _latents()
    dirty = lat.clone()
    dirty[~valid.unsqueeze(-1).expand_as(lat)] = 17.0
    nasty = lat.clone()
    nasty[1, 2] = float("inf")
    for arm in C6_ARMS:
        pool = arm_pooling(arm, 8).eval()
        with torch.no_grad():
            assert torch.equal(pool(dirty, valid), pool(lat, valid)), arm
            assert bool(torch.isfinite(pool(nasty, valid)).all()), arm


def test_empty_valid_row_stays_finite():
    lat, _ = _latents()
    valid = torch.zeros((2, 4), dtype=torch.bool)
    for arm in C6_ARMS:
        pool = arm_pooling(arm, 8).eval()
        with torch.no_grad():
            out = pool(lat, valid)
        assert out.shape == (2, 8)
        assert bool(torch.isfinite(out).all()), arm
    pool_b = arm_pooling("C6-B", 8).eval()
    with torch.no_grad():
        assert torch.equal(pool_b(lat, valid), torch.zeros(2, 8))


def test_input_validation_rejects_malformed_inputs():
    lat, valid = _latents()
    for arm in C6_ARMS:
        pool = arm_pooling(arm, 8).eval()
        with pytest.raises(ValueError):
            pool(lat.reshape(2, 32), valid)
        with pytest.raises(ValueError):
            pool(lat, valid.to(torch.float32))
        with pytest.raises(ValueError):
            pool(lat[:, :2], valid)
    with pytest.raises(ValueError):
        arm_pooling("C6-C")
    with pytest.raises(ValueError):
        MeanStdProjectionPooling(0)
    with pytest.raises(ValueError):
        GatedAttentionPooling(8, 0)


def test_model_overrides_only_pooling_and_defaults_hold_b0():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=128,
                   sequence_layers=1, attention_heads=2,
                   use_conditional_norm=False, seed=5)
    default = V1RepresentationModel(cfg)
    assert type(default)._pool_file is V1RepresentationModel._pool_file
    for arm in C6_ARMS:
        torch.manual_seed(171701)
        model = C6ArmModel(cfg, arm_id=arm)
        assert model.c6_arm_id == arm
        assert count_pooling_params(model.c6_pooling) == ARM_POOLING_PARAMS[arm]
        lat = torch.randn(2, 5, 128)
        valid = torch.ones(2, 5, dtype=torch.bool)
        valid[1, 4] = False
        with torch.no_grad():
            assert torch.equal(model._pool_file(lat, valid),
                               model.c6_pooling(lat, valid))
            assert not torch.equal(
                model._pool_file(lat, valid),
                V1RepresentationModel._pool_file(lat, valid)), arm
    with pytest.raises(ValueError):
        C6ArmModel(cfg, arm_id="C6-C")
    model = C6ArmModel(cfg, arm_id="C6-A")
    with pytest.raises(ValueError):
        model._pool_file(torch.randn(1, 2, 128),
                         torch.ones(1, 2, dtype=torch.bool),
                         weights=torch.ones(1, 2))


def test_registry_and_parity_declarations():
    from representation import sprint17_ablation as A
    from representation.sprint17_ablation import TrainingParity
    for arm in C6_ARMS:
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C6",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300


def test_flop_counter_reproduces_b0_and_passes_arm_envelope():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "experiments"))
    import sprint17_task13_c6 as T13
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=128,
                   sequence_layers=4, attention_heads=4, dropout=0.1,
                   n_robots=9, n_programs=8, seed=171701)
    b0 = V1RepresentationModel(cfg)
    assert T13.count_flops_reference(b0) == T13.B0_FLOPS_REFERENCE == 182016709
    for arm in C6_ARMS:
        model, _, _ = T13.build_model(T13.arm_config_dict(arm, 171701),
                                      torch.device("cpu"))
        params = T13.count_parameters(model)
        flops = T13.count_flops_reference(model)
        assert abs(params - T13.B0_PARAMS) / T13.B0_PARAMS <= 0.05, arm
        assert abs(flops - T13.B0_FLOPS_REFERENCE) / T13.B0_FLOPS_REFERENCE <= 0.10, arm
