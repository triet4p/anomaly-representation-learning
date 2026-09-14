"""Focused Task 11 guards: C3 parameter matching, encoder contract,
mask/position behavior, model isolation, and registry parity."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from representation import sprint17_ablation as A
from representation.layers.patch_encoder import LocalPatchEncoder
from representation.sprint17_c3 import (
    ARM_ADAPTER_DESCRIPTION,
    ARM_LOCAL_ENCODER_PARAMS,
    B0_LOCAL_ENCODER_PARAMS,
    WithinPatchTransformerEncoder,
    arm_local_encoder,
    count_local_encoder_params,
)

B0_PARAMS_TOTAL = 1821698


def _inputs(b=2, n=3, c=6, w=32, seed=3, pad_tail=0):
    rng = np.random.default_rng(seed)
    patches = torch.from_numpy(rng.standard_normal((b, n, c, w)).astype(np.float32))
    mask = torch.zeros((b, n, w), dtype=torch.bool)
    if pad_tail:
        mask[:, :, -pad_tail:] = True
    return patches, mask


def test_b0_local_encoder_count_is_frozen_reference():
    assert count_local_encoder_params(LocalPatchEncoder(6, 128, 0.1)) == B0_LOCAL_ENCODER_PARAMS


def test_arm_counts_match_and_envelope_passes():
    for arm, want in ARM_LOCAL_ENCODER_PARAMS.items():
        got = count_local_encoder_params(arm_local_encoder(arm))
        assert got == want, arm
        assert abs(got - B0_LOCAL_ENCODER_PARAMS) / B0_PARAMS_TOTAL <= 0.05, arm


def test_encoder_contract_shapes_finite():
    for arm in ("C3-A", "C3-B"):
        enc = arm_local_encoder(arm).eval()
        patches, mask = _inputs()
        with torch.no_grad():
            out = enc(patches, mask)
        assert tuple(out.shape) == (2, 3, 128)
        assert bool(torch.isfinite(out).all())


def test_padded_values_cannot_change_valid_embeddings():
    for arm in ("C3-A", "C3-B"):
        enc = arm_local_encoder(arm).eval()
        patches, mask = _inputs(pad_tail=7)
        dirty = patches.clone()
        dirty[mask.unsqueeze(2).expand_as(patches)] = 999.0
        with torch.no_grad():
            assert torch.equal(enc(dirty, mask), enc(patches, mask))


def test_fully_padded_patch_has_no_usable_embedding():
    for arm in ("C3-A", "C3-B"):
        enc = arm_local_encoder(arm).eval()
        patches, mask = _inputs()
        mask[:, 1, :] = True
        with torch.no_grad():
            out = enc(patches, mask)
        assert torch.equal(out[:, 1, :], torch.zeros_like(out[:, 1, :]))


def test_c3b_explicit_position_makes_location_matter():
    enc = arm_local_encoder("C3-B").eval()
    patches, mask = _inputs()
    shifted = torch.roll(patches, shifts=5, dims=-1)
    with torch.no_grad():
        assert not torch.equal(enc(shifted, mask), enc(patches, mask))


def test_c3b_rejects_non_native_width():
    enc = arm_local_encoder("C3-B").eval()
    patches, mask = _inputs(w=48)
    with pytest.raises(ValueError):
        enc(patches, mask)


def test_arms_differ_from_b0_on_fixed_input():
    b0 = LocalPatchEncoder(6, 128, 0.1).eval()
    patches, mask = _inputs()
    with torch.no_grad():
        ref = b0(patches, mask)
        for arm in ("C3-A", "C3-B"):
            enc = arm_local_encoder(arm).eval()
            assert not torch.equal(enc(patches, mask), ref), arm


def test_input_validation_rejects_malformed_batches():
    enc = arm_local_encoder("C3-A").eval()
    patches, mask = _inputs()
    with pytest.raises(ValueError):
        enc(patches.reshape(2, 3 * 6, 32), mask)
    with pytest.raises(ValueError):
        enc(patches[:, :, :5, :], mask)
    with pytest.raises(ValueError):
        enc(patches, torch.zeros((2, 3, 32)))
    with pytest.raises(ValueError):
        arm_local_encoder("C3-C")


def test_model_uses_injected_encoder_and_defaults_to_b0():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=128,
                   sequence_layers=1, attention_heads=2,
                   use_conditional_norm=False, seed=5)
    default = V1RepresentationModel(cfg)
    assert isinstance(default.patch_encoder, LocalPatchEncoder)
    injected = V1RepresentationModel(
        cfg, patch_encoder=arm_local_encoder("C3-A"))
    assert injected.patch_encoder is not default.patch_encoder
    assert count_local_encoder_params(injected.patch_encoder) == 34688


def test_registry_and_parity_declarations():
    from representation.sprint17_ablation import TrainingParity
    for arm in ("C3-A", "C3-B"):
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C3",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300


def test_flop_counter_reproduces_b0_and_passes_arm_envelope():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import sprint17_task11_c3 as T11
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=128,
                   sequence_layers=4, attention_heads=4, dropout=0.1,
                   n_robots=9, n_programs=8, seed=171701)
    b0 = V1RepresentationModel(cfg)
    # Groups-aware counting leaves every non-grouped module untouched.
    assert T11.count_flops_reference(b0) == T11.B0_FLOPS_REFERENCE == 182016709
    for arm in ("C3-A", "C3-B"):
        model, _, _ = T11.build_model(T11.arm_config_dict(arm, 171701),
                                      torch.device("cpu"))
        params = T11.count_parameters(model)
        flops = T11.count_flops_reference(model)
        assert abs(params - T11.B0_PARAMS) / T11.B0_PARAMS <= 0.05, arm
        assert abs(flops - T11.B0_FLOPS_REFERENCE) / T11.B0_FLOPS_REFERENCE <= 0.10, arm
