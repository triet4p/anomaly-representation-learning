"""Focused Task 10 guards: C2-A lattice parity, C2-B unit-mass support,
localization preservation, weighted-pooling isolation, and registry parity."""

import numpy as np
import pytest
import torch

from representation import sprint17_ablation as A
from representation.data import collate_variable_files
from representation.sprint17_c2 import (
    ARM_ADAPTER_DESCRIPTION,
    C2A_GRIDS,
    C2B_GRIDS,
    Sprint17C2Patchifier,
    arm_grids,
    arm_patchifier,
    unit_mass_weights,
)
from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, SampleLabel

B0_PATCH = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
LENGTH_BATTERY = (1, 7, 16, 20, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65,
                  100, 511, 512, 513, 1000)


def _sample(timesteps: int, seed: int = 7) -> FileSample:
    rng = np.random.default_rng(seed)
    return FileSample(
        x=rng.standard_normal((6, timesteps)).astype(np.float32),
        file_id=f"C2-TEST-{timesteps}",
        file_label=SampleLabel.NORMAL,
        seed=seed,
        generator_version="task10-test",
        config_hash="none",
        regime_sequence=[],
        robot_idx=0,
        program_idx=0,
    )


def _b0_batch(timesteps: int):
    return B0_PATCH.patchify(_sample(timesteps))


def test_c2a_lattice_bitwise_matches_b0():
    c2a = Sprint17C2Patchifier(C2A_GRIDS)
    for t in LENGTH_BATTERY:
        got = c2a.patchify(_sample(t))
        want = _b0_batch(t)
        assert got.N == want.N, f"T={t}"
        np.testing.assert_array_equal(got.starts, want.starts)
        np.testing.assert_array_equal(got.valid_len, want.valid_len)
        np.testing.assert_array_equal(got.pad_mask, want.pad_mask)
        np.testing.assert_array_equal(got.patches, want.patches)


def test_single_grid_arm_emits_no_weights_key():
    c2a = Sprint17C2Patchifier(C2A_GRIDS)
    assert c2a.support_weights_for(np.array([0]), np.array([32]), 40) is None
    batch = collate_variable_files([_sample(100)], c2a)
    assert "patch_support_weights" not in batch
    b0 = collate_variable_files([_sample(100)], B0_PATCH)
    assert "patch_support_weights" not in b0


def test_c2b_union_lattice_matches_b0_spacing_at_b0_count_scale():
    c2b = Sprint17C2Patchifier(C2B_GRIDS)
    got = c2b.patchify(_sample(512))
    want = _b0_batch(512)
    assert sorted(got.starts.tolist()) == got.starts.tolist()
    assert set(int(s) % 32 for s in got.starts.tolist()) <= {0, 16}
    # Same lattice family (every-16 starts) at ~B0 count: no envelope breach.
    assert want.N == 31
    assert got.N == 32
    assert c2b.grid_ids_for(512).tolist().count(0) == 16
    assert c2b.grid_ids_for(512).tolist().count(1) == 16


def test_c2b_weights_have_unit_timestep_mass():
    c2b = Sprint17C2Patchifier(C2B_GRIDS)
    for t in LENGTH_BATTERY:
        pb = c2b.patchify(_sample(t))
        w = c2b.support_weights_for(pb.starts, pb.valid_len, t)
        assert w is not None
        assert w.dtype == np.float32
        assert float(w.sum()) == pytest.approx(float(t), rel=1e-5)
        # Unit-mass proof: patch p carries sum(1/k) over its valid steps, so
        # the weight total equals the timestep count (each step contributes
        # k(t) x 1/k(t) = 1); full coverage is asserted alongside.
        counts = np.zeros(t, dtype=np.int64)
        for s, vl in zip(pb.starts.tolist(), pb.valid_len.tolist()):
            counts[s:min(s + vl, t)] += 1
        assert bool((counts >= 1).all()), f"T={t} has uncovered timesteps"


def test_c2b_reference_weights_match_hand_values():
    c2b = Sprint17C2Patchifier(C2B_GRIDS)
    pb = c2b.patchify(_sample(512))
    w = c2b.support_weights_for(pb.starts, pb.valid_len, 512)
    by_start = {int(s): float(v)
                for s, v in zip(pb.starts.tolist(), w.tolist())}
    # Interior timesteps are double-covered: each covering patch takes 1/2.
    assert by_start[32] == pytest.approx(16.0)
    # Leading grid-0 patch covers 16 single + 16 double timesteps.
    assert by_start[0] == pytest.approx(24.0)
    # Grid-1 padded tail covers 16 double-covered timesteps.
    assert by_start[496] == pytest.approx(8.0)


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        arm_grids("C2-C")
    with pytest.raises(ValueError):
        arm_patchifier("B0")
    assert arm_grids("C2-A") == C2A_GRIDS
    assert arm_grids("C2-B") == C2B_GRIDS


def test_weighted_pool_with_unit_weights_equals_plain_mean():
    from representation.model import V1RepresentationModel
    latents = torch.randn(2, 5, 4)
    valid = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    plain = V1RepresentationModel._pool_file(latents, valid)
    ones = torch.where(valid, torch.ones_like(latents[..., 0]),
                       torch.zeros_like(latents[..., 0]))
    assert torch.equal(V1RepresentationModel._pool_file(latents, valid, ones),
                       plain)


def test_weighted_pool_matches_hand_computation():
    from representation.model import V1RepresentationModel
    latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    valid = torch.tensor([[True, True, True]])
    weights = torch.tensor([[24.0, 16.0, 8.0]])
    got = V1RepresentationModel._pool_file(latents, valid, weights)
    want = (latents * weights.unsqueeze(-1)).sum(1) / 48.0
    assert torch.equal(got, want)
    with pytest.raises(ValueError):
        V1RepresentationModel._pool_file(latents, valid,
                                         torch.ones(1, 2))


def test_collate_c2b_weights_rows_sum_to_file_length():
    c2b = arm_patchifier("C2-B")
    batch = collate_variable_files([_sample(100, 7), _sample(513, 8)], c2b)
    w = batch["patch_support_weights"]
    assert w.dtype == torch.float32
    assert tuple(w.shape) == (2, batch["patches"].shape[1])
    assert float(w[0].sum()) == pytest.approx(100.0, rel=1e-5)
    assert float(w[1].sum()) == pytest.approx(513.0, rel=1e-5)
    # Padded slots beyond each file's patch count carry zero weight.
    n0 = c2b.patchify(_sample(100, 7)).N
    assert bool((w[0, n0:] == 0).all())
    assert float(w[0, :n0].sum()) == pytest.approx(100.0, rel=1e-5)


def test_localization_contract_preserved_for_both_arms():
    for arm in ("C2-A", "C2-B"):
        patchifier = arm_patchifier(arm)
        for t in (33, 100, 512):
            pb = patchifier.patchify(_sample(t))
            covered = set()
            for t_step in range(t):
                hits = Patchifier.which_patches(
                    t_step, pb.starts, pb.valid_len)
                assert len(hits) >= 1, f"{arm} T={t} t={t_step} uncovered"
                covered.update(hits.tolist())
            assert covered == set(range(pb.N)), f"{arm} T={t}"
            scores = np.arange(pb.N, dtype=np.float64)
            ts = Patchifier.patch_to_timestep_scores(
                scores, pb.starts, pb.valid_len, t)
            assert ts.shape == (t,)
            assert bool(np.isfinite(ts).all())


def test_registry_and_parity_declarations():
    from representation.sprint17_ablation import TrainingParity
    for arm in ("C2-A", "C2-B"):
        spec = A.get_arm(arm)
        A.validate_arm(spec)
        assert spec.kind == "single" and spec.components == ("C2",)
        assert ARM_ADAPTER_DESCRIPTION[arm]
    parity = A.validate_training_parity(TrainingParity())
    assert parity.model_seeds == (171701, 171702, 171703)
    assert parity.optimizer_steps == 300 and parity.checkpoint_step == 300


def test_c2a_end_to_end_matches_b0_through_model():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=8,
                   sequence_layers=1, attention_heads=2,
                   use_conditional_norm=False, seed=11)
    torch.manual_seed(11)
    model_b0 = V1RepresentationModel(cfg, patchifier=B0_PATCH).eval()
    torch.manual_seed(11)
    model_c2a = V1RepresentationModel(
        cfg, patchifier=arm_patchifier("C2-A")).eval()
    assert [tuple(p.shape) for p in model_b0.parameters()] == [
        tuple(p.shape) for p in model_c2a.parameters()]
    for p, q in zip(model_b0.parameters(), model_c2a.parameters()):
        q.data.copy_(p.data)
    files = [_sample(100, 7), _sample(513, 8)]
    with torch.no_grad():
        out_b0 = model_b0(collate_variable_files(
            files, B0_PATCH, masking_config=cfg, masking_seed=5))
        out_a = model_c2a(collate_variable_files(
            files, arm_patchifier("C2-A"), masking_config=cfg,
            masking_seed=5))
    assert torch.equal(out_a["file_embedding"], out_b0["file_embedding"])
    assert torch.equal(out_a["context_latents"], out_b0["context_latents"])


def test_c2b_forward_uses_weights_for_file_embedding():
    from representation.config import V1Config
    from representation.model import V1RepresentationModel
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=8,
                   sequence_layers=1, attention_heads=2,
                   use_conditional_norm=False, seed=11)
    torch.manual_seed(11)
    model = V1RepresentationModel(
        cfg, patchifier=arm_patchifier("C2-B")).eval()
    files = [_sample(200, 7)]
    batch = collate_variable_files(files, model.patchifier,
                                   masking_config=cfg, masking_seed=5)
    assert "patch_support_weights" in batch
    with torch.no_grad():
        out = model(batch)
    assert bool(torch.isfinite(out["file_embedding"]).all())
    ctx = out["context_latents"]
    w = batch["patch_support_weights"].to(ctx.dtype).unsqueeze(-1)
    want = (ctx * w).sum(1) / w.sum(1).clamp_min(1e-6)
    assert torch.equal(out["file_embedding"], want)
