"""
Focused behavioral tests for patchification.

Contracts tested:
  - Patch shape: [N, C, W]
  - Start indices: monotonically increasing, correctly spaced
  - Valid length: real (non-padded) steps per patch
  - Padding mask: True exactly at padded positions
  - Round-trip: can reconstruct signal from patches + starts + valid_len
  - Timestep→patch mask: all anomalous timesteps map to at least one True patch
  - Patch-level mask: patches covering anomaly region are marked True
  - Variable-length files: different T values patchify without error
  - End padding: when pad_end=True, last patch always present even if partial
"""

from __future__ import annotations

import numpy as np
import pytest

from synth.config import SynthConfig, PatchConfig
from synth.generator import SessionGenerator
from synth.patchify import Patchifier
from synth.schema import AnomalyFamily


@pytest.fixture(scope="module")
def gen():
    return SessionGenerator(SynthConfig())


def test_patch_shape(gen):
    """Output has shape [N, C, W]."""
    cfg = PatchConfig(patch_size=32, stride=16)
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=1, split="train")
    batch = p.patchify(s)
    N, C, W = batch.patches.shape
    assert C == s.C
    assert W == 32
    assert N >= 1


def test_start_indices_monotonic(gen):
    cfg = PatchConfig(patch_size=32, stride=16)
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=2, split="train")
    batch = p.patchify(s)
    starts = batch.starts
    assert (np.diff(starts) > 0).all(), "Start indices not strictly increasing"


def test_start_indices_spacing(gen):
    """Consecutive starts differ by exactly stride."""
    cfg = PatchConfig(patch_size=32, stride=16)
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=3, split="train")
    batch = p.patchify(s)
    diffs = np.diff(batch.starts)
    # All except possibly the last should be stride
    assert all(d == 16 for d in diffs), f"Unexpected strides: {diffs.tolist()}"


def test_valid_len_correct(gen):
    """valid_len[i] = min(W, T - starts[i])."""
    cfg = PatchConfig(patch_size=32, stride=16, pad_end=True)
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=4, split="train")
    batch = p.patchify(s)
    T = s.T
    for i, (start, vlen) in enumerate(zip(batch.starts, batch.valid_len)):
        expected = min(32, T - int(start))
        assert int(vlen) == expected, f"Patch {i}: vlen={vlen}, expected={expected}"


def test_pad_mask_correct(gen):
    """pad_mask[i, j] = True iff j >= valid_len[i]."""
    cfg = PatchConfig(patch_size=32, stride=16, pad_end=True)
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=5, split="train")
    batch = p.patchify(s)
    for i, vlen in enumerate(batch.valid_len):
        pm = batch.pad_mask[i]
        expected = np.zeros(32, dtype=bool)
        expected[int(vlen):] = True
        np.testing.assert_array_equal(pm, expected, err_msg=f"Patch {i} pad_mask wrong")


def test_round_trip_reconstruction(gen):
    """Reconstructing signal from patches+starts+valid_len should match original."""
    cfg = PatchConfig(patch_size=32, stride=32, pad_end=True)  # non-overlapping
    p = Patchifier(cfg)
    s = gen.generate_normal(seed=6, split="train")
    batch = p.patchify(s)
    C, T = s.C, s.T

    recon = np.zeros((C, T), dtype=np.float32)
    for i, (start, vlen) in enumerate(zip(batch.starts, batch.valid_len)):
        start, vlen = int(start), int(vlen)
        end = min(start + vlen, T)
        recon[:, start:end] = batch.patches[i, :, :vlen][:, :end - start]

    np.testing.assert_allclose(
        recon[:, :T], s.x[:, :T],
        atol=1e-5,
        err_msg="Round-trip reconstruction failed",
    )


def test_anomaly_mask_to_patch_mask(gen):
    """Every anomalous timestep must appear in at least one True patch."""
    cfg = PatchConfig(patch_size=32, stride=16, pad_end=True)
    p = Patchifier(cfg)

    for i in range(10):
        s = gen.generate_anomaly(seed=i * 11, split="test",
                                 family=AnomalyFamily.REALISTIC_STUCK)
        if s.anomaly_mask is None:
            continue
        batch = p.patchify(s)
        patch_mask = batch.timestep_to_patch_mask()
        if patch_mask is None:
            continue

        # For each timestep that is anomalous in any channel, check it's covered
        ts_anomalous = s.anomaly_mask.any(axis=0)  # [T]
        for t in np.where(ts_anomalous)[0]:
            covering = Patchifier.which_patches(int(t), batch.starts, batch.valid_len)
            assert any(patch_mask[i] for i in covering), \
                f"Anomalous timestep {t} not covered by any True patch"
        return

    pytest.skip("No accepted anomalous samples generated")


def test_variable_lengths_no_error(gen):
    """Patchification must succeed for all session lengths."""
    cfg = PatchConfig(patch_size=32, stride=16, pad_end=True)
    p = Patchifier(cfg)
    for i in range(30):
        s = gen.generate_normal(seed=i * 19, split="train")
        batch = p.patchify(s)
        assert batch.N >= 1
        assert batch.patches.dtype == np.float32


def test_no_padding_when_exact_multiple(gen):
    """When T is exact multiple of W and stride=W, no patch is padded."""
    # Craft a sample of exact length multiple of W
    W = 32
    cfg = SynthConfig()
    cfg.regime.min_total_steps = W * 5
    cfg.regime.max_total_steps = W * 5
    gen2 = SessionGenerator(cfg)
    s = gen2.generate_normal(seed=0, split="train")
    # May not be exactly W*5 due to rounding, but test pad_mask logic
    pcfg = PatchConfig(patch_size=W, stride=W, pad_end=True)
    p = Patchifier(pcfg)
    batch = p.patchify(s)
    # With stride=W every full window is emitted and no redundant tail exists.
    assert batch.starts.tolist() == [0, 32, 64, 96, 128]
    assert batch.valid_len.tolist() == [32, 32, 32, 32, 32]
    assert not batch.pad_mask.any(), "Exact-length patch batch contains padding"

def test_timestep_to_patch_score_aggregation():
    """patch_to_timestep_scores round-trips: aggregated scores sum correctly."""
    import numpy as np
    from synth.patchify import Patchifier
    starts = np.array([0, 8, 16], dtype=np.int64)
    valid_len = np.array([16, 16, 8], dtype=np.int64)
    T = 24
    patch_scores = np.array([1.0, 2.0, 3.0])
    ts_scores = Patchifier.patch_to_timestep_scores(patch_scores, starts, valid_len, T)
    # t=8..15 covered by patches 0 and 1 → mean of 1.0 and 2.0 = 1.5
    np.testing.assert_allclose(ts_scores[8:16], 1.5, atol=1e-6)
    # t=16..23 covered by patch 1 and 2 → mean of 2.0 and 3.0 = 2.5
    np.testing.assert_allclose(ts_scores[16:24], 2.5, atol=1e-6)
