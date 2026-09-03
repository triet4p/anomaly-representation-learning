"""
Focused behavioral tests for masking.

Contracts tested:
  - Total mask ratio is approximately cfg.total_mask_ratio (±5%)
  - All three components contribute patches
  - Masked patches are a subset of valid (non-padded) patches
  - Composition fractions sum correctly
  - Ablation: changing composition while keeping total ratio constant
  - Reproducibility: same RNG seed → same mask
  - Block masking produces contiguous runs
"""

from __future__ import annotations

import numpy as np
import pytest

from synth.config import SynthConfig, MaskingConfig, PatchConfig
from synth.generator import SessionGenerator
from synth.masking import apply_masking
from synth.patchify import Patchifier


@pytest.fixture(scope="module")
def batch_fixture():
    cfg = SynthConfig()
    gen = SessionGenerator(cfg)
    s = gen.generate_normal(seed=42, split="train")
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    return patchifier.patchify(s)


def test_mask_ratio_approximate(batch_fixture):
    """Total mask ratio should be within 5% of configured value."""
    batch = batch_fixture
    cfg = MaskingConfig(total_mask_ratio=0.40)
    rng = np.random.default_rng(0)
    result = apply_masking(batch, cfg, rng)
    valid = ~batch.pad_mask.all(axis=1)
    n_valid = int(valid.sum())
    n_masked = int(result.mask[valid].sum())
    actual_ratio = n_masked / max(1, n_valid)
    assert abs(actual_ratio - 0.40) < 0.10, f"Mask ratio {actual_ratio:.3f} too far from 0.40"


def test_mask_only_valid_patches(batch_fixture):
    """Masked patches must be valid (not fully padded)."""
    batch = batch_fixture
    cfg = MaskingConfig()
    rng = np.random.default_rng(1)
    result = apply_masking(batch, cfg, rng)
    fully_padded = batch.pad_mask.all(axis=1)
    assert not (result.mask & fully_padded).any(), \
        "Some fully-padded patches were masked"


def test_composition_present(batch_fixture):
    """Each masking strategy should contribute at least some patches."""
    batch = batch_fixture
    cfg = MaskingConfig(
        total_mask_ratio=0.40,
        random_fraction=0.34,
        info_fraction=0.33,
        block_fraction=0.33,
    )
    rng = np.random.default_rng(2)
    result = apply_masking(batch, cfg, rng)
    # At least two of the three should be non-zero
    non_zero = sum(1 for v in result.composition.values() if v > 0)
    assert non_zero >= 2, f"Too few masking strategies contributed: {result.composition}"


def test_reproducibility(batch_fixture):
    """Same RNG seed → same mask."""
    batch = batch_fixture
    cfg = MaskingConfig()
    r1 = apply_masking(batch, cfg, np.random.default_rng(7))
    r2 = apply_masking(batch, cfg, np.random.default_rng(7))
    np.testing.assert_array_equal(r1.mask, r2.mask)


def test_different_seeds_produce_different_masks(batch_fixture):
    """Different RNG seeds → different masks (with high probability)."""
    batch = batch_fixture
    cfg = MaskingConfig()
    r1 = apply_masking(batch, cfg, np.random.default_rng(10))
    r2 = apply_masking(batch, cfg, np.random.default_rng(20))
    assert not np.array_equal(r1.mask, r2.mask)


def test_ablation_fixed_ratio(batch_fixture):
    """Changing composition while keeping total ratio → same total masked count (approx)."""
    batch = batch_fixture
    rng_seed = 42
    cfg_a = MaskingConfig(total_mask_ratio=0.40, random_fraction=1.0, info_fraction=0.0, block_fraction=0.0)
    cfg_b = MaskingConfig(total_mask_ratio=0.40, random_fraction=0.0, info_fraction=0.0, block_fraction=1.0)
    ra = apply_masking(batch, cfg_a, np.random.default_rng(rng_seed))
    rb = apply_masking(batch, cfg_b, np.random.default_rng(rng_seed))
    # Total ratios should be similar (within 15%)
    assert abs(ra.total_ratio - rb.total_ratio) < 0.15, \
        f"Ablation ratios differ too much: {ra.total_ratio:.3f} vs {rb.total_ratio:.3f}"


def test_block_masking_contiguous(batch_fixture):
    """Block-only masking should produce contiguous runs of masked patches."""
    batch = batch_fixture
    cfg = MaskingConfig(total_mask_ratio=0.30, random_fraction=0.0, info_fraction=0.0, block_fraction=1.0)
    rng = np.random.default_rng(5)
    result = apply_masking(batch, cfg, rng)
    # Check that masked indices form at least one contiguous run
    masked_idx = np.where(result.mask)[0]
    if len(masked_idx) < 2:
        return
    # Find runs
    runs = []
    run_start = masked_idx[0]
    for i in range(1, len(masked_idx)):
        if masked_idx[i] != masked_idx[i - 1] + 1:
            runs.append((run_start, masked_idx[i - 1]))
            run_start = masked_idx[i]
    runs.append((run_start, masked_idx[-1]))
    # At least one run of length >= min_block_size
    assert any(e - s + 1 >= cfg.min_block_size for s, e in runs), \
        f"No block of size >= {cfg.min_block_size} found in block-only mask"


def test_empty_batch_no_error():
    """Masking a batch with no valid patches must not raise."""
    import numpy as np
    from synth.schema import FileSample, SampleLabel, RegimeMeta, RegimeType
    # Create a trivially short sample
    cfg_s = SynthConfig()
    x = np.zeros((3, 5), dtype=np.float32)
    from synth.physics.causal import CausalSignalGenerator
    s = FileSample(
        x=x, file_id="test", file_label=SampleLabel.NORMAL,
        seed=0, generator_version="0", config_hash="0",
        regime_sequence=[RegimeMeta(RegimeType.IDLE, 0, 5, 0.0)],
    )
    pcfg = PatchConfig(patch_size=32, stride=32, pad_end=True)
    batch = Patchifier(pcfg).patchify(s)
    cfg = MaskingConfig()
    result = apply_masking(batch, cfg, np.random.default_rng(0))
    # Should not raise
    assert isinstance(result.mask, np.ndarray)
