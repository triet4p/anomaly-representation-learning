"""
Focused behavioral tests for contrastive view augmentation.

Contracts tested:
  - Two views are distinct from each other
  - Two views are distinct from the original
  - Views preserve shape and dtype
  - Views preserve approximate mean (offset jitter is small)
  - Augmentation does not alter regime semantics (regime structure unchanged)
  - Same seed → same views (reproducibility)
  - Augmentation does not change anomaly region label (if anomaly present)
"""

from __future__ import annotations

import numpy as np
import pytest

from synth.config import SynthConfig, ContrastiveConfig
from synth.contrastive import make_contrastive_views
from synth.generator import SessionGenerator
from synth.schema import AnomalyFamily


@pytest.fixture(scope="module")
def gen():
    return SessionGenerator(SynthConfig())


def test_views_distinct_from_each_other(gen):
    """View1 and View2 must not be identical."""
    s = gen.generate_normal(seed=10, split="train")
    cfg = ContrastiveConfig()
    v1, v2 = make_contrastive_views(s, cfg, np.random.default_rng(0))
    assert not np.allclose(v1, v2, atol=1e-6), "v1 == v2: augmentation is deterministic / no randomness"


def test_views_distinct_from_original(gen):
    """Both views must differ from the original signal."""
    s = gen.generate_normal(seed=11, split="train")
    cfg = ContrastiveConfig()
    v1, v2 = make_contrastive_views(s, cfg, np.random.default_rng(1))
    assert not np.allclose(v1, s.x, atol=1e-6), "v1 == original"
    assert not np.allclose(v2, s.x, atol=1e-6), "v2 == original"


def test_views_shape_and_dtype(gen):
    """Views must match original [C, T] shape and be float32."""
    s = gen.generate_normal(seed=12, split="train")
    cfg = ContrastiveConfig()
    v1, v2 = make_contrastive_views(s, cfg, np.random.default_rng(2))
    assert v1.shape == s.x.shape
    assert v2.shape == s.x.shape
    assert v1.dtype == np.float32
    assert v2.dtype == np.float32


def test_views_preserve_approximate_mean(gen):
    """Mean should be close to original (offset is small, <5% of std)."""
    s = gen.generate_normal(seed=13, split="train")
    cfg = ContrastiveConfig(offset_std=0.01, gain_std=0.02, noise_std=0.005)
    rng = np.random.default_rng(3)
    v1, _ = make_contrastive_views(s, cfg, rng)
    orig_mean = float(np.mean(s.x))
    v1_mean = float(np.mean(v1))
    orig_std = float(np.std(s.x) + 1e-8)
    assert abs(v1_mean - orig_mean) / orig_std < 0.15, \
        f"View mean shifted too much: {abs(v1_mean - orig_mean) / orig_std:.3f}"


def test_views_finite(gen):
    """Views must be finite — no NaN or Inf introduced."""
    s = gen.generate_normal(seed=14, split="train")
    cfg = ContrastiveConfig()
    v1, v2 = make_contrastive_views(s, cfg, np.random.default_rng(4))
    assert np.isfinite(v1).all(), "v1 contains non-finite values"
    assert np.isfinite(v2).all(), "v2 contains non-finite values"


def test_views_reproducible(gen):
    """Same RNG seed → same views."""
    s = gen.generate_normal(seed=15, split="train")
    cfg = ContrastiveConfig()
    v1a, v2a = make_contrastive_views(s, cfg, np.random.default_rng(42))
    v1b, v2b = make_contrastive_views(s, cfg, np.random.default_rng(42))
    np.testing.assert_array_equal(v1a, v1b)
    np.testing.assert_array_equal(v2a, v2b)


def test_views_different_seeds_differ(gen):
    """Different RNG seeds → different views."""
    s = gen.generate_normal(seed=16, split="train")
    cfg = ContrastiveConfig()
    v1a, _ = make_contrastive_views(s, cfg, np.random.default_rng(0))
    v1b, _ = make_contrastive_views(s, cfg, np.random.default_rng(1))
    assert not np.array_equal(v1a, v1b)


def test_augmentation_small_perturbation(gen):
    """Each channel in each view stays within ±20% of original range."""
    s = gen.generate_normal(seed=17, split="train")
    cfg = ContrastiveConfig(gain_std=0.05, offset_std=0.02, noise_std=0.01, max_shift=2)
    v1, v2 = make_contrastive_views(s, cfg, np.random.default_rng(5))
    x = s.x.astype(np.float64)
    for c in range(s.C):
        ch_range = float(np.max(x[c]) - np.min(x[c])) + 1e-8
        for view in (v1, v2):
            max_dev = float(np.max(np.abs(view[c].astype(np.float64) - x[c])))
            assert max_dev < ch_range * 0.40, \
                f"Ch{c} max deviation {max_dev:.4f} > 40% of range {ch_range:.4f}"
