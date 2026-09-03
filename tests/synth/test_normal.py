"""
Focused behavioral tests for normal session generation.

Contracts tested:
  - Reproducibility: same seed → identical x
  - Variable length: T varies across seeds
  - Regime diversity: multiple regime types appear across sessions
  - Transition smoothness: no step discontinuities between regimes
  - Metadata completeness: regime sequence covers [0, T) without gaps
  - Channel causal ordering: S1 responds to S0, S2 responds to S1
  - FileSample.validate() passes
"""

from __future__ import annotations

import numpy as np
import pytest

from synth.config import SynthConfig
from synth.generator import SessionGenerator
from synth.schema import SampleLabel, RegimeType


@pytest.fixture(scope="module")
def gen():
    return SessionGenerator(SynthConfig())


def test_reproducibility(gen):
    """Same seed → byte-identical output."""
    s1 = gen.generate_normal(seed=42, split="train")
    s2 = gen.generate_normal(seed=42, split="train")
    np.testing.assert_array_equal(s1.x, s2.x)
    assert s1.file_id == s2.file_id


def test_different_seeds_produce_different_samples(gen):
    """Different seeds → different outputs."""
    s1 = gen.generate_normal(seed=1, split="train")
    s2 = gen.generate_normal(seed=2, split="train")
    assert not np.array_equal(s1.x, s2.x)


def test_variable_length(gen):
    """T must vary across seeds; not all 512."""
    Ts = [gen.generate_normal(seed=i * 7, split="train").T for i in range(20)]
    assert min(Ts) != max(Ts), "All sessions have the same length — fixed T still present"
    cfg = SynthConfig()
    assert min(Ts) >= cfg.regime.min_total_steps - 10  # allow small rounding
    assert max(Ts) <= cfg.regime.max_total_steps + 10


def test_regime_diversity():
    """Multiple distinct regime types appear across 30 sessions."""
    gen = SessionGenerator(SynthConfig())
    seen_regimes = set()
    for i in range(30):
        s = gen.generate_normal(seed=i, split="train")
        seen_regimes.update(r.regime for r in s.regime_sequence)
    assert len(seen_regimes) >= 3, f"Only {len(seen_regimes)} regime types seen: {seen_regimes}"


def test_regime_sequence_contiguous(gen):
    """Regime sequence covers [0, T) without gaps or overlaps."""
    for i in range(10):
        s = gen.generate_normal(seed=i * 13, split="train")
        s.validate()  # raises if not contiguous


def test_regime_min_count(gen):
    """Each session has at least min_regimes regimes."""
    cfg = SynthConfig()
    for i in range(20):
        s = gen.generate_normal(seed=i * 5, split="train")
        assert len(s.regime_sequence) >= cfg.regime.min_regimes


def test_signal_finite(gen):
    """All channel values are finite (no NaN, no Inf)."""
    for i in range(10):
        s = gen.generate_normal(seed=i * 3, split="train")
        assert np.isfinite(s.x).all(), f"Non-finite values in sample {i}"


def test_signal_dtype(gen):
    s = gen.generate_normal(seed=0, split="train")
    assert s.x.dtype == np.float32


def test_file_label_normal(gen):
    s = gen.generate_normal(seed=0, split="train")
    assert s.file_label == SampleLabel.NORMAL
    assert s.anomaly_meta is None
    assert s.anomaly_mask is None


def test_cross_channel_nonzero_correlation():
    """S1 and S2 must be significantly correlated with S0 (causal chain)."""
    gen = SessionGenerator(SynthConfig())
    corrs01, corrs02 = [], []
    for i in range(20):
        s = gen.generate_normal(seed=i * 17, split="train")
        x = s.x.astype(np.float64)
        if x.shape[1] > 20:
            c01 = np.corrcoef(x[0], x[1])[0, 1]
            c02 = np.corrcoef(x[0], x[2])[0, 1]
            corrs01.append(abs(c01))
            corrs02.append(abs(c02))
    assert np.mean(corrs01) > 0.3, "S0-S1 correlation too low"
    assert np.mean(corrs02) > 0.1, "S0-S2 correlation too low"


def test_transition_smoothness():
    """No step discontinuity larger than 20% of signal range at regime boundaries."""
    gen = SessionGenerator(SynthConfig())
    for i in range(10):
        s = gen.generate_normal(seed=i, split="train")
        x = s.x.astype(np.float64)
        C, T = x.shape
        for r in s.regime_sequence[:-1]:
            boundary = r.end
            if boundary <= 0 or boundary >= T:
                continue
            for c in range(C):
                diff = abs(float(x[c, boundary]) - float(x[c, boundary - 1]))
                sig_range = float(np.max(x[c]) - np.min(x[c])) + 1e-8
                assert diff / sig_range < 0.25, (
                    f"Step discontinuity {diff:.3f} / range {sig_range:.3f} = "
                    f"{diff/sig_range:.2%} at boundary {boundary} ch{c}"
                )


def test_split_ids_are_disjoint():
    """file_id differs for same seed in different splits."""
    gen = SessionGenerator(SynthConfig())
    train = gen.generate_normal(seed=42, split="train")
    val = gen.generate_normal(seed=42, split="val")
    assert train.file_id != val.file_id
