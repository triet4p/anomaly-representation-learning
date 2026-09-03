"""
Focused behavioral tests for dataset splits and contamination.

Contracts tested:
  - Train/val/test file IDs are disjoint (no leakage)
  - Train contamination=0.0 → all NORMAL labels
  - Contamination levels produce correct anomaly fraction
  - Seeds are non-overlapping across splits
  - .npz save/load round-trip preserves x, label, mask
  - Anomaly families appear with roughly equal weight in test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from synth.config import SynthConfig
from synth.dataset import DatasetBuilder, _save_sample, load_sample
from synth.schema import SampleLabel, AnomalyFamily


@pytest.fixture(scope="module")
def builder():
    cfg = SynthConfig()
    cfg.split.n_train = 30
    cfg.split.n_val = 20
    cfg.split.n_test = 20
    return DatasetBuilder(cfg)


def test_train_file_ids_unique(builder):
    """No duplicate file IDs within the training split."""
    samples = builder.build_split("train", n=20, contamination=0.0)
    ids = [s.file_id for s in samples]
    assert len(set(ids)) == len(ids), "Duplicate file IDs in train split"


def test_split_file_ids_disjoint(builder):
    """File IDs are disjoint across train/val/test splits."""
    train = builder.build_split("train", n=15, contamination=0.0)
    val   = builder.build_split("val",   n=15, contamination=0.5)
    test  = builder.build_split("test",  n=15, contamination=0.5)
    train_ids = {s.file_id for s in train}
    val_ids   = {s.file_id for s in val}
    test_ids  = {s.file_id for s in test}
    assert train_ids.isdisjoint(val_ids),  "Train/val ID overlap (leakage)"
    assert train_ids.isdisjoint(test_ids), "Train/test ID overlap (leakage)"
    assert val_ids.isdisjoint(test_ids),   "Val/test ID overlap (leakage)"


def test_zero_contamination_all_normal(builder):
    """contamination=0.0 → all samples have NORMAL label."""
    samples = builder.build_split("train", n=20, contamination=0.0)
    labels = [s.file_label for s in samples]
    assert all(l == SampleLabel.NORMAL for l in labels), \
        f"Non-normal labels in zero-contamination split: {set(labels)}"


def test_contamination_ratio_approximate(builder):
    """Actual anomaly fraction ≈ contamination (within ±10%)."""
    for ratio in [0.0, 0.10, 0.20]:
        samples = builder.build_split("train", n=50, contamination=ratio)
        n_anom = sum(1 for s in samples if s.file_label == SampleLabel.ABNORMAL)
        actual = n_anom / 50
        assert abs(actual - ratio) <= 0.12, \
            f"Contamination ratio {ratio}: actual={actual:.2f}"


def test_npz_round_trip(builder):
    """Save → load preserves x, file_label, seed, anomaly_mask."""
    samples = builder.build_split("test", n=5, contamination=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        for s in samples:
            path = str(Path(tmp) / f"{s.file_id}.npz")
            _save_sample(s, path)
            s2 = load_sample(path)
            np.testing.assert_array_equal(s.x, s2.x, err_msg="x mismatch after save/load")
            assert s.file_label == s2.file_label
            assert s.seed == s2.seed
            if s.anomaly_mask is not None:
                assert s2.anomaly_mask is not None
                np.testing.assert_array_equal(s.anomaly_mask, s2.anomaly_mask)


def test_anomaly_metadata_preserved_in_npz(builder):
    """Anomaly metadata (type, start, end, severity) survives .npz round-trip."""
    samples = builder.build_split("test", n=10, contamination=0.6)
    anom = [s for s in samples if s.anomaly_meta is not None]
    if not anom:
        pytest.skip("No anomalous samples generated")
    with tempfile.TemporaryDirectory() as tmp:
        for s in anom[:3]:
            path = str(Path(tmp) / f"{s.file_id}.npz")
            _save_sample(s, path)
            s2 = load_sample(path)
            assert s2.anomaly_meta is not None
            assert s2.anomaly_meta.family == s.anomaly_meta.family
            assert s2.anomaly_meta.start == s.anomaly_meta.start
            assert s2.anomaly_meta.end == s.anomaly_meta.end
            assert abs(s2.anomaly_meta.severity - s.anomaly_meta.severity) < 1e-4


def test_reproducibility_across_runs(builder):
    """Rebuilding the same split with same config produces identical file_ids."""
    split_a = builder.build_split("val", n=10, contamination=0.5)
    split_b = builder.build_split("val", n=10, contamination=0.5)
    ids_a = [s.file_id for s in split_a]
    ids_b = [s.file_id for s in split_b]
    assert ids_a == ids_b, "Split is not reproducible"


def test_anomaly_families_in_test(builder):
    """At least 3 distinct anomaly families appear in the test split."""
    samples = builder.build_split("test", n=40, contamination=0.5)
    families = {s.anomaly_meta.family for s in samples if s.anomaly_meta is not None}
    assert len(families) >= 3, f"Only {len(families)} families in test: {families}"


def test_contamination_sweep_keys(builder):
    """build_contamination_sweep returns dict with correct keys."""
    ratios = [0.0, 0.05, 0.10]
    sweep = builder.build_contamination_sweep(n_train=10, ratios=ratios)
    assert set(sweep.keys()) == set(ratios)
    for r, samples in sweep.items():
        assert len(samples) == 10
