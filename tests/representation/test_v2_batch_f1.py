"""Focused tests for Sprint 11 Batch F1 aggregate helpers (Tasks 33-34)."""

import math

import numpy as np
import pytest

from representation.v2_batch_f1 import (
    anisotropy_ratio,
    confusion,
    describe,
    effective_rank,
    group_rates,
    health_bin,
    mask_overlap_fraction,
    rates_with_counts,
    severity_bin,
    summarize_deltas,
    top_tail_mass,
)


def test_effective_rank_isotropic_vs_collapsed():
    rng = np.random.RandomState(0)
    isotropic = rng.normal(size=(400, 8))
    collapsed = np.zeros((400, 8))
    collapsed[:, 0] = rng.normal(size=400)
    assert effective_rank(isotropic) > effective_rank(collapsed)
    assert effective_rank(isotropic) <= 8.0
    with pytest.raises(ValueError):
        effective_rank(np.full((10, 4), np.nan))


def test_anisotropy_spread_direction():
    rng = np.random.RandomState(1)
    tight = rng.normal(scale=1.0, size=(300, 4))
    stretched = rng.normal(size=(300, 4)) @ np.diag([10.0, 1.0, 1.0, 1.0])
    assert anisotropy_ratio(stretched) > anisotropy_ratio(tight)
    assert anisotropy_ratio(tight) >= 1.0


def test_mask_overlap_fraction_edges():
    flagged = [False] * 100
    for i in range(20, 40):
        flagged[i] = True
    assert mask_overlap_fraction(flagged, 20, 40) == pytest.approx(1.0)
    assert mask_overlap_fraction(flagged, 0, 10) == pytest.approx(0.0)
    assert mask_overlap_fraction(flagged, 30, 50) == pytest.approx(0.5)
    assert mask_overlap_fraction([], 0, 10) == pytest.approx(0.0)


def test_top_tail_mass_sparse_vs_diffuse():
    valid = np.ones(20, dtype=bool)
    sparse = np.zeros(20)
    sparse[0] = 10.0
    diffuse = np.ones(20)
    assert top_tail_mass(sparse, valid, 0.1) > top_tail_mass(diffuse, valid, 0.1)
    assert top_tail_mass(diffuse, valid, 0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        top_tail_mass(np.ones(5), np.ones(4, dtype=bool), 0.1)


def test_bins_are_fixed_and_documented():
    assert severity_bin(0.1) == "low(<0.5)"
    assert severity_bin(0.7) == "mid[0.5,1.0)"
    assert severity_bin(2.0) == "high[>=1.0)"
    assert health_bin(0.05) == "healthy(<0.2)"
    assert health_bin(0.3) == "worn[0.2,0.5)"
    assert health_bin(0.9) == "degraded[>=0.5)"


def test_confusion_and_rates():
    stats = confusion([True, True, False, False], [True, False, True, False])
    assert (stats["tp"], stats["fp"], stats["fn"], stats["tn"]) == (1, 1, 1, 1)
    assert stats["f1"] == pytest.approx(0.5)
    rates = rates_with_counts([])
    assert rates == {"n": 0, "flagged": 0, "rate": 0.0}
    grouped = group_rates(["a", "a", "b"], [True, False, True])
    assert grouped["a"]["rate"] == pytest.approx(0.5)
    assert grouped["b"]["n"] == 1


def test_summarize_deltas_and_describe():
    deltas = summarize_deltas(np.array([1.0, 2.0, 3.0]))
    assert deltas["mean"] == pytest.approx(2.0)
    assert deltas["median"] == pytest.approx(2.0)
    assert deltas["n"] == 3
    desc = describe([1.0, 2.0, 3.0, 4.0])
    assert desc["n"] == 4 and desc["min"] == pytest.approx(1.0)
    assert describe([]) == {"n": 0}
    with pytest.raises(ValueError):
        summarize_deltas(np.array([]))
    with pytest.raises(ValueError):
        describe([1.0, math.inf])
