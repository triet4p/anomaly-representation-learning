"""Focused tests for Sprint 11 Batch F2 aggregate helpers (Tasks 35-36)."""

import math

import numpy as np
import pytest

from representation.v2_batch_f2 import (
    brier_score,
    censoring_counts,
    concordance_index,
    count_warning_runs,
    describe_or_null,
    equal_width_ece,
    group_recall,
    maintenance_proximity,
)


def test_concordance_perfect_vs_inverted():
    times = [1.0, 2.0, 3.0, 4.0]
    events = [True, True, True, False]
    perfect = concordance_index(times, events, [0.9, 0.7, 0.5, 0.1])
    assert perfect["c"] == pytest.approx(1.0)
    inverted = concordance_index(times, events, [0.1, 0.5, 0.7, 0.9])
    assert inverted["c"] == pytest.approx(0.0)
    assert perfect["n_comparable_pairs"] == inverted["n_comparable_pairs"] > 0


def test_concordance_ties_and_censoring():
    # Tied scores count half; censored-later files still serve as comparators.
    out = concordance_index([1.0, 2.0, 3.0], [True, False, False], [0.5, 0.5, 0.1])
    assert out["c"] == pytest.approx(0.75)
    assert out["n_tied"] == 1
    # Non-finite times are excluded, not fabricated.
    excluded = concordance_index(
        [1.0, float("nan")], [True, False], [0.8, 0.9]
    )
    assert excluded["n_files"] == 1
    assert excluded["n_comparable_pairs"] == 0
    assert excluded["c"] is None
    with pytest.raises(ValueError):
        concordance_index([], [], [])
    with pytest.raises(ValueError):
        concordance_index([1.0], [True], [float("nan")])


def test_describe_or_null_empty_and_edges():
    assert describe_or_null([]) == {"n": 0}
    desc = describe_or_null([1.0, 2.0, 3.0, 4.0])
    assert desc["n"] == 4 and desc["median"] == pytest.approx(2.5)
    with pytest.raises(ValueError):
        describe_or_null([1.0, math.inf])


def test_ece_and_brier_edges():
    ece = equal_width_ece([0.05, 0.95, 0.05, 0.95], [False, True, False, True])
    assert ece["ece"] == pytest.approx(0.05)
    assert ece["n"] == 4 and len(ece["bins"]) == 10
    assert brier_score([0.0, 1.0], [False, True]) == pytest.approx(0.0)
    assert brier_score([1.0, 0.0], [False, True]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        equal_width_ece([], [])
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])


def test_warning_runs_merge_consecutive():
    assert count_warning_runs([False, True, True, False, True, False]) == 2
    assert count_warning_runs([False, False]) == 0
    assert count_warning_runs([True, True, True]) == 1


def test_group_recall_and_censoring_counts():
    grouped = group_recall(
        ["a", "a", "b", "b"], [True, True, True, False], [True, False, True, True]
    )
    assert grouped["a"]["rate"] == pytest.approx(0.5)
    assert grouped["b"]["n"] == 1 and "c" not in grouped
    counts = censoring_counts([1.0, 2.0, None], [True, False, False])
    assert counts == {
        "n_files": 3,
        "n_observed_events": 1,
        "n_censored": 2,
        "n_nan_times": 1,
    }


def test_maintenance_proximity_windows():
    flags = maintenance_proximity([100.0, 200.0, 1000.0], [105.0], 10.0)
    assert flags == [True, False, False]
    assert maintenance_proximity([1.0], [], 5.0) == [False]
    with pytest.raises(ValueError):
        maintenance_proximity([1.0], [1.0], -1.0)
