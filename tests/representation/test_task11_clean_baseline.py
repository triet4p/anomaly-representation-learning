"""Focused tests for Task 11 clean-baseline exclusion and causal windows."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from sprint12_task11_precursors import (  # noqa: E402
    baseline_eligible,
    intervals_overlap,
    in_window_causal,
    tail_energy,
)

DAY = 86400.0
F = 100.0 * DAY  # failure start


def _base_kwargs(**over):
    kw = {
        "fail_start": F,
        "other_starts": [],
        "degradations": [],
        "maint_list": [],
        "quarantined": False,
    }
    kw.update(over)
    return kw


def test_intervals_overlap_boundaries():
    assert intervals_overlap(0.0, 10.0, 5.0, 15.0) is True
    assert intervals_overlap(0.0, 10.0, 10.0, 20.0) is False  # touching is disjoint
    assert intervals_overlap(10.0, 20.0, 0.0, 10.0) is False
    assert intervals_overlap(0.0, 10.0, 2.0, 4.0) is True  # containment


def test_clean_file_eligible():
    ok, reason = baseline_eligible(F - 20 * DAY, F - 19 * DAY, **_base_kwargs())
    assert (ok, reason) == (True, "ok")


def test_quarantined_excluded():
    ok, reason = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY, **_base_kwargs(quarantined=True))
    assert (ok, reason) == (False, "quarantined")


def test_maintenance_overlap_excluded():
    ok, reason = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY,
        **_base_kwargs(maint_list=[(F - 19.5 * DAY, F - 18 * DAY)]))
    assert (ok, reason) == (False, "maintenance")
    # touching maintenance edge is clean
    ok, _ = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY,
        **_base_kwargs(maint_list=[(F - 19 * DAY, F - 18 * DAY)]))
    assert ok is True


def test_degradation_overlap_excluded():
    ok, reason = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY,
        **_base_kwargs(degradations=[(F - 25 * DAY, F - 10 * DAY)]))
    assert (ok, reason) == (False, "degradation")
    # degradation fully before the file is clean
    ok, _ = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY,
        **_base_kwargs(degradations=[(F - 40 * DAY, F - 30 * DAY)]))
    assert ok is True


def test_other_failure_window_excluded():
    other = F - 40 * DAY
    ok, reason = baseline_eligible(
        other - 8 * DAY, other - 7 * DAY,
        **_base_kwargs(other_starts=[other]))
    assert (ok, reason) == (False, "other_failure_window")
    # file older than the other window is clean
    ok, _ = baseline_eligible(
        other - 40 * DAY, other - 39 * DAY,
        **_base_kwargs(other_starts=[other]))
    assert ok is True


def test_exclusion_priority_order():
    # quarantine is reported first even when several reasons apply
    ok, reason = baseline_eligible(
        F - 20 * DAY, F - 19 * DAY,
        **_base_kwargs(quarantined=True,
                       maint_list=[(F - 21 * DAY, F - 18 * DAY)],
                       degradations=[(F - 25 * DAY, F - 10 * DAY)],
                       other_starts=[F - 30 * DAY]))
    assert (ok, reason) == (False, "quarantined")


def test_window_causality_boundaries():
    assert in_window_causal(F - DAY, F, DAY) is True
    assert in_window_causal(F, F, DAY) is True  # ends exactly at failure start
    assert in_window_causal(F - DAY, F, DAY) is True  # window open edge inclusive
    assert in_window_causal(F + 1.0, F, DAY) is False  # future leak excluded
    assert in_window_causal(F - DAY - 1.0, F, DAY) is False  # too old


def test_tail_energy_top_q():
    e = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    v = np.ones(10, dtype=bool)
    assert tail_energy(e, v, top_q=0.1) == 10.0
    assert tail_energy(e, v, top_q=0.2) == 9.5
    v[8:] = False
    assert tail_energy(e, v, top_q=0.1) == 8.0  # invalid patches ignored
    assert np.isnan(tail_energy(e, np.zeros(10, dtype=bool)))
