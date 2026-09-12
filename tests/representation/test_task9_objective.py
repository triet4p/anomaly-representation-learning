"""Focused tests for Task 9 fail-closed gates (no checkpoint/data needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from sprint16_task9_objective import (  # noqa: E402
    C5_VARIANTS,
    MAX_GRAD_RUNS,
    MAX_GRAD_STEPS,
    check_c5_knob,
    tracking_cosine,
)


def test_tracking_perfect_and_bounds():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 4))
    t = tracking_cosine(a, a)
    assert t["median"] == 1.0
    assert t["lcb"] <= 1.0 <= t["ucb"]
    assert t["n"] == 50


def test_tracking_rejects_bad_shapes():
    a = np.zeros((10, 4))
    with pytest.raises(ValueError):
        tracking_cosine(a, np.zeros((5, 4)))
    with pytest.raises(ValueError):
        tracking_cosine(np.zeros((0, 4)), np.zeros((0, 4)))
    with pytest.raises(ValueError):
        tracking_cosine(np.full((4, 2), np.nan), np.zeros((4, 2)))


def test_c5_knobs_fail_closed():
    assert set(C5_VARIANTS) == {"mask015", "nocontrast"}
    for variant in C5_VARIANTS:
        with pytest.raises(ValueError, match="absent"):
            check_c5_knob(variant)
    with pytest.raises(KeyError):
        check_c5_knob("unapproved-variant")


def test_budget_constants_match_protocol():
    assert MAX_GRAD_RUNS == 2
    assert MAX_GRAD_STEPS == 300
