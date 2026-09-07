"""Focused tests for Task 3 geometry reuse (conditional fit leakage faults)."""

import pytest
import torch

from representation.v2_geometry import HierarchicalMahalanobisGeometry


def _toy(n_healthy=40, n_abnormal=10, dim=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    healthy = torch.randn((n_healthy, dim), generator=gen)
    abnormal = torch.randn((n_abnormal, dim), generator=gen) + 5.0
    rows = torch.cat([healthy, abnormal])
    aux = torch.zeros((rows.shape[0], 3), dtype=torch.long)
    mask = torch.zeros((rows.shape[0],), dtype=torch.bool)
    mask[:n_healthy] = True
    return rows, aux, mask


def test_fit_uses_only_healthy_rows():
    rows, aux, mask = _toy()
    geo = HierarchicalMahalanobisGeometry(6)
    geo.fit(rows, aux[:, 0], aux[:, 1], aux[:, 2], mask)
    snap = geo.snapshot()
    total = snap["fleet"]["n"] + sum(s["n"] for s in snap["groups"].values())
    assert total >= 40
    # abnormal rows must not shift the fleet mean toward +5
    assert float(snap["fleet"]["mu"].mean().abs()) < 1.0


def test_fit_rejects_empty_healthy_cohort():
    rows, aux, mask = _toy()
    mask[:] = False
    geo = HierarchicalMahalanobisGeometry(6)
    with pytest.raises(ValueError):
        geo.fit(rows, aux[:, 0], aux[:, 1], aux[:, 2], mask)


def test_frozen_geometry_rejects_suspect_refit():
    rows, aux, mask = _toy()
    geo = HierarchicalMahalanobisGeometry(6)
    geo.fit(rows, aux[:, 0], aux[:, 1], aux[:, 2], mask)
    frozen = geo.frozen()
    with pytest.raises(ValueError):
        frozen._geometry.fit(rows, aux[:, 0], aux[:, 1], aux[:, 2], mask)


def test_population_energy_zero_on_invalid_patches():
    rows, aux, mask = _toy()
    geo = HierarchicalMahalanobisGeometry(6)
    geo.fit(rows, aux[:, 0], aux[:, 1], aux[:, 2], mask)
    frozen = geo.frozen()
    lat = torch.randn((2, 5, 6))
    valid = torch.ones((2, 5), dtype=torch.bool)
    valid[0, 3:] = False
    out = frozen.population_energy(
        lat, valid,
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        torch.zeros((2, 5), dtype=torch.long),
    )
    assert bool((out["population_energy"][~valid] == 0.0).all())
    assert bool(torch.isfinite(out["population_energy"][valid]).all())

def test_roc_auc_matches_sklearn_convention():
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    from sprint12_task3_baselines import roc_auc_or_nan

    rng = np.random.default_rng(0)
    # ties: must match sklearn exactly (the inverted code failed here by symmetry)
    scores = np.array([0.1, 0.4, 0.4, 0.9, 0.2, 0.7])
    labels = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert roc_auc_or_nan(scores, labels) == roc_auc_score(labels, scores)
    # random scores with ties from rounding
    s = np.round(rng.normal(size=200), 1)
    y = (rng.random(200) < 0.4).astype(float)
    assert roc_auc_or_nan(s, y) == roc_auc_score(y, s)
    # perfect ranking -> 1.0 (inverted code returned 0.0)
    assert roc_auc_or_nan(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])) == 1.0
    # reversed ranking -> 0.0 (inverted code returned 1.0)
    assert roc_auc_or_nan(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    # single-class input -> nan, never a fabricated rank
    assert np.isnan(roc_auc_or_nan(np.array([1.0, 2.0]), np.array([0.0, 0.0])))
    assert np.isnan(roc_auc_or_nan(np.array([1.0, 2.0]), np.array([1.0, 1.0])))
