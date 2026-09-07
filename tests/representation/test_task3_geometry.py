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
