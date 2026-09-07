"""Focused tests for Sprint 12 Task 2 diagnostic helpers (pure, CPU-only)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from sprint12_task2_shortcut_diag import (  # noqa: E402
    centered_within_dim_coupling,
    clamp_fractions,
    describe,
    energy_terms,
    file_corruption_mask,
    file_direction,
    ordinary_least_squares_slope,
    pearson,
    seed_from,
)


def test_energy_terms_split_matches_nll():
    gen = torch.Generator().manual_seed(0)
    z = torch.randn((5, 7, 4), generator=gen)
    mu = torch.randn((5, 7, 4), generator=gen)
    lv = torch.randn((5, 7, 4), generator=gen).clamp(-6.0, 6.0)
    terms = energy_terms(z, mu, lv)
    assert torch.allclose(terms["nll"], terms["residual_term"] + terms["logvar_term"])
    # residual term is non-negative; logvar term sign follows the logvar sum
    assert bool((terms["residual_term"] >= 0).all())
    manual = 0.5 * (((z - mu).pow(2) / lv.exp()).sum(dim=-1) + lv.sum(dim=-1))
    assert torch.allclose(terms["nll"], manual)


def test_energy_terms_self_copy_gives_flat_nll():
    z = torch.randn(4, 6)
    mu = z.clone()
    lv = torch.full((4, 6), -6.0)
    terms = energy_terms(z, mu, lv)
    assert bool((terms["residual_term"] == 0).all())
    # constant variance -> constant energy regardless of latent value
    assert terms["nll"].std().item() == 0.0


def test_clamp_fractions_counts_edges():
    lv = torch.tensor([[[ -6.0, -5.999, 0.0, 5.999, 6.0]]])
    valid = torch.tensor([[True]])
    out = clamp_fractions(lv, valid.unsqueeze(-1).expand_as(lv))
    assert out["n"] == 5
    assert abs(out["frac_at_lo"] - 0.4) < 1e-6
    assert abs(out["frac_at_hi"] - 0.4) < 1e-6

def test_clamp_fractions_empty():
    lv = torch.zeros((2, 3, 4))
    valid = torch.zeros((2, 3), dtype=torch.bool)
    out = clamp_fractions(lv, valid.unsqueeze(-1).expand_as(lv))
    assert out == {"n": 0, "frac_at_lo": 0.0, "frac_at_hi": 0.0}


def test_ols_recovers_identity_tracking():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = x + rng.normal(scale=1e-6, size=200)
    fit = ordinary_least_squares_slope(x, y)
    assert abs(fit["slope"] - 1.0) < 1e-3
    assert fit["r2"] > 0.99


def test_pearson_degenerate_returns_nan():
    assert math.isnan(pearson(np.ones(10), np.arange(10, dtype=float)))
    assert pearson(np.arange(10, dtype=float), np.arange(10, dtype=float)) == (
        __import__("pytest").approx(1.0)
    )

def test_centered_coupling_rejects_centroid_offset_confound():
    # mu shares z's per-dim centroids but carries no within-dim signal:
    # pooled cross-dim OLS looks strong while centered coupling is ~null.
    rng = np.random.default_rng(0)
    z = rng.normal(loc=0.0, scale=1.0, size=(300, 8))
    z = z + np.arange(8)[None, :] * 5.0  # separated per-dim centroids
    mu = np.roll(z, shift=150, axis=0)  # same centroids, unrelated rows
    pooled = ordinary_least_squares_slope(z.ravel(), mu.ravel())
    assert pooled["r2"] > 0.9  # confounded statistic looks like tracking
    out = centered_within_dim_coupling(z, mu)
    assert out["n_dims"] == 8
    assert abs(out["centered_corr_pooled"]) < 0.2  # centered view rejects it


def test_centered_coupling_keeps_true_tracking():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(300, 8)) + np.arange(8)[None, :] * 5.0
    mu = z + rng.normal(scale=1e-3, size=z.shape)
    out = centered_within_dim_coupling(z, mu)
    assert out["centered_corr_pooled"] > 0.99
    assert out["centered_slope"]["median"] == __import__("pytest").approx(1.0, abs=1e-2)


def test_file_mask_deterministic_and_partition_independent():
    valid = np.array([True] * 40 + [False] * 5)
    m1 = file_corruption_mask(45, valid, 0.25, "S-op-000001-abc", 0)
    m2 = file_corruption_mask(45, valid, 0.25, "S-op-000001-abc", 0)
    assert m1.dtype == bool and m1.shape == (45,)
    assert (m1 == m2).all()  # same file -> same support, any call order
    assert not m1[40:].any()  # never masks invalid patches
    m3 = file_corruption_mask(45, valid, 0.25, "S-op-000002-def", 0)
    assert not (m1 == m3).all()  # distinct files differ
    assert seed_from("0", "S-op-000001-abc", "mask") == seed_from("0", "S-op-000001-abc", "mask")


def test_file_direction_fixed_across_severities():
    d1 = file_direction(6, 16, "S-op-000001-abc", 0)
    d2 = file_direction(6, 16, "S-op-000001-abc", 0)
    assert d1.shape == (6, 16) and (d1 == d2).all()


def test_pooled_describe_not_mean_of_shard_medians():
    # Guards Finding 5: pool actual rows, never average shard summaries.
    rng = np.random.default_rng(2)
    shards = [rng.normal(loc=k, scale=1.0, size=50) for k in range(4)]
    pooled_median = describe(np.concatenate(shards))["median"]
    mean_of_medians = float(np.mean([describe(s)["median"] for s in shards]))
    assert pooled_median != mean_of_medians  # the bug changes the number
    # ...and pooling is invariant to how rows are grouped.
    regrouped = [np.concatenate(shards)[:100], np.concatenate(shards)[100:]]
    assert describe(np.concatenate(regrouped))["median"] == pooled_median
