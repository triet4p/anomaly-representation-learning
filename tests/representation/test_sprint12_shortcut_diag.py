"""Focused tests for Sprint 12 Task 2 diagnostic helpers (pure, CPU-only)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from sprint12_task2_shortcut_diag import (  # noqa: E402
    clamp_fractions,
    energy_terms,
    ordinary_least_squares_slope,
    pearson,
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
