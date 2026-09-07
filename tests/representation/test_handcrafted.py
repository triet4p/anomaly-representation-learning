"""Focused tests for handcrafted patch features (mask/padding/leakage faults)."""

import numpy as np

from representation.handcrafted import (
    batch_patch_features,
    feature_dim,
    feature_names,
    patch_features,
)


def test_dim_and_names_agree():
    assert feature_dim(6) == 85
    assert len(feature_names(6)) == 85
    assert len(set(feature_names(6))) == 85


def test_padded_steps_do_not_leak_into_features():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(6, 32))
    pad = np.zeros(32, dtype=bool)
    pad[24:] = True
    poisoned = values.copy()
    poisoned[:, 24:] = 1e6  # padded region must be invisible
    a = patch_features(values, pad)
    b = patch_features(poisoned, pad)
    assert np.array_equal(a, b)


def test_fully_padded_patch_is_finite_zeros():
    out = patch_features(np.zeros((6, 32)), np.ones(32, dtype=bool))
    assert out.shape == (85,)
    assert np.isfinite(out).all()
    assert (out == 0.0).all()


def test_degenerate_single_step_is_finite():
    values = np.full((6, 32), 3.0)
    pad = np.ones(32, dtype=bool)
    pad[0] = False
    out = patch_features(values, pad)
    assert np.isfinite(out).all()
    assert out[0] == 3.0  # ch0 mean sees the single valid step


def test_constant_patch_has_zero_spread_and_corr():
    values = np.full((6, 32), 2.0)
    out = patch_features(values, np.zeros(32, dtype=bool))
    names = feature_names(6)
    stds = out[[names.index(f"ch{c}_std") for c in range(6)]]
    slopes = out[[names.index(f"ch{c}_slope") for c in range(6)]]
    assert (stds == 0.0).all()
    assert (slopes == 0.0).all()
    assert out[names.index("corr_ch0_ch1")] == 0.0


def test_slope_sign_and_level_tracking():
    w = 32
    ramp = np.tile(np.linspace(-1.0, 1.0, w), (6, 1))
    out = patch_features(ramp, np.zeros(w, dtype=bool))
    names = feature_names(6)
    assert out[names.index("ch0_slope")] > 0.0
    assert out[names.index("ch0_mean")] == __import__("pytest").approx(0.0, abs=1e-12)


def test_batch_shape_and_finite():
    rng = np.random.default_rng(1)
    patches = rng.normal(size=(5, 6, 32))
    pad = np.zeros((5, 32), dtype=bool)
    pad[3, 20:] = True
    feats = batch_patch_features(patches, pad)
    assert feats.shape == (5, 85)
    assert np.isfinite(feats).all()


def test_batch_rejects_shape_mismatch():
    import pytest

    with pytest.raises(ValueError):
        batch_patch_features(np.zeros((4, 6, 32)), np.zeros((4, 31), dtype=bool))

def test_standardizer_fit_apply_no_leakage():
    from representation.handcrafted import Standardizer

    rng = np.random.default_rng(3)
    train = rng.normal(loc=5.0, scale=2.0, size=(100, 4))
    std = Standardizer.fit(train)
    # frozen params come from train rows only
    assert np.allclose(std.center, train.mean(axis=0))
    assert np.allclose(std.scale, train.std(axis=0))
    # eval rows far outside train range transform with TRAIN stats, not their own
    eval_rows = np.full((10, 4), 100.0)
    out = std.apply(eval_rows)
    assert np.allclose(out, (100.0 - train.mean(axis=0)) / train.std(axis=0))
    assert np.isfinite(out).all()


def test_standardizer_zero_scale_finite_policy():
    from representation.handcrafted import Standardizer

    rows = np.ones((20, 3))
    rows[:, 0] = np.arange(20, dtype=float)
    std = Standardizer.fit(rows)
    assert std.scale[1] == 1.0 and std.scale[2] == 1.0
    out = std.apply(rows)
    assert np.isfinite(out).all()
    assert (out[:, 1:] == 0.0).all()  # constant dims center to exact zero


def test_standardizer_rejects_refit_shapes_and_nonfinite():
    import pytest

    from representation.handcrafted import Standardizer

    std = Standardizer.fit(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        std.apply(np.zeros((5, 5)))  # width mismatch, not silent truncation
    with pytest.raises(ValueError):
        Standardizer.fit(np.zeros((0, 4)))
