"""Interpretable valid-patch telemetry features (Sprint 12 Task 3).

Fixed-dimension, pad-aware, deterministic patch descriptors over observable
telemetry: per-channel level/RMS/variance/slope/extremes, cross-channel
correlation and mean-difference structure, spectral-band energy, transitions,
and valid duration. Statistics consume valid steps only; fully-padded patches
yield a finite all-zero vector. Labels and masks never enter this module.
"""

from __future__ import annotations

import numpy as np

N_CHANNEL_FEATURES = ("mean", "std", "rms", "slope", "min", "max")
SPECTRAL_BANDS = ("low", "high")


def feature_dim(n_channels: int) -> int:
    """Fixed descriptor width for a channel count."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    n_pair = n_channels * (n_channels - 1) // 2
    return 6 * n_channels + 2 * n_pair + 2 * n_channels + n_channels + 1


def feature_names(n_channels: int) -> list[str]:
    """Human-readable name per descriptor column (interpretability)."""
    names: list[str] = []
    for kind in N_CHANNEL_FEATURES:
        names += [f"ch{c}_{kind}" for c in range(n_channels)]
    for a in range(n_channels):
        for b in range(a + 1, n_channels):
            names.append(f"corr_ch{a}_ch{b}")
    for a in range(n_channels):
        for b in range(a + 1, n_channels):
            names.append(f"meandiff_ch{a}_ch{b}")
    for band in SPECTRAL_BANDS:
        names += [f"ch{c}_spec_{band}" for c in range(n_channels)]
    names += [f"ch{c}_maxstep" for c in range(n_channels)]
    names.append("valid_fraction")
    assert len(names) == feature_dim(n_channels)
    return names


def _slope(steps: np.ndarray) -> float:
    n = steps.shape[0]
    if n < 2 or not np.isfinite(steps).all():
        return 0.0
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    denom = float((t * t).sum())
    if denom == 0.0:
        return 0.0
    return float((t * (steps - steps.mean())).sum() / denom)


def _pairwise_corr(valid: np.ndarray) -> np.ndarray:
    """Upper-triangle cross-channel Pearson correlation (0 on degeneracy)."""
    c = valid.shape[0]
    out = np.zeros((c * (c - 1) // 2,), dtype=np.float64)
    k = 0
    for a in range(c):
        for b in range(a + 1, c):
            xa = valid[a] - valid[a].mean()
            xb = valid[b] - valid[b].mean()
            denom = float(np.sqrt((xa * xa).sum() * (xb * xb).sum()))
            out[k] = float((xa * xb).sum() / denom) if denom > 0.0 else 0.0
            k += 1
    return out


def _spectral_bands(valid: np.ndarray) -> np.ndarray:
    """Log1p power in low/high halves of the one-sided spectrum, per channel."""
    c, n = valid.shape
    out = np.zeros((2 * c,), dtype=np.float64)
    if n < 2:
        return out
    spec = np.abs(np.fft.rfft(valid, axis=1)) ** 2
    cut = max(1, spec.shape[1] // 2)
    out[:c] = np.log1p(spec[:, :cut].sum(axis=1))
    out[c:] = np.log1p(spec[:, cut:].sum(axis=1))
    return out


def patch_features(
    values: np.ndarray, pad_mask: np.ndarray
) -> np.ndarray:
    """Describe one patch ``[C, W]`` with boolean pad mask ``[W]`` (True=padded)."""
    values = np.asarray(values, dtype=np.float64)
    pad_mask = np.asarray(pad_mask, dtype=bool)
    if values.ndim != 2 or pad_mask.ndim != 1 or values.shape[1] != pad_mask.shape[0]:
        raise ValueError("values must be [C, W] with pad_mask [W]")
    c, w = values.shape
    dim = feature_dim(c)
    real = values[:, ~pad_mask]
    n_v = real.shape[1]
    out = np.zeros((dim,), dtype=np.float64)
    if n_v == 0:
        return out
    pos = 0
    means = real.mean(axis=1)
    stds = real.std(axis=1) if n_v > 1 else np.zeros(c)
    out[pos:pos + c] = means
    pos += c
    out[pos:pos + c] = stds
    pos += c
    out[pos:pos + c] = np.sqrt((real * real).mean(axis=1))
    pos += c
    out[pos:pos + c] = np.array([_slope(real[i]) for i in range(c)])
    pos += c
    out[pos:pos + c] = real.min(axis=1)
    pos += c
    out[pos:pos + c] = real.max(axis=1)
    pos += c
    n_pair = c * (c - 1) // 2
    out[pos:pos + n_pair] = _pairwise_corr(real) if n_v > 1 else 0.0
    pos += n_pair
    k = 0
    for a in range(c):
        for b in range(a + 1, c):
            out[pos + k] = abs(float(means[a] - means[b]))
            k += 1
    pos += n_pair
    out[pos:pos + 2 * c] = _spectral_bands(real)
    pos += 2 * c
    if n_v > 1:
        out[pos:pos + c] = np.abs(np.diff(real, axis=1)).max(axis=1)
    pos += c
    out[pos] = n_v / w
    finite = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert finite.shape == (dim,)
    return finite


def batch_patch_features(
    patches: np.ndarray, patch_pad_mask: np.ndarray
) -> np.ndarray:
    """Describe ``[N, C, W]`` patches with ``[N, W]`` pad masks → ``[N, D]``."""
    patches = np.asarray(patches)
    patch_pad_mask = np.asarray(patch_pad_mask, dtype=bool)
    if patches.ndim != 3 or patch_pad_mask.ndim != 2:
        raise ValueError("patches must be [N, C, W] with patch_pad_mask [N, W]")
    if patches.shape[0] != patch_pad_mask.shape[0] or patches.shape[2] != patch_pad_mask.shape[1]:
        raise ValueError("patch/pad batch shapes disagree")
    n, c, _ = patches.shape
    feats = np.zeros((n, feature_dim(c)), dtype=np.float64)
    for i in range(n):
        feats[i] = patch_features(patches[i], patch_pad_mask[i])
    return feats
