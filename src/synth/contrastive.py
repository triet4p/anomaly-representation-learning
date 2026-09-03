"""
Semantics-preserving contrastive view augmentation (DATA.md §8).

Two positive views from the same file:  X → X⁽¹⁾, X⁽²⁾

Augmentations preserve behavioral semantics (regime structure, anomaly
identity) while creating representation diversity:

  - small per-channel multiplicative gain jitter around each channel baseline
  - small per-channel additive offset jitter
  - tiny additional white noise (fraction of signal std)
  - very small edge-preserving temporal shift

Prohibited: augmentations that would change regime sequence, transition
timing, or anomaly semantics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.config import ContrastiveConfig
from synth.schema import FileSample


def make_contrastive_views(
    sample: FileSample,
    cfg: ContrastiveConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Create two augmented views of sample.x.

    Returns (view1, view2) each as float32 [C, T].
    Both views share the same anomaly semantics as the original.
    """
    x = sample.x.astype(np.float64)  # [C, T]
    C, T = x.shape

    view1 = _augment(x, cfg, rng)
    view2 = _augment(x, cfg, rng)

    return view1.astype(np.float32), view2.astype(np.float32)


def _augment(
    x: npt.NDArray[np.float64],
    cfg: ContrastiveConfig,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Apply one set of augmentations to x [C, T]."""
    C, T = x.shape
    result = x.copy()

    # ── Per-channel gain jitter ───────────────────────────────────────
    # Scale the dynamic component around each channel's level rather than
    # scaling its physical DC level.  This keeps high-offset sensors (for
    # example, arc voltage) within a comparable perturbation budget.
    ch_baselines = np.mean(x, axis=1)
    gain = 1.0 + rng.normal(0, cfg.gain_std, C)
    gain = np.clip(gain, 0.90, 1.10)    # hard limits to prevent extreme gain
    result = ch_baselines[:, None] + (result - ch_baselines[:, None]) * gain[:, None]

    # ── Per-channel additive offset ────────────────────────────────────
    # Offset is scaled by per-channel std to be amplitude-relative
    ch_stds = np.std(x, axis=1) + 1e-8
    offsets = rng.normal(0, cfg.offset_std, C) * ch_stds
    result = result + offsets[:, None]

    # ── Additional white noise ────────────────────────────────────────
    if cfg.noise_std > 0:
        noise_scale = cfg.noise_std * ch_stds
        noise = rng.normal(0, 1, (C, T)) * noise_scale[:, None]
        result = result + noise

    # ── Very small temporal shift (edge-preserving) ───────────────────
    # Do not wrap the final samples to the beginning: a circular roll would
    # splice unrelated end/start regimes and alter anomaly context.  Replicate
    # the nearest endpoint instead, preserving session semantics.
    if cfg.max_shift > 0 and T > 1:
        max_shift = min(int(cfg.max_shift), T - 1)
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift > 0:
            shifted = result.copy()
            shifted[:, shift:] = result[:, :-shift]
            shifted[:, :shift] = result[:, shift:shift + 1]
            result = 0.25 * shifted + 0.75 * result
        elif shift < 0:
            n = -shift
            shifted = result.copy()
            shifted[:, :-n] = result[:, n:]
            shifted[:, -n:] = result[:, -n - 1:-n]
            result = 0.25 * shifted + 0.75 * result
    return result
