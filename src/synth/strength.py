"""
Intervention strength measurement and accept/reject gate (DATA.md §5).

Pipeline:
  generate → measure intervention strength → too weak? reject
                                           → too strong? reject
                                           → accept

Strength is a composite of two metrics so it works across all anomaly families:

  metric_1 = mean|x_mod - x_orig| / (std(x_orig in region) + 1e-6)
             (MAD-based — catches spike, drift, phase, lag, gain anomalies)

  metric_2 = |σ_orig - σ_mod| / (σ_orig + 1e-6)
             (std-ratio change — catches stuck, over-regularity anomalies)

  strength = max(metric_1, metric_2)

This ensures that anomalies that change either the signal level OR the
variance are correctly measured, and that pure variance-reduction anomalies
(stuck, over-regularity) are not erroneously rejected.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import AnomalyResult
from synth.config import AnomalyConfig


def composite_strength(
    x_orig: npt.NDArray[np.float64],
    x_mod: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> float:
    """
    Composite intervention strength — computed per channel, then aggregated.

    For each channel c that has any masked timesteps:
      mad_c    = mean|x_mod - x_orig| / (std(x_orig in region) + 1e-6)
      std_ch_c = |σ_orig_c - σ_mod_c| / (σ_orig_c + 1e-6)

    strength = mean over affected channels of max(mad_c, std_ch_c)

    Per-channel computation prevents large-magnitude channels (e.g. temperature
    in °C) from dominating the metric for small-magnitude channels (feed speed).
    """
    if not mask.any():
        return 0.0

    C = x_orig.shape[0]
    # Duration anomalies can legitimately change T.  Interpolate the clean
    # reference to the modified timeline so the exact output mask remains
    # usable without silently dropping the intervention from the gate.
    if x_orig.shape[1] != x_mod.shape[1]:
        src_t = np.linspace(0.0, 1.0, x_orig.shape[1])
        dst_t = np.linspace(0.0, 1.0, x_mod.shape[1])
        orig_aligned = np.vstack([
            np.interp(dst_t, src_t, x_orig[c]) for c in range(C)
        ])
    else:
        orig_aligned = x_orig
    channel_strengths = []

    for c in range(C):
        ch_mask_1d = mask[c]   # [T] bool
        if not ch_mask_1d.any():
            continue
        orig_seg = orig_aligned[c][ch_mask_1d]
        mod_seg  = x_mod[c][ch_mask_1d]

        orig_std = float(np.std(orig_seg) + 1e-6)
        mod_std  = float(np.std(mod_seg)  + 1e-6)

        # Metric 1: normalised MAD
        mad = float(np.mean(np.abs(mod_seg - orig_seg))) / orig_std
        # Metric 2: relative std change (variance-reduction anomalies)
        std_change = abs(orig_std - mod_std) / orig_std
        channel_strengths.append(max(mad, std_change))

    if not channel_strengths:
        return 0.0
    return float(np.mean(channel_strengths))


class StrengthGate:
    """
    Evaluates whether an anomaly injection has appropriate strength.

    Acceptance criterion (DATA.md §5):
        weak_min  ≤ composite_strength(x_orig, x_mod, mask) ≤ strong_max

    Both limits are read from AnomalyConfig; callers can override per-call.
    """

    def __init__(self, cfg: AnomalyConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        x_orig: npt.NDArray[np.float64],
        result: AnomalyResult,
        weak_min: float | None = None,
        strong_max: float | None = None,
    ) -> tuple[bool, float]:
        """
        Returns (accepted, strength).

        accepted=True  → strength is in [weak_min, strong_max]
        accepted=False → too weak or too strong
        """
        wmin = weak_min if weak_min is not None else self.cfg.strength_weak_min
        smax = strong_max if strong_max is not None else self.cfg.strength_strong_max
        strength = composite_strength(x_orig, result.x_modified, result.mask)
        accepted = wmin <= strength <= smax
        return accepted, strength


def generate_with_strength_gate(
    generate_fn,            # Callable[[], AnomalyResult]
    x_orig: npt.NDArray[np.float64],
    cfg: AnomalyConfig,
    weak_min: float | None = None,
    strong_max: float | None = None,
) -> tuple[AnomalyResult, float, int]:
    """
    Repeatedly call ``generate_fn()`` until the strength gate accepts,
    or until max_rejection_attempts is reached.

    Returns (result, strength, n_attempts).
    On exhaustion, returns the last generated result with accepted=False.
    """
    gate = StrengthGate(cfg)
    wmin = weak_min if weak_min is not None else cfg.strength_weak_min
    smax = strong_max if strong_max is not None else cfg.strength_strong_max

    last_result = None
    last_strength = 0.0
    for attempt in range(1, cfg.max_rejection_attempts + 1):
        result = generate_fn()
        if not result.accepted:
            # Injector itself declared rejection (e.g. degenerate session)
            last_result = result
            last_strength = 0.0
            continue
        accepted, strength = gate.evaluate(x_orig, result, wmin, smax)
        last_result = result
        last_strength = strength
        if accepted:
            result.accepted = True
            return result, strength, attempt
        result.accepted = False

    # Return last attempt as rejected
    if last_result is not None:
        last_result.accepted = False
    return last_result, last_strength, cfg.max_rejection_attempts
