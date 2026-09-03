"""
Cross-Channel Inconsistency anomaly (DATA.md §4.7).

Each channel individually looks normal but the relation between channels
is broken: lag, gain, or phase-shift of one channel relative to another.

Critical constraint: ONLY inject into regions with sufficient dynamics
and inter-channel correlation — idle/flat regions are excluded.

Causal consistency is maintained by modifying only one channel's
relationship to the others, using a valid-trajectory from a different
time point rather than pure noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import (
    InjectionContext, AnomalyResult, find_dynamic_region
)
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily


def inject_cross_channel_inconsistency(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Break the relationship between channels while keeping each channel
    individually plausible.  Three sub-modes:
      - lag:   one channel is time-lagged relative to expected
      - gain:  gain relation between channels is wrong
      - phase: phase relation shifted in the frequency domain
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # ── Find dynamic region where cross-channel relation is meaningful ──
    # Use channel 0 (feed speed) to detect dynamics
    threshold = cfg.cc_dynamics_threshold * float(np.std(x[0]) + 1e-8)
    dyn_regions = find_dynamic_region(x, channel=0, min_local_std=threshold, min_duration=20)

    if not dyn_regions:
        # Relax: use any region with std > small threshold
        threshold_fallback = float(np.std(x[0]) * 0.05 + 1e-6)
        dyn_regions = find_dynamic_region(
            x, channel=0, min_local_std=threshold_fallback, min_duration=10,
        )

    if not dyn_regions:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.CROSS_CHANNEL_INCONSISTENCY, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    # Pick a dynamic region
    reg_idx = int(rng.integers(0, len(dyn_regions)))
    dyn_start, dyn_end = dyn_regions[reg_idx]

    # Intervention length: severity-scaled fraction of dynamic region
    dyn_len = dyn_end - dyn_start
    anom_len = max(15, int(dyn_len * severity * 0.7))
    anom_start = int(rng.integers(dyn_start, max(dyn_start + 1, dyn_end - anom_len)))
    anom_end = min(anom_start + anom_len, T)
    anom_len = anom_end - anom_start

    # ── Choose mode and target channel ───────────────────────────────
    mode = rng.choice(["lag", "gain", "phase"])
    # Pick one channel to distort (not all — individual channel plausibility)
    target_ch = int(rng.integers(0, C))
    affected = [target_ch]

    x_mod = x.copy()
    blend_n = min(6, anom_len // 4)

    if mode == "lag":
        lo_lag, hi_lag = cfg.cc_lag_range
        lag = int(rng.integers(lo_lag, hi_lag + 1))
        lag = min(lag, anom_start)   # cannot lag past beginning

        # Replace target channel in anomaly region with a lagged version
        lag_start = max(0, anom_start - lag)
        lag_source = x[target_ch, lag_start:lag_start + anom_len]
        if len(lag_source) < anom_len:
            pad_len = anom_len - len(lag_source)
            lag_source = np.concatenate([np.full(pad_len, lag_source[0] if len(lag_source) else 0.0), lag_source])

        seg = lag_source[:anom_len].copy()
        # Blend boundaries
        if blend_n > 0 and anom_start > 0:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg[:blend_n] = x[target_ch, anom_start - 1] * (1 - ab) + seg[:blend_n] * ab
        if blend_n > 0 and anom_end < T:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg[-blend_n:] = seg[-blend_n:] * (1 - ab) + x[target_ch, anom_end] * ab
        x_mod[target_ch, anom_start:anom_end] = seg
        lag_val = float(lag)
        gain_val, phase_val = None, None

    elif mode == "gain":
        lo_g, hi_g = cfg.cc_gain_delta_range
        gain_delta = rng.uniform(lo_g, hi_g) * severity
        sign = rng.choice([-1.0, 1.0])
        gain_factor = 1.0 + sign * gain_delta

        seg = x[target_ch, anom_start:anom_end].copy()
        seg_mean = float(np.mean(seg))
        # Rescale around local mean
        seg_mod = seg_mean + (seg - seg_mean) * gain_factor
        # Boundary blend
        if blend_n > 0 and anom_start > 0:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg_mod[:blend_n] = x[target_ch, anom_start - 1] * (1 - ab) + seg_mod[:blend_n] * ab
        if blend_n > 0 and anom_end < T:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg_mod[-blend_n:] = seg_mod[-blend_n:] * (1 - ab) + x[target_ch, anom_end] * ab
        x_mod[target_ch, anom_start:anom_end] = seg_mod
        gain_val = float(gain_factor)
        lag_val, phase_val = None, None

    else:  # phase
        lo_ph, hi_ph = cfg.cc_phase_delta_range
        phase_delta = rng.uniform(lo_ph, hi_ph) * np.pi * severity

        seg = x[target_ch, anom_start:anom_end].copy()
        seg_mean = float(np.mean(seg))
        seg_demean = seg - seg_mean
        n = anom_len
        if n >= 4:
            fft = np.fft.rfft(seg_demean)
            freqs = np.fft.rfftfreq(n)
            fft_shifted = fft * np.exp(1j * phase_delta)
            shifted = np.fft.irfft(fft_shifted, n=n)
            # Preserve local stats
            orig_std = float(np.std(seg_demean) + 1e-8)
            new_std = float(np.std(shifted) + 1e-8)
            seg_mod = seg_mean + shifted * (orig_std / new_std)
        else:
            seg_mod = seg

        # Boundary blend
        if blend_n > 0 and anom_start > 0:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg_mod[:blend_n] = x[target_ch, anom_start - 1] * (1 - ab) + seg_mod[:blend_n] * ab
        if blend_n > 0 and anom_end < T:
            t = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t)) / 2
            seg_mod[-blend_n:] = seg_mod[-blend_n:] * (1 - ab) + x[target_ch, anom_end] * ab
        x_mod[target_ch, anom_start:anom_end] = seg_mod
        phase_val = float(phase_delta)
        lag_val, gain_val = None, None

    mask = np.zeros((C, T), dtype=bool)
    mask[target_ch, anom_start:anom_end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.CROSS_CHANNEL_INCONSISTENCY,
        start=anom_start, end=anom_end, severity=severity,
        affected_channels=affected,
        lag=int(lag_val) if lag_val is not None else None,
        gain_change=gain_val,
        phase_shift=phase_val,
        extra={"mode": mode},
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
