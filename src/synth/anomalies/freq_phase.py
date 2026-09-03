"""
Frequency / Phase Mismatch anomaly (DATA.md §4.6).

Frequency or phase of a periodic component is slightly wrong.
Hard mode: only one or two channels affected, severity moderate,
amplitude and local statistics near normal.

Conceptually related to legacy archived phase_shift.py but:
- Continuous severity via blend fraction
- Affects only a subset of channels
- Causal consistency maintained (S0 modified, S1/S2 recomputed if needed)
- No boundary artifacts
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_freq_phase_mismatch(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    In a periodic region, substitute a version with slightly wrong
    frequency or phase.  Amplitude is preserved; local stats remain similar.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # Find a periodic or active region with some oscillation structure
    periodic_regs = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.PERIODIC, RegimeType.ACTIVE)
        and r.duration >= cfg.fp_min_duration
    ]
    if not periodic_regs:
        # Try any sufficiently dynamic region
        periodic_regs = [r for r in ctx.regimes if r.duration >= cfg.fp_min_duration]

    if not periodic_regs:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.FREQ_PHASE_MISMATCH, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    reg = periodic_regs[int(rng.integers(0, len(periodic_regs)))]
    dur = max(cfg.fp_min_duration, int(reg.duration * min(0.8, severity * 1.2)))
    start = int(rng.integers(reg.start, max(reg.start + 1, reg.end - dur)))
    end = min(start + dur, T)
    dur = end - start

    # Affect a subset of channels (not all — hard mode)
    n_aff = int(rng.integers(1, C + 1))
    affected = sorted(rng.choice(C, size=n_aff, replace=False).tolist())

    # Choose distortion type
    mode = rng.choice(["phase", "freq"])
    x_mod = x.copy()

    for c in affected:
        seg = x[c, start:end].copy()
        seg_mean = float(np.mean(seg))
        seg_demean = seg - seg_mean
        n = len(seg_demean)
        if n < 4:
            continue

        if mode == "phase":
            # Shift the signal in the frequency domain
            lo_p, hi_p = cfg.phase_delta_range
            phase_delta = rng.uniform(lo_p, hi_p) * np.pi * severity
            # Apply phase shift via FFT
            fft = np.fft.rfft(seg_demean)
            freqs = np.fft.rfftfreq(n)
            # Shift higher-frequency components more
            phase_shift = phase_delta * (freqs / (np.max(freqs) + 1e-8))
            fft_shifted = fft * np.exp(1j * phase_shift * 2 * np.pi)
            shifted = np.fft.irfft(fft_shifted, n=n)
            new_seg = shifted + seg_mean

        else:  # freq
            # Stretch time-axis to change effective frequency
            lo_f, hi_f = cfg.freq_delta_range
            freq_delta = 1.0 + rng.uniform(lo_f, hi_f) * severity * rng.choice([-1, 1])
            t_orig = np.arange(n)
            t_new = np.clip(t_orig / freq_delta, 0, n - 1)
            new_seg = np.interp(t_orig, t_new, seg)

        # Blend back by severity (lower severity → more blend with original)
        blend_alpha = severity * 0.8
        blended = blend_alpha * new_seg + (1 - blend_alpha) * seg

        # Preserve amplitude envelope
        orig_env = np.abs(seg_demean) + 1e-8
        new_env = np.abs(blended - seg_mean) + 1e-8
        blended = seg_mean + (blended - seg_mean) * (orig_env / new_env)

        # Boundary cosine blend
        blend_n = min(5, dur // 5)
        if blend_n > 0 and start > 0:
            t_b = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t_b)) / 2
            blended[:blend_n] = x[c, start - 1] * (1 - ab) + blended[:blend_n] * ab
        if blend_n > 0 and end < T:
            t_b = np.linspace(0, 1, blend_n)
            ab = (1 - np.cos(np.pi * t_b)) / 2
            blended[-blend_n:] = blended[-blend_n:] * (1 - ab) + x[c, end] * ab

        x_mod[c, start:end] = blended

    mask = np.zeros((C, T), dtype=bool)
    for c in affected:
        mask[c, start:end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.FREQ_PHASE_MISMATCH,
        start=start, end=end, severity=severity,
        affected_channels=affected,
        phase_shift=float(phase_delta) if mode == "phase" else None,
        extra={"mode": mode},
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
