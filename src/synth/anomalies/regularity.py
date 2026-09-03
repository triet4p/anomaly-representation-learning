"""
Over-Regularity anomaly (DATA.md §4.3 / §4.4).

Signal stays within correct range and regime but becomes unnaturally
smooth: phase jitter disappears, cycle-to-cycle variation collapses,
waveform becomes too clean.

This is anomalous because it is EASIER to reconstruct than normal,
not harder — the exact failure mode that reconstruction-centric models miss.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_over_regularity(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Suppress all high-frequency jitter and cycle-to-cycle variation in
    a region, making the signal unnaturally smooth while keeping mean/range.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # Target: active or periodic region with enough duration
    candidate = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.ACTIVE, RegimeType.PERIODIC)
        and r.duration >= cfg.reg_min_duration
    ]
    if not candidate:
        candidate = [r for r in ctx.regimes if r.duration >= cfg.reg_min_duration]

    if not candidate:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.OVER_REGULARITY, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    reg = candidate[int(rng.integers(0, len(candidate)))]
    dur = max(cfg.reg_min_duration, int(reg.duration * severity * 0.8))
    start = int(rng.integers(reg.start, max(reg.start + 1, reg.end - dur)))
    end = min(start + dur, T)
    dur = end - start

    # Residual jitter fraction after smoothing (severity → less jitter)
    lo, hi = cfg.reg_jitter_suppress_range
    jitter_fraction = lo + (1 - severity) * (hi - lo)  # low severity → more jitter left

    x_mod = x.copy()
    affected = list(range(C))

    for c in range(C):
        seg = x[c, start:end].copy()
        seg_mean = float(np.mean(seg))

        # Low-pass smooth the segment — use a rolling mean
        kernel_size = max(3, int(dur * 0.15))
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(seg, size=kernel_size, mode="reflect")

        # Blend between smooth and original: severity controls how smooth
        alpha = 1.0 - jitter_fraction   # high alpha → more smoothing
        new_seg = alpha * smooth + (1 - alpha) * seg

        # Add tiny residual jitter (not zero) to avoid exact constant
        residual = rng.normal(0, float(np.std(seg)) * jitter_fraction * 0.3, dur)
        new_seg = new_seg + residual

        # Boundary blend
        blend_n = min(5, dur // 4)
        if blend_n > 0 and start > 0:
            t = np.linspace(0, 1, blend_n)
            alpha_b = (1 - np.cos(np.pi * t)) / 2
            new_seg[:blend_n] = x[c, start - 1] * (1 - alpha_b) + new_seg[:blend_n] * alpha_b
        if blend_n > 0 and end < T:
            t = np.linspace(0, 1, blend_n)
            alpha_b = (1 - np.cos(np.pi * t)) / 2
            new_seg[-blend_n:] = new_seg[-blend_n:] * (1 - alpha_b) + x[c, end] * alpha_b

        x_mod[c, start:end] = new_seg

    mask = np.zeros((C, T), dtype=bool)
    mask[:, start:end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.OVER_REGULARITY,
        start=start, end=end, severity=severity,
        affected_channels=affected,
        extra={"jitter_fraction": jitter_fraction},
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
