"""
Realistic Stuck anomaly (DATA.md §4.3).

Reduces dynamics in a region WITHOUT using exact constants.

σ_abnormal ≈ 0.3–0.6 × σ_expected.

Residual noise, slow drift, and small AR(1) dynamics are preserved.
NOT a flatline. The easy_sanity version (exact zero) is in easy_sanity.py.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_realistic_stuck(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Reduce channel dynamics in an active region.  σ_abn ≈ 0.3–0.6 × σ_expected.
    Residual noise and slow drift are added to avoid exact constant.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # ── Choose target region in an active/periodic regime ────────────
    active_regs = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.ACTIVE, RegimeType.PERIODIC)
        and r.duration >= cfg.stuck_min_duration
    ]
    if not active_regs:
        active_regs = [r for r in ctx.regimes if r.duration >= cfg.stuck_min_duration]

    if not active_regs:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.REALISTIC_STUCK, start=0, end=0,
            severity=severity, affected_channels=list(range(C)),
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    reg = active_regs[int(rng.integers(0, len(active_regs)))]
    dur = int(rng.integers(
        min(cfg.stuck_min_duration, reg.duration),
        min(cfg.stuck_max_duration, reg.duration) + 1,
    ))
    start = int(rng.integers(reg.start, max(reg.start + 1, reg.end - dur)))
    end = min(start + dur, reg.end, T)
    dur = end - start

    # ── σ ratio: interpolate from severity ───────────────────────────
    lo, hi = cfg.stuck_sigma_ratio_range
    sigma_ratio = hi - severity * (hi - lo)   # high severity → lower sigma
    sigma_ratio = float(np.clip(sigma_ratio, lo, hi))

    x_mod = x.copy()
    affected = []

    # Apply to channels where the region is dynamic
    for c in range(C):
        seg = x[c, start:end]
        seg_mean = float(np.mean(seg))
        seg_std = float(np.std(seg))
        if seg_std < 1e-4:
            continue   # already flat — skip channel
        affected.append(c)

        # New signal: mean stays, variance suppressed
        target_std = sigma_ratio * seg_std

        # Suppress dynamics: scale deviations around the segment mean
        new_seg = seg_mean + (seg - seg_mean) * (target_std / seg_std)

        # Add residual noise: small AR(1)
        phi = 0.7
        residual_std = target_std * 0.3
        noise = np.zeros(dur)
        innov = rng.normal(0, residual_std, dur)
        noise[0] = innov[0]
        for i in range(1, dur):
            noise[i] = phi * noise[i - 1] + innov[i]

        # Add very slow drift
        drift_amp = seg_std * 0.05 * severity
        drift = drift_amp * np.linspace(-1, 1, dur) * rng.uniform(-1, 1)

        # Cosine blend at boundaries
        blend_n = min(6, dur // 4)
        if blend_n > 0 and start > 0:
            t = np.linspace(0, 1, blend_n)
            alpha = (1 - np.cos(np.pi * t)) / 2
            new_seg[:blend_n] = x[c, start - 1] * (1 - alpha) + new_seg[:blend_n] * alpha
        if blend_n > 0 and end < T:
            t = np.linspace(0, 1, blend_n)
            alpha = (1 - np.cos(np.pi * t)) / 2
            new_seg[-blend_n:] = new_seg[-blend_n:] * (1 - alpha) + x[c, end] * alpha

        x_mod[c, start:end] = new_seg + noise + drift

    if not affected:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.REALISTIC_STUCK, start=start, end=end,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    mask = np.zeros((C, T), dtype=bool)
    for c in affected:
        mask[c, start:end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.REALISTIC_STUCK,
        start=start, end=end, severity=severity,
        affected_channels=affected,
        stuck_sigma_ratio=sigma_ratio,
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
