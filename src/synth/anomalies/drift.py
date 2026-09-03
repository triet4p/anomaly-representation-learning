"""
Subtle Drift anomaly (DATA.md §4.5).

Trajectory-level drift: each individual timestep appears within normal
range but the overall trend is wrong.  Amplitude threshold cannot detect it.

Adapted from legacy thermal_drift but:
- Continuous severity
- Bounded to avoid exceeding signal range
- Applied post-noise to prevent noise from masking the drift
- Smooth ramp-up to avoid boundary artifact
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_subtle_drift(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Add a slow trajectory drift starting at some point in an active region.
    Drift is bounded to prevent obvious amplitude anomaly.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # Find a region to drift from
    active = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.ACTIVE, RegimeType.PERIODIC)
        and r.duration >= cfg.drift_min_duration
    ]
    if not active:
        active = [r for r in ctx.regimes if r.duration >= cfg.drift_min_duration]

    if not active:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.SUBTLE_DRIFT, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    reg = active[int(rng.integers(0, len(active)))]
    # Start drift at about 40-70% into the regime
    frac_start = rng.uniform(0.3, 0.6)
    drift_start = int(reg.start + reg.duration * frac_start)
    drift_start = max(0, min(drift_start, T - cfg.drift_min_duration))
    drift_end = T   # drift continues to end of session

    # Rate from severity
    lo, hi = cfg.drift_rate_range
    drift_rate = lo + severity * (hi - lo)
    direction = rng.choice([-1.0, 1.0])
    drift_len = drift_end - drift_start

    x_mod = x.copy()
    affected = []

    for c in range(C):
        seg = x[c, drift_start:drift_end]
        if len(seg) == 0:
            continue
        affected.append(c)

        # Local signal range and std to bound the drift
        local_range = float(np.max(x[c]) - np.min(x[c])) + 1e-4
        local_std = float(np.std(seg) + 1e-6)
        # Cap drift so MAD/orig_std stays below ~0.5 (strength gate max ~0.6)
        # MAD of linear ramp ≈ max_drift/2 → max_drift ≤ orig_std * 0.80
        max_total_drift_std = local_std * 0.80
        max_total_drift_range = local_range * 0.15   # also don't exceed 15% of range
        max_total_drift = min(max_total_drift_std, max_total_drift_range)
        actual_rate = min(drift_rate * abs(direction), max_total_drift / max(drift_len, 1)) * direction

        # Smooth ramp-in over first 20% of drift region
        ramp_steps = max(3, drift_len // 5)
        t_arr = np.arange(drift_len, dtype=float)
        drift_signal = actual_rate * t_arr
        # Ramp in
        ramp = np.minimum(t_arr / ramp_steps, 1.0)
        drift_signal *= ramp

        x_mod[c, drift_start:drift_end] = seg + drift_signal

    if not affected:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.SUBTLE_DRIFT, start=drift_start, end=drift_end,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    mask = np.zeros((C, T), dtype=bool)
    for c in affected:
        mask[c, drift_start:drift_end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.SUBTLE_DRIFT,
        start=drift_start, end=drift_end,
        severity=severity,
        affected_channels=affected,
        drift_rate=float(actual_rate),
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
