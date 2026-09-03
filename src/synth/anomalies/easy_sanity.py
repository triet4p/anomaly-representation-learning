"""
Easy sanity anomalies (DATA.md §4.1).

These are NOT hard-benchmark defaults.  They are included only when
``AnomalyConfig.include_easy_sanity=True`` and are clearly labelled.
They serve as sanity-check targets to verify the detection pipeline works
at all — if the system can't detect these, nothing harder will work.

  easy_spike:    amplitude 4× local amplitude for 5 steps
  easy_flatline: exact zero (or near-zero) for 40–60 steps
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_easy_spike(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """Sharp spike at 4× local amplitude.  Easy to detect via threshold."""
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    duration = cfg.easy_spike_duration
    active = [r for r in ctx.regimes if r.regime == RegimeType.ACTIVE and r.duration > duration + 10]
    if active:
        reg = active[int(rng.integers(0, len(active)))]
        start = int(rng.integers(reg.start + 5, reg.end - duration - 5))
    else:
        start = int(rng.integers(5, max(6, T - duration - 5)))
    end = min(start + duration, T)

    x_mod = x.copy()
    affected = list(range(C))
    for c in range(C):
        local_amp = float(np.std(x[c, max(0, start - 20):start + 20]) + 1e-4)
        spike = cfg.easy_spike_amplitude * local_amp * float(rng.choice([-1.0, 1.0]))
        x_mod[c, start:end] += spike

    mask = np.zeros((C, T), dtype=bool)
    mask[:, start:end] = True
    meta = AnomalyMeta(
        family=AnomalyFamily.EASY_SPIKE, start=start, end=end,
        severity=1.0, affected_channels=affected,
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)


def inject_easy_flatline(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """Exact-zero flatline.  Easy to detect via amplitude threshold."""
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    lo_d, hi_d = cfg.easy_flatline_duration_range
    duration = int(rng.integers(lo_d, hi_d + 1))
    start = int(rng.integers(5, max(6, T - duration - 5)))
    end = min(start + duration, T)

    x_mod = x.copy()
    affected = list(range(C))
    for c in range(C):
        x_mod[c, start:end] = 0.0

    mask = np.zeros((C, T), dtype=bool)
    mask[:, start:end] = True
    meta = AnomalyMeta(
        family=AnomalyFamily.EASY_FLATLINE, start=start, end=end,
        severity=1.0, affected_channels=affected,
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
