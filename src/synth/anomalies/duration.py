"""
Duration Anomaly (DATA.md §4.8).

A regime lasts much too long or too short.

NOT patch-tiling — actual time-stretch/compress via interpolation.
The signal waveform is physically resampled so it looks like the same
process running slower or faster, then inserted back at the correct position.
Surrounding regimes are adjusted accordingly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_duration_anomaly(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """Time-stretch or compress one regime and update all metadata."""
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape
    candidates = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.ACTIVE, RegimeType.PERIODIC)
        and r.duration >= cfg.dur_min_regime_steps
    ] or [r for r in ctx.regimes if r.duration >= cfg.dur_min_regime_steps]
    if not candidates:
        empty = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.DURATION_ANOMALY, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=empty, meta=meta, accepted=False)

    reg = candidates[int(rng.integers(0, len(candidates)))]
    start, end = int(reg.start), int(reg.end)
    original_dur = end - start
    mode = str(rng.choice(["stretch", "compress"]))
    if mode == "stretch":
        lo, hi = cfg.dur_stretch_range
    else:
        lo, hi = cfg.dur_compress_range
    factor = float(lo + float(severity) * (hi - lo))
    new_dur = max(2, int(round(original_dur * factor)))
    segment = x[:, start:end]
    old_t = np.linspace(0.0, 1.0, original_dur)
    new_t = np.linspace(0.0, 1.0, new_dur)
    warped = np.vstack([np.interp(new_t, old_t, segment[c]) for c in range(C)])
    x_mod = np.concatenate((x[:, :start], warped, x[:, end:]), axis=1)

    # Shift the selected and subsequent regime boundaries by the exact length
    # delta.  The list is private to this injection attempt; callers only
    # adopt it after the strength gate accepts the result.
    delta = new_dur - original_dur
    new_regimes = []
    for current in ctx.regimes:
        if current is reg:
            new_regimes.append(type(current)(
                regime=current.regime, start=current.start,
                end=current.end + delta, target_level=current.target_level,
                frequency=current.frequency, phase=current.phase,
            ))
        elif current.start >= end:
            new_regimes.append(type(current)(
                regime=current.regime, start=current.start + delta,
                end=current.end + delta, target_level=current.target_level,
                frequency=current.frequency, phase=current.phase,
            ))
        else:
            new_regimes.append(current)

    mask = np.zeros((C, x_mod.shape[1]), dtype=bool)
    mask[:, start:start + new_dur] = True
    meta = AnomalyMeta(
        family=AnomalyFamily.DURATION_ANOMALY,
        start=start,
        end=start + new_dur,
        severity=severity,
        affected_channels=list(range(C)),
        extra={
            "mode": mode,
            "stretch_factor": factor,
            "original_dur": original_dur,
            "new_dur": new_dur,
        },
    )
    return AnomalyResult(
        x_modified=x_mod, mask=mask, meta=meta, regimes_modified=new_regimes
    )
