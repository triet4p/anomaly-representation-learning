"""
Missing Event anomaly (DATA.md §4.9).

An event/regime that should appear is absent.  Instead of flat
interpolation, the missing slot is filled with a *valid behavior* from
a different context — making it locally plausible but contextually wrong.

Modes:
  wrong_regime:    replace the expected regime with a different valid regime
  continuation:    the preceding regime simply continues without the expected event
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult, cosine_blend
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_missing_event(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Remove an expected regime and replace it with a contextually-wrong but
    locally-plausible signal segment.  No flat interpolation is used.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape
    regimes = ctx.regimes

    # ── Find a regime to "miss" ─────────────────────────────────────
    # Best candidates: ramp_up, active, periodic, recovery (transitions that matter)
    candidate_idx = [
        i for i, r in enumerate(regimes)
        if r.regime in (RegimeType.RAMP_UP, RegimeType.ACTIVE, RegimeType.PERIODIC, RegimeType.RECOVERY)
        and r.duration >= 15
        and i > 0  # can't miss the very first
        and i < len(regimes) - 1  # can't miss the very last
    ]

    if not candidate_idx:
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.MISSING_EVENT, start=0, end=0,
            severity=severity, affected_channels=[],
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)
    ci = int(rng.choice(candidate_idx))

    missing_reg = regimes[ci]
    prev_reg = regimes[ci - 1]
    full_start = missing_reg.start
    full_end = missing_reg.end
    full_dur = full_end - full_start
    # At low severity only part of the expected event is replaced; at high
    # severity almost the whole event is missing.  Centering keeps both
    # boundaries local and leaves the untouched context intact.
    target_dur = max(5, int(round(full_dur * (0.2 + 0.8 * float(severity)))))
    target_dur = min(full_dur, target_dur)
    anom_start = full_start + (full_dur - target_dur) // 2
    anom_end = anom_start + target_dur

    mode = cfg.me_replacement_mode
    x_mod = x.copy()
    affected = list(range(C))
    blend_n = min(8, target_dur // 4)

    if mode == "continuation" or len(regimes) < 3:
        # Fill with continuation of previous regime — same dynamics, no transition
        dur = anom_end - anom_start
        for c in range(C):
            # Take the last `dur` steps of the previous regime as continuation
            prev_seg = x[c, max(0, prev_reg.start):prev_reg.end]
            if len(prev_seg) == 0:
                continue
            # Repeat from end of prev_reg
            if len(prev_seg) >= dur:
                fill = prev_seg[-dur:].copy()
            else:
                # Tile with wrap (not exact repetition — interpolate smoothly)
                reps = (dur // len(prev_seg)) + 2
                t_orig = np.linspace(0, 1, len(prev_seg))
                t_new = np.linspace(0, reps, dur) % 1.0
                fill = np.interp(t_new, t_orig, prev_seg)

            # Add small noise to avoid exact repetition
            fill = fill + rng.normal(0, float(np.std(fill)) * 0.05, dur)

            # Boundary blend
            if blend_n > 0 and anom_start > 0:
                t = np.linspace(0, 1, blend_n)
                ab = (1 - np.cos(np.pi * t)) / 2
                fill[:blend_n] = x[c, anom_start - 1] * (1 - ab) + fill[:blend_n] * ab
            if blend_n > 0 and anom_end < T:
                t = np.linspace(0, 1, blend_n)
                ab = (1 - np.cos(np.pi * t)) / 2
                fill[-blend_n:] = fill[-blend_n:] * (1 - ab) + x[c, anom_end] * ab

            x_mod[c, anom_start:anom_end] = fill

    else:  # wrong_regime
        # Find another valid-but-wrong regime from a different part of the session
        other_regs = [
            r for i, r in enumerate(regimes)
            if i not in (ci - 1, ci, ci + 1)
            and r.regime != missing_reg.regime
            and r.duration >= 15
        ]
        dur = anom_end - anom_start
        if other_regs:
            donor = other_regs[int(rng.integers(0, len(other_regs)))]
            donor_dur = donor.end - donor.start
        else:
            # Use the segment before the missing event from a different offset
            donor = prev_reg
            donor_dur = prev_reg.end - prev_reg.start

        for c in range(C):
            d_start = donor.start
            d_end = donor.end
            donor_seg = x[c, d_start:d_end].copy()
            if len(donor_seg) == 0:
                continue
            # Resample to match the missing region length
            t_orig = np.linspace(0, 1, len(donor_seg))
            t_new = np.linspace(0, 1, dur)
            fill = np.interp(t_new, t_orig, donor_seg)

            # Mean-match: shift donor to continue from surrounding
            target_mean = float(np.mean(x[c, anom_start:anom_end]))
            donor_mean = float(np.mean(fill))
            fill = fill - donor_mean + target_mean

            # Boundary blend
            if blend_n > 0 and anom_start > 0:
                t = np.linspace(0, 1, blend_n)
                ab = (1 - np.cos(np.pi * t)) / 2
                fill[:blend_n] = x[c, anom_start - 1] * (1 - ab) + fill[:blend_n] * ab
            if blend_n > 0 and anom_end < T:
                t = np.linspace(0, 1, blend_n)
                ab = (1 - np.cos(np.pi * t)) / 2
                fill[-blend_n:] = fill[-blend_n:] * (1 - ab) + x[c, anom_end] * ab

            x_mod[c, anom_start:anom_end] = fill

    mask = np.zeros((C, T), dtype=bool)
    mask[:, anom_start:anom_end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.MISSING_EVENT,
        start=anom_start, end=anom_end,
        severity=severity,
        affected_channels=affected,
        extra={"mode": mode, "missing_regime": missing_reg.regime.value},
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
