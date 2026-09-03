"""
Wrong Transition anomaly (DATA.md §4.3).

Timing/ordering of a regime transition is perturbed: too fast, too slow,
too early, or too late.  The resulting waveform is physically plausible but
contextually wrong — no plateau or step artifacts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import InjectionContext, AnomalyResult
from synth.config import AnomalyConfig
from synth.regimes import _cosine_ease
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_wrong_transition(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Pick a transition between two adjacent regimes and distort its timing
    by making it too fast, too slow, shifted early, or shifted late.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape

    # ── Find transitions between meaningful regimes ──────────────────
    regimes = ctx.regimes
    transition_indices = []
    for i in range(len(regimes) - 1):
        r_cur = regimes[i]
        r_nxt = regimes[i + 1]
        # Interesting transitions: ramp_up/down boundaries, regime changes
        if r_cur.regime != r_nxt.regime and r_cur.duration >= 15 and r_nxt.duration >= 15:
            transition_indices.append(i)

    if not transition_indices:
        # No suitable transition — degenerate
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.WRONG_TRANSITION, start=0, end=0,
            severity=severity, affected_channels=list(range(C)),
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    ti = int(rng.choice(transition_indices))
    r_before = regimes[ti]
    r_after = regimes[ti + 1]
    boundary = r_after.start   # exact transition boundary

    # ── Choose distortion mode ───────────────────────────────────────
    mode = rng.choice(["too_fast", "too_slow", "too_early", "too_late"])

    # Normal transition span inferred from the ramp structure of x[0]
    normal_span = max(int((r_before.duration + r_after.duration) * 0.15), 6)
    speed_lo, speed_hi = cfg.wt_speed_range
    timing_lo, timing_hi = cfg.wt_timing_shift_range

    x_mod = x.copy()
    affected = list(range(C))

    if mode in ("too_fast", "too_slow"):
        # Rescale the transition — warp the boundary ± span region
        span = normal_span
        anom_start = max(0, boundary - span)
        anom_end = min(T, boundary + span)

        target_speed = float(rng.uniform(speed_lo, speed_hi))
        speed_frac = 1.0 - float(severity) * (1.0 - target_speed)
        if mode == "too_fast":
            new_span = max(3, int(span * speed_frac))
        else:
            new_span = min(int(span / speed_frac), (anom_end - anom_start) - 2)

        for c in range(C):
            seg_before = x[c, anom_start:boundary]
            seg_after = x[c, boundary:anom_end]
            if len(seg_before) == 0 or len(seg_after) == 0:
                continue
            # Replace transition segment with a faster/slower cosine blend
            start_val = float(x[c, anom_start])
            end_val = float(x[c, min(anom_end - 1, T - 1)])
            # Build new transition
            new_tran_len = anom_end - anom_start
            t_norm = np.linspace(0, 1, new_tran_len)
            trans = start_val + (end_val - start_val) * _cosine_ease(t_norm)
            # Insert within new_span at boundary
            blend = new_span
            out = x[c, anom_start:anom_end].copy()
            t_blend = np.linspace(0, 1, min(blend, new_tran_len))
            blend_zone = start_val + (end_val - start_val) * _cosine_ease(t_blend)
            blend_start = max(0, len(out) // 2 - blend // 2)
            out[blend_start:blend_start + len(blend_zone)] = blend_zone
            x_mod[c, anom_start:anom_end] = out
        anom_mask_start, anom_mask_end = anom_start, anom_end

    else:  # too_early / too_late
        shift_frac = float(severity) * float(rng.uniform(timing_lo, timing_hi))
        if mode == "too_early":
            shift = -int(r_before.duration * shift_frac)
        else:
            shift = int(r_after.duration * shift_frac)

        new_boundary = int(np.clip(boundary + shift, span := max(10, normal_span), T - span))
        anom_start = min(boundary, new_boundary) - 5
        anom_end = max(boundary, new_boundary) + 5
        anom_start = max(0, anom_start)
        anom_end = min(T, anom_end)

        for c in range(C):
            start_val = float(x[c, anom_start])
            end_val = float(x[c, min(anom_end - 1, T - 1)])
            new_len = anom_end - anom_start
            if new_len <= 0:
                continue
            t_norm = np.linspace(0, 1, new_len)
            x_mod[c, anom_start:anom_end] = start_val + (end_val - start_val) * _cosine_ease(t_norm)
        anom_mask_start, anom_mask_end = anom_start, anom_end

    mask = np.zeros((C, T), dtype=bool)
    mask[:, anom_mask_start:anom_mask_end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.WRONG_TRANSITION,
        start=anom_mask_start, end=anom_mask_end,
        severity=severity,
        affected_channels=affected,
        transition_speed=float(speed_frac) if mode in ("too_fast", "too_slow") else None,
        extra={"mode": mode},
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
