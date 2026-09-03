"""
Contextual Replacement anomaly (DATA.md §4.3 / §4.2).

Replaces a region with a normal-looking segment taken from a *different*
context window within the same session (different regime position),
matching local mean/std and using cosine-boundary blending so the
replacement region is locally plausible but contextually wrong.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.anomalies.base import (
    InjectionContext, AnomalyResult, cosine_blend, cosine_blend_end, local_stats
)
from synth.config import AnomalyConfig
from synth.schema import AnomalyMeta, AnomalyFamily, RegimeType


def inject_contextual_replacement(
    ctx: InjectionContext,
    severity: float,
    cfg: AnomalyConfig,
) -> AnomalyResult:
    """
    Replace a contiguous region with a mean/std-matched segment drawn
    from a different regime position in the same session.
    """
    x = ctx.x.astype(np.float64)
    rng = ctx.rng
    C, T = x.shape
    blend = cfg.ctx_boundary_blend_steps

    # ── Find candidate target regions (one per active/periodic regime) ──
    active_regimes = [
        r for r in ctx.regimes
        if r.regime in (RegimeType.ACTIVE, RegimeType.PERIODIC, RegimeType.RECOVERY)
        and r.duration >= 30
    ]
    if len(active_regimes) < 2:
        # Fall back: any two non-overlapping regions of length >= 20
        third = T // 3
        candidates = [(0, third), (third, 2 * third), (2 * third, T)]
        candidates = [(s, e) for s, e in candidates if e - s >= 20]
        if len(candidates) < 2:
            # Session too short — use a simple approach
            mid = T // 2
            region_len = max(15, int((T // 4) * severity))
            start = max(0, mid - region_len // 2)
            end = min(T, start + region_len)
            donor_start = 0 if start > T // 2 else max(0, T - region_len - blend)
            donor_end = donor_start + (end - start)
        else:
            idx = int(rng.integers(0, len(candidates)))
            di = (idx + 1) % len(candidates)
            r_s, r_e = candidates[idx]
            d_s, d_e = candidates[di]
            region_len = max(15, int((r_e - r_s) * severity * 0.7))
            start = int(rng.integers(r_s, max(r_s + 1, r_e - region_len)))
            end = start + region_len
            donor_start = int(rng.integers(d_s, max(d_s + 1, d_e - region_len)))
            donor_end = donor_start + region_len
    else:
        # Pick two different active-regime windows
        idx = int(rng.integers(0, len(active_regimes)))
        candidates_other = [i for i in range(len(active_regimes)) if i != idx]
        di = int(rng.choice(candidates_other))
        reg = active_regimes[idx]
        donor_reg = active_regimes[di]
        region_len = max(15, int(min(reg.duration, donor_reg.duration) * severity * 0.7))
        start = int(rng.integers(reg.start, max(reg.start + 1, reg.end - region_len)))
        end = start + region_len
        donor_start = int(rng.integers(donor_reg.start, max(donor_reg.start + 1, donor_reg.end - region_len)))
        donor_end = donor_start + region_len

    # Clamp
    end = min(end, T)
    donor_end = min(donor_end, T)
    region_len = min(end - start, donor_end - donor_start)
    end = start + region_len
    donor_end = donor_start + region_len

    if region_len < 10:
        # Degenerate — return unmodified (will be rejected by strength gate)
        mask = np.zeros((C, T), dtype=bool)
        meta = AnomalyMeta(
            family=AnomalyFamily.CONTEXTUAL_REPLACEMENT,
            start=start, end=end, severity=severity, affected_channels=list(range(C)),
        )
        return AnomalyResult(x_modified=x, mask=mask, meta=meta, accepted=False)

    x_mod = x.copy()
    affected = list(range(C))

    for c in range(C):
        # Donor segment
        donor_seg = x[c, donor_start:donor_end].copy()

        # Match local mean/std of the target region
        t_mean, t_std = local_stats(x, start, end, c)
        d_mean, d_std = float(np.mean(donor_seg)), float(np.std(donor_seg) + 1e-8)
        donor_matched = (donor_seg - d_mean) / d_std * t_std + t_mean

        # Cosine blend at boundaries
        if start > 0:
            donor_matched = cosine_blend(x[c, max(0, start - blend):start], donor_matched, blend)
        if end < T:
            donor_matched = cosine_blend_end(donor_matched, float(x[c, end]), blend)

        x_mod[c, start:end] = donor_matched

    mask = np.zeros((C, T), dtype=bool)
    mask[:, start:end] = True

    meta = AnomalyMeta(
        family=AnomalyFamily.CONTEXTUAL_REPLACEMENT,
        start=start, end=end, severity=severity,
        affected_channels=affected,
        donor_file_id=f"self:{donor_start}-{donor_end}",
    )
    return AnomalyResult(x_modified=x_mod, mask=mask, meta=meta)
