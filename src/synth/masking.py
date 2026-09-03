"""
Masking policy for latent prediction (DATA.md §7).

Three-component masking with fixed total mask ratio:
  M = M_random ∪ M_info ∪ M_block

The total mask ratio is always cfg.total_mask_ratio regardless of which
composition fractions are used — enabling ablation by changing composition
while keeping total masked tokens constant.

Information-aware masking stratifies patches into four categories based on
local statistics, then selects proportionally from each stratum.

Block masking selects contiguous runs of patches to force long-range context.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.config import MaskingConfig
from synth.schema import PatchBatch, MaskResult


def _patch_stats(patch: npt.NDArray[np.float32]) -> dict[str, float]:
    """Compute local statistics for one patch [C, W]."""
    flat = patch.flatten().astype(np.float64)
    std = float(np.std(flat))
    rng = float(np.max(flat) - np.min(flat))
    energy = float(np.mean(flat ** 2))
    deriv = float(np.mean(np.abs(np.diff(flat))))
    slope = float(np.polyfit(np.arange(len(flat)), flat, 1)[0]) if len(flat) >= 2 else 0.0
    return {"std": std, "range": rng, "energy": energy, "derivative_energy": deriv, "slope": abs(slope)}


def compute_all_patch_stats(batch: PatchBatch) -> npt.NDArray[np.float64]:
    """
    Compute a composite information score for each patch.
    Score = std * range * (1 + derivative_energy).
    Higher score → more informative / dynamic patch.

    Returns float64 [N].
    """
    N = batch.N
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    flat = batch.patches.reshape(N, -1).astype(np.float64)
    std = np.std(flat, axis=1)
    rng = np.ptp(flat, axis=1)
    deriv = np.mean(np.abs(np.diff(flat, axis=1)), axis=1)
    return (std + 1e-8) * (rng + 1e-8) * (1.0 + deriv)


def _stratify_patches(
    scores: npt.NDArray[np.float64],
    valid_indices: npt.NDArray[np.int64],
    cfg: MaskingConfig,
) -> dict[str, npt.NDArray[np.int64]]:
    """
    Stratify valid (non-padding) patches into four categories by score quartile.

    Returns dict with keys: stable, transition, dynamic, extreme.
    """
    if len(valid_indices) == 0:
        return {k: np.array([], dtype=np.int64) for k in ("stable", "transition", "dynamic", "extreme")}
    vs = scores[valid_indices]
    q25, q50, q75 = np.percentile(vs, [25, 50, 75])
    stable     = valid_indices[vs <= q25]
    transition = valid_indices[(vs > q25) & (vs <= q50)]
    dynamic    = valid_indices[(vs > q50) & (vs <= q75)]
    extreme    = valid_indices[vs > q75]
    return {"stable": stable, "transition": transition, "dynamic": dynamic, "extreme": extreme}


def apply_masking(
    batch: PatchBatch,
    cfg: MaskingConfig,
    rng: np.random.Generator,
) -> MaskResult:
    """Apply a disjoint random/info-aware/block masking composition.

    The requested total is computed from valid (non-padding) patches and is
    filled exactly whenever at least that many valid patches exist.  Strategy
    selections are disjoint, so ``composition`` is an auditable partition of
    the returned mask rather than a count of overlapping attempts.
    """
    N = batch.N
    valid_indices = np.flatnonzero(~batch.pad_mask.all(axis=1)).astype(np.int64)
    n_valid = int(valid_indices.size)
    masked = np.zeros(N, dtype=bool)
    composition = {"random": 0, "info": 0, "block": 0}
    if n_valid == 0:
        return MaskResult(mask=masked, composition=composition, total_ratio=0.0)

    ratio = float(np.clip(cfg.total_mask_ratio, 0.0, 1.0))
    target = min(n_valid, int(round(ratio * n_valid)))
    if target == 0:
        return MaskResult(mask=masked, composition=composition, total_ratio=0.0)

    fractions = np.asarray(
        [cfg.random_fraction, cfg.info_fraction, cfg.block_fraction],
        dtype=float,
    )
    if np.any(fractions < 0) or not np.isfinite(fractions).all() or fractions.sum() <= 0:
        raise ValueError("masking composition fractions must be finite and non-negative")
    fractions /= fractions.sum()
    quotas = np.floor(fractions * target).astype(int)
    for i in np.argsort(-(fractions * target - quotas))[: target - int(quotas.sum())]:
        quotas[i] += 1
    n_random, n_info, n_block = map(int, quotas)

    # Random selection.
    remaining = valid_indices.copy()
    if n_random:
        chosen = rng.choice(remaining, size=n_random, replace=False)
        masked[chosen] = True
        composition["random"] = int(chosen.size)
        remaining = remaining[~np.isin(remaining, chosen)]

    # Information-aware selection: stratified quotas, followed by a
    # deterministic fill from remaining strata if a small batch has an empty
    # quartile.  This never silently loses the requested mask count.
    if n_info and remaining.size:
        scores = compute_all_patch_stats(batch)
        strata = _stratify_patches(scores, remaining, cfg)
        names = ("stable", "transition", "dynamic", "extreme")
        info_fracs = np.asarray(
            [cfg.info_stable_fraction, cfg.info_transition_fraction,
             cfg.info_dynamic_fraction, cfg.info_extreme_fraction],
            dtype=float,
        )
        if np.any(info_fracs < 0) or info_fracs.sum() <= 0:
            raise ValueError("information-aware fractions must be non-negative")
        info_fracs /= info_fracs.sum()
        info_quota = np.floor(info_fracs * n_info).astype(int)
        for i in np.argsort(-(info_fracs * n_info - info_quota))[: n_info - int(info_quota.sum())]:
            info_quota[i] += 1
        selected: list[int] = []
        for name, quota in zip(names, info_quota):
            pool = strata[name]
            take = min(int(quota), int(pool.size))
            if take:
                selected.extend(int(v) for v in rng.choice(pool, size=take, replace=False))
        selected_set = set(selected)
        if len(selected) < n_info:
            fill_pool = np.array([v for v in remaining if int(v) not in selected_set], dtype=np.int64)
            extra = n_info - len(selected)
            if fill_pool.size:
                selected.extend(int(v) for v in rng.choice(fill_pool, size=min(extra, fill_pool.size), replace=False))
        chosen = np.asarray(selected[:n_info], dtype=np.int64)
        masked[chosen] = True
        composition["info"] = int(chosen.size)
        remaining = remaining[~np.isin(remaining, chosen)]

    # Block selection.  Since patch starts are ordered, contiguous slices of
    # remaining indices form true patch blocks; split around already-selected
    # patches and fill each requested block quota exactly.
    if n_block and remaining.size:
        selected: list[int] = []
        available = remaining.copy()
        while len(selected) < n_block and available.size:
            take = min(
                n_block - len(selected),
                available.size,
                max(1, int(rng.integers(cfg.min_block_size, cfg.max_block_size + 1))),
            )
            start = int(rng.integers(0, available.size - take + 1))
            chosen = available[start:start + take]
            selected.extend(int(v) for v in chosen)
            available = available[~np.isin(available, chosen)]
        chosen_arr = np.asarray(selected[:n_block], dtype=np.int64)
        masked[chosen_arr] = True
        composition["block"] = int(chosen_arr.size)

    # Defensive fill for tiny/degenerate batches where a strategy had no
    # remaining pool.  Attribute the fill to the first configured strategy
    # with a positive fraction, preserving the exact partition invariant.
    shortfall = target - int(masked[valid_indices].sum())
    if shortfall:
        fill = valid_indices[~masked[valid_indices]][:shortfall]
        masked[fill] = True
        for name, frac in zip(("random", "info", "block"), fractions):
            if frac > 0:
                composition[name] += int(fill.size)
                break

    actual = int(masked[valid_indices].sum())
    return MaskResult(
        mask=masked,
        composition=composition,
        total_ratio=actual / n_valid,
    )
