"""Sprint 17 Task 16 — C9 patch-to-file aggregation alternatives (C9-A / C9-B).

Both arms rescore the *identical* valid B0 per-patch MSE energies (the frozen
``score_batch`` formula) with a different file-level aggregation only:

- ``C9-A top-k``: mean of the largest ``TOP_K_FRACTION`` share of valid patch
  scores (``k = max(1, ceil(fraction * n_valid))``).
- ``C9-B contiguous-tail``: maximum mean over fixed-duration contiguous patch
  windows of ``WINDOW_DURATION_TIMESTEPS`` timesteps, using patch starts and
  valid lengths for coverage; deterministic earliest-window tie-break.

Frozen Task 16 numerics: the protocol registry (sprint17-ablation-v4 §2) fixes
only the *mechanisms* ("fixed-fraction top-k mean", "maximum fixed-duration
contiguous-window mean"); no numeric fraction/duration exists anywhere in the
frozen protocol, config, or registry, so this module freezes them explicitly
*before outcomes*: ``TOP_K_FRACTION = 0.25`` (upper-quartile tail readout) and
``WINDOW_DURATION_TIMESTEPS = 64`` (4 strides / 2 patch widths at the frozen
B0 lattice ``patch_size=32, stride=16`` — short enough to stay localizable,
long enough to span a short anomalous run).

Scope and non-goals: only the ``S_pred`` patch-to-file aggregation changes.
``S_pop`` is a file-level bank distance with no patch scores, so the C9
mechanism cannot apply to it — it is reused bitwise from the hash-verified B0
caches (never fused, never substituted). Patch scores, prediction masks,
patch→timestep mapping, timestep scores, and metric code stay frozen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

#: Registered C9 arm IDs (protocol sprint17-ablation-v4 §2).
C9_ARMS = ("C9-A", "C9-B")

#: Frozen Task 16 top-k fraction (upper-quartile tail readout).
TOP_K_FRACTION = 0.25

#: Frozen Task 16 contiguous-window duration in timesteps (4 B0 strides).
WINDOW_DURATION_TIMESTEPS = 64

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C9-A": ("fixed-fraction top-k mean over valid patch scores "
             f"(fraction={TOP_K_FRACTION})"),
    "C9-B": ("maximum fixed-duration contiguous-window mean using patch "
             f"starts/valid lengths (duration={WINDOW_DURATION_TIMESTEPS} "
             "timesteps, earliest-window tie-break)"),
}


def b0_patch_energies(predicted: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """B0 per-patch MSE energies (the replaced baseline; fidelity oracle).

    Verbatim ``RepresentationInference.score_batch`` formula
    (``src/representation/inference.py``): mean MSE over latent dim, zero
    outside the valid masked set.
    """
    if (predicted.ndim != 3 or target.ndim != 3 or mask.ndim != 2
            or predicted.shape != target.shape
            or predicted.shape[:2] != mask.shape):
        raise ValueError("C9 latents/mask have incompatible shapes")
    if mask.dtype is not torch.bool:
        raise ValueError("prediction mask must be bool")
    errors = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
    return errors.masked_fill(~mask, 0.0)


def file_mean(energies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """B0 file aggregation: mean energy over valid masked patches.

    Empty valid sets score exactly 0.0 (the ``clamp_min(1)`` convention).
    """
    if energies.ndim != 2 or mask.ndim != 2 or energies.shape != mask.shape:
        raise ValueError("C9 energies/mask have incompatible shapes")
    if mask.dtype is not torch.bool:
        raise ValueError("prediction mask must be bool")
    counts = mask.sum(dim=1).to(energies.dtype)
    return (energies * mask.to(energies.dtype)).sum(dim=1) / counts.clamp_min(1.0)


def _check_1d(scores: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if scores.ndim != 1 or mask.ndim != 1 or scores.shape != mask.shape:
        raise ValueError("C9 patch scores and mask must be 1-D with equal length")
    valid = scores[mask]
    if valid.size and not np.isfinite(valid).all():
        raise ValueError("C9 valid patch scores must be finite")
    return scores, mask


def topk_mean(scores: np.ndarray, mask: np.ndarray,
              fraction: float = TOP_K_FRACTION) -> float:
    """C9-A file score: mean of the top ``fraction`` share of valid scores.

    ``k = max(1, ceil(fraction * n_valid))``; empty valid sets score 0.0
    (B0 convention). Deterministic: ``argsort`` with stable (index-order)
    tie-break, so equal scores always resolve to the same set.
    """
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
        raise ValueError("C9-A fraction must be finite")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("C9-A fraction must be in (0, 1]")
    scores, mask = _check_1d(scores, mask)
    valid = scores[mask]
    if valid.size == 0:
        return 0.0
    k = max(1, int(math.ceil(float(fraction) * valid.size)))
    order = np.argsort(-valid, kind="stable")
    return float(valid[order[:k]].mean())


def window_max_mean(scores: np.ndarray, mask: np.ndarray,
                    starts: np.ndarray, valid_len: np.ndarray,
                    duration: int = WINDOW_DURATION_TIMESTEPS) -> tuple[float, dict]:
    """C9-B file score: max fixed-duration contiguous-window mean.

    Candidate windows start at ``t = 0`` and at every valid patch start; a
    window covers ``[t, t + duration)`` and includes the valid patches whose
    valid interval ``[s_i, s_i + l_i)`` intersects it. The file score is the
    maximum window mean (ties resolve to the earliest ``t``). Empty valid
    sets score 0.0 (B0 convention). Padding slots (``starts == -1`` /
    ``valid_len == 0``) have empty intervals and can never be covered.

    Returns ``(score, provenance)`` with the winning window ``start``,
    ``n_covered`` valid patches, and ``covered_span`` timesteps (duration
    accounting for the evidence record).
    """
    if isinstance(duration, bool) or not isinstance(duration, (int, np.integer)):
        raise ValueError("C9-B duration must be an integer timestep count")
    duration = int(duration)
    if duration < 1:
        raise ValueError("C9-B duration must be >= 1 timestep")
    scores, mask = _check_1d(scores, mask)
    starts = np.asarray(starts)
    valid_len = np.asarray(valid_len)
    n = scores.shape[0]
    if starts.shape != (n,) or valid_len.shape != (n,):
        raise ValueError("C9-B starts/valid_len must match the patch count")
    if bool((valid_len < 0).any()) or bool((starts < -1).any()):
        raise ValueError("C9-B starts must be >= -1 and valid_len >= 0")
    valid_idx = np.nonzero(mask)[0]
    if valid_idx.size == 0:
        return 0.0, {"start": None, "n_covered": 0, "covered_span": 0}
    s = starts[valid_idx].astype(np.int64)
    ln = valid_len[valid_idx].astype(np.int64)
    v = scores[valid_idx]
    # Valid patches with empty coverage (padding slots) never join a window.
    nonempty = ln > 0
    s, ln, v = s[nonempty], ln[nonempty], v[nonempty]
    if v.size == 0:
        return 0.0, {"start": None, "n_covered": 0, "covered_span": 0}
    candidates = sorted({0} | {int(t) for t in s if int(t) >= 0})
    best_mean = -math.inf
    best: dict = {"start": None, "n_covered": 0, "covered_span": 0}
    for t in candidates:
        end = t + duration
        hit = (s < end) & (s + ln > t)
        if not bool(hit.any()):
            continue
        mean = float(v[hit].mean())
        # Strict improvement only: first (earliest t) wins ties.
        if mean > best_mean:
            lo = int(s[hit].min())
            hi = int((s[hit] + ln[hit]).max())
            best_mean = mean
            best = {"start": t, "n_covered": int(hit.sum()),
                    "covered_span": hi - lo}
    if best["start"] is None:
        return 0.0, best
    return best_mean, best


def arm_file_scores(arm_id: str, patch_scores: np.ndarray, mask: np.ndarray,
                    starts: np.ndarray | None = None,
                    valid_len: np.ndarray | None = None) -> tuple[float, dict]:
    """File score for one registered C9 arm over identical valid patch scores.

    Each score branch calls this independently on its own patch scores (in
    practice only the context branch has patch scores; the population branch
    is reused bitwise and never passes through here — no fusion by
    construction).
    """
    if arm_id == "C9-A":
        return topk_mean(patch_scores, mask), {"mechanism": "top-k",
                                               "fraction": TOP_K_FRACTION}
    if arm_id == "C9-B":
        if starts is None or valid_len is None:
            raise ValueError("C9-B requires patch starts and valid lengths")
        score, prov = window_max_mean(patch_scores, mask, starts, valid_len)
        return score, {"mechanism": "contiguous-window",
                       "duration": WINDOW_DURATION_TIMESTEPS, **prov}
    raise ValueError(f"unknown Sprint 17 C9 arm: {arm_id!r}")
