"""Deterministic aggregate helpers for Sprint 11 Batch F1 (Tasks 33-34).

Pure, dependency-light functions over in-memory tensors/lists. No model,
checkpoint, or dataset access lives here; the server runner
(``experiments/v2_staged/batch_f1_analyze.py``) calls these and transfers
only the bounded aggregates they return. Row-level latents/scores never
leave the server.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def effective_rank(rows: np.ndarray) -> float:
    """Exponential-entropy effective rank of centered rows (float64 SVD)."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("effective_rank requires a [M, D] matrix with M >= 2")
    if not np.isfinite(values).all():
        raise ValueError("effective_rank requires finite rows")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= 0.0:
        return 1.0
    probs = singular / total
    probs = probs[probs > 0.0]
    return float(math.exp(-float(np.sum(probs * np.log(probs)))))


def anisotropy_ratio(rows: np.ndarray, *, floor: float = 1e-6) -> float:
    """Largest/smallest singular value of centered rows (spread measure)."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("anisotropy_ratio requires a [M, D] matrix with M >= 2")
    if not np.isfinite(values).all():
        raise ValueError("anisotropy_ratio requires finite rows")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    top = float(singular[0])
    bottom = float(singular[-1])
    if top <= 0.0:
        return 1.0
    return float(top / max(bottom, floor))


def mask_overlap_fraction(
    flagged: Sequence[bool] | np.ndarray, start: int, end: int
) -> float:
    """Fraction of timestep range [start, end) covered by a boolean mask."""
    span = max(1, int(end) - max(0, int(start)))
    if len(flagged) == 0:
        return 0.0
    lo = max(0, int(start))
    hi = min(len(flagged), int(end))
    if hi <= lo:
        return 0.0
    return float(np.asarray(flagged)[lo:hi].sum()) / float(span)


def top_tail_mass(energies: np.ndarray, valid: np.ndarray, top_q: float) -> float:
    """Softmax-normalized mass of the top-Q valid patches (sparse retention).

    Over the valid signed energies ``E`` (e.g. negative NLL-scale context or
    population energies), weights are ``w_i = exp(E_i - max E) / sum_j
    exp(E_j - max E)`` — additive-shift-invariant, nonnegative, and summing
    to one — and the mass is the weight sum over the top-Q patches ranked by
    energy (``Q = max(1, ceil(top_q * n_valid))``). Uniform energies give
    ``Q / n_valid`` regardless of sign or offset. The prior total-fraction
    definition is invalid for signed energies (it returned 0.0 whenever the
    energy total was non-positive) and must not be used.
    """
    energy = np.asarray(energies, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if energy.shape != mask.shape:
        raise ValueError("energies and valid must share shape")
    values = energy[mask]
    if values.size == 0:
        raise ValueError("top_tail_mass requires at least one valid patch")
    if not np.isfinite(values).all():
        raise ValueError("top_tail_mass requires finite energies on valid patches")
    top_k = max(1, int(math.ceil(values.size * float(top_q))))
    shifted = values - float(values.max())
    weights = np.exp(shifted)
    weights /= float(weights.sum())
    order = np.argsort(values)[::-1][:top_k]
    return float(weights[order].sum())


def severity_bin(severity: float) -> str:
    """Fixed severity bins (documented; no data-driven edges)."""
    value = float(severity)
    if value < 0.5:
        return "low(<0.5)"
    if value < 1.0:
        return "mid[0.5,1.0)"
    return "high[>=1.0)"


def health_bin(health_value: float) -> str:
    """Fixed robot-health bins (documented; no data-driven edges)."""
    value = float(health_value)
    if value < 0.2:
        return "healthy(<0.2)"
    if value < 0.5:
        return "worn[0.2,0.5)"
    return "degraded[>=0.5)"


def summarize_deltas(paired: np.ndarray) -> dict[str, float]:
    """Mean/median/std/min/max/count of matched hybrid-minus-control deltas."""
    values = np.asarray(paired, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summarize_deltas requires a non-empty finite vector")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def rates_with_counts(flags: Sequence[bool]) -> dict[str, float | int]:
    """Rate + counts for a boolean decision list (empty-safe)."""
    items = [bool(v) for v in flags]
    return {
        "n": len(items),
        "flagged": int(sum(items)),
        "rate": float(sum(items) / len(items)) if items else 0.0,
    }


def confusion(
    decisions: Sequence[bool], labels: Sequence[bool]
) -> dict[str, int | float]:
    """Confusion counts + precision/recall/F1 (zero-division safe)."""
    if len(decisions) != len(labels):
        raise ValueError("decisions and labels must align")
    tp = sum(1 for d, y in zip(decisions, labels) if d and y)
    fp = sum(1 for d, y in zip(decisions, labels) if d and not y)
    fn = sum(1 for d, y in zip(decisions, labels) if not d and y)
    tn = sum(1 for d, y in zip(decisions, labels) if not d and not y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def group_rates(
    keys: Sequence[str], decisions: Sequence[bool]
) -> dict[str, dict[str, float | int]]:
    """Per-key flag rates with counts (deterministic key order)."""
    if len(keys) != len(decisions):
        raise ValueError("keys and decisions must align")
    buckets: dict[str, list[bool]] = {}
    for key, decision in zip(keys, decisions):
        buckets.setdefault(str(key), []).append(bool(decision))
    return {key: rates_with_counts(buckets[key]) for key in sorted(buckets)}


def describe(values: Sequence[float]) -> dict[str, float | int]:
    """Count/mean/median/std/min/max/p25/p75 of a finite list (empty-safe)."""
    items = [float(v) for v in values]
    if not items:
        return {"n": 0}
    array = np.asarray(items, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("describe requires finite values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def check_mapping_nonempty(mapping: Mapping[str, object], name: str) -> None:
    """Fail fast on an empty aggregate mapping (never silently publish gaps)."""
    if not mapping:
        raise ValueError(f"{name} is empty; refusing to publish a vacuous aggregate")
