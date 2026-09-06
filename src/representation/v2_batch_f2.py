"""Deterministic aggregate helpers for Sprint 11 Batch F2 (Tasks 35-36).

Chronological early-warning aggregates over the untouched temporal view:
concordance, lead-time distributions, false-warning runs, maintenance-boundary
windows, calibration error, censoring counts, and per-group slices. All helpers
are pure NumPy/Python with fail-fast guards; row-level timelines never leave
the server runner (``experiments/v2_staged/batch_f2_analyze.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def check_finite_list(values: Sequence[float], name: str) -> list[float]:
    """Return finite floats or raise (never silently publish gaps)."""
    out = [float(v) for v in values]
    if any(not math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite, got non-finite entries")
    return out


def concordance_index(
    days_to_event: Sequence[float],
    event_observed: Sequence[bool],
    scores: Sequence[float],
) -> dict[str, float | int]:
    """Harrell's C-index for failure-risk ranking (higher score = sooner failure).

    A pair (i, j) is comparable when the earlier time is an observed failure;
    it is concordant when the earlier failure carries the higher risk score
    (ties count 1/2). Files with non-finite times are excluded. Returns the
    index plus comparable/concordant/tied counts; zero comparable pairs
    yields ``c=None`` (recorded as null, never fabricated).
    """
    times = np.asarray(list(days_to_event), dtype=np.float64)
    events = np.asarray(list(event_observed), dtype=bool)
    risk = np.asarray(list(scores), dtype=np.float64)
    if not (times.shape == events.shape == risk.shape):
        raise ValueError("days_to_event, event_observed, and scores must share shape")
    if times.size == 0:
        raise ValueError("concordance requires at least one file")
    if not np.all(np.isfinite(risk)):
        raise ValueError("risk scores must be finite")
    finite = np.isfinite(times)
    times, events, risk = times[finite], events[finite], risk[finite]
    n_pairs = 0
    n_concordant = 0.0
    n_tied = 0
    for i in range(times.size):
        if not bool(events[i]):
            continue
        later = np.nonzero(times > times[i])[0]
        for j in later:
            n_pairs += 1
            if risk[i] > risk[j]:
                n_concordant += 1.0
            elif risk[i] == risk[j]:
                n_concordant += 0.5
                n_tied += 1
    return {
        "c": (float(n_concordant / n_pairs) if n_pairs else None),  # type: ignore[dict-item]
        "n_comparable_pairs": int(n_pairs),
        "n_concordant": float(n_concordant),
        "n_tied": int(n_tied),
        "n_files": int(times.size),
    }


def describe_or_null(values: Sequence[float]) -> dict[str, float | int | None]:
    """Count/mean/median/std/min/max/p25/p75, or ``{"n": 0}`` when empty."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("describe_or_null requires finite values")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def equal_width_ece(
    scores: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> dict[str, object]:
    """Expected calibration error over equal-width bins with per-bin rows."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    scores_list = check_finite_list(list(scores), "scores")
    labels_list = [bool(v) for v in labels]
    if len(scores_list) != len(labels_list):
        raise ValueError("scores and labels must share length")
    if not scores_list:
        raise ValueError("calibration requires at least one file")
    score_arr = np.asarray(scores_list)
    label_arr = np.asarray(labels_list, dtype=np.float64)
    bins: list[list[float]] = []
    ece_num = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b < n_bins - 1:
            sel = (score_arr >= lo) & (score_arr < hi)
        else:
            sel = (score_arr >= lo) & (score_arr <= hi)
        count = int(sel.sum())
        if count == 0:
            bins.append([lo + 0.5 / n_bins, float("nan"), 0.0])
            continue
        acc = float(label_arr[sel].mean())
        conf = float(score_arr[sel].mean())
        bins.append([(b + 0.5) / n_bins, acc, count / len(scores_list)])
        ece_num += abs(acc - conf) * count
    return {"ece": float(ece_num / len(scores_list)), "bins": bins, "n": len(scores_list)}


def brier_score(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between predicted probabilities and 0/1 labels."""
    scores_list = check_finite_list(list(scores), "scores")
    labels_list = [bool(v) for v in labels]
    if len(scores_list) != len(labels_list):
        raise ValueError("scores and labels must share length")
    if not scores_list:
        raise ValueError("brier requires at least one file")
    arr = np.asarray(scores_list)
    truth = np.asarray([1.0 if v else 0.0 for v in labels_list])
    return float(((arr - truth) ** 2).mean())


def count_warning_runs(warned: Sequence[bool]) -> int:
    """Count contiguous warned runs (a run starts on a False→True edge)."""
    flags = [bool(v) for v in warned]
    runs = 0
    in_run = False
    for flag in flags:
        if flag and not in_run:
            runs += 1
            in_run = True
        elif not flag:
            in_run = False
    return runs


def group_recall(
    keys: Sequence[str],
    in_window: Sequence[bool],
    warned: Sequence[bool],
) -> dict[str, dict[str, float | int]]:
    """Per-key warned fraction over files inside pre-failure windows."""
    result: dict[str, dict[str, float | int]] = {}
    buckets: dict[str, list[bool]] = {}
    for key, inside, flag in zip(keys, in_window, warned):
        if bool(inside):
            buckets.setdefault(str(key), []).append(bool(flag))
    for key in sorted(buckets):
        flags = buckets[key]
        result[key] = {
            "n": len(flags),
            "warned": sum(1 for f in flags if f),
            "rate": (sum(1 for f in flags if f) / len(flags)) if flags else 0.0,
        }
    return result


def censoring_counts(
    days_to_event: Sequence[float | None],
    event_observed: Sequence[bool],
) -> dict[str, int]:
    """Bounded censoring composition of one temporal cohort."""
    days = list(days_to_event)
    events = [bool(v) for v in event_observed]
    if len(days) != len(events):
        raise ValueError("days_to_event and event_observed must share length")
    n_nan = sum(1 for d in days if d is None or not math.isfinite(float(d)))
    n_censored = sum(1 for d, e in zip(days, events) if not e)
    n_observed = sum(1 for e in events if e)
    return {
        "n_files": len(days),
        "n_observed_events": int(n_observed),
        "n_censored": int(n_censored),
        "n_nan_times": int(n_nan),
    }


def maintenance_proximity(
    file_end_times: Sequence[float],
    boundary_times: Sequence[float],
    window_s: float,
) -> list[bool]:
    """Flag files whose end time falls within ±window of a boundary time."""
    if window_s < 0.0:
        raise ValueError("window_s must be non-negative")
    ends = check_finite_list(list(file_end_times), "file_end_times")
    bounds = check_finite_list(list(boundary_times), "boundary_times")
    if not bounds:
        return [False] * len(ends)
    bound_arr = np.asarray(bounds)
    return [bool(np.any(np.abs(np.asarray(e) - bound_arr) <= window_s)) for e in ends]


def check_mapping_nonempty(mapping: Mapping[str, object], name: str) -> None:
    """Fail fast on an empty aggregate mapping (never publish vacuous gaps)."""
    if not mapping:
        raise ValueError(f"{name} is empty; refusing to publish a vacuous aggregate")
