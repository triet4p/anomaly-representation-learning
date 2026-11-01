"""Encoder-free evaluation for the S24-7-M20 population benchmark.

Everything here recomputes from persisted per-file readout items: no model,
no torch, no generation. The same functions produce the official statistics
and the fresh-process replay, so a replay mismatch is a real failure.

The single movement statistic is the unbiased MMD^2 between a window
population and the fixed history reference population under a Gaussian kernel.
Each readout defines its own item space and bandwidth; the calibration and
gate procedure is identical for all readouts, which is what makes the three
comparisons comparable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from population_benchmark import spec

#: Item-key prefix for per-encoder readouts.
POOLED_PREFIX = "pooled_"
PREPOOL_PREFIX = "prepool_"
RAW_KEY = "raw"
LENGTH_KEY = "length"


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def item_keys(encoders: Sequence[int] = spec.ENCODER_SEEDS) -> list[str]:
    keys = [RAW_KEY, LENGTH_KEY]
    for seed in encoders:
        keys.append(f"{POOLED_PREFIX}{int(seed)}")
        keys.append(f"{PREPOOL_PREFIX}{int(seed)}")
    return keys


def readout_key(readout: str, seed: int | None = None) -> str:
    if readout == "RAW":
        return RAW_KEY
    if readout == spec.DIAGNOSTIC_READOUT:
        return LENGTH_KEY
    if readout == "POOLED":
        return f"{POOLED_PREFIX}{int(seed)}"
    if readout == "PREPOOL":
        return f"{PREPOOL_PREFIX}{int(seed)}"
    raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown readout {readout!r}")


# --- kernel / statistic --------------------------------------------------------


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise _refused("REFUSED-ERROR-BOUNDARY", "distance operands must be [n,d] with equal d")
    d2 = (
        np.einsum("ij,ij->i", a, a)[:, None]
        + np.einsum("ij,ij->i", b, b)[None, :]
        - 2.0 * (a @ b.T)
    )
    np.maximum(d2, 0.0, out=d2)
    return d2


def gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise _refused("REFUSED-ERROR-BOUNDARY", "bandwidth must be finite and positive")
    return np.exp(-squared_distances(a, b) / (2.0 * float(sigma) ** 2))


def median_pairwise_distance(items: np.ndarray) -> float:
    """Median Euclidean distance over unique pairs (frozen bandwidth rule)."""
    values = np.asarray(items, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "bandwidth needs two or more items")
    d2 = squared_distances(values, values)
    upper = d2[np.triu_indices(values.shape[0], k=1)]
    if upper.size == 0:
        raise _refused("REFUSED-ERROR-BOUNDARY", "bandwidth has no pairs")
    median = float(np.median(np.sqrt(np.maximum(upper, 0.0))))
    if not math.isfinite(median) or median <= 0.0:
        raise _refused("REFUSED-ERROR-BOUNDARY", f"degenerate bandwidth {median}")
    return median


def standardizer(items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(items, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "standardizer needs two or more items")
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return center, scale


def apply_standardizer(items: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(items, dtype=np.float64)
    return (values - center[None, :]) / scale[None, :]


@dataclass(frozen=True)
class ReferenceSpace:
    """One (history, readout, encoder) kernel space with frozen nuisance fits."""

    center: np.ndarray
    scale: np.ndarray
    sigma: float
    reference: np.ndarray
    reference_kernel_sum: float
    reference_sq: np.ndarray


def fit_reference_space(reference: np.ndarray, bandwidth: np.ndarray) -> ReferenceSpace:
    center, scale = standardizer(reference)
    ref_std = apply_standardizer(reference, center, scale)
    bw_std = apply_standardizer(bandwidth, center, scale)
    sigma = median_pairwise_distance(bw_std)
    kernel = gaussian_kernel(ref_std, ref_std, sigma)
    return ReferenceSpace(
        center=center,
        scale=scale,
        sigma=float(sigma),
        reference=ref_std,
        reference_kernel_sum=float(kernel.sum()),
        reference_sq=np.einsum("ij,ij->i", ref_std, ref_std),
    )


def mmd2_unbiased(window: np.ndarray, space: ReferenceSpace) -> float:
    """Unbiased MMD^2 between a window population and the reference population."""
    x = np.asarray(window, dtype=np.float64)
    y = space.reference
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "MMD needs two or more items per population")
    sigma = space.sigma
    x_sq = np.einsum("ij,ij->i", x, x)
    d2_xy = x_sq[:, None] + space.reference_sq[None, :] - 2.0 * (x @ y.T)
    np.maximum(d2_xy, 0.0, out=d2_xy)
    cross = float(np.exp(-d2_xy / (2.0 * sigma * sigma)).sum())
    d2_xx = x_sq[:, None] + x_sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2_xx, 0.0, out=d2_xx)
    within = float(np.exp(-d2_xx / (2.0 * sigma * sigma)).sum())
    within_term = (within - n) / (n * (n - 1))
    reference_term = (space.reference_kernel_sum - m) / (m * (m - 1))
    return 2.0 * cross / (n * m) - within_term - reference_term


def null_threshold(values: Sequence[float], quantile: float = spec.CALIBRATION_QUANTILE) -> dict[str, Any]:
    """Order-statistic threshold with its finite-sample rank uncertainty."""
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    if n < 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "calibration needs two or more windows")
    if not 0.0 < float(quantile) < 1.0:
        raise _refused("REFUSED-UNKNOWN-DESIGN", "quantile must lie in (0,1)")
    rank = int(math.ceil(float(quantile) * n)) - 1
    rank = max(0, min(rank, n - 1))
    threshold = ordered[rank]
    # Probability that a fresh null draw exceeds the threshold, as an interval
    # from the order-statistic Beta(k, n-k+1) law.
    from scipy.stats import beta

    lower = float(beta.ppf(0.025, rank + 1, n - rank))
    upper = float(beta.ppf(0.975, rank + 1, n - rank))
    return {
        "threshold": float(threshold),
        "n": n,
        "rank_0based": rank,
        "quantile": float(quantile),
        "nominal_exceedance": float(1.0 - quantile),
        "exceedance_interval_95": [1.0 - upper, 1.0 - lower],
    }


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> dict[str, float | int | None]:
    """Exact binomial interval; None bounds are reported as null."""
    if trials < 0 or successes < 0 or successes > trials:
        raise _refused("REFUSED-ERROR-BOUNDARY", "invalid binomial counts")
    if trials == 0:
        return {"successes": 0, "trials": 0, "rate": None, "lower": None, "upper": None}
    from scipy.stats import beta

    alpha = 1.0 - float(level)
    rate = successes / trials
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return {
        "successes": int(successes),
        "trials": int(trials),
        "rate": float(rate),
        "lower": lower,
        "upper": upper,
    }


def majority_alarm(alarms: Mapping[int, bool]) -> bool:
    return sum(1 for value in alarms.values() if value) >= 2


# --- loaded chunk model --------------------------------------------------------


@dataclass
class Chunk:
    """One persisted chunk: ordered items plus its custody metadata."""

    kind: str
    history_id: str
    chunk_id: str
    meta: dict[str, Any]
    items: dict[str, np.ndarray]

    @property
    def file_ids(self) -> list[str]:
        return [str(record["file_id"]) for record in self.meta["files"]]

    @property
    def windows(self) -> list[dict[str, Any]]:
        return list(self.meta.get("windows", []))


def _group_by_history(chunks: Sequence[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.history_id, []).append(chunk)
    return grouped


@dataclass
class HistorySpace:
    """All items for one history with per-readout kernel spaces."""

    history_id: str
    file_ids: list[str]
    row_of: dict[str, int]
    matrix: dict[str, np.ndarray]
    spaces: dict[str, ReferenceSpace]

    def rows(self, key: str, file_ids: Sequence[str]) -> np.ndarray:
        matrix = self.matrix[key]
        try:
            index = [self.row_of[str(file_id)] for file_id in file_ids]
        except KeyError as exc:
            raise _refused("REFUSED-IDENTITY", f"window references unknown file {exc}") from None
        return matrix[index]


def build_history_space(
    history_id: str,
    reference: Chunk,
    bandwidth: Chunk,
    others: Sequence[Chunk],
    encoders: Sequence[int] = spec.ENCODER_SEEDS,
    readouts: Sequence[str] = spec.READOUTS,
) -> HistorySpace:
    """Concatenate items, fit nuisance transforms on reference only, freeze spaces."""
    ordered = [reference, bandwidth] + list(others)
    file_ids = [file_id for chunk in ordered for file_id in chunk.file_ids]
    if len(set(file_ids)) != len(file_ids):
        raise _refused("REFUSED-IDENTITY", "duplicate file IDs across chunks of one history")
    row_of = {file_id: index for index, file_id in enumerate(file_ids)}
    reference_rows = np.arange(len(reference.file_ids))
    bandwidth_rows = np.arange(len(reference.file_ids), len(reference.file_ids) + len(bandwidth.file_ids))
    matrix: dict[str, np.ndarray] = {}
    spaces: dict[str, ReferenceSpace] = {}
    for readout in readouts:
        keys = [readout_key(readout, seed) for seed in encoders] if readout != "RAW" and readout != spec.DIAGNOSTIC_READOUT else [readout_key(readout)]
        for key in keys:
            values = np.concatenate([np.asarray(chunk.items[key], dtype=np.float64) for chunk in ordered], axis=0)
            matrix[key] = values
            spaces[key] = fit_reference_space(values[reference_rows], values[bandwidth_rows])
    if LENGTH_KEY not in matrix:
        values = np.concatenate([np.asarray(chunk.items[LENGTH_KEY], dtype=np.float64) for chunk in ordered], axis=0)
        matrix[LENGTH_KEY] = values
        spaces[LENGTH_KEY] = fit_reference_space(values[reference_rows], values[bandwidth_rows])
    return HistorySpace(history_id, file_ids, row_of, matrix, spaces)


# --- evaluation ----------------------------------------------------------------


def _cell_label(index: int) -> tuple[str, float]:
    return spec.FAMILIES[index // len(spec.SEVERITIES)], spec.SEVERITIES[index % len(spec.SEVERITIES)]


def _window_statistics(
    space: HistorySpace,
    file_ids: Sequence[str],
    encoders: Sequence[int] = spec.ENCODER_SEEDS,
    readouts: Sequence[str] = spec.READOUTS,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for readout in readouts:
        if readout == "RAW":
            keys = [RAW_KEY]
        else:
            keys = [readout_key(readout, seed) for seed in encoders]
        values: dict[str, float] = {}
        for key in keys:
            matrix_key = key
            rows = space.rows(matrix_key, file_ids)
            values[key] = float(mmd2_unbiased(rows, space.spaces[matrix_key]))
        output[readout] = values
    if spec.DIAGNOSTIC_READOUT not in output:
        rows = space.rows(LENGTH_KEY, file_ids)
        output[spec.DIAGNOSTIC_READOUT] = {LENGTH_KEY: float(mmd2_unbiased(rows, space.spaces[LENGTH_KEY]))}
    return output


def evaluate_chunks(
    chunks: Sequence[Chunk],
    plan: spec.Plan,
    *,
    readouts: Sequence[str] = spec.READOUTS,
    encoders: Sequence[int] = spec.ENCODER_SEEDS,
) -> dict[str, Any]:
    """Recompute every statistic, threshold, alarm, aggregate and gate."""
    grouped = _group_by_history(chunks)
    missing = [name for name in plan.history_ids if name not in grouped]
    if missing:
        raise _refused("REFUSED-PREREQUISITE", f"missing history chunks: {missing}")

    spaces: dict[str, HistorySpace] = {}
    for history_id, history_chunks in grouped.items():
        reference = [chunk for chunk in history_chunks if chunk.kind == "reference"]
        bandwidth = [chunk for chunk in history_chunks if chunk.kind == "bandwidth"]
        if len(reference) != 1 or len(bandwidth) != 1:
            raise _refused("REFUSED-PREREQUISITE", f"history {history_id} needs one reference and one bandwidth chunk")
        others = [chunk for chunk in history_chunks if chunk.kind in ("calibration", "test")]
        spaces[history_id] = build_history_space(
            history_id, reference[0], bandwidth[0], others, encoders=encoders, readouts=tuple(readouts)
        )

    calibration_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.kind not in ("calibration", "test"):
            continue
        space = spaces[chunk.history_id]
        for window in chunk.windows:
            file_ids = [str(value) for value in window["file_ids"]]
            stats = _window_statistics(space, file_ids, encoders=encoders, readouts=tuple(readouts))
            record = {
                **window,
                "history_id": chunk.history_id,
                "chunk_id": chunk.chunk_id,
                "cell_index": int(window["cell_index"]),
                "family": str(window["family"]),
                "severity": float(window["severity"]),
                "stats": stats,
            }
            (calibration_records if chunk.kind == "calibration" else test_records).append(record)

    all_readouts = list(readouts) + [spec.DIAGNOSTIC_READOUT]
    thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    for readout in all_readouts:
        thresholds[readout] = {}
        for horizon in plan.horizons:
            thresholds[readout][horizon] = {}
            if readout == spec.DIAGNOSTIC_READOUT:
                values = [
                    float(record["stats"][readout][LENGTH_KEY])
                    for record in calibration_records
                    if record["horizon"] == horizon and record["burden"] == spec.BURDEN_LABELS[0]
                ]
                if not values:
                    raise _refused("REFUSED-PREREQUISITE", f"no calibration windows for {readout}/{horizon}")
                thresholds[readout][horizon]["length"] = null_threshold(values, plan.calibration_quantile)
                continue
            for seed in encoders:
                key = readout_key(readout, seed) if readout not in ("RAW",) else RAW_KEY
                values = [
                    float(record["stats"][readout][key])
                    for record in calibration_records
                    if record["horizon"] == horizon and record["burden"] == spec.BURDEN_LABELS[0]
                ]
                if not values:
                    raise _refused("REFUSED-PREREQUISITE", f"no calibration windows for {readout}/{horizon}")
                thresholds[readout][horizon][str(seed)] = null_threshold(values, plan.calibration_quantile)

    for record in test_records:
        record["alarms"] = {}
        record["combined"] = {}
        for readout in all_readouts:
            if readout == spec.DIAGNOSTIC_READOUT:
                threshold = float(thresholds[readout][record["horizon"]]["length"]["threshold"])
                alarm = bool(float(record["stats"][readout][LENGTH_KEY]) > threshold)
                record["alarms"][readout] = {"length": alarm}
                record["combined"][readout] = alarm
                continue
            alarms: dict[int, bool] = {}
            for seed in encoders:
                key = readout_key(readout, seed) if readout not in ("RAW",) else RAW_KEY
                threshold = float(thresholds[readout][record["horizon"]][str(seed)]["threshold"])
                alarms[int(seed)] = bool(float(record["stats"][readout][key]) > threshold)
            record["alarms"][readout] = {str(seed): value for seed, value in alarms.items()}
            record["combined"][readout] = bool(majority_alarm(alarms))

    aggregates = {
        readout: _aggregate_readout(readout, test_records, plan, encoders)
        for readout in list(readouts) + [spec.DIAGNOSTIC_READOUT]
    }
    return {
        "calibration_records": calibration_records,
        "test_records": test_records,
        "thresholds": thresholds,
        "aggregates": aggregates,
        "plan_digest": spec.enumeration_digest(plan),
        "reference_spaces": {
            history_id: {
                key: {
                    "sigma": space.sigma,
                    "center": [float(value) for value in space.center],
                    "scale": [float(value) for value in space.scale],
                }
                for key, space in history.spaces.items()
            }
            for history_id, history in spaces.items()
        },
    }


def _coverage_ok(record: Mapping[str, Any]) -> bool:
    return str(record.get("coverage")) == "COVERED"


def _aggregate_readout(
    readout: str,
    records: Sequence[Mapping[str, Any]],
    plan: spec.Plan,
    encoders: Sequence[int],
) -> dict[str, Any]:
    horizons = list(plan.horizons)
    burdens = list(spec.BURDEN_LABELS)
    cell_count = plan.cell_count
    coverage: dict[str, dict[str, dict[str, int]]] = {
        horizon: {burden: {str(cell): 0 for cell in range(cell_count)} for burden in burdens}
        for horizon in horizons
    }
    alarms: dict[str, dict[str, dict[str, list[bool]]]] = {
        horizon: {burden: {str(cell): [] for cell in range(cell_count)} for burden in burdens}
        for horizon in horizons
    }
    per_encoder_alarms: dict[str, dict[str, dict[str, dict[str, list[bool]]]]] = {
        horizon: {burden: {str(cell): {str(seed): [] for seed in encoders} for cell in range(cell_count)} for burden in burdens}
        for horizon in horizons
    }
    for record in records:
        horizon, burden, cell = str(record["horizon"]), str(record["burden"]), str(record["cell_index"])
        if not _coverage_ok(record):
            continue
        coverage[horizon][burden][cell] += 1
        combined = bool(record["combined"].get(readout, False))
        alarms[horizon][burden][cell].append(combined)
        if readout != spec.DIAGNOSTIC_READOUT:
            for seed in encoders:
                per_encoder_alarms[horizon][burden][cell][str(seed)].append(
                    bool(record["alarms"][readout][str(seed)])
                )

    def rates(alarm_map: Mapping[str, Mapping[str, list[bool]]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for horizon in horizons:
            out[horizon] = {}
            for burden in burdens:
                cell_rates: dict[str, Any] = {}
                for cell in range(cell_count):
                    values = alarm_map[horizon][burden][str(cell)]
                    cell_rates[str(cell)] = clopper_pearson(sum(1 for value in values if value), len(values))
                out[horizon][burden] = cell_rates
        return out

    cell_rates = rates(alarms)
    macro: dict[str, dict[str, Any]] = {}
    family_rates: dict[str, dict[str, dict[str, Any]]] = {}
    non_duration: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        macro[horizon] = {}
        family_rates[horizon] = {}
        non_duration[horizon] = {}
        for burden in burdens:
            available = [
                float(cell_rates[horizon][burden][str(cell)]["rate"])
                for cell in range(cell_count)
                if cell_rates[horizon][burden][str(cell)]["rate"] is not None
            ]
            macro[horizon][burden] = {
                "rate": float(np.mean(available)) if available else None,
                "cells": len(available),
            }
            family_rates[horizon][burden] = {}
            for family in spec.FAMILIES:
                family_cells = [
                    cell
                    for cell in range(cell_count)
                    if _cell_label(cell)[0] == family
                ]
                family_values = [
                    float(cell_rates[horizon][burden][str(cell)]["rate"])
                    for cell in family_cells
                    if cell_rates[horizon][burden][str(cell)]["rate"] is not None
                ]
                family_rates[horizon][burden][family] = {
                    "rate": float(np.mean(family_values)) if family_values else None,
                    "cells": len(family_values),
                }
            non_duration_values = [
                family_rates[horizon][burden][family]["rate"]
                for family in spec.FAMILIES
                if family != spec.DURATION_FAMILY and family_rates[horizon][burden][family]["rate"] is not None
            ]
            non_duration[horizon][burden] = {
                "rate": float(np.mean(non_duration_values)) if non_duration_values else None,
                "families": len(non_duration_values),
            }

    min_realized = max(1, int(plan.episodes_per_cell * spec.COVERAGE_FLOOR))
    coverage_checks = [
        {
            "horizon": horizon,
            "burden": burden,
            "cell": cell,
            "realized": coverage[horizon][burden][str(cell)],
            "required": min_realized,
            "pass": coverage[horizon][burden][str(cell)] >= min_realized,
        }
        for horizon in horizons
        for burden in burdens
        for cell in range(cell_count)
    ]
    coverage_ok = all(item["pass"] for item in coverage_checks)

    def gate_checks(rate_map: Mapping[str, Any]) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for horizon in horizons:
            fpr = macro_for(rate_map, horizon, spec.BURDEN_LABELS[0])
            checks.append({
                "check": f"fpr:{horizon}",
                "value": fpr,
                "threshold": spec.FPR_MAX,
                "pass": fpr is not None and fpr <= spec.FPR_MAX,
            })
            for burden in (spec.BURDEN_LABELS[1], spec.BURDEN_LABELS[2]):
                detection = macro_for(rate_map, horizon, burden)
                checks.append({
                    "check": f"detection:{horizon}:{burden}",
                    "value": detection,
                    "threshold": spec.DETECTION_MIN,
                    "pass": detection is not None and detection >= spec.DETECTION_MIN,
                })
                for family in spec.FAMILIES:
                    value = rate_map[horizon][burden]["family"][family]["rate"]
                    checks.append({
                        "check": f"family:{horizon}:{burden}:{family}",
                        "value": value,
                        "threshold": spec.FAMILY_FLOOR,
                        "pass": value is not None and value >= spec.FAMILY_FLOOR,
                    })
                non_duration_value = non_duration_map_for(rate_map, horizon, burden)
                checks.append({
                    "check": f"non_duration:{horizon}:{burden}",
                    "value": non_duration_value,
                    "threshold": spec.NON_DURATION_FLOOR,
                    "pass": non_duration_value is not None and non_duration_value >= spec.NON_DURATION_FLOOR,
                })
            checks.append({
                "check": f"monotonic:{horizon}",
                "value": [
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[0]),
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[1]),
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[2]),
                ],
                "pass": _monotonic(
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[0]),
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[1]),
                    macro_for(rate_map, horizon, spec.BURDEN_LABELS[2]),
                ),
            })
        return checks

    # Build the same nested structure for per-encoder rates.
    per_encoder_rates: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed in encoders:
        encoder_map = {
            horizon: {
                burden: {
                    str(cell): per_encoder_alarms[horizon][burden][str(cell)][str(seed)]
                    for cell in range(cell_count)
                }
                for burden in burdens
            }
            for horizon in horizons
        }
        per_encoder_rates[str(seed)] = _rates_from_alarm_map(encoder_map, horizons, burdens, cell_count)

    combined_checks = gate_checks(_macro_view(cell_rates, horizons, burdens, cell_count))
    encoder_pass: dict[str, bool] = {}
    encoder_checks: dict[str, list[dict[str, Any]]] = {}
    for seed in encoders:
        rates_view = _macro_view(per_encoder_rates[str(seed)], horizons, burdens, cell_count)
        checks = gate_checks(rates_view)
        encoder_checks[str(seed)] = checks
        core = [item for item in checks if item["check"].split(":")[0] in ("fpr", "detection", "family", "non_duration")]
        encoder_pass[str(seed)] = all(item["pass"] for item in core)
    robustness = sum(1 for value in encoder_pass.values() if value) >= 2

    return {
        "readout": readout,
        "coverage": coverage,
        "coverage_checks": coverage_checks,
        "coverage_ok": coverage_ok,
        "cell_rates": cell_rates,
        "macro": macro,
        "family_rates": family_rates,
        "non_duration": non_duration,
        "checks": combined_checks,
        "encoder_checks": encoder_checks,
        "encoder_pass": encoder_pass,
        "encoder_robustness_ok": robustness,
        "qualifies": bool(
            coverage_ok
            and all(item["pass"] for item in combined_checks)
            and robustness
        ) if readout != spec.DIAGNOSTIC_READOUT else False,
    }


def _rates_from_alarm_map(
    alarm_map: Mapping[str, Mapping[str, Mapping[str, list[bool]]]],
    horizons: Sequence[str],
    burdens: Sequence[str],
    cell_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        horizon: {
            burden: {
                str(cell): clopper_pearson(
                    sum(1 for value in alarm_map[horizon][burden][str(cell)] if value),
                    len(alarm_map[horizon][burden][str(cell)]),
                )
                for cell in range(cell_count)
            }
            for burden in burdens
        }
        for horizon in horizons
    }


def _macro_view(
    cell_rates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    horizons: Sequence[str],
    burdens: Sequence[str],
    cell_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    view: dict[str, dict[str, dict[str, Any]]] = {}
    for horizon in horizons:
        view[horizon] = {}
        for burden in burdens:
            family_map: dict[str, dict[str, Any]] = {}
            available: list[float] = []
            for family in spec.FAMILIES:
                family_cells = [cell for cell in range(cell_count) if _cell_label(cell)[0] == family]
                values = [
                    float(cell_rates[horizon][burden][str(cell)]["rate"])
                    for cell in family_cells
                    if cell_rates[horizon][burden][str(cell)]["rate"] is not None
                ]
                family_map[family] = {"rate": float(np.mean(values)) if values else None}
                available.extend(values)
            view[horizon][burden] = {
                "macro": float(np.mean(available)) if available else None,
                "family": family_map,
            }
    return view


def macro_for(rate_map: Mapping[str, Any], horizon: str, burden: str) -> float | None:
    return rate_map[horizon][burden]["macro"]


def non_duration_map_for(rate_map: Mapping[str, Any], horizon: str, burden: str) -> float | None:
    family_map = rate_map[horizon][burden]["family"]
    values = [
        family_map[family]["rate"]
        for family in spec.FAMILIES
        if family != spec.DURATION_FAMILY and family_map[family]["rate"] is not None
    ]
    return float(np.mean(values)) if values else None


def _monotonic(control: float | None, mid: float | None, elevated: float | None) -> bool:
    if control is None or mid is None or elevated is None:
        return False
    return bool(elevated >= mid >= control)


def aggregate_test_records(
    readout: str,
    records: Sequence[Mapping[str, Any]],
    plan: spec.Plan,
    encoders: Sequence[int] = spec.ENCODER_SEEDS,
) -> dict[str, Any]:
    """Public wrapper used by behavior fixtures and independent checks."""
    return _aggregate_readout(readout, records, plan, encoders)


# --- decision ------------------------------------------------------------------


def decide(
    aggregates: Mapping[str, Mapping[str, Any]],
    coverage_ok: bool,
    statuses: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Frozen action rule; never a model negative for invalid coverage."""
    resolved: dict[str, str] = {}
    for readout in spec.READOUTS:
        if statuses is not None and readout in statuses:
            resolved[readout] = str(statuses[readout])
            continue
        aggregate = aggregates.get(readout)
        if aggregate is None or not aggregate.get("checks"):
            resolved[readout] = spec.READOUT_INVALID
        elif aggregate.get("qualifies") and coverage_ok:
            resolved[readout] = spec.READOUT_QUALIFIES
        else:
            resolved[readout] = spec.READOUT_FAILS
    statuses = resolved
    if not coverage_ok or any(value == spec.READOUT_INVALID for value in statuses.values()):
        action = spec.ACTION_INVALID_IMPLEMENTATION
        rationale = "coverage invalid or a readout could not be computed; implementation defect"
    elif statuses["POOLED"] == spec.READOUT_QUALIFIES:
        action = spec.ACTION_KEEP_POOLED
        rationale = "pooled file embedding passes both gates; retain it"
    elif statuses["PREPOOL"] == spec.READOUT_QUALIFIES:
        action = spec.ACTION_CHANGE_READOUT
        rationale = "only the pre-pooling distribution readout passes; change aggregation"
    elif statuses["RAW"] == spec.READOUT_QUALIFIES:
        action = spec.ACTION_TARGET_REPRESENTATION
        rationale = "raw features pass while both learned readouts fail; target representation changes"
    else:
        action = spec.ACTION_STOP_SCALE
        rationale = "no readout passes; stop scaling this configuration (never an absence claim)"
    return {
        "action": action,
        "rationale": rationale,
        "statuses": statuses,
        "learned_route_valid": statuses["POOLED"] == spec.READOUT_QUALIFIES
        or statuses["PREPOOL"] == spec.READOUT_QUALIFIES,
        "raw_qualifies": statuses["RAW"] == spec.READOUT_QUALIFIES,
        "coverage_ok": bool(coverage_ok),
        "priority": list(spec.READOUT_PRIORITY),
    }


def decision_from_fixture(statuses: Mapping[str, str], coverage_ok: bool = True) -> dict[str, Any]:
    """Behavior fixture entry: evaluate the decision rule on throwaway statuses."""
    aggregates = {
        readout: {
            "checks": [{"check": "fixture", "pass": statuses.get(readout) == spec.READOUT_QUALIFIES}],
            "qualifies": statuses.get(readout) == spec.READOUT_QUALIFIES,
        }
        for readout in spec.READOUTS
    }
    return decide(aggregates, coverage_ok=coverage_ok, statuses=statuses)


__all__ = [
    "Chunk",
    "HistorySpace",
    "LENGTH_KEY",
    "POOLED_PREFIX",
    "PREPOOL_PREFIX",
    "RAW_KEY",
    "ReferenceSpace",
    "aggregate_test_records",
    "apply_standardizer",
    "build_history_space",
    "clopper_pearson",
    "decide",
    "decision_from_fixture",
    "evaluate_chunks",
    "fit_reference_space",
    "gaussian_kernel",
    "item_keys",
    "macro_for",
    "majority_alarm",
    "median_pairwise_distance",
    "mmd2_unbiased",
    "non_duration_map_for",
    "null_threshold",
    "readout_key",
    "squared_distances",
    "standardizer",
]
