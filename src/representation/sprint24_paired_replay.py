"""Offline replay of persisted Sprint 24 paired capture, without model calls."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from representation.sprint24_paired_journal import load_capture_batches


def _validate_calendar(event: dict[str, Any]) -> None:
    """Reject missing, overlapping or out-of-window synthetic source schedules."""
    from representation import sprint24_paired_official as official
    from representation.sprint24_controlled_data_execution import build_history_calendar
    from representation.sprint24_business_horizon import DAY_S
    from synth.generator import derive_pair_seed

    calendar = build_history_calendar(str(event["history_id"]))
    if event.get("calendar") != calendar:
        raise ValueError("REFUSED-IDENTITY: synthetic calendar drift")
    c1 = float(calendar["c1_rel_s"])
    kind = event["kind"]
    pre_roles = {"reference_pairs": ("reference", official.REFERENCE_INDEX_BASE, 0.0),
                 "bandwidth_pairs": ("bandwidth", official.BANDWIDTH_INDEX_BASE, 0.5),
                 "null_pairs": ("null", official.NULL_INDEX_BASE, 0.25)}
    if kind == "stage2_pairs":
        collections = [(event["abnormal"], event["mixture_scenario_id"], "mixture-abnormal"),
                       (event["ordinary"], event["mixture_scenario_id"], "mixture-ordinary"),
                       (event["control"], event["control_scenario_id"], "control")]
    elif kind == "stage1_pairs":
        collections = [(event["pairs"], event["scenario_id"], "scored-pair")]
    else:
        role, _, _ = pre_roles[kind]
        collections = [(event["pairs"], event["scenario_id"], role)]
    occupied: dict[str, list[tuple[float, float]]] = {}
    indices: set[int] = set()
    windows = {w["period_key"]: w for w in calendar["weeks"] + calendar["months"]}
    for pairs, scenario, role in collections:
        if not scenario:
            raise ValueError("REFUSED-IDENTITY: missing scenario identity")
        for pair in pairs:
            index = int(pair["pair_index"])
            if index in indices or int(pair["pair_seed"]) != derive_pair_seed(int(event["history_seed"]), index):
                raise ValueError("REFUSED-IDENTITY: repeated or mismatched source seed")
            indices.add(index)
            if pair.get("scenario_id") != scenario or pair.get("role") != role or pair.get("time_basis") != "planned synthetic calendar":
                raise ValueError("REFUSED-IDENTITY: missing scenario or role linkage")
            start, end = float(pair["planned_start_rel_s"]), float(pair["planned_end_rel_s"])
            _equal(float(pair["c1_rel_s"]), c1)
            _equal(float(pair["sample_interval_s"]), 0.125)
            if kind in pre_roles:
                _, base_index, fraction = pre_roles[kind]
                slot = index - base_index
                spacing = 38.5 * DAY_S / official.REFERENCE_SLOTS_PER_HISTORY
                _equal(float(pair["window_start_rel_s"]), c1 - 38.5 * DAY_S)
                _equal(float(pair["window_end_rel_s"]), c1)
                if not 0 <= slot < official.REFERENCE_SLOTS_PER_HISTORY or end > c1:
                    raise ValueError("REFUSED-IDENTITY: invalid pre-c1 source")
                _equal(end, c1 - (official.REFERENCE_SLOTS_PER_HISTORY - slot - fraction) * spacing)
                _equal(end - start, int(pair["observed_waveform_steps"]) * 0.125)
                if end - start >= spacing / 4:
                    raise ValueError("REFUSED-IDENTITY: pre-c1 slot overlap")
            else:
                window = windows.get(pair["window_key"])
                if window is None:
                    raise ValueError("REFUSED-IDENTITY: unrecognized scenario window")
                ws, we = float(window["start_rel_s"]), float(window["end_rel_s"])
                _equal(float(pair["window_start_rel_s"]), ws)
                _equal(float(pair["window_end_rel_s"]), we)
                slot, count = int(pair["slot"]), int(pair["n_slots"])
                if kind == "stage1_pairs":
                    expected_window = calendar["weeks"][1 + index // 40]
                    if count != 40 or slot != index % 40 or window != expected_window:
                        raise ValueError("REFUSED-IDENTITY: stage-1 scenario slot drift")
                elif kind == "stage2_pairs":
                    expected_window = calendar["weeks"][1] if event["horizon"] == "week" else calendar["months"][1]
                    if count != int(event["size"]) or window != expected_window:
                        raise ValueError("REFUSED-IDENTITY: stage-2 scenario window drift")
                if not 0 <= slot < count or start <= c1 or start < ws or end > we:
                    raise ValueError("REFUSED-IDENTITY: source outside evaluation window")
                _equal(end, ws + (slot + 1) * (we - ws) / count)
                _equal(end - start, max(int(pair["twin_waveform_steps"]), int(pair["variant_waveform_steps"])) * 0.125)
            occupied.setdefault(str(scenario), []).append((start, end))
    for intervals in occupied.values():
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("REFUSED-IDENTITY: overlapping sources within a scenario")


def _validate_attempts(
    event: dict[str, Any], pending: dict[tuple[str, str, int], list[dict[str, Any]]],
    source_starts: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    """Bind actual started/completed attempt events to the selected pair."""
    kind, history = event["kind"], str(event["history_id"])
    if kind == "stage2_pairs":
        collections = [(event["abnormal"], "mixture-abnormal"), (event["ordinary"], "mixture-ordinary"),
                       (event["control"], "control")]
    else:
        roles = {"reference_pairs": "reference", "bandwidth_pairs": "bandwidth",
                 "null_pairs": "null", "stage1_pairs": "stage1"}
        collections = [(event["pairs"], roles[kind])]
    for pairs, role in collections:
        for pair in pairs:
            key = (history, role, int(pair["pair_index"]))
            source_start = source_starts.pop(key, None)
            if source_start is None or int(source_start["pair_seed"]) != int(pair["pair_seed"]):
                raise ValueError("REFUSED-IDENTITY: missing source generation start")
            records = pending.pop(key, [])
            attempts = pair.get("attempts")
            if not isinstance(attempts, list) or len(attempts) != int(pair["n_attempts"]) or len(records) != 2 * len(attempts):
                raise ValueError("REFUSED-IDENTITY: missing actual retry events")
            for index, attempt in enumerate(attempts):
                started, completed = records[2 * index:2 * index + 2]
                if started["status"] != "STARTED" or completed["status"] not in ("ACCEPTED", "REJECTED"):
                    raise ValueError("REFUSED-IDENTITY: invalid attempt lifecycle")
                for field in ("pair_seed", "source_id", "group_id"):
                    if started[field] != pair[field] or completed[field] != pair[field]:
                        raise ValueError("REFUSED-IDENTITY: retry source identity mismatch")
                for field in ("attempt", "attempt_seed", "status", "strength", "reason"):
                    if completed[field] != attempt[field]:
                        raise ValueError("REFUSED-IDENTITY: retry outcome mismatch")
                if started["attempt"] != attempt["attempt"] or started["attempt_seed"] != attempt["attempt_seed"]:
                    raise ValueError("REFUSED-IDENTITY: retry start mismatch")
            if attempts and ((attempts[-1]["status"] == "ACCEPTED") != (pair["disposition"] == "PAIRED")):
                raise ValueError("REFUSED-IDENTITY: retry disposition mismatch")


def replay_journal(root: Path) -> dict[str, Any]:
    """Verify capture files and recompute calibration, groups and population scores."""
    from representation import sprint24_paired_official as official
    from representation.sprint24_weekly_monthly_endpoints import mmd2_unbiased

    batches = iter(load_capture_batches(root))
    rows: dict[tuple[str, int, str], np.ndarray] = {}
    matrices: dict[tuple[str, int], np.ndarray] = {}
    nulls: dict[tuple[str, int], float] = {}
    current_pairs: dict[str, dict[int, dict[str, Any]]] = {}
    disjoint: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    populations: list[dict[str, Any]] = []
    completion: dict[str, Any] | None = None
    captured_rows = 0
    expected_files: dict[str, list[str]] = {}
    pre_sources: dict[str, dict[str, set[int]]] = {}
    stage2_pending: dict[str, dict[str, Any]] = {}
    pending_attempts: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    source_starts: dict[tuple[str, str, int], dict[str, Any]] = {}
    with (Path(root) / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["kind"]
            if kind == "source_started":
                key = (str(event["history_id"]), str(event["role"]), int(event["pair_index"]))
                if key in source_starts:
                    raise ValueError("REFUSED-IDENTITY: duplicate source generation start")
                source_starts[key] = event
            if kind == "injection_attempt":
                key = (str(event["history_id"]), str(event["role"]), int(event["pair_index"]))
                pending_attempts.setdefault(key, []).append(event)
            if kind in ("reference_pairs", "bandwidth_pairs", "null_pairs", "stage1_pairs", "stage2_pairs"):
                _validate_calendar(event)
                _validate_attempts(event, pending_attempts, source_starts)
                if kind in ("reference_pairs", "bandwidth_pairs", "null_pairs"):
                    history = str(event["history_id"])
                    suffix = {"reference_pairs": "-REF", "bandwidth_pairs": "-BANDWIDTH",
                              "null_pairs": "-NULL"}[kind]
                    if not history.endswith(suffix):
                        raise ValueError("REFUSED-IDENTITY: pre-c1 role history mismatch")
                    base_history = history.removesuffix(suffix)
                    source_sets = pre_sources.setdefault(base_history, {})
                    source_set = {int(pair["pair_seed"]) for pair in event["pairs"]}
                    if any(source_set & previous for previous in source_sets.values()):
                        raise ValueError("REFUSED-IDENTITY: pre-c1 sources overlap")
                    expected_n = 64 if kind == "null_pairs" else 220
                    if len(source_set) != expected_n:
                        raise ValueError("REFUSED-IDENTITY: pre-c1 source count mismatch")
                    source_sets[kind] = source_set
                    expected_files[history] = [pair["twin_file_id"] for pair in event["pairs"]]
                elif kind == "stage1_pairs":
                    history = str(event["history_id"])
                    if set(pre_sources.get(history, {})) != {"reference_pairs", "bandwidth_pairs", "null_pairs"}:
                        raise ValueError("REFUSED-IDENTITY: missing pre-c1 source lineage")
                    source_set = {int(pair["pair_seed"]) for pair in event["pairs"]}
                    if any(source_set & previous for previous in pre_sources[history].values()):
                        raise ValueError("REFUSED-IDENTITY: scored and pre-c1 sources overlap")
                    admitted = [pair for pair in event["pairs"] if pair["disposition"] == "PAIRED"]
                    expected_files[history] = ([pair["twin_file_id"] for pair in admitted]
                                               + [pair["abnormal_file_id"] for pair in admitted])
                elif kind == "stage2_pairs":
                    history = str(event["history_id"])
                    base_history = history.removesuffix("-STAGE2")
                    if set(pre_sources.get(base_history, {})) != {"reference_pairs", "bandwidth_pairs", "null_pairs"}:
                        raise ValueError("REFUSED-IDENTITY: missing population reference lineage")
                    source_set = {int(pair["pair_seed"]) for role in ("abnormal", "ordinary", "control") for pair in event[role]}
                    if any(source_set & previous for previous in pre_sources[base_history].values()):
                        raise ValueError("REFUSED-IDENTITY: population and pre-c1 sources overlap")
                    stage2_pending[history] = event
                    expected_files[history] = (
                        [pair["abnormal_file_id"] for pair in event["abnormal"] if pair["disposition"] == "PAIRED"]
                        + [pair["twin_file_id"] for pair in event["ordinary"] if pair["disposition"] == "PAIRED"]
                        + [pair["twin_file_id"] for pair in event["control"] if pair["disposition"] == "PAIRED"]
                    )
            if kind == "capture":
                saved, matrix = next(batches)
                captured_rows += len(matrix)
                if saved != event:
                    raise ValueError("REFUSED-IDENTITY: capture journal changed during reload")
                history, seed = str(event["history_id"]), int(event["model_seed"])
                if [str(item["file_id"]) for item in event["files"]] != expected_files.get(history):
                    raise ValueError("REFUSED-IDENTITY: captured files do not match source lineage")
                matrices[(history, seed)] = matrix.astype(np.float64)
                for file, vector in zip(event["files"], matrix):
                    rows[(history, seed, str(file["file_id"]))] = vector.astype(np.float64)
            elif kind == "null_scores":
                history, seed = str(event["history_id"]), int(event["model_seed"])
                ref_history = history.removesuffix("-NULL") + "-BANDWIDTH"
                reference = {seed: {"matrix": matrices[(ref_history, seed)]}}
                # Recompute the reference bandwidth independently of the saved threshold.
                from scipy.spatial.distance import pdist
                sigma = float(np.median(pdist(reference[seed]["matrix"])))
                _equal(sigma, event["sigma"])
                matrix = matrices[(history, seed)]
                k, count = official.base.GROUP_SIZE_K, official.NULL_GROUPS_PER_CELL
                scores = [official.paired_mmd2_unbiased(matrix[i*k:(i+1)*k],
                          matrix[(i+count)*k:(i+count+1)*k], sigma) for i in range(count)]
                for actual, expected in zip(scores, event["scores"], strict=True):
                    _equal(actual, expected)
                q95 = float(np.quantile(scores, 0.95))
                _equal(q95, event["q95"])
                nulls[(history.removesuffix("-NULL"), seed)] = q95
            elif kind == "stage1_pairs":
                current_pairs[str(event["history_id"])] = {int(p["pair_index"]): p for p in event["pairs"]}
            elif kind == "stage1_disjoint_test":
                from representation.sprint24_paired_disjoint import source_disjoint_test
                from scipy.spatial.distance import pdist
                history = str(event["history_id"])
                pairs = list(current_pairs[history].values())
                admitted = [p for p in pairs if p["disposition"] == "PAIRED"]
                features: dict[int, dict[str, np.ndarray]] = {}
                sigma: dict[int, float] = {}
                for seed in official.base.FROZEN_ENCODER_SEEDS:
                    seed = int(seed)
                    sigma[seed] = float(np.median(pdist(matrices[(history + "-BANDWIDTH", seed)])))
                    if admitted:
                        features[seed] = {
                            "twin": np.stack([rows[(history, seed, p["twin_file_id"])] for p in admitted]),
                            "abnormal": np.stack([rows[(history, seed, p["abnormal_file_id"])] for p in admitted]),
                        }
                actual = source_disjoint_test(
                    [p["pair_index"] for p in pairs], [p["pair_index"] for p in admitted],
                    features, sigma, int(event["history_seed"]), str(event["family"]), float(event["severity"]),
                )
                if actual != event["result"]:
                    raise ValueError("REFUSED-IDENTITY: source-disjoint replay mismatch")
                disjoint[history] = actual
            elif kind == "stage1_groups":
                history = str(event["history_id"])
                pairs = current_pairs[history]
                null_median = float(np.median([nulls[(history, int(s))] for s in official.base.FROZEN_ENCODER_SEEDS]))
                outcomes: list[dict[str, Any]] = []
                for group in event["outcomes"]:
                    per_seed: list[float] = []
                    for seed in official.base.FROZEN_ENCODER_SEEDS:
                        seed = int(seed)
                        ref = matrices[(history + "-BANDWIDTH", seed)]
                        from scipy.spatial.distance import pdist
                        sigma = float(np.median(pdist(ref)))
                        selected = [pairs[int(i)] for i in group["pair_indices"]]
                        x = np.stack([rows[(history, seed, p["twin_file_id"])] for p in selected])
                        y = np.stack([rows[(history, seed, p["abnormal_file_id"])] for p in selected])
                        per_seed.append(official.paired_mmd2_unbiased(x, y, sigma))
                    for actual, expected in zip(per_seed, group["e1_per_seed"], strict=True):
                        _equal(actual, expected)
                    median = float(np.median(per_seed))
                    _equal(median, group["median_e1"])
                    _equal(null_median, group["null_q95_median"])
                    if bool(median > null_median) != group["response"]:
                        raise ValueError("REFUSED-IDENTITY: group response mismatch")
                    outcomes.append({"pair_indices": group["pair_indices"], "per_seed": per_seed,
                                     "median_e1": median, "response": bool(median > null_median)})
                summary = official.descriptive_cell_summary([v["median_e1"] for v in outcomes], null_median)
                groups.append({"history_id": history, "family": event["family"], "severity": event["severity"], "outcomes": outcomes, "summary": summary})
                groups[-1]["disjoint_test"] = disjoint[history]
            elif kind == "stage1_sensitivity":
                history = str(event["history_id"])
                group = groups[-1]
                expected = event["sensitivity"]
                if group["outcomes"]:
                    seed = int(official.base.FROZEN_ENCODER_SEEDS[0])
                    selected = [current_pairs[history][int(i)] for i in group["outcomes"][0]["pair_indices"]]
                    twin = np.stack([rows[(history, seed, p["twin_file_id"])] for p in selected])
                    abnormal = np.stack([rows[(history, seed, p["abnormal_file_id"])] for p in selected])
                    reference = matrices[(history + "-REF", seed)]
                    e2 = official.e2_occupancy(twin, abnormal, reference)
                    d = official.within_prototype_diagnostics(twin, abnormal, reference)
                    _equal(e2["e2"], expected["e2"]["e2"])
                    _equal(d["coverage"], expected["d"]["coverage"])
                    if bool(d["valid"]) != expected["d"]["valid"]:
                        raise ValueError("REFUSED-IDENTITY: sensitivity coverage mismatch")
                    if d["valid"]:
                        _equal(d["d_loc"], expected["d"]["d_loc"])
                    group["sensitivity"] = expected
                elif expected.get("status") != "UNAVAILABLE":
                    raise ValueError("REFUSED-IDENTITY: unexpected sensitivity")
                else:
                    group["sensitivity"] = expected
            elif kind == "stage2_scores":
                from representation.sprint24_paired_estimators import stratified_mmd2_unbiased
                if event.get("estimator") != "stratified-realized-mixture-mmd2-v1":
                    raise ValueError("REFUSED-IDENTITY: unsupported historical population estimator")
                history = str(event["history_id"])
                source_event = stage2_pending.pop(history, None)
                if source_event is None:
                    raise ValueError("REFUSED-IDENTITY: missing population scenario lineage")
                mixture_expected = (
                    [p["abnormal_file_id"] for p in source_event["abnormal"] if p["disposition"] == "PAIRED"]
                    + [p["twin_file_id"] for p in source_event["ordinary"] if p["disposition"] == "PAIRED"]
                )
                control_expected = [p["twin_file_id"] for p in source_event["control"] if p["disposition"] == "PAIRED"]
                if event["mixture_ids"] != mixture_expected or event["control_ids"] != control_expected:
                    raise ValueError("REFUSED-IDENTITY: population membership disagrees with source lineage")
                ref_history = history.removesuffix("-STAGE2") + "-REF"
                actual: dict[str, Any] = {}
                if set(event["mixture_ids"]) & set(event["control_ids"]):
                    raise ValueError("REFUSED-IDENTITY: overlapping population arms")
                for seed in official.base.FROZEN_ENCODER_SEEDS:
                    seed = int(seed)
                    reference = matrices[(ref_history, seed)]
                    from scipy.spatial.distance import pdist
                    bandwidth = matrices[(history.removesuffix("-STAGE2") + "-BANDWIDTH", seed)]
                    sigma = float(np.median(pdist(bandwidth)))
                    mixture = np.stack([rows[(history, seed, f)] for f in event["mixture_ids"]])
                    control = np.stack([rows[(history, seed, f)] for f in event["control_ids"]])
                    n_abnormal = int(event["n_abnormal_paired"])
                    if n_abnormal + int(event["n_ordinary_paired"]) != len(mixture):
                        raise ValueError("REFUSED-IDENTITY: mixture component denominator mismatch")
                    _equal(n_abnormal / len(mixture), event["realized_abnormal_fraction"])
                    transition = stratified_mmd2_unbiased(
                        mixture[:n_abnormal], mixture[n_abnormal:], reference, sigma,
                    )
                    control_score = mmd2_unbiased(control, reference, sigma)
                    saved = event["per_seed"][str(seed)]
                    _equal(transition, saved["e1_transition"])
                    _equal(control_score, saved["e1_control"])
                    if (len(mixture), len(control), len(reference)) != (saved["n_mixture"], saved["n_control"], saved["n_reference"]):
                        raise ValueError("REFUSED-IDENTITY: population denominator mismatch")
                    actual[str(seed)] = {"e1_transition": transition, "e1_control": control_score}
                populations.append({"history_id": history, "horizon": event["horizon"], "size": event["size"], "burden": event["burden"], "per_seed": actual})
            elif kind == "stage2_abstention":
                history = str(event["history_id"])
                if stage2_pending.pop(history, None) is None:
                    raise ValueError("REFUSED-IDENTITY: missing abstention scenario lineage")
            elif kind in ("source_failed", "run_failure", "terminal_failure"):
                raise ValueError("REFUSED-IDENTITY: journal records a failed run")
            elif kind == "capture_complete":
                completion = event
    if completion is None or stage2_pending or pending_attempts or source_starts or int(completion["forwards"]) != captured_rows:
        raise ValueError("REFUSED-IDENTITY: missing completion or forward denominator mismatch")
    expected = {f"{h}|{f}|{s}" for h in official.base.PAIRED_HISTORY_IDS for f, s in official.CELL_GRID}
    keyed = {f"{g['history_id']}|{g['family']}|{g['severity']}": g for g in groups}
    if set(keyed) == expected:
        keys = sorted(expected)
        holm: dict[str, Any] = {
            "status": "CONDITIONAL-ON-EXCHANGEABILITY",
            "family": "162 history-by-family-by-severity cells", "alpha": 0.05,
            "cell_order": keys,
            "adjustments": official.holm_over_families(
                [float(keyed[key]["disjoint_test"]["p_value"]) for key in keys], alpha=0.05),
        }
    else:
        holm = {"status": "UNSUPPORTED-SUBSET", "reason": "global 162-cell family incomplete"}
    arms: dict[str, list[float]] = {}
    for population in populations:
        key = f"{population['horizon']}|{population['size']}|{population['burden']}"
        median = float(np.median([value["e1_transition"] for value in population["per_seed"].values()]))
        arms.setdefault(key, []).append(median)
    return {"status": "REPLAY-VERIFIED", "stage1": groups, "stage2": populations,
            "holm_by_family": holm,
            "stage2_bootstraps": {key: official.cluster_bootstrap_median(values) for key, values in arms.items()}}


def _equal(actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"REFUSED-IDENTITY: replay mismatch {actual!r} != {expected!r}")
