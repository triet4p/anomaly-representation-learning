#!/usr/bin/env python3
"""Hash-guarded no-write Sprint 18 control/eligibility diagnostic.

--smoke evaluates only the fixed disposable fixture and proves instrumentation
and temporary persistence. --study is the one-shot, review-gated Task 62 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DIAGNOSTIC_ID = "sprint18-control-eligibility-mechanism-diagnostic-v1"
CONTRACT_PATH = ROOT / "experiments/sprint18-control-eligibility-diagnostic-v1.md"
SCHEMA_PATH = ROOT / "experiments/sprint18-control-eligibility-diagnostic-v1.schema.json"
V2_RUNNER_PATH = ROOT / "experiments/sprint18_task58_feasibility-v2.py"
V2_CONTRACT_PATH = ROOT / "experiments/sprint18-isolated-feasibility-study-v2.md"
V1_RUNNER_PATH = ROOT / "experiments/sprint18_task58_feasibility-v1.py"
V1_CONTRACT_PATH = ROOT / "experiments/sprint18-isolated-feasibility-study-v1.md"
OUTPUT_DIR = ROOT / "artifacts/sprint-18/control-eligibility-diagnostic-v1"
ATTEMPT_PATH = OUTPUT_DIR / "attempt.json"
RESULT_PATH = OUTPUT_DIR / "result.json"
FIXTURE_SEED = 93061
STUDY_SEEDS = tuple(range(918100, 918108))
PROFILE_ORDER = ("sprint15-v7", "sprint18-c2")
PROFILE_RATE = {"sprint15-v7": 2.2e-5, "sprint18-c2": 3.3e-5}
PROFILE_CONFIG_HASH = {"sprint15-v7": "7de530a28d23", "sprint18-c2": "720e31e94681"}
BASE_SOURCE_CLOSURE_SHA256 = "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834"
V2_RUNNER_SHA256 = "d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6"
V2_CONTRACT_SHA256 = "8b8e28e489c393cd0ccf33d5844bab9a1d5e562904d671f5b94c122fc4b5ec42"
V1_RUNNER_SHA256 = "d65f01ee703d2f89b46c5b0138b3690f811edce304878204d9b1cbf14fa83111"
V1_CONTRACT_SHA256 = "1252707032aabfd95d99d1944140e57f8eff3610f7c8d35810e649c65a276bd0"
COHORTS = ("P", "W", "A")
RESET_CLASSES = ("preventive", "corrective", "overlap", "undetermined")
EXPECTED_COORDINATES = tuple(
    (seed, profile) for seed in STUDY_SEEDS for profile in PROFILE_ORDER
)


class ContractError(RuntimeError):
    """A frozen-input, trace, persistence, or one-shot guard failed."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise ContractError(f"cannot hash required frozen input {path}: {exc}") from exc


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object at {path}")
    return cast("dict[str, object]", value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not a JSON object")
    return cast("dict[str, object]", value)


def _load_core() -> dict[str, object]:
    if _file_sha256(V2_RUNNER_PATH) != V2_RUNNER_SHA256:
        raise ContractError("frozen v2 diagnostic runner digest mismatch")
    if _file_sha256(V2_CONTRACT_PATH) != V2_CONTRACT_SHA256:
        raise ContractError("frozen v2 diagnostic contract digest mismatch")
    try:
        return cast(
            "dict[str, object]",
            runpy.run_path(str(V2_RUNNER_PATH), run_name="sprint18_task61_v2_core"),
        )
    except Exception as exc:
        raise ContractError(f"cannot load frozen v2 diagnostic core: {exc}") from exc


def _contract_binding() -> dict[str, object]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- release-binding -->\s*```json\s*(.*?)\s*```", text, flags=re.DOTALL
    )
    if match is None:
        raise ContractError("diagnostic contract has no release-binding block")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid release-binding JSON: {exc}") from exc
    return _mapping(value, "diagnostic release binding")


def _verify_task_namespace(core: Mapping[str, object]) -> dict[str, object]:
    if tuple(sorted(set(STUDY_SEEDS))) != STUDY_SEEDS or len(STUDY_SEEDS) != 8:
        raise ContractError("the fixed Task 61 seed namespace is not eight ordered seeds")
    if tuple(sorted(set(EXPECTED_COORDINATES))) != EXPECTED_COORDINATES:
        raise ContractError("the fixed profile-seed coordinate order is inconsistent")
    base_namespace_fn = cast("object", core["_verify_namespace_disjointness"])
    base_namespace = cast("dict[str, object]", base_namespace_fn())  # type: ignore[operator]
    bindings = cast("Sequence[str]", core["BINDING_PATHS"])
    load_json = cast("object", core["_load_json"])
    recorded_seeds = cast("object", core["_recorded_seeds"])
    protected_direct: dict[str, list[int]] = {}
    for relative in bindings:
        binding = cast("dict[str, object]", load_json(relative))  # type: ignore[operator]
        seeds = set(cast("set[int]", recorded_seeds(binding, relative)))  # type: ignore[operator]
        overlap = sorted(seeds.intersection(STUDY_SEEDS))
        fixture_overlap = FIXTURE_SEED in seeds
        if overlap or fixture_overlap:
            raise ContractError(
                f"Task 61 namespace collides with {relative}: {overlap}, fixture={fixture_overlap}"
            )
        protected_direct[relative] = sorted(seeds)

    c3 = cast(
        "dict[str, object]",
        load_json("experiments/sprint18-role-binding-c3-v1.json"),  # type: ignore[operator]
    )
    ranges = c3.get("disjoint_seed_ranges_checked")
    if not isinstance(ranges, list):
        raise ContractError("frozen C3 binding has no recorded seed-range checks")
    checked_ranges: list[dict[str, object]] = []
    for item in ranges:
        if not isinstance(item, dict) or item.get("result") != "disjoint":
            raise ContractError("frozen C3 namespace evidence is malformed")
        source = str(item.get("source", ""))
        if "seeds" in item:
            occupied = {int(seed) for seed in cast("list[object]", item["seeds"])}
            study_overlap = sorted(occupied.intersection(STUDY_SEEDS))
            fixture_overlap = FIXTURE_SEED in occupied
            detail: dict[str, object] = {"seeds": sorted(occupied)}
        elif "min" in item and "max" in item:
            low, high = int(item["min"]), int(item["max"])
            if low > high:
                raise ContractError(f"invalid frozen seed range {source}")
            study_overlap = [seed for seed in STUDY_SEEDS if low <= seed <= high]
            fixture_overlap = low <= FIXTURE_SEED <= high
            detail = {"min": low, "max": high}
        else:
            raise ContractError(f"unsupported frozen namespace record {source}")
        if study_overlap or fixture_overlap:
            raise ContractError(
                f"Task 61 seed collision with {source}: {study_overlap}, fixture={fixture_overlap}"
            )
        checked_ranges.append({
            "source": source,
            **detail,
            "diagnostic_seed_overlap": study_overlap,
            "fixture_overlap": fixture_overlap,
            "result": "disjoint",
        })

    prior_study_seeds = set(cast("Sequence[int]", core["STUDY_SEEDS"]))
    if set(STUDY_SEEDS).intersection(prior_study_seeds):
        raise ContractError("Task 61 seeds collide with the frozen Task 59 study block")
    if FIXTURE_SEED in set(STUDY_SEEDS) | prior_study_seeds | {93058}:
        raise ContractError("disposable fixture collides with a frozen research seed")
    return {
        "diagnostic_seeds": list(STUDY_SEEDS),
        "profile_order": list(PROFILE_ORDER),
        "coordinate_order": [list(coordinate) for coordinate in EXPECTED_COORDINATES],
        "fixture_seed": FIXTURE_SEED,
        "recorded_binding_seed_values": protected_direct,
        "c3_recorded_range_checks": checked_ranges,
        "prior_task59_seeds": sorted(prior_study_seeds),
        "sealed_inputs_opened": False,
        "v1_v2_namespace_guard": base_namespace,
    }


def _verify_static_guards() -> dict[str, object]:
    core = _load_core()
    core_static_fn = cast("object", core["_verify_static_guards"])
    core_static = cast("dict[str, object]", core_static_fn())  # type: ignore[operator]
    base_closure = cast(
        "dict[str, str]", _mapping(core_static.get("source_closure"), "v2 source closure")
    )
    base_digest = str(core_static.get("source_closure_sha256", ""))
    if len(base_closure) != 47 or base_digest != BASE_SOURCE_CLOSURE_SHA256:
        raise ContractError("the frozen 47-file v2 source/config closure changed")

    extra_paths = {
        "experiments/sprint18_task58_feasibility-v1.py": V1_RUNNER_PATH,
        "experiments/sprint18-isolated-feasibility-study-v1.md": V1_CONTRACT_PATH,
        "experiments/sprint18_task58_feasibility-v2.py": V2_RUNNER_PATH,
        "experiments/sprint18-isolated-feasibility-study-v2.md": V2_CONTRACT_PATH,
        "experiments/sprint18-control-eligibility-diagnostic-v1.schema.json": SCHEMA_PATH,
    }
    expected_extra = {
        "experiments/sprint18_task58_feasibility-v1.py": V1_RUNNER_SHA256,
        "experiments/sprint18-isolated-feasibility-study-v1.md": V1_CONTRACT_SHA256,
        "experiments/sprint18_task58_feasibility-v2.py": V2_RUNNER_SHA256,
        "experiments/sprint18-isolated-feasibility-study-v2.md": V2_CONTRACT_SHA256,
    }
    source_closure = dict(base_closure)
    for relative, path in extra_paths.items():
        actual = _file_sha256(path)
        expected = expected_extra.get(relative)
        if expected is not None and actual != expected:
            raise ContractError(f"frozen historical input mismatch at {relative}")
        source_closure[relative] = actual

    schema_digest = source_closure[
        "experiments/sprint18-control-eligibility-diagnostic-v1.schema.json"
    ]
    runner_digest = _file_sha256(Path(__file__).resolve())
    contract_digest = _file_sha256(CONTRACT_PATH)
    source_digest = _sha256(_canonical_json(dict(sorted(source_closure.items()))))
    namespace = _verify_task_namespace(core)
    config_contract = _mapping(core_static.get("config_contract"), "frozen v2 config")
    if config_contract.get("profile_config_hashes") != PROFILE_CONFIG_HASH:
        raise ContractError("v7/C2 profile config hashes differ from the frozen diagnostic values")
    if config_contract.get("profile_rates") != PROFILE_RATE:
        raise ContractError("v7/C2 abrupt-A rates differ from the frozen diagnostic values")
    binding = _contract_binding()
    expected_binding = {
        "runner_sha256": runner_digest,
        "schema_sha256": schema_digest,
        "base_source_closure_sha256": BASE_SOURCE_CLOSURE_SHA256,
        "source_closure_sha256": source_digest,
        "profile_config_hashes": PROFILE_CONFIG_HASH,
        "profile_rates": PROFILE_RATE,
        "quota": config_contract.get("quota"),
        "diagnostic_id": DIAGNOSTIC_ID,
        "seed_order": list(STUDY_SEEDS),
        "profile_order": list(PROFILE_ORDER),
        "fixture_seed": FIXTURE_SEED,
        "coordinate_budget": len(EXPECTED_COORDINATES),
        "v1_runner_sha256": V1_RUNNER_SHA256,
        "v1_contract_sha256": V1_CONTRACT_SHA256,
        "v2_runner_sha256": V2_RUNNER_SHA256,
        "v2_contract_sha256": V2_CONTRACT_SHA256,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise ContractError(
                f"release-binding mismatch for {key}: {binding.get(key)!r} != {expected!r}"
            )
    try:
        schema = _read_object(SCHEMA_PATH)
    except ContractError:
        raise
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ContractError("diagnostic schema is not JSON Schema 2020-12")
    if schema.get("$id") != DIAGNOSTIC_ID:
        raise ContractError("diagnostic schema identifier changed")
    return {
        "source_closure": dict(sorted(source_closure.items())),
        "base_source_closure_sha256": base_digest,
        "source_closure_sha256": source_digest,
        "runner_sha256": runner_digest,
        "contract_sha256": contract_digest,
        "schema_sha256": schema_digest,
        "namespace": namespace,
        "config_contract": config_contract,
        "release_binding": binding,
        "core": core,
    }


def _freeze_digest(static: Mapping[str, object]) -> str:
    contract_digest = str(static["contract_sha256"])
    runner_digest = str(static["runner_sha256"])
    schema_digest = str(static["schema_sha256"])
    closure = cast("dict[str, str]", static["source_closure"])
    material = (
        b"SPRINT18-CONTROL-ELIGIBILITY-DIAGNOSTIC-FREEZE-v1\n"
        + bytes.fromhex(contract_digest)
        + bytes.fromhex(runner_digest)
        + bytes.fromhex(schema_digest)
        + _canonical_json({
            "source_closure": dict(sorted(closure.items())),
            "namespace": static["namespace"],
            "profile_config_hashes": PROFILE_CONFIG_HASH,
            "profile_rates": PROFILE_RATE,
            "coordinate_order": [list(item) for item in EXPECTED_COORDINATES],
            "fixture_seed": FIXTURE_SEED,
            "budget_unique_persisted_coordinates": len(EXPECTED_COORDINATES),
        })
    )
    return _sha256(material)


def _assert_expected_freeze(expected: str, actual: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != actual:
        raise ContractError("expected freeze digest does not match the reviewed diagnostic freeze")


def _count_map(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _nested_count_map(
    records: Sequence[Mapping[str, object]], outer: str, inner: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record[outer])][str(record[inner])] += 1
    return {
        key: {inner_key: counts[key][inner_key] for inner_key in sorted(counts[key])}
        for key in sorted(counts)
    }


def _maintenance_records(health: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    episodes = cast("Sequence[object]", getattr(health, "episodes"))
    for episode in episodes:
        kind_obj = getattr(episode, "kind")
        kind = str(getattr(kind_obj, "value", kind_obj))
        if kind != "maintenance":
            continue
        end_time = getattr(episode, "end_time")
        if end_time is None:
            raise ContractError("maintenance reset has no finite end")
        records.append({
            "episode_id": str(getattr(episode, "episode_id")),
            "robot_id": str(getattr(episode, "robot_id")),
            "start_time": float(getattr(episode, "start_time")),
            "end_time": float(end_time),
            "kind": kind,
        })
    return records


def _reset_intersections(
    robot_id: str,
    start: float,
    end: float,
    maintenance: Mapping[str, Sequence[Sequence[float]]],
    episodes: Sequence[Mapping[str, object]],
    health: object,
    ledger: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from synth import events as E

    wins = cast("dict[str, list[list[float]]]", dict(maintenance))
    if not E.window_intersects_reset(start, end, robot_id, wins):
        return {"intersects": False, "classification": "not_applicable", "windows": []}
    intersecting = [
        episode for episode in episodes
        if episode["robot_id"] == robot_id
        and float(episode["start_time"]) < end
        and start < float(episode["end_time"])
    ]
    if not intersecting:
        raise ContractError("reset predicate intersected but no maintenance episode reconciled")

    all_episodes = cast("Sequence[object]", getattr(health, "episodes"))
    classified: list[dict[str, object]] = []
    known_types: set[str] = set()
    unresolved = False
    for episode in intersecting:
        episode_id = str(episode["episode_id"])
        source_type = "undetermined"
        basis = "maintenance episode identity does not match a frozen generator class"
        linked_failure: str | None = None
        if episode_id.startswith("pmaint-"):
            source_type = "preventive"
            basis = "pmaint episode emitted by RobotHealthProcess._preventive_blocks"
        elif episode_id.startswith("maint-"):
            matches = [
                item for item in all_episodes
                if str(getattr(getattr(item, "kind"), "value", getattr(item, "kind"))) == "failure"
                and str(getattr(item, "robot_id")) == robot_id
                and float(getattr(item, "start_time")) == float(episode["start_time"])
            ]
            linked_rows = [
                event for event in ledger
                if event.get("maintenance_episode_id") == episode_id
                and event.get("robot_id") == robot_id
                and float(event["failure_time"]) == float(episode["start_time"])
            ]
            if len(matches) == 1 and len(linked_rows) <= 1:
                source_type = "corrective"
                linked_failure = str(linked_rows[0]["failure_id"]) if linked_rows else str(
                    getattr(matches[0], "episode_id")
                )
                basis = "maint episode emitted at a same-robot failure by RobotHealthProcess._run_robot"
            else:
                unresolved = True
                basis = "corrective episode lacks a unique same-time failure linkage"
        else:
            unresolved = True
        if source_type == "undetermined":
            unresolved = True
        else:
            known_types.add(source_type)
        classified.append({
            **dict(episode),
            "source_type": source_type,
            "cause_basis": basis,
            "associated_failure_id_or_episode": linked_failure,
        })

    if unresolved:
        classification = "undetermined"
    elif len(intersecting) > 1:
        classification = "overlap"
    elif known_types == {"preventive"}:
        classification = "preventive"
    elif known_types == {"corrective"}:
        classification = "corrective"
    else:
        classification = "undetermined"
    return {
        "intersects": True,
        "classification": classification,
        "windows": classified,
        "distinct_reset_window_count": len(intersecting),
        "known_source_types": sorted(known_types),
        "single_cause_assigned": classification in ("preventive", "corrective"),
    }


def _program_by_event(
    rows: Sequence[Mapping[str, object]], ledger: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    operations: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        operations[(str(row["robot_id"]), float(row["end_time"]))].append(row)
    result: dict[str, str] = {}
    for event in ledger:
        key = (str(event["robot_id"]), float(event["failure_time"]))
        matches = operations.get(key, [])
        if len(matches) != 1:
            raise ContractError(
                f"failure {event['failure_id']} maps to {len(matches)} operation rows"
            )
        result[str(event["failure_id"])] = str(matches[0]["program_id"])
    return result


def _event_trace(
    rows: Sequence[dict[str, object]],
    ledger: Sequence[dict[str, object]],
    maintenance: dict[str, list[list[float]]],
    maintenance_episodes: Sequence[Mapping[str, object]],
    health: object,
    canonical_eligible: Sequence[Mapping[str, object]],
    canonical_rejections: Mapping[str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from synth import balanced as B
    from synth import events as E

    program_lookup = _program_by_event(rows, ledger)
    eligible_ids = [str(event["failure_id"]) for event in canonical_eligible]
    eligible_set = set(eligible_ids)
    if len(eligible_ids) != len(eligible_set):
        raise ContractError("canonical eligible pool contains duplicate failure ids")
    traces: list[dict[str, object]] = []
    traced_eligible: list[str] = []
    reason_counts: Counter[str] = Counter()
    reset_by_cohort: Counter[str] = Counter()
    reset_by_subtype: Counter[str] = Counter()
    reset_by_class: Counter[str] = Counter()
    for index, event in enumerate(ledger):
        failure_id = str(event["failure_id"])
        cohort = str(event["cohort"])
        subtype = str(event["subtype"])
        time_of_failure = float(event["failure_time"])
        reset_detail = _reset_intersections(
            str(event["robot_id"]),
            time_of_failure - float(E.HORIZON_S),
            time_of_failure,
            maintenance,
            maintenance_episodes,
            health,
            ledger,
        )
        if bool(reset_detail["intersects"]) != E.positive_window_intersects_reset(
            event, maintenance
        ):
            raise ContractError("reset attribution changed the canonical positive-window predicate")
        detail: dict[str, object] | None = None
        positive_file_count: int | None = None
        if reset_detail["intersects"]:
            reason = "reset-horizon"
            reset_by_cohort[cohort] += 1
            reset_by_subtype[f"{cohort}/{subtype}"] += 1
            reset_by_class[str(reset_detail["classification"])] += 1
        else:
            detail = cast(
                "dict[str, object]",
                B._lead_detail(rows, event, maintenance, ledger),
            )
            positive_files = E.pos_files(rows, event, maintenance)
            positive_file_count = len(positive_files)
            if not positive_files:
                reason = "no-endpoints"
            elif cohort in ("P", "W") and not (
                int(detail["endpoints"]) >= 3 and bool(detail["early_endpoint"])
            ):
                reason = "lead-shortfall"
            else:
                reason = "eligible"
                traced_eligible.append(failure_id)
        if (reason == "eligible") != (failure_id in eligible_set):
            raise ContractError(f"event trace disagrees with eligible_anchors for {failure_id}")
        reason_counts[reason] += 1
        traces.append({
            "raw_ledger_index": index,
            "failure_id": failure_id,
            "robot_id": str(event["robot_id"]),
            "program_id": program_lookup[failure_id],
            "failure_time": time_of_failure,
            "cohort": cohort,
            "subtype": subtype,
            "stage": "eligible-anchor" if reason == "eligible" else "excluded",
            "disposition": reason,
            "reset_horizon": reset_detail,
            "positive_window_file_count": positive_file_count,
            "lead_support_detail": detail,
        })
    if traced_eligible != eligible_ids:
        raise ContractError("event trace changed canonical eligible pool order or membership")
    actual_rejections = {str(key): int(value) for key, value in canonical_rejections.items()}
    if {key: value for key, value in reason_counts.items() if key != "eligible"} != actual_rejections:
        raise ContractError("event-level rejection trace does not reconcile with eligible_anchors")
    return traces, {
        "reason_counts_including_eligible": dict(sorted(reason_counts.items())),
        "canonical_rejection_counts": dict(sorted(actual_rejections.items())),
        "reset_exclusions_by_cohort": dict(sorted(reset_by_cohort.items())),
        "reset_exclusions_by_subtype": dict(sorted(reset_by_subtype.items())),
        "reset_exclusions_by_attribution_class": {
            key: reset_by_class.get(key, 0) for key in RESET_CLASSES
        },
    }


def _anchor_row_trace(
    rows: Sequence[dict[str, object]],
    maintenance: dict[str, list[list[float]]],
    maintenance_episodes: Sequence[Mapping[str, object]],
    health: object,
    ledger: Sequence[Mapping[str, object]],
    canonical_anchors: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from synth import events as E

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    first_reason_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        checks = {
            "program_excluded": False,
            "robot_excluded": False,
            "temporal_view_present": E.TEMPORAL_VIEW in row.get("member_views", []),
            "censored": bool(row["is_censored"]),
            "quarantined": bool(row["is_quarantined"]),
            "maintenance_overlap": E._overlaps_maintenance(row, maintenance),
        }
        for name, hit in checks.items():
            if hit:
                predicate_counts[name] += 1
        reason: str | None = None
        if checks["program_excluded"]:
            reason = "program-excluded"
        elif checks["robot_excluded"]:
            reason = "robot-excluded"
        elif not checks["temporal_view_present"]:
            reason = "not-in-temporal-view"
        elif checks["censored"]:
            reason = "censored"
        elif checks["quarantined"]:
            reason = "quarantined"
        elif checks["maintenance_overlap"]:
            reason = "maintenance-overlap"
        record: dict[str, object] = {
            "manifest_row_index": index,
            "file_id": str(row["file_id"]),
            "robot_id": str(row["robot_id"]),
            "program_id": str(row["program_id"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "predicate_flags": checks,
            "first_exclusion_reason": reason,
        }
        if reason is None:
            accepted.append(row)
        else:
            first_reason_counts[reason] += 1
            record["maintenance_intersection"] = (
                _reset_intersections(
                    str(row["robot_id"]),
                    float(row["start_time"]),
                    float(row["end_time"]),
                    maintenance,
                    maintenance_episodes,
                    health,
                    ledger,
                ) if checks["maintenance_overlap"] else None
            )
            rejected.append(record)
    expected = [str(row["file_id"]) for row in canonical_anchors]
    traced = [str(row["file_id"]) for row in accepted]
    if traced != expected:
        raise ContractError("row-level anchor trace disagrees with anchor_rows")
    if len(rows) != len(accepted) + len(rejected):
        raise ContractError("anchor-row eligibility partition does not reconcile")
    return accepted, rejected, {
        "input_manifest_rows": len(rows),
        "eligible_anchor_rows": len(accepted),
        "rejected_anchor_rows": len(rejected),
        "first_exclusion_reason_counts": dict(sorted(first_reason_counts.items())),
        "overlapping_predicate_flag_counts": dict(sorted(predicate_counts.items())),
        "reconciliation": "input_manifest_rows = eligible_anchor_rows + rejected_anchor_rows",
    }


def _control_trace(
    anchors: Sequence[dict[str, object]],
    failures: Sequence[dict[str, object]],
    maintenance: dict[str, list[list[float]]],
    maintenance_episodes: Sequence[Mapping[str, object]],
    health: object,
    ledger: Sequence[Mapping[str, object]],
    canonical_windows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from synth import events as E

    ordered = sorted(anchors, key=lambda row: row["end_time"])
    last_kept: dict[str, float] = {}
    candidate_trace: list[dict[str, object]] = []
    traced_windows: list[dict[str, object]] = []
    first_reason_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    candidates_by_robot: Counter[str] = Counter()
    candidates_by_program: Counter[str] = Counter()
    selected_by_robot: Counter[str] = Counter()
    selected_anchor_programs: Counter[str] = Counter()
    for ordinal, anchor in enumerate(ordered):
        end_time = float(anchor["end_time"])
        robot_id = str(anchor["robot_id"])
        program_id = str(anchor["program_id"])
        candidates_by_robot[robot_id] += 1
        candidates_by_program[program_id] += 1
        prior_end = last_kept.get(robot_id, -math.inf)
        spacing_conflict = end_time - prior_end < float(E.HORIZON_S)
        failure_conflicts = [
            str(failure["failure_id"])
            for failure in failures
            if str(failure["robot_id"]) == robot_id
            and float(failure["failure_time"]) >= end_time - float(E.HORIZON_S)
            and float(failure["failure_time"]) < end_time + float(E.HORIZON_S)
        ]
        reset_detail = _reset_intersections(
            robot_id,
            end_time - float(E.HORIZON_S),
            end_time,
            maintenance,
            maintenance_episodes,
            health,
            ledger,
        )
        flags = {
            "prior_selected_window_spacing_conflict": spacing_conflict,
            "same_robot_failure_horizon_conflict": bool(failure_conflicts),
            "maintenance_reset_conflict": bool(reset_detail["intersects"]),
        }
        for name, hit in flags.items():
            if hit:
                predicate_counts[name] += 1
        reason: str | None = None
        if spacing_conflict:
            reason = "prior-selected-window-spacing"
        elif failure_conflicts:
            reason = "same-robot-failure-horizon"
        elif reset_detail["intersects"]:
            reason = "maintenance-reset-span"
        record: dict[str, object] = {
            "candidate_ordinal_after_stable_end_time_sort": ordinal,
            "candidate_anchor_file_id": str(anchor["file_id"]),
            "robot_id": robot_id,
            "program_id": program_id,
            "anchor_end": end_time,
            "window_start": end_time - float(E.HORIZON_S),
            "window_end": end_time,
            "selection_predicate_flags": flags,
            "same_robot_failure_conflict_ids": failure_conflicts,
            "maintenance_reset_intersection": reset_detail,
            "selected": reason is None,
            "first_rejection_reason": reason,
        }
        if reason is None:
            members = [
                row for row in ordered
                if str(row["robot_id"]) == robot_id
                and end_time - float(E.HORIZON_S) <= float(row["end_time"]) <= end_time
            ]
            member_rows = [
                {
                    "file_id": str(member["file_id"]),
                    "program_id": str(member["program_id"]),
                    "end_time": float(member["end_time"]),
                }
                for member in members
            ]
            record["members"] = member_rows
            last_kept[robot_id] = end_time
            selected_by_robot[robot_id] += 1
            selected_anchor_programs[program_id] += 1
            traced_windows.append({
                "anchor_end": end_time,
                "robot_id": robot_id,
                "members": [str(member["file_id"]) for member in members],
            })
        else:
            first_reason_counts[reason] += 1
        candidate_trace.append(record)

    def signature(window: Mapping[str, object]) -> tuple[object, ...]:
        members = cast("Sequence[object]", window["members"])
        member_ids = tuple(
            str(item["file_id"]) if isinstance(item, Mapping) else str(item)
            for item in members
        )
        return (float(window["anchor_end"]), str(window["robot_id"]), member_ids)

    canonical = [signature(window) for window in canonical_windows]
    traced = [signature(window) for window in traced_windows]
    if traced != canonical:
        raise ContractError("control candidate trace disagrees with select_control_windows")
    total = len(candidate_trace)
    selected_count = len(traced_windows)
    if total != selected_count + sum(first_reason_counts.values()):
        raise ContractError("control candidate disposition totals do not reconcile")

    sorted_by_robot: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for window in canonical_windows:
        sorted_by_robot[str(window["robot_id"])].append(window)
    positive_duration_overlaps = 0
    shared_boundary_member_ids: set[str] = set()
    selected_spans: list[dict[str, object]] = []
    for robot_id, robot_windows in sorted_by_robot.items():
        by_end = sorted(robot_windows, key=lambda item: float(item["anchor_end"]))
        for left, right in zip(by_end, by_end[1:]):
            left_end = float(left["anchor_end"])
            right_start = float(right["anchor_end"]) - float(E.HORIZON_S)
            if left_end - right_start > 0.0:
                positive_duration_overlaps += 1
            left_members = {str(member["file_id"]) for member in cast("Sequence[Mapping[str, object]]", left["members"])}
            right_members = {str(member["file_id"]) for member in cast("Sequence[Mapping[str, object]]", right["members"])}
            shared_boundary_member_ids.update(left_members.intersection(right_members))
        for window in by_end:
            selected_spans.append({
                "robot_id": robot_id,
                "window_start": float(window["anchor_end"]) - float(E.HORIZON_S),
                "window_end": float(window["anchor_end"]),
                "anchor_end": float(window["anchor_end"]),
            })
    if positive_duration_overlaps:
        raise ContractError("canonical selected control windows overlap in positive duration")

    membership_records = [
        member
        for window in canonical_windows
        for member in cast("Sequence[Mapping[str, object]]", window["members"])
    ]
    unique_members: dict[str, Mapping[str, object]] = {}
    for member in membership_records:
        file_id = str(member["file_id"])
        unique_members[file_id] = member
    membership_programs = Counter(str(member["program_id"]) for member in membership_records)
    unique_programs = Counter(str(member["program_id"]) for member in unique_members.values())
    selected_controls = [
        {
            "anchor_end": float(window["anchor_end"]),
            "robot_id": str(window["robot_id"]),
            "members": [
                {
                    "file_id": str(member["file_id"]),
                    "program_id": str(member["program_id"]),
                    "end_time": float(member["end_time"]),
                }
                for member in cast("Sequence[Mapping[str, object]]", window["members"])
            ],
        }
        for window in canonical_windows
    ]
    return {
        "candidate_opportunity_count": total,
        "candidate_opportunity_denominator": "eligible anchor rows after temporal-view, uncensored, unquarantined, and maintenance-overlap gates",
        "candidate_opportunities": candidate_trace,
        "first_rejection_reason_counts": dict(sorted(first_reason_counts.items())),
        "first_rejection_reason_total": sum(first_reason_counts.values()),
        "overlapping_predicate_flag_counts": dict(sorted(predicate_counts.items())),
        "selected_control_window_count": selected_count,
        "selected_control_windows": selected_controls,
        "selected_window_spans": selected_spans,
        "nonoverlap_audit": {
            "positive_duration_overlapping_window_pairs": positive_duration_overlaps,
            "shared_member_file_ids_at_or_across_window_boundaries": sorted(shared_boundary_member_ids),
            "boundary_touch_is_allowed_by_spacing_predicate": True,
            "interpretation": "window spans have no positive-duration overlap; closed member ranges can share a boundary row",
        },
        "distributions": {
            "candidate_opportunities_by_robot": {key: candidates_by_robot[key] for key in sorted(candidates_by_robot)},
            "candidate_opportunities_by_program": {key: candidates_by_program[key] for key in sorted(candidates_by_program)},
            "first_rejected_opportunities_by_robot": _nested_count_map(
                [item for item in candidate_trace if item["first_rejection_reason"] is not None],
                "robot_id",
                "first_rejection_reason",
            ),
            "first_rejected_opportunities_by_program": _nested_count_map(
                [item for item in candidate_trace if item["first_rejection_reason"] is not None],
                "program_id",
                "first_rejection_reason",
            ),
            "selected_windows_by_robot": {key: selected_by_robot[key] for key in sorted(selected_by_robot)},
            "selected_windows_by_anchor_program": {key: selected_anchor_programs[key] for key in sorted(selected_anchor_programs)},
            "selected_member_occurrences_by_program": {key: membership_programs[key] for key in sorted(membership_programs)},
            "selected_distinct_member_files_by_program": {key: unique_programs[key] for key in sorted(unique_programs)},
            "selected_member_occurrence_denominator": len(membership_records),
            "selected_distinct_member_file_denominator": len(unique_members),
        },
        "reconciliation": "candidate_opportunity_count = selected_control_window_count + first_rejection_reason_count_total",
    }


def _event_pool_summary(
    traces: Sequence[Mapping[str, object]],
    disposition: str,
    bucket_order: Sequence[tuple[str, str]],
) -> dict[str, object]:
    selected = [
        trace for trace in traces
        if disposition == "raw" or trace["disposition"] == disposition
    ]
    by_cohort = _count_map([str(trace["cohort"]) for trace in selected])
    by_subtype = _count_map([f"{trace['cohort']}/{trace['subtype']}" for trace in selected])
    for cohort, subtype in bucket_order:
        by_subtype.setdefault(f"{cohort}/{subtype}", 0)
    by_subtype = dict(sorted(by_subtype.items()))
    return {
        "count": len(selected),
        "event_ids_in_ledger_order": [str(trace["failure_id"]) for trace in selected],
        "by_cohort": by_cohort,
        "by_cohort_subtype": by_subtype,
        "by_robot": _count_map([str(trace["robot_id"]) for trace in selected]),
        "by_program": _count_map([str(trace["program_id"]) for trace in selected]),
        "by_robot_program": _nested_count_map(selected, "robot_id", "program_id"),
        "denominator": "all raw failure-ledger events" if disposition != "eligible" else "eligible-anchor events",
    }


def _selected_allocation_summary(
    allocation: Mapping[str, object],
    event_traces: Sequence[Mapping[str, object]],
    bucket_order: Sequence[tuple[str, str]],
    quota: Mapping[str, object],
) -> dict[str, object] | None:
    if allocation.get("status") != "FEASIBLE":
        return None
    selected = _mapping(allocation.get("selected"), "exact allocation selected buckets")
    events_by_id = {str(trace["failure_id"]): trace for trace in event_traces}
    picked: list[Mapping[str, object]] = []
    selected_by_bucket: dict[str, int] = {}
    seen: set[str] = set()
    for cohort, subtype in bucket_order:
        cohort_map = _mapping(selected.get(cohort), f"selected {cohort} buckets")
        ids_value = cohort_map.get(subtype)
        if not isinstance(ids_value, list):
            raise ContractError(f"exact allocation omits selected {cohort}/{subtype}")
        ids = [str(item) for item in ids_value]
        selected_by_bucket[f"{cohort}/{subtype}"] = len(ids)
        for failure_id in ids:
            if failure_id in seen or failure_id not in events_by_id:
                raise ContractError(f"exact allocation has duplicate or unknown event {failure_id}")
            seen.add(failure_id)
            trace = events_by_id[failure_id]
            if trace["disposition"] != "eligible" or trace["cohort"] != cohort or trace["subtype"] != subtype:
                raise ContractError(f"exact allocation selected an ineligible or misbucketed event {failure_id}")
            picked.append(trace)
        quota_field = {
            "P/P1": "p1", "P/P2": "p2", "W/W1": "w1",
            "W/W2": "w2", "A/A1": "a1", "A/A2": "a2",
        }[f"{cohort}/{subtype}"]
        if len(ids) != int(_mapping(quota, "quota")[quota_field]):
            raise ContractError(f"exact allocation selected wrong quota for {cohort}/{subtype}")
    allocator_ids = {str(item) for item in cast("Sequence[object]", allocation.get("selected_ids", []))}
    if seen != allocator_ids:
        raise ContractError("selected quota buckets disagree with exact allocator selected_ids")
    return {
        "selected_positive_event_count": len(picked),
        "selected_by_cohort_subtype": dict(sorted(selected_by_bucket.items())),
        "selected_by_robot": _count_map([str(item["robot_id"]) for item in picked]),
        "selected_by_program": _count_map([str(item["program_id"]) for item in picked]),
        "selected_by_robot_program": _nested_count_map(picked, "robot_id", "program_id"),
        "denominator": "all exact-allocator selected positive event anchors",
    }


def _evaluate_coordinate(
    seed: int,
    profile: str,
    freeze_digest: str,
    *,
    failure_probe: bool = False,
) -> tuple[dict[str, object], bool]:
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import (
        Patchifier,
        _failure_to_dict,
        _maintenance_windows,
        _manifest_file_rows,
        build_chronological,
    )

    started = time.perf_counter()
    row: dict[str, object] = {
        "history_seed": seed,
        "profile": profile,
        "diagnostic_id": DIAGNOSTIC_ID,
        "freeze_digest": freeze_digest,
        "protocol": "sprint15-benchmark-protocol-v7",
        "config_hash": None,
        "a_abrupt_rate": None,
        "no_write": True,
        "generation_writes": False,
        "generated_data_roots_written": False,
        "waveform_artifacts_written": False,
        "model_trained": False,
        "sealed_inputs_opened": False,
        "benchmark_paths_touched": False,
        "evaluation_started": True,
    }
    try:
        core = _load_core()
        configs = cast("object", core["_resolve_pair_configs"])(seed)  # type: ignore[operator]
        config = _mapping(configs, "resolved profile configs")[profile]
        resolved_config_hash = str(getattr(config, "hash")())
        if not resolved_config_hash:
            raise ContractError("resolved profile config has no hash")
        a_rate_fn = cast("Callable[[object], float]", core["_a_rate"])
        if a_rate_fn(config) != PROFILE_RATE[profile]:
            raise ContractError("profile A abrupt rate changed after static config guard")
        row["config_hash"] = resolved_config_hash
        row["a_abrupt_rate"] = PROFILE_RATE[profile]
        _, health, labeled, _ = build_chronological(config)
        maintenance = _maintenance_windows(health)
        maintenance_episodes = _maintenance_records(health)
        patchifier = Patchifier(getattr(config, "patch"))
        patch_counts = {
            sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
            for sample in labeled
        }
        rows = _manifest_file_rows(labeled, maintenance, patch_counts)
        ledger = [_failure_to_dict(event) for event in health.failure_events]
        inputs = B.project_allocation_inputs(rows)
        eligible, rejection, _lead = B.eligible_anchors(inputs, ledger, maintenance)
        event_trace, event_audit = _event_trace(
            inputs,
            ledger,
            maintenance,
            maintenance_episodes,
            health,
            eligible,
            rejection,
        )
        raw_pool = _event_pool_summary(event_trace, "raw", B.BUCKET_ORDER)
        eligible_pool = _event_pool_summary(event_trace, "eligible", B.BUCKET_ORDER)
        if raw_pool["count"] != len(ledger) or eligible_pool["count"] != len(eligible):
            raise ContractError("raw or eligible event-pool denominator does not reconcile")
        if sum(cast("dict[str, int]", raw_pool["by_cohort"]).values()) != len(ledger):
            raise ContractError("raw event cohort distribution does not reconcile")
        if sum(cast("dict[str, int]", eligible_pool["by_cohort_subtype"]).values()) != len(eligible):
            raise ContractError("eligible event subtype distribution does not reconcile")

        rejection_by_reason: Counter[str] = Counter(
            str(trace["disposition"]) for trace in event_trace
            if trace["disposition"] != "eligible"
        )
        if {key: int(value) for key, value in rejection.items()} != dict(rejection_by_reason):
            raise ContractError("event reason counts disagree with eligible_anchors")
        raw_by_cohort = cast("dict[str, int]", raw_pool["by_cohort"])
        eligible_by_cohort = cast("dict[str, int]", eligible_pool["by_cohort"])
        raw_by_subtype = cast("dict[str, int]", raw_pool["by_cohort_subtype"])
        eligible_by_subtype = cast("dict[str, int]", eligible_pool["by_cohort_subtype"])
        cohort_rejections: dict[str, dict[str, int]] = {}
        subtype_rejections: dict[str, dict[str, int]] = {}
        for cohort in COHORTS:
            cohort_rejections[cohort] = {
                reason: sum(
                    1 for trace in event_trace
                    if trace["cohort"] == cohort and trace["disposition"] == reason
                )
                for reason in ("reset-horizon", "no-endpoints", "lead-shortfall")
            }
            if raw_by_cohort.get(cohort, 0) != eligible_by_cohort.get(cohort, 0) + sum(cohort_rejections[cohort].values()):
                raise ContractError(f"raw-to-eligible event reconciliation failed for {cohort}")
        for cohort, subtype in B.BUCKET_ORDER:
            key = f"{cohort}/{subtype}"
            subtype_rejections[key] = {
                reason: sum(
                    1 for trace in event_trace
                    if trace["cohort"] == cohort and trace["subtype"] == subtype
                    and trace["disposition"] == reason
                )
                for reason in ("reset-horizon", "no-endpoints", "lead-shortfall")
            }
            if raw_by_subtype.get(key, 0) != eligible_by_subtype.get(key, 0) + sum(subtype_rejections[key].values()):
                raise ContractError(f"raw-to-eligible subtype reconciliation failed for {key}")

        quota = B.DEFAULT_QUOTA
        quota_dict = quota.as_dict()
        if quota_dict != _mapping(
            _mapping(_contract_binding().get("quota"), "frozen quota"), "frozen quota"
        ):
            raise ContractError("active quota changed from the frozen diagnostic contract")
        subtype_margins = []
        quota_fields = {
            ("P", "P1"): "p1", ("P", "P2"): "p2",
            ("W", "W1"): "w1", ("W", "W2"): "w2",
            ("A", "A1"): "a1", ("A", "A2"): "a2",
        }
        for cohort, subtype in B.BUCKET_ORDER:
            key = f"{cohort}/{subtype}"
            need = int(quota_dict[quota_fields[(cohort, subtype)]])
            count = int(eligible_by_subtype.get(key, 0))
            subtype_margins.append({
                "cohort": cohort,
                "subtype": subtype,
                "eligible_pool_count": count,
                "quota": need,
                "eligible_minus_quota_margin": count - need,
            })

        canonical_anchors = E.anchor_rows(inputs, maintenance)
        accepted_anchors, rejected_anchor_rows, anchor_audit = _anchor_row_trace(
            inputs,
            maintenance,
            maintenance_episodes,
            health,
            ledger,
            canonical_anchors,
        )
        if [str(row["file_id"]) for row in accepted_anchors] != [
            str(row["file_id"]) for row in canonical_anchors
        ]:
            raise ContractError("candidate-control anchor ordering mismatch")
        canonical_controls = E.select_control_windows(canonical_anchors, ledger, maintenance)
        controls = _control_trace(
            accepted_anchors,
            ledger,
            maintenance,
            maintenance_episodes,
            health,
            ledger,
            canonical_controls,
        )
        controls["eligible_anchor_row_exclusions"] = rejected_anchor_rows
        controls["anchor_row_stage"] = anchor_audit
        controls["quota"] = int(quota.controls)
        controls["margin"] = int(controls["selected_control_window_count"]) - int(quota.controls)
        controls["margin_denominator"] = "selected_control_windows before positive allocation"
        reset_total = int(event_audit["reason_counts_including_eligible"].get("reset-horizon", 0))
        raw_total = len(ledger)
        reset_fraction = reset_total / raw_total if raw_total else None
        eligible_total = len(eligible)
        eligible_mix = {
            cohort: (eligible_by_cohort.get(cohort, 0) / eligible_total if eligible_total else None)
            for cohort in COHORTS
        }
        rejection_by_cohort_reason = {
            cohort: dict(sorted(cohort_rejections[cohort].items())) for cohort in COHORTS
        }
        exact_record: dict[str, object]
        blocked = False
        try:
            allocation = B.allocate_quotas(
                rows, ledger, maintenance, seed, quota, method="exact"
            )
        except B.InfeasibleCandidate as exc:
            failure = _mapping(exc.record, "exact allocator infeasibility")
            exact_record = {
                "status": "INFEASIBLE",
                "feasible": False,
                "reason": failure.get("reason"),
                "missing": failure.get("missing"),
                "details": failure,
            }
        except B.AllocationUnavailable as exc:
            failure = _mapping(exc.record, "exact allocator unavailable record")
            exact_record = {
                "status": "UNAVAILABLE",
                "feasible": None,
                "reason": failure.get("reason"),
                "missing": failure.get("missing"),
                "details": failure,
            }
            blocked = True
        except RuntimeError as exc:
            exact_record = {
                "status": "WITNESS-BREACH",
                "feasible": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            blocked = True
        else:
            exact_record = {**allocation, "status": "FEASIBLE", "feasible": True}
            exact_control_projection = [
                {
                    "anchor_end": float(item["anchor_end"]),
                    "robot_id": str(item["robot_id"]),
                    "members": [str(value) for value in cast("Sequence[object]", item["members"])],
                }
                for item in cast("Sequence[Mapping[str, object]]", allocation["controls"])
            ]
            trace_control_projection = [
                {
                    "anchor_end": float(item["anchor_end"]),
                    "robot_id": str(item["robot_id"]),
                    "members": [str(member["file_id"]) for member in cast("Sequence[Mapping[str, object]]", item["members"])],
                }
                for item in cast("Sequence[Mapping[str, object]]", controls["selected_control_windows"])
            ]
            if exact_control_projection != trace_control_projection:
                raise ContractError("exact allocator final controls disagree with the independent control trace")
            allocator_margins = _mapping(allocation.get("margins"), "exact allocator margins")
            if int(allocator_margins.get("controls", -1)) != int(controls["selected_control_window_count"]):
                raise ContractError("control trace count disagrees with exact allocator control margin")

        selected_summary = _selected_allocation_summary(
            exact_record, event_trace, B.BUCKET_ORDER, quota_dict
        )
        row.update({
            "manifest_file_row_count": len(rows),
            "maintenance_window_count": sum(len(intervals) for intervals in maintenance.values()),
            "raw_event_pool": raw_pool,
            "eligible_event_pool": eligible_pool,
            "event_trace": event_trace,
            "eligibility_rejections": {
                "by_reason": dict(sorted(rejection_by_reason.items())),
                "by_cohort_and_reason": rejection_by_cohort_reason,
                "by_subtype_and_reason": dict(sorted(subtype_rejections.items())),
                "reset_exclusion_attribution_denominator": "events rejected by reset-horizon",
                "reset_exclusions_by_attribution_class": event_audit["reset_exclusions_by_attribution_class"],
                "reset_exclusions_by_cohort": event_audit["reset_exclusions_by_cohort"],
                "reset_exclusions_by_subtype": event_audit["reset_exclusions_by_subtype"],
                "reset_excluded_event_count": reset_total,
                "reset_fraction_of_raw_events": reset_fraction,
                "raw_to_eligible_event_reconciliation": "raw = eligible + reset-horizon + no-endpoints + lead-shortfall",
            },
            "eligible_set_mix": {
                "denominator_count": eligible_total,
                "denominator_name": "eligible_event_pool.count",
                "fractions_by_cohort": eligible_mix,
            },
            "eligible_subtype_quota_margins": {
                "denominator": "eligible-event subtype pool",
                "margins": subtype_margins,
            },
            "control_opportunities": controls,
            "exact_allocation": exact_record,
            "exact_allocation_selected_distribution": selected_summary,
            "allocator_margin_source": "exact_allocation.margins when feasible; otherwise the unmodified allocator failure record",
        })
        if failure_probe:
            row["disposable_failure_probe"] = _impossible_quota_probe(
                rows,
                ledger,
                maintenance,
                seed,
                int(eligible_by_subtype.get("P/P1", 0)),
                row,
            )
        row["execution_status"] = "BLOCKED" if blocked else "RECORDED"
        if blocked:
            row["block_reason"] = "exact allocator did not prove feasibility or infeasibility"
    except Exception as exc:
        row["execution_status"] = "BLOCKED"
        row["block_reason"] = "evaluation infrastructure, trace reconciliation, or audit error"
        row["infrastructure_error"] = {"type": type(exc).__name__, "message": str(exc)}
        blocked = True
    row["elapsed_s"] = round(time.perf_counter() - started, 3)
    return row, blocked


def _row_sha256(row: Mapping[str, object]) -> str:
    unsealed = dict(row)
    unsealed.pop("record_sha256", None)
    return _sha256(_canonical_json(unsealed))


def _seal_row(row: Mapping[str, object]) -> dict[str, object]:
    sealed = dict(row)
    sealed.pop("record_sha256", None)
    sealed["record_sha256"] = _row_sha256(sealed)
    return sealed


def _write_json_atomic(path: Path, value: object, *, exclusive: bool = False) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _claim_attempt(output_dir: Path, metadata: Mapping[str, object]) -> Path:
    if output_dir.exists():
        raise ContractError(f"one-shot output already exists; no retry/resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt_path = output_dir / "attempt.json"
    try:
        _write_json_atomic(attempt_path, metadata, exclusive=True)
    except Exception as exc:
        raise ContractError(f"one-shot output claimed but attempt marker could not be persisted: {exc}") from exc
    return attempt_path


def _coordinate_json(coordinate: tuple[int, str]) -> dict[str, object]:
    return {"history_seed": coordinate[0], "profile": coordinate[1]}


def _validate_rows(
    rows: Sequence[Mapping[str, object]],
    expected_coordinates: Sequence[tuple[int, str]],
) -> None:
    if len(rows) > len(expected_coordinates):
        raise ContractError("result contains more coordinates than the frozen budget")
    actual: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        coordinate = (int(row.get("history_seed", -1)), str(row.get("profile", "")))
        actual.append(coordinate)
        if coordinate != expected_coordinates[index]:
            raise ContractError(f"row coordinate/order mismatch at {index}: {coordinate}")
        if row.get("no_write") is not True or row.get("generation_writes") is not False:
            raise ContractError(f"row {index} breached no-write isolation")
        if any(row.get(flag) is not False for flag in (
            "generated_data_roots_written", "waveform_artifacts_written", "model_trained",
            "sealed_inputs_opened", "benchmark_paths_touched",
        )):
            raise ContractError(f"row {index} breached an isolation boundary")
        digest = row.get("record_sha256")
        if not isinstance(digest, str) or digest != _row_sha256(row):
            raise ContractError(f"row {index} has an invalid canonical SHA-256 seal")
    if len(set(actual)) != len(actual):
        raise ContractError("result contains duplicate profile-seed coordinates")


def _result_document(
    static: Mapping[str, object],
    freeze_digest: str,
    rows: Sequence[Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    _validate_rows(rows, EXPECTED_COORDINATES)
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "schema_version": "1.0.0",
        "status": status,
        "freeze_digest": freeze_digest,
        "contract_sha256": static["contract_sha256"],
        "runner_sha256": static["runner_sha256"],
        "schema_sha256": static["schema_sha256"],
        "base_source_closure_sha256": static["base_source_closure_sha256"],
        "source_closure_sha256": static["source_closure_sha256"],
        "source_closure": static["source_closure"],
        "namespace": static["namespace"],
        "coordinate_order": [list(coordinate) for coordinate in EXPECTED_COORDINATES],
        "profile_order": list(PROFILE_ORDER),
        "budget_unique_persisted_coordinates": len(EXPECTED_COORDINATES),
        "persisted_coordinate_count": len(rows),
        "results": list(rows),
        "interpretation_boundary": "mechanism diagnostics only; not qualification, benchmark eligibility, training authorization, or causal method effect",
    }


def _attempt_metadata(static: Mapping[str, object], freeze_digest: str, mode: str) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "attempt": 1,
        "mode": mode,
        "status": "RUNNING",
        "started_at_utc": now,
        "updated_at_utc": now,
        "freeze_digest": freeze_digest,
        "contract_sha256": static["contract_sha256"],
        "runner_sha256": static["runner_sha256"],
        "schema_sha256": static["schema_sha256"],
        "base_source_closure_sha256": static["base_source_closure_sha256"],
        "source_closure_sha256": static["source_closure_sha256"],
        "source_closure": static["source_closure"],
        "seed_order": list(STUDY_SEEDS),
        "profile_order": list(PROFILE_ORDER),
        "coordinate_order": [_coordinate_json(item) for item in EXPECTED_COORDINATES],
        "budget_unique_persisted_coordinates": len(EXPECTED_COORDINATES),
        "persisted_coordinate_count": 0,
        "evaluation_invocations_started": 0,
        "next_coordinate_index": 0,
        "in_progress_coordinate": None,
        "remaining_coordinates": [_coordinate_json(item) for item in EXPECTED_COORDINATES],
        "result_sha256": None,
        "generated_data_roots_written": False,
        "waveform_artifacts_written": False,
        "model_trained": False,
        "sealed_inputs_opened": False,
        "benchmark_paths_touched": False,
        "interrupted_invocation_count_caveat": "an interrupted coordinate may have started computation without a persisted row; automatic continuation is prohibited",
    }


def _persist_result_and_attempt(
    result_path: Path,
    attempt_path: Path,
    result: dict[str, object],
    attempt: dict[str, object],
) -> None:
    _write_json_atomic(result_path, result)
    attempt["result_sha256"] = _file_sha256(result_path)
    attempt["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(attempt_path, attempt)


def _mark_blocked_before_coordinate(
    attempt_path: Path,
    attempt: dict[str, object],
    reason: str,
    result_path: Path,
) -> None:
    attempt.update({
        "status": "BLOCKED",
        "stop_reason": reason,
        "in_progress_coordinate": None,
        "result_sha256": _file_sha256(result_path) if result_path.is_file() else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_json_atomic(attempt_path, attempt)


def _execute_study(static: Mapping[str, object], freeze_digest: str) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        raise ContractError(f"one-shot study output already exists; no retry/resume: {OUTPUT_DIR}")
    attempt = _attempt_metadata(static, freeze_digest, "study")
    attempt_path = _claim_attempt(OUTPUT_DIR, attempt)
    result_path = OUTPUT_DIR / "result.json"
    rows: list[dict[str, object]] = []
    result = _result_document(static, freeze_digest, rows, "RUNNING")
    _persist_result_and_attempt(result_path, attempt_path, result, attempt)

    for index, coordinate in enumerate(EXPECTED_COORDINATES):
        try:
            current_static = _verify_static_guards()
            current_freeze = _freeze_digest(current_static)
            _assert_expected_freeze(freeze_digest, current_freeze)
        except Exception as exc:
            _mark_blocked_before_coordinate(
                attempt_path, attempt, f"source/config/freeze mismatch before coordinate: {exc}", result_path
            )
            return {"status": "BLOCKED", "persisted_coordinates": len(rows), "stop_reason": attempt["stop_reason"]}

        attempt["in_progress_coordinate"] = _coordinate_json(coordinate)
        attempt["next_coordinate_index"] = index
        attempt["evaluation_invocations_started"] = int(attempt["evaluation_invocations_started"]) + 1
        attempt["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(attempt_path, attempt)
        row, blocked = _evaluate_coordinate(coordinate[0], coordinate[1], freeze_digest)
        row["source_closure_sha256_at_start"] = current_static["source_closure_sha256"]
        try:
            after_static = _verify_static_guards()
            _assert_expected_freeze(freeze_digest, _freeze_digest(after_static))
            row["source_closure_sha256_after_evaluation"] = after_static["source_closure_sha256"]
        except Exception as exc:
            row["execution_status"] = "BLOCKED"
            row["block_reason"] = "source/config/freeze mismatch after coordinate evaluation"
            row["post_evaluation_guard_error"] = {"type": type(exc).__name__, "message": str(exc)}
            blocked = True
        sealed = _seal_row(row)
        rows.append(sealed)
        _validate_rows(rows, EXPECTED_COORDINATES)
        next_status = "BLOCKED" if blocked else ("COMPLETE" if len(rows) == len(EXPECTED_COORDINATES) else "RUNNING")
        result = _result_document(static, freeze_digest, rows, next_status)
        attempt.update({
            "status": next_status,
            "persisted_coordinate_count": len(rows),
            "next_coordinate_index": len(rows),
            "in_progress_coordinate": None,
            "remaining_coordinates": [
                _coordinate_json(item) for item in EXPECTED_COORDINATES[len(rows):]
            ],
        })
        if blocked:
            attempt["stop_reason"] = str(row.get("block_reason", "coordinate blocked"))
        _persist_result_and_attempt(result_path, attempt_path, result, attempt)
        if blocked:
            return {
                "status": "BLOCKED",
                "persisted_coordinates": len(rows),
                "stop_reason": attempt["stop_reason"],
                "result_sha256": attempt["result_sha256"],
            }

    _validate_rows(rows, EXPECTED_COORDINATES)
    return {
        "status": "COMPLETE",
        "persisted_coordinates": len(rows),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": attempt["result_sha256"],
        "attempt_sha256": _file_sha256(attempt_path),
    }


def _expect_rejected(
    rejected: list[str], label: str, action: Callable[[], object]
) -> None:
    try:
        action()
    except ContractError:
        rejected.append(label)
        return
    raise ContractError(f"smoke accepted invalid boundary: {label}")


def _impossible_quota_probe(
    rows: Sequence[dict[str, object]],
    ledger: Sequence[dict[str, object]],
    maintenance: dict[str, list[list[float]]],
    seed: int,
    eligible_p1_count: int,
    profile_row: Mapping[str, object],
) -> dict[str, object]:
    from synth import balanced as B

    input_digest = _sha256(
        _canonical_json({"rows": rows, "ledger": ledger, "maintenance": maintenance})
    )
    quota_before = _canonical_json(B.DEFAULT_QUOTA.as_dict())
    row_before = _row_sha256(profile_row)

    probe_quota = replace(B.DEFAULT_QUOTA, p1=eligible_p1_count + 1)
    try:
        B.allocate_quotas(rows, ledger, maintenance, seed, probe_quota, method="exact")
    except B.InfeasibleCandidate as exc:
        failure = _mapping(exc.record, "smoke exact-allocation failure")
        if failure.get("reason") != "quota-shortfall":
            raise ContractError("disposable exact allocator probe failed for an unexpected reason")
        if not isinstance(failure.get("solver"), dict) and not str(
            failure.get("missing", "")
        ).startswith("exact CSP: empty eligible pool"):
            raise ContractError("disposable quota failure did not preserve a proof record")
        inputs_unchanged = input_digest == _sha256(
            _canonical_json({"rows": rows, "ledger": ledger, "maintenance": maintenance})
        )
        quota_unchanged = quota_before == _canonical_json(B.DEFAULT_QUOTA.as_dict())
        row_unchanged = row_before == _row_sha256(profile_row)
        if not inputs_unchanged or not quota_unchanged or not row_unchanged:
            raise ContractError("impossible-quota probe mutated fixture inputs, quota, or profile row")
        return {
            "status": "PROVEN-INFEASIBLE",
            "reason": failure.get("reason"),
            "allocator_inputs_unchanged": inputs_unchanged,
            "profile_default_quota_unchanged": quota_unchanged,
            "profile_row_unchanged": row_unchanged,
            "solver_or_empty_pool_proof": failure.get("solver", failure.get("missing")),
        }
    raise ContractError("disposable impossible-quota probe was not proven infeasible")


def run_smoke(static: Mapping[str, object], freeze_digest: str) -> dict[str, object]:
    if OUTPUT_DIR.exists():
        raise ContractError("smoke refuses to inspect or alter the fixed one-shot study output")
    rejected: list[str] = []
    _expect_rejected(
        rejected,
        "freeze mismatch",
        lambda: _assert_expected_freeze("0" * 64, freeze_digest),
    )
    _expect_rejected(
        rejected,
        "source closure mismatch",
        lambda: _assert_expected_freeze(
            freeze_digest,
            _freeze_digest({
                **dict(static),
                "source_closure": {
                    **cast("Mapping[str, str]", static["source_closure"]),
                    "smoke-corrupted-source": "0" * 64,
                },
            }),
        ),
    )
    _expect_rejected(
        rejected,
        "missing independent-review acknowledgement",
        lambda: _require_study_acknowledgement(False, freeze_digest, freeze_digest),
    )
    with tempfile.TemporaryDirectory(prefix="s18-control-eligibility-task61-") as temp:
        temp_root = Path(temp).resolve()
        if temp_root.is_relative_to(ROOT):
            raise ContractError("disposable persistence directory is inside the project")
        output_dir = temp_root / "fixture-output"
        attempt = _attempt_metadata(static, freeze_digest, "disposable-smoke")
        attempt["seed_order"] = [FIXTURE_SEED]
        attempt["coordinate_order"] = [
            _coordinate_json((FIXTURE_SEED, profile)) for profile in PROFILE_ORDER
        ]
        attempt["budget_unique_persisted_coordinates"] = len(PROFILE_ORDER)
        attempt["remaining_coordinates"] = [
            _coordinate_json((FIXTURE_SEED, profile)) for profile in PROFILE_ORDER
        ]
        attempt_path = _claim_attempt(output_dir, attempt)
        result_path = output_dir / "result.json"
        fixture_coordinates = tuple((FIXTURE_SEED, profile) for profile in PROFILE_ORDER)
        fixture_rows: list[dict[str, object]] = []
        result = _fixture_result_document(static, freeze_digest, fixture_rows, "RUNNING")
        _persist_result_and_attempt(result_path, attempt_path, result, attempt)

        from unittest.mock import patch
        import synth.chronicle as chronicle

        generator_calls: list[tuple[int, str]] = []
        profile_rows: dict[str, dict[str, object]] = {}
        with patch.object(
            chronicle,
            "materialize_chronological",
            side_effect=AssertionError("persistent generator/materializer called in no-write diagnostic"),
        ), patch.object(
            chronicle,
            "write_seal",
            side_effect=AssertionError("seal writer called in research-only diagnostic"),
        ):
            for index, coordinate in enumerate(fixture_coordinates):
                attempt["in_progress_coordinate"] = _coordinate_json(coordinate)
                attempt["evaluation_invocations_started"] = index + 1
                _write_json_atomic(attempt_path, attempt)
                generator_calls.append(coordinate)
                row, blocked = _evaluate_coordinate(
                    coordinate[0],
                    coordinate[1],
                    freeze_digest,
                    failure_probe=coordinate[1] == "sprint15-v7",
                )
                if blocked:
                    raise ContractError(f"real fixture diagnostic blocked: {row.get('block_reason')}")
                row["source_closure_sha256_at_start"] = static["source_closure_sha256"]
                row["source_closure_sha256_after_evaluation"] = static["source_closure_sha256"]
                if int(row.get("manifest_file_row_count", 0)) <= 0:
                    raise ContractError("real fixture produced no manifest file rows")
                if int(_mapping(row.get("raw_event_pool"), "raw pool").get("count", 0)) <= 0:
                    raise ContractError("real fixture produced no failure events")
                sealed = _seal_row(row)
                fixture_rows.append(sealed)
                profile_rows[coordinate[1]] = sealed
                _validate_rows(fixture_rows, fixture_coordinates)
                result = _fixture_result_document(
                    static, freeze_digest, fixture_rows,
                    "SMOKE-PASS" if len(fixture_rows) == len(fixture_coordinates) else "RUNNING",
                )
                attempt.update({
                    "status": result["status"],
                    "persisted_coordinate_count": len(fixture_rows),
                    "next_coordinate_index": len(fixture_rows),
                    "in_progress_coordinate": None,
                    "remaining_coordinates": [
                        _coordinate_json(item) for item in fixture_coordinates[len(fixture_rows):]
                    ],
                })
                _persist_result_and_attempt(result_path, attempt_path, result, attempt)

        saved_result_sha = _file_sha256(result_path)
        saved_attempt_sha = _file_sha256(attempt_path)
        reloaded = _read_object(result_path)
        stored_rows = cast("list[dict[str, object]]", reloaded["results"])
        _validate_rows(stored_rows, fixture_coordinates)
        if reloaded.get("status") != "SMOKE-PASS" or len(stored_rows) != 2:
            raise ContractError("temporary real-fixture result did not persist/finalize")
        stored_attempt = _read_object(attempt_path)
        if (
            stored_attempt.get("result_sha256") != saved_result_sha
            or int(stored_attempt.get("persisted_coordinate_count", -1)) != len(stored_rows)
            or stored_attempt.get("status") != "SMOKE-PASS"
        ):
            raise ContractError("attempt marker does not bind the finalized temporary result")

        bad_order = list(reversed(stored_rows))
        _expect_rejected(
            rejected,
            "reordered fixture rows",
            lambda: _validate_rows(bad_order, fixture_coordinates),
        )
        duplicate_rows = [dict(stored_rows[0]), dict(stored_rows[0])]
        _expect_rejected(
            rejected,
            "duplicate fixture coordinate",
            lambda: _validate_rows(duplicate_rows, fixture_coordinates),
        )
        corrupt_row = dict(stored_rows[0])
        corrupt_row["no_write"] = False
        _expect_rejected(
            rejected,
            "no-write boundary",
            lambda: _validate_rows([corrupt_row], fixture_coordinates),
        )
        corrupt_digest_row = dict(stored_rows[0])
        corrupt_digest_row["record_sha256"] = "0" * 64
        _expect_rejected(
            rejected,
            "invalid canonical row digest",
            lambda: _validate_rows([corrupt_digest_row], fixture_coordinates),
        )
        before_duplicate_claim = (saved_result_sha, saved_attempt_sha)
        _expect_rejected(
            rejected,
            "duplicate one-shot output claim",
            lambda: _claim_attempt(output_dir, attempt),
        )
        after_duplicate_claim = (_file_sha256(result_path), _file_sha256(attempt_path))
        if before_duplicate_claim != after_duplicate_claim:
            raise ContractError("duplicate output claim changed persisted fixture evidence")

        impossible_probe = profile_rows["sprint15-v7"].get("disposable_failure_probe")
        if not isinstance(impossible_probe, dict) or impossible_probe.get("status") != "PROVEN-INFEASIBLE":
            raise ContractError("real fixture did not exercise the exact-allocation rejection path")
        if any(impossible_probe.get(key) is not True for key in (
            "allocator_inputs_unchanged",
            "profile_default_quota_unchanged",
            "profile_row_unchanged",
        )):
            raise ContractError("impossible-quota probe did not preserve fixture inputs and row")

    if OUTPUT_DIR.exists():
        raise ContractError("disposable smoke created the fixed study output")
    if _freeze_digest(_verify_static_guards()) != freeze_digest:
        raise ContractError("source/config closure changed during the disposable smoke")
    return {
        "status": "SMOKE-PASS",
        "fixture_seed": FIXTURE_SEED,
        "fixture_profile_order": list(PROFILE_ORDER),
        "fixture_generator_coordinates": [list(item) for item in generator_calls],
        "real_rows_persisted_in_temporary_directory": 2,
        "temporary_result_sha256": saved_result_sha,
        "temporary_attempt_sha256": saved_attempt_sha,
        "real_fixture_raw_events_by_profile": {
            profile: _mapping(profile_rows[profile]["raw_event_pool"], "raw pool")["count"]
            for profile in PROFILE_ORDER
        },
        "real_fixture_eligible_events_by_profile": {
            profile: _mapping(profile_rows[profile]["eligible_event_pool"], "eligible pool")["count"]
            for profile in PROFILE_ORDER
        },
        "real_fixture_control_windows_by_profile": {
            profile: _mapping(profile_rows[profile]["control_opportunities"], "control trace")["selected_control_window_count"]
            for profile in PROFILE_ORDER
        },
        "exact_allocator_failure_probe": impossible_probe,
        "rejected_guard_boundaries": rejected,
        "source_closure_sha256": static["source_closure_sha256"],
        "freeze_digest": freeze_digest,
        "generated_data_roots_written": False,
        "benchmark_paths_touched": False,
        "sealed_inputs_opened": False,
        "fixed_study_output_created": False,
    }


def _fixture_result_document(
    static: Mapping[str, object],
    freeze_digest: str,
    rows: Sequence[Mapping[str, object]],
    status: str,
) -> dict[str, object]:
    fixture_order = tuple((FIXTURE_SEED, profile) for profile in PROFILE_ORDER)
    _validate_rows(rows, fixture_order)
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "schema_version": "1.0.0",
        "status": status,
        "freeze_digest": freeze_digest,
        "contract_sha256": static["contract_sha256"],
        "runner_sha256": static["runner_sha256"],
        "schema_sha256": static["schema_sha256"],
        "base_source_closure_sha256": static["base_source_closure_sha256"],
        "source_closure_sha256": static["source_closure_sha256"],
        "source_closure": static["source_closure"],
        "namespace": static["namespace"],
        "coordinate_order": [list(coordinate) for coordinate in fixture_order],
        "profile_order": list(PROFILE_ORDER),
        "budget_unique_persisted_coordinates": len(PROFILE_ORDER),
        "persisted_coordinate_count": len(rows),
        "results": list(rows),
        "disposable_fixture_only": True,
        "interpretation_boundary": "disposable proof fixture; not a study coordinate, qualification, benchmark eligibility, training authorization, or causal method effect",
    }


def _require_study_acknowledgement(
    review_passed: bool, expected_freeze: str, actual_freeze: str
) -> None:
    if not review_passed:
        raise ContractError("--study requires --review-passed after independent Task 61 evidence review")
    _assert_expected_freeze(expected_freeze, actual_freeze)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run the disposable real-fixture proof only")
    mode.add_argument("--study", action="store_true", help="run the fixed one-shot Task 62 coordinates")
    parser.add_argument("--review-passed", action="store_true", help="acknowledge the independent evidence-review gate")
    parser.add_argument("--expected-freeze", help="exact freeze SHA-256 from the reviewed Task 61 evidence")
    args = parser.parse_args(argv)
    if args.smoke and (args.review_passed or args.expected_freeze is not None):
        raise ContractError("--smoke does not accept study review or freeze arguments")
    if args.study and (not args.review_passed or args.expected_freeze is None):
        raise ContractError("--study requires both --review-passed and --expected-freeze")
    static = _verify_static_guards()
    freeze_digest = _freeze_digest(static)
    if args.smoke:
        outcome = run_smoke(static, freeze_digest)
    else:
        _require_study_acknowledgement(args.review_passed, args.expected_freeze, freeze_digest)
        outcome = _execute_study(static, freeze_digest)
    print(json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if outcome.get("status") in ("SMOKE-PASS", "COMPLETE") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
