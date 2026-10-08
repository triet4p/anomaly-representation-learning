"""Sprint 18 Task 5 measurability qualification for an existing frozen roster.

The runner consumes only the exact 16 role-bound Sprint 18 roots. It never
creates data, fits or scores a representation, or accesses any other history.
The one-time ``--correct-existing-result`` path audits EG1 against the
inherited per-robot and manifest-reload contracts, preserves the superseded
record, and does not rerun the full observable-probe qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BINDING_PATH = Path("experiments/sprint18-role-binding-v1.json")
PROTOCOL_V1_PATH = Path("experiments/sprint18-full-training-protocol-v1.md")
PROTOCOL_V2_PATH = Path("experiments/sprint18-full-training-protocol-v2.md")
MATERIALIZATION_PATH = Path("artifacts/sprint-18/task4-materialization.json")
PREFLIGHT_PATH = Path("artifacts/sprint-18/task4-preflight.json")
OUTPUT_PATH = Path("artifacts/sprint-18/task5-measurability.json")
CORRECTION_ARCHIVE_PATH = Path(
    "artifacts/sprint-18/task5-measurability-global-chronology-superseded.json"
)
EXPECTED_PRIOR_OUTPUT_SHA256 = "8072898493e1ce0b964ca4ad14db224bdc15498882199534521e32d3ac8e591d"
EXPECTED_PRIOR_RUNNER_SHA256 = "84cf4610034d47bf0cdc5e48084f7179401876886f60c823ed4c915fed70a49e"
EXPECTED_PROTOCOL_V2_SHA256 = "e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2"
EXPECTED_BINDING_SHA256 = "d2a398e223cbdfc14fd64cd582513aa71dc9c4e44737f6e66abf73954f270bdf"
EXPECTED_PREFLIGHT_SHA256 = "7a15d9d41852f1a9646e62ccb8be8230f92049dd8397614882d2c095001fae0d"
EXPECTED_GATE_SHA256 = "d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c"
EXPECTED_PROBE_CODE_SHA256 = "a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b"
EXPECTED_ROLE_COUNTS = {
    "DESIGN": 4,
    "FIT": 3,
    "CALIBRATION": 1,
    "DEVELOPMENT": 4,
    "CONFIRMATION": 4,
}
EXPECTED_ROLE_IDS = (
    [f"H-S18-DES-{i:02d}" for i in range(1, 5)]
    + [f"H-S18-FIT-{i:02d}" for i in range(1, 4)]
    + ["H-S18-CAL-01"]
    + [f"H-S18-DEV-{i:02d}" for i in range(1, 5)]
    + [f"H-S18-CONF-{i:02d}" for i in range(1, 5)]
)
EXPECTED_SEEDS = list(range(31800, 31816))
EXPECTED_ROLE_PERMISSIONS = {
    "DESIGN": "Structural/measurability qualification; healthy-only coefficient-pilot retention evaluation. Never final fitting.",
    "FIT": "Verified-healthy representation fitting, EMA targets, Fit reference banks/standardizers, Fit-only auxiliary fitting and cross-fitting.",
    "CALIBRATION": "Verified-healthy per-arm/per-seed q95 thresholds only.",
    "DEVELOPMENT": "Complete descriptive comparison and diagnostics after all configs are frozen. No coefficient, arm, mask, checkpoint, branch, or threshold selection.",
    "CONFIRMATION": "One-shot locked final comparison only.",
}
EXPECTED_REQUIRED_MANIFEST_FIELDS = {
    "format", "generator_version", "protocol", "role", "config_hash",
    "resolved_config", "seeds", "counts", "schedule", "episodes",
    "failure_events", "maintenance_windows", "calendar", "splits",
    "files", "shards",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def record_check(checks: dict[str, bool], failures: list[str], name: str,
                 passed: bool) -> None:
    checks[name] = bool(passed)
    if not passed:
        failures.append(name)


def load_bound_inputs() -> tuple[dict, dict, dict, list[dict], dict]:
    from synth.chronicle import sprint15_v7_history_config

    binding, binding_raw = load_json(BINDING_PATH)
    materialization, _ = load_json(MATERIALIZATION_PATH)
    preflight, preflight_raw = load_json(PREFLIGHT_PATH)
    unsigned = {key: value for key, value in binding.items()
                if key != "binding_sha256"}
    if sha256_bytes(canonical_json(unsigned)) != binding["binding_sha256"]:
        raise ValueError("canonical Task 3 binding digest mismatch")
    if binding["binding_sha256"] != EXPECTED_BINDING_SHA256:
        raise ValueError("Task 3 binding is not the accepted digest")
    if sha256_file(PROTOCOL_V2_PATH) != EXPECTED_PROTOCOL_V2_SHA256:
        raise ValueError("accepted Sprint 18 v2 protocol digest mismatch")
    if binding["protocol_sha256"] != EXPECTED_PROTOCOL_V2_SHA256:
        raise ValueError("binding does not name the accepted Sprint 18 protocol")
    if sha256_file(PROTOCOL_V1_PATH) != binding["base_protocol"]["sha256"]:
        raise ValueError("inherited Sprint 18 v1 protocol digest mismatch")
    if sha256_file(Path("docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md")) != EXPECTED_GATE_SHA256:
        raise ValueError("inherited normative measurability gates digest mismatch")
    if sha256_file(Path("src/synth/probe15.py")) != EXPECTED_PROBE_CODE_SHA256:
        raise ValueError("frozen observable probe implementation digest mismatch")

    specs = {
        "N1-HCC": "experiments/sprint18-hierarchical-conditional-contrastive-spec-v2.md",
        "N2-MSP": "experiments/sprint18-multiscale-latent-prediction-spec-v2.md",
        "N3-CCRP": "experiments/sprint18-cross-channel-relational-prediction-spec-v2.md",
        "N4-CRN": "experiments/sprint18-context-residual-normalization-spec-v2.md",
    }
    for arm, path in specs.items():
        if sha256_file(Path(path)) != binding["mechanism_spec_sha256"][arm]:
            raise ValueError(f"accepted mechanism specification digest mismatch: {arm}")
    for source, expected in binding["generator_digest_expectations"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError(f"bound Candidate 7 source digest mismatch: {source}")
    for source, expected in materialization["source_hashes"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError(f"Task 4 materialization source digest mismatch: {source}")
    if len(materialization["source_hashes"]) != materialization["source_hash_check_count"]:
        raise ValueError("Task 4 source-hash coverage mismatch")
    preflight_digest = sha256_bytes(preflight_raw)
    if preflight_digest != EXPECTED_PREFLIGHT_SHA256:
        raise ValueError("Task 4 raw preflight digest mismatch")
    if (preflight["verdict"] != "PREFLIGHT-PASS"
            or preflight["feasible_count"] != "16/16"
            or len(preflight["results"]) != 16
            or any(not row.get("no_write") for row in preflight["results"])):
        raise ValueError("persisted one-shot rejection-only preflight is not 16/16 PASS")
    if (materialization["status"] != "MATERIALIZATION_COMPLETE"
            or materialization["attempt_number"] != 1
            or materialization["binding_sha256"] != binding["binding_sha256"]
            or materialization["accepted_protocol_sha256"] != EXPECTED_PROTOCOL_V2_SHA256
            or materialization["preflight_sha256"] != preflight_digest
            or materialization["preflight_verdict"] != "PREFLIGHT-PASS"
            or materialization["preflight_feasible_count"] != "16/16"
            or materialization["current_history_id"] is not None
            or materialization["current_root"] is not None
            or materialization["current_index"] != 16):
        raise ValueError("Task 4 materialization record is incomplete or not bound")

    entries = binding["role_binding"]
    if ([entry["history_id"] for entry in entries] != EXPECTED_ROLE_IDS
            or [entry["data_seed"] for entry in entries] != EXPECTED_SEEDS
            or Counter(entry["role"] for entry in entries) != EXPECTED_ROLE_COUNTS
            or len({entry["directory"] for entry in entries}) != 16
            or len({entry["history_id"] for entry in entries}) != 16
            or len({entry["data_seed"] for entry in entries}) != 16):
        raise ValueError("bound Sprint 18 roster is not the exact accepted 16-root roster")
    for entry in entries:
        if entry["permitted_use"] != EXPECTED_ROLE_PERMISSIONS[entry["role"]]:
            raise ValueError(f"role permission mismatch: {entry['history_id']}")
        if entry["directory"] != (
                f"data/generated/sprint18-full-training-v1/{entry['role']}/"
                f"{entry['history_id']}"):
            raise ValueError(f"noncanonical bound root path: {entry['history_id']}")
        cfg = sprint15_v7_history_config(entry["data_seed"])
        if cfg.hash() != entry["expected_config_hash"]:
            raise ValueError(f"bound Candidate 7 configuration mismatch: {entry['history_id']}")
    if set(EXPECTED_SEEDS).intersection(binding["model_seeds"]):
        raise ValueError("data and model seed namespaces overlap")

    roots = materialization["expected_roots"]
    expected_roots = [entry["directory"] for entry in entries]
    if roots != expected_roots:
        raise ValueError("Task 4 expected-root list is not the bound roster")
    records = materialization["root_records"]
    if len(records) != 16 or materialization["completed_roots"] != EXPECTED_ROLE_IDS:
        raise ValueError("Task 4 root record coverage is not the exact bound roster")
    for entry, record in zip(entries, records):
        if (record["history_id"] != entry["history_id"]
                or record["role"] != entry["role"]
                or record["data_seed"] != entry["data_seed"]
                or record["directory"] != entry["directory"]
                or record["permitted_use"] != entry["permitted_use"]):
            raise ValueError(f"Task 4 root record binding mismatch: {entry['history_id']}")

    base = Path("data/generated/sprint18-full-training-v1")
    expected_role_dirs = set(EXPECTED_ROLE_COUNTS)
    if {child.name for child in base.iterdir()} != expected_role_dirs:
        raise ValueError("Sprint 18 generated root has unbound role directories")
    for role in EXPECTED_ROLE_COUNTS:
        expected_ids = {entry["history_id"] for entry in entries if entry["role"] == role}
        if {child.name for child in (base / role).iterdir()} != expected_ids:
            raise ValueError(f"Sprint 18 {role} directory contains missing or unbound roots")

    return binding, materialization, preflight, entries, {
        "binding_sha256": binding["binding_sha256"],
        "binding_file_sha256": sha256_bytes(binding_raw),
        "protocol_v1_sha256": sha256_file(PROTOCOL_V1_PATH),
        "protocol_v2_sha256": sha256_file(PROTOCOL_V2_PATH),
        "gate_document_sha256": EXPECTED_GATE_SHA256,
        "observable_probe_spec_sha256": sha256_file(Path("experiments/sprint15-observable-probe-v7.md")),
        "observable_probe_code_sha256": EXPECTED_PROBE_CODE_SHA256,
        "mechanism_spec_sha256": binding["mechanism_spec_sha256"],
        "candidate7_source_digests_verified": len(binding["generator_digest_expectations"]),
        "materialization_source_digests_verified": len(materialization["source_hashes"]),
        "preflight_sha256": preflight_digest,
        "preflight_verdict": preflight["verdict"],
        "preflight_feasible_count": preflight["feasible_count"],
        "materialization_status": materialization["status"],
        "materialization_attempt_number": materialization["attempt_number"],
        "materialized_roots": len(records),
        "materialized_manifest_rows": sum(r["manifest_file_row_count"] for r in records),
        "materialized_shards": sum(r["shard_count"] for r in records),
    }


def validate_manifest(entry: dict, record: dict, binding: dict,
                      materialization: dict) -> tuple[dict, dict]:
    from synth.chronicle import sprint15_v7_history_config

    root = Path(entry["directory"])
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    expected_fields = set(binding["manifest_expectations"]["required_top_level_fields"])
    if expected_fields != EXPECTED_REQUIRED_MANIFEST_FIELDS:
        raise ValueError("bound manifest required-field contract changed")
    if (manifest_path.as_posix() != record["manifest_path"]
            or sha256_bytes(raw) != record["manifest_sha256"]
            or len(raw) != record["manifest_bytes"]):
        raise ValueError(f"manifest bytes differ from Task 4: {entry['history_id']}")
    if (set(manifest) != expected_fields
            or manifest.get("format") != binding["manifest_expectations"]["format"]
            or manifest.get("generator_version") != binding["manifest_expectations"]["generator_version"]
            or manifest.get("protocol") != binding["manifest_expectations"]["protocol"]
            or manifest.get("role") != entry["history_id"]
            or manifest.get("config_hash") != entry["expected_config_hash"]
            or "sprint15" in manifest):
        raise ValueError(f"manifest provenance or top-level contract mismatch: {entry['history_id']}")
    cfg = sprint15_v7_history_config(entry["data_seed"])
    if cfg.hash() != manifest["config_hash"]:
        raise ValueError(f"manifest config hash differs from exact constructor: {entry['history_id']}")
    expected_seeds = {name: entry["data_seed"] for name in
                      ("factory", "scheduler", "health", "signal", "temporal")}
    if manifest["seeds"] != expected_seeds:
        raise ValueError(f"manifest seed fields mismatch: {entry['history_id']}")
    files = manifest["files"]
    if (record["manifest_file_row_count"] != len(files)
            or record["manifest_top_level_fields"] != sorted(expected_fields)
            or record["sprint15_provenance_block"] != "absent (sprint15=None)"
            or record["shard_count"] != len(manifest["shards"])
            or len(files) != manifest["counts"]["total"]):
        raise ValueError(f"manifest summary differs from Task 4 record: {entry['history_id']}")
    allowed = {"manifest.json", "files"}
    if {child.name for child in root.iterdir()} != allowed:
        raise ValueError(f"root layout contains an unbound path: {entry['history_id']}")
    expected_shard_names = {Path(shard["path"]).name for shard in manifest["shards"]}
    actual_shard_names = {child.name for child in (root / "files").iterdir()}
    if actual_shard_names != expected_shard_names:
        raise ValueError(f"shard roster differs from manifest: {entry['history_id']}")
    expected_shards = record["shards"]
    if len(expected_shards) != len(manifest["shards"]):
        raise ValueError(f"Task 4 shard record count mismatch: {entry['history_id']}")
    expected_ids = [row["file_id"] for row in files]
    for index, (shard, frozen_shard) in enumerate(zip(manifest["shards"], expected_shards)):
        start = index * binding["manifest_expectations"]["shard_size"]
        end = min(start + binding["manifest_expectations"]["shard_size"], len(files))
        if (shard["path"] != frozen_shard["path"]
                or shard["start"] != frozen_shard["start"]
                or shard["end"] != frozen_shard["end"]
                or shard["count"] != frozen_shard["count"]
                or shard["sha256"] != frozen_shard["sha256"]
                or shard["start"] != start or shard["end"] != end
                or shard["count"] != end - start
                or shard["file_ids"] != expected_ids[start:end]):
            raise ValueError(f"shard manifest or Task 4 binding mismatch: {entry['history_id']}:{index}")
    for name in ("factory", "scheduler", "health", "signal", "temporal"):
        if manifest["seeds"][name] != entry["data_seed"]:
            raise ValueError(f"seed field mismatch: {entry['history_id']}:{name}")
    return manifest, {
        "manifest_sha256": sha256_bytes(raw),
        "manifest_bytes": len(raw),
        "file_rows": len(files),
        "shards": len(manifest["shards"]),
        "config_hash": manifest["config_hash"],
        "protocol": manifest["protocol"],
        "sprint15_provenance_absent": True,
    }


def history_signature(samples: list, manifest: dict, manifest_path: Path) -> dict:
    from synth import events as E

    waveforms = hashlib.sha256()
    sample_ids = []
    for sample in samples:
        sample_ids.append(sample.file_id)
        waveforms.update(sample.file_id.encode("utf-8"))
        waveforms.update(b"\0")
        values = sample.x
        waveforms.update(str(values.dtype).encode("ascii"))
        waveforms.update(canonical_json(list(values.shape)))
        waveforms.update(values.tobytes(order="C"))
    ledger_digest = sha256_bytes(canonical_json(E.failure_ledger(manifest)))
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "event_ledger_sha256": ledger_digest,
        "waveform_stream_sha256": waveforms.hexdigest(),
        "sample_count": len(samples),
        "sample_order_matches_manifest": sample_ids == [row["file_id"] for row in manifest["files"]],
        "shard_count": len(manifest["shards"]),
    }


def structural_summary(entry: dict, manifest: dict) -> dict:
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import _last_reset_time

    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]
    checks: dict[str, bool] = {}
    failures: list[str] = []
    windows_bounded = all(
        isinstance(window, list) and len(window) == 2
        and math.isfinite(float(window[0])) and math.isfinite(float(window[1]))
        and float(window[1]) >= float(window[0])
        for intervals in wins.values() for window in intervals
    )
    windows_sorted = all(intervals == sorted(intervals)
                         for intervals in wins.values())
    overlapping_window_robots = []
    for robot_id, intervals in wins.items():
        ordered_windows = sorted(intervals)
        if any(current[0] < previous[1]
               for previous, current in zip(ordered_windows, ordered_windows[1:])):
            overlapping_window_robots.append(robot_id)
    record_check(checks, failures, "bounded_maintenance_windows", windows_bounded)
    record_check(checks, failures, "maintenance_windows_sorted", windows_sorted)

    expected_robot_ids = [f"robot-{i:02d}" for i in range(1, 10)]
    allowed = B.ALLOWED_ROW_KEYS
    row_ids = [row["file_id"] for row in rows]
    record_check(checks, failures, "exact_model_visible_row_keys",
                 all(set(row) == allowed for row in rows))
    record_check(checks, failures, "no_cohort_or_subtype_in_model_rows",
                 all("cohort" not in row and "subtype" not in row for row in rows))
    record_check(checks, failures, "unique_file_ids", len(row_ids) == len(set(row_ids)))
    record_check(checks, failures, "all_expected_cohorts",
                 {failure["cohort"] for failure in ledger} == {"P", "W", "A"})
    record_check(checks, failures, "nine_robot_topology",
                 sorted({row["robot_id"] for row in rows}) == expected_robot_ids)
    record_check(checks, failures, "reserved_robot_present",
                 B.ROBOT_RESERVE in {row["robot_id"] for row in rows})
    record_check(checks, failures, "reserved_program_present",
                 B.PROGRAM_RESERVE in {row["program_id"] for row in rows})
    by_robot: dict[str, list[dict]] = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    serial = True
    for robot, robot_rows in by_robot.items():
        ordered = sorted(robot_rows, key=lambda row: (row["start_time"], row["end_time"]))
        serial &= all(cur["start_time"] >= prev["end_time"] - 1e-6
                      for prev, cur in zip(ordered, ordered[1:]))
    record_check(checks, failures, "per_robot_operation_serialization", serial)
    timing = all(
        math.isfinite(float(row["start_time"]))
        and math.isfinite(float(row["end_time"]))
        and math.isfinite(float(row["last_reset_time"]))
        and row["end_time"] >= row["start_time"]
        and row["last_reset_time"] <= row["start_time"]
        and int(row["n_valid_patches"]) > 0
        for row in rows
    )
    record_check(checks, failures, "finite_file_timing_and_patch_support", timing)
    reset_consistency = all(
        abs(float(row["last_reset_time"]) - _last_reset_time(
            wins, row["robot_id"], float(row["start_time"]))) <= 1e-6
        for row in rows
    )
    record_check(checks, failures, "last_reset_matches_frozen_windows", reset_consistency)
    counts = Counter(row["file_label"] for row in rows)
    splits = manifest["splits"]
    count_record = manifest["counts"]
    split_counts = {
        "dev_train": len(splits["dev_train"]),
        "dev_val": len(splits["dev_val"]),
        "test_static": len(splits["test_static"]),
        "test_temporal": len(splits["test_temporal"]),
        "quarantined": len(splits["quarantined"]),
    }
    count_consistency = (
        count_record["total"] == len(rows)
        and count_record["normal"] == counts["normal"]
        and count_record["abnormal"] == counts["abnormal"]
        and count_record["quarantined"] == sum(row["is_quarantined"] for row in rows)
        and all(count_record[name] == value for name, value in split_counts.items())
        and len(manifest["schedule"]) == len(rows)
    )
    record_check(checks, failures, "manifest_counts_and_splits_consistent", count_consistency)

    subtype_counts = Counter(failure["subtype"] for failure in ledger)
    cohort_counts = Counter(failure["cohort"] for failure in ledger)
    # User-accepted policy S18-T69-B01 (DR01 provenance; nominal P upper 15d,
    # accepted band [15d, 16d) i.e. 2 <= duration_d < 16): the nominal bound
    # is reported, not hidden. Boundary duration_d == 16.0 and < 2.0 FAIL.
    nominal_p_exceedances = sorted(
        failure["duration_d"]
        for failure in ledger
        if failure["cohort"] == "P" and failure["duration_d"] > 15.0
    )
    physical = all(
        (failure["duration_d"] == 0.0 and failure["degradation_onset"] is None
         and failure["subtype"] in ("A1", "A2"))
        if failure["cohort"] == "A"
        else ((failure["subtype"] in ("P1", "P2")
               and 2.0 <= failure["duration_d"] < 16.0)
              if failure["cohort"] == "P"
              else (failure["subtype"] in ("W1", "W2")
                    and 6.0 <= failure["duration_d"] <= 28.0))
        for failure in ledger
    )
    p_durations = sorted(failure["duration_d"] for failure in ledger if failure["cohort"] == "P")
    w_durations = sorted(failure["duration_d"] for failure in ledger if failure["cohort"] == "W")
    physical &= bool(p_durations and w_durations)
    if p_durations:
        physical &= 5.0 <= statistics.median(p_durations) <= 10.0
    if w_durations:
        physical &= 12.0 <= statistics.median(w_durations) <= 24.0
    record_check(checks, failures, "cohort_subtype_physical_shapes_and_duration_bounds", physical)
    nuisance = B.check_nuisance_envelopes(manifest["resolved_config"])
    margin = B.analytic_signal_margin()
    record_check(checks, failures, "candidate7_nuisance_envelopes", bool(nuisance["pass"]))
    record_check(checks, failures, "candidate7_analytic_signal_margin", bool(margin["pass"]))

    eligible, rejection, lead_details = B.eligible_anchors(rows, ledger, wins)
    pos_eval_by_robot: Counter = Counter()
    pos_by_cohort: Counter = Counter()
    sub_eval: Counter = Counter()
    per_program: dict[str, Counter] = {"P": Counter(), "W": Counter()}
    program_at_end = {(row["robot_id"], row["end_time"]): row["program_id"] for row in rows}
    lead_eligible: dict[str, dict] = {}
    positive_event_ids: list[str] = []
    failure_by_id = {failure["failure_id"]: failure for failure in ledger}
    reserved_eligible = Counter()
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        candidates = E.pos_files(rows, failure, wins)
        if not candidates:
            continue
        positive_event_ids.append(failure["failure_id"])
        pos_by_cohort[failure["cohort"]] += 1
        pos_eval_by_robot[failure["robot_id"]] += 1
        sub_eval[failure["subtype"]] += 1
        if failure["cohort"] in ("P", "W"):
            program = program_at_end.get((failure["robot_id"], failure["failure_time"]))
            per_program[failure["cohort"]][program if program is not None else "unknown"] += 1
            lead_eligible[failure["failure_id"]] = lead_details.get(failure["failure_id"], {})
            if failure["robot_id"] == B.ROBOT_RESERVE:
                reserved_eligible["robot"] += 1
            if program == B.PROGRAM_RESERVE:
                reserved_eligible["program"] += 1
            if failure["subtype"] == B.P_SUBTYPE_RESERVE:
                reserved_eligible["P_subtype"] += 1
            if failure["subtype"] == B.W_SUBTYPE_RESERVE:
                reserved_eligible["W_subtype"] += 1
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    controls_by_robot = Counter(window["robot_id"] for window in controls)
    robot_days = {
        (row["robot_id"], int(row["end_time"] // B.DAY))
        for row in rows if E.eligible_operational_row(row, wins)
    }
    programs_by_cohort = {cohort: sorted(program for program, count in per_program[cohort].items() if count)
                          for cohort in ("P", "W")}
    program_shares = {
        cohort: (max(per_program[cohort].values()) / pos_by_cohort[cohort]
                 if pos_by_cohort[cohort] else 1.0)
        for cohort in ("P", "W")
    }
    cohort_mix = {
        cohort: (pos_by_cohort[cohort] / sum(pos_by_cohort.values())
                 if sum(pos_by_cohort.values()) else 0.0)
        for cohort in ("P", "W", "A")
    }
    max_positive_share = (max(pos_eval_by_robot.values()) / sum(pos_by_cohort.values())
                          if sum(pos_by_cohort.values()) else 1.0)
    max_negative_share = (max(controls_by_robot.values()) / len(controls)
                          if controls else 1.0)
    lead_support: dict[str, dict[str, float | int]] = {}
    for cohort in ("P", "W"):
        cohort_events = [detail for failure_id, detail in lead_eligible.items()
                         if failure_by_id[failure_id]["cohort"] == cohort]
        good = sum(
            1 for detail in cohort_events
            if detail.get("endpoints", 0) >= 3
            and detail.get("early_endpoint")
            and detail.get("clean_baseline")
        )
        lead_support[cohort] = {
            "eligible_events": len(cohort_events),
            "events_meeting_three_endpoints_early_endpoint_and_clean_baseline": good,
            "fraction": good / len(cohort_events) if cohort_events else 0.0,
            "pass_80_percent": bool(cohort_events) and good / len(cohort_events) >= 0.80,
        }
    support = {
        "positives_by_cohort": {cohort: pos_by_cohort.get(cohort, 0) for cohort in ("P", "W", "A")},
        "positives_total": sum(pos_by_cohort.values()),
        "evaluable_positive_events": len(eligible),
        "unevaluable_positive_events": len(ledger) - len(eligible),
        "evaluable_by_subtype": {subtype: sub_eval.get(subtype, 0)
                                  for subtype in ("P1", "P2", "W1", "W2", "A1", "A2")},
        "negative_control_windows": len(controls),
        "negative_controls_by_robot": dict(sorted(controls_by_robot.items())),
        "evaluable_robot_days": len(robot_days),
        "positive_contributing_robots": sum(value > 0 for value in pos_eval_by_robot.values()),
        "negative_contributing_robots": sum(value > 0 for value in controls_by_robot.values()),
        "positive_events_by_robot": dict(sorted(pos_eval_by_robot.items())),
        "programs_by_cohort": programs_by_cohort,
        "program_shares": program_shares,
        "cohort_mix": cohort_mix,
        "max_positive_share_by_robot": max_positive_share,
        "max_negative_share_by_robot": max_negative_share,
        "lead_support": lead_support,
        "eligible_anchor_count": len(eligible),
        "reserved_identity_eligible_events": {
            "reserved_robot_P_or_W": reserved_eligible["robot"],
            "reserved_program_P_or_W": reserved_eligible["program"],
            "reserved_P_subtype": reserved_eligible["P_subtype"],
            "reserved_W_subtype": reserved_eligible["W_subtype"],
        },
        "eligibility_rejections": dict(sorted(rejection.items())),
    }
    support["support_sha256"] = sha256_bytes(canonical_json({
        "history_id": entry["history_id"],
        "positive_event_ids": sorted(positive_event_ids),
        "control_windows": [
            {"anchor_end": w["anchor_end"], "robot_id": w["robot_id"],
             "member_file_ids": [row["file_id"] for row in w["members"]]}
            for w in controls
        ],
        "robot_days": sorted([robot, day] for robot, day in robot_days),
    }))
    core_pass = all(checks.values())
    return {
        "history_id": entry["history_id"],
        "role": entry["role"],
        "data_seed": entry["data_seed"],
        "checks": checks,
        "core_integrity_pass": core_pass,
        "core_integrity_failures": failures,
        "nominal_p_upper_15d_exceedance_count": len(nominal_p_exceedances),
        "nominal_p_upper_15d_exceedance_durations_d": list(nominal_p_exceedances),
        "nominal_p_upper_15d_exceedance_allowance_d": 1.0,
        "nuisance": nuisance,
        "analytic_signal_margin": margin,
        "maintenance_window_count": sum(len(intervals) for intervals in wins.values()),
        "maintenance_overlap_robots_diagnostic_only": overlapping_window_robots,
        "support": support,
    }


def design_promotion_checks(summary: dict) -> dict[str, bool]:
    support = summary["support"]
    p, w, a = (support["positives_by_cohort"][c] for c in ("P", "W", "A"))
    mix = support["cohort_mix"]
    lead = support["lead_support"]
    programs = support["programs_by_cohort"]
    return {
        "P_ge_13": p >= 13,
        "W_ge_13": w >= 13,
        "A_ge_10": a >= 10,
        "total_ge_38": p + w + a >= 38,
        "negative_controls_ge_32": support["negative_control_windows"] >= 32,
        "robot_days_ge_188": support["evaluable_robot_days"] >= 188,
        "positive_robots_ge_6": support["positive_contributing_robots"] >= 6,
        "negative_robots_ge_6": support["negative_contributing_robots"] >= 6,
        "at_least_2_programs_each_P_W": len(programs["P"]) >= 2 and len(programs["W"]) >= 2,
        "positive_robot_share_le_35_percent": support["max_positive_share_by_robot"] <= 0.35,
        "negative_robot_share_le_40_percent": support["max_negative_share_by_robot"] <= 0.40,
        "program_share_le_60_percent_each_P_W": all(v <= 0.60 for v in support["program_shares"].values()),
        "each_cohort_mix_15_to_60_percent": all(0.15 <= mix[c] <= 0.60 for c in ("P", "W", "A")),
        "P_lead_support_ge_80_percent": lead["P"]["pass_80_percent"],
        "W_lead_support_ge_80_percent": lead["W"]["pass_80_percent"],
    }


def confirmation_hard_floor_checks(summary: dict) -> dict[str, bool]:
    support = summary["support"]
    p, w, a = (support["positives_by_cohort"][c] for c in ("P", "W", "A"))
    lead = support["lead_support"]
    programs = support["programs_by_cohort"]
    mix = support["cohort_mix"]
    return {
        "P_ge_10": p >= 10,
        "W_ge_10": w >= 10,
        "A_ge_8": a >= 8,
        "total_ge_30": p + w + a >= 30,
        "negative_controls_ge_25": support["negative_control_windows"] >= 25,
        "robot_days_ge_150": support["evaluable_robot_days"] >= 150,
        "positive_robots_ge_6": support["positive_contributing_robots"] >= 6,
        "negative_robots_ge_6": support["negative_contributing_robots"] >= 6,
        "at_least_2_programs_each_P_W": len(programs["P"]) >= 2 and len(programs["W"]) >= 2,
        "positive_robot_share_le_35_percent": support["max_positive_share_by_robot"] <= 0.35,
        "negative_robot_share_le_40_percent": support["max_negative_share_by_robot"] <= 0.40,
        "program_share_le_60_percent_each_P_W": all(v <= 0.60 for v in support["program_shares"].values()),
        "each_cohort_mix_15_to_60_percent": all(0.15 <= mix[c] <= 0.60 for c in ("P", "W", "A")),
        "P_lead_support_ge_80_percent": lead["P"]["pass_80_percent"],
        "W_lead_support_ge_80_percent": lead["W"]["pass_80_percent"],
        "reserved_robot_has_eligible_P_or_W_event": support["reserved_identity_eligible_events"]["reserved_robot_P_or_W"] > 0,
        "reserved_program_has_eligible_P_or_W_event": support["reserved_identity_eligible_events"]["reserved_program_P_or_W"] > 0,
        "reserved_P_subtype_has_eligible_event": support["reserved_identity_eligible_events"]["reserved_P_subtype"] > 0,
        "reserved_W_subtype_has_eligible_event": support["reserved_identity_eligible_events"]["reserved_W_subtype"] > 0,
    }


def healthy_features(samples: list, manifest: dict) -> tuple[list[list[float]], list[str], int, dict[str, int]]:
    import numpy as np
    from synth import balanced as B
    from synth import events as E
    from synth import probe15 as P

    feature_count = len(P.feature_names())
    by_id = {sample.file_id: sample for sample in samples}
    features: list[list[float]] = []
    ids: list[str] = []
    patches = 0
    hygiene = {
        "verified_healthy_source_files": 0,
        "censored_healthy_rows_excluded": 0,
        "maintenance_overlapping_healthy_rows_excluded": 0,
        "probe_input_healthy_files": 0,
        "probe_input_valid_patches": 0,
    }
    for row in manifest["files"]:
        if row["file_label"] != "normal" or row["is_quarantined"]:
            continue
        if row["program_id"] == B.PROGRAM_RESERVE or row["robot_id"] == B.ROBOT_RESERVE:
            continue
        hygiene["verified_healthy_source_files"] += 1
        is_censored = bool(row["is_censored"])
        overlaps_maintenance = E._overlaps_maintenance(row, manifest["maintenance_windows"])
        if is_censored:
            hygiene["censored_healthy_rows_excluded"] += 1
            continue
        if overlaps_maintenance:
            hygiene["maintenance_overlapping_healthy_rows_excluded"] += 1
            continue
        patch_count = int(row["n_valid_patches"])
        sample = by_id[row["file_id"]]
        feature = P.extract_features(
            np.asarray(sample.x, dtype=np.float64),
            row["end_time"] - row["start_time"],
            row["end_time"] - row["last_reset_time"],
        )
        if len(feature) != feature_count or not np.isfinite(feature).all():
            raise ValueError(f"invalid healthy fixed-probe feature: {manifest['role']}/{row['file_id']}")
        features.append(feature.tolist())
        ids.append(row["file_id"])
        patches += patch_count
        hygiene["probe_input_healthy_files"] += 1
        hygiene["probe_input_valid_patches"] += patch_count
    return features, ids, patches, hygiene


def evaluate_role_block(entries: list[dict], manifests: dict[str, dict],
                        structural: dict[str, dict], signatures: dict[str, dict],
                        stats: dict, centroid, threshold: float) -> tuple[dict, dict[str, dict]]:
    import numpy as np
    from synth import events as E
    from synth import probe15 as P
    from synth.chronicle import load_chronological
    feature_count = len(P.feature_names())

    file_scores: dict[str, float] = {}
    rows_by_history: list[list[dict]] = []
    ledgers_by_history: list[list[dict]] = []
    wins_by_history: list[dict] = []
    support_by_history: list[dict] = []
    for entry in entries:
        history_id = entry["history_id"]
        root = Path(entry["directory"])
        samples, loaded_manifest = load_chronological(root)
        if canonical_json(loaded_manifest) != canonical_json(manifests[history_id]):
            raise ValueError(f"loader manifest differs from provenance read: {history_id}")
        signatures[history_id] = history_signature(samples, loaded_manifest, root / "manifest.json")
        by_id = {sample.file_id: sample for sample in samples}
        if len(by_id) != len(samples):
            raise ValueError(f"duplicate sample ID in {history_id}")
        rows = loaded_manifest["files"]
        rows_by_history.append(rows)
        ledger = E.failure_ledger(loaded_manifest)
        ledgers_by_history.append(ledger)
        wins = loaded_manifest["maintenance_windows"]
        wins_by_history.append(wins)
        support_summary = structural[history_id]["support"]
        support_by_history.append(support_summary)
        for row in rows:
            sample = by_id[row["file_id"]]
            feature = P.extract_features(
                np.asarray(sample.x, dtype=np.float64),
                row["end_time"] - row["start_time"],
                row["end_time"] - row["last_reset_time"],
            )
            if len(feature) != feature_count or not np.isfinite(feature).all():
                raise ValueError(f"invalid fixed-probe feature vector: {history_id}/{row['file_id']}")
            score = float(P.score_files(P.apply_standardization(feature[None, :], stats), centroid)[0])
            if not math.isfinite(score):
                raise ValueError(f"non-finite fixed-probe score: {history_id}/{row['file_id']}")
            if row["file_id"] in file_scores:
                raise ValueError(f"file ID collision across evaluation histories: {row['file_id']}")
            file_scores[row["file_id"]] = score
        del by_id, samples

    result = P.evaluate_histories(file_scores, rows_by_history, ledgers_by_history,
                                  wins_by_history, threshold)
    for entry, history_metrics in zip(entries, result["per_history"]):
        history_metrics["history_id"] = entry["history_id"]
    return result, {entry["history_id"]: support for entry, support in
                    zip(entries, support_by_history)}


def observable_gate_checks(result: dict, support: dict[str, dict]) -> dict:
    per_history = result["per_history"]
    support_ids = set(support)
    metric_ids = [history.get("history_id") for history in per_history]
    metrics_by_history = {
        history["history_id"]: history
        for history in per_history if history.get("history_id") is not None
    }
    checks: dict[str, bool] = {}
    required_metrics = [
        result.get("macro_auc_pw"), result.get("macro_auc_p"),
        result.get("macro_auc_w"), result.get("lcb_pw"),
    ]
    checks["all_required_block_metrics_finite"] = all(
        value is not None and math.isfinite(float(value)) for value in required_metrics
    )
    checks["exactly_four_histories"] = len(per_history) == 4
    checks["all_history_ids_present"] = len(support) == 4 and set(metric_ids) == support_ids
    checks["all_histories_have_P_W_and_controls"] = all(
        support[history_id]["positives_by_cohort"]["P"] > 0
        and support[history_id]["positives_by_cohort"]["W"] > 0
        and support[history_id]["negative_control_windows"] > 0
        for history_id in support
    )
    checks["macro_PW_ge_0_65"] = checks["all_required_block_metrics_finite"] and result["macro_auc_pw"] >= 0.65
    checks["PW_history_block_LCB95_gt_0_50"] = checks["all_required_block_metrics_finite"] and result["lcb_pw"] > 0.50
    checks["at_least_3_of_4_PW_AUROC_gt_0_55"] = sum(
        h["auc_pw"] is not None and h["auc_pw"] > 0.55 for h in per_history
    ) >= 3
    checks["macro_P_ge_0_70"] = checks["all_required_block_metrics_finite"] and result["macro_auc_p"] >= 0.70
    checks["macro_W_ge_0_60"] = checks["all_required_block_metrics_finite"] and result["macro_auc_w"] >= 0.60
    per_history_gates = []
    for history_id, support_summary in support.items():
        values = metrics_by_history.get(history_id, {})
        finite = all(values.get(name) is not None and math.isfinite(float(values[name]))
                     for name in ("auc_pw", "auc_p", "auc_w", "recall_p", "recall_w",
                                  "lead_p_median", "lead_w_median", "false_episodes", "far",
                                  "robot_days"))
        per_history_gates.append({
            "history_id": history_id,
            "required_metrics_finite": finite,
            "P_recall_ge_0_50": finite and values["recall_p"] >= 0.50,
            "P_median_lead_ge_1_day": finite and values["lead_p_median"] >= 1.0,
            "W_recall_ge_0_25": finite and values["recall_w"] >= 0.25,
            "W_median_lead_ge_0_5_day": finite and values["lead_w_median"] >= 0.5,
            "FAR_le_0_05": finite and values["far"] <= 0.05,
            "A_companion_reported": "auc_a" in values,
            "A_AUROC": values.get("auc_a"),
            "support": support_summary,
            "metrics": values,
        })
    checks["per_history_gates_pass"] = all(
        h["required_metrics_finite"] and h["P_recall_ge_0_50"]
        and h["P_median_lead_ge_1_day"] and h["W_recall_ge_0_25"]
        and h["W_median_lead_ge_0_5_day"] and h["FAR_le_0_05"]
        and h["A_companion_reported"]
        for h in per_history_gates
    )
    support_hash = sha256_bytes(canonical_json({
        history: value["support_sha256"] for history, value in sorted(support.items())
    }))
    checks["all_gates_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass": checks["all_gates_pass"],
        "support_sha256": support_hash,
        "per_history": per_history_gates,
        "metrics": result,
    }


def reload_only() -> int:
    from synth import events as E
    from synth.chronicle import load_chronological

    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    signatures = {}
    for entry in binding["role_binding"]:
        root = Path(entry["directory"])
        samples, manifest = load_chronological(root)
        signatures[entry["history_id"]] = history_signature(samples, manifest, root / "manifest.json")
    print(json.dumps(signatures, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def main() -> int:
    import numpy as np
    from synth import balanced as B
    from synth import events as E
    from synth import probe15 as P
    from synth.chronicle import load_chronological

    if OUTPUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite prior Task 5 diagnostics: {OUTPUT_PATH}")
    binding, materialization, preflight, entries, lineage = load_bound_inputs()
    manifests: dict[str, dict] = {}
    manifest_provenance: dict[str, dict] = {}
    root_records = {row["history_id"]: row for row in materialization["root_records"]}
    structural: dict[str, dict] = {}
    eg3: dict[str, dict] = {}
    failures: list[str] = []

    for entry in entries:
        manifest, provenance = validate_manifest(entry, root_records[entry["history_id"]],
                                                 binding, materialization)
        manifests[entry["history_id"]] = manifest
        manifest_provenance[entry["history_id"]] = provenance
        summary = structural_summary(entry, manifest)
        structural[entry["history_id"]] = summary
        if entry["role"] in ("DESIGN", "CONFIRMATION"):
            eg3[entry["history_id"]] = B.eg3_fixtures(
                manifest["files"], E.failure_ledger(manifest), manifest["maintenance_windows"]
            )

    design = {history: structural[history] for history in EXPECTED_ROLE_IDS[:4]}
    fit_entries = [entry for entry in entries if entry["role"] == "FIT"]
    calibration_entries = [entry for entry in entries if entry["role"] == "CALIBRATION"]
    development_entries = [entry for entry in entries if entry["role"] == "DEVELOPMENT"]
    confirmation_entries = [entry for entry in entries if entry["role"] == "CONFIRMATION"]


    for history, fixtures in eg3.items():
        if not fixtures["pass"]:
            failures.extend(f"EG3/{history}/{name}" for name, passed in fixtures["checks"].items()
                            if not passed)

    design_checks = {history: design_promotion_checks(summary) for history, summary in design.items()}
    for history, checks in design_checks.items():
        failures.extend(f"EG2/{history}/{name}" for name, passed in checks.items() if not passed)
    reload_signatures: dict[str, dict] = {}
    for entry in entries:
        if entry["role"] != "DESIGN":
            continue
        history = entry["history_id"]
        root = Path(entry["directory"])
        samples, manifest = load_chronological(root)
        if canonical_json(manifest) != canonical_json(manifests[history]):
            raise ValueError(f"loaded Design manifest changed: {history}")
        reload_signatures[history] = history_signature(samples, manifest, root / "manifest.json")
        del samples
    fit_features: list[list[float]] = []
    fit_ids: list[str] = []
    fit_patches = 0
    fit_hygiene: Counter = Counter()
    fit_per_root = []
    for entry in fit_entries:
        history = entry["history_id"]
        samples, manifest = load_chronological(Path(entry["directory"]))
        if canonical_json(manifest) != canonical_json(manifests[history]):
            raise ValueError(f"loaded Fit manifest changed: {history}")
        reload_signatures[history] = history_signature(samples, manifest, Path(entry["directory"]) / "manifest.json")
        feats, ids, patches, hygiene = healthy_features(samples, manifest)
        fit_features.extend(feats)
        fit_ids.extend(f"{history}/{file_id}" for file_id in ids)
        fit_patches += patches
        fit_hygiene.update(hygiene)
        fit_per_root.append({"history_id": history, "healthy_files": len(ids),
                             "valid_patches": patches, "hygiene": hygiene})
        del samples
    fit_count = len(fit_features)
    fit_support = {
        "histories": fit_per_root,
        "verified_healthy_source_file_count": fit_hygiene["verified_healthy_source_files"],
        "probe_input_healthy_file_count": fit_count,
        "probe_input_valid_patch_count": fit_patches,
        "healthy_file_count": fit_count,
        "valid_patch_count": fit_patches,
        "censored_healthy_rows_excluded": fit_hygiene["censored_healthy_rows_excluded"],
        "maintenance_overlapping_healthy_rows_excluded":
            fit_hygiene["maintenance_overlapping_healthy_rows_excluded"],
        "aggregate_healthy_files_ge_200": fit_count >= 200,
        "aggregate_valid_patches_ge_6000": fit_patches >= 6000,
        "fit_file_id_sha256": sha256_bytes(canonical_json(sorted(fit_ids))),
    }
    fit_support["pass"] = all(fit_support[name] for name in (
        "aggregate_healthy_files_ge_200", "aggregate_valid_patches_ge_6000"))
    if not fit_support["pass"]:
        failures.append("Fit healthy support floor failed")


    fit_matrix = np.asarray(fit_features, dtype=np.float64)
    if fit_matrix.ndim != 2 or fit_matrix.shape[1] != len(P.feature_names()) or not np.isfinite(fit_matrix).all():
        raise ValueError("Fit-only observable feature matrix is invalid")
    fit_stats = P.standardize_fit(fit_matrix)
    centroid = P.fit_centroid(P.apply_standardization(fit_matrix, fit_stats))
    del fit_matrix, fit_features

    cal_entry = calibration_entries[0]
    cal_samples, cal_manifest = load_chronological(Path(cal_entry["directory"]))
    if canonical_json(cal_manifest) != canonical_json(manifests[cal_entry["history_id"]]):
        raise ValueError("loaded Calibration manifest changed")
    reload_signatures[cal_entry["history_id"]] = history_signature(
        cal_samples, cal_manifest, Path(cal_entry["directory"]) / "manifest.json")
    cal_features, cal_ids, cal_patches, cal_hygiene = healthy_features(cal_samples, cal_manifest)
    cal_matrix = np.asarray(cal_features, dtype=np.float64)
    cal_scores = P.score_files(P.apply_standardization(cal_matrix, fit_stats), centroid)
    threshold = P.select_threshold(cal_scores)
    calibration_support = {
        "history_id": cal_entry["history_id"],
        "verified_healthy_source_file_count": cal_hygiene["verified_healthy_source_files"],
        "probe_input_healthy_file_count": len(cal_ids),
        "probe_input_valid_patch_count": cal_patches,
        "healthy_file_count": len(cal_ids),
        "valid_patch_count": cal_patches,
        "censored_healthy_rows_excluded": cal_hygiene["censored_healthy_rows_excluded"],
        "maintenance_overlapping_healthy_rows_excluded":
            cal_hygiene["maintenance_overlapping_healthy_rows_excluded"],
        "healthy_file_id_sha256": sha256_bytes(canonical_json(sorted(cal_ids))),
        "support_floor_ge_40": len(cal_ids) >= 40,
        "threshold_quantile": P.THRESHOLD_QUANTILE,
        "threshold": float(threshold),
    }
    calibration_support["pass"] = calibration_support["support_floor_ge_40"]
    if not calibration_support["pass"]:
        failures.append("Calibration healthy support floor failed")
    if not np.isfinite(cal_scores).all() or not math.isfinite(float(threshold)):
        failures.append("Calibration threshold or scores are non-finite")
    del cal_samples, cal_features, cal_matrix, cal_scores

    development_result, development_support = evaluate_role_block(
        development_entries, manifests, structural, reload_signatures,
        fit_stats, centroid, float(threshold))
    confirmation_result, confirmation_support = evaluate_role_block(
        confirmation_entries, manifests, structural, reload_signatures,
        fit_stats, centroid, float(threshold))

    development_gate = observable_gate_checks(development_result, development_support)
    confirmation_gate = observable_gate_checks(confirmation_result, confirmation_support)
    for block_name, gate in (("Development", development_gate), ("Confirmation", confirmation_gate)):
        if not gate["pass"]:
            failures.extend(f"EG4/{block_name}/{name}" for name, passed in gate["checks"].items()
                            if not passed)

    confirmation_checks = {
        history: confirmation_hard_floor_checks(structural[history])
        for history in EXPECTED_ROLE_IDS[-4:]
    }
    for history, checks in confirmation_checks.items():
        failures.extend(f"EG5/{history}/{name}" for name, passed in checks.items() if not passed)
    stability_measures = {
        "P": ("positives_by_cohort", "P"),
        "W": ("positives_by_cohort", "W"),
        "A": ("positives_by_cohort", "A"),
        "negative_controls": ("negative_control_windows",),
        "robot_days": ("evaluable_robot_days",),
    }
    design_values = {
        metric: [
            summary["support"][path[0]][path[1]] if len(path) == 2
            else summary["support"][path[0]]
            for summary in design.values()
        ]
        for metric, path in stability_measures.items()
    }
    confirmation_values = {
        metric: [
            structural[history]["support"][path[0]][path[1]] if len(path) == 2
            else structural[history]["support"][path[0]]
            for history in EXPECTED_ROLE_IDS[-4:]
        ]
        for metric, path in stability_measures.items()
    }
    design_medians = {metric: statistics.median(values)
                      for metric, values in design_values.items()}
    confirmation_medians = {metric: statistics.median(values)
                            for metric, values in confirmation_values.items()}
    stability = {}
    for metric in stability_measures:
        baseline = design_medians[metric]
        ratio = confirmation_medians[metric] / baseline if baseline else None
        stability[metric] = {
            "design_values": design_values[metric],
            "design_median": baseline,
            "confirmation_values": confirmation_values[metric],
            "confirmation_median": confirmation_medians[metric],
            "confirmation_to_design_median_ratio": ratio,
            "within_0_75_to_1_25": ratio is not None and 0.75 <= ratio <= 1.25,
        }
    stability_pass = all(value["within_0_75_to_1_25"] for value in stability.values())
    if not stability_pass:
        failures.extend(f"EG5/stability/{metric}" for metric, value in stability.items()
                        if not value["within_0_75_to_1_25"])

    # This process is the first fresh reload. A second independent Python process
    # reloads exactly the same 16 bound roots; no path outside that roster is passed.
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--reload-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if child.returncode != 0:
        reload_identity = {"pass": False, "error": child.stderr[-4000:]}
        failures.append("EG1/fresh-process reload check failed")
    else:
        second_signatures = json.loads(child.stdout.strip())
        comparisons = {}
        for history in EXPECTED_ROLE_IDS:
            first = reload_signatures.get(history)
            second = second_signatures.get(history)
            comparisons[history] = {
                "manifest_sha256_equal": bool(first and second and first["manifest_sha256"] == second["manifest_sha256"]),
                "event_ledger_sha256_equal": bool(first and second and first["event_ledger_sha256"] == second["event_ledger_sha256"]),
                "waveform_stream_sha256_equal": bool(first and second and first["waveform_stream_sha256"] == second["waveform_stream_sha256"]),
                "sample_order_matches_manifest_both_passes": bool(first and second and first["sample_order_matches_manifest"] and second["sample_order_matches_manifest"]),
                "first_process": first,
                "second_process": second,
            }
        reload_pass = all(
            item["manifest_sha256_equal"] and item["event_ledger_sha256_equal"]
            and item["waveform_stream_sha256_equal"]
            and item["sample_order_matches_manifest_both_passes"]
            for item in comparisons.values()
        )
        reload_identity = {"pass": reload_pass, "fresh_processes": 2,
                           "histories_verified": len(comparisons), "per_history": comparisons}
        if not reload_pass:
            failures.append("EG1/fresh-process byte identity mismatch")
    reload_order_by_history = {
        history: comparisons[history]["sample_order_matches_manifest_both_passes"]
        for history in comparisons
    } if child.returncode == 0 else {
        history: False for history in EXPECTED_ROLE_IDS
    }
    for history, summary in structural.items():
        reload_order = reload_order_by_history[history]
        record_check(
            summary["checks"], summary["core_integrity_failures"],
            "manifest_shard_reload_order_matches_manifest", reload_order,
        )
        summary["core_integrity_pass"] = all(summary["checks"].values())
        if not summary["core_integrity_pass"]:
            failures.extend(
                f"EG1/{history}/{name}" for name in summary["core_integrity_failures"]
            )

    support_by_history = {
        history: summary["support"] for history, summary in structural.items()
    }
    support_block_hashes = {
        "development": development_gate["support_sha256"],
        "confirmation": confirmation_gate["support_sha256"],
    }
    global_checks = {
        "EG0_protocol_binding_and_materialization_provenance": True,
        "EG1_all_roots_core_integrity": all(summary["core_integrity_pass"] for summary in structural.values()),
        "EG1_two_fresh_process_identical_reloads": reload_identity["pass"],
        "EG2_each_design_promotion_target_and_concentration": all(all(checks.values()) for checks in design_checks.values()),
        "EG2_fit_support_floors": fit_support["pass"],
        "EG2_calibration_support_floor": calibration_support["pass"],
        "EG3_design_confirmation_metric_computability": all(
            eg3[history]["pass"] for history in EXPECTED_ROLE_IDS[:4] + EXPECTED_ROLE_IDS[-4:]
        ),
        "EG4_development_fixed_probe": development_gate["pass"],
        "EG4_confirmation_fixed_probe": confirmation_gate["pass"],
        "EG5_confirmation_hard_floors_and_concentration": all(all(checks.values()) for checks in confirmation_checks.values()),
        "EG5_confirmation_median_stability": stability_pass,
    }
    role_entries = {
        role: [entry for entry in entries if entry["role"] == role]
        for role in EXPECTED_ROLE_COUNTS
    }

    def role_support_aggregate(role: str) -> dict:
        summaries = [structural[entry["history_id"]]["support"]
                     for entry in role_entries[role]]
        return {
            "histories": len(summaries),
            "positive_events_by_cohort_sum_diagnostic_only": {
                cohort: sum(item["positives_by_cohort"][cohort] for item in summaries)
                for cohort in ("P", "W", "A")
            },
            "negative_control_windows_sum_diagnostic_only":
                sum(item["negative_control_windows"] for item in summaries),
            "evaluable_robot_days_sum_diagnostic_only":
                sum(item["evaluable_robot_days"] for item in summaries),
        }

    def role_core_pass(role: str) -> bool:
        return all(structural[entry["history_id"]]["core_integrity_pass"]
                   for entry in role_entries[role])

    design_pass = all(all(checks.values()) for checks in design_checks.values())
    confirmation_hard_floor_pass = all(
        all(checks.values()) for checks in confirmation_checks.values())
    role_outcomes = {
        "DESIGN": {
            "history_ids": [entry["history_id"] for entry in role_entries["DESIGN"]],
            "core_integrity_pass": role_core_pass("DESIGN"),
            "promotion_checks_by_history": design_checks,
            "promotion_pass": design_pass,
            "support_aggregate_diagnostic_only": role_support_aggregate("DESIGN"),
            "pass": role_core_pass("DESIGN") and design_pass,
        },
        "FIT": {
            "history_ids": [entry["history_id"] for entry in role_entries["FIT"]],
            "core_integrity_pass": role_core_pass("FIT"),
            "healthy_probe_support": fit_support,
            "support_aggregate_diagnostic_only": role_support_aggregate("FIT"),
            "pass": role_core_pass("FIT") and fit_support["pass"],
        },
        "CALIBRATION": {
            "history_ids": [entry["history_id"] for entry in role_entries["CALIBRATION"]],
            "core_integrity_pass": role_core_pass("CALIBRATION"),
            "healthy_probe_support_and_threshold": calibration_support,
            "support_aggregate_diagnostic_only": role_support_aggregate("CALIBRATION"),
            "pass": role_core_pass("CALIBRATION") and calibration_support["pass"],
        },
        "DEVELOPMENT": {
            "history_ids": [entry["history_id"] for entry in role_entries["DEVELOPMENT"]],
            "core_integrity_pass": role_core_pass("DEVELOPMENT"),
            "fixed_probe": development_gate,
            "support_aggregate_diagnostic_only": role_support_aggregate("DEVELOPMENT"),
            "pass": role_core_pass("DEVELOPMENT") and development_gate["pass"],
        },
        "CONFIRMATION": {
            "history_ids": [entry["history_id"] for entry in role_entries["CONFIRMATION"]],
            "core_integrity_pass": role_core_pass("CONFIRMATION"),
            "hard_floor_checks_by_history": confirmation_checks,
            "hard_floor_pass": confirmation_hard_floor_pass,
            "median_stability_pass": stability_pass,
            "fixed_probe": confirmation_gate,
            "support_aggregate_diagnostic_only": role_support_aggregate("CONFIRMATION"),
            "pass": (
                role_core_pass("CONFIRMATION") and confirmation_hard_floor_pass
                and stability_pass and confirmation_gate["pass"]
            ),
        },
    }
    eligible = all(global_checks.values())
    if eligible:
        scientific_verdict = "ELIGIBLE_FOR_REPRESENTATION_EXECUTION"
        stop_implication = (
            "No Task 5 measurability stop. This record does not itself authorize representation "
            "execution; the required evidence review and remaining sprint gates still apply."
        )
    elif (not global_checks["EG0_protocol_binding_and_materialization_provenance"]
          or not global_checks["EG1_all_roots_core_integrity"]
          or not global_checks["EG1_two_fresh_process_identical_reloads"]
          or not global_checks["EG2_each_design_promotion_target_and_concentration"]
          or not global_checks["EG2_fit_support_floors"]
          or not global_checks["EG2_calibration_support_floor"]):
        scientific_verdict = "REJECTED-CANDIDATE"
        stop_implication = (
            "STOP representation execution. No waiver, pooling, retry, replacement, omission, or "
            "favorable-root rescue; any next candidate requires prospective reviewed methodology "
            "and a new authorized roster."
        )
    elif not global_checks["EG3_design_confirmation_metric_computability"]:
        scientific_verdict = "UNAVAILABLE-CANDIDATE"
        stop_implication = (
            "STOP representation execution. Preserve this candidate; resolve metric computability "
            "through a prospective reviewed protocol change before any new candidate."
        )
    else:
        scientific_verdict = "REJECTED-CANDIDATE"
        stop_implication = (
            "STOP representation execution. No waiver, pooling, retry, replacement, omission, or "
            "favorable-history rescue; any next candidate requires prospective reviewed methodology "
            "and a new authorized roster."
        )
    outcome = ("ELIGIBLE_FOR_REPRESENTATION_EXECUTION" if eligible
               else "NOT_ELIGIBLE_FOR_REPRESENTATION_EXECUTION")
    diagnostics = {
        "measurement_complete": True,
        "qualification_outcome": outcome,
        "scientific_verdict": scientific_verdict,
        "eligible_for_representation_execution": eligible,
        "representation_execution_authorized_by_this_record": False,
        "stop_implication": stop_implication,
        "lineage": lineage,
        "diagnostic_runner_sha256": sha256_file(Path(__file__)),
        "materialization_record_sha256": sha256_file(MATERIALIZATION_PATH),
        "role_roster": [
            {"role": entry["role"], "history_id": entry["history_id"],
             "data_seed": entry["data_seed"], "directory": entry["directory"],
             "manifest": manifest_provenance[entry["history_id"]]}
            for entry in entries
        ],
        "reload_identity": reload_identity,
        "structural_by_history": structural,
        "eg3_by_history": eg3,
        "role_outcomes": role_outcomes,
        "design_promotion_checks": design_checks,
        "fit_healthy_support": fit_support,
        "calibration_healthy_support_and_threshold": calibration_support,
        "observable_probe": {
            "specification": "sprint15-observable-probe-v7",
            "code_sha256": EXPECTED_PROBE_CODE_SHA256,
            "implementation_probe_id": P.PROBE_ID,
            "features": len(P.feature_names()),
            "fit_only_standardization_and_centroid": True,
            "calibration_only_q95_threshold": True,
            "fit_feature_mean_sha256": sha256_bytes(np.asarray(fit_stats["mean"], dtype=np.float64).tobytes()),
            "fit_feature_std_sha256": sha256_bytes(np.asarray(fit_stats["std"], dtype=np.float64).tobytes()),
            "healthy_centroid_sha256": sha256_bytes(np.asarray(centroid, dtype=np.float64).tobytes()),
            "development": development_gate,
            "confirmation": confirmation_gate,
        },
        "confirmation_structural_hard_floor_checks": confirmation_checks,
        "confirmation_stability": {
            "measurements": stability,
            "pass": stability_pass,
        },
        "measurability_support_sha256": support_block_hashes,
        "global_gate_checks": global_checks,
        "failed_gates": sorted(set(failures)),
        "execution_boundaries": {
            "generated_or_regenerated_roots": False,
            "preflight_rerun": False,
            "representation_training_or_scoring": False,
            "fixed_observable_probe_only": True,
            "sprint15_sealed_paths_or_histories_accessed": False,
            "development_and_confirmation_used_only_for_fixed_probe_qualification": True,
            "design_excluded_from_fit_and_calibration": True,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(diagnostics, indent=2, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "qualification_outcome": outcome,
        "global_gate_checks": global_checks,
        "failed_gates": diagnostics["failed_gates"],
        "fit_support": fit_support,
        "calibration_support": calibration_support,
        "development_probe": {
            "pass": development_gate["pass"],
            "metrics": development_gate["metrics"],
            "failed_checks": [name for name, passed in development_gate["checks"].items() if not passed],
        },
        "confirmation_probe": {
            "pass": confirmation_gate["pass"],
            "metrics": confirmation_gate["metrics"],
            "failed_checks": [name for name, passed in confirmation_gate["checks"].items() if not passed],
        },
        "artifact": OUTPUT_PATH.as_posix(),
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0 if eligible else 2


def correct_existing_result() -> int:
    from synth.chronicle import load_chronological

    prior_bytes = OUTPUT_PATH.read_bytes()
    prior_sha256 = sha256_bytes(prior_bytes)
    if prior_sha256 != EXPECTED_PRIOR_OUTPUT_SHA256:
        raise ValueError(
            "correction requires the exact reviewed 34-failure record; "
            f"found {prior_sha256}"
        )
    prior = json.loads(prior_bytes.decode("utf-8"))
    if prior.get("diagnostic_runner_sha256") != EXPECTED_PRIOR_RUNNER_SHA256:
        raise ValueError("prior Task 5 runner provenance does not match the frozen record")
    prior_failures = set(prior["failed_gates"])
    obsolete_checks = {
        f"EG1/{history}/{name}"
        for history in EXPECTED_ROLE_IDS
        for name in ("chronological_file_order", "chronological_failure_event_order")
    }
    observed_obsolete = prior_failures.intersection(obsolete_checks)
    if len(prior_failures) != 34 or observed_obsolete != obsolete_checks:
        raise ValueError("prior Task 5 result is not the expected 34-failure candidate")

    binding, materialization, _, entries, _ = load_bound_inputs()
    root_records = {row["history_id"]: row for row in materialization["root_records"]}
    prior_structural = prior["structural_by_history"]
    corrected_structural: dict[str, dict] = {}
    reload_order_by_history: dict[str, bool] = {}
    for entry in entries:
        history = entry["history_id"]
        manifest, _ = validate_manifest(entry, root_records[history], binding, materialization)
        summary = structural_summary(entry, manifest)
        samples, loaded_manifest = load_chronological(Path(entry["directory"]))
        reload_order = (
            canonical_json(loaded_manifest) == canonical_json(manifest)
            and [sample.file_id for sample in samples]
            == [row["file_id"] for row in manifest["files"]]
        )
        summary["checks"]["manifest_shard_reload_order_matches_manifest"] = reload_order
        if not reload_order:
            summary["core_integrity_failures"].append(
                "manifest_shard_reload_order_matches_manifest"
            )
        summary["core_integrity_pass"] = all(summary["checks"].values())
        reload_order_by_history[history] = reload_order
        del samples

        old = prior_structural[history]
        expected_checks = dict(old["checks"])
        for name in ("chronological_file_order", "chronological_failure_event_order"):
            expected_checks.pop(name, None)
        expected_checks["per_robot_operation_serialization"] = expected_checks.pop(
            "one_operation_per_robot"
        )
        expected_checks["manifest_shard_reload_order_matches_manifest"] = (
            prior["reload_identity"]["per_history"][history][
                "sample_order_matches_manifest_both_passes"
            ]
        )
        if summary["checks"] != expected_checks:
            raise ValueError(f"unexpected structural check change in {history}")
        if set(old["core_integrity_failures"]) != {
            "chronological_file_order", "chronological_failure_event_order"
        }:
            raise ValueError(f"unexpected prior EG1 failure set in {history}")
        for field in (
            "nuisance", "analytic_signal_margin", "maintenance_window_count",
            "maintenance_overlap_robots_diagnostic_only", "support",
        ):
            if summary[field] != old[field]:
                raise ValueError(f"non-chronology evidence changed in {history}: {field}")
        if not summary["core_integrity_pass"]:
            raise ValueError(f"normative EG1 check failed in {history}")
        corrected_structural[history] = summary

    if not all(reload_order_by_history.values()):
        raise ValueError("manifest/shard reload order did not pass for all bound roots")
    if (not prior["reload_identity"]["pass"]
            or prior["reload_identity"]["histories_verified"] != len(EXPECTED_ROLE_IDS)
            or not all(
                item["sample_order_matches_manifest_both_passes"]
                for item in prior["reload_identity"]["per_history"].values()
            )):
        raise ValueError("prior two-process manifest/shard reload evidence is not complete")

    design_checks = {
        history: design_promotion_checks(corrected_structural[history])
        for history in EXPECTED_ROLE_IDS[:4]
    }
    confirmation_checks = {
        history: confirmation_hard_floor_checks(corrected_structural[history])
        for history in EXPECTED_ROLE_IDS[-4:]
    }
    if design_checks != prior["design_promotion_checks"]:
        raise ValueError("Design promotion evidence changed outside EG1 chronology")
    if confirmation_checks != prior["confirmation_structural_hard_floor_checks"]:
        raise ValueError("Confirmation floor evidence changed outside EG1 chronology")
    if (set(name for name, checks in design_checks.items()
            if not all(checks.values())) != {"H-S18-DES-01"}
            or set(name for name, checks in confirmation_checks.items()
                   if not all(checks.values())) != {"H-S18-CONF-03"}):
        raise ValueError("cohort-mix rejection set does not match the reviewed candidate")

    remaining_failures = prior_failures - obsolete_checks
    if len(remaining_failures) != 2 or not all(
        "cohort_mix" in name for name in remaining_failures
    ):
        raise ValueError("remaining failures are not exactly the two cohort-mix rejections")
    global_checks = prior["global_gate_checks"]
    if not (
        global_checks["EG1_all_roots_core_integrity"] is False
        and global_checks["EG1_two_fresh_process_identical_reloads"] is True
    ):
        raise ValueError("prior global EG1 state is not the expected chronology mismatch")
    expected_false_gates = {
        name for name, passed in global_checks.items() if not passed
    }
    if expected_false_gates != {
        "EG1_all_roots_core_integrity",
        "EG2_each_design_promotion_target_and_concentration",
        "EG5_confirmation_hard_floors_and_concentration",
    }:
        raise ValueError("unexpected failing global gate outside the corrected EG1 checks")

    corrected = prior
    corrected["structural_by_history"] = corrected_structural
    corrected["design_promotion_checks"] = design_checks
    corrected["confirmation_structural_hard_floor_checks"] = confirmation_checks
    corrected["global_gate_checks"]["EG1_all_roots_core_integrity"] = True
    corrected["failed_gates"] = sorted(remaining_failures)
    role_entries = {
        role: [entry for entry in entries if entry["role"] == role]
        for role in EXPECTED_ROLE_COUNTS
    }
    for role, outcome in corrected["role_outcomes"].items():
        outcome["core_integrity_pass"] = all(
            corrected_structural[entry["history_id"]]["core_integrity_pass"]
            for entry in role_entries[role]
        )
    corrected["role_outcomes"]["DESIGN"]["pass"] = (
        corrected["role_outcomes"]["DESIGN"]["core_integrity_pass"]
        and all(all(checks.values()) for checks in design_checks.values())
    )
    corrected["role_outcomes"]["FIT"]["pass"] = (
        corrected["role_outcomes"]["FIT"]["core_integrity_pass"]
        and corrected["fit_healthy_support"]["pass"]
    )
    corrected["role_outcomes"]["CALIBRATION"]["pass"] = (
        corrected["role_outcomes"]["CALIBRATION"]["core_integrity_pass"]
        and corrected["calibration_healthy_support_and_threshold"]["pass"]
    )
    corrected["role_outcomes"]["DEVELOPMENT"]["pass"] = (
        corrected["role_outcomes"]["DEVELOPMENT"]["core_integrity_pass"]
        and corrected["observable_probe"]["development"]["pass"]
    )
    corrected["role_outcomes"]["CONFIRMATION"]["pass"] = (
        corrected["role_outcomes"]["CONFIRMATION"]["core_integrity_pass"]
        and all(all(checks.values()) for checks in confirmation_checks.values())
        and corrected["confirmation_stability"]["pass"]
        and corrected["observable_probe"]["confirmation"]["pass"]
    )
    corrected["eligible_for_representation_execution"] = all(
        corrected["global_gate_checks"].values()
    )
    if (corrected["eligible_for_representation_execution"]
            or corrected["qualification_outcome"]
            != "NOT_ELIGIBLE_FOR_REPRESENTATION_EXECUTION"
            or corrected["scientific_verdict"] != "REJECTED-CANDIDATE"
            or corrected["measurement_complete"] is not True):
        raise ValueError("corrected candidate must remain a complete scientific rejection")
    corrected["diagnostic_runner_sha256"] = sha256_file(Path(__file__))
    corrected["correction_audit"] = {
        "scope": "EG1 chronology interpretation only; all other full-run evidence retained",
        "superseded_record": {
            "artifact": CORRECTION_ARCHIVE_PATH.as_posix(),
            "sha256": prior_sha256,
            "diagnostic_runner_sha256": EXPECTED_PRIOR_RUNNER_SHA256,
            "prior_failed_gate_count": len(prior_failures),
            "unsupported_global_sort_failure_count": len(observed_obsolete),
        },
        "audited_existing_roots": len(entries),
        "per_robot_operation_serialization_pass_count": sum(
            summary["checks"]["per_robot_operation_serialization"]
            for summary in corrected_structural.values()
        ),
        "manifest_shard_reload_order_pass_count": sum(reload_order_by_history.values()),
        "two_fresh_process_reload_identity_retained": True,
        "remaining_true_failed_gate_count": len(remaining_failures),
        "remaining_true_failed_gates": sorted(remaining_failures),
        "representation_execution_authorized": False,
    }

    if CORRECTION_ARCHIVE_PATH.exists():
        if CORRECTION_ARCHIVE_PATH.read_bytes() != prior_bytes:
            raise FileExistsError("superseded-record archive already exists with different bytes")
    else:
        CORRECTION_ARCHIVE_PATH.write_bytes(prior_bytes)
    OUTPUT_PATH.write_text(
        json.dumps(corrected, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "superseded_record_sha256": prior_sha256,
        "prior_failed_gate_count": len(prior_failures),
        "unsupported_global_sort_failure_count": len(observed_obsolete),
        "audited_existing_roots": len(entries),
        "per_robot_operation_serialization_pass_count":
            corrected["correction_audit"]["per_robot_operation_serialization_pass_count"],
        "manifest_shard_reload_order_pass_count":
            corrected["correction_audit"]["manifest_shard_reload_order_pass_count"],
        "remaining_true_failed_gate_count": len(remaining_failures),
        "remaining_true_failed_gates": corrected["failed_gates"],
        "scientific_verdict": corrected["scientific_verdict"],
        "qualification_outcome": corrected["qualification_outcome"],
        "artifact": OUTPUT_PATH.as_posix(),
        "superseded_record": CORRECTION_ARCHIVE_PATH.as_posix(),
    }, indent=2, sort_keys=True))
    return 2



def fail_closed(error: Exception) -> int:
    error_record = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_id": "sprint18-task5-measurability-v1",
            "task": "Sprint 18 Task 5",
            "measurement_complete": False,
            "qualification_outcome": "DIAGNOSTIC_INCOMPLETE",
            "scientific_verdict": None,
            "eligible_for_representation_execution": None,
            "diagnostic_error": error_record,
            "execution_boundaries": {
                "generated_or_regenerated_roots": False,
                "preflight_rerun": False,
                "representation_training_or_scoring": False,
                "sprint15_sealed_paths_or_histories_accessed": False,
            },
            "diagnostic_runner_sha256": sha256_file(Path(__file__)),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps({
        "qualification_outcome": "DIAGNOSTIC_INCOMPLETE",
        "measurement_complete": False,
        "scientific_verdict": None,
        "diagnostic_error": error_record,
        "artifact": OUTPUT_PATH.as_posix(),
    }, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--correct-existing-result":
        raise SystemExit(correct_existing_result())
    if len(sys.argv) > 1 and sys.argv[1] == "--reload-only":
        raise SystemExit(reload_only())
    try:
        result = main()
    except Exception as error:
        result = fail_closed(error)
    raise SystemExit(result)
