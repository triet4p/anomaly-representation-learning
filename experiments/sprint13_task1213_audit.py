"""Structural causality (Task 12) and coverage (Task 13) audit — no scores.

Reads the 13 materialized Protocol v4 manifests plus deterministic reloads
and in-memory rebuilds. Inspects only permitted structural metadata
(Task 6 section 4): counts, tallies, windows, digests. No file scores, no
thresholds, no model comparisons. Writes one bounded JSON summary.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

DAY = 86400.0
HORIZON_S = 7.0 * DAY

ROSTER = [
    ("H-VAL-DESIGN", 300), ("H-DEV-1", 301), ("H-DEV-2", 302),
    ("H-DEV-3", 303), ("H-FIT-1", 304), ("H-FIT-2", 305),
    ("H-FIT-3", 306), ("H-CAL-1", 307), ("H-CONF-1", 308),
    ("H-SEAL-1", 400), ("H-SEAL-2", 401), ("H-SEAL-3", 402),
    ("H-SEAL-4", 403),
]

BASE = "data/generated/sprint13"


def sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check(cond: bool, name: str, errors: list) -> None:
    if not cond:
        errors.append(name)


def audit_history(role: str, seed: int) -> dict:
    from synth.chronicle import load_chronological
    from synth import events as E

    errors: list = []
    root = f"{BASE}/{role}"
    manifest = json.load(open(f"{root}/manifest.json"))
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]

    # --- identity / provenance ---
    check(manifest["role"] == role, "role-match", errors)
    check(manifest.get("protocol") == "sprint13-protocol-v4", "protocol-tag", errors)
    check(manifest["seeds"]["health"] == seed, "seed-match", errors)
    samples, _ = load_chronological(root)
    check([s.file_id for s in samples] == [r["file_id"] for r in rows],
          "reload-order", errors)
    for sample in samples:
        sample.validate()

    # --- route/robot/program causality: one op per robot at a time ---
    by_robot: dict = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    robots = sorted(by_robot)
    check(len(robots) == 8, "eight-robots", errors)
    check("robot-08" in by_robot, "robot-08-present", errors)
    programs = {r["program_id"] for r in rows}
    routes = {e["route_id"] for e in manifest["schedule"]}
    check(routes <= {"route-A", "route-B", "route-C", "route-D", "route-E"},
          "route-vocabulary", errors)
    check("program-03" in programs, "program-03-present", errors)
    for robot, file_rows in by_robot.items():
        ordered = sorted(file_rows, key=lambda r: (r["start_time"], r["end_time"]))
        for prev, cur in zip(ordered, ordered[1:]):
            check(cur["start_time"] >= prev["end_time"] - 1e-6,
                  f"robot-serialization:{robot}", errors)
            if errors and errors[-1].startswith("robot-serialization"):
                break

    # --- chronological sanity / leakage guards ---
    for row in rows:
        if not (row["end_time"] >= row["start_time"] >= 0.0):
            errors.append("file-time-order")
            break
    allowed_keys = {"file_id", "operation_id", "robot_id", "program_id",
                    "start_time", "end_time", "file_label", "is_quarantined",
                    "quarantine_reason", "is_censored", "member_views",
                    "last_reset_time"}
    check(all(set(r) <= allowed_keys for r in rows), "no-score-keys", errors)
    views = manifest["splits"]
    check(not (set(views["dev_train"]) & set(views["quarantined"])),
          "quarantine-dev-disjoint", errors)
    check(not (set(views["dev_val"]) & set(views["quarantined"])),
          "quarantine-dev-disjoint", errors)
    cutoff = manifest["calendar"]["cutoff_time"]
    dev_ids = set(views["dev_train"]) | set(views["dev_val"])
    by_id = {r["file_id"]: r for r in rows}
    check(all(by_id[i]["end_time"] <= cutoff + 1e-6 for i in dev_ids),
          "dev-pre-cutoff", errors)

    # --- maintenance behavior ---
    for robot, intervals in wins.items():
        for start, end in intervals:
            check(end > start, f"maint-bounded:{robot}", errors)
    maint_states = 0
    for row in rows:
        reset = row["last_reset_time"]
        check(reset <= row["start_time"] + 1e-6, "reset-causal", errors)
        if any(s <= row["start_time"] < e
               for s, e in wins.get(row["robot_id"], [])):
            maint_states += 1
    check(maint_states > 0, "maint-states-exist", errors)

    # --- category assignment ---
    cohorts = Counter(r["cohort"] for r in ledger)
    check(set(cohorts) == {"P", "W", "A"}, "all-cohorts", errors)
    for record in ledger:
        if record["cohort"] == "P":
            check(7.0 <= record["duration_d"] <= 14.0, "p-duration", errors)
        elif record["cohort"] == "W":
            check(14.0 <= record["duration_d"] <= 28.0, "w-duration", errors)
        else:
            check(record["duration_d"] == 0.0
                  and record["degradation_onset"] is None
                  and record["subtype"] in ("A1", "A2"), "a-shape", errors)

    # --- Task 13: coverage floors (overall ledger, no holdout exclusion) ---
    pos_eval, uneval = 0, 0
    pos_cat: Counter = Counter()
    for failure in ledger:
        cands = E.pos_files(rows, failure, wins)
        if cands:
            pos_eval += 1
            pos_cat[failure["cohort"]] += 1
        else:
            uneval += 1
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger)
    ctrl_by_robot = Counter(w["robot_id"] for w in controls)
    base_cat: Counter = Counter()
    for failure in ledger:
        t_end = failure["failure_time"]
        robot = failure["robot_id"]
        cands = [x for x in rows
                 if x["robot_id"] == robot
                 and t_end - 42 * DAY <= x["end_time"] < t_end - 7 * DAY
                 and not x["is_quarantined"]
                 and not E._overlaps_maintenance(x, wins)]
        degs = [(g["degradation_onset"], g["failure_time"]) for g in ledger
                if g["robot_id"] == robot and g["degradation_onset"] is not None]
        if any(not any(o <= x["end_time"] < t for o, t in degs) for x in cands):
            base_cat[failure["cohort"]] += 1
    eval_days = set()
    for row in rows:
        if not E.eligible_operational_row(row, wins):
            continue
        eval_days.add((row["robot_id"], int(row["end_time"] // DAY)))
    healthy = [r for r in rows
               if r["file_label"] == "normal" and not r["is_quarantined"]
               and r["end_time"] <= cutoff]
    fit_groups: Counter = Counter()
    for r in healthy:
        fit_groups[(r["robot_id"], r["program_id"])] += 1

    HOLDOUT_PROGRAMS = ("program-03",)
    HOLDOUT_ROBOTS = ("robot-08",)
    healthy_eligible = [
        r for r in healthy
        if r["program_id"] not in HOLDOUT_PROGRAMS
        and r["robot_id"] not in HOLDOUT_ROBOTS
    ]
    dev_val_ids = set(manifest["splits"]["dev_val"])
    cal_eligible = [
        r for r in rows
        if r["file_id"] in dev_val_ids
        and r["program_id"] not in HOLDOUT_PROGRAMS
        and r["robot_id"] not in HOLDOUT_ROBOTS
    ]
    fit_groups_eligible: Counter = Counter()
    for r in healthy_eligible:
        fit_groups_eligible[(r["robot_id"], r["program_id"])] += 1
    from synth.config import PatchConfig as _PatchConfig
    from synth.patchify import Patchifier as _Patchifier
    _pcfg = manifest["resolved_config"]["patch"]
    _patchifier = _Patchifier(_PatchConfig(
        patch_size=int(_pcfg["patch_size"]), stride=int(_pcfg["stride"]),
        pad_end=bool(_pcfg["pad_end"]), pad_value=float(_pcfg["pad_value"])))
    _by_file = {s.file_id: s for s in samples}
    patch_rows_eligible = sum(
        int(_patchifier.patchify(_by_file[r["file_id"]]).patches.shape[0])
        for r in healthy_eligible
    )
    floors = {
        "positives_ge_25": pos_eval >= 25,
        "negatives_ge_25": len(controls) >= 25,
        "robot_days_ge_150": len(eval_days) >= 150,
        "category_ge_8": all(pos_cat.get(c, 0) >= 8 for c in "PWA"),
        "triggers_ok": pos_eval >= 2 and len(controls) >= 6,
    }
    struct = ("STRUCT-PASS" if all(floors.values()) else
              ("UNAVAILABLE" if not floors["triggers_ok"] else "STRUCT-FAIL"))

    per_robot = {}
    for robot in robots:
        robot_failures = [f for f in ledger if f["robot_id"] == robot]
        per_robot[robot] = {
            "failures": len(robot_failures),
            "cohorts": dict(Counter(f["cohort"] for f in robot_failures)),
            "controls": ctrl_by_robot.get(robot, 0),
            "files": len(by_robot[robot]),
        }
    return {
        "role": role, "seed": seed,
        "manifest_sha256": sha(f"{root}/manifest.json"),
        "config_hash": manifest["config_hash"],
        "counts": manifest["counts"],
        "quarantine_reasons": dict(Counter(
            r["quarantine_reason"] for r in rows if r["is_quarantined"])),
        "failures_total": len(ledger),
        "cohorts": dict(cohorts),
        "task12_errors": errors,
        "task12_pass": not errors,
        "positives_evaluable": pos_eval,
        "positives_unevaluable": uneval,
        "positives_by_cohort": dict(pos_cat),
        "negatives": len(controls),
        "negatives_by_robot": dict(ctrl_by_robot),
        "clean_baselines_by_cohort": dict(base_cat),
        "evaluated_robot_days": len(eval_days),
        "healthy_precutoff": len(healthy),
        "healthy_eligible": len(healthy_eligible),
        "cal_eligible_rows": len(cal_eligible),
        "eligible_patch_rows": patch_rows_eligible,
        "fit_group_min": min(fit_groups.values()) if fit_groups else 0,
        "fit_group_min_eligible": (
            min(fit_groups_eligible.values()) if fit_groups_eligible else 0),
        "floors": floors,
        "structural_state": struct,
        "per_robot": per_robot,
    }


def main() -> int:
    results = []
    for role, seed in ROSTER:
        try:
            results.append(audit_history(role, seed))
        except Exception as exc:  # noqa: BLE001 — audit must report, not crash
            results.append({"role": role, "seed": seed, "audit_error": str(exc)})
        print(f"audited {role}: "
              f"{'ERROR ' + results[-1].get('audit_error', '') if 'audit_error' in results[-1] else ('task12=' + str(results[-1]['task12_pass']) + ' struct=' + results[-1]['structural_state'])}",
              flush=True)
    with open("/tmp/s13_audit1213.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
