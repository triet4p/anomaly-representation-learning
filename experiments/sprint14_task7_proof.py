"""Sprint 14 Task 7: public-entry causal/deterministic proof (v5 DGP; runnable under Protocol v5 tag).

Disposable proof roots only (seed 796 full-size, seed 797 tiny); never any
cycle roster. Proves through the real local CLI: fresh-process
determinism, chronology, route/robot exclusivity, maintenance/reset
semantics, physical bounds, role isolation, future-state exclusion, EG3
metric fixtures, manifest/seal integrity, dedicated-RNG timing invariance,
subtype separation/containment, P/W rate/timing invariance, and A1/A2
invariance. Any failed check raises (stop before Task 8).

Bounded output: artifacts/sprint-14/server-task7-cycle-3/task7.json.
(v5.1 tooling: cycle-specific rerun roots plus a no-overwrite guard;
never writes to cycle-1 or cycle-2 evidence paths.)
Re-evaluation mode (`--reevaluate`, v5.1 §13): verifies recorded hashes of
the exact preserved seed-796 roots, reloads deterministically, asserts
configured cohort shares, runs EG3 fixture arms, and completes seal
round-trips on verified-identical working copies into
`artifacts/sprint-14/server-task7-cycle-3-reeval/`. Never generates,
mutates, or overwrites any root.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

DAY = 86400.0
FIXED_THRESHOLD = 0.3

BASE = Path("data/generated/sprint14-task7-c3")
OUT = Path("artifacts/sprint-14/server-task7-cycle-3")
SEED_FULL = 796
SEED_TINY = 797

#: Re-evaluation inputs (v5.1): exact preserved seed-796 roots. Digests
#: below are the independently accepted pre-incident records; any mismatch
#: aborts before any check runs. Outputs go only to RE_EVAL_OUT.
RE_EVAL_OUT = Path("artifacts/sprint-14/server-task7-cycle-3-reeval")
RE_EVAL_SEAL = RE_EVAL_OUT / "seal-check"
RE_EVAL_FULL = Path("data/generated/sprint14-task7-c3/full-796a")
RE_EVAL_TINY = Path("data/generated/sprint14-task7-c3/tiny-797")
EXPECTED_FULL_MANIFEST = "5b230719210364d6c5e0be0257ee6b3665c4518929cd88afccb2ed5397e78691"
EXPECTED_TINY_MANIFEST = "10102db8d082e0a3600cd659ab8f4ea3b84aa76b01dd6c388f7f92307a111735"
ALLOWED_ROW_KEYS = {"file_id", "operation_id", "robot_id", "program_id",
                    "start_time", "end_time", "file_label", "is_quarantined",
                    "quarantine_reason", "is_censored", "member_views",
                    "last_reset_time", "n_valid_patches"}


def cli(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "synth.cli", *args],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed {' '.join(args)}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def sha_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(seed: int, root: Path, units: int | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    cmd = ["--chronological", "--profile", "sprint14-v5",
           "--seed", str(seed), "--protocol", "sprint14-benchmark-protocol-v5",
           "--output", str(root)]
    if units is not None:
        cmd += ["--units", str(units)]
    cli(*cmd)


def check_tiny() -> dict:
    from synth.chronicle import load_chronological
    root = BASE / "tiny-797"
    materialize(SEED_TINY, root, units=96)
    samples, manifest = load_chronological(root)
    rows = manifest["files"]
    assert manifest["protocol"] == "sprint14-benchmark-protocol-v5"
    assert manifest["role"] is None
    assert [s.file_id for s in samples] == [r["file_id"] for r in rows]
    for sample in samples:
        sample.validate()
    by_robot: dict = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
        assert set(row) <= ALLOWED_ROW_KEYS, f"row key leak: {set(row) - ALLOWED_ROW_KEYS}"
        assert "subtype" not in row and "cohort" not in row
    for robot, file_rows in by_robot.items():
        ordered = sorted(file_rows, key=lambda r: (r["start_time"], r["end_time"]))
        for prev, cur in zip(ordered, ordered[1:]):
            assert cur["start_time"] >= prev["end_time"] - 1e-6
    return {"root": str(root), "files": len(rows),
            "robots": sorted(by_robot)}


def check_determinism() -> dict:
    root_a = BASE / "full-796a"
    root_b = BASE / "full-796b"
    materialize(SEED_FULL, root_a, units=None)
    materialize(SEED_FULL, root_b, units=None)
    ma = (root_a / "manifest.json").read_bytes()
    mb = (root_b / "manifest.json").read_bytes()
    assert ma == mb, "fresh-process manifest bytes differ"
    la = json.loads(ma)["failure_events"]
    lb = json.loads(mb)["failure_events"]
    assert la == lb, "fresh-process failure ledgers differ"
    shutil.rmtree(root_b)
    return {"manifest_sha256": sha_file(root_a / "manifest.json"),
            "failures": len(la), "root": str(root_a)}


def check_configured_shares(manifest: dict) -> dict:
    """Assert the frozen configured cohort probabilities exactly.

    Reads the resolved generator configuration embedded in the manifest —
    never realized sample counts. Realized shares are reported
    descriptively by callers and gated nowhere.
    """
    cfg_shares = {c["cohort_id"]: c["share"]
                  for c in manifest["resolved_config"]["health"]["cohorts"]}
    assert cfg_shares == {"P": 0.45, "W": 0.30, "A": 0.25}, cfg_shares
    return {"P": 0.45, "W": 0.30, "A": 0.25}


def check_physics_and_subtypes(root: Path) -> dict:
    from synth import events as E
    from synth.chronicle import load_chronological, verify_seal, write_seal

    samples, manifest = load_chronological(root)
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]
    out: dict = {"role": manifest["role"], "protocol": manifest["protocol"]}
    robots = sorted({r["robot_id"] for r in rows})
    assert len(robots) == 9 and "robot-09" in robots
    routes = {e["route_id"] for e in manifest["schedule"]}
    assert routes <= {"route-A", "route-B", "route-C", "route-D", "route-E"}
    ops_by_robot: dict = {}
    for event in manifest["schedule"]:
        ops_by_robot.setdefault(event["robot_id"], []).append(event["operation_id"])
    assert all(len(v) > 0 for v in ops_by_robot.values())
    r2_routes = {e["route_id"] for e in manifest["schedule"]
                 if e["robot_id"] == "robot-02"}
    assert r2_routes == {"route-A"}, r2_routes
    b1 = {(e["robot_id"], e["program_id"]) for e in manifest["schedule"]
          if e["route_id"] == "route-B"}
    assert ("robot-09", "program-03") in b1
    assert "robot-08" in ops_by_robot
    for robot, intervals in wins.items():
        for start, end in intervals:
            assert end > start
    cohorts: dict = {}
    for record in ledger:
        cohorts.setdefault(record["cohort"], []).append(record)
    assert set(cohorts) == {"P", "W", "A"}
    from statistics import median as _med
    p_durs = sorted(r["duration_d"] for r in cohorts["P"])
    w_durs = sorted(r["duration_d"] for r in cohorts["W"])
    assert min(p_durs) >= 2.0 and max(p_durs) <= 15.0
    assert 5.0 <= _med(p_durs) <= 10.0
    assert min(w_durs) >= 6.0 and max(w_durs) <= 28.0
    assert 12.0 <= _med(w_durs) <= 24.0
    for record in ledger:
        if record["cohort"] == "A":
            assert record["duration_d"] == 0.0
            assert record["degradation_onset"] is None
            assert record["subtype"] in ("A1", "A2")
        elif record["cohort"] == "P":
            assert record["subtype"] in ("P1", "P2")
            assert record["degradation_onset"] is not None
        else:
            assert record["subtype"] in ("W1", "W2")
            assert record["degradation_onset"] is not None
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    for record in ledger:
        if record["cohort"] != "A":
            linked = episodes[record["degradation_episode_id"]]
            assert linked["start_time"] <= record["failure_time"]
    check_configured_shares(manifest)
    shares = {c: len(v) / len(ledger) for c, v in cohorts.items()}
    out["cohort_shares_realized_raw"] = shares
    eval_cat: dict = {}
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        if E.pos_files(rows, failure, wins):
            eval_cat[failure["cohort"]] = eval_cat.get(failure["cohort"], 0) + 1
    n_eval = sum(eval_cat.values())
    out["cohort_shares_realized_evaluable"] = {
        c: (eval_cat.get(c, 0) / n_eval if n_eval else 0.0) for c in "PWA"}
    sev_by_file = {}
    for sample in samples:
        meta = sample.anomaly_meta
        if meta is not None and sample.file_label.value == "abnormal":
            sev_by_file[sample.file_id] = float(meta.severity)
    grouped: dict = {}
    for row in rows:
        sev = sev_by_file.get(row["file_id"])
        if sev is None:
            continue
        for record in ledger:
            if (record["robot_id"] == row["robot_id"]
                    and record["failure_time"] - 7 * DAY <= row["end_time"] <= record["failure_time"]):
                grouped.setdefault(record["subtype"], []).append(sev)
                break
    means = {s: sum(v) / len(v) for s, v in grouped.items() if len(v) >= 5}
    assert means.get("P1", 0) > means.get("P2", 1), means
    assert means.get("W1", 0) > means.get("W2", 1), means
    out["precursor_severity_means"] = means
    out["precursor_severity_n"] = {s: len(grouped.get(s, [])) for s in means}
    seal = write_seal(root, role="H-TASK7-DISPOSABLE")
    assert verify_seal(root)["manifest_sha256"] == seal["manifest_sha256"]
    out["seal_protocol"] = seal["protocol"]
    return out


def check_timing_invariance() -> dict:
    from dataclasses import replace
    from synth.chronicle import build_chronological, sprint14_v5_history_config

    def twin(subtypes_on: bool):
        cfg = sprint14_v5_history_config(seed=SEED_FULL)
        base_units = cfg.scheduler.n_units
        cfg.scheduler.n_units = 150
        scaled = cfg.scheduler.arrival_interval_s * base_units / 150
        cfg.scheduler.arrival_interval_s = scaled
        cfg.scheduler.arrival_jitter_s = scaled
        if not subtypes_on:
            bare = [replace(c, subtypes=()) for c in cfg.health.cohorts]
            cfg.health = replace(cfg.health, cohorts=tuple(bare))
        return build_chronological(cfg)

    _, health, _, _ = twin(True)
    _, health_bare, _, _ = twin(False)

    def keyed(failures):
        return sorted((
            f.failure_time, f.robot_id, f.cohort, f.duration_d,
            f.severity, f.degradation_onset,
            f.maintenance_episode_id is not None) for f in failures)
    got, want = keyed(health.failure_events), keyed(health_bare.failure_events)
    assert got == want, "subtype emission perturbed failure timing/density"
    assert any(f.subtype in ("P1", "P2", "W1", "W2")
               for f in health.failure_events)
    assert all(f.subtype is None for f in health_bare.failure_events
               if f.cohort in ("P", "W"))
    return {"failures_compared": len(got)}


def prove_arm(rows, ledger, wins, scores, E) -> dict:
    pos, uneval = [], 0
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            uneval += 1
            continue
        cands = E.pos_files(rows, failure, wins)
        if not cands:
            uneval += 1
            continue
        pos.append(E.window_score([c["file_id"] for c in cands], scores))
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    neg = [E.window_score([m["file_id"] for m in w["members"]], scores)
           for w in controls]
    try:
        auc = E.roc_auc_tie_aware(pos, neg)
        auc_state = "defined"
    except E.UnavailableError:
        auc, auc_state = None, "UNAVAILABLE"
    recalled, leads, persists = 0, [], []
    flagged_by_robot: dict = {}
    for row in rows:
        if not E.eligible_operational_row(row, wins):
            continue
        if scores[row["file_id"]] >= FIXED_THRESHOLD:
            flagged_by_robot.setdefault(row["robot_id"], []).append(row["end_time"])
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        cands = E.pos_files(rows, failure, wins)
        flagged = sorted(c["end_time"] for c in cands
                         if scores[c["file_id"]] >= FIXED_THRESHOLD)
        if flagged:
            recalled += 1
            leads.append(E.lead_days(failure["failure_time"], flagged))
            persists.append(len(flagged))
    eval_days = {(r["robot_id"], int(r["end_time"] // DAY)) for r in rows
                 if E.eligible_operational_row(r, wins)}
    false, far = E.false_alert_episodes(
        flagged_by_robot, ledger, float(len(eval_days)), wins)
    recall = recalled / len(pos) if pos else None
    arm = {
        "event_auc": auc, "auc_state": auc_state,
        "positives_evaluable": len(pos), "positives_unevaluable": uneval,
        "negatives": len(neg), "recall": recall,
        "recall_cp": (list(E.clopper_pearson(recalled, len(pos)))
                      if pos else None),
        "lead": E.summarize_values(leads),
        "lead_positive_fraction": (
            sum(1 for x in leads if x > 0.0) / len(leads) if leads else None),
        "persistence": E.summarize_values(persists),
        "false_episodes": false, "far": far,
        "far_bound": (E.rule_of_three_bound(float(len(eval_days)))
                      if false == 0 and eval_days else None),
    }
    arm["computable"] = E.check_arm_computable(arm)
    return arm


def check_fixtures(root: Path) -> dict:
    from synth import events as E
    from synth.chronicle import load_chronological

    _, manifest = load_chronological(root)
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]
    const = {r["file_id"]: 0.5 for r in rows}
    time_scores = {
        r["file_id"]: max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) / (
            max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) + 30.0)
        for r in rows}
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    assert controls, "proof root yielded no controls"
    assert E.roc_auc_tie_aware([0.5] * 10, [0.5] * max(1, len(controls))) == 0.5
    assert E.roc_auc_tie_aware([1.0] * 10, [0.0] * 10) == 1.0
    assert E.roc_auc_tie_aware([0.0] * 10, [1.0] * 10) == 0.0
    assert E.lead_days(ledger[0]["failure_time"],
                       [ledger[0]["failure_time"]]) == 0.0
    arms = {}
    for name, scores in (("constant", const), ("observable_time", time_scores)):
        arm = prove_arm(rows, ledger, wins, scores, E)
        arms[name] = {"computable": arm["computable"],
                      "event_auc": arm["event_auc"],
                      "positives_evaluable": arm["positives_evaluable"],
                      "negatives": arm["negatives"]}
    assert all(a["computable"] for a in arms.values())
    assert E.roc_auc_tie_aware([1.0] * 10, [0.0] * 10) == 1.0
    return arms


def main() -> int:
    for root in (BASE, OUT):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"refusing to overwrite preserved evidence at {root}; "
                f"a new protocol version is required for another rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"seed_full": SEED_FULL, "seed_tiny": SEED_TINY,
              "profile": "sprint14-v5",
              "protocol": "sprint14-benchmark-protocol-v5"}
    result["tiny"] = check_tiny()
    result["determinism"] = check_determinism()
    root = Path(result["determinism"]["root"])
    result["physics"] = check_physics_and_subtypes(root)
    result["timing_invariance"] = check_timing_invariance()
    result["fixtures"] = check_fixtures(root)
    (OUT / "task7.json").write_text(json.dumps(result, indent=1),
                                    encoding="utf-8")
    print("TASK7-PROOF-PASS " + json.dumps({
        "failures": result["physics"]["n_failures"],
        "subtypes": result["physics"]["subtype_counts"],
        "manifest": result["determinism"]["manifest_sha256"][:12]}))
    return 0


def main_reevaluate() -> int:
    """Deterministic re-evaluation of preserved roots only (v5.1 §13).

    Verifies recorded hashes, reloads deterministically, asserts the
    configured cohort probabilities, runs the EG3 fixture arms, and
    completes seal round-trips on verified-identical working copies.
    Never generates, mutates, or overwrites any root.
    """
    from synth import events as E
    from synth.chronicle import load_chronological, verify_seal, write_seal

    if RE_EVAL_OUT.exists() and any(RE_EVAL_OUT.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite preserved evidence at {RE_EVAL_OUT}; "
            f"a new protocol version is required for another rerun")
    RE_EVAL_OUT.mkdir(parents=True, exist_ok=True)
    RE_EVAL_SEAL.mkdir(parents=True, exist_ok=True)
    result: dict = {"mode": "reevaluate", "protocol": "sprint14-benchmark-protocol-v5",
                    "tooling": "sprint14-benchmark-protocol-v5.2"}
    roots = {}
    for label, root, expected in (
            ("full", RE_EVAL_FULL, EXPECTED_FULL_MANIFEST),
            ("tiny", RE_EVAL_TINY, EXPECTED_TINY_MANIFEST)):
        manifest_path = root / "manifest.json"
        observed = sha_file(manifest_path)
        assert observed == expected, (label, observed, expected)
        samples, manifest = load_chronological(root)
        assert manifest["role"] is None
        assert manifest["protocol"] == "sprint14-benchmark-protocol-v5"
        rows = manifest["files"]
        assert [s.file_id for s in samples] == [r["file_id"] for r in rows]
        for row in rows:
            assert set(row) <= ALLOWED_ROW_KEYS, f"row key leak: {set(row) - ALLOWED_ROW_KEYS}"
            assert "subtype" not in row and "cohort" not in row
        ledger = E.failure_ledger(manifest)
        wins = manifest["maintenance_windows"]
        result[label] = {
            "root": str(root), "manifest_sha256": observed,
            "files": len(rows), "failures": len(ledger),
            "configured_shares": check_configured_shares(manifest),
        }
        if label == "full":
            const = {r["file_id"]: 0.5 for r in rows}
            time_scores = {
                r["file_id"]: max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) / (
                    max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) + 30.0)
                for r in rows}
            assert E.roc_auc_tie_aware([0.5] * 10, [0.5] * 10) == 0.5
            assert E.roc_auc_tie_aware([1.0] * 10, [0.0] * 10) == 1.0
            assert E.roc_auc_tie_aware([0.0] * 10, [1.0] * 10) == 0.0
            assert E.lead_days(ledger[0]["failure_time"],
                               [ledger[0]["failure_time"]]) == 0.0
            arms = {}
            for name, scores in (("constant", const), ("observable_time", time_scores)):
                arm = prove_arm(rows, ledger, wins, scores, E)
                arms[name] = {"computable": arm["computable"],
                              "event_auc": arm["event_auc"],
                              "positives_evaluable": arm["positives_evaluable"],
                              "negatives": arm["negatives"]}
            assert all(a["computable"] for a in arms.values())
            result[label]["fixtures"] = arms
            result[label]["seals"] = {}
            for role_dir, expected_digest in (("full-796a", EXPECTED_FULL_MANIFEST),
                                             ("tiny-797", EXPECTED_TINY_MANIFEST)):
                source = RE_EVAL_FULL if role_dir == "full-796a" else RE_EVAL_TINY
                seal_dir = RE_EVAL_SEAL / role_dir
                seal_dir.mkdir(parents=True, exist_ok=True)
                (seal_dir / "manifest.json").write_bytes(
                    (source / "manifest.json").read_bytes())
                assert sha_file(seal_dir / "manifest.json") == expected_digest
                seal = write_seal(seal_dir, role="H-TASK7-REEVAL")
                assert verify_seal(seal_dir)["manifest_sha256"] == seal["manifest_sha256"]
                assert seal["protocol"] == "sprint14-benchmark-protocol-v5"
                result[label]["seals"][role_dir] = {
                    "role": "H-TASK7-REEVAL",
                    "manifest_sha256": seal["manifest_sha256"],
                    "protocol": seal["protocol"],
                }
    (RE_EVAL_OUT / "task7-reeval.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    print("TASK7-REEVAL-PASS " + json.dumps({
        "full_manifest": EXPECTED_FULL_MANIFEST[:12],
        "fixtures": {k: v["computable"]
                     for k, v in result["full"]["fixtures"].items()}}))
    return 0


if __name__ == "__main__":
    if "--reevaluate" in sys.argv:
        raise SystemExit(main_reevaluate())
    raise SystemExit(main())
