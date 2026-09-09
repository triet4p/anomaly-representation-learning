"""Sprint 14 Task 9: cycle-2 Design promotion audit under Protocol v4.

Audits each Design history (742-745) independently against every hard
floor, every promotion target (negatives >= 32), concentration limits,
80% lead-support predicates, Fit/Calibration support projection, subtype
presence/eligibility, per-robot evaluable-positive concentration,
control-rejection reasons, and EG1/EG3. No pooling: DESIGN-PASS requires
every target on that history. No scores beyond deterministic fixtures;
no probe execution.

Reads: data/generated/sprint14-v4/H-DESIGN-<k>/manifest.json
Writes: artifacts/sprint-14/design-audit-cycle-2.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DAY = 86400.0
HORIZON_S = 7.0 * DAY
FIXED_THRESHOLD = 0.3

BASE = Path("data/generated/sprint14-v4")
OUT = Path("artifacts/sprint-14/design-audit-cycle-2.json")
ROSTER = [("H-DESIGN-5", 742), ("H-DESIGN-6", 743),
          ("H-DESIGN-7", 744), ("H-DESIGN-8", 745)]
ALLOWED_ROW_KEYS = {"file_id", "operation_id", "robot_id", "program_id",
                    "start_time", "end_time", "file_label", "is_quarantined",
                    "quarantine_reason", "is_censored", "member_views",
                    "last_reset_time", "n_valid_patches"}

PROMOTION = {"P": 13, "W": 13, "A": 10, "total": 38, "negatives": 32,
             "robot_days": 188, "robots_pos": 6, "robots_neg": 6,
             "programs": 2}


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


def audit_history(role: str, seed: int) -> dict:
    from synth import events as E
    from synth.chronicle import load_chronological

    errors: list = []
    root = BASE / role
    manifest = json.load(open(root / "manifest.json"))
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]

    def check(cond: bool, name: str) -> None:
        if not cond:
            errors.append(name)

    check(manifest["role"] == role, "role-match")
    check(manifest.get("protocol") == "sprint14-benchmark-protocol-v3",
          "protocol-tag")
    check(manifest["seeds"]["health"] == seed, "seed-match")
    samples, _ = load_chronological(root)
    check([s.file_id for s in samples] == [r["file_id"] for r in rows],
          "reload-order")
    by_robot: dict = {}
    for row in rows:
        by_robot.setdefault(row["robot_id"], []).append(row)
    check(sorted(by_robot) == [f"robot-{i:02d}" for i in range(1, 10)],
          "nine-robots")
    check("robot-09" in by_robot, "robot-09-present")
    check("robot-08" in by_robot, "robot-08-present")
    check("program-03" in {r["program_id"] for r in rows}, "program-03-present")
    check(all(set(r) <= ALLOWED_ROW_KEYS for r in rows), "no-score-keys")
    check(all("subtype" not in r and "cohort" not in r for r in rows),
          "no-subtype-in-rows")
    for robot, file_rows in by_robot.items():
        ordered = sorted(file_rows, key=lambda r: (r["start_time"], r["end_time"]))
        for prev, cur in zip(ordered, ordered[1:]):
            if not cur["start_time"] >= prev["end_time"] - 1e-6:
                errors.append(f"robot-serialization:{robot}")
                break
    for robot, intervals in wins.items():
        for start, end in intervals:
            if not end > start:
                errors.append(f"maint-bounded:{robot}")
                break
    cohorts = Counter(r["cohort"] for r in ledger)
    check(set(cohorts) == {"P", "W", "A"}, "all-cohorts")
    for record in ledger:
        if record["cohort"] == "A":
            check(record["duration_d"] == 0.0
                  and record["degradation_onset"] is None
                  and record["subtype"] in ("A1", "A2"), "a-shape")
        elif record["cohort"] == "P":
            check(record["subtype"] in ("P1", "P2"), "p-subtype")
        else:
            check(record["subtype"] in ("W1", "W2"), "w-subtype")
    from statistics import median as _med
    p_durs = sorted(r["duration_d"] for r in ledger if r["cohort"] == "P")
    w_durs = sorted(r["duration_d"] for r in ledger if r["cohort"] == "W")
    if p_durs:
        check(min(p_durs) >= 2.0 and max(p_durs) <= 15.0, "p-dist-shape")
        check(5.0 <= _med(p_durs) <= 10.0, "p-dist-median")
    if w_durs:
        check(min(w_durs) >= 6.0 and max(w_durs) <= 28.0, "w-dist-shape")
        check(12.0 <= _med(w_durs) <= 24.0, "w-dist-median")

    pos_eval, uneval = 0, 0
    pos_cat: Counter = Counter()
    pos_eval_by_robot: Counter = Counter()
    sub_eval: Counter = Counter()
    sub_total: Counter = Counter()
    lead_support: dict = {}
    for failure in ledger:
        if failure["subtype"]:
            sub_total[failure["subtype"]] += 1
        if E.positive_window_intersects_reset(failure, wins):
            uneval += 1
            continue
        cands = E.pos_files(rows, failure, wins)
        if cands:
            pos_eval += 1
            pos_cat[failure["cohort"]] += 1
            pos_eval_by_robot[failure["robot_id"]] += 1
            if failure["subtype"]:
                sub_eval[failure["subtype"]] += 1
            if failure["cohort"] in ("P", "W"):
                ends = sorted(c["end_time"] for c in cands)
                t_end = failure["failure_time"]
                robot = failure["robot_id"]
                base_cands = [x for x in rows
                              if x["robot_id"] == robot
                              and t_end - 42 * DAY <= x["end_time"] < t_end - 7 * DAY
                              and not x["is_quarantined"]
                              and not E._overlaps_maintenance(x, wins)]
                degs = [(g["degradation_onset"], g["failure_time"])
                        for g in ledger
                        if g["robot_id"] == robot
                        and g["degradation_onset"] is not None]
                clean = any(not any(o <= x["end_time"] < t for o, t in degs)
                            for x in base_cands)
                lead_support[failure["failure_id"]] = {
                    "endpoints": len(ends),
                    "early_endpoint": any(e <= t_end - DAY for e in ends),
                    "clean_baseline": clean,
                }
        else:
            uneval += 1
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger, wins)
    ctrl_by_robot = Counter(w["robot_id"] for w in controls)
    rejection = Counter()
    starts: dict = {}
    for failure in ledger:
        starts.setdefault(failure["robot_id"], []).append(failure["failure_time"])
    ordered_anchors = sorted(anchors, key=lambda r: r["end_time"])
    last_kept: dict = {}
    import math as _math
    for anchor in ordered_anchors:
        end, robot = anchor["end_time"], anchor["robot_id"]
        if end - last_kept.get(robot, -_math.inf) < 7 * DAY:
            rejection["greedy-spacing"] += 1
            continue
        if any(s >= end - 7 * DAY and s < end + 7 * DAY
               for s in starts.get(robot, [])):
            rejection["clearance"] += 1
            continue
        if E.window_intersects_reset(end - 7 * DAY, end, robot, wins):
            rejection["reset-span"] += 1
            continue
        last_kept[robot] = end
    assert sum(rejection.values()) + len(controls) == len(ordered_anchors)
    anchor_excluders = Counter()
    anchor_ids = {r["file_id"] for r in anchors}
    for row in rows:
        if row["file_id"] in anchor_ids:
            continue
        if "test-temporal" not in row["member_views"]:
            anchor_excluders["non-temporal-view"] += 1
        elif row["is_censored"]:
            anchor_excluders["censored"] += 1
        elif row["is_quarantined"]:
            anchor_excluders["quarantined"] += 1
        elif E._overlaps_maintenance(row, wins):
            anchor_excluders["maintenance-overlap"] += 1
    eval_days = set()
    for row in rows:
        if not E.eligible_operational_row(row, wins):
            continue
        eval_days.add((row["robot_id"], int(row["end_time"] // DAY)))
    prog_by_op = {}
    for row in rows:
        prog_by_op[(row["robot_id"], row["end_time"])] = row["program_id"]

    def failure_program(failure):
        """Program of the operation ending at the failure onset."""
        return prog_by_op.get((failure["robot_id"], failure["failure_time"]))

    progs: dict = {}
    for failure in ledger:
        if E.positive_window_intersects_reset(failure, wins):
            continue
        cands = E.pos_files(rows, failure, wins)
        if cands and failure["cohort"] in ("P", "W"):
            prog = failure_program(failure)
            if prog is not None:
                progs.setdefault(failure["cohort"], set()).add(prog)
    cutoff = manifest["calendar"]["cutoff_time"]
    healthy = [r for r in rows
               if r["file_label"] == "normal" and not r["is_quarantined"]
               and r["end_time"] <= cutoff]
    healthy_eligible = [r for r in healthy
                        if r["program_id"] != "program-03"
                        and r["robot_id"] != "robot-08"]
    dev_val_ids = set(manifest["splits"]["dev_val"])
    cal_rows = [r for r in rows
                if r["file_id"] in dev_val_ids
                and r["program_id"] != "program-03"
                and r["robot_id"] != "robot-08"]
    total_pos = pos_eval
    robots_pos = sum(1 for v in pos_eval_by_robot.values() if v > 0)
    robots_neg = sum(1 for v in ctrl_by_robot.values() if v > 0)
    checks = {
        "P_ge_13": pos_cat.get("P", 0) >= 13,
        "W_ge_13": pos_cat.get("W", 0) >= 13,
        "A_ge_10": pos_cat.get("A", 0) >= 10,
        "total_ge_38": total_pos >= 38,
        "negatives_ge_32": len(controls) >= 32,
        "robot_days_ge_188": len(eval_days) >= 188,
        "robots_pos_ge_6": robots_pos >= 6,
        "robots_neg_ge_6": robots_neg >= 6,
        "programs_ge_2": all(len(progs.get(c, set())) >= 2 for c in ("P", "W")),
    }
    max_pos_share = (max(pos_eval_by_robot.values()) / total_pos
                     if total_pos else 1.0)
    max_neg_share = (max(ctrl_by_robot.values()) / len(controls)
                     if controls else 1.0)
    prog_shares = {}
    for cohort in ("P", "W"):
        tot = pos_cat.get(cohort, 0)
        per_prog: Counter = Counter()
        for failure in ledger:
            if failure["cohort"] != cohort:
                continue
            if E.positive_window_intersects_reset(failure, wins):
                continue
            if not E.pos_files(rows, failure, wins):
                continue
            prog = failure_program(failure)
            per_prog[prog if prog is not None else "unknown"] += 1
        prog_shares[cohort] = (max(per_prog.values()) / tot if tot else 1.0)
    cohort_shares = {c: pos_cat.get(c, 0) / total_pos if total_pos else 0.0
                     for c in "PWA"}
    checks["robot_pos_le_35"] = max_pos_share <= 0.35
    checks["robot_neg_le_40"] = max_neg_share <= 0.40
    checks["program_le_60"] = all(v <= 0.60 for v in prog_shares.values())
    checks["cohort_mix_15_60"] = all(0.15 <= v <= 0.60
                                     for v in cohort_shares.values())
    p_events = [v for k, v in lead_support.items()
                if next(f["cohort"] for f in ledger
                        if f["failure_id"] == k) == "P"]
    w_events = [v for k, v in lead_support.items()
                if next(f["cohort"] for f in ledger
                        if f["failure_id"] == k) == "W"]
    def lead_ok(events):
        if not events:
            return False
        good = sum(1 for v in events if v["endpoints"] >= 3
                   and v["early_endpoint"] and v["clean_baseline"])
        return good / len(events) >= 0.80
    checks["lead_P_80"] = lead_ok(p_events)
    checks["lead_W_80"] = lead_ok(w_events)
    checks["fit_support_projection"] = (
        len(healthy_eligible) >= 200)
    checks["cal_support_projection"] = len(cal_rows) >= 40
    checks["eg1_pass"] = not errors
    const = {r["file_id"]: 0.5 for r in rows}
    time_scores = {
        r["file_id"]: max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) / (
            max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) + 30.0)
        for r in rows}
    eg3 = {}
    for name, scores in (("constant", const), ("observable_time", time_scores)):
        arm = prove_arm(rows, ledger, wins, scores, E)
        eg3[name] = {"computable": arm["computable"],
                     "event_auc": arm["event_auc"],
                     "positives_evaluable": arm["positives_evaluable"],
                     "negatives": arm["negatives"]}
    checks["eg3_computable"] = all(a["computable"] for a in eg3.values())
    structural = all(checks.values())
    return {
        "role": role, "seed": seed,
        "task12_errors": errors, "task12_pass": not errors,
        "positives_evaluable": pos_eval,
        "positives_by_cohort": dict(pos_cat),
        "positives_unevaluable": uneval,
        "negatives": len(controls),
        "negatives_by_robot": dict(ctrl_by_robot),
        "evaluated_robot_days": len(eval_days),
        "per_robot_evaluable_positives": dict(pos_eval_by_robot),
        "per_subtype": {s: {"evaluable": sub_eval.get(s, 0),
                            "total": sub_total.get(s, 0)}
                        for s in ("P1", "P2", "W1", "W2", "A1", "A2")},
        "lead_support": {"P_events": len(p_events), "W_events": len(w_events),
                         "P_ok": checks["lead_P_80"], "W_ok": checks["lead_W_80"]},
        "rejection_reasons": dict(rejection),
        "anchor_excluders": dict(anchor_excluders),
        "concentration": {"max_pos_share": max_pos_share,
                          "max_neg_share": max_neg_share,
                          "program_shares": prog_shares,
                          "cohort_shares": cohort_shares},
        "support_projection": {"healthy_eligible": len(healthy_eligible),
                               "cal_eligible_rows": len(cal_rows)},
        "promotion_checks": checks,
        "eg3": eg3,
        "design_state": ("DESIGN-PASS" if structural else "DESIGN-FAIL"),
    }


def main() -> int:
    results = []
    for role, seed in ROSTER:
        results.append(audit_history(role, seed))
        print(f"audited {role}: {results[-1]['design_state']} "
              f"errors={results[-1]['task12_errors']}", flush=True)
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
