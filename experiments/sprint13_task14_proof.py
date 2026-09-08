"""Task 14: score-independent metric computability proof — fixtures only.

Runs the frozen event-window pipeline on every materialized manifest with two
non-fitted, non-selected score arms: a constant arm (0.5) and an
observable-time arm (days-since-reset mapped into [0, 1)). Proves E1–E5 are
defined on real window structure, including lead/persistence/FAR semantics
and tie behavior. Companion threshold 0.6 is ARBITRARY and fixed (never
selected); outputs are computability evidence, never model selection or
scientific success. Structural reads only; no fitting, no selection.
"""

from __future__ import annotations

import json
DAY = 86400.0
HORIZON_S = 7.0 * DAY
# ARBITRARY fixed companion threshold (never selected, never a verdict).
# 0.3 (not 0.6: time scores saturate at 0.5) so recalled events exist and
# lead/persistence medians are exercised on real windows.
FIXED_THRESHOLD = 0.3

ROSTER = [
    ("H-VAL-DESIGN", 300), ("H-DEV-1", 301), ("H-DEV-2", 302),
    ("H-DEV-3", 303), ("H-FIT-1", 304), ("H-FIT-2", 305),
    ("H-FIT-3", 306), ("H-CAL-1", 307), ("H-CONF-1", 308),
    ("H-SEAL-1", 400), ("H-SEAL-2", 401), ("H-SEAL-3", 402),
    ("H-SEAL-4", 403),
]

BASE = "data/generated/sprint13"


def prove_history(role: str) -> dict:
    from synth import events as E

    manifest = json.load(open(f"{BASE}/{role}/manifest.json"))
    rows = manifest["files"]
    ledger = E.failure_ledger(manifest)
    wins = manifest["maintenance_windows"]

    const = {r["file_id"]: 0.5 for r in rows}
    time_scores = {
        r["file_id"]: max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) / (
            max(0.0, (r["end_time"] - r["last_reset_time"]) / DAY) + 30.0)
        for r in rows
    }
    anchors = E.anchor_rows(rows, wins)
    controls = E.select_control_windows(anchors, ledger)

    arms = {}
    for name, scores in (("constant", const), ("observable_time", time_scores)):
        pos, uneval = [], 0
        for failure in ledger:
            cands = E.pos_files(rows, failure, wins)
            if not cands:
                uneval += 1
                continue
            pos.append(E.window_score([c["file_id"] for c in cands], scores))
        neg = [E.window_score([m["file_id"] for m in w["members"]], scores)
               for w in controls]
        try:
            auc = E.roc_auc_tie_aware(pos, neg)
            auc_state = "defined"
        except E.UnavailableError:
            auc, auc_state = None, "UNAVAILABLE"
        recalled, leads, persists = 0, [], []
        flagged_by_robot: dict = {}
        for failure in ledger:
            cands = E.pos_files(rows, failure, wins)
            flagged = sorted(c["end_time"] for c in cands
                             if scores[c["file_id"]] >= FIXED_THRESHOLD)
            for c in cands:
                if scores[c["file_id"]] >= FIXED_THRESHOLD:
                    flagged_by_robot.setdefault(c["robot_id"], []).append(
                        c["end_time"])
            if flagged:
                recalled += 1
                leads.append(E.lead_days(failure["failure_time"], flagged))
                persists.append(len(flagged))
        eval_days = {(r["robot_id"], int(r["end_time"] // DAY)) for r in rows
                     if not any(s <= r["start_time"] < e
                                for s, e in wins.get(r["robot_id"], []))}
        false, far = E.false_alert_episodes(
            flagged_by_robot, ledger, float(len(eval_days)))
        arms[name] = {
            "event_auc": auc, "auc_state": auc_state,
            "positives_evaluable": len(pos), "positives_unevaluable": uneval,
            "negatives": len(neg),
            "recall_at_fixed_threshold": recalled / len(pos) if pos else None,
            "lead_median": sorted(leads)[len(leads) // 2] if leads else None,
            "lead_positive_fraction": (
                sum(1 for x in leads if x > 0.0) / len(leads) if leads else None),
            "persistence_median": (
                sorted(persists)[len(persists) // 2] if persists else None),
            "false_episodes": false, "far_per_robot_day": far,
        }
    return {"role": role, "arms": arms,
            "computable": all(a["auc_state"] == "defined"
                              for a in arms.values())}


def main() -> int:
    results = [prove_history(role) for role, _ in ROSTER]
    with open("/tmp/s13_task14.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    for result in results:
        states = {name: arm["auc_state"] for name, arm in result["arms"].items()}
        print(f"{result['role']}: computable={result['computable']} {states}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
