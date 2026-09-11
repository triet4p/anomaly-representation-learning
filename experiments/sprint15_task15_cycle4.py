"""Sprint 15 Task 15 cycle-4: one-shot frozen-probe execution (candidate 4).

Fit standardization + centroid on Fit verified-healthy files only;
operating threshold on Calibration verified-healthy files only; single
evaluation over all four Confirmation histories via
``probe15.evaluate_histories``. No threshold tuning, no pooling rescue, no
learned models. Writes
``artifacts/sprint-15/observable-confirmation-cycle-4.json`` and prints a
summary (recorded in ``artifacts/sprint-15/task-15-cycle-4.md``).

Sealed roots are never touched here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FIT = (("H-FIT-19", 1304), ("H-FIT-20", 1305), ("H-FIT-21", 1306))
CAL = ("H-CAL-7", 1307)
CONF = (("H-CONF-22", 1308), ("H-CONF-23", 1309), ("H-CONF-24", 1310),
        ("H-CONF-25", 1311))
BASE = Path("data/generated/sprint15-v4")
OUT = Path("artifacts/sprint-15/observable-confirmation-cycle-4.json")


def load_root(role: str, group: str):
    """Load samples plus manifest for one materialized root."""
    from synth.chronicle import load_chronological

    root = BASE / group / role
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"missing root (refusing to proceed): {root}")
    samples, _ = load_chronological(root)
    manifest = json.loads((root / "manifest.json").read_text())
    return samples, manifest


def healthy_features(samples, manifest):
    """Features of verified-healthy, non-reserve files (Fit/Cal inputs)."""
    from synth import balanced as B
    from synth import probe15 as P

    by_id = {s.file_id: s for s in samples}
    feats, ids = [], []
    for row in manifest["files"]:
        if row["file_label"] != "normal" or row["is_quarantined"]:
            continue
        if (row["program_id"] == B.PROGRAM_RESERVE
                or row["robot_id"] == B.ROBOT_RESERVE):
            continue
        sample = by_id[row["file_id"]]
        feats.append(P.extract_features(
            np.asarray(sample.x, dtype=np.float64),
            row["end_time"] - row["start_time"],
            row["end_time"] - row["last_reset_time"]))
        ids.append(row["file_id"])
    return np.array(feats), ids


def main() -> int:
    from synth import balanced as B
    from synth import probe15 as P

    fit_feats, fit_ids, fit_patches = [], [], 0
    fit_evaluated, fit_missing = [], []
    for role, seed in FIT:
        root = BASE / "FIT" / role
        if not (root / "manifest.json").is_file():
            fit_missing.append({"role": role, "seed": seed,
                                "reason": "no root: fail-fast quota "
                                          "infeasibility (see "
                                          "task-14-cycle-4)"})
            continue
        samples, manifest = load_root(role, "FIT")
        fit_evaluated.append(role)
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V4, f"tag:{role}"
        feats, ids = healthy_features(samples, manifest)
        fit_feats.append(feats)
        fit_ids.extend([f"{role}/{i}" for i in ids])
        idset = set(ids)
        fit_patches += sum(
            r["n_valid_patches"] for r in manifest["files"]
            if r["file_id"] in idset)
    fit_matrix = np.vstack(fit_feats)
    stats = P.standardize_fit(fit_matrix)
    centroid = P.fit_centroid(P.apply_standardization(fit_matrix, stats))
    cal_role, cal_seed = CAL
    cal_samples, cal_manifest = load_root(cal_role, "CALIBRATION")
    assert cal_manifest["seeds"]["health"] == cal_seed, "seed-match:cal"
    cal_feats, cal_ids = healthy_features(cal_samples, cal_manifest)
    cal_scores = P.score_files(P.apply_standardization(cal_feats, stats),
                               centroid)
    threshold = P.select_threshold(cal_scores)

    file_scores: dict[str, float] = {}
    rows_by_history, ledgers_by_history, wins_by_history = [], [], []
    seen: set[str] = set()
    evaluated, missing = [], []
    for role, seed in CONF:
        root = BASE / "CONFIRMATION" / role
        if not (root / "manifest.json").is_file():
            missing.append({"role": role, "seed": seed,
                            "reason": "no root: fail-fast quota infeasibility "
                                      "(see task-14-cycle-4)"})
            continue
        samples, manifest = load_root(role, "CONFIRMATION")
        evaluated.append(role)
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V4, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if row["file_id"] in seen:
                raise ValueError(f"file-id collision: {row['file_id']}")
            seen.add(row["file_id"])
            sample = by_id[row["file_id"]]
            feat = P.extract_features(
                np.asarray(sample.x, dtype=np.float64),
                row["end_time"] - row["start_time"],
                row["end_time"] - row["last_reset_time"])
            file_scores[row["file_id"]] = float(
                P.score_files(P.apply_standardization(feat[None, :], stats),
                              centroid)[0])
        from synth import events as E
        rows_by_history.append(manifest["files"])
        ledgers_by_history.append(E.failure_ledger(manifest))
        wins_by_history.append(manifest["maintenance_windows"])

    result = P.evaluate_histories(file_scores, rows_by_history,
                                  ledgers_by_history, wins_by_history,
                                  threshold)
    record = {
        "probe": P.PROBE_ID,
        "protocol": B.S15_PROTOCOL_V4,
        "fit": {"roots": [r for r, _ in FIT],
                "fit_evaluated": fit_evaluated,
                "fit_missing": fit_missing,
                "fit_coverage": f"{len(fit_evaluated)}/3",
                "n_healthy_files": int(len(fit_ids)),
                "n_healthy_patches": int(fit_patches)},
        "calibration": {"root": cal_role,
                        "n_healthy_rows": int(len(cal_ids)),
                        "threshold": threshold},
        "confirmation": [r for r, _ in CONF],
        "confirmation_evaluated": evaluated,
        "confirmation_missing": missing,
        "coverage": f"{len(evaluated)}/4",
        "eg4_satisfiable": len(evaluated) == 4 and len(fit_evaluated) == 3,
        "metrics": result,
    }
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps({
        "threshold": threshold,
        "fit_coverage": f"{len(fit_evaluated)}/3",
        "fit_missing": [m["role"] for m in fit_missing],
        "coverage": f"{len(evaluated)}/4",
        "missing": [m["role"] for m in missing],
        "macro_auc_pw": result["macro_auc_pw"],
        "macro_auc_p": result["macro_auc_p"],
        "macro_auc_w": result["macro_auc_w"],
        "lcb_pw": result["lcb_pw"],
        "directional": result["directional_histories"],
        "per_history": [
            {k: h[k] for k in ("auc_pw", "auc_p", "auc_w", "auc_a",
                               "recall_p", "recall_w", "lead_p_median",
                               "lead_w_median", "false_episodes", "far",
                               "robot_days")}
            for h in result["per_history"]],
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
