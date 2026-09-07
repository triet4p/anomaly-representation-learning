"""Sprint 12 Task 12 — causal warning vs operating-history baselines.

Fit on DEV histories ONLY (frozen thereafter); evaluate the FROZEN warning
comparison per sealed TEMPORAL history (approved Task 12 evaluation — temporal
views only, never the static detection task, never sealed static lists, never
learned geometry, which has no upstream standing).

Arms (boring, causal, observable-only):
- (a) constant-risk: global dev 7d event rate (AUROC 0.5 by construction);
- (b) operating-history: logistic on [usage_h, time_since_maint_d];
- (c) trend/persistence: logistic on [usage_h, tsm_d, trail_max7d,
  trail_frac90_7d, persist_count] from trailing robot files (strictly prior,
  causal windows).

All features use only raw signal + past schedule events; future_targets enter
ONLY as supervised fit labels on dev / eval labels on sealed temporal (never
features — covered by a shuffle-invariance test). Maintenance-overlapping and
censored files are excluded from ranking metrics with counts recorded.
G-rank (frozen): per sealed history, 7d AUROC >= 0.70 AND >= constant + 0.10;
median/IQR across the four; 1-day reported only, no gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.handcrafted import Standardizer  # noqa: E402

DAY = 86400.0
TRAIL_S = 7 * DAY
TSM_CAP_D = 90.0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    """Live Git-derived execution provenance (fails loudly, never a CLI claim)."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no",
         "--", "experiments", "src", "tests"],
        capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "refusing to run with tracked modifications under "
            f"experiments/src/tests:\n{dirty}"
        )
    return out


def roc_auc_or_nan(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard ROC AUROC (repo convention: sklearn), nan on single-class."""
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if labels.min() == labels.max():
        return float("nan")
def operating_points(scores: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, float]]:
    """Tie-safe ROC operating points: (threshold, fpr, tpr), starting at (+inf, 0, 0).

    Thresholds sweep distinct score values descending; tied blocks move as one
    unit, so constant-score arms yield only (0,0) and (1,1) — never fictitious
    intermediate recall. Deterministic (no row-order dependence).
    """
    sc = np.asarray(scores, dtype=np.float64).ravel()
    lb = np.asarray(labels, dtype=np.float64).ravel()
    n_pos = int(lb.sum())
    n_neg = int(lb.size - n_pos)
    points = [(float("inf"), 0.0, 0.0)]
    if n_pos == 0 or n_neg == 0:
        return points
    for thr in sorted(set(sc.tolist()), reverse=True):
        flagged = sc >= thr
        fpr = float((lb[flagged] == 0).sum() / n_neg)
        tpr = float((lb[flagged] == 1).sum() / n_pos)
        points.append((float(thr), fpr, tpr))
    return points


def recall_at_fpr(points: list[tuple[float, float, float]], fp: float) -> float:
    """Max TPR among operating points with FPR <= fp (0.0 if only (0,0) qualifies)."""
    best = 0.0
    for _, fpr, tpr in points:
        if fpr <= fp and tpr > best:
            best = tpr
    return float(best)


def threshold_for_fpr(points: list[tuple[float, float, float]], fp: float) -> float:
    """Highest threshold whose operating point keeps FPR <= fp (+inf default)."""
    best_thr, best_tpr = float("inf"), -1.0
    for thr, fpr, tpr in points:
        if fpr <= fp and (tpr > best_tpr or (tpr == best_tpr and thr > best_thr)):
            best_thr, best_tpr = thr, tpr
    return float(best_thr)


def group_alert_episodes(sorted_ends: list[float], max_gap_s: float) -> list[list[float]]:
    """Maximal runs of flagged file end-times with consecutive gaps <= max_gap_s."""
    episodes: list[list[float]] = []
    for t in sorted_ends:
        if episodes and t - episodes[-1][-1] <= max_gap_s:
            episodes[-1].append(t)
        else:
            episodes.append([t])
    return episodes


def event_recall_lead(
    alert_ends_by_robot: dict[str, list[float]],
    failures: list[tuple[str, float]],
    window_s: float,
) -> dict:
    """Event recall + lead times: an event is recalled if >=1 alert ends in [T-w, T]."""
    recalled, leads = 0, []
    for robot, t_fail in failures:
        hits = [t for t in alert_ends_by_robot.get(robot, [])
                if t_fail - window_s <= t <= t_fail]
        if hits:
            recalled += 1
            leads.append((t_fail - max(hits)) / 86400.0)
    return {"n_events": len(failures), "n_recalled": recalled,
            "recall": recalled / len(failures) if failures else float("nan"),
            "lead_time_d_median": float(np.median(leads)) if leads else float("nan")}


def _apply_std(saved: dict, rows: np.ndarray) -> np.ndarray:
    """Apply a FROZEN fitted center/scale mapping (never refit on eval rows)."""
    center = np.asarray(saved["center"], dtype=np.float64)
    scale = np.asarray(saved["scale"], dtype=np.float64)
    out = (np.asarray(rows, dtype=np.float64) - center[None, :]) / scale[None, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def describe(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def file_score(x: np.ndarray) -> float:
    """Zero-fit causal observable: max centered patch amplitude (W=32/S=16)."""
    med = np.median(x, axis=1, keepdims=True)
    dev = np.abs(x - med)
    _, t = dev.shape
    w, s = 32, 16
    best = 0.0
    for start in range(0, max(1, t - w + 1), s):
        best = max(best, float(dev[:, start:start + w].mean()))
    if t >= w:
        best = max(best, float(dev[:, -w:].mean()))
    return best


def build_history(root: str):
    """Load one history: samples + manifest + per-robot time index."""
    samples, manifest = load_chronological(root)
    files = {row["file_id"]: row for row in manifest["files"]}
    by_robot: dict[str, list] = {}
    for s in samples:
        by_robot.setdefault(str(files[s.file_id]["robot_id"]), []).append(s)
    for evs in by_robot.values():
        evs.sort(key=lambda s: float(files[s.file_id]["end_time"]))
    maint: dict[str, list] = {}
    for e in manifest.get("episodes", []):
        if str(e.get("kind")) == "maintenance":
            maint.setdefault(str(e["robot_id"]), []).append(
                (float(e["start_time"]), float(e.get("end_time") or 0.0)))
    sched: dict[str, list] = {}
    for e in manifest.get("schedule", []):
        sched.setdefault(str(e["robot_id"]), []).append(e)
    return {"samples": samples, "manifest": manifest, "files": files,
            "by_robot": by_robot, "maint": maint, "sched": sched,
            "episodes": manifest.get("episodes", [])}


def in_maint(maint: dict, robot: str, start: float, end: float) -> bool:
    """Whether [start, end) overlaps a maintenance window (boundary-safe)."""
    return any(s < end and start < e for s, e in maint.get(robot, []))


def featurize(h: dict, score_of: dict, q90: float | None = None):
    """Causal per-file features. future_targets NEVER consulted here."""
    files, out = h["files"], []
    for s in h["samples"]:
        fid = s.file_id
        robot = str(files[fid]["robot_id"])
        start = float(files[fid]["start_time"])
        end = float(files[fid]["end_time"])
        usage = sum(float(e["duration"]) for e in h["sched"].get(robot, [])
                    if float(e["end_time"]) <= end) / 3600.0
        past_maint: list = []
        mends = [e for _, e in h["maint"].get(robot, []) if e <= start]
        tsm = min(TSM_CAP_D, (start - max(mends)) / DAY) if mends else TSM_CAP_D
        prior = [p for p in h["by_robot"][robot]
                 if float(files[p.file_id]["end_time"]) < end
                 and float(files[p.file_id]["end_time"]) >= end - TRAIL_S]
        tscores = [score_of[p.file_id] for p in prior]
        persist = 0
        for p in sorted(prior, key=lambda q: float(files[q.file_id]["end_time"]), reverse=True):
            if q90 is not None and score_of[p.file_id] >= q90:
                persist += 1
            else:
                break
        out.append({
            "file_id": fid, "robot": robot,
            "usage_h": usage, "tsm_d": tsm,
            "trail_max": float(max(tscores)) if tscores else 0.0,
            "trail_frac90": (float(np.mean([t >= q90 for t in tscores]))
                             if (tscores and q90 is not None) else 0.0),
            "trail_n": len(prior), "persist": persist,
            "in_maint": in_maint(h["maint"], robot, start, end),
        })
        _ = past_maint
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-roots", nargs=4, required=True)
    ap.add_argument("--seal-roots", nargs=4, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = [build_history(r) for r in args.dev_roots]
    dev_scores = {}
    for h in dev:
        for s in h["samples"]:
            dev_scores[s.file_id] = file_score(s.x)
    q90 = float(np.quantile(list(dev_scores.values()), 0.90))

    # supervised fit set: dev files with known outcomes (censored excluded).
    # Cold-start holdout: program-03 NEVER enters the fit. Maintenance
    # operations are excluded for train/eval symmetry. Post-cutoff precursor
    # files stay IN (they carry the positive outcomes a warning model needs;
    # restricting to healthy-only would remove all positives — see task notes).
    from sklearn.linear_model import LogisticRegression
    Xb, Xc, y = [], [], []
    n_fit_p3_excluded = n_fit_maint_excluded = 0
    for h in dev:
        feats = {f["file_id"]: f for f in featurize(h, dev_scores, q90)}
        for s in h["samples"]:
            ft = s.future_targets
            if ft is None or bool(ft.is_censored):
                continue
            if str(h["files"][s.file_id]["program_id"]) == "program-03":
                n_fit_p3_excluded += 1
                continue
            f = feats[s.file_id]
            if f["in_maint"]:
                n_fit_maint_excluded += 1
                continue
            Xb.append([f["usage_h"], f["tsm_d"]])
            Xc.append([f["usage_h"], f["tsm_d"], f["trail_max"],
                       f["trail_frac90"], f["persist"]])
            y.append(1.0 if bool(ft.failure_within_7d) else 0.0)
    Xb_raw = np.array(Xb)
    Xc_raw = np.array(Xc)
    y = np.array(y)
    const_rate = float(y.mean())
    std_b = Standardizer.fit(Xb_raw)
    std_c = Standardizer.fit(Xc_raw)
    Xb, Xc = std_b.apply(Xb_raw), std_c.apply(Xc_raw)
    clf_b = LogisticRegression().fit(Xb, y)
    clf_c = LogisticRegression().fit(Xc, y)
    frozen = {
        "const_rate": const_rate,
        "q90": q90,
        "standardizer_b": std_b.to_dict(),
        "standardizer_c": std_c.to_dict(),
        "coef_b": clf_b.coef_.ravel().tolist(),
        "intercept_b": float(clf_b.intercept_[0]),
        "coef_c": clf_c.coef_.ravel().tolist(),
        "intercept_c": float(clf_c.intercept_[0]),
        "n_fit": int(y.size),
        "n_pos_fit": int(y.sum()),
        "n_fit_program03_excluded": n_fit_p3_excluded,
        "n_fit_maint_excluded": n_fit_maint_excluded,
    }
    # DEV operating thresholds at FPR <= 0.10 per arm (tie-safe, frozen).
    dev_points = {
        "b": operating_points(Xb @ np.array(frozen["coef_b"]) + frozen["intercept_b"], y),
        "c": operating_points(Xc @ np.array(frozen["coef_c"]) + frozen["intercept_c"], y),
    }
    frozen["dev_threshold_fpr010"] = {
        arm: threshold_for_fpr(pts, 0.10) for arm, pts in dev_points.items()
    }

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    histories = []
    for root in args.seal_roots:
        h = build_history(root)
        test_ids = set(h["manifest"]["splits"]["test_temporal"])
        scores = {s.file_id: file_score(s.x) for s in h["samples"]}
        feats = {f["file_id"]: f for f in featurize(h, scores, q90)}
        rows = []
        n_censored = n_maint_excl = n_no_target = 0
        for s in h["samples"]:
            if s.file_id not in test_ids:
                continue
            ft = s.future_targets
            if ft is None:
                n_no_target += 1
                continue
            if bool(ft.is_censored):
                n_censored += 1
                continue
            f = feats[s.file_id]
            if f["in_maint"]:
                n_maint_excl += 1
                continue
            xb_raw = np.array([f["usage_h"], f["tsm_d"]])
            xc_raw = np.array(
                [f["usage_h"], f["tsm_d"], f["trail_max"], f["trail_frac90"], f["persist"]])
            xb = float(np.array(frozen["coef_b"]) @ _apply_std(
                frozen["standardizer_b"], xb_raw) + frozen["intercept_b"])
            xc = float(np.array(frozen["coef_c"]) @ _apply_std(
                frozen["standardizer_c"], xc_raw) + frozen["intercept_c"])
            rows.append({
                "file_id": s.file_id, "robot": f["robot"],
                "end": float(h["files"][s.file_id]["end_time"]),
                "program": str(h["files"][s.file_id]["program_id"]),
                "y7": bool(ft.failure_within_7d), "y1": bool(ft.failure_within_1d),
                "a": frozen["const_rate"], "b": float(sigmoid(xb)), "c": float(sigmoid(xc)),
            })
        yy7 = np.array([r["y7"] for r in rows], dtype=float)
        out_h: dict = {"root": root, "n_eval": len(rows),
                       "n_censored_excluded": n_censored,
                       "n_maint_excluded": n_maint_excl,
                       "n_no_target": n_no_target,
                       "n_pos_7d": int(yy7.sum()),
                       "n_pos_1d": int(sum(1 for r in rows if r["y1"]))}
        for arm in ("a", "b", "c"):
            sc = np.array([r[arm] for r in rows])
            auc7 = roc_auc_or_nan(sc, yy7)
            yy1 = np.array([r["y1"] for r in rows], dtype=float)
            pts = operating_points(sc, yy7)
            out_h[arm] = {"auroc_7d": auc7, "auroc_1d": roc_auc_or_nan(sc, yy1)}
            for fp in (0.05, 0.10):
                out_h[arm][f"recall_at_fpr_{fp:.2f}"] = recall_at_fpr(pts, fp)
            p3 = [r for r in rows if r["program"] == "program-03"]
            if p3:
                out_h[arm]["auroc_7d_program03"] = roc_auc_or_nan(
                    np.array([r[arm] for r in p3]),
                    np.array([r["y7"] for r in p3], dtype=float))
            # Event metrics at the frozen DEV FPR<=0.10 threshold (arm b/c only;
            # arm a is constant and alerts either everywhere or nowhere).
            if arm in frozen["dev_threshold_fpr010"]:
                thr = frozen["dev_threshold_fpr010"][arm]
                flagged_by_robot: dict[str, list[float]] = {}
                for r in rows:
                    if r[arm] >= thr:
                        flagged_by_robot.setdefault(r["robot"], []).append(r["end"])
                alert_episodes = {rb: group_alert_episodes(sorted(e), 2 * DAY)
                                  for rb, e in flagged_by_robot.items()}
                failures = [(str(e["robot_id"]), float(e["start_time"]))
                            for e in h["episodes"] if str(e.get("kind")) == "failure"]
                flat_alerts = {rb: sorted(t for ep in eps for t in ep)
                               for rb, eps in alert_episodes.items()}
                out_h[arm]["event_7d"] = event_recall_lead(flat_alerts, failures, 7 * DAY)
                n_ep = sum(len(eps) for eps in alert_episodes.values())
                fails_by_robot: dict[str, list[float]] = {}
                for r2, t2 in failures:
                    fails_by_robot.setdefault(r2, []).append(t2)
                n_false = 0
                for rb, eps in alert_episodes.items():
                    for ep in eps:
                        if not any(t2 - 7 * DAY <= t <= t2
                                   for t in ep for t2 in fails_by_robot.get(rb, [])):
                            n_false += 1
                robot_days = 0.0
                for rb in {r["robot"] for r in rows}:
                    ends = [r["end"] for r in rows if r["robot"] == rb]
                    robot_days += (max(ends) - min(ends)) / DAY if len(ends) > 1 else 0.0
                out_h[arm]["n_alert_episodes"] = n_ep
                out_h[arm]["false_alert_episodes_per_robot_day"] = (
                    n_false / robot_days if robot_days > 0 else float("nan"))
        auc_c = out_h["c"]["auroc_7d"]
        auc_a = out_h["a"]["auroc_7d"]
        out_h["g_rank_pass"] = bool(auc_c >= 0.70 and auc_c >= auc_a + 0.10)
        histories.append(out_h)

    auc7 = np.array([h["c"]["auroc_7d"] for h in histories])
    result = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "dev_roots": list(args.dev_roots),
            "seal_roots": list(args.seal_roots),
            "frozen": frozen,
            "note": "sealed TEMPORAL views only (approved Task 12 evaluation); "
                    "never sealed static lists, never learned geometry",
        },
        "histories": histories,
        "g_rank": {
            "per_history_pass": [h["g_rank_pass"] for h in histories],
            "auroc_7d_median_iqr": [float(np.median(auc7)),
                                    float(np.percentile(auc7, 25)),
                                    float(np.percentile(auc7, 75))],
            "pass": bool(all(h["g_rank_pass"] for h in histories)),
        },
    }
    (out_dir / "task12_diag.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task12_diag.json'} g_rank_pass={result['g_rank']['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
