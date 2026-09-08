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
from synth.chronicle import load_chronological  # noqa: E402

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
    return float(roc_auc_score(labels, scores))


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


def select_control_windows(
    file_ends: list[float],
    failures: list[float],
    window_s: float,
) -> list[tuple[float, float]]:
    """Deterministic non-overlapping control windows of length window_s.

    Candidates anchor at each file end e (ascending): window [e-window_s, e]
    is kept iff no failure start lies in [e-window_s, e+window_s) (neither a
    pre-failure horizon nor post-failure aftermath) and it does not overlap
    an already kept window ([kept_end-window_s, kept_end)). Pure function of
    sorted inputs: identical inputs always yield identical windows.
    """
    kept: list[tuple[float, float]] = []
    for e in sorted(file_ends):
        if any(e - window_s <= t < e + window_s for t in failures):
            continue
        if kept and e - kept[-1][1] < window_s:
            continue
        kept.append((e - window_s, e))
    return kept


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
        recommissioned = max([e for _, e in h["maint"].get(robot, []) if e <= start],
                             default=0.0)
        usage = sum(float(e["duration"]) for e in h["sched"].get(robot, [])
                    if float(e["end_time"]) <= end
                    and float(e["start_time"]) >= recommissioned) / 3600.0
        mends = [e for _, e in h["maint"].get(robot, []) if e <= start]
        tsm = min(TSM_CAP_D, (start - max(mends)) / DAY) if mends else TSM_CAP_D
        prior = [p for p in h["by_robot"][robot]
                 if float(files[p.file_id]["end_time"]) < end
                 and float(files[p.file_id]["end_time"]) >= end - TRAIL_S
                 and float(files[p.file_id]["end_time"]) > recommissioned]
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
    fit_seeds = sorted(int(h["manifest"]["seeds"]["temporal"]) for h in dev[:3])
    val_seed = int(dev[3]["manifest"]["seeds"]["temporal"])
    assert fit_seeds == [100, 101, 102], fit_seeds
    assert val_seed == 103, val_seed
    fit_hists, val_hists = dev[:3], [dev[3]]

    def eligible_rows(histories):
        """Eligible temporal rows: test_temporal view, known uncensored outcome."""
        rows = []
        for h in histories:
            temporal = set(h["manifest"]["splits"]["test_temporal"])
            for s in h["samples"]:
                if s.file_id not in temporal:
                    continue
                ft = s.future_targets
                if ft is None or bool(ft.is_censored):
                    continue
                rows.append((h, s))
        return rows

    # q90 and scores: file scores need no fitting; q90 threshold value comes
    # from RISK-FIT eligible rows only.
    dev_scores = {}
    for h in dev:
        for s in h["samples"]:
            dev_scores[s.file_id] = file_score(s.x)
    fit_pool = eligible_rows(fit_hists)
    q90 = float(np.quantile([dev_scores[s.file_id] for _, s in fit_pool], 0.90))

    # supervised fit set: RISK-FIT eligible temporal rows with known outcomes.
    # Cold-start holdout: program-03 NEVER enters the fit. Maintenance
    # operations are excluded for train/eval symmetry. Post-cutoff precursor
    # rows stay IN (they carry the positive outcomes a warning model needs;
    # healthy-only restriction would remove all positives — rejected in v3).
    from sklearn.linear_model import LogisticRegression
    Xb, Xc, y = [], [], []
    n_fit_nontemporal = n_fit_p3_excluded = n_fit_maint_excluded = 0
    fit_feats = {}
    for h in fit_hists:
        for f in featurize(h, dev_scores, q90):
            fit_feats[f["file_id"]] = f
    for h, s in fit_pool:
        if str(h["files"][s.file_id]["program_id"]) == "program-03":
            n_fit_p3_excluded += 1
            continue
        f = fit_feats[s.file_id]
        if f["in_maint"]:
            n_fit_maint_excluded += 1
            continue
        ft = s.future_targets
        Xb.append([f["usage_h"], f["tsm_d"]])
        Xc.append([f["usage_h"], f["tsm_d"], f["trail_max"],
                   f["trail_frac90"], f["persist"]])
        y.append(1.0 if bool(ft.failure_within_7d) else 0.0)
    n_fit_nontemporal = sum(
        1 for h in fit_hists for s in h["samples"]
        if s.file_id not in set(h["manifest"]["splits"]["test_temporal"]))
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
        "protocol": "sprint12-protocol-v3",
        "risk_fit_seeds": [100, 101, 102],
        "risk_val_seed": 103,
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
        "n_fit_nontemporal_excluded": int(n_fit_nontemporal),
        "n_fit_program03_excluded": n_fit_p3_excluded,
        "n_fit_maint_excluded": n_fit_maint_excluded,
    }
    # RISK-VAL operating thresholds at FPR <= 0.10 per arm (tie-safe, frozen).
    # VAL rows use the same eligibility (temporal, known outcome, no program-03,
    # no maintenance); scored with the FROZEN standardizer + model.
    val_scores_b, val_scores_c, val_y = [], [], []
    n_val_excluded = 0
    val_feats: dict[str, dict] = {}
    for h in val_hists:
        for f in featurize(h, dev_scores, q90):
            val_feats[h["manifest"]["seeds"]["temporal"], f["file_id"]] = f
    for h, s in eligible_rows(val_hists):
        f = val_feats[(h["manifest"]["seeds"]["temporal"], s.file_id)]
        if str(h["files"][s.file_id]["program_id"]) == "program-03":
            n_val_excluded += 1
            continue
        ft = s.future_targets
        val_scores_b.append(float(
            np.array(frozen["coef_b"]) @ _apply_std(
                frozen["standardizer_b"],
                np.array([[f["usage_h"], f["tsm_d"]]]))[0]
            + frozen["intercept_b"]))
        val_scores_c.append(float(
            np.array(frozen["coef_c"]) @ _apply_std(
                frozen["standardizer_c"],
                np.array([[f["usage_h"], f["tsm_d"], f["trail_max"],
                           f["trail_frac90"], f["persist"]]]))[0]
            + frozen["intercept_c"]))
        val_y.append(1.0 if bool(ft.failure_within_7d) else 0.0)
    val_points = {
        "b": operating_points(np.array(val_scores_b), np.array(val_y)),
        "c": operating_points(np.array(val_scores_c), np.array(val_y)),
    }
    frozen["val_threshold_fpr010"] = {
        arm: threshold_for_fpr(pts, 0.10) for arm, pts in val_points.items()
    }
    frozen["n_val"] = len(val_y)
    frozen["n_val_pos"] = int(np.sum(val_y))
    frozen["n_val_excluded"] = n_val_excluded

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
                frozen["standardizer_b"], xb_raw.reshape(1, -1))[0]
                + frozen["intercept_b"])
            xc = float(np.array(frozen["coef_c"]) @ _apply_std(
                frozen["standardizer_c"], xc_raw.reshape(1, -1))[0]
                + frozen["intercept_c"])
            rows.append({
                "file_id": s.file_id, "robot": f["robot"],
                "end": float(h["files"][s.file_id]["end_time"]),
                "program": str(h["files"][s.file_id]["program_id"]),
                "quarantined": bool(h["files"][s.file_id]["is_quarantined"]),
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
        failures = [(str(e["robot_id"]), float(e["start_time"]))
                    for e in h["episodes"] if str(e.get("kind")) == "failure"]
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
            # Event-level positives: one score per failure = max file score
            # among eligible files of the same robot ending in [T-7d, T].
            pos_scores, n_pos_unevaluable = [], 0
            for robot, t_fail in failures:
                in_window = [r[arm] for r in rows
                             if r["robot"] == robot
                             and t_fail - 7 * DAY <= r["end"] <= t_fail]
                if in_window:
                    pos_scores.append(float(max(in_window)))
                else:
                    n_pos_unevaluable += 1
            # Event-level negatives: deterministic same-robot control windows.
            neg_scores, n_neg_windows = [], 0
            for rb in sorted({r["robot"] for r in rows}):
                rb_ends = sorted(r["end"] for r in rows
                                 if r["robot"] == rb and not r["quarantined"])
                rb_fails = sorted(t for r2, t in failures if r2 == rb)
                by_end = {r["end"]: r[arm] for r in rows if r["robot"] == rb}
                for w0, w1 in select_control_windows(rb_ends, rb_fails, 7 * DAY):
                    member = [by_end[e] for e in rb_ends if w0 <= e <= w1]
                    if member:
                        neg_scores.append(float(max(member)))
                        n_neg_windows += 1
            out_h[arm]["event_n_pos"] = len(pos_scores)
            out_h[arm]["event_n_pos_unevaluable"] = n_pos_unevaluable
            out_h[arm]["event_n_neg"] = len(neg_scores)
            if len(pos_scores) >= 2 and len(neg_scores) >= 6:
                labels = np.array([1.0] * len(pos_scores) + [0.0] * len(neg_scores))
                out_h[arm]["event_auroc_7d"] = roc_auc_or_nan(
                    np.array(pos_scores + neg_scores), labels)
                out_h[arm]["event_status"] = "scored"
            else:
                out_h[arm]["event_auroc_7d"] = float("nan")
                out_h[arm]["event_status"] = "UNAVAILABLE"
            # Thresholded event companions at the frozen RISK-VAL threshold
            # (arms b/c only; arm a is constant and alerts everywhere/nowhere).
            if arm in frozen["val_threshold_fpr010"]:
                thr = frozen["val_threshold_fpr010"][arm]
                flagged_by_robot: dict[str, list[float]] = {}
                for r in rows:
                    if r[arm] >= thr:
                        flagged_by_robot.setdefault(r["robot"], []).append(r["end"])
                alert_episodes = {rb: group_alert_episodes(sorted(e), 2 * DAY)
                                  for rb, e in flagged_by_robot.items()}
                flat_alerts = {rb: sorted(t for ep in eps for t in ep)
                               for rb, eps in alert_episodes.items()}
                out_h[arm]["event_7d"] = event_recall_lead(flat_alerts, failures, 7 * DAY)
                persist_counts = []
                for rb, eps in alert_episodes.items():
                    for ep in eps:
                        if any(t2 - 7 * DAY <= t <= t2
                               for t in ep
                               for _, t2 in [x for x in failures if x[0] == rb]):
                            persist_counts.append(len(ep))
                out_h[arm]["persistence_median"] = (
                    float(np.median(persist_counts)) if persist_counts else float("nan"))
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
        auc_c = out_h["c"].get("event_auroc_7d", float("nan"))
        auc_a = out_h["a"].get("event_auroc_7d", float("nan"))
        status_c = out_h["c"].get("event_status")
        out_h["g_rank_pass"] = bool(
            status_c == "scored"
            and np.isfinite(auc_c) and np.isfinite(auc_a)
            and auc_c >= 0.70 and auc_c >= auc_a + 0.10)
        out_h["g_rank_status"] = (
            "PASS" if out_h["g_rank_pass"]
            else ("UNAVAILABLE" if status_c == "UNAVAILABLE" else "FAIL"))
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
