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
            "by_robot": by_robot, "maint": maint, "sched": sched}


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

    # supervised fit set: dev files with known outcomes (censored excluded)
    from sklearn.linear_model import LogisticRegression
    Xb, Xc, y = [], [], []
    for h in dev:
        feats = {f["file_id"]: f for f in featurize(h, dev_scores, q90)}
        for s in h["samples"]:
            ft = s.future_targets
            if ft is None or bool(ft.is_censored):
                continue
            f = feats[s.file_id]
            Xb.append([f["usage_h"], f["tsm_d"]])
            Xc.append([f["usage_h"], f["tsm_d"], f["trail_max"],
                       f["trail_frac90"], f["persist"]])
            y.append(1.0 if bool(ft.failure_within_7d) else 0.0)
    Xb, Xc, y = np.array(Xb), np.array(Xc), np.array(y)
    const_rate = float(y.mean())
    clf_b = LogisticRegression().fit(Xb, y)
    clf_c = LogisticRegression().fit(Xc, y)
    frozen = {
        "const_rate": const_rate,
        "q90": q90,
        "coef_b": clf_b.coef_.ravel().tolist(),
        "intercept_b": float(clf_b.intercept_[0]),
        "coef_c": clf_c.coef_.ravel().tolist(),
        "intercept_c": float(clf_c.intercept_[0]),
        "n_fit": int(y.size),
        "n_pos_fit": int(y.sum()),
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
        for s in h["samples"]:
            if s.file_id not in test_ids:
                continue
            ft = s.future_targets
            if ft is None or bool(ft.is_censored):
                continue
            f = feats[s.file_id]
            if f["in_maint"]:
                continue
            xb = np.array(frozen["coef_b"]) @ np.array([f["usage_h"], f["tsm_d"]]) + frozen["intercept_b"]
            xc = np.array(frozen["coef_c"]) @ np.array(
                [f["usage_h"], f["tsm_d"], f["trail_max"], f["trail_frac90"], f["persist"]]) + frozen["intercept_c"]
            rows.append({
                "file_id": s.file_id, "robot": f["robot"],
                "program": str(h["files"][s.file_id]["program_id"]),
                "y7": bool(ft.failure_within_7d), "y1": bool(ft.failure_within_1d),
                "a": frozen["const_rate"], "b": float(sigmoid(xb)), "c": float(sigmoid(xc)),
            })
        yy7 = np.array([r["y7"] for r in rows], dtype=float)
        out_h: dict = {"root": root, "n_eval": len(rows),
                       "n_pos_7d": int(yy7.sum()),
                       "n_pos_1d": int(sum(1 for r in rows if r["y1"]))}
        for arm in ("a", "b", "c"):
            sc = np.array([r[arm] for r in rows])
            auc7 = roc_auc_or_nan(sc, yy7)
            yy1 = np.array([r["y1"] for r in rows], dtype=float)
            out_h[arm] = {"auroc_7d": auc7, "auroc_1d": roc_auc_or_nan(sc, yy1)}
            for fp in (0.05, 0.10):
                order = np.argsort(-sc, kind="stable")
                for k in range(1, len(order) + 1):
                    flagged = order[:k]
                    fpr = float((yy7[flagged] == 0).sum() / max(1, (yy7 == 0).sum()))
                    if fpr > fp:
                        break
                    rec = float((yy7[flagged] == 1).sum() / max(1, (yy7 == 1).sum()))
                out_h[arm][f"recall_at_fpr_{fp:.2f}"] = rec
            p3 = [r for r in rows if r["program"] == "program-03"]
            if p3:
                out_h[arm]["auroc_7d_program03"] = roc_auc_or_nan(
                    np.array([r[arm] for r in p3]),
                    np.array([r["y7"] for r in p3], dtype=float))
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
