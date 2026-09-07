"""Sprint 12 Task 11 — observable precursor diagnostics by failure category.

DEV histories only (never sealed). For each failure episode, post-hoc
simulator state (episodes manifest) assigns ONE category:

- progressive: a degradation episode on the same robot overlapping
  [fail_start − 14d, fail_start] with duration ≥ 3d;
- abrupt: no degradation episode ending within 14d before fail_start
  (deliberately sudden failures stay in the overall ledger here);
- weak: anything in between.

Observable precursor visibility uses ONE zero-fit causal file score (max
centered patch amplitude; file end_time ≤ window end, strictly causal).
Windows: files from the same robot ending in [fail−7d, fail) and [fail−1d,
fail); maintenance-overlapping files excluded (recorded); quarantine status
recorded but NOT exclusionary (precursors are quarantined by design).
Background: same-robot files with no failure within ±14d, non-maintenance.
Censored files (unknown forward outcome) are excluded from background counts
and reported. Labels/health/episodes enter ONLY post-hoc stratification —
never any input (the score needs no input but raw signal).

Bounded output: task11_diag.json. No model, no thresholds, no sealed reads.
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
LOOKBACK_S = 14 * DAY
PROG_MIN_DUR_S = 3 * DAY
WINDOWS = {"w1d": 1 * DAY, "w7d": 7 * DAY}


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
    c, t = dev.shape
    w, s = 32, 16
    best = 0.0
    for start in range(0, max(1, t - w + 1), s):
        best = max(best, float(dev[:, start:start + w].mean()))
    tail = dev[:, (max(0, t - w)):].mean() if t >= w else best
    return float(max(best, float(tail)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-roots", nargs=4, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_failure: list[dict] = []
    for root in args.dev_roots:
        samples, manifest = load_chronological(root)
        by_id = {s.file_id: s for s in samples}
        files = {row["file_id"]: row for row in manifest["files"]}
        episodes = manifest.get("episodes", [])
        failures = [e for e in episodes if str(e.get("kind")) == "failure"]
        degs: dict[str, list] = {}
        for e in episodes:
            if str(e.get("kind")) == "degradation":
                degs.setdefault(str(e["robot_id"]), []).append(e)
        maint: dict[str, list] = {}
        for e in episodes:
            if str(e.get("kind")) == "maintenance":
                maint.setdefault(str(e["robot_id"]), []).append(
                    (float(e["start_time"]), float(e.get("end_time") or 0.0)))

        def in_maint(robot: str, start: float, end: float) -> bool:
            return any(s < end and start < e for s, e in maint.get(robot, []))

        # per-robot file index (causal lookups)
        by_robot: dict[str, list] = {}
        for s in samples:
            robot = str(files[s.file_id]["robot_id"])
            by_robot.setdefault(robot, []).append(s)
        for evs in by_robot.values():
            evs.sort(key=lambda s: float(files[s.file_id]["end_time"]))

        for f in failures:
            robot = str(f["robot_id"])
            fstart = float(f["start_time"])
            # category from post-hoc simulator state only
            recent = [e for e in degs.get(robot, [])
                      if float(e.get("end_time") or 0.0) >= fstart - LOOKBACK_S
                      and float(e["start_time"]) <= fstart]
            long_recent = [e for e in recent
                           if float(e.get("end_time") or 0.0) - float(e["start_time"]) >= PROG_MIN_DUR_S]
            if long_recent:
                cat = "progressive"
            elif not recent:
                cat = "abrupt"
            else:
                cat = "weak"
            rec: dict = {"root": str(root), "episode": f["episode_id"],
                         "robot": robot, "category": cat, "fail_start": fstart}
            for wname, wlen in WINDOWS.items():
                win = [s for s in by_robot.get(robot, [])
                       if fstart - wlen <= float(files[s.file_id]["end_time"]) <= fstart
                       and not in_maint(robot, float(files[s.file_id]["start_time"]),
                                        float(files[s.file_id]["end_time"]))]
                rec[wname + "_n"] = len(win)
                rec[wname] = [file_score(s.x) for s in win]
            bg = [s for s in by_robot.get(robot, [])
                  if not any(abs(float(files[s.file_id]["end_time"]) - ff) <= LOOKBACK_S
                             for ff in [float(e["start_time"]) for e in failures
                                        if str(e["robot_id"]) == robot])
                  and not in_maint(robot, float(files[s.file_id]["start_time"]),
                                   float(files[s.file_id]["end_time"]))]
            bg_scores = np.array([file_score(s.x) for s in bg])
            rec["bg_n"] = len(bg)
            rec["bg_median"] = float(np.median(bg_scores)) if bg_scores.size else float("nan")
            per_failure.append(rec)

    cats: dict[str, dict] = {}
    for cat in ("progressive", "weak", "abrupt"):
        rows = [r for r in per_failure if r["category"] == cat]
        entry: dict = {"n_failures": len(rows)}
        for wname in WINDOWS:
            gaps = np.array([np.median(r[wname]) - r["bg_median"] for r in rows
                             if r[wname + "_n"] > 0 and np.isfinite(r["bg_median"])])
            entry[wname] = describe(gaps)
            entry[wname + "_window_n_median"] = float(
                np.median([r[wname + "_n"] for r in rows])) if rows else 0.0
        cats[cat] = entry
    result = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "dev_roots": list(args.dev_roots),
            "rule": "progressive = degradation overlapping [fail-14d, fail] dur>=3d; "
                    "abrupt = none ending within 14d; else weak",
            "score": "max centered patch amplitude (zero-fit, causal windows)",
        },
        "n_failures": len(per_failure),
        "categories": cats,
    }
    (out_dir / "task11_diag.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task11_diag.json'} ({len(per_failure)} failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
