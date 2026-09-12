"""Sprint 16 Task 13 (X7): oracle features through production file reduction.

FROZEN MATRIX (fixed before any outcome; this docstring + the committed
driver are the freeze record — no protocol amendment: v3 §7 cell X7 names
exactly this comparison).

Cell X7 (the only Task 13 cell needing new execution; X1–X6/X8 reuse the
accepted Tasks 4/7/8/11/12 artifacts on their own identical supports):
oracle/controlled patch features (synth.probe15.extract_features, the Task
7/11 observable side) through the PRODUCTION file-reduction form — top-10%
tail mean (top_q_fraction 0.1, the accepted checkpoint config value both
checkpoints share) with per-event max reporting — vs diagnostic reductions
(mean = Task 11 R-simple form, max, top-4, median) on IDENTICAL valid
patches. Question: does file reduction erase localized oracle signal?

The tail form replicates the production aggregate_file_state tail_energy
row math exactly (top-q mean over valid patch energies); --self-test
asserts equality vs aggregate_file_state on synthetic rows, so no
canonical-label provenance is misused on oracle energies. Per
history/category event AUROC + paired-bootstrap CI (B=2000/seed 20260202),
gaps vs mean within this run, valid-patch support, background/stability per
arm, excluded support, Fit-healthy provenance token. No geometry/scorer
fusion, no threshold work, no verdict (Task 16 owns verdicts). Sealed roots
never touched (FIT + CONFIRMATION only).

Usage (local canonical form, CPU-only — no checkpoint needed):
  uv run --no-sync python experiments/sprint16_task13_matrix.py \\
    --data-root data/generated/sprint15-v7 --out artifacts/sprint-16/task13-local
  --self-test runs synthetic checks without data.

Writes <out>/{metrics.json,run.log}. Deterministic single run (fixed seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
METHODS = ("mean", "tail", "max", "top4", "median")
TOP_Q = 0.1  # production top_q_fraction, both accepted checkpoints
TOPK4 = 4


def reduce_scores(s, method):
    """Frozen file reductions over one file's valid-patch energies."""
    import numpy as np

    a = np.asarray(s, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("reduce_scores requires non-empty finite energies")
    if method == "mean":
        return float(a.mean())
    if method == "tail":
        k = max(1, int(np.ceil(TOP_Q * a.size)))
        return float(np.partition(a, -k)[-k:].mean())
    if method == "max":
        return float(a.max())
    if method == "top4":
        if a.size < TOPK4:
            raise ValueError("top4 needs >=%d patches, got %d" % (TOPK4, a.size))
        return float(np.partition(a, -TOPK4)[-TOPK4:].mean())
    if method == "median":
        return float(np.percentile(a, 50.0, method="linear"))
    raise ValueError("unknown reduction %r" % (method,))


def self_test():
    """Synthetic checks: reductions, guards, tail == production tail."""
    import numpy as np
    import torch

    from representation.v2_aggregation import aggregate_file_state

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert reduce_scores(a, "mean") == 5.5
    assert reduce_scores(a, "tail") == 10.0
    assert reduce_scores(a, "max") == 10.0
    assert reduce_scores(a, "top4") == 8.5
    assert reduce_scores(a, "median") == 5.5
    b = np.arange(1.0, 24.0)
    assert reduce_scores(b, "tail") == float(b[-3:].mean())
    for bad, meth in (([], "mean"), ([1.0, np.inf], "max"),
                      ([1.0, 2.0, 3.0], "top4")):
        try:
            reduce_scores(bad, meth)
        except ValueError:
            pass
        else:
            raise AssertionError("guards must refuse")
    try:
        reduce_scores(a, "p99")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method must raise")
    torch.manual_seed(0)
    for n in (5, 12, 30):
        e = torch.randn(2, n).abs()
        lat = torch.randn(2, n, 4)
        st = aggregate_file_state(
            lat, e, torch.ones((2, n), dtype=torch.bool),
            torch.zeros((2, n), dtype=torch.long),
            energy_source="context_energy", top_q_fraction=TOP_Q,
            elevated_threshold=3.0, n_regimes=7)
        for i in range(2):
            assert abs(float(st["tail_energy"][i].item())
                       - reduce_scores(e[i].numpy(), "tail")) < 1e-5, \
                "tail form must equal production tail_energy"
    print(json.dumps({"self_test": "PASS"}))
    return 0


def _auc_ci(pos, neg):
    import numpy as _np

    from representation import attribution_metrics as _M

    if not pos or not neg:
        raise ValueError("auc requires non-empty pos and neg")
    scores = _np.array(list(pos) + list(neg), dtype=_np.float64)
    labels = _np.array([1.0] * len(pos) + [0.0] * len(neg))
    rng = _np.random.default_rng(_M.BOOTSTRAP_SEED)
    draws = rng.integers(0, scores.size,
                         size=(_M.BOOTSTRAP_REPLICATES, scores.size))
    aucs = _np.asarray([_M.tie_auc(scores[r], labels[r]) for r in draws])
    aucs = aucs[_np.isfinite(aucs)]
    if aucs.size == 0:
        raise ValueError("no finite bootstrap draws")
    return {"point": float(_M.tie_auc(scores, labels)),
            "lcb": float(_np.quantile(aucs, 0.025)),
            "ucb": float(_np.quantile(aucs, 0.975)),
            "n_pos": len(pos), "n_neg": len(neg)}


def _wrap(x):
    from types import SimpleNamespace

    import numpy as _np

    return SimpleNamespace(x=_np.asarray(x, dtype=_np.float32))


def run_fit(patchifier, load_root):
    """Fit universe: healthy patch observables + per-file vectors."""
    import numpy as np

    from synth import balanced as B

    rows_out = []
    file_vecs = []
    n = 0
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:%s" % role
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            keep, vecs = file_vecs_for(by_id[row["file_id"]], row,
                                       patchifier)
            if keep == 0:
                continue
            rows_out.append(vecs)
            file_vecs.append(vecs)
            n += 1
    return np.vstack(rows_out).astype(np.float64), file_vecs, n


def file_vecs_for(sample, row, patchifier):
    """Observable vectors for one file's valid patches (0 count if none)."""
    import numpy as np

    from synth import probe15 as P

    batch = patchifier.patchify(_wrap(np.asarray(sample.x, dtype=np.float64)))
    keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
    if keep.size == 0:
        return 0, np.zeros((0, 1))
    patches = np.asarray(batch.patches, dtype=np.float64)[keep]
    full = np.asarray(sample.x, dtype=np.float64)
    n_t = full.shape[1]
    dt = (row["end_time"] - row["start_time"]) / n_t
    since_reset = row["end_time"] - row["last_reset_time"]
    valid = np.asarray(batch.valid_len, dtype=int)[keep]
    vecs = np.array([P.extract_features(
        patches[p][:, :int(valid[p])],
        int(valid[p]) * dt, since_reset) for p in range(keep.size)],
        dtype=np.float64)
    return int(keep.size), vecs


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.data_root or not args.out:
        ap.error("--data-root and --out are required without --self-test")

    import numpy as np

    from representation import attribution_metrics as M
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"top_q": TOP_Q, "methods": list(METHODS), "steps": []}
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role, group):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    fit_mat, fit_file_vecs, fit_n = run_fit(patchifier, load_root)
    st = M.FrozenStandardizer.fit(fit_mat, source="FIT-healthy-patches")
    cen = st.apply(fit_mat).mean(axis=0)
    token = st.token()
    fit_ref = {m: [] for m in METHODS}
    for vecs in fit_file_vecs:
        d = np.linalg.norm(st.apply(vecs) - cen, axis=1)
        for m in METHODS:
            try:
                fit_ref[m].append(reduce_scores(d, m))
            except ValueError:
                continue
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_mat.shape[0]),
                         "feature_dim": int(fit_mat.shape[1]),
                         "provenance_token": token})

    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:%s" % role
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        scores = {m: {} for m in METHODS}
        patch_counts = {}
        n_patches = 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        for row in rows:
            n_keep, vecs = file_vecs_for(by_id[row["file_id"]], row,
                                         patchifier)
            if n_keep == 0:
                raise ValueError("zero valid patches: %s" % row["file_id"])
            n_patches += n_keep
            d = np.linalg.norm(st.apply(vecs) - cen, axis=1)
            for m in METHODS:
                try:
                    scores[m][row["file_id"]] = reduce_scores(d, m)
                except ValueError:
                    scores[m][row["file_id"]] = float("nan")
            patch_counts[row["file_id"]] = int(n_keep)
        short_ids = [fid for fid, sm in scores["top4"].items()
                     if not np.isfinite(sm)]
        for m in METHODS:
            for fid in short_ids:
                scores[m].pop(fid, None)
        n_short = len(short_ids)

        def window_reader(score_map):
            scored = {k: v for k, v in score_map.items() if np.isfinite(v)}
            kept, dropped = [], 0
            for w in controls:
                members = [m["file_id"] for m in w["members"]
                           if m["file_id"] in scored]
                if not members:
                    dropped += 1
                    continue
                kept.append(E.window_score(members, scored))
            neg = [v for v in kept if v is not None and np.isfinite(v)]
            window_reader.dropped = dropped

            def event_score(failure):
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = [c for c in E.pos_files(rows, failure, wins)
                         if c["file_id"] in scored]
                if not cands:
                    return None
                return E.window_score([c["file_id"] for c in cands], score_map)
            return neg, event_score

        results = {}
        dropped_windows = 0
        for m in METHODS:
            neg, ev = window_reader(scores[m])
            dropped_windows = max(dropped_windows, window_reader.dropped)
            cell = {}
            for cat in CATEGORIES:
                cohort = cat[0] if len(cat) == 2 else cat
                subtype = cat if len(cat) == 2 else None
                pos = [s for f in ledger
                       if f["cohort"] == cohort
                       and (subtype is None or f["subtype"] == subtype)
                       and (s := ev(f)) is not None]
                if not pos or not neg:
                    cell[cat] = {"reason": "no support",
                                 "n_pos": len(pos), "n_neg": len(neg)}
                else:
                    ci = _auc_ci(pos, neg)
                    cell[cat] = {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in ci.items()}
            results[m] = cell
        for m in ("tail", "max", "top4", "median"):
            base, other = results["mean"], results[m]
            for cat in CATEGORIES:
                if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                    other["gap_vs_mean_%s" % cat] = round(
                        other[cat]["point"] - base[cat]["point"], 4)
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background = {}
        for m in METHODS:
            bg = np.array([scores[m][i] for i in bg_ids if i in scores[m]])
            bg = bg[np.isfinite(bg)]
            ref = np.array(fit_ref[m])
            ref = ref[np.isfinite(ref)]
            if bg.size == 0 or ref.size == 0:
                background[m] = {"reason": "no support"}
                continue
            sc = M.stability_ci(bg, ref)
            background[m] = {
                "median": round(float(np.median(bg)), 4),
                "n": int(bg.size), "n_ref": int(ref.size),
                "stability": {k: round(v, 4) for k, v in sc.items()}}
        vcounts = np.array(list(patch_counts.values()))
        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_patches": int(n_patches),
            "n_dropped_short_files": int(n_short),
            "n_dropped_control_windows": int(dropped_windows),
            "patch_dist": {"min": int(vcounts.min()),
                           "median": float(np.median(vcounts)),
                           "max": int(vcounts.max())},
            "categories": results,
            "background": background,
        })
    metrics = {"top_q": TOP_Q, "provenance_token": token,
               "histories": per_history,
               "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                    default=str,
                                                    sort_keys=True))
    with open(outdir / "run.log", "w") as f:
        f.write(json.dumps(log, indent=1, sort_keys=True, default=str))
    print(json.dumps({"histories": len(per_history),
                      "elapsed_s": metrics["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
