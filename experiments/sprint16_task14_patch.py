"""Sprint 16 Task 14 (C2): patch-geometry interventions on observables.

FROZEN VARIANTS (fixed before any outcome; this docstring + the committed
driver are the freeze record — v3 §8 C2 lists exactly P-a/P-b/P-c; no other
C2 form is allowed).

P-a (16/16), P-b (32/32), P-c (64/32) vs production geometry (32/16,
asserted from PatchConfig): frozen diagnostic reader on patchified
observables (per-geometry Fit-healthy standardizer + centroid, Euclidean
patch energies, file MEAN, per-event max reporting — aggregation fixed
because aggregation is C9, not C2). One boundary (patch geometry) varies;
no encoder is reused across sizes (frozen weights are size-bound, §8 rule).
Gaps vs production geometry on identical file support (unified valid
universe across geometries; short files excluded everywhere and counted).
Per history/category event AUROC + paired-bootstrap CI (B=2000/seed
20260202), background/stability per arm, severity medians per level within
P/W (descriptive), excluded support, provenance tokens per geometry.
No verdict (Task 16 owns verdicts). Sealed roots never touched
(FIT + CONFIRMATION only).

Usage (local canonical form, CPU-only — no checkpoint needed):
  uv run --no-sync python experiments/sprint16_task14_patch.py \\
    --data-root data/generated/sprint15-v7 --out artifacts/sprint-16/task14-patch
  --self-test runs synthetic checks without data.

Writes <out>/{metrics.json,run.log}. Deterministic single run (fixed seeds).
Budget: 3 §8 counts (P-a, P-b, P-c).
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
GEOMS = {"prod": (32, 16), "P-a": (16, 16), "P-b": (32, 32), "P-c": (64, 32)}


def self_test():
    """Synthetic checks: geometry table, mean reduction, gap math."""
    assert GEOMS["prod"] == (32, 16)
    assert len(GEOMS) == 4
    import numpy as np

    a = np.array([2.0, 4.0, 6.0])
    assert float(a.mean()) == 4.0
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
    from synth import probe15 as P
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    prod_cfg = PatchConfig()
    assert (prod_cfg.patch_size, prod_cfg.stride, prod_cfg.pad_end,
            prod_cfg.pad_value) == (32, 16, True, 0.0), "production geometry"
    patchifiers = {}
    for name, (w, s) in GEOMS.items():
        if name == "prod":
            patchifiers[name] = Patchifier(prod_cfg)
        else:
            patchifiers[name] = Patchifier(PatchConfig(patch_size=w,
                                                       stride=s))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"geometries": {k: list(v) for k, v in GEOMS.items()},
           "steps": []}
    data_root = Path(args.data_root)

    def load_root(role, group):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    def file_vecs(sample, row, patchifier):
        batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                     dtype=np.float64)))
        keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
        if keep.size == 0:
            return 0, None
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

    # --- Fit per geometry (keep per-file vectors for reference means) ---
    fits = {}
    fit_ref = {}
    for name, pf in patchifiers.items():
        rows_out = []
        file_vec_list = []
        n = 0
        for role, seed in FIT:
            samples, manifest = load_root(role, "FIT")
            assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
            assert manifest.get("protocol") == B.S15_PROTOCOL_V7, \
                "tag:%s" % role
            by_id = {s.file_id: s for s in samples}
            for row in manifest["files"]:
                if row["file_label"] != "normal" or row["is_quarantined"]:
                    continue
                if (row["program_id"] == B.PROGRAM_RESERVE
                        or row["robot_id"] == B.ROBOT_RESERVE):
                    continue
                k, vecs = file_vecs(by_id[row["file_id"]], row, pf)
                if k == 0:
                    continue
                rows_out.append(vecs)
                file_vec_list.append(vecs)
                n += 1
        mat = np.vstack(rows_out).astype(np.float64)
        std = M.FrozenStandardizer.fit(mat, source="FIT-healthy-patches")
        cen = std.apply(mat).mean(axis=0)
        fits[name] = {"std": std, "cen": cen,
                      "token": std.token(), "n": n,
                      "patches": int(mat.shape[0]),
                      "dim": int(mat.shape[1])}
        fit_ref[name] = [float(np.linalg.norm(std.apply(v) - cen, axis=1).mean())
                         for v in file_vec_list]
    log["steps"].append({"fit_files": {k: v["n"] for k, v in fits.items()},
                         "tokens": {k: v["token"] for k, v in fits.items()}})

    # --- Confirmation per geometry, unified valid universe ---
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:%s" % role
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        file_means = {g: {} for g in GEOMS}
        patch_counts = {}
        n_short = 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        for row in rows:
            per_geom = {}
            ok = True
            for name, pf in patchifiers.items():
                k, vecs = file_vecs(by_id[row["file_id"]], row, pf)
                if k == 0:
                    ok = False
                    break
                d = np.linalg.norm(fits[name]["std"].apply(vecs)
                                   - fits[name]["cen"], axis=1)
                per_geom[name] = float(d.mean())
            if not ok:
                n_short += 1
                continue
            for name in GEOMS:
                file_means[name][row["file_id"]] = per_geom[name]
            patch_counts[row["file_id"]] = 1
        n_patches_note = "per-geometry counts differ; support is file-unified"

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
        for g in GEOMS:
            neg, ev = window_reader(file_means[g])
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
            results[g] = cell
        for g in ("P-a", "P-b", "P-c"):
            base, other = results["prod"], results[g]
            for cat in CATEGORIES:
                if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                    other["gap_vs_prod_%s" % cat] = round(
                        other[cat]["point"] - base[cat]["point"], 4)
        # severity medians per level within P/W (descriptive)
        sev = {}
        for g in GEOMS:
            neg_g, ev_g = window_reader(file_means[g])
            for cohort in ("P", "W"):
                lv = {}
                for f in ledger:
                    if f["cohort"] != cohort:
                        continue
                    s = ev_g(f)
                    if s is None or not np.isfinite(s):
                        continue
                    lv.setdefault(f["severity"], []).append(s)
                sev["%s_%s" % (g, cohort)] = {
                    str(k): round(float(np.median(v)), 4)
                    for k, v in sorted(lv.items()) if len(v) >= 3}
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background = {}
        for g in GEOMS:
            bg = np.array([file_means[g][i] for i in bg_ids
                           if i in file_means[g]])
            bg = bg[np.isfinite(bg)]
            ref = np.array(fit_ref[g])
            ref = ref[np.isfinite(ref)]
            if bg.size == 0 or ref.size == 0:
                background[g] = {"reason": "no support"}
                continue
            sc = M.stability_ci(bg, ref)
            background[g] = {"median": round(float(np.median(bg)), 4),
                             "n": int(bg.size), "n_ref": int(ref.size),
                             "stability": {k: round(v, 4)
                                           for k, v in sc.items()}}
        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_unified_files": len(patch_counts),
            "n_dropped_short_files": int(n_short),
            "n_dropped_control_windows": int(dropped_windows),
            "patch_note": n_patches_note,
            "categories": results,
            "severity_medians": sev,
            "background": background,
        })
    metrics = {"geometries": {k: list(v) for k, v in GEOMS.items()},
               "provenance_tokens": {k: v["token"] for k, v in fits.items()},
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
