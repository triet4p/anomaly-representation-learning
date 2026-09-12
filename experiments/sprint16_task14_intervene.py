"""Sprint 16 Task 14 (C6/C7/C8/C9): bounded scorer-side interventions.

FROZEN VARIANTS (fixed before any outcome; this docstring + the committed
driver are the freeze record — v3 §8 lists exactly these forms; C2 runs in
sprint16_task14_patch.py; C1/C3/C4/C5 are explicit non-executions below).

Each variant changes ONE principal mechanism; references hold everything
else fixed. Families never fused. Gaps stay within (stage, family).

C6 (frozen latents, both stages, both checkpoints; 3 counts): file
  mean (reference) vs top-8 tail mean (k=8), median, max. Production
  mean-embedding readout vs diagnostic poolings.
C7 (same latents; 2 counts): G-prod reference (restored-bank mixture,
  file mean) vs G-a diagnostic regularized reader (global Fit-healthy
  centroid Euclidean on frozen-standardized latents, file mean) vs G-b
  global unconditional healthy centroid in production distance form
  (collapsed-conditioning Mahalanobis, file mean). Minimal
  bypass/replacement of the shipped conditional geometry.
C8 (in-vocabulary subset only — head asserts out of vocabulary, Task 12
  rule, counted never remapped; 2 counts): P-max reference (production NLL
  + per-event max, aggregation held per §8) vs S-a query-ablated NLL
  (latent block zeroed pre-head, per-event max) vs S-b patch-MSE
  (per-event max).
C9 (3 counts): P-max reference (shared with C8) vs event-window top-4
  mean (k=4), median, 90th percentile on NLL (linear percentiles);
  C9 <8-patch exclusion with counts.

Per history/category event AUROC + paired-bootstrap CI (B=2000/seed
20260202), within-reference gaps, background/stability per arm (arm's own
file reduction), severity medians per level within P/W (descriptive),
excluded support, provenance tokens, restored-bank rows/source. No
threshold fitting, no fusion, no verdict (Tasks 15/16 own replication and
verdicts). Labels/masks/categories post-hoc only. Sealed roots never
touched (FIT + CONFIRMATION only).

NON-EXECUTIONS (frozen reasons, 0 counts):
- C1 N-a/N-b/N-c: v4 binds stage-1 to verified identity (Task 5 gaps all
  0.0); additive replacements cannot establish C1 causality → UNRESOLVED.
- C3: P3 caps at SUSPECT; no bypass exists without redesign → no execution.
- C4 B-a/B-b: Task 8 refutes dilution (gaps ~0); bypass unmotivated → NOT_RUN.
- C5 background=0/covariance=0 retrains: Task 9 executed both ONCE with no
  replicated recovery (one-shot rule + 2/2 budget exhausted) → cited, capped
  at SUSPECT per v5 (hybrid replication unfunded).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task14_intervene.py \\
    --checkpoint control --data-root <roots> --out <dir> --device cuda
  --self-test runs synthetic checks without checkpoint/data.

Writes <out>/<checkpoint>/{metrics.json,run.log}. Latent banks stay
server-side; only metrics.json + run.log return. Deterministic single run
per checkpoint (fixed seeds, eval mode, no grad). Budget: 10 §8 counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

CHECKPOINTS = {
    "control": {
        "path": ("/tmp/sprint11-task30-corrected-control/full/"
                 "control-normal-only/default/v2_checkpoint.pt"),
        "sha256": "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c",
    },
    "hybrid": {
        "path": ("/tmp/sprint11-task31-corrected-hybrid/full/"
                 "hybrid-boundary/default/v2_checkpoint.pt"),
        "sha256": "76be843b0a5fa94c8cdd646e2734c804498f9c38088ed71c3f16f9aa97e2cc94",
    },
}
FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
TOPK8 = 8
TOPK4 = 4
MIN_WINDOW_PATCHES = 8
C6_METHODS = ("mean", "top8mean", "median", "max")
C9_METHODS = ("max", "top4", "median", "p90")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pool_scores(s, method):
    """Frozen file poolings (C6 reference + variants)."""
    import numpy as np

    a = np.asarray(s, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("pool_scores requires non-empty finite scores")
    if method == "mean":
        return float(a.mean())
    if method == "median":
        return float(np.percentile(a, 50.0, method="linear"))
    if method == "max":
        return float(a.max())
    if method == "top8mean":
        if a.size < TOPK8:
            raise ValueError("top8mean needs >=%d patches" % TOPK8)
        return float(np.partition(a, -TOPK8)[-TOPK8:].mean())
    raise ValueError("unknown pooling %r" % (method,))


def reduce_patches(a, method):
    """Frozen C9-form reductions over valid patch energies."""
    import numpy as np

    v = np.asarray(a, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("reduce_patches requires non-empty finite input")
    if method == "max":
        return float(v.max())
    if method == "top4":
        if v.size < TOPK4:
            raise ValueError("top4 needs >=%d patches" % TOPK4)
        return float(np.partition(v, -TOPK4)[-TOPK4:].mean())
    if method == "median":
        return float(np.percentile(v, 50.0, method="linear"))
    if method == "p90":
        return float(np.percentile(v, 90.0, method="linear"))
    raise ValueError("unknown reduction %r" % (method,))


def self_test():
    """Synthetic checks: poolings, reductions, guards, gap math."""
    import numpy as np

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert pool_scores(a, "mean") == 5.0
    assert pool_scores(a, "top8mean") == 5.5
    assert pool_scores(a, "median") == 5.0
    assert pool_scores(a, "max") == 9.0
    assert reduce_patches(a, "top4") == 7.5
    assert abs(reduce_patches(a, "p90") - 8.2) < 1e-9
    for bad, meth, fn in (([], "mean", pool_scores),
                          (a[:7], "top8mean", pool_scores),
                          ([1.0, np.inf], "max", reduce_patches)):
        try:
            fn(bad, meth)
        except ValueError:
            pass
        else:
            raise AssertionError("guards must refuse")
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
    ap.add_argument("--checkpoint", choices=("control", "hybrid"),
                    default="control")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.data_root or not args.out:
        ap.error("--data-root and --out are required without --self-test")

    import numpy as np
    import torch

    from representation import attribution_metrics as M
    from representation.v2_geometry import HierarchicalMahalanobisGeometry
    from representation.v2_inference import V2InferencePipeline, patch_regime_ids
    from sprint16_task11_geometry import (batched_mixture_energy,
                                          batched_single_energy)
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    spec = CHECKPOINTS[args.checkpoint]
    ck_path = Path(spec["path"])
    if not ck_path.is_file():
        raise FileNotFoundError("checkpoint missing: %s" % ck_path)
    digest = sha256_file(ck_path)
    if digest != spec["sha256"]:
        raise ValueError("checkpoint hash mismatch: %s" % digest)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    @torch.no_grad()
    def file_data(sample, row, batch, keep):
        """Frozen LOCAL + CONTEXTUAL latents, head/mixture signals, cond."""
        K = keep.size
        pw = torch.asarray(np.asarray(batch.patches, dtype=np.float64)[keep],
                           dtype=torch.float32).unsqueeze(0).to(device)
        pm = torch.asarray(np.asarray(batch.pad_mask, dtype=bool)[keep],
                           dtype=torch.bool).unsqueeze(0).to(device)
        vm = torch.ones((1, K), dtype=torch.bool).to(device)
        loc = model.local(pw, pm)
        if isinstance(loc, dict):
            loc = loc["patch_latents"]
        ctx = model.context_encoder(loc, vm)
        if isinstance(ctx, dict):
            ctx = ctx["patch_latents"]
        loc = loc.detach().cpu()
        ctxl = ctx.detach().cpu()
        rb = torch.tensor([int(sample.robot_idx)])
        pr = torch.tensor([int(sample.program_idx)])
        starts = torch.as_tensor(np.asarray(batch.starts,
                                            dtype=np.int64)[keep]).unsqueeze(0)
        rg = patch_regime_ids([sample], starts, K)
        in_range = bool(rb.item() < NR and pr.item() < NP)
        nan = np.full(K, np.nan)
        nll = abl = mse = nan
        if in_range:
            try:
                cvec = model._context_vectors(rb.to(device), pr.to(device),
                                              rg.to(device))
                params = model.head(ctxl.to(device), cvec)
                cm = params["cond_mean"].detach().cpu()
                lv = params["cond_logvar"].detach().cpu()
                nll = (0.5 * (((ctxl - cm).pow(2) / lv.exp() + lv).sum(-1))
                       ).numpy()[0].astype(np.float64)
                mse = ((ctxl - cm).pow(2)).sum(-1).numpy()[0].astype(np.float64)
                ab = model.head(torch.zeros_like(ctxl.to(device)), cvec)
                abl = (0.5 * (((ctxl.to(device) - ab["cond_mean"]).pow(2)
                               / ab["cond_logvar"].exp()
                               + ab["cond_logvar"]).sum(-1))
                       ).detach().cpu().numpy()[0].astype(np.float64)
            except (RuntimeError, IndexError, ValueError):
                in_range = False
        ctx_f = ctxl.reshape(K, -1)
        loc_f = loc.reshape(K, -1)
        rbk = rb.repeat_interleave(K)
        prk = pr.repeat_interleave(K)
        mix, _ = batched_mixture_energy(prod_geo, ctx_f, rbk, prk)
        mix_l, _ = batched_mixture_energy(prod_geo, loc_f, rbk, prk)
        return {"local": loc_f.numpy().astype(np.float64),
                "context": ctx_f.numpy().astype(np.float64),
                "nll": np.asarray(nll, dtype=np.float64),
                "abl": np.asarray(abl, dtype=np.float64),
                "mse": np.asarray(mse, dtype=np.float64),
                "mix": mix.numpy().astype(np.float64),
                "mix_local": mix_l.numpy().astype(np.float64),
                "regimes": rg.numpy()[0].astype(int),
                "in_range": in_range}
    outdir = Path(args.out) / args.checkpoint
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"checkpoint": args.checkpoint, "sha256": digest,
           "device": device, "budget_counts": {"C6": 3, "C7": 2,
                                               "C8": 2, "C9": 3},
           "steps": []}

    pipe = V2InferencePipeline.load(str(ck_path), device=device)
    model = pipe.model.eval()
    cfg = pipe.config
    NR, NP = int(cfg.n_robots), int(cfg.n_programs)
    prod_snap = pipe.geometry._geometry.snapshot()
    prod_geo = HierarchicalMahalanobisGeometry(
        int(cfg.d_model), shrinkage=float(cfg.shrinkage),
        covariance_eps=float(cfg.covariance_eps),
        min_group_samples=int(cfg.min_group_samples),
        diag_min_samples=int(cfg.diag_min_samples))
    prod_geo.restore_snapshot(prod_snap)
    bank = {"source": "restored-checkpoint-geometry",
            "n_groups": len(prod_snap["groups"]),
            "n_pairs": len(prod_snap["pair"]),
            "n_robots": len(prod_snap["robot"]),
            "fleet_n": prod_snap["fleet"]["n"]}
    log["steps"].append({"restored_bank": bank,
                         "conditioning_vocab": {"n_robots": NR,
                                                "n_programs": NP}})
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role, group):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest


    # --- Fit: per-stage standardizers/centroids + collapsed G-b fits ---
    fit_loc, fit_ctx = [], []
    fit_bank = {}
    fit_n = 0
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
            sample = by_id[row["file_id"]]
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                continue
            sig = file_data(sample, row, batch, keep)
            fit_loc.append(sig["local"])
            fit_ctx.append(sig["context"])
            fit_bank[row["file_id"]] = sig
            fit_n += 1
    fit_loc = np.vstack(fit_loc).astype(np.float64)
    fit_ctx = np.vstack(fit_ctx).astype(np.float64)
    stds, cens, geos0 = {}, {}, {}
    for stage, mat in (("local", fit_loc), ("context", fit_ctx)):
        stds[stage] = M.FrozenStandardizer.fit(mat,
                                               source="FIT-healthy-patches")
        cens[stage] = stds[stage].apply(mat).mean(axis=0)
        lat = torch.as_tensor(mat, dtype=torch.float32)
        z = torch.zeros(lat.shape[0], dtype=torch.long)
        g0 = HierarchicalMahalanobisGeometry(
            mat.shape[1], shrinkage=float(cfg.shrinkage),
            covariance_eps=float(cfg.covariance_eps),
            min_group_samples=int(cfg.min_group_samples),
            diag_min_samples=int(cfg.diag_min_samples))
        g0.fit(lat, z, z, z, torch.ones(lat.shape[0], dtype=torch.bool))
        geos0[stage] = g0
    token = stds["local"].token()
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_loc.shape[0]),
                         "provenance_token": token})
    # --- Confirmation: C6/C7 file arms + C8/C9 direct arms ---
    STAGES = ("local", "context")
    file_arms = []
    for stage in STAGES:
        file_arms += ["%s_mean" % stage, "%s_top8" % stage,
                      "%s_median" % stage, "%s_max" % stage,
                      "%s_Gprod" % stage, "%s_Ga" % stage, "%s_Gb" % stage]
    direct_arms = {"Pmax": ("nll", "max"), "SaMax": ("abl", "max"),
                   "SbMax": ("mse", "max"), "Ptop4": ("nll", "top4"),
                   "Pmedian": ("nll", "median"), "Pp90": ("nll", "p90")}
    AL_ARMS = file_arms + sorted(direct_arms)
    gap_ref = {}
    for stage in STAGES:
        for a in ("%s_top8" % stage, "%s_median" % stage, "%s_max" % stage):
            gap_ref[a] = "%s_mean" % stage
        for a in ("%s_Ga" % stage, "%s_Gb" % stage):
            gap_ref[a] = "%s_Gprod" % stage
    for a in ("SaMax", "SbMax", "Ptop4", "Pmedian", "Pp90"):
        gap_ref[a] = "Pmax"

    def file_arm_scores(sig):
        """All 14 file-level arm scores from one file's signals."""
        out = {}
        z = torch.zeros(sig["local"].shape[0], dtype=torch.long)
        for stage in STAGES:
            feat = sig[stage]
            e = np.linalg.norm(stds[stage].apply(feat) - cens[stage], axis=1)
            out["%s_mean" % stage] = pool_scores(e, "mean")
            out["%s_top8" % stage] = pool_scores(e, "top8mean")
            out["%s_median" % stage] = pool_scores(e, "median")
            out["%s_max" % stage] = pool_scores(e, "max")
            out["%s_Ga" % stage] = pool_scores(e, "mean")
            rows_t = torch.as_tensor(feat, dtype=torch.float32)
            gb, _ = batched_single_energy(geos0[stage], rows_t, z, z, z)
            out["%s_Gb" % stage] = float(gb.numpy().mean())
        rows_t = None
        out["context_Gprod"] = float(np.mean(sig["mix"]))
        out["local_Gprod"] = float(np.mean(sig["mix_local"]))
        return out
    fit_file_scores = {fid: file_arm_scores(s)
                       for fid, s in fit_bank.items()}

    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:%s" % role
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        patch_bank = {}
        scores = {a: {} for a in AL_ARMS}
        n_patches = 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        for row in rows:
            sample = by_id[row["file_id"]]
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                raise ValueError("zero valid patches: %s" % row["file_id"])
            sig = file_data(sample, row, batch, keep)
            n_patches += keep.size
            patch_bank[row["file_id"]] = sig
            if keep.size < TOPK8:
                for a in file_arms:
                    scores[a][row["file_id"]] = float("nan")
                continue
            for a, v in file_arm_scores(sig).items():
                scores[a][row["file_id"]] = v

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

        direct_excluded = 0

        def direct_reader(base, method):
            only_in_range = base in ("nll", "abl", "mse")

            def members_ok(fid):
                return (fid in patch_bank and
                        (not only_in_range or patch_bank[fid]["in_range"]))

            neg, drops = [], 0
            for w in controls:
                qual = [m["file_id"] for m in w["members"]
                        if members_ok(m["file_id"])]
                if not qual:
                    drops += 1
                    continue
                pool = np.concatenate([patch_bank[f][base] for f in qual])
                if pool.size < MIN_WINDOW_PATCHES:
                    drops += 1
                    continue
                neg.append(reduce_patches(pool, method))

            def event_score(failure):
                nonlocal direct_excluded
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = [c for c in E.pos_files(rows, failure, wins)
                         if members_ok(c["file_id"])]
                if not cands:
                    return None
                pool = np.concatenate([patch_bank[c["file_id"]][base]
                                       for c in cands])
                if pool.size < MIN_WINDOW_PATCHES:
                    direct_excluded += 1
                    return None
                return reduce_patches(pool, method)
            return neg, event_score, drops

        readers = {}
        tail_dropped, direct_dropped = 0, 0
        for a in file_arms:
            neg, ev = window_reader(scores[a])
            tail_dropped = max(tail_dropped, window_reader.dropped)
            readers[a] = (neg, ev)
        for a, (base, method) in direct_arms.items():
            neg, ev, dr = direct_reader(base, method)
            readers[a] = (neg, ev)
            direct_dropped = max(direct_dropped, dr)

        results = {}
        for a in AL_ARMS:
            neg, ev = readers[a]
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
            results[a] = cell
        for a, ref in gap_ref.items():
            base, other = results[ref], results[a]
            for cat in CATEGORIES:
                if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                    other["gap_vs_%s_%s" % (ref, cat)] = round(
                        other[cat]["point"] - base[cat]["point"], 4)
        sev = {}
        for a in AL_ARMS:
            neg_a, ev_a = readers[a]
            for cohort in ("P", "W"):
                lv = {}
                for f in ledger:
                    if f["cohort"] != cohort:
                        continue
                    s = ev_a(f)
                    if s is None or not np.isfinite(s):
                        continue
                    lv.setdefault(f["severity"], []).append(s)
                sev["%s_%s" % (a, cohort)] = {
                    str(k): round(float(np.median(v)), 4)
                    for k, v in sorted(lv.items()) if len(v) >= 3}
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background = {}
        for a in file_arms:
            bg = np.array([scores[a][i] for i in bg_ids if i in scores[a]])
            bg = bg[np.isfinite(bg)]
            ref = np.array([fit_file_scores[f][a] for f in fit_file_scores])
            ref = ref[np.isfinite(ref)]
            if bg.size == 0 or ref.size == 0:
                background[a] = {"reason": "no support"}
                continue
            sc = M.stability_ci(bg, ref)
            background[a] = {"median": round(float(np.median(bg)), 4),
                             "n": int(bg.size), "n_ref": int(ref.size),
                             "stability": {k: round(v, 4)
                                           for k, v in sc.items()}}
        for a, (base, method) in direct_arms.items():
            floor = TOPK4 if method == "top4" else 1
            pred_ok = (lambda fid: True) if base == "mix" else (
                lambda fid: patch_bank[fid]["in_range"])
            bg = np.array([reduce_patches(patch_bank[i][base], method)
                           if patch_bank[i][base].size >= floor
                           and pred_ok(i) else np.nan for i in bg_ids
                           if i in patch_bank])
            bg = bg[np.isfinite(bg)]
            ref = np.array([reduce_patches(fit_bank[f][base], method)
                            if fit_bank[f][base].size >= floor
                            and (base == "mix"
                                 or fit_bank[f]["in_range"])
                            else np.nan for f in fit_bank])
            ref = ref[np.isfinite(ref)]
            if bg.size == 0 or ref.size == 0:
                background[a] = {"reason": "no support"}
                continue
            sc = M.stability_ci(bg, ref)
            background[a] = {"median": round(float(np.median(bg)), 4),
                             "n": int(bg.size), "n_ref": int(ref.size),
                             "stability": {k: round(v, 4)
                                           for k, v in sc.items()}}
        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_patches": int(n_patches),
            "n_in_range_files": int(sum(1 for s in patch_bank.values()
                                       if s["in_range"])),
            "n_out_of_vocab_files": int(sum(1 for s in patch_bank.values()
                                           if not s["in_range"])),
            "n_dropped_control_windows": int(tail_dropped),
            "n_direct_control_dropped": int(direct_dropped),
            "n_excluded_direct_windows": int(direct_excluded),
            "categories": results,
            "severity_medians": sev,
            "background": background,
        })
    metrics = {"checkpoint": args.checkpoint, "sha256": digest,
               "restored_bank": bank,
               "provenance_token": token, "histories": per_history,
               "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                    default=str,
                                                    sort_keys=True))
    with open(outdir / "run.log", "w") as f:
        f.write(json.dumps(log, indent=1, sort_keys=True, default=str))
    print(json.dumps({"checkpoint": args.checkpoint,
                      "elapsed_s": metrics["elapsed_s"],
                      "histories": len(per_history)}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
