"""Sprint 16 Task 15: C7 replication nuisance/background cells.

FROZEN REPLICATION RULE (fixed before any NEW outcome; this docstring +
the committed driver are the freeze record — the per-history recovery
tallies it consumes are accepted Task 14 numbers, not new outcomes; no
protocol amendment: v3 §10(iv)/(ii)/(iii) govern).

Candidates (ONLY C7 G-a/G-b — the sole Task 14 recovery direction;
C2/C6/C8/C9 showed no recovery and are non-candidates, never promoted):
per (variant × stage): local_Ga, local_Gb, context_Ga, context_Gb, each on
both checkpoints (encoder dimension). A candidate replicates iff ALL of:
(R1) P-macro recovery (gap ≥ +0.10) on ≥3/4 histories AND W-macro recovery
  on ≥3/4 histories (accepted Task 14 cells);
(R2) post-intervention LCB > 0.55 on every counted recovery cell;
(R3) no nuisance alarm: background exceedance @ frozen 138.03 ≤ 0.05 AND
  protocol per-robot-day FAR ≤ 0.05 AND ΔFAR vs Gprod ≤ +0.02 on identical
  support (NEW cells, this driver; veto-signal only — Task 16 gates);
(R4) severity medians reported (descriptive; non-monotone never fails —
  oracle precedent Task 4);
(R5) A-only and threshold-edge cells (e.g. 0.0993-type) never count.

This driver executes ONLY the missing R3 cells: per-file C7 file scores
(mean reduction, frozen latents, restored bank for Gprod) for
background-eligible Confirmation files + Fit-healthy reference, both
checkpoints. Event scores are NOT recomputed (Task 14 identical support).
Exceedance via M.exceedance_fraction @ 138.03; protocol FAR via frozen
E.false_alert_episodes (≤2-day grouping + reset-split, Task 4 form);
stability via M.stability_ci. Deterministic single run per checkpoint
(fixed seeds, eval, no grad). No threshold fitting, no fusion, no verdict
(Task 16 owns labels). Sealed roots never touched (FIT + CONFIRMATION).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task15_replicate.py \\
    --checkpoint control --data-root <roots> --out <dir> --device cuda
  --self-test runs synthetic checks without checkpoint/data.

Writes <out>/<checkpoint>/{metrics.json,run.log}. Latents stay server-side.
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
THRESHOLD = 138.03  # frozen operating threshold (display rounding; §4/§5)
ARMS = ("local_Gprod", "local_Ga", "local_Gb", "context_Gprod",
        "context_Ga", "context_Gb")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def self_test():
    """Synthetic checks: exceedance, FAR plumbing, guards (no data)."""
    import numpy as np

    from representation import attribution_metrics as M
    from synth import events as E

    assert M.exceedance_fraction(np.array([1.0, 200.0, 3.0]), 138.03) == \
        1.0 / 3.0
    assert M.exceedance_fraction(np.array([1.0, 2.0]), 138.03) == 0.0
    eps, far = E.false_alert_episodes({}, [], 100.0, {})
    assert int(eps) == 0 and float(far) == 0.0
    print(json.dumps({"self_test": "PASS"}))
    return 0


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
    outdir = Path(args.out) / args.checkpoint
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"checkpoint": args.checkpoint, "sha256": digest,
           "device": device, "threshold": THRESHOLD, "arms": list(ARMS),
           "steps": []}

    pipe = V2InferencePipeline.load(str(ck_path), device=device)
    model = pipe.model.eval()
    cfg = pipe.config
    prod_snap = pipe.geometry._geometry.snapshot()
    prod_geo = HierarchicalMahalanobisGeometry(
        int(cfg.d_model), shrinkage=float(cfg.shrinkage),
        covariance_eps=float(cfg.covariance_eps),
        min_group_samples=int(cfg.min_group_samples),
        diag_min_samples=int(cfg.diag_min_samples))
    prod_geo.restore_snapshot(prod_snap)
    log["steps"].append({"restored_bank_n_groups": len(prod_snap["groups"])})
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role, group):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    @torch.no_grad()
    def file_c7_scores(sample, row, batch, keep, stds, cens, geos0):
        """C7 file scores (mean) for background/reference files."""
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
        out = {}
        z = torch.zeros(K, dtype=torch.long)
        for stage, mat in (("local", loc.reshape(K, -1)),
                           ("context", ctxl.reshape(K, -1))):
            e = np.linalg.norm(stds[stage].apply(mat.numpy()) - cens[stage],
                               axis=1)
            out["%s_Ga" % stage] = float(e.mean())
            gb, _ = batched_single_energy(
                geos0[stage], torch.as_tensor(mat.numpy(), dtype=torch.float32),
                z, z, z)
            out["%s_Gb" % stage] = float(gb.numpy().mean())
        loc_f = loc.reshape(K, -1)
        ctx_f = ctxl.reshape(K, -1)
        rbk = rb.repeat_interleave(K)
        prk = pr.repeat_interleave(K)
        mix, _ = batched_mixture_energy(prod_geo, ctx_f, rbk, prk)
        mix_l, _ = batched_mixture_energy(prod_geo, loc_f, rbk, prk)
        out["context_Gprod"] = float(mix.numpy().mean())
        out["local_Gprod"] = float(mix_l.numpy().mean())
        return out

    # --- Fit pass 1: latent mats (no batch storage) ---
    loc_mat = []
    ctx_mat = []
    fit_n = 0
    with torch.no_grad():
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
                sample = by_id[row["file_id"]]
                batch = patchifier.patchify(_wrap(np.asarray(
                    sample.x, dtype=np.float64)))
                keep = np.flatnonzero(np.asarray(batch.valid_len,
                                                 dtype=int) > 0)
                if keep.size == 0:
                    continue
                K = keep.size
                pw = torch.asarray(np.asarray(batch.patches,
                                              dtype=np.float64)[keep],
                                   dtype=torch.float32).unsqueeze(0).to(device)
                pm = torch.asarray(np.asarray(batch.pad_mask,
                                              dtype=bool)[keep],
                                   dtype=torch.bool).unsqueeze(0).to(device)
                vm = torch.ones((1, K), dtype=torch.bool).to(device)
                loc = model.local(pw, pm)
                if isinstance(loc, dict):
                    loc = loc["patch_latents"]
                ctx = model.context_encoder(loc, vm)
                if isinstance(ctx, dict):
                    ctx = ctx["patch_latents"]
                loc_mat.append(loc.detach().cpu().numpy()[0])
                ctx_mat.append(ctx.detach().cpu().numpy()[0])
                fit_n += 1
    loc_mat = np.vstack(loc_mat).astype(np.float64)
    ctx_mat = np.vstack(ctx_mat).astype(np.float64)
    stds, cens, geos0 = {}, {}, {}
    for stage, mat in (("local", loc_mat), ("context", ctx_mat)):
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
    log["steps"].append({"fit_files": int(fit_n),
                         "provenance_token": token})
    # --- Fit pass 2: reference file scores per arm ---
    ref_scores = {a: [] for a in ARMS}
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
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
            fs = file_c7_scores(sample, row, batch, keep, stds, cens, geos0)
            for a in ARMS:
                ref_scores[a].append(fs[a])

    # --- Confirmation background files only (events already scored T14) ---
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, "seed-match:%s" % role
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, "tag:%s" % role
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        bg_scores = {a: {} for a in ARMS}
        flagged = {a: {} for a in ARMS}
        for row in rows:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            sample = by_id[row["file_id"]]
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                continue
            fs = file_c7_scores(sample, row, batch, keep, stds, cens, geos0)
            for a in ARMS:
                bg_scores[a][row["file_id"]] = fs[a]
                if fs[a] >= THRESHOLD and E.eligible_operational_row(row, wins):
                    flagged[a].setdefault(row["robot_id"], []).append(
                        row["end_time"])
        eval_days = {(r["robot_id"], int(r["end_time"] // 86400.0))
                     for r in rows if E.eligible_operational_row(r, wins)}
        background = {}
        for a in ARMS:
            bg = np.array([bg_scores[a][i] for i in bg_scores[a]])
            bg = bg[np.isfinite(bg)]
            ref = np.array(ref_scores[a])
            ref = ref[np.isfinite(ref)]
            exc = M.exceedance_fraction(bg, THRESHOLD) if bg.size else 0.0
            eps, far = E.false_alert_episodes(
                flagged[a], ledger, float(len(eval_days)), wins)
            sc = M.stability_ci(bg, ref) if bg.size and ref.size else {}
            background[a] = {
                "median": round(float(np.median(bg)), 4) if bg.size else None,
                "n": int(bg.size), "n_ref": int(ref.size),
                "exceedance_at_frozen_threshold": round(float(exc), 4),
                "false_episodes": int(eps),
                "far_per_robot_day": round(float(far), 4),
                "robot_days": float(len(eval_days)),
                "stability": {k: round(v, 4) for k, v in sc.items()}}
        per_history.append({"role": role, "seed": seed,
                            "background": background})
    metrics = {"checkpoint": args.checkpoint, "sha256": digest,
               "threshold": THRESHOLD, "histories": per_history,
               "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                    default=str,
                                                    sort_keys=True))
    with open(outdir / "run.log", "w") as f:
        f.write(json.dumps(log, indent=1, sort_keys=True, default=str))
    print(json.dumps({"checkpoint": args.checkpoint,
                      "elapsed_s": metrics["elapsed_s"],
                      "histories": len(per_history)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
