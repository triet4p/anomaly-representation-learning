"""Sprint 16 Task 10 (M10/M6): file-embedding and pooling retention.

Holds frozen local/contextual latents fixed (both accepted checkpoints) and
compares score-then-pool readouts on IDENTICAL valid patches with the Task 3
contract reader: production-reference MEAN vs predeclared alternatives
top-8 tail mean (k=8 frozen), median, max — all on centroid distances from
Fit-healthy patch statistics. Reports per history/category event AUROC (+
paired-bootstrap CI), pooling-only gaps vs mean, valid-patch distributions,
duration slices (below/above median duration_d within P and within W),
sparse-support slices (below/above median candidate valid-patch count, P+W
pooled), background/stability per pooling, and excluded support. No
geometry/scorer/head changes, no encoder changes, no verdict (Task 16 owns
verdicts). Sealed roots are never touched (only FIT + CONFIRMATION paths).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task10_pool.py \\
    --checkpoint control --data-root <roots> --out <dir> --device cuda
  --self-test runs synthetic checks without checkpoint/data.

Writes <out>/<checkpoint>/{metrics.json,run.log}. Latent banks stay
server-side; only metrics.json + run.log return. Deterministic single run
per checkpoint (fixed seeds, eval mode, no grad).
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
TOPK = 8  # frozen tail width for top-8 mean


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pool_scores(s: object, method: str) -> float:
    """Reduce one file's valid-patch scores (frozen methods only)."""
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
        if a.size < TOPK:
            raise ValueError(f"top8mean needs >={TOPK} patches, got {a.size}")
        return float(np.partition(a, -TOPK)[-TOPK:].mean())
    raise ValueError(f"unknown pooling method {method!r}")


def self_test() -> int:
    """Synthetic checks of pooling math (no checkpoint/data)."""
    import numpy as np

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert pool_scores(a, "mean") == 5.0
    assert pool_scores(a, "median") == 5.0
    assert pool_scores(a, "max") == 9.0
    assert pool_scores(a, "top8mean") == 5.5
    try:
        pool_scores(a[:7], "top8mean")
    except ValueError:
        pass
    else:
        raise AssertionError("top8mean must refuse short input")
    try:
        pool_scores(a, "p99")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method must raise")
    try:
        pool_scores([], "mean")
    except ValueError:
        pass
    else:
        raise AssertionError("empty input must raise")
    print(json.dumps({"self_test": "PASS"}))
    return 0


def _auc_ci(pos: list[float], neg: list[float]) -> dict[str, float]:
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


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=("control", "hybrid"), default="control")
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
    from representation.v2_inference import V2InferencePipeline
    from synth import balanced as B
    from synth import events as E
    from synth.chronicle import load_chronological
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    spec = CHECKPOINTS[args.checkpoint]
    ck_path = Path(spec["path"])
    if not ck_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ck_path}")
    digest = sha256_file(ck_path)
    if digest != spec["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {digest}")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    outdir = Path(args.out) / args.checkpoint
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"checkpoint": args.checkpoint, "sha256": digest,
           "device": device, "steps": []}

    pipe = V2InferencePipeline.load(str(ck_path), device=device)
    model = pipe.model.eval()
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    @torch.no_grad()
    def encode_both(padded: np.ndarray, pad_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = padded.shape[0]
        pw = torch.asarray(padded, dtype=torch.float32).unsqueeze(0).to(device)
        pm = torch.asarray(pad_mask, dtype=torch.bool).unsqueeze(0).to(device)
        vm = torch.ones((1, K), dtype=torch.bool).to(device)
        local = model.local(pw, pm)
        if isinstance(local, dict):
            local = local["patch_latents"]
        ctx = model.context_encoder(local, vm)
        if isinstance(ctx, dict):
            ctx = ctx["patch_latents"]
        return (local.detach().cpu().numpy()[0].astype(np.float64),
                ctx.detach().cpu().numpy()[0].astype(np.float64))

    # --- Fit: healthy patch latents, both stages ---
    fit_loc, fit_ctx = [], []
    fit_files: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_n = 0
    for role, seed in FIT:
        samples, manifest = load_root(role, "FIT")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        for row in manifest["files"]:
            if row["file_label"] != "normal" or row["is_quarantined"]:
                continue
            if (row["program_id"] == B.PROGRAM_RESERVE
                    or row["robot_id"] == B.ROBOT_RESERVE):
                continue
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = valid > 0
            if not keep.any():
                continue
            loc, ctx = encode_both(patches[keep], pad[keep])
            fit_loc.append(loc)
            fit_ctx.append(ctx)
            fit_files[row["file_id"]] = (loc, ctx)
            fit_n += 1
    fit_loc = np.vstack(fit_loc).astype(np.float64)
    fit_ctx = np.vstack(fit_ctx).astype(np.float64)
    st_loc = M.FrozenStandardizer.fit(fit_loc, source="FIT-healthy-patches")
    st_ctx = M.FrozenStandardizer.fit(fit_ctx, source="FIT-healthy-patches")
    cen_loc = st_loc.apply(fit_loc).mean(axis=0)
    cen_ctx = st_ctx.apply(fit_ctx).mean(axis=0)
    token = st_loc.token()
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_loc.shape[0]),
                         "provenance_token": token})

    def patch_scores_of(feats: np.ndarray, st, cen) -> np.ndarray:
        return np.linalg.norm(st.apply(feats) - cen, axis=1)

    # --- Confirmation: four pooling readouts per stage ---
    METHODS = ("mean", "top8mean", "median", "max")
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        # scores[stage][method][file_id]
        scores: dict[str, dict[str, dict[str, float]]] = {
            "local": {m: {} for m in METHODS},
            "context": {m: {} for m in METHODS}}
        patch_counts: dict[str, int] = {}
        n_patches, n_short = 0, 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)

        def window_reader(score_map: dict[str, float]):
            scored = {k: v for k, v in score_map.items() if np.isfinite(v)}
            neg = [E.window_score([m["file_id"] for m in w["members"]
                                   if m["file_id"] in scored], scored)
                   for w in controls]
            neg = [v for v in neg if v is not None and np.isfinite(v)]

            def event_score(failure: dict) -> float | None:
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = [c for c in E.pos_files(rows, failure, wins)
                         if c["file_id"] in scored]
                if not cands:
                    return None
                return E.window_score([c["file_id"] for c in cands], score_map)
            return neg, event_score

        results: dict[str, dict] = {}
        for stage in ("local", "context"):
            for m in METHODS:
                neg, ev = window_reader(scores[stage][m])
                cell: dict[str, dict] = {}
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
                results[f"{stage}_{m}"] = cell
        # pooling-only gaps vs production mean, per stage and category
        for stage in ("local", "context"):
            base = results[f"{stage}_mean"]
            for m in ("top8mean", "median", "max"):
                other = results[f"{stage}_{m}"]
                for cat in CATEGORIES:
                    if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                        other[f"gap_vs_mean_{cat}"] = round(
                            other[cat]["point"] - base[cat]["point"], 4)
        # slices: duration + sparse support (deterministic median splits)
        pw_events = [f for f in ledger if f["cohort"] in "PW"]
        dur_med = float(np.median([f["duration_d"] for f in pw_events]))
        sup_counts = {}
        for f in pw_events:
            cands = E.pos_files(rows, f, wins)
            sup_counts[f["failure_id"]] = sum(
                patch_counts.get(c["file_id"], 0) for c in cands)
        sup_med = float(np.median(list(sup_counts.values())))
        slices: dict[str, dict] = {}
        slice_defs = [
            ("short_duration", lambda f: f["duration_d"] < dur_med),
            ("long_duration", lambda f: f["duration_d"] >= dur_med),
            ("sparse_support", lambda f: sup_counts.get(f["failure_id"], 0) < sup_med),
            ("dense_support", lambda f: sup_counts.get(f["failure_id"], 0) >= sup_med),
        ]
        for label, pred in slice_defs:
            cell: dict[str, object] = {}
            for key in ("local_mean", "context_mean", "local_top8mean",
                        "local_median", "local_max"):
                stage, m = key.split("_", 1)
                neg_s, ev_s = window_reader(scores[stage][m])
                pos = [s for f in pw_events if pred(f)
                       and (s := ev_s(f)) is not None]
                if len(pos) < 3 or not neg_s:
                    cell[key] = {"reason": "no support", "n_pos": len(pos)}
                else:
                    ci = _auc_ci(pos, neg_s)
                    cell[key] = {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in ci.items()}
            cell["n_events"] = sum(1 for f in pw_events if pred(f))
            slices[label] = cell
        # background / stability per stage+pooling (file-level background
        # vs Fit-healthy file scores under the SAME pooling)
        fit_scores: dict[str, dict[str, list[float]]] = {
            stage: {m: [] for m in METHODS} for stage in ("local", "context")}
        for _fid, (flat, fctx) in fit_files.items():
            for stage, arr, st, cen in (
                    ("local", flat, st_loc, cen_loc),
                    ("context", fctx, st_ctx, cen_ctx)):
                ps = np.linalg.norm(st.apply(arr) - cen, axis=1)
                for m in METHODS:
                    try:
                        fit_scores[stage][m].append(pool_scores(ps, m))
                    except ValueError:
                        continue
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background: dict[str, dict] = {}
        for stage in ("local", "context"):
            for m in METHODS:
                bg = np.array([scores[stage][m][i] for i in bg_ids])
                bg = bg[np.isfinite(bg)]
                ref = np.array(fit_scores[stage][m])
                if bg.size == 0 or ref.size == 0:
                    background[f"{stage}_{m}"] = {"reason": "no support"}
                    continue
                sc = M.stability_ci(bg, ref)
                background[f"{stage}_{m}"] = {
                    "median": round(float(np.median(bg)), 4),
                    "n": int(bg.size), "n_ref": int(ref.size),
                    "stability": {k: round(v, 4) for k, v in sc.items()}}
        vcounts = np.array(list(patch_counts.values()))
        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_patches": int(n_patches),
            "n_short_top8_files": int(n_short),
            "patch_dist": {"min": int(vcounts.min()),
                           "median": float(np.median(vcounts)),
                           "max": int(vcounts.max())},
            "duration_median_d": dur_med,
            "support_median_patches": sup_med,
            "categories": results,
            "slices": slices,
        })
    metrics = {"checkpoint": args.checkpoint, "sha256": digest,
               "provenance_token": token, "histories": per_history,
               "elapsed_s": round(time.time() - t0, 1)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                    sort_keys=True))
    with open(outdir / "run.log", "w") as f:
        f.write(json.dumps(log, indent=1, sort_keys=True))
    print(json.dumps({"checkpoint": args.checkpoint,
                      "elapsed_s": metrics["elapsed_s"],
                      "histories": len(per_history)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
