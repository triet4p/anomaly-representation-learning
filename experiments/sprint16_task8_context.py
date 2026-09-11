"""Sprint 16 Task 8 (M8): contextual-encoder retention vs local latents.

Runs where the accepted checkpoints live (GPU server, verified commit):
loads the accepted control/hybrid V2 checkpoints (sha256 asserted before
load) and compares, at EXACTLY matched valid patch positions/support,
frozen LOCAL latents (LocalPatchEncoder) against frozen CONTEXTUAL latents
(direct ``model.context_encoder(local, valid_mask)`` call — robot/program
independent, bit-identical to the full-forward ``patch_latents``) with the
shared diagnostic reader: linear (Fit-healthy centroid distance) and simple
nonlinear (kNN distance, k=5, deterministic ≤30k bank). File/event scores
use the frozen per-event max; per history/category event AUROC comes from
the Task 3 contract reader. FULL Task 7 support (all files/patches, no
index filter — the encoder call needs no conditioning). No objective
intervention, no geometry/scorer/pooling fusion, labels post-hoc, FIT-only
fitting, CONF-only evaluation, A companion only, no verdict (Task 16 owns
verdicts). Sealed roots never touched (only FIT + CONFIRMATION paths).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task8_context.py \\
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
KNN_K = 5
BANK_MAX = 30000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def self_test() -> int:
    """Synthetic checks of comparative logic (no checkpoint/data)."""
    import numpy as np

    from representation import attribution_metrics as M

    rng = np.random.default_rng(1)
    healthy = rng.normal(0, 1, size=(400, 32))
    shift = np.array([2.0] + [0.0] * 31)
    st = M.FrozenStandardizer.fit(healthy, source="SELFTEST8")
    context_like = healthy + 0.1 * rng.normal(size=healthy.shape)
    for name, mat in (("local", healthy + np.vstack(
            [shift] * 100 + [[0.0] * 32] * 300)[:400]),
                      ("context", context_like + np.vstack(
            [shift] * 100 + [[0.0] * 32] * 300)[:400])):
        z = st.apply(mat)
        d = np.linalg.norm(z - z[100:].mean(axis=0), axis=1)
        auc = M.tie_auc(d, np.array([1.0] * 100 + [0.0] * 300))
        assert auc > 0.6, (name, auc)
    gap = M.movement([0.06, -0.02])
    assert gap["magnitude"] == abs(0.06)
    print(json.dumps({"self_test": "PASS"}))
    return 0


def _auc_ci(pos: list[float], neg: list[float]) -> dict[str, float]:
    """Contract-reader event AUROC with paired-bootstrap CI (frozen B/seed)."""
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
    """Minimal patchify input (only .x is consumed)."""
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
    from synth import probe15 as P
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
    n_r, n_p = pipe.config.n_robots, pipe.config.n_programs
    log["steps"].append({"n_robots": n_r, "n_programs": n_p})
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest


    @torch.no_grad()
    def encode_both(padded: np.ndarray, pad_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Frozen local + contextual latents on identical patches.

        ``padded`` is [K,C,W] full windows, ``pad_mask`` bool [K,W]
        (True = padded). Per-patch validity is valid_len > 0, enforced by
        the caller subset; both encoders receive all-True masks. The
        contextual call is robot/program independent
        (``context_encoder(local, valid_mask)``) — no conditioning indices,
        no invented mapping; bit-identical to the full-forward
        ``patch_latents`` by construction (same submodule, same inputs).
        """
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
    # --- Fit: healthy patch latents (local + contextual), same patches ---
    fit_loc, fit_ctx = [], []
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
            sample = by_id[row["file_id"]]
            x = np.asarray(sample.x, dtype=np.float64)
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
            fit_n += 1
    fit_loc = np.vstack(fit_loc).astype(np.float64)
    fit_ctx = np.vstack(fit_ctx).astype(np.float64)
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_loc.shape[0])})
    st_loc = M.FrozenStandardizer.fit(fit_loc, source="FIT-healthy-patches")
    st_ctx = M.FrozenStandardizer.fit(fit_ctx, source="FIT-healthy-patches")
    cen_loc = st_loc.apply(fit_loc).mean(axis=0)
    cen_ctx = st_ctx.apply(fit_ctx).mean(axis=0)
    bank_loc = st_loc.apply(fit_loc)[:: max(1, fit_loc.shape[0] // BANK_MAX)]
    bank_ctx = st_ctx.apply(fit_ctx)[:: max(1, fit_ctx.shape[0] // BANK_MAX)]
    token = st_loc.token()
    log["steps"].append({"latent_dim": int(fit_loc.shape[1]),
                         "provenance_token": token})

    def score_patches(feats: np.ndarray, st, cen, bank) -> tuple[np.ndarray, np.ndarray]:
        z = st.apply(feats)
        lin = np.linalg.norm(z - cen, axis=1)
        out = []
        with torch.no_grad():
            zt = torch.asarray(z, dtype=torch.float32)
            bt = torch.asarray(bank, dtype=torch.float32)
            if device == "cuda":
                zt, bt = zt.cuda(), bt.cuda()
            for i in range(0, zt.shape[0], 2048):
                d = torch.cdist(zt[i:i + 2048], bt)
                out.append(d.topk(min(KNN_K, bt.shape[0]),
                                  largest=False).values.mean(1).cpu().numpy())
        return lin, np.concatenate(out)

    # --- Confirmation: local vs contextual on identical patches ---
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        file_lin_loc, file_knn_loc = {}, {}
        file_lin_ctx, file_knn_ctx = {}, {}
        n_patches = 0
        for row in rows:
            sample = by_id[row["file_id"]]
            x = np.asarray(sample.x, dtype=np.float64)
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = np.flatnonzero(valid > 0)
            if keep.size == 0:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            loc, ctx = encode_both(patches[keep], pad[keep])
            n_patches += keep.size
            ll, lk = score_patches(loc, st_loc, cen_loc, bank_loc)
            cl, ck = score_patches(ctx, st_ctx, cen_ctx, bank_ctx)
            file_lin_loc[row["file_id"]] = float(ll.max())
            file_knn_loc[row["file_id"]] = float(lk.max())
            file_lin_ctx[row["file_id"]] = float(cl.max())
            file_knn_ctx[row["file_id"]] = float(ck.max())
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        arms = {"local_linear": file_lin_loc, "local_knn": file_knn_loc,
                "context_linear": file_lin_ctx, "context_knn": file_knn_ctx}

        def window_reader(score_map: dict[str, float]):
            neg = [E.window_score([m["file_id"] for m in w["members"]],
                                  score_map) for w in controls]

            def event_score(failure: dict) -> float | None:
                if E.positive_window_intersects_reset(failure, wins):
                    return None
                cands = E.pos_files(rows, failure, wins)
                if not cands:
                    return None
                return E.window_score([c["file_id"] for c in cands], score_map)
            return neg, event_score

        neg_map, ev_map = {}, {}
        for name, sm in arms.items():
            neg_map[name], ev_map[name] = window_reader(sm)
        cats: dict[str, dict] = {}
        for cat in CATEGORIES:
            cohort = cat[0] if len(cat) == 2 else cat
            subtype = cat if len(cat) == 2 else None
            cell: dict[str, object] = {}
            for name in arms:
                pos = [s for f in ledger
                       if f["cohort"] == cohort
                       and (subtype is None or f["subtype"] == subtype)
                       and (s := ev_map[name](f)) is not None]
                if not pos or not neg_map[name]:
                    cell[name] = {"reason": "no support",
                                  "n_pos": len(pos),
                                  "n_neg": len(neg_map[name])}
                else:
                    ci = _auc_ci(pos, neg_map[name])
                    cell[name] = {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in ci.items()}
            for pair in (("local_linear", "context_linear"),
                         ("local_knn", "context_knn")):
                a, b = pair
                if "point" in cell[a] and "point" in cell[b]:
                    cell[f"gap_{a}_vs_{b}"] = round(
                        cell[a]["point"] - cell[b]["point"], 4)
            cats[cat] = cell
        sev: dict[str, dict] = {}
        for cohort in ("P", "W"):
            pts = [(f["severity"], s) for f in ledger
                   if f["cohort"] == cohort
                   and (s := ev_map["context_linear"](f)) is not None]
            if len(pts) >= 3:
                lv = np.array([p[0] for p in pts])
                sc = np.array([p[1] for p in pts])
                sev[cohort] = {"rho_context_linear": round(
                    float(M.severity_spearman(sc, lv)), 4), "n": len(pts)}
            else:
                sev[cohort] = {"rho_context_linear": None, "n": len(pts)}
        per_history.append({"role": role, "seed": seed,
                            "n_files": len(rows),
                            "n_patches": int(n_patches),
                            "categories": cats, "severity": sev})
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
