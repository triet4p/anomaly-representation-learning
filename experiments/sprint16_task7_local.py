"""Sprint 16 Task 7 (M7): frozen local-latent readability vs observables.

Runs where the accepted checkpoints live (GPU server, verified commit):
loads the accepted control/hybrid V2 checkpoints (sha256 asserted before
load), extracts FROZEN local patch latents (LocalPatchEncoder only — no
context encoder, no scorer, no geometry, no fusion) for every valid
production patch of the Fit + Confirmation roots, and evaluates them with
the shared diagnostic reader beside post-patchification observables on the
IDENTICAL patches: linear reader (Fit-healthy centroid distance) and simple
nonlinear reader (kNN distance to a deterministic Fit-healthy bank, k=5).
File/event scores use the frozen per-event max; per history/category event
AUROC comes from the Task 3 contract reader. Compares latent vs observable
arms per physical family (P/W subtypes + severity behavior). No
context/scorer/geometry fusion, no bottleneck verdict (Task 16 owns
verdicts). Sealed roots are never touched (only FIT + CONFIRMATION paths).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task7_local.py \\
    --checkpoint control --data-root data/generated/sprint15-v7 \\
    --out /tmp/sprint16-task7
  --self-test runs synthetic checks without checkpoint/data.

Writes <out>/<checkpoint>/{metrics.json,run.log}. Latent banks stay
server-side only if --keep-latents is given (default off); only
metrics.json + run.log return. Deterministic single run per checkpoint
(fixed seeds, eval mode, no grad).
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
    """Synthetic checks of reader/aggregation logic (no checkpoint/data)."""
    import numpy as np

    from representation import attribution_metrics as M

    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(500, 32))
    abnormal = rng.normal(0, 1, size=(200, 32)) + np.array(
        [3.0] + [0.0] * 31)
    st = M.FrozenStandardizer.fit(healthy, source="SELFTEST")
    za = st.apply(abnormal)
    zh = st.apply(healthy)
    center = zh.mean(axis=0)
    da = np.linalg.norm(za - center, axis=1)
    dh = np.linalg.norm(zh - center, axis=1)
    auc = M.tie_auc(np.concatenate([da, dh]),
                    np.array([1.0] * 200 + [0.0] * 500))
    assert auc > 0.7, auc
    bank = zh[:: max(1, len(zh) // 1000)]
    import torch

    with torch.no_grad():
        d = torch.cdist(torch.asarray(za[:50], dtype=torch.float32),
                        torch.asarray(bank, dtype=torch.float32))
        k = d.topk(min(KNN_K, bank.shape[0]), largest=False).values.mean(1)
    assert bool((np.asarray(k) > 0).all())
    tok = st.token()
    assert M.assert_same_provenance(
        M.DiagnosticSet(history_id="s", ids=np.array(["a"]),
                        families={"x": np.array([1.0])},
                        labels=np.array([1.0]), provenance=tok),
        M.DiagnosticSet(history_id="s", ids=np.array(["a"]),
                        families={"y": np.array([2.0])},
                        labels=np.array([1.0]), provenance=tok)) == tok
    print(json.dumps({"self_test": "PASS", "auc": round(float(auc), 4),
                      "token": tok[:12]}))
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
    local_enc = pipe.model.local.eval()
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    @torch.no_grad()
    def encode_local(padded: np.ndarray, pad_mask: np.ndarray) -> np.ndarray:
        """Frozen LocalPatchEncoder on [N,C,W] + bool [N,W] pad mask."""
        pw = torch.asarray(padded, dtype=torch.float32).unsqueeze(0)
        pm = torch.asarray(pad_mask, dtype=torch.bool).unsqueeze(0)
        out = local_enc(pw.to(device), pm.to(device))
        if isinstance(out, dict):
            out = out["patch_latents"]
        return out.detach().cpu().numpy()[0].astype(np.float64)

    # --- Fit: healthy patch latents + patch observables, same patches ---
    fit_lat, fit_obs = [], []
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
            C, T = x.shape
            dt = (row["end_time"] - row["start_time"]) / T
            since_reset = row["end_time"] - row["last_reset_time"]
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = valid > 0
            if not keep.any():
                continue
            lat = encode_local(patches[keep], pad[keep])
            vecs = np.array([P.extract_features(
                patches[p][:, :int(valid[p])],
                int(valid[p]) * dt, since_reset) for p in np.flatnonzero(keep)])
            fit_lat.append(lat)
            fit_obs.append(vecs)
            fit_n += 1
    fit_lat = np.vstack(fit_lat).astype(np.float64)
    fit_obs = np.vstack(fit_obs).astype(np.float64)
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_lat.shape[0])})
    st_lat = M.FrozenStandardizer.fit(fit_lat, source="FIT-healthy-patches")
    st_obs = M.FrozenStandardizer.fit(fit_obs, source="FIT-healthy-patches")
    cen_lat = st_lat.apply(fit_lat).mean(axis=0)
    cen_obs = st_obs.apply(fit_obs).mean(axis=0)
    bank_lat = st_lat.apply(fit_lat)[:: max(1, fit_lat.shape[0] // BANK_MAX)]
    bank_obs = st_obs.apply(fit_obs)[:: max(1, fit_obs.shape[0] // BANK_MAX)]
    token = st_lat.token()
    log["steps"].append({"latent_dim": int(fit_lat.shape[1]),
                         "obs_dim": int(fit_obs.shape[1]),
                         "bank_lat": [int(v) for v in bank_lat.shape],
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

    # --- Confirmation: both arms on identical patches ---
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        file_lin_lat, file_knn_lat = {}, {}
        file_lin_obs, file_knn_obs = {}, {}
        n_patches = 0
        for row in rows:
            x = np.asarray(by_id[row["file_id"]].x, dtype=np.float64)
            C, T = x.shape
            dt = (row["end_time"] - row["start_time"]) / T
            since_reset = row["end_time"] - row["last_reset_time"]
            batch = patchifier.patchify(_wrap(x))
            patches = np.asarray(batch.patches, dtype=np.float64)
            valid = np.asarray(batch.valid_len, dtype=int)
            pad = np.asarray(batch.pad_mask, dtype=bool)
            keep = np.flatnonzero(valid > 0)
            if keep.size == 0:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            lat = encode_local(patches[keep], pad[keep])
            vecs = np.array([P.extract_features(
                patches[p][:, :int(valid[p])],
                int(valid[p]) * dt, since_reset) for p in keep])
            n_patches += keep.size
            ll, lk = score_patches(lat, st_lat, cen_lat, bank_lat)
            ol, ok = score_patches(vecs, st_obs, cen_obs, bank_obs)
            file_lin_lat[row["file_id"]] = float(ll.max())
            file_knn_lat[row["file_id"]] = float(lk.max())
            file_lin_obs[row["file_id"]] = float(ol.max())
            file_knn_obs[row["file_id"]] = float(ok.max())
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        arms = {"latent_linear": file_lin_lat, "latent_knn": file_knn_lat,
                "obs_linear": file_lin_obs, "obs_knn": file_knn_obs}

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
            for pair in (("latent_linear", "obs_linear"),
                         ("latent_knn", "obs_knn")):
                a, b = pair
                if "point" in cell[a] and "point" in cell[b]:
                    cell[f"gap_{a}_vs_{b}"] = round(
                        cell[b]["point"] - cell[a]["point"], 4)
            cats[cat] = cell
        sev: dict[str, dict] = {}
        for cohort in ("P", "W"):
            pts = [(f["severity"], s) for f in ledger
                   if f["cohort"] == cohort
                   and (s := ev_map["latent_linear"](f)) is not None]
            if len(pts) >= 3:
                lv = np.array([p[0] for p in pts])
                sc = np.array([p[1] for p in pts])
                sev[cohort] = {"rho_latent_linear": round(
                    float(M.severity_spearman(sc, lv)), 4), "n": len(pts)}
            else:
                sev[cohort] = {"rho_latent_linear": None, "n": len(pts)}
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
