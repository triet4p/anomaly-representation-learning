"""Sprint 16 Task 11 (M7/C7): conditional-geometry readability.

FROZEN COMPARISON (fixed before any outcome is inspected; this docstring +
the committed driver are the freeze record — no protocol amendment: v3 §6 M7
plus v3 §8 C7 G-a/G-b cover every arm below; v4/v5 do not touch C7):

  Families (3): oracle patch observables (synth.probe15.extract_features, the
  Task 7 observable side), frozen local latents, frozen contextual latents
  (both accepted checkpoints, eval, no grad).

  Arms per family (identical valid-patch support; ONE frozen file reduction —
  per-file mean of valid-patch energies — held fixed for every arm because
  patch-to-file aggregation is Task 12's boundary, not this task's):
    R-simple : global Fit-healthy centroid, Euclidean, on frozen-standardized
               features (Task 3 contract reader; G-a spirit).
    G-b      : production-form HierarchicalMahalanobisGeometry with collapsed
               conditioning (all ids zeroed) on the same Fit-healthy rows —
               global Mahalanobis; isolates covariance/scaling vs R-simple.
    G-hier   : production-form geometry refit on Fit-healthy family rows with
               true (robot, program, regime) conditioning, single-resolve
               population_energy; isolates centroids+hierarchy+fallback.
    G-mix    : same refit geometry, mixture_energy (the shipped S_pop patch
               scoring form); isolates mixture-density scaling vs G-hier.
    G-prod   : restored checkpoint geometry snapshot (the shipped bank),
               mixture_energy — learned families ONLY. Oracle family G-prod is
               NOT_RUN (no oracle-dim bank exists in any accepted checkpoint).

  Production per-patch loops are evaluated with batched-exact mirrors of the
  shipped formulas (same resolve rule, same solve, same logdet/logsumexp);
  --self-test asserts numeric equality vs the production methods on synthetic
  groups plus fallback-level parity.

Reports per history/category event AUROC (+ paired-bootstrap CI, B=2000 seed
20260202), gaps vs R-simple within family, fallback-level counts, mixture
component stats, bank rows/source, covariance provenance
(shrinkage/eps/min_group/diag_min from the checkpoint config; per-group n,
level, diag-vs-full flag), background/stability per arm, excluded support.
No representation/scorer redesign, no score fusion, no threshold tuning, no
verdict (Task 16 owns verdicts). Labels/masks/categories are post-hoc
diagnostics only. Sealed roots never touched (FIT + CONFIRMATION only).

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task11_geometry.py \\
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
FAMILIES = ("oracle", "local", "context")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Batched-exact mirrors of the shipped geometry formulas ----------------
# Same resolve rule, same linear solve, same logdet/logsumexp as
# src/representation/v2_geometry.py; batched over rows for tractable runtimes.
# Equality vs the production methods is asserted in self_test().

def _resolve_stats(geo, robot: int, program: int, regime: int):
    key = (int(robot), int(program), int(regime))
    group = geo._groups.get(key)
    if group is not None and group.n >= geo.min_group_samples:
        return group, "robot_program_regime"
    pair = geo._pair.get((int(robot), int(program)))
    if pair is not None and pair.n >= geo.min_group_samples:
        return pair, "robot_program"
    rb = geo._robot.get(int(robot))
    if rb is not None and rb.n >= geo.min_group_samples:
        return rb, "robot"
    assert geo._fleet is not None
    return geo._fleet, "fleet_low_confidence"


def batched_single_energy(geo, rows, robots, programs, regimes):
    """Exact batched mirror of population_energy (resolve + Mahalanobis)."""
    import torch

    out = torch.empty(rows.shape[0], dtype=torch.float32)
    levels = []
    order = {}
    for i in range(rows.shape[0]):
        st, lv = _resolve_stats(geo, int(robots[i]), int(programs[i]),
                                int(regimes[i]))
        levels.append(lv)
        order.setdefault((st.mu.data_ptr(), st.cov.data_ptr()), (st, []))[1].append(i)
    for st, idx in order.values():
        sub = rows[idx].to(dtype=torch.float32)
        diff = sub - st.mu.to(sub.device, sub.dtype)
        cov = st.cov.to(sub.device, sub.dtype)
        try:
            solved = torch.linalg.solve(cov, diff.T).T
        except (RuntimeError, AttributeError):
            diag = torch.diagonal(cov).clamp_min(1e-6)
            solved = diff / diag.unsqueeze(0)
        vals = (diff * solved).sum(dim=-1)
        vals = torch.where(torch.isfinite(vals), vals,
                           torch.full_like(vals, 1e6))
        out[idx] = vals.to(dtype=torch.float32)
    return out, levels


def batched_mixture_energy(geo, rows, robots, programs):
    """Exact batched mirror of mixture_energy (regime-mixture density)."""
    import math

    import torch

    d = rows.shape[1]
    const = d * math.log(2.0 * math.pi)
    out = torch.empty(rows.shape[0], dtype=torch.float32)
    n_comp = torch.zeros(rows.shape[0], dtype=torch.long)
    order = {}
    for i in range(rows.shape[0]):
        order.setdefault((int(robots[i]), int(programs[i])), []).append(i)
    for (robot, program), idx in order.items():
        comps = [(k, s) for k, s in geo._groups.items()
                 if k[0] == robot and k[1] == program and s.n >= 1]
        if not comps:
            st, _ = _resolve_stats(geo, robot, program, 0)
            comps = [((robot, program, 0), st)]
        total = sum(s.n for _, s in comps)
        sub = rows[idx].to(dtype=torch.float32)
        log_terms = []
        for _, st in comps:
            diff = sub - st.mu.to(sub.device, sub.dtype)
            cov = st.cov.to(sub.device, sub.dtype)
            try:
                solved = torch.linalg.solve(cov, diff.T).T
            except (RuntimeError, AttributeError):
                diag = torch.diagonal(cov).clamp_min(1e-6)
                solved = diff / diag.unsqueeze(0)
            d2 = (diff * solved).sum(dim=-1)
            sign, logdet = torch.linalg.slogdet(cov)
            logdet_v = float(logdet.item()) if sign.item() > 0 and math.isfinite(
                logdet.item()) else 0.0
            log_terms.append(math.log(st.n / total)
                             - 0.5 * (d2 + logdet_v + const))
        stacked = torch.stack(log_terms)
        vals = -(torch.logsumexp(stacked, dim=0))
        vals = torch.where(torch.isfinite(vals), vals,
                           torch.full_like(vals, 1e6))
        out[idx] = vals.to(dtype=torch.float32)
        n_comp[idx] = len(comps)
    return out, n_comp


def self_test() -> int:
    """Synthetic checks: batched mirrors equal production methods; guards."""
    import numpy as np
    import torch

    from representation.v2_geometry import HierarchicalMahalanobisGeometry

    torch.manual_seed(0)
    d = 4
    rng = np.random.default_rng(0)
    lat = torch.as_tensor(rng.normal(0, 1, size=(120, d)), dtype=torch.float32)
    rob = torch.as_tensor([0] * 60 + [1] * 60)
    pro = torch.as_tensor([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
    reg = torch.as_tensor(([0] * 15 + [1] * 15) * 4)
    hm = torch.ones(120, dtype=torch.bool)
    geo = HierarchicalMahalanobisGeometry(d, shrinkage=0.2,
                                          covariance_eps=1e-4,
                                          min_group_samples=8,
                                          diag_min_samples=32)
    geo.fit(lat, rob, pro, reg, hm)
    frozen = geo.frozen()
    q = torch.as_tensor(rng.normal(0.5, 1, size=(2, 5, d)), dtype=torch.float32)
    vm = torch.ones((2, 5), dtype=torch.bool)
    rr = torch.as_tensor([0, 1])
    pp = torch.as_tensor([1, 0])
    gg = torch.as_tensor([[0, 1, 0, 1, 2], [1, 0, 9, 1, 0]])
    prod_single = frozen.population_energy(q, vm, rr, pp, gg)["population_energy"]
    flat = q.reshape(10, d)
    b_single, b_levels = batched_single_energy(
        geo, flat, rr.repeat_interleave(5), pp.repeat_interleave(5),
        gg.reshape(10))
    assert torch.allclose(prod_single.reshape(10),
                          b_single, atol=1e-4), "single mirror mismatch"
    prod_levels = frozen.population_energy(q, vm, rr, pp,
                                           gg)["fallback_level"]
    assert [c for row in prod_levels for c in row] == b_levels, "level mismatch"
    prod_mix = frozen.mixture_energy(q, vm, rr, pp, gg)["population_energy"]
    b_mix, b_nc = batched_mixture_energy(geo, flat,
                                          rr.repeat_interleave(5),
                                          pp.repeat_interleave(5))
    assert torch.allclose(prod_mix.reshape(10), b_mix, atol=1e-3), \
        "mixture mirror mismatch"
    # collapsed-conditioning G-b: fit path is production code; the mirror must
    # match a manual solve with the resolved group stats.
    geo0 = HierarchicalMahalanobisGeometry(d, shrinkage=0.2,
                                           covariance_eps=1e-4,
                                           min_group_samples=8,
                                           diag_min_samples=32)
    z = torch.zeros(120, dtype=torch.long)
    geo0.fit(lat, z, z, z, hm)
    st0, lv0 = _resolve_stats(geo0, 0, 0, 0)
    assert lv0 == "robot_program_regime", "collapsed fit must resolve group"
    g0, _ = batched_single_energy(geo0, flat, torch.zeros(10, dtype=torch.long),
                                  torch.zeros(10, dtype=torch.long),
                                  torch.zeros(10, dtype=torch.long))
    man = ((flat - st0.mu) @ torch.linalg.inv(st0.cov)
           * (flat - st0.mu)).sum(dim=-1)
    assert torch.allclose(g0, man, atol=1e-3), "G-b global mismatch"
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


def _bank_provenance(geo, source: str) -> dict:
    snap = geo.snapshot()
    det = {"source": source, "d_model": snap["d_model"],
           "shrinkage": snap["shrinkage"],
           "covariance_eps": snap["covariance_eps"],
           "min_group_samples": snap["min_group_samples"],
           "diag_min_samples": snap["diag_min_samples"]}
    for key, label in (("groups", "n_groups"), ("pair", "n_pairs"),
                       ("robot", "n_robots")):
        det[label] = len(snap[key])
        det[f"{key}_n"] = sorted(v["n"] for v in snap[key].values())
    det["fleet_n"] = snap["fleet"]["n"]
    det["total_bank_rows"] = (sum(v["n"] for v in snap["groups"].values())
                              + sum(v["n"] for v in snap["pair"].values())
                              + sum(v["n"] for v in snap["robot"].values())
                              + snap["fleet"]["n"])
    diag_min = snap["diag_min_samples"]
    for key in ("groups", "pair", "robot"):
        det[f"{key}_diag_fallback"] = sum(1 for v in snap[key].values()
                                          if v["n"] < diag_min)
    return det


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
    from representation.v2_geometry import HierarchicalMahalanobisGeometry
    from representation.v2_inference import V2InferencePipeline, patch_regime_ids
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
    cfg = pipe.config
    geo_cfg = {"shrinkage": float(cfg.shrinkage),
               "covariance_eps": float(cfg.covariance_eps),
               "min_group_samples": int(cfg.min_group_samples),
               "diag_min_samples": int(cfg.diag_min_samples),
               "d_model": int(cfg.d_model),
               "n_robots": int(cfg.n_robots),
               "n_programs": int(cfg.n_programs),
               "n_regimes": int(cfg.n_regimes)}
    log["steps"].append({"geometry_config": geo_cfg})
    # Restored shipped bank (arm G-prod, learned families only).
    prod_snap = pipe.geometry._geometry.snapshot()
    prod_geo = HierarchicalMahalanobisGeometry(
        geo_cfg["d_model"], shrinkage=geo_cfg["shrinkage"],
        covariance_eps=geo_cfg["covariance_eps"],
        min_group_samples=geo_cfg["min_group_samples"],
        diag_min_samples=geo_cfg["diag_min_samples"])
    prod_geo.restore_snapshot(prod_snap)
    prod_bank = _bank_provenance(prod_geo, "restored-checkpoint-geometry")
    log["steps"].append({"restored_bank": prod_bank})
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    @torch.no_grad()
    def encode_both(padded: np.ndarray, pad_mask: np.ndarray):
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

    def file_features(sample, row, batch, keep) -> dict[str, np.ndarray]:
        """All three families on identical kept patches + conditioning ids."""
        patches = np.asarray(batch.patches, dtype=np.float64)[keep]
        pad = np.asarray(batch.pad_mask, dtype=bool)[keep]
        loc, ctx = encode_both(patches, pad)
        C, T = np.asarray(sample.x, dtype=np.float64).shape
        dt = (row["end_time"] - row["start_time"]) / T
        since_reset = row["end_time"] - row["last_reset_time"]
        valid = np.asarray(batch.valid_len, dtype=int)[keep]
        vecs = np.array([P.extract_features(
            patches[p][:, :int(valid[p])],
            int(valid[p]) * dt, since_reset) for p in range(keep.size)],
            dtype=np.float64)
        starts = torch.as_tensor(np.asarray(batch.starts,
                                            dtype=np.int64)[keep]).unsqueeze(0)
        regimes = patch_regime_ids([sample], starts,
                                   int(keep.size)).numpy()[0].astype(int)
        n = keep.size
        # Conditioning indices are the integer training-time indices carried
        # by the sample (manifest rows hold string codes only).
        cond = {"robot": np.full(n, int(sample.robot_idx)),
                "program": np.full(n, int(sample.program_idx)),
                "regime": regimes}
        return {"oracle": vecs, "local": loc, "context": ctx, "cond": cond}
    # --- Fit: healthy rows per family + refit production-form geometries ---
    fit_rows: dict[str, list] = {f: [] for f in FAMILIES}
    fit_cond: dict[str, list] = {f: [] for f in FAMILIES}
    fit_files: dict[str, dict] = {}
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
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                continue
            feats = file_features(sample, row, batch, keep)
            for f in FAMILIES:
                fit_rows[f].append(feats[f])
                fit_cond[f].append(feats["cond"])
            fit_files[row["file_id"]] = feats
            fit_n += 1
    fit_rows = {f: np.vstack(fit_rows[f]).astype(np.float64) for f in FAMILIES}
    for f in FAMILIES:
        fit_cond[f] = {k: np.concatenate([c[k] for c in fit_cond[f]])
                       for k in ("robot", "program", "regime")}
    stds, cens = {}, {}
    for f in FAMILIES:
        stds[f] = M.FrozenStandardizer.fit(fit_rows[f],
                                           source="FIT-healthy-patches")
        cens[f] = stds[f].apply(fit_rows[f]).mean(axis=0)
    token = stds["local"].token()
    # Refit production-form geometries (raw family space, true conditioning)
    # plus collapsed-conditioning G-b variants.
    geos, geos0, banks = {}, {}, {}
    for f in FAMILIES:
        d = fit_rows[f].shape[1]
        lat = torch.as_tensor(fit_rows[f], dtype=torch.float32)
        rb = torch.as_tensor(fit_cond[f]["robot"])
        pr = torch.as_tensor(fit_cond[f]["program"])
        rg = torch.as_tensor(fit_cond[f]["regime"])
        hm = torch.ones(lat.shape[0], dtype=torch.bool)
        g = HierarchicalMahalanobisGeometry(
            d, shrinkage=geo_cfg["shrinkage"],
            covariance_eps=geo_cfg["covariance_eps"],
            min_group_samples=geo_cfg["min_group_samples"],
            diag_min_samples=geo_cfg["diag_min_samples"])
        g.fit(lat, rb, pr, rg, hm)
        geos[f] = g
        banks[f] = _bank_provenance(g, "refit-Fit-healthy-production-form")
        z = torch.zeros(lat.shape[0], dtype=torch.long)
        g0 = HierarchicalMahalanobisGeometry(
            d, shrinkage=geo_cfg["shrinkage"],
            covariance_eps=geo_cfg["covariance_eps"],
            min_group_samples=geo_cfg["min_group_samples"],
            diag_min_samples=geo_cfg["diag_min_samples"])
        g0.fit(lat, z, z, z, hm)
        geos0[f] = g0
    log["steps"].append({"fit_files": fit_n,
                         "fit_patches": int(fit_rows["local"].shape[0]),
                         "family_dims": {f: int(fit_rows[f].shape[1])
                                         for f in FAMILIES},
                         "provenance_token": token,
                         "refit_banks": banks})

    ARMS = ("R-simple", "G-b", "G-hier", "G-mix", "G-prod")

    def patch_energies(fam: str, feats: np.ndarray, cond: dict) -> dict:
        """All arms' valid-patch energies for one file (mean reduction later)."""
        n = feats.shape[0]
        rows_t = torch.as_tensor(feats, dtype=torch.float32)
        rb = torch.as_tensor(cond["robot"])
        pr = torch.as_tensor(cond["program"])
        rg = torch.as_tensor(cond["regime"])
        z = torch.zeros(n, dtype=torch.long)
        e_simple = np.linalg.norm(stds[fam].apply(feats) - cens[fam], axis=1)
        e_b, _ = batched_single_energy(geos0[fam], rows_t, z, z, z)
        e_h, _ = batched_single_energy(geos[fam], rows_t, rb, pr, rg)
        e_m, _ = batched_mixture_energy(geos[fam], rows_t, rb, pr)
        out = {"R-simple": e_simple.astype(np.float64),
               "G-b": e_b.numpy().astype(np.float64),
               "G-hier": e_h.numpy().astype(np.float64),
               "G-mix": e_m.numpy().astype(np.float64)}
        if fam != "oracle":
            e_p, _ = batched_mixture_energy(prod_geo, rows_t, rb, pr)
            out["G-prod"] = e_p.numpy().astype(np.float64)
        return out

    # --- Confirmation ---
    per_history = []
    for role, seed in CONF:
        samples, manifest = load_root(role, "CONFIRMATION")
        assert manifest["seeds"]["health"] == seed, f"seed-match:{role}"
        assert manifest.get("protocol") == B.S15_PROTOCOL_V7, f"tag:{role}"
        by_id = {s.file_id: s for s in samples}
        rows = manifest["files"]
        wins = manifest["maintenance_windows"]
        ledger = E.failure_ledger(manifest)
        # scores[fam][arm][file_id], fixed mean reduction for every arm
        scores: dict[str, dict[str, dict[str, float]]] = {
            f: ({a: {} for a in ARMS} if f != "oracle"
                else {a: {} for a in ARMS if a != "G-prod"})
            for f in FAMILIES}
        n_patches = 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        fb_counts = {"robot_program_regime": 0, "robot_program": 0,
                     "robot": 0, "fleet_low_confidence": 0}
        fb_lowconf = 0
        fb_total = 0
        mix_ncomp: list[int] = []
        for row in rows:
            sample = by_id[row["file_id"]]
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            feats = file_features(sample, row, batch, keep)
            n_patches += keep.size
            for f in FAMILIES:
                en = patch_energies(f, feats[f], feats["cond"])
                for a, v in en.items():
                    scores[f][a][row["file_id"]] = float(np.mean(v))
            # fallback accounting on local refit geometry (representative;
            # identical resolve rule serves all refit families)
            rb = feats["cond"]["robot"]
            pr = feats["cond"]["program"]
            rg = feats["cond"]["regime"]
            for i in range(keep.size):
                _, lv = _resolve_stats(geos["local"], int(rb[i]),
                                       int(pr[i]), int(rg[i]))
                fb_counts[lv] += 1
                st, _ = _resolve_stats(geos["local"], int(rb[i]),
                                       int(pr[i]), int(rg[i]))
                fb_lowconf += int(st.low_confidence)
                fb_total += 1
            _, nc = batched_mixture_energy(
                geos["local"], torch.as_tensor(feats["local"],
                                              dtype=torch.float32),
                torch.as_tensor(rb), torch.as_tensor(pr))
            mix_ncomp.extend(nc.tolist())

        def window_reader(score_map: dict[str, float]):
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
        dropped_windows = 0
        for f in FAMILIES:
            for a in scores[f]:
                neg, ev = window_reader(scores[f][a])
                dropped_windows = max(dropped_windows, window_reader.dropped)
                cell: dict[str, dict] = {}
                for cat in CATEGORIES:
                    cohort = cat[0] if len(cat) == 2 else cat
                    subtype = cat if len(cat) == 2 else None
                    pos = [s for f_ in ledger
                           if f_["cohort"] == cohort
                           and (subtype is None or f_["subtype"] == subtype)
                           and (s := ev(f_)) is not None]
                    if not pos or not neg:
                        cell[cat] = {"reason": "no support",
                                     "n_pos": len(pos), "n_neg": len(neg)}
                    else:
                        ci = _auc_ci(pos, neg)
                        cell[cat] = {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in ci.items()}
                results[f"{f}_{a}"] = cell
        # geometry-readability gaps vs R-simple within family
        for f in FAMILIES:
            base = results[f"{f}_R-simple"]
            for a in scores[f]:
                if a == "R-simple":
                    continue
                other = results[f"{f}_{a}"]
                for cat in CATEGORIES:
                    if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                        other[f"gap_vs_R-simple_{cat}"] = round(
                            other[cat]["point"] - base[cat]["point"], 4)
        # background / stability per family+arm (file-level background
        # vs Fit-healthy file means under the SAME arm)
        fit_file_means: dict[str, dict[str, list[float]]] = {
            f: {a: [] for a in scores[f]} for f in FAMILIES}
        for feats in fit_files.values():
            for f in FAMILIES:
                en = patch_energies(f, feats[f], feats["cond"])
                for a, v in en.items():
                    fit_file_means[f][a].append(float(np.mean(v)))
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background: dict[str, dict] = {}
        for f in FAMILIES:
            for a in scores[f]:
                bg = np.array([scores[f][a][i] for i in bg_ids
                               if i in scores[f][a]])
                bg = bg[np.isfinite(bg)]
                ref = np.array(fit_file_means[f][a])
                if bg.size == 0 or ref.size == 0:
                    background[f"{f}_{a}"] = {"reason": "no support"}
                    continue
                sc = M.stability_ci(bg, ref)
                background[f"{f}_{a}"] = {
                    "median": round(float(np.median(bg)), 4),
                    "n": int(bg.size), "n_ref": int(ref.size),
                    "stability": {k: round(v, 4) for k, v in sc.items()}}
        import collections as _c

        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_patches": int(n_patches),
            "n_dropped_control_windows": int(dropped_windows),
            "fallback_counts": fb_counts,
            "fallback_low_confidence_rate": round(fb_lowconf / max(fb_total, 1), 4),
            "mixture_components": dict(_c.Counter(int(v) for v in mix_ncomp)),
            "categories": results,
            "background": background,
        })
    metrics = {"checkpoint": args.checkpoint, "sha256": digest,
               "geometry_config": geo_cfg,
               "restored_bank": prod_bank,
               "refit_banks": banks,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
