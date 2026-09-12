"""Sprint 16 Task 12 (M8/C8/C9): scorer and aggregation readability.

FROZEN ARM MATRIX (fixed before any outcome; this docstring + the committed
driver are the freeze record — no protocol amendment: v3 §6 M8 plus v3 §8 C8
S-a/S-b and C9 top-4/median/p90 cover every arm; v4/v5 do not touch C8/C9).

Representations (frozen contextual latents, both accepted checkpoints) and
geometry (restored checkpoint bank) are HELD FIXED. S_pred/context-mismatch
and S_pop/population energies are NEVER fused — gaps stay within family.

S_pred family (frozen head outputs on contextual latents):
  P-tail      production S_pred path: context_energy -> production
              aggregate_file_state (config top_q + restored elevated
              threshold, tail_energy) -> per-event max reporting.
  P-abl-tail  S-a query ablation: head evaluated with the latent block
              zeroed pre-mixing (conditioning context intact), NLL scored at
              the TRUE latent; same production aggregation/reporting.
              Isolates self-conditioning (residual sensitivity).
  P-mse-tail  S-b energy form: patch-MSE ||z - cond_mean||^2 through the same
              production aggregation/reporting. Isolates the energy term
              (variance-scaled NLL vs raw MSE) vs P-tail.
  P-mse-max   S-b literal: patch-MSE with per-event max (protocol C8 holds
              aggregation at per-event max). Isolates aggregation vs P-mse-tail.
  P-max       context_energy with per-event max over pooled window patches.
              Isolates aggregation vs P-tail.
  P-top4      context_energy, event-window top-4 mean (k=4 frozen, C9).
  P-median    context_energy, event-window median, linear (C9).
  P-p90       context_energy, event-window 90th percentile, linear (C9).
S_pop family (restored-bank mixture_energy on the same frozen latents):
  G-tail      production S_pop path (reference for this family).
  G-max / G-top4 / G-median / G-p90  same C9 direct-window reductions.
Tail behavior = max/top4/p90/median/tail contrasts; duration dependence +
localized support = short/long + sparse/dense slices (one pooled P+W median
each, Task 10 form) on P-tail, P-max, G-tail, G-max; residual sensitivity =
P-abl-tail vs P-tail; energy components = NLL vs MSE (mixture-vs-single was
Task 11). C9 <8-valid-patch window exclusion with per-history counts.

Latents come from the production local/context encoder path; context_energy
and cond_mean/logvar come from the production head module on those latents.
S-a reuses the production head with masked input (no new weights); mixture
energies use the Task 11 batched-exact mirror (equality proven by that
task's --self-test at the same code lineage); production
aggregate_file_state is called directly. Deterministic single run per
checkpoint (fixed seeds, eval, no grad). No threshold fitting, no fusion,
no verdict (Task 16 owns verdicts). Labels/masks/categories are post-hoc
diagnostics only. Sealed roots never touched (FIT + CONFIRMATION only).
EXECUTABILITY SCOPE (pre-outcome): the checkpoint conditioning vocabulary
(n_robots x n_programs embeddings) covers only part of the evaluation
population, and the production head asserts on out-of-vocabulary indices.
S_pred arms therefore serve the in-vocabulary file subset only (counted
per history, never remapped — Task 9 v5 skip-rule precedent); S_pop arms
serve the full universe (Task 11 fallback rule). Identical support holds
within each family; the vocabulary shortfall itself is a Task 12 finding.

Usage (server, from the verified repo root):
  .venv/bin/python experiments/sprint16_task12_scorer.py \\
    --checkpoint control --data-root <roots> --out <dir> --device cuda
  --self-test runs synthetic checks without checkpoint/data.

Writes <out>/<checkpoint>/{metrics.json,run.log}. Latent banks stay
server-side; only metrics.json + run.log return.
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
P_ARMS = ("P-tail", "P-abl-tail", "P-mse-tail", "P-mse-max", "P-max",
          "P-top4", "P-median", "P-p90")
FIT = (("H-FIT-28", 1604), ("H-FIT-29", 1605), ("H-FIT-30", 1606))
CONF = (("H-CONF-34", 1608), ("H-CONF-35", 1609), ("H-CONF-36", 1610),
        ("H-CONF-37", 1611))
CATEGORIES = ["P1", "P2", "W1", "W2", "A1", "A2", "P", "W", "A"]
TOPK4 = 4  # frozen C9 tail width
MIN_WINDOW_PATCHES = 8  # frozen C9 exclusion floor
REF = {"P": "P-tail", "G": "G-tail"}  # gap denominators; families never fused
ARMS = ("P-tail", "P-abl-tail", "P-mse-tail", "P-mse-max", "P-max", "P-top4",
        "P-median", "P-p90", "G-tail", "G-max", "G-top4", "G-median", "G-p90")
TAIL_ARMS = ("P-tail", "P-abl-tail", "P-mse-tail", "G-tail")
ARM_BASE = {"P-max": ("nll", "max"), "P-top4": ("nll", "top4"),
            "P-median": ("nll", "median"), "P-p90": ("nll", "p90"),
            "P-mse-max": ("mse", "max"),
            "G-max": ("mix", "max"), "G-top4": ("mix", "top4"),
            "G-median": ("mix", "median"), "G-p90": ("mix", "p90")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_patches(a, method: str) -> float:
    """Frozen C9-form reductions over valid patch energies."""
    import numpy as np

    v = np.asarray(a, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("reduce_patches requires non-empty finite input")
    if method == "max":
        return float(v.max())
    if method == "top4":
        if v.size < TOPK4:
            raise ValueError(f"top4 needs >={TOPK4} patches")
        return float(np.partition(v, -TOPK4)[-TOPK4:].mean())
    if method == "median":
        return float(np.percentile(v, 50.0, method="linear"))
    if method == "p90":
        return float(np.percentile(v, 90.0, method="linear"))
    raise ValueError(f"unknown reduction {method!r}")


def self_test() -> int:
    """Synthetic checks of reductions, exclusion, S-a/MSE math (no data)."""
    import numpy as np
    import torch

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert reduce_patches(a, "max") == 9.0
    assert reduce_patches(a, "top4") == 7.5
    assert reduce_patches(a, "median") == 5.0
    assert abs(reduce_patches(a, "p90") - 8.2) < 1e-9
    for bad, meth in (([], "max"), ([1.0, np.inf], "max"),
                      ([1.0, 2.0, 3.0], "top4")):
        try:
            reduce_patches(bad, meth)
        except ValueError:
            pass
        else:
            raise AssertionError("guards must refuse")
    try:
        reduce_patches(a, "p99")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown method must raise")
    from representation.v2_patch import PatchDistributionHead

    torch.manual_seed(0)
    head = PatchDistributionHead(8, 2, 4).eval()
    z = torch.randn(1, 5, 8)
    c = torch.randn(1, 5, 4)
    p0 = head(z, c)
    p1 = head(torch.zeros_like(z), c)
    assert not torch.allclose(p0["cond_mean"], p1["cond_mean"]), \
        "ablation must move cond_mean"
    mse = ((z - p0["cond_mean"]).pow(2)).sum(-1)
    nll = 0.5 * (((z - p0["cond_mean"]).pow(2) / p0["cond_logvar"].exp()
                  + p0["cond_logvar"]).sum(-1))
    assert torch.isfinite(mse).all() and torch.isfinite(nll).all()
    assert not torch.allclose(mse, nll), "MSE and NLL must differ"
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
    from representation.v2_aggregation import aggregate_file_state
    from representation.v2_geometry import HierarchicalMahalanobisGeometry
    from representation.v2_inference import V2InferencePipeline, patch_regime_ids
    from sprint16_task11_geometry import batched_mixture_energy
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
    cfg = pipe.config
    agg_cfg = {"top_q_fraction": float(cfg.top_q_fraction),
               "elevated_threshold": float(pipe.elevated_threshold),
               "elevated_source": "restored-operating-threshold",
               "n_regimes": int(cfg.n_regimes)}
    log["steps"].append({"aggregation_config": agg_cfg,
                         "arms": list(ARMS)})
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
    log["steps"].append({"restored_bank": bank})
    NR, NP = int(cfg.n_robots), int(cfg.n_programs)
    log["steps"].append({"conditioning_vocab": {"n_robots": NR,
                                                "n_programs": NP}})
    patchifier = Patchifier(PatchConfig())
    data_root = Path(args.data_root)

    def load_root(role: str, group: str):
        root = data_root / group / role
        manifest = json.loads((root / "manifest.json").read_text())
        samples, _ = load_chronological(root)
        return samples, manifest

    @torch.no_grad()
    def file_signals(sample, row, batch, keep) -> dict[str, np.ndarray]:
        """Frozen base patch signals + regimes + latents for one file.

        Conditioning path (head: NLL, query-ablation, MSE) executes ONLY for
        files whose training-time indices fall inside the checkpoint's
        fitted conditioning vocabulary (robot < n_robots and program <
        n_programs); out-of-range files get NaN head signals, are counted,
        and are NEVER remapped (Task 9 v5 skip-rule precedent). The
        geometry path (mixture) serves the full universe (Task 11 rule).
        """
        K = keep.size
        pw = torch.asarray(np.asarray(batch.patches, dtype=np.float64)[keep],
                           dtype=torch.float32).unsqueeze(0).to(device)
        pm = torch.asarray(np.asarray(batch.pad_mask, dtype=bool)[keep],
                           dtype=torch.bool).unsqueeze(0).to(device)
        vm = torch.ones((1, K), dtype=torch.bool).to(device)
        local = model.local(pw, pm)
        if isinstance(local, dict):
            local = local["patch_latents"]
        lat = model.context_encoder(local, vm)
        if isinstance(lat, dict):
            lat = lat["patch_latents"]
        lat = lat.detach().cpu()
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
                ctx = model._context_vectors(rb.to(device), pr.to(device),
                                             rg.to(device))
                params = model.head(lat.to(device), ctx)
                cm = params["cond_mean"].detach().cpu()
                lv = params["cond_logvar"].detach().cpu()
                zl = lat
                nll = (0.5 * (((zl - cm).pow(2) / lv.exp() + lv).sum(-1))
                       ).numpy()[0].astype(np.float64)
                mse = ((zl - cm).pow(2)).sum(-1).numpy()[0].astype(np.float64)
                ab = model.head(torch.zeros_like(lat.to(device)), ctx)
                abl = (0.5 * (((zl.to(device) - ab["cond_mean"]).pow(2)
                               / ab["cond_logvar"].exp()
                               + ab["cond_logvar"]).sum(-1))
                       ).detach().cpu().numpy()[0].astype(np.float64)
            except (RuntimeError, IndexError, ValueError):
                in_range = False
        lat_f = lat.reshape(K, -1)
        mix, _ = batched_mixture_energy(
            prod_geo, lat_f,
            rb.repeat_interleave(K), pr.repeat_interleave(K))
        return {"nll": np.asarray(nll, dtype=np.float64),
                "abl": np.asarray(abl, dtype=np.float64),
                "mse": np.asarray(mse, dtype=np.float64),
                "mix": mix.numpy().astype(np.float64),
                "regimes": rg.numpy()[0].astype(int),
                "lat": lat_f.numpy().astype(np.float32),
                "in_range": in_range}

    def tail_energy(lat1: np.ndarray, e1: np.ndarray, rg1: np.ndarray,
                    source: str) -> float:
        """Production file-aggregation scalar via aggregate_file_state."""
        st = aggregate_file_state(
            torch.as_tensor(np.asarray(lat1, dtype=np.float32),
                            dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(np.asarray(e1, dtype=np.float64),
                            dtype=torch.float32).unsqueeze(0),
            torch.ones((1, np.asarray(e1).size), dtype=torch.bool),
            torch.as_tensor(np.asarray(rg1, dtype=int),
                            dtype=torch.long).unsqueeze(0),
            energy_source=source,
            top_q_fraction=agg_cfg["top_q_fraction"],
            elevated_threshold=agg_cfg["elevated_threshold"],
            n_regimes=agg_cfg["n_regimes"])
        return float(st["tail_energy"][0].item())
    def arm_file_scores(sig: dict) -> dict[str, float]:
        """All 13 arms' file-level scores from one file's base signals."""
        lat1, rg = sig["lat"], sig["regimes"]

        if sig["in_range"]:
            p = {
                "P-tail": tail_energy(lat1, sig["nll"], rg, "context_energy"),
                "P-abl-tail": tail_energy(lat1, sig["abl"], rg, "context_energy"),
                "P-mse-tail": tail_energy(lat1, sig["mse"], rg, "context_energy"),
                "P-mse-max": reduce_patches(sig["mse"], "max"),
                "P-max": reduce_patches(sig["nll"], "max"),
                "P-top4": reduce_patches(sig["nll"], "top4"),
                "P-median": reduce_patches(sig["nll"], "median"),
                "P-p90": reduce_patches(sig["nll"], "p90"),
            }
        else:
            p = {a: float("nan") for a in P_ARMS}
        p.update({
            "G-tail": tail_energy(lat1, sig["mix"], rg, "population_energy"),
            "G-max": reduce_patches(sig["mix"], "max"),
            "G-top4": reduce_patches(sig["mix"], "top4"),
            "G-median": reduce_patches(sig["mix"], "median"),
            "G-p90": reduce_patches(sig["mix"], "p90"),
        })
        assert set(p) == set(ARMS), "arm coverage"
        return p

    # --- Fit: healthy base signals per file (background reference) ---
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
            fit_files[row["file_id"]] = file_signals(sample, row, batch, keep)
            fit_n += 1
    fit_ref = {f: arm_file_scores(s) for f, s in fit_files.items()}
    log["steps"].append({"fit_files": fit_n})

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
        patch_bank: dict[str, dict] = {}
        scores: dict[str, dict[str, float]] = {a: {} for a in ARMS}
        n_patches = 0
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        for row in rows:
            sample = by_id[row["file_id"]]
            batch = patchifier.patchify(_wrap(np.asarray(sample.x,
                                                         dtype=np.float64)))
            keep = np.flatnonzero(np.asarray(batch.valid_len, dtype=int) > 0)
            if keep.size == 0:
                raise ValueError(f"zero valid patches: {row['file_id']}")
            sig = file_signals(sample, row, batch, keep)
            n_patches += keep.size
            patch_bank[row["file_id"]] = sig
            for a, v in arm_file_scores(sig).items():
                scores[a][row["file_id"]] = v

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

        direct_excluded = 0
        direct_control_dropped = 0

        def direct_reader(base: str, method: str):
            """Negatives + event scorer over pooled window patches.

            S_pred bases (nll/abl/mse) pool in-range files only (head
            signals are NaN out of vocabulary); mix serves the universe.
            """
            only_in_range = base in ("nll", "abl", "mse")

            def members_ok(fid: str) -> bool:
                return (fid in patch_bank and
                        (not only_in_range or patch_bank[fid]["in_range"]))

            neg = []
            for w in controls:
                qual = [m["file_id"] for m in w["members"]
                        if members_ok(m["file_id"])]
                if not qual:
                    direct_control_dropped += 1
                    continue
                pool = np.concatenate([patch_bank[f][base] for f in qual])
                if pool.size < MIN_WINDOW_PATCHES:
                    direct_control_dropped += 1
                    continue
                neg.append(reduce_patches(pool, method))

            def event_score(failure: dict) -> float | None:
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
            return neg, event_score

        readers: dict[str, tuple] = {}
        tail_dropped = 0
        for a in TAIL_ARMS:
            neg, ev = window_reader(scores[a])
            tail_dropped = max(tail_dropped, window_reader.dropped)
            readers[a] = (neg, ev)
        for a, (base, method) in ARM_BASE.items():
            readers[a] = direct_reader(base, method)

        results: dict[str, dict] = {}
        for a in ARMS:
            neg, ev = readers[a]
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
            results[a] = cell
        for a in ARMS:
            fam = "P" if a.startswith("P-") else "G"
            if a == REF[fam]:
                continue
            base, other = results[REF[fam]], results[a]
            for cat in CATEGORIES:
                if "point" in base.get(cat, {}) and "point" in other.get(cat, {}):
                    other[f"gap_vs_{REF[fam]}_{cat}"] = round(
                        other[cat]["point"] - base[cat]["point"], 4)
        # slices on P-tail, P-max, G-tail, G-max (Task 10 splits)
        pw_events = [f for f in ledger if f["cohort"] in "PW"]
        dur_med = float(np.median([f["duration_d"] for f in pw_events]))
        sup_counts = {}
        for f in pw_events:
            cands = E.pos_files(rows, f, wins)
            sup_counts[f["failure_id"]] = sum(
                patch_bank[c["file_id"]]["nll"].size for c in cands
                if c["file_id"] in patch_bank)
        sup_med = float(np.median(list(sup_counts.values())))
        slice_defs = [
            ("short_duration", lambda f: f["duration_d"] < dur_med),
            ("long_duration", lambda f: f["duration_d"] >= dur_med),
            ("sparse_support", lambda f: sup_counts.get(f["failure_id"], 0) < sup_med),
            ("dense_support", lambda f: sup_counts.get(f["failure_id"], 0) >= sup_med),
        ]
        slice_arms = {k: readers[k] for k in ("P-tail", "P-max",
                                              "G-tail", "G-max")}
        slices: dict[str, dict] = {}
        for label, pred in slice_defs:
            cell = {}
            for key, (neg_s, ev_s) in slice_arms.items():
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
        # background / stability per arm (arm's own file-level reduction)
        bg_ids = [row["file_id"] for row in rows
                  if row["file_label"] == "normal" and not row["is_quarantined"]]
        background: dict[str, dict] = {}
        for a in ARMS:
            if a in TAIL_ARMS:
                bg = np.array([scores[a][i] for i in bg_ids if i in scores[a]])
                bg = bg[np.isfinite(bg)]
                ref = np.array([fit_ref[f][a] for f in fit_ref])
                ref = ref[np.isfinite(ref)]
            else:
                base, method = ARM_BASE[a]
                floor = TOPK4 if method == "top4" else 1
                bg = np.array([reduce_patches(patch_bank[i][base], method)
                               if patch_bank[i][base].size >= floor
                               and (base == "mix"
                                    or patch_bank[i]["in_range"])
                               else np.nan for i in bg_ids
                               if i in patch_bank])
                bg = bg[np.isfinite(bg)]
                ref = np.array([reduce_patches(fit_files[f][base], method)
                                if fit_files[f][base].size >= floor
                                and (base == "mix"
                                     or fit_files[f]["in_range"])
                                else np.nan for f in fit_files])
                ref = ref[np.isfinite(ref)]
            if bg.size == 0 or ref.size == 0:
                background[a] = {"reason": "no support"}
                continue
            sc = M.stability_ci(bg, ref)
            background[a] = {
                "median": round(float(np.median(bg)), 4),
                "n": int(bg.size), "n_ref": int(ref.size),
                "stability": {k: round(v, 4) for k, v in sc.items()}}
        per_history.append({
            "role": role, "seed": seed, "n_files": len(rows),
            "n_patches": int(n_patches),
            "n_in_range_files": int(sum(1 for s in patch_bank.values()
                                       if s["in_range"])),
            "n_out_of_vocab_files": int(sum(1 for s in patch_bank.values()
                                           if not s["in_range"])),
            "n_dropped_control_windows": int(tail_dropped),
            "n_direct_control_dropped": int(direct_control_dropped),
            "n_excluded_direct_windows": int(direct_excluded),
            "duration_median_d": dur_med,
            "support_median_patches": sup_med,
            "categories": results,
            "slices": slices,
            "background": background,
        })
    metrics = {"checkpoint": args.checkpoint, "sha256": digest,
               "aggregation_config": agg_cfg,
               "restored_bank": bank,
               "histories": per_history,
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
