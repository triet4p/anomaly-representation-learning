"""Sprint 12 Task 3 — handcrafted vs frozen-learned conditional baselines.

Three matched arms share one pipeline: regularized hierarchical healthy-only
geometry (same hyperparameters as the accepted checkpoint config), dev-val-only
upper-tail calibration (fixed 0.05 rule), and fixed upper-tail file aggregation
with explicit energy identity. Only the patch features differ:

- handcrafted: interpretable observable descriptors (Task 3 feature set);
- learned-control / learned-hybrid: frozen patch latents of the accepted ckpts.

Plus as-shipped restored reference columns (both checkpoints, context and
population file tails) for continuity — no new fits there.

Historical Sprint 11 views are a diagnostic benchmark, not a sealed test.
Labels/masks are evaluation ground truth only (post-hoc).

Bounded outputs: task3_diag.json + per_file3.csv. No training, no refit of
accepted references. Provenance from live `git rev-parse HEAD` in this checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.data import collate_variable_files  # noqa: E402
from representation.handcrafted import batch_patch_features, feature_dim  # noqa: E402
from representation.v2_aggregation import (  # noqa: E402
    aggregate_file_state,
    calibrate_elevated_threshold_with_provenance,
)
from representation.v2_contracts import POPULATION_ENERGY_FIELD  # noqa: E402
from representation.v2_geometry import HierarchicalMahalanobisGeometry  # noqa: E402
from representation.v2_inference import (  # noqa: E402
    V2InferencePipeline,
    patch_regime_ids,
)
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

TAIL_PROBABILITY = 0.05


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


def single_batch(sample, patchifier, device: str) -> dict:
    base = collate_variable_files([sample], patchifier)
    count = base["patches"].shape[1]
    regimes = patch_regime_ids([sample], base["starts"], count)
    return {
        "patches": base["patches"].to(device),
        "patch_pad_mask": base["patch_pad_mask"].to(device),
        "patch_valid_mask": base["patch_valid_mask"].to(device),
        "robot_idx": base["robot_idx"].to(device),
        "program_idx": base["program_idx"].to(device),
        "regime_ids": regimes.to(device),
        "starts": base["starts"],
        "valid_len": base["valid_len"],
    }


@torch.no_grad()
def frozen_latents(pipe: V2InferencePipeline, batch: dict) -> torch.Tensor:
    out = pipe.model(
        batch["patches"], batch["patch_pad_mask"], batch["patch_valid_mask"],
        batch["robot_idx"], batch["program_idx"], batch["regime_ids"],
    )
    return out["patch_latents"].cpu()


def fit_geometry(
    rows: torch.Tensor, robot: torch.Tensor, program: torch.Tensor,
    regimes: torch.Tensor, cfg: dict,
) -> HierarchicalMahalanobisGeometry:
    geo = HierarchicalMahalanobisGeometry(
        rows.shape[1],
        shrinkage=cfg["shrinkage"],
        covariance_eps=cfg["covariance_eps"],
        min_group_samples=cfg["min_group_samples"],
        diag_min_samples=cfg["diag_min_samples"],
    )
    healthy = torch.ones((rows.shape[0],), dtype=torch.bool)
    geo.fit(rows, robot, program, regimes, healthy)
    return geo.frozen()


def file_tail(
    feats: torch.Tensor, energy: torch.Tensor, valid: torch.Tensor,
    regimes: torch.Tensor, threshold: float, top_q: float, n_regimes: int,
) -> float:
    state = aggregate_file_state(
        feats, energy, valid, regimes,
        energy_source=POPULATION_ENERGY_FIELD,
        top_q_fraction=top_q,
        elevated_threshold=threshold,
        n_regimes=n_regimes,
    )
    return float(state["tail_energy"][0])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranked = labels[order]
    n_pos = int(ranked.sum())
    n_neg = ranked.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum = np.cumsum(ranked)
    return float((cum[ranked == 0].sum()) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(float)
    n_pos = ranked.sum()
    if n_pos == 0 or n_pos == ranked.size:
        return float("nan")
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, ranked.size + 1)
    recall = tp / n_pos
    return float(np.trapezoid(precision, recall))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--control-ckpt", required=True)
    ap.add_argument("--hybrid-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    manifest_sha = sha256_of(data_root / "manifest.json")
    samples, _ = load_chronological(data_root)
    by_id = {s.file_id: s for s in samples}
    splits = manifest["splits"]
    train_files = [by_id[i] for i in sorted(splits["dev_train"])]
    val_files = [by_id[i] for i in sorted(splits["dev_val"])]
    static_files = [by_id[i] for i in sorted(splits["test_static"])]
    for s in train_files + val_files:
        if s.file_label is not SampleLabel.NORMAL:
            raise RuntimeError(f"development cohort must be verified-healthy: {s.file_id}")

    ckpts = {"control": Path(args.control_ckpt), "hybrid": Path(args.hybrid_ckpt)}
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
        if sha256_of(path) not in (
            "8bdb845b17a788ad39101e0f98c0654edb55727f0eec540ca44046b63dba2b7c",
            "76be843b0a5fa94c8cdd646e2734c804498f9c38088ed71c3f16f9aa97e2cc94",
        ):
            raise ValueError(f"{name} checkpoint hash is not an accepted Sprint 11 hash")

    pipes = {name: V2InferencePipeline.load(p, device=args.device) for name, p in ckpts.items()}
    base_cfg = pipes["control"].config
    geo_cfg = {
        "shrinkage": float(base_cfg.shrinkage),
        "covariance_eps": float(base_cfg.covariance_eps),
        "min_group_samples": int(base_cfg.min_group_samples),
        "diag_min_samples": int(base_cfg.diag_min_samples),
    }
    top_q = float(base_cfg.top_q_fraction)
    n_regimes = int(base_cfg.n_regimes)
    patchifier = Patchifier(PatchConfig(patch_size=int(base_cfg.patch_size), stride=int(base_cfg.stride)))

    arms = ["handcrafted", "learned-control", "learned-hybrid"]
    # --- featurize development views once ---------------------------------
    dev_feats: dict[str, list[torch.Tensor]] = {a: [] for a in arms}
    dev_aux: dict[str, list[torch.Tensor]] = {a: [] for a in arms}  # robot/program/regime rows
    for sample in train_files:
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"][0]
        n = v.shape[0]
        hc = torch.from_numpy(batch_patch_features(
            batch["patches"][0].cpu().numpy(), batch["patch_pad_mask"][0].cpu().numpy()
        )).float()
        assert hc.shape[1] == feature_dim(6), (hc.shape, feature_dim(6))
        lat_c = frozen_latents(pipes["control"], batch)[0]
        lat_h = frozen_latents(pipes["hybrid"], batch)[0]
        rr = batch["robot_idx"].expand(n)
        pp = batch["program_idx"].expand(n)
        gg = batch["regime_ids"][0]
        for key, mat in (("handcrafted", hc), ("learned-control", lat_c), ("learned-hybrid", lat_h)):
            dev_feats[key].append(mat[v])
            dev_aux[key].append(torch.stack([rr[v], pp[v], gg[v]], dim=1))

    geos, thresholds, threshold_prov = {}, {}, {}
    for arm in arms:
        rows = torch.cat(dev_feats[arm])
        aux = torch.cat(dev_aux[arm])
        geos[arm] = fit_geometry(rows, aux[:, 0], aux[:, 1], aux[:, 2], geo_cfg)
        # calibrate on dev-val healthy patch energies (fixed 0.05 rule)
        e_parts, v_parts = [], []
        for sample in val_files:
            batch = single_batch(sample, patchifier, args.device)
            v = batch["patch_valid_mask"]
            if arm == "handcrafted":
                mat = torch.from_numpy(batch_patch_features(
                    batch["patches"][0].cpu().numpy(), batch["patch_pad_mask"][0].cpu().numpy()
                )).float().unsqueeze(0)
            else:
                pipe = pipes["control"] if arm == "learned-control" else pipes["hybrid"]
                mat = frozen_latents(pipe, batch)
            e = geos[arm].population_energy(
                mat, v, batch["robot_idx"].cpu(), batch["program_idx"].cpu(),
                batch["regime_ids"].cpu(),
            )["population_energy"]
            e_parts.append(e)
            v_parts.append(v.cpu())
        ee = torch.cat([e.reshape(1, -1) for e in e_parts], dim=1)
        vv = torch.cat([v.reshape(1, -1) for v in v_parts], dim=1)
        thr, prov = calibrate_elevated_threshold_with_provenance(
            ee, vv, tail_probability=TAIL_PROBABILITY, cohort=f"dev-val ({arm})",
        )
        thresholds[arm] = thr
        threshold_prov[arm] = prov

    # --- score static benchmark --------------------------------------------
    rows_out: list[dict] = []
    for sample in static_files:
        batch = single_batch(sample, patchifier, args.device)
        v = batch["patch_valid_mask"]
        regimes = batch["regime_ids"].cpu()
        n = v.shape[1]
        hc = torch.from_numpy(batch_patch_features(
            batch["patches"][0].cpu().numpy(), batch["patch_pad_mask"][0].cpu().numpy()
        )).float()
        lat_c = frozen_latents(pipes["control"], batch)
        lat_h = frozen_latents(pipes["hybrid"], batch)
        mats = {
            "handcrafted": hc.unsqueeze(0),
            "learned-control": lat_c,
            "learned-hybrid": lat_h,
        }
        rec: dict = {
            "file_id": sample.file_id,
            "label": sample.file_label.value,
            "robot": int(sample.robot_idx),
            "program": int(sample.program_idx),
            "family": sample.anomaly_meta.family.value if sample.anomaly_meta else "normal",
            "severity": float(sample.anomaly_meta.severity) if sample.anomaly_meta else 0.0,
        }
        for arm in arms:
            e = geos[arm].population_energy(
                mats[arm], v.cpu(), batch["robot_idx"].cpu(),
                batch["program_idx"].cpu(), regimes,
            )
            rec[f"tail_{arm}"] = file_tail(
                mats[arm].cpu(), e["population_energy"], v.cpu(), regimes,
                thresholds[arm], top_q, n_regimes,
            )
            rec[f"flag_{arm}"] = bool(rec[f"tail_{arm}"] >= thresholds[arm])
        # as-shipped restored references (continuity, no new fits)
        for name, pipe in pipes.items():
            scored = pipe.score_patches(
                batch["patches"], batch["patch_pad_mask"], batch["patch_valid_mask"],
                batch["robot_idx"], batch["program_idx"], batch["regime_ids"],
            )
            rec[f"shipped_context_{name}"] = float(scored["file"]["tail_energy"][0])
            rec[f"shipped_population_{name}"] = float(scored["file_population"]["tail_energy"][0])
        # localization: top-1 energy patch vs anomaly mask (post-hoc)
        if sample.anomaly_mask is not None:
            ts_any = sample.anomaly_mask.any(axis=0)
            affected = Patchifier.timestep_mask_to_patch_mask(
                ts_any, batch["starts"][0].numpy(), batch["valid_len"][0].numpy(),
                sample.x.shape[0], sample.x.shape[1],
            )
            for arm in arms:
                e = geos[arm].population_energy(
                    mats[arm], v.cpu(), batch["robot_idx"].cpu(),
                    batch["program_idx"].cpu(), regimes,
                )["population_energy"][0].numpy()
                vv = v[0].numpy()
                e_masked = np.where(vv, e, -np.inf)
                top1 = int(np.argmax(e_masked))
                rec[f"lochit_{arm}"] = bool(affected[top1]) if vv[top1] else False
                rec[f"locgap_{arm}"] = float(
                    np.median(e[vv & affected]) - np.median(e[vv & ~affected])
                ) if affected.any() and (~affected & vv).any() else float("nan")
        rows_out.append(rec)

    labels = np.array([1.0 if r["label"] == "abnormal" else 0.0 for r in rows_out])
    score_cols = [f"tail_{a}" for a in arms] + [
        f"shipped_{s}_{n}" for s in ("context", "population") for n in ("control", "hybrid")
    ]
    metrics: dict = {}
    for col in score_cols:
        s = np.array([r[col] for r in rows_out])
        metrics[col] = {
            "auroc": auroc(s, labels),
            "auprc": auprc(s, labels),
            "median_normal": float(np.median(s[labels == 0])),
            "median_abnormal": float(np.median(s[labels == 1])),
        }
    for arm in arms:
        s = np.array([r[f"tail_{arm}"] for r in rows_out])
        dec = np.array([r[f"flag_{arm}"] for r in rows_out])
        metrics[f"tail_{arm}"].update({
            "threshold": thresholds[arm],
            "fpr_normal": float(dec[labels == 0].mean()),
            "recall_abnormal": float(dec[labels == 1].mean()),
            "lochit_rate": float(np.mean([r[f"lochit_{arm}"] for r in rows_out if f"lochit_{arm}" in r])),
            "locgap_median": float(np.nanmedian([r[f"locgap_{arm}"] for r in rows_out if f"locgap_{arm}" in r])),
        })
    # group slices (post-hoc)
    robots = sorted({r["robot"] for r in rows_out})
    fams = sorted({r["family"] for r in rows_out})
    slices = {"robot": {}, "family": {}}
    for rb in robots:
        idx = [i for i, r in enumerate(rows_out) if r["robot"] == rb]
        lab = labels[idx]
        slices["robot"][str(rb)] = {
            "n": len(idx), "n_abnormal": int(lab.sum()),
            **{c: auroc(np.array([rows_out[i][c] for i in idx]), lab) for c in score_cols[:3]},
        }
    for fm in fams:
        idx = [i for i, r in enumerate(rows_out) if r["family"] == fm]
        lab = labels[idx]
        slices["family"][fm] = {
            "n": len(idx),
            **{c: float(np.median([rows_out[i][c] for i in idx])) for c in score_cols[:3]},
        }

    result = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
            "data_root": str(data_root),
            "manifest_sha256": manifest_sha,
            "config_hash": manifest.get("config_hash"),
            "checkpoints": {n: {"path": str(p), "sha256": sha256_of(p)} for n, p in ckpts.items()},
            "device": args.device,
            "torch": torch.__version__,
            "geometry_hyperparams": geo_cfg,
            "top_q_fraction": top_q,
            "n_regimes": n_regimes,
            "tail_probability": TAIL_PROBABILITY,
            "primary_energy": "population_energy (per-group conditional Mahalanobis + fallback)",
            "primary_file_score": "tail_energy (fixed upper-tail rule)",
            "threshold_provenance": threshold_prov,
            "n_train": len(train_files),
            "n_val": len(val_files),
            "n_static": len(static_files),
        },
        "metrics": metrics,
        "slices": slices,
    }
    (out_dir / "task3_diag.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "per_file3.csv").open("w", newline="") as f:
        keys = list(rows_out[0].keys())
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    print(f"WROTE {out_dir / 'task3_diag.json'} + per_file3.csv ({len(rows_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
