"""Sprint 12 Task 2 — self-conditioned scorer shortcut diagnostics.

Loads the two accepted Sprint 11 checkpoints (control + hybrid) with
restored frozen references and measures, on real historical files:

- squared latent-minus-mean residual ``||z - mu||^2`` (per-patch, per-dim);
- predicted log-variance stats + fractions pinned at the clamp edges;
- energy decomposition (residual term vs log-variance term of the NLL);
- clean/corrupt latent and mean displacement, correlation, and tracking
  (does ``mu`` follow ``z`` under paired corruption?) across healthy,
  abnormal, and paired-corruption inputs;
- the same response on the independent population (Mahalanobis) signal.

Read-only w.r.t. checkpoints and references: eval mode, no grad, no refit.
Historical test files are a diagnostic benchmark here, not a sealed test.

Produces a bounded JSON summary + per-file CSV; row-level tensors stay in
memory and are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.data import collate_variable_files  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from representation.v2_objectives import synthesize_corrupted_patches  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

LOGVAR_LO, LOGVAR_HI = -6.0, 6.0
CLAMP_TOL = 0.01


# ---------------------------------------------------------------- pure helpers
def energy_terms(
    latents: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Split the Gaussian NLL into residual and log-variance terms."""
    var = logvar.exp()
    residual_term = 0.5 * ((latents - mean).pow(2) / var).sum(dim=-1)
    logvar_term = 0.5 * logvar.sum(dim=-1)
    return {
        "residual_term": residual_term,
        "logvar_term": logvar_term,
        "nll": residual_term + logvar_term,
    }


def clamp_fractions(logvar: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Fraction of valid patch-dims pinned at each clamp edge."""
    vals = logvar[valid]
    total = vals.numel()
    if total == 0:
        return {"n": 0, "frac_at_lo": 0.0, "frac_at_hi": 0.0}
    return {
        "n": int(total),
        "frac_at_lo": float((vals <= LOGVAR_LO + CLAMP_TOL).float().mean()),
        "frac_at_hi": float((vals >= LOGVAR_HI - CLAMP_TOL).float().mean()),
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ordinary_least_squares_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Slope/intercept/R^2 of y ~ x (used for mean-on-latent tracking)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or x.std() == 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (slope * x + intercept)
    r2 = float(1.0 - (resid.var() / y.var())) if y.var() > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2}


# ---------------------------------------------------------------- data handling
def select_files(by_id: dict, splits: dict, n_healthy: int, n_abnormal: int, seed: int):
    healthy_ids = sorted(splits["dev_val"])[: n_healthy // 2]
    healthy_ids += sorted(splits["dev_train"])[: n_healthy - len(healthy_ids)]
    static_ids = sorted(splits["test_static"])
    abnormal_ids: list[str] = []
    for fid in static_ids:
        if by_id[fid].file_label is SampleLabel.ABNORMAL:
            abnormal_ids.append(fid)
        if len(abnormal_ids) >= n_abnormal:
            break
    if len(abnormal_ids) < n_abnormal:
        raise RuntimeError(
            f"only {len(abnormal_ids)} abnormal static files; need {n_abnormal}"
        )
    gen = torch.Generator().manual_seed(seed)
    _ = gen  # selection is deterministic-by-id; generator reserved for masks
    return [by_id[i] for i in healthy_ids], [by_id[i] for i in abnormal_ids]


def make_batch(files, patchifier, device: str) -> dict:
    base = collate_variable_files(files, patchifier)
    count = base["patches"].shape[1]
    regimes = patch_regime_ids(files, base["starts"], count)
    return {
        "patches": base["patches"].to(device),
        "patch_pad_mask": base["patch_pad_mask"].to(device),
        "patch_valid_mask": base["patch_valid_mask"].to(device),
        "robot_idx": base["robot_idx"].to(device),
        "program_idx": base["program_idx"].to(device),
        "regime_ids": regimes.to(device),
        "file_ids": base["file_ids"],
    }

@torch.no_grad()
def encode(pipe: V2InferencePipeline, batch: dict) -> dict[str, torch.Tensor]:
    out = pipe.model(
        batch["patches"],
        batch["patch_pad_mask"],
        batch["patch_valid_mask"],
        batch["robot_idx"],
        batch["program_idx"],
        batch["regime_ids"],
    )
    pop = pipe.geometry.mixture_energy(
        out["patch_latents"],
        batch["patch_valid_mask"],
        batch["robot_idx"],
        batch["program_idx"],
        batch["regime_ids"],
    )
    return {
        "latents": out["patch_latents"].cpu(),
        "mean": out["cond_mean"].cpu(),
        "logvar": out["cond_logvar"].cpu(),
        "context_energy": out["context_energy"].cpu(),
        "population_energy": pop["population_energy"].cpu(),
        "valid": batch["patch_valid_mask"].cpu(),
    }


def cohort_stats(enc: dict, name: str) -> dict:
    valid = enc["valid"]
    z, mu, lv = enc["latents"], enc["mean"], enc["logvar"]
    resid_sq = (z - mu).pow(2)
    resid_sq_patch = resid_sq.sum(dim=-1)[valid]
    terms = energy_terms(z, mu, lv)
    resid_term = terms["residual_term"][valid]
    logvar_term = terms["logvar_term"][valid]
    nll = terms["nll"][valid]
    # mean-on-latent tracking: pool valid patch-dims
    ols = ordinary_least_squares_slope(
        z[valid].numpy(), mu[valid].numpy()
    )
    cos = torch.nn.functional.cosine_similarity(
        z[valid].float(), mu[valid].float(), dim=-1
    )
    return {
        "cohort": name,
        "n_patches": int(valid.sum()),
        "resid_sq_per_patch": describe(resid_sq_patch.numpy()),
        "resid_sq_per_dim_mean": float(resid_sq[valid].mean()),
        "residual_term": describe(resid_term.numpy()),
        "logvar_term": describe(logvar_term.numpy()),
        "nll_check_max_abs": float(
            (nll - enc["context_energy"][valid]).abs().max()
        ),
        "context_energy": describe(enc["context_energy"][valid].numpy()),
        "population_energy": describe(enc["population_energy"][valid].numpy()),
        "clamp": clamp_fractions(lv, valid.unsqueeze(-1).expand_as(lv)),
        "logvar_per_dim_mean": describe(lv[valid].mean(dim=0).numpy()),
        "mean_on_latent_ols": ols,
        "cosine_z_mu": describe(cos.numpy()),
    }


def paired_stats(
    pipe: V2InferencePipeline,
    batch: dict,
    severities: list[float],
    rate: float,
    seed: int,
) -> dict:
    clean = encode(pipe, batch)
    valid = clean["valid"]
    out: dict = {}
    for sev in severities:
        gen = torch.Generator().manual_seed(seed + int(round(sev * 1000)))
        mask = (
            torch.rand(valid.shape, generator=gen) < rate
        ) & valid
        corrupted = synthesize_corrupted_patches(
            batch["patches"],
            batch["patch_pad_mask"],
            valid.to(batch["patches"].device),
            mask.to(batch["patches"].device),
            float(sev),
            generator=torch.Generator().manual_seed(seed + int(round(sev * 1000))),
        )
        cb = dict(batch, patches=corrupted)
        corr = encode(pipe, cb)
        dz = (corr["latents"] - clean["latents"])[mask].pow(2).sum(dim=-1).sqrt()
        dm = (corr["mean"] - clean["mean"])[mask].pow(2).sum(dim=-1).sqrt()
        dz_n, dm_n = dz.numpy(), dm.numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(dm_n, dz_n, out=np.full_like(dm_n, np.nan), where=dz_n > 0)
        gap_ctx = (corr["context_energy"] - clean["context_energy"])[mask]
        gap_pop = (corr["population_energy"] - clean["population_energy"])[mask]
        bg = ~mask & valid
        bg_gap_ctx = (corr["context_energy"] - clean["context_energy"])[bg]
        out[f"severity_{sev}"] = {
            "n_masked_patches": int(mask.sum()),
            "latent_disp": describe(dz_n),
            "mean_disp": describe(dm_n),
            "tracking_ratio_mean_over_latent": describe(ratio),
            "tracking_ratio_median": float(np.nanmedian(ratio)) if ratio.size else float("nan"),
            "disp_correlation": float(pearson(dz_n, dm_n)),
            "resid_sq_clean": describe(
                (clean["latents"] - clean["mean"])[mask].pow(2).sum(dim=-1).numpy()
            ),
            "resid_sq_corrupt": describe(
                (corr["latents"] - corr["mean"])[mask].pow(2).sum(dim=-1).numpy()
            ),
            "context_energy_gap": describe(gap_ctx.numpy()),
            "population_energy_gap": describe(gap_pop.numpy()),
            "background_context_gap": describe(bg_gap_ctx.numpy()),
        }
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--control-ckpt", required=True)
    ap.add_argument("--hybrid-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-healthy", type=int, default=16)
    ap.add_argument("--n-abnormal", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--corruption-rate", type=float, default=0.25)
    ap.add_argument("--severities", default="1.0,2.0,4.0")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    severities = [float(s) for s in args.severities.split(",") if s.strip()]

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    manifest_sha = sha256_of(data_root / "manifest.json")
    samples, _ = load_chronological(data_root)
    by_id = {s.file_id: s for s in samples}
    healthy, abnormal = select_files(
        by_id, manifest["splits"], args.n_healthy, args.n_abnormal, args.seed
    )

    ckpts = {"control": Path(args.control_ckpt), "hybrid": Path(args.hybrid_ckpt)}
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    patchifier = Patchifier(PatchConfig())
    result: dict = {
        "provenance": {
            "commit": commit,
            "data_root": str(data_root),
            "manifest_sha256": manifest_sha,
            "config_hash": manifest.get("config_hash"),
            "checkpoints": {
                name: {"path": str(p), "sha256": sha256_of(p)}
                for name, p in ckpts.items()
            },
            "device": args.device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "seed": args.seed,
            "severities": severities,
            "corruption_rate": args.corruption_rate,
            "healthy_ids": [s.file_id for s in healthy],
            "abnormal_ids": [s.file_id for s in abnormal],
        },
        "checkpoints": {},
    }

    file_rows: list[dict] = []
    for name, path in ckpts.items():
        pipe = V2InferencePipeline.load(path, device=args.device)
        thr = getattr(pipe, "elevated_threshold", None)
        cal = getattr(pipe, "elevated_threshold_source", None)
        block: dict = {
            "restored_threshold": thr,
            "restored_threshold_source": cal,
            "cohorts": {},
            "paired": {},
        }
        for cohort_name, files in (("healthy", healthy), ("abnormal", abnormal)):
            encs = []
            for s in range(0, len(files), args.batch_size):
                encs.append(encode(pipe, make_batch(files[s : s + args.batch_size], patchifier, args.device)))
            # Batches carry different patch counts; pool flattened valid rows.
            flat = {k: [] for k in ("latents", "mean", "logvar", "context_energy", "population_energy")}
            for e in encs:
                v = e["valid"]
                flat["latents"].append(e["latents"][v])
                flat["mean"].append(e["mean"][v])
                flat["logvar"].append(e["logvar"][v])
                flat["context_energy"].append(e["context_energy"][v])
                flat["population_energy"].append(e["population_energy"][v])
            n_all = int(sum(t.shape[0] for t in flat["latents"]))
            enc = {
                "latents": torch.cat(flat["latents"]),
                "mean": torch.cat(flat["mean"]),
                "logvar": torch.cat(flat["logvar"]),
                "context_energy": torch.cat(flat["context_energy"]),
                "population_energy": torch.cat(flat["population_energy"]),
                "valid": torch.ones((n_all,), dtype=torch.bool),
            }
            block["cohorts"][cohort_name] = cohort_stats(enc, cohort_name)
            for s in range(0, len(files), args.batch_size):
                e = encs[s // args.batch_size]
                v = e["valid"]
                for j, f in enumerate(files[s : s + args.batch_size]):
                    vi = v[j]
                    r2 = (
                        (e["latents"][j][vi] - e["mean"][j][vi]).pow(2).sum(dim=-1).mean().item()
                        if int(vi.sum())
                        else float("nan")
                    )
                    file_rows.append(
                        {
                            "checkpoint": name,
                            "cohort": cohort_name,
                            "file_id": f.file_id,
                            "mean_resid_sq_per_patch": float(r2),
                            "mean_context_energy": float(e["context_energy"][j][vi].mean())
                            if int(vi.sum())
                            else float("nan"),
                            "mean_population_energy": float(e["population_energy"][j][vi].mean())
                            if int(vi.sum())
                            else float("nan"),
                        }
                    )
        for cohort_name, files in (("healthy", healthy), ("abnormal", abnormal)):
            sev_block = []
            for s in range(0, len(files), args.batch_size):
                sev_block.append(
                    paired_stats(
                        pipe,
                        make_batch(files[s : s + args.batch_size], patchifier, args.device),
                        severities,
                        args.corruption_rate,
                        args.seed,
                    )
                )
            # merge batch shards by concatenating counts (report per-severity pooled means)
            merged: dict = {}
            for sev in severities:
                key = f"severity_{sev}"
                shards = [b[key] for b in sev_block]
                merged[key] = {
                    "n_masked_patches": int(sum(s["n_masked_patches"] for s in shards)),
                    "latent_disp": _pool(shards, "latent_disp"),
                    "mean_disp": _pool(shards, "mean_disp"),
                    "tracking_ratio_median": float(
                        np.mean([s["tracking_ratio_median"] for s in shards])
                    ),
                    "disp_correlation": float(np.mean([s["disp_correlation"] for s in shards])),
                    "resid_sq_clean": _pool(shards, "resid_sq_clean"),
                    "resid_sq_corrupt": _pool(shards, "resid_sq_corrupt"),
                    "context_energy_gap": _pool(shards, "context_energy_gap"),
                    "population_energy_gap": _pool(shards, "population_energy_gap"),
                    "background_context_gap": _pool(shards, "background_context_gap"),
                }
            block["paired"][cohort_name] = merged
        result["checkpoints"][name] = block

    (out_dir / "shortcut_diag.json").write_text(json.dumps(result, indent=2))
    import csv

    with (out_dir / "per_file.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
        w.writeheader()
        w.writerows(file_rows)
    print(f"WROTE {out_dir / 'shortcut_diag.json'} + per_file.csv "
          f"({len(file_rows)} rows)")
    return 0


def _pool(shards: list[dict], key: str) -> dict:
    """Pooled mean-of-means across batch shards (bounded; shards equal-size)."""
    vals = ["mean", "median", "std", "min", "max"]
    return {
        v: float(np.mean([s[key][v] for s in shards if v in s[key]])) for v in vals
    } | {"n_shards": len(shards)}


if __name__ == "__main__":
    raise SystemExit(main())
