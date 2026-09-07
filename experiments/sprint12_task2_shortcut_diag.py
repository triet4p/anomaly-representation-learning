"""Sprint 12 Task 2 — self-conditioned scorer shortcut diagnostics (corrected).

Loads the two accepted Sprint 11 checkpoints (control + hybrid) with
restored frozen references and measures, on real historical files:

- squared latent-minus-mean residual ``||z - mu||^2`` (per-patch, per-dim);
- predicted log-variance stats + fractions pinned at the clamp edges;
- energy decomposition (residual term vs log-variance term of the NLL);
- linear head-block inspection (latent block vs identity, context block norm);
- centered within-dimension latent/mean coupling (demeaned per-dim correlation
  and per-dim OLS slopes — not confounded by cross-dimension centroid offsets);
- query-exclusion interventions (zeroed/permuted query latent into the head with
  context fixed) that characterize shortcut freedom without assuming a cause;
- paired corruptions on a FIXED per-file mask with a FIXED per-file direction,
  scaled by severity, across healthy, abnormal, and paired-corruption inputs;
- directional response (cosine between latent and mean displacements) and a
  residual-attribution bound separating encoder invariance from mean cancellation;
- the same response on the independent population (Mahalanobis) signal.

Read-only w.r.t. checkpoints and references: eval mode, no grad, no refit.
Historical test files are a diagnostic benchmark here, not a sealed test.

Produces a bounded JSON summary + per-file CSV; row-level tensors stay in
memory and are never written. All aggregation pools actual patch rows; per-file
masks/directions derive deterministically from file ids, so results are
independent of batch partition (every file is encoded singly).

Provenance comes from ``git rev-parse HEAD`` in this checkout — this script
MUST run inside the canonical repository checkout, never from a copied tree.
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
# Same-checkout source first (this tree IS the verified commit, not a copy).
sys.path.insert(0, str(REPO_ROOT / "src"))

from representation.data import collate_variable_files  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
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
    """Slope/intercept/R^2 of y ~ x on pooled (possibly cross-dim) samples."""
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


def centered_within_dim_coupling(
    z: np.ndarray, mu: np.ndarray
) -> dict[str, float | int]:
    """Demean each latent dim, then correlate/slope-pool within dimensions.

    Pooled cross-dimension statistics conflate between-dimension centroid offsets
    (manifold alignment) with dynamic coupling. Centering each dimension first
    removes the centroid contribution; what remains is within-dimension tracking.
    """
    z = np.asarray(z, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    assert z.shape == mu.shape and z.ndim == 2
    zc = z - z.mean(axis=0, keepdims=True)
    muc = mu - mu.mean(axis=0, keepdims=True)
    per_dim_corr: list[float] = []
    per_dim_slope: list[float] = []
    for d in range(z.shape[1]):
        c = pearson(zc[:, d], muc[:, d])
        if np.isfinite(c):
            per_dim_corr.append(c)
        fit = ordinary_least_squares_slope(zc[:, d], muc[:, d])
        if np.isfinite(fit["slope"]):
            per_dim_slope.append(fit["slope"])
    return {
        "n_dims": int(z.shape[1]),
        "n_dims_kept": int(len(per_dim_corr)),
        "centered_corr": describe(np.asarray(per_dim_corr)),
        "centered_slope": describe(np.asarray(per_dim_slope)),
        "centered_corr_pooled": pearson(zc.ravel(), muc.ravel()),
    }


def seed_from(*parts: str) -> int:
    """Stable 63-bit seed from string parts (partition/order independent)."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:16], 16) % (2**63)


def file_corruption_mask(
    n_patches: int, valid: np.ndarray, rate: float, file_id: str, seed: int
) -> np.ndarray:
    """Fixed per-file corruption mask (same support at every severity)."""
    gen = torch.Generator().manual_seed(seed_from(str(seed), file_id, "mask"))
    draw = torch.rand((n_patches,), generator=gen).numpy()
    return (draw < rate) & np.asarray(valid, dtype=bool)


def file_direction(
    n_channels: int, width: int, file_id: str, seed: int
) -> np.ndarray:
    """Fixed per-file perturbation direction (scaled by severity)."""
    gen = torch.Generator().manual_seed(seed_from(str(seed), file_id, "direction"))
    return torch.randn((n_channels, width), generator=gen).numpy()


# ---------------------------------------------------------------- data handling
def select_files(by_id: dict, splits: dict, n_healthy: int, n_abnormal: int):
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
    return [by_id[i] for i in healthy_ids], [by_id[i] for i in abnormal_ids]


def make_single_batch(sample, patchifier, device: str) -> dict:
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
        "file_id": sample.file_id,
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
@torch.no_grad()
def head_block_stats(pipe: V2InferencePipeline) -> dict:
    """Parameter-level identity-coupling inspection of the Gaussian head."""
    d = pipe.model.d_model
    out = {}
    for name in ("mean", "logvar"):
        layer = getattr(pipe.model.head, name)
        w = layer.weight.detach().cpu().double()
        b = layer.bias.detach().cpu().double()
        wz, wc = w[:, :d], w[:, d:]
        eye = torch.eye(d, dtype=torch.float64)
        out[name] = {
            "latent_block_fro": float(wz.norm()),
            "latent_block_minus_identity_fro": float((wz - eye).norm()),
            "latent_block_identity_relative": float((wz - eye).norm() / wz.norm()),
            "latent_block_diag_mean": float(torch.diag(wz).mean()),
            "latent_block_diag_std": float(torch.diag(wz).std()),
            "latent_block_offdiag_rms": float(
                (wz - torch.diag(torch.diag(wz))).pow(2).mean().sqrt()
            ),
            "context_block_fro": float(wc.norm()),
            "context_block_relative": float(wc.norm() / wz.norm()),
            "bias_norm": float(b.norm()),
        }
    return out




@torch.no_grad()
def query_interventions(
    pipe: V2InferencePipeline, batch: dict, enc: dict
) -> dict:
    """Target-exclusion probes: how much does the scorer use the query latent?

    Re-runs only the Gaussian head with the query-latent input zeroed or
    permuted (context fixed), scoring the ORIGINAL latent. Large energy shifts
    mean the reference depends on the query (self-conditioning); near-zero
    shifts mean shortcut freedom w.r.t. the query. Characterizes, never assumes.
    """
    model = pipe.model
    valid = batch["patch_valid_mask"]
    local = model.local(batch["patches"], batch["patch_pad_mask"])
    latents = model.context_encoder(local, valid)
    context = model._context_vectors(
        batch["robot_idx"], batch["program_idx"], batch["regime_ids"]
    )
    z = latents
    out: dict = {}
    n = z.shape[1]
    intact = enc["context_energy"][enc["valid"]].numpy()
    out["intact"] = intact
    valid0 = valid[0]
    zv = z[0][valid0]
    rolled_all = z.clone()
    rolled_all[0][valid0] = torch.roll(zv, shifts=int(zv.shape[0] // 2), dims=0)
    for tag, z_mod in (
        ("zeroed", torch.zeros_like(z)),
        ("rolled", rolled_all),
    ):
        params = model.head(z_mod, context)
        e = model._gaussian_nll(z, params["cond_mean"], params["cond_logvar"])
        e = e.masked_fill(~valid, 0.0).cpu()
        moved = e[enc["valid"]].numpy()
        out[tag] = {"abs_shift": np.abs(moved - intact), "moved": moved}
    return out


def cohort_stats(enc: dict, name: str) -> dict:
    valid = enc["valid"]
    z, mu, lv = enc["latents"], enc["mean"], enc["logvar"]
    resid_sq = (z - mu).pow(2)
    resid_sq_patch = resid_sq.sum(dim=-1)[valid]
    terms = energy_terms(z, mu, lv)
    resid_term = terms["residual_term"][valid]
    logvar_term = terms["logvar_term"][valid]
    nll = terms["nll"][valid]
    ols = ordinary_least_squares_slope(z[valid].numpy(), mu[valid].numpy())
    cos = torch.nn.functional.cosine_similarity(
        z[valid].float(), mu[valid].float(), dim=-1
    )
    centered = centered_within_dim_coupling(z[valid].numpy(), mu[valid].numpy())
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
        "pooled_mean_on_latent_ols": ols,
        "pooled_note": (
            "pooled cross-dimension fit describes manifold alignment, "
            "NOT dynamic copying; see centered_within_dim + head_blocks."
        ),
        "centered_within_dim": centered,
        "cosine_z_mu": describe(cos.numpy()),
    }


def corrupt_patches(
    patches: torch.Tensor,
    pad_mask: torch.Tensor,
    mask: np.ndarray,
    direction: np.ndarray,
    severity: float,
) -> torch.Tensor:
    """Scale the FIXED per-file direction on the FIXED mask by severity."""
    active = (
        torch.from_numpy(mask.astype(np.float32))
        .to(patches.device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    real = (~pad_mask).unsqueeze(2).to(patches.dtype)
    direction_t = (
        torch.from_numpy(direction.astype(np.float32))
        .to(device=patches.device, dtype=patches.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    corrupted = patches + float(severity) * direction_t * real * active
    return corrupted.masked_fill(pad_mask.unsqueeze(2), 0.0)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    """Live Git-derived execution provenance (fails loudly, never a CLI claim)."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "experiments", "src", "tests"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            f"refusing to run on a dirty source tree re experiments/src/tests:\n{dirty}"
        )
    return out


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
    ap.add_argument("--corruption-rate", type=float, default=0.25)
    ap.add_argument("--severities", default="1.0,2.0,4.0")
    args = ap.parse_args()

    commit = repo_commit()  # live provenance; raises outside a clean checkout
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    severities = [float(s) for s in args.severities.split(",") if s.strip()]

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    manifest_sha = sha256_of(data_root / "manifest.json")
    samples, _ = load_chronological(data_root)
    by_id = {s.file_id: s for s in samples}
    healthy, abnormal = select_files(
        by_id, manifest["splits"], args.n_healthy, args.n_abnormal
    )

    ckpts = {"control": Path(args.control_ckpt), "hybrid": Path(args.hybrid_ckpt)}
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    patchifier = Patchifier(PatchConfig())
    result: dict = {
        "provenance": {
            "commit": commit,
            "commit_source": "git rev-parse HEAD in execution checkout (dirty source tree refused)",
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
            "encoding": "one single-file batch per file (no batch-partition dependence)",
            "corruption": "fixed per-file mask+direction from file-id seeds; severity scales only",
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
            "head_blocks": head_block_stats(pipe),
            "cohorts": {},
            "paired": {},
            "interventions": {},
        }
        for cohort_name, files in (("healthy", healthy), ("abnormal", abnormal)):
            # Pool ACTUAL valid rows across all files (never averages of summaries).
            acc: dict[str, list[torch.Tensor]] = {
                k: [] for k in (
                    "latents", "mean", "logvar", "context_energy", "population_energy",
                )
            }
            for sample in files:
                batch = make_single_batch(sample, patchifier, args.device)
                e = encode(pipe, batch)
                v = e["valid"]
                acc["latents"].append(e["latents"][v])
                acc["mean"].append(e["mean"][v])
                acc["logvar"].append(e["logvar"][v])
                acc["context_energy"].append(e["context_energy"][v])
                acc["population_energy"].append(e["population_energy"][v])
                n_v = int(v.sum())
                file_rows.append(
                    {
                        "checkpoint": name,
                        "cohort": cohort_name,
                        "file_id": sample.file_id,
                        "n_valid_patches": n_v,
                        "mean_resid_sq_per_patch": float(
                            (e["latents"][v] - e["mean"][v]).pow(2).sum(dim=-1).mean()
                        ),
                        "mean_context_energy": float(e["context_energy"][v].mean()),
                        "mean_population_energy": float(
                            e["population_energy"][v].mean()
                        ),
                    }
                )
            n_all = int(sum(t.shape[0] for t in acc["latents"]))
            enc = {
                "latents": torch.cat(acc["latents"]),
                "mean": torch.cat(acc["mean"]),
                "logvar": torch.cat(acc["logvar"]),
                "context_energy": torch.cat(acc["context_energy"]),
                "population_energy": torch.cat(acc["population_energy"]),
                "valid": torch.ones((n_all,), dtype=torch.bool),
            }
            block["cohorts"][cohort_name] = cohort_stats(enc, cohort_name)
            # Query-exclusion interventions: pool ACTUAL per-patch rows across files.
            intact_parts: list[np.ndarray] = []
            moved_parts: dict[str, list[np.ndarray]] = {"zeroed": [], "rolled": []}
            shift_parts: dict[str, list[np.ndarray]] = {"zeroed": [], "rolled": []}
            for sample in files:
                batch = make_single_batch(sample, patchifier, args.device)
                e = encode(pipe, batch)
                iv = query_interventions(pipe, batch, e)
                intact_parts.append(iv["intact"])
                for tag in moved_parts:
                    moved_parts[tag].append(iv[tag]["moved"])
                    shift_parts[tag].append(iv[tag]["abs_shift"])
            intact_all = np.concatenate(intact_parts)
            block["interventions"][cohort_name] = {
                "intact_energy": describe(intact_all),
                "tags": {
                    tag: {
                        "abs_shift": describe(np.concatenate(shift_parts[tag])),
                        "corr_with_intact": pearson(
                            intact_all, np.concatenate(moved_parts[tag])
                        ),
                    }
                    for tag in moved_parts
                },
            }
        for cohort_name, files in (("healthy", healthy), ("abnormal", abnormal)):
            # Pool ACTUAL masked-patch rows across all files and severities share
            # one fixed mask+direction per file (severity scales only).
            pool: dict[str, list[np.ndarray]] = {
                k: [] for k in (
                    "dz", "dm", "ratio", "dircos",
                    "r2clean", "r2corr", "gap_ctx", "gap_pop", "bg_gap",
                )
            }
            n_masked = 0
            for sample in files:
                batch = make_single_batch(sample, patchifier, args.device)
                clean = encode(pipe, batch)
                n = batch["patches"].shape[1]
                valid_np = clean["valid"][0].numpy()
                mask = file_corruption_mask(
                    n, valid_np, args.corruption_rate, sample.file_id, args.seed
                )
                if not mask.any():
                    continue
                direction = file_direction(
                    batch["patches"].shape[2],
                    batch["patches"].shape[3],
                    sample.file_id,
                    args.seed,
                )
                for sev in severities:
                    corrupted = corrupt_patches(
                        batch["patches"], batch["patch_pad_mask"],
                        mask, direction, sev,
                    )
                    corr = encode(pipe, dict(batch, patches=corrupted))
                    m = torch.from_numpy(mask)
                    dz_v = (corr["latents"][0][m] - clean["latents"][0][m])
                    dm_v = (corr["mean"][0][m] - clean["mean"][0][m])
                    dz_n = dz_v.pow(2).sum(dim=-1).sqrt().numpy()
                    dm_n = dm_v.pow(2).sum(dim=-1).sqrt().numpy()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.divide(
                            dm_n, dz_n,
                            out=np.full_like(dm_n, np.nan), where=dz_n > 0,
                        )
                    dc = torch.nn.functional.cosine_similarity(
                        dz_v.float(), dm_v.float(), dim=-1
                    ).numpy()
                    key = f"severity_{sev}"
                    pool.setdefault(key + "_dz", []).append(dz_n)
                    pool.setdefault(key + "_dm", []).append(dm_n)
                    pool.setdefault(key + "_ratio", []).append(ratio)
                    pool.setdefault(key + "_dircos", []).append(dc)
                    pool.setdefault(key + "_r2clean", []).append(
                        (clean["latents"][0][m] - clean["mean"][0][m])
                        .pow(2).sum(dim=-1).numpy()
                    )
                    pool.setdefault(key + "_r2corr", []).append(
                        (corr["latents"][0][m] - corr["mean"][0][m])
                        .pow(2).sum(dim=-1).numpy()
                    )
                    pool.setdefault(key + "_gap_ctx", []).append(
                        (corr["context_energy"][0][m] - clean["context_energy"][0][m]).numpy()
                    )
                    pool.setdefault(key + "_gap_pop", []).append(
                        (corr["population_energy"][0][m] - clean["population_energy"][0][m]).numpy()
                    )
                    bg = ~m & clean["valid"][0]
                    pool.setdefault(key + "_bg", []).append(
                        (corr["context_energy"][0][bg] - clean["context_energy"][0][bg]).numpy()
                    )
                n_masked += int(mask.sum())
            merged: dict = {"n_masked_patches": n_masked}
            for sev in severities:
                key = f"severity_{sev}"
                dz = np.concatenate(pool[key + "_dz"])
                dm = np.concatenate(pool[key + "_dm"])
                ratio = np.concatenate(pool[key + "_ratio"])
                r2c = np.concatenate(pool[key + "_r2clean"])
                r2x = np.concatenate(pool[key + "_r2corr"])
                # Residual-attribution bound: flatness forced by tiny dz?
                # |D(r2)| <= 2||r|| (|dz|+|dm|) + (|dz|+|dm|)^2, evaluated per patch.
                r_norm = np.sqrt(r2c)
                step = dz + dm
                bound = 2 * r_norm * step + step**2
                obs = np.abs(r2x - r2c)
                merged[key] = {
                    "latent_disp": describe(dz),
                    "mean_disp": describe(dm),
                    "tracking_ratio": describe(ratio),
                    "directional_cosine_dz_dm": describe(
                        np.concatenate(pool[key + "_dircos"])
                    ),
                    "disp_correlation": pearson(dz, dm),
                    "resid_sq_clean": describe(r2c),
                    "resid_sq_corrupt": describe(r2x),
                    "resid_abs_change": describe(obs),
                    "resid_attribution_bound_median": float(np.median(bound)),
                    "resid_change_within_bound_frac": float(np.mean(obs <= bound + 1e-9)),
                    "context_energy_gap": describe(np.concatenate(pool[key + "_gap_ctx"])),
                    "population_energy_gap": describe(np.concatenate(pool[key + "_gap_pop"])),
                    "background_context_gap": describe(np.concatenate(pool[key + "_bg"])),
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


if __name__ == "__main__":
    raise SystemExit(main())
