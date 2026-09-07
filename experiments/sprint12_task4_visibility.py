"""Sprint 12 Task 4 — observable-signal visibility through production stages (corrected).

Paired per-file contrast trace on real historical abnormal files (diagnostic
benchmark, not a sealed test). The production chain is stated truthfully:

- slicing/padding: Patchifier windows raw telemetry into overlapping patches
  with zero-padding. It applies NO filtering, detrending, or normalization —
  S0→S1 differences reflect windowing plus the metric, never a transform stage.
- V2 inference applies NO normalization to patches (verified: score_patches
  passes raw patches; only LayerNorm inside LocalPatchEncoder).
- feature extraction: LocalPatchEncoder (conv + masked pooling + LayerNorm);
- contextual encoding: SequenceContextEncoder mixing across patches;
- scoring: Gaussian head (context) + frozen hierarchical geometry (population).

Stages (same CENTERED amplitude statistic at S0/S1: |x − file channel median|):

- S0 raw spans: centered amplitude gap (masked span vs same-file background),
  reported per family with absolute/two-sided effects;
- S0 cross-channel: corr-matrix distance of the masked span vs background, PLUS
  a matched within-normal null (deterministic same-length background segments)
  with effect, rank, and spread — no claim without it;
- S1 windowed patches: same centered amplitude on patches, affected vs unaffected;
- S2/S3 latents: CONDITIONAL healthy hierarchy (dev-train-only fits on local and
  context latents per checkpoint) scoring affected vs unaffected patches —
  controls operating context instead of a global centroid;
- S4 scoring: context + population energy gaps, affected vs unaffected.

Benign nuisance controls on healthy files use the COMPARABLE statistics
(S1 amplitude delta; S2/S3 conditional-energy deltas; S4 deltas).

No stage is declared the exact loss site unless the contrasts identify it;
causal uncertainty from accepted A3 is preserved.
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
from representation.v2_geometry import HierarchicalMahalanobisGeometry  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import SampleLabel  # noqa: E402

NUISANCE = (
    ("gain_1.05", {"kind": "gain", "value": 1.05}),
    ("offset_0.1sigma", {"kind": "offset", "value": 0.1}),
    ("noise_0.1sigma", {"kind": "noise", "value": 0.1}),
)
N_NULL_SEGMENTS = 8


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


def seed_from(*parts: str) -> int:
    """Stable 63-bit seed from string parts (deterministic null placement)."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], 16) % (2**63)


def describe(values: np.ndarray) -> dict:
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


def single_batch(sample, patchifier, device: str, signal: np.ndarray | None = None) -> dict:
    if signal is not None:
        import dataclasses

        sample = dataclasses.replace(sample, x=np.asarray(signal, dtype=np.float32))
    base = collate_variable_files([sample], patchifier)
    count = base["patches"].shape[1]
    regimes = patch_regime_ids([sample], base["starts"], count)
    return {
        "sample": sample,
        "patches": base["patches"].to(device),
        "patch_pad_mask": base["patch_pad_mask"].to(device),
        "patch_valid_mask": base["patch_valid_mask"].to(device),
        "robot_idx": base["robot_idx"].to(device),
        "program_idx": base["program_idx"].to(device),
        "regime_ids": regimes.to(device),
        "starts": base["starts"][0].cpu().numpy(),
        "valid_len": base["valid_len"][0].cpu().numpy(),
    }


@torch.no_grad()
def stage_latents(pipe: V2InferencePipeline, batch: dict) -> dict:
    model = pipe.model
    valid = batch["patch_valid_mask"]
    local = model.local(batch["patches"], batch["patch_pad_mask"])
    latents = model.context_encoder(local, valid)
    context = model._context_vectors(
        batch["robot_idx"], batch["program_idx"], batch["regime_ids"]
    )
    params = model.head(latents, context)
    energy = model._gaussian_nll(latents, params["cond_mean"], params["cond_logvar"])
    energy = energy.masked_fill(~valid, 0.0).cpu()
    pop = pipe.geometry.mixture_energy(
        latents, valid, batch["robot_idx"], batch["program_idx"], batch["regime_ids"]
    )
    return {
        "local": local.cpu(),
        "context": latents.cpu(),
        "context_energy": energy,
        "population_energy": pop["population_energy"].cpu(),
        "valid": valid.cpu(),
    }

def centered_dev(x: np.ndarray) -> np.ndarray:
    """Same centered amplitude statistic everywhere: |x − file channel median|."""
    med = np.median(x, axis=1, keepdims=True)
    return np.abs(x - med)


def corr_matrix(block: np.ndarray) -> np.ndarray | None:
    """Sample cross-channel correlation, or None when degenerate."""
    if block.shape[1] < 8:
        return None
    c = np.corrcoef(block)
    return c if np.isfinite(c).all() else None


def corr_dist(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = corr_matrix(a), corr_matrix(b)
    if ca is None or cb is None:
        return float("nan")
    return float(np.linalg.norm(ca - cb))


def matched_null_dists(
    x: np.ndarray, span_start: int, span_len: int, file_id: str
) -> dict:
    """Matched within-normal null: deterministic same-length background segments.

    Draws N_NULL_SEGMENTS non-overlapping-with-span background segments of the
    same length, each scored against the remaining background. Returns the null
    distribution plus effect (observed − null median) and rank
    ((1 + #{null ≥ obs}) / (K + 1)). Unavailable when the background is short.
    """
    t = x.shape[1]
    span_end = min(t, span_start + span_len)
    bg = np.ones(t, dtype=bool)
    bg[span_start:span_end] = False
    bg_idx = np.flatnonzero(bg)
    if span_len < 8 or bg_idx.size < 2 * span_len:
        return {"null": [], "available": False}
    gen = torch.Generator().manual_seed(seed_from(file_id, "null-segments"))
    nulls: list[float] = []
    tries = 0
    while len(nulls) < N_NULL_SEGMENTS and tries < 4 * N_NULL_SEGMENTS:
        tries += 1
        s = int(torch.randint(0, max(1, bg_idx.size - span_len + 1), (1,), generator=gen))
        seg_idx = bg_idx[s:s + span_len]
        if seg_idx.size < 8:
            continue
        rest = np.ones(t, dtype=bool)
        rest[seg_idx] = False
        d = corr_dist(x[:, seg_idx], x[:, rest])
        if np.isfinite(d):
            nulls.append(d)
    return {"null": nulls, "available": len(nulls) >= 3}


def apply_nuisance(x: np.ndarray, kind: str, value: float, seed: int) -> np.ndarray:
    sig = x.std(axis=1, keepdims=True) + 1e-8
    if kind == "gain":
        return (x * float(value)).astype(np.float32)
    if kind == "offset":
        return (x + float(value) * sig).astype(np.float32)
    if kind == "noise":
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(x.shape, generator=gen).numpy().astype(np.float64)
        return (x + float(value) * sig * noise).astype(np.float32)
    raise ValueError(f"unknown nuisance {kind}")


def fit_conditional(
    train_files, pipes: dict, patchifier, device: str, stage: str
) -> dict:
    """Healthy conditional hierarchy per checkpoint on local/context latents."""
    out = {}
    for name, pipe in pipes.items():
        rows, aux = [], []
        for sample in train_files:
            batch = single_batch(sample, patchifier, device)
            st = stage_latents(pipe, batch)
            v = st["valid"][0]
            n = int(v.sum())
            rows.append(st[stage][0][v])
            cnt = int(v.shape[0])
            aux.append(torch.stack([
                batch["robot_idx"].cpu().expand(cnt)[v],
                batch["program_idx"].cpu().expand(cnt)[v],
                batch["regime_ids"][0].cpu()[v],
            ], dim=1))
        cfg = pipe.config
        geo = HierarchicalMahalanobisGeometry(
            32, shrinkage=float(cfg.shrinkage), covariance_eps=float(cfg.covariance_eps),
            min_group_samples=int(cfg.min_group_samples),
            diag_min_samples=int(cfg.diag_min_samples),
        )
        all_rows = torch.cat(rows)
        all_aux = torch.cat(aux)
        geo.fit(all_rows, all_aux[:, 0], all_aux[:, 1], all_aux[:, 2],
                torch.ones((all_rows.shape[0],), dtype=torch.bool))
        out[name] = geo.frozen()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--control-ckpt", required=True)
    ap.add_argument("--hybrid-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-nuisance", type=int, default=32)
    args = ap.parse_args()

    commit = repo_commit()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    manifest_sha = sha256_of(data_root / "manifest.json")
    samples, _ = load_chronological(data_root)
    by_id = {s.file_id: s for s in samples}
    splits = manifest["splits"]
    train_files = [by_id[i] for i in sorted(splits["dev_train"])]
    static_files = [by_id[i] for i in sorted(splits["test_static"])]
    abnormal = [s for s in static_files if s.file_label is SampleLabel.ABNORMAL]
    normal_static = [s for s in static_files if s.file_label is SampleLabel.NORMAL]
    if len(normal_static) < args.n_nuisance:
        raise RuntimeError("not enough normal static files for nuisance controls")
    nuisance_files = normal_static[: args.n_nuisance]

    ckpts = {"control": Path(args.control_ckpt), "hybrid": Path(args.hybrid_ckpt)}
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    pipes = {n: V2InferencePipeline.load(p, device=args.device) for n, p in ckpts.items()}
    base_cfg = pipes["control"].config
    patchifier = Patchifier(PatchConfig(patch_size=int(base_cfg.patch_size), stride=int(base_cfg.stride)))

    cond_local = fit_conditional(train_files, pipes, patchifier, args.device, "local")
    cond_context = fit_conditional(train_files, pipes, patchifier, args.device, "context")

    per_stage: dict[str, list[float]] = {}
    fam_s0: dict[str, list[float]] = {}
    corr_obs, corr_null_med, corr_effect, corr_rank = [], [], [], []
    null_unavailable = 0
    for sample in abnormal:
        batch = single_batch(sample, patchifier, args.device)
        x = sample.x
        dev = centered_dev(x)
        mask_any = sample.anomaly_mask.any(axis=0)
        gap_s0 = float(np.median(dev[:, mask_any]) - np.median(dev[:, ~mask_any])) \
            if mask_any.any() and (~mask_any).any() else float("nan")
        fam = sample.anomaly_meta.family.value if sample.anomaly_meta else "unknown"
        fam_s0.setdefault(fam, []).append(gap_s0)
        per_stage.setdefault("S0_amp_gap", []).append(gap_s0)
        per_stage.setdefault("S0_amp_abs", []).append(abs(gap_s0))
        # cross-channel observed + matched null
        span = np.flatnonzero(mask_any)
        s_len = int(span.size)
        s_start = int(span[0]) if s_len else 0
        bg = np.ones(x.shape[1], dtype=bool)
        bg[mask_any] = False
        obs = corr_dist(x[:, mask_any], x[:, bg]) if s_len >= 8 and bg.sum() >= 8 else float("nan")
        per_stage.setdefault("S0_corr_obs", []).append(obs)
        null = matched_null_dists(x, s_start, s_len, sample.file_id)
        if null["available"]:
            nulls = np.array(null["null"])
            corr_obs.append(obs)
            corr_null_med.append(float(np.median(nulls)))
            corr_effect.append(float(obs - np.median(nulls)))
            corr_rank.append(float((1 + int((nulls >= obs).sum())) / (len(nulls) + 1)))
        else:
            null_unavailable += 1
        # S1: same centered statistic on windowed patches
        pv = batch["patches"][0].cpu().numpy()
        pad = batch["patch_pad_mask"][0].cpu().numpy()
        med = np.median(x, axis=1, keepdims=True)
        centered_patches = np.abs(pv - med[None, :, :])
        centered_patches[np.broadcast_to(pad[:, None, :], centered_patches.shape)] = np.nan
        with np.errstate(all="ignore"):
            patch_amp = np.nanmean(centered_patches, axis=(1, 2))
        vv = batch["patch_valid_mask"][0].cpu().numpy()
        affected = Patchifier.timestep_mask_to_patch_mask(
            sample.anomaly_mask, batch["starts"], batch["valid_len"], x.shape[0], x.shape[1]
        )
        if affected.any() and (vv & ~affected).any():
            per_stage.setdefault("S1_patch_amp_gap", []).append(float(
                np.nanmedian(patch_amp[vv & affected]) - np.nanmedian(patch_amp[vv & ~affected])))
            per_stage.setdefault("S1_patch_amp_abs", []).append(float(abs(
                np.nanmedian(patch_amp[vv & affected]) - np.nanmedian(patch_amp[vv & ~affected]))))
        # S2/S3 conditional hierarchy gaps + S4 energy gaps
        for name, pipe in pipes.items():
            st = stage_latents(pipe, batch)
            mats = {"S2_local": st["local"], "S3_context": st["context"]}
            geos = {"S2_local": cond_local[name], "S3_context": cond_context[name]}
            for stage, mat in mats.items():
                e = geos[stage].population_energy(
                    mat, st["valid"], batch["robot_idx"].cpu(),
                    batch["program_idx"].cpu(), batch["regime_ids"].cpu(),
                )["population_energy"][0].numpy()
                per_stage.setdefault(f"{stage}_{name}", []).append(float(
                    np.median(e[vv & affected]) - np.median(e[vv & ~affected])))
            for sig in ("context_energy", "population_energy"):
                e = st[sig][0].numpy()
                per_stage.setdefault(f"S4_{sig}_{name}", []).append(float(
                    np.median(e[vv & affected]) - np.median(e[vv & ~affected])))

    # nuisance controls with COMPARABLE statistics
    nuisance_resp: dict[str, list[float]] = {}
    for sample in nuisance_files:
        x0 = sample.x
        med0 = np.median(x0, axis=1, keepdims=True)
        base = single_batch(sample, patchifier, args.device)
        vv = base["patch_valid_mask"][0].cpu().numpy()
        fid_seed = int(hashlib.sha256(sample.file_id.encode()).hexdigest()[:8], 16)
        for tag, spec in NUISANCE:
            x1 = apply_nuisance(x0, spec["kind"], spec["value"], seed=fid_seed)
            b1 = single_batch(sample, patchifier, args.device, signal=x1)
            d1 = np.abs(np.abs(b1["patches"][0].cpu().numpy() - med0[None, :, :]).mean(axis=(1, 2))
                       - np.abs(base["patches"][0].cpu().numpy() - med0[None, :, :]).mean(axis=(1, 2)))
            nuisance_resp.setdefault(f"S1_{tag}", []).append(float(np.median(d1[vv])))
            for name, pipe in pipes.items():
                st0 = stage_latents(pipe, base)
                st1 = stage_latents(pipe, b1)
                for stage, key, geo in (("S2_local", "local", cond_local[name]),
                                        ("S3_context", "context", cond_context[name])):
                    e0 = geo.population_energy(
                        st0[key], st0["valid"], base["robot_idx"].cpu(),
                        base["program_idx"].cpu(), base["regime_ids"].cpu(),
                    )["population_energy"][0].numpy()
                    e1 = geo.population_energy(
                        st1[key], st1["valid"], b1["robot_idx"].cpu(),
                        b1["program_idx"].cpu(), b1["regime_ids"].cpu(),
                    )["population_energy"][0].numpy()
                    nuisance_resp.setdefault(f"{stage}_{name}_{tag}", []).append(
                        float(np.median(np.abs(e1[vv] - e0[vv]))))
                for sig in ("context_energy", "population_energy"):
                    d = np.abs(st1[sig][0].numpy()[vv] - st0[sig][0].numpy()[vv])
                    nuisance_resp.setdefault(f"S4_{sig}_{name}_{tag}", []).append(float(np.median(d)))

    stages = {k: describe(np.array(v)) for k, v in per_stage.items()}
    stages["S0_corr_null_median"] = describe(np.array(corr_null_med))
    stages["S0_corr_effect_obs_minus_null"] = describe(np.array(corr_effect))
    stages["S0_corr_rank_empirical_p"] = describe(np.array(corr_rank))
    stages["S0_corr_null_unavailable"] = {"n": null_unavailable}
    nuisance = {k: describe(np.array(v)) for k, v in nuisance_resp.items()}
    fams = {k: {"gap": describe(np.array(v)),
                "abs": describe(np.abs(np.array(v)))} for k, v in fam_s0.items()}
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
            "n_abnormal": len(abnormal),
            "n_nuisance_files": len(nuisance_files),
            "nuisance_probes": [t for t, _ in NUISANCE],
            "conditional_fit": "dev-train-only hierarchy on local/context latents per checkpoint",
            "null": f"matched within-file same-length background segments (K={N_NULL_SEGMENTS}, deterministic)",
            "note": "masks/labels post-hoc only; patchifier is slicing/padding only (no filtering/normalization); V2 inference applies no patch normalization",
        },
        "stages": stages,
        "nuisance": nuisance,
        "family_S0": fams,
    }
    (out_dir / "task4_diag.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task4_diag.json'} ({len(abnormal)} abnormal + {len(nuisance_files)} nuisance files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
