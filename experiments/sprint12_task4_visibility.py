"""Sprint 12 Task 4 — observable-signal visibility through production stages.

Paired per-file contrast trace across the actual production chain on real
historical abnormal files (diagnostic benchmark, not a sealed test):

- S0 raw telemetry: amplitude deviation vs file median + cross-channel
  correlation distance, masked span vs background (post-hoc masks only);
- S1 patches: patch amplitude of affected vs unaffected patches;
- S2 local latents: distance to the healthy centroid, affected vs unaffected;
- S3 context latents: same on the context-encoded latents;
- S4 scoring: context + population energy gaps, affected vs unaffected.

Plus benign nuisance controls on healthy files (gain/offset/noise within
normal variation): stage-wise response vs true-anomaly contrast.

Bounded outputs: task4_diag.json. No training, no refit, no redesign.
Provenance from live `git rev-parse HEAD` in this checkout.
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

NUISANCE = (
    ("gain_1.05", {"kind": "gain", "value": 1.05}),
    ("offset_0.1sigma", {"kind": "offset", "value": 0.1}),
    ("noise_0.1sigma", {"kind": "noise", "value": 0.1}),
)


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
        sample = _with_signal(sample, signal)
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
        "starts": base["starts"][0].numpy(),
        "valid_len": base["valid_len"][0].numpy(),
    }


def _with_signal(sample, signal: np.ndarray):
    import dataclasses

    return dataclasses.replace(sample, x=np.asarray(signal, dtype=np.float32))


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


def raw_contrast(x: np.ndarray, mask_any: np.ndarray) -> dict:
    """S0: amplitude deviation vs file median + cross-channel corr distance."""
    c, t = x.shape
    med = np.median(x, axis=1, keepdims=True)
    dev = np.abs(x - med)
    in_m = mask_any & np.ones(t, dtype=bool)
    out = {
        "amp_in": float(np.median(dev[:, in_m])) if in_m.any() else float("nan"),
        "amp_out": float(np.median(dev[:, ~in_m])) if (~in_m).any() else float("nan"),
        "mask_fraction": float(in_m.mean()),
    }
    span = np.flatnonzero(in_m)
    if span.size >= 8:
        seg = x[:, span[0]:span[-1] + 1]
        ref = x[:, ~in_m][:, : seg.shape[1]]
        if ref.shape[1] >= 8:
            ci = np.corrcoef(seg)
            co = np.corrcoef(ref)
            if np.isfinite(ci).all() and np.isfinite(co).all():
                out["corr_dist"] = float(np.linalg.norm(ci - co))
            else:
                out["corr_dist"] = float("nan")
        else:
            out["corr_dist"] = float("nan")
    else:
        out["corr_dist"] = float("nan")
    return out


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

    # healthy centroids per checkpoint (dev_train, unconditional — coarse trace)
    centroids: dict[str, dict[str, torch.Tensor]] = {}
    for name, pipe in pipes.items():
        locs, ctxs = [], []
        for sample in train_files:
            batch = single_batch(sample, patchifier, args.device)
            st = stage_latents(pipe, batch)
            v = st["valid"][0]
            locs.append(st["local"][0][v])
            ctxs.append(st["context"][0][v])
        centroids[name] = {
            "local": torch.cat(locs).mean(dim=0),
            "context": torch.cat(ctxs).mean(dim=0),
        }

    per_stage: dict[str, dict[str, list[float]]] = {}
    fam_contrast: dict[str, list[float]] = {}
    file_count = 0
    for sample in abnormal:
        batch = single_batch(sample, patchifier, args.device)
        x = sample.x
        mask_any = sample.anomaly_mask.any(axis=0)
        s0 = raw_contrast(x, mask_any)
        affected = Patchifier.timestep_mask_to_patch_mask(
            mask_any, batch["starts"], batch["valid_len"], x.shape[0], x.shape[1]
        )
        patch_amp = np.abs(batch["patches"][0].cpu().numpy()).mean(axis=(1, 2))
        vv = batch["patch_valid_mask"][0].cpu().numpy()
        gaps: dict[str, float] = {
            "S0_amp": s0["amp_in"] - s0["amp_out"],
            "S1_patch_amp": float(np.median(patch_amp[vv & affected]) - np.median(patch_amp[vv & ~affected]))
            if affected.any() and (vv & ~affected).any() else float("nan"),
        }
        if np.isfinite(s0.get("corr_dist", float("nan"))):
            gaps["S0_corr_dist"] = float(s0["corr_dist"])
        for name, pipe in pipes.items():
            st = stage_latents(pipe, batch)
            for stage, key in (("S2_local", "local"), ("S3_context", "context")):
                d = ((st[key][0] - centroids[name][key]) ** 2).sum(dim=-1).sqrt().numpy()
                gaps[f"{stage}_{name}"] = float(
                    np.median(d[vv & affected]) - np.median(d[vv & ~affected])
                ) if affected.any() and (vv & ~affected).any() else float("nan")
            for sig in ("context_energy", "population_energy"):
                e = st[sig][0].numpy()
                gaps[f"S4_{sig}_{name}"] = float(
                    np.median(e[vv & affected]) - np.median(e[vv & ~affected])
                ) if affected.any() and (vv & ~affected).any() else float("nan")
        gaps["mask_fraction"] = float(s0["mask_fraction"])
        fam = sample.anomaly_meta.family.value if sample.anomaly_meta else "unknown"
        fam_contrast.setdefault(fam, []).append(gaps.get("S1_patch_amp", float("nan")))
        for k, val in gaps.items():
            if k == "mask_fraction":
                continue
            per_stage.setdefault(k, {}).setdefault("values", []).append(val)
        file_count += 1

    # nuisance controls on healthy files
    nuisance_resp: dict[str, dict[str, list[float]]] = {}
    for sample in nuisance_files:
        x0 = sample.x
        base = single_batch(sample, patchifier, args.device)
        vv = base["patch_valid_mask"][0].cpu().numpy()
        amp0 = np.abs(base["patches"][0].cpu().numpy()).mean(axis=(1, 2))[vv]
        for tag, spec in NUISANCE:
            x1 = apply_nuisance(x0, spec["kind"], spec["value"], seed=hash(sample.file_id) % (2**31))
            b1 = single_batch(sample, patchifier, args.device, signal=x1)
            amp1 = np.abs(b1["patches"][0].cpu().numpy()).mean(axis=(1, 2))[vv]
            nuisance_resp.setdefault(f"S1_{tag}", []).append(float(np.median(np.abs(amp1 - amp0))))
            for name, pipe in pipes.items():
                st0 = stage_latents(pipe, base)
                st1 = stage_latents(pipe, b1)
                for sig in ("context_energy", "population_energy"):
                    d = np.abs(st1[sig][0].numpy()[vv] - st0[sig][0].numpy()[vv])
                    nuisance_resp.setdefault(f"S4_{sig}_{name}_{tag}", []).append(float(np.median(d)))

    stages = {k: describe(np.array(v["values"])) for k, v in per_stage.items()}
    nuisance = {k: describe(np.array(v)) for k, v in nuisance_resp.items()}
    fams = {k: describe(np.array(v)) for k, v in fam_contrast.items()}
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
            "note": "masks/labels post-hoc only; centroids unconditional dev-train (coarse trace)",
        },
        "stages": stages,
        "nuisance": nuisance,
        "family_S1": fams,
    }
    (out_dir / "task4_diag.json").write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_dir / 'task4_diag.json'} ({file_count} abnormal + {len(nuisance_files)} nuisance files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
