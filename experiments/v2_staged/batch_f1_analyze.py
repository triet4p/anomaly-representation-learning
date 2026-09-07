"""Sprint 11 Batch F1 server runner (Tasks 33-34).

Scores the frozen control/hybrid checkpoints over dev + sealed static views
at one verified commit and publishes bounded aggregates only. Patch latents,
per-row scores, masks, and bulk traces stay in memory on the server and are
never written to disk or transferred.

Outputs under <output-root>/ (all review-sized):
  run_manifest.json, task33_geometry.json, task33_slices.csv,
  task34_static.json, task34_slices.csv, figures/*.png, stdout log (caller).

Read-only inputs: chronological dataset, two frozen v2 checkpoints.
No training, no retuning, no threshold optimization on test, no temporal
view access (Task 35 owns it).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Resolved repository source: this script lives in experiments/v2_staged/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from representation.data import collate_variable_files  # noqa: E402
from representation.v2_batch_f1 import (  # noqa: E402
    anisotropy_ratio,
    confusion,
    describe,
    effective_rank,
    group_rates,
    health_bin,
    mask_overlap_fraction,
    rates_with_counts,
    severity_bin,
    summarize_deltas,
    top_tail_mass,
)
from representation.v2_checkpoint import load_v2_checkpoint  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from representation.v2_objectives import synthesize_corrupted_patches  # noqa: E402
from representation.v2_trajectory import TrajectoryTracker  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import RegimeType, SampleLabel  # noqa: E402

REGIME_NAMES: tuple[str, ...] = tuple(r.value for r in RegimeType)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()


def _regime_name(rid: int) -> str:
    idx = int(rid)
    return REGIME_NAMES[idx] if 0 <= idx < len(REGIME_NAMES) else f"regime-{idx}"


def _file_meta(sample) -> dict[str, object]:
    meta = sample.anomaly_meta
    episode = getattr(sample, "episode", None)
    health = getattr(sample, "health", None)
    operation = getattr(sample, "operation", None)
    start = getattr(operation, "start_time", None) if operation is not None else None
    return {
        "file_id": sample.file_id,
        "robot": str(getattr(sample, "robot_code", f"robot-{sample.robot_idx}")),
        "program": str(
            getattr(sample, "program_number", f"program-{sample.program_idx}")
        ),
        "robot_idx": int(sample.robot_idx),
        "program_idx": int(sample.program_idx),
        "abnormal": bool(sample.file_label is SampleLabel.ABNORMAL),
        "family": str(meta.family.value) if meta is not None else "normal",
        "severity": float(getattr(meta, "severity", 0.0))
        if meta is not None
        else 0.0,
        "health_value": float(getattr(health, "health_value", float("nan")))
        if health is not None
        else float("nan"),
        "episode_kind": str(getattr(episode, "kind", "none"))
        if episode is not None
        else "none",
        "start_time": float(start) if start is not None else float("nan"),
        "has_mask": bool(getattr(sample, "anomaly_mask", None) is not None),
    }


@torch.no_grad()
def score_variant(
    ckpt: Path,
    dev_train_files,
    dev_val_files,
    static_files,
    patchifier: Patchifier,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    """Score dev-train/dev-val/static views; row data stays in memory only."""
    pipeline = V2InferencePipeline.load(ckpt, device=device)
    payload = load_v2_checkpoint(ckpt, pipeline.model, expected_config=pipeline.config)
    tracker_state = payload.get("tracker")
    if not isinstance(tracker_state, dict):
        raise RuntimeError(f"Checkpoint {ckpt} carries no trajectory baseline.")
    files: list[dict[str, object]] = []
    latents_pool: list[np.ndarray] = []
    keys: dict[str, list[str]] = {
        "robot": [],
        "pair": [],
        "regime": [],
        "health": [],
        "family": [],
        "label": [],
    }
    fallback_counts: dict[str, int] = defaultdict(int)
    fallback_total = 0
    top_q = float(pipeline.config.top_q_fraction)
    width = int(pipeline.config.patch_size)

    def pass_files(samples, split: str) -> None:
        nonlocal fallback_total
        tracker = TrajectoryTracker(pipeline.config.d_model)
        tracker.load_state_dict(tracker_state)
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            base = collate_variable_files(chunk, patchifier)
            count = base["patches"].shape[1]
            regimes = patch_regime_ids(chunk, base["starts"], count)
            out = pipeline.score_patches(
                base["patches"],
                base["patch_pad_mask"],
                base["patch_valid_mask"],
                base["robot_idx"],
                base["program_idx"],
                regimes,
                tracker=tracker,
            )
            lat = out["patch"]["patch_latents"].cpu()
            energy = out["patch"]["population_energy"].cpu()
            valid = out["patch"]["patch_valid_mask"].cpu()
            starts = base["starts"].cpu()
            resp = pipeline.geometry.population_energy(
                lat,
                valid,
                base["robot_idx"].cpu(),
                base["program_idx"].cpu(),
                regimes.cpu(),
            )
            levels = resp["fallback_level"]
            file_state = out["file"]["file_state"].cpu()
            tail = out["file"]["tail_energy"].cpu()
            elevated = out["file"]["elevated_fraction"].cpu()
            mean_e = out["file"]["mean_energy"].cpu()
            quant = out["file"]["energy_quantiles"].cpu()
            conf = out["confidence"]["confidence"].cpu()
            disp = out["trajectory"]["displacement"].cpu()
            for k, sample in enumerate(chunk):
                info = _file_meta(sample)
                info["split"] = split
                valid_k = valid[k].numpy()
                energy_k = energy[k].numpy()
                lat_k = lat[k].numpy()
                valid_lat = lat_k[valid_k]
                reg_k = regimes[k][valid_k]
                if bool(valid_k.any()):
                    dom = _regime_name(int(reg_k.mode().values.item()))
                else:
                    dom = "none"
                info["dominant_regime"] = dom
                info["n_patches"] = int(valid_k.sum())
                info["tail_energy"] = float(tail[k].item())
                info["elevated_fraction"] = float(elevated[k].item())
                info["mean_energy"] = float(mean_e[k].item())
                info["quantiles"] = [float(v) for v in quant[k].tolist()]
                info["confidence"] = float(conf[k].item())
                info["displacement"] = float(disp[k].item())
                info["file_state"] = file_state[k].numpy().astype(np.float64)
                info["top_tail_mass"] = float(
                    top_tail_mass(energy_k, valid_k, top_q)
                )
                if info["has_mask"] and info["abnormal"]:
                    flagged = sample.anomaly_mask.any(axis=0).tolist()
                    order = np.argsort(energy_k[valid_k])[::-1]
                    valid_idx = np.nonzero(valid_k)[0]
                    starts_k = starts[k].tolist()
                    ranges = [
                        (starts_k[valid_idx[t]], starts_k[valid_idx[t]] + width)
                        for t in order[:3].tolist()
                    ]
                    hits = [mask_overlap_fraction(flagged, a, b) for a, b in ranges]
                    info["argmax_hit"] = bool(hits[0] > 0.0)
                    info["top3_overlap"] = float(sum(hits) / 3.0)
                else:
                    info["argmax_hit"] = None
                    info["top3_overlap"] = None
                files.append(info)
                if valid_lat.shape[0]:
                    latents_pool.append(valid_lat.astype(np.float64))
                    count_rows = valid_lat.shape[0]
                    keys["robot"].extend([str(info["robot"])] * count_rows)
                    keys["pair"].extend(
                        [f"{info['robot']}/{info['program']}"] * count_rows
                    )
                    hv = float(info["health_value"])
                    keys["health"].extend(
                        [health_bin(hv) if np.isfinite(hv) else "unknown"]
                        * count_rows
                    )
                    keys["family"].extend([str(info["family"])] * count_rows)
                    keys["label"].extend(
                        ["abnormal" if info["abnormal"] else "normal"] * count_rows
                    )
                    keys["regime"].extend(
                        [_regime_name(int(r)) for r in reg_k.tolist()]
                    )
                for j in range(valid.shape[1]):
                    if bool(valid[k, j].item()):
                        fallback_counts[str(levels[k][j])] += 1
                        fallback_total += 1

    pass_files(dev_train_files, "dev_train")
    pass_files(dev_val_files, "dev_val")
    pass_files(static_files, "static")
    all_lat = (
        np.concatenate(latents_pool, axis=0) if latents_pool else np.zeros((0, 32))
    )
    return {
        "pipeline": pipeline,
        "files": files,
        "all_latents": all_lat,
        "keys": keys,
        "fallback_counts": dict(fallback_counts),
        "fallback_total": int(fallback_total),
        "config": pipeline.config,
    }


@torch.no_grad()
def corrupt_probe_for_variant(
    pipeline: V2InferencePipeline,
    samples,
    patchifier: Patchifier,
    seed: int,
    batch_size: int,
    rate: float = 0.1,
    severity: float = 2.0,
) -> dict[str, object]:
    """Paired clean/corrupt energy ordering with identical seeded draws.

    Masks and perturbations derive from ``seed``-keyed generators advanced
    per batch, so both variants see identical corruption positions when run
    with the same seed (matched comparison).
    """
    device = pipeline.device
    gaps: list[float] = []
    n_masked = 0
    n_ordered = 0
    background: list[float] = []
    n_files = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        base = collate_variable_files(chunk, patchifier)
        count = base["patches"].shape[1]
        regimes = patch_regime_ids(chunk, base["starts"], count)
        patches = base["patches"].to(device)
        pad = base["patch_pad_mask"].to(device)
        valid = base["patch_valid_mask"].to(device)
        robot = base["robot_idx"].to(device)
        program = base["program_idx"].to(device)
        reg = regimes.to(device)
        clean = pipeline.model(patches, pad, valid, robot, program, reg)
        clean_lat = clean["patch_latents"]
        clean_e = pipeline.geometry.mixture_energy(
            clean_lat, valid, robot, program, reg
        )["population_energy"]
        mask_gen = torch.Generator().manual_seed(seed + start)
        mask = (
            torch.rand(valid.shape, generator=mask_gen) < rate
        ).to(device) & valid
        noise_gen = torch.Generator().manual_seed(seed + 999 + start)
        corrupted = synthesize_corrupted_patches(
            patches, pad, valid, mask, severity, generator=noise_gen
        )
        corrupt = pipeline.model(corrupted, pad, valid, robot, program, reg)
        corrupt_lat = corrupt["patch_latents"]
        corrupt_e = pipeline.geometry.mixture_energy(
            corrupt_lat, valid, robot, program, reg
        )["population_energy"]
        region = (mask & valid).cpu()
        gap = (corrupt_e - clean_e).cpu()
        gaps.extend(gap[region].tolist())
        n_masked += int(region.sum().item())
        n_ordered += int((gap[region] > 0.0).sum().item())
        outside = (valid & ~mask).cpu()
        diff = (clean_lat - corrupt_lat).cpu()
        per_file_bg = (diff[outside].pow(2).mean().item() if bool(outside.any()) else 0.0)
        background.append(float(per_file_bg))
        n_files += len(chunk)
    gap_arr = np.asarray(gaps, dtype=np.float64)
    return {
        "n_files": n_files,
        "n_masked_patches": n_masked,
        "ordering_rate": float(n_ordered / n_masked) if n_masked else 0.0,
        "gap": describe(gaps) if gaps else {"n": 0},
        "background_latent_mse": describe(background),
        "gap_positive_fraction": float((gap_arr > 0.0).mean()) if gaps else 0.0,
    }


def _pooled_by_mask(lat: np.ndarray, keys: list[str], want: str) -> np.ndarray:
    sel = np.array([k == want for k in keys])
    return lat[sel]


def within_between(
    lat: np.ndarray, keys: dict[str, list[str]], cap: int = 2000
) -> dict[str, object]:
    """Mean within/between Euclidean distances per axis (strided, capped)."""
    result: dict[str, object] = {}
    rows = np.asarray(lat, dtype=np.float64)
    n = rows.shape[0]
    if n < 4:
        return result
    for axis in ("robot", "pair", "regime", "label"):
        labs = np.array(keys[axis])
        pool = np.arange(n)[:: max(1, n // min(n, 4000))]
        within: list[float] = []
        between: list[float] = []
        step = max(1, len(pool) // cap)
        half = len(pool) // 2
        for a_pos in range(0, len(pool), step):
            a = int(pool[a_pos])
            b = int(pool[(a_pos + half) % len(pool)])
            if a == b:
                continue
            dist = float(np.linalg.norm(rows[a] - rows[b]))
            if labs[a] == labs[b]:
                within.append(dist)
            else:
                between.append(dist)
            if len(within) + len(between) >= cap:
                break
        result[axis] = {
            "within": describe(within),
            "between": describe(between),
            "ratio_between_within": (
                float(np.mean(between) / max(1e-12, np.mean(within)))
                if within and between
                else float("nan")
            ),
        }
    return result


def geometry_aggregates(scored: dict[str, object]) -> dict[str, object]:
    """Task 33 conditional geometry aggregates (bounded; no row transfer)."""
    pipeline: V2InferencePipeline = scored["pipeline"]  # type: ignore[assignment]
    lat = np.asarray(scored["all_latents"], dtype=np.float64)
    keys = scored["keys"]
    out: dict[str, object] = {}
    out["n_patches"] = int(lat.shape[0])
    out["d_model"] = int(lat.shape[1]) if lat.shape[0] else 0
    if lat.shape[0] < 2:
        raise RuntimeError("Geometry pool is empty; refusing vacuous aggregates.")
    out["effective_rank_all"] = float(effective_rank(lat))
    out["anisotropy_all"] = float(anisotropy_ratio(lat))
    for axis in ("robot", "pair", "regime", "health", "family", "label"):
        block: dict[str, object] = {}
        for key in sorted(set(keys[axis])):
            rows = _pooled_by_mask(lat, keys[axis], key)
            if rows.shape[0] < 2:
                block[key] = {"n": int(rows.shape[0])}
                continue
            block[key] = {
                "n": int(rows.shape[0]),
                "effective_rank": float(effective_rank(rows)),
                "anisotropy": float(anisotropy_ratio(rows)),
            }
        out[f"spectra_by_{axis}"] = block
    snap = pipeline.geometry._geometry.snapshot()  # noqa: SLF001 (same-process introspection)
    cond_all: list[float] = []
    level_cond: dict[str, list[float]] = defaultdict(list)
    for group in ("groups", "pair", "robot"):
        section = snap[group]
        assert isinstance(section, dict)
        for _key, stats in section.items():
            assert isinstance(stats, dict)
            cov = stats["cov"].to(dtype=torch.float64)
            try:
                cond = float(torch.linalg.cond(cov).item())
            except RuntimeError:
                cond = float("inf")
            cond_all.append(cond)
            level_cond[str(stats["level"])].append(cond)
    fleet = snap["fleet"]
    assert isinstance(fleet, dict)
    try:
        fleet_cond = float(torch.linalg.cond(fleet["cov"].to(dtype=torch.float64)).item())
    except RuntimeError:
        fleet_cond = float("inf")
    cond_all.append(fleet_cond)
    level_cond[str(fleet["level"])].append(fleet_cond)
    finite = [c for c in cond_all if np.isfinite(c)]
    if not finite:
        raise RuntimeError("All reference covariances are non-finite.")
    out["cov_conditioning"] = {
        "n_groups": len(cond_all),
        "median": float(np.median(finite)),
        "worst": float(max(finite)),
        "by_level": {
            level: {
                "n": len(v),
                "median": float(np.median([x for x in v if np.isfinite(x)]))
                if any(np.isfinite(x) for x in v)
                else float("nan"),
                "worst": float(max([x for x in v if np.isfinite(x)]))
                if any(np.isfinite(x) for x in v)
                else float("nan"),
            }
            for level, v in sorted(level_cond.items())
        },
    }
    total = int(scored["fallback_total"])
    if total <= 0:
        raise RuntimeError("Fallback pool is empty.")
    out["fallback_coverage"] = {
        "total_patches": total,
        "levels": {
            level: {"n": int(n), "fraction": float(n / total)}
            for level, n in sorted(scored["fallback_counts"].items())
        },
    }
    groups = snap["groups"]
    assert isinstance(groups, dict)
    n_components: dict[str, int] = defaultdict(int)
    for key in groups:
        robot, program, _regime = (int(a) for a in str(key).split(","))
        n_components[f"robot-{robot:02d}/program-{program:02d}"] += 1
    out["mixture_components_per_pair"] = dict(sorted(n_components.items()))
    out["within_between"] = within_between(lat, keys)
    return out


def _split_rows(scored: dict[str, object], split: str) -> list[dict[str, object]]:
    files = scored["files"]
    assert isinstance(files, list)
    return [f for f in files if f["split"] == split]


def build_task33(
    control: dict[str, object],
    hybrid: dict[str, object],
    seed: int,
) -> dict[str, object]:
    """Conditional geometry comparison (no detection claims)."""
    geo_control = geometry_aggregates(control)
    geo_hybrid = geometry_aggregates(hybrid)
    retention = retention_probe(control, hybrid)
    continuity = continuity_probe(control, hybrid)
    severity_table = severity_energy(control, hybrid)
    corrupt_c = control.get("corrupt_probe", {})
    corrupt_h = hybrid.get("corrupt_probe", {})
    assert isinstance(corrupt_c, dict) and isinstance(corrupt_h, dict)
    return {
        "control": geo_control,
        "hybrid": geo_hybrid,
        "deltas": {
            "effective_rank_all": float(
                geo_hybrid["effective_rank_all"] - geo_control["effective_rank_all"]
            ),
            "anisotropy_all": float(
                geo_hybrid["anisotropy_all"] - geo_control["anisotropy_all"]
            ),
            "cov_median": float(
                geo_hybrid["cov_conditioning"]["median"]
                - geo_control["cov_conditioning"]["median"]
            ),
            "cov_worst": float(
                geo_hybrid["cov_conditioning"]["worst"]
                - geo_control["cov_conditioning"]["worst"]
            ),
        },
        "clean_corrupt_ordering": {
            "control": corrupt_c,
            "hybrid": corrupt_h,
            "matched": bool(
                corrupt_c.get("n_files") == corrupt_h.get("n_files")
                and corrupt_c.get("n_files", 0) > 0
            ),
            "protocol": (
                "Seeded random valid-patch subsets (rate 0.1, severity 2.0), "
                "identical seeds per batch for both variants (matched masks "
                "and perturbations); mixture-density energy gap on masked "
                "patches plus background latent stability outside the mask."
            ),
        },
        "sparse_retention": retention,
        "trajectory_continuity": continuity,
        "severity_energy": severity_table,
        "held_out_family_protocol": (
            "UNAVAILABLE — training synthesis applies family-agnostic random "
            "patch perturbations (v2_staged corruption_rate over valid patches), "
            "never conditions on real anomaly families and never reserves one; "
            "all eight test families are equally unseen by both variants, so no "
            "family slice may be relabeled as held-out evidence."
        ),
        "noncollapse_note": (
            "Effective rank / anisotropy / conditioning describe representation "
            "health only; they are not detection evidence and are never equated "
            "with separation here."
        ),
    }


def retention_probe(
    control: dict[str, object], hybrid: dict[str, object]
) -> dict[str, object]:
    """Tail-vs-mean sparse-evidence preservation on static abnormal files."""
    out: dict[str, object] = {}
    for name, scored in (("control", control), ("hybrid", hybrid)):
        dev = _split_rows(scored, "dev_train") + _split_rows(scored, "dev_val")
        static = _split_rows(scored, "static")
        thr_tail = float(np.quantile([f["tail_energy"] for f in dev], 0.95))
        thr_mean = float(np.quantile([f["mean_energy"] for f in dev], 0.95))
        abn = [f for f in static if f["abnormal"]]
        nor = [f for f in static if not f["abnormal"]]
        if not abn or not nor:
            raise RuntimeError("Static view lacks abnormal/normal files.")
        out[name] = {
            "dev_tail_p95": thr_tail,
            "dev_mean_p95": thr_mean,
            "tail_hit": rates_with_counts([f["tail_energy"] > thr_tail for f in abn]),
            "mean_hit": rates_with_counts([f["mean_energy"] > thr_mean for f in abn]),
            "tail_energy_abnormal": describe([f["tail_energy"] for f in abn]),
            "tail_energy_normal": describe([f["tail_energy"] for f in nor]),
            "mean_energy_abnormal": describe([f["mean_energy"] for f in abn]),
            "mean_energy_normal": describe([f["mean_energy"] for f in nor]),
            "top_tail_mass_abnormal": describe([f["top_tail_mass"] for f in abn]),
            "top_tail_mass_normal": describe([f["top_tail_mass"] for f in nor]),
        }
    return out


def continuity_probe(
    control: dict[str, object], hybrid: dict[str, object]
) -> dict[str, object]:
    """Successive-vs-random file-state distances per robot (dev+static order)."""
    out: dict[str, object] = {}
    rng = np.random.RandomState(0)
    for name, scored in (("control", control), ("hybrid", hybrid)):
        files = scored["files"]
        assert isinstance(files, list)
        timed = [f for f in files if np.isfinite(float(f["start_time"]))]
        if not timed:
            raise RuntimeError("No timestamped files for continuity.")
        by_robot: dict[str, list[dict[str, object]]] = defaultdict(list)
        for f in timed:
            by_robot[str(f["robot"])].append(f)
        per_robot = {}
        for robot in sorted(by_robot):
            ordered = sorted(by_robot[robot], key=lambda f: float(f["start_time"]))
            states = [np.asarray(f["file_state"], dtype=np.float64) for f in ordered]
            if len(states) < 3:
                per_robot[robot] = {"n": len(states)}
                continue
            successive = [
                float(np.linalg.norm(states[i + 1] - states[i]))
                for i in range(len(states) - 1)
            ]
            pairs = rng.choice(len(states), size=(min(len(states) * 2, 200), 2))
            random_pairs = [
                float(np.linalg.norm(states[int(a)] - states[int(b)]))
                for a, b in pairs
                if int(a) != int(b)
            ]
            per_robot[robot] = {
                "n": len(states),
                "successive": describe(successive),
                "random_pair": describe(random_pairs),
                "ratio_successive_random": float(
                    np.mean(successive) / max(1e-12, np.mean(random_pairs))
                )
                if random_pairs
                else float("nan"),
            }
        out[name] = per_robot
    out["note"] = (
        "Dev + static files ordered by operation start_time per robot; "
        "the temporal test view is untouched (Task 35)."
    )
    return out


def severity_energy(
    control: dict[str, object], hybrid: dict[str, object]
) -> dict[str, object]:
    """Mean tail energy by severity bin and family (static abnormal)."""
    out: dict[str, object] = {}
    for name, scored in (("control", control), ("hybrid", hybrid)):
        static = [f for f in _split_rows(scored, "static") if f["abnormal"]]
        by_sev: dict[str, list[float]] = defaultdict(list)
        by_fam: dict[str, list[float]] = defaultdict(list)
        for f in static:
            by_sev[severity_bin(float(f["severity"]))].append(float(f["tail_energy"]))
            by_fam[str(f["family"])].append(float(f["tail_energy"]))
        out[name] = {
            "by_severity": {k: describe(v) for k, v in sorted(by_sev.items())},
            "by_family": {k: describe(v) for k, v in sorted(by_fam.items())},
        }
    return out


def write_task33_slices(task33: dict[str, object], path: Path) -> None:
    """Spectra slices as CSV (bounded)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "axis", "key", "n", "effective_rank", "anisotropy"])
        for variant in ("control", "hybrid"):
            for axis in ("robot", "pair", "regime", "health", "family", "label"):
                block = task33[variant][f"spectra_by_{axis}"]
                assert isinstance(block, dict)
                for key in sorted(block):
                    row = block[key]
                    assert isinstance(row, dict)
                    writer.writerow(
                        [
                            variant,
                            axis,
                            key,
                            row.get("n", 0),
                            f"{float(row.get('effective_rank', float('nan'))):.4f}",
                            f"{float(row.get('anisotropy', float('nan'))):.4f}",
                        ]
                    )


def render_task33_figures(task33: dict[str, object], fig_dir: Path) -> None:
    """Representative geometry figures (bounded count)."""
    for axis in ("robot", "family"):
        labels = sorted(task33["control"][f"spectra_by_{axis}"])
        assert isinstance(labels, list)
        xc = np.arange(len(labels))
        width = 0.35
        c_vals = [
            float(task33["control"][f"spectra_by_{axis}"][k].get("effective_rank", float("nan")))
            for k in labels
        ]
        h_vals = [
            float(task33["hybrid"][f"spectra_by_{axis}"][k].get("effective_rank", float("nan")))
            for k in labels
        ]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
        ax.bar(xc - width / 2, c_vals, width, label="control")
        ax.bar(xc + width / 2, h_vals, width, label="hybrid")
        ax.set_xticks(xc)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("effective rank")
        ax.set_title(f"Effective rank by {axis} (control vs hybrid)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"task33_effrank_{axis}.png", dpi=120)
        plt.close(fig)


def build_task34(
    control: dict[str, object], hybrid: dict[str, object]
) -> dict[str, object]:
    """Sealed static-test comparison with validation-only calibration."""
    out: dict[str, object] = {}
    for name, scored in (("control", control), ("hybrid", hybrid)):
        static = _split_rows(scored, "static")
        dev_val_rows = _split_rows(scored, "dev_val")
        if not dev_val_rows:
            raise RuntimeError("Empty dev-val view; cannot calibrate.")
        val_conf = [float(f["confidence"]) for f in dev_val_rows]
        threshold = float(np.quantile(val_conf, 0.95))
        labels = [bool(f["abnormal"]) for f in static]
        scores = [float(f["confidence"]) for f in static]
        decisions = [s >= threshold for s in scores]
        stats = confusion(decisions, labels)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auroc = float(roc_auc_score(labels, scores))
            auprc = float(average_precision_score(labels, scores))
        except Exception:  # noqa: BLE001 (report NaN, never crash the gate)
            auroc, auprc = float("nan"), float("nan")
        tails = [float(f["tail_energy"]) for f in static]
        means = [float(f["mean_energy"]) for f in static]
        elev = [float(f["elevated_fraction"]) for f in static]
        try:
            from sklearn.metrics import average_precision_score as _ap
            from sklearn.metrics import roc_auc_score as _roc

            tail_auroc = float(_roc(labels, tails))
            tail_auprc = float(_ap(labels, tails))
        except Exception:  # noqa: BLE001
            tail_auroc, tail_auprc = float("nan"), float("nan")
        normals = [f for f in static if not f["abnormal"]]
        normal_flags = [bool(f["confidence"] >= threshold) for f in normals]
        fp_robot = group_rates([str(f["robot"]) for f in normals], normal_flags)
        fp_pair = group_rates(
            [f"{f['robot']}/{f['program']}" for f in normals], normal_flags
        )
        fp_regime = group_rates(
            [str(f["dominant_regime"]) for f in normals], normal_flags
        )
        abn = [f for f in static if f["abnormal"]]
        abn_flags = [bool(f["confidence"] >= threshold) for f in abn]
        fam = group_rates([str(f["family"]) for f in abn], abn_flags)
        sev = group_rates(
            [f"{f['family']}/{severity_bin(float(f['severity']))}" for f in abn],
            abn_flags,
        )
        masked = [f for f in abn if f["has_mask"]]
        argmax_hits = [
            bool(f["argmax_hit"]) for f in masked if f["argmax_hit"] is not None
        ]
        top3 = [
            float(f["top3_overlap"]) for f in masked if f["top3_overlap"] is not None
        ]
        out[name] = {
            "n_static": len(static),
            "n_abnormal": int(sum(labels)),
            "n_dev_val": len(dev_val_rows),
            "operating_threshold": threshold,
            "calibration_source": (
                "dev-val confidences only, 95th percentile (restored "
                "conformal calibrator maps displacement to confidence; test "
                "labels only score the operating point, never fit it)"
            ),
            "reference_source": "restored-checkpoint",
            "auroc": auroc,
            "auprc": auprc,
            "tail_auroc": tail_auroc,
            "tail_auprc": tail_auprc,
            **stats,
            "confidence_abnormal": describe(
                [s for s, y in zip(scores, labels) if y]
            ),
            "confidence_normal": describe(
                [s for s, y in zip(scores, labels) if not y]
            ),
            "tail_abnormal": describe([t for t, y in zip(tails, labels) if y]),
            "tail_normal": describe([t for t, y in zip(tails, labels) if not y]),
            "elevated_abnormal": describe([e for e, y in zip(elev, labels) if y]),
            "elevated_normal": describe([e for e, y in zip(elev, labels) if not y]),
            "mean_abnormal": describe([m for m, y in zip(means, labels) if y]),
            "mean_normal": describe([m for m, y in zip(means, labels) if not y]),
            "fp_by_robot": fp_robot,
            "fp_by_pair": fp_pair,
            "fp_by_regime": fp_regime,
            "recall_by_family": fam,
            "recall_by_family_severity": sev,
            "localization": {
                "n_masked_abnormal": len(masked),
                "argmax_hit": rates_with_counts(argmax_hits),
                "top3_overlap": describe(top3),
                "top_tail_mass_abnormal": describe(
                    [float(f["top_tail_mass"]) for f in abn]
                ),
            },
        }
    static_c = _split_rows(control, "static")
    static_h = _split_rows(hybrid, "static")
    if [f["file_id"] for f in static_c] != [f["file_id"] for f in static_h]:
        raise RuntimeError("Variant file order diverged; matched deltas refused.")
    out["deltas_hybrid_minus_control"] = {
        "confidence": summarize_deltas(
            np.array([float(f["confidence"]) for f in static_h])
            - np.array([float(f["confidence"]) for f in static_c])
        ),
        "tail_energy": summarize_deltas(
            np.array([float(f["tail_energy"]) for f in static_h])
            - np.array([float(f["tail_energy"]) for f in static_c])
        ),
        "mean_energy": summarize_deltas(
            np.array([float(f["mean_energy"]) for f in static_h])
            - np.array([float(f["mean_energy"]) for f in static_c])
        ),
        "auroc": float(out["hybrid"]["auroc"] - out["control"]["auroc"]),
        "auprc": float(out["hybrid"]["auprc"] - out["control"]["auprc"]),
        "f1": float(out["hybrid"]["f1"] - out["control"]["f1"]),
        "tp": int(out["hybrid"]["tp"] - out["control"]["tp"]),
        "fp": int(out["hybrid"]["fp"] - out["control"]["fp"]),
        "fn": int(out["hybrid"]["fn"] - out["control"]["fn"]),
        "tn": int(out["hybrid"]["tn"] - out["control"]["tn"]),
    }
    out["threshold_note"] = (
        "Operating point fixed on validation confidences only; no test-label "
        "tuning, no per-slice thresholds."
    )
    return out


def write_task34_slices(task34: dict[str, object], path: Path) -> None:
    """Family/severity/FP slices for both variants."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "slice", "kind", "n", "flagged", "rate"])
        for variant in ("control", "hybrid"):
            block = task34[variant]
            assert isinstance(block, dict)
            for slice_name, kind in (
                ("recall_by_family", "family"),
                ("recall_by_family_severity", "severity"),
                ("fp_by_robot", "fp_robot"),
                ("fp_by_pair", "fp_pair"),
                ("fp_by_regime", "fp_regime"),
            ):
                section = block[slice_name]
                assert isinstance(section, dict)
                for key in sorted(section):
                    row = section[key]
                    writer.writerow(
                        [
                            variant,
                            key,
                            kind,
                            row["n"],
                            row["flagged"],
                            f"{float(row['rate']):.4f}",
                        ]
                    )


def render_task34_figures(task34: dict[str, object], fig_dir: Path) -> None:
    """Representative static figures (bounded count)."""
    families = sorted(task34["control"]["recall_by_family"])
    xc = np.arange(len(families))
    width = 0.35
    c_vals = [float(task34["control"]["recall_by_family"][k]["rate"]) for k in families]
    h_vals = [float(task34["hybrid"]["recall_by_family"][k]["rate"]) for k in families]
    fig, ax = plt.subplots(figsize=(max(7, len(families) * 1.1), 4))
    ax.bar(xc - width / 2, c_vals, width, label="control")
    ax.bar(xc + width / 2, h_vals, width, label="hybrid")
    ax.set_xticks(xc)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("recall at validation operating point")
    ax.set_title("Static recall by anomaly family (validation-calibrated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "task34_recall_by_family.png", dpi=120)
    plt.close(fig)
    robots = sorted(task34["control"]["fp_by_robot"])
    xc = np.arange(len(robots))
    c_fp = [float(task34["control"]["fp_by_robot"][k]["rate"]) for k in robots]
    h_fp = [float(task34["hybrid"]["fp_by_robot"][k]["rate"]) for k in robots]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xc - width / 2, c_fp, width, label="control")
    ax.bar(xc + width / 2, h_fp, width, label="hybrid")
    ax.set_xticks(xc)
    ax.set_xticklabels(robots)
    ax.set_ylabel("false-positive rate (normal static files)")
    ax.set_title("False positives by robot (validation-calibrated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "task34_fp_by_robot.png", dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 11 Batch F1 analysis")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--control-ckpt", required=True)
    parser.add_argument("--hybrid-ckpt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", default="both", choices=("33", "34", "both"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-files", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 (documented aggregate sampling)

    data_root = Path(args.data_root).expanduser()
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    control_ckpt = Path(args.control_ckpt)
    hybrid_ckpt = Path(args.hybrid_ckpt)
    for path in (control_ckpt, hybrid_ckpt):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    samples, manifest = load_chronological(data_root)
    by_id = {s.file_id: s for s in samples}
    dev_train_ids = list(manifest["splits"]["dev_train"])
    dev_val_ids = list(manifest["splits"]["dev_val"])
    static_ids = list(manifest["splits"]["test_static"])
    if "test_temporal" not in manifest["splits"]:
        raise RuntimeError("Chronicle manifest lacks the temporal view.")
    dev_train_files = [by_id[i] for i in dev_train_ids]
    dev_val_files = [by_id[i] for i in dev_val_ids]
    static_files = [by_id[i] for i in static_ids]
    probe = V2InferencePipeline.load(control_ckpt, device="cpu")
    patchifier = Patchifier(
        PatchConfig(patch_size=probe.config.patch_size, stride=probe.config.stride)
    )
    top_q = float(probe.config.top_q_fraction)
    del probe

    commit = _git_commit(REPO_ROOT)
    manifest_sha = _sha256(manifest_path)
    control_sha = _sha256(control_ckpt)
    hybrid_sha = _sha256(hybrid_ckpt)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    print(f"[Provenance] commit={commit}", flush=True)
    print(f"[Provenance] manifest_sha={manifest_sha}", flush=True)
    print(f"[Provenance] control_sha={control_sha}", flush=True)
    print(f"[Provenance] hybrid_sha={hybrid_sha}", flush=True)
    print(
        f"[Provenance] config_hash={manifest['config_hash']} device={device_name}",
        flush=True,
    )
    print(
        f"[Views] dev_train={len(dev_train_files)} dev_val={len(dev_val_files)} "
        f"static={len(static_files)} (temporal untouched)",
        flush=True,
    )

    control = score_variant(
        control_ckpt,
        dev_train_files,
        dev_val_files,
        static_files,
        patchifier,
        args.device,
        args.batch_size,
    )
    hybrid = score_variant(
        hybrid_ckpt,
        dev_train_files,
        dev_val_files,
        static_files,
        patchifier,
        args.device,
        args.batch_size,
    )
    print("[Score] both variants scored (dev + static); rows stay in memory.", flush=True)

    if args.tasks in ("33", "both"):
        probe_ids = static_ids[: args.probe_files]
        probe_samples = [by_id[i] for i in probe_ids]
        control["corrupt_probe"] = corrupt_probe_for_variant(
            control["pipeline"], probe_samples, patchifier, args.seed, args.batch_size
        )
        hybrid["corrupt_probe"] = corrupt_probe_for_variant(
            hybrid["pipeline"], probe_samples, patchifier, args.seed, args.batch_size
        )
        control["corrupt_probe"]["file_ids_hash"] = hashlib.sha256(
            ",".join(probe_ids).encode()
        ).hexdigest()
        hybrid["corrupt_probe"]["file_ids_hash"] = control["corrupt_probe"][
            "file_ids_hash"
        ]
        task33 = build_task33(control, hybrid, args.seed)
        (out_root / "task33_geometry.json").write_text(
            json.dumps(task33, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_task33_slices(task33, out_root / "task33_slices.csv")
        render_task33_figures(task33, out_root / "figures")
        print("[Task33] wrote task33_geometry.json + slices + figures", flush=True)

    if args.tasks in ("34", "both"):
        task34 = build_task34(control, hybrid)
        (out_root / "task34_static.json").write_text(
            json.dumps(task34, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_task34_slices(task34, out_root / "task34_slices.csv")
        render_task34_figures(task34, out_root / "figures")
        print("[Task34] wrote task34_static.json + slices + figures", flush=True)

    run_manifest = {
        "commit": commit,
        "data_root": str(data_root),
        "manifest_sha256": manifest_sha,
        "config_hash": manifest["config_hash"],
        "control_checkpoint": str(control_ckpt),
        "control_sha256": control_sha,
        "hybrid_checkpoint": str(hybrid_ckpt),
        "hybrid_sha256": hybrid_sha,
        "control_config": control["config"].to_dict(),
        "hybrid_config": hybrid["config"].to_dict(),
        "device": device_name,
        "torch": torch.__version__,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "probe_files": args.probe_files,
        "n_dev_train": len(dev_train_files),
        "n_dev_val": len(dev_val_files),
        "n_static": len(static_files),
        "top_q_fraction": top_q,
        "temporal_view_accessed": False,
    }
    (out_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[Done] outputs under", str(out_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
