"""
Data quality experiment (DATA.md §12).

Generates representative normal and anomaly samples, produces:
  1. Normal channel plots (regime diversity, variable T, channel relation)
  2. Per-family anomaly plots (signal + mask + boundaries)
  3. Patch statistics distributions
  4. Weak baseline AUC per anomaly family
  5. Intervention strength acceptance/rejection report
  6. Machine-readable JSON metrics

Usage::

    uv run python experiments/quality_check.py

Outputs are written to ``experiments/artifacts/`` (gitignored).
Figures are saved as PNG.  Metrics are saved as ``metrics.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synth.config import SynthConfig
from synth.generator import SessionGenerator
from synth.dataset import DatasetBuilder
from synth.patchify import Patchifier
from synth.masking import apply_masking, compute_all_patch_stats
from synth.contrastive import make_contrastive_views
from synth.diagnostics import (
    diagnose_normal,
    diagnose_anomaly,
    diagnose_patch_stats,
    compute_weak_baseline_auc,
    strength_acceptance_report,
)
from synth.anomalies.registry import HARD_FAMILIES
from synth.schema import AnomalyFamily


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 2025
N_NORMAL = 60          # normal samples to generate
N_ANOMALY_PER_FAMILY = 8  # anomaly samples per family
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[quality_check] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = SynthConfig()
    cfg.split.train_seed = SEED
    cfg.split.val_seed = SEED + 1_000_000
    cfg.split.test_seed = SEED + 2_000_000

    gen = SessionGenerator(cfg)
    patchifier = Patchifier(cfg.patch)
    metrics: dict = {}

    # ── 1. Generate normal samples ────────────────────────────────────
    log(f"Generating {N_NORMAL} normal samples...")
    rng = np.random.default_rng(SEED)
    normal_seeds = [int(rng.integers(0, 2**28)) for _ in range(N_NORMAL)]
    normals = [gen.generate_normal(s, split="train") for s in normal_seeds]
    Ts = [s.T for s in normals]
    n_regimes = [len(s.regime_sequence) for s in normals]
    log(f"  T range: [{min(Ts)}, {max(Ts)}], mean={np.mean(Ts):.1f}, std={np.std(Ts):.1f}")
    log(f"  Regime count range: [{min(n_regimes)}, {max(n_regimes)}]")
    metrics["normal_T_mean"] = float(np.mean(Ts))
    metrics["normal_T_std"] = float(np.std(Ts))
    metrics["normal_T_min"] = int(min(Ts))
    metrics["normal_T_max"] = int(max(Ts))
    metrics["normal_n_regimes_mean"] = float(np.mean(n_regimes))

    # ── 2. Normal visual diagnostics ──────────────────────────────────
    log("Running normal diagnostics...")
    norm_diag = diagnose_normal(normals, max_plot=6)
    for i, fig in enumerate(norm_diag["figures"]):
        path = OUT_DIR / f"normal_sample_{i:02d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
    log(f"  Saved {len(norm_diag['figures'])} normal plots")
    metrics["channel_correlation"] = norm_diag["channel_correlation_mean"]

    # ── 3. Generate anomaly samples ───────────────────────────────────
    log(f"Generating {N_ANOMALY_PER_FAMILY} samples per family × {len(HARD_FAMILIES)} families...")
    anomaly_samples = []
    rng2 = np.random.default_rng(SEED + 1)
    family_seeds: dict[str, list[int]] = {}
    for family in HARD_FAMILIES:
        seeds = [int(rng2.integers(0, 2**28)) for _ in range(N_ANOMALY_PER_FAMILY)]
        family_seeds[family.value] = seeds
        for s in seeds:
            sample = gen.generate_anomaly(seed=s, split="test", family=family)
            anomaly_samples.append(sample)

    accepted = [s for s in anomaly_samples if s.anomaly_meta is not None and s.anomaly_meta.extra.get("accepted")]
    log(f"  Accepted: {len(accepted)} / {len(anomaly_samples)}")
    metrics["total_generated_anomalies"] = len(anomaly_samples)
    metrics["accepted_anomalies"] = len(accepted)
    metrics["acceptance_rate"] = len(accepted) / max(1, len(anomaly_samples))

    # ── 4. Anomaly visual diagnostics ─────────────────────────────────
    log("Running anomaly diagnostics...")
    anom_diag = diagnose_anomaly(accepted, max_per_family=2)
    for key, fig in anom_diag["figures"].items():
        path = OUT_DIR / f"anomaly_{key}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
    log(f"  Families plotted: {anom_diag['families_present']}")

    # ── 5. Patch statistics ───────────────────────────────────────────
    log("Computing patch statistics...")
    patch_diag = diagnose_patch_stats(normals[:30], accepted[:30], cfg.patch)
    for i, fig in enumerate(patch_diag["figures"]):
        path = OUT_DIR / f"patch_stats_{i:02d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
    for k, v in patch_diag.items():
        if k != "figures":
            metrics[f"patch_{k}"] = v
    log("  Patch statistics saved")

    # ── 6. Weak baseline AUC ─────────────────────────────────────────
    log("Computing weak baseline AUC...")
    if accepted:
        auc_results = compute_weak_baseline_auc(normals[:40], accepted, cfg.patch)
        log(f"  Overall AUC: {auc_results['overall_auc']:.4f}")
        for fam, auc in auc_results["per_family_auc"].items():
            log(f"    {fam}: AUC={auc:.4f}")
        if auc_results.get("warning"):
            log(f"  WARNING: {auc_results['warning']}")
        metrics["weak_baseline"] = auc_results
    else:
        log("  Skipped: no accepted anomalies")
        metrics["weak_baseline"] = {}

    # ── 7. Strength acceptance report ────────────────────────────────
    log("Generating strength acceptance report...")
    strength_rep = strength_acceptance_report(anomaly_samples)
    log(f"  Acceptance rate: {strength_rep['acceptance_rate']:.1%}")
    log(f"  Strength: mean={strength_rep['strength_mean']:.4f}, std={strength_rep['strength_std']:.4f}")
    metrics["strength_report"] = strength_rep

    # ── 8. Masking composition example ───────────────────────────────
    log("Generating masking composition example...")
    if normals:
        s = normals[0]
        batch = patchifier.patchify(s)
        mask_result = apply_masking(batch, cfg.masking, np.random.default_rng(SEED))
        metrics["masking_example"] = {
            "n_patches": int(batch.N),
            "n_masked": int(mask_result.mask.sum()),
            "total_ratio": float(mask_result.total_ratio),
            "composition": mask_result.composition,
        }
        log(f"  Masked {mask_result.mask.sum()}/{batch.N} patches "
            f"(ratio={mask_result.total_ratio:.3f}), "
            f"composition={mask_result.composition}")

    # ── 9. Contrastive views example ─────────────────────────────────
    log("Generating contrastive views example...")
    if normals:
        s = normals[0]
        v1, v2 = make_contrastive_views(s, cfg.contrastive, np.random.default_rng(SEED))
        x = s.x.astype(np.float64)
        cos_sim_v1_v2 = float(np.dot(v1.flatten(), v2.flatten()) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        ))
        cos_sim_v1_orig = float(np.dot(v1.flatten(), x.flatten()) / (
            np.linalg.norm(v1) * np.linalg.norm(x) + 1e-8
        ))
        metrics["contrastive_cosine_v1_v2"] = cos_sim_v1_v2
        metrics["contrastive_cosine_v1_orig"] = cos_sim_v1_orig
        log(f"  Cosine(v1,v2)={cos_sim_v1_v2:.4f}, Cosine(v1,orig)={cos_sim_v1_orig:.4f}")

        # Plot one pair of views
        fig, axes = plt.subplots(s.C, 1, figsize=(12, 2.5 * s.C), sharex=True)
        if s.C == 1:
            axes = [axes]
        t = np.arange(s.T)
        for c, ax in enumerate(axes):
            ax.plot(t, s.x[c], label="original", lw=0.8, alpha=0.8)
            ax.plot(t, v1[c], label="view1", lw=0.8, alpha=0.7, ls="--")
            ax.plot(t, v2[c], label="view2", lw=0.8, alpha=0.7, ls=":")
            if c == 0:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel("timestep")
        fig.suptitle("Contrastive views (same file, different augmentations)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "contrastive_views.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── 10. Save metrics ─────────────────────────────────────────────
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log(f"Metrics saved to {metrics_path}")
    log("Done.")


if __name__ == "__main__":
    main()
