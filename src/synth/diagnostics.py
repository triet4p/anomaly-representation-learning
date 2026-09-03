"""
Data quality diagnostics (DATA.md §12).

Four diagnostic groups:

  1. Normal visual check:    regime diversity, variable length, channel relation
  2. Anomaly visual check:   per-family plots with mask and boundaries
  3. Patch statistics:       std, range, energy, derivative_energy, slope
  4. Normal vs abnormal:     weak baseline (5 statistics), per-family AUC,
                             intervention strength acceptance/rejection report

All functions return structured dicts of results and/or produce matplotlib
figures.  The calling experiment script decides what to save/display.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from synth.schema import FileSample, SampleLabel, AnomalyFamily
from synth.patchify import Patchifier
from synth.masking import compute_all_patch_stats
from synth.config import SynthConfig, PatchConfig, CHANNEL_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_stats(sample: FileSample) -> dict[str, float]:
    """Scalar stats for one file, supporting any channel count."""
    x = sample.x.astype(np.float64)
    stats = {
        "T": float(sample.T),
        "n_regimes": float(len(sample.regime_sequence)),
        "global_std": float(np.std(x)),
        "global_range": float(np.max(x) - np.min(x)),
    }
    for c in range(sample.C):
        stats[f"ch{c}_std"] = float(np.std(x[c]))
    return stats


def _patch_stats_array(
    samples: list[FileSample], pcfg: PatchConfig
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute patch statistics across real (non-padding) timesteps."""
    patchifier = Patchifier(pcfg)
    values = {name: [] for name in ("std", "range", "energy", "derivative_energy", "slope")}
    for sample in samples:
        batch = patchifier.patchify(sample)
        for i in range(batch.N):
            valid = int(batch.valid_len[i])
            flat = batch.patches[i, :, :valid].astype(np.float64).ravel()
            if flat.size == 0:
                continue
            values["std"].append(float(np.std(flat)))
            values["range"].append(float(np.ptp(flat)))
            values["energy"].append(float(np.mean(flat ** 2)))
            values["derivative_energy"].append(
                float(np.mean(np.abs(np.diff(flat)))) if flat.size > 1 else 0.0
            )
            values["slope"].append(
                abs(float(np.polyfit(np.arange(flat.size), flat, 1)[0]))
                if flat.size >= 2 else 0.0
            )
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}

# ---------------------------------------------------------------------------
# 1. Normal visual diagnostics
# ---------------------------------------------------------------------------

def diagnose_normal(
    samples: list[FileSample],
    max_plot: int = 6,
) -> dict[str, Any]:
    """
    Return summary statistics and matplotlib figures for normal samples.

    Returns dict with keys:
      length_distribution, regime_count_distribution,
      channel_correlation_mean, figures (list of Figure objects).
    """
    import matplotlib.pyplot as plt
    if not samples:
        return {
            "length_mean": 0.0, "length_std": 0.0,
            "length_min": 0.0, "length_max": 0.0,
            "n_regimes_mean": 0.0,
            "channel_correlation_mean": {},
            "figures": [],
        }
    Ts = [s.T for s in samples]
    n_regimes = [len(s.regime_sequence) for s in samples]
    corrs: list[tuple[int, int, float]] = []
    for s in samples:
        x = s.x.astype(np.float64)
        if x.shape[1] <= 10:
            continue
        for c1 in range(s.C):
            for c2 in range(c1 + 1, s.C):
                if np.std(x[c1]) > 0 and np.std(x[c2]) > 0:
                    corrs.append((c1, c2, float(np.corrcoef(x[c1], x[c2])[0, 1])))

    pair_means = {
        f"c{c1}_c{c2}": float(np.mean([v for a, b, v in corrs if a == c1 and b == c2]))
        for c1 in range(samples[0].C) for c2 in range(c1 + 1, samples[0].C)
        if any(a == c1 and b == c2 for a, b, _ in corrs)
    }
    results: dict[str, Any] = {
        "length_mean": float(np.mean(Ts)),
        "length_std": float(np.std(Ts)),
        "length_min": float(np.min(Ts)),
        "length_max": float(np.max(Ts)),
        "n_regimes_mean": float(np.mean(n_regimes)),
        "channel_correlation_mean": pair_means,
    }
    figs = []
    # Plot up to max_plot samples
    for i, s in enumerate(samples[:max_plot]):
        fig, axes = plt.subplots(s.C, 1, figsize=(12, 2.5 * s.C), sharex=True)
        if s.C == 1:
            axes = [axes]
        t = np.arange(s.T)
        titles = [f"S{c}: {CHANNEL_NAMES[c]}" if c < len(CHANNEL_NAMES) else f"ch{c}"
                  for c in range(s.C)]
        for c, ax in enumerate(axes):
            ax.plot(t, s.x[c], lw=0.7, color="steelblue", alpha=0.9)
            # Shade regime boundaries
            for r in s.regime_sequence:
                ax.axvline(r.start, color="gray", lw=0.5, alpha=0.4)
            ax.set_ylabel(titles[c] if c < len(titles) else f"ch{c}")
            ax.set_title(f"Normal sample {i} | T={s.T} | {len(s.regime_sequence)} regimes")
        axes[-1].set_xlabel("timestep")
        fig.tight_layout()
        figs.append(fig)
        plt.close(fig)

    # Length distribution plot
    fig_l, ax = plt.subplots(figsize=(6, 3))
    ax.hist(Ts, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Session length T")
    ax.set_ylabel("Count")
    ax.set_title("Normal session length distribution")
    fig_l.tight_layout()
    figs.append(fig_l)
    plt.close(fig_l)

    results["figures"] = figs
    return results


# ---------------------------------------------------------------------------
# 2. Anomaly visual diagnostics
# ---------------------------------------------------------------------------

def diagnose_anomaly(
    samples: list[FileSample],
    max_per_family: int = 2,
) -> dict[str, Any]:
    """
    Plot each anomaly family with channel signals, anomaly mask, and boundaries.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    by_family: dict[str, list[FileSample]] = {}
    for s in samples:
        if s.anomaly_meta is not None:
            fam = s.anomaly_meta.family.value
            by_family.setdefault(fam, []).append(s)

    figs = {}
    for fam_name, fam_samples in by_family.items():
        for i, s in enumerate(fam_samples[:max_per_family]):
            fig, axes = plt.subplots(s.C, 1, figsize=(12, 2.5 * s.C), sharex=True)
            if s.C == 1:
                axes = [axes]
            t = np.arange(s.T)
            m = s.anomaly_meta
            assert m is not None
            titles = [f"S{c}: {CHANNEL_NAMES[c]}" if c < len(CHANNEL_NAMES) else f"ch{c}"
                      for c in range(s.C)]
            for c, ax in enumerate(axes):
                ax.plot(t, s.x[c], lw=0.7, color="steelblue", label="signal")
                if s.anomaly_mask is not None and s.anomaly_mask[c].any():
                    ax.fill_between(
                        t,
                        ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else float(np.min(s.x[c])) - 0.1,
                        float(np.max(s.x[c])) + 0.1,
                        where=s.anomaly_mask[c],
                        color="red", alpha=0.15, label="anomaly mask",
                    )
                ax.axvline(m.start, color="red", lw=1, ls="--", alpha=0.7)
                ax.axvline(m.end, color="red", lw=1, ls="--", alpha=0.7)
                ax.set_ylabel(titles[c] if c < len(titles) else f"ch{c}")
            axes[0].set_title(
                f"{fam_name} | severity={m.severity:.2f} | "
                f"T={s.T} | strength={m.extra.get('strength', '?'):.3f}"
                if isinstance(m.extra.get("strength"), float) else
                f"{fam_name} | severity={m.severity:.2f} | T={s.T}"
            )
            axes[-1].set_xlabel("timestep")
            fig.tight_layout()
            key = f"{fam_name}_{i}"
            figs[key] = fig
            plt.close(fig)

    return {"figures": figs, "families_present": list(by_family.keys())}


# ---------------------------------------------------------------------------
# 3. Patch statistics diagnostics
# ---------------------------------------------------------------------------

def diagnose_patch_stats(
    normal_samples: list[FileSample],
    anomaly_samples: list[FileSample],
    pcfg: PatchConfig,
) -> dict[str, Any]:
    """Compare patch statistics while excluding padded timesteps."""
    import matplotlib.pyplot as plt

    normal_stats = _patch_stats_array(normal_samples[:100], pcfg)
    anomaly_stats = _patch_stats_array(anomaly_samples[:100], pcfg)
    stat_names = ["std", "range", "energy", "derivative_energy", "slope"]
    results: dict[str, Any] = {}
    for stat in stat_names:
        results[f"normal_{stat}_mean"] = (
            float(np.mean(normal_stats[stat])) if normal_stats[stat].size else 0.0
        )
        results[f"anomaly_{stat}_mean"] = (
            float(np.mean(anomaly_stats[stat])) if anomaly_stats[stat].size else 0.0
        )

    figs = []
    if any(normal_stats[name].size or anomaly_stats[name].size for name in stat_names):
        fig, axes = plt.subplots(1, len(stat_names), figsize=(14, 3))
        for ax, stat in zip(axes, stat_names):
            if normal_stats[stat].size:
                ax.hist(normal_stats[stat], bins=30, alpha=0.6, label="normal", color="steelblue", density=True)
            if anomaly_stats[stat].size:
                ax.hist(anomaly_stats[stat], bins=30, alpha=0.6, label="anomaly", color="tomato", density=True)
            ax.set_title(stat)
            ax.legend(fontsize=7)
        fig.suptitle("Patch statistics: normal vs anomaly")
        fig.tight_layout()
        figs.append(fig)
        plt.close(fig)
    results["figures"] = figs
    return results


# ---------------------------------------------------------------------------
# 4. Weak baseline AUC
# ---------------------------------------------------------------------------

def compute_weak_baseline_auc(
    normal_samples: list[FileSample],
    anomaly_samples: list[FileSample],
    pcfg: PatchConfig,
) -> dict[str, Any]:
    """Measure orientation-independent separation with standardized features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    patchifier = Patchifier(pcfg)

    def file_feature(sample: FileSample) -> npt.NDArray[np.float64]:
        batch = patchifier.patchify(sample)
        values = []
        for i in range(batch.N):
            flat = batch.patches[i, :, :int(batch.valid_len[i])].astype(np.float64).ravel()
            if flat.size:
                values.append([
                    np.std(flat), np.ptp(flat), np.mean(flat ** 2),
                    np.mean(np.abs(np.diff(flat))) if flat.size > 1 else 0.0,
                    np.mean(np.abs(flat - np.mean(flat))),
                ])
        return np.mean(values, axis=0) if values else np.zeros(5)

    if not normal_samples or not anomaly_samples:
        return {"overall_auc": float("nan"), "overall_separability": float("nan"),
                "per_family_auc": {}, "separability": {},
                "too_easy_families": {}, "warning": "insufficient samples"}
    normal_feats = np.stack([file_feature(s) for s in normal_samples])

    def fit_score(anom_feats: npt.NDArray[np.float64]) -> tuple[float, float]:
        X = np.vstack((normal_feats, anom_feats))
        y = np.r_[np.zeros(len(normal_feats)), np.ones(len(anom_feats))]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=0
        ))
        counts = np.bincount(y.astype(int))
        if counts.size == 2 and counts.min() >= 2:
            cv = StratifiedKFold(min(5, int(counts.min())), shuffle=True, random_state=0)
            score = cross_val_predict(model, X, y, cv=cv, method="decision_function")
        else:
            model.fit(X, y)
            score = model.decision_function(X)
        auc = float(roc_auc_score(y, score))
        return auc, max(auc, 1.0 - auc)

    by_family: dict[str, list[FileSample]] = {}
    for sample in anomaly_samples:
        if sample.anomaly_meta is not None:
            by_family.setdefault(sample.anomaly_meta.family.value, []).append(sample)
    if not by_family:
        return {"overall_auc": float("nan"), "overall_separability": float("nan"),
                "per_family_auc": {}, "separability": {},
                "too_easy_families": {}, "warning": "anomaly metadata unavailable"}
    family_aucs, separability = {}, {}
    for family, samples in by_family.items():
        family_aucs[family], separability[family] = fit_score(
            np.stack([file_feature(s) for s in samples])
        )
    all_features = np.stack([file_feature(s) for samples in by_family.values() for s in samples])
    overall_auc, overall_sep = fit_score(all_features) if len(all_features) else (float("nan"), float("nan"))
    too_easy = {family: value for family, value in separability.items() if value > 0.95}
    return {
        "overall_auc": overall_auc,
        "overall_separability": overall_sep,
        "per_family_auc": family_aucs,
        "separability": separability,
        "too_easy_families": too_easy,
        "warning": (
            f"Families {list(too_easy)} have near-perfect orientation-independent "
            "simple-statistics separation." if too_easy else None
        ),
    }


# ---------------------------------------------------------------------------
# 5. Strength acceptance/rejection report
# ---------------------------------------------------------------------------

def strength_acceptance_report(samples: list[FileSample]) -> dict[str, Any]:
    """Summarise accepted and rejected injection provenance by family."""
    accepted = [s.anomaly_meta for s in samples if s.anomaly_meta is not None]
    rejected = [s.rejection_meta for s in samples if s.rejection_meta is not None]
    accepted_strengths = [
        float(m.extra["strength"]) for m in accepted
        if isinstance(m.extra.get("strength"), (float, int))
    ]
    rejected_strengths = [
        float(m.extra["strength"]) for m in rejected
        if isinstance(m.extra.get("strength"), (float, int))
    ]
    per_family: dict[str, dict[str, int]] = {}
    for meta, key in [(m, "accepted") for m in accepted] + [(m, "rejected") for m in rejected]:
        family = meta.family.value
        per_family.setdefault(family, {"accepted": 0, "rejected": 0})[key] += 1
    total = len(accepted) + len(rejected)
    return {
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "acceptance_rate": len(accepted) / max(1, total),
        "strength_mean": float(np.mean(accepted_strengths)) if accepted_strengths else 0.0,
        "strength_std": float(np.std(accepted_strengths)) if accepted_strengths else 0.0,
        "rejected_strength_mean": float(np.mean(rejected_strengths)) if rejected_strengths else 0.0,
        "per_family": per_family,
    }
