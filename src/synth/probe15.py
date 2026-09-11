"""Sprint 15 candidate-2 frozen observable probe (Task 13 cycle-2).

One simple causal-telemetry probe, frozen before any downstream (Fit,
Calibration, Confirmation) outcome exists. It reuses the repository metric
conventions in :mod:`synth.events` (tie-aware AUROC, window max-aggregation,
first-alert lead, median/IQR summaries, false-alert episode grouping) and
introduces no second convention: every evaluation number is produced by those
frozen primitives.

Leakage contract: :func:`extract_features` accepts only a waveform array plus
two timing scalars. Labels, cohorts, subtypes, failure times, health states,
and simulator state cannot enter feature construction structurally.

Fitting contract: standardization statistics and the healthy centroid are fit
on Fit-role verified-healthy files only. The single operating threshold is the
frozen 95th percentile of Calibration-role verified-healthy file scores. Both
rules are fixed here before outcomes; Task 14 applies them once.
"""

from __future__ import annotations

import numpy as np

from synth import events as E

#: Frozen probe identifier bound to candidate 2.
PROBE_ID = "sprint15-observable-probe-v2"

#: Frozen operating-point rule: threshold = this quantile of the
#: Calibration verified-healthy file-score distribution.
THRESHOLD_QUANTILE = 0.95

#: Frozen history-block bootstrap size and seed (deterministic LCB).
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260202

#: Floor guarding standardization against constant features.
STD_FLOOR = 1e-6

#: Spectral bands as thirds of the one-sided FFT bins: low/mid/high.
N_BANDS = 3


def feature_names(n_channels: int = 6) -> list[str]:
    """Return the frozen ordered feature names for ``n_channels``."""
    names: list[str] = []
    for c in range(n_channels):
        names.append(f"ch{c}_rms")
    for c in range(n_channels):
        names.append(f"ch{c}_std")
    for c in range(n_channels):
        names.append(f"ch{c}_slope")
    for c in range(n_channels):
        names.append(f"ch{c}_zcr")
    for c in range(n_channels):
        for b in range(N_BANDS):
            names.append(f"ch{c}_band{b}")
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            names.append(f"ch{i}xch{j}_corr")
    names.extend(["duration_s", "time_since_reset_s"])
    return names


def extract_features(
    x: np.ndarray,
    duration_s: float,
    time_since_reset_s: float,
) -> np.ndarray:
    """Compute the frozen causal observable feature vector.

    ``x`` is a ``[C, T]`` float waveform; the scalars are file duration and
    seconds since the last recommissioning reset (both model-visible timing
    metadata). Output order matches :func:`feature_names`. Pure NumPy,
    deterministic, RNG-free; non-finite outputs are mapped to ``0.0``.
    """
    x = np.asarray(x, dtype=np.float64)
    n_channels, n_times = x.shape
    t = np.arange(n_times, dtype=np.float64)
    t_centered = t - t.mean() if n_times > 1 else t
    denom = float(np.dot(t_centered, t_centered)) if n_times > 1 else 1.0
    feats: list[float] = []
    for c in range(n_channels):
        row = x[c]
        centered = row - row.mean()
        feats.append(float(np.sqrt(np.mean(row * row))))
        feats.append(float(row.std()))
        feats.append(float(np.dot(t_centered, centered) / denom))
        signs = np.sign(centered)
        signs[signs == 0.0] = 1.0
        feats.append(float(np.count_nonzero(np.diff(signs)) / max(n_times - 1, 1)))
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    n_bins = spectrum.shape[1]
    edges = np.linspace(0, n_bins, N_BANDS + 1).astype(int)
    for c in range(n_channels):
        total = float(spectrum[c].sum())
        for b in range(N_BANDS):
            lo, hi = int(edges[b]), int(edges[b + 1])
            band = float(spectrum[c, lo:hi].sum()) if hi > lo else 0.0
            feats.append(band / total if total > 0.0 else 0.0)
    stds = x.std(axis=1)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if stds[i] <= 0.0 or stds[j] <= 0.0:
                feats.append(0.0)
            else:
                cov = float(np.mean((x[i] - x[i].mean()) * (x[j] - x[j].mean())))
                feats.append(cov / float(stds[i] * stds[j]))
    feats.append(float(duration_s))
    feats.append(float(max(time_since_reset_s, 0.0)))
    out = np.array(feats, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def standardize_fit(features_healthy: np.ndarray) -> dict[str, np.ndarray]:
    """Fit frozen standardization on verified-healthy files only."""
    features_healthy = np.asarray(features_healthy, dtype=np.float64)
    mean = features_healthy.mean(axis=0)
    std = features_healthy.std(axis=0)
    std = np.maximum(std, STD_FLOOR)
    return {"mean": mean, "std": std}


def apply_standardization(
    features: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    """Standardize features with frozen Fit statistics."""
    return (np.asarray(features, dtype=np.float64) - stats["mean"]) / stats["std"]


def fit_centroid(standardized_healthy: np.ndarray) -> np.ndarray:
    """Fit the healthy centroid (the probe's only model)."""
    return np.asarray(standardized_healthy, dtype=np.float64).mean(axis=0)


def score_files(
    standardized: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    """Score files by squared distance from the healthy centroid."""
    diff = np.asarray(standardized, dtype=np.float64) - np.asarray(
        centroid, dtype=np.float64
    )
    return np.einsum("ij,ij->i", diff, diff)


def select_threshold(calibration_healthy_scores: np.ndarray) -> float:
    """Select the operating threshold: frozen quantile of healthy scores."""
    return float(
        np.quantile(np.asarray(calibration_healthy_scores, dtype=np.float64),
                    THRESHOLD_QUANTILE)
    )


def history_block_lcb(
    history_aucs: list[float],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """History-block bootstrap 95% LCB of the macro (mean) AUROC."""
    aucs = np.asarray(history_aucs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(aucs), size=(replicates, len(aucs)))
    macros = aucs[draws].mean(axis=1)
    return {
        "macro": float(aucs.mean()),
        "lcb": float(np.quantile(macros, 0.025)),
        "replicates": float(replicates),
    }


def evaluate_histories(
    file_scores: dict[str, float],
    rows_by_history: list[list[dict]],
    ledgers_by_history: list[list[dict]],
    wins_by_history: list[dict],
    threshold: float,
) -> dict:
    """Evaluate frozen EG4 metrics over Confirmation histories.

    ``file_scores`` maps file id to probe score across all histories. Per
    history: P+W / P / W event AUROCs (tie-aware, control windows as
    negatives), directional flag, P/W recall + median lead at ``threshold``,
    false-alert episodes per robot-day, and separate abrupt companion metrics.
    Macro AUROCs are unweighted means; the P+W LCB is history-block
    bootstrapped. Deterministic given inputs.
    """
    per_history = []
    for rows, ledger, wins in zip(rows_by_history, ledgers_by_history,
                                  wins_by_history):
        pos_pw, pos_p, pos_w, pos_a = [], [], [], []
        for failure in ledger:
            if E.positive_window_intersects_reset(failure, wins):
                continue
            cands = E.pos_files(rows, failure, wins)
            if not cands:
                continue
            score = E.window_score([c["file_id"] for c in cands], file_scores)
            if failure["cohort"] == "P":
                pos_p.append(score)
                pos_pw.append(score)
            elif failure["cohort"] == "W":
                pos_w.append(score)
                pos_pw.append(score)
            else:
                pos_a.append(score)
        anchors = E.anchor_rows(rows, wins)
        controls = E.select_control_windows(anchors, ledger, wins)
        neg = [E.window_score([m["file_id"] for m in w["members"]], file_scores)
               for w in controls]
        def _auc(pos):
            if not pos or not neg:
                return None
            return E.roc_auc_tie_aware(pos, neg)

        entry: dict = {
            "auc_pw": _auc(pos_pw),
            "auc_p": _auc(pos_p),
            "auc_w": _auc(pos_w),
            "auc_a": _auc(pos_a),
        }
        for cohort in ("P", "W"):
            recalled = 0
            lead_list = []
            total = 0
            for failure in ledger:
                if failure["cohort"] != cohort:
                    continue
                if E.positive_window_intersects_reset(failure, wins):
                    continue
                cands = E.pos_files(rows, failure, wins)
                if not cands:
                    continue
                total += 1
                flagged = sorted(c["end_time"] for c in cands
                                 if file_scores[c["file_id"]] >= threshold)
                if flagged:
                    recalled += 1
                    lead_list.append(
                        E.lead_days(failure["failure_time"], flagged))
            entry[f"recall_{cohort.lower()}"] = recalled / total if total else 0.0
            entry[f"lead_{cohort.lower()}_median"] = (
                float(np.median(lead_list)) if lead_list else 0.0)
            entry[f"n_{cohort.lower()}"] = total
        flagged_by_robot: dict = {}
        for row in rows:
            if not E.eligible_operational_row(row, wins):
                continue
            if file_scores[row["file_id"]] >= threshold:
                flagged_by_robot.setdefault(row["robot_id"], []).append(
                    row["end_time"])
        eval_days = {(r["robot_id"], int(r["end_time"] // 86400.0))
                     for r in rows if E.eligible_operational_row(r, wins)}
        false, far = E.false_alert_episodes(
            flagged_by_robot, ledger, float(len(eval_days)), wins)
        entry["false_episodes"] = false
        entry["far"] = far
        entry["robot_days"] = float(len(eval_days))
        per_history.append(entry)
    macro_pw = [h["auc_pw"] for h in per_history]
    macro_p = [h["auc_p"] for h in per_history]
    macro_w = [h["auc_w"] for h in per_history]
    lcb = history_block_lcb(macro_pw)
    return {
        "probe": PROBE_ID,
        "threshold": threshold,
        "macro_auc_pw": float(np.mean(macro_pw)),
        "macro_auc_p": float(np.mean(macro_p)),
        "macro_auc_w": float(np.mean(macro_w)),
        "lcb_pw": lcb["lcb"],
        "directional_histories": sum(1 for v in macro_pw if v > 0.55),
        "per_history": per_history,
    }
