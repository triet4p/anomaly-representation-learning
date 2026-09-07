"""Sprint 11 Batch F2 server runner (Tasks 35-36, CORRECTED rerun).

Chronological early-warning/calibration analysis (Task 35) for the frozen
corrected control/hybrid checkpoints on the untouched temporal view, plus the
metric payload Task 36 integrates. Mirrors the canonical
``notebooks/trajectory_v2_early_warning.ipynb`` pipeline exactly (per-robot
chronological scoring, suspect guard, maintenance resets, allowed pre-cutoff
survival fit, censored 1d/7d evaluation) and extends it with the full Task 35
metric set: lead-time distributions, per-robot/per-program slices, concordance,
per-group calibration, censoring counts, cold-start assessment, and
maintenance-boundary behavior.

Corrected-rerun contracts (prior ``e669e2a`` Batch F2 results SUPERSEDED):
canonical ``context_energy`` acute chain with explicit ``energy_source`` at
every seam plus a separately labeled ``population_energy`` view (population
elevated fractions stay excluded from detection claims — the restored
operating threshold is calibrated on ``context_energy``); restored dev-val
operating threshold and dev-val-only conformal calibrator carried from each
checkpoint payload with distinct provenance (``confidence_calibrator_fit_cohort``
vs ``operating_threshold_fit_cohort``); independent conformal healthy coverage
on a held-out healthy cohort disjoint from the dev-val calibrator-fit rows,
with any fit-cohort coverage labeled in-sample diagnostic.

Row-level timelines stay in server memory; only bounded aggregates, slices,
and review-sized figures are written.
"""

from __future__ import annotations

import argparse
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
from representation.v2_batch_f2 import (  # noqa: E402
    brier_score,
    censoring_counts,
    check_mapping_nonempty,
    concordance_index,
    constant_brier_reference,
    count_warning_runs,
    describe_or_null,
    equal_width_ece,
    group_recall,
    held_out_healthy_indices,
    maintenance_proximity,
)
from representation.v2_checkpoint import load_v2_checkpoint  # noqa: E402
from representation.v2_inference import V2InferencePipeline, patch_regime_ids  # noqa: E402
from representation.v2_risk import (  # noqa: E402
    CensoredSurvivalRisk,
    expected_feature_width,
    trajectory_feature_matrix,
)
from representation.v2_trajectory import TrajectoryTracker  # noqa: E402
from synth.chronicle import load_chronological  # noqa: E402
from synth.config import PatchConfig  # noqa: E402
from synth.patchify import Patchifier  # noqa: E402
from synth.schema import EpisodeKind, SampleLabel  # noqa: E402

TRAJECTORY_KEYS = ("displacement", "velocity", "trend", "persistence", "disagreement")
SUSPECT_RULE = "file_label is ABNORMAL or quarantined"
MAINT_WINDOW_S = 3.0 * 86400.0
CONTEXT_ENERGY_FIELD = "context_energy"
POPULATION_ENERGY_FIELD = "population_energy"
CONFIDENCE_CALIBRATOR_FIT_COHORT = "dev-val"
RISK_OPERATING_FIT_COHORT = "allowed-pre-cutoff-nontest"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()


def _robot_of(sample) -> str:
    return str(sample.operation.robot_id)


def _program_of(sample) -> str:
    return str(sample.operation.program_id)


def _is_suspect(sample) -> bool:
    if sample.file_label is SampleLabel.ABNORMAL:
        return True
    prov = getattr(sample, "split_provenance", None)
    return bool(prov is not None and prov.is_quarantined)


def _is_maintenance_reset(sample) -> bool:
    episode = getattr(sample, "episode", None)
    return bool(episode is not None and episode.kind is EpisodeKind.MAINTENANCE)


@torch.no_grad()
def score_temporal(pipeline, tracker_state, temporal_files, patchifier, device):
    """Score the temporal view per robot in chronological order (in memory)."""
    robots = sorted({_robot_of(s) for s in temporal_files})
    file_rows: list[dict[str, object]] = []
    n_breaks = 0
    for robot in robots:
        stream = sorted(
            (s for s in temporal_files if _robot_of(s) == robot),
            key=lambda s: (s.operation.start_time, s.operation.end_time, s.file_id),
        )
        starts = [s.operation.start_time for s in stream]
        if any(b < a for a, b in zip(starts, starts[1:])):
            raise RuntimeError(f"temporal stream for {robot} is not chronological")
        tracker = TrajectoryTracker(pipeline.config.d_model)
        tracker.load_state_dict(tracker_state)
        for sample in stream:
            reset = _is_maintenance_reset(sample)
            n_breaks += int(reset)
            suspect = _is_suspect(sample)
            base = collate_variable_files([sample], patchifier)
            count = base["patches"].shape[1]
            regimes = patch_regime_ids([sample], base["starts"], count)
            out = pipeline.score_patches(
                base["patches"],
                base["patch_pad_mask"],
                base["patch_valid_mask"],
                base["robot_idx"],
                base["program_idx"],
                regimes,
                tracker=tracker,
                suspect_flags=torch.tensor([suspect]),
                maintenance_resets=torch.tensor([reset]),
            )
            if out["file"].get("energy_source") != CONTEXT_ENERGY_FIELD:
                raise RuntimeError("Context file chain lost its energy_source label.")
            if out["file_population"].get("energy_source") != POPULATION_ENERGY_FIELD:
                raise RuntimeError("Population file view lost its energy_source label.")
            if not out["trajectory"]:
                raise RuntimeError("trajectory scoring requires the commissioned baseline")
            traj = {k: float(out["trajectory"][k][0]) for k in TRAJECTORY_KEYS}
            if any(v != v for v in traj.values()):
                raise RuntimeError(f"non-finite trajectory features for {sample.file_id}")
            meta = sample.anomaly_meta
            health = getattr(sample, "health", None)
            prov = getattr(sample, "split_provenance", None)
            file_rows.append(
                {
                    "file_id": sample.file_id,
                    "robot": robot,
                    "program": _program_of(sample),
                    "start_time": float(sample.operation.start_time),
                    "end_time": float(sample.operation.end_time),
                    "suspect": bool(suspect),
                    "maintenance_reset": bool(reset),
                    "abnormal": bool(sample.file_label is SampleLabel.ABNORMAL),
                    "quarantined": bool(prov is not None and prov.is_quarantined),
                    "family": str(getattr(meta, "family", "normal"))
                    if meta is not None
                    else "normal",
                    "health_value": float(getattr(health, "health_value", float("nan")))
                    if health is not None
                    else float("nan"),
                    "energy_source": CONTEXT_ENERGY_FIELD,
                    "context_tail_energy": float(out["file"]["tail_energy"][0]),
                    "context_elevated_fraction": float(out["file"]["elevated_fraction"][0]),
                    "context_mean_energy": float(out["file"]["mean_energy"][0]),
                    "population_energy_source": POPULATION_ENERGY_FIELD,
                    "population_tail_energy": float(out["file_population"]["tail_energy"][0]),
                    "population_mean_energy": float(out["file_population"]["mean_energy"][0]),
                    "population_elevated_fraction": float(
                        out["file_population"]["elevated_fraction"][0]
                    ),
                    "confidence": float(out["confidence"]["confidence"][0]),
                    **traj,
                }
            )
    return file_rows, n_breaks


def fit_allowed_risk(pipeline, tracker_state, fit_candidates, patchifier, seed):
    """Fit the survival head on allowed pre-cutoff data only (never test)."""
    by_robot: dict[str, list] = defaultdict(list)
    for sample in fit_candidates:
        by_robot[_robot_of(sample)].append(sample)
    feat_list, days_list, event_list = [], [], []
    for robot in sorted(by_robot):
        stream = sorted(
            by_robot[robot],
            key=lambda s: (s.operation.start_time, s.operation.end_time, s.file_id),
        )
        tracker = TrajectoryTracker(pipeline.config.d_model)
        tracker.load_state_dict(tracker_state)
        for sample in stream:
            base = collate_variable_files([sample], patchifier)
            count = base["patches"].shape[1]
            regimes = patch_regime_ids([sample], base["starts"], count)
            with torch.no_grad():
                out = pipeline.score_patches(
                    base["patches"],
                    base["patch_pad_mask"],
                    base["patch_valid_mask"],
                    base["robot_idx"],
                    base["program_idx"],
                    regimes,
                    tracker=tracker,
                    suspect_flags=torch.tensor([_is_suspect(sample)]),
                    maintenance_resets=torch.tensor([_is_maintenance_reset(sample)]),
                )
            row = trajectory_feature_matrix(
                out["trajectory"],
                out["file"]["tail_energy"],
                out["file"]["elevated_fraction"],
            )
            feat_list.append(row)
            assert sample.future_targets is not None
            assert sample.future_targets.time_to_next_failure is not None
            days_list.append(sample.future_targets.time_to_next_failure / 86400.0)
            event_list.append(not sample.future_targets.is_censored)
    features = torch.cat(feat_list)
    days = torch.tensor(days_list, dtype=torch.float32)
    events = torch.tensor(event_list, dtype=torch.bool)
    width = expected_feature_width()
    if features.shape[1] != width:
        raise RuntimeError(f"risk feature width {features.shape[1]} != canonical {width}")
    risk = CensoredSurvivalRisk(width, horizons_days=tuple(pipeline.config.risk_horizons_days))
    try:
        risk.fit(features, days, events, seed=seed)
        return risk, features, days, events, None
    except ValueError as exc:
        return risk, features, days, events, (
            f"survival fit on allowed pre-cutoff data failed ({exc}); "
            "no test-data fallback was attempted"
        )


def _safe_auroc_auprc(truth: list[bool], scores: list[float]):
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_pos = sum(1 for t in truth if t)
    if n_pos == 0 or n_pos == len(truth):
        return float("nan"), float("nan")
    return float(roc_auc_score(truth, scores)), float(average_precision_score(truth, scores))


def build_variant_temporal(
    name: str,
    pipeline,
    file_rows: list[dict[str, object]],
    samples_by_id: dict,
    manifest: dict,
    risk,
    fit_features,
    fit_days,
    fit_events,
    risk_fit_error: str | None,
    restored_operating: dict | None,
    restored_calibrator: dict | None,
    dev_val_ids: set,
    fit_file_ids: list,
) -> dict[str, object]:
    """Full Task 35 metric set for one variant (bounded aggregates only)."""
    n_files = len(file_rows)
    robots = sorted({str(r["robot"]) for r in file_rows})
    programs = sorted({str(r["program"]) for r in file_rows})

    days_all = torch.tensor(
        [
            samples_by_id[str(r["file_id"])].future_targets.time_to_next_failure / 86400.0
            if samples_by_id[str(r["file_id"])].future_targets is not None
            and samples_by_id[str(r["file_id"])].future_targets.time_to_next_failure
            is not None
            else float("nan")
            for r in file_rows
        ],
        dtype=torch.float32,
    )
    event_all = torch.tensor(
        [
            samples_by_id[str(r["file_id"])].future_targets is not None
            and not samples_by_id[str(r["file_id"])].future_targets.is_censored
            for r in file_rows
        ],
        dtype=torch.bool,
    )
    censor = censoring_counts(
        [
            None if not np.isfinite(float(v)) else float(v)
            for v in days_all.tolist()
        ],
        event_all.tolist(),
    )

    traj_batch = {
        k: torch.tensor([float(r[k]) for r in file_rows]) for k in TRAJECTORY_KEYS
    }
    # Canonical acute chain: the risk features ride the context signal (the
    # same chain feeding trajectory/confidence/risk); the population view is
    # reported separately and never enters detection claims.
    risk_features = trajectory_feature_matrix(
        traj_batch,
        torch.tensor([float(r["context_tail_energy"]) for r in file_rows]),
        torch.tensor([float(r["context_elevated_fraction"]) for r in file_rows]),
    )

    out: dict[str, object] = {
        "variant": name,
        "n_files": n_files,
        "n_robots": len(robots),
        "robots": robots,
        "programs": programs,
        "censoring": censor,
        "suspect_rule": SUSPECT_RULE,
        "energy_sources": {
            "context": CONTEXT_ENERGY_FIELD,
            "population": POPULATION_ENERGY_FIELD,
        },
        "restored_operating_threshold": restored_operating,
        "restored_confidence_cohort": restored_calibrator,
        "confidence_calibrator_fit_cohort": CONFIDENCE_CALIBRATOR_FIT_COHORT,
        "operating_threshold_fit_cohort": RISK_OPERATING_FIT_COHORT,
    }

    horizons_cfg = tuple(pipeline.config.risk_horizons_days)
    cohort_composition: dict[str, dict[str, int]] = {}
    for horizon in horizons_cfg:
        included, labels = CensoredSurvivalRisk.horizon_cohort(days_all, event_all, int(horizon))
        cohort_composition[f"{int(horizon)}d"] = {
            "n_usable": int(included.sum().item()),
            "n_events": int(labels[included].sum().item()),
            "n_excluded_censored": int((~included).sum().item()),
        }
    out["cohort_composition"] = cohort_composition

    calibrated = risk_fit_error is None
    out["risk_status"] = "calibrated" if calibrated else "uncalibrated"
    out["risk_fit_error"] = risk_fit_error

    if not calibrated:
        out.update(
            {
                "operating_threshold": None,
                "operating_point_source": None,
                "horizons": {},
                "concordance_7d": {"c": None, "reason": risk_fit_error},
                "ece_1d": None,
                "ece_7d": None,
                "calibration_bins_7d": [],
                "event_recall_1d": None,
                "event_recall_7d": None,
                "lead_time_1d_days": {"n": 0},
                "lead_time_7d_days": {"n": 0},
                "warning_persistence": {"n": 0},
                "false_warning_runs": 0,
                "false_warning_runs_per_robot_day": 0.0,
                "false_warning_runs_per_30_robot_days": 0.0,
                "robot_days_total": 0.0,
                "by_robot": {},
                "by_program": {},
                "calibration_by_robot_7d": {},
                "maintenance_boundary": {},
                "population_view": {},
                "conformal_healthy_coverage_independent": None,
                "conformal_healthy_coverage_fit_insample": None,
            }
        )
        return out

    assert risk is not None and fit_features is not None
    proba = risk.predict_proba(risk_features)
    brier = risk.brier_score(risk_features, days_all, event_all)
    fit_proba = risk.predict_proba(fit_features)
    _, fit_label_7d = CensoredSurvivalRisk.horizon_cohort(
        fit_days, fit_events, horizons_cfg[1]
    )
    fit_negatives = fit_proba["risk_7d"][~fit_label_7d]
    if fit_negatives.numel() == 0:
        raise RuntimeError("allowed fit cohort carries no 7d negatives for the operating point")
    operating_threshold = float(torch.quantile(fit_negatives, 0.95))
    out["operating_threshold"] = operating_threshold
    out["operating_threshold_n_negatives"] = int(fit_negatives.numel())
    out["operating_point_source"] = (
        "95th percentile of allowed-fit 7d negatives "
        f"(fit cohort {RISK_OPERATING_FIT_COHORT}; temporal risk operating point, "
        "distinct from the restored dev-val static operating threshold on "
        "context_energy)"
    )

    risk_1d = proba["risk_1d"].tolist()
    risk_7d = proba["risk_7d"].tolist()

    horizons: dict[str, dict[str, object]] = {}
    for horizon, key, scores in (
        (horizons_cfg[0], "risk_1d", risk_1d),
        (horizons_cfg[1], "risk_7d", risk_7d),
    ):
        included, labels = CensoredSurvivalRisk.horizon_cohort(
            days_all, event_all, int(horizon)
        )
        truth = labels[included].tolist()
        sel_scores = [scores[i] for i in torch.nonzero(included).flatten().tolist()]
        auroc, auprc = _safe_auroc_auprc(truth, sel_scores)
        horizons[key] = {
            "horizon_days": float(horizon),
            "n_usable": int(included.sum().item()),
            "n_events": int(labels[included].sum().item()),
            "prevalence": float(labels[included].sum().item() / max(1, int(included.sum().item()))),
            "auroc": auroc,
            "auprc": auprc,
            "brier": float(brier[key]),
            "brier_reference_const": constant_brier_reference(
                [bool(v) for v in labels[included].tolist()]
            ),
        }
    out["horizons"] = horizons

    # Concordance on finite-time files with the 7d risk score.
    finite = [i for i, r in enumerate(file_rows) if np.isfinite(float(days_all[i]))]
    conc = concordance_index(
        [float(days_all[i]) for i in finite],
        [bool(event_all[i]) for i in finite],
        [risk_7d[i] for i in finite],
    )
    out["concordance_7d"] = conc

    # Calibration: ECE + bins for both horizons; Brier already in horizons.
    ece_rows = {}
    for horizon, key, scores in ((horizons_cfg[0], "risk_1d", risk_1d), (horizons_cfg[1], "risk_7d", risk_7d)):
        included, labels = CensoredSurvivalRisk.horizon_cohort(
            days_all, event_all, int(horizon)
        )
        idx = torch.nonzero(included).flatten().tolist()
        ece_rows[key] = equal_width_ece(
            [scores[i] for i in idx], [bool(labels[i]) for i in idx]
        )
    out["ece_1d"] = ece_rows["risk_1d"]["ece"]
    out["ece_7d"] = ece_rows["risk_7d"]["ece"]
    out["calibration_bins_7d"] = ece_rows["risk_7d"]["bins"]
    out["calibration_bins_1d"] = ece_rows["risk_1d"]["bins"]

    # Event recall / lead time / persistence over recorded failure episodes.
    failures = [e for e in manifest["episodes"] if e["kind"] == "failure"]
    out["n_failure_episodes"] = len(failures)
    recall_1d, recall_7d, leads_1d, leads_7d, persist = [], [], [], [], []
    per_robot_events: dict[str, dict[str, list]] = defaultdict(
        lambda: {"recall_1d": [], "recall_7d": [], "leads_7d": []}
    )
    for failure in failures:
        robot, fail_t = failure["robot_id"], float(failure["start_time"])
        idx = [i for i, r in enumerate(file_rows) if str(r["robot"]) == robot]
        for horizon_s, scores, recalls, leads in (
            (86400.0, risk_1d, recall_1d, leads_1d),
            (604800.0, risk_7d, recall_7d, leads_7d),
        ):
            window = [
                i
                for i in idx
                if 0.0 <= fail_t - float(file_rows[i]["end_time"]) <= horizon_s
            ]
            warned = [i for i in window if scores[i] >= operating_threshold]
            hit = bool(warned)
            recalls.append(hit)
            key = "recall_1d" if horizon_s == 86400.0 else "recall_7d"
            per_robot_events[robot][key].append(hit)
            if hit:
                first = min(warned, key=lambda i: float(file_rows[i]["end_time"]))
                lead = (fail_t - float(file_rows[first]["end_time"])) / 86400.0
                leads.append(lead)
                if horizon_s == 604800.0:
                    per_robot_events[robot]["leads_7d"].append(lead)
                persist.append(
                    sum(1 for i in window if scores[i] >= operating_threshold)
                    / len(window)
                )
    out["event_recall_1d"] = sum(recall_1d) / max(1, len(recall_1d)) if recall_1d else None
    out["event_recall_7d"] = sum(recall_7d) / max(1, len(recall_7d)) if recall_7d else None
    out["event_hits_1d"] = f"{sum(recall_1d)}/{len(recall_1d)}"
    out["event_hits_7d"] = f"{sum(recall_7d)}/{len(recall_7d)}"
    out["lead_time_1d_days"] = describe_or_null(leads_1d)
    out["lead_time_7d_days"] = describe_or_null(leads_7d)
    out["warning_persistence"] = describe_or_null(persist)

    # False-warning runs per robot; robot-day/month normalization.
    by_robot: dict[str, dict[str, object]] = {}
    false_runs = 0
    robot_days_total = 0.0
    for robot in robots:
        rows = sorted(
            (r for r in file_rows if str(r["robot"]) == robot),
            key=lambda r: (float(r["start_time"]), float(r["end_time"])),
        )
        span = (max(float(r["end_time"]) for r in rows) - min(float(r["start_time"]) for r in rows)) / 86400.0
        robot_days_total += span
        robot_fails = [float(e["start_time"]) for e in failures if e["robot_id"] == robot]
        order = [file_rows.index(r) for r in rows]
        warned_flags = []
        for i in order:
            warned = risk_7d[i] >= operating_threshold
            future_fail = any(
                0.0 <= f - float(file_rows[i]["end_time"]) <= 604800.0
                for f in robot_fails
            )
            warned_flags.append(bool(warned and not future_fail))
        runs = count_warning_runs(warned_flags)
        false_runs += runs
        ev = per_robot_events.get(robot, {"recall_1d": [], "recall_7d": [], "leads_7d": []})
        by_robot[robot] = {
            "n_files": len(rows),
            "robot_days": span,
            "n_failure_episodes": len(robot_fails),
            "recall_1d": (sum(ev["recall_1d"]) / len(ev["recall_1d"])) if ev["recall_1d"] else None,
            "recall_7d": (sum(ev["recall_7d"]) / len(ev["recall_7d"])) if ev["recall_7d"] else None,
            "lead_7d_days": describe_or_null(ev["leads_7d"]),
            "false_warning_runs": runs,
            "false_runs_per_robot_day": (runs / span) if span > 0 else None,
            "false_runs_per_30d": (runs / (span / 30.0)) if span > 0 else None,
            "population_tail_median": float(
                np.median([float(r["population_tail_energy"]) for r in rows])
            ),
            "population_energy_source": POPULATION_ENERGY_FIELD,
        }
    out["by_robot"] = by_robot
    out["false_warning_runs"] = false_runs
    out["robot_days_total"] = robot_days_total
    out["false_warning_runs_per_robot_day"] = false_runs / max(1e-6, robot_days_total)
    out["false_warning_runs_per_30_robot_days"] = false_runs / max(1e-6, robot_days_total / 30.0)

    # Per-robot horizon discrimination on the 7d usable cohort (may be null).
    for robot in robots:
        idx = [i for i, r in enumerate(file_rows) if str(r["robot"]) == robot]
        included, labels = CensoredSurvivalRisk.horizon_cohort(days_all, event_all, int(horizons_cfg[1]))
        sel = [i for i in idx if bool(included[i])]
        truth = [bool(labels[i]) for i in sel]
        scores = [risk_7d[i] for i in sel]
        auroc, auprc = _safe_auroc_auprc(truth, scores)
        by_robot[robot]["auroc_7d"] = auroc
        by_robot[robot]["auprc_7d"] = auprc
        by_robot[robot]["n_usable_7d"] = len(sel)
        by_robot[robot]["n_events_7d"] = sum(1 for t in truth if t)
        # Per-robot 7d calibration where defined.
        if len(sel) and any(truth) and not all(truth):
            by_robot[robot]["brier_7d"] = brier_score(scores, truth)
            by_robot[robot]["ece_7d"] = equal_width_ece(scores, truth)["ece"]
            by_robot[robot]["brier_reference_const_7d"] = constant_brier_reference(truth)
        else:
            by_robot[robot]["brier_7d"] = float("nan")
            by_robot[robot]["ece_7d"] = float("nan")
            by_robot[robot]["brier_reference_const_7d"] = float("nan")
    out["calibration_by_robot_7d"] = {
        r: {"brier_7d": by_robot[r]["brier_7d"], "ece_7d": by_robot[r]["ece_7d"]}
        for r in robots
    }

    # Per-program slices: warned rate inside pre-failure windows + 7d discrimination.
    window_7d = set()
    for failure in failures:
        fail_t = float(failure["start_time"])
        for i, r in enumerate(file_rows):
            if str(r["robot"]) == failure["robot_id"] and 0.0 <= fail_t - float(r["end_time"]) <= 604800.0:
                window_7d.add(i)
    in_window = [i in window_7d for i in range(n_files)]
    warned_7d = [s >= operating_threshold for s in risk_7d]
    by_program: dict[str, dict[str, object]] = {}
    prog_recall = group_recall(
        [str(r["program"]) for r in file_rows], in_window, warned_7d
    )
    for program in programs:
        idx = [i for i, r in enumerate(file_rows) if str(r["program"]) == program]
        included, labels = CensoredSurvivalRisk.horizon_cohort(days_all, event_all, int(horizons_cfg[1]))
        sel = [i for i in idx if bool(included[i])]
        truth = [bool(labels[i]) for i in sel]
        scores = [risk_7d[i] for i in sel]
        auroc, auprc = _safe_auroc_auprc(truth, scores)
        entry: dict[str, object] = {
            "n_files": len(idx),
            "n_in_7d_window": prog_recall.get(program, {}).get("n", 0),
            "warned_rate_in_7d_window": prog_recall.get(program, {}).get("rate"),
            "auroc_7d": auroc,
            "auprc_7d": auprc,
            "n_usable_7d": len(sel),
            "n_events_7d": sum(1 for t in truth if t),
        }
        by_program[program] = entry
    out["by_program"] = by_program

    # Maintenance-boundary behavior: ±3d around maintenance episode boundaries.
    maint = [e for e in manifest["episodes"] if e["kind"] == "maintenance"]
    bounds: list[float] = []
    for e in maint:
        bounds.append(float(e["start_time"]))
        if e.get("end_time") is not None:
            bounds.append(float(e["end_time"]))
    near = maintenance_proximity(
        [float(r["end_time"]) for r in file_rows], bounds, MAINT_WINDOW_S
    )
    near_idx = [i for i, flag in enumerate(near) if flag]
    disp_all = [float(r["displacement"]) for r in file_rows]
    out["maintenance_boundary"] = {
        "n_maintenance_episodes": len(maint),
        "window_days": 3.0,
        "n_files_within_window": len(near_idx),
        "warned_fraction_within_window": (
            sum(1 for i in near_idx if risk_7d[i] >= operating_threshold) / len(near_idx)
        )
        if near_idx
        else None,
        "displacement_median_within_window": float(np.median([disp_all[i] for i in near_idx]))
        if near_idx
        else None,
        "displacement_median_overall": float(np.median(disp_all)),
        "maintenance_breaks_scored": sum(1 for r in file_rows if bool(r["maintenance_reset"])),
    }

    # Conformal healthy coverage with exact provenance. The restored calibrator
    # was fit on dev-val rows only, so the independent measurement uses a
    # held-out healthy temporal cohort (normal, non-quarantined, disjoint from
    # every fit row by assertion). Coverage on the allowed-fit 7d negatives is
    # kept as a separately labeled in-sample diagnostic (those rows fit the
    # survival head and may overlap dev-val; the overlap count is reported,
    # never pooled into the independent claim).
    temporal_ids = [str(r["file_id"]) for r in file_rows]
    fit_id_list = [str(v) for v in fit_file_ids]
    n_fit_overlaps_devval = sum(1 for v in fit_id_list if v in dev_val_ids)
    if set(temporal_ids) & set(dev_val_ids):
        raise RuntimeError("temporal view overlaps the dev-val calibrator-fit rows")
    if set(temporal_ids) & set(fit_id_list):
        raise RuntimeError("temporal view overlaps the allowed survival-fit rows")
    held_idx = held_out_healthy_indices(
        temporal_ids,
        [bool(r["abnormal"]) for r in file_rows],
        [bool(r["quarantined"]) for r in file_rows],
        list(dev_val_ids) + fit_id_list,
    )
    if not held_idx:
        out["conformal_healthy_coverage_independent"] = {
            "coverage": None,
            "reason": "no held-out healthy temporal files",
            "n": 0,
        }
    else:
        held_disp = torch.tensor(
            [float(file_rows[i]["displacement"]) for i in held_idx],
            dtype=torch.float32,
        )
        out["conformal_healthy_coverage_independent"] = {
            "coverage": float(pipeline.calibrator.coverage(held_disp, level=0.95)),
            "level": 0.95,
            "n": len(held_idx),
            "cohort": "held-out healthy temporal (normal, non-quarantined)",
            "confidence_calibrator_fit_cohort": CONFIDENCE_CALIBRATOR_FIT_COHORT,
            "disjoint_from_dev_val": True,
            "disjoint_from_survival_fit": True,
        }
    _, eval_label_7d = CensoredSurvivalRisk.horizon_cohort(
        fit_days, fit_events, horizons_cfg[1]
    )
    fit_neg_mask = ~eval_label_7d
    if int(fit_neg_mask.sum().item()) == 0:
        out["conformal_healthy_coverage_fit_insample"] = {
            "coverage": None,
            "reason": "allowed fit cohort carries no 7d negatives",
            "n": 0,
        }
    else:
        out["conformal_healthy_coverage_fit_insample"] = {
            "coverage": float(
                pipeline.calibrator.coverage(
                    fit_features[:, 0][fit_neg_mask], level=0.95
                )
            ),
            "level": 0.95,
            "n": int(fit_neg_mask.sum().item()),
            "cohort": "allowed-fit 7d negatives",
            "note": (
                "IN-SAMPLE DIAGNOSTIC: these rows fit the survival head; "
                "not independent coverage"
            ),
            "n_fit_overlaps_dev_val_calibrator_rows": n_fit_overlaps_devval,
        }
    # Separately labeled population trajectory evidence (independent view;
    # elevated fractions excluded from detection claims — the restored
    # operating threshold is calibrated on context_energy, uncalibrated
    # cross-signal).
    pop_tail = [float(r["population_tail_energy"]) for r in file_rows]
    pop_tail_abn = [
        float(r["population_tail_energy"]) for r in file_rows if bool(r["abnormal"])
    ]
    pop_tail_norm = [
        float(r["population_tail_energy"])
        for r in file_rows
        if not bool(r["abnormal"])
    ]
    out["population_view"] = {
        "energy_source": POPULATION_ENERGY_FIELD,
        "tail_median_overall": float(np.median(pop_tail)),
        "tail_p95_overall": float(np.quantile(pop_tail, 0.95)),
        "tail_median_abnormal": float(np.median(pop_tail_abn)) if pop_tail_abn else None,
        "tail_median_normal": float(np.median(pop_tail_norm)) if pop_tail_norm else None,
        "mean_median_overall": float(
            np.median([float(r["population_mean_energy"]) for r in file_rows])
        ),
        "elevated_fraction_excluded": (
            "EXCLUDED from detection claims — the restored operating threshold "
            "is calibrated on context_energy; applied to population-scale "
            "energies it is uncalibrated cross-signal"
        ),
    }
    out["trajectory_summary"] = {
        "energy_source": CONTEXT_ENERGY_FIELD,
        "displacement_median": float(np.median(disp_all)),
        "displacement_p95": float(np.quantile(disp_all, 0.95)),
        "displacement_max": float(np.max(disp_all)),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 11 Batch F2 temporal analysis")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--control-ckpt", required=True)
    parser.add_argument("--hybrid-ckpt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
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
    for view in ("dev_train", "dev_val", "test_static", "test_temporal"):
        if view not in manifest["splits"] or not manifest["splits"][view]:
            raise RuntimeError(f"Chronicle view {view!r} is empty; regenerate the dataset.")
    dev_ids = set(manifest["splits"]["dev_train"]) | set(manifest["splits"]["dev_val"])
    temporal_ids = list(manifest["splits"]["test_temporal"])
    temporal_files = [by_id[i] for i in temporal_ids]
    sealed_ids = set(manifest["splits"]["test_static"]) | set(temporal_ids)
    cutoff = float(manifest["calendar"]["cutoff_time"])

    probe = V2InferencePipeline.load(control_ckpt, device="cpu")
    patchifier = Patchifier(
        PatchConfig(patch_size=probe.config.patch_size, stride=probe.config.stride)
    )
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
        f"[Views] temporal={len(temporal_files)} dev={len(dev_ids)} "
        f"cutoff={cutoff} (static untouched)",
        flush=True,
    )

    # Cold-start assessment from persisted provenance (no model access needed).
    dev_robots = {_robot_of(by_id[i]) for i in dev_ids}
    temporal_robots = {_robot_of(by_id[i]) for i in temporal_ids}
    dev_programs = {_program_of(by_id[i]) for i in dev_ids}
    temporal_programs = {_program_of(by_id[i]) for i in temporal_ids}
    cold_robots = sorted(temporal_robots - dev_robots)
    cold_programs = sorted(temporal_programs - dev_programs)
    cold_start = {
        "dev_robots": sorted(dev_robots),
        "temporal_robots": sorted(temporal_robots),
        "cold_start_robots": cold_robots,
        "dev_programs": sorted(dev_programs),
        "temporal_programs": sorted(temporal_programs),
        "cold_start_programs": cold_programs,
        "status": (
            "available"
            if cold_robots
            else "UNAVAILABLE: every temporal robot appears in dev views; "
            "no genuine held-out robot exists in this materialization"
        ),
    }
    print(f"[Cold-start] robots cold={cold_robots} programs cold={cold_programs}", flush=True)

    # Allowed survival-fit cohort (identical rule to the canonical notebook).
    fit_candidates = [
        s
        for s in samples
        if s.file_id not in sealed_ids
        and s.operation.start_time < cutoff
        and s.future_targets is not None
        and s.future_targets.time_to_next_failure is not None
        and not s.future_targets.is_censored
    ]
    if any(s.file_id in sealed_ids for s in fit_candidates):
        raise RuntimeError("sealed test files reached the survival fit cohort")
    if not fit_candidates:
        raise RuntimeError("allowed pre-cutoff survival cohort is empty")
    n_fit_quarantined = sum(
        1
        for s in fit_candidates
        if s.split_provenance is not None and s.split_provenance.is_quarantined
    )
    print(
        f"[Survival-fit] allowed cohort={len(fit_candidates)} "
        f"quarantined={n_fit_quarantined}",
        flush=True,
    )

    results: dict[str, dict[str, object]] = {}
    dev_val_ids = set(manifest["splits"]["dev_val"])
    fit_file_ids = [s.file_id for s in fit_candidates]
    for name, ckpt in (("control", control_ckpt), ("hybrid", hybrid_ckpt)):
        pipeline = V2InferencePipeline.load(ckpt, device=args.device)
        payload = load_v2_checkpoint(ckpt, pipeline.model, expected_config=pipeline.config)
        tracker_state = payload.get("tracker")
        if not isinstance(tracker_state, dict):
            raise RuntimeError(f"Checkpoint {ckpt} carries no trajectory baseline.")
        if pipeline.calibrator is None:
            raise RuntimeError(f"Checkpoint {ckpt} carries no confidence calibrator.")
        operating_record = payload.get("operating_threshold")
        calibrator_record = payload.get("confidence_calibrator_fit_cohort")
        restored_operating = (
            dict(operating_record) if isinstance(operating_record, dict) else None
        )
        restored_calibrator = (
            dict(calibrator_record) if isinstance(calibrator_record, dict) else None
        )
        file_rows, n_breaks = score_temporal(
            pipeline, tracker_state, temporal_files, patchifier, args.device
        )
        risk, fit_features, fit_days, fit_events, fit_error = fit_allowed_risk(
            pipeline, tracker_state, fit_candidates, patchifier, args.seed
        )
        variant = build_variant_temporal(
            name, pipeline, file_rows, by_id, manifest,
            None if fit_error else risk,
            fit_features, fit_days, fit_events, fit_error,
            restored_operating, restored_calibrator, dev_val_ids, fit_file_ids,
        )
        variant["maintenance_breaks_scored"] = n_breaks
        variant["n_fit_files"] = int(fit_features.shape[0])
        variant["n_fit_quarantined"] = n_fit_quarantined
        variant["fit_source"] = "pre-cutoff non-test files with observed failure times only"
        results[name] = variant
        print(
            f"[{name}] files={len(file_rows)} breaks={n_breaks} "
            f"status={variant['risk_status']} "
            f"recall1d={variant.get('event_recall_1d')} "
            f"recall7d={variant.get('event_recall_7d')}",
            flush=True,
        )
        del pipeline

    check_mapping_nonempty(results, "variant results")
    task35 = {
        "provenance": {
            "commit": commit,
            "manifest_sha256": manifest_sha,
            "config_hash": manifest["config_hash"],
            "control_sha256": control_sha,
            "hybrid_sha256": hybrid_sha,
            "device": device_name,
            "torch": torch.__version__,
            "seed": args.seed,
            "n_temporal": len(temporal_files),
            "n_fit": len(fit_candidates),
            "confidence_calibrator_fit_cohort": CONFIDENCE_CALIBRATOR_FIT_COHORT,
            "operating_threshold_fit_cohort": RISK_OPERATING_FIT_COHORT,
            "control_restored_operating_threshold": results["control"].get(
                "restored_operating_threshold"
            ),
            "hybrid_restored_operating_threshold": results["hybrid"].get(
                "restored_operating_threshold"
            ),
            "control_restored_confidence_cohort": results["control"].get(
                "restored_confidence_cohort"
            ),
            "hybrid_restored_confidence_cohort": results["hybrid"].get(
                "restored_confidence_cohort"
            ),
            "prior_f2_superseded": (
                "e669e2a (legacy tail/elevated names without energy_source, "
                "ambiguous calibration provenance, pooled fit-cohort coverage)"
            ),
            "corrected_f1_evidence": "e725250 (canonical context/population signals)",
        },
        "cold_start": cold_start,
        "control": results["control"],
        "hybrid": results["hybrid"],
    }
    (out_root / "task35_temporal.json").write_text(
        json.dumps(task35, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    # Bounded per-group slices CSV (robot + program rows, both variants).
    import csv

    slice_path = out_root / "task35_slices.csv"
    with open(slice_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant", "group_kind", "group",
                "n_files", "recall_1d", "recall_7d",
                "auroc_7d", "auprc_7d", "brier_7d", "brier_reference_const_7d", "ece_7d",
                "false_runs", "false_runs_per_30d",
                "warned_rate_in_7d_window", "lead_7d_median_days", "lead_7d_n",
                "population_tail_median",
            ]
        )
        for variant in ("control", "hybrid"):
            res = results[variant]
            by_robot = res.get("by_robot", {})
            for robot in sorted(by_robot):
                row = by_robot[robot]
                lead = row.get("lead_7d_days", {})
                writer.writerow(
                    [
                        variant, "robot", robot, row.get("n_files"),
                        row.get("recall_1d"), row.get("recall_7d"),
                        row.get("auroc_7d"), row.get("auprc_7d"),
                        row.get("brier_7d"), row.get("brier_reference_const_7d"),
                        row.get("ece_7d"),
                        row.get("false_warning_runs"), row.get("false_runs_per_30d"),
                        "", lead.get("median"), lead.get("n"),
                        row.get("population_tail_median"),
                    ]
                )
            by_program = res.get("by_program", {})
            for program in sorted(by_program):
                row = by_program[program]
                writer.writerow(
                    [
                        variant, "program", program, row.get("n_files"),
                        "", "",
                        row.get("auroc_7d"), row.get("auprc_7d"), "", "", "",
                        "", "",
                        row.get("warned_rate_in_7d_window"), "", "", "",
                    ]
                )
    print("[Slices] wrote task35_slices.csv", flush=True)

    # Bounded figures (review-sized).
    fig_dir = out_root / "figures"
    for variant in ("control", "hybrid"):
        res = results[variant]
        bins = res.get("calibration_bins_7d", [])
        if bins:
            fig, ax = plt.subplots(figsize=(5, 4))
            xs = [b[0] for b in bins if np.isfinite(b[1])]
            ys = [b[1] for b in bins if np.isfinite(b[1])]
            ax.plot(xs, ys, marker="o", label="observed 7d")
            ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
            ax.set_title(f"{variant}: 7d risk calibration (censored cohort)")
            ax.set_xlabel("mean predicted risk")
            ax.set_ylabel("observed failure rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"task35_calibration_7d_{variant}.png", dpi=100)
            plt.close(fig)
        lead = res.get("lead_time_7d_days", {})
        # Lead-time histogram needs row data; distribution summary ships in JSON.
        _ = lead
    # Lead-time distribution comparison from JSON summaries is not plottable
    # without row data; emit warned-rate bars per robot instead.
    fig, ax = plt.subplots(figsize=(6, 4))
    robots = sorted(results["control"].get("by_robot", {}))
    x = np.arange(len(robots))
    width = 0.35
    for k, variant in enumerate(("control", "hybrid")):
        rates = [
            results[variant]["by_robot"][r].get("false_runs_per_30d") or 0.0
            for r in robots
        ]
        ax.bar(x + (k - 0.5) * width, rates, width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(robots)
    ax.set_ylabel("false-warning runs per 30 robot-days")
    ax.set_title("False-warning runs by robot (7d operating point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "task35_false_alerts_by_robot.png", dpi=100)
    plt.close(fig)
    print("[Figures] wrote calibration curves + false-alert bars", flush=True)

    run_manifest = {
        "commit": commit,
        "data_root": str(data_root),
        "manifest_sha256": manifest_sha,
        "config_hash": manifest["config_hash"],
        "control_checkpoint": str(control_ckpt),
        "control_sha256": control_sha,
        "hybrid_checkpoint": str(hybrid_ckpt),
        "hybrid_sha256": hybrid_sha,
        "device": device_name,
        "torch": torch.__version__,
        "seed": args.seed,
        "scoring": "per-file chronological per-robot (tracker state, suspect guard)",
        "signals": "context_energy acute chain + separately labeled population_energy view",
        "confidence_calibrator_fit_cohort": CONFIDENCE_CALIBRATOR_FIT_COHORT,
        "operating_threshold_fit_cohort": RISK_OPERATING_FIT_COHORT,
        "n_temporal": len(temporal_files),
        "n_fit": len(fit_candidates),
        "n_fit_quarantined": n_fit_quarantined,
        "prior_f2_superseded": "e669e2a",
        "corrected_f1_evidence": "e725250",
        "static_view_accessed": False,
        "temporal_view_accessed": True,
    }
    (out_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[Done] outputs under", str(out_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
