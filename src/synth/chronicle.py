"""Chronological dataset materialization and manifest (Task 7).

Public generation entry point for client-scale and server-scale
chronological factory datasets (methodology section 20). Runs the full
accepted pipeline — scheduler, health, scheduled signals, temporal
anomalies, labels/quarantine/splits — then persists sharded archives
plus schedule, episode, split, checksum, seed, count, and configuration
manifests.

Roots are always explicit: materialization fails fast on invalid roots
or existing manifests (unless ``overwrite``), and loading fails fast on
missing manifests, hash mismatches, or incomplete shards. Nothing scans
mounts and nothing silently falls back. Generated data roots stay
outside version control (``data/generated/``, ``*.npz``).
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from synth.config import (
    GENERATOR_VERSION,
    CohortConfig,
    FactoryCalendarConfig,
    HealthConfig,
    RouteConfig,
    RouteStageConfig,
    SchedulerConfig,
    SynthConfig,
)
from synth.dataset import (
    _atomic_json,
    _episode_to_dict,
    _hash_file,
    _operation_to_dict,
    _sample_bytes,
    load_sample_bytes,
)
from synth.health import RobotHealthProcess
from synth.patchify import Patchifier
from synth.scheduled import ScheduledSignalGenerator
from synth.scheduler import FactoryScheduler
from synth.schema import EpisodeKind, SampleLabel
from synth.splits import ChronologicalSplitter
from synth.temporal import TemporalAnomalyProcess

#: Manifest format version for chronological datasets.
CHRONICLE_FORMAT = 1

#: Fixed ZIP entry timestamp so shard layouts stay comparable.
_SHARD_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def client_config(seed: int = 0) -> SynthConfig:
    """Return the tiny multi-robot routed client profile.

    Three robots on two alternating two-stage routes; the 60-unit
    calendar spans the 45-day development cutoff with failures on both
    line robots, exercising quarantine, splits, and both test views.
    """
    cfg = SynthConfig()
    cfg.factory = FactoryCalendarConfig(
        span_days=90.0, dev_cutoff_days=45.0, quarantine_days=7.0, seed=seed
    )
    cfg.scheduler = SchedulerConfig(
        n_units=60,
        arrival_interval_s=86400.0,
        routes=[
            RouteConfig(
                route_id="route-A",
                product_type="sedan",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-01", program_id="program-01",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-02", program_id="program-02",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
            RouteConfig(
                route_id="route-B",
                product_type="hatch",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-02", program_id="program-03",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-03", program_id="program-01",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
        ],
        seed=seed,
    )
    cfg.health = HealthConfig(
        seed=seed, aging_rate=5e-7, wear_rate=1e-4, noise_scale=1e-3,
        base_rate=2e-6, alpha=2.5, beta=0.0, abrupt_rate=0.0,
    )
    cfg.signal.seed = seed
    cfg.temporal.seed = seed
    return cfg


def server_config(seed: int = 0) -> SynthConfig:
    """Return the larger server profile over the same line topology."""
    cfg = client_config(seed=seed)
    cfg.scheduler.n_units = 300
    cfg.scheduler.arrival_interval_s = 21600.0
    cfg.scheduler.arrival_jitter_s = 21600.0
    return cfg

#: Sprint 13 operating point (Protocol v4 §2): 180-day histories on eight
#: robots with three competing failure cohorts at 22-day per-robot MTBF.
SPRINT13_SEEDS = (300, 301, 302, 303, 304, 305, 306, 307, 308,
                  400, 401, 402, 403)


def sprint13_history_config(seed: int = 0) -> SynthConfig:
    """Return the Protocol v4 benchmark history configuration.

    Eight robots on five two-stage routes (program-03 rides route-B,
    robot-08 closes route-E); 180-day span with day-120 development cutoff;
    competing P/W/A cohort hazards; preventive maintenance every 30 days
    """

    cfg = SynthConfig()
    cfg.factory = FactoryCalendarConfig(
        span_days=180.0, dev_cutoff_days=120.0, quarantine_days=7.0, seed=seed
    )
    cfg.scheduler = SchedulerConfig(
        n_units=1152,
        arrival_interval_s=11200.0,
        arrival_jitter_s=11200.0,
        routes=[
            RouteConfig(
                route_id="route-A",
                product_type="sedan",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-01", program_id="program-01",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-02", program_id="program-02",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
            RouteConfig(
                route_id="route-B",
                product_type="hatch",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-02", program_id="program-03",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-03", program_id="program-04",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
            RouteConfig(
                route_id="route-C",
                product_type="sedan",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-04", program_id="program-05",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-05", program_id="program-06",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
            RouteConfig(
                route_id="route-D",
                product_type="hatch",
                stages=[
                    RouteStageConfig(
                        robot_id="robot-06", program_id="program-07",
                        duration_s=600.0, travel_after_s=60.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-07", program_id="program-08",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                    RouteStageConfig(
                        robot_id="robot-08", program_id="program-01",
                        duration_s=600.0, travel_after_s=0.0,
                    ),
                ],
            ),
        ],
        seed=seed,
    )
    cfg.fleet.n_robots = 8
    cfg.health = HealthConfig(
        seed=seed, aging_rate=5e-7, wear_rate=1e-4, noise_scale=1e-3,
        base_rate=0.0, alpha=2.5, beta=0.0, abrupt_rate=0.0,
        degradation_onset=0.3, severity_scale=2.0,
        maintenance_duration_s=172800.0,
        preventive_interval_s=30.0 * 86400.0,
        preventive_duration_s=86400.0,
        cohorts=(
            CohortConfig(cohort_id="P", share=0.45, base_rate=3.0e-7,
                         degradation_min_d=7.0, degradation_max_d=14.0),
            CohortConfig(cohort_id="W", share=0.30, base_rate=3.6e-7,
                         amplitude_scale=0.3,
                         degradation_min_d=14.0, degradation_max_d=28.0),
            CohortConfig(cohort_id="A", share=0.25, abrupt_rate=1.6e-5,
                         subtypes=("A1", "A2")),
        ),
    )
    cfg.signal.seed = seed
    cfg.temporal.seed = seed
    return cfg


#: Sprint 13 Protocol v4.1 roster (fresh identities; v4 300s/400s retired).
V41_SEEDS = (500, 501, 502, 503, 504, 505, 506, 507, 508,
             600, 601, 602, 603)

#: Sprint 14 Protocol v3 roster (cycle-1 Design 700–703; Fit 710–712;
#: Calibration 713; Confirmation 720–723; Sealed 730–733).
S14_SEEDS = (700, 701, 702, 703, 710, 711, 712, 713,
             720, 721, 722, 723, 730, 731, 732, 733)


def sprint13_v41_history_config(seed: int = 0) -> SynthConfig:
    """Return the Protocol v4.1 benchmark history configuration.

    Same 180-day, 8-robot, five-route factory as v4, with v4.1 §2–§4
    mechanics: analytic hazard rates, cohort-specific wear
    (w_P = 3.0e-4, w_W = 1.0e-4, base 2.5e-5, aging 1e-7), gated base
    hazard on open degradation episodes, and pre-failure upcoming-cohort
    draws. Only the seed varies between histories.
    """
    cfg = sprint13_history_config(seed=seed)
    cfg.health = HealthConfig(
        seed=seed, aging_rate=1e-7, wear_rate=3.0e-5, noise_scale=1e-3,
        base_rate=0.0, alpha=5.0, beta=0.0, abrupt_rate=0.0,
        degradation_onset=0.3, severity_scale=2.0, upcoming_p=0.55,
        maintenance_duration_s=172800.0,
        preventive_interval_s=30.0 * 86400.0,
        preventive_duration_s=86400.0,
        cohorts=(
            CohortConfig(cohort_id="P", share=0.45, base_rate=5.0e-10,
                         wear_rate=2.0e-4, failure_threshold_h=2.0,
                         degradation_min_d=7.0, degradation_max_d=14.0),
            CohortConfig(cohort_id="W", share=0.30, base_rate=3.0e-9,
                         wear_rate=5.0e-5, failure_threshold_h=1.35,
                         amplitude_scale=0.3,
                         degradation_min_d=14.0, degradation_max_d=28.0),
            CohortConfig(cohort_id="A", share=0.25, abrupt_rate=1.1e-5,
                         wear_rate=0.0, subtypes=("A1", "A2")),
        ),
    )
    return cfg


def sprint14_v3_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 14 Protocol v3 benchmark history configuration.

    v4.1 factory with the cycle-1 Option A amendment: a ninth robot ends
    robot-02 double duty (route-B first stage moves to ``robot-09`` with
    ``program-03`` staying on route-B); P/W cohorts declare physical
    manifestation subtypes (protocol v3 §1). All rates, physics, units,
    cadence, and route volumes identical to v4.1.
    """
    cfg = sprint13_v41_history_config(seed=seed)

    routes = []
    for route in cfg.scheduler.routes:
        stages = [
            RouteStageConfig(
                robot_id=("robot-09" if (route.route_id == "route-B"
                                        and stage.robot_id == "robot-02")
                          else stage.robot_id),
                program_id=stage.program_id,
                duration_s=stage.duration_s,
                travel_after_s=stage.travel_after_s,
            )
            for stage in route.stages
        ]
        routes.append(RouteConfig(
            route_id=route.route_id,
            product_type=route.product_type,
            stages=stages,
        ))
    cfg.scheduler.routes = routes
    cfg.fleet.n_robots = 9
    subtype_labels = {"P": ("P1", "P2"), "W": ("W1", "W2"),
                      "A": ("A1", "A2")}
    cohorts = [
        replace(cohort, subtypes=subtype_labels[cohort.cohort_id])
        for cohort in cfg.health.cohorts
    ]
    cfg.health = replace(cfg.health, cohorts=tuple(cohorts))
    return cfg


def sprint14_v5_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 14 Protocol v5 benchmark history configuration.

    v3/v4 DGP with the cycle-3 A+C amendment: preventive cadence 30 d
    → 60 d (1 d duration kept), corrective downtime 2 d → 1 d, and a
    uniform 240-day calendar (cutoff 160 d, 1536 units at unchanged
    arrival cadence). All cohort rates, physics, routes, holdouts,
    quarantine, and eligibility semantics identical to v4/v4.1.
    """
    cfg = sprint14_v3_history_config(seed=seed)
    cfg.factory = replace(
        cfg.factory, span_days=240.0, dev_cutoff_days=160.0)
    cfg.scheduler = replace(cfg.scheduler, n_units=1536)
    cfg.health = replace(
        cfg.health, maintenance_duration_s=86400.0,
        preventive_interval_s=60.0 * 86400.0)
    return cfg


def sprint15_v1_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v1 balanced history configuration.

    v5 factory with the Batch B hypothesis-3 amendment (H1 measured
    infeasible at 240 d: P/P2 shortfall, 32 < 48 controls with ~70% of
    robot-days clearance-blocked; H2 blocked by the 250-day code guard):
    360-day span with units 1536 -> 2304 at unchanged arrival cadence
    (density, durations, and wear physics preserved; the guard widens
    250 -> 400 d with all eligibility predicates unchanged), development
    cutoff at 50% (180 d), abrupt rate 1.1e-5 -> 2.2e-5 ~s-1 (A pool covers
    the exact 8/8 quota after reset exclusions), upcoming P draw 0.55 ->
    0.52 with W episode wear 5.0e-5 -> 6.2e-5 (W pool margin for the exact
    12/12 split). Routes, robots, cadence, maintenance, quarantine, gains,
    thresholds, and eligibility semantics identical to v5. Predicted:
    P ~25/25, W ~23/23, A eligible ~13/13, controls ~55, duration shapes
    preserved; verified on proof roots before any Design materialization
    (Tasks 9/11).
    """
    cfg = sprint14_v5_history_config(seed=seed)
    cfg.factory = replace(
        cfg.factory, span_days=360.0, dev_cutoff_days=180.0)
    cfg.scheduler = replace(cfg.scheduler, n_units=2304)
    cohorts = []
    for cohort in cfg.health.cohorts:
        if cohort.cohort_id == "W":
            cohorts.append(replace(cohort, wear_rate=6.2e-5))
        elif cohort.cohort_id == "A":
            cohorts.append(replace(cohort, abrupt_rate=2.2e-5))
        else:
            cohorts.append(cohort)
    cfg.health = replace(
        cfg.health, upcoming_p=0.52, cohorts=tuple(cohorts))
    return cfg


def sprint15_v2_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v2 balanced history configuration.

    Identical to ``sprint15_v1_history_config`` (Batch B hypothesis-3
    settings carried forward unchanged) with exactly one addition:
    ``health.min_duration_gate`` enabled per protocol v2 §7a, so non-abrupt
    cohorts cannot fire before the frozen audit minimum durations
    (P 2.0 d, W 6.0 d). All rates, wear, thresholds, gains, routes, calendar,
    maintenance, quarantine, and eligibility semantics identical to v1.
    """
    cfg = sprint15_v1_history_config(seed=seed)
    cfg.health = replace(cfg.health, min_duration_gate=True)
    return cfg


def sprint15_v3_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v3 balanced history configuration.

    Identical to ``sprint15_v2_history_config`` (Batch B hypothesis-3
    settings plus the §7a minimum-duration firing gate carried forward
    unchanged). Protocol v3 changes only acceptance scheduling (§3a joint
    fair rounds, implemented in the allocator) — no generator physics,
    rates, wear, thresholds, gains, routes, calendar, maintenance,
    quarantine, or eligibility semantics differ from v2.
    """
    cfg = sprint15_v2_history_config(seed=seed)
    return cfg


def sprint15_v4_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v4 balanced history configuration.

    Identical to ``sprint15_v3_history_config`` (Batch B hypothesis-3
    settings plus the §7a minimum-duration firing gate carried forward
    unchanged). Protocol v4 changes only acceptance scheduling (§3b exact
    deterministic CSP, implemented in the allocator) — no generator physics,
    rates, wear, thresholds, gains, routes, calendar, maintenance,
    quarantine, or eligibility semantics differ from v3.
    """
    cfg = sprint15_v3_history_config(seed=seed)
    return cfg


def sprint15_v5_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v5 balanced history configuration.

    Identical to ``sprint15_v4_history_config`` except the §1 calendar-volume
    margin: operation volume scales 1.5× at frozen per-unit mechanics
    (``n_units`` 2304 → 3456 with the arrival interval/jitter rescaled
    inversely to preserve the frozen 360-day span). Routes, fleet, span, dev
    cutoff, quarantine, per-unit failure physics/rates/wear/thresholds/gains,
    maintenance, §7a gate, and all eligibility semantics identical to v4.
    """
    cfg = sprint15_v4_history_config(seed=seed)
    scale = 2304 / 3456
    cfg.scheduler = replace(
        cfg.scheduler,
        n_units=3456,
        arrival_interval_s=cfg.scheduler.arrival_interval_s * scale,
        arrival_jitter_s=cfg.scheduler.arrival_jitter_s * scale,
    )
    return cfg


#: Candidate-6 fixed-budget cohort outcome shares (protocol v6 §1a).
#: Documentary targets summing to 1.0; realized mix emerges from the rates.
S15_V6_SHARES = {"P": 0.41, "W": 0.28, "A": 0.31}

#: Candidate-6 dominant hazard rates (protocol v6 §1a): abrupt up 30%,
#: P/W base down 8%/9% at fixed total density. All other cohort fields
#: (wear, thresholds, degradation bounds, amplitude, subtypes, upcoming draw)
#: are inherited unchanged from the v4 configuration.
S15_V6_RATES = {"P": 4.6e-10, "W": 2.73e-9, "A": 2.86e-5}


def sprint15_v6_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v6 balanced history configuration.

    Built from ``sprint15_v4_history_config`` (the v5 densification is fully
    reverted: ``n_units`` 2304 with arrival interval/jitter ``11200.0 s``),
    with exactly one change: the §1a fixed-budget cohort reallocation
    (documentary shares P 0.41 / W 0.28 / A 0.31 via dominant hazard rates
    A 2.86e-5, P 4.6e-10, W 2.73e-9). Upcoming draw, wear, thresholds, gains,
    routes, fleet, span, dev cutoff, quarantine, maintenance, §7a gate, and
    all eligibility semantics identical to v4.
    """

    cfg = sprint15_v4_history_config(seed=seed)
    cohorts = []
    for cohort in cfg.health.cohorts:
        if cohort.cohort_id == "A":
            cohorts.append(replace(
                cohort, share=S15_V6_SHARES["A"],
                abrupt_rate=S15_V6_RATES["A"]))
        elif cohort.cohort_id in ("P", "W"):
            cohorts.append(replace(
                cohort, share=S15_V6_SHARES[cohort.cohort_id],
                base_rate=S15_V6_RATES[cohort.cohort_id]))
        else:  # pragma: no cover - cohort alphabet fixed to P/W/A
            cohorts.append(cohort)
    cfg.health = replace(cfg.health, cohorts=tuple(cohorts))
    return cfg


def sprint15_v7_history_config(seed: int = 0) -> SynthConfig:
    """Return the Sprint 15 Protocol v7 balanced history configuration.

    Identical to ``sprint15_v4_history_config`` in every scheduler, fleet,
    factory, cohort-share/rate, upcoming-draw, wear, threshold, gain, route,
    maintenance, and §7a setting, with exactly one change: the §1a
    deterministic subtype-stratification flag
    (``health.stratified_subtype_emission = True``). The v5 densification and
    v6 rate shifts are fully reverted by construction (built from v4, never
    from v5/v6).
    """
    cfg = sprint15_v4_history_config(seed=seed)
    cfg.health = replace(cfg.health, stratified_subtype_emission=True)
    return cfg


def _require_root(root: str | Path, *, must_exist: bool) -> Path:
    if root is None or (isinstance(root, str) and not root.strip()):
        raise ValueError("chronological root must be an explicit path")
    path = Path(root)
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"chronological root not found: {path}")
    return path


def _failure_to_dict(record) -> dict[str, object]:
    """Render one FailureEvent ledger record for the manifest."""
    return {
        "failure_id": record.failure_id,
        "robot_id": record.robot_id,
        "failure_time": record.failure_time,
        "cohort": record.cohort,
        "subtype": record.subtype,
        "degradation_onset": record.degradation_onset,
        "duration_d": record.duration_d,
        "severity": record.severity,
        "degradation_episode_id": record.degradation_episode_id,
        "maintenance_episode_id": record.maintenance_episode_id,
    }


def _maintenance_windows(health) -> dict[str, list[list[float]]]:
    """Index bounded maintenance windows per robot for reset derivation."""
    windows: dict[str, list[list[float]]] = {}
    for episode in health.episodes:
        if episode.kind is not EpisodeKind.MAINTENANCE:
            continue
        assert episode.end_time is not None
        windows.setdefault(episode.robot_id, []).append(
            [episode.start_time, episode.end_time])
    for intervals in windows.values():
        intervals.sort()
    return windows

def _last_reset_time(
    windows: dict[str, list[list[float]]], robot_id: str, start_time: float
) -> float:
    """Return the last recommissioning end at or before a file start."""
    reset = 0.0
    for window_start, window_end in windows.get(robot_id, []):
        if window_end <= start_time:
            reset = max(reset, window_end)
        elif window_start > start_time:
            break
    return reset


def write_seal(root: str | Path, role: str) -> dict[str, object]:
    """Seal one materialized history against training/selection access.

    Records the manifest digest, config hash, seeds, role, and the
    manifest's own protocol tag in ``seal.json``. The protocol must be a
    known benchmark version (Sprint 13 v4/v4.1, Sprint 14 v3/v4/v5, or
    Sprint 15 v1/v2/v3/v4/v5/v6/v7);
    manifests without a tag keep the v4 default so old seals stay valid.
    Deterministic: no timestamps.
    """
    out = _require_root(root, must_exist=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol") or "sprint13-protocol-v4"
    if protocol not in ("sprint13-protocol-v4", "sprint13-protocol-v4.1",
                        "sprint14-benchmark-protocol-v3",
                        "sprint14-benchmark-protocol-v4",
                        "sprint15-benchmark-protocol-v1",
                        "sprint15-benchmark-protocol-v2",
                        "sprint15-benchmark-protocol-v3",
                        "sprint15-benchmark-protocol-v4",
                        "sprint15-benchmark-protocol-v5",
                        "sprint15-benchmark-protocol-v6",
                        "sprint15-benchmark-protocol-v7"):
        raise ValueError(f"unknown benchmark protocol {protocol!r} at {out}")
    seal = {
        "protocol": protocol,
        "role": role,
        "manifest_sha256": _hash_file(manifest_path),
        "config_hash": manifest["config_hash"],
        "seeds": manifest["seeds"],
    }
    _atomic_json(out / "seal.json", seal)
    return seal


def verify_seal(root: str | Path) -> dict[str, object]:
    """Verify a sealed history root; raise on any mismatch or tamper."""
    out = _require_root(root, must_exist=True)
    seal_path = out / "seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError(f"seal not found: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if seal.get("manifest_sha256") != _hash_file(manifest_path):
        raise ValueError(f"seal manifest digest mismatch at {out}")
    if seal.get("config_hash") != manifest.get("config_hash"):
        raise ValueError(f"seal config hash mismatch at {out}")
    if seal.get("protocol") != (manifest.get("protocol")
                                or "sprint13-protocol-v4"):
        raise ValueError(f"seal protocol mismatch at {out}")
    return seal


def build_chronological(cfg: SynthConfig):
    """Run the full scheduler → splits pipeline for one config."""
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    base = ScheduledSignalGenerator(cfg).generate(schedule, health)
    manifested, _ = TemporalAnomalyProcess(cfg).apply(
        schedule, health, base
    )
    labeled, splits = ChronologicalSplitter(cfg).build(
        schedule, health, manifested
    )
    return schedule, health, labeled, splits


def _manifest_file_rows(labeled, maint_windows, patch_counts):
    """Build the model-visible file-row table (shared persist/allocate path)."""
    return [
        {
            "file_id": s.file_id,
            "operation_id": s.operation.operation_id,  # type: ignore[union-attr]
            "robot_id": s.operation.robot_id,  # type: ignore[union-attr]
            "program_id": s.operation.program_id,  # type: ignore[union-attr]
            "start_time": s.operation.start_time,  # type: ignore[union-attr]
            "end_time": s.operation.end_time,  # type: ignore[union-attr]
            "file_label": s.file_label.value,
            "is_quarantined": s.split_provenance.is_quarantined,  # type: ignore[union-attr]
            "quarantine_reason": s.split_provenance.quarantine_reason,  # type: ignore[union-attr]
            "is_censored": s.future_targets.is_censored,  # type: ignore[union-attr]
            "member_views": list(s.split_provenance.member_views),  # type: ignore[union-attr]
            "last_reset_time": _last_reset_time(
                maint_windows,
                s.operation.robot_id,  # type: ignore[union-attr]
                s.operation.start_time,  # type: ignore[union-attr]
            ),
            "n_valid_patches": patch_counts[s.file_id],
        }
        for s in labeled
    ]


def materialize_chronological(
    cfg: SynthConfig,
    root: str | Path,
    *,
    shard_size: int = 64,
    overwrite: bool = False,
    role: str | None = None,
    protocol: str | None = None,
    sprint15=None,
) -> dict[str, object]:
    """Persist one chronological dataset plus complete manifests.

    ``role`` names the whole-history assignment (or None when unassigned);
    ``protocol`` names the frozen benchmark version (defaults to
    ``sprint13-protocol-v4`` when a role is given — pass explicitly,
    e.g. ``sprint13-protocol-v4.1``, for amended runs). Both recorded verbatim.
    ``sprint15`` carries a ``synth.balanced.Sprint15Binding`` for Sprint 15
    histories: quota allocation runs fail-fast before shard I/O and a
    ``sprint15`` provenance block joins the manifest. ``None`` (default)
    preserves the legacy path byte-for-byte.
    """
    out = _require_root(root, must_exist=False)
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"manifest already exists: {manifest_path}; use overwrite"
        )
    schedule, health, labeled, splits = build_chronological(cfg)
    config_hash = cfg.hash()
    maint_windows = _maintenance_windows(health)
    if protocol is None and role is not None:
        protocol = "sprint13-protocol-v4"
    patchifier = Patchifier(cfg.patch)
    patch_counts = {
        sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
        for sample in labeled
    }
    sprint15_block = None
    if sprint15 is not None:
        from synth.balanced import prepare_sprint15_block

        if not overwrite and out.exists() and any(out.iterdir()):
            raise FileExistsError(
                f"sprint15 target root exists and is non-empty: {out}; "
                "refusing to start without overwrite"
            )
        pre_rows = _manifest_file_rows(labeled, maint_windows, patch_counts)
        pre_ledger = [_failure_to_dict(r) for r in health.failure_events]
        resolved = json.loads(
            json.dumps(asdict(cfg), sort_keys=True, default=str)
        )
        sprint15_block = prepare_sprint15_block(
            rows=pre_rows,
            ledger=pre_ledger,
            wins=maint_windows,
            schedule_events=[
                _operation_to_dict(e) for e in schedule.events
            ],
            resolved_config=resolved,
            config_hash=config_hash,
            history_seed=cfg.health.seed,
            role=role,
            quota=sprint15.quota,
            profile=sprint15.profile,
            protocol=sprint15.protocol,
            method=sprint15.method,
        )
    files_dir = out / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    total = len(labeled)
    expected_n = (total + shard_size - 1) // shard_size
    for shard_idx in range(expected_n):
        start = shard_idx * shard_size
        end = min(total, (shard_idx + 1) * shard_size)
        rel = f"files/shard-{shard_idx:05d}.zip"
        final_path = out / rel
        tmp_path = final_path.with_suffix(".zip.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        ids = [s.file_id for s in labeled[start:end]]
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for sample in labeled[start:end]:
                info = zipfile.ZipInfo(
                    f"{sample.file_id}.npz", date_time=_SHARD_DATE_TIME
                )
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, _sample_bytes(sample))
        os.replace(tmp_path, final_path)
        shards.append(
            {
                "path": rel,
                "start": start,
                "end": end,
                "count": len(ids),
                "file_ids": ids,
                "sha256": _hash_file(final_path),
            }
        )

    manifest: dict[str, object] = {
        "format": CHRONICLE_FORMAT,
        "generator_version": GENERATOR_VERSION,
        "protocol": protocol,
        "role": role,
        "config_hash": config_hash,
        "resolved_config": json.loads(
            json.dumps(asdict(cfg), sort_keys=True, default=str)
        ),
        "seeds": {
            "factory": cfg.factory.seed,
            "scheduler": cfg.scheduler.seed,
            "health": cfg.health.seed,
            "signal": cfg.signal.seed,
            "temporal": cfg.temporal.seed,
        },
        "counts": {
            "total": total,
            "normal": sum(
                1 for s in labeled
                if s.file_label is SampleLabel.NORMAL
            ),
            "abnormal": sum(
                1 for s in labeled
                if s.file_label is SampleLabel.ABNORMAL
            ),
            "dev_train": len(splits.dev_train),
            "dev_val": len(splits.dev_val),
            "test_static": len(splits.test_static),
            "test_temporal": len(splits.test_temporal),
            "quarantined": len(splits.quarantined),
        },
        "schedule": [_operation_to_dict(e) for e in schedule.events],
        "episodes": [_episode_to_dict(e) for e in health.episodes],
        "failure_events": [_failure_to_dict(r) for r in health.failure_events],
        "maintenance_windows": maint_windows,
        "calendar": {
            "span_s": schedule.span_s,
            "cutoff_time": splits.cutoff_time,
            "quarantine_s": splits.quarantine_s,
        },
        "splits": {
            "dev_train": list(splits.dev_train),
            "dev_val": list(splits.dev_val),
            "test_static": list(splits.test_static),
            "test_temporal": list(splits.test_temporal),
            "quarantined": list(splits.quarantined),
            "failed_episode_ids": list(splits.failed_episode_ids),
        },
        "files": _manifest_file_rows(labeled, maint_windows, patch_counts),
        "shards": shards,
    }
    if sprint15_block is not None:
        manifest["sprint15"] = sprint15_block
    _atomic_json(manifest_path, manifest)

    reloaded, _ = load_chronological(out)
    if [s.file_id for s in reloaded] != [s.file_id for s in labeled]:
        raise RuntimeError(f"reload order mismatch at {out}")
    return manifest


def load_chronological(root: str | Path):
    """Reload one materialized chronological dataset deterministically."""
    out = _require_root(root, must_exist=True)
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"chronological manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CHRONICLE_FORMAT:
        raise ValueError(
            f"unsupported chronicle format {manifest.get('format')!r}"
        )
    samples = []
    for shard in manifest["shards"]:
        shard_path = out / str(shard["path"])
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"chronological shard not found: {shard_path}"
            )
        digest = _hash_file(shard_path)
        if digest != shard["sha256"]:
            raise ValueError(
                f"checksum mismatch for {shard['path']}: "
                f"manifest {shard['sha256']} != file {digest}"
            )
        with zipfile.ZipFile(shard_path) as archive:
            for file_id in shard["file_ids"]:
                samples.append(
                    load_sample_bytes(archive.read(f"{file_id}.npz"))
                )
    for sample in samples:
        sample.validate()
    expected_ids = [row["file_id"] for row in manifest["files"]]
    if [s.file_id for s in samples] != expected_ids:
        raise ValueError("shard file order does not match the manifest")
    return samples, manifest
