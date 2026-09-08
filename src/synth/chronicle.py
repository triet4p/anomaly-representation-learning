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
from dataclasses import asdict
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

    Records the manifest digest, config hash, seeds, and role in
    ``seal.json``. Deterministic: no timestamps. Task 11 writes seals;
    Task 15 verifies them.
    """
    out = _require_root(root, must_exist=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = {
        "protocol": "sprint13-protocol-v4",
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


def materialize_chronological(
    cfg: SynthConfig,
    root: str | Path,
    *,
    shard_size: int = 64,
    overwrite: bool = False,
    role: str | None = None,
    protocol: str | None = None,
) -> dict[str, object]:
    """Persist one chronological dataset plus complete manifests.

    ``role`` names the whole-history assignment (or None when unassigned);
    ``protocol`` names the frozen benchmark version (defaults to
    ``sprint13-protocol-v4`` when a role is given — pass explicitly,
    e.g. ``sprint13-protocol-v4.1``, for amended runs). Both recorded verbatim.
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

    maint_windows = _maintenance_windows(health)
    if protocol is None and role is not None:
        protocol = "sprint13-protocol-v4"
    patchifier = Patchifier(cfg.patch)
    patch_counts = {
        sample.file_id: int(patchifier.patchify(sample).patches.shape[0])
        for sample in labeled
    }

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
        "files": [
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
        ],
        "shards": shards,
    }
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
