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
from synth.scheduled import ScheduledSignalGenerator
from synth.scheduler import FactoryScheduler
from synth.schema import SampleLabel
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
    return cfg


def _require_root(root: str | Path, *, must_exist: bool) -> Path:
    if root is None or (isinstance(root, str) and not root.strip()):
        raise ValueError("chronological root must be an explicit path")
    path = Path(root)
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"chronological root not found: {path}")
    return path


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
) -> dict[str, object]:
    """Persist one chronological dataset plus complete manifests."""
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

    manifest: dict[str, object] = {
        "format": CHRONICLE_FORMAT,
        "generator_version": GENERATOR_VERSION,
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
        "calendar": {
            "span_s": schedule.span_s,
            "cutoff_time": splits.cutoff_time,
            "quarantine_s": splits.quarantine_s,
        },
        "schedule": [_operation_to_dict(e) for e in schedule.events],
        "episodes": [_episode_to_dict(e) for e in health.episodes],
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
                "member_views": list(s.split_provenance.member_views),  # type: ignore[union-attr]
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
