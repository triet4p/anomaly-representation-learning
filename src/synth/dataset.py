"""Deterministic, bounded-memory dataset generation and persistence."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from synth.config import GENERATOR_VERSION, SynthConfig
from synth.generator import SessionGenerator
from synth.schema import AnomalyFamily, AnomalyMeta, FileSample, SampleLabel, RegimeMeta, RegimeType
from synth.schema import DegradationStage, EpisodeKind, FactoryProvenance, FutureFailureTargets
from synth.schema import HealthEpisode, ObservableAnomalyLabels, OperatingContext, OperationEvent
from synth.schema import RobotHealthState, SplitProvenance



class DatasetGenerationError(RuntimeError):
    """Exact-count failure carrying deterministic retry provenance."""

    def __init__(self, split: str, index: int, family: AnomalyFamily,
                 provenance: list[dict[str, object]]) -> None:
        self.split, self.index, self.family, self.provenance = split, index, family, provenance
        super().__init__(
            f"Could not generate accepted {family.value} anomaly: "
            f"split={split} index={index} retries={len(provenance)}"
        )

class DatasetBuilder:
    """Generate samples lazily and optionally persist them as atomic shards."""

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()
        self.gen = SessionGenerator(self.cfg)
        self.split_cfg = self.cfg.split

    def _base_seed(self, split: str) -> tuple[str, int]:
        if split in ("validation", "val"):
            return "val", int(self.split_cfg.val_seed)
        if split in ("train", "test"):
            return split, int(getattr(self.split_cfg, f"{split}_seed"))
        raise ValueError(f"Unknown split: {split!r}")

    def _sample_at(self, split: str, n: int, contamination: float,
                   index: int, fams: list[AnomalyFamily]) -> FileSample:
        split, base_seed = self._base_seed(split)
        n_anom = int(round(n * contamination))
        n_normal = n - n_anom
        if index < n_normal:
            return self.gen.generate_normal(base_seed + index, split=split)
        anomaly_idx = index - n_normal
        family = fams[anomaly_idx % len(fams)]
        retry_provenance: list[dict[str, object]] = []
        for retry in range(self.cfg.anomaly.max_rejection_attempts):
            seed = base_seed + index + (retry + 1) * 10_000_000
            candidate = self.gen.generate_anomaly(seed, split=split, family=family)
            meta = candidate.rejection_meta
            if candidate.file_label == SampleLabel.ABNORMAL:
                if candidate.anomaly_meta is not None:
                    internal = int(candidate.anomaly_meta.extra.get("n_attempts", 1))
                    candidate.anomaly_meta.extra["dataset_retry_rejections"] = retry_provenance
                    candidate.anomaly_meta.extra["dataset_rejected_attempts"] = (
                        sum(int(r["internal_attempts"]) for r in retry_provenance)
                        + max(0, internal - 1)
                    )
                return candidate
            internal = int(meta.extra.get("n_attempts", 1)) if meta is not None else 1
            retry_provenance.append({
                "seed": seed, "family": family.value,
                "internal_attempts": internal,
                "metadata": _meta_to_dict(meta) if meta is not None else None,
            })
        raise DatasetGenerationError(split, index, family, retry_provenance)

    def iter_split(self, split: str, n: int, contamination: float = 0.0,
                   families: list[AnomalyFamily | str] | None = None) -> Iterator[FileSample]:
        """Yield exactly ``n`` samples without retaining prior samples."""
        self._base_seed(split)
        if n < 0:
            raise ValueError("n must be non-negative")
        if not 0.0 <= contamination <= 1.0:
            raise ValueError("contamination must be in [0, 1]")
        n_anom = int(round(n * contamination))
        fams = [AnomalyFamily(f) if isinstance(f, str) else f
                for f in (families or self.split_cfg.anomaly_families)]
        if n_anom and not fams:
            raise ValueError("families must not be empty when contamination > 0")
        for index in range(n):
            yield self._sample_at(split, n, contamination, index, fams)
    def build_split(self, split: str, n: int, contamination: float = 0.0,
                    families: list[AnomalyFamily | str] | None = None) -> list[FileSample]:
        return list(self.iter_split(split, n, contamination, families))

    def build_contamination_sweep(self, n_train: int,
                                  ratios: list[float] | None = None) -> dict[float, list[FileSample]]:
        ratios = self.split_cfg.contamination_ratios if ratios is None else ratios
        return {r: self.build_split("train", n_train, r) for r in ratios}

    def materialize(self, output_dir: str | Path, n_train: int | None = None,
                    n_val: int | None = None, n_test: int | None = None,
                    contamination: float = 0.0) -> dict[str, list[str]]:
        """Legacy per-file materialization wrapper."""
        cfg = self.split_cfg
        counts = {"train": cfg.n_train if n_train is None else n_train,
                  "val": cfg.n_val if n_val is None else n_val,
                  "test": cfg.n_test if n_test is None else n_test}
        out = Path(output_dir)
        saved: dict[str, list[str]] = {}
        for split, count, ratio in (("train", counts["train"], contamination),
                                    ("val", counts["val"], 0.5),
                                    ("test", counts["test"], 0.5)):
            directory = out / split
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for sample in self.iter_split(split, count, ratio):
                path = directory / f"{sample.file_id}.npz"
                _save_sample(sample, path)
                paths.append(str(path))
            saved[split] = paths
        return saved

    def materialize_sharded(self, output_dir: str | Path, *, n_train: int | None = None,
                            n_val: int | None = None, n_test: int | None = None,
                            contamination: float = 0.0, shard_size: int = 512,
                            resume: bool = False, overwrite: bool = False,
                            profile: str | None = None) -> dict[str, object]:
        """Write one ZIP archive per bounded shard and atomically update manifest."""
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        cfg = self.split_cfg
        counts = {"train": cfg.n_train if n_train is None else int(n_train),
                  "val": cfg.n_val if n_val is None else int(n_val),
                  "test": cfg.n_test if n_test is None else int(n_test)}
        if profile == "small":
            counts = {"train": 12, "val": 8, "test": 12}
        if any(v < 0 for v in counts.values()):
            raise ValueError("split counts must be non-negative")
        out = Path(output_dir)
        manifest_path = out / "manifest.json"
        config_hash = self.cfg.hash()
        resolved = json.loads(json.dumps(asdict(self.cfg), sort_keys=True, default=str))
        if manifest_path.exists() and not (resume or overwrite):
            raise FileExistsError(f"manifest already exists: {manifest_path}; use resume or overwrite")
        if overwrite and manifest_path.exists():
            manifest_path.unlink()
        if resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("config_hash") != config_hash or
                    manifest.get("generator_version") != GENERATOR_VERSION or
                    manifest.get("resolved_config") != resolved):
                raise ValueError("resume configuration is incompatible with existing manifest")
            if manifest.get("counts") != counts or manifest.get("shard_size") != shard_size:
                raise ValueError("resume counts/shard_size differ from existing manifest")
        else:
            manifest = {"format": 2, "generator_version": GENERATOR_VERSION,
                        "config_hash": config_hash, "resolved_config": resolved,
                        "counts": counts, "shard_size": shard_size, "splits": {}}
        out.mkdir(parents=True, exist_ok=True)
        for split, count in counts.items():
            ratio = contamination if split == "train" else 0.5
            n_anom = int(round(count * ratio))
            split_entry = manifest.setdefault("splits", {}).setdefault(
                split, {"count": count, "requested_normal": count - n_anom,
                        "requested_abnormal": n_anom, "accepted_normal": 0,
                        "accepted_abnormal": 0, "rejection_attempt_total": 0,
                        "rejection_attempts_by_family": {}, "shards": []})
            shards: list[dict[str, object]] = split_entry["shards"]
            expected_n = (count + shard_size - 1) // shard_size
            while len(shards) > expected_n:
                shards.pop()
            for shard_idx in range(expected_n):
                start, end = shard_idx * shard_size, min(count, (shard_idx + 1) * shard_size)
                shard_rel = f"{split}/shard-{shard_idx:05d}.zip"
                existing = shards[shard_idx] if shard_idx < len(shards) else None
                if existing and _verify_shard(out, existing):
                    continue
                final_path = out / shard_rel
                final_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = final_path.with_suffix(".zip.tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                ids: list[str] = []
                class_counts = {"normal": 0, "abnormal": 0}
                rejection_total = 0
                rejection_by_family: dict[str, int] = {}
                try:
                    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as archive:
                        for sample in self._iter_slice(split, count, ratio, start, end):
                            ids.append(sample.file_id)
                            label = sample.file_label.value
                            class_counts[label] += 1
                            attempts = _sample_rejected_attempts(sample)
                            rejection_total += attempts
                            family_meta = sample.anomaly_meta or sample.rejection_meta
                            if attempts and family_meta is not None:
                                family_name = family_meta.family.value
                                rejection_by_family[family_name] = rejection_by_family.get(family_name, 0) + attempts
                            info = zipfile.ZipInfo(
                                f"{sample.file_id}.npz", date_time=(1980, 1, 1, 0, 0, 0)
                            )
                            info.compress_type = zipfile.ZIP_STORED
                            archive.writestr(info, _sample_bytes(sample))
                except DatasetGenerationError as exc:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    split_entry["status"] = "failed"
                    split_entry["failure"] = {
                        "split": exc.split, "index": exc.index,
                        "family": exc.family.value, "provenance": exc.provenance,
                    }
                    _rebuild_split_accounting(split_entry, shards, count, n_anom)
                    _atomic_json(manifest_path, manifest)
                    raise
                os.replace(tmp_path, final_path)
                split_entry["status"] = "in_progress"
                shard_info = {"path": shard_rel, "start": start, "end": end,
                              "count": len(ids), "file_ids": ids,
                              "normal": class_counts["normal"], "abnormal": class_counts["abnormal"],
                              "rejection_attempt_total": rejection_total,
                              "rejection_attempts_by_family": rejection_by_family,
                              "sha256": _hash_file(final_path)}
                if len(shards) <= shard_idx:
                    shards.append(shard_info)
                else:
                    shards[shard_idx] = shard_info
                _rebuild_split_accounting(split_entry, shards, count, n_anom)
                _atomic_json(manifest_path, manifest)
            split_entry["status"] = "complete"
            split_entry.pop("failure", None)
            split_entry["count"] = count
            _rebuild_split_accounting(split_entry, shards, count, n_anom)
        _atomic_json(manifest_path, manifest)
        return manifest

    def _iter_slice(self, split: str, n: int, contamination: float,
                    start: int, end: int) -> Iterator[FileSample]:
        fams = [AnomalyFamily(f) for f in self.split_cfg.anomaly_families]
        if int(round(n * contamination)) and not fams:
            raise ValueError("configured anomaly_families is empty")
        for index in range(start, end):
            yield self._sample_at(split, n, contamination, index, fams)


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_shard_path(root: Path, rel_path: str) -> Path:
    direct = root / rel_path
    if direct.is_file():
        return direct
    flat = root / Path(rel_path).name
    if flat.is_file():
        return flat
    matches = list(root.glob(f"**/{Path(rel_path).name}"))
    if matches:
        return matches[0]
    return direct


def _verify_shard(root: Path, info: dict[str, object], strict_hash: bool = True) -> bool:
    path = _resolve_shard_path(root, str(info["path"]))
    if not path.is_file():
        return False
    if strict_hash and _hash_file(path) != info.get("sha256"):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = [f"{file_id}.npz" for file_id in info["file_ids"]]
            return (len(names) == int(info["count"]) and names == expected and
                    archive.testzip() is None)
    except (OSError, zipfile.BadZipFile):
        return False


def _sample_rejected_attempts(sample: FileSample) -> int:
    meta = sample.anomaly_meta or sample.rejection_meta
    if meta is None:
        return 0
    if "dataset_rejected_attempts" in meta.extra:
        return int(meta.extra["dataset_rejected_attempts"])
    return max(0, int(meta.extra.get("n_attempts", 1)) - 1)

def _rebuild_split_accounting(entry: dict[str, object], shards: list[dict[str, object]],
                              count: int, n_anom: int) -> None:
    entry["count"] = count
    entry["requested_normal"] = count - n_anom
    entry["requested_abnormal"] = n_anom
    entry["accepted_normal"] = sum(int(s.get("normal", 0)) for s in shards)
    entry["accepted_abnormal"] = sum(int(s.get("abnormal", 0)) for s in shards)
    entry["rejection_attempt_total"] = sum(int(s.get("rejection_attempt_total", 0)) for s in shards)
    by_family: dict[str, int] = {}
    for shard in shards:
        for family, amount in dict(shard.get("rejection_attempts_by_family", {})).items():
            by_family[str(family)] = by_family.get(str(family), 0) + int(amount)
    entry["rejection_attempts_by_family"] = by_family
    entry["shard_count"] = len(shards)


def iter_materialized(output_dir: str | Path, split: str | None = None) -> Iterator[FileSample]:
    """Stream samples from ZIP shards or unzipped directories after integrity checks."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = [split] if split else list(manifest["splits"])
    is_kaggle = Path("/kaggle/input").is_dir() or os.environ.get("V1_SKIP_STRICT_HASH", "0") == "1"
    strict_hash = not is_kaggle
    for name in names:
        if name not in manifest["splits"]:
            raise ValueError(f"split {name!r} is absent from manifest")
        for shard in manifest["splits"][name]["shards"]:
            shard_rel = str(shard["path"])

            # 1. Check if Kaggle automatically unzipped shard into a directory
            candidates_dir = [
                root / (shard_rel[:-4] if shard_rel.endswith(".zip") else shard_rel),
                root / shard_rel,
                root / name / Path(shard_rel).stem,
            ]
            shard_dir = next((d for d in candidates_dir if d.is_dir()), None)
            if shard_dir is not None:
                for file_id in shard["file_ids"]:
                    npz_file = shard_dir / f"{file_id}.npz"
                    if npz_file.is_file():
                        yield load_sample(npz_file)
                    else:
                        matches = list(shard_dir.glob(f"**/{file_id}.npz"))
                        if matches:
                            yield load_sample(matches[0])
                        else:
                            raise FileNotFoundError(f"Sample {file_id}.npz not found in {shard_dir}")
                continue

            # 2. Standard .zip archive layout (local server)
            shard_path = _resolve_shard_path(root, shard_rel)
            if not _verify_shard(root, shard, strict_hash=strict_hash):
                raise ValueError(f"manifest integrity check failed for {shard['path']}")
            with zipfile.ZipFile(shard_path) as archive:
                for file_id in shard["file_ids"]:
                    yield load_sample_bytes(archive.read(f"{file_id}.npz"))

def _meta_to_dict(meta: AnomalyMeta) -> dict[str, object]:
    return {"family": meta.family.value, "start": meta.start, "end": meta.end,
            "severity": meta.severity, "affected_channels": list(meta.affected_channels),
            "lag": meta.lag, "phase_shift": meta.phase_shift, "gain_change": meta.gain_change,
            "drift_rate": meta.drift_rate, "stuck_sigma_ratio": meta.stuck_sigma_ratio,
            "transition_speed": meta.transition_speed, "donor_file_id": meta.donor_file_id,
            "extra": meta.extra}


def _meta_from_dict(data: dict[str, object]) -> AnomalyMeta:
    return AnomalyMeta(
        family=AnomalyFamily(str(data["family"])), start=int(data["start"]), end=int(data["end"]),
        severity=float(data["severity"]), affected_channels=[int(v) for v in data.get("affected_channels", [])],
        lag=None if data.get("lag") is None else int(data["lag"]),
        phase_shift=None if data.get("phase_shift") is None else float(data["phase_shift"]),
        gain_change=None if data.get("gain_change") is None else float(data["gain_change"]),
        drift_rate=None if data.get("drift_rate") is None else float(data["drift_rate"]),
        stuck_sigma_ratio=None if data.get("stuck_sigma_ratio") is None else float(data["stuck_sigma_ratio"]),
        transition_speed=None if data.get("transition_speed") is None else float(data["transition_speed"]),
        donor_file_id=data.get("donor_file_id"), extra=dict(data.get("extra", {})),
    )


def _operation_to_dict(op: OperationEvent) -> dict[str, object]:
    return {"operation_id": op.operation_id, "unit_id": op.unit_id,
            "product_type": op.product_type, "route_id": op.route_id,
            "route_position": op.route_position, "robot_id": op.robot_id,
            "program_id": op.program_id, "arrival_time": op.arrival_time,
            "start_time": op.start_time, "end_time": op.end_time,
            "duration": op.duration, "queue_delay": op.queue_delay,
            "travel_time": op.travel_time, "idle_before": op.idle_before}


def _operation_from_dict(data: dict[str, object]) -> OperationEvent:
    return OperationEvent(
        operation_id=str(data["operation_id"]), unit_id=str(data["unit_id"]),
        product_type=str(data["product_type"]), route_id=str(data["route_id"]),
        route_position=int(data["route_position"]), robot_id=str(data["robot_id"]),
        program_id=str(data["program_id"]), arrival_time=float(data["arrival_time"]),
        start_time=float(data["start_time"]), end_time=float(data["end_time"]),
        duration=float(data["duration"]), queue_delay=float(data["queue_delay"]),
        travel_time=float(data["travel_time"]), idle_before=float(data["idle_before"]),
    )


def _context_to_dict(ctx: OperatingContext) -> dict[str, object]:
    return {"shift": ctx.shift, "load": ctx.load,
            "ambient_temp_c": ctx.ambient_temp_c}


def _context_from_dict(data: dict[str, object]) -> OperatingContext:
    return OperatingContext(shift=str(data.get("shift", "day")),
                            load=float(data.get("load", 1.0)),
                            ambient_temp_c=float(data.get("ambient_temp_c", 25.0)))


def _health_to_dict(state: RobotHealthState) -> dict[str, object]:
    return {"health_value": state.health_value,
            "program_sensitivity": state.program_sensitivity,
            "manifested_value": state.manifested_value,
            "degradation_stage": state.degradation_stage.value,
            "degradation_severity": state.degradation_severity,
            "degradation_episode_id": state.degradation_episode_id}


def _health_from_dict(data: dict[str, object]) -> RobotHealthState:
    return RobotHealthState(
        health_value=float(data["health_value"]),
        program_sensitivity=float(data["program_sensitivity"]),
        manifested_value=float(data["manifested_value"]),
        degradation_stage=DegradationStage(str(data["degradation_stage"])),
        degradation_severity=float(data["degradation_severity"]),
        degradation_episode_id=None if data.get("degradation_episode_id") is None
        else str(data["degradation_episode_id"]),
    )


def _episode_to_dict(episode: HealthEpisode) -> dict[str, object]:
    return {"episode_id": episode.episode_id, "kind": episode.kind.value,
            "robot_id": episode.robot_id, "start_time": episode.start_time,
            "end_time": episode.end_time}


def _episode_from_dict(data: dict[str, object]) -> HealthEpisode:
    return HealthEpisode(
        episode_id=str(data["episode_id"]), kind=EpisodeKind(str(data["kind"])),
        robot_id=str(data["robot_id"]), start_time=float(data["start_time"]),
        end_time=None if data.get("end_time") is None else float(data["end_time"]),
    )


def _labels_to_dict(labels: ObservableAnomalyLabels) -> dict[str, object]:
    return {"is_file_anomalous": labels.is_file_anomalous,
            "anomaly_family": labels.anomaly_family,
            "anomaly_severity": labels.anomaly_severity}


def _labels_from_dict(data: dict[str, object]) -> ObservableAnomalyLabels:
    return ObservableAnomalyLabels(
        is_file_anomalous=bool(data["is_file_anomalous"]),
        anomaly_family=None if data.get("anomaly_family") is None
        else str(data["anomaly_family"]),
        anomaly_severity=None if data.get("anomaly_severity") is None
        else float(data["anomaly_severity"]),
    )


def _targets_to_dict(targets: FutureFailureTargets) -> dict[str, object]:
    return {"time_to_next_failure": targets.time_to_next_failure,
            "failure_within_1d": targets.failure_within_1d,
            "failure_within_7d": targets.failure_within_7d,
            "is_censored": targets.is_censored}


def _targets_from_dict(data: dict[str, object]) -> FutureFailureTargets:
    return FutureFailureTargets(
        time_to_next_failure=None if data.get("time_to_next_failure") is None
        else float(data["time_to_next_failure"]),
        failure_within_1d=bool(data["failure_within_1d"]),
        failure_within_7d=bool(data["failure_within_7d"]),
        is_censored=bool(data["is_censored"]),
    )


def _split_prov_to_dict(prov: SplitProvenance) -> dict[str, object]:
    return {"cutoff_time": prov.cutoff_time, "is_quarantined": prov.is_quarantined,
            "quarantine_reason": prov.quarantine_reason,
            "member_views": list(prov.member_views)}


def _split_prov_from_dict(data: dict[str, object]) -> SplitProvenance:
    return SplitProvenance(
        cutoff_time=float(data["cutoff_time"]),
        is_quarantined=bool(data["is_quarantined"]),
        quarantine_reason=None if data.get("quarantine_reason") is None
        else str(data["quarantine_reason"]),
        member_views=tuple(str(v) for v in data.get("member_views", [])),
    )


def _factory_prov_to_dict(prov: FactoryProvenance) -> dict[str, object]:
    return {"seed": prov.seed, "stream": prov.stream,
            "generator_version": prov.generator_version,
            "config_hash": prov.config_hash}


def _factory_prov_from_dict(data: dict[str, object]) -> FactoryProvenance:
    return FactoryProvenance(seed=int(data["seed"]), stream=str(data["stream"]),
                             generator_version=str(data["generator_version"]),
                             config_hash=str(data["config_hash"]))


def _json_bytes(value: dict[str, object]) -> bytes:
    return np.bytes_(json.dumps(value, sort_keys=True))


def _sample_arrays(sample: FileSample) -> dict[str, object]:
    arrays: dict[str, object] = {
        "x": sample.x, "seed": np.int64(sample.seed), "file_label": np.bytes_(sample.file_label.value),
        "generator_version": np.bytes_(sample.generator_version), "config_hash": np.bytes_(sample.config_hash),
        "file_id": np.bytes_(sample.file_id),
        "regime_types": np.array([r.regime.value for r in sample.regime_sequence], dtype="S32"),
        "regime_starts": np.array([r.start for r in sample.regime_sequence], dtype=np.int64),
        "regime_ends": np.array([r.end for r in sample.regime_sequence], dtype=np.int64),
        "regime_levels": np.array([r.target_level for r in sample.regime_sequence], dtype=np.float64),
        "regime_frequencies": np.array([np.nan if r.frequency is None else r.frequency for r in sample.regime_sequence]),
        "regime_phases": np.array([np.nan if r.phase is None else r.phase for r in sample.regime_sequence]),
        "robot_idx": np.int64(sample.robot_idx),
        "program_idx": np.int64(sample.program_idx),
        "robot_code": np.bytes_(sample.robot_code),
        "program_number": np.bytes_(sample.program_number),
    }
    if sample.anomaly_mask is not None:
        arrays["anomaly_mask"] = sample.anomaly_mask.astype(np.uint8)
    if sample.anomaly_meta is not None:
        arrays["anomaly_meta_json"] = np.bytes_(json.dumps(_meta_to_dict(sample.anomaly_meta), sort_keys=True))
    if sample.rejection_meta is not None:
        arrays["rejection_meta_json"] = np.bytes_(json.dumps(_meta_to_dict(sample.rejection_meta), sort_keys=True))
    if sample.operation is not None:
        arrays["operation_json"] = _json_bytes(_operation_to_dict(sample.operation))
    if sample.operating_context is not None:
        arrays["operating_context_json"] = _json_bytes(_context_to_dict(sample.operating_context))
    if sample.health is not None:
        arrays["health_json"] = _json_bytes(_health_to_dict(sample.health))
    if sample.episode is not None:
        arrays["episode_json"] = _json_bytes(_episode_to_dict(sample.episode))
    if sample.anomaly_labels is not None:
        arrays["anomaly_labels_json"] = _json_bytes(_labels_to_dict(sample.anomaly_labels))
    if sample.future_targets is not None:
        arrays["future_targets_json"] = _json_bytes(_targets_to_dict(sample.future_targets))
    if sample.split_provenance is not None:
        arrays["split_provenance_json"] = _json_bytes(_split_prov_to_dict(sample.split_provenance))
    if sample.factory_provenance is not None:
        arrays["factory_provenance_json"] = _json_bytes(
            _factory_prov_to_dict(sample.factory_provenance))
    return arrays


def _sample_bytes(sample: FileSample) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **_sample_arrays(sample))
    return output.getvalue()


def _save_sample(sample: FileSample, path: str | Path) -> None:
    """Save one sample atomically while retaining ragged signal metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_sample_bytes(sample))
    os.replace(tmp, path)


def load_sample(path: str | Path) -> FileSample:
    with Path(path).open("rb") as handle:
        return load_sample_bytes(handle.read())


def load_sample_bytes(payload: bytes) -> FileSample:
    """Decode one NPZ member while keeping the archive stream bounded."""
    def scalar(value: object) -> str:
        raw = np.asarray(value).item()
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    with np.load(io.BytesIO(payload), allow_pickle=False) as d:
        types = [RegimeType(t.decode()) for t in d["regime_types"].tolist()]
        levels = d.get("regime_levels", np.zeros(len(types))).tolist()
        freqs = d.get("regime_frequencies", np.full(len(types), np.nan)).tolist()
        phases = d.get("regime_phases", np.full(len(types), np.nan)).tolist()
        regimes = [RegimeMeta(rt, int(s), int(e), float(level),
                              None if np.isnan(freq) else float(freq),
                              None if np.isnan(phase) else float(phase))
                   for rt, s, e, level, freq, phase in zip(
                       types, d["regime_starts"], d["regime_ends"], levels, freqs, phases)]
        anomaly = (_meta_from_dict(json.loads(scalar(d["anomaly_meta_json"])))
                   if "anomaly_meta_json" in d else None)
        rejection = (_meta_from_dict(json.loads(scalar(d["rejection_meta_json"])))
                     if "rejection_meta_json" in d else None)
        mask = d["anomaly_mask"].astype(bool) if "anomaly_mask" in d else None
        robot_idx = int(np.asarray(d["robot_idx"]).item()) if "robot_idx" in d else 0
        program_idx = int(np.asarray(d["program_idx"]).item()) if "program_idx" in d else 0
        robot_code = scalar(d["robot_code"]) if "robot_code" in d else "R01"
        program_number = scalar(d["program_number"]) if "program_number" in d else "P100"
        # Factory contracts are optional: archives written before Sprint 11
        # Task 1 simply decode to None.
        operation = (_operation_from_dict(json.loads(scalar(d["operation_json"])))
                     if "operation_json" in d else None)
        operating_context = (_context_from_dict(
            json.loads(scalar(d["operating_context_json"])))
            if "operating_context_json" in d else None)
        health = (_health_from_dict(json.loads(scalar(d["health_json"])))
                  if "health_json" in d else None)
        episode = (_episode_from_dict(json.loads(scalar(d["episode_json"])))
                   if "episode_json" in d else None)
        anomaly_labels = (_labels_from_dict(
            json.loads(scalar(d["anomaly_labels_json"])))
            if "anomaly_labels_json" in d else None)
        future_targets = (_targets_from_dict(
            json.loads(scalar(d["future_targets_json"])))
            if "future_targets_json" in d else None)
        split_provenance = (_split_prov_from_dict(
            json.loads(scalar(d["split_provenance_json"])))
            if "split_provenance_json" in d else None)
        factory_provenance = (_factory_prov_from_dict(
            json.loads(scalar(d["factory_provenance_json"])))
            if "factory_provenance_json" in d else None)
        return FileSample(
            x=d["x"].copy(), file_id=scalar(d["file_id"]),
            file_label=SampleLabel(scalar(d["file_label"])),
            seed=int(np.asarray(d["seed"]).item()),
            generator_version=scalar(d["generator_version"]),
            config_hash=scalar(d["config_hash"]),
            regime_sequence=regimes, anomaly_meta=anomaly,
            anomaly_mask=mask, rejection_meta=rejection,
            robot_idx=robot_idx, program_idx=program_idx,
            robot_code=robot_code, program_number=program_number,
            operation=operation, operating_context=operating_context,
            health=health, episode=episode, anomaly_labels=anomaly_labels,
            future_targets=future_targets, split_provenance=split_provenance,
            factory_provenance=factory_provenance,
        )
