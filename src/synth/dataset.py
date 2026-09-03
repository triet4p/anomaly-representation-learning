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


def _verify_shard(root: Path, info: dict[str, object]) -> bool:
    path = root / str(info["path"])
    if not path.is_file():
        return False
    if _hash_file(path) != info.get("sha256"):
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
    """Stream samples from ZIP shards after hash and member integrity checks."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = [split] if split else list(manifest["splits"])
    for name in names:
        if name not in manifest["splits"]:
            raise ValueError(f"split {name!r} is absent from manifest")
        for shard in manifest["splits"][name]["shards"]:
            if not _verify_shard(root, shard):
                raise ValueError(f"manifest integrity check failed for {shard['path']}")
            with zipfile.ZipFile(root / str(shard["path"])) as archive:
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
    }
    if sample.anomaly_mask is not None:
        arrays["anomaly_mask"] = sample.anomaly_mask.astype(np.uint8)
    if sample.anomaly_meta is not None:
        arrays["anomaly_meta_json"] = np.bytes_(json.dumps(_meta_to_dict(sample.anomaly_meta), sort_keys=True))
    if sample.rejection_meta is not None:
        arrays["rejection_meta_json"] = np.bytes_(json.dumps(_meta_to_dict(sample.rejection_meta), sort_keys=True))
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
        return FileSample(
            x=d["x"].copy(), file_id=scalar(d["file_id"]),
            file_label=SampleLabel(scalar(d["file_label"])),
            seed=int(np.asarray(d["seed"]).item()),
            generator_version=scalar(d["generator_version"]),
            config_hash=scalar(d["config_hash"]),
            regime_sequence=regimes, anomaly_meta=anomaly,
            anomaly_mask=mask, rejection_meta=rejection,
        )
