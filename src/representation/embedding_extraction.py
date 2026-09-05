"""Bounded-memory Stage 1 latent embedding extraction.

This module is intentionally independent from the later geometry analysis.  It
loads a validated V1 checkpoint once, consumes materialized files in bounded
batches, and publishes only numeric embeddings plus auditable tabular metadata.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from representation.checkpoint import load_checkpoint
from representation.config import V1Config
from representation.geometry_contracts import (
    ArtifactInfo,
    DiagnosticConfig,
    GeometryManifest,
    GeometryOutputPaths,
    GeometryRecord,
    RECORD_FIELDS,
)
from representation.inference import NormalReferenceBank, RepresentationInference
from representation.model import V1RepresentationModel
from representation.data import collate_variable_files
from synth.dataset import iter_materialized
from synth.patchify import Patchifier
from synth.schema import FileSample


class GeometryCompatibilityError(ValueError):
    """Raised when checkpoint, reference bank, or dataset features disagree."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_info(path: Path, root: Path) -> ArtifactInfo:
    """Describe one published artifact relative to the cache root."""
    return ArtifactInfo(
        path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_csv(path: Path, rows: Sequence[GeometryRecord]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(RECORD_FIELDS), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        values = row.model_dump(mode="python")
        for key, value in values.items():
            values[key] = "" if value is None else value
        writer.writerow(values)
    _atomic_bytes(path, output.getvalue().encode("utf-8"))


def _npy_payload(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a reproducible NPZ (NumPy's default ZIP timestamps are not stable)."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, _npy_payload(np.asarray(arrays[name])))
    _atomic_bytes(path, output.getvalue())


def _manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise GeometryCompatibilityError(f"dataset manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometryCompatibilityError(f"dataset manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise GeometryCompatibilityError(f"dataset manifest must be a mapping: {path}")
    if value.get("format") != 2:
        raise GeometryCompatibilityError(
            f"dataset manifest has unsupported format {value.get('format')!r}: {path}"
        )
    splits = value.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise GeometryCompatibilityError(f"dataset manifest splits must be a non-empty mapping: {path}")
    return value


def validate_dataset_compatibility(dataset_manifest: Mapping[str, object], model_config: V1Config) -> str:
    """Reject a checkpoint whose feature width differs from materialized data.

    Returns a short fleet precheck status for the extraction manifest. Manifests
    that predate fleet metadata skip the static cardinality comparison: the sample
    loader defaults absent indices to bucket 0 (always in range for positive
    cardinalities) while ``ConditionalBatchNorm._bucket_ids`` still rejects any
    present out-of-range index per batch at runtime.
    """
    resolved = dataset_manifest.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise GeometryCompatibilityError("dataset manifest is missing resolved_config metadata")
    raw_channels = resolved.get("n_channels")
    if raw_channels is None:
        raise GeometryCompatibilityError("dataset manifest is missing n_channels metadata")
    try:
        channels = int(raw_channels)
    except (TypeError, ValueError) as exc:
        raise GeometryCompatibilityError("dataset manifest n_channels is invalid") from exc
    if channels != model_config.n_channels:
        raise GeometryCompatibilityError(
            f"dataset n_channels={channels} is incompatible with checkpoint n_channels={model_config.n_channels}"
        )
    if not model_config.use_conditional_norm:
        return "not-required: conditional normalization disabled"
    fleet = resolved.get("fleet")
    if not isinstance(fleet, Mapping):
        return "skipped: dataset manifest predates fleet metadata"
    for field, model_limit in (("n_robots", model_config.n_robots), ("n_programs", model_config.n_programs)):
        raw_value = fleet.get(field)
        if raw_value is None:
            raise GeometryCompatibilityError(f"dataset manifest is missing fleet.{field}")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise GeometryCompatibilityError(f"dataset fleet.{field} is invalid") from exc
        if value > model_limit:
            raise GeometryCompatibilityError(
                f"dataset fleet.{field}={value} exceeds checkpoint limit {model_limit}"
            )
    return "manifest"


def _checkpoint_model(path: Path) -> tuple[V1RepresentationModel, NormalReferenceBank, dict[str, object]]:
    if not path.is_file():
        raise GeometryCompatibilityError(f"checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GeometryCompatibilityError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise GeometryCompatibilityError("checkpoint payload must be a mapping")
    raw_config = payload.get("config")
    try:
        config = V1Config(**raw_config) if isinstance(raw_config, Mapping) else None
    except Exception as exc:
        raise GeometryCompatibilityError("checkpoint contains an invalid V1 config") from exc
    if config is None:
        raise GeometryCompatibilityError("checkpoint is missing V1 config")
    model = V1RepresentationModel(config)
    bank = NormalReferenceBank()
    try:
        info = load_checkpoint(path, model, expected_config=config, reference_bank=bank)
    except Exception as exc:
        raise GeometryCompatibilityError(f"checkpoint failed coherent V1 loading: {exc}") from exc
    if not info.get("has_reference_bank") or bank.embeddings is None:
        raise GeometryCompatibilityError("checkpoint must contain a fitted normal reference bank")
    if bank.k <= 0:
        raise GeometryCompatibilityError("checkpoint reference bank k must be positive")
    if bank.embeddings.ndim != 2 or bank.embeddings.shape[1] != config.d_model:
        raise GeometryCompatibilityError(
            f"reference bank dimension does not match checkpoint d_model={config.d_model}"
        )
    if bank.embeddings.shape[0] == 0 or not torch.isfinite(bank.embeddings).all():
        raise GeometryCompatibilityError("checkpoint reference bank must be finite and non-empty")
    return model, bank, info


def _reservoir_indices(source: Iterable[FileSample], *, limit: int, seed: int) -> list[int]:
    """Select row indices without retaining any raw ``FileSample`` values."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[int] = []
    rng = random.Random(seed)
    for index, _sample in enumerate(source):
        if index < limit:
            selected.append(index)
            continue
        slot = rng.randint(0, index)
        if slot < limit:
            selected[slot] = index
    return sorted(selected)


def _sample_stream(
    source: Iterable[FileSample], *, limit: int, strategy: str, seed: int
) -> Iterator[FileSample]:
    """Yield head samples; reservoir selection is performed by index replay."""
    if strategy != "head":
        raise ValueError("reservoir sampling requires a replayable source")
    for index, sample in enumerate(source):
        if index >= limit:
            break
        yield sample


def _regime_summary(sample: FileSample) -> str:
    return "|".join(dict.fromkeys(meta.regime.value for meta in sample.regime_sequence))


def _record(row_index: int, split: str, sample: FileSample, pred: float | None, pop: float | None) -> GeometryRecord:
    anomaly = sample.anomaly_meta
    return GeometryRecord(
        row_index=row_index,
        file_id=sample.file_id,
        split=split,
        label=sample.file_label.value,
        anomaly_family="" if anomaly is None else anomaly.family.value,
        fleet=sample.robot_code or (f"robot_{sample.robot_idx}" if sample.robot_idx is not None else ""),
        regime_summary=_regime_summary(sample),
        severity=None if anomaly is None else float(anomaly.severity),
        length=sample.T,
        generator_version=sample.generator_version,
        config_hash=sample.config_hash,
        sample_seed=int(sample.seed),
        robot_idx=int(sample.robot_idx),
        program_idx=int(sample.program_idx),
        S_pred=pred,
        S_pop=pop,
    )

class BoundedEmbeddingExtractor:
    def __init__(
        self,
        model: V1RepresentationModel,
        reference_bank: NormalReferenceBank,
        patchifier: Patchifier,
        config: DiagnosticConfig,
        *,
        checkpoint_info: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.reference_bank = reference_bank
        self.patchifier = patchifier
        self.config = config
        self.checkpoint_info = dict(checkpoint_info or {})
        if reference_bank.embeddings is None or reference_bank.embeddings.ndim != 2:
            raise GeometryCompatibilityError("reference bank must contain a 2D embedding matrix")
        if reference_bank.embeddings.shape[1] != model.config.d_model:
            raise GeometryCompatibilityError("reference bank dimension is incompatible with model")
        if reference_bank.embeddings.shape[0] > config.max_reference_samples:
            reference_bank.embeddings = reference_bank.embeddings[: config.max_reference_samples].clone()
        if patchifier.cfg.patch_size != model.config.patch_size or patchifier.cfg.stride != model.config.stride:
            raise GeometryCompatibilityError("patchifier geometry is incompatible with model")

    @classmethod
    def from_checkpoint(cls, config: DiagnosticConfig) -> "BoundedEmbeddingExtractor":
        model, bank, info = _checkpoint_model(config.checkpoint_path)
        return cls(model, bank, model.patchifier, config, checkpoint_info=info)

    def extract(self) -> dict[str, object]:
        dataset_root = self.config.dataset_root
        dataset_manifest = _manifest(dataset_root)
        fleet_precheck = validate_dataset_compatibility(dataset_manifest, self.model.config)
        splits = dataset_manifest["splits"]
        assert isinstance(splits, Mapping)
        if self.config.reference_split not in splits:
            raise GeometryCompatibilityError(f"reference split {self.config.reference_split!r} is absent from manifest")
        for split in self.config.splits:
            if split not in splits:
                raise GeometryCompatibilityError(f"selected split {split!r} is absent from manifest")

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise GeometryCompatibilityError("CUDA was requested but is unavailable")
        self.model.to(device)
        # Diagnostics must be deterministic: never mutate fleet statistics and
        # always run co-located batches (score_batch re-asserts both per call).
        self.model.eval()
        inference = RepresentationInference(self.model, self.reference_bank, self.patchifier)
        embeddings: list[np.ndarray] = []
        pred_scores: list[np.ndarray] = []
        pop_scores: list[np.ndarray] = []
        records: list[GeometryRecord] = []
        seen_ids: set[str] = set()
        row_index = 0
        for split_index, split in enumerate(self.config.splits):
            if self.config.sampling == "reservoir":
                indices = set(_reservoir_indices(iter_materialized(dataset_root, split), limit=self.config.max_samples, seed=self.config.seed + split_index))
                selected = (
                    sample
                    for index, sample in enumerate(iter_materialized(dataset_root, split))
                    if index in indices
                )
            else:
                selected = _sample_stream(
                    iter_materialized(dataset_root, split),
                    limit=self.config.max_samples,
                    strategy="head",
                    seed=self.config.seed + split_index,
                )
            batch: list[FileSample] = []
            for sample in selected:
                if sample.file_id in seen_ids:
                    raise GeometryCompatibilityError(f"duplicate file_id across selected splits: {sample.file_id}")
                seen_ids.add(sample.file_id)
                batch.append(sample)
                if len(batch) < self.config.batch_size:
                    continue
                row_index = self._process_batch(batch, split, row_index, inference, embeddings, pred_scores, pop_scores, records)
                batch.clear()
            if batch:
                row_index = self._process_batch(batch, split, row_index, inference, embeddings, pred_scores, pop_scores, records)
                batch.clear()

        if not records:
            raise GeometryCompatibilityError("selected splits yielded no samples")
        embedding_array = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
        pred_array = np.concatenate(pred_scores, axis=0).astype(np.float32, copy=False)
        pop_array = np.concatenate(pop_scores, axis=0).astype(np.float32, copy=False)
        if embedding_array.shape != (len(records), self.model.config.d_model):
            raise GeometryCompatibilityError("embedding and records linkage is inconsistent")
        paths = GeometryOutputPaths(root=self.config.output_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        _atomic_npz(
            paths.embeddings,
            {
                "embeddings": embedding_array,
                "row_index": np.arange(len(records), dtype=np.int64),
                "S_pred": pred_array,
                "S_pop": pop_array,
            },
        )
        _atomic_csv(paths.records, records)
        paths.figures.mkdir(parents=True, exist_ok=True)
        _atomic_json(paths.metrics, {"status": "extraction_complete", "records_count": len(records), "feature_dim": int(embedding_array.shape[1])})
        _atomic_bytes(paths.neighbors, b"query_row_index,neighbor_row_index,rank,distance\n")
        files = {
            "embeddings.npz": _artifact_info(paths.embeddings, paths.root),
            "records.csv": _artifact_info(paths.records, paths.root),
            "metrics.json": _artifact_info(paths.metrics, paths.root),
            "neighbors.csv": _artifact_info(paths.neighbors, paths.root),
        }
        counts = {split: sum(row.split == split for row in records) for split in self.config.splits}
        checkpoint_metadata = {
            "sha256": _sha256(self.config.checkpoint_path),
            "name": self.config.checkpoint_path.name,
            "step": self.checkpoint_info.get("step"),
            "config": self.model.config.to_dict(),
            "reference_count": int(self.reference_bank.embeddings.shape[0]),
            "reference_k": int(self.reference_bank.k),
        }
        manifest = GeometryManifest(
            checkpoint=checkpoint_metadata,
            dataset={"manifest": "manifest.json", "sha256": _sha256(dataset_root / "manifest.json"), "config_hash": dataset_manifest.get("config_hash", ""), "generator_version": dataset_manifest.get("generator_version", "")},
            extraction={"seed": self.config.seed, "splits": list(self.config.splits), "reference_split": self.config.reference_split, "max_samples": self.config.max_samples, "max_reference_samples": self.config.max_reference_samples, "batch_size": self.config.batch_size, "sampling": self.config.sampling, "device": device, "fleet_precheck": fleet_precheck},
            feature_dim=int(embedding_array.shape[1]), records_count=len(records), splits=counts, files=files,
        )
        _atomic_bytes(paths.manifest, manifest.to_json().encode("utf-8"))
        return {"paths": paths, "manifest": manifest, "records": records}

    @staticmethod
    def _process_batch(
        batch: Sequence[FileSample], split: str, row_index: int, inference: RepresentationInference,
        embeddings: list[np.ndarray], pred_scores: list[np.ndarray], pop_scores: list[np.ndarray], records: list[GeometryRecord],
    ) -> int:
        unmasked_batch = collate_variable_files(batch, inference.patchifier)
        unmasked_batch.pop("file_samples", None)
        # Match the batch device to the model device (CPU batches against a
        # CUDA model crash inside the first device-sensitive reduction).
        unmasked_batch = inference._move_batch(unmasked_batch)
        with torch.inference_mode():
            unmasked_output = inference.model(unmasked_batch)
        values = unmasked_output.get("file_embedding")
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise GeometryCompatibilityError("model output is missing [B, D] file_embedding")
        values = values.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise GeometryCompatibilityError("model produced non-finite file embeddings")

        masked_batch = collate_variable_files(
            batch,
            inference.patchifier,
            masking_config=inference.model.config,
            masking_seed=row_index,
        )
        # Force one valid target for very short files where ratio rounding yields zero.
        for index in range(masked_batch["mask"].shape[0]):
            if not bool((masked_batch["mask"][index] & masked_batch["patch_valid_mask"][index]).any()):
                valid = torch.flatnonzero(masked_batch["patch_valid_mask"][index])
                if len(valid):
                    masked_batch["mask"][index, valid[0]] = True
        masked_batch.pop("file_samples", None)
        result = inference.score_batch(masked_batch)
        pred = result.get("S_pred")
        if not isinstance(pred, torch.Tensor):
            raise GeometryCompatibilityError("independent prediction score is unavailable")
        pop = inference.reference_bank.score(values)
        pred = pred.detach().cpu().float()
        pop = pop.detach().cpu().float()
        if not torch.isfinite(pred).all() or not torch.isfinite(pop).all():
            raise GeometryCompatibilityError("model produced non-finite diagnostic scores")
        embeddings.append(values.numpy().copy())
        pred_scores.append(pred.numpy().copy())
        pop_scores.append(pop.numpy().copy())
        for offset, sample in enumerate(batch):
            records.append(_record(row_index + offset, split, sample, float(pred[offset]), float(pop[offset])))
        return row_index + len(batch)


def extract_embeddings(config: DiagnosticConfig) -> dict[str, object]:
    """Convenience entry point that loads a coherent checkpoint then extracts."""
    return BoundedEmbeddingExtractor.from_checkpoint(config).extract()


# Short aliases keep the API discoverable without introducing a second implementation.
EmbeddingExtractor = BoundedEmbeddingExtractor
