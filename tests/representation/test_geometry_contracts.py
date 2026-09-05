from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from representation.config import V1Config
from representation.embedding_extraction import _atomic_npz, _reservoir_indices, _sample_stream
from representation.geometry import (
    DiagnosticConfig,
    GeometryManifest,
    GeometryOutputPaths,
    GeometryRecord,
    validate_dataset_compatibility,
)


def _config(tmp_path: Path, **overrides: object) -> DiagnosticConfig:
    values: dict[str, object] = {
        "dataset_root": tmp_path / "dataset",
        "checkpoint_path": tmp_path / "model.pt",
        "output_dir": tmp_path / "cache",
    }
    values.update(overrides)
    return DiagnosticConfig(**values)


def test_diagnostic_config_rejects_empty_duplicate_and_negative_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _config(tmp_path, splits=[])
    with pytest.raises(ValueError, match="duplicates"):
        _config(tmp_path, splits=("test", "test"))
    with pytest.raises(ValueError, match="max_samples"):
        _config(tmp_path, max_samples=0)
    with pytest.raises(ValueError, match="batch_size"):
        _config(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="schema_version"):
        _config(tmp_path, schema_version=2)


def test_fixed_portable_output_names_and_explicit_missing_metadata(tmp_path: Path) -> None:
    paths = GeometryOutputPaths(root=tmp_path / "cache")
    assert paths.manifest.name == "geometry-manifest.json"
    assert paths.embeddings.name == "embeddings.npz"
    assert paths.records.name == "records.csv"
    assert paths.metrics.name == "metrics.json"
    assert paths.neighbors.name == "neighbors.csv"
    assert paths.figures.name == "figures"

    record = GeometryRecord(row_index=0, file_id="f0", split="test", label="normal", length=4)
    dumped = record.model_dump()
    assert dumped["anomaly_family"] == ""
    assert dumped["severity"] is None
    assert dumped["S_pred"] is None


def test_manifest_rejects_split_count_and_schema_mismatch() -> None:
    kwargs = {
        "checkpoint": {},
        "dataset": {},
        "extraction": {},
        "feature_dim": 4,
        "records_count": 2,
        "splits": {"test": 1},
        "files": {
            "embeddings.npz": {"path": "embeddings.npz", "bytes": 1, "sha256": "0" * 64},
            "records.csv": {"path": "records.csv", "bytes": 1, "sha256": "1" * 64},
            "metrics.json": {"path": "metrics.json", "bytes": 1, "sha256": "2" * 64},
            "neighbors.csv": {"path": "neighbors.csv", "bytes": 1, "sha256": "3" * 64},
        },
    }
    with pytest.raises(ValueError, match="sum"):
        GeometryManifest(**kwargs)
    with pytest.raises(ValueError, match="schema_version"):
        GeometryManifest(schema_version=2, **{**kwargs, "records_count": 1, "splits": {"test": 1}})


def test_dataset_feature_compatibility_is_explicit() -> None:
    model_config = V1Config(n_channels=3, d_model=8, attention_heads=2, sequence_layers=1)
    validate_dataset_compatibility({"resolved_config": {"n_channels": 3, "fleet": {"n_robots": 5, "n_programs": 8}}}, model_config)
    with pytest.raises(ValueError, match="n_channels"):
        validate_dataset_compatibility({"resolved_config": {"n_channels": 6}}, model_config)
    with pytest.raises(ValueError, match="missing n_channels"):
        validate_dataset_compatibility({"resolved_config": {}}, model_config)


def test_fleet_precheck_reports_legacy_disabled_and_manifest_status() -> None:
    """Fleet validation must report its basis without weakening channel checks."""
    norm_config = V1Config(n_channels=3, d_model=8, attention_heads=2, sequence_layers=1)
    assert norm_config.use_conditional_norm is True
    assert validate_dataset_compatibility(
        {"resolved_config": {"n_channels": 3, "fleet": {"n_robots": 5, "n_programs": 8}}}, norm_config
    ) == "manifest"
    assert validate_dataset_compatibility(
        {"resolved_config": {"n_channels": 3}}, norm_config
    ) == "skipped: dataset manifest predates fleet metadata"
    plain_config = V1Config(
        n_channels=3, d_model=8, attention_heads=2, sequence_layers=1, use_conditional_norm=False
    )
    assert validate_dataset_compatibility(
        {"resolved_config": {"n_channels": 3}}, plain_config
    ) == "not-required: conditional normalization disabled"
    with pytest.raises(ValueError, match="exceeds checkpoint limit"):
        validate_dataset_compatibility(
            {"resolved_config": {"n_channels": 3, "fleet": {"n_robots": 9, "n_programs": 8}}}, norm_config
        )


def test_reservoir_sampling_is_seeded_and_bounded() -> None:
    source = list(range(100))
    first = list(_reservoir_indices(source, limit=7, seed=42))
    second = list(_reservoir_indices(source, limit=7, seed=42))
    changed = list(_reservoir_indices(source, limit=7, seed=43))
    assert first == second
    assert first != changed
    assert len(first) == 7


def test_numeric_npz_writer_is_byte_deterministic(tmp_path: Path) -> None:
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    arrays = {
        "embeddings": np.arange(6, dtype="float32").reshape(2, 3),
        "row_index": np.arange(2, dtype="int64"),
    }
    _atomic_npz(path_a, arrays)
    _atomic_npz(path_b, arrays)
    assert path_a.read_bytes() == path_b.read_bytes()
