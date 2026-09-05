from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from representation.embedding_extraction import _atomic_bytes, _atomic_csv, _atomic_npz
from representation.geometry_analysis import (
    analyze_geometry_cache,
    compute_health_metrics,
    compute_neighbors,
    compute_projections,
    compute_separation_metrics,
    neighbor_group_rates,
    write_projection_figures,
)
from representation.geometry_contracts import ArtifactInfo, GeometryManifest, GeometryRecord, NeighborConfig, ProjectionConfig


def _records(count: int) -> list[dict[str, object]]:
    return [
        {"row_index": str(i), "file_id": f"f{i}", "label": "normal" if i % 2 == 0 else "abnormal", "fleet": "R01" if i < count // 2 else "R02", "anomaly_family": "", "regime_summary": "active"}
        for i in range(count)
    ]


def test_health_metrics_identify_constant_collapsed_embeddings() -> None:
    result = compute_health_metrics(np.ones((8, 4)))
    assert result["sample_count"] == 8
    assert result["collapse"]["is_collapsed"] is True
    assert result["rank"] == 0
    assert result["effective_rank"] == 0.0


def test_health_metrics_distinguish_low_rank_anisotropic_and_healthy() -> None:
    low_rank = np.column_stack([np.arange(20), np.arange(20) * 2, np.zeros(20), np.zeros(20)]).astype(float)
    low = compute_health_metrics(low_rank)
    assert low["rank"] == 1
    assert low["effective_rank"] == pytest.approx(1.0)
    assert low["anisotropy"] > 1.0

    rng = np.random.default_rng(5)
    healthy = rng.normal(size=(64, 4))
    normal = compute_health_metrics(healthy)
    assert normal["rank"] == 4
    assert normal["effective_rank"] > 2.0
    assert normal["collapse"]["is_collapsed"] is False


def test_neighbors_are_bounded_deterministic_and_post_hoc() -> None:
    embeddings = np.arange(60, dtype=float).reshape(20, 3)
    records = _records(len(embeddings))
    config = NeighborConfig(k=3, max_queries=5, seed=11)
    first = compute_neighbors(embeddings, records, config)
    second = compute_neighbors(embeddings, records, config)
    assert first == second
    assert len(first) == 15
    assert all(row["query_row_index"] != row["neighbor_row_index"] for row in first)
    summary = neighbor_group_rates(first)
    assert summary["sample_count"] == 15
    assert summary["groups"]["label"]["comparisons"] == 15
    assert {"query_S_pred", "neighbor_S_pred", "query_S_pop", "neighbor_S_pop"} <= set(first[0])
    for field in ("label", "fleet", "regime_summary", "anomaly_family", "severity"):
        assert field in summary["groups"]


def test_pca_and_tsne_sampling_is_seeded_and_bounded() -> None:
    rng = np.random.default_rng(12)
    embeddings = rng.normal(size=(12, 5))
    config = ProjectionConfig(max_samples=7, seed=9, perplexity=20)
    first = compute_projections(embeddings, config)
    second = compute_projections(embeddings, config)
    assert np.array_equal(first["row_index"], second["row_index"])
    np.testing.assert_allclose(first["pca"], second["pca"])
    np.testing.assert_allclose(first["tsne"], second["tsne"])
    assert len(first["row_index"]) == 7
    assert len(first["pca_explained_variance_ratio"]) == 2


def test_separation_reports_reference_distance_and_no_false_contrastive_claim() -> None:
    embeddings = np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]], dtype=float)
    records = [
        {"file_id": "n0", "label": "normal"},
        {"file_id": "n1", "label": "normal"},
        {"file_id": "a0", "label": "abnormal"},
        {"file_id": "a1", "label": "abnormal"},
    ]
    result = compute_separation_metrics(embeddings, records)
    assert result["normal_reference_distance"]["count"] == 2
    assert result["same_class_similarity"]["count"] == 2
    assert result["different_class_similarity"]["count"] == 4
    assert result["class_separation_margin"] is not None
    assert result["true_contrastive_view_metrics"]["available"] is False


def test_singleton_reference_and_collapsed_metrics_are_json_safe() -> None:
    result = compute_separation_metrics(np.asarray([[1.0, 0.0], [2.0, 0.0]]), [{"label": "normal"}, {"label": "abnormal"}])
    assert result["normal_reference_distance"]["mean_distance"] is None
    import json
    json.dumps(compute_health_metrics(np.ones((3, 2))), allow_nan=False)


def test_projection_figures_include_continuous_severity_view(tmp_path) -> None:
    records = [
        {"label": "normal", "severity": "", "anomaly_family": "", "fleet": "R01", "regime_summary": "active", "split": "test"},
        {"label": "abnormal", "severity": "0.2", "anomaly_family": "drift", "fleet": "R01", "regime_summary": "active", "split": "test"},
        {"label": "abnormal", "severity": "0.8", "anomaly_family": "drift", "fleet": "R02", "regime_summary": "idle", "split": "test"},
    ]
    projections = {"row_index": np.arange(3), "pca": np.zeros((3, 2)), "tsne": np.zeros((3, 2)), "pca_explained_variance_ratio": np.array([0.75, 0.25])}
    paths = write_projection_figures(projections, records, tmp_path)
    assert (tmp_path / "pca_by_severity.png").is_file()
    assert (tmp_path / "tsne_by_severity.png").is_file()
    assert len(paths) == 12


def _make_tiny_cache(root: Path) -> None:
    root.mkdir()
    embeddings = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]], dtype=np.float32)
    _atomic_npz(root / "embeddings.npz", {
        "embeddings": embeddings,
        "row_index": np.arange(4, dtype=np.int64),
        "S_pred": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "S_pop": np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
    })
    records = [
        GeometryRecord(row_index=i, file_id=f"f{i}", split="test", label="normal" if i < 2 else "abnormal",
                       anomaly_family="" if i < 2 else "drift", fleet="R01", regime_summary="active",
                       length=8, S_pred=float(i) / 10, S_pop=float(4 - i) / 10)
        for i in range(4)
    ]
    _atomic_csv(root / "records.csv", records)
    _atomic_bytes(root / "metrics.json", b"{}\\n")
    _atomic_bytes(root / "neighbors.csv", b"query_row_index,neighbor_row_index,rank,distance\\n")
    files = {}
    for path in (root / "embeddings.npz", root / "records.csv", root / "metrics.json", root / "neighbors.csv"):
        files[path.name] = ArtifactInfo(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = GeometryManifest(checkpoint={}, dataset={}, extraction={}, feature_dim=2, records_count=4, splits={"test": 4}, files=files)
    _atomic_bytes(root / "geometry-manifest.json", manifest.to_json().encode())


def test_analysis_orchestration_keeps_input_cache_read_only(tmp_path: Path) -> None:
    cache, output = tmp_path / "cache", tmp_path / "output"
    _make_tiny_cache(cache)
    before = {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()}
    result = analyze_geometry_cache(cache, output_dir=output, neighbor_config=NeighborConfig(k=2, max_queries=4), projection_config=ProjectionConfig(max_samples=4, seed=3, perplexity=2))
    assert result["output_dir"] == output
    assert (output / "metrics.json").is_file()
    assert (output / "neighbors.csv").is_file()
    assert (output / "figures" / "pca_by_severity.png").is_file()
    assert {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()} == before
    metrics = json.loads((output / "metrics.json").read_text())
    assert "neighbor_group_rates" in metrics
    assert metrics["separation"]["true_contrastive_view_metrics"]["available"] is False
