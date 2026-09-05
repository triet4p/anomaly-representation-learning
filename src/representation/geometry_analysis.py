"""CPU-only analysis of the portable latent-geometry cache.

No function here imports a model or dataset loader.  Labels and provenance are
joined only after embeddings are loaded and are never used to fit projections.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from representation.geometry_contracts import (
    GEOMETRY_SCHEMA_VERSION,
    GeometryManifest,
    NeighborConfig,
    ProjectionConfig,
)

MISSING = "<missing>"
_GROUP_FIELDS = ("label", "anomaly_family", "fleet", "regime_summary", "severity", "split")

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
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
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _value(row: Mapping[str, object], name: str) -> object:
    value = row.get(name)
    return MISSING if value is None or str(value).strip() == "" else value


def _records(root: Path) -> list[dict[str, object]]:
    with (root / "records.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for expected, row in enumerate(rows):
        try:
            row_index = int(row.get("row_index", "-1"))
        except ValueError as exc:
            raise ValueError("records.csv has an invalid row_index") from exc
        if row_index != expected:
            raise ValueError("records.csv row_index is not contiguous")
    return [{str(k): v for k, v in row.items()} for row in rows]


def load_geometry_cache(root: str | Path) -> tuple[np.ndarray, list[dict[str, object]], GeometryManifest]:
    """Load and verify a cache without importing model/checkpoint code."""
    base = Path(root)
    manifest_path = base / "geometry-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"geometry manifest is missing: {manifest_path}")
    try:
        manifest = GeometryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("geometry manifest is invalid or schema-incompatible") from exc
    for info in manifest.files.values():
        path = base / info.path
        if not path.is_file() or _sha256(path) != info.sha256:
            raise ValueError(f"geometry artifact checksum failed: {info.path}")
    try:
        with np.load(base / "embeddings.npz", allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"embeddings", "row_index", "S_pred", "S_pop"}
            if keys != required:
                raise ValueError("embeddings.npz has an incompatible numeric schema")
            embeddings = np.asarray(archive["embeddings"])
            row_index = np.asarray(archive["row_index"])
            pred = np.asarray(archive["S_pred"])
            pop = np.asarray(archive["S_pop"])
    except (OSError, ValueError) as exc:
        raise ValueError("embeddings.npz is unreadable") from exc
    if embeddings.ndim != 2 or embeddings.shape[1] != manifest.feature_dim or embeddings.shape[0] != manifest.records_count:
        raise ValueError("embedding shape does not match geometry manifest")
    if row_index.dtype.kind not in "iu" or not np.array_equal(row_index, np.arange(len(row_index), dtype=row_index.dtype)):
        raise ValueError("embeddings row_index is not contiguous")
    if pred.shape != (len(row_index),) or pop.shape != (len(row_index),):
        raise ValueError("score arrays do not match embedding rows")
    if not np.isfinite(embeddings).all() or not np.isfinite(pred).all() or not np.isfinite(pop).all():
        raise ValueError("cache arrays must be finite")
    records = _records(base)
    if len(records) != len(embeddings):
        raise ValueError("records.csv and embeddings.npz row counts differ")
    return embeddings.astype(np.float64, copy=False), records, manifest


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty [N, D] matrix")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    return values


def _group_summary(values: np.ndarray, records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, dict[str, float | int]]]:
    norms = np.linalg.norm(values, axis=1)
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in _GROUP_FIELDS:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[str(_value(record, field))].append(index)
        output[field] = {
            group: {"count": len(indices), "mean_norm": float(norms[indices].mean())}
            for group, indices in sorted(groups.items())
        }
    return output


def compute_health_metrics(embeddings: np.ndarray, records: Sequence[Mapping[str, object]] | None = None) -> dict[str, object]:
    """Compute stable health/spectrum/collapse metrics with explicit counts."""
    values = _validate_embeddings(embeddings)
    n_samples, dimension = values.shape
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    covariance = (centered.T @ centered) / float(max(n_samples - 1, 1))
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(eigenvalues.sum())
    tolerance = max(total, 1.0) * np.finfo(np.float64).eps * max(n_samples, dimension) * 16.0
    positive = eigenvalues[eigenvalues > tolerance]
    if total > tolerance and positive.size:
        probabilities = positive / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        anisotropy = float(eigenvalues[0] / max(total / dimension, tolerance))
    else:
        effective_rank = 0.0
        anisotropy = 0.0
    variance = np.diag(covariance)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "sample_count": n_samples,
        "feature_dim": dimension,
        "embedding_norm": {"mean": float(np.linalg.norm(values, axis=1).mean()), "std": float(np.linalg.norm(values, axis=1).std())},
        "per_dimension_variance": variance.tolist(),
        "covariance_spectrum": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "anisotropy": anisotropy,
        "rank": rank,
        "collapse": {"is_collapsed": bool(total <= tolerance), "zero_variance_dimensions": int(np.count_nonzero(variance <= tolerance)), "variance_tolerance": tolerance},
        "groups": _group_summary(values, records) if records is not None else {},
        "definitions": {"effective_rank": "exp of Shannon entropy of positive covariance eigenvalue proportions", "anisotropy": "largest covariance eigenvalue divided by mean per-dimension variance", "missing_group": MISSING},
    }


def _sample_indices(count: int, limit: int, seed: int) -> np.ndarray:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=limit, replace=False).astype(np.int64))


def compute_neighbors(embeddings: np.ndarray, records: Sequence[Mapping[str, object]], config: NeighborConfig = NeighborConfig()) -> list[dict[str, object]]:
    """Return a bounded deterministic Euclidean kNN table with post-hoc groups."""
    values = _validate_embeddings(embeddings)
    if len(records) != len(values):
        raise ValueError("records and embeddings row counts differ")
    if len(values) < 2:
        return []
    queries = _sample_indices(len(values), config.max_queries, config.seed)
    k = min(config.k, len(values) - 1)
    rows: list[dict[str, object]] = []
    for query in queries:
        distances = np.sqrt(np.maximum(((values - values[query]) ** 2).sum(axis=1), 0.0))
        distances[query] = np.inf
        nearest = np.argsort(distances, kind="stable")[:k]
        for rank, neighbor in enumerate(nearest, start=1):
            query_record, neighbor_record = records[int(query)], records[int(neighbor)]
            row = {
                "query_row_index": int(query), "neighbor_row_index": int(neighbor), "rank": rank,
                "distance": float(distances[neighbor]), "similarity": float(1.0 / (1.0 + distances[neighbor])),
                "query_file_id": str(query_record.get("file_id", "")), "neighbor_file_id": str(neighbor_record.get("file_id", "")),
                "query_S_pred": _value(query_record, "S_pred"), "neighbor_S_pred": _value(neighbor_record, "S_pred"),
                "query_S_pop": _value(query_record, "S_pop"), "neighbor_S_pop": _value(neighbor_record, "S_pop"),
            }
            for field in ("label", "fleet", "regime_summary", "anomaly_family", "severity", "split"):
                left, right = _value(query_record, field), _value(neighbor_record, field)
                row[f"query_{field}"] = left
                row[f"neighbor_{field}"] = right
                row[f"same_{field}"] = bool(left != MISSING and right != MISSING and left == right)
            rows.append(row)
    return rows


def neighbor_group_rates(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize same-group rates with explicit empty/missing behavior."""
    result: dict[str, object] = {"sample_count": len(rows), "groups": {}}
    for field in ("label", "fleet", "regime_summary", "anomaly_family", "severity", "split"):
        comparable = [
            row for row in rows
            if _value(row, f"query_{field}") != MISSING and _value(row, f"neighbor_{field}") != MISSING
        ]
        result["groups"][field] = {
            "comparisons": len(comparable),
            "same_rate": (
                float(sum(bool(row.get(f"same_{field}")) for row in comparable) / len(comparable))
                if comparable else None
            ),
        }
    return result


def compute_projections(embeddings: np.ndarray, config: ProjectionConfig = ProjectionConfig()) -> dict[str, object]:
    """Fit PCA and bounded deterministic t-SNE on an unlabeled sampled subset."""
    values = _validate_embeddings(embeddings)
    indices = _sample_indices(len(values), config.max_samples, config.seed)
    sample = values[indices]
    if len(sample) < 2:
        raise ValueError("at least two embeddings are required for projections")
    components = min(2, sample.shape[0], sample.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    pca_xy = pca.fit_transform(sample)
    pca_xy = np.nan_to_num(pca_xy, nan=0.0, posinf=0.0, neginf=0.0)
    pca_ratio = np.nan_to_num(np.asarray(pca.explained_variance_ratio_, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if components == 1:
        pca_xy = np.column_stack([pca_xy[:, 0], np.zeros(len(sample))])
        pca_ratio = np.array([pca_ratio[0], 0.0], dtype=np.float64)
    perplexity = min(float(config.perplexity), max(1.0, (len(sample) - 1.0) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=config.seed, init="pca", learning_rate="auto", max_iter=1_000)
    tsne_xy = tsne.fit_transform(sample)
    return {"row_index": indices, "pca": pca_xy, "pca_explained_variance_ratio": pca_ratio, "tsne": tsne_xy}

def compute_separation_metrics(
    embeddings: np.ndarray,
    records: Sequence[Mapping[str, object]],
    config: NeighborConfig = NeighborConfig(),
) -> dict[str, object]:
    """Compute diagnostic-only normal distances and labelled similarities."""
    values = _validate_embeddings(embeddings)
    if len(records) != len(values):
        raise ValueError("records and embeddings row counts differ")
    labels = [str(_value(row, "label")) for row in records]
    normal = np.asarray([index for index, label in enumerate(labels) if label == "normal"], dtype=np.int64)
    reference_result: dict[str, object] = {"count": int(len(normal)), "mean_distance": None}
    if len(normal):
        reference_k = min(config.reference_k, len(normal) - 1)
        if reference_k > 0:
            normal_lookup = {int(row): column for column, row in enumerate(normal)}
            total_distance = 0.0
            for start in range(0, len(values), 256):
                chunk = values[start : start + 256]
                distances = np.sqrt(np.maximum(((chunk[:, None, :] - values[normal][None, :, :]) ** 2).sum(axis=2), 0.0))
                for row in range(len(chunk)):
                    normal_column = normal_lookup.get(start + row)
                    if normal_column is not None:
                        distances[row, normal_column] = np.inf
                total_distance += float(np.partition(distances, reference_k - 1, axis=1)[:, :reference_k].sum())
            reference_result["mean_distance"] = total_distance / (len(values) * reference_k)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float64).tiny)
    positives: list[float] = []
    negatives: list[float] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if len(positives) + len(negatives) >= config.max_pairs:
                break
            if labels[left] == MISSING or labels[right] == MISSING:
                continue
            similarity = float(np.dot(normalized[left], normalized[right]))
            (positives if labels[left] == labels[right] else negatives).append(similarity)
        if len(positives) + len(negatives) >= config.max_pairs:
            break
    positive_mean = float(np.mean(positives)) if positives else None
    negative_mean = float(np.mean(negatives)) if negatives else None
    return {
        "normal_reference_distance": reference_result,
        "same_class_similarity": {"count": len(positives), "mean": positive_mean},
        "different_class_similarity": {"count": len(negatives), "mean": negative_mean},
        "class_separation_margin": (positive_mean - negative_mean) if positive_mean is not None and negative_mean is not None else None,
        "true_contrastive_view_metrics": {"available": False, "reason": "cache contains one unaugmented embedding per file; augmented-view positives are unavailable"},
        "labelled_pair_count": len(positives) + len(negatives),
        "definitions": {"class_separation_margin": "mean same-label cosine similarity minus mean different-label cosine similarity; supervised diagnostic only", "missing_label": MISSING},
    }


def write_projection_figures(projections: Mapping[str, object], records: Sequence[Mapping[str, object]], output_dir: str | Path) -> list[Path]:
    """Render deterministic PNGs colored by each available post-hoc field."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(projections["row_index"], dtype=np.int64)
    outputs: list[Path] = []
    group_fields = (
        ("label", "label"),
        ("anomaly_family", "anomaly-family"),
        ("fleet", "fleet"),
        ("regime_summary", "regime"),
        ("severity", "severity"),
        ("split", "split"),
    )
    for name, title in (("pca", "PCA latent geometry"), ("tsne", "t-SNE latent geometry")):
        xy = np.asarray(projections[name])
        for field, legend_title in group_fields:
            groups = [str(_value(records[int(index)], field)) for index in indices]
            fig, axis = plt.subplots(figsize=(7.0, 5.0), dpi=160)
            if field == "severity":
                numeric = np.asarray([float(value) if value != MISSING else np.nan for value in groups], dtype=float)
                available = np.isfinite(numeric)
                if available.any():
                    plotted = axis.scatter(xy[available, 0], xy[available, 1], s=18, alpha=0.8, c=numeric[available], cmap="viridis")
                    fig.colorbar(plotted, ax=axis, label="severity")
                if (~available).any():
                    axis.scatter(xy[~available, 0], xy[~available, 1], s=18, alpha=0.8, color="0.5", label=MISSING)
            else:
                for group in sorted(set(groups)):
                    selected = np.asarray([value == group for value in groups])
                    axis.scatter(xy[selected, 0], xy[selected, 1], s=18, alpha=0.8, label=group)
                axis.legend(title=legend_title, frameon=False)
            axis.set_title(f"{title} by {legend_title} (n={len(xy)})")
            if name == "pca":
                ratio = np.asarray(projections.get("pca_explained_variance_ratio", [0.0, 0.0]))
                axis.set_xlabel(f"component 1 ({ratio[0] * 100:.1f}% variance)")
                axis.set_ylabel(f"component 2 ({ratio[1] * 100:.1f}% variance)")
            else:
                axis.set_xlabel("component 1")
                axis.set_ylabel("component 2")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            suffix = "" if field == "label" else f"_by_{field}"
            path = root / f"{name}{suffix}.png"
            with tempfile.NamedTemporaryFile(dir=root, suffix=".png.tmp", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                fig.savefig(temporary, format="png", metadata={"Software": "latent-geometry"})
                os.replace(temporary, path)
            finally:
                plt.close(fig)
                temporary.unlink(missing_ok=True)
            outputs.append(path)
    return outputs


def write_neighbors(rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    fields = tuple(rows[0].keys()) if rows else ("query_row_index", "neighbor_row_index", "rank", "distance", "similarity")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic(Path(path), output.getvalue().encode("utf-8"))


def analyze_geometry_cache(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
    neighbor_config: NeighborConfig = NeighborConfig(),
    projection_config: ProjectionConfig = ProjectionConfig(),
) -> dict[str, object]:
    """Analyze a read-only cache and publish results to a separate writable directory."""
    base = Path(root)
    output = base if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings, records, manifest = load_geometry_cache(base)
    metrics = compute_health_metrics(embeddings, records)
    neighbor_rows = compute_neighbors(embeddings, records, neighbor_config)
    metrics["neighbor_group_rates"] = neighbor_group_rates(neighbor_rows)
    metrics["separation"] = compute_separation_metrics(embeddings, records, neighbor_config)
    projections = compute_projections(embeddings, projection_config)
    metrics["projections"] = {
        "sample_count": int(len(projections["row_index"])),
        "pca_explained_variance_ratio": np.asarray(projections["pca_explained_variance_ratio"]).tolist(),
    }
    _atomic(output / "metrics.json", (json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    write_neighbors(neighbor_rows, output / "neighbors.csv")
    figure_paths = write_projection_figures(projections, records, output / "figures")

    refreshed = manifest
    if output.resolve() == base.resolve():
        from representation.geometry_contracts import ArtifactInfo
        files = dict(manifest.files)
        for path in (output / "metrics.json", output / "neighbors.csv", *figure_paths):
            relative = path.relative_to(output).as_posix()
            files[relative] = ArtifactInfo(path=relative, bytes=path.stat().st_size, sha256=_sha256(path))
        refreshed = GeometryManifest(**{**manifest.model_dump(mode="python"), "files": files})
        _atomic(output / "geometry-manifest.json", refreshed.to_json().encode("utf-8"))
    return {"manifest": refreshed, "metrics": metrics, "neighbors": neighbor_rows, "projections": projections, "figures": figure_paths, "output_dir": output}
