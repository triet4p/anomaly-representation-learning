"""Validated contracts for the portable latent-geometry cache.

The contracts in this module deliberately describe *diagnostics* rather than model
inputs.  They keep split selection, sampling and artifact linkage explicit so a
CPU analysis stage can consume a cache without importing or loading a model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GEOMETRY_SCHEMA_VERSION = 1
GEOMETRY_ARTIFACT_NAMES = (
    "geometry-manifest.json",
    "embeddings.npz",
    "records.csv",
    "metrics.json",
    "neighbors.csv",
)


class SamplingConfig(BaseModel):
    """Deterministic bounded sample selection."""

    model_config = ConfigDict(extra="forbid")

    max_samples: int = Field(default=10_000, ge=1)
    seed: int = Field(default=0, ge=0)
    strategy: Literal["head", "reservoir"] = "head"


class BatchConfig(BaseModel):
    """Bound on raw files presented to one model forward."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, ge=1)


class CompatibilityConfig(BaseModel):
    """Version and feature checks shared by extraction and analysis stages."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_schema_version: int = Field(default=1, ge=1)
    dataset_manifest_format: int = Field(default=2, ge=1)
    expected_channels: int | None = Field(default=None, ge=1)
    expected_embedding_dim: int | None = Field(default=None, ge=1)


class ProjectionConfig(BaseModel):
    """Bounded deterministic projection settings for the later analysis stage."""

    model_config = ConfigDict(extra="forbid")

    max_samples: int = Field(default=5_000, ge=1)
    seed: int = Field(default=0, ge=0)
    perplexity: float = Field(default=30.0, gt=0.0)


class NeighborConfig(BaseModel):
    """Bounded nearest-neighbor report settings."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=15, ge=1)
    max_queries: int = Field(default=10_000, ge=1)
    seed: int = Field(default=0, ge=0)
    reference_k: int = Field(default=5, ge=1)
    max_pairs: int = Field(default=10_000, ge=1)


class DiagnosticResult(BaseModel):
    """Small typed result envelope shared by extraction and analysis stages."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=GEOMETRY_SCHEMA_VERSION, ge=1)
    records_count: int = Field(ge=0)
    feature_dim: int = Field(ge=1)
    artifact_root: Path


class DiagnosticConfig(BaseModel):
    """Bounded and reproducible Stage 1 extraction configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset_root: Path
    checkpoint_path: Path
    output_dir: Path
    splits: tuple[str, ...] = ("train", "val", "test")
    reference_split: str = "train"
    max_samples: int = Field(default=10_000, ge=1)
    max_reference_samples: int = Field(default=10_000, ge=1)
    batch_size: int = Field(default=32, ge=1)
    seed: int = Field(default=0, ge=0)
    sampling: Literal["head", "reservoir"] = "head"
    schema_version: int = Field(default=GEOMETRY_SCHEMA_VERSION, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @field_validator("splits")
    @classmethod
    def validate_splits(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("splits must contain at least one split")
        if any(not item or item.strip() != item or any(ch.isspace() for ch in item) for item in value):
            raise ValueError("split names must be non-empty and whitespace-free")
        if len(set(value)) != len(value):
            raise ValueError("splits must not contain duplicates")
        return value
    @field_validator("reference_split")
    @classmethod
    def validate_reference_split(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("reference_split must be a non-empty name")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "DiagnosticConfig":
        if self.schema_version != GEOMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported geometry schema_version {self.schema_version}; "
                f"expected {GEOMETRY_SCHEMA_VERSION}"
            )
        if not self.checkpoint_path.name:
            raise ValueError("checkpoint_path must name a file")
        return self

    @property
    def split_names(self) -> tuple[str, ...]:
        """Alias used by callers that call the selected values split names."""
        return self.splits


class GeometryOutputPaths(BaseModel):
    """Portable, fixed-name artifact locations below one output directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "geometry-manifest.json"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings.npz"

    @property
    def records(self) -> Path:
        return self.root / "records.csv"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def neighbors(self) -> Path:
        return self.root / "neighbors.csv"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def as_dict(self) -> dict[str, str]:
        return {
            "geometry-manifest.json": self.manifest.name,
            "embeddings.npz": self.embeddings.name,
            "records.csv": self.records.name,
            "metrics.json": self.metrics.name,
            "neighbors.csv": self.neighbors.name,
            "figures": self.figures.name,
        }


class GeometryRecord(BaseModel):
    """One row in ``records.csv``; unavailable values remain explicit blanks."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0)
    file_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    label: Literal["normal", "abnormal"]
    anomaly_family: str = ""
    fleet: str = ""
    regime_summary: str = ""
    severity: float | None = Field(default=None, ge=0.0, le=1.0)
    length: int = Field(ge=1)
    generator_version: str = ""
    config_hash: str = ""
    sample_seed: int | None = Field(default=None, ge=0)
    robot_idx: int | None = Field(default=None, ge=0)
    program_idx: int | None = Field(default=None, ge=0)
    S_pred: float | None = None
    S_pop: float | None = None

    @field_validator("file_id", "split")
    @classmethod
    def no_newlines(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("record identifiers cannot contain newlines")
        return value

    @model_validator(mode="after")
    def validate_scores(self) -> "GeometryRecord":
        for name in ("S_pred", "S_pop"):
            score = getattr(self, name)
            if score is not None and not math.isfinite(float(score)):
                raise ValueError(f"{name} must be finite when available")
        return self


class ArtifactInfo(BaseModel):
    """Integrity metadata for one published file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("artifact paths must be portable relative paths")
        return value


class GeometryManifest(BaseModel):
    """Machine-readable linkage and provenance for a geometry cache."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=GEOMETRY_SCHEMA_VERSION, ge=1)
    artifact_type: Literal["latent_geometry_cache"] = "latent_geometry_cache"
    checkpoint: dict[str, object]
    dataset: dict[str, object]
    extraction: dict[str, object]
    feature_dim: int = Field(ge=1)
    records_count: int = Field(ge=0)
    splits: dict[str, int]
    files: dict[str, ArtifactInfo]

    @field_validator("splits")
    @classmethod
    def validate_split_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("split counts must be non-negative")
        return value
    @model_validator(mode="after")
    def validate_manifest(self) -> "GeometryManifest":
        if self.schema_version != GEOMETRY_SCHEMA_VERSION:
            raise ValueError("geometry manifest schema_version is incompatible")
        if sum(self.splits.values()) != self.records_count:
            raise ValueError("split counts must sum to records_count")
        for required in ("embeddings.npz", "records.csv", "metrics.json", "neighbors.csv"):
            if required not in self.files:
                raise ValueError(f"manifest missing artifact entry {required}")
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


RECORD_FIELDS = tuple(GeometryRecord.model_fields)
