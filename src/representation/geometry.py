"""Public latent-geometry extraction API."""

from representation.embedding_extraction import (
    BoundedEmbeddingExtractor,
    EmbeddingExtractor,
    GeometryCompatibilityError,
    extract_embeddings,
    validate_dataset_compatibility,
)
from representation.geometry_contracts import (
    GEOMETRY_ARTIFACT_NAMES,
    GEOMETRY_SCHEMA_VERSION,
    ArtifactInfo,
    BatchConfig,
    CompatibilityConfig,
    DiagnosticConfig,
    DiagnosticResult,
    GeometryManifest,
    GeometryOutputPaths,
    GeometryRecord,
    NeighborConfig,
    ProjectionConfig,
    SamplingConfig,
)

__all__ = [
    "ArtifactInfo",
    "BatchConfig",
    "BoundedEmbeddingExtractor",
    "CompatibilityConfig",
    "DiagnosticConfig",
    "DiagnosticResult",
    "EmbeddingExtractor",
    "GEOMETRY_ARTIFACT_NAMES",
    "GEOMETRY_SCHEMA_VERSION",
    "GeometryCompatibilityError",
    "GeometryManifest",
    "GeometryOutputPaths",
    "GeometryRecord",
    "NeighborConfig",
    "ProjectionConfig",
    "SamplingConfig",
    "extract_embeddings",
    "validate_dataset_compatibility",
]
