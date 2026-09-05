"""Contracts and configuration for the anomaly representation model."""

from representation.layers.normalization import ConditionalBatchNorm
from representation.config import V1Config
from representation.contracts import (
    RepresentationBatch,
    RepresentationOutput,
    validate_batch,
    validate_output,
)
from representation.model import V1RepresentationModel
from representation.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from representation.criterion import (
    FileContrastiveCriterion,
    JointRepresentationCriterion,
    LatentPredictionCriterion,
    ProgressiveLambda,
)
from representation.inference import NormalReferenceBank, ReferenceBank, RepresentationInference, mad_threshold, prepare_reference_bank
from representation.geometry import (
    BatchConfig,
    BoundedEmbeddingExtractor,
    CompatibilityConfig,
    DiagnosticConfig,
    DiagnosticResult,
    EmbeddingExtractor,
    GeometryCompatibilityError,
    GeometryManifest,
    GeometryOutputPaths,
    GeometryRecord,
    NeighborConfig,
    ProjectionConfig,
    SamplingConfig,
    extract_embeddings,
)

__all__ = [
    "ConditionalBatchNorm",
    "V1Config",
    "V1RepresentationModel",
    "RepresentationBatch",
    "FileContrastiveCriterion",
    "JointRepresentationCriterion",
    "LatentPredictionCriterion",
    "ProgressiveLambda",
    "NormalReferenceBank",
    "ReferenceBank",
    "RepresentationInference",
    "mad_threshold",
    "prepare_reference_bank",
    "validate_batch",
    "CheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
    "validate_output",
    "BatchConfig",
    "BoundedEmbeddingExtractor",
    "CompatibilityConfig",
    "DiagnosticConfig",
    "DiagnosticResult",
    "EmbeddingExtractor",
    "GeometryCompatibilityError",
    "GeometryManifest",
    "GeometryOutputPaths",
    "GeometryRecord",
    "NeighborConfig",
    "ProjectionConfig",
    "SamplingConfig",
    "extract_embeddings",
]
