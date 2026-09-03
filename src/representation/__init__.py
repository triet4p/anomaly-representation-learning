"""Contracts and configuration for the anomaly representation model."""

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
from representation.inference import NormalReferenceBank, ReferenceBank, RepresentationInference, mad_threshold

__all__ = [
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
    "validate_batch",
    "CheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
    "validate_output",
]
