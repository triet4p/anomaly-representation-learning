"""Independent context and population inference for V1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from representation.config import V1Config
from representation.contracts import RepresentationBatch
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from synth.config import MaskingConfig
from synth.patchify import Patchifier
from synth.schema import FileSample


class NormalReferenceBank:
    """CPU reference embeddings fitted exclusively from normal files."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.embeddings: torch.Tensor | None = None

    def fit(
        self,
        embeddings: torch.Tensor,
        labels: Sequence[object] | None = None,
    ) -> "NormalReferenceBank":
        """Store normal embeddings, rejecting any abnormal-labelled row."""
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty [M, D] tensor")
        if not torch.isfinite(embeddings).all():
            raise ValueError("reference embeddings must be finite")
        if labels is not None:
            if len(labels) != embeddings.shape[0]:
                raise ValueError("labels length must match embeddings")
            for label in labels:
                value = getattr(label, "value", label)
                if str(value).lower() == "abnormal":
                    raise ValueError("normal reference bank cannot include abnormal samples")
        self.embeddings = embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
        return self

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return mean distance to up to ``k`` nearest normal references."""
        if self.embeddings is None:
            raise ValueError("reference bank has not been fitted")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query embeddings must be [B, D] with matching D")
        distances = torch.cdist(embeddings.float().cpu(), self.embeddings)
        nearest = distances.topk(min(self.k, self.embeddings.shape[0]), largest=False, dim=1).values
        return nearest.mean(dim=1).to(device=embeddings.device)


ReferenceBank = NormalReferenceBank


def mad_threshold(scores: torch.Tensor, multiplier: float = 3.0) -> float:
    """Return a robust median-plus-MAD threshold for one independent score."""
    if multiplier < 0.0:
        raise ValueError("multiplier must be non-negative")
    if scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite and non-empty")
    median = scores.median()
    mad = (scores - median).abs().median()
    return float((median + multiplier * mad).item())


class RepresentationInference:
    """Compute context and population scores without fusing them."""

    def __init__(
        self,
        model: V1RepresentationModel,
        reference_bank: NormalReferenceBank,
        patchifier: Patchifier,
        *,
        masking_config: V1Config | MaskingConfig | None = None,
    ) -> None:
        self.model = model
        self.reference_bank = reference_bank
        self.patchifier = patchifier
        self.masking_config = masking_config

    @torch.no_grad()
    def score_batch(self, batch: RepresentationBatch) -> dict[str, object]:
        """Score a padded batch and return independent patch/timestep localization."""
        self.model.eval()
        output = self.model(batch)
        predicted = output["predicted_latents"]
        target = output["target_latents"]
        prediction_mask = output["prediction_mask"]
        errors = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
        errors = errors.masked_fill(~prediction_mask, 0.0)
        counts = prediction_mask.sum(dim=1)
        context_scores = errors.sum(dim=1) / counts.clamp_min(1).to(errors.dtype)
        population_scores = self.reference_bank.score(output["file_embedding"])
        timestep_scores: list[np.ndarray] = []
        starts = batch["starts"].cpu().numpy()
        valid_len = batch["valid_len"].cpu().numpy()
        patch_errors = errors.cpu().numpy()
        lengths = batch["file_valid_mask"].sum(dim=1).cpu().tolist()
        for index, length in enumerate(lengths):
            timestep_scores.append(
                self.patchifier.patch_to_timestep_scores(
                    patch_errors[index], starts[index], valid_len[index], int(length)
                )
            )
        return {
            "S_pred": context_scores,
            "S_pop": population_scores,
            "context_score": context_scores,
            "population_score": population_scores,
            "patch_scores": errors,
            "timestep_scores": timestep_scores,
            "prediction_mask": prediction_mask,
        }

    @torch.no_grad()
    def score_file(self, sample: FileSample) -> dict[str, object]:
        """Patchify and score one complete variable-length file."""
        batch = collate_variable_files(
            [sample], self.patchifier, masking_config=self.masking_config
        )
        return self.score_batch(batch)

    @staticmethod
    def thresholds(
        result: dict[str, object], multiplier: float = 3.0
    ) -> dict[str, float]:
        """Compute independent MAD thresholds for context and population scores."""
        context = result["S_pred"]
        population = result["S_pop"]
        if not isinstance(context, torch.Tensor) or not isinstance(population, torch.Tensor):
            raise ValueError("inference result has invalid score tensors")
        return {
            "S_pred_threshold": mad_threshold(context, multiplier),
            "S_pop_threshold": mad_threshold(population, multiplier),
        }
