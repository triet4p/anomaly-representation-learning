"""Losses for the V1 latent prediction and contrastive objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class LatentPredictionCriterion(nn.Module):
    """Compare predicted and EMA target latents only at valid masked patches."""

    def __init__(self, empty_policy: Literal["zero", "error"] = "zero") -> None:
        super().__init__()
        if empty_policy not in {"zero", "error"}:
            raise ValueError("empty_policy must be 'zero' or 'error'")
        self.empty_policy = empty_policy

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return loss, per-patch errors, and auditable masked counts."""
        predicted = self._tensor(model_output, "predicted_latents", 3)
        target = self._tensor(model_output, "target_latents", 3).detach()
        requested = model_output.get("prediction_mask", model_output.get("mask"))
        if not isinstance(requested, torch.Tensor):
            raise ValueError("model output must contain prediction_mask")
        valid = model_output.get("patch_valid_mask")
        if valid is None:
            valid = torch.ones_like(requested, dtype=torch.bool)
        if not isinstance(valid, torch.Tensor):
            raise ValueError("patch_valid_mask must be a torch.Tensor")
        if predicted.shape != target.shape:
            raise ValueError("predicted_latents and target_latents must share shape")
        if requested.shape != predicted.shape[:2] or valid.shape != predicted.shape[:2]:
            raise ValueError("prediction masks must match latent batch and count")
        if requested.dtype is not torch.bool or valid.dtype is not torch.bool:
            raise ValueError("prediction masks must have torch.bool dtype")
        mask = requested & valid
        errors = (predicted - target).square().mean(dim=-1).masked_fill(~mask, 0.0)
        counts_per_file = mask.sum(dim=1)
        masked_count = counts_per_file.sum()
        if masked_count.item() == 0:
            if self.empty_policy == "error":
                raise ValueError("no valid masked patches available for prediction loss")
            loss = predicted.sum() * 0.0
        else:
            loss = errors.sum() / masked_count.to(errors.dtype)
        return {"loss": loss, "patch_prediction_error": errors, "masked_count": masked_count, "masked_counts": counts_per_file}

    @staticmethod
    def _tensor(model_output: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D torch.Tensor")
        return value


class FileContrastiveCriterion(nn.Module):
    """Apply symmetric in-batch InfoNCE to two views of each file."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return contrastive loss and normalized same-file view embeddings."""
        view_1 = self._embedding(model_output, "view_embedding_1")
        view_2 = self._embedding(model_output, "view_embedding_2")
        if view_1.shape != view_2.shape:
            raise ValueError("contrastive views must share shape")
        if view_1.shape[0] == 0:
            raise ValueError("contrastive views cannot be empty")
        normalized_1 = F.normalize(view_1, dim=-1)
        normalized_2 = F.normalize(view_2, dim=-1)
        batch_size = view_1.shape[0]
        if batch_size == 1:
            loss = (normalized_1 - normalized_2).square().sum() * 0.0
            similarity = (normalized_1 * normalized_2).sum(dim=-1).mean()
            negative_similarity = torch.tensor(0.0, device=view_1.device)
            margin = similarity
        else:
            sim_12 = normalized_1 @ normalized_2.transpose(0, 1)
            sim_21 = normalized_2 @ normalized_1.transpose(0, 1)
            logits_12 = sim_12 / self.temperature
            logits_21 = sim_21 / self.temperature
            labels = torch.arange(batch_size, device=view_1.device)
            loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
            similarity = sim_12.diag().mean()
            neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=view_1.device)
            negative_similarity = 0.5 * (sim_12[neg_mask].mean() + sim_21[neg_mask].mean())
            margin = similarity - negative_similarity
        return {
            "loss": loss,
            "similarity": similarity,
            "negative_similarity": negative_similarity,
            "margin": margin,
            "normalized_view_1": normalized_1,
            "normalized_view_2": normalized_2,
            "batch_size": torch.tensor(batch_size, device=view_1.device),
        }

    @staticmethod
    def _embedding(model_output: Mapping[str, object], name: str) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a 2D torch.Tensor")
        return value


class ProgressiveLambda:
    """Monotonic linear contrastive-weight schedule with an optional warmup."""

    def __init__(
        self,
        lambda_max: float = 1.0,
        ramp_steps: int = 1_000,
        warmup_steps: int = 0,
    ) -> None:
        if lambda_max < 0.0:
            raise ValueError("lambda_max must be non-negative")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.lambda_max = float(lambda_max)
        self.ramp_steps = int(ramp_steps)
        self.warmup_steps = int(warmup_steps)

    def lambda_at(self, step: int) -> float:
        """Return the exact contrastive weight at a non-negative step."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if step <= self.warmup_steps:
            return 0.0
        if self.ramp_steps == 0:
            return self.lambda_max
        progress = (step - self.warmup_steps) / self.ramp_steps
        return self.lambda_max * min(1.0, progress)

    @property
    def full_ramp_step(self) -> int:
        """Return the first non-negative step at which lambda reaches lambda_max."""
        if self.ramp_steps == 0:
            return self.warmup_steps + (1 if self.lambda_max > 0.0 else 0)
        return self.warmup_steps + self.ramp_steps

    def is_full_ramp(self, step: int) -> bool:
        """Return whether lambda has reached lambda_max at or past the full-ramp boundary."""
        if step < 0:
            raise ValueError("step must be non-negative")
        return step >= self.full_ramp_step and (
            self.lambda_max == 0.0 or self.lambda_at(step) >= self.lambda_max
        )


class JointRepresentationCriterion(nn.Module):
    """Combine masked prediction and progressive file-level contrastive losses."""

    def __init__(self, prediction: LatentPredictionCriterion | None = None, contrastive: FileContrastiveCriterion | None = None, lambda_schedule: ProgressiveLambda | None = None, prediction_weight: float = 1.0) -> None:
        super().__init__()
        if prediction_weight < 0.0:
            raise ValueError("prediction_weight must be non-negative")
        self.prediction = prediction or LatentPredictionCriterion()
        self.contrastive = contrastive or FileContrastiveCriterion()
        self.lambda_schedule = lambda_schedule or ProgressiveLambda()
        self.prediction_weight = float(prediction_weight)

    def forward(self, model_output: Mapping[str, object], step: int = 0) -> dict[str, torch.Tensor | float]:
        """Return each objective term, schedule value, and joint loss."""
        prediction_result = self.prediction(model_output)
        prediction_loss = prediction_result["loss"]
        contrastive_sim = None
        contrastive_neg_sim = None
        contrastive_margin = None
        if "view_embedding_1" in model_output and "view_embedding_2" in model_output:
            contrastive_result = self.contrastive(model_output)
            contrastive_loss = contrastive_result["loss"]
            if "similarity" in contrastive_result:
                contrastive_sim = contrastive_result["similarity"]
            if "negative_similarity" in contrastive_result:
                contrastive_neg_sim = contrastive_result["negative_similarity"]
            if "margin" in contrastive_result:
                contrastive_margin = contrastive_result["margin"]
        else:
            contrastive_loss = prediction_loss * 0.0
        lambda_value = self.lambda_schedule.lambda_at(step)
        weighted_contrastive_loss = lambda_value * contrastive_loss
        joint_loss = self.prediction_weight * prediction_loss + weighted_contrastive_loss
        stationary_lambda = self.lambda_schedule.lambda_max
        stationary_joint_loss = self.prediction_weight * prediction_loss + stationary_lambda * contrastive_loss

        diagnostics: dict[str, torch.Tensor | float] = {
            "prediction_loss": prediction_loss,
            "contrastive_loss": contrastive_loss,
            "weighted_contrastive_loss": weighted_contrastive_loss,
            "joint_loss": joint_loss,
            "stationary_joint_loss": stationary_joint_loss,
            "loss": joint_loss,
            "lambda": lambda_value,
            "effective_lambda": lambda_value,
            "stationary_lambda": stationary_lambda,
            "masked_count": prediction_result["masked_count"],
        }
        if contrastive_sim is not None:
            diagnostics["contrastive_sim"] = contrastive_sim
        if contrastive_neg_sim is not None:
            diagnostics["contrastive_neg_sim"] = contrastive_neg_sim
        if contrastive_margin is not None:
            diagnostics["contrastive_margin"] = contrastive_margin
        pred_mask = model_output.get("prediction_mask")
        valid_mask = model_output.get("patch_valid_mask")

        context = model_output.get("context_latents")
        if isinstance(context, torch.Tensor) and context.ndim == 3:
            if isinstance(valid_mask, torch.Tensor):
                valid_float = valid_mask.to(context.dtype)
                diagnostics["context_norm"] = (context.detach().norm(dim=-1) * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            else:
                diagnostics["context_norm"] = context.detach().norm(dim=-1).mean()

        target = model_output.get("target_latents")
        if isinstance(target, torch.Tensor) and target.ndim == 3:
            if isinstance(valid_mask, torch.Tensor):
                valid_float = valid_mask.to(target.dtype)
                diagnostics["target_norm"] = (target.detach().norm(dim=-1) * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            else:
                diagnostics["target_norm"] = target.detach().norm(dim=-1).mean()

        predicted = model_output.get("predicted_latents")
        if isinstance(predicted, torch.Tensor) and predicted.ndim == 3:
            if isinstance(pred_mask, torch.Tensor):
                pred_float = pred_mask.to(predicted.dtype)
                diagnostics["predicted_norm"] = (predicted.detach().norm(dim=-1) * pred_float).sum() / pred_float.sum().clamp_min(1.0)
            else:
                diagnostics["predicted_norm"] = predicted.detach().norm(dim=-1).mean()

        file_emb = model_output.get("file_embedding")
        if isinstance(file_emb, torch.Tensor) and file_emb.ndim == 2:
            diagnostics["file_embedding_norm"] = file_emb.detach().norm(dim=-1).mean()

        return diagnostics
