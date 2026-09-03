"""Deterministic training loop for the V1 representation model."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from representation.criterion import (
    FileContrastiveCriterion,
    JointRepresentationCriterion,
    ProgressiveLambda,
)
from representation.contracts import RepresentationBatch
from representation.model import V1RepresentationModel


class RepresentationTrainer:
    """Train context/predictor parameters and update EMA only after stepping."""

    def __init__(
        self,
        model: V1RepresentationModel,
        criterion: JointRepresentationCriterion | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_grad_norm: float | None = 1.0,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> None:
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        self.model = model
        if criterion is None:
            config = model.config
            criterion = JointRepresentationCriterion(
                contrastive=FileContrastiveCriterion(temperature=config.contrastive_temperature),
                lambda_schedule=ProgressiveLambda(
                    lambda_max=config.contrastive_weight_max,
                    ramp_steps=config.contrastive_ramp_steps,
                    warmup_steps=config.contrastive_warmup_steps,
                ),
                prediction_weight=config.prediction_weight,
            )
        self.criterion = criterion
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        if optimizer is None:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=1e-3)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.step = 0
        self.history: list[dict[str, float]] = []
        torch.manual_seed(seed)

    def _move_batch(self, batch: RepresentationBatch) -> RepresentationBatch:
        moved: dict[str, object] = dict(batch)
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[name] = value.to(self.device)
        return moved  # type: ignore[return-value]

    @staticmethod
    def _scalar(value: torch.Tensor | float) -> float:
        return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)

    def _terms(self, batch: RepresentationBatch) -> dict[str, torch.Tensor | float]:
        output = self.model(self._move_batch(batch))
        return self.criterion(output, step=self.step)

    def train_epoch(self, batches: Iterable[RepresentationBatch]) -> dict[str, float]:
        """Run one epoch; each EMA update follows its optimizer step."""
        self.model.train()
        totals: dict[str, float] = {"prediction_loss": 0.0, "contrastive_loss": 0.0, "joint_loss": 0.0, "lambda": 0.0}
        count = 0
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            terms = self._terms(batch)
            joint_loss = terms["joint_loss"]
            if not isinstance(joint_loss, torch.Tensor):
                raise TypeError("joint_loss must be a tensor")
            joint_loss.backward()
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.max_grad_norm is not None:
                clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.target_encoder.update(self.model.context_encoder)
            self.step += 1
            count += 1
            for name in totals:
                totals[name] += self._scalar(terms[name])
        if count == 0:
            raise ValueError("training epoch received no batches")
        averaged = {name: value / count for name, value in totals.items()}
        averaged["step"] = float(self.step)
        self.history.append(averaged)
        return averaged

    @torch.no_grad()
    def validate(self, batches: Iterable[RepresentationBatch]) -> dict[str, float]:
        """Evaluate without optimizer or EMA mutation, restoring train mode."""
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {"prediction_loss": 0.0, "contrastive_loss": 0.0, "joint_loss": 0.0, "lambda": 0.0}
        count = 0
        for batch in batches:
            terms = self._terms(batch)
            count += 1
            for name in totals:
                totals[name] += self._scalar(terms[name])
        if was_training:
            self.model.train()
        if count == 0:
            raise ValueError("validation received no batches")
        return {name: value / count for name, value in totals.items()}

    def fit(
        self,
        train_batches: Iterable[RepresentationBatch],
        *,
        epochs: int = 1,
        validation_batches: Iterable[RepresentationBatch] | None = None,
    ) -> list[dict[str, float]]:
        """Train epochs, restore the best validation state, and return history."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        for _ in range(epochs):
            train_metrics = self.train_epoch(train_batches)
            metrics = dict(train_metrics)
            if validation_batches is not None:
                validation = self.validate(validation_batches)
                metrics.update({f"val_{name}": value for name, value in validation.items()})
                if validation["joint_loss"] < best_loss:
                    best_loss = validation["joint_loss"]
                    best_state = deepcopy(self.model.state_dict())
            self.history[-1] = metrics
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return list(self.history)
