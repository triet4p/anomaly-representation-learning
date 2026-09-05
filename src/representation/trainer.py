"""Deterministic training loop for the V1 representation model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from representation.inference import NormalReferenceBank

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
        step: int = 0,
        selection_metric: str = "val_stationary_joint_loss",
    ) -> None:
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        if step < 0:
            raise ValueError("step must be non-negative")
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
        self.step = int(step)
        self.selection_metric = str(selection_metric)
        self.best_state: dict[str, object] | None = None
        self.best_loss: float = float("inf")
        self.best_step: int | None = None
        self.best_epoch: int | None = None
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
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            terms = self._terms(batch)
            joint_loss = terms["joint_loss"]
            if not isinstance(joint_loss, torch.Tensor):
                raise TypeError("joint_loss must be a tensor")
            joint_loss.backward()
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(parameters, self.max_grad_norm)
            else:
                grad_norm = clip_grad_norm_(parameters, float("inf"))
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.target_encoder.update(self.model.context_encoder)
            self.step += 1
            for name, value in terms.items():
                if name in ("loss", "masked_count"):
                    continue
                scalar = self._scalar(value)
                totals[name] = totals.get(name, 0.0) + scalar
                counts[name] = counts.get(name, 0) + 1
            grad_norm_scalar = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            lr_scalar = float(self.optimizer.param_groups[0]["lr"])
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + grad_norm_scalar
            counts["grad_norm"] = counts.get("grad_norm", 0) + 1
            totals["lr"] = totals.get("lr", 0.0) + lr_scalar
            counts["lr"] = counts.get("lr", 0) + 1
        if not counts:
            raise ValueError("training epoch received no batches")
        averaged = {name: totals[name] / counts[name] for name in totals}
        averaged["step"] = float(self.step)
        self.history.append(averaged)
        return averaged

    @torch.no_grad()
    def validate(self, batches: Iterable[RepresentationBatch]) -> dict[str, float]:
        """Evaluate without optimizer or EMA mutation, restoring train mode."""
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for batch in batches:
            terms = self._terms(batch)
            for name, value in terms.items():
                if name in ("loss", "masked_count"):
                    continue
                scalar = self._scalar(value)
                totals[name] = totals.get(name, 0.0) + scalar
                counts[name] = counts.get(name, 0) + 1
        if was_training:
            self.model.train()
        if not counts:
            raise ValueError("validation received no batches")
        return {name: totals[name] / counts[name] for name in totals}

    def record_eval(
        self,
        metrics: Mapping[str, float],
        epoch: int | None = None,
    ) -> bool:
        """Record an evaluation point and update best_state coherently if improved."""
        metric_candidates = [
            self.selection_metric,
            (
                f"val_{self.selection_metric}"
                if not self.selection_metric.startswith("val_")
                else self.selection_metric[4:]
            ),
            "val_stationary_joint_loss",
            "stationary_joint_loss",
            "val_joint_loss",
            "joint_loss",
        ]
        score: float | None = None
        for candidate in metric_candidates:
            if candidate in metrics:
                score = float(metrics[candidate])
                break
        if score is None:
            return False

        schedule = getattr(self.criterion, "lambda_schedule", None)
        if schedule is not None and hasattr(schedule, "is_full_ramp"):
            is_eligible = schedule.is_full_ramp(self.step)
        elif schedule is not None:
            boundary = schedule.warmup_steps + (
                schedule.ramp_steps
                if schedule.ramp_steps > 0
                else (1 if schedule.lambda_max > 0.0 else 0)
            )
            is_eligible = self.step >= boundary and (
                schedule.lambda_max == 0.0
                or schedule.lambda_at(self.step) >= schedule.lambda_max
            )
        else:
            is_eligible = True

        if not is_eligible:
            return False

        improved = score < self.best_loss
        pre_ramp_override = False
        if self.best_state is not None and schedule is not None:
            best_step = int(self.best_state.get("step", 0))
            if hasattr(schedule, "is_full_ramp"):
                if not schedule.is_full_ramp(best_step):
                    pre_ramp_override = True
            else:
                boundary = schedule.warmup_steps + (
                    schedule.ramp_steps
                    if schedule.ramp_steps > 0
                    else (1 if schedule.lambda_max > 0.0 else 0)
                )
                if best_step < boundary:
                    pre_ramp_override = True
        if improved or pre_ramp_override:
            self.best_loss = score
            self.best_step = self.step
            self.best_epoch = epoch
            self.best_state = {
                "model": deepcopy(self.model.state_dict()),
                "optimizer": (
                    deepcopy(self.optimizer.state_dict())
                    if self.optimizer is not None
                    else None
                ),
                "scheduler": (
                    deepcopy(self.scheduler.state_dict())
                    if self.scheduler is not None
                    else None
                ),
                "step": self.step,
                "epoch": epoch,
                "score": score,
                "metrics": dict(metrics),
            }
            return True
        return False

    def restore_best_state(self) -> dict[str, object] | None:
        """Restore model, optimizer, scheduler, and step to the best coherent snapshot."""
        if self.best_state is None:
            return None
        self.model.load_state_dict(self.best_state["model"])
        if (
            self.optimizer is not None
            and self.best_state.get("optimizer") is not None
        ):
            self.optimizer.load_state_dict(self.best_state["optimizer"])
        if (
            self.scheduler is not None
            and self.best_state.get("scheduler") is not None
        ):
            self.scheduler.load_state_dict(self.best_state["scheduler"])
        self.step = int(self.best_state["step"])
        return dict(self.best_state)

    def save_checkpoint(
        self,
        path: str | Path,
        reference_bank: NormalReferenceBank | None = None,
    ) -> None:
        """Save a coherent checkpoint matching active model, optimizer, scheduler, and step."""
        from representation.checkpoint import save_checkpoint

        save_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            reference_bank=reference_bank,
        )

    def fit(
        self,
        train_batches: Iterable[RepresentationBatch],
        *,
        epochs: int = 1,
        validation_batches: Iterable[RepresentationBatch] | None = None,
        restore_best: bool = True,
    ) -> list[dict[str, float]]:
        """Train epochs, evaluate validation, restore coherent best state, and return history."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_batches)
            metrics = dict(train_metrics)
            if validation_batches is not None:
                validation = self.validate(validation_batches)
                metrics.update(
                    {f"val_{name}": value for name, value in validation.items()}
                )
                self.record_eval(metrics, epoch=epoch)
            self.history[-1] = metrics
        if restore_best and self.best_state is not None:
            self.restore_best_state()
        return list(self.history)
