"""Deterministic training loop for the V1 representation model."""

from __future__ import annotations

import math
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
    from representation.sprint18_harness import Sprint18ArmConfig


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
        self.objective_balance_history: list[dict[str, int | float | bool]] = []
        torch.manual_seed(seed)

    def _move_batch(self, batch: RepresentationBatch) -> RepresentationBatch:
        moved: dict[str, object] = dict(batch)
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[name] = value.to(self.device)
        return moved  # type: ignore[return-value]

    @staticmethod
    def _scalar(value: torch.Tensor | float | int) -> float:
        return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)

    def _forward_terms(
        self,
        batch: RepresentationBatch,
        *,
        effective_lambda_override: float | None = None,
    ) -> tuple[Mapping[str, object], dict[str, torch.Tensor | float | int]]:
        output = self.model(self._move_batch(batch))
        terms = self.criterion(
            output,
            step=self.step,
            effective_lambda_override=effective_lambda_override,
        )
        return output, terms

    def _terms(
        self,
        batch: RepresentationBatch,
    ) -> dict[str, torch.Tensor | float | int]:
        return self._forward_terms(batch)[1]

    @staticmethod
    def _embedding_geometry(
        embeddings: torch.Tensor,
        supported_files: torch.Tensor,
    ) -> dict[str, int | float]:
        if embeddings.ndim != 2 or embeddings.shape[1] < 1:
            raise ValueError("file embeddings must be a non-empty-width matrix")
        if (
            supported_files.dtype is not torch.bool
            or supported_files.shape != embeddings.shape[:1]
        ):
            raise ValueError("embedding support must be a bool mask over files")
        values = embeddings.detach()[supported_files].to(dtype=torch.float64)
        sample_count, dimension = values.shape
        if sample_count == 0:
            return {
                "embedding_support_files": 0,
                "embedding_variance_mean": 0.0,
                "embedding_std_median": 0.0,
                "embedding_effective_rank": 0.0,
            }
        if not torch.isfinite(values).all():
            raise ValueError("supported file embeddings must be finite")
        centered = values - values.mean(dim=0, keepdim=True)
        population_variance = centered.square().mean(dim=0)
        median_std = torch.quantile(
            population_variance.sqrt(), 0.5, interpolation="linear"
        )
        covariance = (centered.transpose(0, 1) @ centered) / float(
            max(sample_count - 1, 1)
        )
        covariance = (covariance + covariance.transpose(0, 1)) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0).flip(0)
        total_variance = float(eigenvalues.sum().item())
        tolerance = (
            max(total_variance, 1.0)
            * torch.finfo(torch.float64).eps
            * max(sample_count, dimension)
            * 16.0
        )
        positive = eigenvalues[eigenvalues > tolerance]
        if total_variance > tolerance and positive.numel() > 0:
            probabilities = positive / total_variance
            effective_rank = float(
                torch.exp(-(probabilities * probabilities.log()).sum()).item()
            )
        else:
            effective_rank = 0.0
        return {
            "embedding_support_files": sample_count,
            "embedding_variance_mean": float(population_variance.mean().item()),
            "embedding_std_median": float(median_std.item()),
            "embedding_effective_rank": effective_rank,
        }

    @staticmethod
    def _gradient_norm(
        loss: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
    ) -> float:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm: torch.Tensor | None = None
        for gradient in gradients:
            if gradient is None:
                continue
            contribution = gradient.detach().to(dtype=torch.float64).square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
        if squared_norm is None:
            return 0.0
        return float(torch.sqrt(squared_norm).item())

    def _shared_encoder_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in (
            self.model.patch_encoder,
            self.model.context_encoder,
            self.model.conditional_norm,
        ):
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            raise ValueError("shared online encoder has no trainable parameters")
        return tuple(parameters)

    def _balance_record(
        self,
        output: Mapping[str, object],
        terms: Mapping[str, torch.Tensor | float | int],
        *,
        epoch: int,
        global_step: int,
        learning_rate: float,
        parameters: tuple[nn.Parameter, ...],
    ) -> dict[str, int | float | bool]:
        prediction_loss = terms["prediction_loss"]
        weighted_info_loss = terms["weighted_contrastive_loss"]
        if not isinstance(prediction_loss, torch.Tensor) or not isinstance(
            weighted_info_loss, torch.Tensor
        ):
            raise TypeError("objective losses must be tensors for gradient diagnostics")
        prediction_norm = self._gradient_norm(prediction_loss, parameters)
        info_norm = self._gradient_norm(weighted_info_loss, parameters)
        prediction_mask = output.get("prediction_mask")
        valid_mask = output.get("patch_valid_mask")
        embeddings = output.get("file_embedding")
        if (
            not isinstance(prediction_mask, torch.Tensor)
            or not isinstance(valid_mask, torch.Tensor)
            or not isinstance(embeddings, torch.Tensor)
            or prediction_mask.dtype is not torch.bool
            or valid_mask.dtype is not torch.bool
            or prediction_mask.ndim != 2
            or valid_mask.ndim != 2
            or embeddings.ndim != 2
            or prediction_mask.shape != valid_mask.shape
            or embeddings.shape[0] != valid_mask.shape[0]
        ):
            raise ValueError("model output masks/embeddings do not share valid file support")
        valid_prediction = prediction_mask & valid_mask
        geometry = self._embedding_geometry(
            embeddings,
            valid_mask.any(dim=1),
        )
        effective_prediction = self._scalar(
            terms["effective_prediction_coefficient"]
        )
        raw_info = self._scalar(terms["contrastive_loss"])
        coefficient_info = self._scalar(
            terms["effective_contrastive_coefficient"]
        )
        raw_prediction = self._scalar(prediction_loss)
        ratio = info_norm / (prediction_norm + 1e-12)
        record: dict[str, int | float | bool] = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": learning_rate,
            "raw_L_pred": raw_prediction,
            "effective_coefficient_L_pred": effective_prediction,
            "weighted_L_pred": self._scalar(terms["weighted_prediction_loss"]),
            "raw_L_info": raw_info,
            "effective_coefficient_L_info": coefficient_info,
            "weighted_L_info": self._scalar(weighted_info_loss),
            "gradient_norm_L_pred_shared": prediction_norm,
            "gradient_norm_weighted_L_info_shared": info_norm,
            "r_aux": ratio,
            "support_valid_files": int(valid_mask.any(dim=1).sum().item()),
            "support_valid_patches": int(valid_mask.sum().item()),
            "support_prediction_files": int(valid_prediction.any(dim=1).sum().item()),
            "support_prediction_patches": int(valid_prediction.sum().item()),
            "support_info_files": int(terms["contrastive_support_files"]),
            "contrastive_input_files": int(terms["contrastive_input_files"]),
            "protocol_gradient_observation": (
                26 <= epoch <= 30 and global_step % 40 == 0
            ),
        }
        record.update(geometry)
        return record

    def train_epoch(self, batches: Iterable[RepresentationBatch]) -> dict[str, float]:
        """Run one epoch; each EMA update follows its optimizer step."""
        return self._run_epoch(batches)

    def train_sprint18_epoch(
        self,
        batches: Iterable[RepresentationBatch],
        *,
        arm: Sprint18ArmConfig,
        epoch: int,
        collect_balance: bool = True,
    ) -> dict[str, float]:
        """Run one B0 epoch under the accepted Sprint 18 schedule and diagnostics."""
        from representation.sprint18_checkpoint import (
            coefficient_for_epoch,
            learning_rate_for_epoch,
        )
        from representation.sprint18_harness import validate_arm_config

        validate_arm_config(arm, require_locked=(arm.selection_state == "locked"))
        if arm.arm_id != "B0-FULL":
            raise ValueError("Task8 instrumentation currently supports the implemented B0-FULL path")
        if self.scheduler is not None:
            raise ValueError("Sprint18 uses a constant-within-epoch LR; pass no step scheduler")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise ValueError("Sprint18 representation training requires AdamW")
        if self.max_grad_norm != 1.0:
            raise ValueError("Sprint18 representation training requires global L2 clip 1.0")
        if self.criterion.prediction_weight != 1.0:
            raise ValueError("Sprint18 prediction objective coefficient must remain 1.0")
        coefficient_max = arm.lambda_info_max
        if (
            coefficient_max is None
            or self.criterion.lambda_schedule.lambda_max != coefficient_max
            or self.model.config.contrastive_weight_max != coefficient_max
        ):
            raise ValueError("B0 contrastive coefficient differs from the arm configuration")
        model_shape = (
            self.model.config.n_channels,
            self.model.config.patch_size,
            self.model.config.stride,
            self.model.config.n_robots,
            self.model.config.n_programs,
        )
        arm_shape = (
            arm.n_channels,
            arm.patch_size,
            arm.stride,
            arm.n_robots,
            arm.n_programs,
        )
        if (
            model_shape != arm_shape
            or self.model.config.seed != arm.model_seed
            or self.model.config.ema_decay != arm.ema_decay
            or self.model.target_encoder.decay != arm.ema_decay
            or self.model.patchifier.cfg.patch_size != arm.patch_size
            or self.model.patchifier.cfg.stride != arm.stride
        ):
            raise ValueError("B0 model configuration differs from its frozen Sprint18 arm")
        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        model_parameters = list(self.model.parameters())
        optimizer_parameter_ids = {id(parameter) for parameter in optimizer_parameters}
        model_parameter_ids = {id(parameter) for parameter in model_parameters}
        trainable_parameter_ids = {
            id(parameter) for parameter in model_parameters if parameter.requires_grad
        }
        if (
            any(group["weight_decay"] != 0.0 for group in self.optimizer.param_groups)
            or len(optimizer_parameter_ids) != len(optimizer_parameters)
            or not trainable_parameter_ids.issubset(optimizer_parameter_ids)
            or not optimizer_parameter_ids.issubset(model_parameter_ids)
        ):
            raise ValueError("B0 AdamW parameter/weight-decay groups differ from the frozen contract")
        learning_rate = learning_rate_for_epoch(epoch)
        effective_lambda = coefficient_for_epoch(coefficient_max, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return self._run_epoch(
            batches,
            epoch=epoch,
            effective_lambda=effective_lambda,
            learning_rate=learning_rate,
            collect_balance=collect_balance,
        )

    def _run_epoch(
        self,
        batches: Iterable[RepresentationBatch],
        *,
        epoch: int | None = None,
        effective_lambda: float | None = None,
        learning_rate: float | None = None,
        collect_balance: bool = False,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        epoch_balance_records: list[dict[str, int | float | bool]] = []
        shared_parameters = self._shared_encoder_parameters() if collect_balance else ()
        for batch_index, batch in enumerate(batches):
            next_step = self.step + 1
            self.optimizer.zero_grad(set_to_none=True)
            output, terms = self._forward_terms(
                batch,
                effective_lambda_override=effective_lambda,
            )
            pending_record: dict[str, int | float | bool] | None = None
            if collect_balance and epoch is not None and (
                batch_index == 0
                or (26 <= epoch <= 30 and next_step % 40 == 0)
            ):
                if learning_rate is None:
                    raise ValueError("Sprint18 diagnostics require the active epoch learning rate")
                pending_record = self._balance_record(
                    output,
                    terms,
                    epoch=epoch,
                    global_step=next_step,
                    learning_rate=learning_rate,
                    parameters=shared_parameters,
                )
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
            if pending_record is not None:
                self.objective_balance_history.append(pending_record)
                epoch_balance_records.append(pending_record)
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
        if epoch is not None:
            averaged["epoch"] = float(epoch)
        if epoch_balance_records:
            for name, value in epoch_balance_records[0].items():
                averaged[f"balance_{name}"] = float(value)
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
