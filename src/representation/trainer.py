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
    HierarchicalContrastiveCriterion,
    JointRepresentationCriterion,
    ProgressiveLambda,
    ccrp_assign_channels,
    ccrp_patch_count,
    ccrp_patch_seed,
    ccrp_sample_targets,
    hcc_relation_support,
    msp_allocate_budget,
    msp_fine_mask,
    msp_mask_budget,
    msp_mask_seed,
    msp_place_groups,
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
        sprint18_view_epoch: int | None = None,
    ) -> tuple[Mapping[str, object], dict[str, torch.Tensor | float | int]]:
        moved = self._move_batch(batch)
        if sprint18_view_epoch is None:
            output = self.model(moved)
        else:
            output = self.model(moved, sprint18_view_epoch=sprint18_view_epoch)
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

    @staticmethod
    def hcc_objective_terms(
        output: Mapping[str, object],
        batch: Mapping[str, object],
        *,
        temperature: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        """Compute the frozen HCC relation-target objective from encoder-visible identity only."""
        view_1 = output.get("view_embedding_1")
        view_2 = output.get("view_embedding_2")
        robot_idx = batch.get("robot_idx")
        program_idx = batch.get("program_idx")
        if (
            not isinstance(view_1, torch.Tensor)
            or not isinstance(view_2, torch.Tensor)
            or not isinstance(robot_idx, torch.Tensor)
            or not isinstance(program_idx, torch.Tensor)
        ):
            raise ValueError("HCC objective requires view embeddings and robot/program identity tensors")
        return HierarchicalContrastiveCriterion(temperature=temperature)(
            view_1, view_2, robot_idx.to(device=view_1.device), program_idx.to(device=view_1.device)
        )

    def _hcc_balance_record(
        self,
        output: Mapping[str, object],
        batch: Mapping[str, object],
        terms: Mapping[str, torch.Tensor | float | int],
        hcc_loss: torch.Tensor,
        weighted_hcc_loss: torch.Tensor,
        effective_hcc: float,
        *,
        epoch: int,
        global_step: int,
        learning_rate: float,
        parameters: tuple[nn.Parameter, ...],
    ) -> dict[str, int | float | bool]:
        prediction_loss = terms["prediction_loss"]
        if not isinstance(prediction_loss, torch.Tensor) or not isinstance(weighted_hcc_loss, torch.Tensor):
            raise TypeError("objective losses must be tensors for gradient diagnostics")
        prediction_norm = self._gradient_norm(prediction_loss, parameters)
        hcc_norm = self._gradient_norm(weighted_hcc_loss, parameters)
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
        geometry = self._embedding_geometry(embeddings, valid_mask.any(dim=1))
        effective_prediction = self._scalar(terms["effective_prediction_coefficient"])
        raw_hcc = float(hcc_loss.detach().cpu().item())
        raw_prediction = self._scalar(prediction_loss)
        ratio = hcc_norm / (prediction_norm + 1e-12)
        robot_idx = batch.get("robot_idx")
        program_idx = batch.get("program_idx")
        if not isinstance(robot_idx, torch.Tensor) or not isinstance(program_idx, torch.Tensor):
            raise ValueError("HCC balance diagnostics require robot/program identity tensors")
        support = hcc_relation_support(robot_idx.detach().cpu(), program_idx.detach().cpu())
        record: dict[str, int | float | bool] = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": learning_rate,
            "raw_L_pred": raw_prediction,
            "effective_coefficient_L_pred": effective_prediction,
            "weighted_L_pred": self._scalar(terms["weighted_prediction_loss"]),
            "raw_L_hcc": raw_hcc,
            "effective_coefficient_L_hcc": effective_hcc,
            "weighted_L_hcc": float(weighted_hcc_loss.detach().cpu().item()),
            "gradient_norm_L_pred_shared": prediction_norm,
            "gradient_norm_weighted_L_hcc_shared": hcc_norm,
            "r_aux": ratio,
            "support_valid_files": int(valid_mask.any(dim=1).sum().item()),
            "support_valid_patches": int(valid_mask.sum().item()),
            "support_prediction_files": int(valid_prediction.any(dim=1).sum().item()),
            "support_prediction_patches": int(valid_prediction.sum().item()),
            "support_hcc_files": int(embeddings.shape[0]),
            "contrastive_input_files": int(embeddings.shape[0]),
            "anchors_same_rp": int(support["anchors_same_rp"]),
            "anchors_same_r_diff_p": int(support["anchors_same_r_diff_p"]),
            "anchors_only_same_file": int(support["anchors_only_same_file"]),
            "target_entropy_mean": float(support["target_entropy_mean"]),
            "protocol_gradient_observation": (26 <= epoch <= 30 and global_step % 40 == 0),
        }
        record.update(geometry)
        return record

    @staticmethod
    def msp_batch_masks(
        batch: Mapping[str, object],
        *,
        model_seed: int,
        epoch: int,
        fit_role_sha256: str,
    ) -> dict[str, object]:
        """Build the frozen N2 union mask plus coarse groups for one collated batch.

        Uses per-file N2 mask seeds, seeded capacity-preserving placement, and
        the B0 fine-mask override on remaining free indices. Returns the union
        mask, coarse groups per file, and per-file allocation diagnostics.
        Consumes no labels/outcomes.
        """
        import numpy as np

        from synth.schema import PatchBatch

        file_ids = batch.get("file_ids")
        samples = batch.get("file_samples")
        valid_mask = batch.get("patch_valid_mask")
        pad_mask = batch.get("patch_pad_mask")
        patches = batch.get("patches")
        if not isinstance(file_ids, (list, tuple)) or not file_ids:
            raise ValueError("N2 masking requires batch file IDs")
        if samples is None or len(samples) != len(file_ids):
            raise ValueError("N2 masking requires file_samples metadata")
        if (
            not isinstance(valid_mask, torch.Tensor)
            or not isinstance(pad_mask, torch.Tensor)
            or valid_mask.dtype is not torch.bool
        ):
            raise ValueError("N2 masking requires bool patch_valid_mask/patch_pad_mask")
        batch_size, max_patches = valid_mask.shape
        union = torch.zeros((batch_size, max_patches), dtype=torch.bool)
        groups_4_all: list[list[tuple[int, ...]]] = []
        groups_8_all: list[list[tuple[int, ...]]] = []
        diagnostics: list[dict[str, object]] = []
        for row, file_id in enumerate(file_ids):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("N2 masking requires canonical file IDs")
            n_valid = int(valid_mask[row].sum().item())
            valid_positions = [int(i) for i in range(max_patches) if bool(valid_mask[row, i].item())]
            budget = msp_mask_budget(n_valid)
            allocation = msp_allocate_budget(budget)
            fine_target, g4, g8 = allocation
            seed = msp_mask_seed(model_seed, epoch, fit_role_sha256, file_id)
            rng = np.random.default_rng(seed)
            placement = msp_place_groups(valid_positions, g8, g4, rng)
            groups_8 = [tuple(int(m) for m in g) for g in placement["groups_8"]]  # type: ignore[union-attr]
            groups_4 = [tuple(int(m) for m in g) for g in placement["groups_4"]]  # type: ignore[union-attr]
            remaining = [int(v) for v in placement["remaining_free"]]  # type: ignore[union-attr]
            fine_count = budget - 8 * len(groups_8) - 4 * len(groups_4)
            sample = samples[row]
            if getattr(sample, "file_id", file_id) != file_id:
                raise ValueError("N2 batch file identity does not match its samples")
            # Original patch data for B0 info scores: full [N, C, W] row indexed
            # by original patch position (valid positions need not be a prefix).
            patch_data = patches[row].detach().cpu().numpy() if isinstance(patches, torch.Tensor) else np.zeros((max_patches, 1, 1), dtype=np.float32)
            fine = msp_fine_mask(remaining, fine_count, np.asarray(patch_data), rng)
            members: set[int] = set()
            for group in groups_8:
                members.update(group)
            for group in groups_4:
                members.update(group)
            members.update(int(v) for v in fine["selected"])  # type: ignore[union-attr]
            if len(members) != budget:
                raise ValueError("N2 mask union does not match its exact integer budget")
            for member in members:
                union[row, member] = True
            groups_4_all.append(groups_4)
            groups_8_all.append(groups_8)
            diagnostics.append(
                {
                    "file_id": file_id,
                    "n": n_valid,
                    "M": budget,
                    "allocation": allocation,
                    "q8": len(groups_8),
                    "q4": len(groups_4),
                    "q8_fallback_to_4": placement["q8_fallback_to_4"],
                    "q4_fallback_to_fine": placement["q4_fallback_to_fine"],
                    "fine": len(fine["selected"]),  # type: ignore[union-attr]
                    "composition": fine["composition"],
                }
            )
        return {
            "union_mask": union,
            "groups_4": groups_4_all,
            "groups_8": groups_8_all,
            "diagnostics": diagnostics,
        }

    def _msp_balance_record(
        self,
        output: Mapping[str, object],
        terms: Mapping[str, torch.Tensor | float | int],
        coarse_loss: torch.Tensor,
        weighted_coarse_loss: torch.Tensor,
        effective_alpha: float,
        effective_info: float,
        *,
        epoch: int,
        global_step: int,
        learning_rate: float,
        parameters: tuple[nn.Parameter, ...],
        groups_4_count: int,
        groups_8_count: int,
    ) -> dict[str, int | float | bool]:
        prediction_loss = terms["prediction_loss"]
        if not isinstance(prediction_loss, torch.Tensor) or not isinstance(weighted_coarse_loss, torch.Tensor):
            raise TypeError("objective losses must be tensors for gradient diagnostics")
        prediction_norm = self._gradient_norm(prediction_loss, parameters)
        coarse_norm = self._gradient_norm(weighted_coarse_loss, parameters)
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
        geometry = self._embedding_geometry(embeddings, valid_mask.any(dim=1))
        effective_prediction = self._scalar(terms["effective_prediction_coefficient"])
        raw_coarse = float(coarse_loss.detach().cpu().item())
        raw_prediction = self._scalar(prediction_loss)
        ratio = coarse_norm / (prediction_norm + 1e-12)
        record: dict[str, int | float | bool] = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": learning_rate,
            "raw_L_pred": raw_prediction,
            "effective_coefficient_L_pred": effective_prediction,
            "weighted_L_pred": self._scalar(terms["weighted_prediction_loss"]),
            "raw_L_coarse": raw_coarse,
            "effective_coefficient_L_coarse": effective_alpha,
            "weighted_L_coarse": float(weighted_coarse_loss.detach().cpu().item()),
            "raw_L_info": self._scalar(terms["contrastive_loss"]),
            "effective_coefficient_L_info": effective_info,
            "weighted_L_info": self._scalar(terms["weighted_contrastive_loss"]),
            "gradient_norm_L_pred_shared": prediction_norm,
            "gradient_norm_weighted_L_coarse_shared": coarse_norm,
            "r_aux": ratio,
            "support_valid_files": int(valid_mask.any(dim=1).sum().item()),
            "support_valid_patches": int(valid_mask.sum().item()),
            "support_prediction_files": int(valid_prediction.any(dim=1).sum().item()),
            "support_prediction_patches": int(valid_prediction.sum().item()),
            "support_coarse_g4": int(groups_4_count),
            "support_coarse_g8": int(groups_8_count),
            "contrastive_input_files": int(terms["contrastive_input_files"]),
            "protocol_gradient_observation": (26 <= epoch <= 30 and global_step % 40 == 0),
        }
        record.update(geometry)
        return record

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
        """Run one B0/N1 epoch under the accepted Sprint 18 schedule and diagnostics."""
        from representation.sprint18_checkpoint import (
            coefficient_for_epoch,
            learning_rate_for_epoch,
        )
        from representation.sprint18_harness import (
            TASK9_COEFFICIENT_PILOT_EPOCHS,
            validate_arm_config,
        )

        validate_arm_config(arm, require_locked=(arm.selection_state == "locked"))
        if arm.arm_id == "N1-HCC":
            return self.train_sprint18_hcc_epoch(batches, arm=arm, epoch=epoch, collect_balance=collect_balance)
        if arm.arm_id == "N2-MSP":
            return self.train_sprint18_msp_epoch(batches, arm=arm, epoch=epoch, collect_balance=collect_balance)
        if arm.arm_id == "N3-CCRP":
            return self.train_sprint18_ccrp_epoch(batches, arm=arm, epoch=epoch, collect_balance=collect_balance)
        if arm.selection_state == "coefficient_pilot" and (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= TASK9_COEFFICIENT_PILOT_EPOCHS
        ):
            raise ValueError("Task9 coefficient pilots must stop at the frozen epoch-30 full-ramp checkpoint")
        if (
            arm.selection_state == "coefficient_pilot"
            and arm.lambda_info_max != 0.0
            and not collect_balance
        ):
            raise ValueError("Task9 coefficient pilots require objective-balance diagnostics")
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
            sprint18_view_epoch=epoch,
        )

    def train_sprint18_hcc_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        arm: Sprint18ArmConfig,
        epoch: int,
        collect_balance: bool = True,
    ) -> dict[str, float]:
        """Run one N1-HCC epoch: frozen prediction plus the HCC relation-target objective."""
        from representation.sprint18_checkpoint import (
            coefficient_for_epoch,
            learning_rate_for_epoch,
        )
        from representation.sprint18_harness import (
            TASK9_COEFFICIENT_PILOT_EPOCHS,
            validate_arm_config,
        )

        validate_arm_config(arm, require_locked=(arm.selection_state == "locked"))
        if arm.arm_id != "N1-HCC":
            raise ValueError("HCC training requires the N1-HCC arm configuration")
        if arm.selection_state == "coefficient_pilot" and (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= TASK9_COEFFICIENT_PILOT_EPOCHS
        ):
            raise ValueError("N1 coefficient pilots must stop at the frozen epoch-30 full-ramp checkpoint")
        if (
            arm.selection_state == "coefficient_pilot"
            and arm.lambda_hcc_max != 0.0
            and not collect_balance
        ):
            raise ValueError("N1 coefficient pilots require objective-balance diagnostics")
        if self.scheduler is not None:
            raise ValueError("Sprint18 uses a constant-within-epoch LR; pass no step scheduler")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise ValueError("Sprint18 representation training requires AdamW")
        if self.max_grad_norm != 1.0:
            raise ValueError("Sprint18 representation training requires global L2 clip 1.0")
        if self.criterion.prediction_weight != 1.0:
            raise ValueError("Sprint18 prediction objective coefficient must remain 1.0")
        coefficient_max = arm.lambda_hcc_max
        if coefficient_max is None:
            raise ValueError("N1-HCC pilot requires its frozen HCC coefficient maximum")
        if (
            self.model.config.n_channels != arm.n_channels
            or self.model.config.patch_size != arm.patch_size
            or self.model.config.stride != arm.stride
            or self.model.config.n_robots != arm.n_robots
            or self.model.config.n_programs != arm.n_programs
            or self.model.config.seed != arm.model_seed
            or self.model.config.ema_decay != arm.ema_decay
            or self.model.target_encoder.decay != arm.ema_decay
        ):
            raise ValueError("N1 model configuration differs from its frozen Sprint18 arm")
        learning_rate = learning_rate_for_epoch(epoch)
        effective_hcc = coefficient_for_epoch(coefficient_max, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return self._run_hcc_epoch(
            batches,
            epoch=epoch,
            effective_hcc=effective_hcc,
            learning_rate=learning_rate,
            collect_balance=collect_balance,
            sprint18_view_epoch=epoch,
        )

    def _run_hcc_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        epoch: int,
        effective_hcc: float,
        learning_rate: float,
        collect_balance: bool,
        sprint18_view_epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        epoch_balance_records: list[dict[str, int | float | bool]] = []
        shared_parameters = self._shared_encoder_parameters() if collect_balance else ()
        for batch_index, raw_batch in enumerate(batches):
            next_step = self.step + 1
            self.optimizer.zero_grad(set_to_none=True)
            moved = self._move_batch(raw_batch)  # type: ignore[arg-type]
            output = self.model(moved, sprint18_view_epoch=sprint18_view_epoch)
            terms = self.criterion(output, step=self.step, effective_lambda_override=0.0)
            hcc_terms = self.hcc_objective_terms(output, moved, temperature=self.model.config.contrastive_temperature)
            hcc_loss = hcc_terms["loss"]
            if not isinstance(hcc_loss, torch.Tensor):
                raise TypeError("HCC loss must be a tensor")
            weighted_hcc_loss = effective_hcc * hcc_loss
            joint_loss = terms["weighted_prediction_loss"] + weighted_hcc_loss
            if not isinstance(joint_loss, torch.Tensor):
                raise TypeError("joint_loss must be a tensor")
            pending_record: dict[str, int | float | bool] | None = None
            if collect_balance and (
                batch_index == 0
                or (26 <= epoch <= 30 and next_step % 40 == 0)
            ):
                pending_record = self._hcc_balance_record(
                    output,
                    moved,
                    terms,
                    hcc_loss,
                    weighted_hcc_loss,
                    effective_hcc,
                    epoch=epoch,
                    global_step=next_step,
                    learning_rate=learning_rate,
                    parameters=shared_parameters,
                )
            joint_loss.backward()
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(parameters, self.max_grad_norm)
            else:
                grad_norm = clip_grad_norm_(parameters, float("inf"))
            self.optimizer.step()
            self.model.target_encoder.update(self.model.context_encoder)
            self.step += 1
            if pending_record is not None:
                self.objective_balance_history.append(pending_record)
                epoch_balance_records.append(pending_record)
            for name, value in (
                ("prediction_loss", terms["prediction_loss"]),
                ("contrastive_loss", hcc_loss),
                ("weighted_contrastive_loss", weighted_hcc_loss),
                ("joint_loss", joint_loss),
            ):
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
        averaged["epoch"] = float(epoch)
        if epoch_balance_records:
            for name, value in epoch_balance_records[0].items():
                averaged[f"balance_{name}"] = float(value)
        averaged["step"] = float(self.step)
        self.history.append(averaged)
        return averaged

    def train_sprint18_msp_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        arm: Sprint18ArmConfig,
        epoch: int,
        collect_balance: bool = True,
        fit_role_sha256: str | None = None,
    ) -> dict[str, float]:
        """Run one N2-MSP epoch: frozen fine prediction + coarse + active InfoNCE."""
        from representation.sprint18_checkpoint import (
            coefficient_for_epoch,
            learning_rate_for_epoch,
        )
        from representation.sprint18_harness import (
            TASK9_COEFFICIENT_PILOT_EPOCHS,
            validate_arm_config,
        )

        validate_arm_config(arm, require_locked=(arm.selection_state == "locked"))
        if arm.arm_id != "N2-MSP":
            raise ValueError("MSP training requires the N2-MSP arm configuration")
        if arm.selection_state == "coefficient_pilot" and (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= TASK9_COEFFICIENT_PILOT_EPOCHS
        ):
            raise ValueError("N2 coefficient pilots must stop at the frozen epoch-30 full-ramp checkpoint")
        if arm.alpha is None:
            raise ValueError("N2-MSP pilot requires its frozen alpha coefficient maximum")
        if arm.lambda_info_max is None:
            raise ValueError("N2-MSP pilot requires the frozen common InfoNCE coefficient")
        if self.scheduler is not None:
            raise ValueError("Sprint18 uses a constant-within-epoch LR; pass no step scheduler")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise ValueError("Sprint18 representation training requires AdamW")
        if self.max_grad_norm != 1.0:
            raise ValueError("Sprint18 representation training requires global L2 clip 1.0")
        if self.criterion.prediction_weight != 1.0:
            raise ValueError("Sprint18 prediction objective coefficient must remain 1.0")
        if (
            self.model.config.n_channels != arm.n_channels
            or self.model.config.patch_size != arm.patch_size
            or self.model.config.stride != arm.stride
            or self.model.config.n_robots != arm.n_robots
            or self.model.config.n_programs != arm.n_programs
            or self.model.config.seed != arm.model_seed
            or self.model.config.ema_decay != arm.ema_decay
            or self.model.target_encoder.decay != arm.ema_decay
        ):
            raise ValueError("N2 model configuration differs from its frozen Sprint18 arm")
        if (
            self.criterion.lambda_schedule.lambda_max != arm.lambda_info_max
            or self.model.config.contrastive_weight_max != arm.lambda_info_max
        ):
            raise ValueError("N2 InfoNCE coefficient differs from the arm configuration")
        learning_rate = learning_rate_for_epoch(epoch)
        effective_alpha = coefficient_for_epoch(arm.alpha, epoch)
        effective_info = coefficient_for_epoch(arm.lambda_info_max, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return self._run_msp_epoch(
            batches,
            epoch=epoch,
            effective_alpha=effective_alpha,
            effective_info=effective_info,
            learning_rate=learning_rate,
            collect_balance=collect_balance,
            sprint18_view_epoch=epoch,
            fit_role_sha256=fit_role_sha256,
            model_seed=arm.model_seed,
        )

    def _run_msp_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        epoch: int,
        effective_alpha: float,
        effective_info: float,
        learning_rate: float,
        collect_balance: bool,
        sprint18_view_epoch: int,
        fit_role_sha256: str | None,
        model_seed: int,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        epoch_balance_records: list[dict[str, int | float | bool]] = []
        shared_parameters = self._shared_encoder_parameters() if collect_balance else ()
        for batch_index, raw_batch in enumerate(batches):
            next_step = self.step + 1
            self.optimizer.zero_grad(set_to_none=True)
            moved = self._move_batch(raw_batch)  # type: ignore[arg-type]
            masks: dict[str, object] | None = None
            if fit_role_sha256 is not None:
                masks = self.msp_batch_masks(
                    moved, model_seed=model_seed, epoch=epoch, fit_role_sha256=fit_role_sha256
                )
                union = masks["union_mask"]
                assert isinstance(union, torch.Tensor)
                moved = dict(moved)
                # The N2 union is built on CPU while the collated batch lives on
                # the trainer device; align it before the model call. On CPU this
                # is a no-op, so local numerics are bitwise unchanged.
                moved["mask"] = union.to(dtype=torch.bool, device=moved["mask"].device)
            output = self.model(moved, sprint18_view_epoch=sprint18_view_epoch)
            terms = self.criterion(output, step=self.step, effective_lambda_override=effective_info)
            context = output.get("context_latents")
            target = output.get("target_latents")
            valid = output.get("patch_valid_mask")
            if (
                not isinstance(context, torch.Tensor)
                or not isinstance(target, torch.Tensor)
                or not isinstance(valid, torch.Tensor)
            ):
                raise TypeError("N2 coarse loss requires context/target/valid tensors")
            if masks is None:
                groups_4: list[list[tuple[int, ...]]] = [[] for _ in range(context.shape[0])]
                groups_8: list[list[tuple[int, ...]]] = [[] for _ in range(context.shape[0])]
            else:
                groups_4 = masks["groups_4"]  # type: ignore[assignment]
                groups_8 = masks["groups_8"]  # type: ignore[assignment]
            # Per-file groups address global patch indices; the batch shares one
            # N dimension, so flatten per-file groups into batch-global lists with
            # row offsets. Here rows share geometry only in tests; pilots pass
            # per-file batches via collation. Use row 0 groups for the single-row
            # path and require explicit per-row handling otherwise.
            flat_4: list[tuple[int, ...]] = []
            flat_8: list[tuple[int, ...]] = []
            for row_groups in groups_4:
                flat_4.extend(tuple(int(m) for m in g) for g in row_groups)
            for row_groups in groups_8:
                flat_8.extend(tuple(int(m) for m in g) for g in row_groups)
            coarse_terms = self.model.msp_coarse_terms(context, target, valid, flat_4, flat_8)
            coarse_loss = coarse_terms["coarse_loss"]
            if not isinstance(coarse_loss, torch.Tensor):
                raise TypeError("coarse loss must be a tensor")
            weighted_coarse_loss = effective_alpha * coarse_loss
            joint_loss = terms["weighted_prediction_loss"] + weighted_coarse_loss + terms["weighted_contrastive_loss"]
            if not isinstance(joint_loss, torch.Tensor):
                raise TypeError("joint_loss must be a tensor")
            pending_record: dict[str, int | float | bool] | None = None
            if collect_balance and (
                batch_index == 0
                or (26 <= epoch <= 30 and next_step % 40 == 0)
            ):
                pending_record = self._msp_balance_record(
                    output,
                    terms,
                    coarse_loss,
                    weighted_coarse_loss,
                    effective_alpha,
                    effective_info,
                    epoch=epoch,
                    global_step=next_step,
                    learning_rate=learning_rate,
                    parameters=shared_parameters,
                    groups_4_count=len(flat_4),
                    groups_8_count=len(flat_8),
                )
            joint_loss.backward()
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(parameters, self.max_grad_norm)
            else:
                grad_norm = clip_grad_norm_(parameters, float("inf"))
            self.optimizer.step()
            self.model.target_encoder.update(self.model.context_encoder)
            self.step += 1
            if pending_record is not None:
                self.objective_balance_history.append(pending_record)
                epoch_balance_records.append(pending_record)
            for name, value in (
                ("prediction_loss", terms["prediction_loss"]),
                ("contrastive_loss", terms["contrastive_loss"]),
                ("weighted_contrastive_loss", terms["weighted_contrastive_loss"]),
                ("coarse_loss", coarse_loss),
                ("weighted_coarse_loss", weighted_coarse_loss),
                ("joint_loss", joint_loss),
            ):
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
        averaged["epoch"] = float(epoch)
        if epoch_balance_records:
            for name, value in epoch_balance_records[0].items():
                averaged[f"balance_{name}"] = float(value)
        averaged["step"] = float(self.step)
        self.history.append(averaged)
        return averaged

    @staticmethod
    def ccrp_batch_selections(
        batch: Mapping[str, object],
        *,
        model_seed: int,
        epoch: int,
        fit_role_sha256: str,
    ) -> dict[str, object]:
        """Select frozen N3 relational targets for one collated batch.

        Per file: ``V`` is the valid non-padding patch indices, ``Q`` follows
        ``ccrp_patch_count``, and one seeded permutation picks the initial
        sample. Fully temporally masked initials (current B0 ``mask`` over
        valid patches) are resampled deterministically in initial-sample
        order. Consumes no labels/outcomes.
        """
        import numpy as np

        file_ids = batch.get("file_ids")
        valid_mask = batch.get("patch_valid_mask")
        temporal_mask = batch.get("mask")
        if not isinstance(file_ids, (list, tuple)) or not file_ids:
            raise ValueError("N3 selection requires batch file IDs")
        if (
            not isinstance(valid_mask, torch.Tensor)
            or not isinstance(temporal_mask, torch.Tensor)
            or valid_mask.dtype is not torch.bool
            or temporal_mask.dtype is not torch.bool
        ):
            raise ValueError("N3 selection requires bool patch_valid_mask/mask")
        if tuple(valid_mask.shape) != tuple(temporal_mask.shape):
            raise ValueError("N3 valid and temporal masks must share shape")
        batch_size, max_patches = valid_mask.shape
        if len(file_ids) != batch_size:
            raise ValueError("N3 selection file IDs must match batch size")
        selected_all: list[list[int]] = []
        diagnostics: list[dict[str, object]] = []
        for row, file_id in enumerate(file_ids):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("N3 selection requires canonical file IDs")
            valid_positions = [int(i) for i in range(max_patches) if bool(valid_mask[row, i].item())]
            fully_masked = [
                int(i)
                for i in valid_positions
                if bool(temporal_mask[row, i].item())
            ]
            n_valid = len(valid_positions)
            requested = ccrp_patch_count(n_valid)
            seed = ccrp_patch_seed(model_seed, epoch, fit_role_sha256, file_id)
            rng = np.random.default_rng(seed)
            sampled = ccrp_sample_targets(valid_positions, requested, rng, fully_masked)
            selected_all.append([int(v) for v in sampled["selected"]])  # type: ignore[union-attr]
            diagnostics.append(
                {
                    "file_id": file_id,
                    "n": n_valid,
                    "Q": requested,
                    "initial": len(sampled["initial"]),  # type: ignore[arg-type]
                    "rejected": len(sampled["rejected"]),  # type: ignore[arg-type]
                    "replacements": len(sampled["replacements"]),  # type: ignore[arg-type]
                    "final": len(sampled["selected"]),  # type: ignore[arg-type]
                    "unsupported": requested - len(sampled["selected"]),  # type: ignore[operator]
                }
            )
        return {"selected": selected_all, "diagnostics": diagnostics}

    def _ccrp_balance_record(
        self,
        output: Mapping[str, object],
        terms: Mapping[str, torch.Tensor | float | int],
        channel_loss: torch.Tensor,
        weighted_channel_loss: torch.Tensor,
        effective_beta: float,
        effective_info: float,
        *,
        epoch: int,
        global_step: int,
        learning_rate: float,
        parameters: tuple[nn.Parameter, ...],
        channel_targets: int,
        channel_excluded: int,
        channel_counts: tuple[int, ...],
    ) -> dict[str, int | float | bool]:
        prediction_loss = terms["prediction_loss"]
        if not isinstance(prediction_loss, torch.Tensor) or not isinstance(weighted_channel_loss, torch.Tensor):
            raise TypeError("objective losses must be tensors for gradient diagnostics")
        prediction_norm = self._gradient_norm(prediction_loss, parameters)
        channel_norm = self._gradient_norm(weighted_channel_loss, parameters)
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
        geometry = self._embedding_geometry(embeddings, valid_mask.any(dim=1))
        effective_prediction = self._scalar(terms["effective_prediction_coefficient"])
        raw_channel = float(channel_loss.detach().cpu().item())
        raw_prediction = self._scalar(prediction_loss)
        ratio = channel_norm / (prediction_norm + 1e-12)
        record: dict[str, int | float | bool] = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": learning_rate,
            "raw_L_pred": raw_prediction,
            "effective_coefficient_L_pred": effective_prediction,
            "weighted_L_pred": self._scalar(terms["weighted_prediction_loss"]),
            "raw_L_channel": raw_channel,
            "effective_coefficient_L_channel": effective_beta,
            "weighted_L_channel": float(weighted_channel_loss.detach().cpu().item()),
            "raw_L_info": self._scalar(terms["contrastive_loss"]),
            "effective_coefficient_L_info": effective_info,
            "weighted_L_info": self._scalar(terms["weighted_contrastive_loss"]),
            "gradient_norm_L_pred_shared": prediction_norm,
            "gradient_norm_weighted_L_channel_shared": channel_norm,
            "r_aux": ratio,
            "support_valid_files": int(valid_mask.any(dim=1).sum().item()),
            "support_valid_patches": int(valid_mask.sum().item()),
            "support_prediction_files": int(valid_prediction.any(dim=1).sum().item()),
            "support_prediction_patches": int(valid_prediction.sum().item()),
            "support_channel_targets": int(channel_targets),
            "support_channel_excluded_near_zero": int(channel_excluded),
            "contrastive_input_files": int(terms["contrastive_input_files"]),
            "protocol_gradient_observation": (26 <= epoch <= 30 and global_step % 40 == 0),
        }
        for index, count in enumerate(channel_counts):
            record[f"support_channel_c{index}"] = int(count)
        record.update(geometry)
        return record

    def train_sprint18_ccrp_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        arm: Sprint18ArmConfig,
        epoch: int,
        collect_balance: bool = True,
        fit_role_sha256: str | None = None,
    ) -> dict[str, float]:
        """Run one N3-CCRP epoch: frozen fine prediction + channel + active InfoNCE."""
        from representation.sprint18_checkpoint import (
            coefficient_for_epoch,
            learning_rate_for_epoch,
        )
        from representation.sprint18_harness import (
            TASK9_COEFFICIENT_PILOT_EPOCHS,
            validate_arm_config,
        )

        validate_arm_config(arm, require_locked=(arm.selection_state == "locked"))
        if arm.arm_id != "N3-CCRP":
            raise ValueError("CCRP training requires the N3-CCRP arm configuration")
        if arm.selection_state == "coefficient_pilot" and (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= TASK9_COEFFICIENT_PILOT_EPOCHS
        ):
            raise ValueError("N3 coefficient pilots must stop at the frozen epoch-30 full-ramp checkpoint")
        if arm.beta is None:
            raise ValueError("N3-CCRP pilot requires its frozen beta coefficient maximum")
        if arm.lambda_info_max is None:
            raise ValueError("N3-CCRP pilot requires the frozen common InfoNCE coefficient")
        if self.scheduler is not None:
            raise ValueError("Sprint18 uses a constant-within-epoch LR; pass no step scheduler")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise ValueError("Sprint18 representation training requires AdamW")
        if self.max_grad_norm != 1.0:
            raise ValueError("Sprint18 representation training requires global L2 clip 1.0")
        if self.criterion.prediction_weight != 1.0:
            raise ValueError("Sprint18 prediction objective coefficient must remain 1.0")
        if (
            self.model.config.n_channels != arm.n_channels
            or self.model.config.patch_size != arm.patch_size
            or self.model.config.stride != arm.stride
            or self.model.config.n_robots != arm.n_robots
            or self.model.config.n_programs != arm.n_programs
            or self.model.config.seed != arm.model_seed
            or self.model.config.ema_decay != arm.ema_decay
            or self.model.target_encoder.decay != arm.ema_decay
        ):
            raise ValueError("N3 model configuration differs from its frozen Sprint18 arm")
        if (
            self.criterion.lambda_schedule.lambda_max != arm.lambda_info_max
            or self.model.config.contrastive_weight_max != arm.lambda_info_max
        ):
            raise ValueError("N3 InfoNCE coefficient differs from the arm configuration")
        learning_rate = learning_rate_for_epoch(epoch)
        effective_beta = coefficient_for_epoch(arm.beta, epoch)
        effective_info = coefficient_for_epoch(arm.lambda_info_max, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return self._run_ccrp_epoch(
            batches,
            epoch=epoch,
            effective_beta=effective_beta,
            effective_info=effective_info,
            learning_rate=learning_rate,
            collect_balance=collect_balance,
            sprint18_view_epoch=epoch,
            fit_role_sha256=fit_role_sha256,
            model_seed=arm.model_seed,
        )

    def _run_ccrp_epoch(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        epoch: int,
        effective_beta: float,
        effective_info: float,
        learning_rate: float,
        collect_balance: bool,
        sprint18_view_epoch: int,
        fit_role_sha256: str | None,
        model_seed: int,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        epoch_balance_records: list[dict[str, int | float | bool]] = []
        shared_parameters = self._shared_encoder_parameters() if collect_balance else ()
        materialized = list(batches)
        if not materialized:
            raise ValueError("training epoch received no batches")
        epoch_keys: list[tuple[str, int]] = []
        per_batch_selected: list[list[list[int]]] = []
        for raw_batch in materialized:
            moved_probe = self._move_batch(raw_batch)  # type: ignore[arg-type]
            if fit_role_sha256 is None:
                per_batch_selected.append([])
                continue
            selections = self.ccrp_batch_selections(
                moved_probe, model_seed=model_seed, epoch=epoch, fit_role_sha256=fit_role_sha256
            )
            file_ids = moved_probe.get("file_ids")
            assert isinstance(file_ids, (list, tuple))
            selected = selections["selected"]
            assert isinstance(selected, list)
            per_batch_selected.append(selected)  # type: ignore[arg-type]
            for row, file_id in enumerate(file_ids):
                assert isinstance(file_id, str)
                for patch in selected[row]:  # type: ignore[index]
                    epoch_keys.append((file_id, int(patch)))
        epoch_keys = sorted(epoch_keys, key=lambda item: (item[0], item[1]))
        assignment = ccrp_assign_channels(epoch_keys, model_seed, epoch)
        assigned_channels = assignment["channels"]
        assert isinstance(assigned_channels, list)
        channel_counts = assignment["counts"]
        assert isinstance(channel_counts, tuple)
        channel_by_key = {key: int(channel) for key, channel in zip(epoch_keys, assigned_channels)}
        for batch_index, raw_batch in enumerate(materialized):
            next_step = self.step + 1
            self.optimizer.zero_grad(set_to_none=True)
            moved = self._move_batch(raw_batch)  # type: ignore[arg-type]
            output = self.model(moved, sprint18_view_epoch=sprint18_view_epoch)
            terms = self.criterion(output, step=self.step, effective_lambda_override=effective_info)
            norm = output.get("normalization")
            means = norm.get("mean") if isinstance(norm, dict) else None
            stds = norm.get("std") if isinstance(norm, dict) else None
            raw_patches = moved.get("patches")
            pad_mask = moved.get("patch_pad_mask")
            valid_mask = moved.get("patch_valid_mask")
            temporal_mask = moved.get("mask")
            if (
                not isinstance(raw_patches, torch.Tensor)
                or not isinstance(pad_mask, torch.Tensor)
                or not isinstance(valid_mask, torch.Tensor)
                or not isinstance(temporal_mask, torch.Tensor)
                or raw_patches.ndim != 4
            ):
                raise TypeError("N3 channel forward requires collated patch tensors")
            if means is None or stds is None:
                raise ValueError("N3 channel forward requires the B0 normalization result")
            normalized_full = (raw_patches - means.unsqueeze(1).unsqueeze(-1)) / stds.unsqueeze(1).unsqueeze(-1)
            normalized_full = normalized_full.masked_fill(pad_mask.unsqueeze(2), 0.0)
            aux_patches = normalized_full.clone()
            targets_row: list[int] = []
            targets_patch: list[int] = []
            targets_channel: list[int] = []
            file_ids = moved.get("file_ids")
            if not isinstance(file_ids, (list, tuple)):
                raise ValueError("N3 channel forward requires batch file IDs")
            selected = per_batch_selected[batch_index]
            if fit_role_sha256 is not None:
                for row, file_id in enumerate(file_ids):
                    if not isinstance(file_id, str) or not file_id:
                        raise ValueError("N3 channel forward requires canonical file IDs")
                    for patch in (selected[row] if selected else []):
                        channel = channel_by_key[(file_id, int(patch))]
                        aux_patches[row, int(patch), channel, :] = 0.0
                        targets_row.append(row)
                        targets_patch.append(int(patch))
                        targets_channel.append(channel)
            aux_patches = aux_patches.masked_fill(pad_mask.unsqueeze(2), 0.0)
            channel_loss: torch.Tensor
            excluded_near_zero = 0
            target_count = len(targets_row)
            if target_count:
                full_local_full, zeroed_local_full = self.model.ccrp_aux_local(
                    normalized_full, aux_patches, pad_mask
                )
                full_rows = torch.stack(
                    [full_local_full[row, patch] for row, patch in zip(targets_row, targets_patch)]
                )
                zeroed_rows = torch.stack(
                    [zeroed_local_full[row, patch] for row, patch in zip(targets_row, targets_patch)]
                )
                local_full = full_rows.detach()
                local_without = zeroed_rows.detach()
                aux_visible = self.model.patch_encoder(aux_patches, pad_mask)
                aux_visible = aux_visible.masked_fill(temporal_mask.unsqueeze(-1), 0.0)
                context_aux_full = self.model.context_encoder(aux_visible, valid_mask)
                context_aux = torch.stack(
                    [context_aux_full[row, patch] for row, patch in zip(targets_row, targets_patch)]
                )
                channel_ids = torch.tensor(targets_channel, dtype=torch.long, device=context_aux.device)
                channel_terms = self.model.ccrp_channel_terms(context_aux, local_full, local_without, channel_ids)
                channel_loss = channel_terms["channel_loss"]
                if not isinstance(channel_loss, torch.Tensor):
                    raise TypeError("channel loss must be a tensor")
                excluded_near_zero = int(channel_terms["support_channel_excluded_near_zero"].item())
            else:
                channel_loss = (terms["weighted_prediction_loss"] * 0.0) + (output["context_latents"].sum() * 0.0)
            weighted_channel_loss = effective_beta * channel_loss
            joint_loss = terms["weighted_prediction_loss"] + weighted_channel_loss + terms["weighted_contrastive_loss"]
            if not isinstance(joint_loss, torch.Tensor):
                raise TypeError("joint_loss must be a tensor")
            pending_record: dict[str, int | float | bool] | None = None
            if collect_balance and (
                batch_index == 0
                or (26 <= epoch <= 30 and next_step % 40 == 0)
            ):
                pending_record = self._ccrp_balance_record(
                    output,
                    terms,
                    channel_loss,
                    weighted_channel_loss,
                    effective_beta,
                    effective_info,
                    epoch=epoch,
                    global_step=next_step,
                    learning_rate=learning_rate,
                    parameters=shared_parameters,
                    # Channel-support fields record the epoch's final target count and
                    # epoch-global round-robin counts (normative v2 S2 recording rule);
                    # the selector audits coherence and round-robin balance on every
                    # observation row. The near-zero exclusion stays the actual batch
                    # observation backing this row's loss/gradient diagnostics.
                    channel_targets=len(epoch_keys),
                    channel_excluded=excluded_near_zero,
                    channel_counts=tuple(int(v) for v in channel_counts),  # type: ignore[arg-type]
                )
            joint_loss.backward()
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(parameters, self.max_grad_norm)
            else:
                grad_norm = clip_grad_norm_(parameters, float("inf"))
            self.optimizer.step()
            self.model.target_encoder.update(self.model.context_encoder)
            self.step += 1
            if pending_record is not None:
                self.objective_balance_history.append(pending_record)
                epoch_balance_records.append(pending_record)
            for name, value in (
                ("prediction_loss", terms["prediction_loss"]),
                ("contrastive_loss", terms["contrastive_loss"]),
                ("weighted_contrastive_loss", terms["weighted_contrastive_loss"]),
                ("channel_loss", channel_loss),
                ("weighted_channel_loss", weighted_channel_loss),
                ("joint_loss", joint_loss),
            ):
                scalar = self._scalar(value)
                totals[name] = totals.get(name, 0.0) + scalar
                counts[name] = counts.get(name, 0) + 1
            grad_norm_scalar = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            lr_scalar = float(self.optimizer.param_groups[0]["lr"])
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + grad_norm_scalar
            counts["grad_norm"] = counts.get("grad_norm", 0) + 1
            totals["lr"] = totals.get("lr", 0.0) + lr_scalar
            counts["lr"] = counts.get("lr", 0) + 1
        averaged = {name: totals[name] / counts[name] for name in totals}
        averaged["epoch"] = float(epoch)
        if epoch_balance_records:
            for name, value in epoch_balance_records[0].items():
                averaged[f"balance_{name}"] = float(value)
        averaged["step"] = float(self.step)
        self.history.append(averaged)
        return averaged

    def _run_epoch(
        self,
        batches: Iterable[RepresentationBatch],
        *,
        epoch: int | None = None,
        effective_lambda: float | None = None,
        learning_rate: float | None = None,
        collect_balance: bool = False,
        sprint18_view_epoch: int | None = None,
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
                sprint18_view_epoch=sprint18_view_epoch,
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
