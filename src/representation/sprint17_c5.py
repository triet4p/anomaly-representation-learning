"""Sprint 17 Task 12 — C5 objective/masking alternatives (C5-A / C5-B).

Frozen-registry implementation for the C5 (objective/masking) suspect only.
Total mask ratio (0.40), architecture, inputs, checkpoint rule, and
model-selection basis stay frozen; only the masking structure (C5-A) and the
learning-pressure terms (both arms) change:

- ``C5-A`` structured-predict: the 0.40 quota is selected as seeded
  contiguous channel-time blocks (full-channel width over contiguous
  patch-time spans, exact count, valid-only) instead of the B0
  random/info/block composition; the EMA latent-prediction loss becomes
  multi-horizon (shared head matches stop-gradient EMA targets at offsets
  0/+1/+2 where valid). No new parameters.
- ``C5-B`` redundancy-control: EMA masked prediction is retained unchanged;
  the file-level InfoNCE term is replaced by VICReg-style variance/covariance
  regularization on the same projected view embeddings, under the identical
  frozen warmup schedule. Because the warmup holds the file-level weight at
  exactly 0.0 through step 300, C5-B training is mathematically the
  prediction-only trajectory — a parity control, reported as such. No new
  parameters.

Downstream evaluation (masking for scoring uses the arm policy; S_pred
recomputes horizon-0 errors; S_pop, pooling, geometry, metrics) is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from representation.criterion import (
    FileContrastiveCriterion,
    JointRepresentationCriterion,
    LatentPredictionCriterion,
    ProgressiveLambda,
)

#: Prediction horizons matched against EMA targets (shared head, no new params).
C5A_HORIZONS: tuple[int, int, int] = (0, 1, 2)
#: Number of contiguous spans splitting the frozen mask quota.
C5A_N_BLOCKS = 2

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C5-A": ("seeded contiguous channel-time block masking at the frozen "
             "0.40 ratio plus shared-head multi-horizon (0/+1/+2) EMA latent "
             "prediction; no new parameters; scoring/inference code unchanged"),
    "C5-B": ("VICReg-style variance/covariance file-level term in place of "
             "InfoNCE under the identical frozen warmup (weight exactly 0.0 "
             "through step 300); EMA masked prediction retained; no new "
             "parameters"),
}

_C5_ARMS = ("C5-A", "C5-B")


def channel_time_block_mask(n_patches: int, valid_indices: np.ndarray,
                            target: int, rng: np.random.Generator,
                            n_blocks: int = C5A_N_BLOCKS,
                            ) -> tuple[np.ndarray, dict[str, int]]:
    """Select exactly ``target`` valid patches as contiguous time blocks.

    Blocks are contiguous spans in valid-order (patch index order is time
    order) at full channel width — i.e. channel-time blocks — placed at
    seeded non-overlapping offsets. Deterministic given ``rng``; valid-only;
    disjoint by construction.
    """
    masked = np.zeros(int(n_patches), dtype=bool)
    order = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
    need = int(min(max(int(target), 0), order.size))
    composition = {"time_block": 0}
    if need <= 0 or order.size == 0:
        return masked, composition
    quotas = [need // int(n_blocks)] * int(n_blocks)
    for i in range(need % int(n_blocks)):
        quotas[i] += 1
    quotas = [q for q in quotas if q > 0]
    occupied = np.zeros(order.size, dtype=bool)
    for length in quotas:
        placed = False
        for _ in range(64):
            start = int(rng.integers(0, order.size - length + 1))
            if not occupied[start:start + length].any():
                occupied[start:start + length] = True
                placed = True
                break
        if not placed:
            # Deterministic fallback: first free window (rejection only fails
            # when the order is nearly full, where a free window must exist
            # because the total quota never exceeds the order size).
            for start in range(order.size - length + 1):
                if not occupied[start:start + length].any():
                    occupied[start:start + length] = True
                    placed = True
                    break
        if not placed:
            raise ValueError("channel-time block placement failed")
    masked[order[occupied]] = True
    composition["time_block"] = int(masked.sum())
    return masked, composition


def c5a_block_policy(n_patches: int, valid_indices: np.ndarray, target: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    """Masking-bridge adapter for the C5-A channel-time block policy."""
    return channel_time_block_mask(int(n_patches), np.asarray(valid_indices),
                                   int(target), rng)


class MultiHorizonPredictionCriterion(nn.Module):
    """Match EMA targets at several horizons with one shared head.

    Output keys mirror :class:`LatentPredictionCriterion` so joint training,
    logging, and checkpoint contracts work unchanged. ``patch_prediction_error``
    carries horizon-0 errors for diagnostic compatibility.
    """

    def __init__(self, horizons: tuple[int, ...] = C5A_HORIZONS,
                 empty_policy: str = "zero") -> None:
        super().__init__()
        if not horizons or any(h < 0 for h in horizons):
            raise ValueError("horizons must be non-empty non-negative offsets")
        if empty_policy not in ("zero", "error"):
            raise ValueError("empty_policy must be 'zero' or 'error'")
        self.horizons = tuple(int(h) for h in horizons)
        self.empty_policy = empty_policy

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return mean-over-horizons masked prediction loss."""
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
        base = requested & valid
        batch, count, _ = predicted.shape
        horizon_losses = []
        base_errors = None
        for horizon in self.horizons:
            shifted_target = torch.zeros_like(target)
            if horizon < count:
                shifted_target[:, :count - horizon] = target[:, horizon:]
            shifted_valid = torch.zeros_like(valid)
            if horizon < count:
                shifted_valid[:, :count - horizon] = valid[:, horizon:]
            sel = base & shifted_valid
            errors = (predicted - shifted_target).square().mean(dim=-1).masked_fill(~sel, 0.0)
            if horizon == 0:
                base_errors = (predicted - target).square().mean(dim=-1).masked_fill(~base, 0.0)
            counts = sel.sum(dim=1)
            total = counts.sum()
            if total.item() == 0:
                if self.empty_policy == "error":
                    raise ValueError("no valid masked patches available for prediction loss")
                horizon_losses.append(predicted.sum() * 0.0)
            else:
                horizon_losses.append(errors.sum() / total.to(errors.dtype))
        loss = torch.stack(horizon_losses).mean()
        counts_per_file = base.sum(dim=1)
        masked_count = counts_per_file.sum()
        if base_errors is None:
            base_errors = torch.zeros_like(loss).expand(predicted.shape[:2])
        return {"loss": loss, "patch_prediction_error": base_errors,
                "masked_count": masked_count, "masked_counts": counts_per_file}

    @staticmethod
    def _tensor(model_output: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D torch.Tensor")
        return value


class FileRedundancyCriterion(nn.Module):
    """VICReg-style file-level term: invariance + variance + covariance.

    Consumes the same projected view embeddings as the B0 InfoNCE term and
    exposes the same ``loss`` key, so the joint criterion, schedule, logging,
    and checkpoint contracts work unchanged (the ``similarity`` diagnostics
    are InfoNCE-specific and therefore absent).
    """

    def __init__(self, invariance_weight: float = 1.0, variance_weight: float = 1.0,
                 covariance_weight: float = 1.0, variance_gamma: float = 1.0) -> None:
        super().__init__()
        for name, value in (("invariance_weight", invariance_weight),
                            ("variance_weight", variance_weight),
                            ("covariance_weight", covariance_weight),
                            ("variance_gamma", variance_gamma)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite non-negative")
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_gamma = float(variance_gamma)

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return the redundancy-regularized file-level loss."""
        view_1 = self._embedding(model_output, "view_embedding_1")
        view_2 = self._embedding(model_output, "view_embedding_2")
        if view_1.shape != view_2.shape:
            raise ValueError("redundancy views must share shape")
        if view_1.shape[0] == 0:
            raise ValueError("redundancy views cannot be empty")
        invariance = F.mse_loss(view_1, view_2)
        variance = 0.5 * (self._variance_hinge(view_1) + self._variance_hinge(view_2))
        covariance = 0.5 * (self._covariance(view_1) + self._covariance(view_2))
        loss = (self.invariance_weight * invariance + self.variance_weight * variance
                + self.covariance_weight * covariance)
        return {"loss": loss, "invariance": invariance, "variance": variance,
                "covariance": covariance}

    def _variance_hinge(self, views: torch.Tensor) -> torch.Tensor:
        std = views.std(dim=0, unbiased=False)
        return F.relu(self.variance_gamma - std).mean()

    @staticmethod
    def _covariance(views: torch.Tensor) -> torch.Tensor:
        batch, dim = views.shape
        centered = views - views.mean(dim=0, keepdim=True)
        if batch <= 1:
            return torch.zeros((), device=views.device, dtype=views.dtype)
        cov = (centered.transpose(0, 1) @ centered) / (batch - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag.square().sum() / max(dim * (dim - 1), 1))

    @staticmethod
    def _embedding(model_output: Mapping[str, object], name: str) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a 2D torch.Tensor")
        return value


def arm_criterion(arm_id: str, temperature: float = 0.2, prediction_weight: float = 1.0,
                  lambda_max: float = 0.1, ramp_steps: int = 980,
                  warmup_steps: int = 980) -> JointRepresentationCriterion:
    """Build the frozen joint criterion for a registered C5 arm."""
    schedule = ProgressiveLambda(lambda_max=lambda_max, ramp_steps=ramp_steps,
                                 warmup_steps=warmup_steps)
    if arm_id == "C5-A":
        return JointRepresentationCriterion(
            prediction=MultiHorizonPredictionCriterion(),
            contrastive=FileContrastiveCriterion(temperature=temperature),
            lambda_schedule=schedule, prediction_weight=prediction_weight)
    if arm_id == "C5-B":
        return JointRepresentationCriterion(
            prediction=LatentPredictionCriterion(),
            contrastive=FileRedundancyCriterion(),
            lambda_schedule=schedule, prediction_weight=prediction_weight)
    raise ValueError(f"unknown Sprint 17 C5 arm: {arm_id!r}")
