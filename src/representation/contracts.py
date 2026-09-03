"""Runtime-checked tensor contracts for variable-length V1 batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

import torch

from synth.schema import FileSample, SampleLabel


class RepresentationBatch(TypedDict):
    """Padded minibatch made from complete files.

    Shapes use ``B`` for files, ``C`` for channels, ``T`` for the batch's
    maximum timestep count, ``N`` for its maximum patch count, and ``W`` for
    patch size. Padding is represented by masks, never by a semantic length.
    """

    signals: torch.Tensor  # [B, C, T]
    file_valid_mask: torch.Tensor  # bool [B, T], True for real timesteps
    patches: torch.Tensor  # [B, N, C, W]
    patch_valid_mask: torch.Tensor  # bool [B, N], True for real patches
    patch_pad_mask: torch.Tensor  # bool [B, N, W], True for padded steps
    starts: torch.Tensor  # int64 [B, N]
    valid_len: torch.Tensor  # int64 [B, N]
    mask: torch.Tensor  # bool [B, N], True for selected prediction targets
    file_ids: Sequence[str]
    file_labels: NotRequired[Sequence[SampleLabel]]
    anomaly_meta: NotRequired[Sequence[object | None]]
    anomaly_masks: NotRequired[Sequence[object | None]]
    file_samples: NotRequired[Sequence[FileSample]]
    mask_composition: NotRequired[Sequence[Mapping[str, int]]]
    mask_ratio: NotRequired[torch.Tensor]  # float [B]


class RepresentationOutput(TypedDict):
    """Outputs consumed by V1 criteria and inference."""

    context_latents: torch.Tensor  # [B, N, D]
    target_latents: torch.Tensor  # [B, N, D], stop-gradient
    predicted_latents: torch.Tensor  # [B, N, D]
    prediction_mask: torch.Tensor  # bool [B, N]
    file_embedding: torch.Tensor  # [B, D]
    view_embedding_1: NotRequired[torch.Tensor]  # [B, D]
    view_embedding_2: NotRequired[torch.Tensor]  # [B, D]
    patch_prediction_error: NotRequired[torch.Tensor]  # [B, N]


def _require_tensor(batch: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    return value


def _require_bool(value: torch.Tensor, name: str) -> None:
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must have torch.bool dtype")


def validate_batch(batch: Mapping[str, object]) -> None:
    """Validate shape, dtype, and mask invariants for a model input batch."""
    signals = _require_tensor(batch, "signals", 3)
    file_mask = _require_tensor(batch, "file_valid_mask", 2)
    patches = _require_tensor(batch, "patches", 4)
    patch_valid = _require_tensor(batch, "patch_valid_mask", 2)
    pad_mask = _require_tensor(batch, "patch_pad_mask", 3)
    starts = _require_tensor(batch, "starts", 2)
    valid_len = _require_tensor(batch, "valid_len", 2)
    mask = _require_tensor(batch, "mask", 2)

    b, c, t = signals.shape
    bp, n, cp, w = patches.shape
    if (bp, cp) != (b, c):
        raise ValueError("patches must have matching batch and channel dimensions")
    expected = {
        "file_valid_mask": (b, t),
        "patch_valid_mask": (b, n),
        "patch_pad_mask": (b, n, w),
        "starts": (b, n),
        "valid_len": (b, n),
        "mask": (b, n),
    }
    actual = {
        "file_valid_mask": tuple(file_mask.shape),
        "patch_valid_mask": tuple(patch_valid.shape),
        "patch_pad_mask": tuple(pad_mask.shape),
        "starts": tuple(starts.shape),
        "valid_len": tuple(valid_len.shape),
        "mask": tuple(mask.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} shape {actual[name]} != expected {shape}")

    for name, value in (
        ("file_valid_mask", file_mask),
        ("patch_valid_mask", patch_valid),
        ("patch_pad_mask", pad_mask),
        ("mask", mask),
    ):
        _require_bool(value, name)
    if starts.dtype != torch.int64 or valid_len.dtype != torch.int64:
        raise ValueError("starts and valid_len must have torch.int64 dtype")
    if len(batch.get("file_ids", ())) != b:
        raise ValueError("file_ids length must match the batch size")
    if torch.any(mask & ~patch_valid):
        raise ValueError("masked patches must be valid patches")
    if torch.any(valid_len < 0) or torch.any(valid_len > w):
        raise ValueError("valid_len must be within [0, patch_size]")
    if torch.any(pad_mask[:, :, :].sum(dim=-1) != (w - valid_len)):
        raise ValueError("patch_pad_mask must agree with valid_len")


def validate_output(output: Mapping[str, object]) -> None:
    """Validate the common output shape and target stop-gradient invariants."""
    context = _require_tensor(output, "context_latents", 3)
    target = _require_tensor(output, "target_latents", 3)
    predicted = _require_tensor(output, "predicted_latents", 3)
    prediction_mask = _require_tensor(output, "prediction_mask", 2)
    file_embedding = _require_tensor(output, "file_embedding", 2)

    if target.shape != context.shape or predicted.shape != context.shape:
        raise ValueError("all patch latent tensors must share [B, N, D] shape")
    if tuple(prediction_mask.shape) != tuple(context.shape[:2]):
        raise ValueError("prediction_mask must have shape [B, N]")
    if tuple(file_embedding.shape) != (context.shape[0], context.shape[2]):
        raise ValueError("file_embedding must have shape [B, D]")
    _require_bool(prediction_mask, "prediction_mask")
    if target.requires_grad:
        raise ValueError("target_latents must be stop-gradient tensors")
