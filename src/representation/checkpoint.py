"""Atomic V1 model, optimizer, EMA, and reference-bank checkpoints."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from representation.config import V1Config
from representation.inference import NormalReferenceBank
from representation.model import V1RepresentationModel

SCHEMA_VERSION = 1
_REQUIRED_KEYS = {"schema_version", "config", "model_state", "step"}


def save_checkpoint(
    path: str | Path,
    model: V1RepresentationModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: V1Config | Mapping[str, object] | None = None,
    step: int = 0,
    reference_bank: NormalReferenceBank | None = None,
) -> None:
    """Atomically save all train-state needed to resume a V1 run."""
    if step < 0:
        raise ValueError("step must be non-negative")
    resolved_config: Mapping[str, object]
    if config is None:
        resolved_config = model.config.to_dict()
    elif isinstance(config, V1Config):
        resolved_config = config.to_dict()
    else:
        resolved_config = dict(config)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(resolved_config),
        "model_state": model.state_dict(),
        "step": int(step),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if reference_bank is not None:
        payload["reference_bank"] = {
            "k": reference_bank.k,
            "embeddings": None if reference_bank.embeddings is None else reference_bank.embeddings.clone(),
        }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_checkpoint(
    path: str | Path,
    model: V1RepresentationModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_config: V1Config | Mapping[str, object] | None = None,
    reference_bank: NormalReferenceBank | None = None,
) -> dict[str, object]:
    """Validate and restore a checkpoint, rejecting incompatible metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not _REQUIRED_KEYS.issubset(payload):
        raise ValueError(f"checkpoint missing required keys: {_REQUIRED_KEYS}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("incompatible checkpoint schema_version")
    saved_config = payload["config"]
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config must be a mapping")
    current_config = model.config.to_dict()
    if dict(saved_config) != current_config:
        raise ValueError("checkpoint configuration is incompatible with model")
    if expected_config is not None:
        expected = expected_config.to_dict() if isinstance(expected_config, V1Config) else dict(expected_config)
        if dict(saved_config) != expected:
            raise ValueError("checkpoint configuration does not match expected_config")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("checkpoint missing optimizer_state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if "scheduler_state" not in payload:
            raise ValueError("checkpoint missing scheduler_state")
        scheduler.load_state_dict(payload["scheduler_state"])
    bank_payload = payload.get("reference_bank")
    if reference_bank is not None and bank_payload is not None:
        if not isinstance(bank_payload, Mapping):
            raise ValueError("checkpoint reference_bank must be a mapping")
        embeddings = bank_payload.get("embeddings")
        if embeddings is not None and not isinstance(embeddings, torch.Tensor):
            raise ValueError("checkpoint reference embeddings must be a tensor")
        reference_bank.k = int(bank_payload["k"])
        reference_bank.embeddings = None if embeddings is None else embeddings.detach().cpu().clone()
    return {
        "config": dict(saved_config),
        "step": payload["step"],
        "has_optimizer": "optimizer_state" in payload,
        "has_scheduler": "scheduler_state" in payload,
        "has_reference_bank": bank_payload is not None,
    }


class CheckpointManager:
    """Small stateful wrapper around atomic checkpoint save/load functions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, model: V1RepresentationModel, **kwargs: Any) -> None:
        save_checkpoint(self.path, model, **kwargs)

    def load(self, model: V1RepresentationModel, **kwargs: Any) -> dict[str, object]:
        return load_checkpoint(self.path, model, **kwargs)
