"""Torch batch bridge for the Sprint 1 fixed-ratio masking policy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from representation.config import V1Config
from representation.contracts import RepresentationBatch, validate_batch
from synth.config import MaskingConfig
from synth.masking import apply_masking
from synth.schema import PatchBatch


def _synth_masking_config(config: V1Config | MaskingConfig) -> MaskingConfig:
    if isinstance(config, MaskingConfig):
        return config
    return MaskingConfig(
        total_mask_ratio=config.total_mask_ratio,
        random_fraction=config.random_fraction,
        info_fraction=config.info_fraction,
        block_fraction=config.block_fraction,
    )


def apply_batched_masking(
    batch: RepresentationBatch,
    config: V1Config | MaskingConfig,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> RepresentationBatch:
    """Apply the existing NumPy policy independently to each batch row.

    ``synth.masking.apply_masking`` remains the source of truth for strategy
    composition and exact valid-patch counts. This bridge only converts one
    padded torch row to ``PatchBatch`` and restores the resulting boolean mask
    to the original device.
    """
    validate_batch(batch)
    if rng is not None and seed is not None:
        raise ValueError("provide either seed or rng, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    patches = batch["patches"]
    pad_mask = batch["patch_pad_mask"]
    starts = batch["starts"]
    valid_len = batch["valid_len"]
    file_samples = batch.get("file_samples")
    if file_samples is None or len(file_samples) != patches.shape[0]:
        raise ValueError("file_samples metadata is required for batched masking")

    synth_config = _synth_masking_config(config)
    masks = torch.zeros_like(batch["mask"])
    compositions: list[Mapping[str, int]] = []
    ratios: list[float] = []
    for index, sample in enumerate(file_samples):
        patch_batch = PatchBatch(
            patches=patches[index].detach().cpu().numpy().astype(np.float32, copy=False),
            starts=starts[index].detach().cpu().numpy().astype(np.int64, copy=False),
            valid_len=valid_len[index].detach().cpu().numpy().astype(np.int64, copy=False),
            pad_mask=pad_mask[index].detach().cpu().numpy().astype(bool, copy=False),
            file_sample=sample,
        )
        result = apply_masking(patch_batch, synth_config, rng)
        masks[index] = torch.from_numpy(result.mask).to(device=masks.device)
        compositions.append(dict(result.composition))
        ratios.append(float(result.total_ratio))

    output: RepresentationBatch = dict(batch)
    output["mask"] = masks
    output["mask_composition"] = compositions
    output["mask_ratio"] = torch.tensor(ratios, dtype=torch.float32, device=masks.device)
    validate_batch(output)
    return output
