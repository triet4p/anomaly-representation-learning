"""Composed V1 representation-learning model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from representation.config import V1Config
from representation.contracts import RepresentationBatch, RepresentationOutput, validate_batch, validate_output
from representation.layers import ConditionalBatchNorm, EMATargetEncoder, LocalPatchEncoder, MaskedLatentPredictor, SequenceContextEncoder
from synth.config import ContrastiveConfig, PatchConfig
from synth.contrastive import make_contrastive_views
from synth.patchify import Patchifier


class V1RepresentationModel(nn.Module):
    """Encode complete variable-length files into patch and file latents."""

    def __init__(
        self,
        config: V1Config,
        *,
        patchifier: Patchifier | None = None,
        contrastive_config: ContrastiveConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patchifier = patchifier or Patchifier(
            PatchConfig(patch_size=config.patch_size, stride=config.stride, pad_end=True)
        )
        self.contrastive_config = contrastive_config or ContrastiveConfig(
            gain_std=config.contrastive_gain_std,
            offset_std=config.contrastive_offset_std,
            noise_std=config.contrastive_noise_std,
            max_shift=config.contrastive_max_shift,
        )
        self.patch_encoder = LocalPatchEncoder(
            config.n_channels, config.d_model, dropout=config.dropout
        )
        self.context_encoder = SequenceContextEncoder(
            config.d_model,
            config.attention_heads,
            layers=config.sequence_layers,
            dropout=config.dropout,
        )
        self.target_encoder = EMATargetEncoder(
            self.context_encoder, decay=config.ema_decay
        )
        self.predictor = MaskedLatentPredictor(config.d_model, dropout=config.dropout)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.conditional_norm = (
            ConditionalBatchNorm(
                num_sensors=config.n_channels,
                condition_cardinalities={
                    "robot_idx": config.n_robots,
                    "program_idx": config.n_programs,
                },
                min_bucket_samples=config.min_bucket_samples,
            )
            if config.use_conditional_norm
            else None
        )
        self._view_rng = np.random.default_rng(config.seed)

    def _encode_patches(
        self,
        patches: torch.Tensor,
        patch_pad_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.patch_encoder(patches, patch_pad_mask)
        visible_local = local.masked_fill(mask.unsqueeze(-1), 0.0)
        context = self.context_encoder(visible_local, patch_valid_mask)
        with torch.no_grad():
            target = self.target_encoder(
                self.patch_encoder(patches, patch_pad_mask), patch_valid_mask
            )
        predicted, prediction_mask = self.predictor(context, mask, patch_valid_mask)
        return context, target.detach(), predicted, prediction_mask

    @staticmethod
    def _pool_file(latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(latents.dtype).unsqueeze(-1)
        return (latents * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def get_extra_state(self) -> dict[str, object]:
        """Serialize augmentation RNG state with model checkpoints."""
        return {"view_rng_state": self._view_rng.bit_generator.state}

    def set_extra_state(self, state: object) -> None:
        """Restore augmentation RNG state, or reset it from the config seed."""
        self._view_rng = np.random.default_rng(self.config.seed)
        if isinstance(state, dict) and "view_rng_state" in state:
            self._view_rng.bit_generator.state = state["view_rng_state"]

    def reset_view_rng(self, seed: int | None = None) -> None:
        """Reset view augmentation randomness for a reproducible run."""
        self._view_rng = np.random.default_rng(self.config.seed if seed is None else seed)
    def project_contrastive(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project file-level embeddings into the contrastive optimization space."""
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected embeddings with shape [B, {self.config.d_model}], got {tuple(embeddings.shape)}"
            )
        return self.contrastive_projector(embeddings)


    def _view_embeddings(
        self,
        batch: RepresentationBatch,
        norm_result: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        samples = batch.get("file_samples")
        if samples is None:
            return None
        device = batch["patches"].device
        dtype = batch["patches"].dtype
        if norm_result is None and self.conditional_norm is not None and "signals" in batch:
            norm_inputs: dict[str, torch.Tensor] = {
                "input": batch["signals"],
                "valid_mask": batch.get("file_valid_mask"),
            }
            if "robot_idx" in batch and isinstance(batch["robot_idx"], torch.Tensor):
                norm_inputs["robot_idx"] = batch["robot_idx"]
            if "program_idx" in batch and isinstance(batch["program_idx"], torch.Tensor):
                norm_inputs["program_idx"] = batch["program_idx"]
            norm_result = self.conditional_norm(norm_inputs)
        generated: list[tuple[object, object]] = [
            make_contrastive_views(sample, self.contrastive_config, self._view_rng)
            for sample in samples
        ]
        embeddings: list[list[torch.Tensor]] = [[], []]
        for view_index in range(2):
            patch_batches = [
                self.patchifier.patchify(replace(sample, x=generated[index][view_index]))
                for index, sample in enumerate(samples)
            ]
            max_patches = max(1, max(patch_batch.N for patch_batch in patch_batches))
            channels = samples[0].C
            width = self.patchifier.cfg.patch_size
            patches = torch.zeros(
                (len(samples), max_patches, channels, width), device=device, dtype=dtype
            )
            pad_mask = torch.ones(
                (len(samples), max_patches, width), device=device, dtype=torch.bool
            )
            valid = torch.zeros((len(samples), max_patches), device=device, dtype=torch.bool)
            for index, patch_batch in enumerate(patch_batches):
                count = patch_batch.N
                if count:
                    patches[index, :count] = torch.from_numpy(patch_batch.patches).to(
                        device=device, dtype=dtype
                    )
                    pad_mask[index, :count] = torch.from_numpy(patch_batch.pad_mask).to(
                        device=device
                    )
                    valid[index, :count] = torch.from_numpy(
                        ~patch_batch.pad_mask.all(axis=1)
                    ).to(device=device)
            if norm_result is not None:
                means = norm_result["mean"].to(device=device, dtype=dtype)
                stds = norm_result["std"].to(device=device, dtype=dtype)
                patches = (patches - means.unsqueeze(1).unsqueeze(-1)) / stds.unsqueeze(1).unsqueeze(-1)
                patches = patches.masked_fill(pad_mask.unsqueeze(2), 0.0)
            local = self.patch_encoder(patches, pad_mask)
            contextual = self.context_encoder(local, valid)
            pooled = self._pool_file(contextual, valid)
            embeddings[view_index].append(self.project_contrastive(pooled))
        return torch.cat(embeddings[0], dim=0), torch.cat(embeddings[1], dim=0)
    def forward(self, batch: RepresentationBatch) -> RepresentationOutput:
        """Return context, stop-gradient target, prediction, and file latents."""
        validate_batch(batch)
        patches = batch["patches"]
        norm_result = None
        if self.conditional_norm is not None:
            norm_inputs: dict[str, torch.Tensor] = {
                "input": batch["signals"],
                "valid_mask": batch["file_valid_mask"],
            }
            if "robot_idx" in batch and isinstance(batch["robot_idx"], torch.Tensor):
                norm_inputs["robot_idx"] = batch["robot_idx"]
            if "program_idx" in batch and isinstance(batch["program_idx"], torch.Tensor):
                norm_inputs["program_idx"] = batch["program_idx"]
            norm_result = self.conditional_norm(norm_inputs)
            means = norm_result["mean"]
            stds = norm_result["std"]
            patches = (patches - means.unsqueeze(1).unsqueeze(-1)) / stds.unsqueeze(1).unsqueeze(-1)
            patches = patches.masked_fill(batch["patch_pad_mask"].unsqueeze(2), 0.0)

        context, target, predicted, prediction_mask = self._encode_patches(
            patches,
            batch["patch_pad_mask"],
            batch["patch_valid_mask"],
            batch["mask"],
        )
        file_embedding = self._pool_file(context, batch["patch_valid_mask"])
        output: RepresentationOutput = {
            "context_latents": context,
            "target_latents": target,
            "predicted_latents": predicted,
            "prediction_mask": prediction_mask,
            "file_embedding": file_embedding,
            "projected_file_embedding": self.project_contrastive(file_embedding),
            "patch_valid_mask": batch["patch_valid_mask"],
        }
        if norm_result is not None:
            output["normalization"] = norm_result
        views = self._view_embeddings(batch, norm_result=norm_result)
        if views is not None:
            output["view_embedding_1"], output["view_embedding_2"] = views
        validate_output(output)
        return output
