"""Composed V1 representation-learning model."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import torch
from torch import nn

from representation.config import V1Config
from representation.contracts import RepresentationBatch, RepresentationOutput, validate_batch, validate_output
from representation.layers import ConditionalBatchNorm, EMATargetEncoder, LocalPatchEncoder, MaskedLatentPredictor, SequenceContextEncoder
from synth.config import ContrastiveConfig, PatchConfig
from synth.contrastive import make_contrastive_views
from synth.patchify import Patchifier

def _sprint18_view_rng(model_seed: int, epoch: int, file_id: str, view_index: int) -> np.random.Generator:
    """Create the frozen per-file, per-view Sprint 18 augmentation stream."""
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= 100
        or not isinstance(file_id, str)
        or not file_id
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or view_index not in (0, 1)
    ):
        raise ValueError("Sprint18 view randomness requires a valid seed, epoch, file, and view index")
    key = f"sprint18-view-v1|{model_seed}|{epoch}|{file_id}|{view_index}"
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    return np.random.default_rng(seed)


class V1RepresentationModel(nn.Module):
    """Encode complete variable-length files into patch and file latents."""

    def __init__(
        self,
        config: V1Config,
        *,
        patchifier: Patchifier | None = None,
        contrastive_config: ContrastiveConfig | None = None,
        patch_encoder: nn.Module | None = None,
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
        self.patch_encoder = patch_encoder or LocalPatchEncoder(
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
        # N2-MSP coarse-scale student heads: two separate two-layer predictors
        # with hidden/output width d_model. Teacher side uses functional
        # LayerNorm (no shared teacher/student parameters). Inference
        # (forward/score_batch) stays fine-only; coarse outputs are
        # diagnostics, never fused into S_pred.
        self.coarse_predictor_4 = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.coarse_predictor_8 = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        # N3-CCRP channel predictor/embedding: one shared two-layer head with
        # hidden/output width d_model plus a 6-way channel-ID embedding fed
        # only to the relational predictor, never into the B0 encoder.
        self.channel_embedding = nn.Embedding(6, config.d_model)
        self.channel_predictor = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
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

    def msp_coarse_terms(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        groups_4: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
        groups_8: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> dict[str, torch.Tensor]:
        """Compute N2 coarse-4/8 student predictions and stop-gradient teacher targets.

        The online context path must already hide every coarse member before
        context encoding (union mask); this helper never touches visibility.
        Teacher targets are detached EMA latents; student inputs are masked
        member context outputs only. Separate two-layer heads per scale.
        """
        if context.ndim != 3 or target.ndim != 3:
            raise ValueError("coarse latents must be [B, N, D]")
        if context.shape != target.shape:
            raise ValueError("coarse context and target latents must share shape")
        if (
            not isinstance(patch_valid_mask, torch.Tensor)
            or patch_valid_mask.ndim != 2
            or patch_valid_mask.dtype is not torch.bool
            or tuple(patch_valid_mask.shape) != tuple(context.shape[:2])
        ):
            raise ValueError("patch_valid_mask must be a bool [B, N] tensor")
        width = context.shape[-1]
        if width != self.config.d_model:
            raise ValueError("coarse latents must match the configured d_model")
        detached = target.detach()
        outputs: dict[str, torch.Tensor] = {}
        for scale, groups, head in (
            ("group4", groups_4, self.coarse_predictor_4),
            ("group8", groups_8, self.coarse_predictor_8),
        ):
            span = 4 if scale == "group4" else 8
            if not isinstance(groups, (list, tuple)):
                raise ValueError(f"{scale} groups must be a list/tuple of index tuples")
            students: list[torch.Tensor] = []
            teachers: list[torch.Tensor] = []
            for group in groups:
                members = [int(m) for m in group]
                if len(members) != span:
                    raise ValueError(f"{scale} groups must have exactly {span} members")
                student_in = torch.layer_norm(context[:, members, :].mean(dim=1), (width,))
                teacher = torch.layer_norm(detached[:, members, :].mean(dim=1), (width,))
                students.append(head(student_in))
                teachers.append(teacher)
            if students:
                student_t = torch.stack(students, dim=1)
                teacher_t = torch.stack(teachers, dim=1)
                errors = ((student_t - teacher_t).square().mean(dim=-1)).mean()
            else:
                errors = (context.sum() * 0.0) + 0.0
            outputs[f"{scale}_loss"] = errors
            outputs[f"{scale}_count"] = torch.tensor(len(students))
        if int(outputs["group4_count"].item()) and int(outputs["group8_count"].item()):
            outputs["coarse_loss"] = 0.5 * outputs["group4_loss"] + 0.5 * outputs["group8_loss"]
        elif int(outputs["group4_count"].item()):
            outputs["coarse_loss"] = outputs["group4_loss"]
        else:
            outputs["coarse_loss"] = outputs["group8_loss"]
        return outputs

    def ccrp_channel_terms(
        self,
        context_aux: torch.Tensor,
        local_full: torch.Tensor,
        local_without_channel: torch.Tensor,
        channels: torch.Tensor | Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Map corrupted-view context plus channel IDs to normalized predictions.

        ``context_aux`` is the online context encoding of the corrupted aux
        view; ``local_full``/``local_without_channel`` are already-detached
        EMA local-patch encodings of the full vs zeroed normalized patches.
        The channel-ID embedding reaches only this predictor, never the B0
        encoder. Teacher deltas are normalized inside the channel loss.
        """
        if context_aux.ndim != 2 or local_full.ndim != 2 or local_without_channel.ndim != 2:
            raise ValueError("channel latents must be 2D [K, D] tensors")
        if context_aux.shape != local_full.shape or context_aux.shape != local_without_channel.shape:
            raise ValueError("channel context and local targets must share shape")
        width = context_aux.shape[-1]
        if width != self.config.d_model:
            raise ValueError("channel latents must match the configured d_model")
        if isinstance(channels, torch.Tensor):
            channel_ids = [int(v) for v in channels.detach().cpu().tolist()]
        else:
            channel_ids = [int(v) for v in channels]
        if len(channel_ids) != context_aux.shape[0]:
            raise ValueError("one held-out channel is required per relational target")
        for channel in channel_ids:
            if isinstance(channel, bool) or not 0 <= channel < 6:
                raise ValueError("held-out channels must be in [0, 6)")
        channel_emb = self.channel_embedding(
            torch.tensor(channel_ids, dtype=torch.long, device=context_aux.device)
        )
        predicted = self.channel_predictor(torch.cat([context_aux, channel_emb], dim=-1))
        predicted = torch.nn.functional.normalize(predicted, dim=-1)
        detached_full = local_full.detach()
        detached_without = local_without_channel.detach()
        full_norm = torch.layer_norm(detached_full, (width,))
        without_norm = torch.layer_norm(detached_without, (width,))
        from representation.criterion import CrossChannelRelationalCriterion

        terms = CrossChannelRelationalCriterion()(predicted, full_norm, without_norm)
        terms["predicted"] = predicted
        return terms

    def ccrp_aux_local(
        self,
        normalized_full: torch.Tensor,
        normalized_zeroed: torch.Tensor,
        patch_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode full vs zeroed normalized views with the shared local path.

        Inputs are already-normalized patch tensors, so no CBN stat update
        occurs here. Both views pass through the online patch-encoder
        weights under ``no_grad`` (stop-gradient teacher read); no
        independent channel encoder is introduced.
        """
        for name, value in (("normalized_full", normalized_full), ("normalized_zeroed", normalized_zeroed)):
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError(f"{name} must be a 4D [B, N, C, W] tensor")
        if normalized_full.shape != normalized_zeroed.shape:
            raise ValueError("full and zeroed normalized views must share shape")
        if (
            not isinstance(patch_pad_mask, torch.Tensor)
            or patch_pad_mask.ndim != 3
            or patch_pad_mask.dtype is not torch.bool
        ):
            raise ValueError("patch_pad_mask must be a bool [B, N, W] tensor")
        with torch.no_grad():
            full_local = self.patch_encoder(normalized_full, patch_pad_mask)
            zeroed_local = self.patch_encoder(normalized_zeroed, patch_pad_mask)
        return full_local.detach(), zeroed_local.detach()


    @staticmethod
    def _pool_file(latents: torch.Tensor, valid: torch.Tensor,
                   weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is None:
            w = valid.to(latents.dtype).unsqueeze(-1)
            return (latents * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        if tuple(weights.shape) != tuple(valid.shape):
            raise ValueError(
                f"support weights shape {tuple(weights.shape)} != valid shape "
                f"{tuple(valid.shape)}")
        w = weights.to(device=latents.device, dtype=latents.dtype).unsqueeze(-1)
        w = w * valid.to(latents.dtype).unsqueeze(-1)
        return (latents * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

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
        *,
        sprint18_view_epoch: int | None = None,
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
        if sprint18_view_epoch is None:
            generated = [
                make_contrastive_views(sample, self.contrastive_config, self._view_rng)
                for sample in samples
            ]
        else:
            if isinstance(sprint18_view_epoch, bool) or not isinstance(sprint18_view_epoch, int) or not 1 <= sprint18_view_epoch <= 100:
                raise ValueError("Sprint18 view epoch must be in [1, 100]")
            file_ids = [sample.file_id for sample in samples]
            if len(set(file_ids)) != len(file_ids):
                raise ValueError("Sprint18 view batches require unique canonical file IDs")
            generated = [
                make_contrastive_views(
                    sample,
                    self.contrastive_config,
                    (
                        _sprint18_view_rng(self.config.seed, sprint18_view_epoch, sample.file_id, 0),
                        _sprint18_view_rng(self.config.seed, sprint18_view_epoch, sample.file_id, 1),
                    ),
                )
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
            view_weights = None
            hook = getattr(self.patchifier, "support_weights_for", None)
            if hook is not None:
                rows = [hook(pb.starts, pb.valid_len, int(pb.file_sample.T))
                        for pb in patch_batches]
                if any(r is not None for r in rows):
                    view_weights = torch.zeros(
                        (len(patch_batches), max_patches),
                        device=device, dtype=dtype)
                    for index, (pb, row) in enumerate(zip(patch_batches, rows)):
                        if row is None:
                            continue
                        view_weights[index, :pb.N] = torch.from_numpy(
                            np.asarray(row, dtype=np.float32)).to(
                                device=device, dtype=dtype)
            pooled = self._pool_file(contextual, valid, view_weights)
            embeddings[view_index].append(self.project_contrastive(pooled))
        return torch.cat(embeddings[0], dim=0), torch.cat(embeddings[1], dim=0)
    def forward(
        self,
        batch: RepresentationBatch,
        *,
        sprint18_view_epoch: int | None = None,
    ) -> RepresentationOutput:
        """Return representation outputs, optionally using a frozen Sprint18 view epoch."""
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
        weights = batch.get("patch_support_weights")
        file_embedding = self._pool_file(
            context, batch["patch_valid_mask"],
            weights if isinstance(weights, torch.Tensor) else None)
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
        views = self._view_embeddings(
            batch,
            norm_result=norm_result,
            sprint18_view_epoch=sprint18_view_epoch,
        )
        if views is not None:
            output["view_embedding_1"], output["view_embedding_2"] = views
        validate_output(output)
        return output
