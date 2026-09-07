"""Context-conditioned patch distribution encoder (Sprint 11 Task 10).

Models a conditional normal distribution (Gaussian head) plus regime
prototype assignment per patch. Exposes stable per-patch latents with
variable-length masks and localization; there is no waveform/signal
reconstruction path. Masks, labels, and severities never enter this module.

Energy identity (Deep-Review Finding 1, Batch B1): ``context_energy`` is the
encoder conditional NLL (signed, higher-is-more-anomalous) and the ONLY
energy field this module exposes. It is the exact score optimized by the
localized boundary loss and the monitoring score Batch B2 must expose
unchanged at inference. Hierarchical Mahalanobis/mixture energy lives under
the distinct ``population_energy`` field in Task 11 and MUST NOT silently
replace this score: cross-field presence is rejected by contract.
"""

from __future__ import annotations

import torch
from torch import nn

from representation.layers.patch_encoder import LocalPatchEncoder
from representation.layers.sequence_encoder import SequenceContextEncoder
from representation.v2_contracts import validate_context_patch_output


class PatchDistributionHead(nn.Module):
    """Predict conditional Gaussian params and prototype logits per patch."""

    def __init__(self, d_model: int, n_prototypes: int, context_width: int) -> None:
        super().__init__()
        if d_model <= 0 or n_prototypes <= 0 or context_width <= 0:
            raise ValueError("d_model, n_prototypes, and context_width must be positive")
        self.mean = nn.Linear(d_model + context_width, d_model)
        self.logvar = nn.Linear(d_model + context_width, d_model)
        self.prototypes = nn.Linear(d_model + context_width, n_prototypes)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return conditional distribution parameters for valid patches."""
        joined = torch.cat([latents, context], dim=-1)
        logvar = self.logvar(joined).clamp(-6.0, 6.0)
        return {
            "cond_mean": self.mean(joined),
            "cond_logvar": logvar,
            "prototype_logits": self.prototypes(joined),
        }


class ContextConditionedPatchEncoder(nn.Module):
    """Stable patch latents with context-conditioned normal distributions."""

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_robots: int,
        n_programs: int,
        n_regimes: int,
        n_prototypes: int = 2,
        sequence_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_robots <= 0 or n_programs <= 0 or n_regimes <= 0:
            raise ValueError("context cardinalities must be positive")
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        context_width = min(32, d_model)
        self.local = LocalPatchEncoder(n_channels, d_model, dropout=dropout)
        self.context_encoder = SequenceContextEncoder(
            d_model, attention_heads, layers=sequence_layers, dropout=dropout
        )
        self.robot_embedding = nn.Embedding(n_robots, context_width)
        self.program_embedding = nn.Embedding(n_programs, context_width)
        self.regime_embedding = nn.Embedding(n_regimes, context_width)
        self.context_mixer = nn.Sequential(
            nn.Linear(3 * context_width, context_width),
            nn.GELU(),
            nn.LayerNorm(context_width),
        )
        self.head = PatchDistributionHead(d_model, n_prototypes, context_width)

    def _context_vectors(
        self,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> torch.Tensor:
        file_context = torch.cat(
            [
                self.robot_embedding(robot_idx),
                self.program_embedding(program_idx),
            ],
            dim=-1,
        )  # [B, 2C]
        b, n = regime_ids.shape
        file_broadcast = file_context.unsqueeze(1).expand(b, n, -1)
        regime = self.regime_embedding(regime_ids)  # [B, N, C]
        return self.context_mixer(torch.cat([file_broadcast, regime], dim=-1))

    @staticmethod
    def _gaussian_nll(
        latents: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * ((latents - mean).pow(2) / logvar.exp() + logvar).sum(dim=-1)

    def forward(
        self,
        patches: torch.Tensor,
        patch_pad_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return patch latents, distribution params, and conditional energy.

        Sole energy field: ``context_energy`` (signed conditional NLL, zero
        on invalid patches).
        """
        if patches.ndim != 4:
            raise ValueError(f"patches must be [B, N, C, W], got {patches.shape}")
        b, n = patch_valid_mask.shape
        for name, tensor, shape in (
            ("robot_idx", robot_idx, (b,)),
            ("program_idx", program_idx, (b,)),
            ("regime_ids", regime_ids, (b, n)),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if patch_valid_mask.dtype is not torch.bool:
            raise ValueError("patch_valid_mask must have torch.bool dtype")

        local = self.local(patches, patch_pad_mask)
        latents = self.context_encoder(local, patch_valid_mask)
        context = self._context_vectors(robot_idx, program_idx, regime_ids)
        params = self.head(latents, context)
        energy = self._gaussian_nll(latents, params["cond_mean"], params["cond_logvar"])
        energy = energy.masked_fill(~patch_valid_mask, 0.0)

        output = {
            "patch_latents": latents,
            "context_energy": energy,
            "patch_valid_mask": patch_valid_mask,
            "cond_mean": params["cond_mean"].masked_fill(~patch_valid_mask.unsqueeze(-1), 0.0),
            "cond_logvar": params["cond_logvar"].masked_fill(~patch_valid_mask.unsqueeze(-1), 0.0),
            "prototype_logits": params["prototype_logits"].masked_fill(
                ~patch_valid_mask.unsqueeze(-1), 0.0
            ),
        }
        validate_context_patch_output(output)
        return output
