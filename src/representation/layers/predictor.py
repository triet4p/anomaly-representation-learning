"""Prediction head for masked patch latent targets."""

from __future__ import annotations

import math

import torch
from torch import nn


class MaskedLatentPredictor(nn.Module):
    """Predict masked target latents by attending to visible patch context."""

    def __init__(
        self,
        d_model: int,
        target_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        target_width = target_dim or d_model
        if target_width <= 0:
            raise ValueError("target_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.target_dim = target_width
        self.hidden_dim = hidden_dim or 2 * d_model
        attention_heads = 2 if d_model % 2 == 0 else 1
        self.cross_attention = nn.MultiheadAttention(
            d_model, attention_heads, dropout=dropout, batch_first=True
        )
        self.mask_embedding = nn.Linear(1, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model + 1),
            nn.Linear(2 * d_model + 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, target_width),
        )

    def _positions(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10_000.0) / self.d_model)
        )
        encoding = torch.zeros((length, self.d_model), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(
        self,
        context_latents: torch.Tensor,
        requested_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned predictions and ``requested_mask & patch_valid_mask``."""
        if context_latents.ndim != 3:
            raise ValueError("context_latents must be [B, N, D]")
        if requested_mask.ndim != 2 or patch_valid_mask.ndim != 2:
            raise ValueError("requested_mask and patch_valid_mask must be [B, N]")
        batch, count, width = context_latents.shape
        if width != self.d_model:
            raise ValueError(f"expected latent width {self.d_model}, got {width}")
        if tuple(requested_mask.shape) != (batch, count):
            raise ValueError("requested_mask must match context_latents batch and count")
        if tuple(patch_valid_mask.shape) != (batch, count):
            raise ValueError("patch_valid_mask must match context_latents batch and count")
        if requested_mask.dtype is not torch.bool or patch_valid_mask.dtype is not torch.bool:
            raise ValueError("mask inputs must have torch.bool dtype")

        prediction_mask = requested_mask & patch_valid_mask
        visible = patch_valid_mask & ~requested_mask
        positions = self._positions(count, context_latents.device, context_latents.dtype)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)
        visible_values = context_latents.masked_fill(~visible.unsqueeze(-1), 0.0)
        keys = visible_values + positions
        queries = positions + self.mask_embedding(requested_mask.to(context_latents.dtype).unsqueeze(-1))

        # Multi-head attention returns NaNs when every key is masked. Keep a
        # harmless sentinel key for those rows; all invalid output positions are
        # still zeroed below and cannot contribute to the prediction loss.
        key_padding_mask = ~visible
        no_visible = ~visible.any(dim=1)
        safe_key_padding_mask = key_padding_mask.clone()
        safe_key_padding_mask[no_visible, 0] = False
        attended, _ = self.cross_attention(
            queries, keys, visible_values, key_padding_mask=safe_key_padding_mask
        )
        features = torch.cat(
            [attended, queries, requested_mask.to(context_latents.dtype).unsqueeze(-1)], dim=-1
        )
        predictions = self.head(features)
        return predictions.masked_fill(~prediction_mask.unsqueeze(-1), 0.0), prediction_mask


LatentPredictor = MaskedLatentPredictor
