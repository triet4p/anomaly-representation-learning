"""Variable-length patch sequence/context encoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class SequenceContextEncoder(nn.Module):
    """Contextualize patch embeddings while ignoring padded patch positions."""

    def __init__(
        self,
        d_model: int,
        attention_heads: int,
        layers: int = 2,
        dropout: float = 0.0,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or attention_heads <= 0:
            raise ValueError("d_model and attention_heads must be positive")
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if layers <= 0:
            raise ValueError("layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.layers = layers
        hidden = feedforward_dim or 4 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=norm,
            enable_nested_tensor=False,
        )

    @property
    def norm(self) -> nn.LayerNorm:
        """Return the final LayerNorm module of the transformer encoder."""
        return self.encoder.norm  # type: ignore[return-value]

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

    def forward(self, tokens: torch.Tensor, patch_valid_mask: torch.Tensor) -> torch.Tensor:
        """Return contextual tokens with shape ``[B,N,D]``."""
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, N, D], got {tokens.shape}")
        if patch_valid_mask.ndim != 2:
            raise ValueError("patch_valid_mask must be [B, N]")
        batch, count, width = tokens.shape
        if width != self.d_model:
            raise ValueError(f"expected latent width {self.d_model}, got {width}")
        if tuple(patch_valid_mask.shape) != (batch, count):
            raise ValueError("patch_valid_mask must match tokens batch and count")
        if patch_valid_mask.dtype is not torch.bool:
            raise ValueError("patch_valid_mask must have torch.bool dtype")

        valid = patch_valid_mask
        key_padding_mask = ~valid
        # Transformer attention with every key masked can produce NaNs. Keep a
        # harmless sentinel key for those rows, then zero all invalid outputs.
        no_valid = ~valid.any(dim=1)
        safe_padding_mask = key_padding_mask.clone()
        safe_padding_mask[no_valid, 0] = False
        positioned = tokens + self._positions(count, tokens.device, tokens.dtype).unsqueeze(0)
        encoded = self.encoder(positioned, src_key_padding_mask=safe_padding_mask)
        return encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
