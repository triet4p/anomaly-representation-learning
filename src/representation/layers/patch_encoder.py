"""Channel-aware local encoding of variable-length file patches."""

from __future__ import annotations

import torch
from torch import nn


class LocalPatchEncoder(nn.Module):
    """Map ``[B, N, C, W]`` patches to ``[B, N, D]`` embeddings.

    Padding is masked before convolution and again during temporal pooling, so
    changing values in padded positions cannot alter a valid patch embedding.
    ``N`` and ``W`` are runtime dimensions; neither is a model constructor
    parameter.
    """

    def __init__(self, n_channels: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_channels = n_channels
        self.d_model = d_model
        self.projection = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """Return one finite embedding per patch, preserving batch alignment."""
        if patches.ndim != 4:
            raise ValueError(f"patches must be [B, N, C, W], got {patches.shape}")
        if patch_pad_mask.ndim != 3:
            raise ValueError(
                f"patch_pad_mask must be [B, N, W], got {patch_pad_mask.shape}"
            )
        b, n, channels, width = patches.shape
        if channels != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {channels}"
            )
        if tuple(patch_pad_mask.shape) != (b, n, width):
            raise ValueError("patch_pad_mask must match patches batch, count, and width")
        if patch_pad_mask.dtype is not torch.bool:
            raise ValueError("patch_pad_mask must have torch.bool dtype")

        real_steps = ~patch_pad_mask
        # Mask before convolution to prevent padded values leaking through the
        # local receptive field into a valid timestep at the patch boundary.
        flat = patches.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)
        encoded = self.projection(flat.reshape(b * n, channels, width))
        step_weights = real_steps.reshape(b * n, width).to(encoded.dtype).unsqueeze(1)
        pooled = (encoded * step_weights).sum(dim=-1)
        pooled = pooled / step_weights.sum(dim=-1).clamp_min(1.0)
        output = self.output(pooled).reshape(b, n, self.d_model)
        has_real_steps = real_steps.any(dim=-1, keepdim=True)
        return output.masked_fill(~has_real_steps, 0.0)
