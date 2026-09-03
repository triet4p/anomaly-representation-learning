"""Opt-in RVQ comparison baseline.

Algorithm provenance: narrowly adapted from the legacy
``ResidualFactorizedVectorQuantizerEMA`` in
``anomaly_detection/layers/bottleneck.py``. This module intentionally exposes
only its documented ``[B,C,D]`` quantization interface and no reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RVQBaseline(nn.Module):
    """Per-channel residual VQ baseline, kept outside the V1 default path."""

    def __init__(
        self,
        n_channels: int,
        codebook_size: int = 64,
        num_levels: int = 2,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or codebook_size <= 0 or num_levels <= 0 or embedding_dim <= 0:
            raise ValueError("RVQ dimensions must be positive")
        if commitment_cost < 0.0 or not 0.0 < decay < 1.0:
            raise ValueError("invalid commitment_cost or decay")
        self.n_channels = n_channels
        self.codebook_size = codebook_size
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.register_buffer("codebooks", torch.empty(num_levels, n_channels, codebook_size, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_levels, n_channels, codebook_size))
        self.register_buffer("embed_avg", torch.empty(num_levels, n_channels, codebook_size, embedding_dim))
        self.register_buffer("is_initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, inputs: torch.Tensor) -> None:
        normalized = F.normalize(inputs, dim=-1)
        samples = normalized[:, :, :].permute(1, 0, 2)
        for level in range(self.num_levels):
            for channel in range(self.n_channels):
                values = samples[channel]
                indices = torch.arange(self.codebook_size, device=inputs.device) % values.shape[0]
                chosen = values[indices]
                self.codebooks[level, channel].copy_(chosen)
                self.embed_avg[level, channel].copy_(chosen)
                self.cluster_size[level, channel].fill_(1.0)
        self.is_initialized.fill_(True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(quantized [B,C,D], commitment_loss, indices [L,B,C])``."""
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [B, C, D]")
        batch, channels, width = inputs.shape
        if channels != self.n_channels or width != self.embedding_dim:
            raise ValueError("inputs channel or embedding dimensions do not match baseline")
        if batch == 0:
            raise ValueError("inputs cannot be empty")
        if not self.is_initialized:
            self._initialize(inputs)
        residual = F.normalize(inputs, dim=-1)
        quantized_total = torch.zeros_like(residual)
        all_indices: list[torch.Tensor] = []
        commitment = residual.new_zeros(())
        for level in range(self.num_levels):
            level_quantized = torch.zeros_like(residual)
            level_indices: list[torch.Tensor] = []
            for channel in range(channels):
                codebook = F.normalize(self.codebooks[level, channel], dim=-1)
                distances = 2.0 - 2.0 * residual[:, channel] @ codebook.transpose(0, 1)
                indices = distances.argmin(dim=-1)
                level_indices.append(indices)
                level_quantized[:, channel] = codebook[indices]
                commitment = commitment + F.mse_loss(level_quantized[:, channel].detach(), residual[:, channel])
            quantized_total = quantized_total + level_quantized
            residual = residual - level_quantized.detach()
            all_indices.append(torch.stack(level_indices, dim=1))
        quantized = F.normalize(inputs, dim=-1) + (quantized_total - F.normalize(inputs, dim=-1)).detach()
        return quantized, self.commitment_cost * commitment / (self.num_levels * channels), torch.stack(all_indices)
