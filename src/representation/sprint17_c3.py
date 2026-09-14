"""Sprint 17 Task 11 — C3 local-encoder alternatives (C3-A / C3-B).

Frozen-registry implementation for the C3 (local patch encoder) suspect only.
Both arms map ``[B, N, C, W]`` patches to ``[B, N, D]`` embeddings under the
exact B0 contract (masked pooling, zero fully-padded patches, finite outputs)
and are parameter-matched to the B0 ``LocalPatchEncoder`` (35,712 parameters
at C=6, D=128):

- ``C3-A``: residual depthwise-separable dilated 1D convolution front-end
  (1x1 input projection, depthwise k=3 dilation=2, pointwise 1x1, residual
  add) feeding the unchanged output head. Total 35,688 (−24 vs B0).
- ``C3-B``: lightweight within-patch Transformer (1x1 input projection to
  d=32, learned absolute position table, one encoder layer with an explicit
  padding mask, projection back to D) feeding the unchanged output head.
  Total 35,724 (+12 vs B0; feedforward width 140 chosen for the match).
- ``C3-A``: residual depthwise-separable dilated 1D convolution front-end
  (1x1 input projection, depthwise k=3 dilation=2, pointwise 1x1, residual
  add) feeding the unchanged output head. Total 34,688 (−1,024 vs B0).
either arm through the same ``patch_encoder(patches, patch_pad_mask)`` call.
"""

from __future__ import annotations

import torch
from torch import nn

#: Exact B0 LocalPatchEncoder count at (n_channels=6, d_model=128).
B0_LOCAL_ENCODER_PARAMS = 35712
#: Frozen arm totals at (6, 128); asserted by focused tests.
ARM_LOCAL_ENCODER_PARAMS = {"C3-A": 34688, "C3-B": 35724}

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C3-A": ("residual depthwise-separable dilated-1D-conv patch encoder "
             "(in-proj k1, depthwise k3/dilation-2, pointwise k1, residual); "
             "unchanged output head and masked pooling; no other adapter"),
    "C3-B": ("lightweight within-patch Transformer (d=32, 2 heads, ff=140, "
             "1 layer) with learned absolute positions and explicit padding "
             "masks; unchanged output head and masked pooling; no other "
             "adapter"),
}

_C3_ARMS = ("C3-A", "C3-B")


def _validate_inputs(patches: torch.Tensor, patch_pad_mask: torch.Tensor,
                     n_channels: int) -> tuple[int, int, int, int]:
    if patches.ndim != 4:
        raise ValueError(f"patches must be [B, N, C, W], got {patches.shape}")
    if patch_pad_mask.ndim != 3:
        raise ValueError(
            f"patch_pad_mask must be [B, N, W], got {patch_pad_mask.shape}")
    b, n, channels, width = patches.shape
    if channels != n_channels:
        raise ValueError(f"expected {n_channels} channels, got {channels}")
    if tuple(patch_pad_mask.shape) != (b, n, width):
        raise ValueError("patch_pad_mask must match patches batch, count, and width")
    if patch_pad_mask.dtype is not torch.bool:
        raise ValueError("patch_pad_mask must have torch.bool dtype")
    return b, n, channels, width


def _masked_pool(encoded: torch.Tensor, real_steps: torch.Tensor,
                 d_model: int) -> torch.Tensor:
    """Masked temporal mean over encoded steps ``[BN, D, W]``."""
    step_weights = real_steps.to(encoded.dtype).unsqueeze(1)
    pooled = (encoded * step_weights).sum(dim=-1)
    return pooled / step_weights.sum(dim=-1).clamp_min(1.0)


class ResidualDilatedPatchEncoder(nn.Module):
    """C3-A: residual depthwise-separable dilated convolution encoder."""

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
        self.in_projection = nn.Conv1d(n_channels, d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2,
                                   dilation=2, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """Return one finite embedding per patch, preserving batch alignment."""
        b, n, _, width = _validate_inputs(patches, patch_pad_mask, self.n_channels)
        real_steps = ~patch_pad_mask
        flat = patches.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)
        base = self.in_projection(flat.reshape(b * n, self.n_channels, width))
        refined = self.pointwise(self.depthwise(base))
        encoded = self.activation(base + refined)
        pooled = _masked_pool(encoded, real_steps.reshape(b * n, width), self.d_model)
        output = self.output(pooled).reshape(b, n, self.d_model)
        has_real_steps = real_steps.any(dim=-1, keepdim=True)
        return output.masked_fill(~has_real_steps, 0.0)


class WithinPatchTransformerEncoder(nn.Module):
    """C3-B: lightweight within-patch Transformer with explicit masks."""

    POS_WIDTH = 32
    MODEL_DIM = 32
    HEADS = 2
    FEEDFORWARD_DIM = 140

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
        self.in_projection = nn.Conv1d(n_channels, self.MODEL_DIM, kernel_size=1)
        self.position = nn.Parameter(torch.zeros(1, self.POS_WIDTH, self.MODEL_DIM))
        self.attention = nn.TransformerEncoderLayer(
            d_model=self.MODEL_DIM, nhead=self.HEADS,
            dim_feedforward=self.FEEDFORWARD_DIM, dropout=dropout,
            batch_first=True)
        self.out_projection = nn.Linear(self.MODEL_DIM, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, patches: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """Return one finite embedding per patch, preserving batch alignment."""
        b, n, _, width = _validate_inputs(patches, patch_pad_mask, self.n_channels)
        if width != self.POS_WIDTH:
            raise ValueError(
                f"C3-B position table is width {self.POS_WIDTH}, got {width}")
        real_steps = ~patch_pad_mask
        flat = patches.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)
        steps = self.in_projection(flat.reshape(b * n, self.n_channels, width))
        tokens = steps.transpose(1, 2) + self.position[:, :width, :]
        attended = self.attention(tokens, src_key_padding_mask=patch_pad_mask.reshape(b * n, width))
        encoded = self.out_projection(attended).transpose(1, 2)
        pooled = _masked_pool(encoded, real_steps.reshape(b * n, width), self.d_model)
        output = self.output(pooled).reshape(b, n, self.d_model)
        has_real_steps = real_steps.any(dim=-1, keepdim=True)
        return output.masked_fill(~has_real_steps, 0.0)


def arm_local_encoder(arm_id: str, n_channels: int = 6, d_model: int = 128,
                      dropout: float = 0.1) -> nn.Module:
    """Build the frozen local encoder for a registered C3 arm."""
    if arm_id == "C3-A":
        return ResidualDilatedPatchEncoder(n_channels, d_model, dropout)
    if arm_id == "C3-B":
        return WithinPatchTransformerEncoder(n_channels, d_model, dropout)
    raise ValueError(f"unknown Sprint 17 C3 arm: {arm_id!r}")


def count_local_encoder_params(encoder: nn.Module) -> int:
    """Total trainable parameters of a local patch encoder."""
    return int(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
