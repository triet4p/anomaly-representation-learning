"""Sprint 17 Task 13 — C6 file-pooling alternatives (C6-A / C6-B).

Frozen-registry implementation for the C6 (file pooling) suspect only.
Both arms map ``[B, N, D]`` context latents plus the ``[B, N]`` bool valid
mask to a ``[B, D]`` file embedding under the exact B0 contract (valid
patches only, finite outputs, existing file-embedding dimension):

- ``C6-A`` statistics: valid-patch mean and population standard deviation
  concatenated to ``[B, 2D]`` and linearly projected back to ``[B, D]``.
  The projection is the only adapter (``Linear(2D, D)``, 32,896 params at
  D=128).
- ``C6-B`` gated attention: Ilse-style gated scoring (``tanh(V h)`` gated
  by ``sigmoid(U h)``, linear scorer to one logit) with softmax restricted
  to valid patches. The gate/scorer is the only adapter (8,289 params at
  D=128, gate width 32). Inputs are context latents and the valid mask
  only — no labels, anomaly masks, severity, categories, or simulator
  hidden state enter the pooling.

Invalid positions are masked to zero before any statistic/weight is
formed, so values stored in invalid slots (including non-finite ones)
cannot alter the file embedding; rows with no valid patch yield a finite
fallback (C6-A: projection of the zero statistic; C6-B: the zero vector)
instead of NaN. ``C6ArmModel`` overrides only ``_pool_file``; patch
encoding, context encoding, objective, projector, bank, scorers, and
metric code are untouched.
"""

from __future__ import annotations

import torch
from torch import nn

from representation.model import V1RepresentationModel

#: Registered C6 arm IDs (protocol sprint17-ablation-v4 §2).
C6_ARMS = ("C6-A", "C6-B")

#: Frozen gate width for C6-B; asserted by focused tests via param identity.
C6B_GATE_DIM = 32

#: Frozen adapter parameter totals at d_model=128; asserted by focused tests.
ARM_POOLING_PARAMS = {"C6-A": 32896, "C6-B": 8289}

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C6-A": ("valid-patch mean-plus-population-std concatenated to 2D and "
             "linearly projected to D (one Linear(256,128) + bias, 32,896 "
             "params); invalid patches masked to zero before statistics; "
             "no other adapter"),
    "C6-B": ("Ilse-style gated attention over valid patches (tanh/sigmoid "
             "gates D->32, linear scorer 32->1, 8,289 params; softmax over "
             "valid only, invalid weight exactly 0); inputs are context "
             "latents and the valid mask only; no other adapter"),
}


def _validate_pool_inputs(latents: torch.Tensor, valid: torch.Tensor) -> None:
    if latents.ndim != 3:
        raise ValueError(f"latents must be [B, N, D], got {tuple(latents.shape)}")
    if valid.ndim != 2:
        raise ValueError(f"valid mask must be [B, N], got {tuple(valid.shape)}")
    if tuple(valid.shape) != (latents.shape[0], latents.shape[1]):
        raise ValueError(
            f"valid shape {tuple(valid.shape)} != latents batch/count "
            f"{(latents.shape[0], latents.shape[1])}")
    if valid.dtype is not torch.bool:
        raise ValueError("valid mask must have torch.bool dtype")


class MeanStdProjectionPooling(nn.Module):
    """C6-A: valid-patch mean-plus-std projected to the file-embedding dim."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model
        self.project = nn.Linear(2 * d_model, d_model)

    def forward(self, latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        _validate_pool_inputs(latents, valid)
        # Device invariant: every accumulator shares the latents device/dtype.
        v = valid.to(latents.dtype).unsqueeze(-1)
        safe = latents.masked_fill(~valid.unsqueeze(-1), 0.0)
        n = v.sum(dim=1).clamp_min(1.0)
        mean = (safe * v).sum(dim=1) / n
        diff = (safe - mean.unsqueeze(1)) * v
        var = (diff.square()).sum(dim=1) / n
        std = var.clamp_min(0.0).sqrt()
        return self.project(torch.cat([mean, std], dim=-1))


class GatedAttentionPooling(nn.Module):
    """C6-B: gated attention pooling restricted to valid patches."""

    def __init__(self, d_model: int = 128, gate_dim: int = C6B_GATE_DIM) -> None:
        super().__init__()
        if d_model <= 0 or gate_dim <= 0:
            raise ValueError(
                f"d_model and gate_dim must be positive, got {(d_model, gate_dim)}")
        self.d_model = d_model
        self.gate_dim = gate_dim
        self.gate_v = nn.Linear(d_model, gate_dim)
        self.gate_u = nn.Linear(d_model, gate_dim)
        self.gate_w = nn.Linear(gate_dim, 1)

    def attention_weights(self, latents: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
        """Softmax weights over valid patches; invalid weight is exactly 0."""
        _validate_pool_inputs(latents, valid)
        safe = latents.masked_fill(~valid.unsqueeze(-1), 0.0)
        gate = torch.tanh(self.gate_v(safe)) * torch.sigmoid(self.gate_u(safe))
        scores = self.gate_w(gate).squeeze(-1)
        floor = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid, floor)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        return weights

    def forward(self, latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(latents, valid)
        safe = latents.masked_fill(~valid.unsqueeze(-1), 0.0)
        return (weights.unsqueeze(-1) * safe).sum(dim=1)


def arm_pooling(arm_id: str, d_model: int = 128) -> nn.Module:
    """Build the frozen pooling module for a registered C6 arm."""
    if arm_id == "C6-A":
        return MeanStdProjectionPooling(d_model)
    if arm_id == "C6-B":
        return GatedAttentionPooling(d_model, C6B_GATE_DIM)
    raise ValueError(f"unknown Sprint 17 C6 arm: {arm_id!r}")


def count_pooling_params(pooling: nn.Module) -> int:
    """Total trainable parameters of a C6 pooling module."""
    return int(sum(p.numel() for p in pooling.parameters() if p.requires_grad))


class C6ArmModel(V1RepresentationModel):
    """B0 graph with only the file-pooling function replaced (C6-A / C6-B).

    Patch encoding, context encoding, EMA targets, prediction, contrastive
    views, and the file-embedding dimension are inherited unchanged; the
    override routes every ``_pool_file`` call (main path and contrastive
    views) through the arm pooling module.
    """

    def __init__(self, config, *, arm_id: str,
                 pooling: nn.Module | None = None, **kwargs) -> None:
        if arm_id not in C6_ARMS:
            raise ValueError(f"C6ArmModel runs only {C6_ARMS}, got {arm_id!r}")
        super().__init__(config, **kwargs)
        self.c6_arm_id = arm_id
        self.c6_pooling = pooling if pooling is not None else arm_pooling(
            arm_id, config.d_model)

    def _pool_file(self, latents: torch.Tensor, valid: torch.Tensor,
                   weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is not None:
            raise ValueError(
                "C6 arms pool over valid patches only; support weights are "
                "an undeclared mechanism for C6")
        return self.c6_pooling(latents, valid)
