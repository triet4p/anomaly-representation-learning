"""Exponential-moving-average target encoder for latent learning."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMATargetEncoder(nn.Module):
    """Maintain a stop-gradient exponential-moving-average copy of a context encoder."""

    def __init__(self, context_encoder: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.target = deepcopy(context_encoder)
        self.register_buffer("_decay", torch.tensor(decay, dtype=torch.float64))
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    @property
    def decay(self) -> float:
        """Return the configured target-weight decay."""
        return float(self._decay.item())

    def train(self, mode: bool = True) -> "EMATargetEncoder":
        """Mirror train/eval mode on the target copy without enabling gradients."""
        super().train(mode)
        self.target.requires_grad_(False)
        return self

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, patch_valid_mask: torch.Tensor) -> torch.Tensor:
        """Encode target tokens without constructing an autograd graph."""
        return self.target(tokens, patch_valid_mask)

    @torch.no_grad()
    def update(self, context_encoder: nn.Module) -> None:
        """Update target state from a context encoder after its optimizer step."""
        target_state = self.target.state_dict()
        context_state = context_encoder.state_dict()
        if target_state.keys() != context_state.keys():
            raise ValueError("context encoder state does not match target encoder")
        for name, target_value in target_state.items():
            context_value = context_state[name]
            if target_value.shape != context_value.shape:
                raise ValueError(f"state shape mismatch for {name}")

        decay = self.decay
        for name, target_value in target_state.items():
            context_value = context_state[name].to(
                device=target_value.device, dtype=target_value.dtype
            )
            if target_value.is_floating_point() or target_value.is_complex():
                target_value.mul_(decay).add_(context_value, alpha=1.0 - decay)
            else:
                target_value.copy_(context_value)
        self.target.load_state_dict(target_state, strict=True)
