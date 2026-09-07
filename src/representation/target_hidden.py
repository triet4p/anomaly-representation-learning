"""Target-hidden context scorer without query self-copying (Sprint 12 Task 8).

Predicts each patch's latent from position-aware context that cannot see that
patch's CONTENT: the target (plus waveform-overlapping neighbors j-1/j/j+1 and
pads) is zeroed BEFORE any contextual mixing, while EVERY position — including
the hidden target's own — receives the repo-standard sinusoidal position code
(cf. SequenceContextEncoder._positions), so queries/keys carry order without
content. Targets are stop-gradient: gradients shape the context mixer and the
head, never the targets. Energy uses boring FIXED unit variance
(``0.5 * ||z - mu||^2``): no learned variance head exists to clamp or copy,
and loss terms share coherent declared scales. The scored field is the
distinct experimental ``hidden_context_energy`` — never the canonical
``context_energy`` or ``population_energy``.

Supersession: the first revision (order-blind mixer + learned query-conditioned
variance, whose NLL logvar term at -6.0 drowned the boundary) is REJECTED as
defective; its checkpoints must not be reused.
"""

from __future__ import annotations
import torch
from torch import nn

#: Waveform-overlap radius in patch index units (stride 16 < width 32).
OVERLAP_RADIUS = 1


def sinusoidal_positions(length: int, width: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
    """Repo-standard sinusoidal position codes (SequenceContextEncoder pattern)."""
    import math

    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros((length, width), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding


def allowed_keys(n: int, valid: torch.Tensor) -> torch.Tensor:
    """Bool matrix [B, N, N]: query j may use key k (valid, beyond overlap)."""
    if valid.ndim != 2 or valid.shape[1] != n:
        raise ValueError("valid must have shape [B, N]")
    if valid.dtype is not torch.bool:
        raise ValueError("valid must have torch.bool dtype")
    distance = (torch.arange(n, device=valid.device).unsqueeze(0)
                - torch.arange(n, device=valid.device).unsqueeze(1))
    structural = distance.abs() <= OVERLAP_RADIUS  # [N, N] always blocked
    return valid.unsqueeze(1) & ~structural.unsqueeze(0)  # [B, N, N]


def exclusion_bias(n: int, valid: torch.Tensor) -> torch.Tensor:
    """Additive attention bias [B, N, N]: -inf on target/overlap/padded keys.

    Query j may attend key k only when patch k is valid AND |k - j| >
    OVERLAP_RADIUS. Returns float bias (0.0 allowed, -inf blocked).
    """
    allowed = allowed_keys(n, valid)
    return torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(~allowed, float("-inf"))


class TargetHiddenScorer(nn.Module):
    """Context-mixing scorer with the query target hidden by construction."""

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0:
            raise ValueError("d_model, n_heads, and n_layers must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                dropout=0.0, batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.mean = nn.Linear(d_model, d_model)
        # NOTE: deliberately no variance head (fixed unit variance energy).

    def config(self) -> dict[str, int]:
        """Serializable architecture config for coherent checkpoints."""
        return {"d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers}

    def context_for(self, latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Position-aware context with target CONTENT hidden BEFORE mixing.

        Run j consumes latents with positions {j-1, j, j+1} (and pads) zeroed,
        so no residual stream can carry the target into its own context; keys
        are additionally blocked in attention. EVERY position — including the
        zeroed target's own — receives the sinusoidal position code, so the
        target query gets its POSITION ONLY. context[b, j] is output position
        j of run j (diagonal gather over the batched runs).
        """
        if latents.ndim != 3 or latents.shape[2] != self.d_model:
            raise ValueError(f"latents must be [B, N, {self.d_model}]")
        b, n, _ = latents.shape
        key_ok = allowed_keys(n, valid).to(latents.device)  # [B, N, N] per query
        content = latents.unsqueeze(1).expand(b, n, n, -1) * key_ok.unsqueeze(-1).to(latents.dtype)
        positions = sinusoidal_positions(n, self.d_model, latents.device, latents.dtype)
        h = (content + positions.unsqueeze(0).unsqueeze(0)).reshape(b * n, n, -1)
        key_pad = ~key_ok.reshape(b * n, n)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=key_pad)
        h = h.reshape(b, n, n, -1)
        out = h[torch.arange(b).unsqueeze(1), torch.arange(n), torch.arange(n)]
        # Queries with no allowed keys (degenerate N or isolation) get an
        # explicit zero context — never NaN from fully-masked attention.
        has_keys = key_ok.any(dim=-1)
        return torch.where(has_keys.unsqueeze(-1), out, torch.zeros_like(out))

    def forward(self, latents: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score stop-gradient targets with fixed unit variance (boring energy)."""
        context = self.context_for(latents, valid)
        mu = self.mean(context)
        target = latents.detach()  # stop-gradient: never ease the targets
        energy = 0.5 * (target - mu).pow(2).sum(dim=-1)
        energy = energy.masked_fill(~valid, 0.0)
        return {
            "hidden_context": context,
            "hidden_mean": mu,
            "hidden_context_energy": energy,
            "patch_valid_mask": valid,
        }

    def save(self, path: object) -> None:
        """Persist a coherent checkpoint (config + weights)."""
        torch.save({"config": self.config(), "state": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path: object) -> "TargetHiddenScorer":
        """Restore a checkpoint; fail fast on architecture mismatch."""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("target-hidden checkpoint is missing its config")
        model = cls(d_model=int(config["d_model"]), n_heads=int(config.get("n_heads", 4)),
                    n_layers=int(config.get("n_layers", 2)))
        if model.config() != {k: int(v) for k, v in config.items()}:
            raise ValueError(f"target-hidden config mismatch: {config}")
        model.load_state_dict(payload["state"])
        return model.eval()
