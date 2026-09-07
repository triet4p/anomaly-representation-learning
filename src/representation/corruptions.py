"""Controlled mechanism-based corruptions with nuisance controls (Sprint 12 Task 7).

In-memory paired clean/corrupt waveform views preserving unit, robot, program,
regime, and nuisance context: the corrupted view differs only inside the
synthetic mask on valid, real steps. Five development mechanisms with ordered
severity, exact support, and physical bounds, plus nuisance-only controls that
must NOT flag:

- level_drift: slow additive ramp on a channel subset (drift class);
- relative_gain_drift: slow parametric relative-gain change between channel
  pairs (drift class; NOT a swap/contradiction injection);
- timing_shift: local resampling phase shift inside the masked span;
- transient: brief bounded spike/dip on a sub-window;
- persistent_degradation: level shift + variance change to patch end;
- nuisance_gain / nuisance_offset / nuisance_noise: benign global probes.

MECHANISM PARTITION (protocol v2 §3): the reserved eval families
(`wrong_transition`, `cross_channel_inconsistency`) are generator-level
mechanisms held out entirely — this module implements NO swap/contradiction
or transition-restructuring corruption, and nothing here reads generator
family code. Masks shape the views only and never enter any encoder.
"""

from __future__ import annotations

import torch

# Physical bounds per mechanism (severity multiplies within these envelopes).
BOUNDS: dict[str, dict[str, float]] = {
    "level_drift": {"max_abs_offset": 3.0, "channels": "subset"},
    "relative_gain_drift": {"max_rel_gain": 0.15, "channels": "pairs"},
    "timing_shift": {"max_shift_steps": 4.0},
    "transient": {"max_abs_spike": 4.0, "max_width_frac": 0.25},
    "persistent_degradation": {"max_abs_shift": 2.0, "max_var_scale": 0.5},
    "nuisance_gain": {"gain": 1.02},
    "nuisance_offset": {"offset_sigma": 0.05},
    "nuisance_noise": {"noise_sigma": 0.05},
}

MECHANISMS: tuple[str, ...] = (
    "level_drift",
    "relative_gain_drift",
    "timing_shift",
    "transient",
    "persistent_degradation",
)

NUISANCES: tuple[str, ...] = ("nuisance_gain", "nuisance_offset", "nuisance_noise")

#: Reserved generator families must never be synthesized here (partition).
RESERVED_FAMILIES: tuple[str, ...] = ("wrong_transition", "cross_channel_inconsistency")


def _require_shapes(
    patches: torch.Tensor,
    patch_pad_mask: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    corruption_mask: torch.Tensor,
) -> tuple[int, int, int, int]:
    if patches.ndim != 4:
        raise ValueError(f"patches must be [B, N, C, W], got {patches.shape}")
    b, n, _, w = patches.shape
    if tuple(patch_valid_mask.shape) != (b, n) or tuple(corruption_mask.shape) != (b, n):
        raise ValueError("patch_valid_mask and corruption_mask must have shape [B, N]")
    if tuple(patch_pad_mask.shape) != (b, n, w):
        raise ValueError("patch_pad_mask must have shape [B, N, W]")
    if corruption_mask.dtype is not torch.bool or patch_valid_mask.dtype is not torch.bool:
        raise ValueError("corruption_mask and patch_valid_mask must be bool")
    return b, n, patches.shape[2], w


def severity_of(value: float) -> float:
    """Validate an ordered severity (finite, non-negative)."""
    sev = float(value)
    if not (sev >= 0.0) or sev == float("inf"):
        raise ValueError("severity must be a finite non-negative float")
    return sev


def _channel_subset(gen: torch.Generator, n_channels: int, device: torch.device) -> torch.Tensor:
    """Deterministic non-empty channel subset mask [C]."""
    k = 1 + int(torch.randint(0, n_channels, (1,), generator=gen, device=device).item())
    perm = torch.randperm(n_channels, generator=gen, device=device)[:k]
    out = torch.zeros((n_channels,), dtype=torch.float32, device=device)
    out[perm.to(device)] = 1.0
    return out


def _blend(
    patches: torch.Tensor,
    patch_pad_mask: torch.Tensor,
    active: torch.Tensor,
    new_values: torch.Tensor,
) -> torch.Tensor:
    """Take ``new_values`` inside the active bool mask, keep input elsewhere."""
    out = torch.where(active.unsqueeze(-1).unsqueeze(-1).expand_as(patches),
                      new_values, patches)
    return out.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)


def _apply(
    patches: torch.Tensor,
    patch_pad_mask: torch.Tensor,
    active: torch.Tensor,
    perturbation: torch.Tensor,
) -> torch.Tensor:
    """Add ``perturbation`` on valid real steps inside the active bool mask."""
    real = (~patch_pad_mask).unsqueeze(2).to(patches.dtype)
    gain = active.unsqueeze(-1).unsqueeze(-1).to(patches.dtype)
    out = patches + perturbation * real * gain
    return out.masked_fill(patch_pad_mask.unsqueeze(2), 0.0)



def corrupt(
    name: str,
    patches: torch.Tensor,
    patch_pad_mask: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    corruption_mask: torch.Tensor,
    severity: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the corrupted view for one mechanism at one severity."""
    sev = severity_of(severity)
    b, n, c, w = _require_shapes(patches, patch_pad_mask, patch_valid_mask, corruption_mask)
    seed = generator.initial_seed() if generator is not None else 0
    try:
        gen = torch.Generator(device=patches.device).manual_seed(seed)
    except RuntimeError:
        gen = torch.Generator().manual_seed(seed)  # device without generator support
    device = patches.device
    active = (corruption_mask & patch_valid_mask).unsqueeze(-1).unsqueeze(-1).to(patches.dtype)
    if name == "level_drift":
        subset = _channel_subset(gen, c, device).reshape(1, 1, c, 1)
        ramp = torch.linspace(0.0, 1.0, w, device=device).reshape(1, 1, 1, w)
        direction = torch.sign(torch.randn((b, n, 1, 1), generator=gen, device=device))
        pert = sev * BOUNDS["level_drift"]["max_abs_offset"] * direction * subset * ramp
        pert = pert.expand(b, n, c, w)
    elif name == "relative_gain_drift":
        a = int(torch.randint(0, c, (1,), generator=gen, device=device).item())
        bb = (a + 1 + int(torch.randint(0, c - 1, (1,), generator=gen, device=device).item())) % c
        ramp = torch.linspace(0.0, 1.0, w, device=device)
        g = sev * BOUNDS["relative_gain_drift"]["max_rel_gain"]
        scale = torch.ones((b, n, c, w), dtype=patches.dtype, device=device)
        scale[:, :, a, :] *= (1.0 + g * ramp)
        scale[:, :, bb, :] *= (1.0 - g * ramp)
        return _blend(patches, patch_pad_mask, corruption_mask & patch_valid_mask,
                      (patches * scale))
    elif name == "timing_shift":
        shift = min(int(round(sev * BOUNDS["timing_shift"]["max_shift_steps"])), w - 1)
        idx = torch.arange(w, device=device).reshape(1, 1, 1, w)
        src = (idx - shift).clamp(0, w - 1).expand(b, n, c, w)
        warped = torch.gather(patches, -1, src.long())
        return _blend(patches, patch_pad_mask, corruption_mask & patch_valid_mask, warped)
    elif name == "transient":
        width = max(1, int(w * BOUNDS["transient"]["max_width_frac"] * min(sev, 1.0) + 1))
        start = int(torch.randint(0, max(1, w - width + 1), (1,), generator=gen, device=device).item())
        window = torch.zeros((w,), device=device)
        window[start:start + width] = 1.0
        subset = _channel_subset(gen, c, device).reshape(1, 1, c, 1)
        direction = torch.sign(torch.randn((b, n, 1, 1), generator=gen, device=device))
        pert = (sev * BOUNDS["transient"]["max_abs_spike"] * direction * subset
                * window.reshape(1, 1, 1, w)).expand(b, n, c, w)
    elif name == "persistent_degradation":
        cut = w // 2
        tail = torch.zeros((w,), device=device)
        tail[cut:] = 1.0
        subset = _channel_subset(gen, c, device).reshape(1, 1, c, 1)
        direction = torch.sign(torch.randn((b, n, 1, 1), generator=gen, device=device))
        noise = torch.randn((b, n, c, w), generator=gen, dtype=patches.dtype, device=device)
        pert = (sev * BOUNDS["persistent_degradation"]["max_abs_shift"] * direction * subset
                * tail.reshape(1, 1, 1, w)).expand(b, n, c, w)
        pert = pert + (sev * BOUNDS["persistent_degradation"]["max_var_scale"] * noise
                       * tail.reshape(1, 1, 1, w))
    elif name == "nuisance_gain":
        g = BOUNDS["nuisance_gain"]["gain"]
        return _blend(patches, patch_pad_mask, corruption_mask & patch_valid_mask,
                      patches * g)
    elif name == "nuisance_offset":
        sig = patches.detach().float().std().item() + 1e-8
        pert = torch.full_like(patches, BOUNDS["nuisance_offset"]["offset_sigma"] * sig)
    elif name == "nuisance_noise":
        sig = patches.detach().float().std().item() + 1e-8
        pert = (torch.randn(patches.shape, generator=gen, dtype=patches.dtype, device=device)
                * BOUNDS["nuisance_noise"]["noise_sigma"] * sig)
    else:
        raise ValueError(f"unknown mechanism {name!r}")
    if name in ("level_drift", "transient", "persistent_degradation",
                "nuisance_offset", "nuisance_noise"):
        return _apply(patches, patch_pad_mask, corruption_mask & patch_valid_mask, pert)
    raise ValueError(f"unhandled mechanism {name!r}")
