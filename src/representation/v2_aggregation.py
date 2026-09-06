"""Distribution-preserving file-state aggregation (Sprint 11 Task 13).

Replace plain mean-only population representation with a distributional
contract over valid patches: energy quantiles, validation-calibrated
upper-tail statistics, elevated-patch fraction, regime summaries,
transition/cross-channel statistics, and duration/persistence. A plain
mean is retained only as one auxiliary field among many — it is never
the sole representation.

All statistics consume valid patches only (``patch_valid_mask``); rows
with no valid patches yield finite all-zero summaries instead of NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from representation.v2_contracts import validate_file_state

#: Default quantile levels: median, upper quartile, and calibrated tail.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 1.0)


def calibrate_elevated_threshold(
    healthy_energies: torch.Tensor,
    healthy_valid_mask: torch.Tensor,
    *,
    tail_probability: float = 0.05,
) -> float:
    """Return the healthy-validation energy threshold for elevation.

    Only verified-healthy validation energies enter this calibration;
    test anomalies must never calibrate aggregation choices. The
    threshold is the ``(1 - tail_probability)`` quantile of the pooled
    valid healthy energies.
    """
    if healthy_energies.ndim != 2 or healthy_valid_mask.shape != healthy_energies.shape:
        raise ValueError("healthy_energies and healthy_valid_mask must share [B, N] shape")
    if healthy_valid_mask.dtype is not torch.bool:
        raise ValueError("healthy_valid_mask must have torch.bool dtype")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must be in (0, 1)")
    pooled = healthy_energies[healthy_valid_mask].detach().to(dtype=torch.float64)
    if pooled.numel() == 0:
        raise ValueError("calibration requires at least one valid healthy energy")
    if not torch.isfinite(pooled).all():
        raise ValueError("healthy calibration energies must be finite")
    return float(torch.quantile(pooled, 1.0 - tail_probability).item())


def _row_quantiles(valid: torch.Tensor, levels: Sequence[float]) -> torch.Tensor:
    if valid.numel() == 0:
        return torch.zeros((len(levels),), dtype=torch.float32)
    q = torch.quantile(
        valid.to(dtype=torch.float64),
        torch.tensor(list(levels), dtype=torch.float64),
    )
    return q.to(dtype=torch.float32)


def aggregate_file_state(
    patch_latents: torch.Tensor,
    patch_energy: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    regime_ids: torch.Tensor,
    *,
    top_q_fraction: float = 0.1,
    elevated_threshold: float = 3.0,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    n_regimes: int = 7,
    channel_energy: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate per-patch geometry into a distributional file state.

    Args:
        patch_latents: ``[B, N, D]`` patch latents, zero on invalid patches.
        patch_energy: ``[B, N]`` signed finite conditional energies.
        patch_valid_mask: ``[B, N]`` bool validity (variable-length) mask.
        regime_ids: ``[B, N]`` long operating-regime ids per patch.
        top_q_fraction: fraction of valid patches forming the upper tail.
        elevated_threshold: validation-calibrated energy elevation cutoff.
        quantile_levels: non-decreasing quantile levels in ``[0, 1]``.
        n_regimes: regime vocabulary size for per-regime summaries.
        channel_energy: optional ``[B, N, C]`` per-channel energies for
            cross-channel consistency; when absent a latent-dispersion
            proxy is reported instead (see notes).
    """
    if patch_latents.ndim != 3:
        raise ValueError(f"patch_latents must be [B, N, D], got {tuple(patch_latents.shape)}")
    b, n, d = patch_latents.shape
    if tuple(patch_energy.shape) != (b, n) or tuple(patch_valid_mask.shape) != (b, n):
        raise ValueError("patch_energy/patch_valid_mask must match [B, N] of patch_latents")
    if tuple(regime_ids.shape) != (b, n):
        raise ValueError("regime_ids must have shape [B, N]")
    if patch_valid_mask.dtype is not torch.bool:
        raise ValueError("patch_valid_mask must have torch.bool dtype")
    if not 0.0 < top_q_fraction <= 1.0:
        raise ValueError("top_q_fraction must be in (0, 1]")
    levels = tuple(float(q) for q in quantile_levels)
    if not levels or any(not 0.0 <= q <= 1.0 for q in levels):
        raise ValueError("quantile_levels must be non-empty values in [0, 1]")
    if any(a_ > b_ for a_, b_ in zip(levels, levels[1:])):
        raise ValueError("quantile_levels must be non-decreasing")
    if channel_energy is not None and tuple(channel_energy.shape[:2]) != (b, n):
        raise ValueError("channel_energy must have shape [B, N, C]")

    latents = patch_latents.detach().to(dtype=torch.float32)
    energies = patch_energy.detach().to(dtype=torch.float32)
    if not torch.isfinite(latents).all():
        raise ValueError("patch_latents must be finite")
    if not torch.isfinite(energies).all():
        raise ValueError("patch_energy must be finite")

    device = latents.device
    q_count = len(levels)
    file_state = torch.zeros((b, d), dtype=torch.float32, device=device)
    quantiles = torch.zeros((b, q_count), dtype=torch.float32, device=device)
    tail_energy = torch.zeros((b,), dtype=torch.float32, device=device)
    elevated_fraction = torch.zeros((b,), dtype=torch.float32, device=device)
    mean_energy = torch.zeros((b,), dtype=torch.float32, device=device)
    regime_mean_energy = torch.zeros((b, n_regimes), dtype=torch.float32, device=device)
    regime_presence = torch.zeros((b, n_regimes), dtype=torch.bool, device=device)
    transition_count = torch.zeros((b,), dtype=torch.float32, device=device)
    transition_jump = torch.zeros((b,), dtype=torch.float32, device=device)
    cross_channel_spread = torch.zeros((b,), dtype=torch.float32, device=device)
    max_run_fraction = torch.zeros((b,), dtype=torch.float32, device=device)
    n_elevated = torch.zeros((b,), dtype=torch.float32, device=device)
    n_valid = torch.zeros((b,), dtype=torch.float32, device=device)

    for i in range(b):
        valid_idx = torch.where(patch_valid_mask[i])[0]
        count = int(valid_idx.numel())
        n_valid[i] = float(count)
        if count == 0:
            continue
        valid_e = energies[i, valid_idx]
        valid_z = latents[i, valid_idx]
        file_state[i] = valid_z.mean(dim=0)
        mean_energy[i] = valid_e.mean()
        quantiles[i] = _row_quantiles(valid_e, levels)
        top_k = max(1, int(torch.ceil(torch.tensor(top_q_fraction * count)).item()))
        tail_energy[i] = torch.topk(valid_e, top_k).values.mean()
        elevated = valid_e >= float(elevated_threshold)
        n_elevated[i] = float(elevated.sum().item())
        elevated_fraction[i] = float(elevated.sum().item()) / count

        row_regimes = regime_ids[i, valid_idx].to(dtype=torch.long)
        for r in range(n_regimes):
            sel = valid_e[row_regimes == r]
            if sel.numel() > 0:
                regime_mean_energy[i, r] = sel.mean()
                regime_presence[i, r] = True

        if count >= 2:
            # Consecutive *valid* patches in index order; invalid gaps break
            # adjacency only insofar as non-consecutive indices still compare
            # neighboring surviving patches (documented, deterministic).
            diffs = valid_e[1:] - valid_e[:-1]
            transition_jump[i] = diffs.abs().max()
            reg = row_regimes
            transition_count[i] = float((reg[1:] != reg[:-1]).sum().item())

        if channel_energy is not None:
            chan = channel_energy[i, valid_idx].detach().to(dtype=torch.float32)
            spread = (chan.max(dim=-1).values - chan.min(dim=-1).values).mean()
            cross_channel_spread[i] = spread if torch.isfinite(spread) else torch.zeros(())
        else:
            # Latent-dispersion proxy: mean per-patch std across latent dims.
            # Honest fallback only — real cross-channel evidence requires
            # per-channel energies from the caller.
            cross_channel_spread[i] = (
                valid_z.std(dim=-1).mean() if d > 1 else torch.zeros(())
            )

        # Duration/persistence: longest index-contiguous run of valid patches
        # that are all elevated, as a fraction of valid-patch count. Sparse
        # single-patch anomalies score 1/count (survive via tail/quantiles);
        # sustained events approach 1.0.
        full = torch.zeros((n,), dtype=torch.bool, device=device)
        full[valid_idx[elevated]] = True
        best = cur = 0
        for flag in full.tolist():
            cur = cur + 1 if flag else 0
            best = max(best, cur)
        max_run_fraction[i] = float(best) / count

    state: dict[str, torch.Tensor] = {
        "file_state": file_state,
        "energy_quantiles": quantiles,
        "tail_energy": tail_energy,
        "elevated_fraction": elevated_fraction,
        "patch_valid_mask": patch_valid_mask,
        # Auxiliary detail beyond the boundary contract: the plain mean is
        # present but never sufficient on its own.
        "mean_energy": mean_energy,
        "regime_mean_energy": regime_mean_energy,
        "regime_presence": regime_presence,
        "transition_count": transition_count,
        "transition_jump": transition_jump,
        "cross_channel_spread": cross_channel_spread,
        "max_run_fraction": max_run_fraction,
        "n_elevated": n_elevated,
        "n_valid": n_valid,
    }
    validate_file_state(state)
    return state
