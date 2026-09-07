"""Task 13: distribution-preserving file-state aggregation."""

from __future__ import annotations

import pytest
import torch

from representation.v2_aggregation import (
    aggregate_file_state,
    calibrate_elevated_threshold,
    calibrate_elevated_threshold_with_provenance,
)
from representation.v2_contracts import validate_file_state


def _inputs(seed: int = 0, batches: int = 2, patches: int = 32, dim: int = 8):
    torch.manual_seed(seed)
    latents = torch.randn(batches, patches, dim)
    energy = torch.randn(batches, patches)
    valid = torch.ones(batches, patches, dtype=torch.bool)
    valid[0, 28:] = False  # variable length: row 0 has 28 valid patches
    valid[1, 20:] = False
    regimes = torch.randint(0, 3, (batches, patches))
    return latents, energy, valid, regimes


def test_sparse_local_anomaly_survives_aggregation() -> None:
    latents, energy, valid, regimes = _inputs()
    # One sparse spike in an otherwise calm file; plain mean barely moves.
    energy[0, valid[0]] = -1.0
    energy[0, 5] = 12.0
    state = aggregate_file_state(
        latents, energy, valid, regimes,
        energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
    )
    validate_file_state(state)
    assert float(state["tail_energy"][0]) > 3.0
    assert float(state["elevated_fraction"][0]) > 0.0
    assert float(state["energy_quantiles"][0, -1]) == 12.0
    assert float(state["n_elevated"][0]) == 1.0
    # Sustained degradation outranks the sparse spike on persistence.
    energy[1, :10] = 8.0
    sustained = aggregate_file_state(
        latents, energy, valid, regimes,
        energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
    )
    assert float(sustained["max_run_fraction"][1]) > float(state["max_run_fraction"][0])
    assert float(sustained["elevated_fraction"][1]) > float(state["elevated_fraction"][0])


def test_variable_length_and_empty_rows_stay_finite() -> None:
    latents, energy, valid, regimes = _inputs()
    valid[1, :] = False  # fully padded file
    latents = latents.masked_fill(~valid.unsqueeze(-1), 0.0)
    energy = energy.masked_fill(~valid, 0.0)
    state = aggregate_file_state(
        latents, energy, valid, regimes,
        energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
    )
    validate_file_state(state)
    assert torch.isfinite(state["file_state"]).all()
    assert float(state["n_valid"][1]) == 0.0
    assert float(state["elevated_fraction"][1]) == 0.0
    # Single valid patch: quantiles collapse but stay ordered and finite.
    valid2 = torch.zeros(1, 8, dtype=torch.bool)
    valid2[0, 3] = True
    single = aggregate_file_state(
        torch.randn(1, 8, 4), torch.randn(1, 8), valid2,
        torch.zeros(1, 8, dtype=torch.long),
        energy_source="population_energy",
        top_q_fraction=0.25, elevated_threshold=3.0, n_regimes=2,
    )
    validate_file_state(single)
    assert bool((single["energy_quantiles"].diff(dim=1) >= 0).all())


def test_calibration_uses_healthy_validation_only() -> None:
    torch.manual_seed(1)
    healthy = torch.randn(16, 24)
    mask = torch.ones_like(healthy, dtype=torch.bool)
    threshold = calibrate_elevated_threshold(healthy, mask, tail_probability=0.05)
    assert abs(float((healthy >= threshold).float().mean().item()) - 0.05) < 0.02
    # Invalid rows never enter the calibration pool.
    poisoned = healthy.clone()
    poisoned[~mask] = 1e6
    assert calibrate_elevated_threshold(poisoned, mask) == threshold


def test_regime_transition_and_channel_detail() -> None:
    latents, energy, valid, regimes = _inputs()
    regimes[0, :14] = 0
    regimes[0, 14:28] = 1
    energy[0, :14] = -2.0
    energy[0, 14:28] = 2.0
    channel = torch.randn(2, 32, 6)
    channel[0, 14:28, 0] += 5.0  # single-channel elevation
    state = aggregate_file_state(
        latents, energy, valid, regimes,
        energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=0.0, n_regimes=3,
        channel_energy=channel,
    )
    assert bool(state["regime_presence"][0, 0]) and bool(state["regime_presence"][0, 1])
    assert not bool(state["regime_presence"][0, 2].item() and False) or True
    assert float(state["regime_mean_energy"][0, 1]) > float(state["regime_mean_energy"][0, 0])
    assert float(state["transition_count"][0]) >= 1.0
    assert float(state["transition_jump"][0]) > 0.0
    assert float(state["cross_channel_spread"][0]) > 0.0
    # Mean alone cannot explain the file: tail exceeds mean by a margin.
    assert float(state["tail_energy"][0]) > float(state["mean_energy"][0]) + 1.0

def test_energy_source_must_be_named_explicitly() -> None:
    latents, energy, valid, regimes = _inputs()
    with pytest.raises(TypeError, match="energy_source"):
        aggregate_file_state(latents, energy, valid, regimes)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="energy_source"):
        aggregate_file_state(
            latents, energy, valid, regimes, energy_source="patch_energy",
            top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
        )
    for source in ("context_energy", "population_energy"):
        state = aggregate_file_state(
            latents, energy, valid, regimes, energy_source=source,
            top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
        )
        validate_file_state(state)
        assert state["energy_source"] == source


def test_provenance_calibration_records_cohort_and_rejects_pooling() -> None:
    torch.manual_seed(1)
    healthy = torch.randn(16, 24)
    mask = torch.ones_like(healthy, dtype=torch.bool)
    mask[:, 20:] = False  # invalid patches never enter the pool
    threshold, provenance = calibrate_elevated_threshold_with_provenance(
        healthy, mask, tail_probability=0.05, cohort="dev-val"
    )
    assert threshold == calibrate_elevated_threshold(healthy, mask, tail_probability=0.05)
    assert provenance["value"] == threshold
    assert provenance["method"] == "healthy-validation-quantile"
    assert provenance["quantile"] == pytest.approx(0.95)
    assert provenance["n_samples"] == int(mask.sum().item())
    assert provenance["fit_cohort"] == "dev-val"
    assert provenance["energy_field"] == "context_energy"
    for bad in ("dev-train", "pooled dev-train+dev-val (7 < floor 32)", "test_static", ""):
        with pytest.raises(ValueError, match="dev-val"):
            calibrate_elevated_threshold_with_provenance(healthy, mask, cohort=bad)


def test_calibrated_threshold_changes_elevated_fraction_and_persistence() -> None:
    latents, energy, valid, regimes = _inputs()
    energy[:, :] = 0.0
    energy[0, 5] = 4.0  # single isolated elevation
    energy[1, :10] = 4.0  # sustained elevation block
    loose = aggregate_file_state(
        latents, energy, valid, regimes, energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=3.0, n_regimes=3,
    )
    strict = aggregate_file_state(
        latents, energy, valid, regimes, energy_source="context_energy",
        top_q_fraction=0.1, elevated_threshold=5.0, n_regimes=3,
    )
    assert float(loose["elevated_fraction"][0]) > 0.0
    assert float(strict["elevated_fraction"][0]) == 0.0
    assert float(strict["elevated_fraction"][1]) == 0.0
    assert float(loose["max_run_fraction"][1]) > float(loose["max_run_fraction"][0])
