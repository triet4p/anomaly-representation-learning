"""Task 9: V2 configuration and observable contracts."""

from __future__ import annotations

import pytest
import torch

from representation.v2_config import (
    FALLBACK_ORDER,
    assert_encoder_inputs_clean,
    V2Config,
)
from representation.v2_contracts import (
    validate_confidence,
    validate_context_patch_output,
    validate_file_state,
    validate_population_patch_output,
    validate_risk,
    validate_trajectory,
)


def test_fallback_order_never_program_only() -> None:
    assert FALLBACK_ORDER == (
        "robot_program_regime",
        "robot_program",
        "robot",
        "fleet_low_confidence",
    )
    assert not any("program" == level for level in FALLBACK_ORDER)
    assert V2Config().fallback_order() == FALLBACK_ORDER


def test_config_forbids_program_only_fallback() -> None:
    with pytest.raises(ValueError, match="program-only"):
        V2Config(allow_program_only_fallback=True)


def test_config_validates_geometry_and_risk_fields() -> None:
    with pytest.raises(ValueError, match="stride"):
        V2Config(patch_size=8, stride=16)
    with pytest.raises(ValueError, match="divisible"):
        V2Config(d_model=7, attention_heads=4)
    with pytest.raises(ValueError, match="diag_min_samples"):
        V2Config(min_group_samples=32, diag_min_samples=8)
    with pytest.raises(ValueError, match="risk_horizons"):
        V2Config(risk_horizons_days=(1, 3))
    cfg = V2Config()
    assert cfg.to_dict()["risk_horizons_days"] == [1, 7]


def test_encoder_leakage_guard_rejects_supervision() -> None:
    with pytest.raises(ValueError, match="leak"):
        assert_encoder_inputs_clean({"robot_idx": 0, "anomaly_mask": 1})
    with pytest.raises(ValueError, match="leak"):
        assert_encoder_inputs_clean({"x": 0, "future_targets": 1})
    assert_encoder_inputs_clean({"robot_idx": 0, "regime_ids": 1})


def test_patch_output_contract_accepts_valid_rejects_leak_shapes() -> None:
    latents = torch.zeros(2, 3, 4)
    energy = torch.tensor([[0.5, 0.0, 1.0], [0.2, 0.3, 0.0]])
    valid = torch.tensor([[True, False, True], [True, True, False]])
    validate_context_patch_output(
        {"patch_latents": latents, "context_energy": energy, "patch_valid_mask": valid}
    )
    # Continuous Gaussian/mixture NLL is signed: tight normal densities score
    # below zero on valid patches without changing invalid-patch semantics.
    negative = torch.tensor([[-2.5, 0.0, -0.25], [-1.0, -3.75, 0.0]])
    validate_context_patch_output(
        {"patch_latents": latents, "context_energy": negative, "patch_valid_mask": valid}
    )
    bad_energy = energy.clone()
    bad_energy[0, 1] = 2.0
    with pytest.raises(ValueError, match="invalid patches"):
        validate_context_patch_output(
            {"patch_latents": latents, "context_energy": bad_energy, "patch_valid_mask": valid}
        )
    # Cross-field presence is rejected: a context output carrying population
    # energy (and vice versa) fails validation.
    with pytest.raises(ValueError, match="distinct signals"):
        validate_context_patch_output(
            {"patch_latents": latents, "context_energy": energy,
             "population_energy": energy, "patch_valid_mask": valid}
        )
    with pytest.raises(ValueError, match="distinct signals"):
        validate_population_patch_output(
            {"population_energy": energy, "context_energy": energy,
             "patch_valid_mask": valid}
        )


def test_file_trajectory_confidence_risk_contracts() -> None:
    validate_file_state(
        {
            "file_state": torch.zeros(2, 4),
            "energy_quantiles": torch.tensor([[0.1, 0.5, 0.9], [0.0, 0.2, 0.4]]),
            "tail_energy": torch.tensor([0.9, 0.4]),
            "elevated_fraction": torch.tensor([0.2, 0.0]),
            "patch_valid_mask": torch.ones(2, 5, dtype=torch.bool),
        }
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_file_state(
            {
                "file_state": torch.zeros(1, 4),
                "energy_quantiles": torch.tensor([[0.9, 0.1]]),
                "tail_energy": torch.tensor([0.5]),
                "elevated_fraction": torch.tensor([0.1]),
                "patch_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            }
        )
    validate_trajectory(
        {
            "displacement": torch.tensor([1.0]),
            "velocity": torch.tensor([0.2]),
            "trend": torch.tensor([-0.1]),
            "persistence": torch.tensor([0.0]),
        }
    )
    validate_confidence({"confidence": torch.tensor([0.9]), "p_normal": torch.tensor([0.1])})
    validate_risk({"risk_1d": torch.tensor([0.2]), "risk_7d": torch.tensor([0.5])})
    with pytest.raises(ValueError, match="at least risk_1d"):
        validate_risk({"risk_1d": torch.tensor([0.8]), "risk_7d": torch.tensor([0.1])})
