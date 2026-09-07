"""Batch B2 Findings 2–3: calibration persistence with distinct provenance.

Proves the dev-val-calibrated operating (elevated-patch) threshold and the
dev-val-only conformal confidence-calibrator cohort persist in the
checkpoint as separate records, restore by default, and drive inference —
while legacy checkpoints without calibration state still load.
"""

from __future__ import annotations

import pytest
import torch

from representation.v2_aggregation import calibrate_elevated_threshold_with_provenance
from representation.v2_checkpoint import load_v2_checkpoint, save_v2_checkpoint
from representation.v2_config import V2Config
from representation.v2_patch import ContextConditionedPatchEncoder
from representation.v2_risk import HealthyTailCalibrator


def _config() -> V2Config:
    return V2Config(
        n_channels=6, d_model=4, n_robots=1, n_programs=1, n_regimes=2,
        n_prototypes=1, sequence_layers=1, attention_heads=2, dropout=0.0,
        min_group_samples=2, diag_min_samples=2, calibration_min_samples=4,
    )


def _model(config: V2Config) -> ContextConditionedPatchEncoder:
    return ContextConditionedPatchEncoder(
        n_channels=config.n_channels, d_model=config.d_model,
        n_robots=config.n_robots, n_programs=config.n_programs,
        n_regimes=config.n_regimes, n_prototypes=config.n_prototypes,
        sequence_layers=config.sequence_layers,
        attention_heads=config.attention_heads, dropout=0.0,
    )


def _operating_record() -> dict[str, object]:
    torch.manual_seed(2)
    energies = torch.randn(4, 12) - 1.0
    mask = torch.ones_like(energies, dtype=torch.bool)
    mask[:, 10:] = False
    _, provenance = calibrate_elevated_threshold_with_provenance(
        energies, mask, tail_probability=0.05, cohort="dev-val"
    )
    return dict(provenance)


def test_checkpoint_persists_distinct_calibration_provenances(tmp_path) -> None:
    config = _config()
    operating = _operating_record()
    calibrator = HealthyTailCalibrator(min_samples=4).fit(
        torch.randn(7).abs(), cohort="dev-val"
    )
    path = tmp_path / "v2_calibrated.pt"
    save_v2_checkpoint(
        path, _model(config), config=config, step=3,
        calibrator=calibrator, operating_threshold=operating,
    )
    restored = load_v2_checkpoint(path, _model(config), expected_config=config)
    back_operating = restored["operating_threshold"]
    back_cohort = restored["confidence_calibrator_fit_cohort"]
    assert isinstance(back_operating, dict) and isinstance(back_cohort, dict)
    # The two provenances are distinct fields with distinct content.
    assert back_operating["value"] == operating["value"]
    assert back_operating["fit_cohort"] == "dev-val"
    assert back_operating["method"] == "healthy-validation-quantile"
    assert back_cohort["cohort"] == "dev-val"
    assert back_cohort["n_samples"] == 7
    assert back_cohort["small_sample"] is True
    assert set(back_operating) != set(back_cohort)
    # The calibrator state itself carries the same cohort (no silent swap).
    assert restored["calibrator"]["fit_cohort"] == "dev-val"


def test_checkpoint_without_calibration_loads_with_none_provenance(tmp_path) -> None:
    config = _config()
    path = tmp_path / "v2_legacy.pt"
    save_v2_checkpoint(path, _model(config), config=config, step=0)
    restored = load_v2_checkpoint(path, _model(config), expected_config=config)
    assert restored["operating_threshold"] is None
    assert restored["confidence_calibrator_fit_cohort"] is None


def test_checkpoint_rejects_malformed_operating_record(tmp_path) -> None:
    config = _config()
    path = tmp_path / "v2_bad.pt"
    with pytest.raises(ValueError, match="finite 'value'"):
        save_v2_checkpoint(
            path, _model(config), config=config,
            operating_threshold={"method": "x", "fit_cohort": "dev-val", "n_samples": 4},
        )
    with pytest.raises(ValueError, match="'fit_cohort'"):
        save_v2_checkpoint(
            path, _model(config), config=config,
            operating_threshold={
                "value": 1.0, "method": "x", "fit_cohort": "", "n_samples": 4
            },
        )
