import pytest

from synth.config import AnomalyConfig, MaskingConfig, PatchConfig, RegimeConfig, SynthConfig


def test_invalid_regime_range_fails_early():
    with pytest.raises(ValueError, match="active_range"):
        SynthConfig(regime=RegimeConfig(active_range=(80, 20)))


def test_invalid_anomaly_range_fails_early():
    with pytest.raises(ValueError, match="wt_speed_range"):
        SynthConfig(anomaly=AnomalyConfig(wt_speed_range=(0.8, 1.2)))


def test_invalid_masking_and_patch_parameters_fail_early():
    with pytest.raises(ValueError, match="composition"):
        SynthConfig(masking=MaskingConfig(random_fraction=0.8, info_fraction=0.8, block_fraction=0.0))
    with pytest.raises(ValueError, match="patch_size"):
        SynthConfig(patch=PatchConfig(patch_size=0))
