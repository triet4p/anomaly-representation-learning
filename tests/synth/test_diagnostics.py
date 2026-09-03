import numpy as np

from synth.config import PatchConfig
from synth.diagnostics import compute_weak_baseline_auc
from synth.schema import AnomalyFamily, AnomalyMeta, FileSample, SampleLabel


def _sample(x, abnormal=False, index=0):
    return FileSample(
        x=np.asarray(x, dtype=np.float32),
        file_id=str(index),
        file_label=SampleLabel.ABNORMAL if abnormal else SampleLabel.NORMAL,
        seed=index,
        generator_version="test",
        config_hash="test",
        regime_sequence=[],
        anomaly_meta=(AnomalyMeta(AnomalyFamily.REALISTIC_STUCK, 2, 6, 0.5, [0, 1, 2])
                      if abnormal else None),
    )


def test_weak_baseline_reports_separability_for_obvious_shift():
    normal = [_sample(np.zeros((3, 16)), index=i) for i in range(6)]
    abnormal = [_sample(np.full((3, 16), 10.0), abnormal=True, index=i + 10) for i in range(6)]
    result = compute_weak_baseline_auc(normal, abnormal, PatchConfig())
    assert result["overall_separability"] > 0.95
    assert "realistic_stuck" in result["too_easy_families"]
