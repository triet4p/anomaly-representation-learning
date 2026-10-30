"""Sprint 24 Task 7-M16 paired-twin constructor checks (consumer-visible only).

Covers the public paired-twin contract on tiny synthetic inputs: shared
pre-injection source and identity, legacy anomaly-path equivalence, signal
dependent noise disclosure, duration-family within-family pairing, sweeps,
retry-selected source, normal-return/missing-event and error boundaries.
No official corpus, inference, forwards, training, refit, remote or scale.
"""

from __future__ import annotations

import pytest

from synth.config import SynthConfig
from synth.generator import SessionGenerator, derive_pair_seed
from synth.schema import AnomalyFamily, SampleLabel


@pytest.fixture(scope="module")
def gen() -> SessionGenerator:
    return SessionGenerator(SynthConfig())


def test_pair_seed_derivation_matches_m15_domain() -> None:
    assert derive_pair_seed(53101, 0) == derive_pair_seed(53101, 0)
    assert len({derive_pair_seed(53101, index) for index in range(96)}) == 96
    assert derive_pair_seed(53101, 1) != derive_pair_seed(53102, 1)


def test_twin_shares_source_identity_with_variant(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    assert pair.disposition == "PAIRED"
    assert pair.twin.file_label == SampleLabel.NORMAL
    assert pair.abnormal.file_label == SampleLabel.ABNORMAL
    assert pair.twin.T == 450
    assert pair.abnormal.T == pair.twin.T
    assert (pair.twin.robot_idx, pair.twin.program_idx) == (pair.abnormal.robot_idx, pair.abnormal.program_idx)
    assert pair.twin.robot_code == pair.abnormal.robot_code
    assert pair.twin.program_number == pair.abnormal.program_number
    assert [(item.start, item.end) for item in pair.twin.regime_sequence] == [
        (item.start, item.end) for item in pair.abnormal.regime_sequence
    ]
    assert pair.group_id == f"s24p-paired-twin-v1:{pair.pair_seed:010d}"
    assert pair.abnormal.anomaly_meta is not None
    assert pair.abnormal.anomaly_meta.extra["pair_seed"] == pair.pair_seed
    assert pair.abnormal.anomaly_meta.extra["group_id"] == pair.group_id
    assert pair.abnormal.anomaly_meta.extra["source_id"] == pair.source_id
    assert pair.abnormal.anomaly_mask is not None
    assert pair.abnormal.anomaly_mask.shape == pair.abnormal.x.shape
    assert pair.twin.anomaly_meta is None
    assert pair.twin.anomaly_mask is None


def test_twin_matches_legacy_anomaly_path_outputs(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    legacy = gen.generate_anomaly(seed=42401, split="test", family=AnomalyFamily.SUBTLE_DRIFT, severity=0.6)
    assert pair.abnormal.file_id == legacy.file_id
    assert (pair.abnormal.anomaly_meta.extra["strength"], pair.abnormal.anomaly_meta.extra["n_attempts"]) == (
        legacy.anomaly_meta.extra["strength"], legacy.anomaly_meta.extra["n_attempts"]
    )


def test_twin_replay_is_deterministic(gen: SessionGenerator) -> None:
    first = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    second = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    assert (first.twin.x == second.twin.x).all()
    assert (first.abnormal.x == second.abnormal.x).all()
    assert first.source_id == second.source_id
    assert first.twin.file_id == second.twin.file_id


def test_severity_sweep_shares_one_source(gen: SessionGenerator) -> None:
    sweep = gen.generate_paired_severity_sweep(42401, AnomalyFamily.SUBTLE_DRIFT, [0.35, 0.60, 0.85])
    twins = [pair.twin for pair in sweep]
    assert {item.T for item in twins} == {450}
    assert {(item.robot_idx, item.program_idx) for item in twins} == {(3, 1)}
    assert {(item.start, item.end) for twin in twins for item in twin.regime_sequence} == {
        (item.start, item.end) for item in twins[0].regime_sequence
    }
    assert {pair.source_id for pair in sweep} == {sweep[0].source_id}
    assert (twins[0].x == twins[1].x).all()


def test_family_sweep_shares_one_source(gen: SessionGenerator) -> None:
    sweep = gen.generate_paired_family_sweep(
        42401, [AnomalyFamily.SUBTLE_DRIFT, AnomalyFamily.REALISTIC_STUCK], 0.6
    )
    assert {pair.source_id for pair in sweep} == {sweep[0].source_id}
    assert (sweep[0].twin.x == sweep[1].twin.x).all()
    assert sweep[0].abnormal.anomaly_meta.family != sweep[1].abnormal.anomaly_meta.family


def test_retry_pairs_selected_source(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42447, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    legacy = gen.generate_anomaly(seed=42447, split="test", family=AnomalyFamily.SUBTLE_DRIFT, severity=0.6)
    assert pair.n_attempts == legacy.anomaly_meta.extra["n_attempts"] == 2
    assert (pair.abnormal.x == legacy.x).all()


def test_duration_pair_is_within_family_only(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42401, AnomalyFamily.DURATION_ANOMALY, 0.6)
    assert pair.disposition == "PAIRED"
    assert pair.abnormal.T != pair.twin.T
    assert (pair.abnormal.robot_idx, pair.abnormal.program_idx) == (pair.twin.robot_idx, pair.twin.program_idx)
    assert "within-family only" in pair.abnormal.anomaly_meta.extra["duration_pairing_note"]


def test_missing_event_normal_return_has_no_fake_success(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42471, AnomalyFamily.MISSING_EVENT, 0.6)
    assert pair.disposition == "NORMAL_RETURN"
    assert pair.abnormal.file_label == SampleLabel.NORMAL
    assert pair.abnormal.anomaly_meta is None
    assert pair.abnormal.rejection_meta is not None
    assert pair.abnormal.rejection_meta.extra["disposition"] == "NORMAL_RETURN"
    assert pair.abnormal.rejection_meta.extra["n_attempts"] == pair.n_attempts


def test_normal_path_is_not_the_twin(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    assert pair.normal_path_diverges is True
    probe = gen.generate_normal(seed=42401, split="test")
    assert probe.T != pair.twin.T


def test_invalid_inputs_fail_closed(gen: SessionGenerator) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gen.generate_paired_twin(-1, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    with pytest.raises(ValueError, match="Unknown anomaly family"):
        gen.generate_paired_twin(7, "nope", 0.6)
    with pytest.raises(ValueError, match=r"severity must be in \[0, 1\]"):
        gen.generate_paired_twin(7, AnomalyFamily.SUBTLE_DRIFT, 9.0)
    with pytest.raises(ValueError, match="at least one"):
        gen.generate_paired_family_sweep(7, [], 0.6)
    with pytest.raises(ValueError, match="at least one"):
        gen.generate_paired_severity_sweep(7, AnomalyFamily.SUBTLE_DRIFT, [])


def test_encoder_inputs_stay_label_free(gen: SessionGenerator) -> None:
    pair = gen.generate_paired_twin(42401, AnomalyFamily.SUBTLE_DRIFT, 0.6)
    for sample in (pair.twin, pair.abnormal):
        inputs = sample.encoder_inputs()
        assert "anomaly_meta" not in inputs
        assert "anomaly_mask" not in inputs
        assert "rejection_meta" not in inputs
        sample.validate()
