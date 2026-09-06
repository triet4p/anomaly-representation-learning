"""Focused Task 1 tests: factory calendar config and event schema.

Covers accepted minimal/full records, validation failures, deterministic
serialization round trips, legacy compatibility, and the encoder-input
separation contract. Scheduler, health simulation, and split assignment
are later tasks and are NOT tested here.
"""
from __future__ import annotations

import pytest
import numpy as np

from synth.config import FactoryCalendarConfig, SynthConfig
from synth.dataset import _sample_bytes, load_sample_bytes
from synth.schema import (
    ALLOWED_SPLIT_VIEWS,
    DIAGNOSTIC_ONLY_FIELD_NAMES,
    MODEL_INPUT_FIELD_NAMES,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    AnomalyFamily,
    AnomalyMeta,
    DegradationStage,
    EpisodeKind,
    FactoryProvenance,
    FileSample,
    FutureFailureTargets,
    HealthEpisode,
    ObservableAnomalyLabels,
    OperatingContext,
    OperationEvent,
    RegimeMeta,
    RegimeType,
    RobotHealthState,
    SampleLabel,
    SplitProvenance,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _operation(**overrides) -> OperationEvent:
    fields: dict[str, object] = {
        "operation_id": "op-0001",
        "unit_id": "unit-0001",
        "product_type": "sedan",
        "route_id": "route-A",
        "route_position": 2,
        "robot_id": "robot-03",
        "program_id": "prog-weld-7",
        "arrival_time": 1000.0,
        "start_time": 1030.0,
        "end_time": 1150.0,
        "duration": 120.0,
        "queue_delay": 30.0,
        "travel_time": 45.0,
        "idle_before": 12.0,
    }
    fields.update(overrides)
    return OperationEvent(**fields)  # type: ignore[arg-type]


def _regimes(timesteps: int = 64) -> list[RegimeMeta]:
    return [RegimeMeta(RegimeType.ACTIVE, 0, timesteps, 0.8)]


def _signal(channels: int = 6, timesteps: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((channels, timesteps)).astype(np.float32)


def _full_sample(label: SampleLabel = SampleLabel.NORMAL) -> FileSample:
    x = _signal()
    mask = None
    meta = None
    if label == SampleLabel.ABNORMAL:
        mask = np.zeros_like(x, dtype=bool)
        mask[:, 8:12] = True
        meta = AnomalyMeta(
            family=AnomalyFamily.SUBTLE_DRIFT, start=8, end=12,
            severity=0.6, affected_channels=[0, 1],
        )
    return FileSample(
        x=x,
        file_id="factory-file-0001",
        file_label=label,
        seed=11,
        generator_version="test",
        config_hash="abc123",
        regime_sequence=_regimes(x.shape[1]),
        anomaly_meta=meta,
        anomaly_mask=mask,
        robot_idx=3,
        program_idx=7,
        robot_code="R03",
        program_number="P107",
        operation=_operation(),
        operating_context=OperatingContext(
            shift="night", load=1.1, ambient_temp_c=27.5),
        health=RobotHealthState(
            health_value=0.42, program_sensitivity=1.5, manifested_value=0.63,
            degradation_stage=DegradationStage.INTERMITTENT,
            degradation_severity=0.3, degradation_episode_id="deg-09"),
        episode=HealthEpisode(
            episode_id="deg-09", kind=EpisodeKind.DEGRADATION,
            robot_id="robot-03", start_time=900.0, end_time=None),
        anomaly_labels=ObservableAnomalyLabels(
            is_file_anomalous=(label == SampleLabel.ABNORMAL),
            anomaly_family=("subtle_drift"
                            if label == SampleLabel.ABNORMAL else None),
            anomaly_severity=(0.6 if label == SampleLabel.ABNORMAL else None),
        ),
        future_targets=FutureFailureTargets(
            time_to_next_failure=3.0 * SECONDS_PER_DAY,
            failure_within_1d=False, failure_within_7d=True, is_censored=False),
        split_provenance=SplitProvenance(
            cutoff_time=5_184_000.0, is_quarantined=False,
            member_views=("test-temporal",)),
        factory_provenance=FactoryProvenance(
            seed=11, stream="calendar", generator_version="test",
            config_hash="abc123"),
    )


# ---------------------------------------------------------------------------
# Calendar configuration
# ---------------------------------------------------------------------------

def test_default_factory_calendar_accepted():
    cfg = SynthConfig()
    factory = cfg.factory
    assert factory.calendar_origin == "2024-01-01"
    assert factory.span_days == 120.0
    assert factory.dev_cutoff_days == 60.0
    assert factory.quarantine_days == 7.0
    assert factory.seed == 0


def test_factory_calendar_bounds_rejected():
    with pytest.raises(ValueError, match="span_days"):
        FactoryCalendarConfig(span_days=89.0)
    with pytest.raises(ValueError, match="span_days"):
        FactoryCalendarConfig(span_days=184.0)
    with pytest.raises(ValueError, match="dev_cutoff_days"):
        FactoryCalendarConfig(dev_cutoff_days=120.0)
    with pytest.raises(ValueError, match="dev_cutoff_days"):
        FactoryCalendarConfig(dev_cutoff_days=0.0)
    with pytest.raises(ValueError, match="quarantine_days"):
        FactoryCalendarConfig(quarantine_days=6.0)
    with pytest.raises(ValueError, match="calendar_origin"):
        FactoryCalendarConfig(calendar_origin="not-a-date")
    with pytest.raises(ValueError, match="seed"):
        FactoryCalendarConfig(seed=-1)
    with pytest.raises(ValueError, match="quarantine_days"):
        SynthConfig(factory=FactoryCalendarConfig(quarantine_days=1.0))


# ---------------------------------------------------------------------------
# Accepted records
# ---------------------------------------------------------------------------

def test_minimal_legacy_record_accepted():
    """Pre-factory FileSample construction still validates."""
    sample = FileSample(
        x=_signal(3, 32),
        file_id="legacy-1",
        file_label=SampleLabel.NORMAL,
        seed=3,
        generator_version="test",
        config_hash="hash",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 32, 0.5)],
    )
    sample.validate()
    assert sample.operation is None
    assert sample.future_targets is None
    inputs = sample.encoder_inputs()
    assert set(inputs) <= MODEL_INPUT_FIELD_NAMES


def test_full_factory_records_accepted():
    for label in (SampleLabel.NORMAL, SampleLabel.ABNORMAL):
        sample = _full_sample(label)
        sample.validate()


def test_open_episode_and_recommissioned_stage_accepted():
    episode = HealthEpisode(
        episode_id="m-1", kind="maintenance",
        robot_id="robot-01", start_time=10.0, end_time=70.0)
    assert episode.kind is EpisodeKind.MAINTENANCE
    state = RobotHealthState(
        health_value=0.0, program_sensitivity=0.0, manifested_value=0.0,
        degradation_stage="recommissioned", degradation_severity=0.0)
    assert state.degradation_stage is DegradationStage.RECOMMISSIONED


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

def test_identity_distinction_rejected():
    with pytest.raises(ValueError, match="distinct"):
        _operation(unit_id="robot-03")
    with pytest.raises(ValueError, match="distinct"):
        _operation(operation_id="prog-weld-7")
    with pytest.raises(ValueError, match="non-empty"):
        _operation(robot_id="  ")
    with pytest.raises(ValueError, match="route_position"):
        _operation(route_position=-1)


def test_timing_invariants_rejected():
    with pytest.raises(ValueError, match="arrival_time <= start_time"):
        _operation(arrival_time=2000.0)
    with pytest.raises(ValueError, match="arrival_time <= start_time"):
        _operation(end_time=1000.0)
    with pytest.raises(ValueError, match="duration"):
        _operation(duration=119.0)
    with pytest.raises(ValueError, match="queue_delay"):
        _operation(queue_delay=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        _operation(travel_time=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        _operation(idle_before=-0.5)


def test_future_target_consistency_rejected():
    with pytest.raises(ValueError, match="contradicts"):
        FutureFailureTargets(
            time_to_next_failure=3600.0, failure_within_1d=True,
            failure_within_7d=False, is_censored=False)
    with pytest.raises(ValueError, match="censored"):
        FutureFailureTargets(
            time_to_next_failure=3600.0, failure_within_1d=True,
            failure_within_7d=True, is_censored=True)
    with pytest.raises(ValueError, match="censored"):
        FutureFailureTargets(
            time_to_next_failure=None, failure_within_1d=False,
            failure_within_7d=True, is_censored=True)
    with pytest.raises(ValueError, match="require time_to_next_failure"):
        FutureFailureTargets(
            time_to_next_failure=None, failure_within_1d=False,
            failure_within_7d=True, is_censored=False)
    with pytest.raises(ValueError, match="contradicts"):
        FutureFailureTargets(
            time_to_next_failure=30.0 * SECONDS_PER_DAY,
            failure_within_1d=False, failure_within_7d=True,
            is_censored=False)


def test_future_target_horizons_accepted():
    one_day = FutureFailureTargets(
        time_to_next_failure=3600.0, failure_within_1d=True,
        failure_within_7d=True, is_censored=False)
    assert one_day.failure_within_1d and one_day.failure_within_7d
    assert SECONDS_PER_WEEK == 7.0 * SECONDS_PER_DAY
    week_only = FutureFailureTargets(
        time_to_next_failure=3.0 * SECONDS_PER_DAY,
        failure_within_1d=False, failure_within_7d=True, is_censored=False)
    assert not week_only.failure_within_1d
    far = FutureFailureTargets(
        time_to_next_failure=30.0 * SECONDS_PER_DAY,
        failure_within_1d=False, failure_within_7d=False, is_censored=False)
    assert not far.failure_within_7d
    censored = FutureFailureTargets(
        time_to_next_failure=None, failure_within_1d=False,
        failure_within_7d=False, is_censored=True)
    assert censored.is_censored


def test_quarantine_provenance_rejected():
    with pytest.raises(ValueError, match="quarantined files MUST NOT"):
        SplitProvenance(
            cutoff_time=100.0, is_quarantined=True,
            quarantine_reason="precursor window", member_views=("dev-train",))
    with pytest.raises(ValueError, match="quarantine_reason"):
        SplitProvenance(cutoff_time=100.0, is_quarantined=True)
    with pytest.raises(ValueError, match="must not carry quarantine_reason"):
        SplitProvenance(
            cutoff_time=100.0, is_quarantined=False,
            quarantine_reason="stale")
    with pytest.raises(ValueError, match="unknown split view"):
        SplitProvenance(
            cutoff_time=100.0, is_quarantined=False,
            member_views=("dev-train", "nope"))
    with pytest.raises(ValueError, match="duplicates"):
        SplitProvenance(
            cutoff_time=100.0, is_quarantined=False,
            member_views=("test-static", "test-static"))
    assert set(ALLOWED_SPLIT_VIEWS) == {
        "dev-train", "dev-val", "test-static", "test-temporal"}


def test_label_agreement_rejected():
    normal = _full_sample(SampleLabel.NORMAL)
    normal.anomaly_labels = ObservableAnomalyLabels(
        is_file_anomalous=True, anomaly_family="subtle_drift",
        anomaly_severity=0.6)
    with pytest.raises(AssertionError, match="must agree"):
        normal.validate()
    abnormal = _full_sample(SampleLabel.ABNORMAL)
    abnormal.anomaly_labels = ObservableAnomalyLabels(
        is_file_anomalous=True, anomaly_family="wrong_transition",
        anomaly_severity=0.6)
    with pytest.raises(AssertionError, match="must match injected"):
        abnormal.validate()
    with pytest.raises(ValueError, match="must not carry"):
        ObservableAnomalyLabels(
            is_file_anomalous=False, anomaly_family="subtle_drift")


def test_episode_bounds_rejected():
    with pytest.raises(ValueError, match="precedes start_time"):
        HealthEpisode(
            episode_id="e-1", kind=EpisodeKind.FAILURE,
            robot_id="robot-01", start_time=50.0, end_time=49.0)
    with pytest.raises(ValueError, match="unknown episode kind"):
        HealthEpisode(
            episode_id="e-1", kind="outage",
            robot_id="robot-01", start_time=50.0)
    with pytest.raises(ValueError, match="unknown degradation_stage"):
        RobotHealthState(
            health_value=0.1, program_sensitivity=1.0, manifested_value=0.1,
            degradation_stage="mild", degradation_severity=0.1)


def test_provenance_agreement_rejected():
    sample = _full_sample()
    assert sample.factory_provenance is not None
    sample.factory_provenance.seed = 999
    with pytest.raises(AssertionError, match="provenance.seed"):
        sample.validate()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _assert_factory_round_trip(sample: FileSample) -> FileSample:
    payload = _sample_bytes(sample)
    assert payload == _sample_bytes(sample), "serialization must be deterministic"
    restored = load_sample_bytes(payload)
    restored.validate()
    return restored


def test_full_record_round_trip():
    sample = _full_sample(SampleLabel.ABNORMAL)
    restored = _assert_factory_round_trip(sample)
    assert restored.operation == sample.operation
    assert restored.operating_context == sample.operating_context
    assert restored.health == sample.health
    assert restored.episode == sample.episode
    assert restored.anomaly_labels == sample.anomaly_labels
    assert restored.future_targets == sample.future_targets
    assert restored.split_provenance == sample.split_provenance
    assert restored.factory_provenance == sample.factory_provenance
    np.testing.assert_array_equal(restored.x, sample.x)


def test_legacy_record_round_trip():
    """Archives written before Task 1 decode new contracts to None."""
    sample = FileSample(
        x=_signal(3, 32),
        file_id="legacy-2",
        file_label=SampleLabel.NORMAL,
        seed=5,
        generator_version="test",
        config_hash="hash",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 32, 0.5)],
    )
    payload = _sample_bytes(sample)
    assert b"operation_json" not in payload
    restored = load_sample_bytes(payload)
    restored.validate()
    assert restored.operation is None
    assert restored.health is None
    assert restored.future_targets is None
    assert restored.split_provenance is None
    assert restored.factory_provenance is None


# ---------------------------------------------------------------------------
# Model-input separation
# ---------------------------------------------------------------------------

def test_every_sample_field_classified():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(FileSample)}
    assert fields == MODEL_INPUT_FIELD_NAMES | DIAGNOSTIC_ONLY_FIELD_NAMES
    assert not (MODEL_INPUT_FIELD_NAMES & DIAGNOSTIC_ONLY_FIELD_NAMES)


def test_encoder_inputs_exclude_diagnostics_and_future():
    sample = _full_sample(SampleLabel.ABNORMAL)
    inputs = sample.encoder_inputs()
    assert set(inputs) <= MODEL_INPUT_FIELD_NAMES
    assert not (set(inputs) & DIAGNOSTIC_ONLY_FIELD_NAMES)
    for name in DIAGNOSTIC_ONLY_FIELD_NAMES:
        assert name not in inputs
    # Scheduling/identity context stays visible; health and targets do not.
    assert inputs["operation"] == sample.operation
    assert inputs["operating_context"] == sample.operating_context
