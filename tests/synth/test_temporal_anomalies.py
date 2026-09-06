"""Focused tests for temporal anomaly and degradation policies (Task 5)."""

from __future__ import annotations

import numpy as np
import pytest

from synth.anomalies.base import InjectionContext
from synth.anomalies.registry import HARD_FAMILIES, inject_anomaly
from synth.config import (
    FactoryCalendarConfig,
    HealthConfig,
    SchedulerConfig,
    SynthConfig,
    TemporalAnomalyConfig,
)
from synth.health import FactoryHealth, RobotHealthProcess
from synth.scheduled import ScheduledSignalGenerator
from synth.scheduler import FactoryScheduler
from synth.schema import (
    AnomalyFamily,
    DegradationStage,
    EpisodeKind,
    SampleLabel,
)
from synth.strength import StrengthGate
from synth.temporal import (
    TEMPORAL_FAMILIES,
    TemporalAnomalyProcess,
    precursor_probability,
    progressive_severity,
)


def _aggressive_config(**overrides) -> SynthConfig:
    cfg = SynthConfig()
    cfg.factory = FactoryCalendarConfig(
        span_days=90.0, dev_cutoff_days=45.0, quarantine_days=7.0, seed=0
    )
    cfg.scheduler = SchedulerConfig(
        n_units=30, arrival_interval_s=86400.0, seed=0
    )
    cfg.health = HealthConfig(
        seed=0, aging_rate=5e-7, wear_rate=1e-4, noise_scale=1e-3,
        base_rate=2e-6, alpha=2.5, beta=0.0, abrupt_rate=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg.temporal, key, value)
    return cfg


@pytest.fixture(scope="module")
def pipeline():
    cfg = _aggressive_config()
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    base = ScheduledSignalGenerator(cfg).generate(schedule, health)
    samples, annotation = TemporalAnomalyProcess(cfg).apply(
        schedule, health, base
    )
    return cfg, schedule, health, base, samples, annotation


def _next_failure_times(schedule, health):
    failures_by_robot: dict[str, list[float]] = {}
    for episode in health.episodes:
        if episode.kind is EpisodeKind.FAILURE:
            failures_by_robot.setdefault(episode.robot_id, []).append(
                episode.start_time
            )
    for times in failures_by_robot.values():
        times.sort()
    result = []
    for event in schedule.events:
        future = [
            t for t in failures_by_robot.get(event.robot_id, [])
            if t >= event.end_time - 1e-6
        ]
        result.append(min(future) - event.end_time if future else None)
    return result


# ---------------------------------------------------------------------------
# Determinism and stream independence
# ---------------------------------------------------------------------------


def test_temporal_apply_is_deterministic(pipeline):
    cfg, schedule, health, base, samples, annotation = pipeline
    again, annotation2 = TemporalAnomalyProcess(cfg).apply(
        schedule, health, base
    )
    assert annotation2.policies == annotation.policies
    for first, second in zip(samples, again):
        assert first.file_label is second.file_label
        assert np.array_equal(first.x, second.x)
        if first.anomaly_mask is not None:
            assert np.array_equal(first.anomaly_mask, second.anomaly_mask)
        else:
            assert second.anomaly_mask is None


def test_temporal_seed_changes_decisions_without_moving_calendar(pipeline):
    cfg, schedule, health, base, samples, annotation = pipeline
    cfg2 = _aggressive_config(seed=7)
    assert cfg2.hash() != cfg.hash()
    again, annotation2 = TemporalAnomalyProcess(cfg2).apply(
        schedule, health, base
    )
    assert annotation2.policies != annotation.policies
    assert [e.operation_id for e in schedule.events] == [
        s.operation.operation_id for s in again
    ]


def test_apply_rejects_misaligned_samples(pipeline):
    cfg, schedule, health, base, _, _ = pipeline
    with pytest.raises(ValueError):
        TemporalAnomalyProcess(cfg).apply(schedule, health, base[:-1])


def test_temporal_config_validation():
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(isolated_rate=1.5)
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(precursor_slope=-1.0)
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(severity_scale=0.0)
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(failure_severity=2.0)
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(seed=-1)
    with pytest.raises(ValueError):
        TemporalAnomalyConfig(max_attempts=0)


# ---------------------------------------------------------------------------
# Policy shape: program sensitivity and ordered severity
# ---------------------------------------------------------------------------


def test_precursor_probability_monotone_and_program_sensitive():
    cfg = TemporalAnomalyConfig()
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    probs = [precursor_probability(g, cfg) for g in grid]
    assert all(0.0 < p < 1.0 for p in probs)
    assert all(b >= a for a, b in zip(probs, probs[1:]))
    # Same latent health, sensitive program manifests with higher probability.
    assert precursor_probability(1.5 * 0.8, cfg) > precursor_probability(
        0.5 * 0.8, cfg
    )


def test_progressive_severity_ordered_and_bounded():
    cfg = TemporalAnomalyConfig()
    grid = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    sevs = [progressive_severity(g, cfg) for g in grid]
    assert all(cfg.severity_floor <= s <= 0.95 for s in sevs)
    assert all(b >= a for a, b in zip(sevs, sevs[1:]))


def test_realized_precursor_severity_monotone_in_manifested_health(pipeline):
    _, _, _, _, samples, annotation = pipeline
    pairs = [
        (s.health.manifested_value, s.anomaly_meta.severity)
        for s, policy in zip(samples, annotation.policies)
        if policy == "precursor"
    ]
    assert len(pairs) >= 3
    ordered = sorted(pairs, key=lambda pair: pair[0])
    severities = [sev for _, sev in ordered]
    assert all(b >= a - 1e-12 for a, b in zip(severities, severities[1:]))


def test_higher_severity_moves_signal_more(pipeline):
    _, _, _, base, _, _ = pipeline
    cfg = _aggressive_config()
    target = next(s for s in base if s.T > 100)
    x_clean = target.x.astype(np.float64)
    shifts = []
    for severity in (0.2, 0.8):
        rng = np.random.default_rng(1234)
        ctx = InjectionContext(
            x=x_clean.copy(), regimes=list(target.regime_sequence), rng=rng
        )
        result = inject_anomaly(
            AnomalyFamily.SUBTLE_DRIFT, ctx, severity, cfg.anomaly
        )
        assert result.accepted
        gate = StrengthGate(cfg.anomaly)
        accepted, _ = gate.evaluate(x_clean, result)
        assert accepted
        masked = result.mask
        shifts.append(float(np.mean(np.abs(
            result.x_modified[masked] - x_clean[masked]
        ))))
    assert shifts[1] > shifts[0]


# ---------------------------------------------------------------------------
# Manifestation behavior: localized, provenanced, not forced
# ---------------------------------------------------------------------------


def test_masks_stay_localized(pipeline):
    _, _, _, _, samples, _ = pipeline
    abnormal = [
        s for s in samples if s.file_label is SampleLabel.ABNORMAL
    ]
    assert len(abnormal) >= 5
    for sample in abnormal:
        assert sample.anomaly_mask is not None
        assert sample.anomaly_mask.shape == sample.x.shape
        assert bool(sample.anomaly_mask.any())
        assert not bool(sample.anomaly_mask.all())
        assert sample.anomaly_meta.family in TEMPORAL_FAMILIES
        assert sample.anomaly_meta.family in HARD_FAMILIES


def test_not_every_prefailure_file_is_binary_abnormal(pipeline):
    _, schedule, health, _, samples, _ = pipeline
    assert any(
        e.kind is EpisodeKind.FAILURE for e in health.episodes
    )
    horizons = _next_failure_times(schedule, health)
    horizon_normals = [
        sample
        for sample, horizon in zip(samples, horizons)
        if horizon is not None
        and 0.0 < horizon <= 7.0 * 86400.0
        and sample.file_label is SampleLabel.NORMAL
    ]
    assert horizon_normals, (
        "precursor quarantine horizon must contain visibly normal files"
    )


def test_episode_provenance_resolves(pipeline):
    _, schedule, health, _, samples, annotation = pipeline
    degradation_ids = {
        e.episode_id for e in health.episodes
        if e.kind is EpisodeKind.DEGRADATION
    }
    for event, state, sample, policy in zip(
        schedule.events, health.states, samples, annotation.policies
    ):
        if policy in ("isolated", "precursor"):
            assert sample.episode is not None, event.operation_id
            assert sample.episode.episode_id in degradation_ids
            assert sample.episode.episode_id == state.degradation_episode_id
            assert sample.anomaly_meta.extra["temporal_policy"] == policy
        if state.degradation_stage is DegradationStage.FAILED:
            assert policy == "failure_manifestation"
            assert sample.file_label is SampleLabel.ABNORMAL
            assert sample.episode is not None
            assert sample.episode.kind is EpisodeKind.FAILURE
        if state.degradation_stage is DegradationStage.IN_MAINTENANCE:
            assert policy == "none"
            assert sample.file_label is SampleLabel.NORMAL
            assert sample.anomaly_mask is None


def test_abrupt_failure_without_precursor_episode():
    cfg = _aggressive_config()
    cfg.health = HealthConfig(
        seed=3, aging_rate=0.0, wear_rate=0.0, noise_scale=0.0,
        degradation_onset=10.0, severity_scale=2.0,
        base_rate=0.0, alpha=0.0, beta=0.0, abrupt_rate=2e-4,
    )
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    failures = [
        e for e in health.episodes if e.kind is EpisodeKind.FAILURE
    ]
    assert failures, "abrupt config must produce failures"
    assert not [
        e for e in health.episodes if e.kind is EpisodeKind.DEGRADATION
    ], "abrupt failures must not open degradation episodes"
    base = ScheduledSignalGenerator(cfg).generate(schedule, health)
    samples, annotation = TemporalAnomalyProcess(cfg).apply(
        schedule, health, base
    )
    horizons = _next_failure_times(schedule, health)
    prefailure = [
        (sample, horizon)
        for sample, horizon in zip(samples, horizons)
        if horizon is not None and horizon > 0.0
    ]
    assert prefailure
    normal_fraction = sum(
        1 for sample, _ in prefailure
        if sample.file_label is SampleLabel.NORMAL
    ) / len(prefailure)
    assert normal_fraction >= 0.9
    for event, state, sample, policy in zip(
        schedule.events, health.states, samples, annotation.policies
    ):
        if state.degradation_stage is DegradationStage.FAILED:
            assert sample.file_label is SampleLabel.ABNORMAL
            assert sample.anomaly_meta.extra["temporal_policy"] == (
                "failure_manifestation"
            )


def test_encoder_inputs_exclude_temporal_diagnostics(pipeline):
    _, _, _, _, samples, _ = pipeline
    for sample in samples:
        sample.validate()
        inputs = sample.encoder_inputs()
        assert "anomaly_meta" not in inputs
        assert "anomaly_mask" not in inputs
        assert "anomaly_labels" not in inputs
        assert "health" not in inputs
        assert "episode" not in inputs
        assert "future_targets" not in inputs
