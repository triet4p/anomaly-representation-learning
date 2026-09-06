"""Focused tests for temporal labels, quarantine, and splits (Task 6)."""

from __future__ import annotations

import pytest

from synth.config import (
    FactoryCalendarConfig,
    HealthConfig,
    SchedulerConfig,
    SynthConfig,
)
from synth.health import RobotHealthProcess
from synth.scheduled import ScheduledSignalGenerator
from synth.scheduler import FactoryScheduler
from synth.schema import (
    DegradationStage,
    EpisodeKind,
    SampleLabel,
)
from synth.splits import (
    ChronologicalSplitter,
    ChronologicalSplits,
)
from synth.temporal import TemporalAnomalyProcess

DAY = 86400.0
WEEK = 604800.0


def _config() -> SynthConfig:
    cfg = SynthConfig()
    cfg.factory = FactoryCalendarConfig(
        span_days=90.0, dev_cutoff_days=45.0, quarantine_days=7.0, seed=0
    )
    cfg.scheduler = SchedulerConfig(
        n_units=60, arrival_interval_s=86400.0, seed=0
    )
    cfg.health = HealthConfig(
        seed=0, aging_rate=5e-7, wear_rate=1e-4, noise_scale=1e-3,
        base_rate=2e-6, alpha=2.5, beta=0.0, abrupt_rate=0.0,
    )
    return cfg


@pytest.fixture(scope="module")
def pipeline():
    cfg = _config()
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    base = ScheduledSignalGenerator(cfg).generate(schedule, health)
    manifested, _ = TemporalAnomalyProcess(cfg).apply(
        schedule, health, base
    )
    labeled, splits = ChronologicalSplitter(cfg).build(
        schedule, health, manifested
    )
    return cfg, schedule, health, labeled, splits


def _by_id(labeled):
    return {s.file_id: s for s in labeled}


# ---------------------------------------------------------------------------
# Exact membership and validation
# ---------------------------------------------------------------------------


def test_splits_validate_passes(pipeline):
    _, schedule, health, labeled, splits = pipeline
    assert isinstance(splits, ChronologicalSplits)
    splits.validate(schedule, health, labeled)


def test_views_are_nontrivial(pipeline):
    _, _, _, _, splits = pipeline
    assert splits.dev_train and splits.dev_val
    assert splits.test_static and splits.test_temporal
    assert splits.quarantined and splits.failed_episode_ids


def test_exact_membership_recomputed(pipeline):
    cfg, schedule, health, labeled, splits = pipeline
    cutoff = cfg.factory.dev_cutoff_days * DAY
    failed_points = {
        (e.robot_id, e.start_time)
        for e in health.episodes
        if e.kind is EpisodeKind.FAILURE
    }
    failed_degs = {
        e.episode_id
        for e in health.episodes
        if e.kind is EpisodeKind.DEGRADATION
        and e.end_time is not None
        and (e.robot_id, e.end_time) in failed_points
    }
    assert sorted(failed_degs) == sorted(splits.failed_episode_ids)
    ordered = sorted(
        labeled,
        key=lambda s: (
            s.operation.start_time, s.operation.end_time, s.file_id
        ),
    )
    pool = [
        s.file_id for s in ordered
        if s.operation.start_time < cutoff
        and s.file_label is SampleLabel.NORMAL
        and not s.split_provenance.is_quarantined
    ]
    assert splits.dev_train == pool[: int(len(pool) * 0.8)]
    assert splits.dev_val == pool[int(len(pool) * 0.8):]
    assert splits.test_static == [
        s.file_id for s in ordered
        if s.file_label is SampleLabel.ABNORMAL
        or (
            s.operation.start_time >= cutoff
            and s.file_label is SampleLabel.NORMAL
        )
    ]
    assert splits.test_temporal == [
        s.file_id for s in ordered
        if s.operation.start_time >= cutoff
        or (
            (
                s.health.degradation_episode_id in failed_degs
                or s.health.degradation_stage
                in (DegradationStage.FAILED, DegradationStage.IN_MAINTENANCE)
            )
            and (
                s.file_label is SampleLabel.ABNORMAL
                or s.split_provenance.is_quarantined
            )
        )
    ]


def test_all_postcutoff_normals_and_all_abnormals_in_static(pipeline):
    cfg, _, _, labeled, splits = pipeline
    cutoff = cfg.factory.dev_cutoff_days * DAY
    static = set(splits.test_static)
    for sample in labeled:
        if sample.file_label is SampleLabel.ABNORMAL:
            assert sample.file_id in static, sample.file_id
        elif sample.operation.start_time >= cutoff:
            assert sample.file_id in static, sample.file_id
        else:
            if not sample.split_provenance.is_quarantined:
                assert sample.file_id not in static, sample.file_id


# ---------------------------------------------------------------------------
# Causality and leakage
# ---------------------------------------------------------------------------


def test_development_data_is_verified_healthy_precutoff(pipeline):
    cfg, _, _, labeled, splits = pipeline
    cutoff = cfg.factory.dev_cutoff_days * DAY
    by_id = _by_id(labeled)
    for fid in splits.dev_train + splits.dev_val:
        sample = by_id[fid]
        assert sample.operation.start_time < cutoff
        assert sample.file_label is SampleLabel.NORMAL
        assert not sample.split_provenance.is_quarantined
        assert sample.future_targets.is_censored or (
            sample.future_targets.time_to_next_failure is not None
            and sample.future_targets.time_to_next_failure
            > cfg.factory.quarantine_days * DAY
        )


def test_train_val_is_chronological_80_20(pipeline):
    _, _, _, labeled, splits = pipeline
    by_id = _by_id(labeled)
    pool_size = len(splits.dev_train) + len(splits.dev_val)
    assert len(splits.dev_train) == int(pool_size * 0.8)
    latest_train = max(
        by_id[fid].operation.start_time for fid in splits.dev_train
    )
    earliest_val = min(
        by_id[fid].operation.start_time for fid in splits.dev_val
    )
    assert latest_train <= earliest_val


def test_no_leakage_between_views(pipeline):
    _, _, _, _, splits = pipeline
    dev = set(splits.dev_train) | set(splits.dev_val)
    assert set(splits.dev_train) & set(splits.dev_val) == set()
    assert dev & set(splits.test_static) == set()
    assert dev & set(splits.test_temporal) == set()
    assert dev & set(splits.quarantined) == set()


def test_temporal_view_is_chronological_and_unshuffled(pipeline):
    _, _, _, labeled, splits = pipeline
    by_id = _by_id(labeled)
    keys = [
        (
            by_id[fid].operation.start_time,
            by_id[fid].operation.end_time,
            fid,
        )
        for fid in splits.test_temporal
    ]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Labels, targets, quarantine
# ---------------------------------------------------------------------------


def test_future_targets_are_causally_coherent(pipeline):
    _, schedule, health, labeled, _ = pipeline
    failures: dict[str, list[float]] = {}
    for episode in health.episodes:
        if episode.kind is EpisodeKind.FAILURE:
            failures.setdefault(episode.robot_id, []).append(
                episode.start_time
            )
    for event, sample in zip(schedule.events, labeled):
        targets = sample.future_targets
        future = [
            t for t in failures.get(event.robot_id, [])
            if t >= event.end_time - 1e-6
        ]
        expected = max(0.0, min(future) - event.end_time) if future else None
        if expected is None:
            assert targets.is_censored
            assert targets.time_to_next_failure is None
            assert not targets.failure_within_1d
            assert not targets.failure_within_7d
        else:
            assert not targets.is_censored
            assert targets.time_to_next_failure == pytest.approx(
                expected, abs=1e-6
            )
            assert targets.failure_within_1d == (expected <= DAY)
            assert targets.failure_within_7d == (expected <= WEEK)
        if targets.failure_within_1d:
            assert targets.failure_within_7d


def test_critical_early_warning_example_exists(pipeline):
    _, _, _, labeled, splits = pipeline
    quarantined = set(splits.quarantined)
    examples = [
        s for s in labeled
        if not s.anomaly_labels.is_file_anomalous
        and s.future_targets.failure_within_7d
    ]
    assert examples, "section 20.9 example must occur in the data"
    assert all(s.file_id in quarantined for s in examples)


def test_quarantine_reasons_and_maintenance_boundaries(pipeline):
    _, _, _, labeled, _ = pipeline
    for sample in labeled:
        prov = sample.split_provenance
        stage = sample.health.degradation_stage
        if stage is DegradationStage.FAILED:
            assert prov.is_quarantined
            assert prov.quarantine_reason == "failure_event"
        elif stage is DegradationStage.IN_MAINTENANCE:
            assert prov.is_quarantined
            assert prov.quarantine_reason == "maintenance_transition"
            assert "dev-train" not in prov.member_views
            assert "dev-val" not in prov.member_views
        elif prov.is_quarantined:
            assert prov.quarantine_reason == "precursor_horizon"
        else:
            assert prov.quarantine_reason is None


def test_labels_complete_and_agree(pipeline):
    _, _, _, labeled, _ = pipeline
    for sample in labeled:
        assert sample.anomaly_labels is not None
        assert sample.future_targets is not None
        assert sample.split_provenance is not None
        labels = sample.anomaly_labels
        if sample.file_label is SampleLabel.ABNORMAL:
            assert labels.is_file_anomalous
            assert labels.anomaly_family == sample.anomaly_meta.family.value
        else:
            assert not labels.is_file_anomalous
            assert labels.anomaly_family is None
            assert labels.anomaly_severity is None
            assert sample.anomaly_mask is None


def test_split_is_deterministic(pipeline):
    cfg, schedule, health, labeled, splits = pipeline
    base = [s for s in labeled]
    relabeled, splits2 = ChronologicalSplitter(cfg).build(
        schedule, health, base
    )
    assert splits2.dev_train == splits.dev_train
    assert splits2.dev_val == splits.dev_val
    assert splits2.test_static == splits.test_static
    assert splits2.test_temporal == splits.test_temporal
    assert splits2.quarantined == splits.quarantined
    for first, second in zip(labeled, relabeled):
        assert first.future_targets == second.future_targets
        assert first.anomaly_labels == second.anomaly_labels


def test_build_rejects_misaligned_samples(pipeline):
    cfg, schedule, health, labeled, _ = pipeline
    with pytest.raises(ValueError):
        ChronologicalSplitter(cfg).build(schedule, health, labeled[:-1])


def test_encoder_inputs_exclude_label_and_future_diagnostics(pipeline):
    _, _, _, labeled, _ = pipeline
    for sample in labeled:
        sample.validate()
        inputs = sample.encoder_inputs()
        for forbidden in (
            "anomaly_meta",
            "anomaly_mask",
            "anomaly_labels",
            "health",
            "episode",
            "future_targets",
            "split_provenance",
            "rejection_meta",
        ):
            assert forbidden not in inputs
