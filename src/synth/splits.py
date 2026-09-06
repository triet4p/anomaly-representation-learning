"""Temporal labels, precursor quarantine, and split indices (Task 6).

Materializes the three separate label groups of methodology section 20.9
over the accepted Task 5 anomaly annotation, then builds the chronological
split indices of sections 20.10-20.12:

- observable anomaly labels (static detection only);
- latent diagnostic state, already carried on ``FileSample.health`` /
  ``episode`` and completed here for files Task 5 left unmanifested;
- future-failure targets (``time_to_next_failure``, one-day/seven-day
  flags, censoring) for survival/risk supervision only;
- precursor quarantine (at least seven days): files inside the warning
  horizon, failure events, and maintenance transitions are excluded from
  verified healthy development data even when not binary abnormal;
- chronological pre-cutoff verified-healthy development data split
  earliest/latest 80/20 into train/validation;
- static test: all post-cutoff normals plus every abnormal file at any
  time;
- temporal test: all post-cutoff files plus held-out pre-cutoff failure
  episodes, kept in chronological order and never shuffled.

Static and temporal views answer distinct questions and may overlap; they
are never conflated. Labels, health, episodes, masks, and future targets
are diagnostics only and never enter encoder inputs (section 20.9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from synth.config import SynthConfig
from synth.health import FactoryHealth
from synth.scheduler import FactorySchedule
from synth.schema import (
    DegradationStage,
    EpisodeKind,
    FileSample,
    FutureFailureTargets,
    HealthEpisode,
    ObservableAnomalyLabels,
    SampleLabel,
    SplitProvenance,
)
from synth.temporal import TemporalAnomalyProcess

#: Split views recorded in ``SplitProvenance.member_views``.
DEV_TRAIN_VIEW = "dev-train"
DEV_VAL_VIEW = "dev-val"
TEST_STATIC_VIEW = "test-static"
TEST_TEMPORAL_VIEW = "test-temporal"

#: Quarantine reasons recorded in ``SplitProvenance.quarantine_reason``.
PRECURSOR_REASON = "precursor_horizon"
FAILURE_REASON = "failure_event"
MAINTENANCE_REASON = "maintenance_transition"


@dataclass
class ChronologicalSplits:
    """Chronological file index over one generated calendar.

    Every list holds ``file_id`` values in chronological
    ``(start_time, end_time, operation_id)`` order. ``quarantined`` holds
    every quarantined file id; ``failed_episode_ids`` names the
    degradation episodes that culminate in a failure.
    """

    cutoff_time: float
    quarantine_s: float
    dev_train: list[str] = field(default_factory=list)
    dev_val: list[str] = field(default_factory=list)
    test_static: list[str] = field(default_factory=list)
    test_temporal: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    failed_episode_ids: list[str] = field(default_factory=list)

    def validate(
        self,
        schedule: FactorySchedule,
        health: FactoryHealth,
        samples: list[FileSample],
    ) -> None:
        """Assert exact-membership, causality, and leakage invariants."""
        schedule.validate()
        health.validate(schedule)
        by_id = {s.file_id: s for s in samples}
        assert len(by_id) == len(samples), "file ids must be unique"
        assert [s.operation.operation_id for s in samples] == [  # type: ignore[union-attr]
            e.operation_id for e in schedule.events
        ], "samples must align 1:1 with schedule order"

        expected = _expected_views(
            schedule, health, samples, self.cutoff_time, self.quarantine_s
        )
        for view in ("dev_train", "dev_val", "test_static", "test_temporal"):
            assert getattr(self, view) == expected[view], (
                f"{view} membership mismatch"
            )
        assert sorted(self.quarantined) == sorted(expected["quarantined"]), (
            "quarantined membership mismatch"
        )
        assert sorted(self.failed_episode_ids) == sorted(
            expected["failed_episode_ids"]
        ), "failed episode set mismatch"

        times = {
            s.file_id: (
                s.operation.start_time,  # type: ignore[union-attr]
                s.operation.end_time,  # type: ignore[union-attr]
            )
            for s in samples
        }
        for view in ("dev_train", "dev_val", "test_static", "test_temporal"):
            ordered = sorted(
                getattr(self, view),
                key=lambda fid: (times[fid][0], times[fid][1], fid),
            )
            assert getattr(self, view) == ordered, (
                f"{view} must be in chronological order"
            )

        dev = set(self.dev_train) | set(self.dev_val)
        assert set(self.dev_train) & set(self.dev_val) == set(), (
            "train/val must be disjoint"
        )
        for fid in dev:
            sample = by_id[fid]
            assert sample.operation.start_time < self.cutoff_time, (  # type: ignore[union-attr]
                f"{fid}: development files must precede the cutoff"
            )
            assert sample.file_label is SampleLabel.NORMAL, (
                f"{fid}: development files must be verified healthy"
            )
        if self.dev_train and self.dev_val:
            latest_train = max(times[fid] for fid in self.dev_train)
            earliest_val = min(times[fid] for fid in self.dev_val)
            assert latest_train <= earliest_val, (
                "train/validation must be a chronological 80/20 split"
            )
        for sample in samples:
            assert sample.future_targets is not None, (
                f"{sample.file_id}: missing future targets"
            )
            assert sample.anomaly_labels is not None, (
                f"{sample.file_id}: missing observable labels"
            )
            assert sample.split_provenance is not None, (
                f"{sample.file_id}: missing split provenance"
            )
            views = set(sample.split_provenance.member_views)
            in_dev = bool(views & {DEV_TRAIN_VIEW, DEV_VAL_VIEW})
            assert in_dev == (sample.file_id in dev), (
                f"{sample.file_id}: split provenance must match split indices"
            )
            assert (TEST_STATIC_VIEW in views) == (
                sample.file_id in self.test_static
            ), f"{sample.file_id}: static provenance must match"
            assert (TEST_TEMPORAL_VIEW in views) == (
                sample.file_id in self.test_temporal
            ), f"{sample.file_id}: temporal provenance must match"
            sample.encoder_inputs()


def _expected_views(
    schedule: FactorySchedule,
    health: FactoryHealth,
    samples: list[FileSample],
    cutoff: float,
    quarantine_s: float,
) -> dict[str, list[str]]:
    """Recompute the exact expected views without reusing split internals.
    Horizons, quarantine flags, failed-episode membership, and view rules
    are derived again from health episodes, labels, and timestamps, so
    `ChronologicalSplits.validate` guards against wiring mistakes rather
    than repeating the same code path.
    """
    failures: dict[str, list[float]] = {}
    for episode in health.episodes:
        if episode.kind is EpisodeKind.FAILURE:
            failures.setdefault(episode.robot_id, []).append(episode.start_time)
    for times in failures.values():
        times.sort()
    failed_degs: set[str] = set()
    failure_points = {
        (episode.robot_id, episode.start_time)
        for episode in health.episodes
        if episode.kind is EpisodeKind.FAILURE
    }
    for episode in health.episodes:
        if episode.kind is EpisodeKind.DEGRADATION and episode.end_time is not None:
            if (episode.robot_id, episode.end_time) in failure_points:
                failed_degs.add(episode.episode_id)
    def horizon_of(sample: FileSample) -> float | None:
        assert sample.operation is not None
        future = [
            t for t in failures.get(sample.operation.robot_id, [])
            if t >= sample.operation.end_time - 1e-6
        ]
        if not future:
            return None
        return max(0.0, min(future) - sample.operation.end_time)
    ordered = sorted(
        samples,
        key=lambda s: (
            s.operation.start_time,  # type: ignore[union-attr]
            s.operation.end_time,  # type: ignore[union-attr]
            s.file_id,
        ),
    )
    dev_pool: list[str] = []
    static: list[str] = []
    temporal: list[str] = []
    quarantined: list[str] = []
    for sample in ordered:
        assert sample.operation is not None
        assert sample.health is not None
        assert sample.split_provenance is not None
        horizon = horizon_of(sample)
        stage = sample.health.degradation_stage
        if stage is DegradationStage.IN_MAINTENANCE:
            expected_q: tuple[bool, str | None] = (True, MAINTENANCE_REASON)
        elif stage is DegradationStage.FAILED:
            expected_q = (True, FAILURE_REASON)
        elif horizon is not None and horizon <= quarantine_s:
            expected_q = (True, PRECURSOR_REASON)
        else:
            expected_q = (False, None)
        assert sample.split_provenance.is_quarantined == expected_q[0], (
            f"{sample.file_id}: quarantine flag mismatch"
        )
        assert sample.split_provenance.quarantine_reason == expected_q[1], (
            f"{sample.file_id}: quarantine reason mismatch"
        )
        pre = sample.operation.start_time < cutoff
        normal = sample.file_label is SampleLabel.NORMAL
        if pre and normal and not expected_q[0]:
            dev_pool.append(sample.file_id)
        if not normal or not pre:
            static.append(sample.file_id)
        in_failed = (
            sample.health.degradation_episode_id in failed_degs
            or stage in (
                DegradationStage.FAILED,
                DegradationStage.IN_MAINTENANCE,
            )
        )
        held_out = in_failed and (not normal or expected_q[0])
        if not pre or held_out:
            temporal.append(sample.file_id)
        if expected_q[0]:
            quarantined.append(sample.file_id)
    n_train = int(len(dev_pool) * 0.8)
    return {
        "dev_train": dev_pool[:n_train],
        "dev_val": dev_pool[n_train:],
        "test_static": static,
        "test_temporal": temporal,
        "quarantined": quarantined,
        "failed_episode_ids": sorted(failed_degs),
    }


class ChronologicalSplitter:
    """Derives labels, targets, quarantine, and split indices.

    Usage::

        schedule = FactoryScheduler(config).build()
        health = RobotHealthProcess(config).run(schedule)
        base = ScheduledSignalGenerator(config).generate(schedule, health)
        manifested, _ = TemporalAnomalyProcess(config).apply(
            schedule, health, base)
        labeled, splits = ChronologicalSplitter(config).build(
            schedule, health, manifested)
        splits.validate(schedule, health, labeled)
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()

    @property
    def cutoff_time(self) -> float:
        """Development cutoff in seconds since the calendar origin."""
        return self.cfg.factory.dev_cutoff_days * 86400.0

    @property
    def quarantine_s(self) -> float:
        """Precursor quarantine horizon in seconds (at least seven days)."""
        return self.cfg.factory.quarantine_days * 86400.0

    def build(
        self,
        schedule: FactorySchedule,
        health: FactoryHealth,
        samples: list[FileSample],
    ) -> tuple[list[FileSample], ChronologicalSplits]:
        """Attach labels/targets/provenance and compute split indices."""
        schedule.validate()
        health.validate(schedule, self.cfg.health)
        if len(samples) != len(schedule.events):
            raise ValueError(
                f"samples ({len(samples)}) must align 1:1 with "
                f"schedule events ({len(schedule.events)})"
            )
        cutoff = self.cutoff_time
        quarantine = self.quarantine_s
        failures_by_robot = self._failures_by_robot(health)
        failed_degs = self._failed_degradation_ids(health)
        episodes = {e.episode_id: e for e in health.episodes}
        failures = [e for e in health.episodes if e.kind is EpisodeKind.FAILURE]
        maintenances = [
            e for e in health.episodes if e.kind is EpisodeKind.MAINTENANCE
        ]

        labeled: list[FileSample] = []
        for event, state, sample in zip(
            schedule.events, health.states, samples
        ):
            if sample.operation is None or (
                sample.operation.operation_id != event.operation_id
            ):
                raise ValueError(
                    f"sample/schedule misalignment at {event.operation_id}"
                )
            horizon = self._time_to_next_failure(
                event, failures_by_robot.get(event.robot_id, [])
            )
            targets = FutureFailureTargets(
                time_to_next_failure=horizon,
                failure_within_1d=(
                    horizon is not None and horizon <= 86400.0
                ),
                failure_within_7d=(
                    horizon is not None and horizon <= 604800.0
                ),
                is_censored=horizon is None,
            )
            if sample.file_label is SampleLabel.ABNORMAL:
                labels = sample.anomaly_labels
                if labels is None:
                    assert sample.anomaly_meta is not None
                    labels = ObservableAnomalyLabels(
                        is_file_anomalous=True,
                        anomaly_family=sample.anomaly_meta.family.value,
                        anomaly_severity=float(sample.anomaly_meta.severity),
                    )
            else:
                labels = ObservableAnomalyLabels(is_file_anomalous=False)
            quarantined, reason = self._quarantine(
                event, state, horizon, quarantine
            )
            labeled.append(
                replace(
                    sample,
                    anomaly_labels=labels,
                    future_targets=targets,
                    episode=self._episode_for(
                        event, state, sample,
                        episodes, failures, maintenances,
                    ),
                    split_provenance=SplitProvenance(
                        cutoff_time=cutoff,
                        is_quarantined=quarantined,
                        quarantine_reason=reason,
                        member_views=(),
                    ),
                )
            )

        splits = self._index(schedule, health, labeled, failed_degs)
        for sample in labeled:
            views = self._views_for(sample.file_id, splits)
            sample.split_provenance = replace(
                sample.split_provenance,  # type: ignore[arg-type]
                member_views=tuple(views),
            )
        splits.validate(schedule, health, labeled)
        return labeled, splits

    @staticmethod
    def _failures_by_robot(health: FactoryHealth) -> dict[str, list[float]]:
        """Return sorted failure times per robot."""
        out: dict[str, list[float]] = {}
        for episode in health.episodes:
            if episode.kind is EpisodeKind.FAILURE:
                out.setdefault(episode.robot_id, []).append(episode.start_time)
        for times in out.values():
            times.sort()
        return out

    @staticmethod
    def _failed_degradation_ids(health: FactoryHealth) -> set[str]:
        """Return degradation episodes closed exactly by a failure."""
        failures = [
            (e.robot_id, e.start_time)
            for e in health.episodes
            if e.kind is EpisodeKind.FAILURE
        ]
        out: set[str] = set()
        for episode in health.episodes:
            if episode.kind is not EpisodeKind.DEGRADATION:
                continue
            if episode.end_time is None:
                continue
            if (episode.robot_id, episode.end_time) in failures:
                out.add(episode.episode_id)
        return out

    @staticmethod
    def _time_to_next_failure(
        event: object, failure_times: list[float]
    ) -> float | None:
        """Return seconds from operation end to the next failure, if any."""
        end = event.end_time  # type: ignore[union-attr]
        future = [t for t in failure_times if t >= end - 1e-6]
        if not future:
            return None
        return max(0.0, min(future) - end)

    @staticmethod
    def _quarantine(
        event: object,
        state: object,
        horizon: float | None,
        quarantine: float,
    ) -> tuple[bool, str | None]:
        """Return ``(is_quarantined, reason)`` for one file."""
        stage = state.degradation_stage  # type: ignore[union-attr]
        if stage is DegradationStage.IN_MAINTENANCE:
            return True, MAINTENANCE_REASON
        if stage is DegradationStage.FAILED:
            return True, FAILURE_REASON
        if horizon is not None and horizon <= quarantine:
            return True, PRECURSOR_REASON
        return False, None

    @staticmethod
    def _episode_for(
        event: object,
        state: object,
        sample: FileSample,
        episodes: dict[str, HealthEpisode],
        failures: list[HealthEpisode],
        maintenances: list[HealthEpisode],
    ) -> HealthEpisode | None:
        """Keep Task 5 episode provenance, completing unmanifested files."""
        if sample.episode is not None:
            return sample.episode
        return TemporalAnomalyProcess._resolve_episode(
            event,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            episodes,
            failures,
            maintenances,
        )

    def _index(
        self,
        schedule: FactorySchedule,
        health: FactoryHealth,
        samples: list[FileSample],
        failed_degs: set[str],
    ) -> ChronologicalSplits:
        """Compute the exact chronological split indices."""
        cutoff = self.cutoff_time

        def key(sample: FileSample) -> tuple[float, float, str]:
            assert sample.operation is not None
            return (
                sample.operation.start_time,
                sample.operation.end_time,
                sample.file_id,
            )

        ordered = sorted(samples, key=key)
        dev_pool = [
            s
            for s in ordered
            if s.operation.start_time < cutoff  # type: ignore[union-attr]
            and s.file_label is SampleLabel.NORMAL
            and s.split_provenance is not None
            and not s.split_provenance.is_quarantined
        ]
        n_train = int(len(dev_pool) * 0.8)
        dev_train = [s.file_id for s in dev_pool[:n_train]]
        dev_val = [s.file_id for s in dev_pool[n_train:]]
        test_static = [
            s.file_id
            for s in ordered
            if (
                s.file_label is SampleLabel.ABNORMAL
                or (
                    s.operation.start_time >= cutoff  # type: ignore[union-attr]
                    and s.file_label is SampleLabel.NORMAL
                )
            )
        ]
        test_temporal = [
            s.file_id
            for s in ordered
            if s.operation.start_time >= cutoff  # type: ignore[union-attr]
            or self._held_out(s, failed_degs)
        ]
        quarantined = [
            s.file_id
            for s in samples
            if s.split_provenance is not None
            and s.split_provenance.is_quarantined
        ]
        return ChronologicalSplits(
            cutoff_time=cutoff,
            quarantine_s=self.quarantine_s,
            dev_train=dev_train,
            dev_val=dev_val,
            test_static=test_static,
            test_temporal=test_temporal,
            quarantined=quarantined,
            failed_episode_ids=sorted(failed_degs),
        )
    @staticmethod
    def _in_failed_episode(
        sample: FileSample, failed_degs: set[str]
    ) -> bool:
        """Return whether a file belongs to a failure-culminating episode."""
        assert sample.health is not None
        if sample.health.degradation_episode_id in failed_degs:
            return True
        return sample.health.degradation_stage in (
            DegradationStage.FAILED,
            DegradationStage.IN_MAINTENANCE,
        )

    @classmethod
    def _held_out(
        cls, sample: FileSample, failed_degs: set[str]
    ) -> bool:
        """Return whether a pre-cutoff file joins the temporal view.

        Held-out pre-cutoff failure episodes contribute their quarantined
        and abnormal members to early-warning evaluation, but verified
        healthy development members are never reused there: a dev file in
        the temporal view would be train-test contamination.
        """
        assert sample.split_provenance is not None
        if not cls._in_failed_episode(sample, failed_degs):
            return False
        return (
            sample.file_label is SampleLabel.ABNORMAL
            or sample.split_provenance.is_quarantined
        )

    @staticmethod
    def _views_for(
        file_id: str, splits: ChronologicalSplits
    ) -> list[str]:
        """Return the member views of one file in canonical order."""
        views = []
        if file_id in splits.dev_train:
            views.append(DEV_TRAIN_VIEW)
        if file_id in splits.dev_val:
            views.append(DEV_VAL_VIEW)
        if file_id in splits.test_static:
            views.append(TEST_STATIC_VIEW)
        if file_id in splits.test_temporal:
            views.append(TEST_TEMPORAL_VIEW)
        return views
