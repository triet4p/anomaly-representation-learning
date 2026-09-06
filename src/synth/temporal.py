"""Temporal anomaly and degradation policies (Sprint 11 Task 5).

Maps scheduled healthy signals onto observable anomaly manifestations
(methodology section 20.7) without touching the accepted Task 2 calendar,
Task 3 health trajectories, or Task 4 signal composition:

- isolated local anomalies strike single operations at a fixed rate;
- intermittent precursors appear with ``sigmoid(a*G + b)`` in the
  manifested health ``G``, so sensitive programs manifest earlier;
- progressive severity ``clip(floor + G / scale)`` keeps symptom
  strength ordered along a degradation trajectory;
- ``FAILED`` operations always receive an obvious manifestation,
  including deliberately abrupt failures at near-zero health;
- ``IN_MAINTENANCE`` operations never receive a manifestation.

Injection reuses the accepted per-family injectors plus the anomaly
strength gate, so masks stay localized and not every pre-failure file
becomes visibly abnormal (probabilistic manifestation plus gate
rejections leave files normal). Every decision uses a dedicated RNG
stream seeded from ``TemporalAnomalyConfig.seed`` plus operation
identity, never shared with scheduling, health, or signal streams
(section 20.14). Labels, masks, episode provenance, and future-target
inputs are diagnostics only and never enter encoder inputs (section
20.9); future targets and split indices themselves belong to Task 6.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

from synth.anomalies.base import AnomalyResult, InjectionContext

from synth.anomalies.registry import HARD_FAMILIES, inject_anomaly
from synth.config import SynthConfig, TemporalAnomalyConfig
from synth.health import FactoryHealth
from synth.scheduler import FactorySchedule
from synth.schema import (
    AnomalyFamily,
    DegradationStage,
    EpisodeKind,
    FileSample,
    HealthEpisode,
    ObservableAnomalyLabels,
    OperationEvent,
    RobotHealthState,
    SampleLabel,
)
from synth.strength import generate_with_strength_gate

#: Policies recorded in ``anomaly_meta.extra["temporal_policy"]``.
ISOLATED_POLICY = "isolated"
PRECURSOR_POLICY = "precursor"
FAILURE_POLICY = "failure_manifestation"


#: Temporal injection preserves scheduled provenance (regime coverage,
#: mask alignment), so duration interventions that reshape time are out
#: of scope here; the accepted IID generator still covers that family.
TEMPORAL_FAMILIES: list[AnomalyFamily] = [
    family for family in HARD_FAMILIES
    if family is not AnomalyFamily.DURATION_ANOMALY
]


def _stable_int(*parts: str) -> int:
    """Derive a stable 32-bit seed from identity parts (no RNG cost)."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def precursor_probability(manifested: float, cfg: TemporalAnomalyConfig) -> float:
    """Return ``P(manifest | G) = sigmoid(slope * G + intercept)``."""
    arg = cfg.precursor_slope * manifested + cfg.precursor_intercept
    arg = max(-50.0, min(50.0, arg))
    return 1.0 / (1.0 + math.exp(-arg))


def progressive_severity(manifested: float, cfg: TemporalAnomalyConfig) -> float:
    """Return the ordered severity for one manifested health value."""
    manifested = max(0.0, manifested)
    return min(0.95, cfg.severity_floor + manifested / cfg.severity_scale)


@dataclass
class TemporalAnomalies:
    """Observable anomaly annotation of one scheduled batch.

    ``operation_ids`` pins 1:1 alignment with the schedule order;
    ``policies`` records the per-file policy (``isolated``,
    ``precursor``, ``failure_manifestation``, or ``none``).
    """

    operation_ids: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)

    def validate(
        self,
        schedule: FactorySchedule,
        health: FactoryHealth,
        samples: list[FileSample],
    ) -> None:
        """Assert temporal anomaly structural invariants."""
        schedule.validate()
        health.validate(schedule)
        assert len(samples) == len(schedule.events) == len(self.operation_ids), (
            "temporal annotation must align 1:1 with schedule events"
        )
        assert self.operation_ids == [e.operation_id for e in schedule.events], (
            "temporal alignment must follow schedule order"
        )
        assert len(self.policies) == len(samples), "one policy per file required"
        for event, state, sample, policy in zip(
            schedule.events, health.states, samples, self.policies
        ):
            assert sample.operation is not None, f"{event.operation_id}: missing operation"
            assert sample.operation.operation_id == event.operation_id, (
                f"{event.operation_id}: sample/schedule operation mismatch"
            )
            assert sample.health is not None, f"{event.operation_id}: missing health"
            if policy not in ("isolated", "precursor", "failure_manifestation", "none"):
                raise AssertionError(f"{event.operation_id}: unknown policy {policy!r}")
            if sample.file_label == SampleLabel.ABNORMAL:
                assert sample.anomaly_meta is not None, (
                    f"{event.operation_id}: abnormal file needs anomaly_meta"
                )
                assert sample.anomaly_mask is not None, (
                    f"{event.operation_id}: abnormal file needs anomaly_mask"
                )
                assert sample.anomaly_mask.shape == sample.x.shape, (
                    f"{event.operation_id}: mask shape must match signal shape"
                )
                assert bool(sample.anomaly_mask.any()), (
                    f"{event.operation_id}: anomaly mask must be non-empty"
                )
                assert not bool(sample.anomaly_mask.all()), (
                    f"{event.operation_id}: anomaly mask must stay localized"
                )
                assert sample.anomaly_labels is not None, (
                    f"{event.operation_id}: abnormal file needs observable labels"
                )
                assert sample.anomaly_labels.is_file_anomalous, (
                    f"{event.operation_id}: labels must agree with file_label"
                )
                assert 0.0 <= float(sample.anomaly_meta.severity) <= 1.0, (
                    f"{event.operation_id}: severity must lie in [0, 1]"
                )
                assert sample.anomaly_meta.extra.get("temporal_policy") == policy, (
                    f"{event.operation_id}: meta policy must match annotation"
                )
            else:
                assert policy != "failure_manifestation", (
                    f"{event.operation_id}: FAILED operations must manifest"
                )
                assert sample.anomaly_mask is None, (
                    f"{event.operation_id}: normal file must not carry a mask"
                )
                assert sample.anomaly_labels is None or (
                    not sample.anomaly_labels.is_file_anomalous
                ), f"{event.operation_id}: labels must agree with file_label"
            if state.degradation_stage is DegradationStage.IN_MAINTENANCE:
                assert policy == "none", (
                    f"{event.operation_id}: maintenance files must stay unmanifested"
                )
                assert sample.file_label == SampleLabel.NORMAL, (
                    f"{event.operation_id}: maintenance files must stay normal"
                )
            if state.degradation_stage is DegradationStage.FAILED:
                assert policy == "failure_manifestation", (
                    f"{event.operation_id}: FAILED operations must manifest"
                )
            if state.degradation_episode_id is not None and sample.episode is not None:
                assert sample.episode.episode_id == state.degradation_episode_id or (
                    sample.episode.kind is EpisodeKind.FAILURE
                ), f"{event.operation_id}: episode provenance must resolve"


class TemporalAnomalyProcess:
    """Injects scheduled temporal anomaly manifestations.

    Usage::

        schedule = FactoryScheduler(config).build()
        health = RobotHealthProcess(config).run(schedule)
        base = ScheduledSignalGenerator(config).generate(schedule, health)
        out = TemporalAnomalyProcess(config).apply(schedule, health, base)
        out.validate(schedule, health, samples)

    The input samples are never mutated: abnormal files are rebuilt as
    new ``FileSample`` records sharing the base identity (file id, seed,
    provenance); normal files are returned as-is.
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()

    @property
    def temporal_cfg(self) -> TemporalAnomalyConfig:
        """Configured temporal anomaly policy parameters."""
        return self.cfg.temporal

    def apply(
        self,
        schedule: FactorySchedule,
        health: FactoryHealth,
        samples: list[FileSample],
    ) -> tuple[list[FileSample], TemporalAnomalies]:
        """Annotate every scheduled sample with temporal manifestations."""
        tcfg = self.cfg.temporal
        schedule.validate()
        health.validate(schedule, self.cfg.health)
        if len(samples) != len(schedule.events):
            raise ValueError(
                f"samples ({len(samples)}) must align 1:1 with "
                f"schedule events ({len(schedule.events)})"
            )
        for event, sample in zip(schedule.events, samples):
            if sample.operation is None or (
                sample.operation.operation_id != event.operation_id
            ):
                raise ValueError(
                    f"sample/schedule misalignment at {event.operation_id}"
                )
        episodes = {e.episode_id: e for e in health.episodes}
        failures = [e for e in health.episodes if e.kind is EpisodeKind.FAILURE]
        maintenances = [
            e for e in health.episodes if e.kind is EpisodeKind.MAINTENANCE
        ]
        out: list[FileSample] = []
        policies: list[str] = []
        for event, state, sample in zip(schedule.events, health.states, samples):
            manifested, families, policy = self._decide(event, state, tcfg)
            if not families:
                out.append(sample)
                policies.append(policy)
                continue
            assert manifested is not None
            abnormal = self._inject(
                event, state, sample, families, manifested, policy,
                episodes, failures, maintenances,
            )
            if abnormal is None:
                if policy == FAILURE_POLICY:
                    raise RuntimeError(
                        f"{event.operation_id}: FAILED operation rejected every "
                        f"candidate family; widen the anomaly strength gate"
                    )
                out.append(sample)
                policies.append("none")
            else:
                out.append(abnormal)
                policies.append(policy)
        annotation = TemporalAnomalies(
            operation_ids=[e.operation_id for e in schedule.events],
            policies=policies,
        )
        annotation.validate(schedule, health, out)
        return out, annotation

    def _decide(
        self,
        event: OperationEvent,
        state: RobotHealthState,
        tcfg: TemporalAnomalyConfig,
    ) -> tuple[float | None, list[AnomalyFamily], str]:
        """Return ``(severity, candidate families, policy)`` for one operation."""
        rng = np.random.default_rng(
            _stable_int("temporal-anomaly", str(tcfg.seed), event.operation_id)
        )
        if state.degradation_stage is DegradationStage.IN_MAINTENANCE:
            return None, [], "none"
        if state.degradation_stage is DegradationStage.FAILED:
            # A failure must manifest even when the strength gate rejects
            # the first family: rotate through every shape-preserving
            # family from a deterministic offset, so abrupt failures at
            # near-zero health still carry an obvious localized symptom.
            offset = int(rng.integers(len(TEMPORAL_FAMILIES)))
            rotated = TEMPORAL_FAMILIES[offset:] + TEMPORAL_FAMILIES[:offset]
            return tcfg.failure_severity, rotated, FAILURE_POLICY
        manifested = progressive_severity(state.manifested_value, tcfg)
        if rng.random() < tcfg.isolated_rate:
            family = TEMPORAL_FAMILIES[int(rng.integers(len(TEMPORAL_FAMILIES)))]
            severity = float(rng.uniform(0.3, 0.8))
            return severity, [family], ISOLATED_POLICY
        if rng.random() < precursor_probability(state.manifested_value, tcfg):
            family = TEMPORAL_FAMILIES[int(rng.integers(len(TEMPORAL_FAMILIES)))]
            return manifested, [family], PRECURSOR_POLICY
        return None, [], "none"

    def _inject(
        self,
        event: OperationEvent,
        state: RobotHealthState,
        sample: FileSample,
        families: list[AnomalyFamily],
        severity: float,
        policy: str,
        episodes: dict[str, HealthEpisode],
        failures: list[HealthEpisode],
        maintenances: list[HealthEpisode],
    ) -> FileSample | None:
        """Build the abnormal counterpart of one sample, or None if rejected.
        Candidate families are tried in order; the first gate-accepted
        injection wins. Single-family precursor/isolated decisions keep
        exactly one candidate, while failure manifestations rotate through
        every shape-preserving family so abrupt failures still manifest.
        """
        acfg = self.cfg.anomaly
        gate_cfg = replace(
            acfg,
            max_rejection_attempts=max(
                1, min(acfg.max_rejection_attempts, self.cfg.temporal.max_attempts)
            ),
        )
        x_clean = sample.x.astype(np.float64)
        regimes = sample.regime_sequence
        for family in families:
            rng = np.random.default_rng(
                _stable_int(
                    "temporal-inject", str(self.cfg.temporal.seed),
                    event.operation_id, family.value,
                )
            )
            def make_attempt(family: AnomalyFamily = family) -> object:
                ctx = InjectionContext(
                    x=x_clean.copy(),
                    regimes=list(regimes),
                    rng=rng,
                )
                return inject_anomaly(family, ctx, severity, acfg)
            result, strength, n_attempts = generate_with_strength_gate(
                make_attempt, x_clean, gate_cfg
            )
            if result.accepted and result.mask.shape == x_clean.shape:
                return self._assemble(
                    event, state, sample, result, strength, n_attempts,
                    severity, policy, episodes, failures, maintenances,
                )
        return None

    def _assemble(
        self,
        event: OperationEvent,
        state: RobotHealthState,
        sample: FileSample,
        result: AnomalyResult,
        strength: float,
        n_attempts: int,
        severity: float,
        policy: str,
        episodes: dict[str, HealthEpisode],
        failures: list[HealthEpisode],
        maintenances: list[HealthEpisode],
    ) -> FileSample:
        """Assemble one gate-accepted injection into an abnormal FileSample."""
        meta = result.meta
        meta.extra["strength"] = float(strength)
        meta.extra["n_attempts"] = n_attempts
        meta.extra["accepted"] = True
        meta.extra["temporal_policy"] = policy
        meta.extra["operation_id"] = event.operation_id
        meta.extra["manifested_health"] = float(state.manifested_value)
        meta.extra["degradation_episode_id"] = state.degradation_episode_id
        mask = result.mask.astype(bool)
        x_out = result.x_modified.astype(np.float32)
        episode = self._resolve_episode(
            event, state, episodes, failures, maintenances
        )
        abnormal = FileSample(
            x=x_out,
            file_id=sample.file_id,
            file_label=SampleLabel.ABNORMAL,
            seed=sample.seed,
            generator_version=sample.generator_version,
            config_hash=sample.config_hash,
            regime_sequence=(
                result.regimes_modified
                if result.regimes_modified is not None
                else sample.regime_sequence
            ),
            anomaly_meta=meta,
            anomaly_mask=mask,
            robot_idx=sample.robot_idx,
            program_idx=sample.program_idx,
            robot_code=sample.robot_code,
            program_number=sample.program_number,
            operation=sample.operation,
            operating_context=sample.operating_context,
            health=sample.health,
            episode=episode,
            anomaly_labels=ObservableAnomalyLabels(
                is_file_anomalous=True,
                anomaly_family=meta.family.value,
                anomaly_severity=float(meta.severity),
            ),
            factory_provenance=sample.factory_provenance,
        )
        abnormal.validate()
        return abnormal

    @staticmethod
    def _resolve_episode(
        event: OperationEvent,
        state: RobotHealthState,
        episodes: dict[str, HealthEpisode],
        failures: list[HealthEpisode],
        maintenances: list[HealthEpisode],
    ) -> HealthEpisode | None:
        """Resolve the episode provenance for one manifested file."""
        if state.degradation_stage is DegradationStage.FAILED:
            for failure in failures:
                if failure.robot_id == event.robot_id and math.isclose(
                    failure.start_time, event.end_time,
                    rel_tol=1e-9, abs_tol=1e-6,
                ):
                    return failure
            return None
        if state.degradation_stage is DegradationStage.IN_MAINTENANCE:
            for maintenance in maintenances:
                if maintenance.robot_id != event.robot_id:
                    continue
                end = (
                    maintenance.end_time
                    if maintenance.end_time is not None
                    else math.inf
                )
                if maintenance.start_time <= event.start_time < end:
                    return maintenance
            return None
        if state.degradation_episode_id is not None:
            return episodes.get(state.degradation_episode_id)
        return None
