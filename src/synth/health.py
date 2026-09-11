"""Robot-wide health, failure, and maintenance process (Sprint 11 Task 3).

One latent trajectory ``H_r(t)`` per robot over a fixed Task 2 factory
calendar (methodology §20.5–20.8). Health evolves causally through
calendar aging, operation-dependent wear, and small stochastic variation::

    H_next = H + aging_rate * dt_wall + wear_rate * duration + noise

Programs reveal the shared trajectory with deterministic per-pair
sensitivity ``G = γ * H`` (§20.5); programs never own independent health.
Failures arise from a hazard coupled to health and accumulated usage::

    λ = abrupt_rate + base_rate * exp(alpha * H + beta * W)
    P(fail in op) = 1 - exp(-λ * duration)

never from independently randomized timestamps (§20.8). A nonzero
``abrupt_rate`` supports failures with little or no precursor (§20.7).
Each failure opens an explicit maintenance episode; the trajectory is
frozen across the maintenance window and restarts from a recommissioned
draw afterwards, so pre- and post-maintenance paths are never connected
(§20.7).

Every emitted state is a Task 1 ``RobotHealthState`` and every boundary a
Task 1 ``HealthEpisode``; cross-state validation lives in
``FactoryHealth.validate``. The schedule is read-only: health annotates
the Task 2 calendar and can never reorder it. One dedicated
``numpy.random.Generator`` seeded from ``HealthConfig.seed`` drives all
health randomness in deterministic (robot, time) order, never shared with
scheduling or signal streams (§20.14). Signal synthesis, anomaly
injection, labels, splits, and materialization are later tasks and MUST
NOT live here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from synth.config import HealthConfig, SynthConfig
from synth.scheduler import FactorySchedule
from synth.schema import (
    DegradationStage,
    EpisodeKind,
    FailureEvent,
    HealthEpisode,
    OperationEvent,
    RobotHealthState,
)

#: Cap on the hazard exponent; beyond it the per-operation failure
#: probability already saturates at 1.
_HAZARD_ARG_CAP = 50.0

#: Minimum degradation duration in days before a non-abrupt cohort may fire.
#: Sprint 15 protocol v2 §7a; values sourced verbatim from the frozen audit
#: floors (``synth.balanced.audit_sprint15``: P ``min >= 2.0 d``,
#: W ``min >= 6.0 d``). Consulted only when
#: ``HealthConfig.min_duration_gate`` is enabled (candidate-2 profiles);
#: legacy and candidate-1 paths (flag off) never observe this table.
MIN_DEGRADATION_DURATION_D = {"P": 2.0, "W": 6.0}

#: Tolerance (seconds) for health-episode boundary comparisons.
_TIME_ABS_TOL = 1e-6

#: Severity bands mapping capped severity in [0, 1] onto the causal
#: progression healthy → drift → intermittent → persistent → obvious.
_STAGE_BANDS = (
    (0.15, DegradationStage.HEALTHY),
    (0.35, DegradationStage.LATENT_DRIFT),
    (0.55, DegradationStage.INTERMITTENT),
    (0.75, DegradationStage.PERSISTENT),
)


def program_sensitivity(robot_id: str, program_id: str, health: HealthConfig) -> float:
    """Return the deterministic program sensitivity γ for one robot-program pair.

    Derived from a stable hash of the identity pair, so sensitivities are
    structural (identical on every run and seed) and cost no RNG draws.
    """
    digest = hashlib.sha256(f"{robot_id}\x00{program_id}".encode()).hexdigest()
    unit = int(digest[:12], 16) / 16**12
    return health.sensitivity_min + unit * (health.sensitivity_max - health.sensitivity_min)


def _stage_for_severity(severity: float) -> DegradationStage:
    """Map capped severity in [0, 1] onto the causal stage progression."""
    for bound, stage in _STAGE_BANDS:
        if severity < bound:
            return stage
    return DegradationStage.OBVIOUS


@dataclass
class FactoryHealth:
    """Health annotation of one fixed factory calendar.

    ``states`` aligns 1:1 with the schedule's events in schedule order;
    ``operation_ids`` pins that alignment. ``episodes`` holds every
    degradation, failure, and maintenance boundary. ``validate`` enforces
    the Task 3 structural invariants against the schedule the run used.
    """

    operation_ids: list[str] = field(default_factory=list)
    states: list[RobotHealthState] = field(default_factory=list)
    episodes: list[HealthEpisode] = field(default_factory=list)
    failure_events: list[FailureEvent] = field(default_factory=list)
    seed: int = 0
    span_s: float = 0.0

    def by_robot(
        self, schedule: FactorySchedule
    ) -> dict[str, list[tuple[OperationEvent, RobotHealthState]]]:
        """Group (event, state) pairs per robot in causal time order."""
        by_event = {e.operation_id: e for e in schedule.events}
        grouped: dict[str, list[tuple[OperationEvent, RobotHealthState]]] = {}
        for operation_id, state in zip(self.operation_ids, self.states):
            source = by_event[operation_id]
            grouped.setdefault(source.robot_id, []).append((source, state))
        for pairs in grouped.values():
            pairs.sort(key=lambda pair: (pair[0].start_time, pair[0].end_time))
        return grouped

    def validate(self, schedule: FactorySchedule, health: HealthConfig | None = None) -> None:
        """Assert health-over-schedule structural invariants.

        ``health`` optionally supplies the process config so the
        no-drop-across-operations check tolerates per-step stochastic
        variation; without it, drops are only allowed across explicit
        maintenance boundaries.
        """
        drop_tol = 1e-9 if health is None else 10.0 * health.noise_scale + 1e-9
        assert self.states, "health must annotate at least one operation"
        assert len(self.states) == len(schedule.events) == len(self.operation_ids), (
            "states must align 1:1 with schedule events"
        )
        assert self.operation_ids == [e.operation_id for e in schedule.events], (
            "health alignment must follow schedule order"
        )
        assert self.span_s == schedule.span_s, "health span must match the calendar"
        episode_ids = [e.episode_id for e in self.episodes]
        assert len(set(episode_ids)) == len(episode_ids), "episode ids must be unique"
        known_robots = {e.robot_id for e in schedule.events}
        degradation_ids: dict[str, HealthEpisode] = {}
        failures: list[HealthEpisode] = []
        maintenances: list[HealthEpisode] = []
        for episode in self.episodes:
            assert episode.robot_id in known_robots, (
                f"{episode.episode_id}: unknown robot {episode.robot_id}"
            )
            if episode.kind is EpisodeKind.DEGRADATION:
                degradation_ids[episode.episode_id] = episode
            elif episode.kind is EpisodeKind.FAILURE:
                assert episode.end_time == episode.start_time, (
                    f"{episode.episode_id}: failures are point episodes"
                )
                failures.append(episode)
            else:
                assert episode.end_time is not None and episode.end_time >= episode.start_time, (
                    f"{episode.episode_id}: maintenance needs a bounded window"
                )
                maintenances.append(episode)
        if self.failure_events:
            unmatched = list(failures)
            for record in self.failure_events:
                if record.cohort == "A":
                    assert record.degradation_onset is None, (
                        f"{record.failure_id}: abrupt records carry no onset")
                else:
                    assert record.degradation_onset is not None, (
                        f"{record.failure_id}: non-abrupt records need onset")
                hits = [
                    f for f in unmatched
                    if f.robot_id == record.robot_id and math.isclose(
                        f.start_time, record.failure_time,
                        rel_tol=1e-9, abs_tol=_TIME_ABS_TOL)
                ]
                assert len(hits) == 1, (
                    f"{record.failure_id}: needs exactly one failure episode")
                unmatched.remove(hits[0])
            assert not unmatched, "every failure episode needs a ledger record"
        by_event = {e.operation_id: e for e in schedule.events}
        for operation_id, state in zip(self.operation_ids, self.states):
            event = by_event[operation_id]
            assert math.isclose(
                state.manifested_value,
                state.program_sensitivity * state.health_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ), f"{operation_id}: manifested value must equal γ·H"
            if state.degradation_episode_id is not None:
                assert state.degradation_episode_id in degradation_ids, (
                    f"{operation_id}: unknown degradation episode "
                    f"{state.degradation_episode_id}"
                )
                assert degradation_ids[state.degradation_episode_id].robot_id == event.robot_id, (
                    f"{operation_id}: degradation episode must belong to the same robot"
                )
            if state.degradation_stage is DegradationStage.FAILED:
                assert any(
                    f.robot_id == event.robot_id
                    and math.isclose(f.start_time, event.end_time,
                                     rel_tol=1e-9, abs_tol=_TIME_ABS_TOL)
                    for f in failures
                ), f"{operation_id}: FAILED state needs a failure at operation end"
            if state.degradation_stage is DegradationStage.IN_MAINTENANCE:
                assert any(
                    m.robot_id == event.robot_id
                    and m.start_time <= event.start_time < (m.end_time or math.inf)
                    for m in maintenances
                ), f"{operation_id}: maintenance state must sit inside a maintenance window"
        grouped = self.by_robot(schedule)
        for robot_id, pairs in grouped.items():
            windows = sorted(
                (m.start_time, m.end_time if m.end_time is not None else math.inf)
                for m in maintenances if m.robot_id == robot_id
            )
            previous: tuple[OperationEvent, RobotHealthState] | None = None
            for event, state in pairs:
                if previous is not None:
                    prev_event, prev_state = previous
                    if state.health_value < prev_state.health_value - drop_tol:
                        assert any(
                            prev_event.start_time < end <= event.start_time + _TIME_ABS_TOL
                            for _, end in windows
                        ), (
                            f"robot {robot_id}: health must not drop except "
                            f"across a maintenance boundary"
                        )
                previous = (event, state)


class RobotHealthProcess:
    """Simulates one latent health trajectory per robot over a fixed schedule.

    Usage::

        schedule = FactoryScheduler(config).build()
        health = RobotHealthProcess(config).run(schedule)
        health.validate(schedule, config.health)

    The same config and schedule always produce identical trajectories.
    Changing the health seed never reorders the schedule: the calendar is
    read-only input built solely from ``SchedulerConfig.seed``.
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()

    @property
    def health_cfg(self) -> HealthConfig:
        """Configured health process parameters."""
        return self.cfg.health

    def run(self, schedule: FactorySchedule) -> FactoryHealth:
        """Annotate every scheduled operation with robot health state."""
        hcfg = self.cfg.health
        rng = np.random.default_rng(hcfg.seed)
        if not schedule.events:
            raise ValueError("health requires a schedule with at least one operation")
        by_robot: dict[str, list[int]] = {}
        for index, event in enumerate(schedule.events):
            by_robot.setdefault(event.robot_id, []).append(index)
        for indices in by_robot.values():
            indices.sort(key=lambda i: (schedule.events[i].start_time,
                                        schedule.events[i].end_time))
        states: list[RobotHealthState | None] = [None] * len(schedule.events)
        episodes: list[HealthEpisode] = []
        failure_events: list[FailureEvent] = []
        counters: dict[str, int] = {}
        for robot_id in sorted(by_robot):
            trajectory = self._run_robot(
                robot_id, [schedule.events[i] for i in by_robot[robot_id]],
                hcfg, rng, episodes, failure_events, counters,
            )
            for index, state in zip(by_robot[robot_id], trajectory):
                states[index] = state
        assert all(s is not None for s in states)
        result = FactoryHealth(
            operation_ids=[e.operation_id for e in schedule.events],
            states=[s for s in states if s is not None],
            episodes=episodes,
            failure_events=failure_events,
            seed=hcfg.seed,
            span_s=schedule.span_s,
        )
        result.validate(schedule, hcfg)
        return result

    def _run_robot(
        self,
        robot_id: str,
        ordered: list[OperationEvent],
        hcfg: HealthConfig,
        rng: np.random.Generator,
        episodes: list[HealthEpisode],
        failure_events: list[FailureEvent],
        counters: dict[str, int],
    ) -> list[RobotHealthState]:
        """Evolve one robot trajectory in causal time order."""
        out: list[RobotHealthState] = []
        health = 0.0
        usage = 0.0
        last_end = 0.0
        maint_until: float | None = None
        frozen = 0.0
        open_degradation: HealthEpisode | None = None
        open_deg_id: str | None = None
        preventive = self._preventive_blocks(
            robot_id, ordered, hcfg, episodes, counters)
        by_id = ({c.cohort_id: c for c in hcfg.cohorts} if hcfg.cohorts else {})

        def _draw_upcoming() -> str | None:
            """Draw the next non-abrupt cohort before it manifests."""
            if not by_id or "P" not in by_id or "W" not in by_id:
                return None
            return "P" if rng.random() < hcfg.upcoming_p else "W"

        deg_sub_ordinal: dict[tuple[str, str], int] = {}
        abrupt_ordinal: dict[str, int] = {}

        def _draw_pw_subtype(cohort: str | None) -> str | None:
            """Draw one precursor-manifestation subtype at episode opening.

            Sprint 14 protocol v3: causal selection on a dedicated sub-stream
            derived from (health seed, robot, cohort, per-cohort episode
            ordinal). The shared health RNG is never touched, so hazard
            draws — and hence failure timing and density — are invariant
            to subtype emission. Returns None for legacy configs whose
            cohort declares no subtypes (v4.1 behavior preserved exactly).

            Sprint 15 protocol v7 (flag-gated): with
            ``stratified_subtype_emission`` set, the uniform pick is replaced
            by deterministic round-robin alternation
            ``labels[(ordinal + digest[0]) % len(labels)]`` on the same
            dedicated inputs, so emitted per-(robot, cohort) subtype counts
            differ by at most one. Flag-off behavior is byte-identical to v6
            and earlier.
            """
            if cohort is None or cohort not in by_id:
                return None
            labels = by_id[cohort].subtypes
            if not labels:
                return None
            key = (robot_id, cohort)
            deg_sub_ordinal[key] = deg_sub_ordinal.get(key, 0) + 1
            digest = hashlib.sha256(
                "|".join(["sprint14-subtype", str(hcfg.seed), robot_id,
                          cohort, str(deg_sub_ordinal[key])]).encode()
            ).digest()
            if getattr(hcfg, "stratified_subtype_emission", False):
                return labels[
                    (deg_sub_ordinal[key] + digest[0]) % len(labels)]
            sub_rng = np.random.default_rng(
                int.from_bytes(digest[:8], "big"))
            return labels[int(sub_rng.integers(len(labels)))]

        def _next_id(kind: str) -> str:
            key = f"{robot_id}:{kind}"
            counters[key] = counters.get(key, 0) + 1
            return f"{kind}-{robot_id}-{counters[key]:04d}"

        upcoming = _draw_upcoming()

        for event in ordered:
            gamma = program_sensitivity(event.robot_id, event.program_id, hcfg)
            for block_start, block_end in preventive:
                if block_start <= event.start_time < block_end:
                    frozen = health
                    if maint_until is None or block_end > maint_until:
                        maint_until = block_end
                    break
            if maint_until is not None and event.start_time < maint_until:
                severity = min(1.0, frozen / hcfg.severity_scale)
                out.append(RobotHealthState(
                    health_value=frozen,
                    program_sensitivity=gamma,
                    manifested_value=gamma * frozen,
                    degradation_stage=DegradationStage.IN_MAINTENANCE,
                    degradation_severity=severity,
                    degradation_episode_id=None,
                ))
                last_end = event.end_time
                continue
            recommissioned = False
            if maint_until is not None and event.start_time >= maint_until:
                if open_degradation is not None:
                    open_degradation.end_time = maint_until
                    open_degradation = None
                    open_deg_id = None
                drawn = (
                    rng.normal(hcfg.recommission_mean, hcfg.recommission_scale)
                    if hcfg.recommission_scale > 0.0 else hcfg.recommission_mean
                )
                health = max(0.0, drawn)
                usage = 0.0
                maint_until = None
                recommissioned = True
                upcoming = _draw_upcoming()
            health += hcfg.aging_rate * max(0.0, event.start_time - last_end)
            if open_degradation is not None and upcoming is not None and by_id:
                cohort_wear = by_id[upcoming].wear_rate
                health += (cohort_wear if cohort_wear is not None
                           else hcfg.wear_rate) * event.duration
            else:
                health += hcfg.wear_rate * event.duration
            if hcfg.noise_scale > 0.0:
                health += rng.normal(0.0, hcfg.noise_scale)
            health = max(0.0, health)
            usage += event.duration
            severity = min(1.0, health / hcfg.severity_scale)
            if health >= hcfg.degradation_onset and open_degradation is None:
                open_degradation = HealthEpisode(
                    episode_id=_next_id("deg"),
                    kind=EpisodeKind.DEGRADATION,
                    robot_id=robot_id,
                    start_time=event.start_time,
                    subtype=_draw_pw_subtype(upcoming),
                )
                episodes.append(open_degradation)
                open_deg_id = open_degradation.episode_id
            exponent = min(
                _HAZARD_ARG_CAP, hcfg.alpha * health + hcfg.beta * usage)
            fired = None
            legacy_fires = False
            if by_id:
                if upcoming is not None:
                    upcoming_cohort = by_id.get(upcoming)
                else:
                    upcoming_cohort = None
                duration_gate_open = (
                    not hcfg.min_duration_gate
                    or open_degradation is None
                    or upcoming_cohort is None
                    or upcoming_cohort.cohort_id == "A"
                    or (event.end_time - open_degradation.start_time) / 86400.0
                    >= MIN_DEGRADATION_DURATION_D.get(
                        upcoming_cohort.cohort_id, 0.0)
                )
                if (open_degradation is not None and upcoming_cohort is not None
                        and upcoming_cohort.failure_threshold_h > 0.0
                        and duration_gate_open
                        and health >= upcoming_cohort.failure_threshold_h):
                    fired = upcoming_cohort
                if fired is None:
                    abrupt_cfg = by_id.get("A")
                    if abrupt_cfg is not None and abrupt_cfg.abrupt_rate > 0.0:
                        if rng.random() < 1.0 - math.exp(
                                -abrupt_cfg.abrupt_rate * event.duration):
                            fired = abrupt_cfg
                if (fired is None and open_degradation is not None
                        and upcoming_cohort is not None
                        and duration_gate_open):
                    cohort = upcoming_cohort
                    rate = cohort.base_rate * math.exp(exponent)
                    if rng.random() < 1.0 - math.exp(-rate * event.duration):
                        fired = cohort
            else:
                hazard = hcfg.abrupt_rate + hcfg.base_rate * math.exp(exponent)
                threshold = 1.0 - math.exp(-hazard * event.duration)
                legacy_fires = rng.random() < threshold
            if fired is not None or legacy_fires:
                failure_id = _next_id("fail")
                cohort_id: str | None = None
                subtype: str | None = None
                onset: float | None = None
                duration_d = 0.0
                if fired is not None and fired.cohort_id == "A":
                    cohort_id = "A"
                    if getattr(hcfg, "stratified_subtype_emission", False):
                        abrupt_ordinal[robot_id] = (
                            abrupt_ordinal.get(robot_id, 0) + 1)
                        offset = hashlib.sha256(
                            "|".join(["sprint15-stratified-A",
                                      str(hcfg.seed),
                                      robot_id]).encode()).digest()[0]
                        labels = by_id["A"].subtypes
                        subtype = labels[
                            (abrupt_ordinal[robot_id] + offset)
                            % len(labels)]
                        rng.random()  # stream-preserving no-op: keeps legacy
                        # shared-RNG consumption bit-identical so failure
                        # timing/density are invariant to emission.
                    else:
                        subtype = "A1" if rng.random() < 0.5 else "A2"
                    sev_level = (1.0, 2.0, 4.0)[int(rng.integers(3))]
                elif fired is not None:
                    assert open_degradation is not None and open_deg_id is not None
                    cohort_id = fired.cohort_id
                    subtype = open_degradation.subtype
                    assert subtype is None or subtype in by_id[cohort_id].subtypes, (
                        f"episode subtype {subtype!r} not declared by "
                        f"cohort {cohort_id!r}")
                    onset = open_degradation.start_time
                    duration_d = (event.end_time - onset) / 86400.0
                    sev_level = (1.0, 2.0, 4.0)[int(rng.integers(3))]
                    open_degradation.end_time = event.end_time
                    open_degradation = None
                elif open_degradation is not None:
                    open_degradation.end_time = event.end_time
                    open_degradation = None
                episodes.append(HealthEpisode(
                    episode_id=failure_id,
                    kind=EpisodeKind.FAILURE,
                    robot_id=robot_id,
                    start_time=event.end_time,
                    end_time=event.end_time,
                ))
                maint_until = event.end_time + hcfg.maintenance_duration_s
                maint_id = _next_id("maint")
                episodes.append(HealthEpisode(
                    episode_id=maint_id,
                    kind=EpisodeKind.MAINTENANCE,
                    robot_id=robot_id,
                    start_time=event.end_time,
                    end_time=maint_until,
                ))
                if fired is not None and cohort_id is not None:
                    failure_events.append(FailureEvent(
                        failure_id=failure_id,
                        robot_id=robot_id,
                        failure_time=event.end_time,
                        cohort=cohort_id,
                        subtype=subtype,
                        degradation_onset=onset,
                        duration_d=duration_d,
                        severity=sev_level,
                        degradation_episode_id=(
                            None if cohort_id == "A" else open_deg_id),
                        maintenance_episode_id=maint_id,
                    ))
                frozen = health
                out.append(RobotHealthState(
                    health_value=health,
                    program_sensitivity=gamma,
                    manifested_value=gamma * health,
                    degradation_stage=DegradationStage.FAILED,
                    degradation_severity=severity,
                    degradation_episode_id=open_deg_id,
                ))
                if cohort_id != "A":
                    open_deg_id = None
            else:
                if recommissioned:
                    stage = DegradationStage.RECOMMISSIONED
                    if health >= hcfg.degradation_onset and open_degradation is None:
                        open_degradation = HealthEpisode(
                            episode_id=_next_id("deg"),
                            kind=EpisodeKind.DEGRADATION,
                            robot_id=robot_id,
                            start_time=event.start_time,
                            subtype=_draw_pw_subtype(upcoming),
                        )
                        episodes.append(open_degradation)
                        open_deg_id = open_degradation.episode_id
                else:
                    stage = _stage_for_severity(severity)
                out.append(RobotHealthState(
                    health_value=health,
                    program_sensitivity=gamma,
                    manifested_value=gamma * health,
                    degradation_stage=stage,
                    degradation_severity=severity,
                    degradation_episode_id=open_deg_id,
                ))
            last_end = event.end_time
        return out
    @staticmethod
    def _preventive_blocks(
        robot_id: str,
        ordered: list[OperationEvent],
        hcfg: HealthConfig,
        episodes: list[HealthEpisode],
        counters: dict[str, int],
    ) -> list[tuple[float, float]]:
        """Emit explicit preventive maintenance windows for one robot.

        Blocks recur every ``preventive_interval_s`` from t=0 while the
        interval start precedes the robot's last operation end. Disabled
        (empty) when the interval is 0. Each block is an explicit
        recommissioning boundary like corrective maintenance.
        """
        blocks: list[tuple[float, float]] = []
        if hcfg.preventive_interval_s <= 0.0 or not ordered:
            return blocks
        horizon = max(e.end_time for e in ordered)
        index = 1
        while index * hcfg.preventive_interval_s < horizon:
            start = index * hcfg.preventive_interval_s
            end = start + hcfg.preventive_duration_s
            key = f"{robot_id}:pmaint"
            counters[key] = counters.get(key, 0) + 1
            episodes.append(HealthEpisode(
                episode_id=f"pmaint-{robot_id}-{counters[key]:04d}",
                kind=EpisodeKind.MAINTENANCE,
                robot_id=robot_id,
                start_time=start,
                end_time=end,
            ))
            blocks.append((start, end))
            index += 1
        return blocks
