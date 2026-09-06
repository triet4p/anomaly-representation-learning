"""Focused Task 3 tests: robot-wide health, failure, and maintenance.

Covers one latent trajectory per robot, causal usage progression,
program-sensitive manifestation, hazard-coupled failures, abrupt
failures, maintenance freeze/reset boundaries, deterministic replay,
and RNG-stream independence from scheduling and signal noise. Signal
synthesis, anomalies, labels, splits, and materialization are later
tasks and are NOT tested here.
"""

from __future__ import annotations

import copy

import pytest

from synth.config import (
    HealthConfig,
    NoiseConfig,
    RouteConfig,
    RouteStageConfig,
    SchedulerConfig,
    SynthConfig,
)
from synth.health import FactoryHealth, RobotHealthProcess, program_sensitivity
from synth.scheduler import FactorySchedule, FactoryScheduler
from synth.schema import DegradationStage, EpisodeKind


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _line_scheduler(n_units: int = 8, seed: int = 3) -> SchedulerConfig:
    return SchedulerConfig(
        n_units=n_units,
        arrival_interval_s=60.0,
        seed=seed,
        routes=[RouteConfig(
            route_id="route-S", product_type="sedan", stages=[
                RouteStageConfig(robot_id="robot-01", program_id="program-01",
                                 duration_s=120.0, travel_after_s=10.0),
                RouteStageConfig(robot_id="robot-01", program_id="program-02",
                                 duration_s=120.0, travel_after_s=0.0),
            ])],
    )


def _two_route_config(**overrides) -> SynthConfig:
    route_a = RouteConfig(
        route_id="route-A", product_type="sedan", stages=[
            RouteStageConfig(robot_id="robot-01", program_id="program-01",
                             duration_s=300.0, travel_after_s=60.0),
            RouteStageConfig(robot_id="robot-02", program_id="program-02",
                             duration_s=300.0, travel_after_s=0.0),
        ])
    route_b = RouteConfig(
        route_id="route-B", product_type="hatchback", stages=[
            RouteStageConfig(robot_id="robot-02", program_id="program-03",
                             duration_s=400.0, travel_after_s=30.0),
            RouteStageConfig(robot_id="robot-01", program_id="program-01",
                             duration_s=200.0, travel_after_s=0.0),
        ])
    fields: dict[str, object] = {
        "n_units": 12, "arrival_interval_s": 120.0,
        "routes": [route_a, route_b], "seed": 7,
    }
    fields.update(overrides)
    return SynthConfig(scheduler=SchedulerConfig(**fields))  # type: ignore[arg-type]


def _progressive_health(**overrides) -> HealthConfig:
    fields: dict[str, object] = {
        "seed": 1, "wear_rate": 0.00125, "noise_scale": 0.0,
        "aging_rate": 0.0, "base_rate": 1e-4, "alpha": 3.0,
        "maintenance_duration_s": 300.0,
        "recommission_mean": 0.0, "recommission_scale": 0.0,
    }
    fields.update(overrides)
    return HealthConfig(**fields)  # type: ignore[arg-type]


def _run(scheduler_cfg=None, health_cfg=None) -> tuple:
    cfg = SynthConfig(
        scheduler=scheduler_cfg or _line_scheduler(),
        health=health_cfg or _progressive_health(),
    )
    schedule = FactoryScheduler(cfg).build()
    return schedule, RobotHealthProcess(cfg).run(schedule)


# ---------------------------------------------------------------------------
# Trajectories and progression
# ---------------------------------------------------------------------------

def test_one_trajectory_per_robot():
    schedule = FactoryScheduler(_two_route_config()).build()
    health = RobotHealthProcess(SynthConfig()).run(schedule)
    health.validate(schedule, SynthConfig().health)
    grouped = health.by_robot(schedule)
    assert set(grouped) == {e.robot_id for e in schedule.events}
    assert len(health.states) == len(schedule.events)
    for event, state in zip(schedule.events, health.states):
        assert state.health_value >= 0.0
        assert 0.0 <= state.degradation_severity <= 1.0


def test_health_progresses_with_usage():
    schedule, health = _run()
    pairs = health.by_robot(schedule)["robot-01"]
    pre = []
    for _, state in pairs:
        if state.degradation_stage in (DegradationStage.FAILED,
                                       DegradationStage.IN_MAINTENANCE,
                                       DegradationStage.RECOMMISSIONED):
            break
        pre.append(state)
    assert len(pre) >= 4, "expected a visible pre-failure progression"
    assert [s.health_value for s in pre] == sorted(s.health_value for s in pre)
    assert [s.degradation_severity for s in pre] == sorted(
        s.degradation_severity for s in pre)
    order = [DegradationStage.HEALTHY, DegradationStage.LATENT_DRIFT,
             DegradationStage.INTERMITTENT, DegradationStage.PERSISTENT,
             DegradationStage.OBVIOUS]
    assert pre[0].degradation_stage is DegradationStage.HEALTHY
    assert order.index(pre[-1].degradation_stage) > order.index(pre[0].degradation_stage)


def test_program_sensitivity_is_structural():
    schedule = FactoryScheduler(_two_route_config()).build()
    first = RobotHealthProcess(SynthConfig(health=HealthConfig(seed=1))).run(schedule)
    second = RobotHealthProcess(SynthConfig(health=HealthConfig(seed=999))).run(schedule)
    for (event, state_a), (_, state_b) in zip(
            zip(schedule.events, first.states), zip(schedule.events, second.states)):
        assert state_a.program_sensitivity == state_b.program_sensitivity
        assert state_a.program_sensitivity == pytest.approx(
            program_sensitivity(event.robot_id, event.program_id, HealthConfig()))
        assert 0.5 <= state_a.program_sensitivity <= 1.5
        assert state_a.manifested_value == pytest.approx(
            state_a.program_sensitivity * state_a.health_value)

# ---------------------------------------------------------------------------
# Failures, abrupt support, and maintenance boundaries
# ---------------------------------------------------------------------------

def test_failure_causally_closes_degradation():
    schedule, health = _run()
    degradations = [e for e in health.episodes if e.kind is EpisodeKind.DEGRADATION]
    failures = [e for e in health.episodes if e.kind is EpisodeKind.FAILURE]
    assert failures, "progressive config must produce a failure"
    first_fail = min(failures, key=lambda e: e.start_time)
    prior = [e for e in degradations if e.start_time < first_fail.start_time]
    assert prior, "failure must be preceded by a degradation episode"
    assert prior[0].end_time == pytest.approx(first_fail.start_time)
    failed_states = [s for s in health.states
                     if s.degradation_stage is DegradationStage.FAILED]
    assert failed_states and failed_states[0].health_value >= 0.3


def test_abrupt_failure_without_precursor():
    abrupt = HealthConfig(seed=0, wear_rate=0.0, noise_scale=0.0,
                          aging_rate=0.0, base_rate=0.0, abrupt_rate=0.01,
                          maintenance_duration_s=300.0)
    schedule, health = _run(health_cfg=abrupt)
    failures = [e for e in health.episodes if e.kind is EpisodeKind.FAILURE]
    assert failures, "abrupt hazard must produce failures"
    first_fail = min(failures, key=lambda e: e.start_time)
    assert not [e for e in health.episodes
                if e.kind is EpisodeKind.DEGRADATION and e.start_time <= first_fail.start_time]
    failed = [s for s in health.states if s.degradation_stage is DegradationStage.FAILED][0]
    assert failed.health_value == 0.0
    assert failed.degradation_episode_id is None


def test_maintenance_freezes_and_resets():
    schedule, health = _run()
    maint = [e for e in health.episodes if e.kind is EpisodeKind.MAINTENANCE]
    assert maint, "progressive config must produce maintenance"
    window = maint[0]
    inside = [(e, s) for e, s in zip(schedule.events, health.states)
              if e.robot_id == window.robot_id
              and window.start_time <= e.start_time < (window.end_time or 0.0)]
    assert inside, "maintenance window must cover scheduled operations"
    assert all(s.degradation_stage is DegradationStage.IN_MAINTENANCE for _, s in inside)
    frozen = {s.health_value for _, s in inside}
    assert len(frozen) == 1, "trajectory must freeze across maintenance"
    after = [(e, s) for e, s in zip(schedule.events, health.states)
             if e.robot_id == window.robot_id and e.start_time >= (window.end_time or 0.0)]
    assert after and after[0][1].degradation_stage is DegradationStage.RECOMMISSIONED
    assert after[0][1].health_value < 0.3 < next(iter(frozen))


def test_zero_hazard_never_fails_but_degrades_open():
    calm = HealthConfig(seed=0, wear_rate=0.05, noise_scale=0.0,
                        aging_rate=0.0, base_rate=0.0, abrupt_rate=0.0)
    schedule, health = _run(health_cfg=calm)
    assert not [e for e in health.episodes if e.kind is EpisodeKind.FAILURE]
    assert not [e for e in health.episodes if e.kind is EpisodeKind.MAINTENANCE]
    degradations = [e for e in health.episodes if e.kind is EpisodeKind.DEGRADATION]
    assert degradations
    assert all(e.end_time is None for e in degradations)


def test_workload_coupling_drives_failures_without_health():
    workload = HealthConfig(seed=1, wear_rate=0.0, noise_scale=0.0,
                            aging_rate=0.0, base_rate=1e-4, beta=0.002,
                            maintenance_duration_s=300.0)
    schedule, health = _run(health_cfg=workload)
    pairs = health.by_robot(schedule)["robot-01"]
    failed_idx = [i for i, (_, s) in enumerate(pairs)
                  if s.degradation_stage is DegradationStage.FAILED]
    assert failed_idx and failed_idx[0] > 0, "usage must accumulate before failure"
    assert pairs[failed_idx[0]][1].health_value == 0.0


# ---------------------------------------------------------------------------
# Determinism and stream independence
# ---------------------------------------------------------------------------

def test_deterministic_replay():
    schedule, first = _run()
    _, second = _run()
    assert [s for s in first.states] == [s for s in second.states]
    assert [(e.episode_id, e.kind, e.robot_id, e.start_time, e.end_time)
            for e in first.episodes] == [
            (e.episode_id, e.kind, e.robot_id, e.start_time, e.end_time)
            for e in second.episodes]


def test_health_seed_changes_trajectory_not_calendar():
    noisy = _progressive_health(noise_scale=0.01)
    cfg_a = SynthConfig(scheduler=_line_scheduler(), health=noisy)
    cfg_b = SynthConfig(scheduler=_line_scheduler(),
                        health=_progressive_health(seed=2, noise_scale=0.01))
    sched_a = FactoryScheduler(cfg_a).build()
    sched_b = FactoryScheduler(cfg_b).build()
    assert [e for e in sched_a.events] == [e for e in sched_b.events]
    health_a = RobotHealthProcess(cfg_a).run(sched_a)
    health_b = RobotHealthProcess(cfg_b).run(sched_b)
    assert ([s.health_value for s in health_a.states]
            != [s.health_value for s in health_b.states])


def test_signal_noise_resampling_never_reorders_health():
    base = SynthConfig(scheduler=_line_scheduler(), health=_progressive_health())
    other = SynthConfig(scheduler=_line_scheduler(), health=_progressive_health(),
                        noise=NoiseConfig(s0_gaussian_std=0.5))
    schedule = FactoryScheduler(base).build()
    first = RobotHealthProcess(base).run(schedule)
    second = RobotHealthProcess(other).run(schedule)
    assert [s for s in first.states] == [s for s in second.states]


# ---------------------------------------------------------------------------
# Boundaries and schema compatibility
# ---------------------------------------------------------------------------

def test_empty_schedule_fails_fast():
    with pytest.raises(ValueError, match="at least one operation"):
        RobotHealthProcess(SynthConfig()).run(FactorySchedule(events=[], span_s=1.0))


def test_validate_rejects_tampered_manifestation():
    schedule, health = _run()
    tampered = copy.deepcopy(health)
    tampered.states[0].manifested_value += 1.0
    with pytest.raises(AssertionError, match="γ·H"):
        tampered.validate(schedule, _progressive_health())


def test_invalid_health_configs_rejected():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        HealthConfig(seed=-1)
    with pytest.raises(ValueError, match="seed must be an integer"):
        HealthConfig(seed=True)
    for name in ("aging_rate", "wear_rate", "noise_scale", "base_rate",
                 "abrupt_rate", "maintenance_duration_s", "recommission_scale"):
        with pytest.raises(ValueError, match=name):
            HealthConfig(**{name: -0.5})
    with pytest.raises(ValueError, match="sensitivity"):
        HealthConfig(sensitivity_min=2.0, sensitivity_max=1.0)
    with pytest.raises(ValueError, match="severity_scale must be positive"):
        HealthConfig(severity_scale=0.0)
    with pytest.raises(ValueError, match="degradation_onset"):
        HealthConfig(degradation_onset=-0.1)
