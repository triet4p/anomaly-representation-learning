"""Focused Task 2 tests: deterministic shared-unit factory scheduler.

Covers causal arrivals, shared routes, single-server robot availability
(no same-robot overlap), causal route order, nonnegative queue/travel/idle
times, asynchronous cross-robot execution, fixed-seed determinism, and
clear impossible-config errors. Health, signals, anomalies, splits,
materialization, and models are later tasks and are NOT tested here.
"""
from __future__ import annotations

import pytest

import numpy as np

from synth.config import (
    FactoryCalendarConfig,
    RouteConfig,
    RouteStageConfig,
    SchedulerConfig,
    SynthConfig,
)
from synth.scheduler import FactoryScheduler
from synth.schema import FileSample, RegimeMeta, RegimeType, SampleLabel


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _two_route_config(**overrides) -> SynthConfig:
    route_a = RouteConfig(
        route_id="route-A",
        product_type="sedan",
        stages=[
            RouteStageConfig(robot_id="robot-01", program_id="program-01",
                             duration_s=300.0, travel_after_s=60.0),
            RouteStageConfig(robot_id="robot-02", program_id="program-02",
                             duration_s=300.0, travel_after_s=0.0),
        ],
    )
    route_b = RouteConfig(
        route_id="route-B",
        product_type="hatchback",
        stages=[
            RouteStageConfig(robot_id="robot-02", program_id="program-03",
                             duration_s=400.0, travel_after_s=30.0),
            RouteStageConfig(robot_id="robot-01", program_id="program-01",
                             duration_s=200.0, travel_after_s=0.0),
        ],
    )
    fields: dict[str, object] = {
        "n_units": 12,
        "arrival_interval_s": 120.0,
        "routes": [route_a, route_b],
        "seed": 7,
    }
    fields.update(overrides)
    return SynthConfig(scheduler=SchedulerConfig(**fields))  # type: ignore[arg-type]


def _by_robot(schedule) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for event in schedule.events:
        grouped.setdefault(event.robot_id, []).append(event)
    return grouped


def _by_unit(schedule) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for event in schedule.events:
        grouped.setdefault(event.unit_id, []).append(event)
    return grouped


# ---------------------------------------------------------------------------
# Accepted schedules
# ---------------------------------------------------------------------------

def test_default_schedule_builds_and_validates():
    schedule = FactoryScheduler().build()
    schedule.validate()
    assert len(schedule.events) == 20 * 2
    assert len({event.operation_id for event in schedule.events}) == 40
    assert schedule.seed == 0
    assert schedule.span_s == 120.0 * 86400.0


def test_no_same_robot_overlap():
    schedule = FactoryScheduler(_two_route_config()).build()
    for robot_id, events in _by_robot(schedule).items():
        ordered = sorted(events, key=lambda event: event.start_time)
        assert len(ordered) >= 2, f"{robot_id} should see contention"
        for previous, current in zip(ordered, ordered[1:]):
            assert previous.end_time <= current.start_time, (
                f"robot {robot_id}: overlap {previous.operation_id} / "
                f"{current.operation_id}"
            )


def test_causal_route_ordering_and_stable_identity():
    schedule = FactoryScheduler(_two_route_config()).build()
    for unit_id, events in _by_unit(schedule).items():
        ordered = sorted(events, key=lambda event: event.route_position)
        assert [event.route_position for event in ordered] == [0, 1]
        assert len({event.route_id for event in ordered}) == 1
        assert len({event.product_type for event in ordered}) == 1
        first, second = ordered
        assert second.arrival_time == pytest.approx(
            first.end_time + first.travel_time)
        assert second.start_time >= second.arrival_time


def test_nonnegative_queue_travel_idle_durations():
    schedule = FactoryScheduler(_two_route_config()).build()
    assert schedule.events, "expected queued operations under contention"
    queued = [e for e in schedule.events if e.queue_delay > 0.0]
    assert queued, "shared robots must produce real queues"
    for event in schedule.events:
        assert event.duration > 0.0
        assert event.queue_delay >= 0.0
        assert event.travel_time >= 0.0
        assert event.idle_before >= 0.0
        assert event.queue_delay == pytest.approx(
            event.start_time - event.arrival_time)


def test_cross_robot_execution_is_asynchronous():
    schedule = FactoryScheduler(_two_route_config()).build()
    overlaps = [
        (a.operation_id, b.operation_id)
        for a in schedule.events
        for b in schedule.events
        if a.robot_id < b.robot_id
        and a.start_time < b.end_time
        and b.start_time < a.end_time
    ]
    assert overlaps, "distinct robots must process units simultaneously"


def test_stable_output_under_fixed_seed():
    config = _two_route_config()
    first = FactoryScheduler(config).build()
    second = FactoryScheduler(config).build()
    assert [e for e in first.events] == [e for e in second.events]


def test_seed_changes_route_assignment():
    route_ids = lambda seed: [  # noqa: E731
        e.route_id for e in
        FactoryScheduler(_two_route_config(seed=seed)).build().events
    ]
    assert route_ids(7) != route_ids(8)


def test_events_satisfy_task1_schema_and_file_compat():
    schedule = FactoryScheduler(_two_route_config()).build()
    event = schedule.events[0]
    sample = FileSample(
        file_id="sched-compat",
        file_label=SampleLabel.NORMAL,
        x=np.zeros((6, 32), dtype="float32"),
        seed=schedule.seed,
        generator_version="test",
        config_hash="hash",
        regime_sequence=[RegimeMeta(RegimeType.ACTIVE, 0, 32, 0.5)],
        operation=event,
        factory_provenance=None,
    )
    sample.validate()
    assert sample.encoder_inputs()["operation"] == event


def test_schedule_validate_rejects_overlap_and_broken_chain():
    schedule = FactoryScheduler(_two_route_config()).build()
    broken = FactoryScheduler(_two_route_config()).build()
    first, second = sorted(
        [e for e in broken.events if e.robot_id == "robot-01"],
        key=lambda e: e.start_time)[:2]
    first.end_time = second.end_time + 60.0
    with pytest.raises(AssertionError, match="overlapping operations"):
        broken.validate()
    chained = FactoryScheduler(_two_route_config()).build()
    unit_events = sorted(
        _by_unit(chained)[schedule.events[0].unit_id],
        key=lambda e: e.route_position)
    unit_events[1].arrival_time -= 50.0
    with pytest.raises(AssertionError, match="previous end \\+ travel"):
        chained.validate()


# ---------------------------------------------------------------------------
# Impossible configurations fail clearly
# ---------------------------------------------------------------------------

def test_impossible_scheduler_configs_rejected():
    with pytest.raises(ValueError, match="at least one route"):
        SchedulerConfig(routes=[])
    with pytest.raises(ValueError, match="at least one stage"):
        RouteConfig(route_id="route-X", product_type="sedan", stages=[])
    with pytest.raises(ValueError, match="must be unique"):
        SchedulerConfig(routes=[
            RouteConfig(route_id="route-A", product_type="sedan", stages=[
                RouteStageConfig(robot_id="r", program_id="p", duration_s=1.0)]),
            RouteConfig(route_id="route-A", product_type="coupe", stages=[
                RouteStageConfig(robot_id="r", program_id="p", duration_s=1.0)]),
        ])
    with pytest.raises(ValueError, match="duration_s must be positive"):
        RouteStageConfig(robot_id="r", program_id="p", duration_s=0.0)
    with pytest.raises(ValueError, match="travel_after_s must be non-negative"):
        RouteStageConfig(robot_id="r", program_id="p",
                         duration_s=1.0, travel_after_s=-1.0)
    with pytest.raises(ValueError, match="final stage must have travel_after_s=0"):
        RouteConfig(route_id="route-X", product_type="sedan", stages=[
            RouteStageConfig(robot_id="r", program_id="p",
                             duration_s=1.0, travel_after_s=5.0)])
    with pytest.raises(ValueError, match="arrival_interval_s must be positive"):
        SchedulerConfig(arrival_interval_s=0.0)
    with pytest.raises(ValueError, match="n_units must be a positive integer"):
        SchedulerConfig(n_units=0)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        SchedulerConfig(seed=-3)


def test_schedule_beyond_calendar_span_fails_fast():
    config = SynthConfig(
        factory=FactoryCalendarConfig(span_days=90.0),
        scheduler=SchedulerConfig(
            n_units=1,
            arrival_interval_s=60.0,
            routes=[RouteConfig(
                route_id="route-A", product_type="sedan", stages=[
                    RouteStageConfig(robot_id="robot-01",
                                     program_id="program-01",
                                     duration_s=100.0 * 86400.0)])],
        ),
    )
    with pytest.raises(ValueError, match="beyond the calendar span"):
        FactoryScheduler(config).build()

def test_idle_gaps_cover_zero_and_nonzero():
    schedule = FactoryScheduler(_two_route_config()).build()
    idles = [event.idle_before for event in schedule.events]
    assert any(idle == 0.0 for idle in idles), (
        "consecutive operations must produce zero idle gaps"
    )
    assert any(idle > 0.0 for idle in idles), (
        "robot-waiting intervals must produce nonzero idle gaps"
    )


def test_single_robot_line_serializes_strictly():
    config = SynthConfig(scheduler=SchedulerConfig(
        n_units=5,
        arrival_interval_s=60.0,
        seed=3,
        routes=[RouteConfig(
            route_id="route-S", product_type="sedan", stages=[
                RouteStageConfig(robot_id="robot-01", program_id="program-01",
                                 duration_s=120.0, travel_after_s=10.0),
                RouteStageConfig(robot_id="robot-01", program_id="program-02",
                                 duration_s=120.0, travel_after_s=0.0),
            ])],
    ))
    schedule = FactoryScheduler(config).build()
    schedule.validate()
    assert {event.robot_id for event in schedule.events} == {"robot-01"}
    ordered = sorted(schedule.events,
                     key=lambda event: (event.start_time, event.end_time))
    for previous, current in zip(ordered, ordered[1:]):
        assert previous.end_time <= current.start_time
    queued = [e for e in schedule.events if e.queue_delay > 0.0]
    assert queued, "overlapping arrivals on one robot must queue"
    for event in queued:
        assert event.idle_before == 0.0


def test_global_sortability_and_stable_local_ordering():
    config = _two_route_config()
    first = FactoryScheduler(config).build()
    second = FactoryScheduler(config).build()
    assert [e.operation_id for e in first.events] == [
        e.operation_id for e in second.events]
    key = lambda e: (e.start_time, e.end_time, e.operation_id)  # noqa: E731
    globally_sorted = sorted(first.events, key=key)
    assert [e.start_time for e in globally_sorted] == sorted(
        e.start_time for e in globally_sorted)
    assert globally_sorted == sorted(second.events, key=key)
    assert len({e.operation_id for e in globally_sorted}) == len(first.events)
    for robot_id, events in _by_robot(first).items():
        starts = [e.start_time for e in sorted(
            events, key=lambda e: (e.start_time, e.end_time))]
        assert starts == sorted(starts), (
            f"robot {robot_id} stream must be locally ordered")
    for unit_id, events in _by_unit(first).items():
        assert [e.route_position for e in sorted(
            events, key=lambda e: e.route_position)] == list(range(len(events)))


def test_factory_seed_does_not_reorder_calendar():
    base = _two_route_config()
    other = SynthConfig(
        factory=FactoryCalendarConfig(seed=999),
        scheduler=base.scheduler,
    )
    assert [e for e in FactoryScheduler(base).build().events] == [
        e for e in FactoryScheduler(other).build().events]


def test_accepted_schedule_respects_calendar_bounds():
    schedule = FactoryScheduler(_two_route_config()).build()
    assert 0.0 < schedule.span_s <= 183.0 * 86400.0
    for event in schedule.events:
        assert 0.0 <= event.arrival_time <= event.start_time <= event.end_time
        assert event.end_time <= schedule.span_s
