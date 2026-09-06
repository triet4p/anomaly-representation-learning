"""Deterministic shared-unit factory scheduler (Sprint 11 Task 2).

Causal construction of operation events on a 3–6 month calendar
(methodology §20.2–20.4): physical units arrive at fixed intervals, take a
deterministically assigned production route, and queue for single-server
robots that serve one operation at a time.

Scheduling equations (methodology §20.3) for unit ``u`` at route stage ``k``
on robot ``r``::

    s = max(a, v_r)   e = s + d   v_r := e   a_next = e + tau

These guarantee no same-robot overlap, natural queues and idle gaps,
asynchronous cross-robot execution, and causal per-unit route order.
Every event is a Task 1 ``OperationEvent``; cross-event validation lives
in ``FactorySchedule.validate``. Robot health, signal synthesis, anomaly
injection, splits, and materialization are later tasks and MUST NOT live
here.

Determinism: one ``numpy.random.Generator`` seeded from
``SchedulerConfig.seed`` drives route assignment only, and is never shared
with signal or health streams, so resampling sensor noise cannot reorder
the factory calendar (methodology §20.14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from synth.config import SynthConfig
from synth.schema import OperationEvent

#: Seconds per calendar day; the scheduler works in float seconds.
SECONDS_PER_DAY = 86400.0

#: Absolute tolerance (seconds) for schedule-level chaining checks.
_SCHEDULE_ABS_TOL = 1e-6


@dataclass
class FactorySchedule:
    """One causal factory calendar: validated operation events in build order.

    ``span_s`` is the calendar length in seconds. ``validate`` enforces the
    Task 2 structural invariants: unique operation identities, no
    same-robot overlap, causal per-unit route order with stable unit
    identity, and timestamps inside ``[0, span_s]``.
    """
    events: list[OperationEvent] = field(default_factory=list)
    seed: int = 0
    span_s: float = 0.0

    def validate(self) -> None:
        """Assert schedule-level structural invariants."""
        assert self.events, "schedule must contain at least one operation"
        assert self.span_s > 0.0, "span_s must be positive"
        operation_ids = [event.operation_id for event in self.events]
        assert len(set(operation_ids)) == len(operation_ids), (
            "operation_id values must be unique"
        )
        by_robot: dict[str, list[OperationEvent]] = {}
        by_unit: dict[str, list[OperationEvent]] = {}
        for event in self.events:
            assert 0.0 <= event.arrival_time <= event.start_time, (
                f"{event.operation_id}: arrival_time out of [0, start_time]"
            )
            assert event.end_time <= self.span_s, (
                f"{event.operation_id}: end_time {event.end_time} exceeds "
                f"calendar span {self.span_s}"
            )
            by_robot.setdefault(event.robot_id, []).append(event)
            by_unit.setdefault(event.unit_id, []).append(event)
        for robot_id, robot_events in by_robot.items():
            ordered = sorted(robot_events,
                             key=lambda event: (event.start_time, event.end_time))
            for previous, current in zip(ordered, ordered[1:]):
                assert previous.end_time <= current.start_time, (
                    f"robot {robot_id}: overlapping operations "
                    f"{previous.operation_id} and {current.operation_id}"
                )
        for unit_id, unit_events in by_unit.items():
            ordered = sorted(unit_events,
                             key=lambda event: event.route_position)
            positions = [event.route_position for event in ordered]
            assert positions == list(range(len(ordered))), (
                f"unit {unit_id}: route positions must be 0..k-1, got {positions}"
            )
            route_ids = {event.route_id for event in ordered}
            assert len(route_ids) == 1, (
                f"unit {unit_id}: route identity must be stable, got {route_ids}"
            )
            products = {event.product_type for event in ordered}
            assert len(products) == 1, (
                f"unit {unit_id}: product identity must be stable, got {products}"
            )
            for previous, current in zip(ordered, ordered[1:]):
                expected_arrival = previous.end_time + previous.travel_time
                assert math.isclose(current.arrival_time, expected_arrival,
                                    rel_tol=1e-9, abs_tol=_SCHEDULE_ABS_TOL), (
                    f"unit {unit_id}: stage {current.route_position} arrival "
                    f"{current.arrival_time} != previous end + travel "
                    f"{expected_arrival}"
                )


class FactoryScheduler:
    """Builds causal factory schedules deterministically from ``SynthConfig``.

    Usage::

        scheduler = FactoryScheduler(SynthConfig())
        schedule = scheduler.build()
        schedule.validate()

    The same config always produces identical events. Changing the
    scheduler seed changes route assignment only.
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()
        self._span_s = self.cfg.factory.span_days * SECONDS_PER_DAY

    @property
    def span_s(self) -> float:
        """Calendar length in seconds."""
        return self._span_s

    def build(self) -> FactorySchedule:
        """Build the full operation schedule in causal unit-arrival order."""
        sched = self.cfg.scheduler
        calendar = self.cfg.factory
        rng = np.random.default_rng(sched.seed)
        availability: dict[str, float] = {}
        events: list[OperationEvent] = []
        for unit_index in range(sched.n_units):
            unit_id = f"unit-{unit_index:04d}"
            route = sched.routes[int(rng.integers(len(sched.routes)))]
            factory_arrival = unit_index * sched.arrival_interval_s
            previous_end = 0.0
            previous_travel = 0.0
            for position, stage in enumerate(route.stages):
                arrival = (factory_arrival if position == 0
                           else previous_end + previous_travel)
                robot_available = availability.get(stage.robot_id, 0.0)
                start = max(arrival, robot_available)
                end = start + stage.duration_s
                if end > self._span_s:
                    raise ValueError(
                        f"unit {unit_id} stage {position} on {stage.robot_id} "
                        f"ends at {end:.1f}s, beyond the calendar span of "
                        f"{self._span_s:.1f}s (span_days={calendar.span_days}); "
                        f"reduce n_units, arrival_interval_s, or stage "
                        f"durations, or extend span_days"
                    )
                events.append(OperationEvent(
                    operation_id=f"op-{len(events):06d}",
                    unit_id=unit_id,
                    product_type=route.product_type,
                    route_id=route.route_id,
                    route_position=position,
                    robot_id=stage.robot_id,
                    program_id=stage.program_id,
                    arrival_time=arrival,
                    start_time=start,
                    end_time=end,
                    duration=stage.duration_s,
                    queue_delay=start - arrival,
                    travel_time=stage.travel_after_s,
                    idle_before=start - robot_available,
                ))
                availability[stage.robot_id] = end
                previous_end = end
                previous_travel = stage.travel_after_s
        schedule = FactorySchedule(
            events=events, seed=sched.seed, span_s=self._span_s)
        schedule.validate()
        return schedule
