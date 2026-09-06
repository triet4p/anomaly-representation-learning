"""Scheduled operation signal integration (Sprint 11 Task 4).

Generate one :class:`~synth.schema.FileSample` per scheduled
:class:`~synth.schema.OperationEvent`, composing the methodology §20.6
signal from independently represented causes::

    x_{u,r,p,t} = F(B_{r,p}, U_u, H_r(t), psi_{r,p}, C_t, eta)

where ``B`` is healthy robot-program behavior (causal physics baseline
with robot calibration bias and program speed scale), ``U`` is stable
per-unit variation, ``H``/``psi`` enter only through the manifested
``G = gamma * H`` carried by the Task 3
:class:`~synth.schema.RobotHealthState`, ``C`` is deterministic
operating context derived from the event clock, and ``eta`` is
sensor/process noise from an independent stream.

Channel separation (why unit effects cannot masquerade as degradation):

- ``S0`` (feed) carries the per-unit offset, the load-scaled healthy
  envelope, and a constant night-shift bias. Unit offsets are stable
  per ``unit_id`` across every stage of that unit's route.
- ``S1`` (current) and ``S2`` (temperature) carry the manifested health
  drift ``G`` additively. Unit variation never touches these channels.
- ``S2`` additionally carries a diurnal ambient offset; unlike health
  drift it oscillates with time of day and is identical across robots,
  so the two remain separable by their temporal signatures.
- Six-channel observables (voltage, power, pressure) receive the
  current-consistent linear counterparts of the ``S1`` health shift.

Pipeline order preserves the existing normal-signal boundary exactly:
causal physics, then sensor noise, then regime-boundary smoothing, and
only afterwards the additive scheduled components (unit, health,
context). Determinism and stream independence (§20.14): per-event
structure and noise ``numpy.random.Generator`` instances are siblings
derived from ``SignalConfig.seed`` plus the operation identity only —
never from the full config hash and never shared with the scheduler
(``SchedulerConfig.seed``) or health (``HealthConfig.seed``) streams.
Changing noise parameters therefore never changes regime/structure
draws, and resampling noise cannot reorder the calendar or move failure
episodes. The schedule is read-only input throughout.

Scope: healthy-plus-context signals only. Anomaly injection, observable
labels, future-failure targets, splits, and materialization are later
tasks (Tasks 5+) and MUST NOT live here. ``episode``,
``anomaly_labels``, ``future_targets``, and ``split_provenance`` stay
``None``; every sample is ``NORMAL`` with complete operation, context,
health, and factory provenance.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from synth.config import (
    GENERATOR_VERSION,
    SynthConfig,
    channel_offset_scales,
)
from synth.health import FactoryHealth, RobotHealthProcess
from synth.physics.causal import CausalSignalGenerator
from synth.physics.noise import SensorNoiseGenerator
from synth.regimes import (
    build_feed_envelope,
    sample_regime_sequence,
    smooth_regime_boundaries,
)
from synth.scheduler import FactorySchedule, FactoryScheduler
from synth.schema import (
    FactoryProvenance,
    FileSample,
    OperatingContext,
    OperationEvent,
    RobotHealthState,
    SampleLabel,
)

#: Magnitude (mm/min of feed speed) of the stable per-unit S0 offset.
#: Drawn uniformly from ``[-MAG, +MAG]`` by a hash-derived generator, so
#: one physical unit carries the same offset on every route stage.
UNIT_FEED_OFFSET_MAGNITUDE = 6.0

#: Additive current (S1) shift per unit of manifested health ``G``.
HEALTH_CURRENT_GAIN = 0.6

#: Additive temperature (S2) shift per unit of manifested health ``G``.
HEALTH_TEMP_GAIN = 1.2

#: Additive temperature (S2) response per degree of ambient deviation
#: from the 25 C nominal conditioning point.
CTX_AMBIENT_GAIN = 0.4

#: Constant feed-speed (S0) bias applied to night-shift operations.
#: Shared by every night operation, so it cannot mimic a per-unit
#: offset; it never touches the health channels.
SHIFT_NIGHT_S0_BIAS = 1.5

_SECONDS_PER_DAY = 86400.0


def _stable_int(*parts: str) -> int:
    """Derive a stable 32-bit seed from identity parts (no RNG cost)."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def unit_feed_offset(unit_id: str, stream_key: str) -> float:
    """Return the stable per-unit S0 offset for one physical unit.

    Identical for every event carrying ``unit_id`` under the same signal
    stream key (``str(SignalConfig.seed)``); distinct units draw
    independently. Uses a dedicated hash-derived generator, never a
    shared stream, so noise or physics parameter changes cannot move it.
    """
    rng = np.random.default_rng(_stable_int("scheduled-unit", unit_id, stream_key))
    return float(rng.uniform(-UNIT_FEED_OFFSET_MAGNITUDE, UNIT_FEED_OFFSET_MAGNITUDE))


def robot_baseline(
    robot_id: str, program_id: str, cfg: SynthConfig
) -> tuple[np.ndarray, float, float]:
    """Return stable healthy robot-program conditioning ``(gain, temp, speed)``.

    Per-robot gain bias and temperature offset plus the program speed
    scale. Draws come from hash-derived generators keyed by identity
    plus the signal stream key: no stream is consumed, output is stable
    for a fixed signal seed, and noise parameter changes cannot move it.
    Physical magnitudes still follow ``fleet``/``physics`` parameters.
    """
    stream_key = str(cfg.signal.seed)
    fleet = cfg.fleet
    channels = cfg.n_channels
    robot_rng = np.random.default_rng(
        _stable_int("scheduled-robot", robot_id, stream_key)
    )
    gain_bias = robot_rng.normal(1.0, fleet.robot_gain_std, channels)
    low, high = fleet.robot_temp_offset_range
    temp_offset = float(robot_rng.uniform(low, high))
    scales = fleet.program_speed_scales
    speed_index = _stable_int("scheduled-program", program_id, stream_key) % len(scales)
    return gain_bias, temp_offset, float(scales[speed_index])


def operating_context_for(event: OperationEvent) -> OperatingContext:
    """Derive deterministic encoder-visible operating context from an event.

    Shift follows the roster clock (day for 06:00–18:00, night
    otherwise); load rises with congestion
    ``1 + 0.5 * q / (q + d + 60)`` from queue delay ``q`` and duration
    ``d``; ambient follows a diurnal 16–28 C wave. No RNG is consumed.
    """
    day_time = event.start_time % _SECONDS_PER_DAY
    hour = day_time / 3600.0
    shift = "day" if 6.0 <= hour < 18.0 else "night"
    queue, duration = event.queue_delay, event.duration
    load = 1.0 + 0.5 * queue / (queue + duration + 60.0)
    ambient = 22.0 + 6.0 * math.sin(
        2.0 * math.pi * day_time / _SECONDS_PER_DAY - math.pi / 2.0
    )
    return OperatingContext(shift=shift, load=load, ambient_temp_c=ambient)


def _scheduled_file_id(operation_id: str, config_hash: str) -> str:
    tag = hashlib.sha256(
        f"scheduled:{operation_id}:{config_hash}".encode()
    ).hexdigest()[:10]
    return f"S-{operation_id}-{tag}"


class ScheduledSignalGenerator:
    """Compose scheduled FileSamples from Task 2/3 events and health states.

    Usage::

        schedule = FactoryScheduler(config).build()
        health = RobotHealthProcess(config).run(schedule)
        samples = ScheduledSignalGenerator(config).generate(schedule, health)

    or ``ScheduledSignalGenerator(config).build()`` for the full
    schedule → health → signals chain. Output aligns 1:1 with the
    schedule order; the schedule is read-only input and is never
    reordered here.
    """

    def __init__(self, config: SynthConfig | None = None) -> None:
        self.cfg = config or SynthConfig()
        self._config_hash = self.cfg.hash()

    @property
    def config_hash(self) -> str:
        """Config hash stamped on every generated sample."""
        return self._config_hash

    def generate_one(
        self,
        event: OperationEvent,
        state: RobotHealthState,
        *,
        robot_idx: int,
        program_idx: int,
    ) -> FileSample:
        """Compose one FileSample from one (event, health-state) pair."""
        cfg = self.cfg
        pcfg, ncfg, rcfg = cfg.physics, cfg.noise, cfg.regime
        channels = cfg.n_channels
        config_hash = self._config_hash
        stream_key = str(cfg.signal.seed)

        structure_rng = np.random.default_rng(_stable_int(
            "scheduled-structure", stream_key, event.operation_id,
        ))
        noise_rng = np.random.default_rng(_stable_int(
            "scheduled-noise", stream_key, event.operation_id,
        ))

        context = operating_context_for(event)

        # ── Healthy robot-program behavior B_{r,p} ───────────────────
        gain_bias, robot_temp_offset, speed_scale = robot_baseline(
            event.robot_id, event.program_id, cfg
        )
        gain = (
            1.0 + structure_rng.uniform(-pcfg.channel_gain_variation,
                                        pcfg.channel_gain_variation, channels)
        ) * gain_bias
        off_frac = structure_rng.uniform(-pcfg.channel_offset_variation,
                                         pcfg.channel_offset_variation, channels)
        off_abs = off_frac * channel_offset_scales(channels, pcfg)

        seq_pairs = sample_regime_sequence(structure_rng, rcfg)
        feed_envelope, regime_meta = build_feed_envelope(
            seq_pairs, rcfg, pcfg, structure_rng)
        feed_envelope = feed_envelope * speed_scale * context.load

        causal_rng = np.random.default_rng(int(structure_rng.integers(0, 2**31)))
        causal = CausalSignalGenerator(pcfg, causal_rng,
                                       channel_gain=gain, channel_off=off_abs)
        x = causal.generate(feed_envelope, robot_temp_offset=robot_temp_offset)

        # ── Sensor/process noise η (existing boundary, unchanged) ────
        x = SensorNoiseGenerator(ncfg, noise_rng).add_all(x)
        x = smooth_regime_boundaries(x, regime_meta)

        # ── Scheduled components: unit, health, context ──────────────
        manifested = state.manifested_value
        x[0] += unit_feed_offset(event.unit_id, stream_key)
        if context.shift == "night":
            x[0] += SHIFT_NIGHT_S0_BIAS
        current_shift = HEALTH_CURRENT_GAIN * manifested
        x[1] += current_shift
        x[2] += HEALTH_TEMP_GAIN * manifested
        x[2] += CTX_AMBIENT_GAIN * (context.ambient_temp_c - 25.0)
        if channels == 6:
            # Current-consistent linear counterparts of the S1 shift,
            # mirroring the causal voltage/power/pressure coefficients.
            x[3] += 0.09 * current_shift
            x[4] += 1.25 * current_shift
            x[5] += 0.012 * current_shift

        x = x.astype(np.float32)
        seed = _stable_int("scheduled-seed", stream_key,
                           event.operation_id, config_hash) % (2**31)
        sample = FileSample(
            x=x,
            file_id=_scheduled_file_id(event.operation_id, config_hash),
            file_label=SampleLabel.NORMAL,
            seed=seed,
            generator_version=GENERATOR_VERSION,
            config_hash=config_hash,
            regime_sequence=regime_meta,
            robot_idx=robot_idx,
            program_idx=program_idx,
            robot_code=event.robot_id,
            program_number=event.program_id,
            operation=event,
            operating_context=context,
            health=state,
            factory_provenance=FactoryProvenance(
                seed=seed,
                stream="signal",
                generator_version=GENERATOR_VERSION,
                config_hash=config_hash,
            ),
        )
        sample.validate()
        return sample

    def generate(
        self, schedule: FactorySchedule, health: FactoryHealth
    ) -> list[FileSample]:
        """Compose one FileSample per scheduled operation, in order.

        Validates the schedule and the 1:1 health alignment before
        generating; every emitted sample is validated. Robot/program
        indices follow sorted schedule-registry order so they are
        deterministic for a fixed schedule.
        """
        schedule.validate()
        health.validate(schedule, self.cfg.health)
        robots = sorted({event.robot_id for event in schedule.events})
        programs = sorted({event.program_id for event in schedule.events})
        robot_index = {robot_id: i for i, robot_id in enumerate(robots)}
        program_index = {program_id: i for i, program_id in enumerate(programs)}
        samples = [
            self.generate_one(
                event, state,
                robot_idx=robot_index[event.robot_id],
                program_idx=program_index[event.program_id],
            )
            for event, state in zip(schedule.events, health.states)
        ]
        assert [s.operation.operation_id for s in samples] == [  # type: ignore[union-attr]
            e.operation_id for e in schedule.events
        ], "scheduled signals must align 1:1 with schedule order"
        return samples

    def build(self) -> tuple[FactorySchedule, FactoryHealth, list[FileSample]]:
        """Run scheduler → health → signals from this generator's config."""
        schedule = FactoryScheduler(self.cfg).build()
        health = RobotHealthProcess(self.cfg).run(schedule)
        return schedule, health, self.generate(schedule, health)
