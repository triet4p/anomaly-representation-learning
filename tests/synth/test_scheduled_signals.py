"""Focused Task 4 tests: scheduled operation signal integration.

Covers public scheduled generation aligned 1:1 with accepted
scheduler/health contracts, deterministic replay, stable unit variation
across route events, robot-health/program-sensitivity effects, operating
context, sensor-noise independence, variable lengths with C=6,
provenance with encoder separation, dataset round trip, unchanged patch
mapping, and the preserved causal channel chain. Temporal anomalies,
labels, splits, and materialization are later tasks and are NOT tested
here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from synth.config import (
    HealthConfig,
    NoiseConfig,
    RouteConfig,
    RouteStageConfig,
    SchedulerConfig,
    SignalConfig,
    SynthConfig,
)
from synth.dataset import _sample_bytes, load_sample_bytes
from synth.health import FactoryHealth, RobotHealthProcess, program_sensitivity
from synth.patchify import Patchifier
from synth.scheduler import FactorySchedule, FactoryScheduler
from synth.schema import (
    DegradationStage,
    RobotHealthState,
    SampleLabel,
)
from synth.scheduled import (
    CTX_AMBIENT_GAIN,
    HEALTH_CURRENT_GAIN,
    HEALTH_TEMP_GAIN,
    SHIFT_NIGHT_S0_BIAS,
    ScheduledSignalGenerator,
    operating_context_for,
    unit_feed_offset,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _route_single_robot() -> SchedulerConfig:
    """One unit route visiting the same robot under two programs."""
    return SchedulerConfig(
        n_units=6,
        arrival_interval_s=60.0,
        seed=11,
        routes=[RouteConfig(
            route_id="route-S", product_type="sedan", stages=[
                RouteStageConfig(robot_id="robot-01", program_id="program-01",
                                 duration_s=120.0, travel_after_s=10.0),
                RouteStageConfig(robot_id="robot-01", program_id="program-02",
                                 duration_s=120.0, travel_after_s=0.0),
            ])],
    )


def _frozen_health(**overrides) -> HealthConfig:
    """Health frozen at zero: no aging, wear, noise, or failures."""
    fields: dict[str, object] = {
        "seed": 5, "aging_rate": 0.0, "wear_rate": 0.0, "noise_scale": 0.0,
        "base_rate": 0.0, "abrupt_rate": 0.0,
    }
    fields.update(overrides)
    return HealthConfig(**fields)  # type: ignore[arg-type]


def _chain(cfg: SynthConfig) -> tuple[FactorySchedule, FactoryHealth, list]:
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    samples = ScheduledSignalGenerator(cfg).generate(schedule, health)
    return schedule, health, samples


def _small_cfg(**overrides) -> SynthConfig:
    fields: dict[str, object] = {
        "scheduler": _route_single_robot(), "health": _frozen_health(),
    }
    fields.update(overrides)
    return SynthConfig(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public scheduled generation and 1:1 alignment
# ---------------------------------------------------------------------------

def test_public_generation_aligns_one_to_one():
    """Samples follow schedule order with matching operation/health links."""
    cfg = _small_cfg()
    schedule, health, samples = _chain(cfg)
    schedule.validate()
    health.validate(schedule, cfg.health)
    assert len(samples) == len(schedule.events) == len(health.states)
    for sample, event, state in zip(samples, schedule.events, health.states):
        sample.validate()
        assert sample.file_label is SampleLabel.NORMAL
        assert sample.anomaly_meta is None and sample.anomaly_mask is None
        assert sample.operation is event
        assert sample.health is state
        assert sample.C == 6
        assert sample.robot_code == event.robot_id
        assert sample.program_number == event.program_id


def test_deterministic_replay():
    """Same config replays byte-identical signals and identities."""
    cfg = _small_cfg()
    _, _, first = _chain(cfg)
    _, _, second = _chain(cfg)
    assert [s.file_id for s in first] == [s.file_id for s in second]
    for a, b in zip(first, second):
        assert np.array_equal(a.x, b.x)


def test_signal_seed_sensitivity():
    """A different signal seed changes signals but keeps the calendar."""
    cfg = _small_cfg()
    schedule, health, samples = _chain(cfg)
    cfg2 = dataclasses.replace(cfg, signal=SignalConfig(seed=99))
    schedule2 = FactoryScheduler(cfg2).build()
    health2 = RobotHealthProcess(cfg2).run(schedule2)
    assert [e.operation_id for e in schedule2.events] == [
        e.operation_id for e in schedule.events]
    assert [s.health_value for s in health2.states] == [
        s.health_value for s in health.states]
    resampled = ScheduledSignalGenerator(cfg2).generate(schedule2, health2)
    assert any(not np.array_equal(a.x, b.x) for a, b in zip(samples, resampled))


# ---------------------------------------------------------------------------
# Unit variation: stable, confined, repeatable across route events
# ---------------------------------------------------------------------------

def test_unit_offset_stable_and_distinct():
    """One unit keeps its offset; different units draw independently."""
    cfg = _small_cfg()
    stream_key = str(cfg.signal.seed)
    assert unit_feed_offset("unit-0000", stream_key) == unit_feed_offset(
        "unit-0000", stream_key)
    assert unit_feed_offset("unit-0000", stream_key) != unit_feed_offset(
        "unit-0001", stream_key)


def test_unit_effect_confined_to_feed_channel():
    """Swapping only the unit shifts S0 by a constant; nothing else moves."""
    cfg = _small_cfg()
    schedule, health, _ = _chain(cfg)
    gen = ScheduledSignalGenerator(cfg)
    event, state = schedule.events[0], health.states[0]
    other = dataclasses.replace(event, unit_id="unit-9999")
    base = gen.generate_one(event, state, robot_idx=0, program_idx=0)
    swapped = gen.generate_one(other, state, robot_idx=0, program_idx=0)
    expected = (unit_feed_offset("unit-9999", str(cfg.signal.seed))
                - unit_feed_offset(event.unit_id, str(cfg.signal.seed)))
    for channel in range(1, base.C):
        assert np.array_equal(base.x[channel], swapped.x[channel])
    diff = (swapped.x[0].astype(np.float64) - base.x[0].astype(np.float64))
    assert np.allclose(diff, expected, rtol=1e-5, atol=1e-4)


def test_unit_repeatability_across_route_events():
    """One physical unit traces its route with one stable offset identity."""
    cfg = _small_cfg()
    schedule, _, samples = _chain(cfg)
    by_unit: dict[str, list[int]] = {}
    for index, event in enumerate(schedule.events):
        by_unit.setdefault(event.unit_id, []).append(index)
    assert by_unit["unit-0000"] == [0, 1]
    assert len(by_unit["unit-0000"]) == 2
    stream_key = str(cfg.signal.seed)
    offsets = [unit_feed_offset(schedule.events[i].unit_id, stream_key)
               for i in by_unit["unit-0000"]]
    assert offsets[0] == offsets[1]
    for index in by_unit["unit-0000"]:
        assert samples[index].operation.unit_id == "unit-0000"
    first_stage, second_stage = (schedule.events[i] for i in by_unit["unit-0000"])
    assert first_stage.route_id == second_stage.route_id
    assert (first_stage.route_position, second_stage.route_position) == (0, 1)


# ---------------------------------------------------------------------------
# Robot health and program sensitivity
# ---------------------------------------------------------------------------

def _state_with(symbol_state, *, health_value: float, gamma: float) -> RobotHealthState:
    severity = min(1.0, health_value / 2.0)
    return RobotHealthState(
        health_value=health_value,
        program_sensitivity=gamma,
        manifested_value=gamma * health_value,
        degradation_stage=symbol_state,
        degradation_severity=severity,
    )


def test_robot_health_shifts_degradation_channels():
    """Manifested health adds the documented gain on S1/S2 and nothing on S0."""
    cfg = _small_cfg()
    schedule, health, _ = _chain(cfg)
    gen = ScheduledSignalGenerator(cfg)
    event, state = schedule.events[0], health.states[0]
    gamma = state.program_sensitivity
    healthy = _state_with(DegradationStage.HEALTHY, health_value=0.0, gamma=gamma)
    degraded = _state_with(DegradationStage.PERSISTENT, health_value=2.0, gamma=gamma)
    before = gen.generate_one(event, healthy, robot_idx=0, program_idx=0)
    after = gen.generate_one(event, degraded, robot_idx=0, program_idx=0)
    delta_g = 2.0 * gamma
    assert np.allclose(after.x[0], before.x[0], rtol=1e-6, atol=1e-6)
    assert np.allclose(after.x[1].mean() - before.x[1].mean(),
                       HEALTH_CURRENT_GAIN * delta_g, rtol=1e-4, atol=1e-4)
    assert np.allclose(after.x[2].mean() - before.x[2].mean(),
                       HEALTH_TEMP_GAIN * delta_g, rtol=1e-4, atol=1e-4)


def test_program_sensitivity_tied_to_one_robot_health():
    """Same latent H under two programs manifests in proportion to γ."""
    cfg = _small_cfg()
    schedule, health, _ = _chain(cfg)
    gen = ScheduledSignalGenerator(cfg)
    event = schedule.events[0]
    gamma_a = program_sensitivity(event.robot_id, "program-01", cfg.health)
    gamma_b = program_sensitivity(event.robot_id, "program-02", cfg.health)
    assert gamma_a != gamma_b
    level = 1.5
    zero = _state_with(DegradationStage.HEALTHY, health_value=0.0, gamma=gamma_a)
    state_a = _state_with(DegradationStage.LATENT_DRIFT, health_value=level, gamma=gamma_a)
    state_b = _state_with(DegradationStage.LATENT_DRIFT, health_value=level, gamma=gamma_b)
    ref = gen.generate_one(event, zero, robot_idx=0, program_idx=0)
    sample_a = gen.generate_one(event, state_a, robot_idx=0, program_idx=0)
    sample_b = gen.generate_one(event, state_b, robot_idx=0, program_idx=1)
    shift_a = float(sample_a.x[1].mean() - ref.x[1].mean())
    shift_b = float(sample_b.x[1].mean() - ref.x[1].mean())
    assert shift_a == pytest.approx(HEALTH_CURRENT_GAIN * gamma_a * level, rel=1e-4)
    assert shift_b / shift_a == pytest.approx(gamma_b / gamma_a, rel=1e-4)


def test_health_drift_visible_along_robot_trajectory():
    """Growing H moves later S1 means on the same robot under one seed."""
    cfg = _small_cfg(health=_frozen_health(wear_rate=0.02, degradation_onset=0.0))
    schedule, health, samples = _chain(cfg)
    assert health.states[-1].health_value > health.states[0].health_value + 1.0
    assert samples[-1].x[1].mean() > samples[0].x[1].mean()


# ---------------------------------------------------------------------------
# Operating context
# ---------------------------------------------------------------------------

def _wide_cfg() -> SynthConfig:
    """Schedule spanning day and night shifts for context coverage."""
    return _small_cfg(scheduler=SchedulerConfig(
        n_units=8,
        arrival_interval_s=7200.0,
        seed=11,
        routes=_route_single_robot().routes,
    ))


def test_operating_context_derivation_and_effect():
    """Context follows the event clock and enters the signal deterministically."""
    cfg = _small_cfg()
    schedule, health, _ = _chain(cfg)
    gen = ScheduledSignalGenerator(cfg)
    midnight = operating_context_for(schedule.events[0])
    assert schedule.events[0].start_time == 0.0
    assert midnight.shift == "night"
    assert midnight.ambient_temp_c == pytest.approx(16.0)
    wide_schedule = FactoryScheduler(_wide_cfg()).build()
    wide_contexts = [operating_context_for(e) for e in wide_schedule.events]
    assert {c.shift for c in wide_contexts} == {"day", "night"}
    assert all(16.0 <= c.ambient_temp_c <= 28.0 for c in wide_contexts)
    assert all(c.load >= 1.0 for c in wide_contexts)
    assert all(c.load >= 1.0 for c in
               (operating_context_for(e) for e in schedule.events))
    event, state = schedule.events[0], health.states[0]
    noon = dataclasses.replace(
        event,
        arrival_time=event.arrival_time + 43200.0,
        start_time=event.start_time + 43200.0,
        end_time=event.end_time + 43200.0,
    )
    assert operating_context_for(noon).shift == "day"
    night_sample = gen.generate_one(event, state, robot_idx=0, program_idx=0)
    noon_sample = gen.generate_one(noon, state, robot_idx=0, program_idx=0)
    ambient_delta = (operating_context_for(noon).ambient_temp_c
                     - operating_context_for(event).ambient_temp_c)
    assert np.allclose(noon_sample.x[2].mean() - night_sample.x[2].mean(),
                       CTX_AMBIENT_GAIN * ambient_delta, rtol=1e-4, atol=1e-4)
    assert np.allclose(noon_sample.x[0].mean() - night_sample.x[0].mean(),
                       -SHIFT_NIGHT_S0_BIAS, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Noise independence and stream separation
# ---------------------------------------------------------------------------

def test_noise_changes_never_reorder_schedule_or_health():
    """Resampling sensor noise leaves calendar and health episodes fixed."""
    cfg = _small_cfg()
    schedule = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(schedule)
    noisy = dataclasses.replace(
        cfg, noise=dataclasses.replace(cfg.noise, s1_arc_noise_std=0.5))
    schedule2 = FactoryScheduler(noisy).build()
    health2 = RobotHealthProcess(noisy).run(schedule2)
    assert [(e.operation_id, e.start_time, e.end_time) for e in schedule2.events] == [
        (e.operation_id, e.start_time, e.end_time) for e in schedule.events]
    assert [s.health_value for s in health2.states] == [
        s.health_value for s in health.states]
    assert [e.episode_id for e in health2.episodes] == [
        e.episode_id for e in health.episodes]


def test_noise_params_keep_structure_but_change_signal():
    """Noise resampling preserves regime structure while changing samples."""
    cfg = _small_cfg()
    _, _, samples = _chain(cfg)
    noisy = dataclasses.replace(
        cfg, noise=dataclasses.replace(
            cfg.noise, s1_arc_noise_std=0.9, s0_vibration_amplitude=0.2,
            s0_gaussian_std=0.2, s2_rw_step_std=1.0, s2_glitch_prob=0.01))
    _, _, resampled = _chain(noisy)
    assert [(r.start, r.end) for s in samples for r in s.regime_sequence] == [
        (r.start, r.end) for s in resampled for r in s.regime_sequence]
    assert any(not np.array_equal(a.x, b.x) for a, b in zip(samples, resampled))


# ---------------------------------------------------------------------------
# Lengths, layouts, provenance, persistence, patches, causality
# ---------------------------------------------------------------------------

def test_variable_lengths_with_c6():
    """Scheduled files keep variable lengths on the production layout."""
    cfg = _small_cfg()
    _, _, samples = _chain(cfg)
    assert all(s.C == 6 for s in samples)
    assert len({s.T for s in samples}) > 1
    assert all(cfg.regime.min_total_steps <= s.T <= cfg.regime.max_total_steps
               for s in samples)


def test_three_channel_layout_supported():
    """The legacy three-channel layout stays schema-valid."""
    cfg = _small_cfg(n_channels=3)
    _, _, samples = _chain(cfg)
    assert samples
    for sample in samples:
        sample.validate()
        assert sample.C == 3


def test_provenance_and_encoder_separation():
    """Complete provenance travels with the file; health stays diagnostic."""
    cfg = _small_cfg()
    schedule, health, samples = _chain(cfg)
    sample = samples[0]
    assert sample.operation == schedule.events[0]
    assert sample.operating_context is not None
    assert sample.health == health.states[0]
    assert sample.episode is None
    assert sample.anomaly_labels is None
    assert sample.future_targets is None
    assert sample.split_provenance is None
    provenance = sample.factory_provenance
    assert provenance is not None
    assert provenance.seed == sample.seed
    assert provenance.stream == "signal"
    assert provenance.generator_version == sample.generator_version
    assert provenance.config_hash == sample.config_hash == cfg.hash()
    inputs = sample.encoder_inputs()
    assert inputs["operation"] is sample.operation
    assert inputs["operating_context"] is sample.operating_context
    assert inputs["factory_provenance"] is provenance
    assert "health" not in inputs


def test_scheduled_sample_dataset_round_trip():
    """Scheduled provenance survives the Task 1 persistence boundary."""
    cfg = _small_cfg()
    _, _, samples = _chain(cfg)
    payload = _sample_bytes(samples[0])
    assert payload == _sample_bytes(samples[0])
    restored = load_sample_bytes(payload)
    restored.validate()
    assert restored.operation == samples[0].operation
    assert restored.health == samples[0].health
    assert restored.operating_context == samples[0].operating_context
    assert restored.factory_provenance == samples[0].factory_provenance
    assert np.array_equal(restored.x, samples[0].x)


def test_patch_mapping_unchanged():
    """Scheduled files patchify through the untouched timestep contract."""
    cfg = _small_cfg()
    _, _, samples = _chain(cfg)
    patchifier = Patchifier(cfg.patch)
    batch = patchifier.patchify(samples[0])
    assert batch.starts[0] == 0
    assert batch.pad_mask.shape == (batch.N, cfg.patch.patch_size)
    assert not batch.pad_mask[:-1].any()
    for timestep in (0, samples[0].T // 2, samples[0].T - 1):
        assert len(Patchifier.which_patches(
            timestep, batch.starts, batch.valid_len)) >= 1


def test_causal_channel_chain_preserved():
    """Feed→current→temperature correlation survives composition."""
    cfg = _small_cfg()
    _, _, samples = _chain(cfg)
    corrs01, corrs02 = [], []
    for sample in samples:
        x = sample.x.astype(np.float64)
        corrs01.append(float(np.corrcoef(x[0], x[1])[0, 1]))
        corrs02.append(float(np.corrcoef(x[0], x[2])[0, 1]))
    assert float(np.mean(corrs01)) > 0.5
    assert float(np.mean(corrs02)) > 0.1
