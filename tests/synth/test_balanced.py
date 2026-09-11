"""Focused tests for Sprint 15 balanced quota machinery (Batch B).

Covers allocator exactness/determinism/infeasibility, waveform-blindness
(import and row-key constraints), seed-namespace disjointness, quota
arithmetic, and seal protocol acceptance. No generation; synthetic metadata
only, except the tiny CLI smoke that pins the legacy path byte-contract.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from synth import balanced as B
from synth.balanced import InfeasibleCandidate, QuotaConfig

DAY = 86400.0


def _row(fid, robot, program, end_day, views=("test-temporal",),
         quarantined=False, censored=False, maint_overlap=False):
    # NOTE: maintenance overlap is emulated by placing the row inside a
    # maintenance window anchored at end_day; callers pass wins accordingly.
    _ = maint_overlap
    return {
        "file_id": fid,
        "operation_id": f"op-{fid}",
        "robot_id": robot,
        "program_id": program,
        "start_time": (end_day - 0.01) * DAY,
        "end_time": end_day * DAY,
        "file_label": "abnormal" if quarantined else "normal",
        "is_quarantined": quarantined,
        "quarantine_reason": "precursor_horizon" if quarantined else None,
        "is_censored": censored,
        "member_views": list(views),
        "last_reset_time": 0.0,
        "n_valid_patches": 96,
    }


def _failure(fid, robot, day, cohort, subtype, onset_day=None):
    return {
        "failure_id": fid,
        "robot_id": robot,
        "failure_time": day * DAY,
        "cohort": cohort,
        "subtype": subtype,
        "degradation_onset": (onset_day * DAY if onset_day is not None
                              else None),
        "duration_d": ((day - onset_day) if onset_day is not None else 0.0),
        "severity": 2.0,
        "degradation_episode_id": f"deg-{fid}" if cohort != "A" else None,
        "maintenance_episode_id": f"maint-{fid}",
    }


def _ledger_pool(n_per_subtype=4, robots=("robot-01", "robot-02"),
                 start_day=30.0, step_days=20.0):
    """Build a pool with rich horizons: daily files per robot over span."""
    rows = []
    for robot in robots:
        day = 1.0
        while day < 600.0:
            rows.append(_row(f"{robot}-{day:.0f}", robot, "program-01", day))
            day += 1.0
    ledger = []
    day = start_day
    i = 0
    for cohort, subtypes in (("P", ("P1", "P2")), ("W", ("W1", "W2")),
                             ("A", ("A1", "A2"))):
        for subtype in subtypes:
            for _ in range(n_per_subtype):
                robot = robots[i % len(robots)]
                onset = day - 10.0 if cohort != "A" else None
                ledger.append(_failure(
                    f"fail-{cohort}-{subtype}-{i}", robot, day,
                    cohort, subtype, onset))
                i += 1
                day += step_days
    return rows, ledger, {}


def test_alloc_seed_formula():
    assert B.alloc_seed_for(1000) == 1000 * 31 + 7
    assert B.alloc_seed_for(1100) == 1100 * 31 + 7


def test_quota_mix_inside_gate():
    total = B.DEFAULT_QUOTA.p + B.DEFAULT_QUOTA.w + B.DEFAULT_QUOTA.a
    assert B.DEFAULT_QUOTA.p / total == pytest.approx(0.375)
    assert B.DEFAULT_QUOTA.w / total == pytest.approx(0.375)
    assert B.DEFAULT_QUOTA.a / total == pytest.approx(0.25)
    assert (B.DEFAULT_QUOTA.p1, B.DEFAULT_QUOTA.p2) == (12, 12)
    assert (B.DEFAULT_QUOTA.w1, B.DEFAULT_QUOTA.w2) == (12, 12)
    assert (B.DEFAULT_QUOTA.a1, B.DEFAULT_QUOTA.a2) == (8, 8)


def test_candidate1_seeds_disjoint_from_sprint14():
    seeds = sorted(s for s in B.S15_ROSTER if 1000 <= s <= 1017)
    assert len(seeds) == 18 and len(set(seeds)) == 18
    assert min(seeds) == 1000 and max(seeds) == 1017
    assert max(seeds) > 933  # above every ledgered/test Sprint 14 seed
    groups = Counter(group for s, (group, _) in B.S15_ROSTER.items()
                     if 1000 <= s <= 1017)
    assert groups == {"DESIGN": 4, "FIT": 3, "CALIBRATION": 1,
                      "CONFIRMATION": 4, "SEALED": 4, "PROOF": 2}
    assert B.S15_ROSTER[1016] == ("PROOF", "H-PROOF-1")
    assert B.S15_ROSTER[1017] == ("PROOF", "H-PROOF-2")


def test_projector_rejects_leakage_keys():
    rows, _, _ = _ledger_pool()
    bad = dict(rows[0])
    bad["cohort"] = "P"
    with pytest.raises(ValueError, match="leakage keys"):
        B.project_allocation_inputs([bad])
    missing = dict(rows[0])
    del missing["n_valid_patches"]
    with pytest.raises(ValueError, match="missing row keys"):
        B.project_allocation_inputs([missing])


def test_allocator_exact_splits_and_determinism():
    rows, ledger, wins = _ledger_pool(n_per_subtype=4)
    first = B.allocate_quotas(rows, ledger, wins, 1000,
                              QuotaConfig(p=4, p1=2, p2=2, w=4, w1=2, w2=2,
                                          a=4, a1=2, a2=2, controls=0,
                                          robot_days=0, robots_pos=1,
                                          robots_neg=0, programs=1,
                                          robot_pos_cap=1.0,
                                          program_cap=1.0))
    second = B.allocate_quotas(rows, ledger, wins, 1000,
                               QuotaConfig(p=4, p1=2, p2=2, w=4, w1=2, w2=2,
                                           a=4, a1=2, a2=2, controls=0,
                                           robot_days=0, robots_pos=1,
                                           robots_neg=0, programs=1,
                                           robot_pos_cap=1.0,
                                           program_cap=1.0))
    assert first["selected"] == second["selected"]
    for cohort, subs in (("P", (("P1", 2), ("P2", 2))),
                         ("W", (("W1", 2), ("W2", 2))),
                         ("A", (("A1", 2), ("A2", 2)))):
        for subtype, need in subs:
            assert len(first["selected"][cohort][subtype]) == need


def test_allocator_infeasible_on_short_pool():
    rows, ledger, wins = _ledger_pool(n_per_subtype=1)
    with pytest.raises(InfeasibleCandidate) as exc:
        B.allocate_quotas(rows, ledger, wins, 1000)
    assert exc.value.record["reason"] == "quota-shortfall"


def test_allocator_infeasible_on_control_shortfall():
    rows, ledger, wins = _ledger_pool(n_per_subtype=4)
    tiny = QuotaConfig(p=2, p1=1, p2=1, w=2, w1=1, w2=1, a=2, a1=1,
                       a2=1, controls=10**6, robot_days=0, robots_pos=1,
                       robots_neg=0, programs=1, robot_pos_cap=1.0)
    with pytest.raises(InfeasibleCandidate) as exc:
        B.allocate_quotas(rows, ledger, wins, 1000, tiny)
    assert exc.value.record["reason"] == "control-shortfall"


def test_allocator_enforces_spacing():
    rows = [_row(f"f-{d}", "robot-01", "program-01", float(d))
            for d in range(1, 140)]
    ledger = [_failure(f"fail-{i}", "robot-01", 20.0 + 15 * i, "A",
                       "A1" if i % 2 == 0 else "A2")
              for i in range(8)]
    out = B.allocate_quotas(
        rows, ledger, {}, 1000,
        QuotaConfig(p=0, p1=0, p2=0, w=0, w1=0, w2=0, a=4, a1=2, a2=2,
                    controls=0, robot_days=0, robots_pos=1, robots_neg=0,
                    programs=0, robot_pos_cap=1.0))
    moments = sorted(
        next(f["failure_time"] for f in ledger if f["failure_id"] == fid)
        for subs in out["selected"].values() for ids in subs.values()
        for fid in ids)
    for prev, cur in zip(moments, moments[1:]):
        if prev // 1 == cur // 1:
            continue
        assert cur - prev >= 14 * DAY - 1e-6


def test_balanced_module_waveform_blind_imports():
    tree = ast.parse(Path("src/synth/balanced.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    # Waveform-blindness contract: no signal, temporal-manifestation, dataset
    # persistence, or health/scheduler imports. `time` (stdlib solve clock)
    # and `scipy` (locked MILP backend, deferred import in the exact path)
    # are pure-computation additions that carry no waveform state.
    assert imported <= {"__future__", "collections", "dataclasses",
                        "hashlib", "json", "math", "numpy", "scipy",
                        "statistics", "synth", "time"}
    synth_mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("synth."):
            synth_mods.add(node.module)
    assert synth_mods <= {"synth.events", "synth.config"}, synth_mods


def test_analytic_signal_margin_passes():
    margin = B.analytic_signal_margin()
    assert margin["weakest_subtype"] == "W2"
    assert margin["margin"] >= B.SIGNAL_MARGIN_MIN
    assert margin["pass"] is True


def test_nuisance_envelopes_reject_drift():
    resolved = {"health": {"noise_scale": 1.0e-3},
                "scheduler": {"routes": [{"stages": [
                    {"duration_s": 600.0}]}]}}
    assert B.check_nuisance_envelopes(resolved)["pass"] is True
    resolved["health"]["noise_scale"] = 2.0e-3
    assert B.check_nuisance_envelopes(resolved)["pass"] is False


def test_seal_accepts_sprint15_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v1",
                "config_hash": "abc123", "seeds": {"health": 1016}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-1")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v1"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-1"


def test_seal_still_rejects_unknown_protocol(tmp_path):
    from synth.chronicle import write_seal

    manifest = {"protocol": "no-such-protocol", "config_hash": "x",
                "seeds": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown benchmark protocol"):
        write_seal(tmp_path, "H-X")


def test_roster_rejects_wrong_role_pairing():
    with pytest.raises(ValueError, match="roster mismatch"):
        B.prepare_sprint15_block(
            rows=[], ledger=[], wins={}, schedule_events=[],
            resolved_config={}, config_hash="x", history_seed=1016,
            role="H-DESIGN-13")
    with pytest.raises(ValueError, match="roster mismatch"):
        B.prepare_sprint15_block(
            rows=[], ledger=[], wins={}, schedule_events=[],
            resolved_config={}, config_hash="x", history_seed=999,
            role="H-PROOF-1")


# --- Sprint 15 candidate-2 (§7a minimum-duration firing gate) ---

def _run_ledger(cfg):
    from synth.health import RobotHealthProcess
    from synth.scheduler import FactoryScheduler

    sched = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(sched)
    return [(f.failure_id, f.robot_id, round(f.failure_time, 3), f.cohort,
             f.subtype, round(f.duration_d, 6),
             f.degradation_onset is None) for f in health.failure_events]



def _rigged_cfg(seed=7, gate=False):
    """Tiny fast calendar with huge P/W base rates (gate efficacy stress)."""
    from dataclasses import replace

    from synth.chronicle import sprint15_v1_history_config

    cfg = sprint15_v1_history_config(seed=seed)
    cfg.factory = replace(cfg.factory, span_days=90.0, dev_cutoff_days=45.0)
    cfg.scheduler = replace(cfg.scheduler, n_units=192)
    cohorts = []
    for c in cfg.health.cohorts:
        if c.cohort_id in ("P", "W"):
            cohorts.append(replace(
                c, base_rate=1e-3, wear_rate=(c.wear_rate or 0.0) * 10))
        else:
            cohorts.append(replace(c, abrupt_rate=0.0))
    return replace(cfg, health=replace(
        cfg.health, cohorts=tuple(cohorts), min_duration_gate=gate))


def _rigged_abrupt_cfg(seed=9, gate=True):
    """Tiny calendar with hot abrupt hazard (A-shape preservation stress)."""
    from dataclasses import replace

    from synth.chronicle import sprint15_v1_history_config

    cfg = sprint15_v1_history_config(seed=seed)
    cfg.factory = replace(cfg.factory, span_days=90.0, dev_cutoff_days=45.0)
    cfg.scheduler = replace(cfg.scheduler, n_units=192)
    cohorts = [replace(c, abrupt_rate=1e-3)
               if c.cohort_id == "A" else c
               for c in cfg.health.cohorts]
    return replace(cfg, health=replace(
        cfg.health, cohorts=tuple(cohorts), min_duration_gate=gate))
    from synth.scheduler import FactoryScheduler

    sched = FactoryScheduler(cfg).build()
    health = RobotHealthProcess(cfg).run(sched)
    return [(f.failure_id, f.robot_id, round(f.failure_time, 3), f.cohort,
             f.subtype, round(f.duration_d, 6),
             f.degradation_onset is None) for f in health.failure_events]


def test_gate_blocks_subfloor_firing():
    led = _run_ledger(_rigged_cfg(gate=True))
    p_durs = [r[5] for r in led if r[3] == "P"]
    w_durs = [r[5] for r in led if r[3] == "W"]
    assert p_durs and min(p_durs) >= 2.0
    assert w_durs and min(w_durs) >= 6.0


def test_gate_stream_golden_pins_no_early_draw():
    import hashlib as _hashlib
    import json as _json

    led = _run_ledger(_rigged_cfg(gate=True))
    digest = _hashlib.sha256(
        _json.dumps(led, sort_keys=True).encode()).hexdigest()
    # Golden ledger under gating: any extra or missing RNG draw (e.g. a
    # draw-then-discard gate) shifts every downstream event and breaks this.
    assert digest == "79dd2c907633b2e3339d612f3343b10d5de931c3a61e14666ad6c820275ebf7f"


def test_gate_leaves_abrupt_records_intact():
    led = _run_ledger(_rigged_abrupt_cfg())
    abrupt = [r for r in led if r[3] == "A"]
    assert abrupt, "abrupt cohort must still fire under gating"
    assert all(r[5] == 0.0 and r[6] for r in abrupt)


def test_legacy_no_cohort_path_runs_deterministic():
    from dataclasses import replace

    from synth.chronicle import sprint15_v1_history_config
    from synth.health import RobotHealthProcess
    from synth.scheduler import FactoryScheduler

    def episodes():
        cfg = sprint15_v1_history_config(seed=11)
        cfg.factory = replace(
            cfg.factory, span_days=90.0, dev_cutoff_days=45.0)
        cfg.scheduler = replace(cfg.scheduler, n_units=192)
        cfg.health = replace(cfg.health, cohorts=(), base_rate=1e-4)
        sched = FactoryScheduler(cfg).build()
        health = RobotHealthProcess(cfg).run(sched)
        return [(e.kind.value, e.robot_id, round(e.start_time, 3),
                 round(e.end_time, 3) if e.end_time is not None else None)
                for e in health.episodes]

    first, second = episodes(), episodes()
    assert first == second and len(first) > 0


def test_flag_defaults_off_and_v2_enables_only_flag():
    from dataclasses import asdict, replace

    from synth.chronicle import (
        sprint15_v1_history_config,
        sprint15_v2_history_config,
    )
    from synth.config import HealthConfig

    assert HealthConfig().min_duration_gate is False
    v1 = sprint15_v1_history_config(seed=5).health
    v2 = sprint15_v2_history_config(seed=5).health
    assert v1.min_duration_gate is False
    assert v2.min_duration_gate is True
    d1, d2 = asdict(v1), asdict(v2)
    d2.pop("min_duration_gate")
    d1.pop("min_duration_gate")
    assert d1 == d2


def test_candidate2_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1100 <= s <= 1117}
    assert B.S15_B2 == 1100
    assert len(band) == 18 and len(set(band)) == 18
    assert B.S15_ROSTER[1100] == ("DESIGN", "H-DESIGN-17")
    assert B.S15_ROSTER[1103] == ("DESIGN", "H-DESIGN-20")
    assert B.S15_ROSTER[1107] == ("CALIBRATION", "H-CAL-5")
    assert B.S15_ROSTER[1111] == ("CONFIRMATION", "H-CONF-17")
    assert B.S15_ROSTER[1115] == ("SEALED", "H-SEAL-20")
    assert B.S15_ROSTER[1116] == ("PROOF", "H-PROOF-3")
    assert B.S15_ROSTER[1117] == ("PROOF", "H-PROOF-4")


def test_binding_and_audit_default_to_v1_tags():
    import inspect

    assert B.Sprint15Binding().profile == B.S15_PROFILE
    assert B.Sprint15Binding().protocol == B.S15_PROTOCOL
    assert B.Sprint15Binding(
        profile=B.S15_PROFILE_V2,
        protocol=B.S15_PROTOCOL_V2).protocol == B.S15_PROTOCOL_V2
    sig = inspect.signature(B.audit_sprint15)
    assert sig.parameters["profile"].default == B.S15_PROFILE
    assert sig.parameters["protocol"].default == B.S15_PROTOCOL
    sig = inspect.signature(B.prepare_sprint15_block)
    assert sig.parameters["profile"].default == B.S15_PROFILE
    assert sig.parameters["protocol"].default == B.S15_PROTOCOL


def test_seal_accepts_v2_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v2",
                "config_hash": "abc123", "seeds": {"health": 1116}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-3")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v2"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-3"


# --- Sprint 15 candidate-3 (§3a joint fair constrained allocation) ---


def _starvation_fixture():
    """Bulk-P vs tight-A contention pool (Gate-D-shaped starvation rig).

    P1 holds a 10-anchor bulk pool (need 6); A1 holds 3 anchors (need 2),
    each shadowed by a distinct P1 anchor. Under retired bucket-sequential
    fill, P1's bulk acceptance covers A1's pool and starves it; joint
    round-robin lets A1 claim anchors before P1 saturates.
    """
    rows = []
    day = 1.0
    while day < 400.0:
        rows.append(_row(f"robot-01-{day:.0f}", "robot-01", "program-01",
                         day))
        day += 1.0
    ledger = []
    i = 0
    for day in [20, 50, 80, 110, 140, 170, 200, 230, 260, 290]:
        ledger.append(_failure(f"fail-P1-{i}", "robot-01", float(day),
                               "P", "P1", float(day) - 10.0))
        i += 1
    for day in [21, 51, 205]:
        ledger.append(_failure(f"fail-A1-{i}", "robot-01", float(day),
                               "A", "A1", None))
        i += 1
    quota = QuotaConfig(p=6, p1=6, p2=0, w=0, w1=0, w2=0, a=2, a1=2, a2=0,
                        controls=0, robot_days=0, robots_pos=1, robots_neg=0,
                        programs=0, robot_pos_cap=1.0, program_cap=1.0)
    return rows, ledger, {}, quota

def _sequential_oracle(rows, ledger, wins, history_seed, quota):
    """Retired v2 bucket-sequential scheduler (Gate D failure reference).

    Fills P, then W, then A against one shared spacing ledger — the exact
    order bias protocol v3 §3a removes. Test scaffolding only; the retired
    implementation must never return to production code.
    """
    from synth import events as E

    inputs = B.project_allocation_inputs(rows)
    eligible, rejection, _ = B.eligible_anchors(inputs, ledger, wins)
    by_cohort: dict = {}
    for failure in eligible:
        by_cohort.setdefault(failure["cohort"], []).append(failure)
    seed = B.alloc_seed_for(history_seed)
    rng = np.random.default_rng(seed)
    splits = {"P": (("P1", quota.p1), ("P2", quota.p2)),
              "W": (("W1", quota.w1), ("W2", quota.w2)),
              "A": (("A1", quota.a1), ("A2", quota.a2))}
    selected: dict = {}
    per_robot: Counter = Counter()
    kept: dict = {}
    prog_by_op = {(r["robot_id"], r["end_time"]): r["program_id"]
                  for r in inputs}
    total = quota.p + quota.w + quota.a
    cap = max(1, int(math.ceil(quota.robot_pos_cap * total)))
    per_program = {"P": Counter(), "W": Counter()}

    def chosen():
        return {fid for subs in selected.values()
                for ids in subs.values() for fid in ids}

    for cohort in ("P", "W", "A"):
        pool = list(by_cohort.get(cohort, []))
        rng.shuffle(pool)
        selected[cohort] = {}
        for subtype, need in splits[cohort]:
            got = []
            for failure in pool:
                if failure["subtype"] != subtype:
                    continue
                if failure["failure_id"] in chosen():
                    continue
                robot = failure["robot_id"]
                if per_robot[robot] >= cap:
                    continue
                moment = failure["failure_time"]
                if any(abs(moment - k) < quota.spacing_s
                       for k in kept.get(robot, [])):
                    continue
                if cohort in ("P", "W"):
                    prog = prog_by_op.get((robot, moment))
                    if prog is not None and per_program[cohort][prog] >= math.ceil(
                            quota.program_cap * (quota.p if cohort == "P" else quota.w)):
                        continue
                got.append(failure["failure_id"])
                per_robot[robot] += 1
                kept.setdefault(robot, []).append(moment)
                if cohort in ("P", "W"):
                    prog = prog_by_op.get((robot, moment))
                    if prog is not None:
                        per_program[cohort][prog] += 1
                if len(got) >= need:
                    break
            selected[cohort][subtype] = got
            if len(got) < need:
                raise B.InfeasibleCandidate({
                    "reason": "quota-shortfall",
                    "missing": f"{cohort}/{subtype}: {len(got)} < {need}"})
    return selected


def test_joint_allocator_prevents_abrupt_starvation():
    rows, ledger, wins, quota = _starvation_fixture()
    with pytest.raises(B.InfeasibleCandidate) as exc:
        _sequential_oracle(rows, ledger, wins, 0, quota)
    assert exc.value.record["missing"] == "A/A1: 0 < 2"
    out = B.allocate_quotas(rows, ledger, wins, 0, quota)
    assert len(out["selected"]["P"]["P1"]) == 6
    assert len(out["selected"]["A"]["A1"]) == 2
    assert out["rejection"].get("spacing-skip", 0) > 0
    again = B.allocate_quotas(rows, ledger, wins, 0, quota)
    assert again["selected"] == out["selected"]


def test_bucket_sub_rngs_are_independent():
    rows = [_row(f"r1-{d:.0f}", "robot-01", "program-01", float(d))
            for d in range(1, 400)]
    ledger = [_failure(f"fail-A1-{i}", "robot-01", float(day), "A", "A1",
                       None)
              for i, day in enumerate([21, 51, 205])]
    quota = QuotaConfig(p=0, p1=0, p2=0, w=0, w1=0, w2=0, a=2, a1=2, a2=0,
                        controls=0, robot_days=0, robots_pos=1, robots_neg=0,
                        programs=0, robot_pos_cap=1.0)
    base = B.allocate_quotas(rows, ledger, {}, 3, quota)["selected"]
    extra_ledger = list(ledger) + [
        _failure("fail-P-ghost-1", "robot-02", 250.0, "P", "P1", 240.0),
        _failure("fail-P-ghost-2", "robot-02", 270.0, "P", "P2", 260.0),
    ]
    ghost = B.allocate_quotas(rows, extra_ledger, {}, 3,
                              quota)["selected"]
    assert ghost == base


def test_canonical_order_breaks_same_moment_ties():
    rows = [_row(f"f-{d}", "robot-01", "program-01", float(d))
            for d in range(1, 60)]
    ledger = [_failure("fail-P1", "robot-01", 30.0, "P", "P1", 20.0),
              _failure("fail-A1", "robot-01", 30.0, "A", "A1", None)]
    quota = QuotaConfig(p=1, p1=1, p2=0, w=0, w1=0, w2=0, a=1, a1=1, a2=0,
                        controls=0, robot_days=0, robots_pos=1, robots_neg=0,
                        programs=1, robot_pos_cap=1.0)
    with pytest.raises(B.InfeasibleCandidate) as exc:
        B.allocate_quotas(rows, ledger, wins={}, history_seed=5, quota=quota)
    assert exc.value.record["stalled_buckets"] == ["A/A1"]
    assert exc.value.record["missing"] == "A/A1: 0 < 1"


def test_terminal_stall_record_is_deterministic():
    rows = [_row(f"f-{d}", "robot-01", "program-01", float(d))
            for d in range(1, 60)]
    ledger = [_failure("fail-A1", "robot-01", 30.0, "A", "A1", None)]
    quota = QuotaConfig(p=0, p1=0, p2=0, w=0, w1=0, w2=0, a=2, a1=2, a2=0,
                        controls=0, robot_days=0, robots_pos=1, robots_neg=0,
                        programs=0, robot_pos_cap=1.0)
    records = []
    for _ in range(2):
        with pytest.raises(B.InfeasibleCandidate) as exc:
            B.allocate_quotas(rows, ledger, {}, 9, quota)
        record = dict(exc.value.record)
        record["rejection"] = dict(record["rejection"])
        records.append(record)
    assert records[0] == records[1]
    assert records[0]["reason"] == "quota-shortfall"
    assert records[0]["stalled_buckets"] == ["A/A1"]
    assert records[0]["missing"] == "A/A1: 1 < 2"


def test_candidate3_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1200 <= s <= 1217}
    assert B.S15_B3 == 1200
    assert len(band) == 18 and len(set(band)) == 18
    assert B.S15_ROSTER[1200] == ("DESIGN", "H-DESIGN-21")
    assert B.S15_ROSTER[1203] == ("DESIGN", "H-DESIGN-24")
    assert B.S15_ROSTER[1207] == ("CALIBRATION", "H-CAL-6")
    assert B.S15_ROSTER[1211] == ("CONFIRMATION", "H-CONF-21")
    assert B.S15_ROSTER[1115] == ("SEALED", "H-SEAL-20")
    assert B.S15_ROSTER[1117] == ("PROOF", "H-PROOF-4")
    assert B.S15_ROSTER[1215] == ("SEALED", "H-SEAL-24")
    assert B.S15_ROSTER[1216] == ("PROOF", "H-PROOF-5")
    assert B.S15_ROSTER[1217] == ("PROOF", "H-PROOF-6")


def test_v3_binding_and_profile():
    assert B.S15_PROFILE_V3 == "sprint15-v3"
    assert B.S15_PROTOCOL_V3 == "sprint15-benchmark-protocol-v3"
    binding = B.Sprint15Binding(profile=B.S15_PROFILE_V3,
                                protocol=B.S15_PROTOCOL_V3)
    assert binding.protocol == B.S15_PROTOCOL_V3
    from synth.chronicle import sprint15_v3_history_config
    from synth.chronicle import sprint15_v2_history_config
    import dataclasses

    v2 = sprint15_v2_history_config(seed=5).health
    v3 = sprint15_v3_history_config(seed=5).health
    assert v3.min_duration_gate is True
    assert dataclasses.asdict(v3) == dataclasses.asdict(v2)


def test_seal_accepts_v3_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v3",
                "config_hash": "abc123", "seeds": {"health": 1216}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-5")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v3"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-5"


# --- Sprint 15 candidate-4 (§3b exact deterministic CSP) ---


def test_candidate4_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1300 <= s <= 1317}
    assert B.S15_B4 == 1300
    assert len(band) == 18 and len(set(band)) == 18
    assert min(band) > 1217
    assert B.S15_ROSTER[1300] == ("DESIGN", "H-DESIGN-25")
    assert B.S15_ROSTER[1303] == ("DESIGN", "H-DESIGN-28")
    assert B.S15_ROSTER[1307] == ("CALIBRATION", "H-CAL-7")
    assert B.S15_ROSTER[1311] == ("CONFIRMATION", "H-CONF-25")
    assert B.S15_ROSTER[1315] == ("SEALED", "H-SEAL-28")
    assert B.S15_ROSTER[1316] == ("PROOF", "H-PROOF-7")
    assert B.S15_ROSTER[1317] == ("PROOF", "H-PROOF-8")


def test_v4_binding_profile_and_config():
    assert B.S15_PROFILE_V4 == "sprint15-v4"
    assert B.S15_PROTOCOL_V4 == "sprint15-benchmark-protocol-v4"
    assert B.Sprint15Binding().method == "joint"
    v4 = B.Sprint15Binding(profile=B.S15_PROFILE_V4,
                           protocol=B.S15_PROTOCOL_V4, method="exact")
    assert (v4.profile, v4.protocol, v4.method) == (
        "sprint15-v4", "sprint15-benchmark-protocol-v4", "exact")
    from synth.chronicle import sprint15_v4_history_config
    from synth.chronicle import sprint15_v3_history_config
    import dataclasses

    v3 = sprint15_v3_history_config(seed=5).health
    v4cfg = sprint15_v4_history_config(seed=5).health
    assert v4cfg.min_duration_gate is True
    assert dataclasses.asdict(v4cfg) == dataclasses.asdict(v3)


def test_seal_accepts_v4_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v4",
                "config_hash": "abc123", "seeds": {"health": 1316}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-7")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v4"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-7"


def test_floor_cap_constants_match_v2_gates():
    assert B.ROBOT_POS_CAP_MAX == math.floor(0.35 * 64) == 22
    assert B.PROGRAM_CAP_MAX == math.floor(0.60 * 24) == 14
    assert math.ceil(0.35 * 64) == 23
    assert math.ceil(0.60 * 24) == 15


def test_exact_program_encodes_all_constraint_groups():
    rows, ledger, wins, quota = _starvation_fixture()
    inputs = B.project_allocation_inputs(rows)
    eligible, _, _ = B.eligible_anchors(inputs, ledger, wins)
    program = B.build_allocation_program(eligible, inputs, quota)
    assert [row["need"] for row in program["quota_rows"]] == [6, 0, 0, 0, 2, 0]
    assert [row["bucket"] for row in program["quota_rows"]] == [
        "P/P1", "P/P2", "W/W1", "W/W2", "A/A1", "A/A2"]
    by_id = {f["failure_id"]: f for f in eligible}
    order = program["order"]
    assert order == sorted(
        order, key=lambda fid: (by_id[fid]["failure_time"], fid))
    assert program["spacing_edges"], "contention pool must carry spacing rows"
    edge_set = {frozenset((order[i], order[j]))
                for i, j in program["spacing_edges"]}
    assert frozenset(("fail-P1-0", "fail-A1-10")) in edge_set
    for i, j in program["spacing_edges"]:
        a, b = by_id[order[i]], by_id[order[j]]
        assert a["robot_id"] == b["robot_id"]
        assert abs(a["failure_time"] - b["failure_time"]) < quota.spacing_s
    assert program["robot_cap"] == 22
    assert program["program_cap"] == 14
    assert set(program["robot_rows"]) == {f["robot_id"] for f in eligible}
    assert all(key.startswith(("P/", "W/"))
               for key in program["program_rows"])
    assert program["n_variables"] == len(eligible)
    assert program["n_constraints"] == (
        6 + len(program["spacing_edges"])
        + len(program["robot_rows"]) + len(program["program_rows"]))


def test_exact_solver_recovers_starvation_pool():
    rows, ledger, wins, quota = _starvation_fixture()
    out = B.allocate_quotas(rows, ledger, wins, 0, quota, method="exact")
    assert len(out["selected"]["P"]["P1"]) == 6
    assert len(out["selected"]["A"]["A1"]) == 2
    solver = out["solver"]
    assert solver["solver"] == "scipy.optimize.milp (HiGHS)"
    assert solver["scipy_version"] == "1.18.1"
    assert solver["solver_status"] == 0
    assert set(B.SOLVER_MANIFEST_KEYS) <= set(solver)
    assert "solve_s" in solver
    again = B.allocate_quotas(rows, ledger, wins, 0, quota, method="exact")
    assert again["selected"] == out["selected"]
    assert again["selected_ids"] == out["selected_ids"]
    stable = {k: v for k, v in again["solver"].items() if k != "solve_s"}
    assert stable == {k: v for k, v in solver.items() if k != "solve_s"}


def test_solver_status_mapping_and_proven_infeasible():
    assert B.map_solver_status(0, True) == "optimal"
    assert B.map_solver_status(2, False) == "infeasible"
    assert B.map_solver_status(2, True) == "infeasible"
    assert B.map_solver_status(0, False) == "unavailable"
    assert B.map_solver_status(1, False) == "unavailable"
    program = {
        "order": ["only"],
        "quota_rows": [
            {"bucket": "P/P1", "need": 0, "members": []},
            {"bucket": "P/P2", "need": 0, "members": []},
            {"bucket": "W/W1", "need": 0, "members": []},
            {"bucket": "W/W2", "need": 0, "members": []},
            {"bucket": "A/A1", "need": 2, "members": [0]},
            {"bucket": "A/A2", "need": 0, "members": []},
        ],
        "spacing_edges": [],
        "robot_rows": {"robot-01": [0]},
        "program_rows": {},
        "robot_cap": 22,
        "program_cap": 14,
        "n_variables": 1,
        "n_constraints": 7,
    }
    verdict, witness, provenance = B.solve_allocation_program(program)
    assert verdict == "infeasible"
    assert witness is None
    assert provenance["solver_status"] == 2
    assert provenance["scipy_version"] == "1.18.1"
    assert "infeasible" in provenance["solver_message"].lower()


def test_exact_path_maps_unavailable_without_retry(monkeypatch):
    rows, ledger, wins, quota = _starvation_fixture()
    record = {"solver": "mock", "scipy_version": "mock",
              "solver_status": 7, "solver_message": "mock abort",
              "solve_s": 0.0, "n_variables": 1, "n_constraints": 1}
    monkeypatch.setattr(
        B, "solve_allocation_program",
        lambda program: ("unavailable", None, record))
    with pytest.raises(B.AllocationUnavailable) as exc:
        B.allocate_quotas(rows, ledger, wins, 0, quota, method="exact")
    assert exc.value.record["reason"] == "solver-unavailable"
    assert exc.value.record["solver"]["solver_status"] == 7


def test_exact_path_rejects_invalid_witness(monkeypatch):
    rows, ledger, wins, quota = _starvation_fixture()
    monkeypatch.setattr(
        B, "solve_allocation_program",
        lambda program: ("optimal", ["fail-P1-0", "fail-A1-10"],
                         {"solver": "mock"}))
    with pytest.raises(RuntimeError, match="witness breach"):
        B.allocate_quotas(rows, ledger, wins, 0, quota, method="exact")


def test_witness_checker_catches_breaches():
    rows, ledger, wins, quota = _starvation_fixture()
    inputs = B.project_allocation_inputs(rows)
    eligible, _, _ = B.eligible_anchors(inputs, ledger, wins)
    program = B.build_allocation_program(eligible, inputs, quota)
    by_id = {f["failure_id"]: f for f in eligible}
    out = B.allocate_quotas(rows, ledger, wins, 0, quota, method="exact")
    assert B.validate_allocation_witness(
        out["selected_ids"], program, by_id, quota.spacing_s) == []
    spacing_hit = B.validate_allocation_witness(
        ["fail-P1-0", "fail-A1-10"], program, by_id, quota.spacing_s)
    assert any(b.startswith("spacing:") for b in spacing_hit)
    assert B.validate_allocation_witness(
        ["fail-P1-0", "fail-P1-0"], program, by_id,
        quota.spacing_s) == ["duplicate-ids",
                             "P/P1: 1 != 6", "A/A1: 0 != 2"]
    assert B.validate_allocation_witness(
        ["ghost-id"], program, by_id,
        quota.spacing_s) == ["unknown-ids: ['ghost-id']",
                             "P/P1: 0 != 6", "A/A1: 0 != 2"]
    capped = dict(program, robot_cap=1)
    cap_hit = B.validate_allocation_witness(
        ["fail-P1-0", "fail-P1-1"], capped, by_id, quota.spacing_s)
    assert "robot-cap: robot-01: 2" in cap_hit


def test_allocate_rejects_unknown_method():
    rows, ledger, wins, _ = _starvation_fixture()
    with pytest.raises(ValueError, match="unknown allocation method"):
        B.allocate_quotas(rows, ledger, wins, 0, B.DEFAULT_QUOTA,
                          method="bogus")


# --- Sprint 15 candidate-5 (§1 calendar-volume margin) ---


def test_candidate5_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1400 <= s <= 1417}
    assert B.S15_B5 == 1400
    assert len(band) == 18 and len(set(band)) == 18
    assert min(band) > 1317
    assert B.S15_ROSTER[1400] == ("DESIGN", "H-DESIGN-29")
    assert B.S15_ROSTER[1403] == ("DESIGN", "H-DESIGN-32")
    assert B.S15_ROSTER[1406] == ("FIT", "H-FIT-24")
    assert B.S15_ROSTER[1407] == ("CALIBRATION", "H-CAL-8")
    assert B.S15_ROSTER[1411] == ("CONFIRMATION", "H-CONF-29")
    assert B.S15_ROSTER[1415] == ("SEALED", "H-SEAL-32")
    assert B.S15_ROSTER[1416] == ("PROOF", "H-PROOF-9")
    assert B.S15_ROSTER[1417] == ("PROOF", "H-PROOF-10")


def test_v5_binding_profile_and_volume():
    assert B.S15_PROFILE_V5 == "sprint15-v5"
    assert B.S15_PROTOCOL_V5 == "sprint15-benchmark-protocol-v5"
    v5 = B.Sprint15Binding(profile=B.S15_PROFILE_V5,
                           protocol=B.S15_PROTOCOL_V5, method="exact")
    assert (v5.profile, v5.protocol, v5.method) == (
        "sprint15-v5", "sprint15-benchmark-protocol-v5", "exact")
    from synth.chronicle import sprint15_v5_history_config
    from synth.chronicle import sprint15_v4_history_config
    import dataclasses
    v4 = sprint15_v4_history_config(seed=5)
    cfg = sprint15_v5_history_config(seed=5)
    assert cfg.health.min_duration_gate is True
    assert cfg.scheduler.n_units == 3456
    assert cfg.scheduler.arrival_interval_s == 11200.0 * (2304 / 3456)
    assert cfg.scheduler.arrival_jitter_s == 11200.0 * (2304 / 3456)
    assert cfg.factory.span_days == 360.0
    assert cfg.fleet.n_robots == 9
    rest4 = {k: v for k, v in dataclasses.asdict(v4).items()
             if k != "scheduler"}
    rest5 = {k: v for k, v in dataclasses.asdict(cfg).items()
             if k != "scheduler"}
    assert rest5 == rest4
    sched4 = dataclasses.asdict(v4)["scheduler"]
    sched5 = dataclasses.asdict(cfg)["scheduler"]
    assert {k: v for k, v in sched5.items() if k not in (
        "n_units", "arrival_interval_s", "arrival_jitter_s")} == {
            k: v for k, v in sched4.items() if k not in (
                "n_units", "arrival_interval_s", "arrival_jitter_s")}


def test_seal_accepts_v5_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v5",
                "config_hash": "abc123", "seeds": {"health": 1416}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-9")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v5"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-9"


# --- Sprint 15 candidate-6 (§1a fixed-budget reallocation) ---


def test_candidate6_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1500 <= s <= 1517}
    assert B.S15_B6 == 1500
    assert len(band) == 18 and len(set(band)) == 18
    assert min(band) > 1417
    assert B.S15_ROSTER[1500] == ("DESIGN", "H-DESIGN-33")
    assert B.S15_ROSTER[1503] == ("DESIGN", "H-DESIGN-36")
    assert B.S15_ROSTER[1506] == ("FIT", "H-FIT-27")
    assert B.S15_ROSTER[1507] == ("CALIBRATION", "H-CAL-9")
    assert B.S15_ROSTER[1511] == ("CONFIRMATION", "H-CONF-33")
    assert B.S15_ROSTER[1515] == ("SEALED", "H-SEAL-36")
    assert B.S15_ROSTER[1516] == ("PROOF", "H-PROOF-11")
    assert B.S15_ROSTER[1517] == ("PROOF", "H-PROOF-12")


def test_v6_binding_profile_and_reallocation():
    assert B.S15_PROFILE_V6 == "sprint15-v6"
    assert B.S15_PROTOCOL_V6 == "sprint15-benchmark-protocol-v6"
    v6 = B.Sprint15Binding(profile=B.S15_PROFILE_V6,
                           protocol=B.S15_PROTOCOL_V6, method="exact")
    assert (v6.profile, v6.protocol, v6.method) == (
        "sprint15-v6", "sprint15-benchmark-protocol-v6", "exact")
    from synth.chronicle import S15_V6_RATES, S15_V6_SHARES
    from synth.chronicle import sprint15_v6_history_config
    from synth.chronicle import sprint15_v4_history_config
    import dataclasses

    assert S15_V6_SHARES == {"P": 0.41, "W": 0.28, "A": 0.31}
    assert sum(S15_V6_SHARES.values()) == pytest.approx(1.0)
    assert S15_V6_RATES == {"P": 4.6e-10, "W": 2.73e-9, "A": 2.86e-5}
    v4 = sprint15_v4_history_config(seed=5)
    cfg = sprint15_v6_history_config(seed=5)
    assert cfg.health.min_duration_gate is True
    assert cfg.health.upcoming_p == 0.52
    assert dataclasses.asdict(cfg.scheduler) == dataclasses.asdict(
        v4.scheduler)
    assert cfg.factory.span_days == 360.0
    assert cfg.fleet.n_robots == 9
    by_id = {c.cohort_id: c for c in cfg.health.cohorts}
    assert (by_id["P"].share, by_id["P"].base_rate) == (0.41, 4.6e-10)
    assert (by_id["W"].share, by_id["W"].base_rate) == (0.28, 2.73e-9)
    assert (by_id["A"].share, by_id["A"].abrupt_rate) == (0.31, 2.86e-5)
    old = {c.cohort_id: dataclasses.asdict(c)
           for c in v4.health.cohorts}
    new = {c.cohort_id: dataclasses.asdict(c)
           for c in cfg.health.cohorts}
    for cohort in ("P", "W", "A"):
        changed = {f for f in old[cohort] if old[cohort][f] != new[cohort][f]}
        assert changed <= {"share", "base_rate", "abrupt_rate"}, changed
    rest4 = {k: v for k, v in dataclasses.asdict(v4).items()
             if k not in ("scheduler", "health")}
    rest6 = {k: v for k, v in dataclasses.asdict(cfg).items()
             if k not in ("scheduler", "health")}
    assert rest6 == rest4
    assert {k: v for k, v in dataclasses.asdict(cfg.health).items()
            if k not in ("cohorts",)} == {
                k: v for k, v in dataclasses.asdict(v4.health).items()
                if k not in ("cohorts",)}


def test_seal_accepts_v6_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v6",
                "config_hash": "abc123", "seeds": {"health": 1516}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-11")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v6"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-11"


# --- Sprint 15 candidate-7 (§1a deterministic stratification) ---


def test_candidate7_roster_topology():
    band = {s: v for s, v in B.S15_ROSTER.items() if 1600 <= s <= 1617}
    assert B.S15_B7 == 1600
    assert len(band) == 18 and len(set(band)) == 18
    assert min(band) > 1517
    assert B.S15_ROSTER[1600] == ("DESIGN", "H-DESIGN-37")
    assert B.S15_ROSTER[1603] == ("DESIGN", "H-DESIGN-40")
    assert B.S15_ROSTER[1606] == ("FIT", "H-FIT-30")
    assert B.S15_ROSTER[1607] == ("CALIBRATION", "H-CAL-10")
    assert B.S15_ROSTER[1611] == ("CONFIRMATION", "H-CONF-37")
    assert B.S15_ROSTER[1615] == ("SEALED", "H-SEAL-40")
    assert B.S15_ROSTER[1616] == ("PROOF", "H-PROOF-13")
    assert B.S15_ROSTER[1617] == ("PROOF", "H-PROOF-14")


def test_v7_binding_profile_and_stratification_flag():
    assert B.S15_PROFILE_V7 == "sprint15-v7"
    assert B.S15_PROTOCOL_V7 == "sprint15-benchmark-protocol-v7"
    v7 = B.Sprint15Binding(profile=B.S15_PROFILE_V7,
                           protocol=B.S15_PROTOCOL_V7, method="exact")
    assert (v7.profile, v7.protocol, v7.method) == (
        "sprint15-v7", "sprint15-benchmark-protocol-v7", "exact")
    from synth.chronicle import sprint15_v7_history_config
    from synth.chronicle import sprint15_v4_history_config
    import dataclasses

    v4 = sprint15_v4_history_config(seed=5)
    cfg = sprint15_v7_history_config(seed=5)
    assert cfg.health.min_duration_gate is True
    assert cfg.health.stratified_subtype_emission is True
    assert v4.health.stratified_subtype_emission is False
    assert dataclasses.asdict(cfg.scheduler) == dataclasses.asdict(
        v4.scheduler)
    assert dataclasses.asdict(cfg.factory) == dataclasses.asdict(v4.factory)
    assert dataclasses.asdict(cfg.health) != dataclasses.asdict(v4.health)
    only_flag = {k for k, v in dataclasses.asdict(cfg.health).items()
                 if v != dataclasses.asdict(v4.health)[k]}
    assert only_flag == {"stratified_subtype_emission"}, only_flag
    rest4 = {k: v for k, v in dataclasses.asdict(v4).items()
             if k not in ("health",)}
    rest7 = {k: v for k, v in dataclasses.asdict(cfg).items()
             if k not in ("health",)}
    assert rest7 == rest4


def test_seal_accepts_v7_protocol(tmp_path):
    from synth.chronicle import verify_seal, write_seal
    from synth.dataset import _hash_file

    manifest = {"protocol": "sprint15-benchmark-protocol-v7",
                "config_hash": "abc123", "seeds": {"health": 1616}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    seal = write_seal(tmp_path, "H-PROOF-13")
    assert seal["protocol"] == "sprint15-benchmark-protocol-v7"
    assert seal["manifest_sha256"] == _hash_file(tmp_path / "manifest.json")
    assert verify_seal(tmp_path)["role"] == "H-PROOF-13"
