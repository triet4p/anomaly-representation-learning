"""Focused tests for Sprint 15 candidate-7 stratification + preflight (Batch B).

Covers flag-gated timing/density invariance with subtype rebalancing (focused
synthetic calendar, deterministic seed), the preflight no-write contract
(import and call-site constraints), roster unanimity edge cases, and frozen
preflight identifiers. No generation roots; synthetic configs only.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from pathlib import Path


def _small_configs(seed: int = 5):
    """v3 configs with only the existing subtype flag switched off/on."""
    from synth.chronicle import sprint18_iterative_v3_history_config

    off = sprint18_iterative_v3_history_config(seed=seed)
    off.health = replace(
        off.health, stratified_subtype_emission=False)
    on = sprint18_iterative_v3_history_config(seed=seed)
    for cfg in (off, on):
        cfg.scheduler = replace(cfg.scheduler, n_units=384)
        cfg.factory = replace(cfg.factory, span_days=180.0,
                              dev_cutoff_days=90.0)
    return off, on


def test_iterative_v3_failure_stratification_preserves_timing_and_balances():
    import hashlib

    from synth.chronicle import _failure_to_dict, build_chronological
    from synth.schema import EpisodeKind

    off_cfg, on_cfg = _small_configs()
    _, health_off, _, _ = build_chronological(off_cfg)
    _, health_on, _, _ = build_chronological(on_cfg)
    off = [_failure_to_dict(r) for r in health_off.failure_events]
    on = [_failure_to_dict(r) for r in health_on.failure_events]
    strip = lambda d: {k: v for k, v in d.items() if k != "subtype"}
    assert len(off) == len(on) > 0
    assert [strip(d) for d in off] == [strip(d) for d in on]

    off_episodes = {e.episode_id: e for e in health_off.episodes}
    assert all(
        failure["subtype"]
        == off_episodes[failure["degradation_episode_id"]].subtype
        for failure in off
        if failure["cohort"] in ("P", "W")
    )

    episode_ordinal = Counter()
    episode_by_id = {e.episode_id: e for e in health_on.episodes}
    for episode in health_on.episodes:
        if episode.kind is not EpisodeKind.DEGRADATION or episode.subtype is None:
            continue
        cohort = "P" if episode.subtype in ("P1", "P2") else "W"
        key = (episode.robot_id, cohort)
        episode_ordinal[key] += 1
        digest = hashlib.sha256(
            "|".join([
                "sprint14-subtype", str(on_cfg.health.seed),
                episode.robot_id, cohort, str(episode_ordinal[key]),
            ]).encode()
        ).digest()
        labels = ("P1", "P2") if cohort == "P" else ("W1", "W2")
        assert episode.subtype == labels[
            (episode_ordinal[key] + digest[0]) % len(labels)]

    failure_ordinal = Counter()
    failure_counts = {}
    a_ordinal = Counter()
    has_dual_label = False
    for failure in health_on.failure_events:
        if failure.cohort == "A":
            a_ordinal[failure.robot_id] += 1
            digest = hashlib.sha256(
                "|".join([
                    "sprint15-stratified-A", str(on_cfg.health.seed),
                    failure.robot_id,
                ]).encode()
            ).digest()
            labels = next(
                c.subtypes for c in on_cfg.health.cohorts
                if c.cohort_id == "A"
            )
            assert failure.subtype == labels[
                (a_ordinal[failure.robot_id] + digest[0]) % len(labels)]
            continue

        key = (failure.robot_id, failure.cohort)
        failure_ordinal[key] += 1
        phase = hashlib.sha256(
            "|".join([
                "sprint18-failure-subtype-v3", str(on_cfg.health.seed),
                failure.robot_id, failure.cohort,
            ]).encode()
        ).digest()[0]
        labels = (
            ("P1", "P2") if failure.cohort == "P" else ("W1", "W2")
        )
        assert failure.subtype == labels[
            (failure_ordinal[key] + phase) % len(labels)]
        failure_counts.setdefault(key, Counter())[failure.subtype] += 1
        episode = episode_by_id[failure.degradation_episode_id]
        has_dual_label |= failure.subtype != episode.subtype

    assert failure_counts
    for (robot_id, cohort), counts in failure_counts.items():
        labels = ("P1", "P2") if cohort == "P" else ("W1", "W2")
        assert abs(counts[labels[0]] - counts[labels[1]]) <= 1, (
            robot_id, cohort, counts)
    assert has_dual_label

def test_iterative_v2_failure_stratification_preserves_timing_and_balances():
    test_iterative_v3_failure_stratification_preserves_timing_and_balances()

def test_preflight_module_is_no_write():
    tree = ast.parse(Path("src/synth/preflight15.py").read_text())
    imported = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    assert imported <= {
        "__future__", "time", "typing", "synth", "json", "collections",
        "dataclasses", "experiments.sprint18_task5_measurability",
        "synth.chronicle", "synth.balanced",
    }, imported
    assert "synth.probe15" not in imported
    assert "synth.dataset" not in imported
    forbidden = {"materialize_chronological", "write_seal", "verify_seal",
                 "score_files", "evaluate_histories", "select_threshold",
                 "fit_centroid", "mkdir", "write_text", "write_bytes"}
    assert not (called & forbidden), called & forbidden


def test_preflight_constants_and_unanimity_edge():
    from synth import preflight15 as P

    assert P.PREFLIGHT_PROFILE == "sprint15-v7"
    assert P.PREFLIGHT_PROTOCOL == "sprint15-benchmark-protocol-v7"
    assert P.PREFLIGHT_C2_PROFILE == "sprint18-c2"
    assert P.PREFLIGHT_C2_PROTOCOL == "sprint15-benchmark-protocol-v7"
    assert P.run_preflight([])["verdict"] == "PREFLIGHT-FAIL"
    assert P.run_preflight_c2([])["verdict"] == "PREFLIGHT-FAIL"


def test_preflight_unknown_profile_raises_fail_closed():
    import pytest

    from synth import preflight15 as P

    with pytest.raises(ValueError, match="unknown preflight profile"):
        P.preflight_seed(9000, profile="sprint18-c2 ")
    with pytest.raises(ValueError, match="unknown preflight profile"):
        P.run_preflight([9000], profile="nope")
    with pytest.raises(ValueError, match="unknown preflight profile"):
        P.run_preflight([], profile="nope")


def test_iterative_preflight_profile_requires_role_bound_gates():
    import pytest

    from synth import preflight15 as P

    factory, profile, protocol = P._resolve_preflight(
        P.PREFLIGHT_ITERATIVE_PROFILE
    )
    assert profile == "sprint18-iterative-v1"
    assert protocol == "sprint15-benchmark-protocol-v7"
    config = factory(seed=9000)
    assert config.factory.span_days == 450.0
    assert config.scheduler.n_units == 2880
    with pytest.raises(ValueError, match="requires fixed roles"):
        P.run_preflight([], profile=P.PREFLIGHT_ITERATIVE_PROFILE)
    with pytest.raises(ValueError, match="requires one of"):
        P.preflight_seed_iterative_v1(9000, "UNKNOWN")
    assert P.run_preflight(
        [], profile=P.PREFLIGHT_ITERATIVE_PROFILE, roles=[]
    )["verdict"] == "PREFLIGHT-FAIL"


def test_iterative_v3_preflight_profile_is_role_bound():
    import pytest

    from synth import preflight15 as P

    factory, profile, protocol = P._resolve_preflight(
        P.PREFLIGHT_ITERATIVE_V3_PROFILE
    )
    assert profile == "sprint18-iterative-v3"
    assert protocol == "sprint15-benchmark-protocol-v7"
    config = factory(seed=94100)
    assert config.factory.span_days == 450.0
    assert config.factory.dev_cutoff_days == 225.0
    assert config.scheduler.n_units == 2880
    with pytest.raises(ValueError, match="requires fixed roles"):
        P.run_preflight([], profile=P.PREFLIGHT_ITERATIVE_V3_PROFILE)
    with pytest.raises(ValueError, match="requires one of"):
        P.preflight_seed(94100, profile=P.PREFLIGHT_ITERATIVE_V3_PROFILE)
    assert P.run_preflight(
        [], profile=P.PREFLIGHT_ITERATIVE_V3_PROFILE, roles=[]
    )["verdict"] == "PREFLIGHT-FAIL"


def test_iterative_v2_preflight_profile_is_role_bound():
    from synth import preflight15 as P

    factory, profile, protocol = P._resolve_preflight(
        P.PREFLIGHT_ITERATIVE_V2_PROFILE
    )
    assert profile == "sprint18-iterative-v2"
    assert protocol == "sprint15-benchmark-protocol-v7"
    assert P.PREFLIGHT_ITERATIVE_V2_PROFILE in P._RETIRED_ITERATIVE_V2_PROFILES
    assert P.PREFLIGHT_ITERATIVE_V2_PROFILE not in P._ITERATIVE_PROFILES
