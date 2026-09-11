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
    """v4 (flag-off) and v7 (flag-on) configs on a small fast calendar."""
    from synth.chronicle import sprint15_v4_history_config
    from synth.chronicle import sprint15_v7_history_config

    off = sprint15_v4_history_config(seed=seed)
    on = sprint15_v7_history_config(seed=seed)
    for cfg in (off, on):
        cfg.scheduler = replace(cfg.scheduler, n_units=384)
        cfg.factory = replace(cfg.factory, span_days=180.0,
                              dev_cutoff_days=90.0)
    return off, on


def test_stratification_preserves_timing_and_rebalances():
    from synth.chronicle import _failure_to_dict, build_chronological

    off_cfg, on_cfg = _small_configs()
    _, health_off, _, _ = build_chronological(off_cfg)
    _, health_on, _, _ = build_chronological(on_cfg)
    off = [_failure_to_dict(r) for r in health_off.failure_events]
    on = [_failure_to_dict(r) for r in health_on.failure_events]
    strip = lambda d: {k: v for k, v in d.items() if k != "subtype"}
    assert len(off) == len(on) > 0
    assert [strip(d) for d in off] == [strip(d) for d in on]
    assert [d["subtype"] for d in off] != [d["subtype"] for d in on]
    on_by_cohort = {}
    for d in on:
        on_by_cohort.setdefault(d["cohort"], []).append(d["subtype"])
    assert Counter(on_by_cohort["A"]) == {"A1": 8, "A2": 8}
    for cohort, subs in on_by_cohort.items():
        counts = Counter(subs)
        assert max(counts.values()) - min(counts.values()) <= 3, cohort


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
    assert imported <= {"__future__", "time", "typing", "synth",
                        "synth.chronicle", "synth.balanced"}, imported
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
    assert P.run_preflight([])["verdict"] == "PREFLIGHT-FAIL"
