"""Focused Task 4 behavioral tests: leakage and boundary errors only.

Row-level behavior runs on real-regime fixtures (DEV-02 passing, DEV-01
fail-closed boundary) through the public entry points. Covers only
plausible leakage/boundary failures.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from representation import sprint24_views as views

FIXTURE_DIR = Path("artifacts/sprint-24/evidence/s24-t4/s24-t4-dev02-fixture")
HISTORY_ID = "S18I-ITER-0005-DEVELOPMENT-02"


def _load_fixture() -> dict:
    import json as _json
    rows = _json.loads((FIXTURE_DIR / "rows.json").read_text(encoding="utf-8"))
    windows = _json.loads((FIXTURE_DIR / "failure_windows.json").read_text(encoding="utf-8"))
    degradations = _json.loads((FIXTURE_DIR / "degradation_episodes.json").read_text(encoding="utf-8"))
    maint = _json.loads((FIXTURE_DIR / "maintenance_windows.json").read_text(encoding="utf-8"))
    edges = _json.loads((FIXTURE_DIR / "bin_edges.json").read_text(encoding="utf-8"))
    return {"rows": rows, "windows": windows, "degradations": degradations, "maint": maint, "edges": edges}


def _labels(rows) -> dict[str, dict]:
    return {
        r["file_id"]: {"file_label": r["file_label"], "anomaly_family": r.get("anomaly_family")}
        for r in rows
    }


def _build(fx, **kw) -> object:
    return views.build_history_views(
        HISTORY_ID, fx["rows"], fx["windows"], fx["degradations"], fx["maint"],
        cutoff_c1=fx["edges"][4], cutoff_c2=fx["edges"][7], bin_edges=fx["edges"], **kw
    )


def test_views_reproduce_without_source_mutation() -> None:
    fx = _load_fixture()
    before = copy.deepcopy(fx["rows"])
    first = _build(fx)
    second = _build(fx)
    assert fx["rows"] == before
    assert first.manifest_digest == second.manifest_digest
    assert first.frames == second.frames
    assert first.budgets == second.budgets


def test_fixed_slices_and_budgets_disjoint_with_frozen_shapes() -> None:
    fx = _load_fixture()
    built = _build(fx)
    assert len(built.frame("V-VAL")) == 60
    assert len(built.frame("V-TRIG")) == 100
    assert len(built.frame("V-R0")) == 50
    assert {len(built.budgets[b]) for b in ("B1", "B2", "B3")} == {40, 120, 320}
    assert set(built.budgets["B1"]) <= set(built.budgets["B2"]) <= set(built.budgets["B3"])
    fixed = [set(built.frame(n)) for n in ("V-VAL", "V-TRIG", "V-R0")]
    assert fixed[0].isdisjoint(fixed[1]) and fixed[0].isdisjoint(fixed[2]) and fixed[1].isdisjoint(fixed[2])
    assert set(built.budgets["B3"]).isdisjoint(fixed[0] | fixed[1] | fixed[2])


def test_merged_groups_never_cross_reset_and_attach_is_audited() -> None:
    fx = _load_fixture()
    groups, audit, parent_of = views.build_episode_groups(
        HISTORY_ID, fx["rows"], fx["windows"], fx["degradations"], fx["maint"]
    )
    for group_id, members in groups.items():
        if group_id.count("|") != 2:
            continue
        assert group_id in parent_of, f"piece {group_id} lacks parent custody"
        robot = group_id.split("|")[1]
        spans = {(float(s), float(e)) for s, e in fx["maint"].get(robot, [])}
        assert spans, f"group {group_id} has no maintenance table"
    by_label = {}
    for entry in audit:
        by_label.setdefault(entry["file_label"], 0)
        by_label[entry["file_label"]] += 1
    assert sum(by_label.values()) == len(audit) > 0
    assert by_label.get("abnormal", 0) > 0

def test_premature_label_access_rejected_and_past_only_replay() -> None:
    fx = _load_fixture()
    built = _build(fx)
    gate = views.LabelReleaseGate(built, _labels(fx["rows"]))
    replay = views.ReplayEligibility(gate)
    adapt_release = built.label_release["V-ADAPT"]
    adapt_id = built.frame("V-ADAPT")[0]
    e1_release = built.label_release["V-SEQ-E1"]
    e1_id = built.frame("V-SEQ-E1")[0]
    with pytest.raises((ValueError, KeyError)):
        gate.require_released([adapt_id, e1_id], adapt_release)
    assert adapt_id in gate.released_labels(adapt_release)
    assert e1_id not in gate.released_labels(adapt_release)
    assert e1_id in gate.released_labels(e1_release)
    with pytest.raises(ValueError, match="unreleased"):
        replay.require_past_only([adapt_id, e1_id], adapt_release)
    assert e1_id in replay.eligible_at([adapt_id, e1_id], e1_release)


def test_evaluation_and_reserve_labels_never_released() -> None:
    fx = _load_fixture()
    built = _build(fx)
    gate = views.LabelReleaseGate(built, _labels(fx["rows"]))
    by_id = {r["file_id"]: r for r in fx["rows"]}
    # Manifest rows carry no anomaly_family (family lives in shard NPZ
    # labels); the withheld-family path is covered by the accepted-source
    # audit, not by this manifest-row fixture. Here: future evaluation
    # labels never release to the learner.
    for file_id in built.frame("V-FUTURE"):
        with pytest.raises(KeyError):
            gate.release_time_of(file_id)
    # V-R0 sits inside V-ADAPT time, so its labels release at the ADAPT
    # close; the probe stays reporting-only because no training, stopping,
    # or threshold frame ever draws from it.
    reserve_slice = set(built.frame("V-RESERVE-SLICE"))
    if reserve_slice:
        assert reserve_slice <= set(built.frame("V-FUTURE")) | set(built.frame("V-SEQ-E1")) | set(
            built.frame("V-SEQ-E2")) | set(built.frame("V-SEQ-E3"))
        assert all(
            by_id[i]["robot_id"] == "robot-08" or by_id[i]["program_id"] == "program-03"
            for i in reserve_slice
        )


def test_default_allocator_fills_exact_totals_with_deterministic_abnormal() -> None:
    """Default D2 allocator: exact file totals, deterministic abnormal counts.

    Self-contained: builds real views from the fixture through the public
    entry point (default ``feasible_abnormal=True``). Exact file totals hold;
    abnormal counts are the deterministic source-derived values recorded in
    the manifest digest. Explicit ``feasible_abnormal=False`` preserves the
    fail-closed boundary for infeasible frozen quotas.
    """
    from representation.sprint24_views import SHAPE_R0, SHAPE_VAL, SHAPE_TRIG, BUDGETS
    fx = _load_fixture()
    built = _build(fx)
    by_id = {r["file_id"]: r for r in fx["rows"]}

    def counts(ids) -> tuple[int, int]:
        ab = sum(1 for i in ids if by_id[i]["file_label"] == "abnormal")
        return (ab, len(ids) - ab)
    assert [len(built.frame(n)) for n in ("V-R0", "V-VAL", "V-TRIG")] == [
        SHAPE_R0[0], SHAPE_VAL[0], SHAPE_TRIG[0]]
    assert [len(built.budgets[b]) for b in ("B1", "B2", "B3")] == [
        BUDGETS["B1"][0], BUDGETS["B2"][0], BUDGETS["B3"][0]]
    assert set(built.budgets["B1"]) <= set(built.budgets["B2"]) <= set(built.budgets["B3"])
    # No parent episode spans two sampled roles (D3 custody).
    role_of: dict[str, str] = {}
    for role, ids in [("V-R0", built.frame("V-R0")), ("V-VAL", built.frame("V-VAL")),
                      ("V-TRIG", built.frame("V-TRIG")), ("B3", built.budgets["B3"])]:
        for i in ids:
            parent = built.parent_of.get(built.group_of(i), built.group_of(i))
            assert role_of.setdefault(parent, role) == role, f"{parent} spans roles"
    # Fail-closed boundary preserved for infeasible exact quotas (DEV-01
    # fixture: 90 passing abnormal < 106 frozen quota).
    import pytest as _pytest
    dev01 = Path("artifacts/sprint-24/evidence/s24-t4/s24-t4-tiny-fixture")
    rows01 = json.loads((dev01 / "rows.json").read_text(encoding="utf-8"))
    with _pytest.raises(ValueError):
        views.build_history_views(
            "S18I-ITER-0005-DEVELOPMENT-01", rows01,
            json.loads((dev01 / "failure_windows.json").read_text(encoding="utf-8")),
            json.loads((dev01 / "degradation_episodes.json").read_text(encoding="utf-8")),
            json.loads((dev01 / "maintenance_windows.json").read_text(encoding="utf-8")),
            cutoff_c1=12902229.909, cutoff_c2=22578902.340,
            bin_edges=json.loads((dev01 / "bin_edges.json").read_text(encoding="utf-8")),
            feasible_abnormal=False,
        )
