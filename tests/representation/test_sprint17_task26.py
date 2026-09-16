"""Focused Task 26 contract tests: published matrix integrity.

Read-only: the matrix publisher consumes hash-verified accepted sources and
emits exact values. Tests assert registration completeness, byte-traceable
aggregates, gate/class reproduction, deterministic rematerialization, and
fail-closed behavior — never scoring, training, Sealed access, or verdicts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "experiments" / "sprint17-task26-matrix.json"
PUBLISHER_PATH = REPO_ROOT / "experiments" / "sprint17_task26_matrix.py"


def _load_publisher():
    spec = importlib.util.spec_from_file_location("sprint17_task26_matrix", PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUB = _load_publisher()

EXPECTED_ARMS = (
    "B0",
    "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A",
    "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9",
)

EXPECTED_K_MEMBERS = {
    "K1": ("C2-A", "C3-A"),
    "K2": ("C3-A", "C5-A"),
    "K3": ("C5-A", "C6-A"),
    "K4": ("C6-A", "C7-A"),
    "K5": ("C7-A", "C8-A"),
    "K6": ("C8-A", "C9-B"),
    "K7": ("C2-A", "C3-A", "C5-A", "C6-A"),
    "K8": ("C7-A", "C8-A", "C9-B"),
    "K9": ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"),
}


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_matrix_has_exactly_24_registered_rows() -> None:
    m = _matrix()
    assert tuple(m["rows"]) == EXPECTED_ARMS
    assert len(m["rows"]) == 24
    assert m["lock_sha256"] == "14fab303d78725af061aaa639a558f84b262c76d5b918042618a109d61afd039"
    assert m["development_support_sha256"] == "800fc825ac948c3661c2dc727c83083095a796df8950c94922dafabae318f59a"
    assert m["confirmation_support_sha256"] == "6ed3f08cb644c0548cc539a5f257af1eeb7f7cbdca1c0f57f7fdf680642926b3"
    assert m["qualification"] == "MEASURABLE_WITH_USER_WAIVER"
    assert len(m["sources"]) == 17


def test_membership_and_selection_match_lock() -> None:
    m = _matrix()
    for kid, members in EXPECTED_K_MEMBERS.items():
        assert tuple(m["rows"][kid]["members"]) == members
        assert m["rows"][kid]["arm_kind"] == "combination"
    for aid in EXPECTED_ARMS[1:15]:
        assert m["rows"][aid]["arm_kind"] == "single"
        assert m["rows"][aid]["members"] == ["B0"]
    assert m["rows"]["B0"]["arm_kind"] == "baseline"
    assert m["selection"] == {"C2": "C2-A", "C3": "C3-A", "C5": "C5-A", "C6": "C6-A",
                              "C7": "C7-A", "C8": "C8-A", "C9": "C9-B"}
    assert m["invalid"] == {"count": 0, "arms": []}


def test_branch_separation_and_no_sprint_verdict() -> None:
    m = _matrix()
    text = MATRIX_PATH.read_text(encoding="utf-8")
    assert "NO_REPRODUCIBLE_RECOVERY" not in text
    assert "SINGLE_COMPONENT_RECOVERY" not in text or "single_component_recovery" in text
    assert "UNRESOLVED" not in text.replace("unresolved", "").replace('"unresolved": false', "")
    for aid, row in m["rows"].items():
        if aid == "B0":
            continue
        assert set(row["development"]) >= {"S_pred", "S_pop"}
        assert set(row["confirmation"]) >= {"S_pred", "S_pop"}
        assert row["development"]["S_pred"]["macro_pw"] != row["development"]["S_pop"]["macro_pw"] or aid in (
            "C5-B", "C8-A", "C8-B", "C9-A", "C9-B", "K6")


def test_aggregates_trace_to_raw_sources() -> None:
    m = _matrix()
    pairs = [("experiments/sprint17-task10-c2-summary.json", "C2-A"),
             ("experiments/sprint17-task11-c3-summary.json", "C3-A"),
             ("experiments/sprint17-task12-c5-summary.json", "C5-A"),
             ("experiments/sprint17-task13-c6-summary.json", "C6-A"),
             ("experiments/sprint17-task14-c7-summary.json", "C7-A"),
             ("experiments/sprint17-task15-c8-summary.json", "C8-A"),
             ("experiments/sprint17-task16-c9-summary.json", "C9-B"),
             ("experiments/sprint17-task20-k-summary.json", "K1"),
             ("experiments/sprint17-task21-k-summary.json", "K9")]
    for rel, aid in pairs:
        doc = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        assert m["rows"][aid]["development"]["S_pred"]["delta_pw"] == doc["arms"][aid]["S_pred"]["delta_vs_B0"]
        assert m["rows"][aid]["development"]["S_pop"]["delta_pw"] == doc["arms"][aid]["S_pop"]["delta_vs_B0"]
    s24 = json.loads((REPO_ROOT / "experiments/sprint17-task24-confirmation-summary.json").read_text(encoding="utf-8"))
    s25 = json.loads((REPO_ROOT / "experiments/sprint17-task25-confirmation-summary.json").read_text(encoding="utf-8"))
    for aid in ("C9-B", "C2-A"):
        assert m["rows"][aid]["confirmation"]["S_pred"]["delta_pw"] == s24["arms"][aid]["S_pred"]["delta_pw"]
    for aid in ("K7", "K9"):
        assert m["rows"][aid]["confirmation"]["S_pred"]["delta_pw"] == s25["arms"][aid]["S_pred"]["delta_pw"]
        assert m["rows"][aid]["interaction"] == s25["interactions"][aid]


def test_gates_and_classes_reproduce_frozen_records() -> None:
    m = _matrix()
    assert all(v in ("baseline", "valid_negative", "non_replication",
                     "single_component_recovery", "interaction_only_recovery")
               for v in m["classes"].values())
    assert m["classes"]["B0"] == "baseline"
    # Frozen outcome: nothing passes the +0.05 effect gate, so no recovery class appears.
    assert "single_component_recovery" not in m["classes"].values()
    assert "interaction_only_recovery" not in m["classes"].values()
    # Sign flips without replication are exactly K5/K6/K8.
    assert {a for a, c in m["classes"].items() if c == "non_replication"} == {"K5", "K6", "K8"}
    assert all(m["rows"][a]["confirmation"]["gates"]["recovery"] is False for a in EXPECTED_ARMS)
    assert all(m["rows"][a]["confirmation"]["gates"]["minimum_effect"] is False for a in EXPECTED_ARMS)


def test_deterministic_rematerialization() -> None:
    first = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    rebuilt = PUB.build_matrix()
    canonical = (json.dumps(rebuilt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == first
    assert PUB.check_sources() == rebuilt["sources"]


def test_fail_closed_on_source_drift_and_sealed_markers() -> None:
    import copy
    tampered = copy.deepcopy(PUB.EXPECTED_SOURCES)
    tampered["experiments/sprint17-task24-confirmation-summary.json"] = "0" * 64
    old = PUB.EXPECTED_SOURCES
    PUB.EXPECTED_SOURCES = tampered
    try:
        try:
            PUB.check_sources()
        except ValueError:
            pass
        else:
            raise AssertionError("source drift did not trip")
    finally:
        PUB.EXPECTED_SOURCES = old
    assert PUB.check_no_sealed_contact() is True
