"""Focused behavioral tests for the S24-7-M7 controlled-data protocol design.

These tests protect the invariants the design-only protocol depends on: the
frozen design digest before any validation, fail-closed label-blind planned
identity construction, schedule disjointness, lineage separation from the
immutable old corpus, seed-block non-collision, quota refusal without rescue,
the monthly control requirement, and the forward-budget gate. Fixtures are
small and synthetic: no accepted data is copied and no generation exists.
"""

from __future__ import annotations

from representation import sprint24_controlled_data_protocol as protocol


def test_design_freeze_is_deterministic_and_marks_research_defaults() -> None:
    first = protocol.protocol_freeze()
    second = protocol.protocol_freeze()
    assert first["protocol_sha256"] == second["protocol_sha256"] == protocol.protocol_sha256()
    spec = first["spec"]
    assert isinstance(spec, dict)
    assert spec["research_bands"]["low_burden_fraction_primary"] == 0.10  # type: ignore[index]
    assert spec["research_bands"]["elevated_fraction_primary"] == 0.20  # type: ignore[index]
    assert "not owner-approved" in spec["research_bands"]["assumption_status"]  # type: ignore[index]
    assert spec["calendar"]["timezone"] == "UTC"  # type: ignore[index]
    assert spec["scale"]["n_histories"] == 6  # type: ignore[index]


def test_planned_identity_construction_is_label_blind() -> None:
    protocol.assert_planning_construction_label_blind()
    weeks = protocol.planned_week_identities("S24C-ITER-0001-CONTROLLED-01")
    months = protocol.planned_month_identities("S24C-ITER-0001-CONTROLLED-01")
    assert len(weeks) == protocol.N_PLANNED_WEEKS
    assert len(months) == protocol.N_PLANNED_MONTHS
    assert protocol.planned_reference_identity("S24C-ITER-0001-CONTROLLED-01").endswith("end_le_c1")
    assert protocol.order_planned_identities(list(reversed(weeks))) == weeks


def test_disjoint_support_gate_refuses_shared_weeks() -> None:
    negative = protocol.validate_arm_disjointness({1, 2, 3}, {3, 4, 5})
    positive = protocol.validate_arm_disjointness({1, 2, 3}, {5, 6})
    assert negative["status"] == "REFUSED" and negative["reason"] == "REFUSED-OVERLAP"
    assert positive["status"] == "PASS"
    schedule = protocol.validate_disjoint_support(protocol.DEFAULT_WEEK_BANDS)
    assert schedule["status"] == "PASS"


def test_quota_gate_refuses_shortfall_without_rescue() -> None:
    negative = protocol.validate_quotas(220, 5, 4, 12)
    positive = protocol.validate_quotas(220, 40, 4, 12)
    assert any(check["status"] == "REFUSED" for check in negative)
    assert all(check["status"] == "PASS" for check in positive)


def test_lineage_and_seed_gates_refuse_old_corpus_reuse() -> None:
    lineage_negative = protocol.validate_lineage("S18I-ITER-0005-DEVELOPMENT-01")
    lineage_positive = protocol.validate_lineage("S24C-ITER-0001-CONTROLLED-01")
    assert lineage_negative["status"] == "REFUSED"
    assert lineage_negative["reason"] == "REFUSED-LINEAGE"
    assert lineage_positive["status"] == "PASS"
    seed_negative = protocol.validate_seeds(32064, [42401])
    seed_positive = protocol.validate_seeds(42401, [])
    assert seed_negative["status"] == "REFUSED"
    assert seed_positive["status"] == "PASS"


def test_monthly_gate_requires_two_disjoint_low_runs() -> None:
    bad = {1: "WARMUP-PARTIAL", 2: "LOW", 3: "ELEVATED", 4: "LOW", 5: "LOW", 6: "LOW"}
    negative = protocol.validate_monthly_bands(bad)
    positive = protocol.validate_monthly_bands(protocol.DEFAULT_MONTH_BANDS)
    assert negative["status"] == "REFUSED"
    assert negative["reason"] == "REFUSED-NO-MONTHLY-CONTROL"
    assert positive["status"] == "PASS"


def test_forward_gate_refuses_planned_inference_on_current_headroom() -> None:
    estimate = protocol.cost_estimate()
    assert isinstance(estimate["forwards"], dict)
    assert int(estimate["forwards"]["shortfall_vs_headroom"]) == 20845
    gate = protocol.forward_gate()
    assert gate["gate"] == "INFERENCE-NOT-AUTHORIZED"


def test_planned_scale_is_six_independent_histories_not_a_power_claim() -> None:
    spec = protocol.protocol_spec()
    assert isinstance(spec, dict)
    assert len(spec["scale"]["history_ids"]) == 6  # type: ignore[index]
    assert len(set(spec["scale"]["history_ids"])) == 6  # type: ignore[index]
    assert "No power" in spec["scale"]["scale_justification"]  # type: ignore[index]


def test_planned_pairs_cover_both_estimands_per_horizon() -> None:
    pairs = protocol.default_planned_pairs()
    weekly = [row for row in pairs if row["horizon"] == "week"]
    monthly = [row for row in pairs if row["horizon"] == "month"]
    assert len(weekly) == 5 and len(monthly) == 5
    assert sum(1 for row in weekly if row["kind"] == "CONTROL") == 1
    assert sum(1 for row in weekly if row["kind"] == "TRANSITION") == 4
