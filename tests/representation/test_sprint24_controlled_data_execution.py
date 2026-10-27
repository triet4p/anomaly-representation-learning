"""Focused behavioral tests for the S24-7-M8 v2 execution readiness.

These tests protect the invariants the readiness task depends on: v1 design
preservation, deterministic v2 execution freeze, real-calendar week/month
nesting, burden-band quotas, roster support, label-blind construction,
fail-fast refusal without rescue, non-overwrite behavior, seed lineage,
realized (not copied) annotation provenance, and model-input separation.
Fixtures are small and synthetic: no accepted data is read and no official
corpus is produced.
"""

from __future__ import annotations

import pytest

from representation import sprint24_controlled_data_execution as execution
from representation import sprint24_controlled_data_protocol as v1
from synth.schema import AnomalyFamily
from typing import Any, cast


def test_v1_design_preserved_untouched() -> None:
    assert execution.BASE_PROTOCOL_ID == "S24-CONTROLLED-DATA-v1"
    assert execution.BASE_PROTOCOL_SHA256 == v1.protocol_sha256() == "a5762fc572eaa861190b486d372724fad0fd4c48f8d5ea599a2db8c6803b1493"
    assert v1.N_PLANNED_WEEKS == 26
    assert execution.N_PLANNED_WEEKS == 30
    assert execution.N_PLANNED_MONTHS == 7


def test_execution_freeze_is_deterministic() -> None:
    first = execution.execution_freeze()
    second = execution.execution_freeze()
    assert first["protocol_sha256"] == second["protocol_sha256"]
    first_spec = cast("dict[str, Any]", first["spec"])
    first_cost = cast("dict[str, Any]", first_spec["cost"])
    first_counts = cast("dict[str, Any]", first_cost["planned_counts"])
    assert first_counts["files_per_history"] == 220 + 29 * 40


def test_full_plan_nests_and_meets_bands() -> None:
    used: list[int] = []
    seeds: tuple[int, ...] = v1.PROPOSED_DATA_SEEDS
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, seeds):
        plan = execution.plan_history(str(history_id), int(seed), list(used))
        used.append(int(seed))
        plan_slots = cast("list[dict[str, object]]", plan["slots"])
        plan_checks = cast("list[dict[str, object]]", plan["checks"])
        assert len(plan_slots) == 220 + 29 * 40
        assert all(str(check["status"]) == "PASS" for check in plan_checks)


def test_identity_construction_is_label_blind() -> None:
    execution.assert_execution_construction_label_blind()
    weeks = execution.planned_week_identities("S24C-ITER-0001-CONTROLLED-01")
    months = execution.planned_month_identities("S24C-ITER-0001-CONTROLLED-01")
    assert len(weeks) == execution.N_PLANNED_WEEKS
    assert len(months) == execution.N_PLANNED_MONTHS
    assert v1.planned_reference_identity("S24C-ITER-0001-CONTROLLED-01").endswith("end_le_c1")


def test_quota_shortfall_refuses_without_rescue() -> None:
    with pytest.raises(ValueError, match="REFUSED-SHORTFALL"):
        execution.plan_history(
            v1.NEW_HISTORY_IDS[0], 42401, [], n_weekly=5,
        )


def test_seed_and_lineage_gates_refuse() -> None:
    with pytest.raises(ValueError, match="REFUSED-SEED-COLLISION"):
        execution.plan_history(v1.NEW_HISTORY_IDS[0], 32064, [])
    with pytest.raises(ValueError, match="REFUSED-LINEAGE"):
        execution.plan_history("S18I-ITER-0005-DEVELOPMENT-01", 42401, [])


def test_planned_abnormal_slots_generate_realized_annotations() -> None:
    plan = execution.plan_history(str(v1.NEW_HISTORY_IDS[0]), 42401, [])
    generator = execution.fresh_generator()
    plan_slots = cast("list[dict[str, object]]", plan["slots"])
    abnormal = [slot for slot in plan_slots if str(slot["requested_label"]) == "abnormal"][:4]
    assert abnormal
    for slot in abnormal:
        sample = execution.generate_planned_file(slot, generator)
        annotation = execution.trace_annotation(sample, slot)
        assert sample.anomaly_meta is not None
        assert annotation["family"] == sample.anomaly_meta.family.value
        assert annotation["severity"] == sample.anomaly_meta.severity
        assert annotation["file_label"] == "abnormal" == str(slot["requested_label"])


def test_encoder_inputs_stay_label_free() -> None:
    plan = execution.plan_history(str(v1.NEW_HISTORY_IDS[0]), 42401, [])
    generator = execution.fresh_generator()
    plan_slots = cast("list[dict[str, object]]", plan["slots"])
    slot = next(slot for slot in plan_slots if str(slot["requested_label"]) == "abnormal")
    sample = execution.generate_planned_file(slot, generator)
    execution.verify_encoder_label_free(sample, str(slot["slot_id"]))
    lowered = sample.file_id.lower()
    for token in ("baseline", "elevated", "buffer", "warmup", "low", "arm"):
        assert token not in lowered


def test_non_overwrite_refuses_existing_directory(tmp_path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "sentinel.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED-OVERWRITE"):
        execution.require_fresh_dir(target, purpose="test")


def test_four_slot_reassignment_is_prospective_and_exact() -> None:
    used: list[int] = []
    reassigned: list[str] = []
    for history_id, seed in zip(v1.NEW_HISTORY_IDS, v1.PROPOSED_DATA_SEEDS):
        plan = execution.plan_history(str(history_id), int(seed), list(used))
        used.append(int(seed))
        plan_slots = cast("list[dict[str, object]]", plan["slots"])
        for slot in plan_slots:
            if slot.get("reassigned"):
                assert slot["requested_family_original"] == "missing_event"
                assert slot["requested_family"] == "subtle_drift"
                reassigned.append(str(slot["slot_id"]))
    assert sorted(reassigned) == sorted(
        [
            "S24C-ITER-0001-CONTROLLED-02|W15|038",
            "S24C-ITER-0001-CONTROLLED-04|REFERENCE|163",
            "S24C-ITER-0001-CONTROLLED-05|W15|013",
            "S24C-ITER-0001-CONTROLLED-06|W29|009",
        ]
    )


def test_full_plan_generates_single_shot_without_refusal() -> None:
    plan = execution.plan_history(str(v1.NEW_HISTORY_IDS[0]), 42401, [])
    generator = execution.fresh_generator()
    plan_slots = cast("list[dict[str, object]]", plan["slots"])
    abnormal = [slot for slot in plan_slots if str(slot["requested_label"]) == "abnormal"]
    assert abnormal
    for slot in abnormal:
        sample = execution.generate_planned_file(slot, generator)
        annotation = execution.trace_annotation(sample, slot)
        assert annotation["file_label"] == "abnormal"
