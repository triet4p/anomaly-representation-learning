"""Behavioral tests for the metadata-only context/time support audit.

These tests protect the invariants that make the audit trustworthy rather than
echoing its implementation: label-blind membership, causal windows, dependent
window accounting, zero-support reporting and the frozen output contract. They
run on a small synthetic manifest with the accepted row vocabulary, so they do
not depend on copied accepted data or ignored evidence trees.
"""

from __future__ import annotations

import copy

import pytest

from representation import sprint24_context_audit as audit

DAY = 86_400.0
C1_DAYS = 15.0
SPAN_DAYS = 40.0
HISTORY = "SYNTHETIC-DEV-01"
EDGES = {
    HISTORY: [0.0, 4.0, 8.0, 12.0, C1_DAYS, 20.0, 25.0, 30.0, 35.0, 38.0, SPAN_DAYS],
}
EDGES = {HISTORY: [value * DAY for value in EDGES[HISTORY]]}

#: Fields whose values may legitimately depend on annotations.
POSTHOC_FIELDS = {
    "n_supported_windows_zero_abnormal_files",
    "n_healthy_supported_windows",
    "healthy_window_density_supported",
    "n_supported_windows_covering_failure_events",
    "n_failure_events_covered_unique",
    "failure_event_cover_assignments",
    "failure_event_reuse_factor",
    "n_degradation_onsets_covered_unique",
    "abnormal_member_fraction_supported",
    "n_supported_windows_unknown_label_members",
    "posthoc_healthy_control_present",
}


def _manifest() -> dict[str, object]:
    files: list[dict[str, object]] = []
    maintenance: dict[str, list[list[float]]] = {
        "robot-01": [[10.0 * DAY, 10.5 * DAY]],
        "robot-02": [[20.0 * DAY, 20.5 * DAY]],
    }
    step = 0.2 * DAY
    index = 0
    time = 0.0
    while time < SPAN_DAYS * DAY:
        start = time
        end = time + 0.15 * DAY
        robot = ("robot-01", "robot-02", "robot-03", "robot-08", "robot-09")[index % 5]
        program = {
            "robot-01": "program-01",
            "robot-02": "program-01",
            "robot-03": "program-02",
            "robot-08": "program-02",
            "robot-09": "program-03",
        }[robot]
        last_reset = 0.0
        for span_start, span_end in maintenance.get(robot, []):
            if span_end <= start:
                last_reset = span_end
        files.append(
            {
                "file_id": f"S-op-{index:06d}-{index:08x}",
                "start_time": start,
                "end_time": end,
                "robot_id": robot,
                "program_id": program,
                "last_reset_time": last_reset,
                "is_quarantined": False,
                "is_censored": False,
                "n_valid_patches": 40,
                "file_label": "abnormal" if index % 10 == 7 else "normal",
            }
        )
        index += 1
        time += step
    failure_events = [
        {
            "failure_id": f"fail-{number}",
            "robot_id": robot,
            "failure_time": (18.0 + 4.0 * number) * DAY,
        }
        for number, robot in enumerate(("robot-01", "robot-02", "robot-03"))
    ]
    episodes = [
        {
            "episode_id": f"deg-{number}",
            "kind": "degradation",
            "robot_id": "robot-02",
            "start_time": (17.0 + 5.0 * number) * DAY,
            "end_time": (17.5 + 5.0 * number) * DAY,
        }
        for number in range(3)
    ]
    episodes.append(
        {
            "episode_id": "maint-1",
            "kind": "maintenance",
            "robot_id": "robot-01",
            "start_time": 10.0 * DAY,
            "end_time": 10.5 * DAY,
        }
    )
    return {
        "files": files,
        "maintenance_windows": maintenance,
        "failure_events": failure_events,
        "episodes": episodes,
    }


def _run(manifest: dict[str, object]) -> dict[str, object]:
    return audit.audit_histories({HISTORY: manifest}, EDGES)


def _fleet_row(result: dict[str, object], width: float, stride: float) -> dict[str, object]:
    rows = result["support_rows"]
    assert isinstance(rows, list)
    return next(
        row
        for row in rows
        if row["grouping"] == "fleet"
        and row["width_days"] == width
        and row["stride_days"] == stride
    )


def test_grid_freeze_is_stable_and_admits_only_stride_at_most_width() -> None:
    freeze = audit.grid_freeze()
    assert freeze["grid_sha256"] == audit.grid_sha256()
    assert freeze["grid_sha256"] == audit.grid_freeze()["grid_sha256"]
    cells = [cell for cell in audit.candidate_grid()]
    assert len(cells) == len({cell.candidate_id() for cell in cells})
    assert all(cell.stride_days <= cell.width_days for cell in cells)
    assert {cell.grouping for cell in cells} == set(audit.GROUPINGS)


def test_observable_projection_strips_every_annotation_field() -> None:
    observable = audit.project_observable(_manifest())
    rows = observable["files"]
    assert isinstance(rows, list) and rows
    for row in rows:
        assert set(row) == set(audit.OBSERVABLE_ROW_FIELDS)
        assert not set(row) & set(audit.FORBIDDEN_ROW_FIELDS)
    audit.assert_construction_label_blind()


def test_label_permutation_leaves_membership_and_support_unchanged() -> None:
    manifest = _manifest()
    baseline = _run(manifest)
    permuted = copy.deepcopy(manifest)
    files = permuted["files"]
    assert isinstance(files, list)
    labels = [row["file_label"] for row in files][::-1]
    for row, label in zip(files, labels):
        row["file_label"] = label
    events = permuted["failure_events"]
    assert isinstance(events, list)
    for index, event in enumerate(events):
        event["failure_time"] = events[(index + 1) % len(events)]["failure_time"]
    changed = _run(permuted)

    baseline_rows = baseline["support_rows"]
    changed_rows = changed["support_rows"]
    assert isinstance(baseline_rows, list) and isinstance(changed_rows, list)
    assert len(baseline_rows) == len(changed_rows)
    structural_fields = [
        field for field in audit.SUPPORT_ROW_FIELDS if field not in POSTHOC_FIELDS
    ]
    for before, after in zip(baseline_rows, changed_rows):
        for field in structural_fields:
            assert before[field] == after[field], (field, before["candidate_id"])
    assert any(
        any(before[field] != after[field] for field in POSTHOC_FIELDS)
        for before, after in zip(baseline_rows, changed_rows)
    ), "permuted labels must still be consumed post-hoc"


def test_windows_are_causal_and_reference_stays_before_c1() -> None:
    manifest = _manifest()
    observable = audit.project_observable(manifest)
    meta = audit.prepare_history(HISTORY, observable, C1_DAYS * DAY)
    for context in meta.contexts:
        arrays = audit.prepare_context(meta, context)
        for candidate in audit.candidate_grid():
            if candidate.grouping != context.grouping:
                continue
            membership = audit.build_membership(meta, arrays, candidate)
            ends = {meta.file_ids[index]: meta.ends[index] for index in arrays.ok_index}
            for window in membership.windows:
                assert window.n_eligible == len(window.member_ids)
                for file_id in window.member_ids:
                    assert window.window_start < ends[file_id] <= window.window_end
                if window.n_prior:
                    assert window.n_prior <= len(arrays.ok_ids)
    fleet = next(context for context in meta.contexts if context.grouping == "fleet")
    arrays = audit.prepare_context(meta, fleet)
    prefix_ids = {meta.file_ids[index] for index in arrays.ok_index if meta.ends[index] <= meta.c1}
    assert arrays.prefix_count == len(prefix_ids)
    assert arrays.prefix_digest == audit.sha256_text("\n".join(sorted(prefix_ids)))


def test_overlapping_windows_report_dependence_and_disjoint_stride_does_not() -> None:
    result = _run(_manifest())
    disjoint = _fleet_row(result, 14.0, 14.0)
    overlapping = _fleet_row(result, 14.0, 1.0)
    assert disjoint["mean_adjacent_overlap_fraction"] == 0.0
    assert disjoint["reuse_factor"] == pytest.approx(1.0)
    assert disjoint["effective_independent_windows"] == pytest.approx(
        disjoint["n_supported_windows"], rel=0.15
    )
    assert overlapping["mean_adjacent_overlap_fraction"] > 0.5
    assert overlapping["reuse_factor"] > 5.0
    assert overlapping["effective_independent_windows"] < overlapping["n_windows"]
    assert overlapping["n_windows"] > disjoint["n_windows"]


def test_zero_support_context_is_reported_with_denominators() -> None:
    result = _run(_manifest())
    rows = result["support_rows"]
    assert isinstance(rows, list)
    reserve = [row for row in rows if row["context_id"] == "robot:robot-09"]
    assert reserve, "reserve-identity context must still be reported"
    for row in reserve:
        assert row["n_context_rows"] > 0
        assert row["n_context_eligible"] == 0
        assert row["n_supported_windows"] == 0
        assert row["n_zero_member_windows"] == row["n_windows"]
        assert row["disposition"] == "NO-WINDOW-SUPPORT"
        assert row["excluded_reserve_identity"] > 0
        assert row["posthoc_healthy_control_present"] is False
    assert any(
        row["disposition"] == "NO-WINDOW-SUPPORT" for row in rows
    )


def test_every_output_row_matches_the_frozen_field_spec() -> None:
    result = _run(_manifest())
    rows = result["support_rows"]
    assert isinstance(rows, list) and rows
    for row in rows:
        assert set(row) == set(audit.SUPPORT_ROW_FIELDS)
        assert row["disposition"] in audit.DISPOSITIONS
        for field in audit.SUPPORT_ROW_FIELDS:
            assert field in audit.SUPPORT_FIELD_UNITS
