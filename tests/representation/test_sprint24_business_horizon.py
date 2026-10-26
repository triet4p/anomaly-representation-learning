"""Behavioral tests for business-horizon period membership and exposure.

These tests protect the invariants the prospective design depends on: the
inherited right-inclusive ``end_time`` assignment, real calendar month lengths
(no nominal 28-day month), partial-period abstention, overlap-weighted exposure,
explicit support/exposure abstention states, annotation absence never becoming
"healthy", and label-blind construction. They run on a small synthetic manifest
with the accepted row vocabulary, so they need no copied accepted data.
"""

from __future__ import annotations

import copy

import pytest

from representation import sprint24_business_horizon as bh

DAY = 86_400.0
ANCHOR = 1_704_067_200.0  # 2024-01-01T00:00:00Z, a Monday
C1_DAYS = 30.0
SPAN_DAYS = 120.0
HISTORY = "SYNTHETIC-BIZ-01"
#: Decile edges with the inherited c1 slot (index 4) pinned to C1_DAYS.
EDGES = {
    HISTORY: [
        0.0,
        10.0 * DAY,
        20.0 * DAY,
        25.0 * DAY,
        C1_DAYS * DAY,
        45.0 * DAY,
        60.0 * DAY,
        75.0 * DAY,
        90.0 * DAY,
        105.0 * DAY,
        SPAN_DAYS * DAY,
    ]
}

#: Fields whose values may legitimately depend on annotations.
PERIOD_POSTHOC_FIELDS = {
    "n_abnormal_members",
    "n_unknown_label_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "annotation_state",
}
PAIR_POSTHOC_FIELDS = {
    "previous_annotation_state",
    "current_annotation_state",
    "n_abnormal_members",
    "abnormal_member_fraction",
    "n_failure_events_covered",
    "n_degradation_onsets_covered",
    "attribution_class",
}


def _calendar(**overrides: object) -> bh.CalendarConfig:
    options: dict[str, object] = {
        "anchor_epoch_s": ANCHOR,
        "timezone": "UTC",
        "week_start": "monday",
        "month_anchor_day": 15,
    }
    options.update(overrides)
    return bh.CalendarConfig(**options)


def _manifest() -> dict[str, object]:
    files: list[dict[str, object]] = []
    robots = ("robot-01", "robot-02", "robot-03")
    programs = {"robot-01": "program-01", "robot-02": "program-02", "robot-03": "program-01"}
    step = 2.0 * 3600.0
    index = 0
    time = 1.9 * 3600.0  # places one daily file across each midnight boundary
    while time < SPAN_DAYS * DAY:
        start = time
        end = time + 600.0
        robot = robots[index % len(robots)]
        files.append(
            {
                "file_id": f"S-op-{index:06d}-{index:08x}",
                "start_time": start,
                "end_time": end,
                "robot_id": robot,
                "program_id": programs[robot],
                "last_reset_time": 0.0,
                "is_quarantined": False,
                "is_censored": False,
                "n_valid_patches": 40,
                "file_label": "abnormal" if index % 8 == 3 else "normal",
            }
        )
        index += 1
        time += step
    return {
        "files": files,
        "maintenance_windows": {},
        "failure_events": [
            {
                "failure_id": f"fail-{number}",
                "robot_id": robots[number % len(robots)],
                "failure_time": (C1_DAYS + 3.0 + 5.0 * number) * DAY,
            }
            for number in range(6)
        ],
        "episodes": [
            {
                "episode_id": f"deg-{number}",
                "kind": "degradation",
                "robot_id": "robot-02",
                "start_time": (C1_DAYS + 4.0 + 5.0 * number) * DAY,
                "end_time": (C1_DAYS + 4.5 + 5.0 * number) * DAY,
            }
            for number in range(4)
        ],
    }


def _evaluate(manifest: dict[str, object], calendar: bh.CalendarConfig | None = None) -> dict[str, object]:
    return bh.evaluate({HISTORY: manifest}, EDGES, calendar or _calendar(), modes=("calendar",))


def _rows(result: dict[str, object], key: str, horizon: str) -> list[dict[str, object]]:
    rows = result[key]
    assert isinstance(rows, list)
    return [row for row in rows if row["horizon"] == horizon]


def _meta():
    manifest = _manifest()
    observable = bh.project_observable(manifest)
    return bh.prepare_history(HISTORY, observable, C1_DAYS * DAY)


def test_period_boundaries_are_right_inclusive_on_end_time() -> None:
    grid = bh.PeriodGrid("day", "calendar", _calendar(), 0.0, 10.0 * DAY)
    boundary = DAY
    assert grid.period_for_end_time(boundary).key == "D2024-01-01"
    assert grid.period_for_end_time(boundary - 1.0).key == "D2024-01-01"
    assert grid.period_for_end_time(boundary + 1.0).key == "D2024-01-02"
    # The rule is end_time in (start, end], so a period's own start instant is
    # owned by the previous period, never by itself.
    period = grid.period_for_end_time(boundary)
    assert period.start < boundary <= period.end
    assert not (grid.period_for_end_time(period.start).key == period.key)


def test_calendar_month_lengths_are_real_and_never_a_nominal_month() -> None:
    calendar = _calendar()
    lengths = {}
    for year, month in ((2024, 2), (2025, 2), (2024, 6), (2024, 7), (2023, 2)):
        probe = calendar.relative(calendar.local_midnight(year, month, 10))
        period = bh.PeriodGrid("month", "calendar", calendar, 0.0, 1.0).period_containing(probe)
        lengths[f"{year:04d}-{month:02d}"] = period.length_days
    assert lengths["2024-02"] == 29.0  # leap February
    assert lengths["2025-02"] == 28.0
    assert lengths["2023-02"] == 28.0
    assert lengths["2024-06"] == 30.0
    assert lengths["2024-07"] == 31.0
    assert set(lengths.values()) != {28.0}


def test_rolling_month_keeps_unequal_inter_anchor_gaps() -> None:
    calendar = _calendar()
    rolling = bh.PeriodGrid("month", "rolling", calendar, 0.0, 1.0)
    lengths = []
    for year, month in ((2024, 1), (2024, 2), (2024, 3), (2024, 6), (2024, 7)):
        anchor = calendar.relative(calendar.local_midnight(year, month, 15))
        lengths.append(rolling.period_containing(anchor + 1.0).length_days)
    assert lengths == [31.0, 29.0, 31.0, 30.0, 31.0]
    assert 28.0 not in lengths
    jan = rolling.period_containing(calendar.relative(calendar.local_midnight(2024, 1, 15)) + 1.0)
    assert jan.start == pytest.approx(14.0 * DAY)
    assert jan.end == pytest.approx(45.0 * DAY)


def test_partial_periods_are_flagged_and_cannot_be_compared() -> None:
    result = _evaluate(_manifest())
    month_rows = _rows(result, "period_rows", "month")
    states = {row["period_key"]: row["coverage_state"] for row in month_rows}
    assert states["M2024-01"] == "PARTIAL-PRE"
    assert states["M2024-02"] == "FULL"
    assert states["M2024-03"] == "FULL"
    assert states["M2024-04"] == "PARTIAL-POST"
    pair_rows = _rows(result, "pair_rows", "month")
    leading = next(row for row in pair_rows if row["previous_key"] == "M2024-01")
    assert leading["comparison_state"] == "PARTIAL-PERIOD"
    assert leading["coverage_state"] == "PAIR-PARTIAL"


def test_exposure_is_overlap_weighted_and_reports_straddlers() -> None:
    meta = _meta()
    grid = bh.PeriodGrid("day", "calendar", _calendar(), meta.c1, meta.tmax)
    records = bh.build_period_records(meta, grid)
    straddling = [record for record in records if record.n_straddling_members > 0]
    assert straddling, "two-hour sampling must produce files straddling day boundaries"
    record = straddling[0]
    expected = 0.0
    for position, file_id in enumerate(record.member_ids):
        start = float(record.member_starts[position])
        end = float(record.member_ends[position])
        expected += (
            min(end, record.period.end) - max(start, record.period.start)
        ) / DAY
        assert record.period.start < end <= record.period.end, file_id
    assert record.exposure_days == pytest.approx(expected)
    assert record.exposure_days < record.n_members * (600.0 / DAY) + 1e-9


def test_support_and_exposure_shortfalls_abstain_with_explicit_states() -> None:
    result = _evaluate(_manifest())
    day_rows = _rows(result, "feasibility_rows", "day")
    assert day_rows[0]["n_supported_periods"] == 0
    assert day_rows[0]["horizon_state"] == "INFEASIBLE"
    assert day_rows[0]["n_pairs_comparable_matched"] == 0
    day_pairs = _rows(result, "pair_rows", "day")
    assert {row["comparison_state"] for row in day_pairs} == {
        "INSUFFICIENT-PERIOD-SUPPORT",
        "PARTIAL-PERIOD",
    }
    week_rows = _rows(result, "feasibility_rows", "week")
    assert week_rows[0]["n_supported_periods"] > 0
    week_pairs = _rows(result, "pair_rows", "week")
    assert any(row["comparison_state"] == "COMPARABLE-MATCHED" for row in week_pairs)
    for row in week_pairs:
        if row["comparison_state"] == "COMPARABLE-MATCHED":
            assert row["exposure_ratio"] >= bh.PROPOSED_EXPOSURE_RATIO_FLOOR
        if row["comparison_state"] == "COMPARABLE-UNMATCHED-EXPOSURE":
            assert row["exposure_ratio"] < bh.PROPOSED_EXPOSURE_RATIO_FLOOR


def test_absent_annotations_never_become_a_healthy_control() -> None:
    manifest = _manifest()
    blank = copy.deepcopy(manifest)
    files = blank["files"]
    assert isinstance(files, list)
    for row in files:
        row["file_label"] = None
    result = _evaluate(blank)
    period_rows = _rows(result, "period_rows", "week")
    assert period_rows
    assert all(row["annotation_state"] == "UNANNOTATED" for row in period_rows)
    assert all(row["n_abnormal_members"] == 0 for row in period_rows)
    assert all(row["annotation_state"] != "ANNOTATED-CLEAN" for row in period_rows)
    matrix = result["horizon_matrix"]
    assert isinstance(matrix, list)
    week = next(row for row in matrix if row["horizon"] == "week")
    assert week["n_supported_pairs_annotated_clean_total"] == 0
    assert week["n_supported_pairs_zero_failure_events_total"] == 0
    assert week["control_state"] == "NO-SUPPORTED-PERIODS" or week["control_state"] == "NO-QUIET-PERIOD-CONTROL"
    assert week["control_state"] != "ANNOTATED-CLEAN-CONTROL-AVAILABLE"
    assert week["control_state"] != "QUIET-PERIOD-CONTROL-AVAILABLE"


def test_fully_annotated_clean_period_is_reported_as_clean() -> None:
    manifest = _manifest()
    clean = copy.deepcopy(manifest)
    files = clean["files"]
    assert isinstance(files, list)
    for row in files:
        row["file_label"] = "normal"
    clean["failure_events"] = []
    clean["episodes"] = []
    result = _evaluate(clean)
    period_rows = _rows(result, "period_rows", "week")
    assert all(row["annotation_state"] == "ANNOTATED-CLEAN" for row in period_rows)
    matrix = result["horizon_matrix"]
    assert isinstance(matrix, list)
    week = next(row for row in matrix if row["horizon"] == "week")
    assert week["n_supported_pairs_annotated_clean_total"] > 0
    assert week["control_state"] == "ANNOTATED-CLEAN-CONTROL-AVAILABLE"


def test_period_and_pair_construction_is_label_blind() -> None:
    bh.assert_period_construction_label_blind()
    manifest = _manifest()
    baseline = _evaluate(manifest)
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
    changed = _evaluate(permuted)
    for key, posthoc in (
        ("period_rows", PERIOD_POSTHOC_FIELDS),
        ("pair_rows", PAIR_POSTHOC_FIELDS),
    ):
        baseline_rows = baseline[key]
        changed_rows = changed[key]
        assert isinstance(baseline_rows, list) and isinstance(changed_rows, list)
        assert len(baseline_rows) == len(changed_rows)
        structural = [
            field for field in baseline_rows[0] if field not in posthoc
        ]
        for before, after in zip(baseline_rows, changed_rows):
            for field in structural:
                assert before[field] == after[field], (key, field, before.get("period_key"))
    baseline_periods = _rows(baseline, "period_rows", "week")
    changed_periods = _rows(changed, "period_rows", "week")
    assert any(
        before["n_abnormal_members"] != after["n_abnormal_members"]
        or before["n_failure_events_covered"] != after["n_failure_events_covered"]
        for before, after in zip(baseline_periods, changed_periods)
    ), "permuted annotations must still be consumed post-hoc"


def test_horizon_matrix_covers_every_horizon_and_reports_units() -> None:
    result = _evaluate(_manifest())
    matrix = result["horizon_matrix"]
    assert isinstance(matrix, list)
    assert {row["horizon"] for row in matrix} == set(bh.HORIZONS)
    for row in matrix:
        assert set(row) == set(bh.HORIZON_MATRIX_FIELDS)
        assert row["horizon_state"] in bh.HORIZON_STATES
        assert row["control_state"] in bh.CONTROL_STATES
    periods = _rows(result, "period_rows", "month")
    for row in periods:
        assert set(row) == set(bh.PERIOD_ROW_FIELDS)
        assert row["coverage_state"] in bh.COVERAGE_STATES
        assert row["support_state"] in bh.PERIOD_SUPPORT_STATES
        for field in bh.PERIOD_ROW_FIELDS:
            assert field in bh.PERIOD_FIELD_UNITS
    pairs = _rows(result, "pair_rows", "week")
    for row in pairs:
        assert set(row) == set(bh.PAIR_ROW_FIELDS)
        assert row["comparison_state"] in bh.PAIR_COMPARISON_STATES
        for field in bh.PAIR_ROW_FIELDS:
            assert field in bh.PAIR_ROW_FIELD_UNITS
    for row in _rows(result, "feasibility_rows", "day"):
        assert set(row) == set(bh.FEASIBILITY_ROW_FIELDS)
        for field in bh.FEASIBILITY_ROW_FIELDS:
            assert field in bh.FEASIBILITY_FIELD_UNITS


def test_proposal_freeze_is_stable_and_declares_proposed_defaults() -> None:
    freeze = bh.proposal_freeze()
    assert freeze["proposal_sha256"] == bh.proposal_sha256()
    assert freeze["proposal_sha256"] == bh.proposal_freeze()["proposal_sha256"]
    spec = freeze["spec"]
    assert isinstance(spec, dict)
    assert spec["proposal_id"] == bh.PROPOSAL_ID
    assert spec["proposed_defaults"]["timezone"] == bh.PROPOSED_TIMEZONE
    assert spec["proposed_defaults"]["week_start"] == bh.PROPOSED_WEEK_START
    assert spec["inherited_unchanged"]["min_period_files"] == 30
    assert spec["inherited_unchanged"]["min_reference_files"] == 200
    assert "month" in spec["horizons"]
    assert {item["id"] for item in spec["endpoints"]} == {"E1", "E2", "E3", "E4"}
