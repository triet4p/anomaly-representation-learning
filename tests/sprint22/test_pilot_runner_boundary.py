"""Focused minimum-control consumer regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sprint22_pilot_runner as runner


def _allocation(count: int) -> dict[str, object]:
    controls = [{"members": [f"control-{index:02d}"]} for index in range(count)]
    return {"controls": controls, "margins": {"controls": count}}


def test_minimum_control_consumer_preserves_all_48_and_49_controls() -> None:
    for count in (48, 49):
        allocation = _allocation(count)
        consumed = runner.require_minimum_control_allocation(allocation)
        expected_ids = [f"control-{index:02d}" for index in range(count)]

        assert [window["members"][0] for window in consumed["controls"]] == expected_ids
        assert consumed["margins"]["controls"] == count


def test_minimum_control_consumer_rejects_47_controls() -> None:
    with pytest.raises(runner.PilotError) as exc_info:
        runner.require_minimum_control_allocation(_allocation(47))

    assert exc_info.value.code == "CONTROL_CONSTRUCTION_SHORTFALL"
    assert "47 < 48" in str(exc_info.value)
