"""Focused Task 7 guards: healthy-row predicate, seeds, support, fail-fast I/O."""

import re
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sprint17_task7_b0 as T7
from synth import events as E


def _row(**over):
    base = {
        "file_id": "f", "file_label": "normal", "is_quarantined": False,
        "is_censored": False, "program_id": "program-01",
        "robot_id": "robot-01", "start_time": 0.0, "end_time": 500.0,
        "member_views": ["test-temporal"],
    }
    base.update(over)
    return base


def test_healthy_row_predicate_excludes_each_violation():
    wins: dict = {}
    assert T7.eligible_healthy_row(_row(), wins, "program-03", "robot-08") is True
    assert T7.eligible_healthy_row(_row(file_label="abnormal"), wins,
                                   "program-03", "robot-08") is False
    assert T7.eligible_healthy_row(_row(is_quarantined=True), wins,
                                   "program-03", "robot-08") is False
    assert T7.eligible_healthy_row(_row(is_censored=True), wins,
                                   "program-03", "robot-08") is False
    assert T7.eligible_healthy_row(_row(program_id="program-03"), wins,
                                   "program-03", "robot-08") is False
    assert T7.eligible_healthy_row(_row(robot_id="robot-08"), wins,
                                   "program-03", "robot-08") is False
    overlap_wins = {"robot-01": [[100.0, 200.0]]}
    assert T7.eligible_healthy_row(_row(), overlap_wins,
                                   "program-03", "robot-08") is False


def test_file_seed_deterministic_bounded():
    assert T7.file_seed("abc") == T7.file_seed("abc")
    assert 0 <= T7.file_seed("abc") < 2 ** 31
    assert T7.file_seed("abc") != T7.file_seed("abd")


def test_canonical_hash_stable_and_hex():
    assert T7.canonical_hash({"b": 1, "a": [1, 2]}) == T7.canonical_hash(
        {"a": [1, 2], "b": 1})
    assert re.fullmatch(r"[0-9a-f]{64}", T7.canonical_hash({"x": None}))


def test_binding_digest_recomputes_to_frozen_value():
    binding = T7.check_binding_digest(
        REPO_ROOT / "experiments" / "sprint17-role-binding-v3.json")
    assert binding["binding_sha256"] == T7.BINDING_SHA256


def test_load_verified_root_refuses_confirmation_and_sealed(tmp_path):
    with pytest.raises(ValueError, match="refusing"):
        T7.load_verified_root(tmp_path, "CONFIRMATION", "H-S17-V3-CONF-01", 2812)
    with pytest.raises(ValueError, match="refusing"):
        T7.load_verified_root(tmp_path, "SEALED", "H-SEAL-37", 1612)


def test_load_verified_root_missing_manifest_is_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="no regeneration"):
        T7.load_verified_root(tmp_path, "FIT", "H-S17-V3-FIT-01", 2804)


def test_evaluable_support_finds_members_and_counts_reset_skip():
    horizon = E.HORIZON_S
    failure_time = 10 * horizon
    rows = [
        _row(file_id="f1", end_time=failure_time - 1000.0,
             start_time=failure_time - 1500.0),
        _row(file_id="f2", end_time=failure_time - 2000.0,
             start_time=failure_time - 2500.0),
    ]
    failure = {
        "cohort": "P", "subtype": "P1", "robot_id": "robot-01",
        "failure_time": failure_time, "severity": 2.0,
        "degradation_onset": failure_time - 100000.0, "duration_d": 5.0,
    }
    manifest = {"failure_events": [dict(failure)]}
    ledger = E.failure_ledger(manifest)
    support = T7.evaluable_support(rows, ledger, {})
    assert len(support["pos"]["P"]) == 1
    assert sorted(support["pos"]["P"][0]["members"]) == ["f1", "f2"]
    assert support["skipped"] == 0
    spanning = {"robot-01": [[failure_time - horizon - 100.0,
                              failure_time - 50.0]]}
    support2 = T7.evaluable_support(rows, ledger, spanning)
    assert support2["skipped"] == 1
    assert support2["pos"]["P"] == []


class _ToyScoringPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(192, 8)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(8)

    def forward(self, batch):
        x = batch["patches"].flatten(2)
        return self.act(self.norm(self.lin(x)))


def test_flop_counter_matches_hand_computed_toy():
    # T=512 patchifies to 31 patches (32/16, pad_end); per-patch row = 192.
    model = _ToyScoringPath().eval()
    flops = T7.count_flops_reference(model)
    n_patches, width, out = 31, 192, 8
    expected = (2 * n_patches * out * width  # Linear MACs x2
                + 8 * n_patches * out  # GELU
                + 5 * n_patches * out)  # LayerNorm
    assert flops == expected == 98456


def test_to_jsonable_preserves_booleans_and_numerics():
    import numpy as np
    assert T7.to_jsonable(True) is True
    assert T7.to_jsonable(False) is False
    assert T7.to_jsonable(np.True_) is True
    assert T7.to_jsonable({"g": True, "v": np.float64(0.5),
                           "n": float("nan")}) == {"g": True, "v": 0.5,
                                                   "n": None}
