"""Focused behavioral checks for S24-7-M9 official corpus generation/audit.

These tests protect the invariants the M9 CLIs depend on without producing
the official corpus: the frozen execution digest, digest-drift refusal,
non-overwrite refusal on both output paths, per-slot single-shot
realization with zero fallback, and mask-exact annotation provenance on a
bounded real subset. The exhaustive 8,280-file generation/audit evidence
lives in ``artifacts/sprint-24/evidence/s24-7-m9-a01/``, not in this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from representation import sprint24_controlled_data_execution as execution
from representation import sprint24_controlled_data_protocol as v1
from synth.dataset import _save_sample, load_sample

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"


def _run_cli(script: str, *argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXPERIMENTS / script), *argv],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_m9_generation_cli_refuses_existing_output(tmp_path) -> None:
    target = tmp_path / "corpus"
    target.mkdir()
    (target / "sentinel.txt").write_text("x", encoding="utf-8")
    completed = _run_cli(
        "sprint24_task7m9_controlled_data_generation.py", "--out", str(target),
    )
    assert completed.returncode != 0
    assert "REFUSED-OVERWRITE" in (completed.stdout + completed.stderr)


def test_m9_audit_cli_refuses_existing_output(tmp_path) -> None:
    target = tmp_path / "audit"
    target.mkdir()
    (target / "sentinel.txt").write_text("y", encoding="utf-8")
    completed = _run_cli(
        "sprint24_task7m9_controlled_data_audit.py",
        "--corpus", str(tmp_path / "dummy_corpus"),
        "--out", str(target),
    )
    assert completed.returncode != 0
    assert "REFUSED-OVERWRITE" in (completed.stdout + completed.stderr)


def test_m9_audit_records_every_reloaded_file() -> None:
    import json as _json

    audit_path = REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m9-a01" / "corpus-audit.json"
    if not audit_path.is_file():
        pytest.skip("official M9 audit evidence not present in this checkout")
    audit = _json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["execution_sha256"] == execution.execution_freeze()["protocol_sha256"]
    assert audit["realized_files"] == 8280
    assert audit["reloaded_files"] == 8280
    assert audit["n_audit_issues"] == 0
    assert audit["n_pairs"] == 60

def test_m9_frozen_execution_digest_pinned() -> None:
    assert execution.execution_freeze()["protocol_sha256"] == (
        "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"
    )


def test_m9_generation_cli_refuses_digest_drift(tmp_path) -> None:
    target = tmp_path / "fresh"
    completed = _run_cli(
        "sprint24_task7m9_controlled_data_generation.py",
        "--out", str(target),
        "--expected-execution-sha256", "0" * 64,
    )
    assert completed.returncode != 0
    assert "mismatch" in (completed.stdout + completed.stderr).lower() or "drift" in (completed.stdout + completed.stderr).lower()


def test_m9_bounded_subset_realizes_mask_exact(tmp_path) -> None:
    plan = execution.plan_history(str(v1.NEW_HISTORY_IDS[0]), 42401, [])
    slots = cast("list[dict[str, object]]", plan["slots"])
    generator = execution.fresh_generator()
    abnormal = [s for s in slots if str(s["requested_label"]) == "abnormal"][:4]
    normal = [s for s in slots if str(s["requested_label"]) == "normal"][:4]
    assert len(abnormal) == 4 and len(normal) == 4
    for slot in abnormal + normal:
        sample = execution.generate_planned_file(slot, generator)
        annotation = execution.trace_annotation(sample, slot)
        assert annotation["requested_label"] == str(slot["requested_label"])
        target = tmp_path / f"{sample.file_id}.npz"
        _save_sample(sample, target)
        reloaded = load_sample(target)
        reloaded.validate()
        assert reloaded.file_id == sample.file_id
        assert reloaded.seed == sample.seed
        repeat = execution.trace_annotation(reloaded, slot)
        for key in ("family", "severity", "start_step", "end_step",
                    "affected_channels", "duration_steps", "file_label"):
            assert repeat[key] == annotation[key]
