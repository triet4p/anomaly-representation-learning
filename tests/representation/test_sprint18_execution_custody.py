"""Fail-closed tests for fresh execution custody of the immutable Task 33 lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from representation import sprint18_execution_custody as C

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "experiments" / "sprint18-task33-confirmation-lock.json"
EXECUTION_COMMIT = "58312e616f1f0ed0dcab7bab2304e4f16fa62807"
EXECUTION_TREE = "e96525dfc0e4db9cb7df633dd3a6ee9386c24981"
CHECKPOINT_CHANGES = [
    {"status": "M", "paths": ["CHANGELOG.md"]},
    {"status": "A", "paths": ["experiments/sprint18-task33-confirmation-lock.json"]},
    {"status": "A", "paths": ["src/representation/sprint18_confirmation_lock.py"]},
    {"status": "A", "paths": ["tests/representation/test_sprint18_confirmation_lock.py"]},
]


@pytest.fixture(scope="module")
def lock() -> dict[str, object]:
    return json.loads(LOCK_PATH.read_bytes().decode("utf-8"))


@pytest.fixture()
def report(lock: dict[str, object]) -> dict[str, object]:
    custody = lock["artifact_custody"]
    assert isinstance(custody, dict)
    locked = dict(custody["remote_worktree"])
    local = custody["local_artifacts"]
    assert isinstance(local, dict)
    return {
        "schema_id": C.SCHEMA_ID,
        "task_id": C.TASK_ID,
        "attempt_id": "S18T34A01",
        "agent_id": "S18T34A01",
        "generator": "artifacts/sprint-18/s18-t34-a01/build_execution_custody.py",
        "lock": {
            "path": "experiments/sprint18-task33-confirmation-lock.json",
            "file_sha256": C.LOCK_FILE_SHA256,
            "lock_sha256": C.LOCK_CANONICAL_SHA256,
            "bytes": len(LOCK_PATH.read_bytes()),
        },
        "locked_worktree": locked,
        "execution_worktree": {
            "path": locked["path"],
            "branch": locked["branch"],
            "commit": EXECUTION_COMMIT,
            "tree": EXECUTION_TREE,
        },
        "source_lineage": {
            "locked_commit": locked["commit"],
            "locked_commit_is_ancestor": True,
            "tracked_clean": True,
            "changed_paths": copy.deepcopy(CHECKPOINT_CHANGES),
        },
        "checkpoint_scope": {
            "lock_file_sha256": C.LOCK_FILE_SHA256,
            "publisher_sha256": C.PUBLISHER_SHA256,
            "tests_sha256": "eb201811c49be40bf90857e7752977032fcdd7a7bad0c416a04ccfb8be9af9c1",
            "changelog_sha256": "30021c4603433ebaea7324ec91e911415bb6e85e6415c3765d6db4215b600e91",
        },
        "artifacts": dict(custody["remote_artifacts"]),
        "verified_artifacts": len(custody["remote_artifacts"]),
        "missing": [],
        "mismatched": [],
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "execution-custody.json"
    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    return path


def _verify(tmp_path: Path, payload: dict[str, object]) -> dict[str, object]:
    return C.verify_execution_custody(report_path=_write(tmp_path, payload))


def test_execution_custody_passes(report: dict[str, object], tmp_path: Path) -> None:
    summary = _verify(tmp_path, report)
    assert summary["status"] == "VALID"
    assert summary["lock_file_sha256"] == C.LOCK_FILE_SHA256
    assert summary["lock_sha256"] == C.LOCK_CANONICAL_SHA256
    assert summary["execution_commit"] == EXECUTION_COMMIT
    assert summary["verified_remote_artifacts"] == 118
    assert summary["changed_paths"] == 4


def test_report_schema_differs(report: dict[str, object], tmp_path: Path) -> None:
    report["schema_id"] = "sprint18-confirmation-lock-custody-v1"
    with pytest.raises(C.ExecutionCustodyError, match="schema differs"):
        _verify(tmp_path, report)


def test_unknown_report_key_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    report["P_AUROC"] = 0.5
    with pytest.raises(C.ExecutionCustodyError, match="keys differ"):
        _verify(tmp_path, report)


def test_altered_lock_bytes_are_rejected(report: dict[str, object], tmp_path: Path) -> None:
    altered = tmp_path / "lock.json"
    original = LOCK_PATH.read_bytes()
    altered.write_bytes(original.replace(b'"phase": "pre_confirmation"', b'"phase": "pre_confirmation" ', 1))
    with pytest.raises(C.ExecutionCustodyError, match="lock bytes differ"):
        C.verify_execution_custody(lock_path=altered, report_path=_write(tmp_path, report))


def test_report_lock_digest_differs(report: dict[str, object], tmp_path: Path) -> None:
    lock_identity = report["lock"]
    assert isinstance(lock_identity, dict)
    lock_identity["file_sha256"] = "0" * 64
    with pytest.raises(C.ExecutionCustodyError, match="lock file digest differs"):
        _verify(tmp_path, report)


def test_locked_worktree_differs(report: dict[str, object], tmp_path: Path) -> None:
    reported = report["locked_worktree"]
    assert isinstance(reported, dict)
    reported["commit"] = EXECUTION_COMMIT
    with pytest.raises(C.ExecutionCustodyError, match="locked worktree differs"):
        _verify(tmp_path, report)


def test_execution_worktree_must_keep_locked_branch(report: dict[str, object], tmp_path: Path) -> None:
    execution = report["execution_worktree"]
    assert isinstance(execution, dict)
    execution["branch"] = "main"
    with pytest.raises(C.ExecutionCustodyError, match="branch differs"):
        _verify(tmp_path, report)


def test_execution_commit_must_be_hex(report: dict[str, object], tmp_path: Path) -> None:
    execution = report["execution_worktree"]
    assert isinstance(execution, dict)
    execution["commit"] = "58312e6"
    with pytest.raises(C.ExecutionCustodyError, match="40-hex"):
        _verify(tmp_path, report)


def test_ancestor_requirement_is_enforced(report: dict[str, object], tmp_path: Path) -> None:
    lineage = report["source_lineage"]
    assert isinstance(lineage, dict)
    lineage["locked_commit_is_ancestor"] = False
    with pytest.raises(C.ExecutionCustodyError, match="not a verified descendant"):
        _verify(tmp_path, report)


def test_tracked_clean_requirement_is_enforced(report: dict[str, object], tmp_path: Path) -> None:
    lineage = report["source_lineage"]
    assert isinstance(lineage, dict)
    lineage["tracked_clean"] = False
    with pytest.raises(C.ExecutionCustodyError, match="tracked-clean"):
        _verify(tmp_path, report)


def test_changed_locked_code_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    lineage = report["source_lineage"]
    assert isinstance(lineage, dict)
    lineage["changed_paths"] = [{"status": "M", "paths": ["src/representation/sprint18_harness.py"]}]
    with pytest.raises(C.ExecutionCustodyError, match="changed a locked artifact"):
        _verify(tmp_path, report)


def test_deleting_the_publisher_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    lineage = report["source_lineage"]
    assert isinstance(lineage, dict)
    lineage["changed_paths"] = [{"status": "D", "paths": [C.PUBLISHER_PATH]}]
    with pytest.raises(C.ExecutionCustodyError, match="changed a locked artifact"):
        _verify(tmp_path, report)


def test_absolute_changed_path_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    lineage = report["source_lineage"]
    assert isinstance(lineage, dict)
    lineage["changed_paths"] = [{"status": "A", "paths": ["/etc/passwd"]}]
    with pytest.raises(C.ExecutionCustodyError, match="repository-relative"):
        _verify(tmp_path, report)


def test_missing_artifact_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts.pop(sorted(artifacts)[0])
    with pytest.raises(C.ExecutionCustodyError, match="artifact set differs"):
        _verify(tmp_path, report)


def test_mismatched_artifact_digest_is_rejected(report: dict[str, object], tmp_path: Path) -> None:
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts[sorted(artifacts)[0]] = "0" * 64
    with pytest.raises(C.ExecutionCustodyError, match="digest mismatch"):
        _verify(tmp_path, report)


def test_publisher_checkpoint_scope_is_enforced(report: dict[str, object], tmp_path: Path) -> None:
    scope = report["checkpoint_scope"]
    assert isinstance(scope, dict)
    scope["publisher_sha256"] = "0" * 64
    with pytest.raises(C.ExecutionCustodyError, match="reviewed publisher bytes"):
        _verify(tmp_path, report)


def test_lock_checkpoint_scope_is_enforced(report: dict[str, object], tmp_path: Path) -> None:
    scope = report["checkpoint_scope"]
    assert isinstance(scope, dict)
    scope["lock_file_sha256"] = "0" * 64
    with pytest.raises(C.ExecutionCustodyError, match="immutable lock bytes"):
        _verify(tmp_path, report)


def test_verified_artifact_count_is_enforced(report: dict[str, object], tmp_path: Path) -> None:
    report["verified_artifacts"] = 117
    with pytest.raises(C.ExecutionCustodyError, match="verified-artifact count"):
        _verify(tmp_path, report)
