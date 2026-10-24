"""Fail-closed validation of fresh execution custody for the immutable Task 33 lock.

The published pre-Confirmation lock (``experiments/sprint18-task33-confirmation-lock.json``)
is immutable: it records the lock-time isolated-worktree identity and it
self-pins the publisher module digest, so editing either byte would invalidate
the accepted lock.  After the authorized Task33 checkpoint and sync, the actual
execution worktree is a *descendant* of that locked identity, not the identity
itself.  The lock-time custody validator in
:mod:`representation.sprint18_confirmation_lock` therefore cannot accept a
post-sync execution-custody report: it compares the reported worktree commit and
tree to the locked historical constants.

This module supplies the missing public path without touching the lock or its
publisher.  It validates an append-only execution-custody report against the
immutable lock and enforces:

* the exact immutable lock bytes and canonical digest;
* the exact locked worktree identity recorded by the lock;
* a tracked-clean execution worktree whose commit is a verified descendant of
  the locked commit;
* a git delta since the locked commit that touches no locked artifact path;
* the reviewed lock/publisher bytes present unchanged in the execution source;
* every one of the locked remote artifact digests reproduced at the execution
  source.

Confirmation support content, labels, features and scores are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from representation.sprint18_confirmation_lock import ConfirmationLockError, lock_sha256

__all__ = [
    "ExecutionCustodyError",
    "LOCK_CANONICAL_SHA256",
    "LOCK_FILE_SHA256",
    "PUBLISHER_PATH",
    "PUBLISHER_SHA256",
    "SCHEMA_ID",
    "verify_execution_custody",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = "experiments/sprint18-task33-confirmation-lock.json"
SCHEMA_ID = "sprint18-execution-custody-v1"
TASK_ID = "S18-T34"
LOCK_FILE_SHA256 = "a846e63365db47929d6979075b46a7837c55205d25c3d2e5c8ff25e2ae64c314"
LOCK_CANONICAL_SHA256 = "d8aeae288ae25fb1fc2aa177530bda48b6b895f184d141b6c6f465a5dc869c3e"
PUBLISHER_PATH = "src/representation/sprint18_confirmation_lock.py"
PUBLISHER_SHA256 = "97f299d6aa800aab691732513bc14006fb3fecf70776b6b5a4db0ccff4d6b3c5"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_REPORT_KEYS = frozenset(
    {
        "schema_id",
        "task_id",
        "attempt_id",
        "agent_id",
        "generator",
        "lock",
        "locked_worktree",
        "execution_worktree",
        "source_lineage",
        "checkpoint_scope",
        "artifacts",
        "verified_artifacts",
        "missing",
        "mismatched",
    }
)


class ExecutionCustodyError(ValueError):
    """Raised when execution custody is incomplete, inconsistent or tampered."""


def _expect(condition: object, message: str) -> None:
    if not condition:
        raise ExecutionCustodyError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_root(repository_root: str | Path | None) -> Path:
    return Path(repository_root).resolve() if repository_root is not None else REPOSITORY_ROOT


def _locked_paths(lock: Mapping[str, Any]) -> set[str]:
    """Return every repository-relative path whose bytes the lock pins."""
    custody = cast(Mapping[str, Any], lock["artifact_custody"])
    worktree = cast(Mapping[str, Any], custody["remote_worktree"])
    prefix = f"{worktree['path']}/"
    paths: set[str] = set(cast(Mapping[str, str], custody["local_artifacts"]))
    for absolute in cast(Mapping[str, str], custody["remote_artifacts"]):
        if absolute.startswith(prefix):
            paths.add(absolute[len(prefix) :])
    return paths


def _validate_changed_paths(
    changes: object,
    locked_paths: set[str],
    verified_additions: frozenset[str],
) -> list[tuple[str, tuple[str, ...]]]:
    _expect(isinstance(changes, list), "source lineage has no changed-path list")
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for entry in cast(list[object], changes):
        _expect(isinstance(entry, dict), "source lineage change is not an object")
        change = cast(Mapping[str, Any], entry)
        status = change.get("status")
        paths = change.get("paths")
        _expect(isinstance(status, str) and status, "source lineage change has no status")
        _expect(
            isinstance(paths, list) and paths and all(isinstance(item, str) and item for item in paths),
            "source lineage change has no path list",
        )
        clean: list[str] = []
        for raw in cast(list[str], paths):
            _expect(
                not raw.startswith("/") and "\\" not in raw and ".." not in raw.split("/"),
                f"source lineage change is not repository-relative: {raw!r}",
            )
            if raw in locked_paths:
                # The authorized Task33 publishing commit itself adds the
                # publisher module.  That single addition is admissible only
                # because the execution bytes are separately verified equal to
                # the reviewed publisher digest.
                _expect(
                    raw in verified_additions and status.strip().upper() in {"A", "M"},
                    f"execution source changed a locked artifact: {raw}",
                )
            clean.append(raw)
        parsed.append((status, tuple(clean)))
    return parsed


def verify_execution_custody(
    *,
    repository_root: str | Path | None = None,
    lock_path: str | Path | None = None,
    report_path: str | Path,
) -> dict[str, Any]:
    """Validate one execution-custody report against the immutable Task 33 lock."""
    root = _resolve_root(repository_root)
    candidate_lock = Path(lock_path) if lock_path is not None else root / LOCK_PATH
    _expect(candidate_lock.is_file(), f"lock file is missing: {candidate_lock}")
    lock_bytes = candidate_lock.read_bytes()
    _expect(
        hashlib.sha256(lock_bytes).hexdigest() == LOCK_FILE_SHA256,
        "lock bytes differ from the reviewed Task 33 lock",
    )
    lock = json.loads(lock_bytes.decode("utf-8"))
    _expect(isinstance(lock, dict), "lock is not a JSON object")
    lock = cast(dict[str, Any], lock)
    _expect(
        lock.get("lock_sha256") == LOCK_CANONICAL_SHA256,
        "lock canonical digest differs from the reviewed Task 33 digest",
    )
    _expect(
        lock_sha256(lock) == LOCK_CANONICAL_SHA256,
        "lock payload does not reproduce the reviewed canonical digest",
    )

    report_candidate = Path(report_path)
    _expect(report_candidate.is_file(), f"custody report is missing: {report_candidate}")
    report = json.loads(report_candidate.read_bytes().decode("utf-8"))
    _expect(isinstance(report, dict), "custody report is not a JSON object")
    report = cast(dict[str, Any], report)
    _expect(set(report) == _REPORT_KEYS, f"custody report keys differ: {sorted(set(report) ^ _REPORT_KEYS)}")
    _expect(report.get("schema_id") == SCHEMA_ID, "custody report schema differs")
    _expect(report.get("task_id") == TASK_ID, "custody report task id differs")
    for field in ("attempt_id", "agent_id", "generator"):
        _expect(
            isinstance(report.get(field), str) and bool(report.get(field)),
            f"custody report has no {field}",
        )

    declared_lock = report.get("lock")
    _expect(isinstance(declared_lock, dict), "custody report has no lock identity")
    declared_lock = cast(Mapping[str, Any], declared_lock)
    _expect(
        declared_lock.get("file_sha256") == LOCK_FILE_SHA256,
        "custody report lock file digest differs",
    )
    _expect(
        declared_lock.get("lock_sha256") == LOCK_CANONICAL_SHA256,
        "custody report lock canonical digest differs",
    )

    custody = cast(Mapping[str, Any], lock["artifact_custody"])
    locked_worktree = cast(Mapping[str, Any], custody["remote_worktree"])
    reported_locked = report.get("locked_worktree")
    _expect(isinstance(reported_locked, dict), "custody report has no locked worktree identity")
    _expect(
        dict(cast(Mapping[str, Any], reported_locked)) == dict(locked_worktree),
        "custody report locked worktree differs from the immutable lock",
    )

    execution = report.get("execution_worktree")
    _expect(isinstance(execution, dict), "custody report has no execution worktree identity")
    execution = cast(Mapping[str, Any], execution)
    _expect(
        execution.get("path") == locked_worktree["path"],
        "execution worktree path differs from the locked worktree",
    )
    _expect(
        execution.get("branch") == locked_worktree["branch"],
        "execution worktree branch differs from the locked branch",
    )
    _expect(
        isinstance(execution.get("commit"), str) and bool(_HEX40.match(cast(str, execution["commit"]))),
        "execution commit is not a 40-hex git object id",
    )
    _expect(
        isinstance(execution.get("tree"), str) and bool(_HEX40.match(cast(str, execution["tree"]))),
        "execution tree is not a 40-hex git object id",
    )

    lineage = report.get("source_lineage")
    _expect(isinstance(lineage, dict), "custody report has no source lineage")
    lineage = cast(Mapping[str, Any], lineage)
    _expect(
        lineage.get("locked_commit") == locked_worktree["commit"],
        "source lineage locked commit differs from the immutable lock",
    )
    _expect(
        lineage.get("locked_commit_is_ancestor") is True,
        "execution commit is not a verified descendant of the locked commit",
    )
    _expect(lineage.get("tracked_clean") is True, "execution worktree is not tracked-clean")

    scope = report.get("checkpoint_scope")
    _expect(isinstance(scope, dict), "custody report has no checkpoint scope")
    scope = cast(Mapping[str, Any], scope)
    _expect(
        scope.get("lock_file_sha256") == LOCK_FILE_SHA256,
        "execution source does not contain the immutable lock bytes",
    )
    _expect(
        scope.get("publisher_sha256") == PUBLISHER_SHA256,
        "execution source does not contain the reviewed publisher bytes",
    )
    verified_additions = frozenset({PUBLISHER_PATH})

    locked_paths = _locked_paths(lock)
    changes = _validate_changed_paths(lineage.get("changed_paths"), locked_paths, verified_additions)

    expected = cast(Mapping[str, str], custody["remote_artifacts"])
    artifacts = report.get("artifacts")
    _expect(isinstance(artifacts, dict), "custody report has no artifact digest map")
    artifacts = cast(Mapping[str, str], artifacts)
    _expect(
        set(artifacts) == set(expected),
        f"custody report artifact set differs: {sorted(set(artifacts) ^ set(expected))[:1]}",
    )
    mismatched = sorted(path for path in expected if artifacts[path] != expected[path])
    _expect(
        not mismatched,
        f"remote custody digest mismatch for {len(mismatched)} artifact(s), first: {mismatched[:1]}",
    )
    _expect(report.get("missing") == [], "custody report records missing artifacts")
    _expect(report.get("mismatched") == [], "custody report records mismatched artifacts")
    _expect(
        report.get("verified_artifacts") == len(expected),
        "custody report verified-artifact count differs",
    )

    return {
        "status": "VALID",
        "lock_file_sha256": LOCK_FILE_SHA256,
        "lock_sha256": LOCK_CANONICAL_SHA256,
        "locked_commit": str(locked_worktree["commit"]),
        "execution_commit": str(execution["commit"]),
        "execution_tree": str(execution["tree"]),
        "changed_paths": sum(len(paths) for _, paths in changes),
        "verified_remote_artifacts": len(expected),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--lock", default=None)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = verify_execution_custody(
            repository_root=args.repository_root,
            lock_path=args.lock,
            report_path=args.report,
        )
    except (ExecutionCustodyError, ConfirmationLockError) as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({**summary, "mode": "execution-custody"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
