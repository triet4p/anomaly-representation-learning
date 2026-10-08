"""Consumer regression: stage ledger-Path contract and whole-fresh no-resume.

Guards the Task-69 R06 ledger-Path defect at the consumer boundary plus the
Task-68 candidate-5 whole-fresh cutover:

- Stage callers carry the ledger as a Path from _candidate_paths while the
  event list comes from _load_stage_state (passing the list as a Path raised
  AttributeError before any role write).
- The public dispatcher `dispatch_stage` runs the full load → guard → loop
  control flow (not a private-helper slice); the `--stage` CLI delegates to
  it so both paths are identical.
- The binding is whole-fresh: candidate S18-ITER-0005 seeds 32064–32079,
  strict no-resume on interruption, exact 16-role order/permissions/paths,
  and collision catalog disjointness from all retired seeds.
- A stale host-uv binding pin still refuses.
- The public CLI parser builds without installed package metadata: an eager
  ``--version`` probe inside ``build_parser()`` (Task-69 A06) refused every
  CLI invocation on runtimes whose environment lacks the application
  distribution, including plain ``--help``. ``build_parser()`` must succeed
  with the distribution masked, while ``--version`` alone resolves the real
  installed version (or reports the honest missing-metadata error).
- The pre-contact runtime guard runs the real same-interpreter public CLI
  parser child (``python -m synth.cli --help`` plus the bound origin probe)
  before contact (Task-69 R07 MEDIUM gap); a parser refusal fails closed
  naming the child exit instead of retiring a candidate.

No candidate contact: all ledger/marker state lives under tmp_path, using a
tiny disposable binding whose seed block is disjoint from every bound roster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sprint18_iterative_candidate_v1 as runner


def _tiny_binding() -> dict:
    binding = json.loads(
        (REPO_ROOT / "experiments" / "sprint18-iterative-binding-v1.json")
        .read_text(encoding="utf-8")
    )
    binding = copy.deepcopy(binding)
    binding["candidate_id"] = "S18-ITER-TINY-0001"
    binding["candidate_number"] = 101
    binding["candidate_root_relative"] = (
        "data/generated/sprint18-iterative-v1/S18-ITER-TINY-0001"
    )
    return binding


def _seed_fresh_attempt(root: Path, binding: dict) -> tuple[Path, Path]:
    candidate_root, attempt_root, _ = runner._candidate_paths(binding, root)
    candidate_root.mkdir(parents=True)
    attempt_root.mkdir(parents=True)
    marker = dict(runner._expected_config_fields(binding))
    marker.update(
        {
            "schema_id": "sprint18-iterative-pre-attempt-v1",
            "started_utc": "2026-10-07T02:39:01.402616+00:00",
        }
    )
    (attempt_root / "attempt.json").write_bytes(
        runner.canonical_json(marker) + b"\n"
    )
    _, _, ledger_path = runner._candidate_paths(binding, root)
    runner.append_event(ledger_path, "attempt_started", marker)
    runner.append_event(
        ledger_path,
        "preflight_started",
        {
            "candidate_id": binding["candidate_id"],
            "profile_id": runner.PROFILE,
            "protocol_id": runner.PROTOCOL,
            "in_memory_waveforms_expected": True,
            "persisted_role_roots_expected": False,
            "seed_order": [e["data_seed"] for e in binding["role_binding"]],
            "role_order": [e["role"] for e in binding["role_binding"]],
        },
    )
    return attempt_root, ledger_path


def test_stage_callers_carry_ledger_path_not_event_list(tmp_path: Path) -> None:
    binding = _tiny_binding()
    attempt_root, ledger_path = _seed_fresh_attempt(tmp_path, binding)

    _, dispatch_attempt_root, dispatch_ledger_path = runner._candidate_paths(
        binding, tmp_path
    )
    _, _, events = runner._load_stage_state(binding, tmp_path)
    assert isinstance(dispatch_ledger_path, Path)
    assert isinstance(events, list)
    assert dispatch_attempt_root == attempt_root
    reread = runner.read_ledger(dispatch_ledger_path)
    assert [e["event_type"] for e in reread] == [
        "attempt_started",
        "preflight_started",
    ]

    _, conf_attempt_root, conf_ledger_path = runner._candidate_paths(
        binding, tmp_path
    )
    _, _, conf_events = runner._load_stage_state(binding, tmp_path)
    assert isinstance(conf_ledger_path, Path)
    assert isinstance(conf_events, list)
    assert conf_attempt_root == attempt_root

    with pytest.raises(AttributeError):
        runner.read_ledger(events)  # type: ignore[arg-type]


def test_public_dispatcher_is_not_a_helper_slice(tmp_path: Path) -> None:
    """dispatch_stage must exist, be public, and own the dispatch loop."""
    assert callable(runner.dispatch_stage)
    binding = _tiny_binding()
    with pytest.raises(runner.GuardError, match="unsupported stage"):
        runner.dispatch_stage(binding, tmp_path, "preflight")


def test_fresh_path_stays_strict_on_tampered_marker(tmp_path: Path) -> None:
    binding = _tiny_binding()
    _seed_fresh_attempt(tmp_path, binding)
    _, _, events = runner._load_stage_state(binding, tmp_path)
    assert len(events) == 2

    candidate_root, attempt_root, _ = runner._candidate_paths(binding, tmp_path)
    marker_path = attempt_root / "attempt.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["binding_sha256"] = "0" * 64
    marker_path.write_bytes(runner.canonical_json(marker) + b"\n")
    with pytest.raises(runner.GuardError, match="released binding"):
        runner._load_stage_state(binding, tmp_path)


def test_interrupted_materialization_retires_without_resume(tmp_path: Path) -> None:
    binding = _tiny_binding()
    attempt_root, ledger_path = _seed_fresh_attempt(tmp_path, binding)
    first = binding["role_binding"][0]
    runner.append_event(
        ledger_path,
        "role_materialization_started",
        {
            "candidate_id": binding["candidate_id"],
            "history_id": first["history_id"],
            "role": first["role"],
            "data_seed": first["data_seed"],
            "directory": str(tmp_path / "role-output"),
        },
    )
    _, _, events = runner._load_stage_state(binding, tmp_path)
    with pytest.raises(runner.GuardError, match="no resume"):
        runner._materialize_entries(
            binding,
            tmp_path,
            attempt_root,
            ledger_path,
            events,
            binding["role_binding"][:12],
        )
    assert any(
        e.get("event_type") == "candidate_rejected"
        for e in runner.read_ledger(ledger_path)
    )


def test_host_uv_guard_rejects_stale_binding_pin() -> None:
    binding = _tiny_binding()
    stale = copy.deepcopy(binding)
    stale["runtime"]["uv_version"] = "0.12.20"
    stale["runtime"]["uv_build"] = "2274b80d6"
    with pytest.raises(runner.GuardError, match="observed host uv"):
        runner._validate_host_uv_toolchain(stale)


def test_binding_is_whole_fresh_candidate5() -> None:
    binding = json.loads(
        (REPO_ROOT / "experiments" / "sprint18-iterative-binding-v1.json")
        .read_text(encoding="utf-8")
    )
    assert binding["candidate_id"] == "S18-ITER-0005"
    assert binding["candidate_number"] == 5
    assert binding["contact_authorized"] is False
    assert binding["seed_block"]["seeds"] == list(range(32064, 32080))
    assert [e["data_seed"] for e in binding["role_binding"]] == list(
        range(32064, 32080)
    )
    assert [e["role"] for e in binding["role_binding"]] == (
        ["DESIGN"] * 4 + ["FIT"] * 3 + ["CALIBRATION"]
        + ["DEVELOPMENT"] * 4 + ["CONFIRMATION"] * 4
    )
    assert binding["attempt_policy"]["no_resume"] is True
    assert "resume_rule" not in binding["attempt_policy"]
    catalog = binding["collision_catalog"]
    assert catalog["candidate_seeds"] == list(range(32064, 32080))
    assert catalog["candidate_intersection"] == []
    assert catalog["excluded_unique_count"] == 421
    excluded: set[int] = set()
    for item in catalog["prior_data_seed_ranges"]:
        excluded.update(range(item["first"], item["last"] + 1))
    for item in catalog["prior_model_seed_ranges"]:
        excluded.update(range(item["first"], item["last"] + 1))
    excluded.update(catalog["prior_single_seeds"])
    assert 32032 in excluded and 32047 in excluded
    assert 32048 in excluded and 32063 in excluded
    assert sorted(set(range(32064, 32080)) & excluded) == []
    assert (
        hashlib.sha256(
            runner.canonical_json(sorted(excluded))
        ).hexdigest()
        == catalog["exclusions_sha256"]
    )


def test_release_receipt_rejects_stale_checkpoint_parent() -> None:
    binding = json.loads(
        (REPO_ROOT / "experiments" / "sprint18-iterative-binding-v1.json")
        .read_text(encoding="utf-8")
    )
    stale_release = {
        "schema_id": runner.MAIN_RELEASE_SCHEMA_ID,
        "candidate_id": binding["candidate_id"],
        "binding_sha256": binding["binding_sha256"],
        "binding_file_sha256": "0" * 64,
        "source_closure_sha256": binding["source_closure"]["closure_sha256"],
        "source_identity_scheme": runner.SOURCE_IDENTITY_SCHEME,
        "task67_evidence_path": runner.TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": runner.TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": runner.TASK67_EVIDENCE_REFERENCE,
        "base_commit": "a17bfad5d5c625e2c8745c0d7b6feff4ee928eb4",
        "checkpoint_parent": "a17bfad5d5c625e2c8745c0d7b6feff4ee928eb4",
        "release_scope": "Task69-preflight-and-nonconfirmation",
        "status": "RELEASED_BY_MAIN",
        "evidence_review_verdict": "PASS",
        "actionable_findings": 0,
        "evidence_review_ref": "disposable-regression",
        "checkpoint_commit": "0" * 40,
    }
    with pytest.raises(runner.GuardError, match="base_commit"):
        runner.validate_release_receipt(binding, "0" * 64, stale_release)
    runner._validate_task69_checkpoint_parent(
        "c" * 40,
        "c" * 40,
    )
    with pytest.raises(runner.GuardError, match="does not follow"):
        runner._validate_task69_checkpoint_parent("d" * 40, "c" * 40)


def test_public_cli_parser_needs_no_installed_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_parser() must not probe distribution metadata; only --version may.

    Regression for the Task-69 A06 boundary: an eager
    ``version(_DISTRIBUTION_NAME)`` call inside ``build_parser()`` raised
    ``PackageNotFoundError`` for EVERY CLI invocation (including ``--help``
    and all generation commands) on the deployed runtime whose environment
    has third-party distributions but no application ``dist-info``. The
    parser must construct with the distribution masked, while ``--version``
    alone resolves the real installed version (or, when truly absent, exits
    non-zero naming the missing metadata instead of a fabricated fallback).
    """
    import importlib.metadata

    import synth.cli as cli

    real_version = importlib.metadata.version

    def _masked(name: str, *args: object, **kwargs: object) -> str:
        if name == cli._DISTRIBUTION_NAME:
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name, *args, **kwargs)  # type: ignore[call-arg]

    monkeypatch.setattr(importlib.metadata, "version", _masked)
    # Parser construction is the A06 failure point: it must not probe metadata.
    parser = cli.build_parser()
    args = parser.parse_args(["--output", "dummy"])
    assert args.output.name == "dummy"
    assert "--version" in parser.format_help()
    # Ordinary parsing still works while metadata is absent.
    chronological = parser.parse_args([
        "--chronological", "--profile", "sprint18-iterative-v3",
        "--output", "dummy",
    ])
    assert chronological.profile == "sprint18-iterative-v3"
    assert chronological.output.name == "dummy"

    monkeypatch.undo()
    # With real metadata, --version alone reports the installed version.
    expected = importlib.metadata.version(cli._DISTRIBUTION_NAME)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert expected == "0.1.0"


def test_runtime_guard_exercises_real_cli_entrypoint_before_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_runtime must run the real SAME-interpreter CLI parser child.

    Regression for the Task-69 R07 MEDIUM gap: ``validate_runtime`` passed
    with exit 0 on the deployed runtime even though no CLI invocation could
    succeed there. The guard now runs ``python -m synth.cli --help`` (plus a
    same-interpreter origin probe asserting the child imported the live
    checkout ``src/synth/cli.py`` that owns the guard) as a real subprocess
    before contact; a parser failure must refuse with the child exit named.
    No mocks: the child is the genuine public entrypoint.
    """
    binding = _tiny_binding()
    observed = runner._validate_public_cli_entrypoint(binding, tmp_path)
    expected_cli = (REPO_ROOT / "src" / "synth" / "cli.py").resolve()
    assert Path(observed["cli_module_origin"]).resolve() == expected_cli
    assert observed["cli_help_returncode"] == 0
    assert observed["cli_profile_exposed"] == runner.PROFILE

    real_run = subprocess.run

    def _refusing_run(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        completed = real_run(*args, **kwargs)
        completed.returncode = 1
        completed.stderr = "importlib.metadata.PackageNotFoundError: No package metadata"
        return completed

    monkeypatch.setattr(subprocess, "run", _refusing_run)
    with pytest.raises(runner.GuardError, match="public CLI entrypoint parser refused"):
        runner._validate_public_cli_entrypoint(binding, tmp_path)
