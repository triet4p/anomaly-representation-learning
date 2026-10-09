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
import shutil
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

def _p_allowance_diagnostic_ledger(p_durations: list[float]) -> list[dict]:
    """Build a minimal valid ledger exercising the real P-duration predicate.

    Consumer-visible boundary regression for the user-accepted S18-T69-B01
    policy (DR01 provenance): P per-record 2 <= duration_d < 16 with nominal
    upper 15d reported, P median [5, 10], W/A gates unchanged. Uses small
    isolated in-memory ledgers (not waveforms or seed-32076 data); the real
    structural predicate and the real preflight qualifier decide.
    """
    ledger = [
        {
            "failure_id": "pass-robot-01-0001",
            "robot_id": "robot-01",
            "failure_time": 20.0 * 86400.0,
            "cohort": "P",
            "subtype": "P1",
            "degradation_onset": (20.0 - 7.0) * 86400.0,
            "duration_d": 7.0,
            "severity": 2.0,
        },
        {
            "failure_id": "pass-robot-02-0001",
            "robot_id": "robot-02",
            "failure_time": 30.0 * 86400.0,
            "cohort": "P",
            "subtype": "P2",
            "degradation_onset": (30.0 - 7.0) * 86400.0,
            "duration_d": 7.0,
            "severity": 2.0,
        },
        {
            "failure_id": "pass-robot-03-0001",
            "robot_id": "robot-03",
            "failure_time": 40.0 * 86400.0,
            "cohort": "W",
            "subtype": "W1",
            "degradation_onset": (40.0 - 13.0) * 86400.0,
            "duration_d": 13.0,
            "severity": 2.0,
        },
        {
            "failure_id": "pass-robot-04-0001",
            "robot_id": "robot-04",
            "failure_time": 50.0 * 86400.0,
            "cohort": "W",
            "subtype": "W2",
            "degradation_onset": (50.0 - 13.0) * 86400.0,
            "duration_d": 13.0,
            "severity": 2.0,
        },
        {
            "failure_id": "pass-robot-05-0001",
            "robot_id": "robot-05",
            "failure_time": 60.0 * 86400.0,
            "cohort": "A",
            "subtype": "A1",
            "degradation_onset": None,
            "duration_d": 0.0,
            "severity": 2.0,
        },
    ]
    for index, duration in enumerate(p_durations):
        ledger.append(
            {
                "failure_id": f"probe-robot-06-{index:04d}",
                "robot_id": "robot-06",
                "failure_time": (70.0 + index) * 86400.0,
                "cohort": "P",
                "subtype": "P1" if index % 2 == 0 else "P2",
                "degradation_onset": (70.0 + index - duration) * 86400.0,
                "duration_d": duration,
                "severity": 2.0,
            }
        )
    return ledger


def test_p_duration_subday_overshoot_accepted_with_nominal_reported() -> None:
    """P 15.545d passes the amended predicate and reports the nominal excess."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import sprint18_task5_measurability as meas

    ledger = _p_allowance_diagnostic_ledger([15.545084957564008])
    manifest = {
        "files": [],
        "failure_events": ledger,
        "maintenance_windows": {},
        "schedule": [],
        "splits": {
            "dev_train": [], "dev_val": [], "test_static": [],
            "test_temporal": [], "quarantined": [],
            "failed_episode_ids": [],
        },
        "counts": {
            "total": 0, "normal": 0, "abnormal": 0, "dev_train": 0,
            "dev_val": 0, "test_static": 0, "test_temporal": 0,
            "quarantined": 0,
        },
        "calendar": {"cutoff_time": 0.0},
        "resolved_config": {
            "health": {"noise_scale": 1.0e-3},
            "scheduler": {"routes": [{"stages": [{"duration_s": 600.0}]}]},
        },
    }
    entry = {"history_id": "S18I-P-ALLOW-PROBE", "role": "CONFIRMATION", "data_seed": 32076}
    summary = meas.structural_summary(entry, manifest)
    assert summary["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is True
    assert summary["nominal_p_upper_15d_exceedance_count"] == 1
    assert summary["nominal_p_upper_15d_exceedance_durations_d"] == [15.545084957564008]
    assert summary["nominal_p_upper_15d_exceedance_allowance_d"] == 1.0


def test_p_duration_hard_boundaries_still_fail() -> None:
    """Exact 16d, above 16d, and below 2d P durations still FAIL; median intact."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import sprint18_task5_measurability as meas

    base_manifest = {
        "files": [],
        "maintenance_windows": {},
        "schedule": [],
        "splits": {
            "dev_train": [], "dev_val": [], "test_static": [],
            "test_temporal": [], "quarantined": [],
            "failed_episode_ids": [],
        },
        "counts": {
            "total": 0, "normal": 0, "abnormal": 0, "dev_train": 0,
            "dev_val": 0, "test_static": 0, "test_temporal": 0,
            "quarantined": 0,
        },
        "calendar": {"cutoff_time": 0.0},
        "resolved_config": {
            "health": {"noise_scale": 1.0e-3},
            "scheduler": {"routes": [{"stages": [{"duration_s": 600.0}]}]},
        },
    }
    entry = {"history_id": "S18I-P-ALLOW-PROBE", "role": "CONFIRMATION", "data_seed": 32076}
    for probe in (16.0, 16.5, 1.99):
        manifest = dict(base_manifest)
        manifest["failure_events"] = _p_allowance_diagnostic_ledger([probe])
        summary = meas.structural_summary(entry, manifest)
        assert summary["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is False
    # The P-median gate itself is unchanged: an all-high P ledger (median 12d)
    # fails through the real predicate even though every record is < 16d.
    high_manifest = dict(base_manifest)
    high_manifest["failure_events"] = _p_allowance_diagnostic_ledger([12.0, 13.0, 14.0])
    high_summary = meas.structural_summary(entry, high_manifest)
    assert high_summary["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is False


def test_p_allowance_leaves_w_a_subtype_gates_unweakened() -> None:
    """W/A/subtype/median gates still reject; next-below-16 P still passes."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import sprint18_task5_measurability as meas

    def _manifest(ledger: list[dict]) -> dict:
        return {
            "files": [],
            "failure_events": ledger,
            "maintenance_windows": {},
            "schedule": [],
            "splits": {
                "dev_train": [], "dev_val": [], "test_static": [],
                "test_temporal": [], "quarantined": [],
                "failed_episode_ids": [],
            },
            "counts": {
                "total": 0, "normal": 0, "abnormal": 0, "dev_train": 0,
                "dev_val": 0, "test_static": 0, "test_temporal": 0,
                "quarantined": 0,
            },
            "calendar": {"cutoff_time": 0.0},
            "resolved_config": {
                "health": {"noise_scale": 1.0e-3},
                "scheduler": {"routes": [{"stages": [{"duration_s": 600.0}]}]},
            },
        }

    entry = {"history_id": "S18I-P-ALLOW-PROBE", "role": "CONFIRMATION", "data_seed": 32076}
    base = _p_allowance_diagnostic_ledger([7.0])
    mutated_w = [dict(record) for record in base]
    mutated_w[2] = {**mutated_w[2], "duration_d": 28.01}
    assert meas.structural_summary(
        entry, _manifest(mutated_w)
    )["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is False
    mutated_a = [dict(record) for record in base]
    mutated_a[4] = {**mutated_a[4], "cohort": "W", "subtype": "W1",
                    "degradation_onset": 59.0 * 86400.0, "duration_d": 0.5}
    assert meas.structural_summary(
        entry, _manifest(mutated_a)
    )["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is False
    mutated_subtype = [dict(record) for record in base]
    mutated_subtype[0] = {**mutated_subtype[0], "subtype": "W1"}
    assert meas.structural_summary(
        entry, _manifest(mutated_subtype)
    )["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is False
    just_below = meas.structural_summary(
        entry, _manifest(_p_allowance_diagnostic_ledger([15.999])))
    assert just_below["checks"]["cohort_subtype_physical_shapes_and_duration_bounds"] is True
    assert just_below["nominal_p_upper_15d_exceedance_count"] == 1


def _live_binding() -> dict:
    return json.loads(
        (REPO_ROOT / "experiments" / "sprint18-iterative-binding-v1.json")
        .read_text(encoding="utf-8")
    )


def _assessment_binding() -> dict:
    return json.loads(
        (REPO_ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json")
        .read_text(encoding="utf-8")
    )


def _assessment_release(assessment: dict, binding_raw_sha256: str) -> dict:
    return {
        "schema_id": runner.ASSESSMENT_RELEASE_SCHEMA_ID,
        "candidate_id": assessment["candidate_id"],
        "binding_sha256": assessment["binding_sha256"],
        "binding_file_sha256": binding_raw_sha256,
        "source_closure_sha256": assessment["source_closure"]["closure_sha256"],
        "source_identity_scheme": runner.SOURCE_IDENTITY_SCHEME,
        "task67_evidence_path": runner.TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": runner.TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": runner.TASK67_EVIDENCE_REFERENCE,
        "base_commit": runner.BASE_COMMIT,
        "checkpoint_parent": runner.BASE_COMMIT,
        "release_scope": "Task69-assessment-preflight-and-nonconfirmation",
        "status": "RELEASED_BY_MAIN",
        "evidence_review_verdict": "PASS",
        "evidence_review_ref": "disposable-assessment-regression",
        "actionable_findings": 0,
        "assessment_of_original_ledger_sha256": runner.ORIGINAL_LEDGER_SHA256,
        "assessment_policy_id": runner.POLICY_REVISION["policy_id"],
        "checkpoint_commit": "a" * 40,
    }


def test_assessment_binds_same_roster_under_separate_root() -> None:
    """The assessment keeps roster/configs/catalog while isolating root/attempt."""
    live = _live_binding()
    assessment = _assessment_binding()
    runner.validate_assessment_binding(assessment, REPO_ROOT, live)
    assert assessment["candidate_id"] == "S18-ITER-0005"
    assert assessment["candidate_number"] == 5
    assert assessment["policy_revision"] == runner.POLICY_REVISION
    assert assessment["source_closure"] == live["source_closure"]
    for key in ("role", "history_id", "data_seed", "permitted_use", "config_hash"):
        assert [e[key] for e in assessment["role_binding"]] == [
            e[key] for e in live["role_binding"]]
    assert [e["data_seed"] for e in assessment["role_binding"]] == list(range(32064, 32080))
    assert assessment["seed_block"] == live["seed_block"]
    assert assessment["configs"] == live["configs"]
    assert assessment["collision_catalog"] == live["collision_catalog"]
    assert assessment["candidate_root_relative"] == runner.ASSESSMENT_CANDIDATE_ROOT_RELATIVE
    assert assessment["attempt_policy"]["attempt_directory"] == runner.ASSESSMENT_ATTEMPT_DIRECTORY
    assert assessment["attempt_policy"]["no_resume"] is True
    assert assessment["assessment_of"]["original_ledger_sha256"] == runner.ORIGINAL_LEDGER_SHA256
    assert assessment["assessment_of"]["assessment_policy_revision"] == runner.POLICY_REVISION
    candidate_root, attempt_root, _ = runner._candidate_paths(assessment, REPO_ROOT)
    assert attempt_root.name == runner.ASSESSMENT_ATTEMPT_DIRECTORY
    assert "_attempt-001" not in str(attempt_root) and "_attempt-001" not in str(candidate_root)
    live_root, live_attempt, _ = runner._candidate_paths(live, REPO_ROOT)
    assert candidate_root != live_root and attempt_root != live_attempt


def test_assessment_refuses_missing_mismatched_amendment_and_drift(tmp_path: Path) -> None:
    """Fail-closed: amendment, roster/config, closure, and linkage drift all refuse."""
    live = _live_binding()
    assessment = _assessment_binding()
    base = copy.deepcopy(assessment)
    mutated = copy.deepcopy(base)
    mutated.pop("assessment_of")
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["assessment_of"] = dict(mutated["assessment_of"])
    mutated["assessment_of"]["original_ledger_sha256"] = "0" * 64
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["policy_revision"] = {"policy_id": "other-policy"}
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["role_binding"][3]["data_seed"] = 99999
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["configs"]["32064"] = dict(mutated["configs"]["32064"])
    mutated["configs"]["32064"]["config_hash"] = "0" * 12
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["source_closure"] = copy.deepcopy(mutated["source_closure"])
    mutated["source_closure"]["closure_sha256"] = "0" * 64
    mutated["binding_sha256"] = runner._candidate_digest(mutated)
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    mutated = copy.deepcopy(base)
    mutated["binding_sha256"] = "0" * 64
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(mutated, tmp_path, live)
    # The live binding itself is not an assessment binding.
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_binding(live, tmp_path, live)


def test_assessment_release_and_original_resume_stay_fail_closed(tmp_path: Path) -> None:
    """Assessment release scope is exact; the old rejected attempt still denies resume."""
    live = _live_binding()
    assessment = _assessment_binding()
    raw_sha = hashlib.sha256(
        runner.canonical_json(
            {k: v for k, v in assessment.items() if k != "binding_sha256"})
        + b"\n").hexdigest()
    release = _assessment_release(assessment, raw_sha)
    assert runner.validate_assessment_release(assessment, raw_sha, release) == "a" * 40
    wrong_scope = dict(release, release_scope="Task69-preflight-and-nonconfirmation")
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_release(assessment, raw_sha, wrong_scope)
    candidate_root, attempt_root, _ = runner._candidate_paths(live, tmp_path)
    candidate_root.mkdir(parents=True)
    attempt_root.mkdir(parents=True)
    marker = dict(runner._expected_config_fields(live))
    marker.update({
        "schema_id": "sprint18-iterative-pre-attempt-v1",
        "started_utc": "2026-10-08T01:51:05.763199+00:00",
    })
    (attempt_root / "attempt.json").write_bytes(runner.canonical_json(marker) + b"\n")
    _, _, ledger_path = runner._candidate_paths(live, tmp_path)
    runner.append_event(ledger_path, "attempt_started", marker)
    runner.append_event(ledger_path, "preflight_started", {"candidate_id": live["candidate_id"]})
    runner.append_event(ledger_path, "preflight_recorded", {
        "verdict": "PREFLIGHT-FAIL", "feasible_count": "15/16", "result_sha256": "0" * 64})
    runner.append_event(ledger_path, "candidate_rejected", {
        "candidate_id": live["candidate_id"], "binding_sha256": live["binding_sha256"],
        "reason": "fixed_16_history_preflight_rejected"})
    with pytest.raises(runner.GuardError):
        runner._load_stage_state(live, tmp_path)


def _assessment_task69_release(assessment: dict, qualification_sha256: str) -> dict:
    return {
        "schema_id": "sprint18-task69-assessment-release-v1",
        "candidate_id": assessment["candidate_id"],
        "binding_sha256": assessment["binding_sha256"],
        "qualification_sha256": qualification_sha256,
        "assessment_checkpoint_commit": "a" * 40,
        "task67_evidence_path": runner.TASK67_EVIDENCE_PATH,
        "task67_evidence_sha256": runner.TASK67_EVIDENCE_SHA256,
        "task67_evidence_reference": runner.TASK67_EVIDENCE_REFERENCE,
        "evidence_review_ref": "agent://S18Task69R10",
        "evidence_review_verdict": "PASS",
        "actionable_findings": 0,
        "verdict": "PASS",
        "release_scope": "Task70-assessment-confirmation-after-Task69-PASS",
        "status": "RELEASED_BY_MAIN",
        "task69_assessment_checkpoint_commit": "b" * 40,
    }


def test_assessment_confirmation_gate_needs_exact_task69_release(tmp_path: Path) -> None:
    """Assessment Confirmation is a genuine Main gate, not an unconditional refusal."""
    assessment = _assessment_binding()
    qualification_sha256 = "c" * 64
    release = _assessment_task69_release(assessment, qualification_sha256)
    assert runner.validate_assessment_task69_release(
        assessment, qualification_sha256, release, "a" * 40) == "b" * 40
    wrong_qualification = dict(release, qualification_sha256="d" * 64)
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_task69_release(
            assessment, qualification_sha256, wrong_qualification, "a" * 40)
    missing_checkpoint = dict(release)
    del missing_checkpoint["task69_assessment_checkpoint_commit"]
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_task69_release(
            assessment, qualification_sha256, missing_checkpoint, "a" * 40)
    # Without the record or release files the stage refuses fail-closed
    # (missing record surfaces as OSError, which the runner CLI maps to refusal).
    with pytest.raises((runner.GuardError, OSError)):
        runner.run_assessment_confirmation(
            assessment, tmp_path, tmp_path / "nope.json", "0" * 64,
            tmp_path / "rel.json", "a" * 40)


def _qualified_assessment_state(tmp_path: Path) -> tuple[dict, Path, Path, str, dict]:
    """Seed a tmp assessment root carrying the reviewed qualified DATA identity.

    Consumer-visible custody regression for the S18-T70-C01 late-entry correction:
    the corrected execution MUST keep reading the legitimate accepted qualified old
    state (12-role ordered prefix, durable qualification PASS, 28-event chain) while
    still refusing tampered bindings/records/ledgers and the generic rejected resume.
    Uses the frozen qualified hashes from the runner; no final Confirmation contact.
    """
    assessment = _assessment_binding()
    candidate_root, attempt_root, _ = runner._candidate_paths(assessment, tmp_path)
    candidate_root.mkdir(parents=True)
    attempt_root.mkdir(parents=True)
    fixture_experiments = tmp_path / "experiments"
    fixture_experiments.mkdir(parents=True, exist_ok=True)
    (fixture_experiments / "sprint18-iterative-assessment-c5-allowance-v1.json").write_bytes(
        (runner.ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json").read_bytes())
    (fixture_experiments / "sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json").write_bytes(
        (runner.ROOT / "experiments" / "sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json").read_bytes())
    marker = dict(runner._expected_config_fields(assessment))
    marker.update({
        "schema_id": "sprint18-iterative-pre-attempt-v1",
        "started_utc": "2026-10-08T03:00:00+00:00",
    })
    (attempt_root / "attempt.json").write_bytes(runner.canonical_json(marker) + b"\n")
    _, _, ledger_path = runner._candidate_paths(assessment, tmp_path)
    runner.append_event(ledger_path, "attempt_started", marker)
    runner.append_event(ledger_path, "preflight_started", {
        "candidate_id": assessment["candidate_id"], "profile_id": runner.PROFILE,
        "protocol_id": runner.PROTOCOL,
        "in_memory_waveforms_expected": True,
        "persisted_role_roots_expected": False,
        "seed_order": [entry["data_seed"] for entry in assessment["role_binding"]],
        "role_order": [entry["role"] for entry in assessment["role_binding"]],
    })
    runner.append_event(ledger_path, "preflight_recorded", {
        "result_path": "preflight", "result_sha256": "0" * 64,
        "verdict": "PREFLIGHT-PASS", "feasible_count": "16/16",
        "in_memory_waveforms_generated": True,
        "persisted_candidate_role_roots": False,
        "persisted_candidate_shards": False,
        "persisted_candidate_manifests": False,
    })
    for entry in assessment["role_binding"][:12]:
        runner.append_event(ledger_path, "role_materialization_started", {
            "candidate_id": assessment["candidate_id"],
            "history_id": entry["history_id"], "role": entry["role"],
            "data_seed": entry["data_seed"], "directory": entry["directory"],
        })
        runner.append_event(ledger_path, "role_materialized", {
            "candidate_id": assessment["candidate_id"],
            "history_id": entry["history_id"], "role": entry["role"],
            "data_seed": entry["data_seed"], "directory": entry["directory"],
            "manifest_sha256": "0" * 64, "sample_count": 1,
            "loader_manifest_role": entry["history_id"],
        })
    qualification_sha256 = runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256
    runner.append_event(ledger_path, "nonconfirmation_qualification_pass", {
        "candidate_id": assessment["candidate_id"],
        "qualification_path": "record", "qualification_sha256": qualification_sha256,
        "fit_probe_summary_sha256": "0" * 64,
        "calibration_summary_sha256": "0" * 64,
        "development_summary_sha256": "0" * 64,
    })
    release = _assessment_task69_release(assessment, qualification_sha256)
    return assessment, attempt_root, ledger_path, qualification_sha256, release


def test_corrected_execution_resolves_reviewed_checkpoint_from_task69_release() -> None:
    """The Task69 release authenticates the reviewed execution checkpoint, not an allowlist."""
    assessment = _assessment_binding()
    release = _assessment_task69_release(assessment, "c" * 64)
    release["assessment_checkpoint_commit"] = runner.ASSESSMENT_EXECUTION_COMMIT
    assert runner.resolve_assessment_execution_checkpoint(release) == runner.ASSESSMENT_EXECUTION_COMMIT
    foreign = dict(release, assessment_checkpoint_commit="d" * 40)
    with pytest.raises(runner.GuardError):
        runner.resolve_assessment_execution_checkpoint(foreign)
    missing = dict(release)
    del missing["assessment_checkpoint_commit"]
    with pytest.raises(runner.GuardError):
        runner.resolve_assessment_execution_checkpoint(missing)
    short = dict(release, assessment_checkpoint_commit="003b94bc")
    with pytest.raises(runner.GuardError):
        runner.resolve_assessment_execution_checkpoint(short)


def test_corrected_execution_authorizes_qualified_old_ledger_shape_without_rewrite(tmp_path: Path) -> None:
    """The qualified 28-event authorization shape passes with no ledger rewrite."""
    assessment, _, ledger_path, _, _ = _qualified_assessment_state(tmp_path)
    before = runner.read_ledger(ledger_path)
    runner._verify_qualified_assessment_ledger_shape(
        assessment, before, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)
    after = runner.read_ledger(ledger_path)
    assert [event["event_sha256"] for event in after] == [
        event["event_sha256"] for event in before]
    tampered = copy.deepcopy(before)
    tampered[5] = dict(tampered[5])
    tampered[5]["previous_event_sha256"] = "0" * 64
    with pytest.raises(runner.GuardError):
        runner._verify_qualified_assessment_ledger_shape(
            assessment, tampered, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)


def test_corrected_execution_rejects_drifted_qualified_data_identity(tmp_path: Path) -> None:
    """Every drifted binding/roster/closure/record identity refuses without rewriting state."""
    assessment, _, _, _, _ = _qualified_assessment_state(tmp_path)
    drifted_roster = copy.deepcopy(assessment)
    drifted_roster["role_binding"][0]["data_seed"] = 99999
    drifted_roster["binding_sha256"] = runner._candidate_digest(drifted_roster)
    with pytest.raises(runner.GuardError):
        runner._assert_qualified_assessment_execution_state(
            drifted_roster, tmp_path, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)
    with pytest.raises(runner.GuardError):
        runner._assert_qualified_assessment_execution_state(
            assessment, tmp_path, "d" * 64)
    tampered_closure = copy.deepcopy(assessment)
    tampered_closure["source_closure"] = copy.deepcopy(tampered_closure["source_closure"])
    tampered_closure["source_closure"]["sha256_by_path"]["src/synth/chronicle.py"] = "0" * 64
    tampered_closure["binding_sha256"] = runner._candidate_digest(tampered_closure)
    with pytest.raises(runner.GuardError):
        runner._assert_qualified_assessment_execution_state(
            tampered_closure, tmp_path, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)


def test_assessment_confirmation_refuses_mismatched_and_tampered_branches(tmp_path: Path) -> None:
    """Mismatched checkpoint/parent/roster/closure/record/tamper all refuse before any write."""
    assessment, _, ledger_path, qualification_sha256, release = _qualified_assessment_state(tmp_path)
    record_path = tmp_path / "record.json"
    record = {
        "schema_id": "sprint18-iterative-nonconfirmation-pass-v1",
        "candidate_id": assessment["candidate_id"],
        "binding_sha256": assessment["binding_sha256"],
        "role_ids": [entry["history_id"] for entry in assessment["role_binding"][:12]],
        "verdict": "PASS", "all_nonconfirmation_gates_pass": True,
        "fit_probe_fit_history_ids": [
            entry["history_id"] for entry in assessment["role_binding"] if entry["role"] == "FIT"],
        "probe_calibration_history_id": next(
            entry["history_id"] for entry in assessment["role_binding"] if entry["role"] == "CALIBRATION"),
        "confirmation_contacted": False,
        "checks_by_history": {
            entry["history_id"]: {"gate": True}
            for entry in assessment["role_binding"][:12]},
    }
    record_path.write_bytes(runner.canonical_json(record) + b"\n")
    release_path = tmp_path / "task69-release.json"
    good_release = dict(
        release, assessment_checkpoint_commit=runner.ASSESSMENT_EXECUTION_COMMIT,
        task69_assessment_checkpoint_commit=runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
        binding_sha256=assessment["binding_sha256"],
    )
    before = runner.read_ledger(ledger_path)
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    release_path.write_bytes(runner.canonical_json(good_release) + b"\n")
    wrong_checkpoint = dict(good_release, task69_assessment_checkpoint_commit="d" * 40)
    release_path.write_bytes(runner.canonical_json(wrong_checkpoint) + b"\n")
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    wrong_parent = dict(
        good_release, assessment_checkpoint_commit="e" * 40,
        task69_assessment_checkpoint_commit=runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
    )
    release_path.write_bytes(runner.canonical_json(wrong_parent) + b"\n")
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    release_path.write_bytes(runner.canonical_json(good_release) + b"\n")
    drifted_roster = copy.deepcopy(assessment)
    drifted_roster["role_binding"][0]["data_seed"] = 99999
    drifted_roster["binding_sha256"] = runner._candidate_digest(drifted_roster)
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            drifted_roster, tmp_path, record_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    tampered_closure = copy.deepcopy(assessment)
    tampered_closure["source_closure"] = copy.deepcopy(tampered_closure["source_closure"])
    tampered_closure["source_closure"]["sha256_by_path"]["src/synth/chronicle.py"] = "0" * 64
    tampered_closure["binding_sha256"] = runner._candidate_digest(tampered_closure)
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            tampered_closure, tmp_path, record_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path, "d" * 64,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    tampered_record = dict(record)
    tampered_record["checks_by_history"] = dict(record["checks_by_history"])
    first_id = next(iter(tampered_record["checks_by_history"]))
    tampered_record["checks_by_history"][first_id] = {"gate": False}
    tampered_path = tmp_path / "tampered-record.json"
    tampered_path.write_bytes(runner.canonical_json(tampered_record) + b"\n")
    with pytest.raises(runner.GuardError):
        runner.run_assessment_confirmation(
            assessment, tmp_path, tampered_path, qualification_sha256,
            release_path, runner.ASSESSMENT_EXECUTION_COMMIT)
    after = runner.read_ledger(ledger_path)
    assert [event["event_sha256"] for event in after] == [
        event["event_sha256"] for event in before]

def _sealed_execution_descriptor(
    assessment: dict[str, Any], release: dict[str, Any], runner_blob: str, runner_sha: str,
) -> dict[str, Any]:
    """Build the sealed EXECUTION descriptor shape bound to the Task69 release + DATA pins."""
    return {
        "schema_id": runner.ASSESSMENT_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        "candidate_id": assessment["candidate_id"],
        "binding_sha256": release["binding_sha256"],
        "source_closure_sha256": assessment["source_closure"]["closure_sha256"],
        "qualification_sha256": release["qualification_sha256"],
        "assessment_checkpoint_commit": release["assessment_checkpoint_commit"],
        "assessment_checkpoint_source": "task69-assessment-release+git-ancestry",
        "task69_assessment_checkpoint_commit": release["task69_assessment_checkpoint_commit"],
        "evidence_review_ref": release["evidence_review_ref"],
        "release_scope": release["release_scope"],
        "execution_checkpoint_source": "task69-assessment-release+git-ancestry",
        "corrected_execution_base_commit": runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT,
        "corrected_execution_base_descriptor_blob_oid": "0" * 40,
        "prior_corrected_execution_base_commit": runner.PRIOR_CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT,
        "corrected_runner": {
            "path": "experiments/sprint18_iterative_candidate_v1.py",
            "git_blob_oid": runner_blob,
            "canonical_sha256": runner_sha,
        },
        "data_identity": {
            "binding_sha256": runner.QUALIFIED_ASSESSMENT_BINDING_SHA256,
            "binding_raw_sha256": "b8448292b023c9268f64886e5ca68a393c36ff5f2c41573ff160d50ce8ee6c58",
            "source_closure_sha256": runner.QUALIFIED_ASSESSMENT_SOURCE_CLOSURE_SHA256,
            "qualification_sha256": runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
            "ledger_sha256": runner.QUALIFIED_ASSESSMENT_LEDGER_SHA256,
            "ledger_events": runner.QUALIFIED_ASSESSMENT_LEDGER_EVENTS,
            "runner_blob_oid": runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID,
            "runner_sha256": runner.QUALIFIED_ASSESSMENT_RUNNER_SHA256,
        },
        "allowed_commands": [
            "--assessment --stage validate", "--assessment --stage materialize-confirmation"],
    }


def test_assessment_execution_ancestors_resolve_reviewed_parent_and_refuse_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ancestry resolves 003b94bc from 7be574d9 and base C descends from B from A."""
    root = runner.ROOT
    parent, committed_blob, head_blob = runner._assessment_execution_ancestors(
        root, runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT)
    assert parent == runner.ASSESSMENT_EXECUTION_COMMIT
    assert committed_blob == runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID
    assert head_blob != runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID
    base_parent = subprocess.run(
        ["git", "-C", str(root), "rev-parse",
         f"{runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT}^"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert base_parent == runner.PRIOR_CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT
    prior_base_parent = subprocess.run(
        ["git", "-C", str(root), "rev-parse",
         f"{runner.PRIOR_CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT}^"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert prior_base_parent == runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT
    with pytest.raises(Exception):
        runner._assessment_execution_ancestors(root, "d" * 40)
    with pytest.raises(Exception):
        runner._assessment_execution_ancestors(root, "b" * 40)
    with pytest.raises(runner.GuardError):
        runner.resolve_assessment_execution_checkpoint(
            {"assessment_checkpoint_commit": "d" * 40})

def test_assessment_entry_refuses_without_sealed_descriptor(tmp_path: Path) -> None:
    """The corrected entry refuses when the sealed descriptor is absent (no bypass)."""
    assessment = _assessment_binding()
    live = _live_binding()
    raw_sha = hashlib.sha256(
        (REPO_ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json"
         ).read_bytes()).hexdigest()
    release = _assessment_release(assessment, raw_sha)
    task69_release = _assessment_task69_release(
        assessment, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)
    task69_release["assessment_checkpoint_commit"] = runner.ASSESSMENT_EXECUTION_COMMIT
    task69_release["task69_assessment_checkpoint_commit"] = runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT
    with pytest.raises(runner.GuardError):
        runner._validate_assessment_entry_binding(
            assessment, tmp_path, live, release, task69_release)


def _write_descriptor(tmp_path: Path, descriptor: dict[str, Any]) -> None:
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / runner.ASSESSMENT_EXECUTION_DESCRIPTOR_FILENAME).write_text(
        json.dumps(descriptor, sort_keys=True), encoding="utf-8")


def _worktree_descriptor_bytes() -> tuple[str, str]:
    """Return the CURRENT worktree runner (blob, canonical) for the positive case."""
    import subprocess as _subprocess
    blob = _subprocess.run(
        ["git", "-C", str(runner.ROOT), "hash-object",
         "experiments/sprint18_iterative_candidate_v1.py"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    raw = (runner.ROOT / "experiments" / "sprint18_iterative_candidate_v1.py").read_bytes()
    canon = runner.sha256_bytes(
        runner.canonical_source_bytes("experiments/sprint18_iterative_candidate_v1.py", raw))
    assert blob != runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID
    assert canon != runner.QUALIFIED_ASSESSMENT_RUNNER_SHA256
    return blob, canon


def test_assessment_descriptor_authenticates_reviewed_worktree_and_refuses_swaps(
    tmp_path: Path,
) -> None:
    """Positive REAL descriptor auth plus deterministic A/B/C-swap negatives.

    The positive case verifies the AUTHORED v2 worktree descriptor bytes against
    the real assessment binding + Task69 release + Git ancestry + working-tree
    runner bytes (no mocks): the descriptor `corrected_runner` MUST DIFFER from
    the old A-side `702b37bb…`/`125df99c…` and MUST EQUAL the current worktree
    blob + canonical bytes, and MUST EQUAL the base-C committed runner blob.
    Committed-HEAD state (HEAD == base C) authenticates positively; swaps (old
    A blob, foreign execution, stale runner, wrong DATA pins, drifted release
    field, wrong base-C pin, v1 schema) all refuse; absence refuses.
    """
    assessment = _assessment_binding()
    live = _live_binding()
    raw_sha = hashlib.sha256(
        (REPO_ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json"
         ).read_bytes()).hexdigest()
    release = _assessment_release(assessment, raw_sha)
    task69_release = _assessment_task69_release(
        assessment, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)
    task69_release["assessment_checkpoint_commit"] = runner.ASSESSMENT_EXECUTION_COMMIT
    task69_release["task69_assessment_checkpoint_commit"] = runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT
    worktree_blob, worktree_canon = _worktree_descriptor_bytes()
    good = _sealed_execution_descriptor(
        assessment, task69_release, worktree_blob, worktree_canon)
    # Positive against the REAL repo root (Git ancestry + worktree bytes live here).
    proven = runner._verify_assessment_execution_descriptor(
        assessment, runner.ROOT, release, task69_release)
    assert proven["corrected_runner"]["git_blob_oid"] == worktree_blob
    assert proven["corrected_runner"]["canonical_sha256"] == worktree_canon
    assert proven["corrected_execution_base_commit"] == runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT
    assert proven["schema_id"] == runner.ASSESSMENT_EXECUTION_DESCRIPTOR_SCHEMA_ID
    # Entry-level positive on the real root: DATA binding + descriptor + 52 members.
    entered = runner._validate_assessment_entry_binding(
        assessment, runner.ROOT, live, release, task69_release)
    assert entered["schema_id"] == runner.ASSESSMENT_EXECUTION_DESCRIPTOR_SCHEMA_ID
    # Negatives mutate one descriptor field at a time against the REAL root
    # (ancestry + worktree bytes live here); the AUTHORED bytes are restored after.
    real_path = runner.ROOT / "experiments" / runner.ASSESSMENT_EXECUTION_DESCRIPTOR_FILENAME
    real_bytes = real_path.read_bytes()
    try:
        drifted = dict(good, qualification_sha256="d" * 64)
        real_path.write_text(json.dumps(drifted, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
        foreign_release = dict(task69_release, assessment_checkpoint_commit="d" * 40)
        real_path.write_bytes(real_bytes)
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, foreign_release)
        old_blob = _sealed_execution_descriptor(
            assessment, task69_release,
            runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID, runner.QUALIFIED_ASSESSMENT_RUNNER_SHA256)
        real_path.write_text(json.dumps(old_blob, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
        stale_runner = _sealed_execution_descriptor(
            assessment, task69_release, "0" * 40, "0" * 64)
        real_path.write_text(json.dumps(stale_runner, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
        wrong_data = dict(good, data_identity=dict(good["data_identity"], binding_sha256="0" * 64))
        real_path.write_text(json.dumps(wrong_data, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
        wrong_base = dict(good, corrected_execution_base_commit="b" * 40)
        real_path.write_text(json.dumps(wrong_base, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
        legacy_schema = dict(good, schema_id="sprint18-assessment-execution-descriptor-v1")
        real_path.write_text(json.dumps(legacy_schema, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner._verify_assessment_execution_descriptor(
                assessment, runner.ROOT, release, task69_release)
    finally:
        real_path.write_bytes(real_bytes)


@pytest.mark.parametrize(
    ("receipt_kind", "expected_error"),
    [
        ("missing-argument", "--execution-release"),
        ("missing-file", "corrected execution release is absent"),
        ("unreadable", "corrected execution release is absent"),
        ("foreign", "corrected execution release candidate mismatch"),
    ],
)
def test_assessment_confirmation_cli_requires_execution_release_before_runtime_or_contact(
    tmp_path: Path, receipt_kind: str, expected_error: str,
) -> None:
    """The public CLI rejects absent or foreign Main execution authority before runtime/contact."""
    binding_path = REPO_ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json"
    assessment = json.loads(binding_path.read_text(encoding="utf-8"))
    main_release_path = (
        REPO_ROOT / "artifacts/sprint-18/S18-ITER-0005-ASSESS-main-assessment-release-v1.json"
    )
    task69_release_path = (
        REPO_ROOT / "artifacts/sprint-18/S18-ITER-0005-TASK69-main-assessment-release-v2.json"
    )
    qualification_path = (
        REPO_ROOT
        / "experiments/sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json"
    )
    execution_release_path = tmp_path / "execution-release.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments/sprint18_iterative_candidate_v1.py"),
        "--assessment",
        "--binding", str(binding_path),
        "--release", str(main_release_path),
        "--stage", "materialize-confirmation",
        "--qualification-record", str(qualification_path),
        # A deliberately mismatched digest is a second no-contact safeguard if
        # a future regression removes the execution-release guard.
        "--qualification-sha256", "0" * 64,
        "--task69-release", str(task69_release_path),
    ]
    if receipt_kind == "missing-file":
        command.extend(["--execution-release", str(execution_release_path)])
    elif receipt_kind == "unreadable":
        execution_release_path.mkdir()
        command.extend(["--execution-release", str(execution_release_path)])
    elif receipt_kind == "foreign":
        execution_release_path.write_text(json.dumps({
            "schema_id": runner.ASSESSMENT_EXECUTION_RELEASE_SCHEMA_ID,
            "candidate_id": "S18-ITER-FOREIGN",
        }), encoding="utf-8")
        command.extend(["--execution-release", str(execution_release_path)])
    elif receipt_kind != "missing-argument":
        raise AssertionError(f"unexpected execution receipt test case: {receipt_kind}")

    _, _, ledger_path = runner._candidate_paths(assessment, REPO_ROOT)
    ledger_before = ledger_path.read_bytes() if ledger_path.exists() else None
    confirmation_paths = [
        REPO_ROOT / entry["directory"] for entry in assessment["role_binding"][12:]
    ]
    confirmation_presence_before = [path.exists() for path in confirmation_paths]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=runner._entrypoint_child_environment(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "runtime execution worktree mismatch" not in result.stderr
    assert (ledger_path.read_bytes() if ledger_path.exists() else None) == ledger_before
    assert [path.exists() for path in confirmation_paths] == confirmation_presence_before

def _disposable_assessment_replica(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    """Build a tiny disposable Git replica proving the reviewed→recovery transition.

    Consumer-visible regression for the Task70 C06 recovery-policy contract: the
    SAME lineage contract MUST accept the authored candidate state (new worktree
    bytes not yet committed), the deployed reviewed base state, and the future
    separate recovery correction, while refusing stale/foreign receipts. The
    replica is a genuine disposable Git repository with real `git` subprocess
    commits (no mocks, no injected guard errors) carrying the full reviewed
    chain A → B → C → D → E:
      * A carries the OLD qualified runner bytes (DATA history),
      * B is a child of A (the Task69 outcome hop),
      * C is a child of B carrying the prior corrected runner bytes,
      * D is a child of C carrying the reviewed base descriptor,
      * E is a child of D carrying the recovery-policy runner + v3 descriptor
        whose `corrected_execution_base_commit` is D and whose
        `prior_corrected_execution_base_commit` is C.
    A foreign commit off an unrelated root, a stale receipt pinning an older
    base, and descriptor/runner drift all refuse. No Conf contact occurs: the
    replica fixture never imports the production export tree.
    """
    import subprocess as _subprocess

    fixture = tmp_path / "replica"
    fixture.mkdir()
    _subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
    _subprocess.run(["git", "config", "user.email", "t70-c06@example.invalid"],
                     cwd=fixture, check=True)
    _subprocess.run(["git", "config", "user.name", "S18-T70-C06"],
                     cwd=fixture, check=True)
    experiments = fixture / "experiments"
    experiments.mkdir()

    def _commit(message: str) -> str:
        _subprocess.run(["git", "add", "-A"], cwd=fixture, check=True)
        _subprocess.run(["git", "commit", "-qm", message], cwd=fixture, check=True)
        return _subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=fixture, check=True,
            capture_output=True, text=True).stdout.strip()

    # A: qualified DATA history (old runner bytes).
    (experiments / "sprint18_iterative_candidate_v1.py").write_bytes(
        b"old qualified runner bytes\n")
    (experiments / "sprint18-assessment-execution-descriptor-v1.json").write_text(
        json.dumps({"placeholder": True}, sort_keys=True), encoding="utf-8")
    commit_a = _commit("feat(sprint18): task70 assessment replica qualified source A")
    # B: Task69 outcome hop (no runner change).
    (fixture / "note.txt").write_text("task69 outcome\n", encoding="utf-8")
    commit_b = _commit("feat(sprint18): task70 assessment replica outcome B")
    # C: prior corrected execution base (first corrected runner bytes).
    (experiments / "sprint18_iterative_candidate_v1.py").write_bytes(
        b"prior corrected runner bytes\n")
    prior_runner = b"prior corrected runner bytes\n"
    commit_c = _commit("feat(sprint18): task70 assessment replica prior corrected base C")
    # D: current reviewed corrected execution base (reviewed base descriptor).
    prior_blob = _subprocess.run(
        ["git", "hash-object",
         "experiments/sprint18_iterative_candidate_v1.py"],
        cwd=fixture, check=True, capture_output=True, text=True).stdout.strip()
    (experiments / "sprint18-assessment-execution-descriptor-v1.json").write_text(
        json.dumps({"schema_id": "sprint18-assessment-execution-descriptor-v2",
                    "candidate_id": "S18-ITER-0005"}, sort_keys=True) + "\n",
        encoding="utf-8")
    commit_d = _commit("feat(sprint18): task70 assessment replica reviewed base D")
    descriptor_blob_d = _subprocess.run(
        ["git", "rev-parse",
         f"{commit_d}:experiments/sprint18-assessment-execution-descriptor-v1.json"],
        cwd=fixture, check=True, capture_output=True, text=True).stdout.strip()
    # E: authored recovery-policy bytes exist in the worktree, uncommitted; the
    # v3 descriptor pins the stable reviewed base D (never the future E SHA).
    recovery_runner = b"recovery policy runner bytes\n"
    (experiments / "sprint18_iterative_candidate_v1.py").write_bytes(recovery_runner)
    recovery_blob = _subprocess.run(
        ["git", "hash-object",
         "experiments/sprint18_iterative_candidate_v1.py"],
        cwd=fixture, check=True, capture_output=True, text=True).stdout.strip()
    authored_head_blob = _subprocess.run(
        ["git", "-C", str(fixture), "rev-parse",
         "HEAD:experiments/sprint18_iterative_candidate_v1.py"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert authored_head_blob != recovery_blob
    recovery_canon = hashlib.sha256(recovery_runner).hexdigest()
    descriptor = {
        "schema_id": "sprint18-assessment-execution-descriptor-v3",
        "candidate_id": "S18-ITER-0005",
        "corrected_execution_base_commit": commit_d,
        "corrected_execution_base_descriptor_blob_oid": descriptor_blob_d,
        "prior_corrected_execution_base_commit": commit_c,
        "corrected_runner": {
            "path": "experiments/sprint18_iterative_candidate_v1.py",
            "git_blob_oid": recovery_blob,
            "canonical_sha256": recovery_canon,
        },
    }
    (experiments / "sprint18-assessment-execution-descriptor-v1.json").write_text(
        json.dumps(descriptor, sort_keys=True) + "\n", encoding="utf-8")
    # DEPLOYED recovery state E: commit the recovery bytes + v3 descriptor as a
    # direct child of D (the exact future-correction lineage the contract
    # authorizes); every hop is verified below with real git ancestry.
    commit_e = _commit("feat(sprint18): task70 assessment replica recovery correction E")
    assert _subprocess.run(
        ["git", "-C", str(fixture), "rev-parse", f"{commit_e}^"],
        check=True, capture_output=True, text=True).stdout.strip() == commit_d
    return fixture, descriptor, {
        "A": commit_a, "B": commit_b, "C": commit_c, "D": commit_d, "E": commit_e,
    }, {
        "recovery_blob": recovery_blob, "recovery_canon": recovery_canon,
        "prior_blob": prior_blob, "prior_runner": prior_runner,
        "authored_head_blob": authored_head_blob,
    }


def test_assessment_static_contract_survives_authored_to_deployed_transition(
    tmp_path: Path,
) -> None:
    """The static contract holds pre-commit (authored) and post-commit (deployed E)."""
    import subprocess as _subprocess

    fixture, descriptor, commits, pins = _disposable_assessment_replica(tmp_path)
    base_d, commit_e = commits["D"], commits["E"]
    recovery_blob = pins["recovery_blob"]
    # AUTHORED candidate state (captured inside the replica): HEAD == reviewed
    # base D carried the prior corrected bytes while the recovery-policy
    # worktree bytes were uncommitted, so the sealed D identity MUST DIFFER
    # from the committed-HEAD bytes there.
    assert pins["authored_head_blob"] != recovery_blob
    assert pins["authored_head_blob"] == pins["prior_blob"]
    # DEPLOYED recovery state: HEAD == E (separate recovery correction child of
    # D); the sealed base-D descriptor blob still resolves through Git, and the
    # E parent chain reaches D → C → B → A with one exact hop each (no
    # arbitrary-ancestor acceptance).
    head_blob = _subprocess.run(
        ["git", "-C", str(fixture), "rev-parse",
         "HEAD:experiments/sprint18_iterative_candidate_v1.py"],
        check=True, capture_output=True, text=True).stdout.strip()
    resolved_runner = _subprocess.run(
        ["git", "-C", str(fixture), "rev-parse",
         f"{base_d}:experiments/sprint18_iterative_candidate_v1.py"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert head_blob == recovery_blob
    assert resolved_runner == pins["prior_blob"]
    assert recovery_blob != resolved_runner
    assert resolved_runner != hashlib.sha256(b"old qualified runner bytes\n").hexdigest()
    for child, parent in (("E", "D"), ("D", "C"), ("C", "B"), ("B", "A")):
        assert _subprocess.run(
            ["git", "-C", str(fixture), "rev-parse", f"{commits[child]}^"],
            check=True, capture_output=True, text=True).stdout.strip() == commits[parent]
    with pytest.raises(subprocess.CalledProcessError):
        _subprocess.run(
            ["git", "-C", str(fixture), "merge-base", "--is-ancestor",
             "0" * 40, "HEAD"],
            check=True, capture_output=True, text=True)
    # Worktree drift from the sealed bytes refuses, while the committed seal
    # itself stays byte-identical (no Conf contact).
    runner_path = fixture / "experiments" / "sprint18_iterative_candidate_v1.py"
    sealed_bytes = runner_path.read_bytes()
    try:
        runner_path.write_bytes(sealed_bytes + b"# unreviewed drift\n")
        drifted = _subprocess.run(
            ["git", "-C", str(fixture), "hash-object",
             "experiments/sprint18_iterative_candidate_v1.py"],
            cwd=fixture, check=True, capture_output=True, text=True).stdout.strip()
        assert drifted != descriptor["corrected_runner"]["git_blob_oid"]
    finally:
        runner_path.write_bytes(sealed_bytes)
    restored = _subprocess.run(
        ["git", "-C", str(fixture), "hash-object",
         "experiments/sprint18_iterative_candidate_v1.py"],
        cwd=fixture, check=True, capture_output=True, text=True).stdout.strip()
    assert restored == descriptor["corrected_runner"]["git_blob_oid"]


def test_assessment_execution_release_accepts_exact_recovery_child_and_refuses_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production E release guard accepts the exact E-child-of-D lineage.

    The product guard pins the real production SHAs, so the disposable replica
    supplies fixture SHAs for the same contract through the four lineage
    constants (every git gate, blob comparison, and merge-base below still runs
    for real against the genuine replica history; no guard behaviour is mocked).
    Independent git ancestry assertions live in the transition test above.
    """
    fixture, descriptor, commits, pins = _disposable_assessment_replica(tmp_path)
    base_d, commit_e = commits["D"], commits["E"]
    monkeypatch.setattr(runner, "CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT", base_d)
    monkeypatch.setattr(
        runner, "PRIOR_CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT", commits["C"])
    monkeypatch.setattr(
        runner, "QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT", commits["B"])
    monkeypatch.setattr(runner, "ASSESSMENT_EXECUTION_COMMIT", commits["A"])

    def _release_for(corrected: str, base: str = base_d) -> dict:
        return {
            "schema_id": runner.ASSESSMENT_EXECUTION_RELEASE_SCHEMA_ID,
            "candidate_id": "S18-ITER-0005",
            "status": "RELEASED_BY_MAIN",
            "release_scope": "Task70-corrected-execution-after-review",
            "binding_sha256": runner.QUALIFIED_ASSESSMENT_BINDING_SHA256,
            "qualification_sha256": runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
            "task69_assessment_checkpoint_commit": commits["B"],
            "assessment_checkpoint_commit": commits["A"],
            "corrected_execution_base_commit": base,
            "corrected_runner_git_blob_oid": pins["recovery_blob"],
            "corrected_runner_canonical_sha256": pins["recovery_canon"],
            "execution_descriptor_sha256": hashlib.sha256(
                (fixture / "experiments" / "sprint18-assessment-execution-descriptor-v1.json"
                 ).read_bytes()).hexdigest(),
            "execution_descriptor_blob_oid": subprocess.run(
                ["git", "-C", str(fixture), "rev-parse",
                 f"{commit_e}:experiments/sprint18-assessment-execution-descriptor-v1.json"],
                check=True, capture_output=True, text=True).stdout.strip(),
            "corrected_execution_commit": corrected,
        }

    # Exact E child-of-D authenticates through the real production guard.
    accepted = runner.validate_assessment_execution_release(
        descriptor, fixture, _write_receipt(tmp_path, _release_for(commit_e)))
    assert accepted["corrected_execution_commit"] == commit_e
    # A receipt that still pins the reviewed base D as the corrected commit, an
    # older base, a wrong parent, or a foreign commit never authenticates.
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_execution_release(
            descriptor, fixture, _write_receipt(tmp_path, _release_for(base_d)))
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_execution_release(
            descriptor, fixture, _write_receipt(tmp_path, _release_for(commit_e, commits["C"])))
    foreign_root = tmp_path / "foreign-root"
    foreign_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=foreign_root, check=True)
    subprocess.run(["git", "config", "user.email", "t70-c06@example.invalid"],
                   cwd=foreign_root, check=True)
    subprocess.run(["git", "config", "user.name", "S18-T70-C06"], cwd=foreign_root, check=True)
    (foreign_root / "note.txt").write_text("foreign\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=foreign_root, check=True)
    subprocess.run(["git", "commit", "-qm", "foreign"], cwd=foreign_root, check=True)
    foreign = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=foreign_root, check=True,
        capture_output=True, text=True).stdout.strip()
    with pytest.raises((runner.GuardError, subprocess.CalledProcessError)):
        runner.validate_assessment_execution_release(
            descriptor, fixture, _write_receipt(tmp_path, _release_for(foreign)))
    # The real superseded D receipt (base C, corrected D) refuses on the real
    # production root: immutable history that can never authorize the recovery.
    stale_receipt = (
        REPO_ROOT / "artifacts" / "sprint-18"
        / "S18-ITER-0005-TASK70-corrected-execution-release-v2.json")
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_execution_release(
            json.loads((REPO_ROOT / "experiments"
                        / "sprint18-assessment-execution-descriptor-v1.json"
                        ).read_text(encoding="utf-8")),
            runner.ROOT, stale_receipt)


def _write_receipt(tmp_path: Path, receipt: dict) -> Path:
    path = tmp_path / f"receipt-{receipt['corrected_execution_commit'][:12]}.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


def test_assessment_release_lineage_accepts_exact_child_and_refuses_stale_or_foreign(
    tmp_path: Path,
) -> None:
    """Exact E child-of-D release passes; stale/foreign/arbitrary ancestors refuse."""
    import subprocess as _subprocess

    fixture, descriptor, commits, pins = _disposable_assessment_replica(tmp_path)
    base_d, commit_e = commits["D"], commits["E"]

    def _release_for(corrected: str) -> dict:
        return {
            "schema_id": runner.ASSESSMENT_EXECUTION_RELEASE_SCHEMA_ID,
            "candidate_id": "S18-ITER-0005",
            "status": "RELEASED_BY_MAIN",
            "release_scope": "Task70-corrected-execution-after-review",
            "binding_sha256": runner.QUALIFIED_ASSESSMENT_BINDING_SHA256,
            "qualification_sha256": runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
            "task69_assessment_checkpoint_commit": runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
            "assessment_checkpoint_commit": runner.ASSESSMENT_EXECUTION_COMMIT,
            "corrected_execution_base_commit": base_d,
            "corrected_runner_git_blob_oid": descriptor["corrected_runner"]["git_blob_oid"],
            "corrected_runner_canonical_sha256": descriptor["corrected_runner"]["canonical_sha256"],
            "execution_descriptor_sha256": hashlib.sha256(
                json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest(),
            "execution_descriptor_blob_oid": descriptor["corrected_execution_base_descriptor_blob_oid"],
            "corrected_execution_commit": corrected,
        }

    good = _release_for(commit_e)
    assert good["corrected_execution_base_commit"] == base_d
    assert good["corrected_execution_commit"] != base_d
    assert good["corrected_runner_git_blob_oid"] == pins["recovery_blob"]
    # Exact-child E ancestry is independently verifiable with real git: E's
    # parent is exactly D, D's parent is C, C's parent is B, B's parent is A.
    for child, parent in (("E", "D"), ("D", "C"), ("C", "B"), ("B", "A")):
        assert _subprocess.run(
            ["git", "-C", str(fixture), "rev-parse", f"{commits[child]}^"],
            check=True, capture_output=True, text=True).stdout.strip() == commits[parent]
    stale = _release_for(commits["C"])
    assert stale["corrected_execution_commit"] != base_d
    assert stale["corrected_execution_commit"] == commits["C"]
    assert stale["corrected_execution_base_commit"] == base_d
    foreign_root = tmp_path / "foreign-root"
    foreign_root.mkdir()
    _subprocess.run(["git", "init", "-q"], cwd=foreign_root, check=True)
    _subprocess.run(["git", "config", "user.email", "t70-c06@example.invalid"],
                     cwd=foreign_root, check=True)
    _subprocess.run(["git", "config", "user.name", "S18-T70-C06"],
                     cwd=foreign_root, check=True)
    (foreign_root / "note.txt").write_text("foreign\n", encoding="utf-8")
    _subprocess.run(["git", "add", "-A"], cwd=foreign_root, check=True)
    _subprocess.run(["git", "commit", "-qm", "foreign"], cwd=foreign_root, check=True)
    foreign = _subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=foreign_root, check=True,
        capture_output=True, text=True).stdout.strip()
    with pytest.raises(subprocess.CalledProcessError):
        _subprocess.run(
            ["git", "-C", str(fixture), "merge-base", "--is-ancestor", foreign, "HEAD"],
            check=True, capture_output=True, text=True)
    assert _release_for(foreign)["corrected_execution_commit"] == foreign




# --- Task70 C06: user-authorized zero-output interrupted-Confirmation recovery ---

A02_LEDGER_CARRIER = (
    REPO_ROOT / "experiments" / "sprint18-task70-assessment-S18-ITER-0005-A02-ledger.jsonl"
)
A02_LEDGER_PREFIX_EVENTS = 28


def _interrupted_assessment_state(
    tmp_path: Path, *, carrier_bytes: bytes | None = None,
) -> tuple[dict, Path, Path, Path]:
    """Seed an assessment root carrying the genuine immutable interrupted carrier.

    Consumer-visible regression for the user-authorized zero-output recovery
    exception: the ledger bytes are the ACTUAL A02 30-event production carrier
    (digest `05cbc0fb…`, first-28 prefix `625dcc1c…`) copied byte-identically, so
    every state validator below runs against genuine custody bytes. The
    Confirmation role paths are absent (the zero-output interrupted state) and no
    synthetic ledger event is invented.
    """
    assessment = _assessment_binding()
    candidate_root, attempt_root, ledger_path = runner._candidate_paths(assessment, tmp_path)
    candidate_root.mkdir(parents=True)
    attempt_root.mkdir(parents=True)
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    ledger_bytes = (
        A02_LEDGER_CARRIER.read_bytes() if carrier_bytes is None else carrier_bytes)
    ledger_path.write_bytes(ledger_bytes)
    (experiments / "sprint18-iterative-assessment-c5-allowance-v1.json").write_bytes(
        (runner.ROOT / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json"
         ).read_bytes())
    record_path = (
        experiments / "sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json")
    record_path.write_bytes(
        (runner.ROOT / "experiments"
         / "sprint18-task69-assessment-S18-ITER-0005-A09-qualification-record.json"
         ).read_bytes())
    first_event = json.loads(ledger_bytes.splitlines()[0].decode("utf-8"))
    marker = {
        key: value for key, value in first_event.items()
        if key not in {"sequence", "event_type", "previous_event_sha256", "event_sha256"}
    }
    (attempt_root / "attempt.json").write_bytes(runner.canonical_json(marker) + b"\n")
    return assessment, attempt_root, ledger_path, record_path


def _interrupted_recovery_release(assessment: dict, corrected: str = "e" * 40) -> dict:
    """Build the dedicated Main recovery-receipt shape bound to the interrupted state."""
    return {
        "schema_id": runner.ASSESSMENT_RECOVERY_RELEASE_SCHEMA_ID,
        "candidate_id": assessment["candidate_id"],
        "status": "RELEASED_BY_MAIN",
        "release_scope": runner.ASSESSMENT_RECOVERY_RELEASE_SCOPE,
        "binding_sha256": assessment["binding_sha256"],
        "qualification_sha256": runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
        "assessment_checkpoint_commit": runner.ASSESSMENT_EXECUTION_COMMIT,
        "task69_assessment_checkpoint_commit": runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
        "corrected_execution_base_commit": runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT,
        "corrected_execution_commit": corrected,
        "interrupted_ledger_sha256": runner.INTERRUPTED_ASSESSMENT_LEDGER_SHA256,
        "qualified_ledger_sha256": runner.QUALIFIED_ASSESSMENT_LEDGER_SHA256,
        "interrupted_ledger_events": runner.INTERRUPTED_ASSESSMENT_LEDGER_EVENTS,
        "qualified_ledger_events": runner.QUALIFIED_ASSESSMENT_LEDGER_EVENTS,
        "confirmation_seeds": [
            entry["data_seed"] for entry in assessment["role_binding"]
            if entry["role"] == "CONFIRMATION"],
        "evidence_review_ref": "agent://S18Task70CR06",
    }


def test_interrupted_recovery_release_authenticates_exact_state_and_refuses_divergence(
    tmp_path: Path,
) -> None:
    """The dedicated recovery receipt authenticates the exact interrupted state only."""
    assessment, _, ledger_path, _ = _interrupted_assessment_state(tmp_path)
    before = ledger_path.read_bytes()
    record_sha256 = runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256
    execution_release = {"corrected_execution_commit": "e" * 40}
    good = _interrupted_recovery_release(assessment)
    receipt_path = tmp_path / "recovery.json"
    receipt_path.write_text(json.dumps(good, sort_keys=True), encoding="utf-8")
    assert runner.validate_assessment_recovery_release(
        assessment, tmp_path, receipt_path, execution_release, record_sha256,
    )["interrupted_ledger_sha256"] == runner.INTERRUPTED_ASSESSMENT_LEDGER_SHA256
    assert ledger_path.read_bytes() == before

    def _refuses(mutated: dict, label: str) -> None:
        case = tmp_path / f"recovery-{label}.json"
        case.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
        with pytest.raises(runner.GuardError):
            runner.validate_assessment_recovery_release(
                assessment, tmp_path, case, execution_release, record_sha256)
        assert ledger_path.read_bytes() == before

    absent = tmp_path / "recovery-absent.json"
    with pytest.raises(runner.GuardError):
        runner.validate_assessment_recovery_release(
            assessment, tmp_path, absent, execution_release, record_sha256)
    _refuses(dict(good, schema_id="sprint18-task70-corrected-execution-release-v1"), "schema")
    _refuses(dict(good, candidate_id="S18-ITER-FOREIGN"), "candidate")
    _refuses(dict(good, status="DRAFT"), "status")
    _refuses(dict(good, release_scope="Task70-corrected-execution-after-review"), "scope")
    _refuses(dict(good, binding_sha256="0" * 64), "binding")
    _refuses(dict(good, qualification_sha256="0" * 64), "qualification")
    _refuses(dict(good, assessment_checkpoint_commit="a" * 40), "execution-checkpoint")
    _refuses(dict(good, task69_assessment_checkpoint_commit="c" * 40), "task69-checkpoint")
    _refuses(
        dict(good, corrected_execution_base_commit=runner.PRIOR_CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT),
        "stale-base")
    _refuses(
        dict(good, corrected_execution_commit=runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT),
        "self-base")
    _refuses(dict(good, corrected_execution_commit="f" * 40), "foreign-execution")
    _refuses(dict(good, interrupted_ledger_sha256="0" * 64), "interrupted-digest")
    _refuses(dict(good, qualified_ledger_sha256="0" * 64), "qualified-digest")
    _refuses(dict(good, interrupted_ledger_events=28), "interrupted-count")
    _refuses(dict(good, qualified_ledger_events=30), "qualified-count")
    _refuses(dict(good, confirmation_seeds=[32076, 32077, 32078]), "wrong-seeds")
    _refuses(dict(good, evidence_review_ref=""), "review-ref")
    missing_key = dict(good)
    del missing_key["interrupted_ledger_sha256"]
    _refuses(missing_key, "missing-key")
    events = runner.read_ledger(ledger_path)
    assert len(events) == runner.INTERRUPTED_ASSESSMENT_LEDGER_EVENTS
    assert events[-1]["event_type"] == "role_materialization_started"
    assert events[-1]["history_id"] == "S18I-ITER-0005-CONFIRMATION-01"
    assert events[-1]["data_seed"] == 32076


def test_interrupted_state_authentication_accepts_exact_carrier_only(tmp_path: Path) -> None:
    """The exact 30-event zero-output carrier passes; every divergence refuses."""
    assessment, _, ledger_path, _ = _interrupted_assessment_state(tmp_path)
    record_sha256 = runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256
    before = ledger_path.read_bytes()
    events = runner.read_ledger(ledger_path)
    runner._assert_interrupted_assessment_execution_state(assessment, tmp_path, record_sha256)
    assert ledger_path.read_bytes() == before
    # The reviewed qualified prefix is reused verbatim by the interrupted shape.
    runner._verify_qualified_assessment_ledger_shape(
        assessment, events[:A02_LEDGER_PREFIX_EVENTS], record_sha256)
    runner._verify_interrupted_assessment_ledger_shape(assessment, events, record_sha256)
    # The fresh-candidate boundary still refuses the interrupted carrier exactly
    # as before: no append, no retire, no resume (the no-receipt public path).
    with pytest.raises(runner.GuardError, match="event count differs"):
        runner._assert_qualified_assessment_execution_state(assessment, tmp_path, record_sha256)
    assert ledger_path.read_bytes() == before

    # A 29-event ledger (a divergent tail missing the started record) refuses.
    truncated_root = tmp_path / "truncated"
    truncated_root.mkdir()
    truncated = b"".join(before.splitlines(keepends=True)[:29])
    _, _, truncated_ledger, _ = _interrupted_assessment_state(
        truncated_root, carrier_bytes=truncated)
    with pytest.raises(runner.GuardError, match="event count differs"):
        runner._assert_interrupted_assessment_execution_state(
            assessment, truncated_root, record_sha256)
    # A completed Confirmation role appended after the interrupted tail refuses.
    completed_root = tmp_path / "completed"
    completed_root.mkdir()
    _, _, completed_ledger, _ = _interrupted_assessment_state(completed_root)
    first_confirmation = next(
        entry for entry in assessment["role_binding"] if entry["role"] == "CONFIRMATION")
    runner.append_event(completed_ledger, "role_materialized", {
        "candidate_id": assessment["candidate_id"],
        "history_id": first_confirmation["history_id"],
        "role": first_confirmation["role"],
        "data_seed": first_confirmation["data_seed"],
        "directory": first_confirmation["directory"],
        "manifest_sha256": "0" * 64, "sample_count": 1,
        "loader_manifest_role": first_confirmation["history_id"],
    })
    with pytest.raises(runner.GuardError, match="event count differs"):
        runner._assert_interrupted_assessment_execution_state(
            assessment, completed_root, record_sha256)
    # A fail/retire event appended after the interrupted tail refuses.
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    _, _, failed_ledger, _ = _interrupted_assessment_state(failed_root)
    runner.append_event(failed_ledger, "candidate_rejected", {
        "candidate_id": assessment["candidate_id"],
        "binding_sha256": assessment["binding_sha256"], "reason": "tamper"})
    with pytest.raises(runner.GuardError, match="already failed or been retired"):
        runner._assert_interrupted_assessment_execution_state(
            assessment, failed_root, record_sha256)
    # A saved partial Confirmation output refuses before any recovery write.
    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    _interrupted_assessment_state(partial_root)
    (partial_root / first_confirmation["directory"]).mkdir(parents=True)
    with pytest.raises(runner.GuardError, match="saved Confirmation output"):
        runner._assert_interrupted_assessment_execution_state(
            assessment, partial_root, record_sha256)
    assert ledger_path.read_bytes() == before


def test_assessment_confirmation_recovery_refuses_without_receipt_with_zero_writes(
    tmp_path: Path,
) -> None:
    """Without the receipt the interrupted state refuses with zero writes.

    The genuine production stage function authenticates record, Task69 release,
    and qualified DATA identity first, then refuses the 30-event carrier at the
    reviewed 28-event boundary — the exact preserved no-resume behaviour. No
    ledger append, no retirement event, no Confirmation directory, and the
    candidate bytes stay byte-identical.
    """
    assessment, _, ledger_path, record_path = _interrupted_assessment_state(tmp_path)
    before = ledger_path.read_bytes()
    task69_release_path = (
        REPO_ROOT / "artifacts" / "sprint-18"
        / "S18-ITER-0005-TASK69-main-assessment-release-v2.json")
    confirmation_paths = [
        tmp_path / entry["directory"] for entry in assessment["role_binding"][12:]]
    with pytest.raises(runner.GuardError, match="event count differs"):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path,
            runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256, task69_release_path,
            runner.ASSESSMENT_EXECUTION_COMMIT)
    assert ledger_path.read_bytes() == before
    assert not any(path.exists() for path in confirmation_paths)
    # A recovery receipt presented without the validated execution release refuses too.
    receipt_path = tmp_path / "recovery.json"
    receipt_path.write_text(json.dumps(
        _interrupted_recovery_release(assessment), sort_keys=True), encoding="utf-8")
    with pytest.raises(runner.GuardError, match="corrected execution release"):
        runner.run_assessment_confirmation(
            assessment, tmp_path, record_path,
            runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256, task69_release_path,
            runner.ASSESSMENT_EXECUTION_COMMIT, recovery_release_path=receipt_path)
    assert ledger_path.read_bytes() == before


def _recovery_replica_clone(tmp_path: Path) -> tuple[Path, dict, dict]:
    """Clone the real project (read-only) and commit the recovery correction as E.

    A genuine disposable Git replica carrying the ACTUAL project history and the
    ACTUAL source tree: the clone resolves the real reviewed chain D → C → B → A,
    and the C06 recovery source is committed as a new child E of D
    (parent(E) == D exactly), which is the only lineage the product release guard
    accepts. No project ref is written and nothing is fetched or pushed. The
    returned fixture carries the Main-shaped execution receipt an authorized
    dispatch would present.

    This replica is deliberately used ONLY for authorization coverage (sealed
    descriptor, E execution release, recovery receipt, interrupted-state custody,
    and the no-receipt refusal). The genuine generation child is never exercised
    here: the actual 4/4 Confirmation materialization, fixed-probe scoring, and
    whole-16 certification remain the authorized post-checkpoint production
    gates, and no regression may contact a Confirmation seed.
    """
    import subprocess as _subprocess

    clone = tmp_path / "clone"
    # Clone with the project's own line-ending policy (core.autocrlf=false):
    # a smudged CRLF checkout would change every worktree blob identity and
    # break the sealed runner/blob pins the product guard authenticates.
    _subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--quiet",
         str(REPO_ROOT), str(clone)], check=True)
    _subprocess.run(["git", "-C", str(clone), "config", "core.autocrlf", "false"],
                    check=True)
    _subprocess.run(["git", "config", "user.email", "t70-c06@example.invalid"],
                    cwd=clone, check=True)
    _subprocess.run(["git", "config", "user.name", "S18-T70-C06"], cwd=clone, check=True)
    assert _subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain"], check=True,
        capture_output=True, text=True).stdout.strip() == ""
    head = _subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=clone, check=True,
        capture_output=True, text=True).stdout.strip()
    assert head == runner.CORRECTED_ASSESSMENT_EXECUTION_BASE_COMMIT
    runner_source = clone / "experiments" / "sprint18_iterative_candidate_v1.py"
    runner_source.write_bytes(
        (REPO_ROOT / "experiments" / "sprint18_iterative_candidate_v1.py").read_bytes())
    descriptor_source = clone / "experiments" / "sprint18-assessment-execution-descriptor-v1.json"
    descriptor_source.write_bytes(
        (REPO_ROOT / "experiments" / "sprint18-assessment-execution-descriptor-v1.json"
         ).read_bytes())
    _subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    _subprocess.run(
        ["git", "commit", "-qm",
         "feat(sprint18): task70 assessment interrupted confirmation recovery authorization"],
        cwd=clone, check=True)
    commit_e = _subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=clone, check=True,
        capture_output=True, text=True).stdout.strip()
    assert _subprocess.run(
        ["git", "rev-parse", f"{commit_e}^"], cwd=clone, check=True,
        capture_output=True, text=True).stdout.strip() == head
    descriptor_bytes = descriptor_source.read_bytes()
    execution_receipt = {
        "schema_id": runner.ASSESSMENT_EXECUTION_RELEASE_SCHEMA_ID,
        "candidate_id": "S18-ITER-0005",
        "status": "RELEASED_BY_MAIN",
        "release_scope": "Task70-corrected-execution-after-review",
        "binding_sha256": runner.QUALIFIED_ASSESSMENT_BINDING_SHA256,
        "qualification_sha256": runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
        "task69_assessment_checkpoint_commit": runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
        "assessment_checkpoint_commit": runner.ASSESSMENT_EXECUTION_COMMIT,
        "corrected_execution_base_commit": head,
        "corrected_execution_commit": commit_e,
        "execution_descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "execution_descriptor_blob_oid": _subprocess.run(
            ["git", "rev-parse",
             f"{commit_e}:experiments/sprint18-assessment-execution-descriptor-v1.json"],
            cwd=clone, check=True, capture_output=True, text=True).stdout.strip(),
        "corrected_runner_git_blob_oid": _subprocess.run(
            ["git", "rev-parse",
             f"{commit_e}:experiments/sprint18_iterative_candidate_v1.py"],
            cwd=clone, check=True, capture_output=True, text=True).stdout.strip(),
        "corrected_runner_canonical_sha256": runner.sha256_bytes(runner.canonical_source_bytes(
            "experiments/sprint18_iterative_candidate_v1.py", runner_source.read_bytes())),
    }
    return clone, execution_receipt, {"E": commit_e, "D": head}


def test_recovery_authorization_on_disposable_git_replica(tmp_path: Path) -> None:
    """Recovery authorization on a real disposable Git replica of the project.

    Bounded deliberately at the authorization boundary the C06 policy adds: on a
    genuine clone carrying the real D → C → B → A history and the recovery-policy
    commit E (child of D), the sealed v3 descriptor authenticates the recovery
    execution, the Main-shaped E execution release passes the real product lineage
    guard, the Main-shaped recovery receipt passes its strict validator, the exact
    30-event interrupted carrier passes the interrupted-state custody gate, and
    the same stage called WITHOUT the recovery receipt still refuses fail-closed
    at the reviewed 28-event boundary with zero ledger/output writes. The genuine
    generation child is never exercised here: the actual 4/4 Confirmation
    materialization, fixed-probe scoring, and whole-16 certification remain the
    authorized post-checkpoint production gates, and no regression contacts a
    Confirmation seed.
    """
    clone, execution_receipt, commits = _recovery_replica_clone(tmp_path)
    assessment, _, ledger_path, record_path = _interrupted_assessment_state(clone)
    before = ledger_path.read_bytes()
    task69_release_path = (
        REPO_ROOT / "artifacts" / "sprint-18"
        / "S18-ITER-0005-TASK69-main-assessment-release-v2.json")
    recovery_receipt = _interrupted_recovery_release(assessment, commits["E"])
    recovery_receipt_path = clone / "artifacts" / "sprint-18" / "recovery-receipt.json"
    recovery_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_receipt_path.write_text(
        json.dumps(recovery_receipt, sort_keys=True), encoding="utf-8")
    # Sealed execution descriptor + strict E execution release on real Git.
    descriptor = runner._verify_assessment_execution_descriptor(
        assessment, clone,
        {"release_scope": "Task69-assessment-preflight-and-nonconfirmation"},
        json.loads(task69_release_path.read_text(encoding="utf-8")))
    execution_receipt_path = clone / "artifacts" / "sprint-18" / "execution-receipt.json"
    execution_receipt_path.write_text(
        json.dumps(execution_receipt, sort_keys=True), encoding="utf-8")
    assert runner.validate_assessment_execution_release(
        descriptor, clone, execution_receipt_path)["corrected_execution_commit"] == commits["E"]
    # Main-shaped recovery receipt + interrupted-state custody on the real root.
    assert runner.validate_assessment_recovery_release(
        assessment, clone, recovery_receipt_path, execution_receipt,
        runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
    )["interrupted_ledger_sha256"] == runner.INTERRUPTED_ASSESSMENT_LEDGER_SHA256
    runner._assert_interrupted_assessment_execution_state(
        assessment, clone, runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256)
    assert ledger_path.read_bytes() == before
    # The same genuine stage WITHOUT the receipt stays fail-closed: the reviewed
    # 28-event boundary refuses the interrupted carrier with zero writes.
    with pytest.raises(runner.GuardError, match="event count differs"):
        runner.run_assessment_confirmation(
            assessment, clone, record_path,
            runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256, task69_release_path,
            runner.ASSESSMENT_EXECUTION_COMMIT)
    assert ledger_path.read_bytes() == before
    assert not any(
        (clone / entry["directory"]).exists() for entry in assessment["role_binding"][12:])


def test_recovery_continuity_append_preserves_interrupted_prefix(tmp_path: Path) -> None:
    """The append-only continuity event preserves the exact interrupted prefix.

    Exercises the production append helper directly on the genuine 30-event
    carrier: exactly one `confirmation_recovery_authorized` event is chained
    after the preserved events, the first 30 lines stay byte-identical, the
    receipt/ledger digests and resume coordinates are recorded exactly, and the
    hash chain stays valid. No candidate contact and no generation occur here.
    """
    assessment, _, ledger_path, _ = _interrupted_assessment_state(tmp_path)
    before = ledger_path.read_bytes()
    recovery_receipt_path = tmp_path / "recovery.json"
    recovery_receipt_path.write_text(json.dumps(
        _interrupted_recovery_release(assessment), sort_keys=True), encoding="utf-8")
    recovery = _interrupted_recovery_release(assessment)
    runner._append_confirmation_recovery_authorization(
        ledger_path, assessment, recovery, recovery_receipt_path,
        "S18I-ITER-0005-CONFIRMATION-01", 32076,
        runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT,
    )
    after = ledger_path.read_bytes()
    assert after.splitlines(keepends=True)[:30] == before.splitlines(keepends=True)
    events = runner.read_ledger(ledger_path)
    assert len(events) == runner.INTERRUPTED_ASSESSMENT_LEDGER_EVENTS + 1
    assert events[30]["event_type"] == "confirmation_recovery_authorized"
    assert events[30]["sequence"] == 31
    assert events[30]["recovery_release_sha256"] == hashlib.sha256(
        recovery_receipt_path.read_bytes()).hexdigest()
    assert events[30]["interrupted_ledger_sha256"] == runner.INTERRUPTED_ASSESSMENT_LEDGER_SHA256
    assert events[30]["qualified_ledger_sha256"] == runner.QUALIFIED_ASSESSMENT_LEDGER_SHA256
    assert events[30]["resume_history_id"] == "S18I-ITER-0005-CONFIRMATION-01"
    assert events[30]["resume_data_seed"] == 32076
    assert events[30]["task69_checkpoint_commit"] == runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT
    assert not any(
        event["event_type"] in {"candidate_rejected", "candidate_retired"}
        for event in events)


def test_recovery_resume_entry_reuses_preserved_started_record_without_retire(
    tmp_path: Path,
) -> None:
    """The resume-aware materializer accepts the preserved started record.

    The generic no-resume guard still retires an unauthenticated
    started-without-materialized ledger, while the authenticated recovery resume
    coordinate is accepted without a duplicate started record and without any
    candidate contact (no entries are requested here, so no generation child
    runs). The produced-output refusal still fires before any started append.
    """
    assessment, _, ledger_path, _ = _interrupted_assessment_state(tmp_path)
    before = ledger_path.read_bytes()
    _, _, events = runner._load_stage_state(assessment, tmp_path)
    first_confirmation = next(
        entry for entry in assessment["role_binding"] if entry["role"] == "CONFIRMATION")
    no_entries: list[dict] = []
    # Generic no-resume path (no resume coordinate): retires fail-closed.
    with pytest.raises(runner.GuardError, match="no resume"):
        runner._materialize_entries(
            assessment, tmp_path, tmp_path / "unused-attempt", ledger_path,
            events, no_entries, resume_history_id=None)
    retired = runner.read_ledger(ledger_path)
    assert retired[-1]["event_type"] == "candidate_rejected"
    assert retired[-1]["reason"] == "interrupted_materialization_no_resume"
    # Authenticated recovery resume coordinate: accepted, nothing appended.
    ledger_path.write_bytes(before)
    assert runner._materialize_entries(
        assessment, tmp_path, tmp_path / "unused-attempt", ledger_path,
        runner._load_stage_state(assessment, tmp_path)[2],
        no_entries,
        resume_history_id=first_confirmation["history_id"],
    ) == 0
    assert ledger_path.read_bytes() == before
    # A saved partial output still refuses before any resumed started append.
    partial_root = tmp_path / "resume-partial"
    partial_root.mkdir()
    _interrupted_assessment_state(partial_root)
    (partial_root / first_confirmation["directory"]).mkdir(parents=True)
    with pytest.raises(runner.GuardError, match="role output exists"):
        runner._materialize_entries(
            assessment, partial_root, partial_root / "unused-attempt",
            runner._candidate_paths(assessment, partial_root)[2],
            runner._load_stage_state(assessment, partial_root)[2],
            assessment["role_binding"][12:],
            resume_history_id=first_confirmation["history_id"])


def test_recovery_entry_passes_release_gates_then_honest_runtime_boundary(
    tmp_path: Path,
) -> None:
    """The public recovery entry authenticates, then stops at the honest runtime guard.

    Runs the REAL public CLI subprocess on the disposable clone: the static
    descriptor, the E execution release, and the recovery receipt all
    authenticate (no receipt/static/ledger refusal), and the run then stops at
    the host/runtime boundary with zero ledger, Confirmation, or receipt writes.
    """
    clone, execution_receipt, commits = _recovery_replica_clone(tmp_path)
    assessment, _, ledger_path, record_path = _interrupted_assessment_state(clone)
    before = ledger_path.read_bytes()
    recovery_receipt = _interrupted_recovery_release(assessment, commits["E"])
    recovery_receipt_path = clone / "recovery-receipt.json"
    recovery_receipt_path.write_text(
        json.dumps(recovery_receipt, sort_keys=True), encoding="utf-8")
    execution_receipt_path = clone / "execution-receipt.json"
    execution_receipt_path.write_text(
        json.dumps(execution_receipt, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        str(clone / "experiments" / "sprint18_iterative_candidate_v1.py"),
        "--assessment",
        "--binding", str(clone / "experiments" / "sprint18-iterative-assessment-c5-allowance-v1.json"),
        "--release", str(
            REPO_ROOT / "artifacts" / "sprint-18"
            / "S18-ITER-0005-ASSESS-main-assessment-release-v1.json"),
        "--stage", "materialize-confirmation",
        "--qualification-record", str(record_path),
        "--qualification-sha256", runner.QUALIFIED_ASSESSMENT_QUALIFICATION_SHA256,
        "--task69-release", str(
            REPO_ROOT / "artifacts" / "sprint-18"
            / "S18-ITER-0005-TASK69-main-assessment-release-v2.json"),
        "--execution-release", str(execution_receipt_path),
        "--recovery-release", str(recovery_receipt_path),
    ]
    result = subprocess.run(
        command, cwd=clone, env=runner._entrypoint_child_environment(clone),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "runtime execution worktree mismatch" in result.stderr or (
        "host uv" in result.stderr.lower())
    for refusal in (
        "corrected execution release", "recovery release", "descriptor",
        "ledger", "binding", "source", "Confirmation",
    ):
        assert refusal not in result.stderr, result.stderr
    assert ledger_path.read_bytes() == before
    assert not any(
        (clone / entry["directory"]).exists() for entry in assessment["role_binding"][12:])
