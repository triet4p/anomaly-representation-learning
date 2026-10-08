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
    """Ancestry resolves 003b94bc from 7be574d9; foreign checkpoint/member/parent refuse."""
    root = runner.ROOT
    parent, committed_blob, head_blob = runner._assessment_execution_ancestors(
        root, runner.QUALIFIED_TASK69_ASSESSMENT_CHECKPOINT)
    assert parent == runner.ASSESSMENT_EXECUTION_COMMIT
    assert committed_blob == runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID
    assert head_blob == runner.QUALIFIED_ASSESSMENT_RUNNER_BLOB_OID
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

    The positive case verifies the AUTHORED worktree descriptor bytes against the
    real assessment binding + Task69 release + Git ancestry + working-tree runner
    bytes (no mocks): the descriptor `corrected_runner` MUST DIFFER from the old
    A-side `702b37bb…`/`125df99c…` and MUST EQUAL the current worktree blob +
    canonical bytes. Swaps (old C blob, foreign execution, stale runner, wrong
    DATA pins, drifted release field) all refuse; absence refuses.
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
