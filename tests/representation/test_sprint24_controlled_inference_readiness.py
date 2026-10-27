"""Focused behavioral checks for S24-7-M10 controlled-corpus inference readiness.

C2: metadata-only cache inspection (central-directory + file_ids member only;
never the embeddings member, never whole-file bytes), non-overrideable
internal pins + canonical freeze/content custody validation, callable
controlled-corpus adapter to accepted cache/movement machinery on real pair
structure with same-data measurement, independent numerical re-derivation,
altered-contract refusal and unknown-ID refusal, 5-pair C1 census with
independent support gating, refusal boundaries, deterministic reproduction.
No model, checkpoint, embedding value/score, or waveform signal is read;
zero forwards.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from representation import sprint24_controlled_inference_readiness as readiness

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
CORPUS = REPO_ROOT / "data" / "generated" / "sprint24_controlled_v2"
AUDIT = REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m9-a01" / "corpus-audit.json"
G2_PREFLIGHT = REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01" / "g2-accepted" / "preflight.json"


def _run_cli(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
    )


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert isinstance(audit, dict)
    preflight = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))
    assert isinstance(preflight, dict)
    manifest = json.loads((CORPUS / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    manifest["__manifest_path__"] = str(CORPUS / "corpus-manifest.json")
    return manifest, audit, preflight


def _bound_contract(manifest: dict[str, Any], audit: dict[str, Any], preflight: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build the actual bound prospective-contract payload plus its digest."""
    frozen = readiness.prospective_contract(manifest, audit, preflight)
    contract = frozen["contract"]
    assert isinstance(contract, dict)
    return dict(contract), str(frozen["contract_sha256"])


def test_m10_amendment_identity_pinned() -> None:
    assert readiness.READINESS_ID == "S24-CONTROLLED-INFERENCE-READINESS-v1"
    assert readiness.AMENDMENT_ID == "S24-CONTROLLED-INFERENCE-AMENDMENT-A01"
    assert readiness.EXPOSURE_RATIO_FLOOR == 0.8
    assert readiness.FIVE_PAIR_MIN_TWO_SIDED_TAIL == pytest.approx(2.0 / 120.0)
    assert readiness.PINNED_EXECUTION_SHA256 == "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"


def test_m10_canonical_freeze_and_pins() -> None:
    freeze = json.loads((CORPUS / "execution-freeze.json").read_text(encoding="utf-8"))
    assert isinstance(freeze, dict)
    ok, _ = readiness.verify_canonical_freeze(freeze)
    assert ok
    tampered = json.loads(json.dumps(freeze))
    assert isinstance(tampered["spec"], dict)
    tampered["spec"]["protocol_id"] = "FORGED"
    ok_forged, _ = readiness.verify_canonical_freeze(tampered)
    assert not ok_forged
    ok_manifest, _ = readiness.verify_pinned_artifact(CORPUS / "corpus-manifest.json", "corpus manifest")
    assert ok_manifest
    ok_audit, _ = readiness.verify_pinned_artifact(AUDIT, "corpus audit")
    assert ok_audit


def test_m10_expected_retained_census_admits_all_transitions() -> None:
    expected = readiness.expected_retained_census()
    assert expected["retained_eligible_transitions"] == 48
    assert expected["retained_eligible_controls"] == 12
    raw_limitation = expected["raw_limitation"]
    assert isinstance(raw_limitation, dict)
    assert raw_limitation["raw_eligible_transitions"] == 0
    assert expected["c2_status"] == "ABSTAIN"
    promising = expected["promising_gate"]
    assert isinstance(promising, str) and promising.startswith("NOT-APPLICABLE")


def test_m10_live_census_matches_expected_ranges() -> None:
    manifest, audit, _preflight = _inputs()
    pair_rows = audit["pair_rows"]
    assert isinstance(pair_rows, list)
    end_time_of, duration_of, header_reads = readiness._manifest_file_maps(manifest)  # noqa: SLF001
    assert header_reads == 8280
    census = readiness.eligibility_census(pair_rows, end_time_of, duration_of)
    assert census["n_pairs"] == 60
    counts: dict[str, int] = {}
    c1_members = 0
    for pair in census["pairs"]:  # type: ignore[union-attr]
        assert isinstance(pair, dict)
        counts[str(pair["eligibility"])] = counts.get(str(pair["eligibility"]), 0) + 1
        assert pair["c2_status"] == "ABSTAIN"
        assert isinstance(pair["retained_exposure_ratio"], float)
        assert pair["retained_exposure_ratio"] >= 0.8
        assert pair["meets_support_minima"] is True
        if bool(pair["c1_association_member"]):
            c1_members += 1
    assert counts.get("ELIGIBLE-RETAINED") == 48
    assert counts.get("CONTROL-EXPECTED-NULL") == 12
    assert c1_members == 60


def test_m10_custody_validates_and_derives_identity_plan() -> None:
    manifest, _audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    assert custody["preprocessing_sha256"] == (
        "903a4702c57719769d194d89820566d0bd3fd0ee2ba44e3d87e5dbc011cb97ef"
    )
    assert custody["binding_sha256"] == "383c62114ee6120d5fd19f866dfd585083d1d5899c9ffe91804761bd8a10a2d5"
    assert custody["device"] == "cuda"
    plan = readiness.custody_identity_plan(manifest, custody)
    assert plan["n_groups"] == 18
    assert plan["n_histories"] == 6
    groups = plan["groups"]
    assert isinstance(groups, dict)
    assert len(groups) == 18
    sample = groups["S24C-ITER-0001-CONTROLLED-01|181801"]
    assert isinstance(sample, dict)
    assert sample["required_rows"] == 1380
    assert len(str(sample["cache_identity"])) == 64
    probe = readiness.probe_declared_cache_roots([])
    assert probe["verdict"] == "NOT-CHECKED/UNKNOWN"
    declared = readiness.probe_declared_cache_roots([REPO_ROOT / "definitely-absent-m10"])
    assert declared["verdict"] == "MISSING"


def test_m10_manifest_inspector_never_touches_embedding_bytes(tmp_path: Path) -> None:
    from representation import sprint24_weekly_monthly_execution as execution

    manifest, _audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    inputs = {
        "checkpoints": dict(custody["checkpoints"]),  # type: ignore[index]
        "banks": dict(custody["banks"]),  # type: ignore[index]
        "arms": dict(custody["arms"]),  # type: ignore[index]
        "binding_sha256": str(custody["binding_sha256"]),
        "preprocessing_sha256": str(custody["preprocessing_sha256"]),
        "device": str(custody["device"]),
        "scale_state_sha256": None,
    }
    groups = readiness.custody_identity_plan(manifest, custody)["groups"]
    assert isinstance(groups, dict)
    first_history = sorted(str(key) for key in groups.keys())[0].split("|")[0]
    file_entries = manifest["files"]
    assert isinstance(file_entries, list)
    manifest_files = [e for e in file_entries if isinstance(e, dict) and e.get("history_id") == first_history]
    file_ids = sorted(str(e["file_id"]) for e in manifest_files)
    support = readiness.sha256_object(file_ids)
    matrix = np.random.default_rng(2407).normal(0, 1, size=(len(file_ids), 128)).astype(np.float32)
    root = tmp_path / "store"
    execution.write_cache_entry(
        root,
        history_id=first_history,
        model_seed=181801,
        file_ids=file_ids,
        embeddings=matrix,
        inputs=inputs,
        support_sha256=support,
        provenance=execution.CACHE_PROVENANCE_FIXTURE,
        capture_commit=None,
    )
    accessed_members: list[str] = []
    real_open = zipfile.ZipFile.open

    def _guarded_open(self: zipfile.ZipFile, name: str, *args: Any, **kwargs: Any) -> Any:
        accessed_members.append(str(name))
        if "embeddings" in str(name):
            raise AssertionError("embeddings member must never be opened")
        return real_open(self, name, *args, **kwargs)

    zipfile.ZipFile.open = _guarded_open  # type: ignore[method-assign]
    try:
        inspected = readiness.inspect_cache_manifest_only(root)
    finally:
        zipfile.ZipFile.open = real_open  # type: ignore[method-assign]
    assert inspected["npz_count"] == 1
    assert inspected["valid_entries"] == 1
    assert inspected["embedding_values_opened"] == 0
    assert inspected["embedding_bytes_read"] == 0
    total_rows = inspected["total_manifest_rows"]
    assert isinstance(total_rows, int) and total_rows == len(file_ids)
    assert not any("embeddings" in name for name in accessed_members)
    entries = inspected["entries"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assert "npz_sha256" not in entries[0]


def test_m10_manifest_inspector_classifies_custody_states(tmp_path: Path) -> None:
    missing = readiness.inspect_cache_manifest_only(tmp_path / "no-such-root")
    assert missing["status"] == "MISSING"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert readiness.inspect_cache_manifest_only(empty)["status"] == "ABSENT"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "orphan.npz").write_bytes(b"not-a-zip")
    assert readiness.inspect_cache_manifest_only(bad)["status"] == "INVALID"
    probe = readiness.probe_declared_cache_roots([tmp_path / "no-such-root"])
    assert probe["verdict"] == "MISSING"
    probe_bad = readiness.probe_declared_cache_roots([bad])
    assert probe_bad["verdict"] == "INVALID"

def test_m10_probe_compares_against_frozen_plan_with_near_miss(tmp_path: Path) -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    fixture = tmp_path / "plan"
    adapter = readiness.controlled_adapter_probe(
        fixture, manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    assert adapter["n_near_misses"] == 0
    executable = contract.get("executable_path")
    assert isinstance(executable, dict)
    plan = executable.get("identity_plan")
    assert isinstance(plan, dict)
    assert isinstance(plan.get("groups"), dict)
    probe = readiness.probe_declared_cache_roots([fixture])
    assert probe["verdict"] == "NON-EMPTY"
    comparison = readiness.compare_probe_to_plan(probe, plan)
    assert comparison["verdict"] == "MATCH"
    assert comparison["n_near_misses"] == 0
    victim = sorted(fixture.glob("*.json"))[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["cache_identity"] = "f" * 64
    victim.write_text(json.dumps(payload), encoding="utf-8")
    probe2 = readiness.probe_declared_cache_roots([fixture])
    assert probe2["verdict"] == "INVALID"
    comparison2 = readiness.compare_probe_to_plan(probe2, plan)
    assert comparison2["verdict"] == "REFUSED"
    assert comparison2["n_near_misses"] == 0


def test_m10_probe_reports_near_miss_on_self_consistent_wrong_identity(tmp_path: Path) -> None:
    from representation import sprint24_weekly_monthly_execution as execution

    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    fixture = tmp_path / "nearmiss"
    adapter = readiness.controlled_adapter_probe(
        fixture, manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    assert adapter["n_near_misses"] == 0
    executable = contract.get("executable_path")
    assert isinstance(executable, dict)
    plan = executable.get("identity_plan")
    assert isinstance(plan, dict)
    assert isinstance(plan.get("groups"), dict)
    victim = sorted(fixture.glob("*.json"))[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    wrong_support = "ab" * 32
    payload["support_sha256"] = wrong_support
    payload["cache_identity"] = execution.cache_identity(
        {
            "checkpoints": {str(payload["model_seed"]): str(payload["checkpoint_sha256"])},
            "banks": {str(payload["model_seed"]): str(payload["bank_sha256"])},
            "arms": {str(payload["model_seed"]): str(payload["arm_config_sha256"])},
            "binding_sha256": str(payload["data_binding_sha256"]),
            "preprocessing_sha256": str(payload["preprocessing_sha256"]),
            "device": str(payload["device"]),
            "scale_state_sha256": None if str(payload["scale_state_sha256"]) == "none" else str(payload["scale_state_sha256"]),
        },
        model_seed=int(payload["model_seed"]),
        support_sha256=wrong_support,
    )
    victim.write_text(json.dumps(payload), encoding="utf-8")
    probe = readiness.probe_declared_cache_roots([fixture])
    assert probe["verdict"] == "NON-EMPTY"
    comparison = readiness.compare_probe_to_plan(probe, plan)
    assert comparison["verdict"] == "NEAR-MISS"
    n_near = comparison["n_near_misses"]
    assert isinstance(n_near, int) and n_near >= 1


def test_m10_cli_refuses_invalid_declared_roots(tmp_path: Path) -> None:
    bad = tmp_path / "badroot"
    bad.mkdir()
    (bad / "orphan.npz").write_bytes(b"not-a-zip")
    target = tmp_path / "out"
    completed = _run_cli(
        str(EXPERIMENTS / "sprint24_task7m10_controlled_inference_readiness.py"),
        "--corpus", str(CORPUS),
        "--audit", str(AUDIT),
        "--out", str(target),
        "--cache-root", str(bad),
    )
    assert completed.returncode != 0
    assert "REFUSED-IDENTITY" in (completed.stdout + completed.stderr)

def test_m10_controlled_adapter_wires_same_data_and_refuses_contract(tmp_path: Path) -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    adapter = readiness.controlled_adapter_probe(
        tmp_path / "fixture", manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    assert adapter["kind"] == "SYNTHETIC-FIXTURE"
    assert adapter["plumbing_admission_status"] == "ADMITTED"
    assert bool(adapter["plumbing_admission_complete"]) is True
    invalid_entries = adapter["inspector_invalid_entries"]
    assert isinstance(invalid_entries, int) and invalid_entries == 0
    assert adapter["n_near_misses"] == 0
    assert adapter["status"] == "COMPUTED"
    e1 = adapter["e1"]
    e1_direct = adapter["e1_direct"]
    assert isinstance(e1, float) and isinstance(e1_direct, float)
    assert bool(np.isfinite(e1)) is True
    assert adapter["e1_match"] is True
    assert e1 == e1_direct
    assert adapter["contract_sensitive"] is True
    assert adapter["contract_refused"] is True
    assert adapter["missing_refused"] is True
    assert adapter["binding_ok"] is True
    assert adapter["binding_verified_pre_execution"] is True
    study_admission = adapter["study_admission"]
    assert isinstance(study_admission, dict) and study_admission["status"] == "REFUSED"
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        readiness.controlled_adapter_probe(tmp_path / "fixture-missing", manifest, audit, custody)


def test_m10_adapter_admission_uses_inspector_results_not_fabrication(tmp_path: Path) -> None:
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    fixture = tmp_path / "fixture"
    adapter = readiness.controlled_adapter_probe(
        fixture, manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    invalid0 = adapter["inspector_invalid_entries"]
    assert isinstance(invalid0, int) and invalid0 == 0
    sidecars = sorted(fixture.glob("*.json"))
    assert sidecars
    victim = sidecars[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["cache_identity"] = "0" * 64
    victim.write_text(json.dumps(payload), encoding="utf-8")
    reinspected = readiness.inspect_cache_manifest_only(fixture)
    invalid = reinspected["invalid_entries"]
    assert isinstance(invalid, int) and invalid >= 1
    from representation import sprint24_weekly_monthly_execution as execution

    with pytest.raises(Exception, match="(?i)(refus|invalid|identity|mismatch|no .*reusable|not .*admit)"):
        plan_blocks = {"h": {"required_file_ids": ["a"]}}
        groups = {
            "h|181801": {
                "history_id": "h",
                "model_seed": 181801,
                "required_rows": 1,
                "support_sha256": "0" * 64,
                "cache_identity": "1" * 64,
            }
        }
        entries = [
            {
                "path": "x",
                "history_id": "h",
                "model_seed": 181801,
                "valid": False,
                "file_ids": [],
                "cache_identity": "2" * 64,
            }
        ]
        coverage = execution.cache_coverage(
            {"per_history": plan_blocks}, {"groups": groups}, {"entries": entries}
        )
        execution.admit_execution(coverage, {"invalid_entries": 1, "entries": entries})


def test_m10_altered_actual_contract_refused_before_fit_or_movement(tmp_path: Path) -> None:
    """A changed term in the actual contract payload refuses before admission."""
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    good = readiness.controlled_adapter_probe(
        tmp_path / "fixture-good", manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    assert good["checks"] is True
    assert good["binding_verified_pre_execution"] is True
    altered = json.loads(json.dumps(contract))
    assert isinstance(altered, dict)
    executable = altered.get("executable_path")
    assert isinstance(executable, dict)
    identity_plan = executable.get("identity_plan")
    assert isinstance(identity_plan, dict)
    groups = identity_plan.get("groups")
    assert isinstance(groups, dict)
    victim_key = sorted(groups.keys())[0]
    victim_group = dict(groups[victim_key])  # type: ignore[index]
    victim_group["support_sha256"] = "cd" * 32
    groups[victim_key] = victim_group
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        readiness.controlled_adapter_probe(
            tmp_path / "fixture-altered-term", manifest, audit, custody, expected_contract=altered, expected_contract_sha256=contract_sha
        )
    stale = json.loads(json.dumps(contract))
    assert isinstance(stale, dict)
    stale["binding_digest"] = "00" * 32
    executable_stale = stale.get("executable_path")
    assert isinstance(executable_stale, dict)
    identity_stale = executable_stale.get("identity_plan")
    assert isinstance(identity_stale, dict)
    assert isinstance(identity_stale.get("groups"), dict)
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        readiness.controlled_adapter_probe(
            tmp_path / "fixture-stale-digest", manifest, audit, custody, expected_contract=stale, expected_contract_sha256=contract_sha
        )


def test_m10_public_cli_cache_path_matches_valid_fixture(tmp_path: Path) -> None:
    """The real public CLI path matches a faithful cache and refuses a bad one."""
    manifest, audit, preflight = _inputs()
    custody = readiness.validate_custody_inputs(preflight)
    contract, contract_sha = _bound_contract(manifest, audit, preflight)
    fixture = tmp_path / "valid-cache"
    adapter = readiness.controlled_adapter_probe(
        fixture, manifest, audit, custody, expected_contract=contract, expected_contract_sha256=contract_sha
    )
    assert adapter["n_near_misses"] == 0
    assert adapter["e1_match"] is True
    target = tmp_path / "out"
    completed = _run_cli(
        str(EXPERIMENTS / "sprint24_task7m10_controlled_inference_readiness.py"),
        "--corpus", str(CORPUS),
        "--audit", str(AUDIT),
        "--out", str(target),
        "--cache-root", str(fixture),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_record = json.loads((target / "run.json").read_text(encoding="utf-8"))
    assert isinstance(run_record, dict)
    assert run_record["cache_plan_comparison"]["verdict"] == "MATCH"
    assert run_record["cache_plan_comparison"]["n_near_misses"] == 0
    assert run_record["identity_plan_groups"] == 18
    assert run_record["adapter_binding_verified_pre_execution"] is True


def test_m10_header_reader_never_materializes_signals() -> None:
    target = None
    for path in (CORPUS / "waves").rglob("*.npz"):
        target = path
        break
    assert target is not None
    shape = readiness.read_npy_shape_header(target)
    assert shape[0] == 6 and shape[1] >= 179
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    _ = readiness.read_npy_shape_header(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest


def test_m10_preflight_cli_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for target in (first, second):
        completed = _run_cli(
            str(EXPERIMENTS / "sprint24_task7m10_controlled_inference_readiness.py"),
            "--corpus", str(CORPUS),
            "--audit", str(AUDIT),
            "--out", str(target),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    for name in ("eligibility-census.json", "expected-retained-census.json", "prospective-contract.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_m10_preflight_cli_refuses_overwrite_drift_and_forged_custody(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "sentinel.txt").write_text("occupied", encoding="utf-8")
    completed = _run_cli(
        str(EXPERIMENTS / "sprint24_task7m10_controlled_inference_readiness.py"),
        "--corpus", str(CORPUS),
        "--audit", str(AUDIT),
        "--out", str(target),
    )
    assert completed.returncode != 0
    assert "REFUSED-OVERWRITE" in (completed.stdout + completed.stderr)
    forged = tmp_path / "forged-preflight.json"
    preflight = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))
    assert isinstance(preflight, dict)
    preflight["binding_sha256"] = "0" * 64
    forged.write_text(json.dumps(preflight), encoding="utf-8")
    forged_target = tmp_path / "forged"
    forged_run = _run_cli(
        str(EXPERIMENTS / "sprint24_task7m10_controlled_inference_readiness.py"),
        "--corpus", str(CORPUS),
        "--audit", str(AUDIT),
        "--out", str(forged_target),
        "--g2-preflight", str(forged),
    )
    assert forged_run.returncode != 0
    assert "REFUSED-DIGEST-DRIFT" in (forged_run.stdout + forged_run.stderr)


def test_m10_forged_custody_content_fails_identity() -> None:
    preflight = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))
    assert isinstance(preflight, dict)
    preflight["binding_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        readiness.validate_custody_inputs(preflight)
    preflight["binding_sha256"] = json.loads(G2_PREFLIGHT.read_text(encoding="utf-8"))["binding_sha256"]
    preflight["checkpoints"] = {"181801": "not-a-digest", "181802": "0" * 64, "181803": "0" * 64}
    with pytest.raises(ValueError, match="REFUSED-IDENTITY"):
        readiness.validate_custody_inputs(preflight)


def test_m10_census_abstains_on_missing_support() -> None:
    manifest, audit, _preflight = _inputs()
    pair_rows = audit["pair_rows"]
    assert isinstance(pair_rows, list)
    end_time_of, duration_of, _ = readiness._manifest_file_maps(manifest)  # noqa: SLF001
    tampered = [dict(row) for row in pair_rows]  # type: ignore[union-attr]
    tampered[0] = dict(tampered[0])
    tampered[0]["meets_support_minima"] = False
    census = readiness.eligibility_census(tampered, end_time_of, duration_of)
    states = {str(p["comparison_identity"]): str(p["eligibility"]) for p in census["pairs"]}  # type: ignore[union-attr]
    assert states[str(tampered[0]["comparison_identity"])] == "ABSTAIN"
