"""Behavioral tests for the execution admission gate and cache custody preflight.

These tests protect the invariants the frozen weekly/monthly study's execution
depends on: the required row plan is exactly the frozen per-seed envelope, a
cached row is reusable only under the exact study-required cache identity
(including capture device), malformed or incomplete stores fail closed, and a
failed preflight can never be converted into a scientific disposition.

The corpus and cache fixtures are small and synthetic: no accepted data,
embedding or checkpoint is copied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from representation import sprint24_weekly_monthly_execution as execution
from representation import sprint24_weekly_monthly_study as study
from representation.sprint24_context_audit import sha256_object

DAY = 86_400.0
C1_DAYS = 24.0
SPAN_DAYS = 60.0
ROBOTS = ("robot-01", "robot-02", "robot-03")
PROGRAMS = {"robot-01": "program-01", "robot-02": "program-02", "robot-03": "program-01"}
HISTORIES = tuple(f"SYNTH-WM-{index:02d}" for index in range(1, 5))
EDGES = {
    history_id: [
        0.0,
        6.0 * DAY,
        12.0 * DAY,
        18.0 * DAY,
        C1_DAYS * DAY,
        30.0 * DAY,
        36.0 * DAY,
        42.0 * DAY,
        48.0 * DAY,
        54.0 * DAY,
        SPAN_DAYS * DAY,
    ]
    for history_id in HISTORIES
}


def _manifest(salt: int) -> dict[str, object]:
    files: list[dict[str, object]] = []
    step = 6.0 * 3600.0
    index = 0
    time = 1_800.0
    while time < SPAN_DAYS * DAY:
        robot = ROBOTS[index % len(ROBOTS)]
        files.append(
            {
                "file_id": f"S-op-{salt}-{index:06d}-{(index + salt):08x}",
                "start_time": time,
                "end_time": time + 900.0,
                "robot_id": robot,
                "program_id": PROGRAMS[robot],
                "last_reset_time": 0.0,
                "is_quarantined": False,
                "is_censored": False,
                "n_valid_patches": 40,
                "file_label": "abnormal" if (index + salt) % 7 == 3 else "normal",
            }
        )
        index += 1
        time += step
    return {
        "files": files,
        "maintenance_windows": {},
        "failure_events": [
            {
                "failure_id": f"fail-{salt}-{number}",
                "robot_id": ROBOTS[number % len(ROBOTS)],
                "failure_time": (C1_DAYS + 3.0 + 5.0 * number) * DAY,
            }
            for number in range(6)
        ],
        "episodes": [],
    }


def _identity_inputs() -> dict[str, object]:
    return {
        "checkpoints": {
            str(seed): f"{seed:064x}" for seed in study.REPRESENTATION_SEEDS
        },
        "banks": {
            str(seed): f"{seed + 1:064x}" for seed in study.REPRESENTATION_SEEDS
        },
        "arms": {
            str(seed): f"{seed + 2:064x}" for seed in study.REPRESENTATION_SEEDS
        },
        "binding_sha256": "ab" * 32,
        "preprocessing_sha256": "cd" * 32,
        "device": "cuda",
        "scale_state_sha256": None,
    }


def _plan() -> dict[str, object]:
    manifests = {history_id: _manifest(index) for index, history_id in enumerate(HISTORIES)}
    return execution.required_row_plan(manifests, EDGES)


def _identities(plan: dict[str, object]) -> dict[str, object]:
    return execution.required_cache_identities(plan, _identity_inputs())


def _write_entry(
    root: Path,
    group: dict[str, object],
    file_ids: list[str],
    *,
    inputs: dict[str, object] | None = None,
    cache_identity: str | None = None,
    support_sha256: str | None = None,
    device: str = "cuda",
    embeddings: np.ndarray | None = None,
    drop_sidecar: bool = False,
    duplicate_ids: bool = False,
    provenance: str = execution.CACHE_PROVENANCE_GPU,
    capture_commit: str | None = "a" * 40,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    resolved_inputs = inputs or _identity_inputs()
    seed = str(group["model_seed"])
    ids = list(file_ids)
    if duplicate_ids:
        ids = [ids[0]] * len(ids)
    matrix = (
        embeddings
        if embeddings is not None
        else np.zeros((len(ids), study.EMBEDDING_DIM), dtype=np.float32)
    )
    stem = f"{group['history_id']}__seed-{seed}"
    np.savez(root / f"{stem}.npz", file_ids=np.array(ids, dtype=np.str_), embeddings=matrix)
    if not drop_sidecar:
        sidecar = {
            "schema": execution.CACHE_SIDECAR_SCHEMA,
            "kind": execution.CACHE_SIDECAR_KIND,
            "history_id": group["history_id"],
            "model_seed": group["model_seed"],
            "device": device,
            "checkpoint_sha256": resolved_inputs["checkpoints"][seed],
            "bank_sha256": resolved_inputs["banks"][seed],
            "support_sha256": support_sha256 or group["support_sha256"],
            "preprocessing_sha256": resolved_inputs["preprocessing_sha256"],
            "arm_config_sha256": resolved_inputs["arms"][seed],
            "data_binding_sha256": resolved_inputs["binding_sha256"],
            "scale_state_sha256": "none",
            "cache_identity": cache_identity or group["cache_identity"],
            "n_rows": len(ids),
            "capture_provenance": provenance,
            "capture_commit": capture_commit,
        }
        (root / f"{stem}.json").write_text(
            json.dumps(sidecar, sort_keys=True), encoding="utf-8"
        )
    return root


def _complete_store(root: Path, plan: dict[str, object], identities: dict[str, object]) -> Path:
    for group in identities["groups"].values():
        file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
        _write_entry(root, group, file_ids)
    return root


def test_required_row_plan_reproduces_per_seed_envelope() -> None:
    plan = _plan()
    totals = plan["totals"]
    assert totals["reference_rows_per_seed"] > 0
    assert totals["span_rows_per_seed"] > 0
    assert totals["rows_per_seed"] == totals["reference_rows_per_seed"] + totals["span_rows_per_seed"]
    assert totals["rows_total_three_seeds"] == totals["rows_per_seed"] * len(
        study.REPRESENTATION_SEEDS
    )
    for history_id, block in plan["per_history"].items():
        reference = set(block["reference_ids"])
        span = set(block["span_ids"])
        union = set(block["required_file_ids"])
        assert not reference & span
        assert union == reference | span
        assert block["n_reference_rows"] == len(reference)
        assert block["n_span_rows"] == len(span)
        assert block["union_ids_sha256"] == sha256_object(sorted(union))
        assert block["reference_state"] in {"REFERENCE-OK", "INSUFFICIENT-REFERENCE"}


def test_required_identities_bind_every_group_and_capture_device() -> None:
    plan = _plan()
    identities = _identities(plan)
    assert identities["n_groups"] == len(HISTORIES) * len(study.REPRESENTATION_SEEDS)
    for group in identities["groups"].values():
        assert group["device"] == execution.REQUIRED_DEVICE_IDENTITY
        assert group["support_sha256"] == plan["per_history"][group["history_id"]][
            "union_ids_sha256"
        ]
        assert len(group["cache_identity"]) == 64
    cpu_inputs = {**_identity_inputs(), "device": "cpu"}
    support = next(iter(identities["groups"].values()))["support_sha256"]
    gpu_identity = execution.cache_identity(
        _identity_inputs(), model_seed=181801, support_sha256=support
    )
    cpu_identity = execution.cache_identity(
        cpu_inputs, model_seed=181801, support_sha256=support
    )
    assert gpu_identity != cpu_identity


def test_preprocessing_pin_matches_frozen_configuration() -> None:
    from types import SimpleNamespace

    from representation.config import V1Config
    from representation.sprint24_features import s24_preprocessing_digest

    config = V1Config()
    shim = SimpleNamespace(config=config)
    digest = s24_preprocessing_digest(model=shim, bank_k=config.knn_k)
    assert digest == execution.ACCEPTED_PREPROCESSING_SHA256


def test_complete_identity_matched_store_admits_execution(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    root = _complete_store(tmp_path / "store", plan, identities)
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 0
    assert inventory["valid_entries"] == identities["n_groups"]
    coverage = execution.cache_coverage(plan, identities, inventory)
    assert coverage["coverage_complete"]
    assert coverage["total_available_rows"] == coverage["total_required_rows"]
    card = execution.admit_execution(coverage, inventory)
    assert card["status"] == "ADMITTED"
    assert card["movement_computed"] is False
    assert card["forwards_executed"] == 0


def test_writer_round_trip_admits_identity_matched_store(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    inputs = _identity_inputs()
    root = tmp_path / "written"
    written = []
    for group in identities["groups"].values():
        file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
        matrix = np.zeros((len(file_ids), study.EMBEDDING_DIM), dtype=np.float32)
        record = execution.write_cache_entry(
            root,
            history_id=group["history_id"],
            model_seed=int(group["model_seed"]),
            file_ids=file_ids,
            embeddings=matrix,
            inputs=inputs,
            support_sha256=str(group["support_sha256"]),
            provenance=execution.CACHE_PROVENANCE_GPU,
            capture_commit="b" * 40,
        )
        written.append(record)
    assert all(record["cache_identity"] == group["cache_identity"] for record, group in zip(written, identities["groups"].values()))
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 0
    coverage = execution.cache_coverage(plan, identities, inventory)
    assert coverage["coverage_complete"]
    assert execution.admit_execution(coverage, inventory)["status"] == "ADMITTED"


@pytest.mark.parametrize(
    "provenance",
    [execution.CACHE_PROVENANCE_SMOKE, execution.CACHE_PROVENANCE_FIXTURE],
)
def test_non_gpu_provenance_is_inadmissible(tmp_path: Path, provenance: str) -> None:
    plan = _plan()
    identities = _identities(plan)
    group = next(iter(identities["groups"].values()))
    file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
    root = _write_entry(
        tmp_path / provenance,
        group,
        file_ids,
        provenance=provenance,
        capture_commit=None,
    )
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 1
    assert any(
        "not admissible study input" in issue
        for issue in inventory["entries"][0]["issues"]
    )
    coverage = execution.cache_coverage(plan, identities, inventory)
    with pytest.raises(execution.CachePreflightFailure):
        execution.admit_execution(coverage, inventory)


def test_gpu_provenance_requires_execution_commit(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    group = next(iter(identities["groups"].values()))
    file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
    root = _write_entry(
        tmp_path / "no-commit", group, file_ids, capture_commit=None
    )
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 1
    assert any(
        "execution commit" in issue for issue in inventory["entries"][0]["issues"]
    )


def test_writer_refuses_malformed_data_before_writing(tmp_path: Path) -> None:
    plan = _plan()
    group = next(iter(_identities(plan)["groups"].values()))
    file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
    root = tmp_path / "refused"
    matrix = np.zeros((len(file_ids), study.EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        execution.write_cache_entry(
            root,
            history_id=group["history_id"],
            model_seed=int(group["model_seed"]),
            file_ids=file_ids,
            embeddings=matrix.astype(np.float64),
            inputs=_identity_inputs(),
            support_sha256=str(group["support_sha256"]),
            provenance=execution.CACHE_PROVENANCE_GPU,
            capture_commit="b" * 40,
        )
    with pytest.raises(ValueError):
        execution.write_cache_entry(
            root,
            history_id=group["history_id"],
            model_seed=int(group["model_seed"]),
            file_ids=[file_ids[0], file_ids[0]],
            embeddings=matrix[:2],
            inputs=_identity_inputs(),
            support_sha256=str(group["support_sha256"]),
            provenance=execution.CACHE_PROVENANCE_GPU,
            capture_commit="b" * 40,
        )
    with pytest.raises(ValueError):
        execution.write_cache_entry(
            root,
            history_id=group["history_id"],
            model_seed=int(group["model_seed"]),
            file_ids=file_ids,
            embeddings=matrix,
            inputs=_identity_inputs(),
            support_sha256=str(group["support_sha256"]),
            provenance=execution.CACHE_PROVENANCE_GPU,
            capture_commit=None,
        )
    assert not root.exists()


def test_missing_store_refuses_admission(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    inventory = execution.inspect_cache_stores([tmp_path / "absent"])
    coverage = execution.cache_coverage(plan, identities, inventory)
    assert coverage["total_available_rows"] == 0
    assert coverage["coverage_fraction"] == 0.0
    with pytest.raises(execution.CachePreflightFailure) as excinfo:
        execution.admit_execution(coverage, inventory)
    assert excinfo.value.report["status"] == "REFUSED"
    assert excinfo.value.report["movement_computed"] is False


def test_incomplete_store_reports_exact_missing_rows(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    root = tmp_path / "store"
    for group in identities["groups"].values():
        file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
        _write_entry(root, group, file_ids[:-1])
    inventory = execution.inspect_cache_stores([root])
    coverage = execution.cache_coverage(plan, identities, inventory)
    assert coverage["total_missing_rows"] == identities["n_groups"]
    assert all(value["missing_rows"] == 1 for value in coverage["groups"].values())
    assert not coverage["coverage_complete"]


def test_identity_mismatch_is_a_near_miss_not_a_reusable_row(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    root = tmp_path / "store"
    for group in identities["groups"].values():
        file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
        # Internally consistent but deliberately different support digest.
        wrong_support = hashlib.sha256(b"wrong-support").hexdigest()
        wrong_identity = execution.cache_identity(
            _identity_inputs(),
            model_seed=int(group["model_seed"]),
            support_sha256=wrong_support,
        )
        _write_entry(
            root,
            group,
            file_ids,
            cache_identity=wrong_identity,
            support_sha256=wrong_support,
        )
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 0
    coverage = execution.cache_coverage(plan, identities, inventory)
    assert coverage["total_available_rows"] == 0
    assert any(
        value["near_miss_entries"] for value in coverage["groups"].values()
    )


def test_device_mismatch_fails_closed_as_invalid(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    group = next(iter(identities["groups"].values()))
    file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
    root = _write_entry(tmp_path / "store", group, file_ids, device="cpu")
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 1
    assert any("cache_identity" in issue for issue in inventory["entries"][0]["issues"])
    coverage = execution.cache_coverage(plan, identities, inventory)
    with pytest.raises(execution.CachePreflightFailure):
        execution.admit_execution(coverage, inventory)


def test_missing_sidecar_and_duplicate_ids_fail_closed(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    groups = list(identities["groups"].values())
    no_sidecar = _write_entry(
        tmp_path / "no-sidecar", groups[0], list(plan["per_history"][groups[0]["history_id"]]["required_file_ids"]), drop_sidecar=True
    )
    duplicates = _write_entry(
        tmp_path / "duplicates", groups[1], list(plan["per_history"][groups[1]["history_id"]]["required_file_ids"]), duplicate_ids=True
    )
    inventory = execution.inspect_cache_stores([no_sidecar, duplicates])
    assert inventory["invalid_entries"] == 2
    reasons = " ".join(
        issue for entry in inventory["entries"] for issue in entry["issues"]
    )
    assert "missing identity sidecar" in reasons
    assert "duplicates" in reasons


def test_non_finite_embeddings_fail_closed(tmp_path: Path) -> None:
    plan = _plan()
    identities = _identities(plan)
    group = next(iter(identities["groups"].values()))
    file_ids = list(plan["per_history"][group["history_id"]]["required_file_ids"])
    matrix = np.zeros((len(file_ids), study.EMBEDDING_DIM), dtype=np.float32)
    matrix[0, 0] = np.nan
    root = _write_entry(tmp_path / "store", group, file_ids, embeddings=matrix)
    inventory = execution.inspect_cache_stores([root])
    assert inventory["invalid_entries"] == 1
    assert any("non-finite" in issue for issue in inventory["entries"][0]["issues"])


def test_execution_ledger_accounts_every_cell_without_claiming_outcomes() -> None:
    manifests = {history_id: _manifest(index) for index, history_id in enumerate(HISTORIES)}
    calendar = study.primary_calendar()
    result = study.evaluate_source(manifests, EDGES, calendar, modes=study.MODES)
    label_blind = study.build_label_blind_manifest(result, calendar)
    posthoc = study.build_posthoc_burden(result)
    metadata_coverage = study.coverage_matrix(label_blind, posthoc)
    ledger = execution.execution_ledger(
        metadata_coverage, {"complete": False}, "cache preflight refused admission"
    )
    assert ledger["n_cells"] == 192 == ledger["declared_cell_count"]
    assert ledger["cell_execution_status_counts"] == {
        execution.NOT_EXECUTED_STATUS: 192
    }
    assert all(cell["outcome"] is None for cell in ledger["cells"])
    assert all(
        cell["execution_status"] == execution.NOT_EXECUTED_STATUS
        for cell in ledger["cells"]
    )
    assert set(ledger["contrasts"][0]) >= {"id", "declared_state", "execution_status"}
    assert len(ledger["contrasts"]) == 8
    for disposition in ledger["dispositions"].values():
        assert disposition["disposition"] is None
        assert disposition["status"] == "NOT-COMPUTED"
        assert disposition["specificity_claim"] is False
        assert disposition["scaling_status"] == "NOT-RELEASED"
        assert {criterion["status"] for criterion in disposition["criteria"]} == {
            "NOT-EVALUATED"
        }
        assert len(disposition["criteria"]) == len(execution.DECISION_CRITERIA)


def test_cost_record_keeps_zero_forwards_and_the_inherited_envelope() -> None:
    plan = _plan()
    identities = _identities(plan)
    inventory = execution.inspect_cache_stores([Path("definitely-missing-cache-root")])
    coverage = execution.cache_coverage(plan, identities, inventory)
    cost = execution.cost_record(plan, coverage)
    assert cost["forwards_executed_by_this_attempt"] == 0
    assert cost["model_loads"] == 0
    assert cost["waveform_reads"] == 0
    assert cost["cap_reset"] is False
    accounting = cost["accounting"]
    span_only_total = accounting["span_only_rows_total_three_seeds"]
    expected_excess = max(0, span_only_total - accounting["inherited_headroom_forwards"])
    assert accounting["fresh_span_only_excess_rows"] == expected_excess
    assert accounting["fresh_span_only_exceeds_inherited_headroom"] == (
        expected_excess > 0
    )
    assert cost["required_rows"]["per_seed"] == cost["accounting"]["rows_per_seed"]
    assert cost["missing_rows"]["total_three_seeds"] == coverage["total_missing_rows"]
