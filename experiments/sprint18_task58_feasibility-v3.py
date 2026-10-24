#!/usr/bin/env python3
"""One-shot continuation of the sealed Sprint 18 feasibility study v2 prefix.

``--smoke`` proves the v2 import and v3 persistence/finalization path using
only disposable fixture rows. ``--continue`` is review-gated and evaluates
only the four coordinates absent from the immutable v2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
V2_RUNNER_PATH = ROOT / "experiments/sprint18_task58_feasibility-v2.py"
V2_CONTRACT_PATH = ROOT / "experiments/sprint18-isolated-feasibility-study-v2.md"
V2_ATTEMPT_PATH = ROOT / "artifacts/sprint-18/isolated-feasibility-v2/attempt.json"
V2_RESULT_PATH = ROOT / "artifacts/sprint-18/isolated-feasibility-v2/result.json"
OUTPUT_DIR = ROOT / "artifacts/sprint-18/isolated-feasibility-v3"
ATTEMPT_PATH = OUTPUT_DIR / "attempt.json"
RESULT_PATH = OUTPUT_DIR / "result.json"
FORBIDDEN_ROOT = ROOT / "data/generated/sprint18-isolated-feasibility-v1"

V2_RUNNER_SHA256 = "d5c4dfe3039bf82ec7eec05bf81f083966b81098ed54813b3e1c3d34c35f50b6"
V2_CONTRACT_SHA256 = "8b8e28e489c393cd0ccf33d5844bab9a1d5e562904d671f5b94c122fc4b5ec42"
V2_ATTEMPT_SHA256 = "9bde33a9656f9b0a4d70215f28a44d44fd382f8413eb4fe06bd3366f36040ba1"
V2_RESULT_SHA256 = "2fee65aa8af915db021d63138bb2748172402f125c66478ae4ee803216a6c25a"
V2_FREEZE_SHA256 = "4a34ab48b76b02305a8267c138f6cddaf5f72550d763ef39dfc49d46c4ded535"
V2_LAST_RECORD_SHA256 = "0157b3c980eee4f2697ef0572107d4f7ae7f6989def2662e1a316732d84d9b05"
SOURCE_CLOSURE_SHA256 = "2678d50a328fb4d34c941c38c8d04e91d85ad3c363f4793cd02e3aa0a5b0b834"
V2_AMENDMENT_ID = "sprint18-isolated-feasibility-study-v2-infrastructure-continuation"
V3_AMENDMENT_ID = "sprint18-isolated-feasibility-study-v3-interruption-continuation"

_V2_CORE = runpy.run_path(str(V2_RUNNER_PATH), run_name="sprint18_task58_v2_core")
ContractError = cast("type[RuntimeError]", _V2_CORE["ContractError"])
_V2_CANONICAL_JSON = cast("Callable[[object], bytes]", _V2_CORE["_canonical_json"])
_V2_FILE_SHA256 = cast("Callable[[Path], str]", _V2_CORE["_file_sha256"])
_V2_LEGACY_PROVENANCE = cast(
    "Callable[[], dict[str, str]]", _V2_CORE["_legacy_provenance_hashes"]
)
_V2_VERIFY_STATIC_GUARDS = cast(
    "Callable[[], dict[str, object]]", _V2_CORE["_verify_static_guards"]
)
_V2_VERIFY_PRIOR_EVIDENCE = cast(
    "Callable[..., tuple[dict[str, object], dict[str, object]]]",
    _V2_CORE["_verify_prior_evidence"],
)
_V2_VERIFY_HISTORICAL_BYTES = cast(
    "Callable[[], dict[str, str]]", _V2_CORE["_verify_historical_evidence_bytes"]
)
_V2_RESOLVE_PAIR_CONFIGS = cast(
    "Callable[[int], dict[str, object]]", _V2_CORE["_resolve_pair_configs"]
)
_V2_METRIC_RECORD = cast(
    "Callable[..., tuple[dict[str, object], bool]]", _V2_CORE["_metric_record"]
)
_V2_SEAL_ROW = cast("Callable[[dict[str, object]], dict[str, object]]", _V2_CORE["_seal_row"])
_V2_SUMMARIZE = cast(
    "Callable[..., dict[str, object]]", _V2_CORE["_summarize"]
)
_V2_WRITE_JSON_ATOMIC = cast(
    "Callable[..., None]", _V2_CORE["_write_json_atomic"]
)
_V2_CLAIM_ATTEMPT = cast(
    "Callable[[Path, Mapping[str, object]], Path]", _V2_CORE["_claim_attempt"]
)

STUDY_ID = str(_V2_CORE["STUDY_ID"])
STUDY_SEEDS = cast("tuple[int, ...]", _V2_CORE["STUDY_SEEDS"])
PROFILE_ORDER = cast("tuple[str, ...]", _V2_CORE["PROFILE_ORDER"])
V1_ATTEMPT_PATH = cast("Path", _V2_CORE["ORIGINAL_ATTEMPT_PATH"])
V1_RESULT_PATH = cast("Path", _V2_CORE["ORIGINAL_RESULT_PATH"])
V1_FREEZE_SHA256 = str(_V2_CORE["ORIGINAL_FREEZE_SHA256"])

TOTAL_EVALUATION_BUDGET = len(STUDY_SEEDS) * len(PROFILE_ORDER)
IMPORTED_COORDINATES = tuple(
    (seed, profile)
    for seed in STUDY_SEEDS[:-2]
    for profile in PROFILE_ORDER
)
REMAINING_COORDINATES = (
    (918014, "sprint15-v7"),
    (918014, "sprint18-c2"),
    (918015, "sprint15-v7"),
    (918015, "sprint18-c2"),
)
ALL_COORDINATES = tuple(
    (seed, profile) for seed in STUDY_SEEDS for profile in PROFILE_ORDER
)
if (
    len(IMPORTED_COORDINATES) != 28
    or len(REMAINING_COORDINATES) != 4
    or IMPORTED_COORDINATES + REMAINING_COORDINATES != ALL_COORDINATES
    or len(set(ALL_COORDINATES)) != TOTAL_EVALUATION_BUDGET
):
    raise RuntimeError("the fixed v3 continuation coordinate budget is inconsistent")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"provenance JSON is not an object: {path}")
    return cast("dict[str, object]", value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not an object")
    return cast("dict[str, object]", value)


def _release_binding() -> dict[str, object]:
    text = (ROOT / "experiments/sprint18-isolated-feasibility-study-v3.md").read_text(
        encoding="utf-8"
    )
    marker = "<!-- release-binding -->"
    if marker not in text:
        raise ContractError("v3 contract has no release-binding JSON block")
    tail = text.split(marker, 1)[1]
    if "```json" not in tail or "```" not in tail.split("```json", 1)[1]:
        raise ContractError("v3 contract release-binding JSON block is malformed")
    raw = tail.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"v3 release binding is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("v3 contract release binding is not an object")
    return cast("dict[str, object]", value)


def _expected_v2_provenance() -> dict[str, str]:
    return {
        "runner_sha256": V2_RUNNER_SHA256,
        "contract_sha256": V2_CONTRACT_SHA256,
        "attempt_sha256": V2_ATTEMPT_SHA256,
        "result_sha256": V2_RESULT_SHA256,
        "freeze_sha256": V2_FREEZE_SHA256,
        "last_record_sha256": V2_LAST_RECORD_SHA256,
    }


def _validate_rows(
    rows: Sequence[Mapping[str, object]],
    expected_coordinates: Sequence[tuple[int, str]],
    *,
    require_complete: bool = False,
) -> None:
    if len(rows) > len(expected_coordinates):
        raise ContractError("continuation contains extra persisted coordinates")
    if require_complete and len(rows) != len(expected_coordinates):
        raise ContractError("continuation is missing one or more fixed coordinates")
    actual_coordinates = [
        (int(row.get("history_seed", -1)), str(row.get("profile", "")))
        for row in rows
    ]
    if len(set(actual_coordinates)) != len(actual_coordinates):
        raise ContractError("continuation contains duplicate persisted coordinates")
    for index, row in enumerate(rows):
        coordinate = actual_coordinates[index]
        if coordinate != expected_coordinates[index]:
            raise ContractError(
                f"continuation row/order mismatch at {index}: "
                f"{coordinate} != {expected_coordinates[index]}"
            )
        digest = row.get("record_sha256")
        unsealed = dict(row)
        unsealed.pop("record_sha256", None)
        if not isinstance(digest, str) or _sha256(_V2_CANONICAL_JSON(unsealed)) != digest:
            raise ContractError(f"continuation row {index} has an invalid seal")
        if row.get("no_write") is not True or row.get("generation_writes") is not False:
            raise ContractError(f"continuation row {index} breached no-write isolation")
        if require_complete and row.get("evaluation_started") is not True:
            raise ContractError(f"continuation row {index} is not a completed evaluation")


def _verify_v1_provenance(source_closure_sha256: str) -> dict[str, object]:
    attempt, imported_row = _V2_VERIFY_PRIOR_EVIDENCE(
        V1_ATTEMPT_PATH,
        V1_RESULT_PATH,
        expected_attempt_sha256=str(_V2_CORE["ORIGINAL_ATTEMPT_SHA256"]),
        expected_result_sha256=str(_V2_CORE["ORIGINAL_RESULT_SHA256"]),
        expected_row_sha256=str(_V2_CORE["ORIGINAL_ROW_SHA256"]),
        expected_freeze_sha256=V1_FREEZE_SHA256,
        expected_seed_order=STUDY_SEEDS,
        source_closure_sha256=source_closure_sha256,
    )
    return {"attempt": attempt, "row": imported_row}


def _verify_v2_evidence(
    static: Mapping[str, object],
    *,
    attempt_path: Path = V2_ATTEMPT_PATH,
    result_path: Path = V2_RESULT_PATH,
    expected_attempt_sha256: str = V2_ATTEMPT_SHA256,
    expected_result_sha256: str = V2_RESULT_SHA256,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    actual_attempt_sha256 = _V2_FILE_SHA256(attempt_path) if attempt_path.is_file() else None
    actual_result_sha256 = _V2_FILE_SHA256(result_path) if result_path.is_file() else None
    if actual_attempt_sha256 != expected_attempt_sha256:
        raise ContractError("prior v2 attempt digest does not match the frozen evidence")
    if actual_result_sha256 != expected_result_sha256:
        raise ContractError("prior v2 result digest does not match the frozen evidence")

    attempt = _read_object(attempt_path)
    result = _read_object(result_path)
    closure_sha256 = str(static["source_closure_sha256"])
    expected_attempt = {
        "study_id": STUDY_ID,
        "amendment_id": V2_AMENDMENT_ID,
        "attempt": 1,
        "status": "RUNNING",
        "freeze_digest": V2_FREEZE_SHA256,
        "original_freeze_digest": V1_FREEZE_SHA256,
        "source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "seed_order": list(STUDY_SEEDS),
        "profile_order": list(PROFILE_ORDER),
        "total_evaluation_budget": TOTAL_EVALUATION_BUDGET,
        "imported_evaluations": 1,
        "new_evaluation_budget": 31,
        "completed_evaluations": 28,
        "new_evaluations_completed": 27,
        "remaining_evaluations": 4,
        "last_record_sha256": V2_LAST_RECORD_SHA256,
        "generated_data_roots_written": False,
    }
    for field, value in expected_attempt.items():
        if attempt.get(field) != value:
            raise ContractError(f"prior v2 attempt has mismatched {field}")
    if attempt.get("result_sha256") != expected_result_sha256:
        raise ContractError("prior v2 attempt does not bind its frozen result bytes")

    expected_result = {
        "study_id": STUDY_ID,
        "amendment_id": V2_AMENDMENT_ID,
        "status": "RUNNING",
        "freeze_digest": V2_FREEZE_SHA256,
        "original_freeze_digest": V1_FREEZE_SHA256,
        "source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "profile_order": list(PROFILE_ORDER),
        "seed_order": list(STUDY_SEEDS),
        "generated_data_roots_written": False,
    }
    for field, value in expected_result.items():
        if result.get(field) != value:
            raise ContractError(f"prior v2 result has mismatched {field}")
    if closure_sha256 != SOURCE_CLOSURE_SHA256:
        raise ContractError("v3 static source closure differs from the v2 frozen closure")
    source_closure = _mapping(result.get("source_closure"), "v2 source closure")
    static_closure = _mapping(static.get("source_closure"), "v3 source closure")
    if source_closure != static_closure:
        raise ContractError("prior v2 result source closure does not match the frozen closure")
    namespace = _mapping(result.get("namespace"), "v2 namespace record")
    if namespace.get("sealed_inputs_opened") is not False:
        raise ContractError("prior v2 result does not prove sealed-input isolation")

    raw_rows = result.get("results")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(IMPORTED_COORDINATES):
        raise ContractError("prior v2 result must contain exactly 28 sealed rows")
    if any(not isinstance(row, dict) for row in raw_rows):
        raise ContractError("prior v2 result contains a malformed row")
    rows = cast("list[dict[str, object]]", raw_rows)
    _validate_rows(rows, IMPORTED_COORDINATES, require_complete=True)
    if rows[-1].get("record_sha256") != V2_LAST_RECORD_SHA256:
        raise ContractError("prior v2 last row seal does not match the frozen marker")

    summary = _mapping(result.get("summary"), "v2 summary")
    if (
        summary.get("study_evaluations_planned") != TOTAL_EVALUATION_BUDGET
        or summary.get("study_evaluations_completed") != len(IMPORTED_COORDINATES)
        or summary.get("run_status") != "RUNNING"
    ):
        raise ContractError("prior v2 partial summary is inconsistent with its 28 rows")
    profile_summaries = _mapping(summary.get("profiles"), "v2 profile summaries")
    for profile in PROFILE_ORDER:
        profile_summary = _mapping(profile_summaries.get(profile), f"v2 {profile} summary")
        if (
            profile_summary.get("completed_rows") != 14
            or profile_summary.get("all_seed_pairs_feasible") is not None
            or profile_summary.get("all_16_feasible") is not None
        ):
            raise ContractError(f"prior v2 partial summary is inconsistent for {profile}")

    provenance = _mapping(attempt.get("continuation_provenance"), "v2 attempt provenance")
    result_provenance = _mapping(
        result.get("continuation_provenance"), "v2 result provenance"
    )
    if provenance != result_provenance:
        raise ContractError("prior v2 attempt/result continuation provenance differs")
    expected_v1 = _V2_LEGACY_PROVENANCE()
    expected_links = {
        "prior_attempt_sha256": expected_v1["attempt_sha256"],
        "prior_result_sha256": expected_v1["result_sha256"],
        "prior_row_sha256": expected_v1["row_sha256"],
        "prior_freeze_sha256": expected_v1["freeze_sha256"],
        "imported_evaluation_count": 1,
        "remaining_evaluation_count": 31,
        "total_evaluation_count": TOTAL_EVALUATION_BUDGET,
    }
    for field, value in expected_links.items():
        if provenance.get(field) != value:
            raise ContractError(f"prior v2 continuation has mismatched {field}")
    imported_coordinates = provenance.get("imported_coordinates")
    if imported_coordinates != [
        {"history_seed": STUDY_SEEDS[0], "profile": PROFILE_ORDER[0]}
    ]:
        raise ContractError("prior v2 continuation changed the imported v1 coordinate")
    remaining = provenance.get("remaining_coordinates")
    expected_remaining = [
        {"history_seed": seed, "profile": profile}
        for seed, profile in ALL_COORDINATES[1:]
    ]
    if remaining != expected_remaining:
        raise ContractError("prior v2 continuation provenance changed its fixed coordinate order")

    v1 = _verify_v1_provenance(closure_sha256)
    if rows[0] != v1["row"]:
        raise ContractError("prior v2 imported v1 row differs from the immutable v1 row")
    return attempt, result, rows


def _verify_static_guards() -> dict[str, object]:
    static = _V2_VERIFY_STATIC_GUARDS()
    source_closure = _mapping(static.get("source_closure"), "source closure")
    source_closure_sha256 = str(static.get("source_closure_sha256", ""))
    if (
        len(source_closure) != 47
        or source_closure_sha256 != SOURCE_CLOSURE_SHA256
        or _sha256(_V2_CANONICAL_JSON(source_closure)) != SOURCE_CLOSURE_SHA256
    ):
        raise ContractError("v3 requires the unchanged frozen 47-file source/config closure")
    attempt, result, rows = _verify_v2_evidence(static)
    release = _release_binding()
    runner_sha256 = _V2_FILE_SHA256(Path(__file__).resolve())
    expected_release = {
        "runner_sha256": runner_sha256,
        "source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "historical_v1_provenance": _V2_LEGACY_PROVENANCE(),
        "v2_provenance": _expected_v2_provenance(),
    }
    if release != expected_release:
        raise ContractError("v3 contract release binding differs from the runner or evidence")
    historical_bytes = _V2_VERIFY_HISTORICAL_BYTES()
    if len(historical_bytes) != 4:
        raise ContractError("v1 historical evidence byte verification was incomplete")
    return {
        **static,
        "v2_attempt": attempt,
        "v2_result": result,
        "v2_rows": rows,
        "historical_v1_bytes": historical_bytes,
        "runner_sha256": runner_sha256,
    }


def _freeze_digest(source_closure: Mapping[str, str]) -> str:
    contract_path = ROOT / "experiments/sprint18-isolated-feasibility-study-v3.md"
    contract_sha256 = _V2_FILE_SHA256(contract_path)
    runner_sha256 = _V2_FILE_SHA256(Path(__file__).resolve())
    material = (
        b"SPRINT18-ISOLATED-FEASIBILITY-CONTINUATION-FREEZE-v3\n"
        + bytes.fromhex(contract_sha256)
        + bytes.fromhex(runner_sha256)
        + _V2_CANONICAL_JSON({
            "source_closure": dict(sorted(source_closure.items())),
            "historical_v1_provenance": _V2_LEGACY_PROVENANCE(),
            "v2_provenance": _expected_v2_provenance(),
        })
    )
    return _sha256(material)


def _assert_expected_freeze(expected: str, actual: str) -> None:
    if expected != actual:
        raise ContractError("expected freeze digest does not match the reviewed v3 freeze")


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _continuation_metadata() -> dict[str, object]:
    return {
        "prior_attempt_path": _path_label(V2_ATTEMPT_PATH),
        "prior_attempt_sha256": V2_ATTEMPT_SHA256,
        "prior_result_path": _path_label(V2_RESULT_PATH),
        "prior_result_sha256": V2_RESULT_SHA256,
        "prior_freeze_sha256": V2_FREEZE_SHA256,
        "prior_last_record_sha256": V2_LAST_RECORD_SHA256,
        "imported_coordinates": [
            {"history_seed": seed, "profile": profile}
            for seed, profile in IMPORTED_COORDINATES
        ],
        "remaining_coordinates": [
            {"history_seed": seed, "profile": profile}
            for seed, profile in REMAINING_COORDINATES
        ],
        "imported_evaluation_count": len(IMPORTED_COORDINATES),
        "remaining_evaluation_count": len(REMAINING_COORDINATES),
        "total_persisted_coordinate_budget": TOTAL_EVALUATION_BUDGET,
        "budget_interpretation": (
            "32 unique persisted coordinates; this is not a claim of 32 total "
            "computation starts. The interrupted v2 process may have started "
            "the unpersisted (918014, sprint15-v7) computation."
        ),
    }


def _record_result(
    status: str,
    results: list[dict[str, object]],
    static: Mapping[str, object],
    freeze_digest: str,
    output_dir: Path,
) -> Path:
    result_path = output_dir / "result.json"
    attempt_path = output_dir / "attempt.json"
    result: dict[str, object] = {
        "study_id": STUDY_ID,
        "amendment_id": V3_AMENDMENT_ID,
        "freeze_digest": freeze_digest,
        "prior_v1_freeze_digest": V1_FREEZE_SHA256,
        "prior_v2_freeze_digest": V2_FREEZE_SHA256,
        "status": status,
        "source_closure_sha256": static["source_closure_sha256"],
        "source_closure": static["source_closure"],
        "namespace": static["namespace"],
        "config_contract": static["config_contract"],
        "profile_order": list(PROFILE_ORDER),
        "seed_order": list(STUDY_SEEDS),
        "continuation_provenance": _continuation_metadata(),
        "results": results,
        "summary": _V2_SUMMARIZE(results, status, STUDY_SEEDS),
        "research_evidence_writes_only": [
            _path_label(attempt_path),
            _path_label(result_path),
        ],
        "generated_data_roots_written": False,
    }
    _V2_WRITE_JSON_ATOMIC(result_path, result)
    return result_path


def _run_continuation(
    static: Mapping[str, object],
    *,
    output_dir: Path,
    freeze_digest: str,
    expected_freeze: str,
    fixture_evaluator: Callable[[int, str, object], dict[str, object]] | None = None,
    fixture_only: bool = False,
) -> dict[str, object]:
    _assert_expected_freeze(expected_freeze, freeze_digest)
    if output_dir.exists():
        raise ContractError(f"one-shot v3 output already exists; no retry or resume: {output_dir}")
    if FORBIDDEN_ROOT.exists():
        raise ContractError(f"forbidden persistent data root exists: {FORBIDDEN_ROOT}")
    if fixture_only != (fixture_evaluator is not None):
        raise ContractError("disposable fixture mode requires only its fixture evaluator")
    if fixture_only:
        if output_dir.resolve().is_relative_to(ROOT):
            raise ContractError("disposable fixture output must be outside the project root")
    elif output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ContractError("study continuation may write only to the fixed v3 output directory")

    _, _, imported_rows = _verify_v2_evidence(static)
    results = list(imported_rows)
    _validate_rows(results, ALL_COORDINATES[:len(IMPORTED_COORDINATES)], require_complete=True)
    provenance = _continuation_metadata()
    started = datetime.now(timezone.utc).isoformat()
    attempt_path = _V2_CLAIM_ATTEMPT(output_dir, {
        "study_id": STUDY_ID,
        "amendment_id": V3_AMENDMENT_ID,
        "attempt": 1,
        "status": "RUNNING",
        "started_at_utc": started,
        "freeze_digest": freeze_digest,
        "prior_v1_freeze_digest": V1_FREEZE_SHA256,
        "prior_v2_freeze_digest": V2_FREEZE_SHA256,
        "source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "seed_order": list(STUDY_SEEDS),
        "profile_order": list(PROFILE_ORDER),
        "total_persisted_coordinate_budget": TOTAL_EVALUATION_BUDGET,
        "imported_evaluations": len(IMPORTED_COORDINATES),
        "new_evaluation_budget": len(REMAINING_COORDINATES),
        "completed_evaluations": len(IMPORTED_COORDINATES),
        "new_evaluations_completed": 0,
        "remaining_evaluations": len(REMAINING_COORDINATES),
        "last_record_sha256": V2_LAST_RECORD_SHA256,
        "continuation_provenance": provenance,
        "generated_data_roots_written": False,
    })

    def persist(status: str, block_reason: str | None = None) -> tuple[Path, str]:
        _validate_rows(results, ALL_COORDINATES)
        result_path = _record_result(status, results, static, freeze_digest, output_dir)
        result_sha256 = _V2_FILE_SHA256(result_path)
        completed = sum(row.get("evaluation_started") is True for row in results)
        new_completed = sum(
            row.get("evaluation_started") is True for row in results[len(IMPORTED_COORDINATES):]
        )
        marker: dict[str, object] = {
            "study_id": STUDY_ID,
            "amendment_id": V3_AMENDMENT_ID,
            "attempt": 1,
            "status": status,
            "started_at_utc": started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_digest": freeze_digest,
            "prior_v1_freeze_digest": V1_FREEZE_SHA256,
            "prior_v2_freeze_digest": V2_FREEZE_SHA256,
            "source_closure_sha256": SOURCE_CLOSURE_SHA256,
            "seed_order": list(STUDY_SEEDS),
            "profile_order": list(PROFILE_ORDER),
            "total_persisted_coordinate_budget": TOTAL_EVALUATION_BUDGET,
            "imported_evaluations": len(IMPORTED_COORDINATES),
            "new_evaluation_budget": len(REMAINING_COORDINATES),
            "completed_evaluations": completed,
            "new_evaluations_completed": new_completed,
            "remaining_evaluations": TOTAL_EVALUATION_BUDGET - completed,
            "last_record_sha256": results[-1]["record_sha256"],
            "result_sha256": result_sha256,
            "continuation_provenance": provenance,
            "generated_data_roots_written": False,
        }
        if block_reason is not None:
            marker["block_reason"] = block_reason
        if status == "COMPLETE":
            marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _V2_WRITE_JSON_ATOMIC(attempt_path, marker)
        return result_path, result_sha256

    persist("RUNNING")
    metric = _V2_METRIC_RECORD
    configs_by_seed: dict[int, dict[str, object]] = {}
    for seed, profile in REMAINING_COORDINATES:
        if seed not in configs_by_seed:
            try:
                configs_by_seed[seed] = _V2_RESOLVE_PAIR_CONFIGS(seed)
            except Exception as exc:
                blocked = _V2_SEAL_ROW({
                    "history_seed": seed,
                    "profile": profile,
                    "execution_status": "BLOCKED",
                    "evaluation_started": False,
                    "generation_writes": False,
                    "no_write": True,
                    "block_reason": "profile config resolution failed before evaluation",
                    "infrastructure_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                })
                results.append(blocked)
                persist("BLOCKED", str(blocked["block_reason"]))
                raise ContractError(f"v3 continuation blocked at {(seed, profile)}") from exc
        if fixture_evaluator is not None:
            row = fixture_evaluator(seed, profile, configs_by_seed[seed][profile])
            blocked_result = False
        else:
            raw_row, blocked_result = metric(seed, profile, configs_by_seed[seed][profile])
            row = raw_row
        sealed_row = _V2_SEAL_ROW(row)
        results.append(sealed_row)
        status = "BLOCKED" if blocked_result else "RUNNING"
        block_reason = str(row.get("block_reason")) if blocked_result else None
        persist(status, block_reason)
        if blocked_result or sealed_row.get("evaluation_started") is not True:
            raise ContractError(f"v3 continuation did not complete fixed coordinate {(seed, profile)}")

    if len(results) != TOTAL_EVALUATION_BUDGET:
        raise ContractError("v3 continuation did not persist exactly 32 unique coordinates")
    _validate_rows(results, ALL_COORDINATES, require_complete=True)
    result_path, result_sha256 = persist("COMPLETE")
    result = _read_object(result_path)
    summary = _mapping(result.get("summary"), "v3 final summary")
    if (
        summary.get("study_evaluations_planned") != TOTAL_EVALUATION_BUDGET
        or summary.get("study_evaluations_completed") != TOTAL_EVALUATION_BUDGET
        or summary.get("complete_seed_pairs") is not True
        or summary.get("complete_16_pairs") is not True
    ):
        raise ContractError("v3 final summary does not prove all 32 persisted evaluations")
    return {
        "status": "COMPLETE",
        "result": _path_label(result_path),
        "result_sha256": result_sha256,
        "attempt_sha256": _V2_FILE_SHA256(attempt_path),
        "freeze_digest": freeze_digest,
        "persisted_evaluations": TOTAL_EVALUATION_BUDGET,
        "imported_evaluations": len(IMPORTED_COORDINATES),
        "new_evaluations": len(REMAINING_COORDINATES),
        "summary": summary,
    }


def _verify_final_output(
    output_dir: Path,
    imported_rows: Sequence[Mapping[str, object]],
    fixture_rows_expected: int,
) -> dict[str, object]:
    result_path = output_dir / "result.json"
    attempt_path = output_dir / "attempt.json"
    result = _read_object(result_path)
    attempt = _read_object(attempt_path)
    raw_rows = result.get("results")
    if not isinstance(raw_rows, list) or any(not isinstance(row, dict) for row in raw_rows):
        raise ContractError("v3 output rows are malformed")
    rows = cast("list[dict[str, object]]", raw_rows)
    _validate_rows(rows, ALL_COORDINATES, require_complete=True)
    if rows[:len(IMPORTED_COORDINATES)] != list(imported_rows):
        raise ContractError("v3 output changed one or more imported v2 row objects")
    marked_fixture_rows = sum(row.get("disposable_fixture_projection") is True for row in rows)
    if marked_fixture_rows != fixture_rows_expected:
        raise ContractError("v3 fixture output has an unexpected number of fixture rows")
    summary = _mapping(result.get("summary"), "v3 final summary")
    provenance = _mapping(result.get("continuation_provenance"), "v3 final provenance")
    result_sha256 = _V2_FILE_SHA256(result_path)
    if (
        result.get("status") != "COMPLETE"
        or result.get("amendment_id") != V3_AMENDMENT_ID
        or result.get("generated_data_roots_written") is not False
        or summary.get("study_evaluations_planned") != TOTAL_EVALUATION_BUDGET
        or summary.get("study_evaluations_completed") != TOTAL_EVALUATION_BUDGET
        or summary.get("complete_seed_pairs") is not True
        or summary.get("complete_16_pairs") is not True
        or provenance.get("imported_evaluation_count") != len(IMPORTED_COORDINATES)
        or provenance.get("remaining_evaluation_count") != len(REMAINING_COORDINATES)
        or provenance.get("total_persisted_coordinate_budget") != TOTAL_EVALUATION_BUDGET
    ):
        raise ContractError("v3 final result summary/provenance is inconsistent")
    if (
        attempt.get("status") != "COMPLETE"
        or attempt.get("completed_evaluations") != TOTAL_EVALUATION_BUDGET
        or attempt.get("imported_evaluations") != len(IMPORTED_COORDINATES)
        or attempt.get("new_evaluations_completed") != len(REMAINING_COORDINATES)
        or attempt.get("remaining_evaluations") != 0
        or attempt.get("last_record_sha256") != rows[-1].get("record_sha256")
        or attempt.get("result_sha256") != result_sha256
        or attempt.get("continuation_provenance") != provenance
    ):
        raise ContractError("v3 attempt marker does not bind its final result")
    return {
        "result_sha256": result_sha256,
        "attempt_sha256": _V2_FILE_SHA256(attempt_path),
        "summary": summary,
        "rows": rows,
    }


def _expect_rejected(
    rejected: list[str], label: str, expected_message: str, action: Callable[[], object]
) -> None:
    try:
        action()
    except ContractError as exc:
        if expected_message not in str(exc):
            raise ContractError(
                f"smoke guard {label!r} failed for an unexpected reason: {exc}"
            ) from exc
        rejected.append(label)
    else:
        raise ContractError(f"smoke guard accepted invalid evidence: {label}")


def run_smoke(static: Mapping[str, object], freeze_digest: str) -> dict[str, object]:
    if OUTPUT_DIR.exists() or FORBIDDEN_ROOT.exists():
        raise ContractError("v3 output or generated-data root already exists; smoke refuses")
    historical_before = _V2_VERIFY_HISTORICAL_BYTES()
    v2_attempt_before = _V2_FILE_SHA256(V2_ATTEMPT_PATH)
    v2_result_before = _V2_FILE_SHA256(V2_RESULT_PATH)
    attempt, result, imported_rows = _verify_v2_evidence(static)
    rejected: list[str] = []
    _expect_rejected(
        rejected,
        "freeze mismatch",
        "does not match the reviewed v3 freeze",
        lambda: _assert_expected_freeze("0" * 64, freeze_digest),
    )
    _expect_rejected(
        rejected,
        "prior attempt digest mismatch",
        "attempt digest does not match",
        lambda: _verify_v2_evidence(
            static, expected_attempt_sha256="0" * 64
        ),
    )
    _expect_rejected(
        rejected,
        "reordered imported rows",
        "row/order mismatch",
        lambda: _validate_rows(list(reversed(imported_rows)), IMPORTED_COORDINATES),
    )
    duplicate_rows = [dict(row) for row in imported_rows]
    duplicate_rows[1] = dict(duplicate_rows[0])
    duplicate_coordinates = [
        (int(row["history_seed"]), str(row["profile"])) for row in duplicate_rows
    ]
    _expect_rejected(
        rejected,
        "duplicate persisted coordinate",
        "duplicate persisted coordinates",
        lambda: _validate_rows(duplicate_rows, duplicate_coordinates),
    )
    corrupt_rows = [dict(row) for row in imported_rows]
    corrupt_rows[0]["no_write"] = False
    _expect_rejected(
        rejected,
        "corrupt imported row seal",
        "invalid seal",
        lambda: _validate_rows(corrupt_rows, IMPORTED_COORDINATES),
    )
    _expect_rejected(
        rejected,
        "prior result digest mismatch",
        "result digest does not match",
        lambda: _verify_v2_evidence(static, expected_result_sha256="0" * 64),
    )

    with tempfile.TemporaryDirectory(prefix="s18-feasibility-v3-") as temp:
        temp_root = Path(temp)
        bad_attempt_path = temp_root / "bad-v2-attempt.json"
        bad_attempt = dict(attempt)
        bad_attempt["status"] = "COMPLETE"
        bad_attempt_path.write_text(
            json.dumps(bad_attempt, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        _expect_rejected(
            rejected,
            "completed prior attempt",
            "mismatched status",
            lambda: _verify_v2_evidence(
                static,
                attempt_path=bad_attempt_path,
                expected_attempt_sha256=_V2_FILE_SHA256(bad_attempt_path),
            ),
        )

        output_dir = temp_root / "disposable-v3-output"
        evaluator_calls: list[tuple[int, str]] = []
        templates = {
            profile: next(
                row for row in reversed(imported_rows) if row.get("profile") == profile
            )
            for profile in PROFILE_ORDER
        }

        def fixture_evaluator(seed: int, profile: str, config: object) -> dict[str, object]:
            del config
            evaluator_calls.append((seed, profile))
            fixture_row = dict(templates[profile])
            fixture_row["history_seed"] = seed
            fixture_row["disposable_fixture_projection"] = True
            fixture_row.pop("record_sha256", None)
            return fixture_row

        completion = _run_continuation(
            static,
            output_dir=output_dir,
            freeze_digest=freeze_digest,
            expected_freeze=freeze_digest,
            fixture_evaluator=fixture_evaluator,
            fixture_only=True,
        )
        if completion.get("status") != "COMPLETE":
            raise ContractError("disposable v3 continuation did not complete")
        if evaluator_calls != list(REMAINING_COORDINATES):
            raise ContractError("disposable fixture did not visit exactly the four missing coordinates")
        final = _verify_final_output(output_dir, imported_rows, len(REMAINING_COORDINATES))
        before_duplicate = (
            _V2_FILE_SHA256(output_dir / "attempt.json"),
            _V2_FILE_SHA256(output_dir / "result.json"),
        )
        _expect_rejected(
            rejected,
            "duplicate v3 output attempt",
            "one-shot v3 output already exists",
            lambda: _run_continuation(
                static,
                output_dir=output_dir,
                freeze_digest=freeze_digest,
                expected_freeze=freeze_digest,
                fixture_evaluator=fixture_evaluator,
                fixture_only=True,
            ),
        )
        after_duplicate = (
            _V2_FILE_SHA256(output_dir / "attempt.json"),
            _V2_FILE_SHA256(output_dir / "result.json"),
        )
        if before_duplicate != after_duplicate:
            raise ContractError("duplicate attempt changed disposable v3 evidence")

    if (
        historical_before != _V2_VERIFY_HISTORICAL_BYTES()
        or v2_attempt_before != _V2_FILE_SHA256(V2_ATTEMPT_PATH)
        or v2_result_before != _V2_FILE_SHA256(V2_RESULT_PATH)
    ):
        raise ContractError("v1/v2 historical evidence changed during the disposable smoke")
    if OUTPUT_DIR.exists() or FORBIDDEN_ROOT.exists():
        raise ContractError("disposable smoke created a persistent v3 output or data root")
    return {
        "status": "SMOKE-PASS",
        "imported_v2_rows": len(imported_rows),
        "fixture_evaluations": evaluator_calls,
        "persisted_coordinates": len(cast("list[dict[str, object]]", final["rows"])),
        "v2_attempt_sha256": v2_attempt_before,
        "v2_result_sha256": v2_result_before,
        "v2_last_record_sha256": V2_LAST_RECORD_SHA256,
        "fixture_result_sha256": final["result_sha256"],
        "fixture_attempt_sha256": final["attempt_sha256"],
        "rejected_guard_boundaries": rejected,
        "fixture_evaluator_calls": len(evaluator_calls),
        "generator_invoked": False,
        "generated_data_roots_written": False,
        "persistent_v3_output_created": False,
        "source_closure_sha256": static["source_closure_sha256"],
        "freeze_digest": freeze_digest,
        "prior_v2_attempt_status": attempt.get("status"),
        "prior_v2_result_status": result.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--continue", dest="continue_study", action="store_true")
    parser.add_argument(
        "--review-passed",
        action="store_true",
        help="acknowledge fresh independent Task 58 PASS before v3 continuation",
    )
    parser.add_argument(
        "--expected-freeze",
        help="required v3 freeze digest recorded in the reviewed Task 58 artifact",
    )
    args = parser.parse_args()
    if args.continue_study and not args.review_passed:
        parser.error("--continue requires --review-passed after fresh independent review")
    if args.continue_study and not args.expected_freeze:
        parser.error("--continue requires --expected-freeze from the reviewed Task 58 artifact")
    if args.smoke and (args.review_passed or args.expected_freeze is not None):
        parser.error("--review-passed/--expected-freeze are valid only with --continue")
    if args.review_passed and not args.continue_study:
        parser.error("--review-passed is valid only with --continue")

    static = _verify_static_guards()
    freeze_digest = _freeze_digest(cast("Mapping[str, str]", static["source_closure"]))
    if args.smoke:
        print(json.dumps(run_smoke(static, freeze_digest), indent=2, sort_keys=True))
        return 0
    if args.continue_study:
        _assert_expected_freeze(str(args.expected_freeze), freeze_digest)
        completion = _run_continuation(
            static,
            output_dir=OUTPUT_DIR,
            freeze_digest=freeze_digest,
            expected_freeze=str(args.expected_freeze),
        )
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0
    raise ContractError("select exactly one explicit runner mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
