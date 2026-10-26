"""Weekly/monthly study execution admission — cache preflight (S24-7-M4).

This module implements the part of the frozen ``S24-BIZ-WM-v1`` execution task
that must happen **before any movement value exists**: verify the frozen study
and manifest digests, compile the exact per-file embedding row plan, prove the
accepted G2 capture identity, inspect declared embedding-cache stores for
identity-compatible per-file rows, and fail closed when the required coverage is
absent or invalid.

Binding boundaries:

- **No movement is computed here.** Nothing in this module evaluates a
  distribution distance, a cluster occupancy, a mean/dispersion component, a rank
  association, a bootstrap interval or a disposition. Endpoint computation is
  gated behind :func:`admit_execution`, which this module publishes rather than
  waives.
- **Reuse is identity-strict.** A cached row counts only when its
  ``s24_feature_cache_identity`` record matches the study-required identity for
  that ``(history, representation seed)`` group exactly, including the capture
  device. A near-match (for example the accepted G2 identity that omitted the 13
  uncaptured ``DEVELOPMENT-01`` window-1 files) is reported and never counted.
- **Missing or invalid cache is a data-availability blocker, not a negative
  scientific result.** The gate raises :class:`CachePreflightFailure`; callers
  record the blocker and compute no outcome.
- **Zero new forwards.** Nothing here loads a model, a checkpoint, a waveform or
  a score, or performs an inference step. The accepted G2 capture identity is
  re-derived from recorded digests only.

Declared execution-side cache contract consumed by :func:`inspect_cache_store`
(this contract exists so a cache can be consumed at all; no producer of it
exists in the accepted lineage, which is exactly why the accepted preflight
fails closed at zero rows):

- ``<root>/<history_id>__seed-<model_seed>.npz`` with arrays ``file_ids``
  (Unicode strings) and ``embeddings`` (``float32``, shape ``[n, 128]``);
- ``<root>/<history_id>__seed-<model_seed>.json`` sidecar carrying the exact
  cache-identity inputs, ``cache_identity`` and ``n_rows``.

A root that contains ``.npz`` files without a matching sidecar, a malformed
sidecar, a digest-inconsistent identity, non-finite embeddings, duplicate file
IDs or any identity field that differs from the study-required input fails the
gate as an invalid cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from representation import sprint24_business_horizon as horizon
from representation import sprint24_weekly_monthly_study as study
from representation.sprint24_context_audit import sha256_object
from representation.sprint24_features import s24_feature_cache_identity

#: Execution-record identifier: the frozen study plus this admission contract.
EXECUTION_ID = "S24-BIZ-WM-v1|EXECUTION-ADMISSION-v1"

#: Accepted G2 execution commit that produced the reusable-capture premise.
ACCEPTED_G2_EXECUTION_COMMIT = "24e8d6bd82e83e23f708514eb5b17539951efc0e"

#: Accepted G2 runner source commit (preflight ``source_commit``).
ACCEPTED_G2_SOURCE_COMMIT = "f50f7c6de66181728153d4963c19db103b3372ff"

#: Accepted G2 diagnostic protocol document digest.
ACCEPTED_G2_PROTOCOL_SHA256 = "10c53edb30b0eaa257744bd5123f4fb6cbddb1b6e33f65b3afd8b8a005293f20"

#: Accepted G2 runner content digest recorded by ``state.json``.
ACCEPTED_G2_RUNNER_SHA256 = "b9914f7e3473f6e05ed960f109a3f1309be9e8579c19c68a5b7fefd72ae04248"

#: Accepted Sprint 18 executable binding digest used by the G2 capture.
ACCEPTED_BINDING_SHA256 = "383c62114ee6120d5fd19f866dfd585083d1d5899c9ffe91804761bd8a10a2d5"

#: Preprocessing/input-contract digest of the frozen B0 bundle. Re-derived from
#: the frozen ``V1Config`` defaults (6 channels, patch 32, stride 16, ``knn_k`` 5)
#: and verified against all twelve recorded G2 cache identities.
ACCEPTED_PREPROCESSING_SHA256 = (
    "903a4702c57719769d194d89820566d0bd3fd0ee2ba44e3d87e5dbc011cb97ef"
)

#: Capture device identity bound into every accepted G2 cache identity.
REQUIRED_DEVICE_IDENTITY = "cuda"

#: Accepted G2 review-sized artifacts and their byte digests. A transferred copy
#: must match all of them before any identity claim is made.
ACCEPTED_G2_ARTIFACTS: dict[str, str] = {
    "preflight.json": "26f9e45d26af66fa17f37782278699d95ed61e24e674f5f9aed5432883e426df",
    "state.json": "e174e67e3c79bf5bfb9748d705bafe7c43585a82a33f4b4d3896866db5175478",
    "costs.json": "6c0ff27b65561d761ee0c41b0c70c62ade77a2bd52c8218e26d6c5d5fe2cfd09",
    "calibration.json": "1fb6a15713ec426130113c360efb76a9f3fd711607666bbbe88ee3e9951e3eec",
    "runner-sha256.txt": "783c8e7a587648d30e0ac0b80a06697cfc6ffbe00567f2a0f2b6f8cef4814e70",
    "tables/window-statistics.jsonl": (
        "054ea5cb9177451b58081f75e57597fd8c5acf6162d1d131168202ce79eb87b8"
    ),
    "tables/controls.jsonl": (
        "f7d1747452638c1039f5e1a58e568feced3dd43fc193a84d23312088987af816"
    ),
    "tables/bootstrap.json": (
        "0d7cd09d3bb4c47dad34007133be1ec0872f4232efba6c82bfb840c5e56313be"
    ),
    "manifests/S18I-ITER-0005-DEVELOPMENT-01/window-reference-manifest.json": (
        "ce529e963cc0abf653797fb58599dc8eed1e5be4c1d09605745550c57d834f47"
    ),
    "manifests/S18I-ITER-0005-DEVELOPMENT-02/window-reference-manifest.json": (
        "075d0a4ab63d4099e241adbe209e1b0a38fe43db063ec402ddc6344fb68e4e33"
    ),
    "manifests/S18I-ITER-0005-DEVELOPMENT-03/window-reference-manifest.json": (
        "51ca69fdf9e8fe7fafce87c6e2d92f03763e695e09b983f0da7f49bf7fcd065e"
    ),
    "manifests/S18I-ITER-0005-DEVELOPMENT-04/window-reference-manifest.json": (
        "de1803da96f477f367d887cd3d65b1ebfcf5397840a3f2903ce6c4a96cbb1bd0"
    ),
}

#: Per-history eligible analysis-span row counts declared by the frozen study.
DECLARED_SPAN_ROWS: dict[str, int] = {
    "S18I-ITER-0005-DEVELOPMENT-01": 1_226,
    "S18I-ITER-0005-DEVELOPMENT-02": 1_353,
    "S18I-ITER-0005-DEVELOPMENT-03": 1_118,
    "S18I-ITER-0005-DEVELOPMENT-04": 1_379,
}

#: Per-history causal reference row counts declared by the frozen study.
DECLARED_REFERENCE_ROWS: dict[str, int] = {
    "S18I-ITER-0005-DEVELOPMENT-01": 850,
    "S18I-ITER-0005-DEVELOPMENT-02": 974,
    "S18I-ITER-0005-DEVELOPMENT-03": 935,
    "S18I-ITER-0005-DEVELOPMENT-04": 919,
}

#: Every cache field the sidecar must carry so the identity can be re-derived.
CACHE_SIDECAR_FIELDS: tuple[str, ...] = (
    "schema",
    "kind",
    "history_id",
    "model_seed",
    "device",
    "checkpoint_sha256",
    "bank_sha256",
    "support_sha256",
    "preprocessing_sha256",
    "arm_config_sha256",
    "data_binding_sha256",
    "scale_state_sha256",
    "cache_identity",
    "n_rows",
    "capture_provenance",
    "capture_commit",
)

#: Sidecar schema/kind labels of the declared execution-side cache contract.
CACHE_SIDECAR_SCHEMA = "s24-file-embedding-cache-v1"
CACHE_SIDECAR_KIND = "feature-cache-store"

#: Capture provenance label of a full GPU forward capture. Only this provenance is
#: admissible as study input.
CACHE_PROVENANCE_GPU = "gpu-forward"

#: Capture provenance label of the bounded GPU smoke capture. It proves the path,
#: never the study support, so it is inadmissible as study input.
CACHE_PROVENANCE_SMOKE = "gpu-forward-smoke"

#: Capture provenance label of a local fixture that performs no model forward. It
#: exists to exercise the writer/admission plumbing and is never study input.
CACHE_PROVENANCE_FIXTURE = "fixture-no-forward"

#: Provenance values the admission gate may accept.
CACHE_ADMISSIBLE_PROVENANCE: tuple[str, ...] = (CACHE_PROVENANCE_GPU,)

#: Execution status of a declared study cell that never ran.
NOT_EXECUTED_STATUS = "NOT-EXECUTED-CACHE-PREFLIGHT-FAILED"

#: Execution status of a declared cell whose cache rows were verified but whose
#: endpoint computation is outside this attempt's scope.
ADMITTED_STATUS = "ADMITTED-PENDING-ENDPOINT-COMPUTATION"

#: Decision-rule criteria that stay unevaluated when no observation exists.
DECISION_CRITERIA: tuple[str, ...] = (
    "A.estimability",
    "B.promise-magnitude",
    "C.promise-consistency",
    "D.promise-sensitivity",
    "N1.nonpromise-magnitude",
    "N2.nonpromise-consistency",
    "N3.nonpromise-sensitivity",
)


class CachePreflightFailure(RuntimeError):
    """Raised when the study's required embedding rows are not reusable.

    The exception carries the complete preflight report so a caller can persist
    the blocker without recomputing anything.
    """

    def __init__(self, message: str, *, report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.report = dict(report)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


# --- accepted G2 records and identity ----------------------------------------


def read_accepted_g2(root: Path) -> dict[str, object]:
    """Load and digest-verify the accepted G2 review-sized records.

    Every transferred file must match :data:`ACCEPTED_G2_ARTIFACTS` byte for byte;
    ``state.json`` must agree with the pinned runner/source/device facts. The
    parsed records are returned so later steps can re-derive the capture identity
    without opening a model or a waveform.
    """
    records: dict[str, object] = {}
    digests: dict[str, object] = {}
    for relative, expected in ACCEPTED_G2_ARTIFACTS.items():
        path = root.joinpath(*relative.split("/"))
        if not path.is_file():
            raise RuntimeError(f"accepted G2 artifact missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"accepted G2 artifact digest mismatch for {relative}: "
                f"{observed} != {expected}"
            )
        digests[relative] = {"sha256": observed, "bytes": path.stat().st_size}
    preflight = json.loads(root.joinpath("preflight.json").read_text(encoding="utf-8"))
    state = json.loads(root.joinpath("state.json").read_text(encoding="utf-8"))
    costs = json.loads(root.joinpath("costs.json").read_text(encoding="utf-8"))
    calibration = json.loads(root.joinpath("calibration.json").read_text(encoding="utf-8"))
    runner_declared = root.joinpath("runner-sha256.txt").read_text(encoding="utf-8").strip()
    windows = [
        json.loads(line)
        for line in root.joinpath("tables", "window-statistics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    manifests = {
        history_id: json.loads(
            root.joinpath(
                "manifests", history_id, "window-reference-manifest.json"
            ).read_text(encoding="utf-8")
        )
        for history_id in sorted(DECLARED_SPAN_ROWS)
    }

    _require(
        str(preflight.get("protocol")) == "S24-GEO-TIME-v1",
        "accepted G2 preflight protocol changed",
    )
    _require(
        str(preflight.get("protocol_sha256")) == ACCEPTED_G2_PROTOCOL_SHA256,
        "accepted G2 protocol digest changed",
    )
    _require(
        str(preflight.get("binding_sha256")) == ACCEPTED_BINDING_SHA256,
        "accepted G2 binding digest changed",
    )
    _require(
        str(preflight.get("source_commit")) == ACCEPTED_G2_SOURCE_COMMIT,
        "accepted G2 preflight source commit changed",
    )
    device = preflight.get("device")
    _require(isinstance(device, Mapping), "accepted G2 device record is malformed")
    _require(
        str(device.get("identity")) == REQUIRED_DEVICE_IDENTITY,  # type: ignore[union-attr]
        "accepted G2 capture device identity is not cuda",
    )
    _require(
        int(preflight.get("forwards_per_seed", -1)) == study.DECLARED_MAX_ROWS_PER_SEED,
        "accepted G2 per-seed row count differs from the frozen envelope",
    )
    _require(
        int(preflight.get("forwards_total_3seeds", -1)) == study.DECLARED_MAX_ROWS_TOTAL,
        "accepted G2 total row count differs from the frozen envelope",
    )
    _require(
        str(state.get("runner_sha256")) == ACCEPTED_G2_RUNNER_SHA256,
        "accepted G2 runner digest changed",
    )
    _require(
        runner_declared == ACCEPTED_G2_RUNNER_SHA256,
        "accepted G2 runner digest file disagrees with the pinned runner",
    )
    _require(
        str(state.get("source_commit")) == ACCEPTED_G2_SOURCE_COMMIT,
        "accepted G2 state source commit changed",
    )
    _require(str(state.get("device")) == REQUIRED_DEVICE_IDENTITY, "accepted G2 state device changed")
    _require(str(state.get("state")) == "COMPLETE", "accepted G2 run is not COMPLETE")
    seeds = [str(seed) for seed in study.REPRESENTATION_SEEDS]
    for field in ("checkpoints", "banks", "arms"):
        table = preflight.get(field)
        _require(isinstance(table, Mapping), f"accepted G2 {field} table is malformed")
        _require(
            sorted(str(key) for key in table) == seeds,  # type: ignore[union-attr]
            f"accepted G2 {field} table does not cover the accepted seeds",
        )
    support = preflight.get("support")
    _require(isinstance(support, Mapping), "accepted G2 support table is malformed")
    for history_id, expected_rows in DECLARED_REFERENCE_ROWS.items():
        entry = support[history_id]  # type: ignore[index]
        _require(
            int(entry["n_ref"]) == expected_rows,
            f"accepted G2 reference count changed for {history_id}",
        )
        _require(
            int(entry["n_eval"]) == DECLARED_SPAN_ROWS[history_id],
            f"accepted G2 analysis-span count changed for {history_id}",
        )
    return {
        "root": str(root),
        "digests": digests,
        "preflight": preflight,
        "state": state,
        "costs": costs,
        "calibration": calibration,
        "window_statistics": windows,
        "manifests": manifests,
    }


def identity_inputs(record: Mapping[str, object]) -> dict[str, object]:
    """Return the frozen cache-identity inputs recorded by the accepted G2 run."""
    preflight = record["preflight"]
    assert isinstance(preflight, Mapping)
    return {
        "checkpoints": {
            str(seed): str(preflight["checkpoints"][str(seed)])  # type: ignore[index]
            for seed in study.REPRESENTATION_SEEDS
        },
        "banks": {
            str(seed): str(preflight["banks"][str(seed)])  # type: ignore[index]
            for seed in study.REPRESENTATION_SEEDS
        },
        "arms": {
            str(seed): str(preflight["arms"][str(seed)])  # type: ignore[index]
            for seed in study.REPRESENTATION_SEEDS
        },
        "binding_sha256": ACCEPTED_BINDING_SHA256,
        "preprocessing_sha256": ACCEPTED_PREPROCESSING_SHA256,
        "device": REQUIRED_DEVICE_IDENTITY,
        "scale_state_sha256": None,
    }


def cache_identity(
    inputs: Mapping[str, object],
    *,
    model_seed: int,
    support_sha256: str,
) -> str:
    """Return the identity a cache row must carry to be reusable by the study."""
    checkpoints = inputs["checkpoints"]
    banks = inputs["banks"]
    arms = inputs["arms"]
    assert isinstance(checkpoints, Mapping) and isinstance(banks, Mapping) and isinstance(arms, Mapping)
    scale_state = inputs["scale_state_sha256"]
    return s24_feature_cache_identity(
        checkpoint_sha256=str(checkpoints[str(model_seed)]),
        bank_sha256=str(banks[str(model_seed)]),
        support_sha256=support_sha256,
        preprocessing_sha256=str(inputs["preprocessing_sha256"]),
        arm_config_sha256=str(arms[str(model_seed)]),
        data_binding_sha256=str(inputs["binding_sha256"]),
        model_seed=int(model_seed),
        scale_state_sha256=None if scale_state is None else str(scale_state),
        device=str(inputs["device"]),
    )


# --- required row plan and identities ----------------------------------------


def required_row_plan(
    manifests: Mapping[str, Mapping[str, object]],
    edges: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Compile the exact per-file rows the frozen study needs per representation seed.

    Reference rows are the inherited causal pool (eligible files ending at or
    before each history's ``c1``); analysis-span rows are the eligible files in
    ``(c1, tmax]``. The union per history is exactly the support a reusable cache
    entry must cover, and its canonical digest is the identity's support digest.
    """
    per_history: dict[str, dict[str, object]] = {}
    total_reference = 0
    total_span = 0
    total_rows = 0
    for history_id in sorted(manifests):
        if history_id not in edges:
            raise KeyError(f"missing frozen decile edges for {history_id}")
        edge_values = [float(value) for value in edges[history_id]]
        if len(edge_values) < 5:
            raise ValueError(f"decile edge table too short for {history_id}")
        observable = horizon.project_observable(manifests[history_id])
        meta = horizon.prepare_history(history_id, observable, edge_values[4])
        reference_index = np.flatnonzero(meta.eligible & (meta.ends <= meta.c1))
        span_index = np.flatnonzero(
            meta.eligible & (meta.ends > meta.c1) & (meta.ends <= meta.tmax)
        )
        reference_ids = sorted(meta.file_ids[int(position)] for position in reference_index)
        span_ids = sorted(meta.file_ids[int(position)] for position in span_index)
        union_ids = sorted(set(reference_ids) | set(span_ids))
        reference = horizon.reference_pool(meta)
        per_history[history_id] = {
            "history_id": history_id,
            "c1_rel_s": meta.c1,
            "tmax_rel_s": meta.tmax,
            "n_reference_rows": len(reference_ids),
            "n_span_rows": len(span_ids),
            "n_rows": len(union_ids),
            "reference_state": str(reference["reference_state"]),
            "reference_digest": str(reference["reference_digest"]),
            "span_ids_sha256": sha256_object(span_ids),
            "union_ids_sha256": sha256_object(union_ids),
            "reference_ids": reference_ids,
            "span_ids": span_ids,
            "required_file_ids": union_ids,
        }
        total_reference += len(reference_ids)
        total_span += len(span_ids)
        total_rows += len(union_ids)
    return {
        "grain": "one row per required (history, file) per representation seed",
        "per_history": per_history,
        "totals": {
            "reference_rows_per_seed": total_reference,
            "span_rows_per_seed": total_span,
            "rows_per_seed": total_rows,
            "rows_total_three_seeds": total_rows * len(study.REPRESENTATION_SEEDS),
            "declared_max_rows_per_seed": study.DECLARED_MAX_ROWS_PER_SEED,
            "declared_max_rows_total": study.DECLARED_MAX_ROWS_TOTAL,
            "matches_declared_envelope": total_rows == study.DECLARED_MAX_ROWS_PER_SEED,
            "matches_declared_span": total_span == sum(DECLARED_SPAN_ROWS.values()),
            "matches_declared_reference": total_reference == sum(DECLARED_REFERENCE_ROWS.values()),
            "representation_seeds": list(study.REPRESENTATION_SEEDS),
            "seeds_are_independent_data": False,
        },
    }


def g2_capture_union(
    plan: Mapping[str, object],
    g2_manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Return the exact file set the accepted G2 run captured, per history.

    The accepted capture was ``reference ids + all SUPPORTED window members``; an
    UNSUPPORTED window's members were never captured, which is why the accepted
    identity is not automatically the study-required identity.
    """
    per_history = plan["per_history"]
    assert isinstance(per_history, Mapping)
    out: dict[str, object] = {}
    for history_id, block in sorted(per_history.items()):
        assert isinstance(block, Mapping)
        reference_ids = [str(value) for value in block["reference_ids"]]  # type: ignore[index]
        manifest = g2_manifests[history_id]
        supported = [window for window in manifest["windows"] if window["status"] == "SUPPORTED"]
        captured = list(reference_ids)
        for window in supported:
            captured.extend(str(value) for value in window["member_ids"])
        captured_sorted = sorted(set(captured))
        out[history_id] = {
            "n_captured": len(captured_sorted),
            "capture_ids_sha256": sha256_object(captured_sorted),
            "n_supported_windows": len(supported),
            "n_unsupported_windows": len(manifest["windows"]) - len(supported),
            "n_uncaptured_study_rows": len(set(block["required_file_ids"]) - set(captured_sorted)),  # type: ignore[arg-type]
            "uncaptured_study_rows": sorted(
                set(block["required_file_ids"]) - set(captured_sorted)  # type: ignore[arg-type]
            ),
        }
    return {
        "rule": "reference ids + every SUPPORTED window's member ids",
        "per_history": out,
    }


def required_cache_identities(
    plan: Mapping[str, object],
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """Return the identity every reusable cache group must carry."""
    per_history = plan["per_history"]
    assert isinstance(per_history, Mapping)
    groups: dict[str, dict[str, object]] = {}
    for history_id, block in sorted(per_history.items()):
        assert isinstance(block, Mapping)
        support_sha256 = str(block["union_ids_sha256"])
        for seed in study.REPRESENTATION_SEEDS:
            identity = cache_identity(inputs, model_seed=seed, support_sha256=support_sha256)
            groups[f"{history_id}|{seed}"] = {
                "history_id": history_id,
                "model_seed": int(seed),
                "required_rows": int(block["n_rows"]),
                "support_sha256": support_sha256,
                "cache_identity": identity,
                "device": REQUIRED_DEVICE_IDENTITY,
                "preprocessing_sha256": str(inputs["preprocessing_sha256"]),
                "data_binding_sha256": str(inputs["binding_sha256"]),
                "scale_state_sha256": "none",
            }
    return {"groups": groups, "n_groups": len(groups)}


def accepted_capture_identities(
    record: Mapping[str, object],
    plan: Mapping[str, object],
    inputs: Mapping[str, object],
    capture: Mapping[str, object],
) -> dict[str, object]:
    """Re-derive the accepted capture identity and compare it with its record.

    The accepted window-statistics rows record one identity per
    ``(history, seed)``. Recomputing it from the recorded provenance inputs and the
    reconstructed capture union proves the identity derivation end to end without
    opening the run's features. The comparison also exposes which groups' accepted
    identity is *not* the study-required identity.
    """
    windows = record["window_statistics"]
    assert isinstance(windows, list)
    recorded: dict[str, set[str]] = {}
    for row in windows:
        assert isinstance(row, Mapping)
        if str(row.get("status")) != "SUPPORTED":
            continue
        key = f"{row['history_id']}|{row['encoder_seed']}"
        recorded.setdefault(key, set()).add(str(row["cache_identity"]))
    capture_blocks = capture["per_history"]
    assert isinstance(capture_blocks, Mapping)
    groups: dict[str, dict[str, object]] = {}
    for key, values in sorted(recorded.items()):
        history_id, seed_text = key.split("|")
        block = capture_blocks[history_id]
        assert isinstance(block, Mapping)
        recomputed = cache_identity(
            inputs,
            model_seed=int(seed_text),
            support_sha256=str(block["capture_ids_sha256"]),
        )
        groups[key] = {
            "history_id": history_id,
            "model_seed": int(seed_text),
            "recorded_identity": sorted(values)[0],
            "recorded_distinct_values": len(values),
            "recomputed_identity": recomputed,
            "recomputed_matches_record": len(values) == 1 and recomputed == sorted(values)[0],
            "captured_rows": int(block["n_captured"]),
        }
    return {
        "n_groups": len(groups),
        "all_recorded_identities_reproduced": all(
            bool(entry["recomputed_matches_record"]) for entry in groups.values()
        ),
        "groups": groups,
    }


# --- declared cache store -----------------------------------------------------


def write_cache_entry(
    root: Path,
    *,
    history_id: str,
    model_seed: int,
    file_ids: Sequence[str],
    embeddings: np.ndarray,
    inputs: Mapping[str, object],
    support_sha256: str,
    provenance: str,
    capture_commit: str | None,
) -> dict[str, object]:
    """Persist one ``(history, model seed)`` embedding group under the declared contract.

    The writer refuses malformed data before touching the destination: duplicate or
    empty file IDs, a wrong dtype/shape, non-finite values, an unknown provenance
    label, or a full-capture claim without a 40-character execution commit. It
    returns the written paths, the re-derived identity and the custody digests.
    """
    if provenance not in (
        CACHE_PROVENANCE_GPU,
        CACHE_PROVENANCE_SMOKE,
        CACHE_PROVENANCE_FIXTURE,
    ):
        raise ValueError(f"unknown capture provenance: {provenance!r}")
    if provenance == CACHE_PROVENANCE_GPU:
        if capture_commit is None or not re.fullmatch(r"[0-9a-f]{40}", capture_commit):
            raise ValueError("a full GPU capture requires its 40-character execution commit")
    elif capture_commit is not None:
        raise ValueError("only a full GPU capture may record an execution commit")
    ids = [str(value) for value in file_ids]
    if not ids:
        raise ValueError("a cache entry requires at least one file ID")
    if any(not value for value in ids):
        raise ValueError("cache file IDs must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("cache file IDs must be unique")
    matrix = np.asarray(embeddings)
    if matrix.dtype != np.float32:
        raise ValueError(f"embeddings must be float32, got {matrix.dtype}")
    if matrix.ndim != 2 or matrix.shape[1] != study.EMBEDDING_DIM:
        raise ValueError(
            f"embeddings must be [n, {study.EMBEDDING_DIM}], got {matrix.shape}"
        )
    if matrix.shape[0] != len(ids):
        raise ValueError("embeddings rows differ from the file-ID count")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    if not re.fullmatch(r"[0-9a-f]{64}", support_sha256):
        raise ValueError("support_sha256 must be a lowercase SHA-256 digest")
    identity = cache_identity(inputs, model_seed=model_seed, support_sha256=support_sha256)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{history_id}__seed-{model_seed}"
    npz_path = root / f"{stem}.npz"
    sidecar_path = root / f"{stem}.json"
    np.savez(npz_path, file_ids=np.array(ids, dtype=np.str_), embeddings=matrix)
    sidecar = {
        "schema": CACHE_SIDECAR_SCHEMA,
        "kind": CACHE_SIDECAR_KIND,
        "history_id": history_id,
        "model_seed": int(model_seed),
        "device": str(inputs["device"]),
        "checkpoint_sha256": str(inputs["checkpoints"][str(model_seed)]),  # type: ignore[index]
        "bank_sha256": str(inputs["banks"][str(model_seed)]),  # type: ignore[index]
        "support_sha256": support_sha256,
        "preprocessing_sha256": str(inputs["preprocessing_sha256"]),
        "arm_config_sha256": str(inputs["arms"][str(model_seed)]),  # type: ignore[index]
        "data_binding_sha256": str(inputs["binding_sha256"]),
        "scale_state_sha256": "none" if inputs["scale_state_sha256"] is None else str(inputs["scale_state_sha256"]),
        "cache_identity": identity,
        "n_rows": len(ids),
        "capture_provenance": provenance,
        "capture_commit": capture_commit,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "npz_path": str(npz_path),
        "sidecar_path": str(sidecar_path),
        "cache_identity": identity,
        "support_sha256": support_sha256,
        "n_rows": len(ids),
        "provenance": provenance,
        "npz_sha256": sha256_file(npz_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }


def _load_sidecar(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sidecar is not a JSON object")
    missing = [field for field in CACHE_SIDECAR_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"sidecar is missing fields: {missing}")
    return payload


def _entry_from_path(root: Path, npz_path: Path) -> dict[str, object]:
    issues: list[str] = []
    sidecar_path = npz_path.with_suffix(".json")
    entry: dict[str, object] = {
        "path": str(npz_path),
        "sidecar_path": str(sidecar_path),
        "root": str(root),
        "valid": False,
        "issues": issues,
        "n_rows": 0,
        "file_ids": [],
    }
    if not sidecar_path.is_file():
        issues.append("missing identity sidecar")
        return entry
    try:
        sidecar = _load_sidecar(sidecar_path)
    except (OSError, ValueError) as exc:
        issues.append(f"unreadable sidecar: {exc}")
        return entry
    entry.update(
        {
            "history_id": sidecar["history_id"],
            "model_seed": sidecar["model_seed"],
            "device": sidecar["device"],
            "support_sha256": sidecar["support_sha256"],
            "cache_identity": sidecar["cache_identity"],
            "declared_n_rows": sidecar["n_rows"],
            "capture_provenance": sidecar["capture_provenance"],
            "capture_commit": sidecar["capture_commit"],
        }
    )
    provenance = str(sidecar["capture_provenance"])
    if provenance not in CACHE_ADMISSIBLE_PROVENANCE:
        issues.append(
            f"capture provenance {provenance!r} is not admissible study input"
        )
    capture_commit = sidecar["capture_commit"]
    if provenance == CACHE_PROVENANCE_GPU:
        if not isinstance(capture_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", capture_commit
        ):
            issues.append(
                "a full GPU capture must record its 40-character execution commit"
            )
    elif capture_commit is not None:
        issues.append("non-GPU provenance must not record an execution commit")
    if str(sidecar["schema"]) != CACHE_SIDECAR_SCHEMA:
        issues.append(f"unexpected sidecar schema {sidecar['schema']!r}")
    if str(sidecar["kind"]) != CACHE_SIDECAR_KIND:
        issues.append(f"unexpected sidecar kind {sidecar['kind']!r}")
    try:
        recomputed = s24_feature_cache_identity(
            checkpoint_sha256=str(sidecar["checkpoint_sha256"]),
            bank_sha256=str(sidecar["bank_sha256"]),
            support_sha256=str(sidecar["support_sha256"]),
            preprocessing_sha256=str(sidecar["preprocessing_sha256"]),
            arm_config_sha256=str(sidecar["arm_config_sha256"]),
            data_binding_sha256=str(sidecar["data_binding_sha256"]),
            model_seed=int(sidecar["model_seed"]),  # type: ignore[arg-type]
            scale_state_sha256=(
                None
                if str(sidecar["scale_state_sha256"]) == "none"
                else str(sidecar["scale_state_sha256"])
            ),
            device=str(sidecar["device"]),
        )
    except (TypeError, ValueError) as exc:
        issues.append(f"identity inputs are invalid: {exc}")
        recomputed = None
    if recomputed is not None and recomputed != str(sidecar["cache_identity"]):
        issues.append("sidecar cache_identity does not match its own identity inputs")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            names = set(payload.files)
            if not {"file_ids", "embeddings"} <= names:
                issues.append(f"npz is missing arrays: {sorted({'file_ids', 'embeddings'} - names)}")
                return entry
            file_ids = [str(value) for value in np.asarray(payload["file_ids"]).tolist()]
            embeddings = np.asarray(payload["embeddings"])
    except (OSError, ValueError) as exc:
        issues.append(f"unreadable npz: {exc}")
        return entry
    if embeddings.dtype != np.float32:
        issues.append(f"embeddings dtype is {embeddings.dtype}, expected float32")
    if embeddings.ndim != 2 or embeddings.shape[1] != study.EMBEDDING_DIM:
        issues.append(
            f"embeddings shape is {embeddings.shape}, expected [n, {study.EMBEDDING_DIM}]"
        )
    elif embeddings.shape[0] != len(file_ids):
        issues.append(
            f"embeddings rows {embeddings.shape[0]} differ from file_ids {len(file_ids)}"
        )
    elif not np.isfinite(embeddings).all():
        issues.append("embeddings contain non-finite values")
    if len(set(file_ids)) != len(file_ids):
        issues.append("file_ids contain duplicates")
    if int(sidecar["n_rows"]) != len(file_ids):  # type: ignore[arg-type]
        issues.append(
            f"sidecar n_rows {sidecar['n_rows']} differs from npz rows {len(file_ids)}"
        )
    entry.update(
        {
            "n_rows": len(file_ids),
            "file_ids": file_ids,
            "file_ids_sha256": sha256_object(sorted(file_ids)),
            "valid": not issues,
            "issues": issues,
        }
    )
    return entry


def inspect_cache_store(root: Path) -> dict[str, object]:
    """Inspect one declared cache store and classify every artifact in it."""
    entries: list[dict[str, object]] = []
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "npz_count": 0,
            "entries": [],
            "valid_entries": 0,
            "invalid_entries": 0,
            "total_valid_rows": 0,
            "note": "declared root does not exist; it contributes no rows",
        }
    for npz_path in sorted(root.glob("*.npz")):
        entries.append(_entry_from_path(root, npz_path))
    valid = [entry for entry in entries if entry["valid"]]
    return {
        "root": str(root),
        "exists": True,
        "npz_count": len(entries),
        "entries": entries,
        "valid_entries": len(valid),
        "invalid_entries": len(entries) - len(valid),
        "total_valid_rows": sum(int(entry["n_rows"]) for entry in valid),
    }


def inspect_cache_stores(roots: Sequence[Path]) -> dict[str, object]:
    """Inspect every declared cache root and aggregate the inventory."""
    stores = [inspect_cache_store(Path(root)) for root in roots]
    entries = [entry for store in stores for entry in store["entries"]]  # type: ignore[union-attr]
    valid = [entry for entry in entries if entry["valid"]]
    return {
        "roots": [str(root) for root in roots],
        "stores": stores,
        "npz_count": sum(int(store["npz_count"]) for store in stores),  # type: ignore[arg-type]
        "valid_entries": len(valid),
        "invalid_entries": len(entries) - len(valid),
        "total_valid_rows": sum(int(entry["n_rows"]) for entry in valid),
        "entries": entries,
    }


# --- coverage and admission ---------------------------------------------------


def cache_coverage(
    plan: Mapping[str, object],
    identities: Mapping[str, object],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Report per-group required/available/missing rows under identity-strict matching.

    A row is available only when a *valid* cache entry carries the study-required
    identity for that group and lists that file ID. Entries with a different
    identity (including the accepted G2 identity for the group it under-covered)
    contribute nothing.
    """
    groups_spec = identities["groups"]
    assert isinstance(groups_spec, Mapping)
    per_history = plan["per_history"]
    assert isinstance(per_history, Mapping)
    entries = inventory["entries"]
    assert isinstance(entries, list)
    valid_entries = [entry for entry in entries if entry["valid"]]
    groups: dict[str, dict[str, object]] = {}
    per_seed: dict[str, dict[str, int]] = {
        str(seed): {"required_rows": 0, "available_rows": 0, "missing_rows": 0}
        for seed in study.REPRESENTATION_SEEDS
    }
    total_required = 0
    total_available = 0
    total_missing = 0
    for key, spec in sorted(groups_spec.items()):
        assert isinstance(spec, Mapping)
        history_id = str(spec["history_id"])
        seed_key = str(spec["model_seed"])
        block = per_history[history_id]
        assert isinstance(block, Mapping)
        required_ids = {str(value) for value in block["required_file_ids"]}  # type: ignore[index]
        matching = [
            entry
            for entry in valid_entries
            if str(entry["history_id"]) == history_id
            and int(entry["model_seed"]) == int(spec["model_seed"])
            and str(entry["cache_identity"]) == str(spec["cache_identity"])
        ]
        available: set[str] = set()
        for entry in matching:
            available.update(str(value) for value in entry["file_ids"])  # type: ignore[index]
        available &= required_ids
        missing = sorted(required_ids - available)
        total_required += len(required_ids)
        total_available += len(available)
        total_missing += len(missing)
        per_seed[seed_key]["required_rows"] += len(required_ids)
        per_seed[seed_key]["available_rows"] += len(available)
        per_seed[seed_key]["missing_rows"] += len(missing)
        groups[key] = {
            "history_id": history_id,
            "model_seed": int(spec["model_seed"]),
            "required_rows": len(required_ids),
            "required_support_sha256": str(spec["support_sha256"]),
            "required_cache_identity": str(spec["cache_identity"]),
            "matching_valid_entries": len(matching),
            "available_rows": len(available),
            "missing_rows": len(missing),
            "coverage_complete": not missing,
            "missing_ids_sample": missing[:20],
            "missing_ids_sha256": sha256_object(missing),
            "near_miss_entries": [
                {
                    "path": str(entry["path"]),
                    "cache_identity": str(entry["cache_identity"]),
                    "support_sha256": str(entry["support_sha256"]),
                    "n_rows": int(entry["n_rows"]),
                    "reason": "identity differs from the study-required identity",
                }
                for entry in valid_entries
                if str(entry["history_id"]) == history_id
                and int(entry["model_seed"]) == int(spec["model_seed"])
                and str(entry["cache_identity"]) != str(spec["cache_identity"])
            ],
        }
    return {
        "identity_rule": (
            "a cached row is reusable only when a valid entry carries the exact "
            "study-required cache identity (checkpoint, bank, support, preprocessing, "
            "arm, data binding, model seed, scale state and capture device)"
        ),
        "groups": groups,
        "n_groups": len(groups),
        "groups_complete": sum(1 for value in groups.values() if value["coverage_complete"]),
        "per_seed": per_seed,
        "total_required_rows": total_required,
        "total_available_rows": total_available,
        "total_missing_rows": total_missing,
        "coverage_complete": total_missing == 0 and total_required > 0,
        "coverage_fraction": (total_available / total_required) if total_required else 0.0,
    }


def admit_execution(
    coverage: Mapping[str, object],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Return the admission decision; raise when the required rows are not reusable.

    The gate is strict: incomplete coverage blocks admission, and so does any
    malformed cache artifact or any declared cache root that could not be
    inspected. The card also names the exact blocker so no caller can treat a
    missing cache as a scientific outcome.
    """
    blockers: list[str] = []
    invalid_entries = int(inventory["invalid_entries"])
    if invalid_entries:
        blockers.append(f"{invalid_entries} invalid cache artifact(s) present")
    missing_groups = [
        key
        for key, value in coverage["groups"].items()  # type: ignore[union-attr]
        if not value["coverage_complete"]
    ]
    if missing_groups:
        blockers.append(
            f"{len(missing_groups)} of {coverage['n_groups']} identity groups have no "
            f"complete reusable row set"
        )
    if int(coverage["total_required_rows"]) == 0:
        blockers.append("the required row plan is empty")
    complete = not blockers and bool(coverage["coverage_complete"])
    card = {
        "status": "ADMITTED" if complete else "REFUSED",
        "complete": complete,
        "blockers": blockers,
        "required_rows": int(coverage["total_required_rows"]),
        "available_rows": int(coverage["total_available_rows"]),
        "missing_rows": int(coverage["total_missing_rows"]),
        "coverage_fraction": float(coverage["coverage_fraction"]),
        "movement_computed": False,
        "forwards_executed": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "checkpoint_loads": 0,
        "outcomes_computed": False,
        "dispositions_computed": [],
        "scope": (
            "cache admission and custody preflight only; endpoint computation is not "
            "released by this attempt and no movement value exists"
        ),
    }
    if not complete:
        raise CachePreflightFailure(
            "the frozen study's required embedding rows are not reusable: "
            + "; ".join(blockers),
            report=card,
        )
    return card


# --- declared execution ledger ------------------------------------------------


def execution_ledger(
    metadata_coverage: Mapping[str, object],
    admission: Mapping[str, object],
    reason: str,
) -> dict[str, object]:
    """Publish every declared cell, contrast and disposition with its true state.

    With a refused admission every cell is reported as not executed and every
    decision criterion as not evaluated: unknown is never presented as a pass, and
    no disposition is manufactured from a data-availability blocker.
    """
    cells = metadata_coverage["cells"]
    assert isinstance(cells, list)
    ledger_cells = [
        {
            "horizon": cell["horizon"],
            "mode": cell["mode"],
            "mode_role": cell["mode_role"],
            "endpoint_id": cell["endpoint_id"],
            "endpoint_role": cell["endpoint_role"],
            "decision_relevant": bool(cell["decision_relevant"]),
            "representation_seed": cell["representation_seed"],
            "history_id": cell["history_id"],
            "metadata_precheck_status": cell["status"],
            "n_comparable_pairs": cell["n_comparable_pairs"],
            "execution_status": (
                ADMITTED_STATUS if admission["complete"] else NOT_EXECUTED_STATUS
            ),
            "outcome": None,
        }
        for cell in cells
    ]
    contrasts = metadata_coverage["contrasts"]
    assert isinstance(contrasts, list)
    ledger_contrasts = [
        {
            "id": contrast["id"],
            "name": contrast["name"],
            "declared_state": contrast["state"],
            "execution_status": (
                ADMITTED_STATUS if admission["complete"] else NOT_EXECUTED_STATUS
            ),
            "outcome": None,
            "detail": contrast["note"],
        }
        for contrast in contrasts
    ]
    status_counts: dict[str, int] = {}
    for row in ledger_cells:
        status = str(row["execution_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    dispositions = {
        horizon_name: {
            "horizon": horizon_name,
            "disposition": None,
            "status": "NOT-COMPUTED",
            "reason": reason,
            "criteria": [
                {
                    "id": criterion_id,
                    "status": "NOT-EVALUATED",
                    "detail": (
                        "no observation was produced: the cache preflight failed "
                        "closed before any movement value; unknown is never a pass"
                    ),
                }
                for criterion_id in DECISION_CRITERIA
            ],
            "estimable_histories": [],
            "median_rho_burden": None,
            "median_rho_residual": None,
            "sensitivity_values": {
                variant: None for variant in study.DECISION_SENSITIVITY_VARIANTS
            },
            "resolution": None,
            "specificity_claim": False,
            "specificity_reason": (
                "no calibrated specificity is claimable: zero annotated-clean and zero "
                "quiet-period pairs exist at week or month granularity on the accepted "
                "Development span"
            ),
            "blocked_claims": list(study.BLOCKED_CLAIMS),
            "scaling_status": "NOT-RELEASED",
        }
        for horizon_name in study.HORIZONS
    }
    return {
        "study_id": study.STUDY_ID,
        "study_sha256": study.study_sha256(),
        "execution_id": EXECUTION_ID,
        "declared_cell_count": metadata_coverage["declared_cell_count"],
        "n_cells": len(ledger_cells),
        "cell_execution_status_counts": status_counts,
        "cells": ledger_cells,
        "contrasts": ledger_contrasts,
        "contrast_execution_status_counts": {
            row["id"]: row["execution_status"] for row in ledger_contrasts
        },
        "dispositions": dispositions,
        "decisions_note": (
            "no PROMISING / NOT-PROMISING / INCONCLUSIVE disposition is computed: the "
            "cache preflight failed closed before outcomes, and a blocked preflight is "
            "a data-availability blocker rather than a negative scientific result"
        ),
    }


# --- cost and custody ---------------------------------------------------------


def cost_record(plan: Mapping[str, object], coverage: Mapping[str, object]) -> dict[str, object]:
    """Return the zero-new-forward cost and envelope accounting of this attempt."""
    totals = plan["totals"]
    assert isinstance(totals, Mapping)
    envelope = study.assert_cost_within_declared_bounds(
        int(totals["span_rows_per_seed"]), int(totals["reference_rows_per_seed"])
    )
    return {
        "study_id": study.STUDY_ID,
        "study_sha256": study.study_sha256(),
        "policy": study.cost_policy(),
        "accounting": envelope,
        "required_rows": {
            "per_seed": int(totals["rows_per_seed"]),
            "total_three_seeds": int(totals["rows_total_three_seeds"]),
            "per_seed_breakdown": dict(coverage["per_seed"]),  # type: ignore[arg-type]
        },
        "available_rows": {
            "per_seed": max(
                int(entry["available_rows"])
                for entry in coverage["per_seed"].values()  # type: ignore[union-attr]
            ),
            "per_seed_breakdown": {
                seed: entry["available_rows"]
                for seed, entry in coverage["per_seed"].items()  # type: ignore[union-attr]
            },
            "total_three_seeds": int(coverage["total_available_rows"]),
        },
        "missing_rows": {
            "per_seed": max(
                int(entry["missing_rows"])
                for entry in coverage["per_seed"].values()  # type: ignore[union-attr]
            ),
            "per_seed_breakdown": {
                seed: entry["missing_rows"]
                for seed, entry in coverage["per_seed"].items()  # type: ignore[union-attr]
            },
            "total_three_seeds": int(coverage["total_missing_rows"]),
        },
        "forwards_executed_by_this_attempt": 0,
        "model_loads": 0,
        "waveform_reads": 0,
        "embedding_scores_read": 0,
        "statistical_work_only": True,
        "inherited_cumulative_forwards": study.INHERITED_CUMULATIVE_FORWARDS,
        "inherited_forward_cap": study.INHERITED_FORWARD_CAP,
        "inherited_headroom_forwards": study.INHERITED_HEADROOM_FORWARDS,
        "cap_reset": False,
    }


def custody_record(
    inputs: Mapping[str, Path],
    record: Mapping[str, object],
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Return the digest custody of every input and inspected cache artifact."""
    files = {
        str(relative): {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for relative, path in sorted(inputs.items())
        if path.is_file()
    }
    return {
        "study_id": study.STUDY_ID,
        "study_sha256": study.study_sha256(),
        "inputs": files,
        "accepted_g2": {
            "execution_commit": ACCEPTED_G2_EXECUTION_COMMIT,
            "source_commit": ACCEPTED_G2_SOURCE_COMMIT,
            "runner_sha256": ACCEPTED_G2_RUNNER_SHA256,
            "artifacts": record["digests"],
        },
        "cache_artifacts": [
            {
                "path": str(entry["path"]),
                "sidecar_path": str(entry["sidecar_path"]),
                "n_rows": int(entry["n_rows"]),
                "valid": bool(entry["valid"]),
            }
            for entry in inventory["entries"]  # type: ignore[union-attr]
        ],
        "no_source_mutation": True,
        "no_remote_write": True,
    }


__all__ = [
    "ACCEPTED_BINDING_SHA256",
    "ACCEPTED_G2_ARTIFACTS",
    "ACCEPTED_G2_EXECUTION_COMMIT",
    "ACCEPTED_G2_PROTOCOL_SHA256",
    "ACCEPTED_G2_RUNNER_SHA256",
    "ACCEPTED_G2_SOURCE_COMMIT",
    "ACCEPTED_PREPROCESSING_SHA256",
    "ADMITTED_STATUS",
    "CACHE_SIDECAR_FIELDS",
    "CACHE_SIDECAR_KIND",
    "CACHE_SIDECAR_SCHEMA",
    "CachePreflightFailure",
    "DECISION_CRITERIA",
    "DECLARED_REFERENCE_ROWS",
    "DECLARED_SPAN_ROWS",
    "EXECUTION_ID",
    "NOT_EXECUTED_STATUS",
    "REQUIRED_DEVICE_IDENTITY",
    "accepted_capture_identities",
    "admit_execution",
    "cache_coverage",
    "cache_identity",
    "cost_record",
    "custody_record",
    "execution_ledger",
    "g2_capture_union",
    "identity_inputs",
    "inspect_cache_store",
    "inspect_cache_stores",
    "read_accepted_g2",
    "required_cache_identities",
    "required_row_plan",
    "sha256_file",
]
