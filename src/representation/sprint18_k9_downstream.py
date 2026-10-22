"""Sprint 18 Task 26 — K9-FULL Fit-only downstream scoring path (C7-A/C8-A/C9-B).

``K9-FULL`` is the exact Sprint 17 seven-member stack
``C2-A + C3-A + C5-A + C6-A + C7-A + C8-A + C9-B``. The representation graph,
optimizer, schedule and coherent checkpoints are already served by the
committed K dispatch (:mod:`representation.sprint18_k_dispatch`) and the
common Sprint 18 harness; what the published source still lacks is the
*Sprint 18 evaluation surface* for the three K9 downstream members:

- ``C8-A`` ``HuberStandardizer`` fitted on the K9 Fit forward residuals,
- ``C9-B`` contiguous-window ``window_max_mean`` aggregation over the C8-A
  patch energies  ->  ``S_pred``,
- ``C7-A`` ``RobustShrinkageReference`` fitted on the K9 Fit file embeddings
  ->  ``S_pop``.

The generic harness bank/scoring path (``fit_fit_reference_bank`` +
``score_certified_rows``) serves ``B0-FULL``/``K3-FULL``/``K7-FULL`` with the
``NormalReferenceBank`` + public-inference pattern. Reusing it for K9 would
leave the declared ``C7-A``/``C8-A``/``C9-B`` members inert and make
``K9-FULL`` indistinguishable from ``K7-FULL`` at scoring time, so K9 gets
its own role-enforced path here. The members themselves are never
re-implemented and never re-tuned: this module calls the committed Sprint 17
implementations (``representation.sprint17_c7``/``c8``/``c9``) through the
committed K dispatch, adds only the Sprint 18 role, checkpoint, support and
provenance enforcement, and keeps the two branches strictly independent (no
fusion, no shared threshold, no outcome-selected switch).

Frozen references honoured here (cited for review):

- Membership: ``experiments/sprint17-task23-confirmation-lock.json`` (accepted
  Sprint 17 K9 parents), ``experiments/sprint17-task19-combinations.json`` and
  ``src/representation/sprint17_combinations.py`` (``K_MEMBERS`` /
  ``MEMBER_IMPLEMENTATION``), the Task 14 lock
  ``experiments/sprint18-task14-config-lock.json`` (``K9-FULL`` component set)
  and the Sprint 18 arm registry
  (:data:`representation.sprint18_harness.ARM_REGISTRY`).
- Fit-only roles: Sprint 18 protocol v1 §3 (role table) and §5.3/§8.1 (Fit
  refits each arm's declared bank/standardizers from that arm's final Fit
  state; Calibration only sets q95; Development/Confirmation never refit), plus
  the harness role-action table (``fit_bank``/``fit_standardizer`` are FIT-only,
  ``calibrate_q95`` is CALIBRATION-only, ``score_development`` is
  DEVELOPMENT-only).
- Checkpoint binding: Sprint 18 protocol v1 §5.3 (coherent checkpoint rule;
  the epoch-100 checkpoint is the sole primary) and the harness strict restore
  identity check used by every other arm.

Fail-closed rules enforced by this module:

- only ``K9-FULL`` may use this path;
- the restored checkpoint identity must match arm, model seed, executable DATA
  binding, model bytes and EMA bytes (the same strictness as the generic
  bank/scoring path);
- the downstream members are fitted only on verified-healthy rows of the
  certified Fit histories, and in ``experiment`` scope only on the complete
  frozen three-history Fit support;
- scoring accepts only the declared role action for the requested support
  (``calibrate_q95`` on verified-healthy Calibration rows, ``score_development``
  on the full Development support, ``score_confirmation`` only under the
  frozen lock/review/release bindings);
- every emitted score is finite and both branches carry the exact declared
  common support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import torch

from representation.model import V1RepresentationModel
from representation.sprint17_c7 import RobustShrinkageReference
from representation.sprint17_c8 import HuberStandardizer
from representation.sprint18_checkpoint import RestoredSprint18Checkpoint
from representation.sprint18_harness import (
    ArmId,
    CertifiedSample,
    HarnessScores,
    Sprint18ArmConfig,
    Task6Binding,
    _assert_restored_checkpoint,
    _canonical_sha256,
    _tensor_digest,
    make_common_support,
    validate_arm_config,
    validate_role_permission,
    validate_score_maps,
)
from representation.sprint18_k_dispatch import (
    fit_k9_downstream as _dispatch_fit_k9_downstream,
)
from representation.sprint18_k_dispatch import (
    k9_forward_latents as _dispatch_k9_forward_latents,
)
from representation.sprint18_k_dispatch import (
    score_k9_files as _dispatch_score_k9_files,
)
from synth.patchify import Patchifier

#: Scope vocabulary shared with the Sprint 18 harness.
K9Scope = Literal["experiment", "task6_smoke_only"]

#: Scoring actions this path may serve (role-permission checked per action).
K9ScoringAction = Literal["calibrate_q95", "score_development", "score_confirmation"]

#: The arm identity served by this module.
K9_ARM_ID = "K9-FULL"

#: Frozen C5-A channel-time block masking policy identity used by every K9
#: collation (representation upstream, downstream fitting and downstream
#: scoring) — the same policy the K dispatch applies for K3/K7/K9.
K9_MASKING_POLICY_ID = "c5a-channel-time-block-v1"

__all__ = [
    "K9_ARM_ID",
    "K9_MASKING_POLICY_ID",
    "K9DownstreamProvenance",
    "ProvenancedK9Downstream",
    "fit_k9_downstream_members",
    "load_k9_downstream",
    "save_k9_downstream",
    "score_k9_certified_rows",
]


@dataclass(frozen=True, slots=True)
class K9DownstreamProvenance:
    """Fit-only provenance of the fitted K9 downstream members."""

    arm_id: ArmId
    model_seed: int
    source_role: str
    source_history_ids: tuple[str, ...]
    source_manifest_sha256: tuple[str, ...]
    source_rows: int
    source_rows_sha256: str
    data_binding_sha256: str
    checkpoint_sha256: str
    standardizer_sha256: str
    reference_sha256: str
    window_sha256: str
    masking_policy: str
    scope: K9Scope


@dataclass(frozen=True, slots=True)
class ProvenancedK9Downstream:
    """The fitted K9 downstream members plus their Fit-only provenance."""

    standardizer: HuberStandardizer
    reference: RobustShrinkageReference
    window: Mapping[str, object]
    fit_ids: tuple[str, ...]
    provenance: K9DownstreamProvenance


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardizer_state_digest(standardizer: HuberStandardizer) -> str:
    """Digest the fitted C8-A center/scale state (never re-implemented)."""
    center = standardizer.center
    scale = standardizer.scale
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("C8-A standardizer is not fitted")
    return _canonical_sha256(
        {"center": _tensor_digest(center), "scale": _tensor_digest(scale)}
    )


def _reference_state_digest(reference: RobustShrinkageReference) -> str:
    """Digest the fitted C7-A reference state (embeddings, cells, statistics).

    The frozen C7-A implementation keeps its fitted statistics in private
    instance attributes; this digest reads that state read-only instead of
    re-deriving or re-implementing any C7-A statistic.
    """
    state = vars(reference)
    embeddings = state.get("_embeddings")
    cells = state.get("_cells")
    stats = state.get("_stats")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(cells, dict) or not isinstance(stats, dict):
        raise ValueError("C7-A reference is not fitted")
    payload = {
        "embeddings": _tensor_digest(embeddings),
        "n_total": int(state.get("_n_total", 0)),
        "cells": {
            f"{robot}:{program}": [int(index) for index in members]
            for (robot, program), members in sorted(cells.items())
        },
        "stats": {
            key: [_tensor_digest(location), _tensor_digest(cholesky)]
            for key, (location, cholesky) in sorted(stats.items())
        },
    }
    return _canonical_sha256(payload)


def _fit_rows_digest(rows: Sequence[CertifiedSample]) -> str:
    ordered = sorted(rows, key=lambda row: (row.actual_history_id, row.file_id))
    return _canonical_sha256(
        [
            {
                "history_id": row.actual_history_id,
                "manifest_sha256": row.manifest_sha256,
                "file_id": row.file_id,
                "shard_sha256": row.shard_sha256,
            }
            for row in ordered
        ]
    )


def fit_k9_downstream_members(
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    rows: Sequence[CertifiedSample],
    patchifier: Patchifier,
    *,
    checkpoint: RestoredSprint18Checkpoint,
    scope: K9Scope,
    device: torch.device | str | None = None,
) -> ProvenancedK9Downstream:
    """Fit the exact K9 downstream members on the arm's own Fit state only.

    ``C8-A`` is fitted on the K9 Fit forward residuals and ``C7-A`` on the K9
    Fit file embeddings, both from the arm's restored final checkpoint; the
    upstream model is never mutated (eval + no-grad probes only). Every input
    row must be a verified-healthy row of a certified Fit history, and in
    ``experiment`` scope the complete frozen three-history Fit support is
    required. ``C9-B`` contributes no fitted state (its frozen window duration
    is carried as provenance).
    """
    if arm.arm_id != K9_ARM_ID:
        raise ValueError("the K9 downstream path serves only the K9-FULL arm")
    validate_arm_config(arm, require_locked=(scope == "experiment"))
    checkpoint_sha256 = _assert_restored_checkpoint(
        model, arm, binding, checkpoint, scope=scope
    )
    if not rows:
        raise ValueError("K9 downstream fitting requires a non-empty Fit support")
    fit_histories = {item.actual_history_id for item in binding.role_histories("FIT")}
    file_ids = [row.file_id for row in rows]
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("K9 downstream fitting support contains duplicate file IDs")
    for row in rows:
        if row.actual_history_id not in fit_histories:
            raise ValueError("K9 downstream fitting rows include a non-Fit history")
        if not row.is_verified_healthy:
            raise ValueError("K9 downstream fitting accepts verified-healthy Fit rows only")
        history = validate_role_permission(
            binding, row.actual_history_id, "fit_bank", healthy=True
        )
        validate_role_permission(
            binding, row.actual_history_id, "fit_standardizer", healthy=True
        )
        if row.manifest_sha256 != history.manifest_sha256:
            raise ValueError("K9 downstream fitting row is not bound to its certified manifest")
    histories = {row.actual_history_id for row in rows}
    if scope == "experiment":
        if histories != fit_histories or len(rows) != binding.fit_healthy_rows:
            raise ValueError(
                "experiment K9 downstream fitting requires the complete frozen three-history Fit support"
            )
    fitted = _dispatch_fit_k9_downstream(
        model,
        patchifier,
        model.config,
        [(row.file_id, row.sample) for row in rows],
        device=device,
    )
    standardizer = fitted["standardizer"]
    reference = fitted["reference"]
    if not isinstance(standardizer, HuberStandardizer):
        raise ValueError("K9 downstream standardizer is not a fitted C8-A HuberStandardizer")
    if not isinstance(reference, RobustShrinkageReference):
        raise ValueError("K9 downstream reference is not a fitted C7-A RobustShrinkageReference")
    window = dict(fitted["window"])
    provenance = K9DownstreamProvenance(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        source_role="FIT",
        source_history_ids=tuple(sorted(histories)),
        source_manifest_sha256=tuple(
            sorted({row.manifest_sha256 for row in rows})
        ),
        source_rows=len(rows),
        source_rows_sha256=_fit_rows_digest(rows),
        data_binding_sha256=binding.executable_binding_sha256,
        checkpoint_sha256=checkpoint_sha256,
        standardizer_sha256=_standardizer_state_digest(standardizer),
        reference_sha256=_reference_state_digest(reference),
        window_sha256=_canonical_sha256(window),
        masking_policy=K9_MASKING_POLICY_ID,
        scope=scope,
    )
    return ProvenancedK9Downstream(
        standardizer=standardizer,
        reference=reference,
        window=MappingProxyType(window),
        fit_ids=tuple(row.file_id for row in rows),
        provenance=provenance,
    )


def score_k9_certified_rows(
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    model: V1RepresentationModel,
    downstream: ProvenancedK9Downstream,
    rows: Sequence[CertifiedSample],
    patchifier: Patchifier,
    *,
    checkpoint: RestoredSprint18Checkpoint,
    action: K9ScoringAction,
    scope: K9Scope,
    device: torch.device | str | None = None,
    confirmation_lock_sha256: str | None = None,
    confirmation_review_sha256: str | None = None,
    main_confirmation_release: bool = False,
) -> HarnessScores:
    """Score declared rows with the fitted K9 downstream members (no fusion).

    ``S_pred`` is the C9-B contiguous-window aggregation over the C8-A Huber
    patch energies and ``S_pop`` is the C7-A robust-shrinkage reference
    energy on the K9 file embeddings. Both branches are returned on the exact
    canonical common support with independent provenance digests, so the
    unchanged harness ``fit_calibration_q95`` can set one q95 threshold per
    branch.
    """
    if arm.arm_id != K9_ARM_ID:
        raise ValueError("the K9 downstream path serves only the K9-FULL arm")
    validate_arm_config(arm, require_locked=(scope == "experiment"))
    checkpoint_sha256 = _assert_restored_checkpoint(
        model, arm, binding, checkpoint, scope=scope
    )
    provenance = downstream.provenance
    if (
        provenance.arm_id != arm.arm_id
        or provenance.model_seed != arm.model_seed
        or provenance.checkpoint_sha256 != checkpoint_sha256
        or provenance.data_binding_sha256 != binding.executable_binding_sha256
        or provenance.scope != scope
    ):
        raise ValueError(
            "K9 downstream state does not match the requested arm/seed/checkpoint/binding/scope"
        )
    if not rows:
        raise ValueError("K9 scoring support cannot be empty")
    history_ids = {row.actual_history_id for row in rows}
    if len(history_ids) != 1:
        raise ValueError("one scoring support cannot pool multiple histories")
    actual_history_id = next(iter(history_ids))
    healthy_only = action == "calibrate_q95"
    history = validate_role_permission(
        binding,
        actual_history_id,
        action,
        healthy=all(row.is_verified_healthy for row in rows) if healthy_only else False,
        confirmation_lock_sha256=confirmation_lock_sha256,
        confirmation_review_sha256=confirmation_review_sha256,
        main_confirmation_release=main_confirmation_release,
    )
    if any(row.manifest_sha256 != history.manifest_sha256 for row in rows):
        raise ValueError("K9 scoring support contains rows from a different manifest")
    if healthy_only and any(not row.is_verified_healthy for row in rows):
        raise ValueError("Calibration support must be verified healthy")
    if healthy_only and scope == "experiment" and len(rows) != binding.calibration_healthy_rows:
        raise ValueError(
            "experiment q95 scoring requires the complete frozen Calibration-healthy support"
        )
    ids = [row.file_id for row in rows]
    support = make_common_support(binding, actual_history_id, ids)
    latents = _dispatch_k9_forward_latents(
        model,
        patchifier,
        model.config,
        [(row.file_id, row.sample) for row in rows],
        device=device,
    )
    predicted, population, _ = _dispatch_score_k9_files(
        latents,
        {
            "standardizer": downstream.standardizer,
            "reference": downstream.reference,
            "window": dict(downstream.window),
        },
    )
    branches = {
        "S_pred": MappingProxyType({key: float(value) for key, value in predicted.items()}),
        "S_pop": MappingProxyType({key: float(value) for key, value in population.items()}),
    }
    validate_score_maps(support, branches)
    branch_provenance = {
        "S_pred": _canonical_sha256(
            {
                "branch": "S_pred",
                "score_map": dict(branches["S_pred"]),
                "support_sha256": support.support_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "standardizer_sha256": provenance.standardizer_sha256,
                "window_sha256": provenance.window_sha256,
                "masking_policy": provenance.masking_policy,
                "role_manifest_sha256": history.manifest_sha256,
            }
        ),
        "S_pop": _canonical_sha256(
            {
                "branch": "S_pop",
                "score_map": dict(branches["S_pop"]),
                "support_sha256": support.support_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "reference_sha256": provenance.reference_sha256,
                "masking_policy": provenance.masking_policy,
                "role_manifest_sha256": history.manifest_sha256,
            }
        ),
    }
    if branch_provenance["S_pred"] == branch_provenance["S_pop"]:
        raise ValueError("branch provenance digests unexpectedly alias")
    return HarnessScores(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        actual_history_id=actual_history_id,
        schema_history_id=history.schema_history_id,
        role_manifest_sha256=history.manifest_sha256,
        support=support,
        scores=MappingProxyType(branches),
        checkpoint_sha256=checkpoint_sha256,
        bank_sha256=provenance.reference_sha256,
        data_binding_sha256=binding.executable_binding_sha256,
        branch_provenance_sha256=MappingProxyType(branch_provenance),
        scope=scope,
    )


def save_k9_downstream(path: str | Path, downstream: ProvenancedK9Downstream) -> str:
    """Persist the fitted K9 downstream state so later phases never refit it.

    Returns the SHA-256 of the written artifact.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "s18-k9-downstream-state-v1",
        "standardizer": downstream.standardizer,
        "reference": downstream.reference,
        "window": dict(downstream.window),
        "fit_ids": list(downstream.fit_ids),
        "provenance": asdict(downstream.provenance),
    }
    torch.save(payload, target)
    return _sha256_file(target)


def load_k9_downstream(path: str | Path) -> ProvenancedK9Downstream:
    """Load a previously persisted K9 downstream state (never refitted)."""
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "s18-k9-downstream-state-v1":
        raise ValueError("K9 downstream state artifact schema is not recognised")
    standardizer = payload.get("standardizer")
    reference = payload.get("reference")
    if not isinstance(standardizer, HuberStandardizer) or not isinstance(reference, RobustShrinkageReference):
        raise ValueError("K9 downstream state artifact does not carry the frozen C8-A/C7-A members")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("K9 downstream state artifact is missing its provenance")
    return ProvenancedK9Downstream(
        standardizer=standardizer,
        reference=reference,
        window=MappingProxyType(dict(payload.get("window", {}))),
        fit_ids=tuple(payload.get("fit_ids", ())),
        provenance=K9DownstreamProvenance(**provenance),
    )
