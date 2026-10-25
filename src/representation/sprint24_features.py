"""Frozen B0/N4 feature extraction and score comparators (Sprint 24 Task 5).

Thin wrapper over the existing public loader/inference/extraction paths. It
adds no new modeling, fitting, or bank construction:

- B0 checkpoint restore: :func:`restore_sprint18_checkpoint` (strict,
  ``scope="experiment"``) with the Task14-locked arm config.
- B0 bank restore: custody ``bank.pt`` envelope bytes are digest-verified and
  attached to a :class:`NormalReferenceBank` **without calling
  ``fit``** (no refit, no fallback recipe, no alternate order).
- B0 scoring: :func:`score_certified_rows` (``action="score_development"``,
  ``scope="experiment"``) for the authoritative ``S_pred``/``S_pop`` maps.
- File embeddings: the identical frozen forward through
  :class:`RepresentationInference` with the identical deterministic per-file
  mask seed, cross-checked for exact ``S_pred``/``S_pop`` agreement with the
  harness maps.
- N4 scoring: :func:`n4_score_files` on the frozen B0 encoder plus the
  strictly restored final scale envelope. ``S_pop`` is never recomputed by
  N4; the restored B0 bank map is carried through unchanged.

Fail-fast contract: any digest mismatch, seed/bank/scaler mismatch, empty or
duplicated support, or non-finite output raises ``ValueError``. There is no
random fallback, no checkpoint/bank refit, and no Confirmation contact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from representation.crn import (
    ContextResidualScaleModel,
    load_n4_scale_checkpoint,
    n4_score_files,
)
from representation.data import collate_variable_files
from representation.inference import NormalReferenceBank, RepresentationInference
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import (
    RestoredSprint18Checkpoint,
    restore_sprint18_checkpoint,
    state_sha256,
)
from representation.sprint18_harness import (
    MODEL_SEEDS,
    BankProvenance,
    CertifiedSample,
    HarnessScores,
    ProvenancedBank,
    Sprint18ArmConfig,
    Task6Binding,
    _arm_scoring_masking_policy,
    _canonical_sha256,
    _encoder_batch,
    _tensor_digest,
    arm_config_sha256,
    deterministic_mask_seed,
    make_common_support,
    score_certified_rows,
    validate_arm_config,
)
from representation.sprint18_supplementary import (
    BANK_ARTIFACT_SCHEMA_ID,
    BANK_ORDER_ID,
    BANK_RECIPE_ID,
)

#: Schema identity for every T5 feature/cache record.
S24_FEATURES_SCHEMA_ID = "s24-frozen-features-v1"

#: Frozen per-file inference mask-seed rule (Sprint 18 scoring convention).
S24_MASK_RULE_ID = "sprint18-score-mask-v2"

#: N4 scoring with no eligible masked valid patch is a hard failure.
S24_N4_EMPTY_PATCHES_ERROR = "N4 scoring file has no eligible masked valid patches"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    return value


def s24_preprocessing_digest(*, model: V1RepresentationModel, bank_k: int) -> str:
    """Digest the frozen preprocessing/input contract for cache identity.

    Binds patch geometry, collation padding, the frozen B0 mask composition
    (no arm-specific policy), the deterministic score-mask rule, and the
    reference-bank width ``k``. Per-file mask seeds are derived
    deterministically from ``(seed, history, file)`` under this rule, so the
    rule identity — not each seed — enters the cache key.
    """
    config = model.config
    if not isinstance(bank_k, int) or isinstance(bank_k, bool) or bank_k <= 0:
        raise ValueError("bank k must be a positive integer")
    return _canonical_sha256(
        {
            "schema": S24_FEATURES_SCHEMA_ID,
            "kind": "preprocessing",
            "n_channels": int(config.n_channels),
            "patch_size": int(config.patch_size),
            "stride": int(config.stride),
            "pad_end": True,
            "mask_composition": "b0-frozen-default",
            "mask_seed_rule": S24_MASK_RULE_ID,
            "bank_k": int(bank_k),
        }
    )


def restore_s24_provenanced_bank(
    bank_path: str | Path,
    *,
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    expected_bank_sha256: str,
    expected_checkpoint_sha256: str,
) -> ProvenancedBank:
    """Restore the locked Fit bank from custody bytes without refitting.

    Verifies the ``sprint18-fit-reference-bank-v2`` envelope (schema, frozen
    recipe/order IDs, recomputed tensor digest, checkpoint/model/EMA/arm
    binding, complete Fit-role membership) and attaches the stored embeddings
    to a :class:`NormalReferenceBank`. ``fit`` is never called.
    """
    validate_arm_config(arm, require_locked=True)
    expected_bank = _require_sha256(expected_bank_sha256, "expected bank SHA-256")
    expected_checkpoint = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    path = Path(bank_path)
    if not path.is_file():
        raise ValueError(f"S24 bank custody file is missing: {path}")
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(envelope, Mapping):
        raise ValueError("S24 bank envelope must be a mapping")
    if envelope.get("schema_id") != BANK_ARTIFACT_SCHEMA_ID:
        raise ValueError("S24 bank envelope schema is not the accepted custody schema")
    embeddings = envelope.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("S24 bank envelope has no embedding tensor")
    if envelope.get("bank_sha256") != expected_bank:
        raise ValueError("S24 bank envelope digest differs from the accepted bank digest")
    if _tensor_digest(embeddings) != expected_bank:
        raise ValueError("S24 bank bytes do not reproduce the accepted bank digest")
    provenance = envelope.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("S24 bank envelope is missing its provenance mapping")
    for name, frozen in (
        ("bank_recipe_id", BANK_RECIPE_ID),
        ("bank_order_id", BANK_ORDER_ID),
    ):
        if provenance.get(name) != frozen:
            raise ValueError(f"S24 bank {name} differs from the frozen custody recipe")
    if provenance.get("tensor_digest_id") != (
        "representation.sprint18_harness._tensor_digest-v1"
    ):
        raise ValueError("S24 bank tensor digest identity is unknown")
    if provenance.get("bank_sha256") != expected_bank:
        raise ValueError("S24 bank provenance digest differs from the accepted digest")
    if provenance.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("S24 bank was fitted from a different checkpoint")
    if provenance.get("model_seed") != arm.model_seed:
        raise ValueError("S24 bank seed does not match the requested arm")
    if provenance.get("bank_arm_id") != arm.arm_id:
        raise ValueError("S24 bank arm does not match the requested arm")
    if provenance.get("data_binding_sha256") != binding.executable_binding_sha256:
        raise ValueError("S24 bank uses a different certified DATA binding")
    if provenance.get("scope") != "experiment":
        raise ValueError("S24 bank scope is not the locked experiment scope")
    if provenance.get("source_role") != "FIT":
        raise ValueError("S24 bank source role is not Fit")
    if provenance.get("source_rows") != binding.fit_healthy_rows:
        raise ValueError("S24 bank source-row count differs from the certified Fit support")
    fit_histories = tuple(
        sorted(item.actual_history_id for item in binding.role_histories("FIT"))
    )
    if tuple(provenance.get("source_history_ids") or ()) != fit_histories:
        raise ValueError("S24 bank source histories differ from the certified Fit roster")
    expected_manifests = tuple(
        sorted(binding.history(history_id).manifest_sha256 for history_id in fit_histories)
    )
    if tuple(provenance.get("source_manifest_sha256") or ()) != expected_manifests:
        raise ValueError("S24 bank source manifests differ from the certified binding")
    bank_k = provenance.get("bank_k")
    if (
        isinstance(bank_k, bool)
        or not isinstance(bank_k, int)
        or bank_k <= 0
    ):
        raise ValueError("S24 restored bank k must be a positive integer")
    if envelope.get("k") != bank_k:
        raise ValueError("S24 bank envelope k differs from its provenance k")
    bank = NormalReferenceBank(k=int(bank_k))
    bank.embeddings = embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
    if bank.embeddings.ndim != 2 or bank.embeddings.shape[0] == 0:
        raise ValueError("S24 restored bank embeddings are empty or malformed")
    if not torch.isfinite(bank.embeddings).all():
        raise ValueError("S24 restored bank embeddings are non-finite")
    return ProvenancedBank(
        bank=bank,
        provenance=BankProvenance(
            arm_id=cast(Any, provenance["bank_arm_id"]),
            model_seed=int(provenance["model_seed"]),
            source_role="FIT",
            source_history_ids=tuple(provenance["source_history_ids"]),
            source_manifest_sha256=tuple(provenance["source_manifest_sha256"]),
            source_rows=int(provenance["source_rows"]),
            source_rows_sha256=str(provenance["source_rows_sha256"]),
            data_binding_sha256=str(provenance["data_binding_sha256"]),
            checkpoint_sha256=str(provenance["checkpoint_sha256"]),
            bank_sha256=str(provenance["bank_sha256"]),
            scope="experiment",
        ),
    )


@dataclass(frozen=True)
class S24B0Bundle:
    """Frozen restored B0 representation plus its locked reference bank."""

    arm_id: str
    model_seed: int
    model: V1RepresentationModel = field(compare=False)
    bank: ProvenancedBank = field(compare=False)
    checkpoint: RestoredSprint18Checkpoint = field(compare=False)
    preprocessing_sha256: str = ""
    frozen_state_sha256: str = ""


def s24_live_state_digests(
    model: V1RepresentationModel,
    bank: ProvenancedBank,
    scale_model: ContextResidualScaleModel | None = None,
) -> dict[str, str]:
    """Digest the live immutable reference components (model/EMA/bank/scale)."""
    if bank.bank.embeddings is None:
        raise ValueError("S24 live bank has no embeddings")
    state: dict[str, str] = {
        "model_state_sha256": state_sha256(model.state_dict()),
        "ema_state_sha256": state_sha256(model.target_encoder.state_dict()),
        "bank_sha256": _tensor_digest(bank.bank.embeddings),
        "bank_k": str(bank.bank.k),
    }
    if scale_model is not None:
        state["scale_state_sha256"] = state_sha256(scale_model.state_dict())
    return state


def assert_s24_frozen_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    """Fail closed when any frozen parameter/state digest moved during scoring."""
    if dict(before) != dict(after):
        moved = sorted(
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        )
        raise ValueError(f"S24 frozen state moved during scoring: {moved}")


def restore_s24_b0_bundle(
    *,
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_bank_sha256: str,
) -> S24B0Bundle:
    """Restore the locked B0 checkpoint, normalization, and bank as one bundle.

    The checkpoint is strictly restored (``scope="experiment"``); the bank is
    restored from custody without refit. The model is left in ``eval`` mode so
    the frozen ``ConditionalBatchNorm`` statistics can never update.
    """
    validate_arm_config(arm, require_locked=True)
    if arm.model_seed not in MODEL_SEEDS:
        raise ValueError("S24 model seed is outside the frozen Sprint18 seed namespace")
    expected_checkpoint = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise ValueError(f"S24 checkpoint file is missing: {checkpoint_file}")
    if _sha256_file(checkpoint_file) != expected_checkpoint:
        raise ValueError("S24 checkpoint bytes differ from the accepted checkpoint digest")
    saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or not isinstance(saved.get("model_config"), Mapping):
        raise ValueError("S24 checkpoint payload is missing its model config")
    from representation.config import V1Config

    model = V1RepresentationModel(V1Config(**dict(saved["model_config"])))
    restored = restore_sprint18_checkpoint(
        checkpoint_file,
        model,
        None,
        arm=arm,
        binding=binding,
        allowed_scope="experiment",
        restore_rng=False,
    )
    if restored.checkpoint_sha256 != expected_checkpoint:
        raise ValueError("S24 restored checkpoint identity differs from the accepted digest")
    bank = restore_s24_provenanced_bank(
        bank_path,
        binding=binding,
        arm=arm,
        expected_bank_sha256=expected_bank_sha256,
        expected_checkpoint_sha256=expected_checkpoint,
    )
    if bank.bank.k != model.config.knn_k:
        raise ValueError("S24 restored bank k differs from the model contract")
    if bank.provenance.checkpoint_sha256 != restored.checkpoint_sha256:
        raise ValueError("S24 bank provenance checkpoint differs from the restored checkpoint")
    live = s24_live_state_digests(model, bank)
    if live["model_state_sha256"] != restored.model_state_sha256:
        raise ValueError("S24 live model state differs from the restored checkpoint identity")
    if live["ema_state_sha256"] != restored.ema_state_sha256:
        raise ValueError("S24 live EMA state differs from the restored checkpoint identity")
    model.eval()
    preprocessing = s24_preprocessing_digest(model=model, bank_k=bank.bank.k)
    frozen = _canonical_sha256(
        {
            "schema": S24_FEATURES_SCHEMA_ID,
            "kind": "frozen-state",
            "checkpoint_sha256": restored.checkpoint_sha256,
            "model_state_sha256": live["model_state_sha256"],
            "ema_state_sha256": live["ema_state_sha256"],
            "bank_sha256": live["bank_sha256"],
            "arm_config_sha256": arm_config_sha256(arm),
        }
    )
    return S24B0Bundle(
        arm_id=arm.arm_id,
        model_seed=arm.model_seed,
        model=model,
        bank=bank,
        checkpoint=restored,
        preprocessing_sha256=preprocessing,
        frozen_state_sha256=frozen,
    )


@dataclass(frozen=True)
class S24FileFeatures:
    """One file's frozen embedding, independent scores, and patch identity."""

    file_id: str
    history_id: str
    mask_seed: int
    embedding: torch.Tensor = field(compare=False)
    s_pred: float = 0.0
    s_pop: float = 0.0
    s_pred_n4: float | None = None
    n_patches: int = 0
    n_valid_patches: int = 0
    n_masked_targets: int = 0
    valid_len: tuple[int, ...] = ()
    starts: tuple[int, ...] = ()


def capture_s24_file_features(
    *,
    model: V1RepresentationModel,
    bank: ProvenancedBank,
    samples: Sequence[Any],
    history_id: str,
    model_seed: int,
    masking_policy: Any = None,
) -> list[S24FileFeatures]:
    """Capture frozen file embeddings and B0 scores through public inference.

    Every file is collated with its deterministic
    ``sprint18-score-mask-v2`` seed, filtered to the encoder-visible batch,
    and scored with :class:`RepresentationInference` (frozen ``eval`` mode, no
    gradients). File/patch identity and valid lengths are retained alongside
    the outputs; the batch order always matches the input sample order.
    """
    if model_seed not in MODEL_SEEDS:
        raise ValueError("S24 model seed is outside the frozen Sprint18 seed namespace")
    if not history_id:
        raise ValueError("S24 history identity must be non-empty")
    materialized = list(samples)
    if not materialized:
        raise ValueError("S24 feature capture requires a non-empty file batch")
    file_ids = [sample.file_id for sample in materialized]
    if any(not isinstance(item, str) or not item for item in file_ids):
        raise ValueError("S24 feature capture requires canonical file IDs")
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("S24 feature capture requires unique file IDs")
    if bank.bank.embeddings is None:
        raise ValueError("S24 feature capture requires a restored reference bank")
    model.eval()
    inference = RepresentationInference(
        model, bank.bank, model.patchifier, masking_config=model.config
    )
    captured: list[S24FileFeatures] = []
    with torch.no_grad():
        for sample in materialized:
            seed = deterministic_mask_seed(model_seed, history_id, sample.file_id)
            batch = collate_variable_files(
                [sample],
                model.patchifier,
                masking_config=model.config,
                masking_seed=seed,
                masking_policy=masking_policy,
            )
            clean = _encoder_batch(
                {key: value for key, value in batch.items() if key != "file_samples"}
            )
            scored = inference.score_batch(
                inference.co_locate_batch(clean),  # type: ignore[arg-type]
            )
            embedding = scored.get("file_embedding")
            s_pred = scored.get("S_pred")
            s_pop = scored.get("S_pop")
            if (
                not isinstance(embedding, torch.Tensor)
                or not isinstance(s_pred, torch.Tensor)
                or not isinstance(s_pop, torch.Tensor)
            ):
                raise ValueError("S24 public inference did not return score tensors")
            if (
                embedding.shape != (1, model.config.d_model)
                or s_pred.shape != (1,)
                or s_pop.shape != (1,)
            ):
                raise ValueError("S24 public inference returned malformed output shapes")
            vector = embedding.detach().cpu().float()
            pred = float(s_pred.detach().cpu().reshape(-1)[0].item())
            pop = float(s_pop.detach().cpu().reshape(-1)[0].item())
            if not torch.isfinite(vector).all() or not all(
                __import__("math").isfinite(value) for value in (pred, pop)
            ):
                raise ValueError("S24 frozen features are missing or non-finite")
            if pred < 0.0 or pop < 0.0:
                raise ValueError("S24 branch scores must be non-negative")
            valid_mask = batch["patch_valid_mask"][0]
            target_mask = batch["mask"][0] & valid_mask
            captured.append(
                S24FileFeatures(
                    file_id=sample.file_id,
                    history_id=history_id,
                    mask_seed=seed,
                    embedding=vector.reshape(-1).clone(),
                    s_pred=pred,
                    s_pop=pop,
                    n_patches=int(batch["patches"].shape[1]),
                    n_valid_patches=int(valid_mask.sum().item()),
                    n_masked_targets=int(target_mask.sum().item()),
                    valid_len=tuple(int(v) for v in batch["valid_len"][0].tolist()),
                    starts=tuple(int(v) for v in batch["starts"][0].tolist()),
                )
            )
    return captured


def score_s24_development_support(
    *,
    binding: Task6Binding,
    arm: Sprint18ArmConfig,
    bundle: S24B0Bundle,
    rows: Sequence[CertifiedSample],
) -> HarnessScores:
    """Score Development rows through the public harness entry point.

    Thin call to :func:`score_certified_rows` with the restored bundle
    (``action="score_development"``, ``scope="experiment"``). Returns the two
    independent branch maps on the exact requested support.
    """
    if not rows:
        raise ValueError("S24 development scoring requires a non-empty row batch")
    if len({row.actual_history_id for row in rows}) != 1:
        raise ValueError("S24 development scoring requires a single-history batch")
    return score_certified_rows(
        binding,
        arm,
        bundle.model,
        bundle.bank,
        rows,
        bundle.model.patchifier,
        checkpoint=bundle.checkpoint,
        action="score_development",
        scope="experiment",
    )


def score_s24_n4_normalized(
    *,
    model: V1RepresentationModel,
    scale_model: ContextResidualScaleModel,
    samples: Sequence[Any],
    history_id: str,
    model_seed: int,
    v_floor: float,
    v_cap: float,
) -> dict[str, dict[str, float]]:
    """Score files through the frozen B0 encoder plus the fitted N4 scale head.

    Thin call to :func:`n4_score_files` with one deterministic per-file mask
    seed per single-sample batch (the accepted runner convention). Returns the
    per-file ``{"raw", "scale", "normalized"}`` triples on exactly the
    requested file set; the bank/``S_pop`` path is untouched here.
    """
    if model_seed not in MODEL_SEEDS:
        raise ValueError("S24 model seed is outside the frozen Sprint18 seed namespace")
    materialized = list(samples)
    if not materialized:
        raise ValueError("S24 N4 scoring requires a non-empty file batch")
    merged: dict[str, dict[str, float]] = {}
    for sample in materialized:
        seed = deterministic_mask_seed(model_seed, history_id, sample.file_id)
        triples = n4_score_files(
            model=model,
            samples=[sample],
            scale_model=scale_model,
            v_floor=float(v_floor),
            v_cap=float(v_cap),
            patchifier=model.patchifier,
            masking_seed=seed,
        )
        if set(triples) != {sample.file_id}:
            raise ValueError("S24 N4 scorer returned the wrong file map")
        merged.update(triples)
    return merged


def s24_feature_cache_identity(
    *,
    checkpoint_sha256: str,
    bank_sha256: str,
    support_sha256: str,
    preprocessing_sha256: str,
    arm_config_sha256: str,
    data_binding_sha256: str,
    model_seed: int,
    scale_state_sha256: str | None = None,
) -> str:
    """Bind the exact checkpoint/source-support/preprocessing identity.

    ``scale_state_sha256`` carries the fitted N4 scale digest for N4-scored
    caches and ``"none"`` for B0-only caches. Any substitution changes the
    digest; malformed inputs fail closed.
    """
    if model_seed not in MODEL_SEEDS:
        raise ValueError("S24 model seed is outside the frozen Sprint18 seed namespace")
    record: dict[str, object] = {
        "schema": S24_FEATURES_SCHEMA_ID,
        "kind": "feature-cache",
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint SHA-256"),
        "bank_sha256": _require_sha256(bank_sha256, "bank SHA-256"),
        "support_sha256": _require_sha256(support_sha256, "support SHA-256"),
        "preprocessing_sha256": _require_sha256(
            preprocessing_sha256, "preprocessing SHA-256"
        ),
        "arm_config_sha256": _require_sha256(arm_config_sha256, "arm config SHA-256"),
        "data_binding_sha256": _require_sha256(
            data_binding_sha256, "data binding SHA-256"
        ),
        "model_seed": int(model_seed),
        "scale_state_sha256": (
            "none" if scale_state_sha256 is None else _require_sha256(
                scale_state_sha256, "scale state SHA-256"
            )
        ),
    }
    return _canonical_sha256(record)


__all__ = [
    "S24_FEATURES_SCHEMA_ID",
    "S24_MASK_RULE_ID",
    "S24B0Bundle",
    "S24FileFeatures",
    "assert_s24_frozen_unchanged",
    "capture_s24_file_features",
    "restore_s24_b0_bundle",
    "restore_s24_provenanced_bank",
    "s24_feature_cache_identity",
    "s24_live_state_digests",
    "s24_preprocessing_digest",
    "score_s24_development_support",
    "score_s24_n4_normalized",
]
