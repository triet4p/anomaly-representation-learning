"""Sprint 18 Task 20 — K3/K7/K9 legacy-stack dispatch surface.

Binds the exact Sprint 17 K3/K7/K9 member implementations and numerics to the
common Sprint 18 optimizer/schedule contract. No member substitution, no
combination-specific tuning, no coefficient reselection.

Normative sources (read directly, never redefined here):

- Sprint 17 accepted lock ``experiments/sprint17-task23-confirmation-lock.json``
  (file SHA-256 ``3b2f14...``, canonical ``lock_sha256`` ``14fab3...``):
  K3 parents ``(C5-A, C6-A)``; K7 parents ``(C2-A, C3-A, C5-A, C6-A)``;
  K9 parents ``(C2-A, C3-A, C5-A, C6-A, C7-A, C8-A, C9-B)``.
- Task 19 freeze ``experiments/sprint17-task19-combinations.json`` and
  ``src/representation/sprint17_combinations.py`` ``MEMBER_IMPLEMENTATION`` /
  ``K_MEMBERS``: exact per-member builders and frozen numerics, reused by
  reference (never copied).
- Trainable assembly/oracle ``experiments/sprint17_task20_k.py``
  (``k_config_dict`` / ``build_model`` / ``masking_policy_for`` /
  ``train_seed`` / ``score_items`` / ``embed_items`` / ``latent_items`` /
  K9 rescoring block).
- Sprint 18 common contract
  ``experiments/sprint18-full-training-protocol-v1.md`` §4–§6 (optimizer,
  epoch LR schedule, 10-zero/20-ramp/70-plateau contrastive schedule,
  deterministic sampler, seed-matched paired views) and the Task 14 lock
  ``experiments/sprint18-task14-config-lock.json`` (K3/K7/K9
  ``lambda_info_max=0.05``, 100 representation epochs, AdamW/lr 1.5e-3/
  weight-decay 0.0/clip 1.0, seeds 181801–181803).
- Sprint 18 Harness ``src/representation/sprint18_harness.py`` (arm identity,
  bank/scoring/calibration contracts) and
  ``src/representation/sprint18_checkpoint.py`` (coherent checkpoints).

Scope of this module:

- Representation members only: C2-A patchifier (``arm_patchifier``),
  C3-A local encoder (``arm_local_encoder`` + ``ResidualDilatedPatchEncoder``),
  C5-A masking policy (``c5a_block_policy``) + multi-horizon criterion
  (``MultiHorizonPredictionCriterion``), C6-A pooling (``arm_pooling`` +
  ``C6ArmModel``). C2-A contributes no train-config keys beyond the
  overlap-explicit lattice; C7/C8/C9 contribute no representation keys
  (train-free downstream scoring members fitted from the arm's own final
  Fit state, exact Sprint 17 K9 flow).
- K9 downstream scoring members: C8-A ``HuberStandardizer`` fitted on the
  K9 Fit forward residuals, C9-B ``window_max_mean`` aggregation over C8-A
  patch energies for S_pred, C7-A ``RobustShrinkageReference`` fitted on the
  K9 Fit file embeddings for S_pop. C7/C8/C9 numerics are the frozen
  ``MEMBER_IMPLEMENTATION`` values; no retuning.
- Common Sprint 18 training contract: AdamW, batch 128 files
  (``drop_last=false``), LR ``learning_rate_for_epoch``, InfoNCE
  ``coefficient_for_epoch`` with the Task 14 ``lambda_info_max=0.05``,
  deterministic ``deterministic_sampler_order`` batches, frozen per-file/view
  ``sprint18-view-v1`` contrastive views via the model's
  ``sprint18_view_epoch`` path, seed-matched paired views across arms where
  tensor shapes permit (per-file/view key includes only the model seed,
  epoch, file id and view index, so identical seeds share realizations).
- Downstream members never enter representation dispatch, optimizer
  membership, gradient isolation, checkpoint state, or S_pop computation.

What this module deliberately does NOT do:

- No Sprint 17 weight/bank/threshold/root reuse (fresh per-(arm, seed)
  initialization only).
- No DATA generation, no certified DATA contact, no Confirmation contact.
- No coefficient selection or protocol amendment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import math

import numpy as np
import torch
from torch import nn

from representation.config import V1Config
from representation.sprint17_c2 import arm_patchifier
from representation.sprint17_c3 import (
    ARM_LOCAL_ENCODER_PARAMS,
    arm_local_encoder,
    count_local_encoder_params,
)
from representation.sprint17_c5 import (
    C5A_HORIZONS,
    C5A_N_BLOCKS,
    MultiHorizonPredictionCriterion,
    c5a_block_policy,
)
from representation.sprint17_c6 import (
    ARM_POOLING_PARAMS,
    C6ArmModel,
    arm_pooling,
    count_pooling_params,
)
from representation.sprint17_c7 import (
    RobustShrinkageReference,
    arm_reference,
)
from representation.sprint17_c8 import (
    HuberStandardizer,
    arm_patch_energies,
)
from representation.sprint17_c9 import (
    WINDOW_DURATION_TIMESTEPS,
    arm_file_scores,
)
from representation.sprint17_combinations import (
    K_MEMBERS,
    MEMBER_IMPLEMENTATION,
)


#: Sprint 18 arm id -> Sprint 17 K combination id.
S18_TO_S17_K = {
    "K3-FULL": "K3",
    "K7-FULL": "K7",
    "K9-FULL": "K9",
}

#: Exact Sprint 17 member tuples, re-exported by reference (never redefined).
K_MEMBER_TUPLES: Mapping[str, tuple[str, ...]] = {
    "K3": K_MEMBERS["K3"],
    "K7": K_MEMBERS["K7"],
    "K9": K_MEMBERS["K9"],
}

#: Frozen per-member numerics, re-exported by reference from the Task 19 table.
def _member_numerics() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for k_id in ("K3", "K7", "K9"):
        for member in K_MEMBERS[k_id]:
            out.setdefault(member, dict(MEMBER_IMPLEMENTATION[member]["frozen_numerics"]))
    return out


K_MEMBER_NUMERICS: Mapping[str, Mapping[str, object]] = _member_numerics()

#: Task 14 frozen common InfoNCE maximum copied unchanged to K3/K7/K9.
K_LAMBDA_INFO_MAX = 0.05

#: Frozen B0 masking composition (Sprint 17 Task 7 lineage) for non-C5-A arms.
B0_MASK_COMPOSITION = (0.34, 0.33, 0.33)

__all__ = [
    "S18_TO_S17_K",
    "K_MEMBER_TUPLES",
    "K_MEMBER_NUMERICS",
    "K_LAMBDA_INFO_MAX",
    "B0_MASK_COMPOSITION",
    "s17_k_id_for_arm",
    "members_for_arm",
    "member_numerics",
    "masking_policy_for_arm",
    "build_k_model",
    "build_k_criterion",
    "build_k_config",
    "fit_k9_downstream",
    "score_k9_files",
    "k_reference_provenance",
]


def s17_k_id_for_arm(arm_id: str) -> str:
    """Map one Sprint 18 K arm to its exact Sprint 17 K combination id."""
    try:
        return S18_TO_S17_K[arm_id]
    except KeyError:
        raise ValueError(f"unknown Sprint18 legacy-stack arm: {arm_id!r}") from None


def members_for_arm(arm_id: str) -> tuple[str, ...]:
    """Return the exact frozen Sprint 17 member tuple for one Sprint 18 K arm."""
    k_id = s17_k_id_for_arm(arm_id)
    members = K_MEMBERS[k_id]
    if arm_id == "K3-FULL" and members != ("C5-A", "C6-A"):
        raise ValueError("K3-FULL membership differs from the frozen Sprint17 registry")
    if arm_id == "K7-FULL" and members != ("C2-A", "C3-A", "C5-A", "C6-A"):
        raise ValueError("K7-FULL membership differs from the frozen Sprint17 registry")
    if arm_id == "K9-FULL" and members != (
        "C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"
    ):
        raise ValueError("K9-FULL membership differs from the frozen Sprint17 registry")
    return members


def member_numerics(member_id: str) -> dict[str, object]:
    """Return the frozen Task 19 numerics for one member (no outcome parsing)."""
    try:
        return dict(MEMBER_IMPLEMENTATION[member_id]["frozen_numerics"])
    except KeyError:
        raise ValueError(f"unknown Sprint17 member: {member_id!r}") from None


def masking_policy_for_arm(arm_id: str):
    """Return the exact Sprint 17 masking policy for one Sprint 18 K arm.

    ``c5a_block_policy`` wherever C5-A is a member (K3/K7/K9 all contain
    C5-A); ``None`` is the frozen B0 composition path otherwise. Mirrors
    ``experiments/sprint17_task20_k.py::masking_policy_for`` exactly.
    """
    members = members_for_arm(arm_id)
    if "C5-A" in members:
        return c5a_block_policy
    return None


def build_k_config(
    arm_id: str,
    model_seed: int,
    *,
    d_model: int = 128,
    dropout: float = 0.1,
) -> V1Config:
    """Build the exact Sprint 18 model config for one K arm.

    Architecture fields are the frozen Sprint 17 B0 values (the K members
    change only patchifier/encoder/masking/criterion/pooling, never the
    config scalars); conditioning is the certified Sprint 18 (9, 8);
    ``contrastive_weight_max`` is the Task 14 ``lambda_info_max=0.05``;
    ``seed`` is the Sprint 18 model seed (fresh namespace, never a
    Sprint 17 weight reuse).
    """
    members_for_arm(arm_id)  # fail closed on unknown arm
    if not isinstance(model_seed, int) or isinstance(model_seed, bool):
        raise ValueError("Sprint18 model seed must be an integer")
    return V1Config(
        n_channels=6,
        patch_size=32,
        stride=16,
        d_model=d_model,
        sequence_layers=4,
        attention_heads=4,
        dropout=dropout,
        n_robots=9,
        n_programs=8,
        total_mask_ratio=0.40,
        random_fraction=0.34,
        info_fraction=0.33,
        block_fraction=0.33,
        ema_decay=0.996,
        prediction_weight=1.0,
        contrastive_weight_max=K_LAMBDA_INFO_MAX,
        contrastive_temperature=0.2,
        contrastive_gain_std=0.05,
        contrastive_offset_std=0.03,
        contrastive_noise_std=0.015,
        contrastive_max_shift=4,
        knn_k=5,
        seed=model_seed,
    )


def build_k_model(
    arm_id: str,
    config: V1Config,
    *,
    patchifier=None,
    device: torch.device | str | None = None,
):
    """Assemble the exact Sprint 17 member graph for one Sprint 18 K arm.

    Mirrors ``experiments/sprint17_task20_k.py::build_model`` exactly:

    - C2-A member: overlap-explicit ``arm_patchifier("C2-A")`` (bitwise B0
      lattice parity control); otherwise the plain B0 ``Patchifier``.
    - C3-A member: ``arm_local_encoder("C3-A", ...)`` with the frozen
      ``ARM_LOCAL_ENCODER_PARAMS`` identity check; passed as the model's
      ``patch_encoder``.
    - C6-A member: ``arm_pooling("C6-A", ...)`` with the frozen
      ``ARM_POOLING_PARAMS`` identity check, deployed as
      ``C6ArmModel(cfg, arm_id="C6-A", pooling=...)`` so the C3-A encoder
      (when present) is preserved via the ``patch_encoder`` kwarg.
      Without C6-A the plain ``V1RepresentationModel`` (with the C3-A
      encoder when present) is used.
    - C5-A changes only masking policy + criterion (handled by
      ``masking_policy_for_arm`` / ``build_k_criterion``), never the graph.
    - C7-A/C8-A/C9-B contribute no representation graph keys (train-free
      downstream scoring members; see ``fit_k9_downstream``).

    The ``C6ArmModel.__init__`` signature is
    ``(config, *, arm_id, pooling=None, **kwargs)`` where ``**kwargs`` are
    forwarded to ``V1RepresentationModel`` — so ``patch_encoder`` MUST be
    passed through (never silently dropped) when both C3-A and C6-A are
    members (K7/K9).
    """
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    members = members_for_arm(arm_id)
    if patchifier is None:
        if "C2-A" in members:
            patchifier = arm_patchifier("C2-A")
            if (
                patchifier.cfg.patch_size != config.patch_size
                or patchifier.cfg.stride != config.stride
            ):
                raise ValueError("C2-A patchifier geometry incompatible with K config")
        else:
            patchifier = Patchifier(
                PatchConfig(patch_size=config.patch_size, stride=config.stride, pad_end=True)
            )
    patch_encoder: nn.Module | None = None
    # Frozen identity counts bind d_model=128; CPU fixtures may use a smaller
    # width with the identical builder (shape-scaled, never substituted).
    if "C3-A" in members:
        patch_encoder = arm_local_encoder(
            "C3-A", config.n_channels, config.d_model, config.dropout
        )
        if config.d_model == 128 and count_local_encoder_params(patch_encoder) != ARM_LOCAL_ENCODER_PARAMS["C3-A"]:
            raise ValueError("C3-A local-encoder parameter identity mismatch")
    if "C6-A" in members:
        pooling = arm_pooling("C6-A", config.d_model)
        if config.d_model == 128 and count_pooling_params(pooling) != ARM_POOLING_PARAMS["C6-A"]:
            raise ValueError("C6-A pooling parameter identity mismatch")
        if patch_encoder is None:
            model = C6ArmModel(config, arm_id="C6-A", pooling=pooling, patchifier=patchifier)
        else:
            model = C6ArmModel(
                config,
                arm_id="C6-A",
                pooling=pooling,
                patchifier=patchifier,
                patch_encoder=patch_encoder,
            )
    else:
        from representation.model import V1RepresentationModel

        if patch_encoder is None:
            model = V1RepresentationModel(config, patchifier=patchifier)
        else:
            model = V1RepresentationModel(
                config, patchifier=patchifier, patch_encoder=patch_encoder
            )
    if device is not None:
        model.to(device)
    return model, patchifier


def build_k_criterion(arm_id: str, lambda_info_max: float = K_LAMBDA_INFO_MAX):
    """Build the exact joint criterion for one Sprint 18 K arm.

    K3/K7/K9 all contain C5-A, so the prediction term is always the frozen
    ``MultiHorizonPredictionCriterion`` (horizons ``[0, 1, 2]``, shared
    head, no new parameters) and the contrastive term is the frozen
    ``FileContrastiveCriterion`` (temperature 0.2). The schedule is the
    Sprint 18 epoch-based schedule, carried by the trainer's
    ``effective_lambda_override`` (``coefficient_for_epoch``), so the
    criterion's internal ``ProgressiveLambda`` is constructed with the
    Task 14 maximum only as an identity carrier — the per-epoch effective
    value always comes from the common schedule. Prediction weight stays
    exactly 1.0.
    """
    from representation.criterion import (
        FileContrastiveCriterion,
        JointRepresentationCriterion,
        ProgressiveLambda,
    )

    members = members_for_arm(arm_id)
    if "C5-A" not in members:
        raise ValueError(f"{arm_id} has no frozen C5-A criterion binding")
    if lambda_info_max != K_LAMBDA_INFO_MAX:
        raise ValueError("K InfoNCE maximum differs from the frozen Task14 lock (0.05)")
    return JointRepresentationCriterion(
        prediction=MultiHorizonPredictionCriterion(),
        contrastive=FileContrastiveCriterion(temperature=0.2),
        lambda_schedule=ProgressiveLambda(lambda_max=lambda_info_max, ramp_steps=1, warmup_steps=0),
        prediction_weight=1.0,
    )


def c5a_criterion_identity() -> Mapping[str, object]:
    """Return the frozen C5-A criterion identity (horizons, no new params)."""
    return {"horizons": list(C5A_HORIZONS), "n_blocks": C5A_N_BLOCKS, "new_parameters": 0}


#: Required K9 downstream provenance keys (Fit-only fitting, never warm-start).
K9_DOWNSTREAM_KEYS = frozenset({"standardizer", "reference", "window", "fit_ids"})


def fit_k9_downstream(
    model,
    patchifier,
    config: V1Config,
    fit_items: Sequence[tuple[str, object]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Fit the exact K9 downstream scoring members on the K9 Fit state.

    Exact Sprint 17 K9 flow (``experiments/sprint17_task20_k.py`` K9 block):

    - Forward latents via the trained K9 upstream stack with the C5-A block
      masking policy (``masking_policy_for_arm("K9-FULL")``) and per-file
      deterministic mask seeds (caller's ``latent_items`` equivalent).
    - C8-A ``HuberStandardizer`` fitted on the Fit forward residuals
      (``(predicted - target)[prediction_mask]``), abnormal labels refused
      inside ``fit``.
    - C7-A ``RobustShrinkageReference`` fitted on the Fit file embeddings
      with ``(robot_idx, program_idx)`` conditions, abnormal labels refused.
    - C9-B contributes no fitted state (``window_max_mean`` with frozen
      ``WINDOW_DURATION_TIMESTEPS=64`` applied at scoring time).

    Returns ``{"standardizer", "reference", "window", "fit_ids",
    "provenance"}``. The upstream model is never mutated (eval + no_grad
    probes only).
    """
    from representation.data import collate_variable_files

    members = members_for_arm("K9-FULL")
    if "C7-A" not in members or "C8-A" not in members or "C9-B" not in members:
        raise ValueError("K9 downstream requires the frozen C7-A/C8-A/C9-B members")
    dev = torch.device(device or "cpu")
    policy = masking_policy_for_arm("K9-FULL")
    model.eval()
    lat_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for file_id, sample in fit_items:
            batch = collate_variable_files(
                [sample], patchifier, masking_config=config, masking_seed=_file_seed(file_id),  # type: ignore[arg-type]
                masking_policy=policy,
            )
            moved = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            res = model(moved)
            lat_rows.append(
                {
                    "file_id": file_id,
                    "predicted": res["predicted_latents"][0].detach().cpu(),
                    "target": res["target_latents"][0].detach().cpu(),
                    "prediction_mask": res["prediction_mask"][0].detach().cpu(),
                    "file_embedding": res["file_embedding"][0].detach().cpu(),
                    "starts": batch["starts"][0].detach().cpu().numpy(),
                    "valid_len": batch["valid_len"][0].detach().cpu().numpy(),
                    "robot_idx": int(getattr(sample, "robot_idx", 0)),
                    "program_idx": int(getattr(sample, "program_idx", 0)),
                    "file_label": getattr(sample, "file_label", None),
                }
            )
    residuals = torch.cat(
        [(r["predicted"] - r["target"])[r["prediction_mask"]] for r in lat_rows], dim=0  # type: ignore[index]
    )
    labels = []
    for r in lat_rows:
        n = int(r["prediction_mask"].sum().item())  # type: ignore[union-attr]
        labels += [r["file_label"]] * n
    standardizer = HuberStandardizer().fit(residuals, labels=labels)
    embeddings = torch.stack([r["file_embedding"] for r in lat_rows])  # type: ignore[arg-type]
    robots = [int(r["robot_idx"]) for r in lat_rows]
    programs = [int(r["program_idx"]) for r in lat_rows]
    file_labels = [r["file_label"] for r in lat_rows]
    reference = arm_reference("C7-A").fit(embeddings, robots, programs, labels=file_labels)
    return {
        "standardizer": standardizer,
        "reference": reference,
        "window": {
            "mechanism": "contiguous-window",
            "duration": WINDOW_DURATION_TIMESTEPS,
        },
        "fit_ids": tuple(fid for fid, _ in fit_items),
        "provenance": {
            "standardizer": standardizer.provenance(),
            "reference": reference.provenance(),
            "masking_policy": "channel_time_block",
            "criterion": "multi-horizon-ema",
        },
    }


def _file_seed(file_id: str) -> int:
    """Deterministic per-file masking seed (exact Sprint 17 K rule)."""
    digest = sha256(file_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def k9_forward_latents(
    model,
    patchifier,
    config: V1Config,
    items: Sequence[tuple[str, object]],
    *,
    device: torch.device | str | None = None,
) -> list[dict[str, object]]:
    """Forward latents for K9 downstream scoring (eval, no_grad, no mutation)."""
    from representation.data import collate_variable_files

    dev = torch.device(device or "cpu")
    policy = masking_policy_for_arm("K9-FULL")
    model.eval()
    out: list[dict[str, object]] = []
    with torch.no_grad():
        for file_id, sample in items:
            batch = collate_variable_files(
                [sample], patchifier, masking_config=config, masking_seed=_file_seed(file_id),  # type: ignore[arg-type]
                masking_policy=policy,
            )
            moved = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            res = model(moved)
            out.append(
                {
                    "file_id": file_id,
                    "predicted": res["predicted_latents"][0].detach().cpu(),
                    "target": res["target_latents"][0].detach().cpu(),
                    "prediction_mask": res["prediction_mask"][0].detach().cpu(),
                    "file_embedding": res["file_embedding"][0].detach().cpu(),
                    "starts": batch["starts"][0].detach().cpu().numpy(),
                    "valid_len": batch["valid_len"][0].detach().cpu().numpy(),
                    "file_len": int(batch["patch_valid_mask"][0].sum().item()),
                    "robot_idx": int(getattr(sample, "robot_idx", 0)),
                    "program_idx": int(getattr(sample, "program_idx", 0)),
                }
            )
    return out


def score_k9_files(
    lat_rows: Sequence[Mapping[str, object]],
    downstream: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Score K9 files with the fitted downstream members (no fusion).

    - S_pred: C9-B ``window_max_mean`` over C8-A Huber patch energies.
    - S_pop: C7-A robust-shrinkage Mahalanobis energies on file embeddings.
    Returns ``(s_pred, s_pop, provenance)`` with per-file dicts.
    """
    standardizer = downstream["standardizer"]
    reference = downstream["reference"]
    if not isinstance(standardizer, HuberStandardizer):
        raise ValueError("K9 downstream standardizer is not a fitted C8-A HuberStandardizer")
    if not isinstance(reference, RobustShrinkageReference):
        raise ValueError("K9 downstream reference is not a fitted C7-A RobustShrinkageReference")
    s_pred: dict[str, float] = {}
    patch_store: dict[str, object] = {}
    for row in lat_rows:
        pred = row["predicted"].unsqueeze(0)  # type: ignore[union-attr]
        targ = row["target"].unsqueeze(0)  # type: ignore[union-attr]
        mask = row["prediction_mask"]
        if mask.dtype is not torch.bool:  # type: ignore[union-attr]
            raise ValueError("prediction mask must be bool")
        mask_b = mask.unsqueeze(0)  # type: ignore[union-attr]
        energies = arm_patch_energies("C8-A", pred, targ, mask_b, standardizer)
        e_np = energies[0].detach().cpu().numpy().astype(np.float64)
        m_np = mask_b[0].detach().cpu().numpy().astype(bool)
        score, prov = arm_file_scores(
            "C9-B", e_np, m_np, row["starts"], row["valid_len"]  # type: ignore[arg-type]
        )
        if not math.isfinite(float(score)):
            raise ValueError("non-finite K9 S_pred file score")
        s_pred[str(row["file_id"])] = float(score)
        patch_store[str(row["file_id"])] = {"energies": e_np, "provenance": prov}
    embeddings = torch.stack([r["file_embedding"] for r in lat_rows])  # type: ignore[arg-type]
    robots = [int(r["robot_idx"]) for r in lat_rows]
    programs = [int(r["program_idx"]) for r in lat_rows]
    pop, levels = reference.score(embeddings, robots, programs)
    pop_np = np.asarray(pop.detach().cpu().tolist() if isinstance(pop, torch.Tensor) else pop, dtype=np.float64)
    if not np.isfinite(pop_np).all():
        raise ValueError("non-finite K9 S_pop file scores")
    s_pop = {str(r["file_id"]): float(v) for r, v in zip(lat_rows, pop_np.tolist())}
    if set(s_pred) != set(s_pop):
        raise ValueError("K9 branch support differs between S_pred and S_pop")
    return s_pred, s_pop, {"patch": patch_store, "levels": levels}


def k_reference_provenance(arm_id: str) -> dict[str, object]:
    """Return the frozen member/source provenance for one Sprint 18 K arm."""
    members = members_for_arm(arm_id)
    return {
        "sprint18_arm": arm_id,
        "sprint17_k": s17_k_id_for_arm(arm_id),
        "member_arm_ids": list(members),
        "members": {
            member: {
                "component": MEMBER_IMPLEMENTATION[member]["component"],
                "impl_module": MEMBER_IMPLEMENTATION[member]["impl_module"],
                "builder": MEMBER_IMPLEMENTATION[member]["builder"],
                "adapter_description": MEMBER_IMPLEMENTATION[member]["adapter_description"],
                "frozen_numerics": dict(MEMBER_IMPLEMENTATION[member]["frozen_numerics"]),
                "driver": MEMBER_IMPLEMENTATION[member]["driver"],
                "summary": MEMBER_IMPLEMENTATION[member]["summary"],
            }
            for member in members
        },
        "sprint17_lock": "experiments/sprint17-task23-confirmation-lock.json",
        "task19_freeze": "experiments/sprint17-task19-combinations.json",
        "train_oracle": "experiments/sprint17_task20_k.py",
        "lambda_info_max": K_LAMBDA_INFO_MAX,
    }
