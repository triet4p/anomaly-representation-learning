"""Supervised transfer, random-frozen, and scratch learners (Sprint 24 Task 6).

Implements the frozen ``S24-PROTO-v1`` Section 5/6 method matrix on top of the
Task 4 views/label gate and the Task 5 accepted-asset feature path:

- ``M1`` frozen-head transfer: the accepted B0 encoder is restored by
  :func:`representation.sprint24_features.restore_s24_b0_bundle` and stays in
  ``eval`` mode with every parameter frozen; only a 128 -> 1 linear head plus a
  fixed per-feature standardization (``129`` trainable parameters) is fitted.
- ``M2`` random-frozen control: the identical architecture from a random
  initialization (frozen seeds ``241801``-``241803``), identical head and
  identical head-training protocol; the deterministic per-file mask seed is
  keyed by the paired accepted B0 seed, so the control differs only in the
  encoder weights.
- ``M3`` scratch-supervised control: the identical architecture initialized
  from seed ``2408`` and trained from scratch on exactly the same revealed
  labels with every weight trainable (fixed 100 epochs, AdamW cosine
  ``1.5e-3 -> 1e-5``), sharing the file-embedding read-out contract.

Label and validation boundaries are enforced, not merely documented:

- Training support is exactly one nested budget (``B1``/``B2``/``B3``) and
  validation/stopping support is exactly ``V-VAL``; labels are read only from
  :meth:`representation.sprint24_views.LabelReleaseGate.require_released`, so
  premature or evaluation-frame label access fails closed. ``V-TRIG``,
  ``V-R0``, ``V-FUTURE``, ``V-CONF-EVAL`` and the sequential frames are never
  accepted as stopping/threshold support.
- Feature standardization is fitted on the adaptation-training subset only.
- Frozen-arm parameters and normalization statistics are digested before and
  after fitting/scoring; any movement is an error.
- A checkpoint binds schema, method, model config, encoder/checkpoint, bank,
  preprocessing, support-with-labels and reveal-boundary identities, plus
  optimizer and RNG state. Restoring verifies every identity, reproduces the
  same predictions, and resumes deterministically without re-consuming the same
  label event (:class:`S24LabelConsumptionLedger`).

Out of scope by assignment: the sequential evaluator (Task 8), the locked
comparison (Task 7), conditional fine-tuning (Task 10), any Confirmation
contact, and any performance gate.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from representation.config import V1Config
from representation.data import collate_variable_files
from representation.model import V1RepresentationModel
from representation.sprint18_checkpoint import state_sha256
from representation.sprint18_harness import (
    MODEL_SEEDS,
    _canonical_sha256,
    _encoder_batch,
    deterministic_mask_seed,
)
from representation.sprint24_features import (
    S24B0Bundle,
    assert_s24_frozen_unchanged,
    s24_live_state_digests,
)
from representation.sprint24_views import (
    PROTOCOL_ID,
    HistoryViews,
    LabelReleaseGate,
)

#: Schema identity for learner records and checkpoints.
S24_LEARNERS_SCHEMA_ID = "s24-learners-v1"
S24_LEARNER_CHECKPOINT_SCHEMA_ID = "s24-learner-checkpoint-v1"

#: Frozen method identifiers (``S24-PROTO-v1`` Section 5).
METHOD_FROZEN_HEAD = "M1"
METHOD_RANDOM_FROZEN = "M2"
METHOD_SCRATCH = "M3"
S24_METHODS = (METHOD_FROZEN_HEAD, METHOD_RANDOM_FROZEN, METHOD_SCRATCH)

#: Frozen random-encoder init seeds (``S24-PROTO-v1`` Section 6).
RANDOM_ENCODER_SEEDS = (241801, 241802, 241803)

#: Frozen ``M1``/``M2`` head-training configuration (Section 6, no search).
HEAD_LR = 1e-3
HEAD_BATCH_SIZE = 128
HEAD_MAX_EPOCHS = 200
HEAD_PATIENCE = 20
HEAD_INIT_SEED = 2405
HEAD_SHUFFLE_SEED = 2406
#: ``AdamW`` keyword defaults are part of the frozen config because the
#: protocol freezes the optimizer and forbids a search.
HEAD_WEIGHT_DECAY = 0.01
HEAD_TRAINABLE_PARAMETERS = 129

#: Frozen ``M3`` scratch-supervised configuration (Section 6, no search).
SCRATCH_LR_START = 1.5e-3
SCRATCH_LR_END = 1e-5
SCRATCH_EPOCHS = 100
SCRATCH_BATCH_SIZE = 128
SCRATCH_INIT_SEED = 2408
SCRATCH_SHUFFLE_SEED = 2409
SCRATCH_WEIGHT_DECAY = 0.01

#: Layer-norm-style floor for a zero-variance feature column.
NORMALIZER_STD_FLOOR = 1e-6

#: Support roles a learner may declare.
LABEL_TRAIN = "training"
LABEL_VALIDATION = "validation"
LABEL_EVALUATION = "evaluation"

#: Frames that may be scored as evaluation support (labels never released).
EVALUATION_FRAMES = frozenset(
    {
        "V-FUTURE",
        "V-CONF-EVAL",
        "V-SEQ-E1",
        "V-SEQ-E2",
        "V-SEQ-E3",
        "V-NF-E4",
        "V-NF-F",
    }
)


def _canonical(value: object) -> str:
    """Return the repository's canonical deterministic digest."""
    return _canonical_sha256(value)


def _label_text(value: object) -> str:
    """Return a label's string form for ``SampleLabel`` enums or plain strings."""
    text = getattr(value, "value", value)
    if not isinstance(text, str):
        raise ValueError("S24 released labels must carry a string label")
    return text


def s24_label_target(label: Mapping[str, object]) -> int:
    """Map one released label record to the binary target; unknown fails closed."""
    if not isinstance(label, Mapping):
        raise ValueError("S24 released labels must be label mappings")
    text = _label_text(label.get("file_label"))
    if text == "normal":
        return 0
    if text == "abnormal":
        return 1
    raise ValueError(
        f"S24 released label {text!r} is neither normal nor abnormal"
    )


def assert_s24_support_role(
    views: HistoryViews,
    file_ids: Sequence[str],
    *,
    role: str,
    budget: str | None = None,
    frame: str | None = None,
) -> tuple[str, ...]:
    """Fail closed unless ``file_ids`` sit exactly in one allowed role frame.

    ``training`` requires budget membership (a nested ``B1``/``B2``/``B3``
    set), ``validation`` requires ``V-VAL`` -- the only stopping/threshold
    support -- and ``evaluation`` requires an explicit evaluation frame.
    Duplicate or empty support is rejected, so no file outside the declared
    role can reach training, stopping or scoring.
    """
    ids = tuple(str(item) for item in file_ids)
    if not ids:
        raise ValueError("S24 support must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("S24 support contains duplicate file IDs")
    if role == LABEL_TRAIN:
        if budget is None or budget not in views.budgets:
            raise ValueError(f"S24 training support requires a known budget, got {budget!r}")
        allowed = set(views.budgets[budget])
        name = budget
    elif role == LABEL_VALIDATION:
        if budget is not None:
            raise ValueError("S24 validation support never belongs to a label budget")
        allowed = set(views.frame("V-VAL"))
        name = "V-VAL"
    elif role == LABEL_EVALUATION:
        if budget is not None:
            raise ValueError("S24 evaluation support never belongs to a label budget")
        if frame is None or frame not in EVALUATION_FRAMES:
            raise ValueError(
                f"S24 evaluation support requires an evaluation frame, got {frame!r}"
            )
        allowed = set(views.frame(frame))
        name = frame
    else:
        raise ValueError(f"unknown S24 support role: {role!r}")
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(
            f"S24 {role} support contains {len(outside)} file(s) outside {name} "
            f"(first: {outside[0]!r})"
        )
    if role == LABEL_TRAIN and set(ids) & set(views.frame("V-VAL")):
        raise ValueError("S24 training support overlaps the V-VAL stopping frame")
    return ids


@dataclass(frozen=True)
class S24TrainSupport:
    """Immutable revealed-label support for one method cell."""

    history_id: str
    budget: str
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    training_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    release_time_s: float
    now_s: float
    views_manifest_digest: str = ""
    support_digest: str = ""

    @property
    def label_ids(self) -> tuple[str, ...]:
        """Return every file ID whose label this cell was allowed to read."""
        return self.training_ids + self.validation_ids

    def record(self) -> dict[str, object]:
        """Return the JSON-safe support record bound into checkpoints."""
        return {
            "schema": S24_LEARNERS_SCHEMA_ID,
            "kind": "train-support",
            "protocol_id": PROTOCOL_ID,
            "views_manifest_digest": self.views_manifest_digest,
            "history_id": self.history_id,
            "budget": self.budget,
            "training_ids": list(self.training_ids),
            "training_labels": list(self.training_labels),
            "validation_ids": list(self.validation_ids),
            "validation_labels": list(self.validation_labels),
            "release_time_s": float(self.release_time_s),
            "now_s": float(self.now_s),
        }


def build_s24_train_support(
    *,
    views: HistoryViews,
    gate: LabelReleaseGate,
    budget: str,
    now_s: float,
    waveform_ids: Sequence[str],
) -> S24TrainSupport:
    """Build one cell's revealed support through the label-release boundary.

    Training IDs are exactly ``views.budgets[budget]`` and stopping IDs are
    exactly ``V-VAL``. Every required waveform must be supplied (no silent
    support shrinkage), every label is read through
    :meth:`LabelReleaseGate.require_released` at ``now_s``, and both classes
    must be present. Nothing outside the two allowed roles is touched.
    """
    if gate.views.manifest_digest != views.manifest_digest:
        raise ValueError("S24 label gate is bound to a different view manifest")
    if isinstance(now_s, bool) or not isinstance(now_s, (int, float)):
        raise ValueError("S24 support requires a numeric reveal boundary")
    reveal = float(now_s)
    if not math.isfinite(reveal):
        raise ValueError("S24 reveal boundary must be finite")
    training_ids = assert_s24_support_role(
        views, views.budgets[budget], role=LABEL_TRAIN, budget=budget
    )
    validation_ids = assert_s24_support_role(
        views, views.frame("V-VAL"), role=LABEL_VALIDATION
    )
    available = {str(item) for item in waveform_ids}
    absent = sorted((set(training_ids) | set(validation_ids)) - available)
    if absent:
        raise ValueError(
            f"S24 support is missing {len(absent)} required waveform(s) "
            f"(first: {absent[0]!r})"
        )
    try:
        released = gate.require_released(
            list(training_ids) + list(validation_ids), reveal
        )
    except KeyError as exc:
        raise ValueError("S24 support file has no released label record") from exc
    training_labels = tuple(
        s24_label_target(released[file_id]) for file_id in training_ids
    )
    validation_labels = tuple(
        s24_label_target(released[file_id]) for file_id in validation_ids
    )
    if set(training_labels) != {0, 1}:
        raise ValueError("S24 training support must contain both classes")
    if not validation_labels:
        raise ValueError("S24 validation support must be non-empty")
    support = S24TrainSupport(
        history_id=views.history_id,
        budget=budget,
        training_ids=training_ids,
        validation_ids=validation_ids,
        training_labels=training_labels,
        validation_labels=validation_labels,
        release_time_s=float(
            max(gate.release_time_of(file_id) for file_id in training_ids + validation_ids)
        ),
        now_s=reveal,
        views_manifest_digest=views.manifest_digest,
    )
    digest = _canonical(support.record())
    return S24TrainSupport(
        history_id=support.history_id,
        budget=support.budget,
        training_ids=support.training_ids,
        validation_ids=support.validation_ids,
        training_labels=support.training_labels,
        validation_labels=support.validation_labels,
        release_time_s=support.release_time_s,
        now_s=support.now_s,
        views_manifest_digest=support.views_manifest_digest,
        support_digest=digest,
    )


class S24LinearHead(nn.Module):
    """Fixed ``d_model -> 1`` linear head with a deterministic frozen init."""

    def __init__(self, d_model: int, *, seed: int = HEAD_INIT_SEED) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("S24 head width must be a positive integer")
        self.d_model = int(d_model)
        self.linear = nn.Linear(self.d_model, 1)
        bound = 1.0 / math.sqrt(self.d_model)
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound, generator=generator)
            self.linear.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit per row of ``features`` (``[B, d_model]``)."""
        if features.ndim != 2 or features.shape[1] != self.d_model:
            raise ValueError(
                f"S24 head expects [B, {self.d_model}] features, "
                f"got {tuple(features.shape)}"
            )
        return self.linear(features).reshape(-1)


@dataclass(frozen=True)
class S24FeatureNormalizer:
    """Fixed per-feature standardization fitted on the training subset only."""

    mean: torch.Tensor = field(compare=False)
    std: torch.Tensor = field(compare=False)

    @classmethod
    def fit(cls, features: torch.Tensor) -> "S24FeatureNormalizer":
        """Fit column statistics from adaptation-training embeddings only."""
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("S24 normalizer requires a non-empty [N, D] feature matrix")
        if not torch.isfinite(features).all():
            raise ValueError("S24 normalizer features must be finite")
        mean = features.mean(dim=0).detach().to(dtype=torch.float32).clone()
        std = (
            features.std(dim=0, unbiased=False)
            .detach()
            .to(dtype=torch.float32)
            .clamp_min(NORMALIZER_STD_FLOOR)
            .clone()
        )
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the frozen standardization to embeddings."""
        if features.ndim != 2 or features.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"S24 normalizer expects [B, {self.mean.shape[0]}] features, "
                f"got {tuple(features.shape)}"
            )
        return (features - self.mean) / self.std


class S24LabelConsumptionLedger:
    """Reject double consumption of one revealed-label event.

    Consumption is scoped by method cell (``M1``/``M2``/``M3``): the same
    labels legitimately feed different compared methods, but within one method
    a distinct event over already-consumed labels is a double-counted update
    and fails closed. Re-presenting the identical event (a checkpoint resume)
    is idempotent.
    """

    def __init__(self, events: Mapping[str, Mapping[str, object]] | None = None) -> None:
        self._events: dict[str, dict[str, object]] = {}
        for event_id, record in (events or {}).items():
            self._events[str(event_id)] = dict(record)

    def consume(
        self,
        *,
        scope: str,
        event_id: str,
        label_ids: Sequence[str],
        release_time_s: float,
        resuming: bool = False,
    ) -> None:
        """Register one event or raise when the same labels were already consumed.

        ``resuming`` is only true for a continuation of the same event (a
        checkpoint restore or a further epoch request on the same learner
        object); a fresh training run over an already-recorded event is a
        double-counted update and fails closed.
        """
        if not scope or not event_id:
            raise ValueError("S24 consumption requires a scope and event identity")
        ids = tuple(str(item) for item in label_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("S24 consumption requires unique non-empty label IDs")
        if event_id in self._events:
            if resuming:
                return
            raise ValueError(
                f"S24 label event {event_id!r} was already consumed; restore the "
                "checkpoint to resume instead of re-fitting from scratch"
            )
        for other_id, record in self._events.items():
            if record.get("scope") != scope:
                continue
            overlap = set(ids) & set(cast(Iterable[str], record.get("label_ids", ())))
            if overlap:
                raise ValueError(
                    f"S24 labels already consumed by event {other_id!r} in scope "
                    f"{scope!r} ({len(overlap)} shared file(s))"
                )
        self._events[event_id] = {
            "schema": S24_LEARNERS_SCHEMA_ID,
            "kind": "label-consumption",
            "scope": scope,
            "event_id": event_id,
            "label_ids": list(ids),
            "release_time_s": float(release_time_s),
        }

    def events(self) -> dict[str, dict[str, object]]:
        """Return a defensive copy of every recorded consumption event."""
        return {key: dict(value) for key, value in self._events.items()}


def _event_identity(method: str, support: S24TrainSupport) -> str:
    return _canonical(
        {
            "schema": S24_LEARNERS_SCHEMA_ID,
            "kind": "label-event",
            "method": method,
            "support_digest": support.support_digest,
        }
    )


def _copy_state(state: Mapping[str, object]) -> dict[str, object]:
    """Deep-copy a model/head state dict, preserving tensor and extra state."""
    copied: dict[str, object] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            copied[name] = value.detach().clone()
        else:
            copied[name] = copy.deepcopy(value)
    return copied


class S24Learner:
    """One frozen-head or scratch-supervised learner cell.

    The learner owns exactly the declared trainable scope: ``M1``/``M2`` freeze
    every encoder parameter and keep the encoder in ``eval`` mode (the
    conditional-normalization statistics therefore cannot move) and train the
    linear head only; ``M3`` trains the identical architecture plus head from a
    declared random initialization.
    """

    def __init__(
        self,
        *,
        method: str,
        encoder: V1RepresentationModel,
        support: S24TrainSupport,
        identity: Mapping[str, object],
        mask_model_seed: int,
        shuffle_seed: int,
        max_epochs: int,
        patience: int,
        frozen_embeddings: Mapping[str, torch.Tensor] | None = None,
        learning_rate: float = HEAD_LR,
        batch_size: int = HEAD_BATCH_SIZE,
    ) -> None:
        if method not in S24_METHODS:
            raise ValueError(f"unknown S24 method: {method!r}")
        if not isinstance(encoder, V1RepresentationModel):
            raise ValueError("S24 learner requires a V1 representation encoder")
        if mask_model_seed not in MODEL_SEEDS:
            raise ValueError("S24 mask seed must stay inside the frozen Sprint18 namespace")
        if max_epochs <= 0 or patience < 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("S24 learner configuration is outside the frozen protocol")
        self.method = method
        self.encoder = encoder
        self.head = S24LinearHead(int(encoder.config.d_model))
        self.support = support
        self.identity = dict(identity)
        self.mask_model_seed = int(mask_model_seed)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.normalizer: S24FeatureNormalizer | None = None
        self.epoch = 0
        self.history: list[dict[str, float]] = []
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.best_head_state: dict[str, object] | None = None
        self.patience_left = int(patience)
        self.stop_reason: str | None = None
        self.final_val_loss: float | None = None
        self.event_id = _event_identity(method, support)
        self.consumed: dict[str, dict[str, object]] = {}
        self._frozen_embeddings: dict[str, torch.Tensor] = {}
        self._train_features: torch.Tensor | None = None
        self._val_features: torch.Tensor | None = None
        self._optimizer: torch.optim.AdamW | None = None
        self._generator = torch.Generator().manual_seed(int(shuffle_seed))
        self._pending_rng_state: torch.Tensor | None = None
        if frozen_embeddings is not None:
            for file_id, vector in frozen_embeddings.items():
                tensor = vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                if tensor.shape[0] != encoder.config.d_model:
                    raise ValueError("S24 frozen embedding width differs from the model")
                if not torch.isfinite(tensor).all():
                    raise ValueError("S24 frozen embeddings must be finite")
                self._frozen_embeddings[str(file_id)] = tensor.clone()
        trainable = method == METHOD_SCRATCH
        for parameter in encoder.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        if trainable:
            self.encoder.train()
        else:
            self.encoder.eval()
        self.identity.setdefault("trainables", self.trainable_record())

    # ------------------------------------------------------------------ scope
    def trainable_record(self) -> dict[str, object]:
        """Return the declared/actual trainable scope of this cell.

        ``declared_trainable_parameters`` reproduces the frozen ``M3`` count
        (every encoder weight including the EMA target copy plus the head).
        ``gradient_trainable_parameters`` reports the parameters that actually
        receive gradients; the EMA target copy is updated only by the frozen
        EMA contract and therefore never by a gradient step.
        """
        encoder_names = [
            name for name, p in self.encoder.named_parameters() if p.requires_grad
        ]
        head_names = [
            name for name, p in self.head.named_parameters() if p.requires_grad
        ]
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        gradient_encoder = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        head_total = sum(p.numel() for p in self.head.parameters())
        return {
            "schema": S24_LEARNERS_SCHEMA_ID,
            "kind": "trainables",
            "method": self.method,
            "encoder_trainable_names": encoder_names,
            "head_trainable_names": head_names,
            "encoder_declared_parameters": encoder_total,
            "encoder_gradient_parameters": gradient_encoder,
            "encoder_frozen_parameters": encoder_total - gradient_encoder,
            "head_trainable_parameters": sum(
                p.numel() for p in self.head.parameters() if p.requires_grad
            ),
            "declared_trainable_parameters": (
                encoder_total + head_total
                if self.method == METHOD_SCRATCH
                else head_total
            ),
            "gradient_trainable_parameters": gradient_encoder + head_total,
        }

    def identity_digest(self) -> str:
        """Return the provenance digest of the learner identity record."""
        return _canonical(
            {
                "schema": S24_LEARNERS_SCHEMA_ID,
                "kind": "learner-identity",
                "method": self.method,
                "support_digest": self.support.support_digest,
                "identity": self.identity,
            }
        )

    # -------------------------------------------------------------- inference
    def _encode_samples(
        self, samples: Sequence[Any], *, history_id: str
    ) -> torch.Tensor:
        """Capture file embeddings through the frozen B0 mask/forward rule."""
        materialized = list(samples)
        if not materialized:
            raise ValueError("S24 encoding requires a non-empty file batch")
        file_ids = [sample.file_id for sample in materialized]
        if len(set(file_ids)) != len(file_ids):
            raise ValueError("S24 encoding requires unique file IDs")
        was_training = self.encoder.training
        self.encoder.eval()
        vectors: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for sample in materialized:
                    seed = deterministic_mask_seed(
                        self.mask_model_seed, history_id, sample.file_id
                    )
                    batch = collate_variable_files(
                        [sample],
                        self.encoder.patchifier,
                        masking_config=self.encoder.config,
                        masking_seed=seed,
                    )
                    output = self.encoder(_encoder_batch(batch))
                    embedding = output["file_embedding"]
                    if embedding.shape != (1, self.encoder.config.d_model):
                        raise ValueError("S24 encoder returned a malformed file embedding")
                    vector = (
                        embedding.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .reshape(-1)
                    )
                    if not torch.isfinite(vector).all():
                        raise ValueError("S24 file embedding is non-finite")
                    vectors.append(vector.clone())
        finally:
            if was_training:
                self.encoder.train()
        return torch.stack(vectors, dim=0)

    def encode(self, samples: Sequence[Any], *, history_id: str) -> torch.Tensor:
        """Return embeddings for ``samples``, reusing declared frozen features.

        Frozen arms may supply Task 5 captured embeddings for the training and
        validation support; per-file deterministic capture covers everything
        else. Returned rows always follow the input order.
        """
        materialized = list(samples)
        if not materialized:
            raise ValueError("S24 encoding requires a non-empty file batch")
        needed = [
            sample.file_id
            for sample in materialized
            if sample.file_id not in self._frozen_embeddings
        ]
        computed: dict[str, torch.Tensor] = {}
        if needed:
            batch = self._encode_samples(
                [sample for sample in materialized if sample.file_id in set(needed)],
                history_id=history_id,
            )
            computed = dict(zip(needed, batch))
        rows: list[torch.Tensor] = []
        for sample in materialized:
            vector = self._frozen_embeddings.get(sample.file_id, computed.get(sample.file_id))
            if vector is None:
                raise ValueError(f"S24 embedding for {sample.file_id!r} was not produced")
            rows.append(vector)
        stacked = torch.stack(rows, dim=0).to(dtype=torch.float32)
        if stacked.shape[1] != self.encoder.config.d_model:
            raise ValueError("S24 embedding width differs from the model contract")
        return stacked

    def _forward_embeddings(self, samples: Sequence[Any]) -> torch.Tensor:
        """One batched encoder forward under the caller's module mode."""
        if not samples:
            raise ValueError("S24 batch forward requires a non-empty file batch")
        batch_seed = int(
            torch.randint(0, 2**63 - 1, (1,), generator=self._generator).item()
        )
        batch = collate_variable_files(
            list(samples),
            self.encoder.patchifier,
            masking_config=self.encoder.config,
            masking_seed=batch_seed,
        )
        return self.encoder(_encoder_batch(batch))["file_embedding"]

    def predict_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return anomaly probabilities from already-captured embeddings."""
        if self.normalizer is None:
            raise ValueError("S24 learner has no fitted feature normalizer")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("S24 prediction requires a non-empty [N, D] matrix")
        self.head.eval()
        with torch.no_grad():
            logits = self.head(self.normalizer.transform(embeddings.to(torch.float32)))
            return torch.sigmoid(logits).detach().cpu()

    def predict(self, samples: Sequence[Any], *, history_id: str) -> dict[str, float]:
        """Return per-file anomaly probabilities through the public forward."""
        if self.normalizer is None:
            raise ValueError("S24 learner has no fitted feature normalizer")
        materialized = list(samples)
        ids = [sample.file_id for sample in materialized]
        if len(set(ids)) != len(ids):
            raise ValueError("S24 prediction requires unique file IDs")
        embeddings = self.encode(materialized, history_id=history_id)
        probabilities = self.predict_embeddings(embeddings)
        return {file_id: float(value) for file_id, value in zip(ids, probabilities)}

    # ---------------------------------------------------------------- training
    def _encoder_digest(self) -> str:
        return state_sha256(self.encoder.state_dict())

    def _matrix_for(
        self, ids: Sequence[str], samples_by_id: Mapping[str, Any] | None
    ) -> torch.Tensor:
        missing = [file_id for file_id in ids if file_id not in self._frozen_embeddings]
        computed: dict[str, torch.Tensor] = {}
        if missing:
            if samples_by_id is None:
                raise ValueError(
                    f"S24 frozen support is missing {len(missing)} captured "
                    "embedding(s) and no waveforms were supplied"
                )
            absent = [file_id for file_id in missing if file_id not in samples_by_id]
            if absent:
                raise ValueError(
                    f"S24 training support has no waveform for {len(absent)} "
                    f"file(s) (first: {absent[0]!r})"
                )
            batch = self._encode_samples(
                [samples_by_id[file_id] for file_id in missing],
                history_id=self.support.history_id,
            )
            computed = dict(zip(missing, batch))
        rows: list[torch.Tensor] = []
        for file_id in ids:
            vector = self._frozen_embeddings.get(file_id, computed.get(file_id))
            if vector is None:
                raise ValueError(f"S24 feature for {file_id!r} was not captured")
            rows.append(vector)
        return torch.stack(rows, dim=0).to(dtype=torch.float32)

    def _new_optimizer(self) -> torch.optim.AdamW:
        if self.method == METHOD_SCRATCH:
            return torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.head.parameters()),
                lr=SCRATCH_LR_START,
                weight_decay=SCRATCH_WEIGHT_DECAY,
            )
        return torch.optim.AdamW(
            self.head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
        )

    def _initialise_fit(self, samples_by_id: Mapping[str, Any] | None) -> None:
        if self.method == METHOD_SCRATCH:
            samples = [
                self._require_sample(file_id, samples_by_id)
                for file_id in self.support.training_ids
            ]
            self.encoder.train()
            with torch.no_grad():
                initial = self._forward_embeddings(samples)
            self.normalizer = S24FeatureNormalizer.fit(initial)
            self.encoder.eval()
            self._optimizer = self._new_optimizer()
            return
        self._train_features = self._matrix_for(self.support.training_ids, samples_by_id)
        self._val_features = self._matrix_for(self.support.validation_ids, samples_by_id)
        self.normalizer = S24FeatureNormalizer.fit(self._train_features)
        self._optimizer = self._new_optimizer()

    def _require_sample(
        self, file_id: str, samples_by_id: Mapping[str, Any] | None
    ) -> Any:
        if samples_by_id is None or file_id not in samples_by_id:
            raise ValueError(
                f"S24 scratch training requires the waveform for {file_id!r}"
            )
        return samples_by_id[file_id]

    def _epoch_lr(self, epoch: int) -> float:
        progress = (epoch - 1) / max(1, SCRATCH_EPOCHS - 1)
        return SCRATCH_LR_END + 0.5 * (SCRATCH_LR_START - SCRATCH_LR_END) * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )

    def _frozen_epoch(self, epoch: int) -> dict[str, float]:
        if self._train_features is None or self._val_features is None:
            raise ValueError(
                "S24 frozen head training requires captured support features"
            )
        assert self.normalizer is not None
        assert self._optimizer is not None
        self.head.train()
        self.encoder.eval()
        x = self.normalizer.transform(self._train_features)
        y = torch.tensor(self.support.training_labels, dtype=torch.float32)
        order = torch.randperm(x.shape[0], generator=self._generator)
        criterion = nn.BCEWithLogitsLoss()
        total = 0.0
        for start in range(0, int(order.numel()), self.batch_size):
            index = order[start : start + self.batch_size]
            loss = criterion(self.head(x.index_select(0, index)), y.index_select(0, index))
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
            total += float(loss.detach()) * int(index.numel())
        train_loss = total / int(order.numel())
        self.head.eval()
        with torch.no_grad():
            val_logits = self.head(self.normalizer.transform(self._val_features))
            val_loss = float(
                criterion(
                    val_logits,
                    torch.tensor(self.support.validation_labels, dtype=torch.float32),
                ).detach()
            )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_head_state = _copy_state(self.head.state_dict())
            self.patience_left = self.patience
        else:
            self.patience_left -= 1
        return {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}

    def _scratch_epoch(
        self, epoch: int, samples_by_id: Mapping[str, Any] | None
    ) -> dict[str, float]:
        assert self.normalizer is not None
        assert self._optimizer is not None
        for group in self._optimizer.param_groups:
            group["lr"] = self._epoch_lr(epoch)
        self.encoder.train()
        self.head.train()
        y = torch.tensor(self.support.training_labels, dtype=torch.float32)
        order = torch.randperm(len(self.support.training_ids), generator=self._generator)
        criterion = nn.BCEWithLogitsLoss()
        total = 0.0
        for start in range(0, int(order.numel()), self.batch_size):
            index = order[start : start + self.batch_size].tolist()
            batch_samples = [
                self._require_sample(self.support.training_ids[position], samples_by_id)
                for position in index
            ]
            logits = self.head(
                self.normalizer.transform(self._forward_embeddings(batch_samples))
            )
            loss = criterion(logits, y[index])
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
            self.encoder.target_encoder.update(self.encoder.context_encoder)
            total += float(loss.detach()) * len(index)
        return {"epoch": float(epoch), "train_loss": total / len(self.support.training_ids)}

    def fit(
        self,
        samples_by_id: Mapping[str, Any] | None = None,
        *,
        epochs: int | None = None,
        ledger: S24LabelConsumptionLedger | None = None,
    ) -> dict[str, object]:
        """Fit or resume this learner under the frozen protocol budget.

        ``epochs`` requests a bounded number of additional epochs (for
        checkpoint/resume tests); ``None`` runs to the frozen limit or early
        stop. A resume keeps the original label event, so the ledger never
        treats the continuation as a second consumption of the same labels.
        """
        if self.stop_reason is not None:
            return self.summary()
        if epochs is not None and (
            isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0
        ):
            raise ValueError("S24 epoch request must be a non-negative integer")
        if ledger is not None:
            ledger.consume(
                scope=self.method,
                event_id=self.event_id,
                label_ids=self.support.training_ids,
                release_time_s=self.support.release_time_s,
                resuming=self.epoch > 0,
            )
            self.consumed = ledger.events()
        if self._pending_rng_state is not None:
            torch.set_rng_state(self._pending_rng_state)
            self._pending_rng_state = None
        before = None if self.method == METHOD_SCRATCH else self._encoder_digest()
        if self.epoch == 0:
            self._initialise_fit(samples_by_id)
        elif self.method != METHOD_SCRATCH and self._train_features is None:
            self._train_features = self._matrix_for(
                self.support.training_ids, samples_by_id
            )
            self._val_features = self._matrix_for(
                self.support.validation_ids, samples_by_id
            )
        target = (
            self.max_epochs
            if epochs is None
            else min(self.max_epochs, self.epoch + epochs)
        )
        while self.epoch < target:
            epoch = self.epoch + 1
            record = (
                self._scratch_epoch(epoch, samples_by_id)
                if self.method == METHOD_SCRATCH
                else self._frozen_epoch(epoch)
            )
            self.epoch = epoch
            self.history.append(record)
            if self.method != METHOD_SCRATCH and self.patience_left <= 0:
                self.stop_reason = "early_stopping"
                break
        if self.stop_reason is None and self.epoch == self.max_epochs:
            self.stop_reason = "epoch_limit"
        if self.stop_reason is not None and self.method != METHOD_SCRATCH:
            if self.best_head_state is not None:
                self.head.load_state_dict(cast(Any, _copy_state(self.best_head_state)))
        self.head.eval()
        if self.method == METHOD_SCRATCH:
            self.encoder.eval()
        if before is not None:
            assert_s24_frozen_unchanged(
                {"encoder_state_sha256": before},
                {"encoder_state_sha256": self._encoder_digest()},
            )
        return self.summary()

    def score_validation(
        self, samples_by_id: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        """Return validation loss and the mean abnormal probability."""
        if self.normalizer is None:
            raise ValueError("S24 learner has no fitted feature normalizer")
        if self.method == METHOD_SCRATCH:
            if samples_by_id is None:
                raise ValueError(
                    "S24 scratch validation requires the validation waveforms"
                )
            features = self.encode(
                [
                    self._require_sample(file_id, samples_by_id)
                    for file_id in self.support.validation_ids
                ],
                history_id=self.support.history_id,
            )
        else:
            if self._val_features is None:
                self._val_features = self._matrix_for(
                    self.support.validation_ids, samples_by_id
                )
            features = self._val_features
        with torch.no_grad():
            logits = self.head(self.normalizer.transform(features))
            loss = float(
                nn.BCEWithLogitsLoss()(
                    logits,
                    torch.tensor(self.support.validation_labels, dtype=torch.float32),
                ).detach()
            )
            probabilities = torch.sigmoid(logits).detach().cpu()
        abnormal = [
            float(probability)
            for probability, label in zip(
                probabilities, self.support.validation_labels
            )
            if label == 1
        ]
        self.final_val_loss = loss
        return {
            "val_loss": loss,
            "mean_abnormal_probability": (
                sum(abnormal) / len(abnormal) if abnormal else float("nan")
            ),
        }

    def summary(self) -> dict[str, object]:
        """Return the observable training summary of this cell."""
        return {
            "schema": S24_LEARNERS_SCHEMA_ID,
            "kind": "learner-summary",
            "method": self.method,
            "history_id": self.support.history_id,
            "budget": self.support.budget,
            "support_digest": self.support.support_digest,
            "identity_digest": self.identity_digest(),
            "event_id": self.event_id,
            "epochs_completed": self.epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": (
                None if not math.isfinite(self.best_val_loss) else self.best_val_loss
            ),
            "final_val_loss": self.final_val_loss,
            "stop_reason": self.stop_reason,
            "history": list(self.history),
            "trainables": self.trainable_record(),
        }


def assert_s24_trainables(learner: S24Learner) -> dict[str, object]:
    """Verify the declared trainable scope of one learner cell.

    Frozen arms must have zero encoder parameters requiring grad and an
    ``eval`` encoder; the scratch arm must have every encoder parameter
    trainable and must be in ``eval`` mode between fit steps. The head is
    always the only declared classification head and must have exactly
    ``129`` trainable parameters.
    """
    encoder_trainable = [
        name for name, p in learner.encoder.named_parameters() if p.requires_grad
    ]
    if learner.method == METHOD_SCRATCH:
        student_frozen = [
            name
            for name, p in learner.encoder.named_parameters()
            if not p.requires_grad and not name.startswith("target_encoder.")
        ]
        if student_frozen:
            raise ValueError(
                "S24 scratch learner must train every student encoder weight: "
                f"{student_frozen[:3]}"
            )
        target_trainable = [
            name for name, p in learner.encoder.target_encoder.named_parameters() if p.requires_grad
        ]
        if target_trainable:
            raise ValueError(
                "S24 scratch EMA target must stay under the EMA contract, not gradients"
            )
        if learner.encoder.training and learner.epoch > 0:
            raise ValueError("S24 scratch learner must be in eval mode outside a fit step")
    else:
        if encoder_trainable:
            raise ValueError(
                "S24 frozen learner has trainable encoder parameters: "
                f"{encoder_trainable[:3]}"
            )
        if learner.encoder.training:
            raise ValueError("S24 frozen learner encoder must stay in eval mode")
    head_names = [name for name, p in learner.head.named_parameters() if p.requires_grad]
    if len(head_names) != len(list(learner.head.named_parameters())):
        raise ValueError("S24 head parameters must all be trainable")
    head_count = sum(p.numel() for p in learner.head.parameters() if p.requires_grad)
    expected_head = int(learner.encoder.config.d_model) + 1
    if head_count != expected_head:
        raise ValueError(
            f"S24 head has {head_count} trainable parameters, expected {expected_head}"
        )
    if learner.encoder.config.d_model == 128 and head_count != HEAD_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"S24 protocol head must have {HEAD_TRAINABLE_PARAMETERS} trainable "
            f"parameters, got {head_count}"
        )
    return learner.trainable_record()


def _identity_base(
    *,
    method: str,
    support: S24TrainSupport,
    config: V1Config,
    mask_model_seed: int,
    encoder: Mapping[str, object],
    bank: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": S24_LEARNERS_SCHEMA_ID,
        "kind": "learner-identity",
        "protocol_id": PROTOCOL_ID,
        "method": method,
        "history_id": support.history_id,
        "budget": support.budget,
        "support_digest": support.support_digest,
        "model_config": config.to_dict(),
        "mask_model_seed": int(mask_model_seed),
        "encoder": dict(encoder),
        "bank": dict(bank),
        "head": {
            "kind": "linear",
            "in_features": int(config.d_model),
            "out_features": 1,
            "init_seed": HEAD_INIT_SEED,
            "trainable_parameters": HEAD_TRAINABLE_PARAMETERS,
        },
        "preprocessing": {
            "mask_rule": "sprint18-score-mask-v2",
            "batch_size": (
                SCRATCH_BATCH_SIZE if method == METHOD_SCRATCH else HEAD_BATCH_SIZE
            ),
            "head_lr": HEAD_LR,
            "head_weight_decay": HEAD_WEIGHT_DECAY,
            "head_max_epochs": HEAD_MAX_EPOCHS,
            "head_patience": HEAD_PATIENCE,
        },
    }


def _freeze_encoder(encoder: V1RepresentationModel) -> V1RepresentationModel:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


def build_s24_frozen_head_learner(
    *,
    bundle: S24B0Bundle,
    support: S24TrainSupport,
    frozen_embeddings: Mapping[str, torch.Tensor] | None = None,
) -> S24Learner:
    """Build the ``M1`` frozen-B0 + linear-head cell.

    The encoder is the strictly restored accepted B0 model; it is frozen and
    left in ``eval`` mode so ``ConditionalBatchNorm`` statistics cannot be
    refitted. The bank identity is carried for provenance (this method adds no
    bank computation).
    """
    if bundle.bank.bank.embeddings is None:
        raise ValueError("S24 frozen-head learner requires a restored reference bank")
    live = s24_live_state_digests(bundle.model, bundle.bank)
    identity = _identity_base(
        method=METHOD_FROZEN_HEAD,
        support=support,
        config=bundle.model.config,
        mask_model_seed=bundle.model_seed,
        encoder={
            "kind": "accepted-b0-restored",
            "arm_id": bundle.arm_id,
            "model_seed": bundle.model_seed,
            "checkpoint_sha256": bundle.checkpoint.checkpoint_sha256,
            "model_state_sha256": live["model_state_sha256"],
            "ema_state_sha256": live["ema_state_sha256"],
            "frozen_state_sha256": bundle.frozen_state_sha256,
            "preprocessing_sha256": bundle.preprocessing_sha256,
        },
        bank={
            "kind": "sprint18-fit-reference-bank",
            "sha256": bundle.bank.provenance.bank_sha256,
            "k": int(bundle.bank.bank.k),
            "rows": int(bundle.bank.provenance.source_rows),
        },
    )
    return S24Learner(
        method=METHOD_FROZEN_HEAD,
        encoder=_freeze_encoder(bundle.model),
        support=support,
        identity=identity,
        mask_model_seed=bundle.model_seed,
        shuffle_seed=HEAD_SHUFFLE_SEED,
        max_epochs=HEAD_MAX_EPOCHS,
        patience=HEAD_PATIENCE,
        frozen_embeddings=frozen_embeddings,
        learning_rate=HEAD_LR,
        batch_size=HEAD_BATCH_SIZE,
    )


def build_s24_random_frozen_learner(
    *,
    model_config: V1Config,
    support: S24TrainSupport,
    init_seed: int,
    mask_model_seed: int,
    frozen_embeddings: Mapping[str, torch.Tensor] | None = None,
) -> S24Learner:
    """Build the ``M2`` random-initialized frozen-encoder control.

    The architecture and head protocol are identical to ``M1``; only the
    encoder initialization differs (frozen seeds ``241801``-``241803``). The
    per-file mask seed stays keyed by the paired accepted B0 seed so both arms
    see identical file masks.
    """
    if init_seed not in RANDOM_ENCODER_SEEDS:
        raise ValueError("S24 random encoder seed is outside the frozen control set")
    if mask_model_seed not in MODEL_SEEDS:
        raise ValueError("S24 mask seed must stay inside the frozen Sprint18 namespace")
    torch.manual_seed(int(init_seed))
    encoder = _freeze_encoder(V1RepresentationModel(model_config))
    init_state = state_sha256(encoder.state_dict())
    identity = _identity_base(
        method=METHOD_RANDOM_FROZEN,
        support=support,
        config=model_config,
        mask_model_seed=mask_model_seed,
        encoder={
            "kind": "random-frozen",
            "init_seed": int(init_seed),
            "model_state_sha256": init_state,
            "frozen_state_sha256": init_state,
        },
        bank={"kind": "none", "reason": "random encoder has no fitted reference bank"},
    )
    return S24Learner(
        method=METHOD_RANDOM_FROZEN,
        encoder=encoder,
        support=support,
        identity=identity,
        mask_model_seed=mask_model_seed,
        shuffle_seed=HEAD_SHUFFLE_SEED,
        max_epochs=HEAD_MAX_EPOCHS,
        patience=HEAD_PATIENCE,
        frozen_embeddings=frozen_embeddings,
        learning_rate=HEAD_LR,
        batch_size=HEAD_BATCH_SIZE,
    )


def build_s24_scratch_learner(
    *,
    model_config: V1Config,
    support: S24TrainSupport,
    mask_model_seed: int,
    initial_state_sha256: str | None = None,
) -> S24Learner:
    """Build the ``M3`` scratch-supervised cell (all weights trainable)."""
    if mask_model_seed not in MODEL_SEEDS:
        raise ValueError("S24 mask seed must stay inside the frozen Sprint18 namespace")
    torch.manual_seed(SCRATCH_INIT_SEED)
    encoder = V1RepresentationModel(model_config)
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    init_state = state_sha256(encoder.state_dict())
    if initial_state_sha256 is not None and initial_state_sha256 != init_state:
        raise ValueError("S24 scratch initialization differs from the declared identity")
    identity = _identity_base(
        method=METHOD_SCRATCH,
        support=support,
        config=model_config,
        mask_model_seed=mask_model_seed,
        encoder={
            "kind": "scratch-initialized",
            "init_seed": SCRATCH_INIT_SEED,
            "init_state_sha256": init_state,
        },
        bank={"kind": "none", "reason": "scratch-supervised model has no reference bank"},
    )
    identity["preprocessing"] = dict(
        cast(Mapping[str, object], identity["preprocessing"]),
        scratch_lr_start=SCRATCH_LR_START,
        scratch_lr_end=SCRATCH_LR_END,
        scratch_epochs=SCRATCH_EPOCHS,
        scratch_weight_decay=SCRATCH_WEIGHT_DECAY,
    )
    return S24Learner(
        method=METHOD_SCRATCH,
        encoder=encoder,
        support=support,
        identity=identity,
        mask_model_seed=mask_model_seed,
        shuffle_seed=SCRATCH_SHUFFLE_SEED,
        max_epochs=SCRATCH_EPOCHS,
        patience=0,
        learning_rate=SCRATCH_LR_START,
        batch_size=SCRATCH_BATCH_SIZE,
    )


def fit_s24_learner(
    learner: S24Learner,
    samples_by_id: Mapping[str, Any] | None = None,
    *,
    epochs: int | None = None,
    ledger: S24LabelConsumptionLedger | None = None,
) -> dict[str, object]:
    """Fit or resume one learner; see :meth:`S24Learner.fit`."""
    return learner.fit(samples_by_id, epochs=epochs, ledger=ledger)


def _checkpoint_digest(payload: Mapping[str, object]) -> str:
    state = cast(Mapping[str, object], payload["state"])
    provenance = {
        "schema_id": payload.get("schema_id"),
        "protocol_id": payload.get("protocol_id"),
        "method": payload.get("method"),
        "identity_digest": payload.get("identity_digest"),
        "support": payload.get("support"),
        "support_digest": payload.get("support_digest"),
        "reveal": payload.get("reveal"),
        "trainables": payload.get("trainables"),
        "model_config": payload.get("model_config"),
        "state_digests": payload.get("state_digests"),
        "state": {
            key: state.get(key)
            for key in (
                "epoch",
                "best_epoch",
                "patience_left",
                "stop_reason",
                "event_id",
            )
        },
    }
    return _canonical(provenance)


def save_s24_learner_checkpoint(
    path: str | Path, learner: S24Learner
) -> dict[str, object]:
    """Write a provenance-bound, resumable learner checkpoint.

    The payload binds the exact identity, revealed support with labels, reveal
    boundary, declared trainables, model configuration, tensor-state digests
    and the full resumable state (head/encoder/optimizer/both RNG streams).
    """
    if learner.normalizer is None:
        raise ValueError("S24 checkpoint requires a fitted learner")
    if learner._optimizer is None:
        raise ValueError("S24 checkpoint requires optimizer state")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    head_state = _copy_state(learner.head.state_dict())
    encoder_state = _copy_state(learner.encoder.state_dict())
    payload: dict[str, object] = {
        "schema_id": S24_LEARNER_CHECKPOINT_SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "method": learner.method,
        "identity": dict(learner.identity),
        "identity_digest": learner.identity_digest(),
        "support": learner.support.record(),
        "support_digest": learner.support.support_digest,
        "reveal": {
            "release_time_s": float(learner.support.release_time_s),
            "now_s": float(learner.support.now_s),
            "released_file_ids": list(learner.support.label_ids),
        },
        "trainables": learner.trainable_record(),
        "model_config": learner.encoder.config.to_dict(),
        "state_digests": {
            "head_state_sha256": state_sha256(head_state),
            "encoder_state_sha256": state_sha256(encoder_state),
            "normalizer_mean_sha256": state_sha256(learner.normalizer.mean),
            "normalizer_std_sha256": state_sha256(learner.normalizer.std),
        },
        "state": {
            "epoch": int(learner.epoch),
            "head": head_state,
            "encoder": encoder_state,
            "normalizer_mean": learner.normalizer.mean.detach().clone(),
            "normalizer_std": learner.normalizer.std.detach().clone(),
            "train_features": (
                None if learner._train_features is None else learner._train_features.clone()
            ),
            "val_features": (
                None if learner._val_features is None else learner._val_features.clone()
            ),
            "optimizer": learner._optimizer.state_dict(),
            "shuffle_generator": learner._generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "history": list(learner.history),
            "best_epoch": int(learner.best_epoch),
            "best_val_loss": float(learner.best_val_loss),
            "best_head_state": (
                None
                if learner.best_head_state is None
                else _copy_state(learner.best_head_state)
            ),
            "patience_left": int(learner.patience_left),
            "stop_reason": learner.stop_reason,
            "final_val_loss": learner.final_val_loss,
            "event_id": learner.event_id,
            "consumed": learner.consumed,
        },
    }
    payload["checkpoint_digest"] = _checkpoint_digest(payload)
    torch.save(payload, target)
    return {
        "schema_id": S24_LEARNER_CHECKPOINT_SCHEMA_ID,
        "path": str(target),
        "method": learner.method,
        "checkpoint_digest": payload["checkpoint_digest"],
        "identity_digest": learner.identity_digest(),
        "support_digest": learner.support.support_digest,
        "epoch": learner.epoch,
        "stop_reason": learner.stop_reason,
    }


def _verify_support(
    views: HistoryViews,
    record: Mapping[str, object],
    *,
    stored_digest: object,
    gate: LabelReleaseGate | None,
) -> S24TrainSupport:
    if record.get("schema") != S24_LEARNERS_SCHEMA_ID or record.get("kind") != "train-support":
        raise ValueError("S24 checkpoint support record is malformed")
    if record.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("S24 checkpoint support uses a different protocol")
    if record.get("views_manifest_digest") != views.manifest_digest:
        raise ValueError("S24 checkpoint support was built from a different view manifest")
    if record.get("history_id") != views.history_id:
        raise ValueError("S24 checkpoint support belongs to a different history")
    budget = record.get("budget")
    if not isinstance(budget, str) or budget not in views.budgets:
        raise ValueError("S24 checkpoint support has an unknown budget")
    support = S24TrainSupport(
        history_id=views.history_id,
        budget=budget,
        training_ids=tuple(str(v) for v in cast(Sequence[object], record["training_ids"])),
        validation_ids=tuple(str(v) for v in cast(Sequence[object], record["validation_ids"])),
        training_labels=tuple(
            int(cast(int, v)) for v in cast(Sequence[object], record["training_labels"])
        ),
        validation_labels=tuple(
            int(cast(int, v)) for v in cast(Sequence[object], record["validation_labels"])
        ),
        release_time_s=float(cast(float, record["release_time_s"])),
        now_s=float(cast(float, record["now_s"])),
        views_manifest_digest=views.manifest_digest,
        support_digest=cast(str, stored_digest) if isinstance(stored_digest, str) else "",
    )
    if _canonical(support.record()) != stored_digest:
        raise ValueError("S24 checkpoint support digest does not reproduce")
    if support.training_ids != tuple(views.budgets[budget]):
        raise ValueError("S24 checkpoint training IDs differ from the frozen budget")
    if support.validation_ids != tuple(views.frame("V-VAL")):
        raise ValueError("S24 checkpoint validation IDs differ from V-VAL")
    if support.now_s < support.release_time_s:
        raise ValueError("S24 checkpoint reveals labels before their release time")
    if set(support.training_labels) != {0, 1}:
        raise ValueError("S24 checkpoint training support lost a class")
    if gate is not None:
        if gate.views.manifest_digest != views.manifest_digest:
            raise ValueError("S24 restore gate is bound to a different view manifest")
        released = gate.released_labels(support.now_s)
        unreleased = [i for i in support.label_ids if i not in released]
        if unreleased:
            raise ValueError(
                f"S24 checkpoint support has {len(unreleased)} label(s) unreleased at "
                f"the recorded boundary (first: {unreleased[0]!r})"
            )
        for file_id, label in zip(support.training_ids, support.training_labels):
            if s24_label_target(released[file_id]) != label:
                raise ValueError("S24 checkpoint training label differs from release")
        for file_id, label in zip(support.validation_ids, support.validation_labels):
            if s24_label_target(released[file_id]) != label:
                raise ValueError("S24 checkpoint validation label differs from release")
        schedule = max(gate.release_time_of(i) for i in support.label_ids)
        if schedule != support.release_time_s:
            raise ValueError("S24 checkpoint release time differs from the frozen schedule")
    return support


def restore_s24_learner_checkpoint(
    path: str | Path,
    *,
    views: HistoryViews,
    bundle: S24B0Bundle | None = None,
    gate: LabelReleaseGate | None = None,
    ledger: S24LabelConsumptionLedger | None = None,
) -> S24Learner:
    """Restore a learner checkpoint after verifying every bound identity.

    Support must still be exactly the frozen budget plus ``V-VAL`` for the same
    history, every label must be released at the recorded boundary, the encoder
    identity (accepted checkpoint/model state, random-init state, or scratch
    state) must match, and the checkpoint digest must be intact. Any mismatch
    fails closed before a learner object is returned.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"S24 learner checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_id") != S24_LEARNER_CHECKPOINT_SCHEMA_ID
    ):
        raise ValueError("S24 checkpoint schema is unknown")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("S24 checkpoint protocol differs from the frozen protocol")
    method = payload.get("method")
    if method not in S24_METHODS:
        raise ValueError("S24 checkpoint method is unknown")
    identity = payload.get("identity")
    support_record = payload.get("support")
    state = payload.get("state")
    if not isinstance(identity, Mapping) or not isinstance(support_record, Mapping):
        raise ValueError("S24 checkpoint identity or support is missing")
    if not isinstance(state, Mapping):
        raise ValueError("S24 checkpoint training state is missing")
    if payload.get("checkpoint_digest") != _checkpoint_digest(payload):
        raise ValueError("S24 checkpoint digest does not match its provenance record")
    state_digests = payload.get("state_digests")
    if not isinstance(state_digests, Mapping):
        raise ValueError("S24 checkpoint state digests are missing")
    support = _verify_support(
        views,
        support_record,
        stored_digest=payload.get("support_digest"),
        gate=gate,
    )
    config = V1Config(**cast(Mapping[str, Any], payload.get("model_config")))
    encoder_record = cast(Mapping[str, object], identity["encoder"])
    mask_seed = identity.get("mask_model_seed")
    if not isinstance(mask_seed, int) or isinstance(mask_seed, bool):
        raise ValueError("S24 checkpoint mask seed is malformed")
    if method == METHOD_FROZEN_HEAD:
        if bundle is None:
            raise ValueError("S24 M1 restore requires the restored accepted B0 bundle")
        live = s24_live_state_digests(bundle.model, bundle.bank)
        if encoder_record.get("checkpoint_sha256") != bundle.checkpoint.checkpoint_sha256:
            raise ValueError("S24 checkpoint was fitted from a different accepted checkpoint")
        if encoder_record.get("model_state_sha256") != live["model_state_sha256"]:
            raise ValueError("S24 checkpoint encoder state differs from the restored model")
        if encoder_record.get("ema_state_sha256") != live["ema_state_sha256"]:
            raise ValueError("S24 checkpoint EMA state differs from the restored model")
        bank_record = cast(Mapping[str, object], identity["bank"])
        if bank_record.get("sha256") != bundle.bank.provenance.bank_sha256:
            raise ValueError("S24 checkpoint bank identity differs from the restored bank")
        encoder = _freeze_encoder(bundle.model)
    elif method == METHOD_RANDOM_FROZEN:
        init_seed = encoder_record.get("init_seed")
        if init_seed not in RANDOM_ENCODER_SEEDS:
            raise ValueError("S24 random-frozen checkpoint init seed is unknown")
        torch.manual_seed(int(cast(int, init_seed)))
        encoder = _freeze_encoder(V1RepresentationModel(config))
        if state_sha256(encoder.state_dict()) != encoder_record.get("model_state_sha256"):
            raise ValueError("S24 random-frozen encoder state does not reproduce")
    else:
        torch.manual_seed(SCRATCH_INIT_SEED)
        encoder = V1RepresentationModel(config)
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
    stored_encoder_state = cast(Mapping[str, object], state["encoder"])
    if method == METHOD_SCRATCH:
        encoder.load_state_dict(cast(Any, stored_encoder_state))
        encoder.eval()
    if state_sha256(encoder.state_dict()) != state_digests.get("encoder_state_sha256"):
        raise ValueError("S24 restored encoder state does not match its digest")
    normalizer = S24FeatureNormalizer(
        mean=cast(torch.Tensor, state["normalizer_mean"]).detach().clone(),
        std=cast(torch.Tensor, state["normalizer_std"]).detach().clone(),
    )
    if (
        state_sha256(normalizer.mean) != state_digests.get("normalizer_mean_sha256")
        or state_sha256(normalizer.std) != state_digests.get("normalizer_std_sha256")
    ):
        raise ValueError("S24 restored normalizer statistics do not match their digest")
    learner = S24Learner(
        method=method,
        encoder=encoder,
        support=support,
        identity=identity,
        mask_model_seed=int(mask_seed),
        shuffle_seed=(
            SCRATCH_SHUFFLE_SEED if method == METHOD_SCRATCH else HEAD_SHUFFLE_SEED
        ),
        max_epochs=SCRATCH_EPOCHS if method == METHOD_SCRATCH else HEAD_MAX_EPOCHS,
        patience=0 if method == METHOD_SCRATCH else HEAD_PATIENCE,
        learning_rate=SCRATCH_LR_START if method == METHOD_SCRATCH else HEAD_LR,
        batch_size=SCRATCH_BATCH_SIZE if method == METHOD_SCRATCH else HEAD_BATCH_SIZE,
    )
    learner.head.load_state_dict(cast(Any, state["head"]))
    learner.head.eval()
    if state_sha256(learner.head.state_dict()) != state_digests.get("head_state_sha256"):
        raise ValueError("S24 restored head state does not match its digest")
    if learner.identity_digest() != payload.get("identity_digest"):
        raise ValueError("S24 restored learner identity digest differs from the checkpoint")
    if learner.trainable_record() != payload.get("trainables"):
        raise ValueError("S24 restored learner trainable scope differs from the checkpoint")
    if learner.event_id != state.get("event_id"):
        raise ValueError("S24 restored learner label event differs from the checkpoint")
    learner.normalizer = normalizer
    learner._optimizer = learner._new_optimizer()
    learner._optimizer.load_state_dict(cast(dict[str, Any], state["optimizer"]))
    learner._generator.set_state(cast(torch.Tensor, state["shuffle_generator"]))
    learner._pending_rng_state = cast(torch.Tensor, state["torch_rng_state"]).clone()
    learner.epoch = int(cast(int, state["epoch"]))
    learner.history = [
        dict(cast(Mapping[str, float], row))
        for row in cast(Sequence[Mapping[str, float]], state["history"])
    ]
    learner.best_epoch = int(cast(int, state["best_epoch"]))
    learner.best_val_loss = float(cast(float, state["best_val_loss"]))
    best_state = state.get("best_head_state")
    learner.best_head_state = (
        _copy_state(cast(Mapping[str, object], best_state))
        if isinstance(best_state, Mapping)
        else None
    )
    learner.patience_left = int(cast(int, state["patience_left"]))
    learner.stop_reason = cast(str | None, state.get("stop_reason"))
    final_loss = state.get("final_val_loss")
    learner.final_val_loss = None if final_loss is None else float(cast(float, final_loss))
    consumed = state.get("consumed")
    learner.consumed = (
        {
            str(key): dict(cast(Mapping[str, object], value))
            for key, value in cast(Mapping[str, object], consumed).items()
        }
        if isinstance(consumed, Mapping)
        else {}
    )
    train_features = state.get("train_features")
    if isinstance(train_features, torch.Tensor):
        learner._train_features = train_features.detach().clone()
    val_features = state.get("val_features")
    if isinstance(val_features, torch.Tensor):
        learner._val_features = val_features.detach().clone()
    if learner.method != METHOD_SCRATCH:
        if learner._train_features is None or learner._val_features is None:
            raise ValueError("S24 frozen checkpoint is missing its support features")
    if ledger is not None:
        for event_id, record in learner.consumed.items():
            entry = cast(Mapping[str, object], record)
            ledger.consume(
                scope=cast(str, entry["scope"]),
                event_id=event_id,
                label_ids=cast(Sequence[str], entry["label_ids"]),
                release_time_s=float(cast(float, entry["release_time_s"])),
                resuming=True,
            )
    return learner


__all__ = [
    "EVALUATION_FRAMES",
    "HEAD_BATCH_SIZE",
    "HEAD_INIT_SEED",
    "HEAD_LR",
    "HEAD_MAX_EPOCHS",
    "HEAD_PATIENCE",
    "HEAD_SHUFFLE_SEED",
    "HEAD_TRAINABLE_PARAMETERS",
    "LABEL_EVALUATION",
    "LABEL_TRAIN",
    "LABEL_VALIDATION",
    "METHOD_FROZEN_HEAD",
    "METHOD_RANDOM_FROZEN",
    "METHOD_SCRATCH",
    "RANDOM_ENCODER_SEEDS",
    "S24_LEARNERS_SCHEMA_ID",
    "S24_LEARNER_CHECKPOINT_SCHEMA_ID",
    "S24_METHODS",
    "SCRATCH_BATCH_SIZE",
    "SCRATCH_EPOCHS",
    "SCRATCH_INIT_SEED",
    "SCRATCH_LR_END",
    "SCRATCH_LR_START",
    "SCRATCH_SHUFFLE_SEED",
    "S24FeatureNormalizer",
    "S24LabelConsumptionLedger",
    "S24Learner",
    "S24LinearHead",
    "S24TrainSupport",
    "assert_s24_support_role",
    "assert_s24_trainables",
    "build_s24_frozen_head_learner",
    "build_s24_random_frozen_learner",
    "build_s24_scratch_learner",
    "build_s24_train_support",
    "fit_s24_learner",
    "restore_s24_learner_checkpoint",
    "s24_label_target",
    "save_s24_learner_checkpoint",
]
