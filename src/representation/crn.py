"""Context-dependent residual normalization (N4-CRN) scale-model capacity path.

Sprint 18 Task 13 scope only: the fixed context-only scale model, Fit-only
quantile bounds, Gaussian residual NLL, saturation diagnostics, and the
deterministic Fit-history cross-fit driver. Representation integration
(frozen-cache binding, Calibration q95, ``S_pop`` parity scoring, and the final
all-Fit scoring model) belongs to later tasks and is deliberately absent here.

Normative sources: ``experiments/sprint18-context-residual-normalization-spec.md``
(v1) as amended by
``experiments/sprint18-context-residual-normalization-spec-v2.md`` (v2), under
protocol v2. Role permissions and the fixed optimizer/epoch/schedule contract
are inherited from the exact v2 composition; this module adds no new permission.

Residual-normalization interpretation (explicit, for prerequisite review):
``normalize`` in the spec residual
``e_t = mean_d((normalize(z_target_t) - normalize(z_pred_t))^2)`` is implemented
as per-patch functional layer normalization over the latent width, matching the
N2/N3 teacher-normalization convention in this repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

if TYPE_CHECKING:
    from representation.sprint18_harness import Sprint18ArmConfig, Task6Binding

#: Fixed N4 scale-model capacity: single hidden width, scalar output.
N4_SCALE_HIDDEN_WIDTH = 32
#: Fixed N4 scale-fit contract (no early stopping, no Development selection).
N4_SCALE_EPOCHS = 50
N4_SCALE_BATCH_SIZE = 512
N4_SCALE_LR = 1e-3
N4_SCALE_WEIGHT_DECAY = 1e-4
N4_SCALE_CLIP_NORM = 1.0
#: Structural scoring-validity gate: Development healthy ``v_cap`` saturation
#: above this fraction invalidates the arm; bounds are never retuned.
N4_SATURATION_CAP_GATE = 0.20
#: Absolute variance floor guard from the frozen bound formulas.
N4_VARIANCE_FLOOR_MIN = 1e-6
#: Frozen scoring-capacity envelope for the scale head.
N4_SCALE_PARAM_LIMIT = 100_000
#: Log-scale tolerance identifying clamped (saturated) predictions.
N4_SATURATION_LOG_TOL = 1e-6
#: Frozen v3 initialization parameterization identity (Fit-mean-anchored output bias).
N4_SCALE_PARAMETERIZATION = "fit-mean-anchored-bias-v1"
#: Frozen v3 fail-closed startup guard contract identity.
N4_GUARD_CONTRACT = "n4-v3-guards-v1"
#: Versioned scale-checkpoint envelope identity; bare legacy state dicts are refused.
N4_SCALE_CHECKPOINT_SCHEMA = "s18-n4-scale-checkpoint-v2"
#: Frozen deployed latent width and the exact parameter roster of the scale head.
N4_SCALE_LATENT_WIDTH = 128
N4_SCALE_PARAMETER_ROSTER: tuple[tuple[str, int], ...] = (
    ("norm.weight", N4_SCALE_LATENT_WIDTH),
    ("norm.bias", N4_SCALE_LATENT_WIDTH),
    ("hidden.weight", N4_SCALE_HIDDEN_WIDTH * N4_SCALE_LATENT_WIDTH),
    ("hidden.bias", N4_SCALE_HIDDEN_WIDTH),
    ("out.weight", N4_SCALE_HIDDEN_WIDTH),
    ("out.bias", 1),
)
#: Exact deployed scale-head parameter count at the frozen latent width (4,417).
N4_SCALE_PARAM_COUNT = sum(numel for _name, numel in N4_SCALE_PARAMETER_ROSTER)

#: Certified per-history N4 in-scope record counts (Task13B A02,
#: ``experiments/sprint18-task13-n4-capacity-v3-a02.json``). An N4 training
#: record is one eligible *masked* valid Fit patch record, so this roster is
#: NOT ``sprint18_harness.FIT_VALID_PATCHES`` (every valid Fit patch). The
#: mask budget is count-deterministic, so the identical counts apply to all
#: three model seeds; normative runners re-derive them from the bound final-B0
#: prediction caches before any fit and abort on any difference.
N4_SCALE_HISTORY_RECORD_ROSTER: tuple[tuple[str, int], ...] = (
    ("S18I-ITER-0005-FIT-01", 22_328),
    ("S18I-ITER-0005-FIT-02", 23_098),
    ("S18I-ITER-0005-FIT-03", 21_563),
)
#: Exact certified N4 in-scope record total (66,989).
N4_SCALE_FIT_RECORDS = sum(count for _history_id, count in N4_SCALE_HISTORY_RECORD_ROSTER)
#: Exact certified per-history map for fold-scope expectations.
N4_SCALE_HISTORY_RECORDS: Mapping[str, int] = {
    history_id: count for history_id, count in N4_SCALE_HISTORY_RECORD_ROSTER
}
#: Frozen per-scope optimizer-step rule: ``50 * ceil(N_scope / 512)``.
def n4_scale_scope_steps(record_count: int) -> int:
    """Return the frozen 50-epoch optimizer-step count for one N4 scope."""
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 1:
        raise ValueError("N4 scope record count must be a positive integer")
    return N4_SCALE_EPOCHS * math.ceil(record_count / N4_SCALE_BATCH_SIZE)


class ContextResidualScaleModel(nn.Module):
    """Fixed-capacity context-only residual scale predictor.

    ``log_v = clamp(Linear_2(GELU(Linear_1(LayerNorm(h)))), log(v_floor),
    log(v_cap))`` with hidden width 32 and scalar output. The forward signature
    accepts only detached online context ``h`` plus the frozen Fit-only bounds;
    target latents, residuals, waveforms, labels, and metadata cannot enter by
    API construction (no such parameters exist).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("N4 scale model requires a positive integer latent width")
        self.d_model = int(d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.hidden = nn.Linear(self.d_model, N4_SCALE_HIDDEN_WIDTH)
        self.out = nn.Linear(N4_SCALE_HIDDEN_WIDTH, 1)

    def forward(
        self, context: torch.Tensor, *, v_floor: float, v_cap: float
    ) -> dict[str, torch.Tensor]:
        """Predict clamped log-variance for one batch of context rows."""
        if not isinstance(context, torch.Tensor) or context.ndim != 2 or context.shape[0] < 1:
            raise ValueError("N4 scale context must be a non-empty 2D [N, D] tensor")
        if context.shape[1] != self.d_model:
            raise ValueError("N4 scale context width differs from the bound model width")
        if not torch.isfinite(context).all():
            raise ValueError("N4 scale context must be finite")
        for name, bound in (("v_floor", v_floor), ("v_cap", v_cap)):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"N4 {name} bound must be a number")
            if not math.isfinite(float(bound)) or float(bound) <= 0.0:
                raise ValueError(f"N4 {name} bound must be finite and positive")
        floor = float(v_floor)
        cap = float(v_cap)
        if cap < floor:
            raise ValueError("N4 v_cap must not lie below v_floor")
        raw = self.out(torch.nn.functional.gelu(self.hidden(self.norm(context)))).squeeze(-1)
        if not torch.isfinite(raw).all():
            raise ValueError("N4 scale pre-clamp output is nonfinite")
        log_v = torch.clamp(raw, min=math.log(floor), max=math.log(cap))
        return {"log_v": log_v, "v": torch.exp(log_v)}


@dataclass(frozen=True)
class N4ScaleRecord:
    """One eligible cached Fit-healthy patch record for scale fitting.

    Carries only detached online context and its scalar residual target plus
    canonical identity. File labels, anomaly metadata, waveforms, and target
    latents are never attached.
    """

    history_id: str
    file_id: str
    patch_index: int
    context: torch.Tensor
    residual: float


def init_n4_scale_model(d_model: int, model_seed: int) -> ContextResidualScaleModel:
    """Build a deterministically initialized scale model without RNG side effects.

    Seeds a forked CPU RNG with the bound model seed before construction, so the
    caller-visible global RNG state is unchanged. Same inputs give bitwise
    identical parameters.
    """
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("N4 scale initialization requires an integer model seed")
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(int(model_seed))
        model = ContextResidualScaleModel(d_model)
    return model


class N4ScaleGuardError(RuntimeError):
    """Fail-closed ``n4-v3-guards-v1`` failure carrying its identity and diagnostics.

    Raised before any optimizer state exists (initialized-interiority guard) or
    before the unchanged clip/step of the first normal backward
    (first-step-gradient guard), so a failure applies zero optimizer updates.
    """

    def __init__(self, guard_id: str, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(f"{guard_id}: {message}")
        self.guard_id = str(guard_id)
        self.diagnostics = dict(diagnostics)


def n4_anchor_scale_model(
    scale_model: ContextResidualScaleModel, residuals: Sequence[float]
) -> dict[str, object]:
    """Anchor ``out.bias`` at ``log(v_ref)`` for the Fit-only in-scope residual mean.

    ``v_ref`` is the float64 mean of the exact in-scope Fit training residuals.
    Only ``out.bias`` is written; ``out.weight``, ``hidden`` and ``norm`` keep
    their seeded values (the last layer is never zeroed). The bias anchor is an
    initialization of the scalar offset, not a claim that every predicted
    ``log_v`` equals ``log(v_ref)``. Non-finite or non-positive ``v_ref`` fails
    closed, so a zero residual pool can never produce ``log(0) = -inf``.
    """
    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 scale anchoring requires the context-only scale model")
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.shape[0] < 1:
        raise ValueError("N4 scale anchoring requires a non-empty residual pool")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("N4 anchor residuals must be finite and non-negative")
    v_ref = float(values.mean())
    if not math.isfinite(v_ref) or v_ref <= 0.0:
        raise ValueError("N4 anchor reference v_ref must be finite and positive")
    log_v_ref = math.log(v_ref)
    if not math.isfinite(log_v_ref):
        raise ValueError("N4 anchor log(v_ref) must be finite")
    with torch.no_grad():
        scale_model.out.bias.fill_(log_v_ref)
    return {"v_ref": v_ref, "log_v_ref": log_v_ref, "record_count": int(values.shape[0])}


def _n4_scale_bounds(v_floor: float, v_cap: float) -> tuple[float, float, float, float]:
    """Validate one frozen bound pair and return ``(floor, cap, log_floor, log_cap)``."""
    for name, bound in (("v_floor", v_floor), ("v_cap", v_cap)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"N4 {name} bound must be a number")
        if not math.isfinite(float(bound)) or float(bound) <= 0.0:
            raise ValueError(f"N4 {name} bound must be finite and positive")
    floor = float(v_floor)
    cap = float(v_cap)
    if cap < floor:
        raise ValueError("N4 v_cap must not lie below v_floor")
    log_floor = math.log(floor)
    log_cap = math.log(cap)
    if not log_cap > log_floor:
        raise ValueError("N4 initialized-interiority guard requires log(v_cap) > log(v_floor)")
    return floor, cap, log_floor, log_cap


def n4_guard_initialized_interiority(
    scale_model: ContextResidualScaleModel,
    contexts: Sequence[torch.Tensor],
    *,
    v_floor: float,
    v_cap: float,
) -> dict[str, object]:
    """Guard A: forward-only full-scope initialized-interiority check.

    Runs after the anchor and the canonical context materialization and before
    ``torch.optim.AdamW`` is constructed. Reads the anchored parameters only:
    no parameter/buffer/``.grad``/mode write, no RNG or sampler call, no
    backward, no extra zeroing. The unchanged clamped forward is evaluated
    under ``torch.no_grad()`` in contiguous chunks of at most 512 records over
    the WHOLE in-scope pool, and every returned ``log_v`` must be strictly
    interior by more than the frozen ``1e-6`` tolerance. Structural
    precondition violations raise :class:`ValueError` before any forward; a
    clamp hit or a non-finite/invalid forward (including a non-finite context
    rejected by the public forward's own validation) fails closed with
    :class:`N4ScaleGuardError` after zero optimizer updates.
    """
    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 initialized-interiority guard requires the context-only scale model")
    parameters = sum(p.numel() for p in scale_model.parameters())
    if parameters != N4_SCALE_PARAM_COUNT or parameters > N4_SCALE_PARAM_LIMIT:
        raise ValueError(
            "N4 initialized-interiority guard requires the exact frozen 4417-parameter scale head"
        )
    if any(True for _ in scale_model.buffers()):
        raise ValueError("N4 initialized-interiority guard requires a buffer-free scale model")
    for module in scale_model.modules():
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
             nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            raise ValueError(
                "N4 initialized-interiority guard forbids stochastic or batch-statistics layers"
            )
    _floor, _cap, log_floor, log_cap = _n4_scale_bounds(v_floor, v_cap)
    record_total = len(contexts)
    if record_total < 1:
        raise ValueError("N4 initialized-interiority guard requires a non-empty in-scope pool")
    reference = scale_model.out.bias
    # Structural checks are metadata-only: no per-record tensor value is read
    # and the already-materialized context sequence is never copied.
    for row in contexts:
        if not isinstance(row, torch.Tensor) or row.ndim != 1 or row.shape[0] != scale_model.d_model:
            raise ValueError("N4 guard context must be a 1D latent row of the bound width")
        if row.device != reference.device or row.dtype != reference.dtype:
            raise ValueError("N4 guard context dtype/device must match the scale head")
    tolerance = N4_SATURATION_LOG_TOL
    record_count = 0
    log_v_min = math.inf
    log_v_max = -math.inf
    at_floor = 0
    at_cap = 0

    def _diagnostics() -> dict[str, object]:
        return {
            "record_count": record_count,
            "log_v_min": log_v_min if record_count else None,
            "log_v_max": log_v_max if record_count else None,
            "log_floor": log_floor,
            "log_cap": log_cap,
            "floor_margin": (log_v_min - log_floor) if record_count else None,
            "cap_margin": (log_cap - log_v_max) if record_count else None,
            "frac_at_floor": (at_floor / record_count) if record_count else None,
            "frac_at_cap": (at_cap / record_count) if record_count else None,
            "tolerance": tolerance,
        }

    with torch.no_grad():
        for start in range(0, record_total, N4_SCALE_BATCH_SIZE):
            stop = min(start + N4_SCALE_BATCH_SIZE, record_total)
            chunk = torch.stack([contexts[index] for index in range(start, stop)], dim=0)
            # The unchanged public forward already validates context finiteness
            # exactly once for this chunk; its ValueError is classified below as
            # the guard failure for an invalid context (no duplicate device scan).
            try:
                log_v = scale_model(chunk, v_floor=_floor, v_cap=_cap)["log_v"]
            except ValueError as exc:
                raise N4ScaleGuardError(
                    "n4-scale-init-interiority",
                    f"initialized forward is invalid: {exc}",
                    _diagnostics(),
                ) from exc
            if log_v.shape[0] != chunk.shape[0] or not torch.isfinite(log_v).all():
                raise N4ScaleGuardError(
                    "n4-scale-init-interiority",
                    "initialized forward returned malformed/non-finite log-variance",
                    _diagnostics(),
                )
            record_count += int(log_v.shape[0])
            log_v_min = min(log_v_min, float(log_v.min().item()))
            log_v_max = max(log_v_max, float(log_v.max().item()))
            at_floor += int((log_v <= log_floor + tolerance).sum().item())
            at_cap += int((log_v >= log_cap - tolerance).sum().item())
    diagnostics = _diagnostics()
    if record_count != record_total:
        raise N4ScaleGuardError(
            "n4-scale-init-interiority",
            "initialized forward did not cover the in-scope pool",
            diagnostics,
        )
    if diagnostics["frac_at_floor"] != 0.0 or diagnostics["frac_at_cap"] != 0.0:
        raise N4ScaleGuardError(
            "n4-scale-init-interiority",
            "initialized log-variance is not strictly interior",
            diagnostics,
        )
    return diagnostics


def n4_guard_first_step_gradients(
    scale_model: ContextResidualScaleModel,
) -> dict[str, object]:
    """Guard B: read-only inspection of the first normal backward's gradients.

    Called after the sole ``loss.backward()`` at ``global_step == 0`` and before
    the unchanged ``clip_grad_norm_(..., 1.0)``/``optimizer.step()``. Fails
    closed on a missing gradient, any non-finite gradient element, or an
    aggregate all-exactly-zero gradient. Individual zero coordinates and whole
    zero tensors are legitimate and are recorded only as diagnostics. Never
    calls ``backward``/``zero_grad``, never mutates ``.grad``, draws no RNG, and
    touches no optimizer, mode, or checkpoint state.
    """
    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 first-step gradient guard requires the context-only scale model")
    named = list(scale_model.named_parameters())
    if not named:
        raise ValueError("N4 first-step gradient guard requires a non-empty parameter set")
    tensors: list[dict[str, object]] = []
    missing: list[str] = []
    total_nonzero = 0
    total_nonfinite = 0
    for name, parameter in named:
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            tensors.append(
                {"name": name, "numel": int(parameter.numel()), "nonzero": None,
                 "zero": None, "nonfinite": None, "norm": None}
            )
            continue
        nonzero = int(torch.count_nonzero(grad).item())
        nonfinite = int((~torch.isfinite(grad)).sum().item())
        numel = int(grad.numel())
        norm_value = float(grad.detach().double().norm().item())
        tensors.append(
            {
                "name": name,
                "numel": numel,
                "nonzero": nonzero,
                "zero": numel - nonzero,
                "nonfinite": nonfinite,
                "norm": norm_value if math.isfinite(norm_value) else None,
            }
        )
        total_nonzero += nonzero
        total_nonfinite += nonfinite
    diagnostics: dict[str, object] = {
        "total_nonzero": total_nonzero,
        "total_nonfinite": total_nonfinite,
        "missing_grad_parameters": missing,
        "tensors": tensors,
    }
    if missing:
        raise N4ScaleGuardError(
            "n4-scale-first-step-gradients",
            "a parameter received no gradient",
            diagnostics,
        )
    if total_nonfinite:
        raise N4ScaleGuardError(
            "n4-scale-first-step-gradients",
            "nonfinite gradient before the first optimizer step",
            diagnostics,
        )
    if total_nonzero < 1:
        raise N4ScaleGuardError(
            "n4-scale-first-step-gradients",
            "all parameter gradients are exactly zero",
            diagnostics,
        )
    return diagnostics


def n4_patch_residual(
    target_latents: torch.Tensor, predicted_latents: torch.Tensor
) -> torch.Tensor:
    """Compute ``e_t = mean_d((layer_norm(z_target) - layer_norm(z_pred))^2)``.

    Both inputs are detached defensively so cached residuals can never carry
    gradients back into the frozen representation. Returns a detached 1D
    ``[K]`` tensor.
    """
    for name, value in (("target_latents", target_latents), ("predicted_latents", predicted_latents)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1:
            raise ValueError(f"N4 {name} must be a non-empty 2D [K, D] tensor")
    if target_latents.shape != predicted_latents.shape:
        raise ValueError("N4 target and predicted latents must share shape")
    if not torch.isfinite(target_latents).all() or not torch.isfinite(predicted_latents).all():
        raise ValueError("N4 residual latents must be finite")
    width = target_latents.shape[1]
    detached_target = target_latents.detach()
    detached_pred = predicted_latents.detach()
    normalized_target = torch.nn.functional.layer_norm(detached_target, (width,))
    normalized_pred = torch.nn.functional.layer_norm(detached_pred, (width,))
    residuals = (normalized_target - normalized_pred).square().mean(dim=-1).detach()
    if not torch.isfinite(residuals).all() or bool((residuals < 0.0).any()):
        raise ValueError("N4 patch residuals must be finite and non-negative")
    return residuals

def n4_normalized_patch_scores(
    residuals: torch.Tensor, log_v: torch.Tensor
) -> torch.Tensor:
    """Compute ``S_pred_patch_CRN = e_t / (exp(log_v_t) + 1e-6)``.

    Both inputs are 1D ``[K]`` float tensors with finite non-negative
    residuals and finite log-variance. The ``1e-6`` denominator guard is the
    exact N4 v1 S5 constant: it keeps the normalized score finite when the
    predicted scale is small. Never fuses or returns the raw residual.
    """
    for name, value in (("residuals", residuals), ("log_v", log_v)):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] < 1:
            raise ValueError(f"N4 {name} must be a non-empty 1D tensor")
    if residuals.shape != log_v.shape:
        raise ValueError("N4 residuals and log-variance must share length")
    if not torch.isfinite(residuals).all() or not torch.isfinite(log_v).all():
        raise ValueError("N4 normalized-score inputs must be finite")
    if bool((residuals < 0.0).any()):
        raise ValueError("N4 normalized-score residuals must be non-negative")
    scores = residuals / (torch.exp(log_v) + 1e-6)
    if not torch.isfinite(scores).all():
        raise ValueError("N4 normalized patch scores are nonfinite")
    return scores


def n4_residual_bounds(residuals: np.ndarray | Sequence[float]) -> dict[str, float]:
    """Compute Fit-only quantile bounds with the NumPy linear convention.

    ``v_floor = max(quantile_0.05, 1e-6)``,
    ``v_cap = max(quantile_0.95, 10 * v_floor)``. The held-out history
    contributes no residual: callers pass only in-fold training residuals.
    """
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.shape[0] < 1:
        raise ValueError("N4 bounds require a non-empty residual pool")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("N4 bound residuals must be finite and non-negative")
    q05 = float(np.quantile(values, 0.05, method="linear"))
    q95 = float(np.quantile(values, 0.95, method="linear"))
    v_floor = max(q05, N4_VARIANCE_FLOOR_MIN)
    v_cap = max(q95, 10.0 * v_floor)
    for name, bound in (("q05", q05), ("q95", q95), ("v_floor", v_floor), ("v_cap", v_cap)):
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError(f"N4 {name} bound is invalid")
    return {"q05": q05, "q95": q95, "v_floor": v_floor, "v_cap": v_cap}


def n4_scale_nll(residuals: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
    """Compute ``mean_t[0.5 * (e_t / exp(log_v_t) + log_v_t)]``."""
    for name, value in (("residuals", residuals), ("log_v", log_v)):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] < 1:
            raise ValueError(f"N4 {name} must be a non-empty 1D tensor")
    if residuals.shape != log_v.shape:
        raise ValueError("N4 residuals and log-variance must share length")
    if not torch.isfinite(residuals).all() or not torch.isfinite(log_v).all():
        raise ValueError("N4 NLL inputs must be finite")
    if bool((residuals < 0.0).any()):
        raise ValueError("N4 NLL residuals must be non-negative")
    loss = (0.5 * (residuals / torch.exp(log_v) + log_v)).mean()
    if not torch.isfinite(loss):
        raise ValueError("N4 scale NLL is nonfinite")
    return loss


def n4_saturation(
    log_v: torch.Tensor, *, v_floor: float, v_cap: float
) -> dict[str, float]:
    """Report the fraction of predictions clamped at each frozen bound."""
    if not isinstance(log_v, torch.Tensor) or log_v.ndim != 1 or log_v.shape[0] < 1:
        raise ValueError("N4 saturation requires a non-empty 1D log-variance tensor")
    if not torch.isfinite(log_v).all():
        raise ValueError("N4 saturation inputs must be finite")
    for name, bound in (("v_floor", v_floor), ("v_cap", v_cap)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"N4 {name} bound must be a number")
        if not math.isfinite(float(bound)) or float(bound) <= 0.0:
            raise ValueError(f"N4 {name} bound must be finite and positive")
    log_floor = math.log(float(v_floor))
    log_cap = math.log(float(v_cap))
    if log_cap < log_floor:
        raise ValueError("N4 v_cap must not lie below v_floor")
    count = log_v.shape[0]
    frac_floor = float((log_v <= log_floor + N4_SATURATION_LOG_TOL).sum().item()) / count
    frac_cap = float((log_v >= log_cap - N4_SATURATION_LOG_TOL).sum().item()) / count
    return {"frac_at_floor": frac_floor, "frac_at_cap": frac_cap}


def n4_cap_saturation_pass(frac_at_cap: float) -> bool:
    """Apply the structural gate: pass when at most 20% saturate ``v_cap``."""
    if isinstance(frac_at_cap, bool) or not isinstance(frac_at_cap, (int, float)):
        raise ValueError("N4 saturation fraction must be a number")
    frac = float(frac_at_cap)
    if not math.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError("N4 saturation fraction must lie in [0, 1]")
    return frac <= N4_SATURATION_CAP_GATE


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    count = values.shape[0]
    start = 0
    while start < count:
        stop = start
        while stop + 1 < count and sorted_vals[stop + 1] == sorted_vals[start]:
            stop += 1
        ranked[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranked


def n4_spearman(scales: np.ndarray | Sequence[float], residuals: np.ndarray | Sequence[float]) -> float:
    """Rank correlation between predicted scale and residual in [-1, 1].

    Returns ``0.0`` when no monotonic association is measurable (fewer than two
    records or either side constant). Never returns NaN.
    """
    x = np.asarray(scales, dtype=np.float64).reshape(-1)
    y = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("N4 correlation inputs must share length")
    if x.shape[0] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        if x.shape[0] >= 2 and (not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("N4 correlation inputs must be finite")
        return 0.0
    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    denominator = float(np.sqrt((centered_x**2).sum() * (centered_y**2).sum()))
    if denominator == 0.0:
        return 0.0
    rho = float((centered_x * centered_y).sum() / denominator)
    if not math.isfinite(rho):
        return 0.0
    return max(-1.0, min(1.0, rho))


def n4_canonical_fold_splits(binding: Task6Binding) -> list[tuple[list[str], str]]:
    """Return the three canonical ``(training_ids, heldout_id)`` fold splits.

    The held-out history rotates in canonical Fit order, matching the checkpoint
    validator's fold indexing. Training IDs are sorted ascending.
    """
    fit_ids = [item.actual_history_id for item in binding.role_histories("FIT")]
    if len(fit_ids) != 3 or len(set(fit_ids)) != 3 or any(not h for h in fit_ids):
        raise ValueError("N4 cross-fitting requires exactly three bound Fit histories")
    splits: list[tuple[list[str], str]] = []
    for heldout in fit_ids:
        training = sorted(h for h in fit_ids if h != heldout)
        splits.append((training, heldout))
    return splits


def n4_record_id(history_id: str, file_id: str, patch_index: int) -> str:
    """Build the canonical scale-training record identity."""
    if not history_id or not file_id:
        raise ValueError("N4 record identity requires history and file IDs")
    if isinstance(patch_index, bool) or not isinstance(patch_index, int) or patch_index < 0:
        raise ValueError("N4 record patch index must be a non-negative integer")
    return f"{history_id}|{file_id}|{patch_index:05d}"


def build_n4_scale_records(
    *,
    model: nn.Module,
    samples: Sequence[object],
    history_id: str,
    patchifier: object,
    masking_seed: int,
) -> list[N4ScaleRecord]:
    """Build one canonical scale record per eligible masked valid patch.

    Runs the frozen B0 encoder under ``no_grad`` on one collated batch, then
    emits per-patch ``(detached context, scalar residual)`` pairs joined with
    canonical ``(history_id, file_id, patch_index)`` identity. This helper
    consumes only caller-qualified anomaly-free healthy rows and reads no
    labels, waveforms beyond the encoder-visible collation, or target latents
    outside the detached residual computation. It performs zero representation
    updates: the model is never set to train mode and no gradient is enabled.
    """
    from representation.data import collate_variable_files

    if not history_id:
        raise ValueError("N4 records require a bound history ID")
    materialized = list(samples)
    if not materialized:
        return []
    model.eval()
    records: list[N4ScaleRecord] = []
    with torch.no_grad():
        batch = collate_variable_files(
            materialized,  # type: ignore[arg-type]
            patchifier,  # type: ignore[arg-type]
            masking_config=getattr(model, "config", None),
            masking_seed=masking_seed,
        )
        clean = {
            key: value for key, value in batch.items() if key != "file_samples"
        }
        output = model(clean)  # type: ignore[operator]
        context = output["context_latents"].detach()
        target = output["target_latents"].detach()
        predicted = output["predicted_latents"].detach()
        requested = output.get("prediction_mask")
        valid = output.get("patch_valid_mask")
        if (
            not isinstance(requested, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or requested.dtype is not torch.bool
            or valid.dtype is not torch.bool
        ):
            raise ValueError("N4 encoder output masks are invalid")
        eligible = requested & valid
        file_ids = batch["file_ids"]
        if len(file_ids) != context.shape[0]:
            raise ValueError("N4 batch file IDs differ from the encoded batch")
        width = context.shape[-1]
        normalized_target = torch.nn.functional.layer_norm(target, (width,))
        normalized_pred = torch.nn.functional.layer_norm(predicted, (width,))
        residuals = (
            (normalized_target - normalized_pred).square().mean(dim=-1).detach()
        )
        for row, file_id in enumerate(file_ids):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("N4 records require canonical file IDs")
            for patch_index in torch.nonzero(eligible[row], as_tuple=False).flatten().tolist():
                patch = int(patch_index)
                residual = float(residuals[row, patch].item())
                if not math.isfinite(residual) or residual < 0.0:
                    raise ValueError("N4 cached residual must be finite and non-negative")
                records.append(
                    N4ScaleRecord(
                        history_id=history_id,
                        file_id=file_id,
                        patch_index=patch,
                        context=context[row, patch].detach().cpu().float(),
                        residual=residual,
                    )
                )
    model.eval()
    return sorted(
        records, key=lambda r: (r.history_id, r.file_id, r.patch_index)
    )

def fit_n4_scale_scope(
    *,
    records: Sequence[N4ScaleRecord],
    training_history_ids: Sequence[str],
    scope: str,
    arm: Sprint18ArmConfig,
    binding: Task6Binding,
    scale_model: ContextResidualScaleModel,
    v_floor: float,
    v_cap: float,
    heldout_records: Sequence[N4ScaleRecord] = (),
    device: torch.device | str = "cpu",
    optimizer: torch.optim.AdamW | None = None,
) -> dict[str, object]:
    """Fit one N4 scope for exactly 50 epochs under the frozen scale contract.

    Consumes every eligible in-scope record exactly once per epoch in contiguous
    batches of at most 512, ordered by the frozen
    ``sprint18-n4-scale-order-v2`` CPU-generator permutation for
    ``(model_seed, scope, sorted in-fold history IDs, epoch)`` with
    ``drop_last=false``. Only ``scale_model`` parameters enter the AdamW
    optimizer (``lr=1e-3``, ``weight_decay=1e-4``, global L2 clip ``1.0``); the
    representation never enters this optimizer. Held-out records are scored for
    diagnostics only and never trained.
    """
    from representation.sprint18_harness import deterministic_sampler_order

    if scope != "final" and not scope.startswith("fold:"):
        raise ValueError("N4 fit scope must be 'final' or 'fold:<heldout_history_id>'")
    training_ids = [str(h) for h in training_history_ids]
    if not training_ids or len(set(training_ids)) != len(training_ids) or any(not h for h in training_ids):
        raise ValueError("N4 training history IDs must be non-empty and unique")
    if sorted(training_ids) != training_ids:
        raise ValueError("N4 training history IDs must arrive in sorted canonical order")
    heldout_id = scope[5:] if scope.startswith("fold:") else None
    if (heldout_id is None) != (scope == "final"):
        raise ValueError("N4 fold scope requires a held-out history; final scope forbids one")
    if scope == "final" and heldout_records:
        raise ValueError("N4 final fit accepts no held-out records")
    materialized = list(records)
    if not materialized:
        raise ValueError("N4 fit scope requires a non-empty record pool")
    seen: set[str] = set()
    for record in materialized:
        if not isinstance(record, N4ScaleRecord):
            raise ValueError("N4 fit records must be N4ScaleRecord entries")
        if record.history_id not in set(training_ids):
            raise ValueError("N4 training pool contains a record outside its in-fold histories")
        if not record.file_id or isinstance(record.patch_index, bool) or record.patch_index < 0:
            raise ValueError("N4 record identity is invalid")
        if not isinstance(record.context, torch.Tensor) or record.context.ndim != 1:
            raise ValueError("N4 record context must be a detached 1D latent row")
        # Structural shape/type checks only: context finiteness is validated
        # exactly once per bounded chunk by the forward inside Guard A, which
        # classifies an invalid context as N4ScaleGuardError before the
        # optimizer exists.
        if record.context.shape[0] != scale_model.d_model:
            raise ValueError("N4 record context must match the scale width")
        if (
            isinstance(record.residual, bool)
            or not isinstance(record.residual, (int, float))
            or not math.isfinite(float(record.residual))
            or float(record.residual) < 0.0
        ):
            raise ValueError("N4 record residual must be a finite non-negative number")
        key = n4_record_id(record.history_id, record.file_id, record.patch_index)
        if key in seen:
            raise ValueError("N4 training pool contains a duplicate canonical record")
        seen.add(key)
    for record in heldout_records:
        if not isinstance(record, N4ScaleRecord) or record.history_id != heldout_id:
            raise ValueError("N4 held-out pool must contain only the held-out history")
    ordered = sorted(materialized, key=lambda r: (r.history_id, r.file_id, r.patch_index))
    canonical_ids = [n4_record_id(r.history_id, r.file_id, r.patch_index) for r in ordered]
    if sorted(canonical_ids) != canonical_ids:
        raise ValueError("N4 canonical record order is incoherent")
    for name, bound in (("v_floor", v_floor), ("v_cap", v_cap)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"N4 {name} bound must be a number")
        if not math.isfinite(float(bound)) or float(bound) <= 0.0:
            raise ValueError(f"N4 {name} bound must be finite and positive")
    if float(v_cap) < float(v_floor):
        raise ValueError("N4 v_cap must not lie below v_floor")
    scale_parameters = sum(p.numel() for p in scale_model.parameters())
    if scale_parameters <= 0 or scale_parameters > N4_SCALE_PARAM_LIMIT:
        raise ValueError("N4 scale capacity is outside the frozen parameter envelope")
    target_device = torch.device(device)
    scale_model.to(target_device)
    init_anchor = n4_anchor_scale_model(scale_model, [float(r.residual) for r in ordered])
    contexts = [r.context.detach().to(device=target_device, dtype=torch.float32) for r in ordered]
    targets = torch.tensor([float(r.residual) for r in ordered], dtype=torch.float32, device=target_device)
    scale_model.train()
    # Guard A runs before the optimizer object exists, so optimizer state can
    # never be affected; it is forward-only over the whole in-scope pool.
    init_interiority = n4_guard_initialized_interiority(
        scale_model, contexts, v_floor=float(v_floor), v_cap=float(v_cap)
    )
    id_to_index = {key: index for index, key in enumerate(canonical_ids)}
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            scale_model.parameters(), lr=N4_SCALE_LR, weight_decay=N4_SCALE_WEIGHT_DECAY
        )
    else:
        # Optional caller-owned optimizer: the coherent N4 checkpoint must
        # serialize the optimizer state that actually produced the scale head.
        # Accept only the frozen optimizer class bound to exactly these
        # parameters under the frozen lr/weight-decay contract; never coerce.
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("N4 external optimizer must be the frozen AdamW")
        bound_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        expected_parameters = list(scale_model.parameters())
        if len(bound_parameters) != len(expected_parameters) or any(
            bound is not expected
            for bound, expected in zip(bound_parameters, expected_parameters)
        ):
            raise ValueError(
                "N4 external optimizer must be bound to exactly the scale-model parameters"
            )
        for group in optimizer.param_groups:
            if (
                group.get("lr") != N4_SCALE_LR
                or group.get("weight_decay") != N4_SCALE_WEIGHT_DECAY
            ):
                raise ValueError("N4 external optimizer must use the frozen lr/weight decay")
    record_count = len(ordered)
    per_epoch_steps = math.ceil(record_count / N4_SCALE_BATCH_SIZE)
    per_epoch_nll: list[float] = []
    epoch_order_seeds: list[int] = []
    global_step = 0
    first_step_gradients: dict[str, object] | None = None
    for epoch in range(1, N4_SCALE_EPOCHS + 1):
        seed, order = deterministic_sampler_order(
            binding,
            arm,
            workload="n4_scale",
            scope="experiment",
            role_history_ids=training_ids,
            epoch=epoch,
            eligible_source_ids=canonical_ids,
        )
        if tuple(order) == tuple(canonical_ids) and record_count > 1:
            raise ValueError("N4 epoch permutation is unexpectedly unshuffled")
        epoch_order_seeds.append(seed)
        indices = [id_to_index[key] for key in order]
        if sorted(indices) != list(range(record_count)):
            raise ValueError("N4 epoch permutation must consume every record exactly once")
        epoch_losses: list[float] = []
        for start in range(0, record_count, N4_SCALE_BATCH_SIZE):
            chunk = indices[start : start + N4_SCALE_BATCH_SIZE]
            batch_context = torch.stack([contexts[i] for i in chunk], dim=0)
            batch_residual = targets[chunk]
            optimizer.zero_grad(set_to_none=True)
            predicted = scale_model(batch_context, v_floor=float(v_floor), v_cap=float(v_cap))
            loss = n4_scale_nll(batch_residual, predicted["log_v"])
            loss.backward()
            if global_step == 0:
                # Guard B inspects the sole normal backward result before the
                # unchanged clip/step; it never mutates gradients or state.
                first_step_gradients = n4_guard_first_step_gradients(scale_model)
            clip_grad_norm_(scale_model.parameters(), N4_SCALE_CLIP_NORM)
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu().item()))
        if len(epoch_losses) != per_epoch_steps:
            raise ValueError("N4 epoch step count differs from the frozen batch schedule")
        per_epoch_nll.append(sum(epoch_losses) / len(epoch_losses))
    expected_steps = N4_SCALE_EPOCHS * per_epoch_steps
    if global_step != expected_steps:
        raise ValueError("N4 scale optimizer-step total differs from the frozen scope count")
    scale_model.eval()
    with torch.no_grad():
        full_context = torch.stack(contexts, dim=0)
        full_out = scale_model(full_context, v_floor=float(v_floor), v_cap=float(v_cap))
        final_train_nll = float(n4_scale_nll(targets, full_out["log_v"]).cpu().item())
        saturation_train = n4_saturation(full_out["log_v"].cpu(), v_floor=float(v_floor), v_cap=float(v_cap))
    heldout_nll: float | None = None
    heldout_spearman: float | None = None
    saturation_heldout: dict[str, float] | None = None
    if heldout_records:
        heldout_sorted = sorted(heldout_records, key=lambda r: (r.history_id, r.file_id, r.patch_index))
        with torch.no_grad():
            heldout_context = torch.stack(
                [r.context.detach().to(device=target_device, dtype=torch.float32) for r in heldout_sorted],
                dim=0,
            )
            heldout_target = torch.tensor(
                [float(r.residual) for r in heldout_sorted], dtype=torch.float32, device=target_device
            )
            heldout_out = scale_model(heldout_context, v_floor=float(v_floor), v_cap=float(v_cap))
            heldout_nll = float(n4_scale_nll(heldout_target, heldout_out["log_v"]).cpu().item())
            saturation_heldout = n4_saturation(
                heldout_out["log_v"].cpu(), v_floor=float(v_floor), v_cap=float(v_cap)
            )
            heldout_spearman = n4_spearman(
                heldout_out["v"].detach().cpu().numpy(), heldout_target.detach().cpu().numpy()
            )
    for value in [final_train_nll, *per_epoch_nll] + (
        [heldout_nll, heldout_spearman] if heldout_nll is not None and heldout_spearman is not None else []
    ):
        if not math.isfinite(float(value)):
            raise ValueError("N4 scope fit produced a nonfinite diagnostic")
    if first_step_gradients is None:
        raise ValueError("N4 first-step gradient guard did not run")
    result: dict[str, Any] = {
        "scope": scope,
        "training_history_ids": list(training_ids),
        "heldout_history_id": heldout_id,
        "scale_parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "init_anchor": dict(init_anchor),
        "init_interiority": dict(init_interiority),
        "first_step_gradients": dict(first_step_gradients),
        "record_count": record_count,
        "heldout_count": len(heldout_records),
        "bounds": {"v_floor": float(v_floor), "v_cap": float(v_cap)},
        "optimizer_steps": global_step,
        "per_epoch_steps": per_epoch_steps,
        "per_epoch_nll": list(per_epoch_nll),
        "final_train_nll": final_train_nll,
        "heldout_nll": heldout_nll,
        "heldout_spearman": heldout_spearman,
        "saturation_train": dict(saturation_train),
        "saturation_heldout": dict(saturation_heldout) if saturation_heldout is not None else None,
        "cap_gate_pass_train": n4_cap_saturation_pass(saturation_train["frac_at_cap"]),
        "epoch_order_seeds": list(epoch_order_seeds),
        "order_seed_rule": (
            "sprint18-n4-scale-order-v2|<model_seed>|<scope>|"
            "<comma-joined sorted in-fold Fit history IDs>|<epoch>; one CPU "
            "torch.randperm per epoch; contiguous batches of at most 512 records; "
            "drop_last=false"
        ),
        "scale_parameters": scale_parameters,
    }
    return result

def _n4_validate_initialization(
    initialization: Mapping[str, object],
    *,
    bounds: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Validate the recorded v3 initialization/guard provenance, fail closed.

    Besides key sets and finite fields, every recorded diagnostic must follow
    from the values it claims to summarize: ``log_v_ref`` from ``v_ref``, the
    recorded log bounds from the checkpoint bounds when supplied, the
    interiority margins from the recorded extrema and log bounds, the frozen
    saturation tolerance, and the per-tensor first-step gradient counts from
    the frozen parameter roster and the aggregate totals. A checkpoint can
    therefore not claim a successful guard on contradictory evidence.
    """
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "init_anchor", "init_interiority", "first_step_gradients"
    }:
        raise ValueError("N4 scale initialization diagnostics are incomplete")
    anchor = initialization["init_anchor"]
    interiority = initialization["init_interiority"]
    gradients = initialization["first_step_gradients"]
    if not isinstance(anchor, Mapping) or set(anchor) != {"v_ref", "log_v_ref", "record_count"}:
        raise ValueError("N4 anchor diagnostics are incomplete")
    v_ref = anchor["v_ref"]
    log_v_ref = anchor["log_v_ref"]
    anchor_count = anchor["record_count"]
    if (
        isinstance(v_ref, bool) or not isinstance(v_ref, (int, float))
        or not math.isfinite(float(v_ref)) or float(v_ref) <= 0.0
    ):
        raise ValueError("N4 anchor v_ref is invalid")
    if (
        isinstance(log_v_ref, bool) or not isinstance(log_v_ref, (int, float))
        or not math.isfinite(float(log_v_ref))
    ):
        raise ValueError("N4 anchor log(v_ref) is invalid")
    if float(log_v_ref) != math.log(float(v_ref)):
        raise ValueError("N4 anchor log(v_ref) does not follow from v_ref")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count < 1:
        raise ValueError("N4 anchor record count is invalid")
    if not isinstance(interiority, Mapping) or set(interiority) != {
        "record_count", "log_v_min", "log_v_max", "log_floor", "log_cap",
        "floor_margin", "cap_margin", "frac_at_floor", "frac_at_cap", "tolerance",
    }:
        raise ValueError("N4 initialized-interiority diagnostics are incomplete")
    if interiority["record_count"] != anchor_count:
        raise ValueError("N4 initialized-interiority scope differs from the anchor scope")
    for name in ("log_v_min", "log_v_max", "log_floor", "log_cap", "floor_margin", "cap_margin", "tolerance"):
        value = interiority[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"N4 initialized-interiority {name} is invalid")
    for name in ("frac_at_floor", "frac_at_cap"):
        value = interiority[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 0.0:
            raise ValueError(f"N4 initialized-interiority {name} must be exactly zero")
    log_v_min = float(interiority["log_v_min"])
    log_v_max = float(interiority["log_v_max"])
    log_floor = float(interiority["log_floor"])
    log_cap = float(interiority["log_cap"])
    tolerance = float(interiority["tolerance"])
    if tolerance != N4_SATURATION_LOG_TOL:
        raise ValueError("N4 initialized-interiority tolerance differs from the frozen tolerance")
    if log_cap <= log_floor:
        raise ValueError("N4 initialized-interiority log bounds are not strictly ordered")
    if log_v_min > log_v_max:
        raise ValueError("N4 initialized-interiority extrema are inverted")
    if float(interiority["floor_margin"]) != log_v_min - log_floor:
        raise ValueError("N4 initialized-interiority floor margin does not follow from the extrema")
    if float(interiority["cap_margin"]) != log_cap - log_v_max:
        raise ValueError("N4 initialized-interiority cap margin does not follow from the extrema")
    if log_v_min <= log_floor + tolerance or log_v_max >= log_cap - tolerance:
        raise ValueError("N4 initialized-interiority extrema are not strictly interior")
    if bounds is not None and (
        log_floor != math.log(float(bounds[0])) or log_cap != math.log(float(bounds[1]))
    ):
        raise ValueError("N4 initialized-interiority log bounds differ from the checkpoint bounds")
    if not isinstance(gradients, Mapping) or set(gradients) != {
        "total_nonzero", "total_nonfinite", "missing_grad_parameters", "tensors"
    }:
        raise ValueError("N4 first-step gradient diagnostics are incomplete")
    total_nonzero = gradients["total_nonzero"]
    total_nonfinite = gradients["total_nonfinite"]
    missing = gradients["missing_grad_parameters"]
    tensors = gradients["tensors"]
    if isinstance(total_nonzero, bool) or not isinstance(total_nonzero, int) or total_nonzero < 1:
        raise ValueError("N4 first-step gradients must contain a nonzero coordinate")
    if isinstance(total_nonfinite, bool) or not isinstance(total_nonfinite, int) or total_nonfinite != 0:
        raise ValueError("N4 first-step gradients must be finite")
    if not isinstance(missing, list) or missing:
        raise ValueError("N4 first-step gradients must cover every parameter")
    if not isinstance(tensors, list) or len(tensors) != len(N4_SCALE_PARAMETER_ROSTER):
        raise ValueError("N4 first-step per-tensor diagnostics do not cover the frozen parameter roster")
    roster_nonzero = 0
    roster_nonfinite = 0
    for entry, (expected_name, expected_numel) in zip(tensors, N4_SCALE_PARAMETER_ROSTER):
        if not isinstance(entry, Mapping) or set(entry) != {
            "name", "numel", "nonzero", "zero", "nonfinite", "norm"
        }:
            raise ValueError("N4 first-step per-tensor diagnostics are incomplete")
        if entry["name"] != expected_name:
            raise ValueError("N4 first-step per-tensor names differ from the frozen parameter roster")
        numel = entry["numel"]
        nonzero = entry["nonzero"]
        zero = entry["zero"]
        nonfinite = entry["nonfinite"]
        if (
            isinstance(numel, bool) or not isinstance(numel, int) or numel != expected_numel
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (nonzero, zero, nonfinite)
            )
            or nonzero < 0 or zero < 0 or nonfinite < 0
            or nonzero + zero != numel or nonfinite > numel
        ):
            raise ValueError("N4 first-step per-tensor counts are inconsistent with the frozen roster")
        if nonfinite:
            raise ValueError("N4 first-step per-tensor gradients must be finite")
        norm = entry["norm"]
        if (
            isinstance(norm, bool) or not isinstance(norm, (int, float))
            or not math.isfinite(float(norm)) or float(norm) < 0.0
        ):
            raise ValueError("N4 first-step per-tensor gradient norm is invalid")
        roster_nonzero += nonzero
        roster_nonfinite += nonfinite
    if roster_nonzero != total_nonzero or roster_nonfinite != total_nonfinite:
        raise ValueError("N4 first-step gradient totals do not reconcile with the per-tensor diagnostics")
    return {
        "init_anchor": dict(anchor),
        "init_interiority": dict(interiority),
        "first_step_gradients": {
            "total_nonzero": total_nonzero,
            "total_nonfinite": total_nonfinite,
            "missing_grad_parameters": list(missing),
            "tensors": [dict(entry) for entry in tensors],
        },
    }


_N4_SCALE_CHECKPOINT_KEYS = {
    "schema", "parameterization", "guard_contract", "fit_scope",
    "training_history_ids", "v_floor", "v_cap", "initialization",
    "state_dict_sha256", "state_dict",
}


def _n4_scale_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    from representation.sprint18_checkpoint import state_sha256

    return state_sha256(state)


def _n4_scale_cpu_state(
    scale_model: ContextResidualScaleModel,
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in scale_model.state_dict().items()}


def save_n4_scale_checkpoint(
    path: str | Path,
    scale_model: ContextResidualScaleModel,
    *,
    fit_scope: str,
    training_history_ids: Sequence[str],
    v_floor: float,
    v_cap: float,
    initialization: Mapping[str, object],
) -> dict[str, object]:
    """Write one ``s18-n4-scale-checkpoint-v2`` envelope for a fitted scale head.

    The bare ``state_dict`` form used by the rejected v1 (dead-zone) runs is
    never written again. The envelope carries the exact parameterization and
    guard identities, the Fit scope and history roster, the frozen bounds, the
    recorded initialization/guard diagnostics, and a canonical digest of the
    scale state it serializes.
    """
    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 scale checkpoint save requires the context-only scale model")
    if fit_scope != "final" and not fit_scope.startswith("fold:"):
        raise ValueError("N4 scale checkpoint scope must be 'final' or 'fold:<heldout_history_id>'")
    histories = [str(h) for h in training_history_ids]
    if not histories or len(set(histories)) != len(histories) or any(not h for h in histories):
        raise ValueError("N4 scale checkpoint history IDs must be non-empty and unique")
    if sorted(histories) != histories:
        raise ValueError("N4 scale checkpoint history IDs must arrive in sorted canonical order")
    floor, cap, _log_floor, _log_cap = _n4_scale_bounds(v_floor, v_cap)
    provenance = _n4_validate_initialization(initialization, bounds=(floor, cap))
    state = _n4_scale_cpu_state(scale_model)
    state_digest = _n4_scale_state_sha256(state)
    payload: dict[str, object] = {
        "schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "fit_scope": fit_scope,
        "training_history_ids": histories,
        "v_floor": floor,
        "v_cap": cap,
        "initialization": provenance,
        "state_dict_sha256": state_digest,
        "state_dict": state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return {
        "schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "fit_scope": fit_scope,
        "training_history_ids": histories,
        "v_floor": floor,
        "v_cap": cap,
        "state_dict_sha256": state_digest,
        "initialization": provenance,
    }


def load_n4_scale_checkpoint(
    path: str | Path,
    scale_model: ContextResidualScaleModel,
    *,
    expected_fit_scope: str | None = None,
    expected_training_history_ids: Sequence[str] | None = None,
    expected_bounds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Strictly restore one v3 scale envelope; refuse legacy/unknown checkpoints.

    Accepts only ``s18-n4-scale-checkpoint-v2`` with
    ``parameterization == "fit-mean-anchored-bias-v1"`` and
    ``guard_contract == "n4-v3-guards-v1"``, recomputes the state digest, and
    fails closed on a bare legacy ``state_dict`` (the rejected A03 dead-zone
    form), an unknown parameterization, a stale scope/history/bound binding, or
    corrupted/unsafe bytes. This is an N4 scale-checkpoint refusal, not the
    generic Sprint18 coherent-checkpoint path.
    """
    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 scale checkpoint restore requires the context-only scale model")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"N4 scale checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - normalize unsafe/legacy bytes to one refusal
        raise ValueError(
            "N4 scale checkpoint is not a safe s18-n4-scale-checkpoint-v2 envelope; "
            "legacy or foreign bytes are refused"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("N4 scale checkpoint payload is not an envelope mapping")
    if payload.get("schema") != N4_SCALE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy or incompatible N4 scale checkpoint refused: only "
            f"{N4_SCALE_CHECKPOINT_SCHEMA} is accepted"
        )
    if set(payload) != _N4_SCALE_CHECKPOINT_KEYS:
        raise ValueError("N4 scale checkpoint envelope is incomplete or has unknown keys")
    if payload["parameterization"] != N4_SCALE_PARAMETERIZATION:
        raise ValueError("N4 scale checkpoint parameterization is unknown")
    if payload["guard_contract"] != N4_GUARD_CONTRACT:
        raise ValueError("N4 scale checkpoint guard contract is unknown")
    fit_scope = payload["fit_scope"]
    if not isinstance(fit_scope, str) or (
        fit_scope != "final" and not fit_scope.startswith("fold:")
    ):
        raise ValueError("N4 scale checkpoint fit scope is invalid")
    histories = payload["training_history_ids"]
    if (
        not isinstance(histories, list)
        or not histories
        or any(not isinstance(h, str) or not h for h in histories)
        or sorted(histories) != histories
        or len(set(histories)) != len(histories)
    ):
        raise ValueError("N4 scale checkpoint history roster is invalid")
    floor, cap, _log_floor, _log_cap = _n4_scale_bounds(payload["v_floor"], payload["v_cap"])
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("N4 scale checkpoint state dict is missing")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("N4 scale checkpoint state dict contains non-tensor values")
    observed_state_sha = _n4_scale_state_sha256(state)
    if payload["state_dict_sha256"] != observed_state_sha:
        raise ValueError("N4 scale checkpoint state digest mismatch")
    provenance = _n4_validate_initialization(payload["initialization"], bounds=(floor, cap))
    if expected_fit_scope is not None and fit_scope != expected_fit_scope:
        raise ValueError("N4 scale checkpoint fit scope differs from the requested scope")
    if expected_training_history_ids is not None:
        expected = [str(h) for h in expected_training_history_ids]
        if histories != expected:
            raise ValueError("N4 scale checkpoint histories differ from the requested roster")
    if expected_bounds is not None:
        if (
            not isinstance(expected_bounds, Mapping)
            or set(expected_bounds) != {"v_floor", "v_cap"}
            or float(expected_bounds["v_floor"]) != floor
            or float(expected_bounds["v_cap"]) != cap
        ):
            raise ValueError("N4 scale checkpoint bounds differ from the requested frozen bounds")
    scale_model.load_state_dict({name: value for name, value in state.items()}, strict=True)
    return {
        "schema": N4_SCALE_CHECKPOINT_SCHEMA,
        "parameterization": N4_SCALE_PARAMETERIZATION,
        "guard_contract": N4_GUARD_CONTRACT,
        "fit_scope": fit_scope,
        "training_history_ids": list(histories),
        "v_floor": floor,
        "v_cap": cap,
        "state_dict_sha256": observed_state_sha,
        "initialization": provenance,
    }


def validate_n4_capacity(
    fold_results: Sequence[Mapping[str, object]],
    *,
    record_counts: Sequence[int],
    per_seed_updates: int,
) -> dict[str, object]:
    """Apply the T13 capacity verdict over the three diagnostic folds.

    Requires exactly three successful fold fits with finite held-out NLL, the
    fixed 50-epoch/512-record schedule with exact step accounting, per-fold
    quantile bounds, saturation fields inside ``[0, 1]``, and the ``v_cap`` gate
    on every fold. Zero representation updates is proven by the caller (frozen
    tensors/optimizer state unchanged); this verdict records the structural
    reason (scale-only optimizer over scale parameters). Any failure returns a
    ``FAILED`` verdict with reasons; passage returns ``PASS``. Never tunes
    bounds, capacity, epochs, or folds from outcomes.
    """
    reasons: list[str] = []
    folds = list(fold_results)
    counts = list(record_counts)
    if len(folds) != 3:
        reasons.append("expected exactly three diagnostic folds")
    if len(counts) != 3:
        reasons.append("expected exactly three fold record counts")
    if isinstance(per_seed_updates, bool) or not isinstance(per_seed_updates, int) or per_seed_updates <= 0:
        reasons.append("per-seed scale-update total is invalid")
    details: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        entry: dict[str, object] = {"fold": index}
        if not isinstance(fold, Mapping):
            reasons.append(f"fold {index} result is not a mapping")
            details.append(entry)
            continue
        for key in (
            "scope", "record_count", "optimizer_steps", "per_epoch_steps",
            "per_epoch_nll", "final_train_nll", "heldout_nll",
            "heldout_spearman", "saturation_train", "saturation_heldout",
            "cap_gate_pass_train", "bounds", "scale_parameters",
            "scale_parameterization", "guard_contract", "init_anchor",
            "init_interiority", "first_step_gradients",
        ):
            if key not in fold:
                reasons.append(f"fold {index} is missing {key}")
        if fold.get("scale_parameterization") != N4_SCALE_PARAMETERIZATION:
            reasons.append(f"fold {index} scale parameterization is not the frozen v3 identity")
        if fold.get("guard_contract") != N4_GUARD_CONTRACT:
            reasons.append(f"fold {index} guard contract is not the frozen v3 identity")
        try:
            provenance = _n4_validate_initialization(
                {
                    "init_anchor": fold.get("init_anchor"),
                    "init_interiority": fold.get("init_interiority"),
                    "first_step_gradients": fold.get("first_step_gradients"),
                }
            )
        except ValueError as exc:
            reasons.append(f"fold {index} initialization/guard provenance is invalid: {exc}")
        else:
            entry["init_frac_at_floor"] = provenance["init_interiority"]["frac_at_floor"]
            entry["init_frac_at_cap"] = provenance["init_interiority"]["frac_at_cap"]
            entry["first_step_total_nonzero"] = provenance["first_step_gradients"]["total_nonzero"]
        n_scope = fold.get("record_count")
        steps = fold.get("optimizer_steps")
        per_epoch = fold.get("per_epoch_steps")
        if (
            isinstance(n_scope, int) and not isinstance(n_scope, bool) and n_scope > 0
            and isinstance(per_epoch, int) and not isinstance(per_epoch, bool) and per_epoch > 0
            and isinstance(steps, int) and not isinstance(steps, bool)
        ):
            if per_epoch != math.ceil(n_scope / N4_SCALE_BATCH_SIZE):
                reasons.append(f"fold {index} per-epoch steps differ from ceil(N/512)")
            if steps != N4_SCALE_EPOCHS * per_epoch:
                reasons.append(f"fold {index} optimizer steps differ from 50*ceil(N/512)")
        else:
            reasons.append(f"fold {index} has invalid step accounting fields")
        for key in ("final_train_nll", "heldout_nll"):
            value = fold.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                reasons.append(f"fold {index} {key} is nonfinite")
        per_epoch_nll = fold.get("per_epoch_nll")
        if (
            not isinstance(per_epoch_nll, (list, tuple))
            or len(per_epoch_nll) != N4_SCALE_EPOCHS
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in per_epoch_nll)
        ):
            reasons.append(f"fold {index} per-epoch NLL trace is incomplete/nonfinite")
        for key in ("saturation_train", "saturation_heldout"):
            sat = fold.get(key)
            if not isinstance(sat, Mapping):
                reasons.append(f"fold {index} {key} is missing")
                continue
            for side in ("frac_at_floor", "frac_at_cap"):
                frac = sat.get(side)
                if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not math.isfinite(float(frac)) or not 0.0 <= float(frac) <= 1.0:
                    reasons.append(f"fold {index} {key}.{side} is outside [0, 1]")
        heldout_sat = fold.get("saturation_heldout")
        if isinstance(heldout_sat, Mapping):
            frac = heldout_sat.get("frac_at_cap")
            if isinstance(frac, (int, float)) and not isinstance(frac, bool) and math.isfinite(float(frac)):
                entry["heldout_cap_gate_pass"] = bool(float(frac) <= N4_SATURATION_CAP_GATE)
                if float(frac) > N4_SATURATION_CAP_GATE:
                    reasons.append(f"fold {index} held-out v_cap saturation exceeds 20%")
        bounds = fold.get("bounds")
        if not isinstance(bounds, Mapping) or not all(
            isinstance(bounds.get(k), (int, float)) and not isinstance(bounds.get(k), bool) and math.isfinite(float(bounds.get(k)))  # type: ignore[arg-type]
            for k in ("v_floor", "v_cap")
        ):
            reasons.append(f"fold {index} bounds are invalid")
        scale_params = fold.get("scale_parameters")
        if isinstance(scale_params, bool) or not isinstance(scale_params, int) or scale_params <= 0 or scale_params > N4_SCALE_PARAM_LIMIT:
            reasons.append(f"fold {index} scale capacity is outside the frozen envelope")
        details.append(entry)
    if len(counts) == 3 and all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in counts):
        expected_total = N4_SCALE_EPOCHS * sum(math.ceil(int(c) / N4_SCALE_BATCH_SIZE) for c in counts)
        observed_total = sum(
            int(f.get("optimizer_steps", 0)) for f in folds if isinstance(f, Mapping) and isinstance(f.get("optimizer_steps"), int) and not isinstance(f.get("optimizer_steps"), bool)
        )
        if per_seed_updates != expected_total or observed_total != expected_total:
            reasons.append("per-seed scale-update total differs from 50*sum(ceil(N_fold/512))")
    verdict = "PASS" if not reasons else "FAILED"
    return {"verdict": verdict, "reasons": list(reasons), "folds": details}


def n4_score_files(
    *,
    model: nn.Module,
    samples: Sequence[object],
    scale_model: ContextResidualScaleModel,
    v_floor: float,
    v_cap: float,
    patchifier: object,
    masking_seed: int | None,
) -> dict[str, dict[str, float]]:
    """Score files through the frozen B0 encoder plus the fitted N4 scale head.

    Returns per-file ``{"raw": ..., "scale": ..., "normalized": ...}`` where
    ``raw`` is the file-mean raw B0 residual ``e_t``, ``scale`` is the
    file-mean predicted variance, and ``normalized`` is the file-mean
    ``S_pred_patch_CRN``. Patch filtering, patch-to-file aggregation, and
    patch starts/valid lengths/localization follow the unchanged B0 path:
    only eligible masked valid patches enter, and the file score is the mean
    over those patches. The bank/``S_pop`` path is untouched here; callers
    prove ``S_pop`` parity separately with the restored B0 bank.
    """
    from representation.data import collate_variable_files as _collate

    if not isinstance(scale_model, ContextResidualScaleModel):
        raise ValueError("N4 file scoring requires the fitted context-only scale model")
    for name, bound in (("v_floor", v_floor), ("v_cap", v_cap)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"N4 {name} bound must be a number")
        if not math.isfinite(float(bound)) or float(bound) <= 0.0:
            raise ValueError(f"N4 {name} bound must be finite and positive")
    if float(v_cap) < float(v_floor):
        raise ValueError("N4 v_cap must not lie below v_floor")
    materialized = list(samples)
    if not materialized:
        raise ValueError("N4 file scoring requires a non-empty file batch")
    model.eval()
    scale_model.eval()
    scored: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        batch = _collate(
            materialized,  # type: ignore[arg-type]
            patchifier,  # type: ignore[arg-type]
            masking_config=getattr(model, "config", None),
            masking_seed=masking_seed,
        )
        clean = {key: value for key, value in batch.items() if key != "file_samples"}
        # Co-locate the collated batch onto the deployed model device before the
        # frozen forward, mirroring RepresentationInference._move_batch. CPU-only
        # callers observe a no-op; CUDA-deployed B0/scale heads consume aligned
        # tensors instead of raising a device mismatch inside layer_norm.
        _model_device_param = next(model.parameters(), None)
        _model_device = _model_device_param.device if _model_device_param is not None else None
        if _model_device is not None:
            clean = {
                key: (value.to(_model_device) if isinstance(value, torch.Tensor) else value)
                for key, value in clean.items()
            }
        output = model(clean)  # type: ignore[operator]
        # Fail closed when the fitted scale head is deployed on a different
        # device than the frozen B0 encoder (normative runners co-deploy both).
        _scale_device = next(scale_model.parameters(), None)
        if _scale_device is not None and _model_device is not None:
            if _scale_device.device != _model_device:
                raise ValueError("N4 scoring scale head device differs from the B0 model device")
        context = output["context_latents"].detach()
        target = output["target_latents"].detach()
        predicted = output["predicted_latents"].detach()
        requested = output.get("prediction_mask")
        valid = output.get("patch_valid_mask")
        if (
            not isinstance(requested, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or requested.dtype is not torch.bool
            or valid.dtype is not torch.bool
        ):
            raise ValueError("N4 encoder output masks are invalid")
        eligible = requested & valid
        file_ids = batch["file_ids"]
        if len(file_ids) != context.shape[0]:
            raise ValueError("N4 scored batch file IDs differ from the encoded batch")
        width = context.shape[-1]
        if width != scale_model.d_model:
            raise ValueError("N4 scoring scale width differs from the bound model width")
        residuals = (
            (torch.nn.functional.layer_norm(target, (width,))
             - torch.nn.functional.layer_norm(predicted, (width,))).square().mean(dim=-1).detach()
        )
        scales = scale_model(
            context.detach().reshape(-1, width), v_floor=float(v_floor), v_cap=float(v_cap)
        )["v"].detach().reshape(context.shape[0], context.shape[1])
        normalized = residuals / (scales + 1e-6)
        for row, file_id in enumerate(file_ids):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("N4 scoring requires canonical file IDs")
            rows = torch.nonzero(eligible[row], as_tuple=False).flatten().tolist()
            if not rows:
                raise ValueError("N4 scoring file has no eligible masked valid patches")
            raw_mean = float(residuals[row, rows].mean().item())
            scale_mean = float(scales[row, rows].mean().item())
            norm_mean = float(normalized[row, rows].mean().item())
            for value in (raw_mean, scale_mean, norm_mean):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("N4 file score must be finite and non-negative")
            scored[file_id] = {"raw": raw_mean, "scale": scale_mean, "normalized": norm_mean}
    return scored


def n4_calibration_q95(scores: Mapping[str, float]) -> float:
    """Select the operating threshold as q95 of eligible healthy file scores.

    Uses the exact common-protocol NumPy linear-interpolation convention.
    Callers pass only eligible Calibration healthy ``normalized`` file scores;
    history binding is proven by the caller (see the T19 proof script), never
    inferred here.
    """
    values = np.asarray(list(scores.values()), dtype=np.float64).reshape(-1)
    if values.shape[0] < 1:
        raise ValueError("N4 calibration requires a non-empty healthy score pool")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("N4 calibration scores must be finite and non-negative")
    threshold = float(np.quantile(values, 0.95, method="linear"))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("N4 calibration quantile is invalid")
    return threshold


def n4_cache_binding_digest(
    *,
    checkpoint_sha256: str,
    residual_records_sha256: str,
    fit_role_sha256: str,
    scale_state_sha256: str,
    scale_parameterization: str,
    bounds: Mapping[str, float],
) -> str:
    """Bind the exact paired B0 checkpoint, Fit cache records, and scale state.

    Returns the SHA-256 of the canonical JSON binding record
    ``{"checkpoint_sha256", "residual_records_sha256", "fit_role_sha256",
    "scale_state_sha256", "scale_parameterization", "q05", "q95", "v_floor",
    "v_cap"}``. Any substituted checkpoint, cache, role digest, scale state,
    bound, or a missing/unknown ``scale_parameterization`` fails closed at the
    consumer: :func:`verify_n4_cache_binding` recomputes this digest and
    rejects mismatches. Legacy v1 A03 digests carry no parameterization and are
    not valid v3 bindings. ``q95`` is the exact Fit residual 0.95 quantile name
    used by :func:`n4_residual_bounds`; the ``q99`` spelling is rejected.
    """
    for name, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("residual_records_sha256", residual_records_sha256),
        ("fit_role_sha256", fit_role_sha256),
        ("scale_state_sha256", scale_state_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"N4 {name} must be a lowercase SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"N4 {name} must be a lowercase SHA-256 digest") from exc
    if not isinstance(scale_parameterization, str) or scale_parameterization != N4_SCALE_PARAMETERIZATION:
        raise ValueError("N4 cache binding requires the frozen v3 scale parameterization identity")
    if not isinstance(bounds, Mapping) or set(bounds) != {"q05", "q95", "v_floor", "v_cap"}:
        raise ValueError("N4 cache binding requires the exact q05/q95/v_floor/v_cap bound set")
    for name in ("q05", "q95", "v_floor", "v_cap"):
        value = bounds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"N4 bound {name} is invalid")
    record = {
        "checkpoint_sha256": checkpoint_sha256,
        "fit_role_sha256": fit_role_sha256,
        "q05": float(bounds["q05"]),
        "q95": float(bounds["q95"]),
        "residual_records_sha256": residual_records_sha256,
        "scale_parameterization": scale_parameterization,
        "scale_state_sha256": scale_state_sha256,
        "v_cap": float(bounds["v_cap"]),
        "v_floor": float(bounds["v_floor"]),
    }
    return sha256(
        __import__("json").dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_n4_cache_binding(
    *,
    expected_digest: str,
    checkpoint_sha256: str,
    residual_records_sha256: str,
    fit_role_sha256: str,
    scale_state_sha256: str,
    scale_parameterization: str,
    bounds: Mapping[str, float],
) -> str:
    """Recompute the cache binding digest and fail closed on any substitution."""
    observed = n4_cache_binding_digest(
        checkpoint_sha256=checkpoint_sha256,
        residual_records_sha256=residual_records_sha256,
        fit_role_sha256=fit_role_sha256,
        scale_state_sha256=scale_state_sha256,
        scale_parameterization=scale_parameterization,
        bounds=bounds,
    )
    if not isinstance(expected_digest, str) or observed != expected_digest:
        raise ValueError("N4 checkpoint/cache binding mismatch: refusing substituted cache or scale state")
    return observed
