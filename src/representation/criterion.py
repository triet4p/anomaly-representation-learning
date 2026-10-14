"""Losses for the V1 latent prediction and contrastive objectives."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn


class LatentPredictionCriterion(nn.Module):
    """Compare predicted and EMA target latents only at valid masked patches."""

    def __init__(self, empty_policy: Literal["zero", "error"] = "zero") -> None:
        super().__init__()
        if empty_policy not in {"zero", "error"}:
            raise ValueError("empty_policy must be 'zero' or 'error'")
        self.empty_policy = empty_policy

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return loss, per-patch errors, and auditable masked counts."""
        predicted = self._tensor(model_output, "predicted_latents", 3)
        target = self._tensor(model_output, "target_latents", 3).detach()
        requested = model_output.get("prediction_mask", model_output.get("mask"))
        if not isinstance(requested, torch.Tensor):
            raise ValueError("model output must contain prediction_mask")
        valid = model_output.get("patch_valid_mask")
        if valid is None:
            valid = torch.ones_like(requested, dtype=torch.bool)
        if not isinstance(valid, torch.Tensor):
            raise ValueError("patch_valid_mask must be a torch.Tensor")
        if predicted.shape != target.shape:
            raise ValueError("predicted_latents and target_latents must share shape")
        if requested.shape != predicted.shape[:2] or valid.shape != predicted.shape[:2]:
            raise ValueError("prediction masks must match latent batch and count")
        if requested.dtype is not torch.bool or valid.dtype is not torch.bool:
            raise ValueError("prediction masks must have torch.bool dtype")
        mask = requested & valid
        errors = (predicted - target).square().mean(dim=-1).masked_fill(~mask, 0.0)
        counts_per_file = mask.sum(dim=1)
        masked_count = counts_per_file.sum()
        if masked_count.item() == 0:
            if self.empty_policy == "error":
                raise ValueError("no valid masked patches available for prediction loss")
            loss = predicted.sum() * 0.0
        else:
            loss = errors.sum() / masked_count.to(errors.dtype)
        return {"loss": loss, "patch_prediction_error": errors, "masked_count": masked_count, "masked_counts": counts_per_file}

    @staticmethod
    def _tensor(model_output: Mapping[str, object], name: str, ndim: int) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D torch.Tensor")
        return value


class FileContrastiveCriterion(nn.Module):
    """Apply symmetric in-batch InfoNCE to two views of each file."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, model_output: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Return contrastive loss and normalized same-file view embeddings."""
        view_1 = self._embedding(model_output, "view_embedding_1")
        view_2 = self._embedding(model_output, "view_embedding_2")
        if view_1.shape != view_2.shape:
            raise ValueError("contrastive views must share shape")
        if view_1.shape[0] == 0:
            raise ValueError("contrastive views cannot be empty")
        normalized_1 = F.normalize(view_1, dim=-1)
        normalized_2 = F.normalize(view_2, dim=-1)
        batch_size = view_1.shape[0]
        if batch_size == 1:
            loss = (normalized_1 - normalized_2).square().sum() * 0.0
            similarity = (normalized_1 * normalized_2).sum(dim=-1).mean()
            negative_similarity = torch.tensor(0.0, device=view_1.device)
            margin = similarity
        else:
            sim_12 = normalized_1 @ normalized_2.transpose(0, 1)
            sim_21 = normalized_2 @ normalized_1.transpose(0, 1)
            logits_12 = sim_12 / self.temperature
            logits_21 = sim_21 / self.temperature
            labels = torch.arange(batch_size, device=view_1.device)
            loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
            similarity = sim_12.diag().mean()
            neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=view_1.device)
            negative_similarity = 0.5 * (sim_12[neg_mask].mean() + sim_21[neg_mask].mean())
            margin = similarity - negative_similarity
        return {
            "loss": loss,
            "similarity": similarity,
            "negative_similarity": negative_similarity,
            "margin": margin,
            "normalized_view_1": normalized_1,
            "normalized_view_2": normalized_2,
            "batch_size": torch.tensor(batch_size, device=view_1.device),
        }

    @staticmethod
    def _embedding(model_output: Mapping[str, object], name: str) -> torch.Tensor:
        value = model_output.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a 2D torch.Tensor")
        return value


MSP_TOTAL_MASK_RATIO = 0.40
MSP_POSITION_SHARES = (0.50, 0.25, 0.25)
MSP_FINE_MASK_COMPOSITION = (0.34, 0.33, 0.33)
MSP_FINE_MASK_STRATA = (0.20, 0.30, 0.30, 0.20)
MSP_FINE_MASK_BLOCK_RANGE = (2, 6)
MSP_AUX_COEFFICIENT_GRID = (0.10, 0.25, 0.50)


def msp_mask_budget(n: int) -> int:
    """Return the exact N2 unique-mask budget ``M = min(n, round(0.40*n))``."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("patch count n must be a non-negative integer")
    return min(n, int(round(MSP_TOTAL_MASK_RATIO * n)))


def msp_allocate_budget(M: int) -> tuple[int, int, int]:
    """Allocate ``M`` masked positions to ``(F, g4, g8)`` per N2 v2 section 3."""
    if isinstance(M, bool) or not isinstance(M, int) or M < 0:
        raise ValueError("mask budget M must be a non-negative integer")
    best: tuple[int, int, int] | None = None
    best_score: int | None = None
    for g8 in range(M // 8 + 1):
        for g4 in range((M - 8 * g8) // 4 + 1):
            fine = M - 8 * g8 - 4 * g4
            score = (
                4 * (2 * fine - M) ** 2
                + (16 * g4 - M) ** 2
                + (32 * g8 - M) ** 2
            )
            candidate = (fine, g4, g8)
            if best_score is None or score < best_score or (
                score == best_score and best is not None and candidate > best
            ):
                best = candidate
                best_score = score
    assert best is not None
    return best


def msp_mask_seed(model_seed: int, epoch: int, fit_role_sha256: str, file_id: str) -> int:
    """Derive the frozen per-file/epoch N2 mask seed from identity only."""
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= 100
        or not isinstance(fit_role_sha256, str)
        or len(fit_role_sha256) != 64
        or any(c not in "0123456789abcdef" for c in fit_role_sha256)
        or not isinstance(file_id, str)
        or not file_id
    ):
        raise ValueError("N2 mask seed requires a valid seed, epoch, role digest, and file ID")
    key = f"sprint18-n2-mask-v2|{model_seed}|{epoch}|{fit_role_sha256}|{file_id}"
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _msp_runs(free: set[int]) -> list[list[int]]:
    """Return maximal consecutive runs of still-eligible original patch indices."""
    runs: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for index in sorted(free):
        if previous is None or index == previous + 1:
            current.append(index)
        else:
            runs.append(current)
            current = [index]
        previous = index
    if current:
        runs.append(current)
    return runs


def _msp_capacity(free: set[int], length: int) -> int:
    """Return ``sum(floor(run/ MPI length))``, the exact disjoint-group capacity."""
    return sum(len(run) // length for run in _msp_runs(free))


def _msp_place_groups(
    candidates: list[int],
    order: npt.NDArray[np.int64],
    free: set[int],
    quota: int,
    length: int,
) -> list[tuple[int, ...]]:
    """Place up to ``quota`` length-``length`` groups with capacity-preserving acceptance."""
    selected: list[tuple[int, ...]] = []
    for position in (int(v) for v in order.tolist()):
        if len(selected) >= quota:
            break
        start = candidates[position]
        members = tuple(start + offset for offset in range(length))
        if any(member not in free for member in members):
            continue
        remaining = set(free) - set(members)
        still_needed = quota - len(selected) - 1
        if _msp_capacity(remaining, length) < still_needed:
            continue
        selected.append(members)
        free.difference_update(members)
    return selected


def msp_place_groups(
    valid_indices: Sequence[int],
    g8: int,
    g4_request: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Place seeded non-overlapping coarse-8 then coarse-4 groups per N2 v2.

    Candidates are every length-``L`` window of adjacent original B0 indices
    all in ``valid_indices``, enumerated by ascending first index; selection
    considers them in ``rng.permutation`` order with capacity-preserving
    acceptance (``sum(floor(run/L)) >= groups_still_needed``). Unplaced 8-groups
    are reassigned to two 4-groups each; unplaced 4-positions fall through to
    fine. Records both fallback counts.
    """
    valid = [int(v) for v in valid_indices]
    if any(isinstance(v, bool) for v in valid_indices) or any(v < 0 for v in valid):
        raise ValueError("valid patch indices must be non-negative integers")
    if sorted(valid) != valid or len(set(valid)) != len(valid):
        raise ValueError("valid patch indices must be sorted and unique")
    for name, value in (("g8", g8), ("g4_request", g4_request)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("placement requires a numpy Generator")
    valid_set = set(valid)
    candidates_8 = [s for s in valid if all(s + o in valid_set for o in range(8))]
    free: set[int] = set(valid)
    cap8 = _msp_capacity(free, 8)
    q8_target = min(g8, cap8)
    order_8 = rng.permutation(len(candidates_8)) if candidates_8 else np.zeros(0, dtype=np.int64)
    groups_8 = _msp_place_groups(candidates_8, order_8, free, q8_target, 8)
    fallback_8_to_4 = g8 - len(groups_8)
    request_4 = g4_request + 2 * fallback_8_to_4
    candidates_4 = [s for s in valid if all(s + o in valid_set for o in range(4))]
    cap4 = _msp_capacity(free, 4)
    q4_target = min(request_4, cap4)
    order_4 = rng.permutation(len(candidates_4)) if candidates_4 else np.zeros(0, dtype=np.int64)
    groups_4 = _msp_place_groups(candidates_4, order_4, free, q4_target, 4)
    fallback_4_to_fine = request_4 - len(groups_4)
    return {
        "groups_8": groups_8,
        "groups_4": groups_4,
        "q8": len(groups_8),
        "q4": len(groups_4),
        "q8_fallback_to_4": fallback_8_to_4,
        "q4_fallback_to_fine": fallback_4_to_fine,
        "remaining_free": sorted(free),
    }


def msp_fine_mask(
    remaining_free: Sequence[int],
    f: int,
    patch_data: npt.NDArray[np.float32],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Apply the frozen B0 fine-mask algorithm with the integer target ``f``.

    ``patch_data`` is the full ``[N, C, W]`` original patch array indexed by
    original B0 patch position; ``remaining_free`` positions index into it.
    """
    free = [int(v) for v in remaining_free]
    if any(isinstance(v, bool) for v in remaining_free) or any(v < 0 for v in free):
        raise ValueError("remaining free indices must be non-negative integers")
    if sorted(free) != free or len(set(free)) != len(free):
        raise ValueError("remaining free indices must be sorted and unique")
    if isinstance(f, bool) or not isinstance(f, int) or f < 0:
        raise ValueError("fine target f must be a non-negative integer")
    if f > len(free):
        raise ValueError("fine target exceeds the remaining eligible pool")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("fine masking requires a numpy Generator")
    if not isinstance(patch_data, np.ndarray):
        raise ValueError("patch data must be a numpy array")
    composition = list(MSP_FINE_MASK_COMPOSITION)
    strata_frac = list(MSP_FINE_MASK_STRATA)
    for vector in (composition, strata_frac):
        total = sum(vector)
        if total <= 0 or any(v < 0 for v in vector):
            raise ValueError("masking fractions must be non-negative with positive sum")
    composition = [v / sum(composition) for v in composition]
    strata_frac = [v / sum(strata_frac) for v in strata_frac]
    # Largest-remainder apportion in listed order.
    base = [int(np.floor(c * f)) for c in composition]
    remainders = [c * f - b for c, b in zip(composition, base)]
    order = sorted(range(len(base)), key=lambda i: (-remainders[i], i))
    shortfall = f - sum(base)
    for i in order[:shortfall]:
        base[i] += 1
    n_random, n_info, n_block = (int(v) for v in base)
    selected: list[int] = []
    selected_set: set[int] = set()
    available = list(free)
    if n_random:
        take = min(n_random, len(available))
        chosen = [int(v) for v in rng.choice(np.asarray(available, dtype=np.int64), size=take, replace=False).tolist()] if take else []
        selected.extend(chosen)
        selected_set.update(chosen)
        available = [v for v in available if v not in selected_set]
    if n_info and available:
        flat = np.asarray(patch_data).reshape(len(patch_data), -1).astype(np.float64)
        std = np.std(flat, axis=1)
        span = np.ptp(flat, axis=1)
        deriv = np.mean(np.abs(np.diff(flat, axis=1)), axis=1) if flat.shape[1] >= 2 else np.zeros(flat.shape[0])
        scores = (std + 1e-8) * (span + 1e-8) * (1.0 + deriv)
        free_arr = np.asarray(available, dtype=np.int64)
        # Scores are indexed by original patch position; map free positions directly.
        values = np.asarray([float(scores[v]) if 0 <= v < len(scores) else 0.0 for v in available], dtype=np.float64)
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        strata = {
            "stable": [v for v, s in zip(available, values) if s <= q25],
            "transition": [v for v, s in zip(available, values) if q25 < s <= q50],
            "dynamic": [v for v, s in zip(available, values) if q50 < s <= q75],
            "extreme": [v for v, s in zip(available, values) if s > q75],
        }
        names = ("stable", "transition", "dynamic", "extreme")
        info_base = [int(np.floor(frac * n_info)) for frac in strata_frac]
        info_rem = [frac * n_info - b for frac, b in zip(strata_frac, info_base)]
        info_order = sorted(range(4), key=lambda i: (-info_rem[i], i))
        for i in info_order[: n_info - sum(info_base)]:
            info_base[i] += 1
        for name, quota in zip(names, info_base):
            pool = [v for v in strata[name] if v not in selected_set]
            take = min(int(quota), len(pool))
            if take:
                chosen = [int(v) for v in rng.choice(np.asarray(pool, dtype=np.int64), size=take, replace=False).tolist()]
                selected.extend(chosen)
                selected_set.update(chosen)
        if len([v for v in selected if v in set(available)]) < n_info:
            still_free = [v for v in available if v not in selected_set]
            extra = n_info - len(selected_set.intersection(set(available)))
            if still_free and extra > 0:
                take = min(extra, len(still_free))
                chosen = [int(v) for v in rng.choice(np.asarray(still_free, dtype=np.int64), size=take, replace=False).tolist()]
                selected.extend(chosen)
                selected_set.update(chosen)
        available = [v for v in available if v not in selected_set]
    if n_block and available:
        pool = list(available)
        need = n_block
        while need > 0 and pool:
            take = min(need, len(pool), max(1, int(rng.integers(MSP_FINE_MASK_BLOCK_RANGE[0], MSP_FINE_MASK_BLOCK_RANGE[1] + 1))))
            start = int(rng.integers(0, len(pool) - take + 1))
            chosen = pool[start : start + take]
            selected.extend(int(v) for v in chosen)
            selected_set.update(int(v) for v in chosen)
            pool = [v for v in pool if v not in selected_set]
            need = n_block - len([v for v in selected if v in set(available)])
        available = [v for v in available if v not in selected_set]
    shortfall = f - len(selected)
    filled_from: str | None = None
    if shortfall > 0:
        # Lowest-index defensive fill, attributed to the first positive component.
        rest = sorted(set(free) - selected_set)[:shortfall]
        selected.extend(int(v) for v in rest)
        selected_set.update(int(v) for v in rest)
        for name, frac in zip(("random", "info", "block"), composition):
            if frac > 0:
                filled_from = name
                break
    if len(selected) != f or len(selected_set) != f:
        raise ValueError("fine masking did not reach its exact integer target")
    realized = {"random": 0, "info": 0, "block": 0}
    # Attribute realized counts: random quota first, then info, then block, with
    # any defensive fill on the first positive component (B0 accounting).
    realized["random"] = min(n_random, f)
    remaining_count = f - realized["random"]
    realized["info"] = min(n_info, remaining_count)
    remaining_count -= realized["info"]
    realized["block"] = max(0, remaining_count)
    return {"selected": sorted(int(v) for v in selected_set), "composition": realized, "defensive_fill_component": filled_from}


HCC_AFFINITY_SAME_FILE = 1.0
HCC_AFFINITY_SAME_ROBOT_PROGRAM = 0.5
HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM = 0.1
HCC_AFFINITY_DIFFERENT_ROBOT = 0.0


def hcc_file_affinities(robot_idx: torch.Tensor, program_idx: torch.Tensor) -> torch.Tensor:
    """Return the frozen file-level HCC affinity matrix from robot/program identity only."""
    for name, value in (("robot_idx", robot_idx), ("program_idx", program_idx)):
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"HCC {name} must be a 1D torch.Tensor")
        if value.dtype.is_floating_point:
            raise ValueError(f"HCC {name} must contain integer category IDs")
        if value.shape[0] == 0:
            raise ValueError("HCC relation batch cannot be empty")
    if robot_idx.shape != program_idx.shape:
        raise ValueError("HCC robot and program identities must share batch size")
    if bool((robot_idx < 0).any()) or bool((program_idx < 0).any()):
        raise ValueError("HCC robot and program identities must be non-negative")
    batch_size = int(robot_idx.shape[0])
    same_robot = robot_idx.unsqueeze(1) == robot_idx.unsqueeze(0)
    same_program = program_idx.unsqueeze(1) == program_idx.unsqueeze(0)
    affinities = torch.zeros((batch_size, batch_size), dtype=torch.float32)
    affinities[same_robot & same_program] = HCC_AFFINITY_SAME_ROBOT_PROGRAM
    affinities[same_robot & ~same_program] = HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM
    affinities.fill_diagonal_(HCC_AFFINITY_SAME_FILE)
    return affinities


def hcc_relation_support(robot_idx: torch.Tensor, program_idx: torch.Tensor) -> dict[str, float | int]:
    """Count natural relation support and target mass from robot/program identity only."""
    affinities = hcc_file_affinities(robot_idx, program_idx)
    batch_size = int(affinities.shape[0])
    cross_same_rp = (affinities == HCC_AFFINITY_SAME_ROBOT_PROGRAM).any(dim=1)
    cross_same_r_diff_p = (affinities == HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM).any(dim=1)
    # Diagonal is the same-file view; any positive off-diagonal mass means cross-file support.
    off_diagonal_positive = ((affinities > 0.0) & ~torch.eye(batch_size, dtype=torch.bool)).any(dim=1)
    anchors_same_rp = int(cross_same_rp.sum().item())
    anchors_same_r_diff_p = int(cross_same_r_diff_p.sum().item())
    anchors_only_same_file = int((~off_diagonal_positive).sum().item())
    row_sums = affinities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    targets = affinities / row_sums
    entropy = float((-(targets * targets.clamp_min(1e-12).log()).sum(dim=1)).mean().item())
    mass_same_file = float(targets.diagonal().mean().item())
    mass_same_rp = float(((targets * (affinities == HCC_AFFINITY_SAME_ROBOT_PROGRAM)).sum(dim=1)).mean().item())
    mass_same_r_diff_p = float(((targets * (affinities == HCC_AFFINITY_SAME_ROBOT_DIFF_PROGRAM)).sum(dim=1)).mean().item())
    return {
        "batch_files": batch_size,
        "anchors_same_rp": anchors_same_rp,
        "anchors_same_r_diff_p": anchors_same_r_diff_p,
        "anchors_only_same_file": anchors_only_same_file,
        "target_entropy_mean": entropy,
        "mass_same_file": mass_same_file,
        "mass_same_rp": mass_same_rp,
        "mass_same_r_diff_p": mass_same_r_diff_p,
    }


class HierarchicalContrastiveCriterion(nn.Module):
    """Apply the frozen soft hierarchical relation-target contrastive loss."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self,
        view_1: torch.Tensor,
        view_2: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the symmetric HCC loss plus natural relation-support diagnostics."""
        for name, value in (("view_1", view_1), ("view_2", view_2)):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"HCC {name} must be a 2D torch.Tensor")
        if view_1.shape != view_2.shape:
            raise ValueError("HCC contrastive views must share shape")
        if view_1.shape[0] == 0:
            raise ValueError("HCC contrastive views cannot be empty")
        if view_1.shape[0] != int(robot_idx.shape[0]) or view_1.shape[0] != int(program_idx.shape[0]):
            raise ValueError("HCC view and identity batch sizes differ")
        normalized_1 = F.normalize(view_1, dim=-1)
        normalized_2 = F.normalize(view_2, dim=-1)
        sim_12 = (normalized_1 @ normalized_2.transpose(0, 1)) / self.temperature
        sim_21 = sim_12.transpose(0, 1)
        affinities = hcc_file_affinities(robot_idx, program_idx).to(device=view_1.device)
        row_sums = affinities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        targets = affinities / row_sums
        log_p_12 = F.log_softmax(sim_12, dim=1)
        log_p_21 = F.log_softmax(sim_21, dim=1)
        loss_12 = -(targets * log_p_12).sum(dim=1).mean()
        loss_21 = -(targets * log_p_21).sum(dim=1).mean()
        loss = 0.5 * (loss_12 + loss_21)
        support = hcc_relation_support(robot_idx.detach().cpu(), program_idx.detach().cpu())
        return {
            "loss": loss,
            "similarity": 0.5 * (sim_12.diag().mean() * self.temperature + sim_21.diag().mean() * self.temperature),
            "normalized_view_1": normalized_1,
            "normalized_view_2": normalized_2,
            "batch_size": torch.tensor(int(view_1.shape[0]), device=view_1.device),
            "anchors_same_rp": torch.tensor(support["anchors_same_rp"], device=view_1.device),
            "anchors_same_r_diff_p": torch.tensor(support["anchors_same_r_diff_p"], device=view_1.device),
            "anchors_only_same_file": torch.tensor(support["anchors_only_same_file"], device=view_1.device),
            "target_entropy_mean": torch.tensor(support["target_entropy_mean"], device=view_1.device),
        }


class MultiscaleLatentPredictionCriterion(nn.Module):
    """Apply the frozen N2 coarse-4/8 normalized latent-distance loss."""

    def __init__(self, empty_policy: str = "exclude") -> None:
        super().__init__()
        if empty_policy != "exclude":
            raise ValueError("coarse loss excludes missing scale support from its denominator")
        self.empty_policy = empty_policy

    @staticmethod
    def _group_loss(
        context: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        groups_4: Sequence[Sequence[int]],
        groups_8: Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        for name, value in (("context_latents", context), ("target_latents", target)):
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(f"{name} must be a 3D torch.Tensor")
        if context.shape != target.shape:
            raise ValueError("coarse context and target latents must share shape")
        if not isinstance(valid, torch.Tensor) or valid.ndim != 2 or valid.dtype is not torch.bool:
            raise ValueError("patch_valid_mask must be a bool [B, N] tensor")
        if tuple(valid.shape) != tuple(context.shape[:2]):
            raise ValueError("patch_valid_mask must match latent batch and count")
        batch, count, width = context.shape
        for scale, groups in (("group4", groups_4), ("group8", groups_8)):
            if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
                raise ValueError(f"{scale} groups must be a sequence of index tuples")
            for group in groups:
                if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                    raise ValueError(f"{scale} group members must be index sequences")
                if len(group) not in (4, 8):
                    raise ValueError(f"{scale} groups must have length 4 or 8")
                for member in group:
                    if isinstance(member, bool) or not isinstance(member, int) or not 0 <= member < count:
                        raise ValueError(f"{scale} member indices must be valid patch positions")
        # Teacher targets are stop-gradient; detach once here.
        detached = target.detach()
        per_scale: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        for scale, groups, span in (("group4", groups_4, 4), ("group8", groups_8, 8)):
            errors: list[torch.Tensor] = []
            for group in groups:
                members = [int(m) for m in group]
                if len(members) != span:
                    raise ValueError(f"{scale} groups must have exactly {span} members")
                teacher_rows = detached[:, members, :]
                # Uniform member weighting: N2 v2 valid_weight_t is the valid-timestep
                # fraction per patch; padded weight zero is handled by eligibility
                # (groups never contain padding) and valid-mask gating below.
                teacher = torch.layer_norm(teacher_rows.mean(dim=1), (width,))
                student_in = context[:, members, :].mean(dim=1)
                student = torch.layer_norm(student_in, (width,))
                errors.append(((student - teacher).square().mean(dim=-1)))
            if errors:
                stacked = torch.stack(errors, dim=1)
                per_scale[scale] = stacked.mean()
                counts[scale] = len(groups)
            else:
                per_scale[scale] = (context.sum() * 0.0) + 0.0
                counts[scale] = 0
        if counts["group4"] and counts["group8"]:
            coarse = 0.5 * per_scale["group4"] + 0.5 * per_scale["group8"]
        elif counts["group4"]:
            coarse = per_scale["group4"]
        elif counts["group8"]:
            coarse = per_scale["group8"]
        else:
            coarse = (context.sum() * 0.0) + 0.0
        return {
            "group4_loss": per_scale["group4"],
            "group8_loss": per_scale["group8"],
            "coarse_loss": coarse,
            "group4_count": torch.tensor(counts["group4"]),
            "group8_count": torch.tensor(counts["group8"]),
        }

    def forward(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        groups_4: Sequence[Sequence[int]],
        groups_8: Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        """Return per-scale and combined coarse losses with missing-scale exclusion."""
        return self._group_loss(context_latents, target_latents, patch_valid_mask, groups_4, groups_8)


class ProgressiveLambda:
    """Monotonic linear contrastive-weight schedule with an optional warmup."""

    def __init__(
        self,
        lambda_max: float = 1.0,
        ramp_steps: int = 1_000,
        warmup_steps: int = 0,
    ) -> None:
        if lambda_max < 0.0:
            raise ValueError("lambda_max must be non-negative")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.lambda_max = float(lambda_max)
        self.ramp_steps = int(ramp_steps)
        self.warmup_steps = int(warmup_steps)

    def lambda_at(self, step: int) -> float:
        """Return the exact contrastive weight at a non-negative step."""
        if step < 0:
            raise ValueError("step must be non-negative")
        if step <= self.warmup_steps:
            return 0.0
        if self.ramp_steps == 0:
            return self.lambda_max
        progress = (step - self.warmup_steps) / self.ramp_steps
        return self.lambda_max * min(1.0, progress)

    @property
    def full_ramp_step(self) -> int:
        """Return the first non-negative step at which lambda reaches lambda_max."""
        if self.ramp_steps == 0:
            return self.warmup_steps + (1 if self.lambda_max > 0.0 else 0)
        return self.warmup_steps + self.ramp_steps

    def is_full_ramp(self, step: int) -> bool:
        """Return whether lambda has reached lambda_max at or past the full-ramp boundary."""
        if step < 0:
            raise ValueError("step must be non-negative")
        return step >= self.full_ramp_step and (
            self.lambda_max == 0.0 or self.lambda_at(step) >= self.lambda_max
        )


class JointRepresentationCriterion(nn.Module):
    """Combine masked prediction and progressive file-level contrastive losses."""

    def __init__(self, prediction: LatentPredictionCriterion | None = None, contrastive: FileContrastiveCriterion | None = None, lambda_schedule: ProgressiveLambda | None = None, prediction_weight: float = 1.0) -> None:
        super().__init__()
        if prediction_weight < 0.0:
            raise ValueError("prediction_weight must be non-negative")
        self.prediction = prediction or LatentPredictionCriterion()
        self.contrastive = contrastive or FileContrastiveCriterion()
        self.lambda_schedule = lambda_schedule or ProgressiveLambda()
        self.prediction_weight = float(prediction_weight)

    def forward(
        self,
        model_output: Mapping[str, object],
        step: int = 0,
        *,
        effective_lambda_override: float | None = None,
    ) -> dict[str, torch.Tensor | float | int]:
        """Return raw/weighted objective terms, schedule value, and support."""
        prediction_result = self.prediction(model_output)
        prediction_loss = prediction_result["loss"]
        contrastive_sim = None
        contrastive_neg_sim = None
        contrastive_margin = None
        contrastive_input_files = 0
        contrastive_support_files = 0
        if "view_embedding_1" in model_output and "view_embedding_2" in model_output:
            contrastive_result = self.contrastive(model_output)
            contrastive_loss = contrastive_result["loss"]
            if "similarity" in contrastive_result:
                contrastive_sim = contrastive_result["similarity"]
            if "negative_similarity" in contrastive_result:
                contrastive_neg_sim = contrastive_result["negative_similarity"]
            if "margin" in contrastive_result:
                contrastive_margin = contrastive_result["margin"]
            view_1 = model_output["view_embedding_1"]
            if not isinstance(view_1, torch.Tensor):
                raise TypeError("view_embedding_1 must be a torch.Tensor")
            contrastive_input_files = int(view_1.shape[0])
            contrastive_support_files = (
                contrastive_input_files if contrastive_input_files > 1 else 0
            )
        else:
            contrastive_loss = prediction_loss * 0.0

        if effective_lambda_override is None:
            lambda_value = self.lambda_schedule.lambda_at(step)
        else:
            if (
                isinstance(effective_lambda_override, bool)
                or not isinstance(effective_lambda_override, (int, float))
                or not math.isfinite(float(effective_lambda_override))
                or effective_lambda_override < 0.0
            ):
                raise ValueError("effective contrastive coefficient must be finite and non-negative")
            lambda_value = float(effective_lambda_override)
        weighted_prediction_loss = self.prediction_weight * prediction_loss
        weighted_contrastive_loss = lambda_value * contrastive_loss
        joint_loss = weighted_prediction_loss + weighted_contrastive_loss
        stationary_lambda = self.lambda_schedule.lambda_max
        stationary_joint_loss = self.prediction_weight * prediction_loss + stationary_lambda * contrastive_loss

        diagnostics: dict[str, torch.Tensor | float | int] = {
            "prediction_loss": prediction_loss,
            "effective_prediction_coefficient": self.prediction_weight,
            "weighted_prediction_loss": weighted_prediction_loss,
            "prediction_support_patches": prediction_result["masked_count"],
            "contrastive_loss": contrastive_loss,
            "effective_contrastive_coefficient": lambda_value,
            "contrastive_input_files": contrastive_input_files,
            "contrastive_support_files": contrastive_support_files,
            "weighted_contrastive_loss": weighted_contrastive_loss,
            "joint_loss": joint_loss,
            "stationary_joint_loss": stationary_joint_loss,
            "loss": joint_loss,
            "lambda": lambda_value,
            "effective_lambda": lambda_value,
            "stationary_lambda": stationary_lambda,
            "masked_count": prediction_result["masked_count"],
        }
        if contrastive_sim is not None:
            diagnostics["contrastive_sim"] = contrastive_sim
        if contrastive_neg_sim is not None:
            diagnostics["contrastive_neg_sim"] = contrastive_neg_sim
        if contrastive_margin is not None:
            diagnostics["contrastive_margin"] = contrastive_margin
        pred_mask = model_output.get("prediction_mask")
        valid_mask = model_output.get("patch_valid_mask")

        context = model_output.get("context_latents")
        if isinstance(context, torch.Tensor) and context.ndim == 3:
            if isinstance(valid_mask, torch.Tensor):
                valid_float = valid_mask.to(context.dtype)
                diagnostics["context_norm"] = (context.detach().norm(dim=-1) * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            else:
                diagnostics["context_norm"] = context.detach().norm(dim=-1).mean()

        target = model_output.get("target_latents")
        if isinstance(target, torch.Tensor) and target.ndim == 3:
            if isinstance(valid_mask, torch.Tensor):
                valid_float = valid_mask.to(target.dtype)
                diagnostics["target_norm"] = (target.detach().norm(dim=-1) * valid_float).sum() / valid_float.sum().clamp_min(1.0)
            else:
                diagnostics["target_norm"] = target.detach().norm(dim=-1).mean()

        predicted = model_output.get("predicted_latents")
        if isinstance(predicted, torch.Tensor) and predicted.ndim == 3:
            if isinstance(pred_mask, torch.Tensor):
                pred_float = pred_mask.to(predicted.dtype)
                diagnostics["predicted_norm"] = (predicted.detach().norm(dim=-1) * pred_float).sum() / pred_float.sum().clamp_min(1.0)
            else:
                diagnostics["predicted_norm"] = predicted.detach().norm(dim=-1).mean()

        file_emb = model_output.get("file_embedding")
        if isinstance(file_emb, torch.Tensor) and file_emb.ndim == 2:
            diagnostics["file_embedding_norm"] = file_emb.detach().norm(dim=-1).mean()

        return diagnostics
