"""Sprint 17 Task 10 — C2 patchification alternatives (C2-A / C2-B).

Frozen-registry implementation for the C2 (patchification) suspect only.
Downstream mechanisms (local encoder, context encoder, objective, file
pooling form, geometry, scorers, aggregation, metrics) remain unchanged.

Registered replacements (protocol v4 section 2, immutable):

- ``C2-A``: fixed-width (W=32) windows at 50% overlap (stride 16),
  preserving valid lengths, padding, starts, and file identity.  The frozen
  B0 configuration is already exactly this lattice (patch 32 / stride 16 /
  pad_end), so C2-A is executed as a parity control: a new overlap-explicit
  code path that must reproduce the B0 lattice bitwise.  No support weights
  are emitted, so the downstream pooling path is literally unchanged.
- ``C2-B``: two fixed-width offset grids — ``(offset 0, stride 32)`` and
  ``(offset 16, stride 32)`` — whose union recovers the B0 start lattice
  (every 16 steps) at the same patch count scale (no FLOP-envelope breach),
  with unit-mass support weights so duplicated timesteps do not gain
  pooling weight.  Weights are C2 support metadata (like ``valid_len`` and
  the valid mask the unchanged pooling already consumes); the pooling form
  itself is unchanged.

Collate hook convention (duck-typed so ``collate_variable_files`` stays
generic): a patchifier MAY expose
``support_weights_for(starts, valid_len, n_timesteps)`` returning either
``None`` (no weights; unchanged downstream path) or a float array of
per-patch weights.  When any file yields weights, collation attaches a
zero-padded float32 ``patch_support_weights`` tensor; otherwise the key is
absent and every downstream consumer behaves exactly as B0.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from synth.config import PatchConfig
from synth.patchify import Patchifier
from synth.schema import FileSample, PatchBatch

PATCH_SIZE = 32
PAD_END = True
PAD_VALUE = 0.0

#: C2-A: single 50%-overlap grid; must reproduce the B0 lattice bitwise.
C2A_GRIDS: tuple[tuple[int, int], ...] = ((0, 16),)
#: C2-B: two half-density offset grids; union spacing matches B0 at ~B0 count.
C2B_GRIDS: tuple[tuple[int, int], ...] = ((0, 32), (16, 32),)

ARM_GRIDS: dict[str, tuple[tuple[int, int], ...]] = {
    "C2-A": C2A_GRIDS,
    "C2-B": C2B_GRIDS,
}

ARM_ADAPTER_DESCRIPTION: dict[str, str] = {
    "C2-A": ("overlap-explicit single-grid patchifier (W=32, stride 16, "
             "pad_end); no geometry adapter, no support weights; downstream "
             "pooling, masking, scoring, and metric code unchanged"),
    "C2-B": ("two fixed-width offset grids (W=32; offsets 0/16, stride 32) "
             "with unit-mass support weights consumed only by the unchanged "
             "weighted-mean file pooling; no learned parameters added; "
             "masking, S_pred aggregation, timestep localization, and metric "
             "code unchanged"),
}


def arm_grids(arm_id: str) -> tuple[tuple[int, int], ...]:
    """Return the frozen grid lattice for a registered C2 arm."""
    try:
        return ARM_GRIDS[arm_id]
    except KeyError:
        raise ValueError(f"unknown Sprint 17 C2 arm: {arm_id!r}") from None


def grid_starts(n_timesteps: int, offset: int, stride: int,
                patch_size: int = PATCH_SIZE) -> list[int]:
    """Start lattice for one fixed-width grid under the B0 tail rule.

    For ``offset == 0`` this replicates ``Patchifier`` with the same stride
    exactly (including the pad_end tail rule); non-zero offsets generalize
    the same rule to a shifted grid.
    """
    T = int(n_timesteps)
    W = int(patch_size)
    if T <= 0:
        return []
    if T >= W:
        starts = list(range(int(offset), T - W + 1, int(stride)))
        if not starts:
            # Offset lies beyond the last full window: one partial patch.
            if int(offset) < T:
                starts = [int(offset)]
        else:
            covered_end = starts[-1] + W
            if covered_end < T:
                starts.append(starts[-1] + int(stride))
        return starts
    # Short file: the grid contributes one padded patch iff it starts inside.
    return [int(offset)] if int(offset) < T else []


def unit_mass_weights(starts: npt.NDArray[np.int64],
                      valid_len: npt.NDArray[np.int64],
                      n_timesteps: int) -> npt.NDArray[np.float32]:
    """Per-patch weights giving every covered timestep total mass exactly 1.

    Weight of patch ``p`` is ``sum(1/k(t))`` over its valid timesteps, where
    ``k(t)`` is the number of valid patches covering ``t``.  Fail fast on
    uncovered timesteps: unit mass is provable only under full coverage,
    which pad_end grids provide.
    """
    T = int(n_timesteps)
    s = np.asarray(starts, dtype=np.int64).reshape(-1)
    vl = np.asarray(valid_len, dtype=np.int64).reshape(-1)
    if s.shape != vl.shape:
        raise ValueError("starts and valid_len must share one length")
    if T <= 0:
        raise ValueError("timestep count must be positive")
    if bool((s < 0).any()) or bool((vl < 0).any()):
        raise ValueError("starts and valid_len must be non-negative")
    if bool(((s + vl) > T).any()):
        raise ValueError("patch support must lie inside [0, T)")
    counts = np.zeros(T, dtype=np.int64)
    for start, length in zip(s.tolist(), vl.tolist()):
        counts[start:start + length] += 1
    if bool((counts == 0).any()):
        raise ValueError("uncovered timesteps have no unit-mass weight")
    inv = 1.0 / counts.astype(np.float64)
    weights = np.empty(len(s), dtype=np.float64)
    for i, (start, length) in enumerate(zip(s.tolist(), vl.tolist())):
        weights[i] = float(inv[start:start + length].sum())
    return weights.astype(np.float32)


class Sprint17C2Patchifier(Patchifier):
    """Multi-grid fixed-width patchifier preserving the PatchBatch contract.

    Each ``(offset, stride)`` grid follows :func:`grid_starts`; patches are
    emitted in ``(start, grid)`` order with unchanged ``PatchBatch`` fields
    (patches, starts, valid_len, pad_mask, file identity), so timestep to
    patch localization (``which_patches``, ``patch_to_timestep_scores``)
    keeps working without downstream changes.
    """

    def __init__(self, grids: tuple[tuple[int, int], ...],
                 patch_size: int = PATCH_SIZE,
                 pad_end: bool = PAD_END) -> None:
        if not grids:
            raise ValueError("at least one grid is required")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        for offset, stride in grids:
            if offset < 0 or stride <= 0:
                raise ValueError("grid offsets must be >= 0, strides positive")
        # cfg.stride records the primary grid stride; the authoritative
        # lattice is ``self.grids``.  Only cfg.patch_size is consumed
        # downstream (batch width).
        super().__init__(PatchConfig(patch_size=patch_size,
                                     stride=grids[0][1], pad_end=pad_end))
        self.grids = tuple((int(o), int(s)) for o, s in grids)

    def grid_ids_for(self, n_timesteps: int) -> npt.NDArray[np.int64]:
        """Grid index per emitted patch for a file length (provenance)."""
        order: list[int] = []
        for grid_index, (offset, stride) in enumerate(self.grids):
            order += [grid_index] * len(grid_starts(n_timesteps, offset,
                                                    stride, self.cfg.patch_size))
        starts_all: list[tuple[int, int]] = []
        for grid_index, (offset, stride) in enumerate(self.grids):
            for s in grid_starts(n_timesteps, offset, stride,
                                 self.cfg.patch_size):
                starts_all.append((s, grid_index))
        starts_all.sort(key=lambda item: (item[0], item[1]))
        return np.array([g for _, g in starts_all], dtype=np.int64)

    def patchify(self, sample: FileSample) -> PatchBatch:
        """Slice into the multi-grid lattice, preserving B0 slice semantics."""
        x = sample.x
        C, T = x.shape
        W = self.cfg.patch_size
        plan: list[tuple[int, int]] = []
        for grid_index, (offset, stride) in enumerate(self.grids):
            for s in grid_starts(T, offset, stride, W):
                plan.append((s, grid_index))
        plan.sort(key=lambda item: (item[0], item[1]))

        patches_list: list[npt.NDArray[np.float32]] = []
        starts_out: list[int] = []
        valid_lens: list[int] = []
        pad_masks: list[npt.NDArray[np.bool_]] = []
        for s, _ in plan:
            real_len = min(W, T - s)
            if real_len <= 0:
                continue
            patch = np.full((C, W), self.cfg.pad_value, dtype=np.float32)
            patch[:, :real_len] = x[:, s:s + real_len]
            pm = np.zeros(W, dtype=bool)
            pm[real_len:] = True
            if real_len < W and not self.cfg.pad_end:
                continue
            patches_list.append(patch)
            starts_out.append(s)
            valid_lens.append(real_len)
            pad_masks.append(pm)

        if not patches_list:
            if self.cfg.pad_end:
                patch = np.full((C, W), self.cfg.pad_value, dtype=np.float32)
                patch[:, :T] = x[:, :T]
                pm = np.ones(W, dtype=bool)
                pm[:T] = False
                patches_list = [patch]
                starts_out = [0]
                valid_lens = [T]
                pad_masks = [pm]
            else:
                return PatchBatch(
                    patches=np.empty((0, C, W), dtype=np.float32),
                    starts=np.empty(0, dtype=np.int64),
                    valid_len=np.empty(0, dtype=np.int64),
                    pad_mask=np.empty((0, W), dtype=bool),
                    file_sample=sample,
                )
        return PatchBatch(
            patches=np.stack(patches_list, axis=0),
            starts=np.array(starts_out, dtype=np.int64),
            valid_len=np.array(valid_lens, dtype=np.int64),
            pad_mask=np.stack(pad_masks, axis=0),
            file_sample=sample,
        )

    def support_weights_for(self, starts: npt.NDArray[np.int64],
                            valid_len: npt.NDArray[np.int64],
                            n_timesteps: int) -> npt.NDArray[np.float32] | None:
        """Collate hook: unit-mass weights for multi-grid arms, else None.

        Single-grid arms return ``None`` so collation omits the weights key
        and the downstream pooling path is literally the B0 path.
        """
        if len(self.grids) == 1:
            return None
        return unit_mass_weights(starts, valid_len, n_timesteps)


def arm_patchifier(arm_id: str) -> Sprint17C2Patchifier:
    """Build the frozen patchifier for a registered C2 arm."""
    return Sprint17C2Patchifier(arm_grids(arm_id))


def pad_support_weights(patchifier: object,
                        patch_batches: list[PatchBatch],
                        max_patches: int) -> torch.Tensor | None:
    """Zero-padded float32 support weights, or None when the arm emits none."""
    hook = getattr(patchifier, "support_weights_for", None)
    if hook is None:
        return None
    rows: list[npt.NDArray[np.float32] | None] = [
        hook(pb.starts, pb.valid_len, int(pb.file_sample.T))
        for pb in patch_batches
    ]
    if all(r is None for r in rows):
        return None
    out = torch.zeros((len(patch_batches), max(1, int(max_patches))),
                      dtype=torch.float32)
    for i, (pb, row) in enumerate(zip(patch_batches, rows)):
        if row is None:
            continue
        w = np.asarray(row, dtype=np.float32).reshape(-1)
        if w.shape[0] != pb.N:
            raise ValueError("support weights must match the patch count")
        out[i, :w.shape[0]] = torch.from_numpy(w)
    return out
