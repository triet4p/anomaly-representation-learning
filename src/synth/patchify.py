"""
Patchification for variable-length files (DATA.md §6).

Input:  X ∈ ℝ[C, T]   (variable T)
Output: P ∈ ℝ[N, C, W]  with per-patch start indices, valid lengths,
        padding masks, and an exact timestep ↔ patch anomaly-mask map.

Design:
- Configurable patch size W and stride S (stride < W = overlapping).
- Last partial patch: optionally zero-padded to full W with pad_mask True.
- Every patch records its start timestep so downstream code can map
  patch scores back to exact timestep positions.
- Timestep-to-patch mapping: ``which_patches(t)`` returns the set of patch
  indices that contain timestep t (useful for overlap aggregation).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synth.config import PatchConfig
from synth.schema import FileSample, PatchBatch


class Patchifier:
    """
    Converts a FileSample into a PatchBatch.

    Args:
        config: PatchConfig with patch_size, stride, pad_end, pad_value.
    """

    def __init__(self, config: PatchConfig) -> None:
        if config.patch_size <= 0 or config.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        self.cfg = config

    def patchify(self, sample: FileSample) -> PatchBatch:
        """
        Slice sample.x into overlapping patches.

        Returns PatchBatch with:
          patches  [N, C, W]   float32
          starts   [N]         int64 — timestep index of patch start
          valid_len[N]         int64 — real steps in patch (< W if padded)
          pad_mask [N, W]      bool  — True = padded position
        """
        x = sample.x  # float32 [C, T]
        C, T = x.shape
        W = self.cfg.patch_size
        S = self.cfg.stride

        # Emit all full windows first.  A tail is appended only when the last
        # full window leaves real timesteps uncovered; this avoids a redundant
        # padded patch for T=64, W=32, S=16 (last full start is 32).
        if T >= W:
            starts = list(range(0, T - W + 1, S))
            covered_end = starts[-1] + W if starts else 0
            if self.cfg.pad_end and covered_end < T:
                starts.append(covered_end if not starts else starts[-1] + S)
        elif self.cfg.pad_end:
            starts = [0]
        else:
            starts = []

        patches_list = []
        starts_out = []
        valid_lens = []
        pad_masks = []

        for s in starts:
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

        patches = np.stack(patches_list, axis=0)
        starts_arr = np.array(starts_out, dtype=np.int64)
        valid_arr = np.array(valid_lens, dtype=np.int64)
        pad_arr = np.stack(pad_masks, axis=0)

        return PatchBatch(
            patches=patches,
            starts=starts_arr,
            valid_len=valid_arr,
            pad_mask=pad_arr,
            file_sample=sample,
        )

    # ------------------------------------------------------------------ #
    # Utility: timestep ↔ patch mapping                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def which_patches(
        timestep: int,
        starts: npt.NDArray[np.int64],
        valid_len: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        Return indices of patches that contain the given timestep.
        A patch i contains timestep t iff starts[i] <= t < starts[i] + valid_len[i].
        """
        lo = starts
        hi = starts + valid_len
        mask = (lo <= timestep) & (timestep < hi)
        return np.where(mask)[0].astype(np.int64)

    @staticmethod
    def timestep_mask_to_patch_mask(
        ts_mask: npt.NDArray[np.bool_],      # [C, T]
        starts: npt.NDArray[np.int64],       # [N]
        valid_len: npt.NDArray[np.int64],    # [N]
        C: int,
        T: int,
    ) -> npt.NDArray[np.bool_]:             # [N]
        """
        For each patch, True if it contains ≥1 anomalous (True) timestep
        in any channel.
        """
        N = len(starts)
        patch_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            s = int(starts[i])
            e = s + int(valid_len[i])
            e = min(e, T)
            if s >= T:
                continue
            if ts_mask[:, s:e].any():
                patch_mask[i] = True
        return patch_mask

    @staticmethod
    def patch_to_timestep_scores(
        patch_scores: npt.NDArray[np.float64],  # [N]
        starts: npt.NDArray[np.int64],          # [N]
        valid_len: npt.NDArray[np.int64],       # [N]
        T: int,
    ) -> npt.NDArray[np.float64]:               # [T] — aggregated by mean
        """
        Aggregate patch-level scores back to timestep-level by averaging
        all patch scores that cover each timestep.
        """
        scores = np.zeros(T, dtype=np.float64)
        counts = np.zeros(T, dtype=np.int64)
        for i, (s, vl) in enumerate(zip(starts, valid_len)):
            e = min(int(s) + int(vl), T)
            scores[int(s):e] += patch_scores[i]
            counts[int(s):e] += 1
        nonzero = counts > 0
        scores[nonzero] /= counts[nonzero]
        return scores
