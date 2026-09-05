"""Conditional sequence batch normalization with hierarchical fleet baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
import torch
from torch import nn


class _BucketStatistics(nn.Module):
    value_sum: torch.Tensor
    value_square_sum: torch.Tensor
    value_count: torch.Tensor
    sample_count: torch.Tensor

    """Persistent sufficient statistics for one conditional bucket level."""

    def __init__(self, num_buckets: int, num_sensors: int) -> None:
        super().__init__()
        self.register_buffer(
            "value_sum", torch.zeros(num_buckets, num_sensors, dtype=torch.float64)
        )
        self.register_buffer(
            "value_square_sum",
            torch.zeros(num_buckets, num_sensors, dtype=torch.float64),
        )
        self.register_buffer(
            "value_count",
            torch.zeros(num_buckets, num_sensors, dtype=torch.float64),
        )
        self.register_buffer(
            "sample_count", torch.zeros(num_buckets, dtype=torch.long)
        )


class ConditionalBatchNorm(nn.Module):
    """Normalize multi-sensor sequences against an optionally conditional fleet baseline.

    Conditions are ordered from specific to broad:
    e.g. ("robot_idx", "program_idx") -> fallback: robot+program -> robot -> fleet.
    A bucket is selected only after observing min_bucket_samples normal sequences;
    otherwise the next broader fallback level is used.

    Contract:
    - Input: dict[str, torch.Tensor] containing required "input" [B, C, T]
      and optional "valid_mask" [B, T], "robot_idx" [B], "program_idx" [B].
    - Output: dict[str, torch.Tensor] containing "output" [B, C, T],
      "fallback_level" [B], "bucket_id" [B], "mean" [B, C], "std" [B, C].
    """

    def __init__(
        self,
        num_sensors: int,
        condition_cardinalities: Mapping[str, int] | None = None,
        min_bucket_samples: int = 32,
        eps: float = 1e-5,
        fallback_condition_sets: Sequence[Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        if num_sensors <= 0:
            raise ValueError("num_sensors must be positive")
        if min_bucket_samples <= 0:
            raise ValueError("min_bucket_samples must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        cardinalities = dict(condition_cardinalities or {})
        if any(not name for name in cardinalities):
            raise ValueError("condition names must be non-empty")
        if any(size <= 0 for size in cardinalities.values()):
            raise ValueError("condition cardinalities must be positive")

        self.num_sensors = num_sensors
        self.condition_names = tuple(cardinalities)
        self.condition_cardinalities = cardinalities
        self.min_bucket_samples = min_bucket_samples
        self.eps = eps

        if fallback_condition_sets is None:
            self.fallback_condition_sets = tuple(
                self.condition_names[:size]
                for size in range(len(self.condition_names), -1, -1)
            )
        else:
            sets = tuple(tuple(s) for s in fallback_condition_sets)
            if not sets or sets[-1] != ():
                raise ValueError("fallback_condition_sets must end with the fleet bucket ()")
            self.fallback_condition_sets = sets

        self._statistics = nn.ModuleList(
            _BucketStatistics(self._num_buckets(condition_set), num_sensors)
            for condition_set in self.fallback_condition_sets
        )

    def _num_buckets(self, condition_set: tuple[str, ...]) -> int:
        total = 1
        for name in condition_set:
            total *= self.condition_cardinalities[name]
        return total

    def _bucket_ids(
        self,
        inputs: Mapping[str, torch.Tensor],
        batch_size: int,
        condition_set: tuple[str, ...],
    ) -> torch.Tensor:
        input_tensor = inputs["input"]
        device = input_tensor.device
        if not condition_set:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        bucket_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        for name in condition_set:
            values = inputs.get(name)
            if values is None:
                values = torch.zeros(batch_size, dtype=torch.long, device=device)
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(
                    f"Condition {name!r} must have shape [{batch_size}], got {tuple(values.shape)}"
                )
            if values.dtype.is_floating_point:
                raise ValueError(f"Condition {name!r} must contain integer category IDs")
            values = values.to(device=device, dtype=torch.long)
            cardinality = self.condition_cardinalities[name]
            if (values < 0).any() or (values >= cardinality).any():
                raise ValueError(
                    f"Condition {name!r} contains IDs outside [0, {cardinality})"
                )
            bucket_ids = bucket_ids * cardinality + values
        return bucket_ids
    def _get_stats(self, level: int) -> _BucketStatistics:
        return cast(_BucketStatistics, self._statistics[level])


    @torch.no_grad()
    def _update_statistics(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        bucket_ids: Sequence[torch.Tensor],
    ) -> None:
        mask_3d = valid_mask.unsqueeze(1).expand_as(x).to(dtype=torch.float64)
        stats_x = x.to(dtype=torch.float64) * mask_3d
        per_sample_sum = stats_x.sum(dim=-1)
        per_sample_square_sum = (stats_x**2).sum(dim=-1)
        per_sample_counts = mask_3d.sum(dim=-1)
        sample_increments = torch.ones(x.shape[0], dtype=torch.long, device=x.device)

        for level, buckets in enumerate(bucket_ids):
            stats = self._get_stats(level)
            # Co-locate the small per-batch index/sources with the persistent
            # accumulators. `.to()` is a no-op when already aligned (no copy).
            device = stats.value_sum.device
            level_buckets = buckets.to(device)
            stats.value_sum = torch.index_add(stats.value_sum, 0, level_buckets, per_sample_sum.to(device))
            stats.value_square_sum = torch.index_add(stats.value_square_sum, 0, level_buckets, per_sample_square_sum.to(device))
            stats.value_count = torch.index_add(stats.value_count, 0, level_buckets, per_sample_counts.to(device))
            stats.sample_count = torch.index_add(stats.sample_count, 0, level_buckets, sample_increments.to(device))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize sequence against selected conditional bucket level.

        Args:
            inputs: Dictionary containing:
                - "input": Tensor [B, C, T] to normalize.
                - "valid_mask": Optional boolean Tensor [B, T].
                - Condition tensors (e.g. "robot_idx", "program_idx"): LongTensor [B].

        Returns:
            Dictionary containing:
                - "output": Normalized Tensor [B, C, T].
                - "fallback_level": LongTensor [B] reporting selected fallback hierarchy.
                - "bucket_id": LongTensor [B] reporting category index.
                - "mean": Tensor [B, C] baseline mean applied.
                - "std": Tensor [B, C] baseline std applied.
        """
        x = inputs.get("input")
        if x is None:
            raise ValueError("ConditionalBatchNorm requires an 'input' tensor")
        if x.ndim != 3 or x.shape[1] != self.num_sensors:
            raise ValueError(
                f"Expected input [B, {self.num_sensors}, T], got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("input must be a floating-point tensor")

        valid_mask = inputs.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(
                (x.shape[0], x.shape[2]), dtype=torch.bool, device=x.device
            )

        bucket_ids = [
            self._bucket_ids(inputs, x.shape[0], c_set)
            for c_set in self.fallback_condition_sets
        ]

        if self.training:
            self._update_statistics(x.detach(), valid_mask, bucket_ids)

        # Select most specific level that has observed >= min_bucket_samples
        selected_level = torch.full(
            (x.shape[0],),
            len(self._statistics) - 1,
            dtype=torch.long,
            device=x.device,
        )
        for level in range(len(self._statistics) - 1, -1, -1):
            stats = self._get_stats(level)
            buckets = bucket_ids[level]
            seen = stats.sample_count[buckets.to(stats.sample_count.device)]
            sufficient = seen.to(selected_level.device) >= self.min_bucket_samples
            selected_level = torch.where(
                sufficient,
                torch.full_like(selected_level, level),
                selected_level,
            )

        means = torch.empty(x.shape[0], self.num_sensors, device=x.device, dtype=x.dtype)
        variances = torch.empty_like(means)
        for level, buckets in enumerate(bucket_ids):
            stats = self._get_stats(level)
            selected = selected_level == level
            if not selected.any():
                continue
            level_buckets = buckets[selected].to(stats.value_count.device)
            counts = stats.value_count[level_buckets].clamp_min(1.0)
            mean = stats.value_sum[level_buckets] / counts
            variance = stats.value_square_sum[level_buckets] / counts - mean.square()
            means[selected] = mean.to(device=means.device, dtype=means.dtype)
            variances[selected] = variance.clamp_min(self.eps).to(device=variances.device, dtype=variances.dtype)

        std = variances.sqrt()
        normalized = (x - means.unsqueeze(-1)) / std.unsqueeze(-1)
        normalized = normalized.masked_fill(~valid_mask.unsqueeze(1), 0.0)

        return {
            "output": normalized,
            "fallback_level": selected_level,
            "bucket_id": bucket_ids[0],
            "mean": means,
            "std": std,
        }
