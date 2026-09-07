"""Batch B2 Finding 1: boundary-trained context energy stays the monitor.

Proves production inference preserves the exact encoder ``context_energy``
optimized by the boundary loss, returns hierarchical ``population_energy``
as a separately named signal, and aggregates each under an explicit
``energy_source`` label — so a deliberate score disagreement can never be
silently conflated.
"""

from __future__ import annotations

import torch
from torch import nn

from representation.v2_aggregation import aggregate_file_state
from representation.v2_config import V2Config
from representation.v2_contracts import (
    CONTEXT_ENERGY_FIELD,
    POPULATION_ENERGY_FIELD,
    validate_context_patch_output,
    validate_population_patch_output,
)
from representation.v2_inference import V2InferencePipeline


class _StubGeometry:
    """Hierarchical scorer returning a fixed population energy."""

    def __init__(self, energy: torch.Tensor) -> None:
        self._energy = energy

    def mixture_energy(
        self,
        latents: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"population_energy": self._energy}


class _StubEncoder(nn.Module):
    """Canned boundary-trained score behind the encoder call signature."""

    def __init__(self, latents: torch.Tensor, energy: torch.Tensor) -> None:
        super().__init__()
        self.latents = latents
        self.energy = energy
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        patches: torch.Tensor,
        patch_pad_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
        regime_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"patch_latents": self.latents, "context_energy": self.energy}


def _inputs(batches: int = 2, patches: int = 8):
    return {
        "patches": torch.zeros(batches, patches, 6, 8),
        "patch_pad_mask": torch.zeros(batches, patches, 8, dtype=torch.bool),
        "patch_valid_mask": torch.ones(batches, patches, dtype=torch.bool),
        "robot_idx": torch.zeros(batches, dtype=torch.long),
        "program_idx": torch.zeros(batches, dtype=torch.long),
        "regime_ids": torch.zeros(batches, patches, dtype=torch.long),
    }


def _config() -> V2Config:
    return V2Config(
        n_channels=6, d_model=4, n_robots=1, n_programs=1, n_regimes=2,
        min_group_samples=2, diag_min_samples=2, top_q_fraction=0.25,
        elevated_threshold=3.0, calibration_min_samples=4,
    )


def test_inference_preserves_context_energy_and_names_population() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 8, 4)
    context = torch.randn(2, 8) - 2.0  # calm boundary-trained score
    population = torch.full((2, 8), 9.0)  # disagreeing hierarchical score
    pipeline = V2InferencePipeline(
        _config(), _StubEncoder(latents, context), _StubGeometry(population),  # type: ignore[arg-type]
    )
    out = pipeline.score_patches(**_inputs())  # type: ignore[arg-type]
    # The monitoring score is the encoder output, bit-for-bit.
    torch.testing.assert_close(out["patch"]["context_energy"], context)
    validate_context_patch_output(out["patch"])
    assert POPULATION_ENERGY_FIELD not in out["patch"]
    # Population energy keeps its own name and never overwrites context.
    torch.testing.assert_close(out["population"]["population_energy"], population)
    validate_population_patch_output(out["population"])
    assert CONTEXT_ENERGY_FIELD not in out["population"]
    # Each file state names the signal it consumed.
    assert out["file"]["energy_source"] == CONTEXT_ENERGY_FIELD
    assert out["file_population"]["energy_source"] == POPULATION_ENERGY_FIELD


def test_disagreeing_scores_cannot_be_silently_conflated() -> None:
    torch.manual_seed(1)
    latents = torch.randn(2, 8, 4)
    context = torch.zeros(2, 8)  # nothing elevated per the monitor
    population = torch.full((2, 8), 9.0)  # everything elevated per hierarchy
    pipeline = V2InferencePipeline(
        _config(), _StubEncoder(latents, context), _StubGeometry(population),  # type: ignore[arg-type]
    )
    args = _inputs()
    out = pipeline.score_patches(**args)  # type: ignore[arg-type]
    valid = args["patch_valid_mask"]
    regimes = args["regime_ids"]
    expected_file = aggregate_file_state(
        latents, context, valid, regimes, energy_source="context_energy",
        top_q_fraction=0.25, elevated_threshold=3.0, n_regimes=2,
    )
    expected_population = aggregate_file_state(
        latents, population, valid, regimes, energy_source="population_energy",
        top_q_fraction=0.25, elevated_threshold=3.0, n_regimes=2,
    )
    torch.testing.assert_close(out["file"]["tail_energy"], expected_file["tail_energy"])
    torch.testing.assert_close(
        out["file_population"]["tail_energy"], expected_population["tail_energy"]
    )
    # The disagreement is explicit: the context chain stays calm while the
    # population view reports full elevation.
    assert float(out["file"]["elevated_fraction"][0]) == 0.0
    assert float(out["file_population"]["elevated_fraction"][0]) == 1.0
