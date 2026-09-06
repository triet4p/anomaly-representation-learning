"""Batch C2 Task 22: staged runner, configs, and provenance contracts.

Covers the 2/5/50 epoch budgets, matched control/hybrid variants, the
predeclared 5-epoch coefficient matrix, commit-placeholder resolution,
sealed-test fail-fast behavior, provenance schema completeness, and one
tiny live contract-cell run on a materialized client profile (also
exercised headlessly by the Task 23 smoke; no quality claim here).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from representation.v2_staged import (
    CONTROL_VARIANT,
    HYBRID_VARIANT,
    STAGE_EPOCHS,
    StageSpec,
    assert_no_test_files,
    balance_matrix,
    control_coefficients,
    geometry_health,
    hybrid_coefficients,
    iter_cells,
    load_stage_config,
    resolve_commit,
    run_stage_cell,
    stage_epochs,
)
from representation.v2_trainer import build_v2_training_stack
from representation.v2_config import V2Config
from synth.chronicle import client_config, materialize_chronological

STAGED_DIR = Path("experiments/v2_staged")


def test_stage_epoch_budgets_are_frozen() -> None:
    assert STAGE_EPOCHS == {"contract": 2, "balance": 5, "full": 50}
    assert stage_epochs("contract") == 2
    assert stage_epochs("balance") == 5
    assert stage_epochs("full") == 50
    with pytest.raises(ValueError):
        stage_epochs("nightly")


def test_control_and_hybrid_variants_match_except_boundary() -> None:
    control = control_coefficients()
    hybrid = hybrid_coefficients()
    assert control["boundary_alpha_max"] == 0.0
    assert hybrid["boundary_alpha_max"] > 0.0
    for key in ("background_weight", "variance_weight", "covariance_weight",
                "boundary_margin"):
        assert control[key] == hybrid[key], f"matched pair differs on {key}"
    StageSpec(stage="contract", variant=CONTROL_VARIANT, epochs=2,
              coefficients=control).validate()
    StageSpec(stage="contract", variant=HYBRID_VARIANT, epochs=2,
              coefficients=hybrid).validate()
    with pytest.raises(ValueError):
        StageSpec(stage="contract", variant=CONTROL_VARIANT, epochs=2,
                  coefficients=hybrid).validate()
    with pytest.raises(ValueError):
        StageSpec(stage="full", variant=HYBRID_VARIANT, epochs=7,
                  coefficients=hybrid).validate()


def test_balance_matrix_is_predeclared_with_control_reference() -> None:
    matrix = balance_matrix()
    assert [cell["cell"] for cell in matrix] == [
        "balance-a", "balance-b", "balance-c", "balance-control",
    ]
    variants = {cell["cell"]: cell["coefficients"] for cell in matrix}
    assert float(variants["balance-b"]["boundary_alpha_max"]) == 1.0
    assert float(variants["balance-control"]["boundary_alpha_max"]) == 0.0


def test_committed_configs_match_presets_and_variants() -> None:
    assert STAGED_DIR.is_dir(), "staged configs must be committed source"
    expected = {
        "contract.yaml": ("contract", 2, 2),
        "balance.yaml": ("balance", 5, 4),
        "full_control.yaml": ("full", 50, 1),
        "full_hybrid.yaml": ("full", 50, 1),
    }
    for name, (stage, epochs, n_cells) in expected.items():
        config = load_stage_config(STAGED_DIR / name)
        assert config["stage"] == stage
        assert config["epochs"] == epochs
        assert config["commit"] == "{COMMIT}", f"{name} must keep the runtime placeholder"
        cells = iter_cells(config)
        assert len(cells) == n_cells, f"{name} must declare {n_cells} cells"
    with pytest.raises(FileNotFoundError):
        load_stage_config(STAGED_DIR / "missing.yaml")


def test_commit_placeholder_resolves_at_runtime() -> None:
    info = resolve_commit("{COMMIT}")
    assert set(info) == {"commit", "commit_source", "dirty"}
    assert info["commit_source"] in ("git-head", "unresolved")
    if info["commit_source"] == "git-head":
        assert len(str(info["commit"])) == 40
    explicit = resolve_commit("a" * 40)
    assert explicit == {"commit": "a" * 40, "commit_source": "explicit", "dirty": False}
    with pytest.raises(ValueError):
        resolve_commit("not-a-sha")


def test_sealed_test_violation_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="sealed-test violation"):
        assert_no_test_files(["f1", "f2"], {"f2", "f9"}, context="unit check")
    assert_no_test_files(["f1"], {"f2", "f9"}, context="unit check")


def test_geometry_health_reports_conditioning() -> None:
    config = V2Config(n_channels=6, d_model=4, n_robots=1, n_programs=1,
                      n_regimes=2, min_group_samples=2, diag_min_samples=2)
    torch.manual_seed(0)
    latents = torch.randn(12, 4)
    _, _, trainer = build_v2_training_stack(config, device="cpu", seed=0)
    geometry = trainer.fit_geometry(
        latents, torch.zeros(12, dtype=torch.long), torch.zeros(12, dtype=torch.long),
        torch.zeros(12, dtype=torch.long), torch.ones(12, dtype=torch.bool),
    )
    health = geometry_health(geometry)
    assert health["n_references"] >= 3
    assert health["all_finite_condition"] is True
    assert health["worst_condition_number"] < 1e12


def test_tiny_contract_control_cell_runs_with_exact_provenance(tmp_path) -> None:
    data_root = tmp_path / "chronicle"
    manifest = materialize_chronological(client_config(seed=0), data_root)
    spec = StageSpec(
        stage="contract", variant=CONTROL_VARIANT, epochs=2, seed=0,
        d_model=8, batch_size=8, coefficients=control_coefficients(),
    )
    provenance = run_stage_cell(
        spec, data_root=data_root, output_root=tmp_path / "runs", device="cpu",
    )
    assert provenance["stage"] == "contract"
    assert provenance["epochs"] == 2
    assert provenance["variant"] == CONTROL_VARIANT
    assert provenance["config_hash"] == manifest["config_hash"]
    assert len(provenance["shard_digests"]) == len(manifest["shards"])
    assert len(provenance["history"]) == 2
    for point in provenance["history"]:
        for key in ("loss", "normal_loss", "background_loss", "boundary_loss",
                    "alpha", "grad_norm_mean", "train_stationary", "val_stationary"):
            assert point[key] == point[key], f"non-finite history field {key}"
    assert provenance["coefficients"]["boundary_alpha_max"] == 0.0
    assert provenance["sealed_test"]["forbidden_file_count"] > 0
    assert Path(provenance["checkpoint_path"]).is_file()
    assert (Path(provenance["output_root"]) / "provenance.json").is_file()
    assert provenance["geometry_health"]["all_finite_condition"] is True
    assert provenance["wall_time_s"] > 0.0
