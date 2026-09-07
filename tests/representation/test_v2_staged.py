"""Batch C2 Task 22: staged runner, configs, and provenance contracts.

Covers the 2/5/50 epoch budgets, matched control/hybrid variants, the
predeclared 5-epoch coefficient matrix, commit-placeholder resolution,
sealed-test fail-fast behavior, complete signed-density objective
diagnostics in every history, dev-val-only threshold/calibrator fits with
distinct provenances (no pooled fallback, no fixed-default runtime use),
explicit context-energy identity, the explicit small-sample floor rule,
superseded-selection non-reuse, provenance schema completeness, and tiny
live contract/balance cell runs on a materialized client profile (also
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
    STAGE_BOUNDARY_SCHEDULE,
    STAGE_EPOCHS,
    StageSpec,
    apply_balance_selection,
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
from representation.criterion import ProgressiveLambda
from representation.v2_trainer import build_v2_training_stack
from representation.v2_config import V2Config
from representation.v2_inference import V2InferencePipeline
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
    # Explicit small-sample floor: the 7-file tiny dev-val clears 4 and is
    # disclosed small-sample below the 32 reference floor (Finding 3 rule).
    spec = StageSpec(
        stage="contract", variant=CONTROL_VARIANT, epochs=2, seed=0,
        d_model=8, batch_size=8, coefficients=control_coefficients(),
        calibration_min_samples=4,
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
        # Complete signed-density objective diagnostics flow through the
        # stage: clean conditional-density NLL plus variance/covariance,
        # background, and ramped boundary terms (never squared selection).
        for key in ("loss", "normal_loss", "density_raw",
                    "variance_raw", "covariance_raw",
                    "background_loss", "boundary_loss",
                    "alpha", "grad_norm_mean", "train_stationary", "val_stationary"):
            assert point[key] == point[key], f"non-finite history field {key}"
    assert "signed-likelihood" in provenance["stationary_objective"]
    assert provenance["coefficients"]["boundary_alpha_max"] == 0.0
    assert provenance["sealed_test"]["forbidden_file_count"] > 0
    assert Path(provenance["checkpoint_path"]).is_file()
    assert (Path(provenance["output_root"]) / "provenance.json").is_file()
    assert provenance["geometry_health"]["all_finite_condition"] is True
    assert provenance["wall_time_s"] > 0.0
    # Distinct dev-val-only provenances: operating threshold from dev-val
    # context energies, conformal fit on dev-val displacements, no shared
    # record and no pooled fallback anywhere in the manifest.
    assert "calibration_source" not in provenance
    operating = provenance["operating_threshold"]
    assert operating["fit_cohort"] == "dev-val"
    assert operating["energy_field"] == "context_energy"
    assert operating["method"] == "healthy-validation-quantile"
    cohort = provenance["confidence_calibrator_fit_cohort"]
    assert cohort["cohort"] == "dev-val"
    assert cohort["small_sample"] is True
    assert provenance["calibration_min_samples"] == 4
    assert "pooled" not in json.dumps(provenance, default=str)
    # The calibrated (non-default) cutoff restores by default end to end.
    pipeline = V2InferencePipeline.load(provenance["checkpoint_path"], device="cpu")
    assert pipeline.elevated_threshold == pytest.approx(float(operating["value"]))
    assert "dev-val" in pipeline.elevated_threshold_source


def test_balance_stage_schedule_is_predeclared_and_stage_specific() -> None:
    """Task 28: every varied coefficient must be active inside 5 epochs."""
    assert STAGE_BOUNDARY_SCHEDULE == {
        "contract": (500, 2_000),
        "balance": (5, 10),
        "full": (50, 100),
    }
    balance = StageSpec(
        stage="balance", variant=HYBRID_VARIANT, epochs=5,
        coefficients=hybrid_coefficients(),
    ).validate()
    assert (balance.boundary_warmup_steps, balance.boundary_ramp_steps) == (5, 10)
    contract = StageSpec(
        stage="contract", variant=CONTROL_VARIANT, epochs=2,
        coefficients=control_coefficients(),
    ).validate()
    explicit = StageSpec(
        stage="balance", variant=HYBRID_VARIANT, epochs=5,
        coefficients=hybrid_coefficients(),
        boundary_warmup_steps=0, boundary_ramp_steps=2,
    ).validate()
    assert (explicit.boundary_warmup_steps, explicit.boundary_ramp_steps) == (0, 2)
    with pytest.raises(ValueError):
        StageSpec(
            stage="balance", variant=HYBRID_VARIANT, epochs=5,
            coefficients=hybrid_coefficients(), boundary_warmup_steps=-1,
        ).validate()
    with pytest.raises(ValueError):
        StageSpec(
            stage="balance", variant=HYBRID_VARIANT, epochs=5,
            coefficients=hybrid_coefficients(), boundary_ramp_steps=True,
        ).validate()

def test_balance_yaml_predeclares_budget_active_schedule() -> None:
    """The committed matrix must engage alpha inside the ~30-step budget."""
    config = load_stage_config(STAGED_DIR / "balance.yaml")
    cells = iter_cells(config)
    assert len(cells) == 4
    for spec in cells:
        assert (spec.boundary_warmup_steps, spec.boundary_ramp_steps) == (5, 10)
    # Task 27 measured 12 steps over 2 epochs at batch 8; 5 epochs ~= 30 steps.
    schedule = ProgressiveLambda(lambda_max=1.0, ramp_steps=10, warmup_steps=5)
    alphas = [schedule.lambda_at(step) for step in range(30)]
    assert sum(alpha > 0.0 for alpha in alphas) >= 24
    assert schedule.lambda_at(15) == pytest.approx(1.0)
    assert schedule.lambda_at(29) == pytest.approx(1.0)
    for spec in cells:
        if spec.variant == HYBRID_VARIANT:
            assert float(spec.coefficients["boundary_alpha_max"]) > 0.0

def test_tiny_balance_hybrid_cell_activates_boundary(tmp_path) -> None:
    """Explicit fast schedule must drive alpha to its max with raw terms logged."""
    data_root = tmp_path / "chronicle"
    materialize_chronological(client_config(seed=0), data_root)
    spec = StageSpec(
        stage="balance", variant=HYBRID_VARIANT, epochs=5, seed=0,
        d_model=8, batch_size=8, coefficients=hybrid_coefficients(),
        boundary_warmup_steps=0, boundary_ramp_steps=2,
        calibration_min_samples=4,
    )
    provenance = run_stage_cell(
        spec, data_root=data_root, output_root=tmp_path / "runs", device="cpu",
    )
    assert provenance["boundary_schedule"] == {
        "warmup_steps": 0, "ramp_steps": 2, "alpha_max": 1.0,
    }
    assert len(provenance["history"]) == 5
    for point in provenance["history"]:
        for key in ("loss", "normal_loss", "density_raw",
                    "variance_raw", "covariance_raw",
                    "background_loss", "boundary_loss", "alpha",
                    "grad_norm_mean", "train_stationary", "val_stationary"):
            assert point[key] == point[key], f"non-finite history field {key}"
    assert "signed-likelihood" in provenance["stationary_objective"]
    assert provenance["history"][0]["alpha"] < 1.0
    assert provenance["history"][-1]["alpha"] == pytest.approx(1.0)
    assert provenance["final_step"] >= 2
    assert "calibration_source" not in provenance
    assert provenance["operating_threshold"]["fit_cohort"] == "dev-val"
    assert provenance["operating_threshold"]["energy_field"] == "context_energy"
    assert provenance["confidence_calibrator_fit_cohort"]["cohort"] == "dev-val"
    assert "pooled" not in json.dumps(provenance, default=str)


def _selection_cell(cell: str, variant: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "cell": cell,
        "variant": variant,
        "boundary_alpha_max": 1.0 if variant == HYBRID_VARIANT else 0.0,
        "final_alpha": 1.0 if variant == HYBRID_VARIANT else 0.0,
        "finite_all": True,
        "best_stationary": 100.0,
        "val_stationary_final": 100.0,
        "worst_condition_number": 500.0,
        "all_finite_condition": True,
        "effective_rank": 8.0,
        "clean_corrupt_gap": 0.5,
        "ordering_monotone": True,
        "background_mse": 0.05,
        "latent_scale": 1.0,
    }
    base.update(overrides)
    return base


def test_apply_balance_selection_ranks_gap_then_leak() -> None:
    cells = [
        _selection_cell("balance-control", CONTROL_VARIANT),
        _selection_cell("balance-a", HYBRID_VARIANT, clean_corrupt_gap=0.3),
        _selection_cell("balance-b", HYBRID_VARIANT, clean_corrupt_gap=0.7,
                        background_mse=0.05),
        _selection_cell("balance-c", HYBRID_VARIANT, clean_corrupt_gap=0.7,
                        background_mse=0.09),
    ]
    record = apply_balance_selection(cells)
    assert record["winner"] == "balance-b"
    assert record["insufficient_evidence"] is False
    assert record["ranked"] == ["balance-b", "balance-c", "balance-a"]
    rejected = {row["cell"]: row["reason"] for row in record["rejected"]}
    assert set(rejected) == {"balance-control", "balance-a", "balance-c"}
    assert rejected["balance-control"] == (
        "control reference (not eligible for hybrid selection)"
    )
    assert "ranked below balance-b" in rejected["balance-a"]


def test_apply_balance_selection_fails_honestly_without_passing_hybrid() -> None:
    cells = [
        _selection_cell("balance-control", CONTROL_VARIANT),
        _selection_cell("balance-a", HYBRID_VARIANT, clean_corrupt_gap=-0.1),
        _selection_cell("balance-b", HYBRID_VARIANT, ordering_monotone=False),
    ]
    record = apply_balance_selection(cells)
    assert record["winner"] is None
    assert record["insufficient_evidence"] is True
    assert "no hybrid cell passed" in record["reason"]
    assert any("separation" in failure for failure in record["evaluated"][1]["failures"])
    assert any("ordering" in failure for failure in record["evaluated"][2]["failures"])


def test_apply_balance_selection_invalid_without_control_reference() -> None:
    cells = [
        _selection_cell("balance-control", CONTROL_VARIANT,
                        all_finite_condition=False),
        _selection_cell("balance-b", HYBRID_VARIANT),
    ]
    record = apply_balance_selection(cells)
    assert record["winner"] is None
    assert record["insufficient_evidence"] is True
    assert "control reference" in record["reason"]
    with pytest.raises(ValueError, match="missing"):
        apply_balance_selection([{"cell": "x", "variant": HYBRID_VARIANT}])


def test_frozen_full_hybrid_schedule_activates_within_step_budget() -> None:
    """Task 29 freeze: alpha must engage inside the ~300-step 50-epoch run."""
    config = load_stage_config(STAGED_DIR / "full_hybrid.yaml")
    cells = iter_cells(config)
    assert len(cells) == 1
    spec = cells[0]
    assert spec.variant == HYBRID_VARIANT
    assert (spec.boundary_warmup_steps, spec.boundary_ramp_steps) == (50, 100)
    # Task 27 measured 12 steps over 2 epochs at batch 8 on the same server
    # data; 50 epochs ~= 300 steps. No training here: validate the exact
    # alpha trace the frozen schedule produces over that budget.
    schedule = ProgressiveLambda(
        lambda_max=float(spec.coefficients["boundary_alpha_max"]),
        ramp_steps=spec.boundary_ramp_steps,
        warmup_steps=spec.boundary_warmup_steps,
    )
    alphas = [schedule.lambda_at(step) for step in range(300)]
    assert sum(alpha > 0.0 for alpha in alphas) >= 249
    assert schedule.lambda_at(150) == pytest.approx(1.0)
    assert schedule.lambda_at(299) == pytest.approx(1.0)
    assert schedule.lambda_at(50) == pytest.approx(0.0)


def test_committed_configs_carry_explicit_calibration_floor() -> None:
    """Every 2/5/50 rerun config predeclares its small-sample floor (Finding 3)."""
    for name in ("contract.yaml", "balance.yaml",
                 "full_control.yaml", "full_hybrid.yaml"):
        config = load_stage_config(STAGED_DIR / name)
        assert config["calibration_min_samples"] == 4, (
            f"{name} must predeclare the explicit small-sample floor"
        )
        for spec in iter_cells(config):
            assert spec.calibration_min_samples == 4


def test_stage_config_rejects_invalid_calibration_floor(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "stage: contract\nepochs: 2\ncommit: '{COMMIT}'\n"
        "calibration_min_samples: 0\n"
        "cells:\n  - cell: default\n    variant: control-normal-only\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="calibration_min_samples"):
        load_stage_config(bad)
    with pytest.raises(ValueError, match="calibration_min_samples"):
        StageSpec(
            stage="contract", variant=CONTROL_VARIANT, epochs=2,
            coefficients=control_coefficients(), calibration_min_samples=-3,
        ).validate()


def test_staged_runner_fails_fast_on_insufficient_dev_val_without_pooling(
    tmp_path,
) -> None:
    """Default-floor tiny cohorts fail fast: the pooled fallback is gone."""
    data_root = tmp_path / "chronicle"
    materialize_chronological(client_config(seed=0), data_root)
    spec = StageSpec(
        stage="contract", variant=CONTROL_VARIANT, epochs=2, seed=0,
        d_model=8, batch_size=8, coefficients=control_coefficients(),
    )
    assert spec.calibration_min_samples is None
    with pytest.raises(ValueError, match="too small.*pooling.*rejected"):
        run_stage_cell(
            spec, data_root=data_root, output_root=tmp_path / "runs", device="cpu",
        )


def test_staged_source_has_no_pooled_or_fixed_fallback() -> None:
    """Structural guard: energy identity, dev-val cohorts, no legacy paths."""
    from representation import v2_staged

    src = Path(v2_staged.__file__).read_text(encoding="utf-8")
    assert 'energy_source="context_energy"' in src
    assert src.count('cohort="dev-val"') >= 2
    assert "calibrate_elevated_threshold_with_provenance" in src
    assert "operating_threshold" in src
    assert "confidence_calibrator_fit_cohort" in src
    for forbidden in ('pooled dev-train+dev-val (',
                      "elevated_threshold=config.elevated_threshold",
                      'encoded["patch_energy"]',
                      "clean_energy", "corrupt_energy",
                      "context_energy.pow(2)", "patch_energy.pow(2)"):
        assert forbidden not in src, f"staged runner keeps legacy path {forbidden!r}"


def test_full_hybrid_is_predeclared_default_not_reselected() -> None:
    """The superseded Task 29 winner must not resurface as a frozen config."""
    config = load_stage_config(STAGED_DIR / "full_hybrid.yaml")
    cells = iter_cells(config)
    assert len(cells) == 1
    assert cells[0].coefficients == hybrid_coefficients()
    text = (STAGED_DIR / "full_hybrid.yaml").read_text(encoding="utf-8")
    assert "SUPERSEDED" in text
    assert "PREDECLARED" in text
    assert "balance-b" not in text, (
        "full_hybrid.yaml must not reference the superseded selected cell"
    )
