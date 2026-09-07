"""Batch C1: canonical V2 notebook contracts (Tasks 18-20).

AST/nbformat structural checks only: the three canonical notebooks stay
unexecuted, syntax-valid, headless-nbconvert compatible, call the accepted
public entry points, fail fast on missing roots/checkpoints, and never
duplicate scheduler/model/inference product logic. No training or full
notebook execution happens here (Task 23 owns the client smoke).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

GENERATE = Path("notebooks/generate_chronological_factory.ipynb")
TRAIN = Path("notebooks/train_v2_geometry.ipynb")
INFER = Path("notebooks/infer_v2_static.ipynb")
ALL = {"generate": GENERATE, "train": TRAIN, "infer": INFER}


def _load(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4, f"{path} must be nbformat 4"
    assert len(notebook["cells"]) >= 5, f"{path} must keep markdown + code cells"
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(i, str) and i for i in ids), f"{path} needs string cell ids"
    assert len(set(ids)) == len(ids), f"{path} has duplicate cell ids"
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4, f"{path} must keep at least four code cells"
    for cell in code_cells:
        assert cell.get("outputs") == [], f"{path}:{cell.get('id')} must stay unexecuted"
        assert cell.get("execution_count") is None, f"{path}:{cell.get('id')} must stay unexecuted"
        compile("".join(cell["source"]), f"{path}:{cell.get('id')}", "exec")
    return notebook


def _code(path: Path) -> str:
    notebook = _load(path)
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


def _forbids_tree(path: Path, names: tuple[str, ...]) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        leaked = defined & set(names)
        assert not leaked, f"{path}:{cell.get('id')} duplicates product logic {sorted(leaked)}"


@pytest.mark.parametrize("name", ["generate", "train", "infer"])
def test_batch_c1_notebooks_are_clean_syntax_valid_and_headless(name: str) -> None:
    code = _code(ALL[name])
    assert code
    for line in code.splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith("%"), f"{name}: notebook magics break nbconvert"
        assert not stripped.startswith("!"), f"{name}: shell escapes break nbconvert"
    for forbidden in ("get_ipython", "SRC_ZIP", "/kaggle/input", ".glob(", ".rglob("):
        assert forbidden not in code, f"{name}: forbidden {forbidden!r}"


@pytest.mark.parametrize("name", ["generate", "train", "infer"])
def test_batch_c1_notebooks_keep_first_cell_path_contract(name: str) -> None:
    code = _code(ALL[name])
    assert "V2_REPO_ROOT" in code
    assert "SRC_DIR" in code
    assert "sys.path" in code
    assert "FileNotFoundError" in code
    assert "PYTHONHASHSEED" in code


def test_generate_notebook_uses_public_chronicle_entry_point() -> None:
    code = _code(GENERATE)
    for required in ("materialize_chronological", "client_config", "server_config",
                     "load_chronological", "manifest.json", "dev_train", "dev_val",
                     "test_static", "test_temporal", "quarantined", "config_hash",
                     "sha256", "cutoff", "quarantine", "chronicle-summary.json"):
        assert required in code, f"generate notebook missing {required!r}"
    assert "synth.cli --chronological" in code or "synth.cli" in code
    for forbidden in ("FactoryScheduler", "RobotHealthProcess", "ScheduledSignalGenerator",
                      "TemporalAnomalyProcess", "ChronologicalSplitter"):
        assert forbidden not in code, f"generate notebook must not touch {forbidden}"
    _forbids_tree(GENERATE, ("FactoryScheduler", "ChronologicalSplitter", "build_chronological"))


def test_generate_notebook_audits_schedule_and_split_invariants() -> None:
    code = _code(GENERATE)
    for required in ("no overlaps", "member_views", "is_quarantined", "per_robot",
                     "FileExistsError", "ValueError"):
        assert required in code, f"generate notebook missing invariant {required!r}"


def test_train_notebook_covers_epoch_profiles_and_coherent_checkpoints() -> None:
    code = _code(TRAIN)
    for required in ("V2Config", "build_v2_training_stack", "train_step",
                     "stationary_loss", "record_eval", "restore_best_state",
                     "fit_geometry", "calibrate_elevated_threshold",
                     "HealthyTailCalibrator", "TrajectoryTracker", "fit_commissioning",
                     "patch_regime_ids", "load_chronological", "corruption_mask",
                     "severity", "normal_loss", "background_loss", "boundary_loss",
                     "grad_norm", "cuda", "FileExistsError"):
        assert required in code, f"train notebook missing {required!r}"
    for profile in ('"smoke": 2', '"balance": 5', '"full": 50'):
        assert profile in code, f"train notebook missing epoch profile {profile}"
    assert "torch.manual_seed" in code
    assert "V2_RESUME" in code
    for forbidden in ("torch.randn", "random weights", "RandomWeights"):
        assert forbidden not in code, f"train notebook must not fall back: {forbidden!r}"
    _forbids_tree(TRAIN, ("HierarchicalMahalanobisGeometry", "CounterfactualCriterion",
                          "ContextConditionedPatchEncoder", "CensoredSurvivalRisk"))


def test_train_notebook_fails_fast_on_missing_roots_and_checkpoints() -> None:
    code = _code(TRAIN)
    assert "V2_DATA_ROOT" in code
    assert "V2_CHECKPOINT_PATH" in code
    assert code.count("FileNotFoundError") >= 2


def test_infer_notebook_uses_restored_reference_pipeline() -> None:
    code = _code(INFER)
    for required in ("V2InferencePipeline.load", "score_patches", "population_energy",
                     "context_energy", "file_population", "energy_source",
                     "tail_energy", "elevated_fraction", "confidence", "localiz",
                     "family", "severity", "roc_auc_score", "average_precision",
                     "dev-val", "provenance.json", "metrics.json", "slices.csv",
                     "fallback_order", "reference_source", "operating_threshold_cohort",
                     "confidence_calibrator_fit_cohort", "elevated_threshold_source"):
        assert required in code, f"infer notebook missing {required!r}"
    assert "torch.load" not in code, "inference must load only through V2InferencePipeline"
    assert code.count("pipeline.refit_references(") == 1, "refit must exist only as the explicit opt-in"
    _forbids_tree(INFER, ("HierarchicalMahalanobisGeometry", "HealthyTailCalibrator",
                          "CensoredSurvivalRisk", "V2InferencePipeline"))


def test_train_notebook_uses_complete_stationary_objective() -> None:
    """Deep-Review Finding 1: stationary selection keeps the full signed-likelihood."""
    code = _code(TRAIN)
    for required in ("stationary_loss", "record_eval", "restore_best_state",
                     "density_raw", "variance_raw", "covariance_raw",
                     "background_loss", "boundary_loss", "final alpha"):
        assert required in code, f"train notebook missing stationary term {required!r}"
    for forbidden in ("context_energy.pow(2)", "clean_energy", "corrupt_energy"):
        assert forbidden not in code, f"train notebook must not use squared-NLL/renamed selection: {forbidden!r}"


def test_train_notebook_calibrates_dev_val_only_with_provenance() -> None:
    """Deep-Review Findings 2-3: dev-val threshold + calibrator, distinct provenances."""
    code = _code(TRAIN)
    assert "calibrate_elevated_threshold_with_provenance" in code
    assert "calibrate_elevated_threshold(" not in code.replace(
        "calibrate_elevated_threshold_with_provenance(", "")
    assert code.count('cohort="dev-val"') >= 2, "threshold and calibrator each need dev-val cohort"
    for required in ('energy_source="context_energy"', "fit_cohort_info",
                     "operating_threshold", "confidence_calibrator_cohort",
                     'energy_field"] == "context_energy"', "val_displacements"):
        assert required in code, f"train notebook missing calibration provenance {required!r}"


def test_notebooks_keep_context_and_population_energies_distinct() -> None:
    """Deep-Review Finding 1: explicit energy_source, no compatibility names."""
    train = _code(TRAIN)
    infer = _code(INFER)
    assert 'energy_source="context_energy"' in train
    assert 'out["file"]["energy_source"] == "context_energy"' in infer
    assert 'out["file_population"]["energy_source"] == "population_energy"' in infer
    assert 'out["patch"]["context_energy"]' in infer
    assert 'out["population"]["population_energy"]' in infer
    for name, code in (("train", train), ("infer", infer)):
        for forbidden in ("patch_energy", "clean_energy", "corrupt_energy",
                          "validate_patch_output", "V2PatchOutput", "LEGACY_PATCH"):
            assert forbidden not in code, f"{name} notebook keeps ambiguous energy name {forbidden!r}"
    assert 'out["patch"]["population_energy"]' not in infer, \
        "infer patch dict must carry context_energy; population lives under out['population']"


def test_infer_notebook_restores_threshold_and_separates_cohorts() -> None:
    """Deep-Review Findings 2-3: restored cutoff by default, cohorts never conflated."""
    code = _code(INFER)
    assert "V2InferencePipeline.load(CHECKPOINT_PATH, device=str(device))" in code, \
        "inference must restore without an explicit elevated-threshold override"
    assert "elevated_threshold=3.0" not in code, \
        "inference must never fall back to the fixed 3.0 cutoff after calibration"
    for required in ("operating_record", "confidence_cohort_record",
                     "operating_cohort_label", "confidence_fit_label",
                     "confidence_operating_threshold", "confidence_decision_cohort",
                     "operating_threshold_source"):
        assert required in code, f"infer notebook missing cohort separation {required!r}"
    assert code.count("calibration_source") == 0, \
        "a single shared calibration_source conflates the two provenances"
    assert 'energies = out["patch"]["context_energy"][0]' in code, \
        "localization must rank the monitoring (context) energy"
