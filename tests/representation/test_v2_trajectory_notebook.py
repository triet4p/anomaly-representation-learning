"""Batch C2 Task 21: canonical trajectory/early-warning notebook contracts.

AST/nbformat structural checks only: the trajectory notebook stays
unexecuted, syntax-valid, headless-nbconvert compatible, preserves
chronological robot episodes with maintenance segmentation, reports
fixed/short-term displacement/velocity/trend/persistence, fits survival
only on allowed pre-cutoff data, evaluates censored 1d/7d horizons, and
never duplicates product logic. No training or notebook execution here
(Task 23 owns the client smoke).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

TRAJECTORY = Path("notebooks/trajectory_v2_early_warning.ipynb")


def _load() -> dict[str, object]:
    notebook = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4, "trajectory notebook must be nbformat 4"
    assert len(notebook["cells"]) >= 5, "trajectory notebook must keep markdown + code cells"
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(i, str) and i for i in ids), "trajectory notebook needs string cell ids"
    assert len(set(ids)) == len(ids), "trajectory notebook has duplicate cell ids"
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4, "trajectory notebook must keep at least four code cells"
    for cell in code_cells:
        assert cell.get("outputs") == [], f"{cell.get('id')} must stay unexecuted"
        assert cell.get("execution_count") is None, f"{cell.get('id')} must stay unexecuted"
        compile("".join(cell["source"]), f"{TRAJECTORY}:{cell.get('id')}", "exec")
    return notebook


def _code() -> str:
    notebook = _load()
    return "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )


def test_trajectory_notebook_is_clean_syntax_valid_and_headless() -> None:
    code = _code()
    assert code
    for line in code.splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith("%"), "notebook magics break nbconvert"
        assert not stripped.startswith("!"), "shell escapes break nbconvert"
    for forbidden in ("get_ipython", "SRC_ZIP", "/kaggle/input", ".glob(", ".rglob("):
        assert forbidden not in code, f"forbidden {forbidden!r}"

def test_trajectory_notebook_records_uncalibrated_status_without_test_fallback() -> None:
    code = _code()
    for required in ("risk_fit_error", "uncalibrated", "cohort_composition",
                     "no test-data fallback", "no test fallback"):
        assert required in code, f"trajectory notebook missing {required!r}"


def test_trajectory_notebook_keeps_first_cell_path_contract() -> None:
    code = _code()
    for required in ("V2_REPO_ROOT", "SRC_DIR", "sys.path", "FileNotFoundError",
                     "PYTHONHASHSEED", "torch.device"):
        assert required in code, f"trajectory notebook missing {required!r}"


def test_trajectory_notebook_preserves_chronological_episodes() -> None:
    code = _code()
    for required in ("test_temporal", "EpisodeKind.MAINTENANCE", "maintenance_reset",
                     "chronological", "fresh_tracker", "load_state_dict",
                     "suspect", "score_patches"):
        assert required in code, f"trajectory notebook missing {required!r}"


def test_trajectory_notebook_reports_fixed_short_term_signals() -> None:
    code = _code()
    for required in ("displacement", "velocity", "trend", "persistence",
                     "disagreement", "fixed", "short-term",
                     "V2InferencePipeline.load", "patch_regime_ids"):
        assert required in code, f"trajectory notebook missing {required!r}"
    assert "torch.load" not in code, "must load only through V2InferencePipeline"


def test_trajectory_notebook_fits_survival_on_allowed_data_only() -> None:
    code = _code()
    for required in ("CensoredSurvivalRisk", "trajectory_feature_matrix",
                     "expected_feature_width", "pre-cutoff", "sealed",
                     "future_targets", "horizon_cohort", "risk_1d", "risk_7d",
                     "brier", "Brier", "calibration", "conformal",
                     "trajectory_metrics.json", "risk_metrics.json", "provenance.json"):
        assert required in code, f"trajectory notebook missing {required!r}"


def test_trajectory_notebook_never_duplicates_product_logic() -> None:
    notebook = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        defined = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        leaked = defined & {
            "TrajectoryTracker",
            "HealthyTailCalibrator",
            "CensoredSurvivalRisk",
            "HierarchicalMahalanobisGeometry",
            "ContextConditionedPatchEncoder",
            "V2InferencePipeline",
        }
        assert not leaked, f"{cell.get('id')} duplicates product logic {sorted(leaked)}"


def test_trajectory_notebook_uses_explicit_energy_identities() -> None:
    """Deep-Review Finding 1: context acute chain plus independent population view."""
    code = _code()
    assert 'out["file"]["energy_source"] == "context_energy"' in code
    assert 'out["file_population"]["energy_source"] == "population_energy"' in code
    assert 'out["file"]["tail_energy"]' in code
    assert 'out["file"]["elevated_fraction"]' in code
    assert 'out["file_population"]["tail_energy"]' in code
    for forbidden in ("patch_energy", "clean_energy", "corrupt_energy",
                      "validate_patch_output", "V2PatchOutput", "LEGACY_PATCH"):
        assert forbidden not in code, f"trajectory notebook keeps ambiguous energy name {forbidden!r}"
    assert 'out["patch"]["population_energy"]' not in code, \
        "population energy lives under out['population']/out['file_population'], never out['patch']"

def test_trajectory_notebook_separates_calibration_provenances() -> None:
    """Deep-Review Findings 2-3: distinct threshold/calibrator cohorts, honest small-sample."""
    code = _code()
    for required in ("operating_record", "confidence_cohort_record",
                     "operating_cohort_label", "confidence_fit_label",
                     "small_sample", "n_samples"):
        assert required in code, f"trajectory notebook missing cohort separation {required!r}"
    assert code.count("calibration_source") == 0, \
        "a single shared calibration_source conflates the two provenances"
    assert "calibrate_elevated_threshold" not in code, \
        "trajectory analysis restores calibration; it never refits the threshold"
    assert "torch.load" not in code, "must load only through V2InferencePipeline"


def test_trajectory_notebook_reports_honest_coverage_status() -> None:
    """Deep-Review Finding 3: coverage is an in-sample diagnostic, never independent."""
    code = _code()
    for required in ("in-sample diagnostic", "conformal_coverage_status",
                     "conformal_coverage_cohort", "not independent coverage"):
        assert required in code, f"trajectory notebook missing honest coverage label {required!r}"
