from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = (
    Path("notebooks/train_v1_representation.ipynb"),
    Path("notebooks/infer_v1_representation.ipynb"),
)


def test_v1_notebooks_are_valid_current_package_workflows() -> None:
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(isinstance(cell_id, str) and cell_id for cell_id in ids)
        assert len(ids) == len(set(ids))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "representation" in source
        assert "make_contrastive_views" not in source
        assert "anomaly_detection" not in source
        assert "Spark" not in source
        assert "SessionGenerator" not in source
        assert "generate_normal" not in source
        assert "generate_anomaly" not in source
        assert "DatasetBuilder" not in source
        assert "V1_DATA_ROOT" in source
        assert "data/generated/production" in source
        assert "FileDataset" in source
        assert "manifest.json" in source
        assert "required_splits" in source
        assert "load_split('train')" in source
        assert "load_split('val'" in source
        if path.name.startswith("infer_"):
            assert "load_split('test'" in source
            assert ".fit(reference_output['file_embedding'])" in source
            assert "labels=" not in source
        assert "S_pred" in source or "S_{\\mathrm{pred}}" in source
        assert "S_pop" in source or "S_{\\mathrm{pop}}" in source
        assert "timestep" in source


TRAIN_NOTEBOOKS = (
    Path("notebooks/train_v1_representation.ipynb"),
)


def test_canonical_train_notebook_compiles_every_code_cell() -> None:
    for path in TRAIN_NOTEBOOKS:
        assert path.is_file(), f"Notebook missing: {path}"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4
        assert "cells" in notebook and len(notebook["cells"]) >= 5

        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids)), f"Duplicate cell IDs in {path}"

        code_cell_count = 0
        for cell_idx, cell in enumerate(notebook["cells"]):
            cell_type = cell.get("cell_type")
            cell_id = cell.get("id", f"cell_{cell_idx}")
            if cell_type == "code":
                code_cell_count += 1
                source = "".join(cell.get("source", []))
                # Compile code cell; raises SyntaxError if invalid
                compiled = compile(source, f"{path}:{cell_id}", "exec")
                assert compiled is not None
                # Confirm clean rerun readiness (empty outputs, no stale execution count)
                assert cell.get("outputs") == []
                assert cell.get("execution_count") is None
        assert code_cell_count >= 5


def test_training_notebooks_enforce_stationary_selection_and_prevent_incoherent_checkpoints() -> None:
    """Verify canonical notebooks adopt corrected APIs, stationary selection, and coherent checkpointing."""
    for path in TRAIN_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

        # Correct APIs & Stationary selection semantics
        assert "stationary_joint_loss" in source
        assert "val_stationary_joint_loss" in source
        assert "record_eval" in source
        assert "restore_best_state" in source
        assert "trainer.save_checkpoint" in source
        assert "resume" in source

        # Resume loop derives start epoch and stops cleanly when completed
        assert "start_epoch = start_step // num_batches_per_epoch" in source
        assert "if start_epoch >= params.epochs:" in source
        assert "range(start_epoch + 1, params.epochs + 1)" in source
        assert "contrastive_margin" in source
        # Broken ad-hoc rollback pattern must NOT be present
        assert 'val_metrics["joint_loss"] < best_loss' not in source
        assert "best_loss = val_metrics" not in source


def test_training_notebooks_expose_and_validate_lambda_max() -> None:
    """Verify lambda_max is user-editable, defaults to 0.1, validates non-negative, and drives configs."""
    for path in TRAIN_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

        # User-editable parameter with 0.1 default
        assert "lambda_max: float = float(os.environ.get('V1_LAMBDA_MAX', '0.1'))" in source
        # Non-negative validation in TrainingParams.__post_init__
        assert "if self.lambda_max < 0.0:" in source
        assert "raise ValueError(\"lambda_max must be non-negative\")" in source
        # Passed into V1Config
        assert "contrastive_weight_max=params.lambda_max" in source
        # Present in logs and progress reporting
        assert "lambda_max=" in source


def test_canonical_notebooks_use_explicit_server_paths() -> None:
    """Verify active notebooks use explicit server paths without Kaggle/mount discovery."""
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "SRC_DIR" in code
        assert "V1_REPO_ROOT" in code
        assert "V1_DATA_ROOT" in code
        assert "V1_CHECKPOINT_PATH" in code
        for forbidden in ("SRC_ZIP", "/kaggle/input", "repo_root", "found_src", "_resolve_default_paths"):
            assert forbidden not in code
