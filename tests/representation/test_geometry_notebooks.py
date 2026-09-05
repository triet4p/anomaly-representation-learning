from __future__ import annotations

import json
from pathlib import Path

import pytest

CANONICAL = {
    "extract": Path("notebooks/geometry_extraction.ipynb"),
    "analysis": Path("notebooks/geometry_analysis.ipynb"),
}


def _load(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs") == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert len({cell.get("id") for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
    return notebook


@pytest.mark.parametrize("name", ["extract", "analysis"])
def test_geometry_notebooks_are_clean_and_compilable(name: str) -> None:
    notebook = _load(CANONICAL[name])
    assert len(notebook["cells"]) >= 5


def test_geometry_notebooks_use_explicit_server_paths() -> None:
    for name in ("extract", "analysis"):
        code = "\n".join("".join(cell["source"]) for cell in _load(CANONICAL[name])["cells"] if cell["cell_type"] == "code")
        assert "SRC_DIR" in code
        assert "V1_REPO_ROOT" in code
        for forbidden in ("SRC_ZIP", "/kaggle/input"):
            assert forbidden not in code
    extract_code = "\n".join("".join(cell["source"]) for cell in _load(CANONICAL["extract"])["cells"] if cell["cell_type"] == "code")
    assert ".glob(" not in extract_code and ".rglob(" not in extract_code
    analysis_code = "\n".join("".join(cell["source"]) for cell in _load(CANONICAL["analysis"])["cells"] if cell["cell_type"] == "code")
    assert ".glob(" not in analysis_code
    # The only rglob calls package explicitly configured output directories.
    assert analysis_code.count(".rglob(") == 2
    assert "CACHE_ROOT.rglob" in analysis_code and "OUTPUT_DIR.rglob" in analysis_code


def test_extraction_notebook_exposes_bounded_user_gate() -> None:
    source = "\n".join("".join(cell["source"]) for cell in _load(CANONICAL["extract"])["cells"])
    for required in ("DATASET_ROOT", "CHECKPOINT_PATH", "OUTPUT_DIR", "SELECTED_SPLITS", "MAX_SAMPLES", "BATCH_SIZE", "SEED", "extract_embeddings", "geometry-manifest.json", "embeddings.npz", "records.csv"):
        assert required in source


def test_analysis_notebook_is_cache_only_and_exposes_bounds() -> None:
    notebook = _load(CANONICAL["analysis"])
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "analyze_geometry_cache" in code
    assert "load_geometry_cache" in code
    for required in ("CACHE_ROOT", "MAX_NEIGHBOR_QUERIES", "NEIGHBOR_K", "MAX_PROJECTION_SAMPLES", "PROJECTION_SEED", "NEIGHBOR_SEED", "geometry-diagnostics.zip"):
        assert required in code
    for forbidden in ("load_checkpoint", "V1RepresentationModel", "torch.load", "extract_embeddings"):
        assert forbidden not in code
