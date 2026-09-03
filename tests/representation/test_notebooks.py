from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = (
    Path("notebooks/train_v1_representation.ipynb"),
    Path("notebooks/infer_v1_representation.ipynb"),
    Path("notebooks/local/train_v1_representation.ipynb"),
    Path("notebooks/local/infer_v1_representation.ipynb"),
    Path("notebooks/kaggle/train_v1_representation.ipynb"),
    Path("notebooks/kaggle/infer_v1_representation.ipynb"),
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
