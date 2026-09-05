# Sprint 6 Task 2 — Convert active notebooks to server paths

## Conversion (canonical notebooks only; mirrors deleted in Task 1)

All four canonical notebooks (`train`, `infer`, `geometry_extraction`, `geometry_analysis`) now use explicit editable first-cell variables with fail-fast validation and no Kaggle/mount/archive logic. Cells remain unexecuted.

- Source: `V1_REPO_ROOT` (default `.`) + `SRC_DIR` (default `<repo>/src`). No `SRC_ZIP`, no extraction, no `zipfile` import for packaging (analysis keeps `zipfile` only for its diagnostics bundle), no candidate lists, no globbing. `SRC_DIR` must contain `representation/` or a `FileNotFoundError` names `SRC_DIR`/`V1_REPO_ROOT`. Run from the repository root or set one variable.
- Data: `V1_DATA_ROOT` (train/infer) and `DATASET_ROOT` (extraction), default `data/generated/production`; manifest absence raises `FileNotFoundError` naming the variable.
- Checkpoint: `V1_CHECKPOINT_PATH` (train/infer) and `CHECKPOINT_PATH` (extraction), default `checkpoints/v1_representation_20260904_01.pt` — the published filename, kept as the real server asset. Kaggle Model handle/mount constants removed. Inference keeps fail-fast-on-missing semantics (never random weights); training keeps resume/output semantics.
- Geometry: `V1_GEOMETRY_OUTPUT` default `experiments/20260905/geometry-cache`; `V1_GEOMETRY_CACHE` / `V1_GEOMETRY_ANALYSIS` defaults under `experiments/20260905/`. Analysis stays cache-only (no checkpoint loading, verified).
- Prose updated to server/repository language in all four notebooks (the stale `V1_MODEL_HANDLE` references this exposed in two fail-fast messages were fixed as part of the conversion).

## Preserved contracts

- Task 18 CUDA device fix and Task 20 `_artifact_info` fix untouched in `src/` (no source changes in this task).
- Test tokens intact: geometry gate tokens + analysis cache-only forbiddens; infer/train workflow tokens (`V1_DATA_ROOT`, `data/generated/production`, `load_split`, stationary selection, `lambda_max` 0.1 + validation).
- Relative paths resolve against `Path.cwd()`; all code cells compile with outputs cleared.

## Verification

- Focused suites: `test_notebooks.py` + `test_geometry_notebooks.py` + `test_device_placement.py` + `test_normalization.py` + `test_extraction_finalization.py` → 18 passed, 1 skipped (CUDA unavailable). No formatters/linters/project-wide suites, no notebook execution.
- `grep` over `notebooks/*.ipynb`: zero `V1_MODEL_HANDLE`, `V1_MODEL_MOUNT`, `/kaggle/input`, `SRC_ZIP` references.
