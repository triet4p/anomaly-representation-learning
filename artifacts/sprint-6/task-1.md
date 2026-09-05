# Sprint 6 Task 1 — Remove Kaggle export workflow

## Deletions (explicitly authorized; no regeneration)

- `exports/kaggle-20260904-01/`, `kaggle-20260904-02/`, `kaggle-20260904-03/`, `kaggle-20260904-04/`, `kaggle-20260905-01/` directories — removed with `rm -rf`.
- `exports/kaggle-20260904-04.zip`, `exports/kaggle-20260905-01.zip` — removed. `exports/` held nothing else (verified by listing before deletion); empty parent dirs removed.
- `notebooks/kaggle/` (4 notebook mirrors + `kernel-metadata.json`, Kaggle-only) — removed with `rm -rf`.
- `scripts/package_kaggle_dataset.py`, `scripts/upload_to_kaggle.ps1`, `scripts/upload_to_kaggle.sh` (Kaggle dataset packaging/upload only; `scripts/` held nothing else) — removed.

## Packaging/test/docs assumptions removed

- `tests/representation/test_notebooks.py`: deleted the historical export-bundle test (its `exports/kaggle-20260904-0{1,2,3}` subjects no longer exist), dropped the `notebooks/kaggle/` entry from `TRAIN_NOTEBOOKS`, replaced the Kaggle/export parity test with a server-path contract test; removed the now-unused `pytest` import.
- `tests/representation/test_geometry_notebooks.py`: deleted the `notebooks/kaggle/` mirror map and byte-parity test; kept canonical clean/compile/contract tests and added an explicit server-path test (no `SRC_ZIP`, no `/kaggle/input`, packaging-only `rglob`).
- No `src.zip` generation code, manifest, or archive remains anywhere; no Kaggle artifacts were regenerated.

## Preserved (untouched, verified by listing/grep)

- Unrelated user work: `CHANGELOG.md`, `docs/PLAN.md`, legacy `artifacts/task_*_summary.md` deletions, `.agents/`, `artifacts/sprint-2/`, `artifacts/sprint-4/`, `docs/sprint-plans/sprint-3|4|5.md`, `experiments/20260904/` historical notebooks, `checkpoints/`, `data/`.
- Sprint history docs (CHANGELOG entries, sprint-4/5 plans, sprint-5 evidence) still describe the superseded workflow as history; only active workflow assumptions were removed.
- Intentionally preserved runtime fallback (not packaging): `src/synth/dataset.py::iter_materialized` keeps its `/kaggle/input` read-only-mount tolerance branch. It is inert on the server (`/kaggle/input` absent → strict hashes) and would require a behavior change to dataset verification to remove; out of deletion scope.

## Verification

- `ls exports/ notebooks/kaggle/ scripts/` — all gone (empty parents removed); `git status` shows only the intended `D` entries for tracked files (`notebooks/kaggle/kernel-metadata.json`, `notebooks/kaggle/train_v1_representation.ipynb`, 3 scripts).
- Focused suites after cleanup: `test_notebooks.py` + `test_geometry_notebooks.py` + `test_device_placement.py` + `test_normalization.py` + `test_extraction_finalization.py` → 18 passed, 1 skipped (CUDA unavailable). No formatters/linters/project-wide suites.
