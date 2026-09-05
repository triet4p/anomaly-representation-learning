# Sprint 5 Task 19 — Manual paths in every active notebook

## Inventory (active production notebooks; historical untouched)

- Rewritten to manual paths (canonical + `notebooks/kaggle/` mirror, byte-identical): `notebooks/{train,infer,geometry_extraction,geometry_analysis}.ipynb` (+ 4 mirrors).
- Export `kaggle-20260905-01` root + `notebooks/kaggle/` mirrors: byte-identical to canonicals (the manual `SRC_ZIP`/`SRC_DIR` bootstrap is now universal, so no export divergence remains).
- Untouched: `experiments/20260904/*.ipynb`, `exports/kaggle-20260904-0{1,2,3,4}/`, training algorithms, Sprint 4 (only train path-bootstrap cells changed; no training logic).

## What was removed (all active copies)

- Source discovery: `candidates` lists, `found_src` loops, `SOURCE_CANDIDATES`/`_ARCHIVE_CANDIDATES`/`_WORKING_SRC`/`_BOOT_INPUT_ROOT`/`_INPUT_ROOT` blocks, every `.glob('**/src')`.
- Data/cache discovery: `_resolve_default_paths`, `_resolve_checkpoint*` (exact-filename `rglob` fallback), `**/manifest.json` scans, `repo_root` directory climbing.
- Hidden environment-driven data behavior: the Kaggle auto-unzip hook (`_kaggle_auto_unzip_iter`, `_verify_shard` override, `V1_SKIP_STRICT_HASH`) — users attach a properly materialized dataset; the standard `iter_materialized` path applies to everyone.
- The only remaining `.glob`/`.rglob` calls are output packaging of explicitly configured directories (`CACHE_ROOT.rglob`/`OUTPUT_DIR.rglob` in the analysis download cell) — not input discovery.

## Manual contract (first cells; exact paths used, fail-fast naming the variable)

- All notebooks: `SRC_ZIP` (default `src.zip`) + `SRC_DIR` (default `/kaggle/working/src`), each honoring `V1_SRC_ZIP`/`V1_SRC_DIR` overrides. `SRC_DIR` used as-is when present; otherwise extracted from exactly `SRC_ZIP`; otherwise `FileNotFoundError` naming `SRC_DIR`/`SRC_ZIP`. `representation` presence validated before `sys.path` insert.
- train/infer/extraction: `V1_DATA_ROOT` (defaults `/kaggle/input/anomaly-representation-20260903-01`; extraction `/kaggle/input/v1-materialized-dataset`; local alternative `data/generated/production` documented); manifest absence raises `FileNotFoundError` naming `V1_DATA_ROOT`.
- train: `V1_CHECKPOINT_PATH` (default `/kaggle/working/v1_representation.pt`, writable output; resume reads it).
- infer/extraction: `V1_MODEL_HANDLE` + `V1_MODEL_MOUNT` constants kept; `V1_CHECKPOINT_PATH`/`V1_CHECKPOINT` default to the pinned mount `/kaggle/input/v1-representation-20260904-01/pyTorch/default/1/v1_representation_20260904_01.pt`; extraction fail-fasts naming `V1_CHECKPOINT` when absent; inference keeps its established checkpoint-missing-uses-defaults semantics with the resolved path.
- analysis (cache-only, no checkpoint loading — verified): `V1_GEOMETRY_CACHE` (default `/kaggle/input/geometry-cache`) with manifest fail-fast, `V1_GEOMETRY_ANALYSIS` (default `/kaggle/working/geometry-analysis`).
- Relative paths resolve against `Path.cwd()`; all notebook prose updated to match the manual variables.

## Compatibility preserved

- Published Kaggle Model identity/path unchanged (handle, mount, `model_sources` in ext/infer kernel metadata, guide contract section).
- Downstream test contracts kept verbatim: geometry token sets + analysis forbiddens; infer tokens (`V1_DATA_ROOT`, `data/generated/production`, `load_split('train'/'val'/'test')`, `.fit(...)`, `S_pred`/`S_pop`, `timestep`, no `labels=`); train tokens (stationary selection, resume loop, `lambda_max` 0.1 default + validation).
- Kernel metadata otherwise unchanged; `dataset-metadata.json` unchanged (dataset identity separate).

## Checks (static only; focused tests; no execution/inference/diagnostics)

- All 14 active/exported code cells compile; outputs cleared; cell ids unique.
- Forbidden autodiscovery tokens absent in all 14 (`_resolve_default_paths`, `_resolve_checkpoint`, `repo_root`, `found_src`, `SOURCE_CANDIDATES`, `_ARCHIVE_CANDIDATES`, `_BOOT_INPUT_ROOT`, `_WORKING_SRC`, `_INPUT_ROOT`, `_kaggle_auto_unzip_iter`, `V1_SKIP_STRICT_HASH`, `**/manifest.json`); `.glob/.rglob` only in the analysis packaging cell over explicit dirs.
- Manual variables + fail-fast messages asserted per notebook; parity: canonical==kaggle==export-root==export-mirror for geometry/inference/analysis, canonical==kaggle for train.
- 18 packaged imports AST-resolve against bundled `src/`; zero `__pycache__`/`.pyc`/`.pyo` in export dir, `src.zip` (52 members), outer ZIP (65 entries); manifest equals tracked disk (64 files); inner `src.zip` hash matches disk.
- Focused tests: `test_device_placement.py` + `test_normalization.py` → 7 passed, 1 skipped (CUDA unavailable on CPU-only torch 2.10.0 host). No formatters/linters/project-wide suites.

## Final artifacts

- `exports/kaggle-20260905-01/src.zip`: 108092 bytes, SHA-256 `db3d7abffc4e142760dd344c1170095432b5ebb55f2d1fb72cb4098161b13510`.
- `exports/kaggle-20260905-01.zip`: 245395 bytes, SHA-256 `d8ab83f490eba9db08973fb9b255c58750e1d6663919c3e5484b6d03718d3795`.

## Correction — runtime-path evidence gate (HIGH/MEDIUM/LOW)

- HIGH (random-weight inference fallback removed): all 4 active `infer_v1_representation.ipynb` copies (canonical, `notebooks/kaggle/`, export root, export mirror) no longer build `V1Config`/random weights when the checkpoint is absent. The config cell now raises `FileNotFoundError` naming `V1_CHECKPOINT_PATH`, the exact expected path, and Kaggle Model `V1_MODEL_HANDLE`, stating inference never runs on random weights. Valid checkpoint loading (`torch.load` config restore) and compatibility checks are untouched. Train notebook intentionally unchanged (training from scratch is its valid path).
- MEDIUM (frozen-history test regression): `tests/representation/test_notebooks.py` no longer compares the current canonical train notebook against frozen export `kaggle-20260904-03`. The historical bundle test now asserts only the bundle's own frozen content (`lambda_max` 0.1 default, `start_epoch` derivation); historical export files untouched. New `test_current_canonical_mirror_and_export_parity` asserts byte parity across canonical ↔ Kaggle mirrors and canonical ↔ current-export copies for train/infer/geometry.
- LOW (docs): `RUN_GUIDE.md` inference step and all 4 infer cell-0 bullets now state inference always fails fast with `FileNotFoundError` when the checkpoint is absent and never scores with random weights; all `otherwise runs with configured defaults` / `using configured defaults` language removed from guide and notebooks.
- `src/` source unchanged by this correction, so `src.zip` bytes/hash are preserved.
- Focused tests: `test_notebooks.py` + `test_geometry_notebooks.py` + `test_device_placement.py` + `test_normalization.py` → **19 passed, 1 skipped** (CUDA unavailable on CPU-only torch 2.10.0 host). No project-wide suites, no notebook/cell execution.
- Recheck: all 14 code cells compile with outputs cleared; parity re-verified; forbidden autodiscovery + fallback tokens absent; 4/4 infer copies carry the fail-fast; manifest equals tracked disk (64 files); archives clean with zero bytecode (`PYTHONDONTWRITEBYTECODE=1`, no export imports).

## Final artifacts (after correction)

- `exports/kaggle-20260905-01/src.zip`: 108092 bytes, SHA-256 `db3d7abffc4e142760dd344c1170095432b5ebb55f2d1fb72cb4098161b13510` (unchanged).
- `exports/kaggle-20260905-01.zip`: 245415 bytes, SHA-256 `2a3bea27420e2e3e8a5786690940641ec7e8e77f80110accafb756cea36db392`.
