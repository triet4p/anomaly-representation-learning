# Sprint 5 Task 16 — Self-contained 20260905 Kaggle export

## Changed files (`exports/kaggle-20260905-01/`, in place)

- `geometry_extraction.ipynb` — Stage 1 copy of `notebooks/geometry_extraction.ipynb`; bootstrap cell now unpacks `src.zip` into `/kaggle/working/src` (skipped when present), then resolves `/kaggle/working/src` → local `src/` → `/kaggle/input/*/src`. Algorithm cells byte-identical to canonical.
- `geometry_analysis.ipynb` — Stage 2 copy of `notebooks/geometry_analysis.ipynb`; same `src.zip` bootstrap treatment. Cache-only boundary unchanged.
- `infer_v1_representation.ipynb` — **new**: full inference workflow from `notebooks/infer_v1_representation.ipynb` with packaging/path bootstrap adaptation only (see decisions). All scoring/evaluation cells unchanged.
- `notebooks/kaggle/{geometry_extraction,geometry_analysis,infer_v1_representation}.ipynb` — byte-identical mirrors of the three root notebooks (parity asserted).
- `src.zip` — **new**: 52 Python modules from the matching `src/` tree with `src/` prefix, `ZIP_DEFLATED(9)`.
- `src/` — retained readable tree; verified byte-identical to repo `src/*.py` (52 files, no diffs).
- `kernel-metadata-infer.json` — **new**: inference kernel (`infer_v1_representation.ipynb`, GPU, private); existing `kernel-metadata.json` (extraction/GPU) and `kernel-metadata-analysis.json` (analysis/CPU) untouched.
- `RUN_GUIDE.md` — now documents three notebooks, `src.zip` contents, the `/kaggle/working/src` bootstrap, inference run step, and a "packaged source not found" recovery entry.
- `export-manifest.json` — `files` inventory regenerated (64 entries: sizes + SHA-256); description extended for inference/`src.zip` coverage. Provenance keys (`export_slug`, `git_commit`, `package_version`, `patch_*`, `prior_export_slug`, `dirty_tree*`) preserved.
- `dataset-metadata.json` — unchanged (dataset identity preserved).
- `exports/kaggle-20260905-01.zip` — regenerated outer archive (top-level `kaggle-20260905-01/` prefix).

No changes to `notebooks/`, `src/`, `scripts/`, `tests/`, Sprint 4 files, or scientific algorithms.

## Compatibility decisions

1. `src.zip` bootstrap (all notebooks): search `cwd/src.zip`, `src.zip`, `/kaggle/input/*/src.zip`, `/kaggle/input/*/*/src.zip`; `extractall` to `/kaggle/working` (yields `/kaggle/working/src` since members carry the `src/` prefix); skip when `/kaggle/working/src` exists. Falls back to local `src/` or attached `/kaggle/input/*/src`. No checkout required.
2. Inference notebook bootstrap repair (export copy only): source `notebooks/infer_v1_representation.ipynb` cell 1 has a stray column-0 `from representation.checkpoint import load_checkpoint` inside the `_kaggle_auto_unzip_iter` fallback (compile error) and uses `zipfile` without importing it. Exported copy moves that import into the import block (`load_checkpoint, save_checkpoint`), adds `import zipfile`, and adds `mad_threshold` to the `representation.inference` import (used by scoring/evaluation cells, previously a latent `NameError`). Dataset/checkpoint identities (`anomaly-representation-20260903-01` slugs, `V1_DATA_ROOT`, `V1_CHECKPOINT_PATH`) and all algorithm cells preserved verbatim.
3. `src.zip` content = export `src/` tree exactly (52 `.py` files); `egg-info`, `__pycache__`, `.pyc`, `.pyo` excluded.
4. CRLF line endings kept for export text files (host/previous-attempt convention); hashes computed over on-disk bytes.

## Layout

- Export root: 3 notebooks + `src.zip` + `src/` + `notebooks/kaggle/` (3 mirrors) + 3 kernel metadata + dataset metadata + manifest + guide = 65 files.
- Outer ZIP: `kaggle-20260905-01/` + all 65 files (66 entries incl. dir), `ZIP_DEFLATED(9)`.

## Checks (static only; no cell execution, inference, extraction, or diagnostics ran)

- Every code cell of all 6 exported notebooks `compile()`s; `outputs == []`, `execution_count is None`; cell ids unique.
- Each notebook contains `src.zip` + `/kaggle/working/src` + `ZipFile`/`extractall` + local `src/` fallback (structural grep).
- 19 packaged `representation.*`/`synth.*` imports AST-parsed from notebook cells; all modules/symbols resolve in bundled `src/`; `synth.dataset` hook attrs (`_verify_shard`, `iter_materialized`, `load_sample`, `load_sample_bytes`) present.
- Zero `__pycache__`/`.pyc`/`.pyo` in export dir, `src.zip` (52 members), and outer ZIP (66 entries).
- Root ↔ `notebooks/kaggle/` byte parity for all three notebooks.
- Manifest `files` map == disk (64 entries, size + SHA-256); outer ZIP opens; inner `src.zip` SHA-256 matches disk.
- Formatters, linters, and project-wide suites skipped per assignment.

## Final artifacts

- `exports/kaggle-20260905-01/src.zip`: 107757 bytes, SHA-256 `232df891207694dd59f0fd6d900651300913c1a087efce34864e804f280eff28`.
- `exports/kaggle-20260905-01.zip`: 244086 bytes, SHA-256 `b386cbe2037a7f6f7052690098f493cdde5c26a4ed38ef0c26a07f5f4916e9eb`.
