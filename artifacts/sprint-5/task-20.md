# Sprint 5 Task 20 — Undefined `_artifact_info` finalization fix

## User-reported failure (ground truth, not rerun)

`BoundedEmbeddingExtractor.extract` reaches its final `files` mapping and raises `NameError: name '_artifact_info' is not defined` at `embedding_extraction.py` (~line 352).

## Root cause

`extract()` calls `_artifact_info(path, root)` four times (embeddings/records/metrics/neighbors) but the helper was never defined or imported in `src/representation/embedding_extraction.py` — a latent crash on every successful extraction, reached only after all GPU/CPU streaming work completed. No LSP server exists in this environment (no LSP tool in inventory); definition/reference history was traced with repo-wide search: zero definitions repo-wide, references only at the four call sites (+ the same four in the export mirror). The intended helper matches the established `ArtifactInfo(path, bytes, sha256)` construction used in `geometry_analysis.py:410` and `test_geometry_analysis.py:139`.

## Fix (correct abstraction boundary, no notebook/fallback changes)

- `src/representation/embedding_extraction.py`: import `ArtifactInfo` from `representation.geometry_contracts` and define module-level `_artifact_info(path, root) -> ArtifactInfo` next to `_sha256` (relative posix path, byte size, SHA-256). No notebook special case, no fake fallback.
- Helper audit of the whole finalization path (`extract()` tail, `_record`, `_regime_summary`, writers): `GeometryOutputPaths`/`GeometryManifest`/`GeometryRecord` imported; `_atomic_npz/_atomic_csv/_atomic_json/_atomic_bytes`, `_sha256`, `_record`, `iter_materialized`, `collate_variable_files`, `inference._move_batch` all defined/imported; `_record` touches only sample attributes + `_regime_summary`. `_artifact_info` was the sole undefined runtime symbol.
- Source-provenance behavior unchanged per the user's correction: the reported newest attached source (`.../anomaly-representation-20260903-01/src/src/representation/...`) is used as-is via the manual `SRC_DIR` variable; nothing added or reclassified.

## Regression (synthetic CPU smoke through the public path)

- New `tests/representation/test_extraction_finalization.py::test_extract_embeddings_publishes_complete_cache`: materializes a tiny real dataset (`DatasetBuilder(SynthConfig())`, 4/2/4 files), builds a tiny real model (6ch, d=8, 1 layer), fits a real `NormalReferenceBank` on model embeddings, saves a real checkpoint, then calls public `extract_embeddings(DiagnosticConfig(..., device="cpu"))` end to end. Asserts: 4 records all on `test`; all five artifacts + `figures/` on disk; NPZ keys/shapes (`embeddings` (4,8), `row_index` arange, `S_pred`/`S_pop` finite); manifest `records_count` 4, `splits` {"test": 4}, 4 file entries with byte-exact size + SHA-256 linkage; returned `paths`/`manifest` linkage.
- Pre-fix proof: with the helper renamed away, the test fails with exactly `NameError: name '_artifact_info' is not defined` at `embedding_extraction.py:362` (user's error reproduced); restored byte-identical, test passes. This is a genuine end-to-end public-extraction smoke (real generation, real forward, real finalization) — no mocks on the finalization path, no notebook cells, no real Kaggle data/checkpoint.

## Focused results

- `test_extraction_finalization.py` + `test_device_placement.py` + `test_normalization.py` + `test_geometry_contracts.py` + `test_geometry_analysis.py` + `test_geometry_notebooks.py` → **26 passed, 1 skipped** (CUDA unavailable on CPU-only torch 2.10.0 host). Command: `python -m pytest tests/representation/test_extraction_finalization.py tests/representation/test_device_placement.py tests/representation/test_normalization.py tests/representation/test_geometry_contracts.py tests/representation/test_geometry_analysis.py tests/representation/test_geometry_notebooks.py -q` with `PYTHONDONTWRITEBYTECODE=1`. New test executed full public extraction twice (13.9 s cold incl. dataset generation, 2.8 s warm). No formatters/linters/project-wide suites.

## Packaging (manual paths/parity/model metadata preserved; no notebook changes)

- Changed files: `src/representation/embedding_extraction.py`, `tests/representation/test_extraction_finalization.py` (new), export mirror `src/representation/embedding_extraction.py`, `src.zip`, `export-manifest.json`, outer ZIP, this file.
- Repo↔export `src/*.py` parity re-verified (52 files); notebook parity preserved (canonical==kaggle==export-root==export-mirror for all three gate notebooks).
- Zero `__pycache__`/`.pyc`/`.pyo` in export dir, `src.zip` (52 members), outer ZIP (65 entries) with `PYTHONDONTWRITEBYTECODE=1` and no export imports; manifest equals tracked disk (64 files); inner `src.zip` hash matches disk.

## Final artifacts

- `exports/kaggle-20260905-01/src.zip`: 108194 bytes, SHA-256 `1df15bd8cc005e60f9a87ab9bcfca98b9a71ab98fb1792f4920935e20a4271e6`.
- `exports/kaggle-20260905-01.zip`: 245625 bytes, SHA-256 `67951e433891d1739db36612e919b99afe137af4ec4b6d86542ab9a8f5e528e5`.
