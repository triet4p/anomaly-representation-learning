# Task Summary: Bounded embedding extraction

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 2 — Implement bounded embedding extraction

## Summary of Work
Implemented `BoundedEmbeddingExtractor` and the `extract_embeddings` entry point. Checkpoint loading reconstructs and strictly restores a V1 model, requires a finite fitted normal reference bank, validates reference dimensions and patch geometry, and rejects incompatible dataset manifest channel metadata. Extraction consumes selected materialized splits once, supports deterministic head or seeded reservoir selection, keeps only one configured batch of raw `FileSample` values during forward execution, runs eval/inference mode, and detaches outputs immediately. It publishes deterministic numeric NPZ arrays (`embeddings`, `row_index`, `S_pred`, `S_pop`), provenance-rich records, placeholder metrics/neighbors files for the later analysis stage, and an atomic versioned manifest containing source identities and SHA-256 checksums.

## Files Modified
* `src/representation/embedding_extraction.py` — checkpoint-safe, bounded streaming extractor and atomic artifact writers.
* `src/representation/inference.py` — exposes the already-computed unprojected file embedding alongside independent scores, avoiding a second model forward.
* `src/representation/geometry.py` — public extraction API.
* `tests/representation/test_geometry_contracts.py` — deterministic bounded sampler and byte-stable numeric NPZ writer checks.
* `docs/sprint-plans/sprint-5.md` — marked Tasks 1–2 complete; Task 3 remains pending.

## Testing
* **Test File:** `tests/representation/test_geometry_contracts.py`
* **Status:** Passed (6 tests)
* **Execution Command:** `python -m pytest tests/representation/test_geometry_contracts.py -q`
* **Additional focused check:** `python -m pytest tests/representation/test_inference.py -q` — Passed (3 tests).
* **Static check:** `python -m py_compile src/representation/geometry_contracts.py src/representation/embedding_extraction.py src/representation/geometry.py src/representation/inference.py` — Passed.

## Additional Notes
Per Sprint 5 gate constraints, no real checkpoint, materialized dataset, model inference, embedding extraction, geometry diagnostic, or notebook cell was executed. The extractor is intended for the user-operated Stage 1 gate.
