# Task Summary: Diagnostic contracts

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 1 — Define diagnostic contracts

## Summary of Work
Added validated Pydantic contracts for deterministic split/sample selection, batch bounds, checkpoint/dataset compatibility metadata, projection and neighbor limits, portable artifact paths, per-file diagnostic records, and the versioned geometry manifest. Missing labels/scores are represented as explicit null values in the typed record and blank CSV fields. The fixed artifact names are centralized in the geometry API.

## Files Modified
* `src/representation/geometry_contracts.py` — configuration, result, record, manifest, artifact, and bounded diagnostic contracts.
* `src/representation/geometry.py` — public geometry contract/extraction exports.
* `tests/representation/test_geometry_contracts.py` — focused bound, schema, linkage, missing-metadata, and deterministic sampling tests.
* `docs/sprint-plans/sprint-5.md` — marked Tasks 1–2 complete; Task 3 remains pending.

## Testing
* **Test File:** `tests/representation/test_geometry_contracts.py`
* **Status:** Passed (6 tests)
* **Execution Command:** `python -m pytest tests/representation/test_geometry_contracts.py -q`

## Additional Notes
The contract schema is V1. Portable cache checksums cover generated artifacts other than the manifest itself, avoiding a circular self-checksum.
