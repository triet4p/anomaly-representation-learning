# Task Summary: Kaggle extraction notebook

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 6 — Build Kaggle extraction notebook

## Summary of Work
Created synchronized canonical and Kaggle copies of a clean unexecuted Stage 1 GPU notebook. It exposes dataset/checkpoint/output roots, selected/reference splits, sample/reference bounds, batch size, seed, sampling strategy, and device. It imports the packaged `representation.geometry` API, constructs the validated diagnostic configuration, runs extraction only in the explicit user-run cell, and verifies portable cache outputs and schema metadata for handoff.

## Files Modified
* `notebooks/geometry_extraction.ipynb` — canonical parameterized Stage 1 gate.
* `notebooks/kaggle/geometry_extraction.ipynb` — byte-synchronized Kaggle copy.
* `tests/representation/test_geometry_notebooks.py` — static structure, clean execution state, compilation, synchronization, and parameter gate checks.
* `docs/sprint-plans/sprint-5.md` — marked Task 6 complete.

## Testing
* **Test File:** `tests/representation/test_geometry_notebooks.py`
* **Status:** Passed (4 tests)
* **Execution Command:** `python -m pytest tests/representation/test_geometry_notebooks.py -q`
* **Verification:** all code cells compiled without execution; outputs are empty and execution counts are null; canonical/Kaggle JSON is identical.

## Additional Notes
No checkpoint, dataset, model inference, extraction, diagnostics, or notebook cell was executed.
