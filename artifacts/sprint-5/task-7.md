# Task Summary: Kaggle analysis notebook

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 7 — Build Kaggle analysis notebook

## Summary of Work
Created synchronized canonical and Kaggle copies of a clean unexecuted Stage 2 CPU notebook. It exposes cache/output roots and bounded neighbor/PCA/t-SNE parameters with independent seeds, imports only the cache-only diagnostics API, validates checksums and row linkage before analysis, writes metrics and static figures, and packages cache-linked results as a downloadable ZIP. The notebook documents the Stage 1 boundary, missing/corrupt cache failure message, reproducibility metadata, and post-hoc label policy.

## Files Modified
* `notebooks/geometry_analysis.ipynb` — canonical parameterized Stage 2 gate.
* `notebooks/kaggle/geometry_analysis.ipynb` — byte-synchronized Kaggle copy.
* `tests/representation/test_geometry_notebooks.py` — static cache-only boundary, structure, clean state, compilation, parameter, and synchronization checks.
* `docs/sprint-plans/sprint-5.md` — marked Task 7 complete; Task 8 remains pending.

## Testing
* **Test File:** `tests/representation/test_geometry_notebooks.py`
* **Status:** Passed (4 tests)
* **Execution Command:** `python -m pytest tests/representation/test_geometry_notebooks.py -q`
* **Verification:** all code cells compiled without execution; outputs are empty and execution counts are null; canonical/Kaggle JSON is identical; forbidden model/checkpoint loading strings are absent from Stage 2 code cells.

## Additional Notes
No checkpoint, dataset, model inference, extraction, diagnostics, or notebook cell was executed.
