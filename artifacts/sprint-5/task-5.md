# Task Summary: Deterministic geometry views

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 5 — Implement deterministic geometry views

## Summary of Work
Added bounded seeded projection sampling, full-SVD PCA with explained variance, and sampled scikit-learn t-SNE with deterministic PCA initialization and bounded perplexity. Added publication-readable non-interactive PNG rendering for PCA and t-SNE, with sample counts, axes, legends, and post-hoc label coloring. No UMAP dependency was added.

## Files Modified
* `src/representation/geometry_analysis.py` — PCA/t-SNE projections, figure rendering, and cache analysis orchestration.
* `src/representation/diagnostics.py` — public projection exports.
* `tests/representation/test_geometry_analysis.py` — deterministic bounded PCA/t-SNE fixture.
* `docs/sprint-plans/sprint-5.md` — marked Task 5 complete; Task 6 remains pending.

## Testing
* **Test File:** `tests/representation/test_geometry_analysis.py`
* **Status:** Added; not executed because Sprint 5 explicitly forbids executing diagnostics in this worker gate.
* **Static check:** `python -m py_compile src/representation/geometry_analysis.py src/representation/diagnostics.py`

## Additional Notes
Projection fitting receives only the sampled numeric embedding matrix. Labels affect figure coloring only and cannot leak into PCA or t-SNE fitting.
