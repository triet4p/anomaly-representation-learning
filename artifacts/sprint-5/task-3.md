# Task Summary: Latent health metrics

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 3 — Implement latent health metrics

## Summary of Work
Added CPU-only health metrics over cached numeric embeddings: stable float64 centering/covariance, clipped covariance eigen-spectrum, effective rank, anisotropy, norm summaries, rank and collapse indicators. Group summaries cover label, anomaly family, fleet, and regime with an explicit `<missing>` bucket and sample counts; no group value participates in embedding computation.

## Files Modified
* `src/representation/geometry_analysis.py` — cache-only health metrics and group summaries.
* `src/representation/diagnostics.py` — public diagnostics exports.
* `tests/representation/test_geometry_analysis.py` — collapsed, low-rank, anisotropic, healthy, and missing-label behavioral fixtures.
* `docs/sprint-plans/sprint-5.md` — marked Task 3 complete.

## Testing
* **Test File:** `tests/representation/test_geometry_analysis.py`
* **Status:** Added; not executed because Sprint 5 explicitly forbids executing diagnostics, including synthetic diagnostic runs, in this worker gate.
* **Static check:** `python -m py_compile src/representation/geometry_analysis.py src/representation/diagnostics.py`

## Additional Notes
Metrics are designed for the user-operated CPU analysis stage; scientific interpretation remains blocked until the external Kaggle gate.
