# Task Summary: Neighborhood diagnostic schema

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 11

## Summary of Work
Neighbor rows now persist query/neighbor `S_pred` and `S_pop`, plus post-hoc same/different relationships for label, fleet, regime summary, anomaly family, severity, and split. `analyze_geometry_cache` persists `neighbor_group_rates` in metrics. Supervised class similarity/margin fields are explicitly named `same_class_similarity`, `different_class_similarity`, and `class_separation_margin`; true augmented-view contrastive metrics are explicitly marked unavailable because the cache contains one unaugmented embedding per file.

## Files Modified
* `src/representation/geometry_analysis.py` — neighbor row/rate schema, metrics persistence, separation naming and availability marker.
* `tests/representation/test_geometry_analysis.py` — focused neighbor/separation schema assertions.
* `docs/sprint-plans/sprint-5.md` — marked Task 11 complete; Task 13 remains pending.

## Testing
* `python -m pytest tests/representation/test_geometry_analysis.py -q` — 7 passed.

## Additional Notes
Labels and provenance are consumed only after numeric distances/embeddings are computed; no labels influence neighbor selection or projection fitting.
