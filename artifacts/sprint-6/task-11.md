# Sprint 6 Task 11 — Correct execution evidence

- Recounted `geometry-analysis/figures/` on the server: **12 PNGs** (pca/tsne × anomaly_family, fleet, regime_summary, severity, split, plus uncolored base). The earlier "8" came from a truncated `ls | head` listing.
- `artifacts/sprint-6/task-6.md` rewritten: corrected rerun (commit `a326e48`, exit 0, 20.2 s), 12 figures, rank 127/128 as near-full, `normal_reference_distance` scoped to within-cache normals with self-exclusion, `same_class_similarity` as capped within-cache similarity, checkpoint-bank distance moved to a separate `S_pop` statement (mean 7.3446 ± 0.7053), all-normal limitation kept explicit.
- `artifacts/sprint-6/task-7.md` rewritten: same corrections plus `S_pred` provenance (mean 0.0735 ± 0.0306), corrected pair counts (same 4999 / different 0), rerun commit/timings, and `server-geometry-run-1-prev/` preservation note.
- No source, notebook, test, or data changes in this task; corrections are evidence-only and suspicious numbers were re-derived from server artifacts (figure listing, cache NPZ score stats, refreshed `metrics.json`), not edited by hand.
