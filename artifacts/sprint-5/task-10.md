# Task Summary: Bounded extraction and score validity

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 10

## Summary of Work
Reservoir selection now stores only deterministic source indices during a first pass and reopens the materialized split to stream selected files, avoiding retention of waveform-bearing `FileSample` objects. Extraction performs an unmasked inference pass for stable unprojected file embeddings, then a deterministic configured masking pass for non-degenerate `S_pred`; `S_pop` is computed from the unmasked embedding. Very short files force one valid prediction target when ratio rounding would otherwise produce an empty mask. Dataset compatibility validation now checks fleet robot/program cardinalities against conditional-normalization checkpoint limits before extraction.

## Files Modified
* `src/representation/embedding_extraction.py` — index-only reservoir selection, deterministic masking, unmasked embedding preservation, fleet cardinality checks.
* `tests/representation/test_geometry_contracts.py` — updated deterministic index-sampling fixture and fleet metadata fixture.
* `artifacts/sprint-5/task-10.md` — this evidence record.

## Testing
* `python -m pytest tests/representation/test_geometry_notebooks.py tests/representation/test_geometry_contracts.py -q` — 10 passed.
* `python -m py_compile src/representation/embedding_extraction.py src/representation/geometry_analysis.py` — passed.

## Additional Notes
No real checkpoint/dataset or model inference was executed. The two-pass masking behavior is covered structurally; external Kaggle execution remains required.
