# Task Summary: Kaggle path and publication contracts

**Sprint:** Sprint 5 — Latent Geometry Diagnostics
**Task:** Task 9

## Summary of Work
Stage 2 now treats the attached cache as read-only: `analyze_geometry_cache` accepts a separate writable `output_dir`, writes metrics/neighbors/figures there, and only refreshes the input manifest when output and input are intentionally identical. Both notebooks resolve source from `/kaggle/working/src`, local `src/`, or attached `/kaggle/input/*/src`; the guide documents attaching source and user-owned datasets. CLI publication instructions initialize dataset metadata before create/version, and CPU/GPU metadata files are present.

## Files Modified
* `src/representation/geometry_analysis.py` — read-only cache/output directory split.
* `notebooks/geometry_extraction.ipynb`, `notebooks/geometry_analysis.ipynb` — portable source discovery and writable output parameter.
* `notebooks/kaggle/geometry_extraction.ipynb`, `notebooks/kaggle/geometry_analysis.ipynb` — synchronized copies.
* `exports/kaggle-20260904-04/RUN_GUIDE.md` — source attachment and corrected CLI metadata initialization.
* `exports/kaggle-20260904-04/kernel-metadata*.json` — explicit GPU/CPU gate metadata.
* `tests/representation/test_geometry_notebooks.py` — static source/clean-state boundary checks.

## Testing
* `python -m pytest tests/representation/test_geometry_notebooks.py tests/representation/test_geometry_contracts.py -q` — 10 passed.
* `python -m py_compile src/representation/embedding_extraction.py src/representation/geometry_analysis.py` — passed.

## Additional Notes
No real cache, notebook cell, inference, or diagnostic run was executed.
