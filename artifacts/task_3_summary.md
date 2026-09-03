# Task 3 Summary — Fixed-Ratio Masking Bridge

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 3 marked `[x]`; Task 4 marked `[~]`).

## Changed targets

- `src/representation/masking.py`: added `apply_batched_masking()` and V1-to-Sprint-1 `MaskingConfig` conversion. Each padded torch row is adapted to `PatchBatch`, delegated to `synth.masking.apply_masking()`, and restored as a device-local boolean mask with composition and ratio diagnostics.
- `src/representation/data.py`: added the optional typed masking hook to `collate_variable_files()`; default data-only collation still emits an explicit all-false mask.
- `src/representation/contracts.py`: minimally extended `RepresentationBatch` with `file_samples`, `mask_composition`, and `mask_ratio` fields needed by the bridge.
- `tests/representation/test_masking_bridge.py`: added focused tests for exact ratio/counts, reproducibility, fixed-count ablations, contiguous block selection, collator hook behavior, and required metadata.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_masking_bridge.py -q
```

Result: `6 passed in 6.76s`.

The focused suite verifies that random, information-aware, block, and combined policies preserve the same requested masked count, never mask invalid/padded patches, expose auditable composition and ratio values, remain seed-reproducible, and integrate through the data collator. No project-wide tests, formatter, linter, or build was run.

## Deviations / risks

- The existing Sprint 1 NumPy policy remains the source of truth; this task adds only a torch batch adapter and does not change `src/synth/masking.py`.
- Dense batches still require one channel layout per batch (C=3 or C=6); this is unchanged from Task 2.
- The bridge requires `file_samples` metadata so each row can satisfy the existing `PatchBatch` contract; Task 2 now retains it explicitly.
