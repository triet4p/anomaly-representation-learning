# Task 2 Summary — Full-File Dataset Adapter

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 2 marked `[x]`; Task 3 marked `[~]`).

## Changed targets

- `src/representation/data.py`: added lazy `FileDataset` over `FileSample` iterables, callables, or Sprint 1 materialized directories; worker-aware deterministic iteration; and `collate_variable_files()`.
- `src/representation/contracts.py`: minimally extended `RepresentationBatch` with optional evaluation metadata fields (`file_labels`, `anomaly_meta`, `anomaly_masks`) used by the adapter.
- `tests/representation/test_data.py`: added focused coverage for variable-length collation, C=3/C=6 batches, metadata retention, mixed-channel rejection, callable re-iteration, and Sprint 1 materialized ZIP round-trip.
- `tests/representation/test_contracts.py`: minimally corrected/extended the Task 1 fixture to validate actual C=3 and C=6 `FileSample` channel contracts; this was required to satisfy Task 1's stated acceptance and is within the authorized correction scope.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_data.py -q
```

Result: `6 passed in 4.80s`.

The focused suite verified that files remain complete, only minibatch padding is introduced, patch starts/valid lengths/padding masks are retained, fully padded patches are invalid and unmasked, evaluation metadata is preserved without entering the model mask, and a Sprint 1 materialized ZIP round-trips by ID and signal values. No project-wide validation, formatter, linter, or build was run.

## Deviations / risks

- A batch with mixed C=3 and C=6 files is rejected because a dense tensor cannot represent two channel axes. Both supported layouts are accepted in separate batches, as documented.
- Task 2 leaves the `mask` tensor as an explicit all-false placeholder; Task 3 owns replacing it with the configured random + information-aware + block masking bridge.
- The root `uv.lock` remains intentionally untouched under the current authorization boundary.
