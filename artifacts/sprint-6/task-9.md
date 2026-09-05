# Sprint 6 Task 9 — Restore independent manifest validation

## Change (`src/representation/embedding_extraction.py::_manifest`)

- Restored two independent checks that the earlier reconstruction had dropped: the manifest `format` must equal the supported value `2`, and `splits` must be a non-empty mapping — otherwise `GeometryCompatibilityError` (missing/unreadable/non-mapping manifests already raised).
- Relaxed **only** the missing-legacy-`resolved_config.fleet` case (returns the recorded `skipped` precheck from Task 6's commit); missing `resolved_config`/`n_channels`, channel mismatch, missing/invalid fleet fields, and over-limit cardinalities still raise. Per-batch fleet-range guards in `ConditionalBatchNorm._bucket_ids` are untouched.
- Verified against the real server manifests: `data/generated/production` and `medium` both carry `format: 2` with complete split mappings, so both pass; their missing `fleet` key takes the recorded-skip path.

## Coverage

- `tests/representation/test_geometry_contracts.py::test_manifest_requires_supported_format_and_splits_mapping`: format 1 / missing format / missing / list / empty splits all raise `GeometryCompatibilityError`; a legacy no-fleet manifest loads. (Also repaired two self-inflicted edit faults in the same file: a dropped `numpy` import and a duplicated `pytest` line.)
- Focused run with the Task 8 + notebook suites: 39 passed, 1 skipped (CUDA unavailable).
