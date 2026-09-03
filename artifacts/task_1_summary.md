# Task 1 Summary — V1 Configuration and Contracts

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 1 marked `[x]`; Task 2 marked `[~]`).

## Changed targets

- `pyproject.toml`: declared `torch>=2.11.0`, `pydantic>=2.0`, and `einops>=0.8.1`.
- `src/representation/__init__.py`: exported the V1 configuration and runtime contract helpers.
- `src/representation/config.py`: added `V1Config` with Pydantic v2 validation for supported channel counts, patch geometry, model dimensions, mask composition, EMA, objective, and kNN settings. No semantic file-length field is present.
- `src/representation/contracts.py`: added typed `RepresentationBatch` and `RepresentationOutput` contracts plus runtime validation for tensor axes/dtypes, valid/padded masks, masked-valid invariants, and stop-gradient target latents.
- `tests/representation/test_contracts.py`: added focused behavioral coverage for valid C=3/C=6-compatible configuration, invalid settings, batch shape/mask invariants, and target stop-gradient enforcement.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_contracts.py -q
```

Result: `5 passed in 39.23s`.

The command used an isolated uv environment because the repository environment did not yet contain PyTorch; it did not run project-wide tests, builds, formatters, or linters.

## Risks / blockers

- The root `uv.lock` was intentionally not modified because Task 1 authorization is limited to the listed targets; a later dependency synchronization step may need to update it under explicit authorization.
- The new contracts intentionally describe model-side padded minibatches; the dataset/collation implementation is Task 2 and remains in progress.
