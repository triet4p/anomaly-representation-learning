# Task 14 Summary — Isolated Optional RVQ Baseline

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 14 marked `[x]`; Task 15 marked `[~]`).

## Changed targets

- `src/representation/baselines/rvq.py`: added opt-in `RVQBaseline` with provenance to the legacy `ResidualFactorizedVectorQuantizerEMA`, narrowly implementing the documented `[B,C,D]` residual quantization interface and returning quantized latents, commitment loss, and per-level indices. No decoder or raw reconstruction verdict is present.
- `src/representation/baselines/__init__.py`: exported only the isolated baseline package API; the primary `representation` package and `V1RepresentationModel` do not import or instantiate it.
- `tests/representation/test_rvq_baseline.py`: added focused coverage for explicit construction, C=3/C=6 shape contracts, finite diagnostics, deterministic repeated evaluation, invalid shapes, and reconstruction-head absence.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_rvq_baseline.py -q
```

Result: `2 passed in 3.39s`.

The focused suite verified opt-in baseline construction and smoke-forwarding for both supported channel layouts, `[B,C,D]` output/indices, finite commitment diagnostics, deterministic eval behavior, invalid-input rejection, and absence of reconstruction outputs. No project-wide tests, formatter, linter, or build was run.

## Deviations / risks

- The legacy tree remains read-only; the implementation is a narrow local equivalent rather than a runtime import from the legacy repository.
- The baseline is intentionally not exported from the primary `representation` package, not composed into V1, and not used by the default criterion/trainer/inference paths.
