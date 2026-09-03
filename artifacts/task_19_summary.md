# Task 19 Summary — Project-Mode Lock and Test Reproducibility

## Status

Complete. The project lock and project-mode representation test path now resolve the declared dependencies and discover the `src` package.

## Changed targets

- `uv.lock`: regenerated from `pyproject.toml`, adding the declared `torch`, `pydantic`, and `einops` dependency graph.
- `pyproject.toml`: added pytest `pythonpath = ["src"]` so project-mode `uv run pytest` discovers the src-layout package without ad hoc `PYTHONPATH`.
- `tests/representation/test_checkpoint.py`: updated state round-trip assertions to account for the model's serialized augmentation RNG extra state.
- `docs/sprint-plans/sprint-2.md`: appended completed Atomic Task 19.
- `docs/PLAN.md`: records Sprint 2 complete including Tasks 17–19 remediation.

## Verification

Lock regeneration:

```text
uv lock
```

Observed: resolved 152 packages and added the declared PyTorch/Pydantic/Einops dependency graph.

Lock consistency:

```text
uv lock --check
```

Observed: resolved 152 packages successfully.

Project-mode suite:

```text
uv run pytest tests/representation -q
```

Observed: `56 passed, 1 skipped, 19 warnings in 9.36s`. The one skip is conditional CUDA coverage; warnings are existing PyTorch nested-tensor warnings.

## Residual risks

- The project uses a src layout without a build-system package declaration; pytest's explicit `pythonpath` configuration is the minimal project-mode discovery fix.
- CUDA remains conditionally skipped where unavailable.
- No unrelated synth suite was run under this gate.
