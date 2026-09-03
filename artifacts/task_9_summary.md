# Task 9 Summary — Masked Latent-Prediction Criterion

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 9 marked `[x]`; Task 10 marked `[~]`).

## Changed targets

- `src/representation/criterion.py`: added `LatentPredictionCriterion` using the legacy dict-loss convention. It intersects requested/prediction masks with `patch_valid_mask`, computes per-patch latent MSE, reduces only valid masked patches, detaches targets at the criterion boundary, and returns aggregate loss, per-patch errors, total masked count, and per-file masked counts. Empty-mask behavior is explicit via `empty_policy='zero'` (default) or `'error'`.
- `src/representation/__init__.py`: exported `LatentPredictionCriterion`.
- `tests/representation/test_prediction_criterion.py`: added focused coverage for masked reduction, invalid/visible/padded exclusion, target stop-gradient, masked counts, and both empty-mask policies.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_prediction_criterion.py -q
```

Result: `3 passed in 5.09s`.

The focused suite verified exact masked normalization, target perturbation invariance outside the valid masked set, target gradient exclusion, finite empty-mask zero behavior, and explicit empty-mask errors. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- The criterion expects the model-output mapping to provide `predicted_latents`, `target_latents`, and `prediction_mask`; `patch_valid_mask` is optional only when the prediction mask is already intersected.
- Empty-mask default is a differentiable zero (`predicted.sum() * 0`) so training remains graph-safe without silently using visible or padded patches.
