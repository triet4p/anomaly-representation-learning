# Task 7 Summary — Masked Latent Predictor

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 7 marked `[x]`; Task 8 marked `[~]`).

## Changed targets

- `src/representation/layers/predictor.py`: added `MaskedLatentPredictor`, which combines visible contextual patch states, visible-context pooling, runtime sinusoidal positions, and mask context to predict target-dimensional latents. It returns `[B,N,target_dim]` predictions and an aligned `requested_mask & patch_valid_mask`; visible and invalid positions are zeroed and cannot contribute to prediction loss.
- `src/representation/layers/__init__.py`: exported `MaskedLatentPredictor` and the shorter `LatentPredictor` alias.
- `tests/representation/test_predictor.py`: added focused coverage for target dimensions, positional/alignment-preserving masks, masked/valid selection, finite outputs, visible-context gradient routing, and input validation.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_predictor.py -q
```

Result: `3 passed in 5.25s`.

The focused suite verified aligned target-dimensional predictions, intersection of requested and valid masks, no output/loss contribution from visible or invalid positions, gradient flow through visible context only, and invalid-input handling. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- The head returns a prediction for every aligned position but zeros visible/invalid positions; callers must use the returned prediction mask when computing latent loss.
- The context encoder is responsible for producing context states with masked-region leakage prevented; this head additionally excludes requested masked states from its visible-context pool.
