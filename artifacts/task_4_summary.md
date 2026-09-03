# Task 4 Summary — Local Patch Encoder

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 4 marked `[x]`; Task 5 marked `[~]`).

## Changed targets

- `src/representation/layers/patch_encoder.py`: added `LocalPatchEncoder`, a channel-aware `[B,N,C,W] -> [B,N,D]` encoder with pre-convolution padding masking, valid-timestep mean pooling, runtime `N`/`W`, and zero output for fully padded patches.
- `src/representation/layers/__init__.py`: exported `LocalPatchEncoder`.
- `tests/representation/test_patch_encoder.py`: added focused coverage for C=3/C=6, runtime patch counts, finite outputs, padded-value invariance, all-padded behavior, and invalid shape/configuration errors.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_patch_encoder.py -q
```

Result: `4 passed in 6.53s`.

The focused suite verified both supported channel layouts, variable runtime patch counts, finite embeddings, invariance to altered padded values, fully padded zero embeddings, and input validation. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- Dense model batches still require one channel count per batch; C=3 and C=6 are supported independently.
- The encoder deliberately exposes no raw reconstruction head; sequence context integration is Task 5.
