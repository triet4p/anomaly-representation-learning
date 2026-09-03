# Task 5 Summary — Variable-Length Sequence/Context Encoder

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 5 marked `[x]`; Task 6 marked `[~]`).

## Changed targets

- `src/representation/layers/sequence_encoder.py`: added `SequenceContextEncoder` with runtime positional encodings, Transformer context layers, patch-valid key padding, safe handling for fully padded rows, and zeroed invalid outputs.
- `src/representation/layers/__init__.py`: exported `SequenceContextEncoder` alongside `LocalPatchEncoder`.
- `tests/representation/test_sequence_encoder.py`: added focused coverage for runtime sequence lengths, padded-token invariance, finite output, all-padded rows, and validation errors.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_sequence_encoder.py -q
```

Result: `3 passed in 6.58s` (PyTorch emitted the expected nested-tensor optimization warning because the encoder uses pre-norm layers).

The focused suite verified runtime patch counts, key-padding exclusion, invariance of valid outputs to padded-token perturbations, finite fully-padded handling, and input validation. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- The encoder uses sinusoidal positions generated at runtime; there is no fixed maximum patch count.
- Fully padded rows use a temporary unmasked sentinel key solely to avoid all-masked attention NaNs, then return zeros for every invalid position.
- File pooling and prediction integration remain downstream tasks.
