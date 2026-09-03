# Task 13 Summary — V1 Checkpoint and Reference-Bank Serialization

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 13 marked `[x]`; Task 14 marked `[~]`).

## Changed targets

- `src/representation/checkpoint.py`: added atomic `save_checkpoint`/`load_checkpoint` and `CheckpointManager`. Payloads include schema version, validated V1 config, complete model state (context/EMA/predictor), optional optimizer and scheduler state, training step, and optional normal reference-bank state. Loading validates required keys/schema/config/step before strict model restoration; temporary files are replaced atomically and cleaned up on failure.
- `src/representation/__init__.py`: exported checkpoint APIs.
- `tests/representation/test_checkpoint.py`: added focused coverage for model/EMA/optimizer/scheduler/step/bank round-trip, missing/incompatible metadata, missing optimizer state, and atomic replacement safety under serialization failure.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_checkpoint.py -q
```

Result: `3 passed in 5.09s` (PyTorch emitted only the existing nested-tensor optimization warning from the sequence encoder).

The focused suite verified state/output round-trip, EMA and optimizer restoration, step and reference-bank persistence, incompatible/incomplete payload failures, and retention of a valid existing checkpoint when a replacement serialization fails. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- Checkpoint loading requires a model constructed with the same validated `V1Config`; incompatible channel/model geometry is rejected before state loading.
- Optimizer/scheduler restoration is required only when corresponding objects are supplied to `load_checkpoint`; omitted state remains optional.
- `torch.load(..., weights_only=False)` is used because the payload intentionally contains optimizer/scheduler metadata and configuration.
