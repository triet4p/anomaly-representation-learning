# Task 10 Summary — File Contrastive Criterion and Progressive Lambda

## Status

Complete (`docs/sprint-plans/sprint-2.md` Task 10 marked `[x]`; Task 11 marked `[~]`).

## Changed targets

- `src/representation/criterion.py`: added `FileContrastiveCriterion`, symmetric in-batch InfoNCE over L2-normalized same-file view embeddings, with diagonal same-file positives, unrelated in-batch negatives, positive temperature validation, and finite batch-size-one behavior. Added `ProgressiveLambda` with exact zero-at-step-zero, monotonic linear ramp, endpoint clamping, and validation.
- `src/representation/__init__.py`: exported `FileContrastiveCriterion` and `ProgressiveLambda`.
- `tests/representation/test_contrastive_criterion.py`: added focused coverage for normalization, same-file diagonal/in-batch objective, metadata non-use, batch-size-one stability, temperature validation, schedule endpoints, monotonicity, and invalid schedule values.

## Verification

Command:

```text
uv run --no-project --with 'torch>=2.11.0' --with 'pydantic>=2.0' --with 'einops>=0.8.1' --with 'pytest>=9.1.1' env PYTHONPATH=src python -m pytest tests/representation/test_contrastive_criterion.py -q
```

Result: `3 passed in 4.86s`.

The focused suite verified normalized same-file positives, finite in-batch contrastive loss and gradients, explicit batch-size-one handling, temperature validation, and exact monotonic lambda ramp endpoints. No project-wide tests, formatter, linter, or build was run.

## Risks / blockers

- Batch-size-one contrastive loss is an explicit differentiable zero because no in-batch negative exists; larger batches use symmetric InfoNCE.
- Positive pairs are determined solely by view row alignment; metadata is accepted only as an ignored mapping entry.
