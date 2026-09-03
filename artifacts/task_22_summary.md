# Task 22 Summary — Persisted Dataset Notebook Workflow

## Implementation

- `docs/SYNTH.md` now documents `uv` setup, production generation, the bounded `--small` profile, `--resume`, manifest and ZIP-shard layout, exact train/validation/test semantics, and the ignored `data/generated/` policy.
- Both representation notebooks resolve `V1_DATA_ROOT`, defaulting to `data/generated/production`. Relative roots are resolved from the repository root even when nbconvert starts the kernel from `notebooks/`.
- Both notebooks validate `manifest.json`, require complete `train`, `val`, and `test` split entries, and fail before model construction with actionable errors for missing, incomplete, or empty data.
- The train notebook reads persisted train and validation samples through `FileDataset`, trains on the persisted train batch with persisted validation, and fits its normal reference bank from train embeddings.
- The inference notebook reads persisted train, validation, and test samples through `FileDataset`, fits the reference bank from persisted train embeddings only, and scores persisted validation and test examples without passing labels to `NormalReferenceBank.fit`.
- Notebook source tests enforce stable non-empty unique IDs, persistent-root resolution, required split readers, and the absence of `SessionGenerator`, generator calls, and `DatasetBuilder`.

## Verification

Focused source test:

```text
uv run pytest tests/representation/test_notebooks.py -q
```

Result: `1 passed`.

Bounded persisted-data smoke (PowerShell environment variables):

```powershell
$env:PYTHONPATH = "src"
uv run python -m synth.cli --output data/generated/task22-smoke --small --shard-size 4 --seed 7 --overwrite
$env:V1_DATA_ROOT = "data/generated/task22-smoke"
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output task22-train-executed.ipynb notebooks/train_v1_representation.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output task22-infer-executed.ipynb notebooks/infer_v1_representation.ipynb
Remove-Item -LiteralPath data/generated/task22-smoke -Recurse -Force
Remove-Item -LiteralPath notebooks/task22-train-executed.ipynb,notebooks/task22-infer-executed.ipynb -Force
```

Observed results: the real CLI wrote `manifest.json` with `train=12 val=8 test=12`; both notebooks executed successfully against that persisted root and wrote executed notebook outputs; all temporary dataset and executed-notebook outputs were removed. The documented stable default is `data/generated/production`.
