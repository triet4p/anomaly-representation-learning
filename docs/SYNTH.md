# Synthetic data subsystem

The generator treats each complete variable-length session as one semantic
sample. Patches are a computational view only, and ragged signals are never
flattened into a fixed-length production array.

The supported layouts are explicit: `C=3` (feed, current, temperature) for
legacy fixtures and `C=6` for production (feed, current, temperature,
arc-voltage, arc-power, torch-pressure). The default is coherent `C=6`; all
six channels share regime timing and causal process factors.

## Environment and imports

From the repository root, install the project into the local `uv` environment:

```powershell
uv sync
$env:PYTHONPATH = "src"
uv run python -c "from synth import SessionGenerator, SynthConfig; print(SessionGenerator(SynthConfig()).generate_normal(0).x.shape)"
```

All generation is deterministic for a fixed seed, split, severity, and
configuration. IDs and seeds do not depend on shard size or resume order.

## Persisted dataset workflow

Run the `synth.cli` generator for the production dataset. Its default profile writes
100,000 normal train files, 20,000 balanced validation files, and 100,000
balanced test files:

```powershell
$env:PYTHONPATH = "src"
uv run python -m synth.cli --output data/generated/production --shard-size 512 --seed 7
```

For a bounded local smoke dataset (12 train, 8 validation, and 12 test files):

```powershell
$env:PYTHONPATH = "src"
uv run python -m synth.cli --output data/generated/smoke --small --shard-size 4 --seed 7 --overwrite
```

An interrupted run can continue only when its output manifest, resolved
configuration, counts, and shard size match:

```powershell
$env:PYTHONPATH = "src"
uv run python -m synth.cli --output data/generated/production --shard-size 512 --seed 7 --resume
```

Each dataset root contains one `manifest.json` and split directories with
deterministic ZIP shards:

```text
data/generated/<name>/
├── manifest.json
├── train/shard-00000.zip
├── val/shard-00000.zip
└── test/shard-00000.zip
```

The manifest records the resolved configuration, counts, split accounting,
shard ranges and member IDs, rejection provenance, and SHA-256 archive hashes.
`iter_materialized(root, split=...)` verifies each archive and streams complete
`FileSample` objects; it does not load the entire dataset into memory.

Train is normal-only. Validation and test are balanced normal/abnormal
evaluation splits. Training and reference-bank fitting must read the persisted
train split only; validation and test are evaluation examples and their labels
must not be used to construct a normal reference bank.

The representation notebooks resolve their dataset root from
`V1_DATA_ROOT`, defaulting to `data/generated/production`:

```powershell
$env:V1_DATA_ROOT = "data/generated/smoke"
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 notebooks/train_v1_representation.ipynb
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 notebooks/infer_v1_representation.ipynb
```

Both notebooks fail before model construction with an actionable error if the
root lacks `manifest.json` or any of the required `train`, `val`, and `test`
split entries. Keep generated roots under `data/generated/`; that path is
ignored by Git and no generated dataset is checked in.

## Generate sessions

```python
from synth import SessionGenerator, SynthConfig
from synth.schema import AnomalyFamily

cfg = SynthConfig()
gen = SessionGenerator(cfg)
normal = gen.generate_normal(seed=10, split="train")
anomaly = gen.generate_anomaly(
    seed=20,
    split="test",
    family=AnomalyFamily.CROSS_CHANNEL_INCONSISTENCY,
    severity=0.7,
)
normal.validate()
anomaly.validate()
```

Hard families are selected by default. Easy spike and flatline checks are deliberately opt-in:

```python
cfg.anomaly.include_easy_sanity = True
sample = gen.generate_anomaly(21, family="easy_spike")
```

An accepted anomaly has `file_label == SampleLabel.ABNORMAL`, an exact `[C, T]` `anomaly_mask`, and complete `AnomalyMeta`. A strength-gate rejection is returned as a normal-labelled sample with `rejection_meta` containing family, region, measured strength, and attempt count; dataset materialization retries deterministically instead of silently changing contamination.

## Patches and masking

```python
from synth.patchify import Patchifier
from synth.masking import apply_masking
import numpy as np

batch = Patchifier(cfg.patch).patchify(normal)
patch_anomaly = batch.timestep_to_patch_mask()
mask = apply_masking(batch, cfg.masking, np.random.default_rng(7))
```

`PatchBatch.starts`, `valid_len`, and `pad_mask` preserve variable-length boundaries. Padding is never included in anomaly or patch statistics. Masking assigns a fixed total ratio across disjoint random, information-aware (stratified), and contiguous block components, so ablations change composition rather than the amount masked.

## Splits and sharded NPZ files

Production defaults are explicit: train has 100,000 normal samples;
validation has 20,000 balanced samples (10,000 normal and 10,000 abnormal);
test has 100,000 balanced samples (50,000 normal and 50,000 abnormal).
Counts are exact: rejected anomaly attempts are retried with deterministic
disjoint seeds, and generation fails with rejection provenance if the retry
budget is exhausted.

For a bounded-memory small run:

```powershell
$env:PYTHONPATH = "src"
uv run python -m synth.cli --output .\data\generated\smoke --small --shard-size 4 --seed 7 --overwrite
uv run python -c "from synth.dataset import iter_materialized; print(sum(1 for _ in iter_materialized('data/generated/smoke')))"
```

The writer creates `manifest.json` and one ZIP archive per bounded shard under
each split. Each archive contains compressed NPZ members, preserving ragged
`[C,T]` signals without one filesystem file per sample. Members keep masks,
regimes (including levels/frequency/phase), anomaly metadata, rejection
metadata, seed, ID, and config identity. The manifest records the resolved
configuration, requested/accepted normal and abnormal counts, rejection
attempt totals/by-family, shard ranges, member IDs, and archive hashes.
Every archive and manifest update is atomic. Use `--resume` to verify hashes and
continue an interrupted run; incompatible configuration, counts, or shard size
are rejected. `iter_materialized` streams one member at a time.

The Python API is:

```python
builder = DatasetBuilder(cfg)
builder.materialize_sharded("data/generated/production", shard_size=512)
```

## Quality experiment

Run the deterministic local review experiment:

```powershell
python experiments/quality_check.py
```

It generates all nine hard families, normal and anomaly plots with regime boundaries and masks, patch-statistic plots, contrastive-view plots, weak-statistics AUCs, and a machine-readable report at `experiments/artifacts/metrics.json`. Generated artifacts are ignored by Git.

Focused behavioral tests:

```powershell
python -m pytest tests/synth/test_six_channel_shards.py -q
python -m pytest tests/synth/test_diagnostics.py -q
```
