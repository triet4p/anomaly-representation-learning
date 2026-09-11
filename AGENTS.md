# Repository Guidelines

## Project Overview

Self-supervised representation learning for anomaly detection on variable-length
multi-channel synthetic telemetry (`C=6`). Pipeline: deterministic synthetic
data (`src/synth/`) → V1 patch/context encoder with EMA latent prediction +
file-level contrastive objective (`src/representation/`) → independent anomaly
scores `S_pred` (context mismatch) and `S_pop` (population mismatch). No raw
waveform reconstruction in V1. Workflows run as headless notebooks executed via
`nbconvert`, locally and on a GPU server through a commit → push → pull loop.

## Architecture & Data Flow

- `src/synth/`: Normal signal → regime envelope → anomaly injection →
  noise/sensor artifacts → `FileSample` (`x: [C, T]`, variable length).
- `src/representation/`: conditional normalization (robot/program-aware) →
  patchify (fixed-stride windows) → `LocalPatchEncoder` → `SequenceContextEncoder`
  (Transformer) → masked latent prediction + contrastive file embedding; EMA
  target encoder, stop-gradient targets.
- File → patches (compute units only, keep lengths/masks/starts/file IDs) →
  file embedding + `S_pred`/`S_pop` → geometry cache → metrics/figures.
- Labels are post-hoc diagnostics only, never model inputs.

## Key Directories

- `src/synth/`: generator, regimes, `anomalies/registry.py`, `patchify.py`,
  `dataset.py`, `cli.py`, Pydantic v2 configs.
- `src/representation/`: `model.py`, `layers/normalization.py`,
  `embedding_extraction.py`, `geometry_analysis.py`, `inference.py`,
  `geometry_contracts.py`, Pydantic v2 configs.
- `notebooks/`: canonical `train_`, `infer_`, `geometry_extraction`,
  `geometry_analysis` — explicit first-cell paths, fail-fast, cells unexecuted.
- `tests/`: focused pytest suites mirroring `src/` layout.
- `experiments/<date>/<run>/`: executed notebooks, logs, manifests, metrics,
  `assets/` figures, per-run `README.md`.
- `checkpoints/`: trained weights (never committed); `data/generated/`: sharded
  splits (never committed).
- `docs/PLAN.md`, `docs/sprint-plans/sprint-N.md`: planning source of truth.

## Development Commands

```bash
uv run python -m synth.cli            # generate data -> data/generated/production
uv run -m pytest tests/<path> -q      # focused tests (default; never full suite unasked)
PYTHONDONTWRITEBYTECODE=1 uv run -m pytest tests/...   # when asserting cache-free trees
uv run --no-sync jupyter nbconvert --to notebook --execute notebooks/<nb>.ipynb \
  --output-dir <run> --output <name>.executed.ipynb --ExecutePreprocessor.timeout=-1
```

Key env vars: `V1_DATA_ROOT`, `V1_CHECKPOINT_PATH` / `V1_CHECKPOINT`,
`V1_GEOMETRY_CACHE`, `V1_GEOMETRY_ANALYSIS`. Paths resolve against `Path.cwd()`;
run from the repo root.

## Code Conventions & Common Patterns

- Pydantic v2 for all configs; tensor work internal PyTorch; `einops` only
  where shapes get clearer.
- Smallest coherent change; reuse existing patterns; no second convention
  beside existing code; boring over clever.
- Device invariant: every index/accumulator/source in a reduction shares the
  tensor's device (`.to()` is a no-op when aligned — no avoidable copies).
- Fail fast with `FileNotFoundError` naming the variable to fix; never silent
  fallbacks, random-weight defaults, or mount scanning.
- Reference-bank rule: a restored checkpoint bank is used as-is unless an
  explicit refit flag is set; every `S_pop` states bank rows + source.
- Never import from copied/export trees during verification (bytecode
  pollution); use AST/text/archive inspection.

## Important Files

- `src/synth/cli.py`, `src/synth/config.py`, `src/synth/schema.py`
- `src/representation/model.py`, `config.py`, `inference.py`,
  `embedding_extraction.py`, `geometry_analysis.py`, `geometry_contracts.py`
- `pyproject.toml` (requires Python ≥ 3.12), `uv.lock`, `.python-version`
- `notebooks/infer_v1_representation.ipynb` (fail-fast checkpoint contract)

## Runtime/Tooling Preferences

- Runtime: Python ≥ 3.12 via `uv` (locked by `uv.lock`); server runs use
  `uv run --no-sync` (system python lacks deps).
- Remote execution: Windows PowerShell OpenSSH with key auth
  (`skill://remote-server-execution`); `BatchMode=yes` so auth gaps fail fast.
- Git loop for server work: local commit → push (never force) → `git pull
  --ff-only` on server; execute one verified commit; never reset/clean.

## Testing & QA

- pytest, focused suites only; name tests after observable contracts
  (determinism, bounds, invariants, error paths), not plumbing.
- Mandatory fixture pattern: tiny real dataset + real checkpoint + real bank
  through the public entry point to final outputs (compile-only checks never
  catch `NameError`-class defects).
- CUDA-only tests skip honestly on CPU hosts (`torch.cuda.is_available()`).
- Sprint gates: task evidence in `artifacts/sprint-N/task-M.md`, reviewed by
  `evidence-reviewer`, then `deep-reviewer`; close only on zero actionable
  findings (see `docs/PLAN.md`).
