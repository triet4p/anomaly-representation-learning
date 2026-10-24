# Anomaly Representation Learning

Research on self-supervised representation learning for anomaly detection in
variable-length, multi-channel industrial telemetry. The project uses synthetic
robot telemetry to study anomaly scoring and early-warning behavior without
training on anomaly labels.

## Approach

- Generate deterministic synthetic telemetry with operating regimes, anomalies,
  noise, and sensor artifacts.
- Normalize conditionally on robot/program metadata and split signals into patches.
- Learn representations with a local patch encoder, a Transformer context encoder,
  EMA targets, masked latent prediction, and a file-level contrastive objective.
- Evaluate two independent scores: `S_pred` for contextual prediction mismatch and
  `S_pop` for population mismatch against a healthy reference bank.

The V1 representation objective does not reconstruct raw waveforms. Anomaly labels
are used for post-hoc evaluation, not as inputs to self-supervised training.

## Setup

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
GPU experiments additionally require a compatible PyTorch/CUDA environment.

Run from the repository root:

```bash
uv sync
uv pip install --no-deps -e .
```

## Usage

Inspect the synthetic-data generator before choosing an output path and configuration:

```bash
uv run python -m synth.cli --help
```

Interactive workflows are available through Jupyter:

```bash
uv run jupyter lab
```

Start with `notebooks/train_v1_representation.ipynb` and
`notebooks/infer_v1_representation.ipynb` for the V1 training/inference workflow.
Other notebooks cover geometry extraction, analysis, and V2 workflows.
Set the explicit paths in the notebook before execution; inference requires a
real checkpoint and its compatible data/configuration.

Common path overrides include `V1_DATA_ROOT`, `V1_CHECKPOINT_PATH` /
`V1_CHECKPOINT`, `V1_GEOMETRY_CACHE`, and `V1_GEOMETRY_ANALYSIS`.
Relative paths resolve from the current working directory.

Generated datasets and trained checkpoints are not included in Git. A fresh
checkout is not a ready-to-run copy of the completed experimental campaigns.
Generating a new dataset does not reproduce the certified Sprint 18 benchmark.

## Tests

Run focused suites rather than assuming every experiment is executable locally:

```bash
uv run -m pytest tests/synth/ -q
uv run -m pytest tests/representation/ -q
```

CUDA-only checks require a CUDA-capable host. These commands do not reproduce
the long-running scientific campaigns.

## Repository layout

| Path | Contents |
|---|---|
| `src/synth/` | Synthetic telemetry generation, configuration, datasets, and patchification |
| `src/representation/` | Encoders, normalization, training, inference, and geometry analysis |
| `notebooks/` | Training, inference, and analysis workflows |
| `tests/` | Focused behavioral and contract tests |
| `experiments/` | Versioned protocols, configuration locks, and experiment runners |
| `docs/` | Concepts, plans, and research conclusions |
| `artifacts/` | Local evidence and review records; ignored by Git |
| `data/generated/`, `checkpoints/` | Local generated data and model weights; ignored by Git |

## Research status and limits

Sprint 18 is closed with the bounded scientific verdict **`UNRESOLVED`**.
Seven executable arms completed locked Confirmation evaluation, while N1-HCC
remained stopped. No evaluated branch satisfied the full recovery criteria.
This is not a production-ready industrial fault detector or evidence of
generalization to real factory telemetry.

The learned encoders and independent scores are candidates for reuse as a
baseline when labels become available. Their value for supervised adaptation,
few-label learning, and continual learning has **not yet been established**.

- [Baseline reuse and continual-learning discussion](docs/sprint18-baseline-and-continual-learning.md)
- [Project plan](docs/PLAN.md)
- [Sprint 18 protocol and closeout](docs/sprint-plans/sprint-18.md)
- [Changelog](CHANGELOG.md)

## License

This project is licensed under the [MIT License](LICENSE).