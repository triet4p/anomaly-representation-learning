# Project Plan — Anomaly Representation Learning

## Overview

The project builds a representation-learning anomaly detector for hard, semantically
meaningful anomalies in variable-length synthetic files. The synthetic data subsystem
is complete; the active work is the first latent prediction plus file-level contrastive
model described in [CONCEPT.md](CONCEPT.md).

## Current Sprint

**Sprint 2 — V1 Anomaly Representation Model** — **Complete (including Tasks 17–22 remediation)**

See the dependency-ordered execution plan:
[docs/sprint-plans/sprint-2.md](sprint-plans/sprint-2.md).

## Completed Sprints

- [Sprint 1 — Synthetic Data Generation Subsystem](sprint-plans/sprint-1.md) — **Complete**

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Typed full-file synthetic sample and provenance contract | ✅ Done |
| 2 | Variable-length, multi-regime normal generation with coherent C=6 physics | ✅ Done |
| 3 | Hard anomaly families, strength gate, and deterministic split materialization | ✅ Done |
| 4 | Patchification with padding metadata and timestep↔patch mapping | ✅ Done |
| 5 | Fixed-ratio random + information-aware + block masking | ✅ Done |
| 6 | Same-file semantics-preserving contrastive views | ✅ Done |
| 7 | Dataset persistence, diagnostics, focused tests, and generation CLI | ✅ Done |
| 8 | V1 variable-length patch/context encoder with EMA latent prediction | ✅ Sprint 2 |
| 9 | File-level contrastive objective with progressive λ ramp | ✅ Sprint 2 |
| 10 | Independent context-mismatch and population-mismatch inference scores | ✅ Sprint 2 |
| 11 | CPU smoke, notebook execution, and design review gates | ✅ Sprint 2 (including predictor remediation) |

## High-Level Design Decisions

- **Semantic unit:** A complete variable-length `FileSample` (`x: [C, T]`) is
  the model unit. Patches are computational units only; they must retain valid
  lengths, padding masks, starts, and file identity.
- **Representation objective:** The context encoder predicts target-encoder
  patch latents only at masked positions. The target encoder is an EMA copy and
  target latents are stop-gradient. There is no raw-waveform decoder in V1.
- **Masking:** Keep total mask ratio fixed while composing random,
  information-aware stratified, and contiguous block masks. Composition is an
  auditable model input and an ablation axis.
- **Contrastive schedule:** An optional validated warmup holds λ at exactly zero through
  `T_warmup`; the configured linear ramp begins at the following step and remains
  monotonic through `lambda_max`.
- **Inference:** Report context mismatch (`S_pred`) and population mismatch
  (`S_kNN`/reference-bank distance) independently before considering any fusion.
  Patch scores remain localizable through the existing patch mapping contract.
- **Legacy boundary:** Reuse/adapt only the legacy RVQ quantizer as an isolated,
  optional baseline and genuinely generic trainer/criterion conventions. Do not
  port reconstruction-centric model graphs, decoders, fixed-window datasets,
  Spark/Mosaic data writers, or reconstruction scorers into V1.
- **Configuration and runtime:** Preserve the current top-level `synth` package
  and NumPy synthetic-data contracts. New model configuration is validated with
  Pydantic v2; tensor work is internal PyTorch, with `einops` used only where it
  makes shape transformations explicit.

## Verification Gates

Each atomic task in Sprint 2 owns focused behavioral tests and an
`artifacts/task_<id>_summary.md` record. The sprint closes only after the
full-file variable-length CPU smoke, notebook syntax/execution checks, and a
review of stop-gradient, EMA, mask-ratio, score-separation, and checkpoint
invariants have passed. No reconstruction score is accepted as a substitute for
the two V1 signals.

The notebook gate also requires a persisted dataset root: generation is
performed by `uv run python -m synth.cli`, notebooks resolve
`V1_DATA_ROOT` (default `data/generated/production`), and train/validation/test
split semantics remain explicit in the manifest.

## Out of Scope for V1

- Raw waveform reconstruction or reconstruction-based anomaly verdicts
- RVQ/codebooks in the primary model
- Learned masking networks, discrete priors, metadata-based positive pairs
- Fusion of context and population scores before independent diagnostics
- Real-data deployment before the synthetic hypothesis and ablation gates pass
