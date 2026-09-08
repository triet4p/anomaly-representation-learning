# Project Plan — Anomaly Representation Learning

## Overview

The project builds a representation-learning anomaly detector and calibrated early-warning
system for variable-length multi-channel telemetry. Sprint 11 delivered causal factory
simulation and conditional geometry, but its accepted scientific verdict is 0/4 gates.
Sprint 12 diagnosed that failure: observable telemetry carries more signal than the learned
pipeline retains, while the temporal benchmark lacks enough clean controls and failure-mode
coverage for identifiable early-warning evaluation. Sprint 13 now owns benchmark
measurability; queued Sprint 14 owns causal localization of weak representation-to-score
components. Targeted model recovery is deferred until those two questions are answered.

## Current Sprint

- [Sprint 13 — Measurable Early-Warning Benchmark](sprint-plans/sprint-13.md) — **Active — REMEDIATION (all batches + Task 16 renewed-accepted; Task 17 ready for fresh deep review, not started; verdict `NOT_MEASURABLE`; Sprint 14 BLOCKED). Goal: prove that event ranking, positive lead time, false alerts, and later risk evaluation are structurally measurable under a frozen independent protocol.**

## Queued Sprints

- [Sprint 14 — Representation Failure Localization and Component Attribution](sprint-plans/sprint-14.md) — **Queued behind Sprint 13 `MEASURABLE`; identifies weak normalization, patchification, encoder/objective, pooling, geometry, scorer, or aggregation components before any targeted recovery sprint.**

## Paused Sprints

- [Sprint 4 — Joint Training Stabilization and Rerun Export](sprint-plans/sprint-4.md) — **Paused with Tasks 15–22 pending**

## Completed Sprints
- [Sprint 12 — Scorer Diagnosis, Signal Baselines, and Gated Scientific Recovery](sprint-plans/sprint-12.md) — **Complete (administrative re-finalization; batch gates A3/B2/C2/D2/F3/G2/G3/G4 accepted, Differential Deep Reviews 4+5 PASS with zero actionable findings; Tasks 10/13 retained as user-approved BLOCKED-FINAL scientific stops; no sealed-static reads, no calibrated-risk fit)**
- [Sprint 11 — Chronological Factory Geometry and Early-Warning Validation](sprint-plans/sprint-11.md) — **Complete on corrected evidence (training `e378626`, F1 `e725250`, F2 `3eef3d2`, report `ebde5d0`); Batches A1–F3 pass with zero actionable findings; sprint-wide deep review PASS (confidence 0.98); integrated verdict 0/4 stands as the accepted negative result**
- [Sprint 10 — Results Consolidation](sprint-plans/sprint-10.md) — **Complete (evidence-only gate passed)**
- [Sprint 9 — V1 Detection Improvement](sprint-plans/sprint-9.md) — **Complete (evidence-only gate passed; all tracks verdict retrain)**
- [Sprint 8 — V1 Results Summary](sprint-plans/sprint-8.md) — **Complete (evidence-only gate passed)**
- [Sprint 6 — Server-Native Geometry Execution and Inference Result Review](sprint-plans/sprint-6.md) — **Complete (geometry gates plus Tasks 14–16 evidence-only inference review)**
- [Sprint 3 — CLI Version Reporting](sprint-plans/sprint-3.md) — **Complete**
- [Sprint 2 — V1 Anomaly Representation Model](sprint-plans/sprint-2.md) — **Complete (including Tasks 17–22 remediation)**
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
| 12 | Side-effect-free version reporting from both synthetic-data CLI entry points | ✅ Sprint 3 |
| 13 | Stable, diagnosable joint training with coherent rerun exports | ⏸ Sprint 4 paused |
| 14 | Server-native latent-geometry execution with Git-synchronized runs | ✅ Sprint 6 (superseded the Sprint 5 Kaggle gate) |
| 15 | Deterministic shared-unit factory calendar with health, failures, maintenance, quarantine, and chronological splits | Complete — Sprint 11 |
| 16 | Conditional hierarchical latent geometry with localized synthetic boundary learning | Implemented — Sprint 11; scientific separation failed |
| 17 | Static detection and calibrated one-day/seven-day early warning | Evaluated — Sprint 11 negative result; not scientifically validated |
| 18 | Scorer diagnosis, observable-signal baselines, and independent scientific recovery gates | Complete — Sprint 12 (re-finalized; Tasks 10/13 BLOCKED-FINAL) |
| 19 | Measurable multi-category early-warning benchmark with independent event controls | REMEDIATION — Task 16 renewed-accepted; Task 17 ready, not started; `NOT_MEASURABLE`; Sprint 14 BLOCKED |
| 20 | Causal localization of representation-to-score bottlenecks | Queued — Sprint 14 |

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
- **Joint-training stability:** Prediction and contrastive branches share the same
  input-normalization contract. Predictor-facing contextual latents are insulated
  from whole-file contrastive projection, latent scale remains bounded, and model
  selection never compares checkpoints under different effective objectives.
- **Checkpoint coherence:** Saved model weights, optimizer, scheduler, global step,
  lambda state, and reference bank must describe one training state. Restoring an
  early model with late optimizer metadata is invalid.
- **Chronological factory data:** Files are causal scheduled operation events over a
  deterministic 3–6 month calendar. Physical units traverse shared robot routes; one
  robot processes at most one operation at a time while robot timelines remain
  asynchronous. Robot health is shared across programs and manifests with
  program-specific sensitivity.
- **Temporal split:** Training and validation use only verified healthy,
  non-quarantined files before the cutoff and split chronologically 80/20. Static test
  contains every post-cutoff normal plus every abnormal file from any time; a separate
  unshuffled temporal view preserves complete early-warning episodes.
- **V2 geometry:** Learn `(robot, program, regime)` normal geometry with hierarchical
  shrinkage, regularized Mahalanobis or mixture-density energy, localized synthetic
  clean/corrupt boundary learning, distribution-preserving file states, and guarded
  fixed/short-term baselines.
- **Confidence and risk:** Empirical/conformal anomaly confidence and censored
  one-day/seven-day survival risk are separate outputs. Geometry health, anomaly
  evidence, and future-failure probability must never be conflated.
- **Benchmark before model recovery:** Establish score-independent event measurability,
  positive-lead semantics, clean negative controls, progressive/weak/abrupt cohorts, and
  sealed history roles before resuming representation or calibrated-risk optimization.
- **Attribution before redesign:** Trace signal through normalization, patchification, local
  and contextual encoders, objective, pooling, conditional geometry, scorer, and aggregation.
  A bottleneck claim requires both stagewise loss and replicated recovery under a bounded
  component bypass/replacement; ambiguous evidence remains `UNRESOLVED`.

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

## Sprint 11 Verification Gates

Sprint 11 runs in explicit evidence-reviewed batches defined in its sprint plan.
Client source and tiny public-entry-point smokes precede a Git-synchronized server
preflight. Server execution then proceeds through two-epoch contract runs, five-epoch
coefficient balancing without test access, frozen-config 50-epoch normal-only and
hybrid training, bounded result extraction, conditional geometry analysis, sealed
static inference, and chronological early-warning analysis. Every batch requires a
fresh `evidence-reviewer` PASS with zero actionable findings; sprint completion also
requires a sprint-wide differential `deep-reviewer` PASS.

## Sprint 13–14 Verification Gates

Sprint 13 freezes its measurement contract, statistical floors, failure cohorts, roles, and
stop rules before generation. It closes only with a reviewed `MEASURABLE`,
`NOT_MEASURABLE`, or `UNAVAILABLE` verdict. Model performance and seed search cannot be
used to accept histories, and file AUROC cannot substitute for unavailable event metrics.

Sprint 14 starts benchmark-dependent attribution only after an accepted Sprint 13
`MEASURABLE` verdict. It uses common stagewise metrics, a frozen component-substitution
matrix, and bounded one-component interventions. It closes as `IDENTIFIED`,
`MULTIPLE BOTTLENECKS`, or `UNRESOLVED`; production recovery and calibrated risk remain
future gated work.

## Out of Scope for V1

- Raw waveform reconstruction or reconstruction-based anomaly verdicts
- RVQ/codebooks in the primary model
- Learned masking networks, discrete priors, metadata-based positive pairs
- Fusion of context and population scores before independent diagnostics
- Real-data deployment before the synthetic hypothesis and ablation gates pass
