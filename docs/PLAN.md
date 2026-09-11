# Project Plan — Anomaly Representation Learning

## Overview

The project builds a representation-learning anomaly detector and calibrated early-warning
system for variable-length multi-channel telemetry. Sprint 11 delivered causal factory
simulation and conditional geometry, but its accepted scientific verdict is 0/4 gates.
Sprint 12 diagnosed that failure: observable telemetry carries more signal than the learned
pipeline retains, while the temporal benchmark lacks enough clean controls and failure-mode
coverage for identifiable early-warning evaluation. Sprint 13 and Sprint 14 both returned
reviewed `NOT_MEASURABLE` verdicts. Sprint 15 now owns a fresh balanced causal benchmark
recovery with new protocols, deterministic fresh seed namespaces, and a success-only
candidate loop. Representation attribution has moved to Sprint 16; Sprint 15 delivered final `MEASURABLE` and its reviewed handoff is ACTIVE pending the mandatory post-final evidence review (see Current Sprint).

## Current Sprint

Sprint 15 - Balanced Causal Benchmark Recovery and Measurability Certification is **COMPLETE** with final verdict `MEASURABLE` for Candidate 7 (Gate F-r12 PASS with 0 actionable findings; final differential deep review R3 PASS with 0 actionable findings and 0 unanswered questions; EG7 PASS; Task 22 DONE per `artifacts/sprint-15/task-22.md` on `deep-review-final-r3.md`). Historical correction rounds and all failed/deprecated reviews preserved as indexed evidence. Mandatory post-final evidence review required before Sprint 16 starts benchmark-dependent work.

## Queued Sprints

- [Sprint 15 — Balanced Causal Benchmark Recovery and Measurability Certification](sprint-plans/sprint-15.md) — **COMPLETE (`MEASURABLE`; Task 22 DONE; post-final evidence review required before Sprint 16 starts).**
- [Sprint 16 — Representation Failure Localization and Component Attribution](sprint-plans/sprint-16.md) — **READY (reviewed handoff ACTIVE per `artifacts/sprint-15/task-22.md`; Sealed H-SEAL-37..40 still forbidden) — not started; benchmark-dependent execution may start only after the mandatory post-final evidence review passes.**

## Paused Sprints

- [Sprint 4 — Joint Training Stabilization and Rerun Export](sprint-plans/sprint-4.md) — **Paused with Tasks 15–22 pending**
## Completed Sprints
- [Sprint 14 — Benchmark Measurability Recovery](sprint-plans/sprint-14.md) — **Complete (negative closeout review PASS, zero findings; Tasks 1–19 complete on the negative branch with Tasks 10–15 truthfully NOT_RUN; final verdict `NOT_MEASURABLE` — EG2 0/4 abrupt count/share shortfall, 3/3 cycles exhausted, EG4–EG7 NOT_RUN, no deep review; no attribution handoff)**
- [Sprint 13 — Measurable Early-Warning Benchmark](sprint-plans/sprint-13.md) — **Complete (final differential deep review PASS, Deep Review 7, 0.995, zero findings; Tasks 1–17 accepted; verdict `NOT_MEASURABLE` — 0/13 STRUCT-PASS, all four sealed histories STRUCT-FAIL, zero UNAVAILABLE; no representation, warning-usefulness, or calibrated-risk success)**
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
| 19 | Measurable multi-category early-warning benchmark with independent event controls | Complete — Sprint 13 (final deep review PASS; verdict `NOT_MEASURABLE` — 0/13 STRUCT-PASS, four sealed STRUCT-FAIL, zero UNAVAILABLE) |
| 20 | Benchmark measurability recovery with independent confirmation and sealed generalization | Complete — Sprint 14 (`NOT_MEASURABLE`) |
| 21 | Balanced causal benchmark recovery and measurability certification | Complete — Sprint 15 (final `MEASURABLE`; Task 22 DONE; post-final review required) |
| 22 | Causal localization of representation-to-score bottlenecks | Ready — Sprint 16 (reviewed handoff ACTIVE; not started; post-final review gates start of benchmark work) |

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

## Sprint 13–16 Verification Gates

Sprint 13 and Sprint 14 remain immutable negative benchmark records. Sprint 14 followed
[`Benchmark Measurability Exit Gates`](BENCHMARK_MEASURABILITY_EXIT_GATES.md), exhausted
three prospectively versioned diagnostic cycles, and closed `NOT_MEASURABLE` without deep
review because EG2 failed. Its roots and evidence remain diagnostic-only and cannot be
reopened as a fourth cycle.

Sprint 15 creates a new gate document,
`BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md`, before generating outcomes. It uses a
calendar-first balanced causal evaluation profile with predeclared per-history event quotas,
bounded nuisance envelopes, explicit causal signal margins, fresh disjoint roles, unchanged
causal windows and acceptance thresholds, and one-shot roots within each candidate. Failed
candidates are preserved and retired; a new prospectively reviewed methodology uses the next
deterministic seed namespace. Sprint 15 has no negative terminal branch: finalization requires
EG0–EG6 provisional `MEASURABLE`, every evidence gate PASS, and a zero-actionable sprint-wide
deep review.

Sprint 16 starts benchmark-dependent attribution only after Sprint 15 returns final
`MEASURABLE` with a zero-actionable closeout and explicit handoff. It uses common stagewise
metrics, a frozen component-substitution matrix, and bounded one-component interventions.
It closes as `IDENTIFIED`, `MULTIPLE BOTTLENECKS`, or `UNRESOLVED`; production recovery
and calibrated risk remain future gated work.

## Out of Scope for V1

- Raw waveform reconstruction or reconstruction-based anomaly verdicts
- RVQ/codebooks in the primary model
- Learned masking networks, discrete priors, metadata-based positive pairs
- Fusion of context and population scores before independent diagnostics
- Real-data deployment before the synthetic hypothesis and ablation gates pass
