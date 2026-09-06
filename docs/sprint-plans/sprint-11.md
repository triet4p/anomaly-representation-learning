# Sprint 11 — Chronological Factory Geometry and Early-Warning Validation

**Goal:** Replace IID timestamped synthetic files and the ineffective V1 geometry with a deterministic 3–6 month shared-unit factory simulation, a robot-program-conditioned latent-geometry model with localized synthetic boundary learning, and server-verified static detection plus one-day/seven-day early-warning evidence.

**Status:** Active — planning complete; implementation not started

**Methodology:** [`docs/LATENT_GEOMETRY_METHODOLOGY.md`](../LATENT_GEOMETRY_METHODOLOGY.md)

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

### Chronological factory data

- [x] **Task 1 — Chronological factory configuration and event schema.** Add validated configuration and typed persisted contracts for the 3–6 month calendar, `unit_id`, product/program identity, routes, route position, robot operation intervals, queues, travel/idle time, health/failure/maintenance episodes, temporal labels, and deterministic provenance. Keep `unit_id`, `program_id`, `robot_id`, `operation_id`, and episode identities distinct. Add focused schema and validation tests. Evidence: `artifacts/sprint-11/task-1.md`.
- [x] **Task 2 — Deterministic shared-unit factory scheduler.** Implement causal unit arrivals and shared routes using single-server robot availability: no same-robot overlap, causal route ordering, nonnegative queue/travel/idle durations, asynchronous cross-robot execution, and stable output under a fixed seed. Add boundary and determinism tests. Evidence: `artifacts/sprint-11/task-2.md`.
- [x] **Task 3 — Robot-wide health, failure, and maintenance process.** Implement one calendar/usage health trajectory per robot, program-sensitive manifestation, causally coupled failure hazard, abrupt-failure support, explicit maintenance/recommissioning boundaries, and independent RNG streams that do not reorder the factory calendar when signal noise changes. Add progression, reset, and causality tests. Evidence: `artifacts/sprint-11/task-3.md`.
- [x] **Task 4 — Scheduled operation signal integration.** Generate each `FileSample` from its scheduled operation, healthy robot-program behavior, unit variation, robot health, program sensitivity, operating context, and sensor noise while preserving `C=6`, variable length, provenance, and the existing timestep/patch contracts. Add tests proving unit effects and robot health remain distinguishable. Evidence: `artifacts/sprint-11/task-4.md`.
- [x] **Task 5 — Temporal anomaly and degradation policies.** Implement isolated local anomalies, intermittent precursors, progressive degradation, program-sensitive symptoms, and deliberately abrupt failures. Preserve localized anomaly masks and ordered severity without making every future failure visibly anomalous. Add deterministic episode and monotonic-severity tests. Evidence: `artifacts/sprint-11/task-5.md`.
- [x] **Task 6 — Temporal labels, precursor quarantine, and split indices.** Materialize separate observable anomaly labels, latent diagnostic state, `time_to_next_failure`, one-day/seven-day targets, censoring, and maintenance boundaries. Implement chronological pre-cutoff verified-healthy development data, earliest/latest 80/20 train/validation, at-least-seven-day precursor quarantine, static test containing all post-cutoff normals plus all abnormalities from any time, and an unshuffled temporal test view. Add leakage and exact-membership tests. Evidence: `artifacts/sprint-11/task-6.md`.
- [x] **Task 7 — Chronological dataset materialization and manifest.** Extend the public generation entry point to write sharded client-scale and server-scale chronological datasets plus schedule, episode, split, checksum, seed, and count manifests. Fail fast on invalid roots or incomplete artifacts; never scan mounts or silently fall back. Add a tiny real materialization smoke test. Evidence: `artifacts/sprint-11/task-7.md`.
- [x] **Task 8 — Client data-contract smoke and invariant audit.** Generate a tiny multi-robot routed dataset locally through the public entry point and verify all scheduler, route, health, anomaly, quarantine, split, determinism, and reload invariants from persisted output. Record counts and the exact command without committing generated data. Evidence: `artifacts/sprint-11/task-8.md`.

### V2 latent geometry and longitudinal risk

- [x] **Task 9 — V2 model configuration and observable contracts.** Define validated configuration and typed outputs for pair/regime-conditioned patch energy, distributional file state, hierarchical covariance fallback, fixed/short-term baselines, trajectory statistics, anomaly confidence, and one-day/seven-day risk. Explicitly forbid program-only cross-robot fallback and label/health/future leakage into encoder inputs. Evidence: `artifacts/sprint-11/task-9.md`.
- [x] **Task 10 — Context-conditioned patch distribution encoder.** Implement the V2 patch/context representation that models conditional normal distributions or regime prototypes, retains variable-length masks and localization, and exposes stable patch latents without raw-waveform reconstruction. Add focused shape, padding, gradient, and context tests. Evidence: `artifacts/sprint-11/task-10.md`.
- [x] **Task 11 — Hierarchical regularized Mahalanobis geometry.** Implement verified-healthy `(robot, program, regime)` estimates, fallback through `(robot, program)` then robot then explicitly low-confidence fleet prior, shrinkage covariance, diagonal or low-rank-plus-diagonal fallback for sparse groups, multimodal mixture-density energy, and frozen monitoring references. Add singular-covariance, sparse-group, fallback, nonfinite, and no-cross-robot-program-sharing tests. Evidence: `artifacts/sprint-11/task-11.md`.
- [x] **Task 12 — Localized synthetic counterfactual objectives.** Implement paired in-memory corruptions that preserve unit/robot/program/regime/nuisance context, normal geometry variance/covariance control, unaffected-background consistency, relative corrupted-versus-clean boundary energy, progressive severity ordering, and a ramped boundary coefficient. Masks may shape the loss but never enter the encoder. Add held-out-region, detach/gradient, empty-mask, and monotonic-energy tests. Evidence: `artifacts/sprint-11/task-12.md`.
- [x] **Task 13 — Distribution-preserving file-state aggregation.** Replace plain mean-only population representation with an aggregation contract retaining patch-energy quantiles, calibrated upper-tail statistics, elevated-patch fraction, regime summaries, transitions, cross-channel consistency, and duration/persistence. Calibrate aggregation choices on verified healthy validation data only. Add sparse-local-anomaly and variable-length tests. Evidence: `artifacts/sprint-11/task-13.md`.
- [x] **Task 14 — Longitudinal baseline and trajectory tracker.** Implement fixed commissioning and guarded short-term baselines, maintenance segmentation, Mahalanobis absolute displacement and velocity, rolling trend, one-sided persistence/CUSUM, and disagreement reporting between fixed and adaptive references. Suspect files must not update either healthy reference. Add slow-drift, abrupt-shift, benign-reversal, maintenance-reset, and absorption-prevention tests. Evidence: `artifacts/sprint-11/task-14.md`.
- [x] **Task 15 — Calibrated anomaly confidence and failure-risk layer.** Implement empirical/conformal healthy-tail calibration separately from a discrete-time censored survival layer producing one-day and seven-day failure probabilities from historical trajectory features. Preserve the distinction between anomaly confidence and failure probability and prevent future target leakage. Add calibration, censoring, horizon, and serialization tests. Evidence: `artifacts/sprint-11/task-15.md`.
- [x] **Task 16 — V2 trainer, checkpoint, reference, and inference integration.** Integrate the V2 objectives and schedules with coherent model/optimizer/scheduler/global-step/reference/baseline state, stationary-objective selection, restored-reference-by-default inference, patch/file/trajectory outputs, and fail-fast configuration. Remove obsolete V1-only paths rather than retaining compatibility shims. Add a tiny full-file public-entry-point train-save-load-infer test. Evidence: `artifacts/sprint-11/task-16.md`.
- [x] **Task 17 — V2 architecture contract audit.** Run the focused V2 suites and a tiny real end-to-end CPU smoke; prove no label/model-input leakage, finite gradients and energies, stable covariance, correct localization, coherent checkpoint restoration, and deterministic trajectory outputs. Evidence: `artifacts/sprint-11/task-17.md`.

### Canonical notebooks and experiment source

- [x] **Task 18 — Chronological generation notebook.** Create an unexecuted canonical notebook that resolves explicit data roots, generates client/server profiles through the public entry point, inspects schedule and split invariants, and fails fast on invalid output. Keep the first-cell path contract and headless `nbconvert` compatibility. Evidence: `artifacts/sprint-11/task-18.md`.
- [x] **Task 19 — V2 training notebook.** Create an unexecuted canonical V2 training notebook with explicit data/checkpoint roots, deterministic seeds, CPU/CUDA selection, 2/5/50-epoch profiles, raw and weighted loss/gradient/geometry diagnostics, coherent resume behavior, and no missing-checkpoint fallback. Evidence: `artifacts/sprint-11/task-19.md`.
- [x] **Task 20 — V2 static inference notebook.** Create an unexecuted canonical notebook for restored hierarchical references, independent patch/file energies, conformal anomaly confidence, validation-only calibration, localization, family metrics, and provenance-separated outputs. Evidence: `artifacts/sprint-11/task-20.md`.
- [x] **Task 21 — V2 trajectory and early-warning notebook.** Create an unexecuted canonical notebook that preserves chronological robot episodes, computes fixed/short-term baseline trajectories, displacement/velocity/trend/persistence, fits only the allowed survival calibration data, evaluates one-day/seven-day risks with censoring, and writes review-sized metrics and figures. Evidence: `artifacts/sprint-11/task-21.md`.
- [x] **Task 22 — Staged experiment runner and provenance contracts.** Add committed experiment source/configs for a 2-epoch syntax/logic stage, 5-epoch coefficient-balance stage, and 50-epoch full stage, including a matched normal-only geometry control and hybrid boundary model. Each run records commit, data/checkpoint/reference checksums, device, seeds, wall time, raw/weighted losses, gradient norms, geometry health, and exact output roots. Evidence: `artifacts/sprint-11/task-22.md`.
- [x] **Task 23 — Client notebook and experiment-source smoke.** Execute only tiny client profiles headlessly to verify canonical notebooks, staged runner, manifests, and output schemas before any server synchronization. Do not claim model quality from this smoke. Evidence: `artifacts/sprint-11/task-23.md`.

### Server data and staged training

- [x] **Task 24 — Git-synchronized server preflight.** Execution commit `d31d172` (local == remote, fast-forward only); remote `trietlm@192.168.30.244` repo/runtime/disk recorded read-only; unrelated remote work preserved; explicit repo/data/output roots resolved. Evidence: `artifacts/sprint-11/task-24.md`.
- [x] **Task 25 — Full chronological dataset generation on server.** At `d31d172`, public entry point + locked server profile/seed 0 → 600 files (381 normal / 219 abnormal; dev 44+11, static 364, temporal 523) at server-only `data/generated/sprint11-server/` (manifest `cf788f92…`); bounded manifest/log/checksums transferred with verified digests. Evidence: `artifacts/sprint-11/task-25.md`.
- [x] **Task 26 — Server dataset evidence audit.** Persisted-data audit 49/49 PASS through the public reader (reload, overlap-free robots, route causality, 36 async pairs, G==γH, failure/maintenance causality, prevalence, quarantine, exact memberships, no leakage, decoded determinism). One-slot arrival jitter fix committed as `d31d172`; grid-locked predecessor superseded and documented. Evidence: `artifacts/sprint-11/task-26.md`.
- [ ] **Task 27 — Two-epoch server contract experiments.** Run matched V2 normal-only and hybrid models for two epochs on bounded server data to catch syntax, device, memory, batching, loss, covariance, checkpoint, restored-reference, and inference defects. Require finite metrics and complete artifacts; use no test outcome to tune coefficients. Evidence: `artifacts/sprint-11/task-27.md`.
- [ ] **Task 28 — Five-epoch coefficient-balance experiments.** Run a bounded predeclared coefficient matrix for normal density, variance/covariance, background consistency, boundary, and severity-ordering terms. Log raw/weighted losses, gradient norms, effective rank, anisotropy, clean/corrupt energy ordering, unaffected-background stability, and healthy validation calibration. Keep test labels sealed. Evidence: `artifacts/sprint-11/task-28.md`.
- [ ] **Task 29 — Coefficient and architecture selection gate.** Select one hybrid configuration for full training using only training plus verified-healthy validation diagnostics and predeclared stability/geometry criteria. Record rejected cells and rationale; do not select on static test or failure-horizon test performance. Freeze the full-run config before opening test results. Evidence: `artifacts/sprint-11/task-29.md`.
- [ ] **Task 30 — Fifty-epoch normal-only control training.** Train the matched V2 normal-geometry control for 50 epochs at the verified commit and dataset manifest; save coherent checkpoint/reference/baseline state and complete stationary-objective history. Evidence: `artifacts/sprint-11/task-30.md`.
- [ ] **Task 31 — Fifty-epoch hybrid boundary training.** Train the frozen selected V2 hybrid configuration for 50 epochs under the same commit, data, seed policy, compute profile, and output contract as the control. Save coherent checkpoint/reference/baseline state and complete histories. Evidence: `artifacts/sprint-11/task-31.md`.
- [ ] **Task 32 — Server result extraction and bounded transfer.** Run canonical extraction at the exact training commit and copy back only executed notebooks, logs, manifests, metrics, summaries, and representative figures with verified checksums. Leave datasets, checkpoints, embeddings, row-level scores, and bulk traces at recorded server paths and uncommitted. Evidence: `artifacts/sprint-11/task-32.md`.

### Geometry, detection, early warning, and report

- [ ] **Task 33 — Conditional latent-geometry analysis.** Compare control and hybrid patch/file geometry by robot, robot-program, regime, health stage, known anomaly family, held-out family, and severity. Report effective rank, anisotropy, covariance conditioning, clean/corrupt energy ordering, conditional within/between distances, trajectory continuity, and multimodal coverage without equating noncollapse with detection. Evidence: `artifacts/sprint-11/task-33.md`.
- [ ] **Task 34 — Static inference and localization analysis.** On the sealed static test, report independent patch/file energy and population signals, validation-calibrated confidence, AUROC/AUPRC/F1, false-positive behavior, family/severity slices, localization overlap/top-tail mass, and matched control-versus-hybrid deltas with exact bank/reference provenance. Evidence: `artifacts/sprint-11/task-34.md`.
- [ ] **Task 35 — Chronological early-warning and calibration analysis.** On the untouched temporal view, report event recall at one day/week, warning lead time and persistence, false alert episodes per robot-day/month, time-dependent AUROC/AUPRC, concordance, Brier scores, calibration curves/error, conformal healthy coverage, censoring, known-robot and cold-start slices, and maintenance-boundary behavior. Evidence: `artifacts/sprint-11/task-35.md`.
- [ ] **Task 36 — Integrated hypothesis and failure-mode assessment.** Decide from predeclared gates whether hierarchical geometry separates anomalies, whether synthetic boundary learning improves held-out families rather than generator recognition, whether file aggregation preserves sparse evidence, and whether latent trajectories provide calibrated pre-failure warning. Identify root causes for every failed gate; do not substitute threshold tuning for representation failure. Evidence: `artifacts/sprint-11/task-36.md`.
- [ ] **Task 37 — Final Sprint 11 report and closeout.** Publish a committed experiment report linking exact commits, remote paths, dataset/checkpoint/reference manifests, executed notebooks, control/hybrid tables, figures, limitations, negative results, and the verified verdict. Update `docs/PLAN.md` and this sprint only after every evidence gate and the sprint-wide deep review pass with zero actionable findings. Evidence: `artifacts/sprint-11/task-37.md`.

## Acceptance Criteria

### Data generation

- A fixed seed reproduces the same calendar, routes, schedules, health paths, failures, maintenance, anomaly episodes, labels, and split manifests.
- One physical unit is traceable across its robot route; no robot has overlapping operations; cross-robot timelines are demonstrably asynchronous.
- One health path exists per robot, programs expose it with controlled sensitivity, and unit defects remain distinguishable from robot degradation.
- The chronological 80/20 development split contains only verified healthy, non-quarantined pre-cutoff files; the static and temporal test definitions exactly match the methodology.
- Server-scale data and checkpoints remain on the server; only bounded provenance and results return to the client.

### Representation and geometry

- No anomaly label, mask, latent health, or future-failure target enters encoder inputs; synthetic masks affect loss construction only.
- Hierarchical reference fallback is `(robot, program, regime)` → `(robot, program)` → robot → low-confidence fleet, never program-only across robots.
- Covariance inversion remains finite under sparse and low-rank groups; monitoring references are frozen or guarded against suspected-file absorption.
- Localized corruptions move outward relative to paired clean patches while unaffected background remains stable; held-out-family evidence is reported separately.
- File state preserves sparse/local evidence; plain averaging is not the sole population representation.

### Staged experiments

- Two-epoch runs prove executable contracts only; five-epoch runs balance coefficients without test access; 50-epoch control and hybrid runs use frozen configs and comparable provenance.
- Every remote result names the exact Git commit, data manifest/checksum, checkpoint/reference provenance, device, seed, wall time, and output root.
- Full training checkpoints are coherent and restored references are used as-is unless an explicitly recorded refit experiment is planned.

### Evaluation

- Static detection reports AUROC, AUPRC, validation-calibrated F1, localization, family/severity slices, and false-positive behavior.
- Early warning reports one-day/seven-day event recall, lead time, persistence, false-alert rate, discrimination, calibration, censoring, known-robot, cold-start, and maintenance slices.
- Geometry health, anomaly confidence, and failure probability remain separate concepts; no favorable threshold substitutes for near-chance representation geometry.
- The final verdict compares matched normal-only and hybrid models and records negative results without narrowing the requested scope.

## Explicit Evidence-Review Batches

Each batch requires a fresh `evidence-reviewer` PASS with zero actionable findings before dependent work advances:

- **Batch A1:** Tasks 1–4 — schemas, scheduler, health, and scheduled signals.
- **Batch A2:** Tasks 5–8 — temporal anomalies, splits, materialization, and persisted smoke.
- **Batch B1:** Tasks 9–12 — V2 contracts, patch distribution, hierarchical geometry, and counterfactual objectives.
- **Batch B2:** Tasks 13–17 — file state, trajectories, risk, integration, and architecture audit.
- **Batch C1:** Tasks 18–20 — generation, training, and static inference notebooks.
- **Batch C2:** Tasks 21–23 — trajectory notebook, staged runner, and client smoke.
- **Batch D:** Tasks 24–26 — Git-synchronized server preflight and server data gate.
- **Batch E1:** Task 27 — two-epoch contract runs.
- **Batch E2:** Tasks 28–29 — five-epoch balance and frozen selection.
- **Batch E3:** Tasks 30–32 — full control/hybrid training and bounded extraction.
- **Batch F1:** Tasks 33–34 — geometry and static inference.
- **Batch F2:** Tasks 35–36 — early warning and integrated hypothesis assessment.
- **Batch F3:** Task 37 — final report and plan closeout evidence.

After all batches pass, run one sprint-wide `deep-reviewer` differential gate. Any actionable finding returns to the responsible task/batch, receives a fresh evidence review, and then receives a fresh sprint-wide deep review.

## Execution and Orchestration Constraints

- The main session remains an orchestrator and never edits or debugs product code.
- Use only a `hard-task` implementation worker for Sprint 11 execution; never spawn or substitute a `task` worker. One hard-task worker owns each explicit evidence-review batch end-to-end and receives only that batch's tasks, acceptance criteria, upstream artifact references, and affected boundaries—not the whole repository history.
- Use one `evidence-reviewer` once per completed batch; do not run per-task reviews inside the batch. Retain an effective hard-task worker across consecutive batches when its context remains bounded; replace it only after repeated correction failures or materially overloaded/contradictory context, preserving accepted artifacts and unresolved findings before replacement.
- Keep at most two active subagents: retained worker plus the current evidence/deep reviewer.
- The main session does not poll. It waits indefinitely with `hub wait timeoutMs=0` and wakes only for subagent completion, findings, questions, or user steering.
- Every implementation task follows `skill://implement-atomic-task`: focused source, observable tests, `artifacts/sprint-11/task-<N>.md`, and task status updates.
- Every remote task follows `skill://remote-server-execution`: read-only preflight; local commit → non-force push → remote fast-forward pull; one verified commit per run; explicit roots; bounded transfers; no credential persistence.
- Long remote jobs run detached and are checked at 5–10 minute cadence by the worker, never by the main session.
- Never commit generated datasets, checkpoint binaries, embeddings, bulk traces, secrets, or credentials.

## Notes / Blockers

- Known server endpoint from prior verified work: `trietlm@192.168.30.244`; Task 24 must rediscover and record the exact checkout and runtime without guessing or mutating first.
- Actual server data roots, checkpoint roots, and experiment output roots are intentionally unresolved until the read-only preflight.
- Sprint 4 remains paused and is not implicitly reopened. Sprint 11 supersedes V1 detection improvement work only where the methodology explicitly replaces it.
