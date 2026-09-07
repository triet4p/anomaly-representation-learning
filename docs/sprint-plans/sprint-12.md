# Sprint 12 — Scorer Diagnosis, Signal Baselines, and Gated Scientific Recovery

## Sprint Goal

Diagnose Sprint 11's accepted negative result, establish observable-signal baselines, remove demonstrated scoring shortcuts, and evaluate controlled representation and early-warning alternatives on independent histories without tuning against opened test evidence.

**Status:** Active — planning complete; Batch A assigned first. Scientific success is not assumed. Sprint 11 remains Complete with its corrected 0/4 result; Sprint 4 remains paused.

**Authoritative request:** [`../SPRINT12-RECOMMENDATIONS.md`](../SPRINT12-RECOMMENDATIONS.md) preserves the previous recommendation entirely and verbatim, including original chat-relative links. Resolve its `src/` links from the repository root; do not rewrite that historical text. This plan operationalizes every recommendation, not a reduced subset.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done. Implementation completion is not evidence acceptance; each task artifact records its gate explicitly.

### Batch A — Accepted-checkpoint diagnosis

- [x] **Task 1 — Verify accepted checkpoint and runtime provenance.** Read-only local/server preflight using the remote-server-execution skill. Resolve the actual existing checkout, locked runtime, dataset, and accepted control/hybrid checkpoints from Sprint 11 evidence. Verify training `e378626`, F1 `e725250`, F2 `3eef3d2`, dataset/checkpoint hashes and restored references; preserve all existing user work and historical artifacts. Record reachable roots, exact commands, and runtime readiness in `artifacts/sprint-12/task-1.md`. Never substitute random weights or a new draw for the accepted checkpoint investigation. (Batch A ACCEPTED PASS per review-A3; pull-before-run at `22c005d`.)
- [x] **Task 2 — Measure self-conditioned scorer shortcut evidence.** Execute residual, log-variance/clamp, energy-component, and latent-versus-predicted-mean response diagnostics on both accepted checkpoints across healthy, abnormal, and paired-corruption inputs. Quantify whether the mean tracks corrupted latents and distinguish architectural possibility from observed checkpoint behavior. Keep context and population scores independent; frozen references stay unchanged. Persist bounded summaries, provenance, and a falsifiable diagnosis in `artifacts/sprint-12/task-2.md`; no claim of confirmed causality from coincident medians alone. (Batch A ACCEPTED PASS per review-A3; causality unresolved, no identity-shortcut confirmation.)

### Batch B — Signal visibility and simple baselines (worker complete; awaiting review)

- [x] **Task 3 — Compare handcrafted and frozen-latent conditional baselines.** Implement an interpretable valid-patch feature set covering channel level/RMS/variance/slope, differences/cross-channel relationships, spectral bands, transitions, and duration. Compare handcrafted features and accepted frozen learned features using the same regularized hierarchical healthy-only geometry and fixed upper-tail file aggregation. Retain localization, variable-length masks, sparse fallback, and score provenance. Run both variants on the historical Sprint 11 diagnostic benchmark, report discrimination/localization/false alerts and group slices, and distinguish absent signal from an inadequate baseline. Evidence: `artifacts/sprint-12/task-3.md`. (Implemented; AUROC 0.41/0.49/0.49 — "neither works" cell; awaiting review.)
- [x] **Task 4 — Audit signal visibility through actual preprocessing.** Execute paired observable-signal diagnostics before and after the production preprocessing path, including amplitude and cross-channel changes and benign nuisance controls. Trace whether anomaly information is removed by preprocessing, feature extraction, contextual encoding, or scoring; quantify rather than infer from source alone. Use labels/masks only for post-hoc diagnostics. Evidence: `artifacts/sprint-12/task-4.md`. (Implemented; cross-channel signal at S0, collapse by S2/S4, causality unresolved; awaiting review.)

### Batch C — Independent development and evaluation protocol

- [ ] **Task 5 — Freeze expanded development and sealed evaluation protocol.** Before generating or inspecting new outcomes, persist seed/history assignments, healthy training/calibration requirements by condition, reserved entire corruption mechanisms, held-out robot/program assignments, non-saturated maintenance windows, independent uncertainty units, development selection rules, computational limits, and numeric scientific advancement criteria. Base cohort-size floors on the intended false-alert target, not the former 11-row calibrator. Sprint 11 is historical diagnostic evidence only. Separate healthy fitting, calibration, development selection, and sealed final evaluation. Preserve seven-day precursor quarantine; enlarge healthy exposure instead. Evidence: `artifacts/sprint-12/task-5.md` and a versioned experiment protocol.
- [ ] **Task 6 — Materialize independent histories with healthy calibration coverage.** Generate the protocol's multiple independent histories through the public data entry point; prove route/robot causality, deterministic reloads, quarantine, chronological partitions, conditional healthy counts, genuine cold-start partitions, and unsaturated maintenance contrast. Do not repeatedly select seeds by anomaly-model performance. Seal final histories against training/selection access. Keep bulk data remote and return bounded manifests/counts/checksums. Evidence: `artifacts/sprint-12/task-6.md`.

### Batch D — Bounded representation experiment

- [ ] **Task 7 — Implement controlled mechanisms and nuisance negative controls.** Implement context-preserving development corruptions covering level/gain drift, cross-channel coupling, timing/phase shifts, localized transients, and persistent degradation under the Task 5 mechanism partition. Entire reserved mechanisms must never be used in training or development selection. Record exact localized support, ordered severity, physical bounds, and nuisance-only controls; reuse existing generators/contracts where applicable. Noise alone is not the complete boundary curriculum. Evidence: `artifacts/sprint-12/task-7.md`.
- [ ] **Task 8 — Implement target-hidden context scorer without leakage.** Implement the controlled target-hidden alternative alongside the frozen-feature density experimental baseline. Hide the target before any contextual mixing, including overlapping waveform support; use stop-gradient targets and healthy-conditioned reference statistics without query-conditioned self-copying. Preserve independent context/population scores, valid-patch localization, coherent checkpoints, and explicit restored-reference semantics. Demonstrate no target path through context or preprocessing, nonzero finite learning gradients, and train-save-load-infer behavior with a tiny real fixture. Experimental alternatives are explicit variants, not compatibility shims. Evidence: `artifacts/sprint-12/task-8.md`.
- [ ] **Task 9 — Run bounded learnability and matched representation comparison.** First run the Task 5 small-set gate: paired corruption ranking, development severity ordering, unaffected-background stability, and nuisance false alarms. If it fails, diagnose gradients/objective competition/scorer behavior and run bounded, documented development-only corrections; do not launch a long run on a failed learnability gate. Compare handcrafted, current frozen, and revised features with the same geometry plus separate context-score results on development histories. Freeze the selected complete configuration and hashes before sealed evaluation. Scientific advancement requires the predeclared separation/nuisance criteria, not finite losses or noncollapse alone. Evidence: `artifacts/sprint-12/task-9.md`.

### Batch E — Independent static evidence

- [ ] **Task 10 — Evaluate frozen design on sealed static histories.** Only after Task 9's scientific advancement gate, execute the frozen design and matched baselines on untouched histories, reserved mechanisms, and held-out robots/programs. Report discrimination, localization, severity, false-alert operating points calibrated without test access, and history/robot/episode-level uncertainty. Never bootstrap correlated patches as independent samples. Record whether learned representations improve over the simple baseline; negative results remain negative and do not trigger sealed-test tuning. Evidence: `artifacts/sprint-12/task-10.md`.

### Batch F — Precursor information and causal warning

- [ ] **Task 11 — Diagnose observable precursors across failure categories.** Stratify progressive, weak-precursor, and deliberately abrupt failures using simulator state only for post-hoc diagnostics. Measure observable pre-failure signal visibility with causal windows and proper censoring, keeping sudden failures in the overall ledger rather than silently removing them. This diagnostic may proceed even if learned static geometry fails; it cannot reopen a sealed selection set. Evidence: `artifacts/sprint-12/task-11.md`.
- [ ] **Task 12 — Compare causal warning and operating-history baselines.** Evaluate constant-risk, observable usage/time-since-maintenance, and simple causal trend/persistence baselines before a flexible survival head. Include geometry trajectories only with explicit upstream scientific standing. Use chronological out-of-sample fit/evaluation, maintenance segmentation, censoring, event recall, lead time, persistence, and false alerts per robot-day; quantify added information beyond operating history. Evidence: `artifacts/sprint-12/task-12.md`.
- [ ] **Task 13 — Evaluate calibrated risk after ranking gate passes.** Only after the predeclared useful-ranking gate, fit/calibrate the separate risk model using allowed development cohorts and evaluate frozen one-day/seven-day predictions against the constant and operating-history baselines. Report Brier, reliability/ECE, discrimination, event detection, uncertainty, censoring, known/cold-start and failure-type slices. Recalibration cannot stand in for discriminative signal. If the ranking gate fails, explicitly record this task as blocked by scientific evidence; never claim a risk experiment ran. Evidence: `artifacts/sprint-12/task-13.md`.

### Batch G — Integrated scientific assessment

- [ ] **Task 14 — Consolidate scientific verdict and sprint evidence.** Produce a complete report covering every recommendation, provenance, executed alternatives, positive/negative/unavailable outcomes, causal uncertainties, independent-test discipline, simple-baseline comparisons, and advancement decisions. Preserve Sprint 11 evidence verbatim. Record all batch evidence gates and unresolved dependencies. Do not declare completion while any planned conditional task is blocked without an explicit user-approved plan disposition. Evidence: `artifacts/sprint-12/task-14.md`.
- [ ] **Task 15 — Pass differential review and finalize sprint status.** After every task/batch evidence gate passes, run a sprint-wide differential deep review over the accepted evidence reports. Correct actionable findings through the retained worker, repeat affected evidence gates and then deep review, and update global/sprint status only with zero actionable findings and all task dispositions resolved. Evidence: `artifacts/sprint-12/task-15.md`.

## Acceptance Criteria and Scientific Stop Rules

- Entire prior recommendation remains verbatim in its dedicated document; every recommendation maps to the tasks above.
- Actual accepted-checkpoint execution establishes whether the suspected self-copy shortcut is active; architectural risk alone is not proof.
- Observable telemetry baselines and actual preprocessing diagnostics precede representation conclusions.
- Training, calibration, development selection, and sealed evaluation provenance remain separate; hidden state/labels/future outcomes never enter model inputs.
- Scientific advancement criteria are frozen before new development outcomes; final evaluation is never reused for tuning. No threshold, bigger model, or longer-run substitution for failed representation gates.
- Useful static evidence precedes flexible risk modeling. A failed advancement gate blocks its dependent experiment, not all independent diagnosis. Finish reachable work, record the blocker, and request explicit disposition rather than silently narrowing scope or declaring success.
- Operational PASS and scientific PASS remain separate. Hypotheses may fail; task acceptance requires faithful execution and reporting, not a favorable outcome.
- New scientific gates are not retroactively applied to invalidate Sprint 11's accepted negative record.

## Execution and Review Contracts

- Main session owns planning/orchestration only using its current model; no product inspection, implementation, or debugging by main during this sprint.
- Retain one `hard-task` worker across consecutive assignments. Maximum two active subagents: worker plus one evidence reviewer or one deep reviewer.
- Explicit consecutive evidence batches: A (1–2), B (3–4), C (5–6), D (7–9), E (10), F (11–13, respecting scientific dependencies), G (14); Task 15 is the final differential gate/closeout.
- Review each completed batch before advancing beyond its gate. Any actionable finding returns to the same worker, then a fresh same-level review. Deep review consumes accepted evidence, not a duplicate full audit.
- Worker follows implement-atomic-task and relevant graphify, remote-server-execution, log-decision, and log-lesson skills. Use focused behavioral tests and tiny real public-entry-point smokes; skip formatters, linters, and project-wide suites. Never verify copied/exported trees through imports.
- Remote work follows read-only preflight, local commit → non-force push → remote fast-forward pull, and exact verified execution commit. Preserve unrelated local/remote changes. No generated datasets, checkpoints, bulk embeddings/traces, credentials, or secrets in commits.
- Experiments must run and return bounded output with command, exit status, hashes, seed, device, runtime, and output root. Compilation or fabricated/mock outputs do not satisfy scientific tasks.
- Worker updates task artifacts and implementation status; main accepts review gates. Persist review identities/results in the sprint gate ledger below.

## Gate Ledger

- Planning: recommendation preserved verbatim; Sprint 12 opened. No scientific claim yet.
- Batch A: ACCEPTED PASS with zero actionable findings (reviews artifacts/sprint-12/review-A.md → review-A2.md → review-A3.md by Sprint12EvidenceA2-retained; pull-before-run at 22c005d, identical rerun numerics, interpretation per A3 §§60-67 with unresolved causality retained; no identity-shortcut confirmation claimed).
- Batch B: awaiting review (Tasks 3–4 implemented; task-3.md: matched handcrafted/learned conditional comparison, AUROC 0.41/0.49/0.49, "neither works" cell; task-4.md: raw cross-channel signal present, amplitude absent, collapse by latent/score stages, causality unresolved; worker does not self-accept).
- Batch C: pending.
- Batch D: pending.
- Batch E: pending scientific prerequisites.
- Batch F: pending; Task 13 requires useful-ranking evidence.
- Batch G: pending.
- Sprint-wide deep review: pending.

## Notes / Blockers

- No external prerequisite has yet been verified for Sprint 12. Known Sprint 11 server/root/hash records are starting references, not assumed current availability.
- Scientific advancement failures must remain explicit. Do not silently replace the full recommendation with only Steps 1–2.
