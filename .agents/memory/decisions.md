## [2026-09-04] Keep joint-training objectives coherent at shared boundaries

**Decision:** Prediction and contrastive branches must share the same input-normalization contract, contrastive learning must use a dedicated projection head, contextual/target/predicted latent scales must be bounded, and checkpoint selection must use a stationary objective with atomically coherent model/trainer/reference-bank state.
**Alternatives considered:** Keep raw contrastive views on the shared encoder; apply InfoNCE directly to predictor-facing contextual embeddings; compare scheduled joint losses across changing lambda values; or restore only model weights while retaining final optimizer/scheduler metadata.
**Reason:** The 40-epoch baseline showed prediction degradation accounted for 93.8% of the joint-loss increase, raw physical-scale contrastive inputs shared an encoder with normalized prediction inputs, contrastive loss saturated rapidly, and the selected checkpoint combined epoch-2 weights with step-7840 training state.
**Consequences:** Future model or notebook changes must preserve normalization parity, the projection boundary, bounded latent outputs, fixed-basis model selection, and same-step checkpoint/reference-bank state. Reversing any element requires new training evidence and an explicit architecture decision.

## [2026-09-04] Make contrastive lambda explicit and conservative for reruns

**Decision:** Expose `lambda_max` as one validated notebook parameter and use `0.1` as the next diagnostic-run default across scheduling, stationary selection, resume reporting, and history.
**Alternatives considered:** Keep the old hard-coded value `1.0`, permanently require lambda below one, or tune separate hidden values for training and model selection.
**Reason:** Lambda has no universal upper bound, but after normalization, stronger augmentation, and projection changes the old run cannot establish that `1.0` balances gradients; `0.1` keeps an initially near-`ln(batch_size)` InfoNCE term comparable to prediction loss while making the value explicit for controlled reruns.
**Consequences:** `0.1` is a conservative experiment default, not a proven optimum or permanent mathematical limit. Every run must record the chosen value, and scheduling and stationary checkpoint evaluation must use the same maximum.

## [2026-09-05] Keep Kaggle diagnostic exports self-contained

**Decision:** Every current Kaggle diagnostic export must bundle the complete extraction, analysis, and full inference notebook workflow plus a matching `src.zip` whose notebooks extract source into a writable Kaggle path.
**Alternatives considered:** Require users to attach a separate source dataset, rely only on a readable `src/` tree, or omit the full inference notebook from the geometry package.
**Reason:** A single reviewed upload avoids attachment-name/path drift and lets each notebook import the exact source version recorded by the export manifest, while retaining the full inference workflow needed to diagnose package-level Kaggle failures.
**Consequences:** Future export regeneration must synchronize all notebook copies, `src/`, and `src.zip`; exclude `__pycache__`, `.pyc`, and `.pyo`; and refresh guide, metadata, inventory, sizes, checksums, and the outer ZIP together.

## [2026-09-05] Pin active notebooks to the published Kaggle Model

**Decision:** All active checkpoint-consuming inference and geometry notebooks use Kaggle Model handle `trietp1253201581/v1-representation-20260904-01/pyTorch/default/1` and checkpoint file `v1_representation_20260904_01.pt`, while preserving explicit local path overrides.
**Alternatives considered:** Continue treating the checkpoint as a Kaggle Dataset, bundle the 19.85 MB checkpoint inside the source export, or leave different checkpoint defaults in canonical and exported notebooks.
**Reason:** The user published the trained state as a versioned Kaggle Model, whose `model_sources` attachment gives the notebooks a stable owner/model/framework/variation/version identity without duplicating model bytes or depending on a checkpoint dataset slug.
**Consequences:** Kaggle GPU notebooks must attach that model source and resolve its exact `pyTorch/default/1` mount first; local runs use `V1_CHECKPOINT_PATH` or `V1_CHECKPOINT`; cache-only analysis notebooks remain model-free.

## [2026-09-05] Make notebook input paths explicitly user-configured

**Decision:** Active production notebooks expose source archive/directory, dataset or geometry-cache input, checkpoint where applicable, and writable output paths as editable first-cell variables and never discover inputs by scanning Kaggle mounts.
**Alternatives considered:** Search `/kaggle/input` recursively, select from candidate mount lists, climb parent directories, or hide input selection behind environment-only autodetection.
**Reason:** Kaggle attachment slugs and layouts are user-controlled; automatic selection concealed wrong attachments and made failures difficult to diagnose. Visible exact paths make each run auditable and let errors name the variable the user must change.
**Consequences:** Every notebook must validate configured paths and fail fast; new Kaggle attachments require a deliberate one-line path edit instead of automatic discovery. Output paths remain separate and writable under `/kaggle/working`.

## [2026-09-05] Run notebooks on the GPU server through Git, not Kaggle exports

**Decision:** Notebook workflows execute directly on `trietlm@192.168.30.244` from a repository checkout synchronized by local commit → GitHub push → PowerShell OpenSSH pull; results live under `experiments/<date>/<run>/` with checkpoints under `checkpoints/`, and no source archive is packaged for upload.
**Alternatives considered:** Keep regenerating versioned `src.zip`/outer-ZIP Kaggle exports per fix, or transfer files ad hoc over SSH/scp.
**Reason:** Repeated Kaggle repackaging forced the user to re-attach updated archives every run and could not exercise CUDA locally; the authenticated server provides the real GPU and a stable Git-synchronized source of truth, with executed outputs versioned alongside the repo.
**Consequences:** Every notebook fix must flow through commit → push → server pull, and execution copies live under `experiments/` with explicitly configured paths. Kaggle export tooling must not be reintroduced, and large generated outputs/checkpoints stay uncommitted.

## [2026-09-06] Use conditional hierarchical geometry with localized synthetic boundary learning

**Decision:** Make regularized hierarchical Mahalanobis geometry over robot-program-regime-conditioned normal manifolds the core deviation measure; use mixture-density energy for multimodal normal behavior, empirical/conformal calibration for anomaly confidence, localized clean/corrupt boundary plus background-consistency losses for encoder training, and a separate chronological survival layer for one-day/seven-day failure risk.
**Alternatives considered:** Continue pure normal-only self-supervision with global Euclidean/kNN geometry; use one fleet-global or program-global manifold; train a conventional binary anomaly classifier; retain the existing whole-file contrastive objective unchanged alongside the new boundary loss; use a single-centroid covariance or a single end-to-end temporal classifier.
**Reason:** Normal-only training leaves abnormal instances unconstrained and can place them inside normal clusters, while a binary classifier risks generator-family shortcuts. Paired synthetic counterfactuals hold robot, program, regime, waveform, nuisance, and timestamp context fixed so only localized corruption is pushed outward, while background regions remain stable. The observed representation is anisotropic and low-effective-rank, making naïve covariance inversion unstable; hierarchical shrinkage handles sparse pair/regime data and multimodal normal operation. Reliable timestamps and failure events require chronological movement, calibrated unusualness, and failure probability to remain separate semantic layers.
**Consequences:** Future redesign experiments must not feed anomaly labels/masks/severity into the encoder, must evaluate held-out families/severity/duration/progression/location ranges, must use robot-program-aware fallback (never program-only across robots), must freeze healthy baselines against deterioration, and must use chronological known-robot and new-robot evaluation. Do not retain every existing objective unchanged in the first boundary experiment. Report anomaly confidence separately from survival risk; random file splits, fleet-global positives, plain mean pooling, theoretical chi-square thresholds, and automatically adapting banks are not valid substitutes.

## [2026-09-06] Generate chronological factory events with robot-wide health

**Decision:** Replace random independent timestamps with a deterministic, causal 3–6 month shared-unit factory scheduler; model one robot-wide health process with program-specific sensitivity; generate causal progressive/acute anomaly and failure episodes; quarantine pre-failure states from healthy chronological train/validation; and preserve separate static-detection and chronological early-warning test views.
**Alternatives considered:** Independently timestamped IID files, independent robot streams without shared unit routing, independent `(robot, program)` health states, random 80/20 splits, and one combined anomaly/failure label.
**Reason:** Real production-line constraints require shared units, routes, queues, asynchronous robot operation, and one physical robot health process. Causal scheduling prevents temporal leakage; precursor quarantine prevents transitional degradation from contaminating the healthy manifold; separate anomaly and failure labels preserve the distinction between static unusualness and calibrated one-day/week warning.
**Consequences:** Generation must retain unit/route/operation/episode metadata, deterministic scheduling and health state, maintenance segmentation, program-sensitive manifestations, and dual static/temporal evaluation indices. Future implementation must preserve these contracts and must not begin until sprint planning explicitly scopes it.

## [2026-09-07] Freeze Sprint 12 protocol v1 before generating new histories

**Decision:** Freeze `experiments/sprint12-protocol-v1.md` (3 seeded histories H-DEV/H-SEAL-A/H-SEAL-B at 4x server scale; reserved wrong_transition + cross_channel_inconsistency; program-03 cold-start holdout; 5% FPR cohort floors; frozen numeric gates G-learn/G-static/G-rank/G-risk; LAN transfer-ref Git relay with canonical pull-before-compute) before generating or inspecting any new outcome.
**Alternatives considered:** Reuse the Sprint 11 600-file dataset for development selection; pick seeds/configs after seeing outcomes; reserve mechanisms by random re-seeding instead of entire held-out families.
**Reason:** Sprint 11 test outcomes already inform redesign, so its data is diagnostic-only; development selection on it would leak. Entire held-out families (not new seeds of the same mechanism) are the only honest mechanism-generalization test, and frozen numeric gates prevent post-hoc threshold/seed/model substitution for failed representations.
**Consequences:** No training, calibration, selection, or evaluation may touch sealed roots outside approved runs; any gate change needs a protocol version bump plus explicit user approval; future agents must not re-litigate seeds, floors, or gates without that bump.

## [2026-09-07] Adopt protocol v2 base-physics multi-history design, reject density/wear scaling

**Decision:** Supersede v1 4x-density histories with eight base-physics server histories (H-DEV-1..4 seeds 100-103, H-SEAL-1..4 seeds 200-203; 300 units, 90-day calendar, unchanged health/wear/hazard/quarantine); pool verified-healthy unquarantined dev CAL/FIT across the four dev histories (aggregate CAL>=40, FIT>=120 with history-identity group matrix); keep four sealed histories as separate uncertainty units under base-realistic per-history floors; retire (never delete) the VOID v1 4x roots.
**Alternatives considered:** Inverse wear/hazard scaling at 4x density to hold failure counts down; extending the calendar span; weakening quarantine to recover dev cohorts.
**Reason:** Measured v1 failure: 4x density tripled failures (60-64/history) so 7-day quarantine blanketed 1440/1440 pre-cutoff files (dev cohorts empty) and robot-02 maintenance saturated; inverse wear scaling would invalidate the validated base physics, and quarantine is inviolable per Sprint 12 recommendation. Base profile yields healthy dev cohorts (44/11 per history) that pool to 200 FIT / 51 CAL across four histories.
**Consequences:** Batch D/E/F build on the eight v2 roots only; v1 roots are audit-only; any further scale/health/floor change needs protocol v3 plus explicit user approval.

## [2026-09-08] Freeze protocol v3 split-by-history risk roles with event-level G-rank

**Decision:** Freeze `experiments/sprint12-protocol-v3.md`: RISK-FIT = eligible temporal rows H-DEV-1..3, RISK-VAL = H-DEV-4, sealed temporal evaluation only; program-03 and maintenance excluded from FIT+VAL; history reset at maintenance; q90/standardizer/model from FIT, thresholds from VAL; event-level G-rank (one positive score per failure from causal 7d window, matched same-history/robot negative controls, UNAVAILABLE on insufficient negatives, never file-AUROC substitution); Task 13 only on all-four PASS.
**Alternatives considered:** Keeping pooled-dev fit with file-level G-rank; restricting supervised risk fit to dev_train-healthy rows; reusing unstandardized features.
**Reason:** Batch F review showed role conflation, cold-start leakage, maintenance asymmetry, L2 scale distortion, and file-vs-event metric mismatch; healthy-only restriction would remove all positive outcomes a warning model must learn from.
**Consequences:** Task 12 reruns only under v3; any risk-role or gate change needs v4 plus explicit user approval.

## [2026-09-08] Correct Task 12 scale/holdout defects; keep event G-rank UNAVAILABLE

**Decision:** Fix the logit-vs-probability threshold mismatch by standardizing on probability scale end to end (VAL operating points on sigmoid outputs), enforce the v3 maintenance exclusion in RISK-VAL with separate accounting, sanitize diagnostics to strict JSON, and rerun Task 12; report event G-rank UNAVAILABLE ×4 with genuinely measured companions; keep Task 13 blocked.
**Alternatives considered:** Converting thresholds back to logit scale; dropping the maintenance exclusion as negligible; substituting file AUROC for the unavailable event gate.
**Reason:** Review-F2 proved the zero-alert result was a scale defect (thresholds 5.6 vs scores <1.0), not a transfer failure; the maintenance omission violated v3 §1 eligibility; file-AUROC substitution is forbidden by the frozen v3 UNAVAILABLE rule.
**Consequences:** task-12.md retracts the score-shift claim; prior Task 12 runs (including 106292c) are superseded as evidence; any future warning rerun must preserve single-scale thresholds and VAL maintenance exclusion (locked by regression tests).

## [2026-09-09] Recover benchmark measurability before representation attribution

**Decision:** Sprint 14 owns benchmark measurability recovery under `docs/BENCHMARK_MEASURABILITY_EXIT_GATES.md`; the existing representation-attribution plan moves to Sprint 15, and targeted component recovery moves to Sprint 16.
**Alternatives considered:** Start attribution on Sprint 13 development roles despite the `NOT_MEASURABLE` verdict; relax or pool Sprint 13 structural floors; tune generator seeds or settings against learned-model scores; or use qualitative benchmark acceptance without numeric exits.
**Reason:** Sprint 13 correctly found 0/13 histories structurally passing and all four sealed histories below independent control/category floors, so representation and risk conclusions would remain unidentified. Prospectively frozen per-history floors, a fixed observable-only sanity probe, fresh confirmation histories, and untouched sealed histories make benchmark acceptance falsifiable while preserving cross-history, robot, program, and mechanism generalization.
**Consequences:** Sprint 15 cannot start without a reviewed Sprint 14 `MEASURABLE` verdict; every benchmark change needs a versioned pre-outcome protocol and fresh roots, pooled rescue and favorable-seed replacement are forbidden, sealed histories remain unscored in Sprint 14, and at most three diagnostic cycles may run before a final negative verdict.

## [2026-09-09] Deep-review Sprint 14 only after a provisional measurable result

**Decision:** Between Sprint 14 diagnostic cycles use only evidence reviews, and call a sprint-wide deep reviewer if and only if EG0–EG6 mechanically yield provisional `MEASURABLE`.
**Alternatives considered:** Deep-review every negative diagnostic cycle; deep-review every terminal Sprint 14 verdict; or omit the final differential review even when the benchmark appears measurable.
**Reason:** Evidence reviews are the correction gates for cycle-local methodology and artifacts, while a differential deep review is valuable only when positive benchmark evidence could authorize Sprint 15; spending it on an already mechanical `NOT_MEASURABLE` or `UNAVAILABLE` result cannot change eligibility and risks reinterpreting a scientific stop.
**Consequences:** The orchestrator may direct prospective methodology corrections between evidence-reviewed cycles but performs no sprint implementation; provisional negative or unavailable outcomes close with EG7 recorded `NOT_RUN`, while provisional `MEASURABLE` requires a zero-actionable deep review before becoming final or unblocking Sprint 15.

## [2026-09-09] Execute Sprint 14 locally

**Decision:** Run all Sprint 14 benchmark-recovery workflows on the local workstation through the repository's locked `uv` environment; do not use the remote server.
**Alternatives considered:** Retain the Git-synchronized GPU-server workflow used by compute-heavy representation sprints, or split generation locally and audits remotely.
**Reason:** Sprint 14 performs bounded synthetic generation, structural audits, deterministic fixtures, and a simple observable probe rather than expensive model training, so remote execution adds synchronization and provenance complexity without a compute requirement.
**Consequences:** Protocols must predeclare the local runtime and local roots, all first-attempt materializations and audits run from the repository root, generated bulk roots remain local and uncommitted, and no Sprint 14 task may require SSH, push/pull execution, or remote-only evidence.

## [2026-09-10] Recover the benchmark in Sprint 15 before Sprint 16 attribution

**Decision:** Preserve Sprint 14 as final `NOT_MEASURABLE`, move the existing representation-attribution plan from Sprint 15 to Sprint 16, and use a new Sprint 15 balanced causal case-control benchmark with fresh protocols, roots, seeds, and a two-cycle cap.
**Alternatives considered:** Add a fourth Sprint 14 diagnostic cycle; keep attribution numbered Sprint 15 while inserting an unnumbered recovery effort; continue tuning natural hazard rates and calendar exposure; or guarantee `MEASURABLE` by weakening gates or selecting favorable seeds.
**Reason:** Sprint 14's final evidence isolated preferential reset-window loss of abrupt events after negative-control support was repaired. A prospectively quota-controlled evaluation profile fixes support at the benchmark-sampling layer without rewriting Sprint 14's stop rule, hiding outcome selection, or claiming that balanced case-control frequencies estimate fleet prevalence.
**Consequences:** Sprint 14 artifacts and verdict remain immutable; Sprint 15 must preserve causal windows and acceptance thresholds, use fresh disjoint roles, fail rather than resample infeasible roots, and earn `MEASURABLE` through Design, observable, Confirmation, Sealed, evidence, and conditional deep-review gates. Sprint 16 stays blocked until the reviewed Sprint 15 handoff exists; targeted model recovery moves to Sprint 17.

## [2026-09-10] Make Sprint 15 a measurable-only delivery sprint

**Decision:** Sprint 15 has no fixed candidate-iteration cap and may finalize only as `MEASURABLE`; every rejected candidate is preserved, then replaced by a prospectively reviewed methodology revision using the next deterministic fresh-seed namespace.
**Alternatives considered:** Keep the two-cycle cap and permit final `NOT_MEASURABLE` or `UNAVAILABLE`; guarantee a pass by weakening acceptance thresholds; retry failed seeds within a candidate; or reopen Sprint 14.
**Reason:** The user requires Sprint 15 to deliver a usable measurable benchmark rather than another terminal diagnosis. A success-only candidate loop makes that delivery condition explicit while preserving within-candidate independence, immutable failed evidence, fixed global acceptance gates, and prohibition of favorable-seed replacement.
**Consequences:** Sprint 15 remains open after allocator, Design, observable, Confirmation, Sealed, evidence, or deep-review failure. Only generator/scheduler/allocator/nuisance/signal methodology may change prospectively between candidates; acceptance floors, metrics, probe thresholds, no-pooling rules, and final `MEASURABLE` semantics remain fixed. Sprint 16 stays blocked until the successful handoff exists.

## [2026-09-11] Bind Sprint 16 C1 to the accepted V2 path: no input-normalization stage

**Decision:** Treat the accepted production stage-1 operator as the verified identity (no external/conditional input normalization exists in the V2 control/hybrid checkpoints and path); C1 resolves evidence-insufficient UNRESOLVED/NOT PRESENT at Task 16 and can never be a bottleneck.
**Alternatives considered:** Ratify protocol v3's C1 boundary (ConditionalBatchNorm with V1Config defaults) and measure M1 against it; invent a fitted normalization convention for the comparison; or count in-encoder LayerNorm as C1 signal loss.
**Reason:** Hash-verified inspection of both accepted checkpoints shows no norm flag and no conditional/BatchNorm statistics, and the accepted V2 inference path applies raw patches with only parameter-only LayerNorm inside the encoders/head. Measuring a V1 stage that the accepted bytes do not contain would fabricate a boundary and its verdict.
**Consequences:** Protocol v4 re-binds C1/M1 (M1 gap is identity-derived, computed through the contract reader); parameter-only LayerNorm belongs to C3/C4 and Tasks 7/8; N-a/N-b/N-c are additive diagnostic replacements that cannot establish C1 causality; future agents must not reintroduce a C1 loss claim without explicit instruction and new accepted evidence.
