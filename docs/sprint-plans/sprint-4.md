# Sprint 4 — Joint Training Stabilization and Rerun Export

**Goal:** Identify and correct every evidence-supported cause of unstable joint prediction/contrastive training, then export a coherent source bundle and notebook for a user-operated Kaggle rerun.

**Status:** Paused — user redirected work to Sprint 5 latent-space diagnostics

---

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

- [x] **Task 1 — Audit the complete joint-training path.** Trace prediction and contrastive inputs through normalization, patch/context encoders, projection and pooling; inspect augmentation strength, loss definitions and scale, lambda scheduling, optimizer/scheduler stepping, EMA updates, validation semantics, checkpoint selection/restoration, reference-bank construction, resume metadata, and notebook orchestration. Reconcile every candidate with the recorded 40-epoch history. Produce `artifacts/sprint-4/task-1.md` containing a ranked root-cause matrix with direct evidence, confidence, affected symbols, and a required-fix disposition. Do not execute the training notebook.
- [x] **Task 2 — Enforce encoder input normalization parity.** Ensure prediction and contrastive views enter the shared local encoder under the same normalization contract, including fleet-conditioned behavior and padded/valid timestep semantics. Add focused behavioral coverage that would fail if raw physical-scale contrastive inputs reach the shared encoder. Record evidence in `artifacts/sprint-4/task-2.md`.
- [x] **Task 3 — Stabilize contextual and target latent scale.** Add the minimal normalization required to bound context, EMA-target, and predicted latent scales without violating stop-gradient or EMA invariants. Cover variable-length and padded batches and record exact focused verification in `artifacts/sprint-4/task-3.md`.
- [x] **Task 4 — Isolate the contrastive optimization surface.** Add a dedicated contrastive projection path so whole-file InfoNCE gradients do not directly force predictor-facing contextual latents to solve instance discrimination. Preserve file-embedding and checkpoint contracts intentionally, migrate all callers, and add focused coverage. Record evidence in `artifacts/sprint-4/task-4.md`.
- [x] **Task 5 — Correct contrastive task calibration.** Replace the trivially saturating augmentation/temperature defaults with validated, configurable values suitable for the generated signal scales, while preserving deterministic augmentation and file identity. Add boundary and determinism coverage and record evidence in `artifacts/sprint-4/task-5.md`.
- [x] **Task 6 — Make model selection and checkpoint state coherent.** Stop comparing best checkpoints under changing objective definitions; use a stationary selection policy that retains a genuinely joint-trained model after the lambda ramp. Ensure model weights, step, optimizer, scheduler, lambda state, and reference bank all come from a coherent training state, including resume behavior. Add regression coverage and record evidence in `artifacts/sprint-4/task-6.md`.
- [x] **Task 7 — Add rerun diagnostics.** Report raw and weighted objective components, effective lambda, latent norms, and other bounded diagnostics needed to distinguish prediction drift, contrastive saturation, scale drift, and objective interference. Keep training overhead bounded and output machine-readable. Record evidence in `artifacts/sprint-4/task-7.md`.
- [x] **Task 8 — Update the canonical training notebook.** Apply the corrected APIs and selection semantics, expose the diagnostic history, prevent silent fresh/incoherent checkpoint use, and leave the notebook runnable from a clean Kaggle session. Validate notebook structure and compile code cells only; do not execute training. Record evidence in `artifacts/sprint-4/task-8.md`.
- [x] **Task 9 — Export rerun source and notebook.** Use the repository's existing Kaggle packaging conventions to produce a versioned source bundle and matching training notebook, plus an export manifest containing paths, version/commit provenance, and checksums. Run only packaging, static notebook compilation, focused tests, and import/smoke checks; do not execute the exported training notebook. Record evidence in `artifacts/sprint-4/task-9.md`.
- [x] **Task 10 — Expose the maximum contrastive lambda.** Add one validated, user-editable notebook parameter for `lambda_max`, with a conservative rerun default of `0.1`; use that same value for the training ramp, stationary validation/model selection, resume reporting, and machine-readable history. Preserve the source configuration contract and update canonical/Kaggle notebooks plus focused static coverage. Do not execute notebook cells. Record evidence in `artifacts/sprint-4/task-10.md`.
- [x] **Task 11 — Regenerate the verified rerun export.** Repackage the matching source and updated notebook into a new versioned export, update provenance/instructions/checksums, and repeat archive import and notebook compilation checks without executing training. Record evidence in `artifacts/sprint-4/task-11.md`.
- [x] **Task 12 — Evaluate the user-operated rerun.** Analyze the executed notebook under `experiments/20260409/(v2)` and downloaded checkpoint files under `checkpoints/`; compare prediction loss, raw/weighted contrastive loss, joint loss, lambda progression, latent norms, checkpoint selection/state coherence, and inference diagnostics against the Sprint 4 acceptance criteria. Record the evidence and remaining findings in `artifacts/sprint-4/task-12.md`.
- [x] **Task 13 — Correct rerun findings.** Resolve the three actionable review findings: make resume epoch/range derive coherently from restored step and stop cleanly at configured total epochs; record negative-pair similarity and contrastive margin; record learning rate and unclipped gradient norm without retaining autograd state. Update focused regression coverage and export a replacement source/notebook pair because notebook/runtime behavior changes. Record evidence in `artifacts/sprint-4/task-13.md`.
- [x] **Task 14 — Close the validated training sprint.** Confirm the user-operated run demonstrates coherent checkpoint state and no unresolved training-instability finding, update project documentation and sprint status, and record final evidence in `artifacts/sprint-4/task-14.md`.
- [~] **Task 15 — Restore training-mode semantics.** Ensure every training epoch explicitly enters model training mode so dropout and conditional-normalization updates cannot silently run under evaluation semantics. Add focused regression coverage and record evidence in `artifacts/sprint-4/task-15.md`.
- [ ] **Task 16 — Require full-ramp checkpoint eligibility.** Gate best-state selection until the configured contrastive ramp has reached `lambda_max`, including exact warmup and ramp boundaries, so only genuinely joint-trained states can win. Add focused regression coverage and record evidence in `artifacts/sprint-4/task-16.md`.
- [ ] **Task 17 — Make resume horizons and epoch boundaries explicit.** Reject or correctly handle scheduler-horizon changes and checkpoints saved inside an epoch so resumed training cannot replay batches or step past the configured total. Synchronize canonical/Kaggle notebooks and focused boundary coverage. Record evidence in `artifacts/sprint-4/task-17.md`.
- [ ] **Task 18 — Align learning-rate diagnostics with optimizer steps.** Record the learning rate used by the current optimizer update rather than the next scheduled update, preserving machine-readable aggregation. Add focused regression coverage and record evidence in `artifacts/sprint-4/task-18.md`.
- [ ] **Task 19 — Align the public criterion lambda default.** Make direct `JointRepresentationCriterion` construction use the same `lambda_max=0.1` contract as `V1Config`, or require an explicit schedule without ambiguity. Cover direct construction and serialization compatibility; record evidence in `artifacts/sprint-4/task-19.md`.
- [ ] **Task 20 — Correct closeout and documentation claims.** Qualify inferred v2 contrastive probability/negative geometry, point the plan to relocated Sprint 2 evidence, remove the nonexistent cutout claim, and update Task 14 evidence without overstating diagnostics added only after the run. Record evidence in `artifacts/sprint-4/task-20.md`.
- [ ] **Task 21 — Make export provenance reproducible.** Record dirty-tree/patch provenance sufficient to reproduce uncommitted Sprint 4 sources, or export from a commit containing those exact sources; regenerate a distinct recommended source/notebook package with accurate manifest/checksums. Do not execute notebooks. Record evidence in `artifacts/sprint-4/task-21.md`.
- [ ] **Task 22 — Reconcile final sprint closeout.** Consolidate Tasks 15–21 evidence, update plan/changelog/recommended export status, and leave Sprint 4 ready for a fresh sprint-wide review without declaring that gate passed. Record evidence in `artifacts/sprint-4/task-22.md`.

## Acceptance Criteria

- Prediction and contrastive branches share an explicit, tested input-normalization contract.
- Context and target latent scale is bounded and observable without breaking EMA or stop-gradient behavior.
- Contrastive optimization uses a dedicated projection surface and no longer trivially saturates under the default configuration.
- Validation/model selection uses a stationary metric or fixed evaluation weighting; selected model weights have completed joint training.
- Saved model, optimizer, scheduler, global step, lambda state, and reference bank describe one coherent state.
- The exported notebook reports enough component and latent diagnostics to attribute any loss movement.
- A versioned source bundle and matching notebook are exported with checksums and static/focused verification evidence.
- `lambda_max` is a validated notebook parameter used consistently by scheduling, stationary selection, resume diagnostics, and exported run history; the next diagnostic run defaults to `0.1`.
- The training notebook is never auto-executed by the agent workflow. The external gate is the user's manual run of the Task 11 revised export.

## Review Gates

- [x] Task 1 root-cause audit has a passing `evidence-reviewer` review — PASS (`Sprint4Task1Evidence`; no unresolved actionable findings).
- [x] Tasks 2–7 stabilization batch has a passing `evidence-reviewer` review — PASS (`Sprint4StabilizationEvidence`; high confidence, no unresolved actionable findings).
- [x] Tasks 8–9 rerun-export batch has a passing `evidence-reviewer` review — PASS (`Sprint4ExportEvidence`; high confidence, no unresolved actionable findings).
- [x] Tasks 10–11 configurable-lambda/export batch has a passing `evidence-reviewer` review — PASS (`Sprint4LambdaExportRecheck`; high confidence, zero findings).
- [x] **Mandatory external stop gate:** PASS — the user supplied the executed v2 notebook and downloaded checkpoints for Task 12 analysis.
- [x] Tasks 12–13 user-run evaluation/remediation batch has a passing `evidence-reviewer` review — PASS (`Sprint4RunRemediationEvidence`; high confidence, no unresolved actionable findings).
- [ ] Sprint-wide review findings from `Sprint4FinalReview` are fully remediated and have a passing task-level `evidence-reviewer` review.
- [ ] The complete sprint has a passing fresh `reviewer` review with no unresolved actionable findings — prior review `Sprint4FinalReview` failed with 10 actionable findings.

## Notes / Blockers

- The baseline run is `experiments/20260904/train-v1-representation-kaggle-run-1.ipynb`; the requested `experiments/20260409` directory was absent.
- Baseline evidence: best validation joint loss was 0.239043 at epoch 2 under lambda zero; by epoch 14 it reached 0.563693, with 93.8% of the increase attributable to prediction-loss degradation. The saved checkpoint mixed epoch-2 model weights with step-7840 optimizer/scheduler metadata.
- Initial candidates requiring Task 1 disposition: branch input-scale mismatch, contextual/EMA target latent-scale drift, trivial contrastive augmentation/temperature, direct objective competition on shared latents, non-stationary checkpoint selection, and incoherent restored checkpoint state.
- User-run evidence is available under `experiments/20260409/(v2)` with downloaded checkpoint files under `checkpoints/`. Task 12 analysis must remain read-only.
- User-operated rerun evaluated in Task 12 (`train-v1-representation-kaggle-run-2.ipynb` and `v1_representation_20260904_01.pt`): validation prediction loss reduced to 0.0578, InfoNCE stabilized at 0.6975 (49.78% positive probability, orthogonal negative similarity ~0.027), latent norms bounded at ~11.31, checkpoint coherent at step 7840 with matching reference bank (norm 7.085).
- Task 13 remediated resume loop epoch derivation, negative similarity/margin diagnostics, and unclipped grad norm/lr logging, producing verified replacement export `exports/kaggle-20260904-03/`.
- Task 14 consolidated all evidence into `artifacts/sprint-4/task-14.md`; Sprint 4 is ready for sprint-wide review.
- `Sprint4FinalReview` failed the first sprint-wide gate with 10 actionable findings spanning training mode, full-ramp model selection, resume boundaries/scheduler horizon, LR diagnostic alignment, criterion defaults, export provenance, and documentation accuracy. Tasks 15–22 own remediation.
- Tasks 15–22 remain pending. The user explicitly stopped Sprint 4 remediation and later authorized only the new Sprint 5 latent-geometry goal; do not resume these tasks without a new instruction.
