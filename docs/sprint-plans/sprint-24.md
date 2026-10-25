# Sprint 24 — Label-Efficient Transfer and Continual Learning on the Sprint 18 Baseline

## Status and authority

**Status:** Phase A and Task 4 complete; Phase B Task 5 in progress and Tasks 6–7 pending. Tasks 8–12 are not released for execution. No empirical model result is claimed.

**Why Sprint 24:** Sprints 19–23 already exist. This is the next unused sprint number, continuing the scientific direction of Sprint 18 without overwriting the intervening research/workflow records.

**Owner direction (2026-10-25):** Study the reuse of the current representation with a small amount of labels and subsequent sequential updates. Carry over Sprint 18 data unchanged; every new data modification must derive from that baseline. A small miss of the old recovery gate does not by itself disqualify a useful baseline. This does not retrospectively alter Sprint 18's `UNRESOLVED` verdict or approve a replacement numeric gate.

**Current authorization:** Phase A Tasks 1–3 completed their required gates. Owner authorized Phase B Tasks 4–7 with Gold/Gold/Gold/Silver workers respectively, independent evidence-review PASS and separate Bronze exact-snapshot checkpoints per task. Scoped commits on local `master`, non-force push of reviewed commits (including Task 3 checkpoint), safe exact-commit remote synchronization, tiny behavioral smokes and the locked Task 7 comparison are authorized for Phase B-owned content and Main plan updates. Preserve foreign memory and all unrelated changes; never force-add ignored artifacts. Execute only the owner-accepted protocol v1; no source deletion, data generation, Phase C execution, gate relaxation or reuse of consumed Sprint 18 releases.

**Workflow for execution:** Full route because label exposure, chronological evaluation and artifact custody interact. Each task requires an independent evidence-review PASS and a separate Bronze exact-snapshot checkpoint before the next independent task; final differential deep review is Task 12. Evidence belongs under `artifacts/sprint-24/`, not in this plan. Select worker roles and obtain execution/checkpoint authority at dispatch; this plan grants neither automatic routing nor Git authority. Apply the default two-review-round / bounded-third-round policy from the installed workflow. Planning itself is a Main-owned, documentation-only change and does not mark any execution task complete.

## Research question

Does the existing self-supervised representation let us learn useful anomaly decisions from fewer labels than training from scratch, and improve through sequential feedback without losing previous capabilities?

Source interpretation: [Sprint 18 baseline and continual learning](../sprint18-baseline-and-continual-learning.md).

Two connected questions must be answered separately:

1. **Label-efficient transfer:** Freeze the encoder, train a small supervised head, and evaluate on later data. Is the information already present in the embedding, even if the existing anomaly scores do not exploit it?
2. **Continual learning:** Update with progressively revealed labels; measure gains on new faults/regimes and retention of old ones after every update. A frozen encoder does not prevent the head from forgetting.

Completion means a reproducible answer, including a negative or unsupported answer. It does not require manufacturing a positive performance result.

## Scope and non-goals

- Primary representation: the three accepted B0-FULL model seeds `181801`, `181802`, `181803`, selected as the baseline identity, not by ranking Confirmation outcomes.
- B0 `S_pred`, B0 `S_pop`, and N4-CRN `S_pred` are separate no-new-label score comparators. N4 shares the frozen B0 representation; it is not a separate encoder candidate. No score fusion is assumed.
- Primary supervised method: a frozen B0 encoder with a small linear classification head on the existing file embedding. Any feature normalization is fitted only on the permitted adaptation-training subset.
- Fault detection is the primary endpoint. Fault-type analysis and new-fault/regime introduction require supported labels and counts; no synthetic simulator variable becomes a predictor input. Simulator labels are an explicitly simulated feedback source, not evidence of real annotator quality.
- Include a matched randomly initialized frozen encoder + identical head control, and a same-architecture model trained from scratch on the same revealed labels. These distinguish representation value from head capacity and extra pretraining. Report pretraining cost separately; identical label budgets do not imply identical total compute.
- Partial/full encoder fine-tuning is a conditional diagnostic, not the default. Task 3 must freeze its trigger, scope and budget before outcome inspection. If triggered, report it separately from frozen-encoder transfer; if not triggered, record `NOT_TRIGGERED`, not an unrun positive claim.
- No broad architecture search, resumption of N1, repair of Sprint 18 gates, real-data deployment, calibrated-risk claim or Sprint 15 Sealed access.
- Keep reference banks fixed in the primary study. Online bank adaptation is excluded so that label learning and bank drift are not confounded. Any later bank study requires a separate confirmed-healthy update/contamination policy.

## Immutable Sprint 18 data inheritance

**Carry over the exact accepted Sprint 18 data, not merely the generator settings or a fresh dataset with the same name.**

Authoritative lineage:

- [Accepted candidate binding](../../experiments/sprint18-iterative-assessment-c5-allowance-v1.json): `S18-ITER-0005`; canonical binding digest `8e2705140d07d9c603989ec709de5ebb766e580a8619b37c7462f6dd2751dd3b`.
- [Certified data handoff](../../experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json) and [handoff description](../../experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.md).
- Source root: `/home/trietlm/anomaly-representation-learning-s18t68-a02-worktree/data/generated/sprint18-iterative-v1/S18-ITER-0005-ASSESS-P-ALLOWANCE-V1`.
- Preserve all 16 histories: 4 Design, 3 Fit, 1 Calibration, 4 Development and 4 Confirmation, including original role IDs, seeds, raw shards, annotations, manifests and assessment/qualification custody. Preserve the original role semantics as provenance even where the new research defines additional views.
- [Sprint 18 Confirmation lock](../../experiments/sprint18-task33-confirmation-lock.json) binds the accepted model/scorer artifacts; [final verdict](../../artifacts/sprint-18/task-38.md) records limits and final identities.

Rules:

1. Source data and Sprint 18 records are read-only. Copying for storage relocation must be byte-preserving and hash-verified; never regenerate and call the result the original data.
2. Label budgets, label-release times, chronological partitions and experience schedules are new versioned manifests referencing original history/file/event IDs. Do not rewrite waveform shards, annotations or original split assignments to implement a new view.
3. Any actual data change is a separately versioned child of this exact source: record parent hashes, transformation/configuration, seeds, affected IDs, purpose and output hashes. Preserve an unchanged baseline comparison. New generation is not part of the primary study and needs a prospective amendment; no silent repair of class balance or weak results.
4. Audit the available anomalies, healthy support, event identities, timestamps, robot/program support and pretraining exposure before fixing numeric label budgets or experience boundaries. Do not assume that a healthy-only training selection means the entire Fit history contains no labeled faults.
5. Group all patches and repeated views of a file together. Keep related files from one fault episode/unit trajectory from leaking across adaptation and evaluation boundaries; define temporal exclusion windows around overlapping events before model runs.
6. Map every adaptation, validation, retention-test and future-test ID to its source role and prior exposure. Do not use future test labels for sampling, balancing, threshold choice, replay selection or stopping. A withheld-fault scenario may use labels for a declared offline scenario construction, but the learner sees only labels released by that scenario.
7. Sprint 18 Development/Confirmation results have already been inspected. Reuse is **secondary research on an exposed benchmark**, not fresh independent Confirmation. The new protocol must explicitly reconcile permitted reuse with the historical role restrictions; it cannot reopen or modify the old one-shot experiment. Keep original Confirmation evaluation-only, with no adaptation, tuning or checkpoint selection. No fresh generalization claim follows from changing a manifest name.
8. Audit the temporal exposure of the inherited encoder, normalization and bank as well as the supervised head. If pretraining used observations later than a proposed evaluation cutoff, call the result sequential label adaptation with prior unlabeled exposure, not a strictly prospective end-to-end temporal experiment. A strictly causal claim needs a separately approved training design, not concealment of this exposure.

## Prospective comparison and measurement contract

Task 3 freezes exact executable choices after the support audit and before running model comparisons. Numeric performance targets are **not yet approved**; do not import Sprint 18's recovery conjunction or invent a positive-success gate.

### Label efficiency

- Use multiple nested label budgets with the same eligible IDs, label-release policy and sampling seeds across compared methods. Include healthy labels in the stated budget; state whether the annotation unit is a file, event or episode and also report effective independent episode counts.
- Separate training labels, validation/threshold labels and untouched evaluation labels. Report the total annotation cost, not only head-training labels. The smallest budgets must be feasible without replacement or hidden oversampling of labeled episodes.
- Evaluate frozen-pretrained head, random-frozen head and scratch-supervised model on identical future support. Preserve the existing score comparators under their original thresholds for continuity; any new operating-point comparison must use a shared prospectively declared calibration rule, never test tuning.
- Freeze architecture, optimizer, training/early-stopping policy, seeds, search budget and compute limits. The scratch baseline must receive a fair declared training budget, not an intentionally undertrained strawman.

### Sequential updates

- Build experiences from the inherited timeline, with label availability and delayed feedback explicit. Each observation is evaluated before its label can update the learner; next-period evaluation remains untouched by the current update.
- Compare: (a) initial head with no updates, (b) sequential head updates without replay, (c) sequential updates with a fixed-size replay buffer. Freeze buffer size in annotation/storage units and the past-only selection policy before outcomes. No future examples enter replay.
- Add a cumulative-label retraining reference using all labels revealed so far, under a declared compute budget. It is a retrospective upper reference for the available labels, not an online method and not access to future labels.
- Evaluate fixed, disjoint old-task retention probes after every update plus new/current and next-period support. Probe results are reporting-only, never a model-selection signal. Freeze the schedule without consulting those results.
- New-fault and new-regime scenarios must have actual source support. Distinguish newly labeled from previously unseen in raw pretraining. Unsupported scenarios are reported with counts and limits; do not fabricate examples or claim that ordinary repeated batches test learning new faults.

### Required outputs

- Label-budget curves: PR-AUC, ROC-AUC where defined, recall and false-positive rate at the declared operating point; per-fault/per-regime support and metrics. Report undefined single-class endpoints as unavailable, not zero.
- Sequential performance matrix by update and experience; old-task forgetting (best prior performance minus current performance on the same fixed probe), new-task acquisition, forward evaluation and false alarms after each update.
- Where event structure supports it, report event recall, false alarm episodes per healthy exposure time and detection delay, with units/denominators and censoring rules. Do not silently substitute file-level false-positive rate for event false alarms.
- Paired method differences on matched source histories/episodes and seeds; uncertainty resampled at the justified independent grouping level, not patches/files treated as independent replicas. State small-history limitations and do not interpret a within-arm interval as a paired interval.
- Per-seed and aggregate results, prevalence and class counts, labeling/replay memory costs, update/inference time, trainable parameters and pretraining versus adaptation compute. Publish all planned methods/budgets, failures and unsupported cells; no favorable-seed selection.

## Ordered atomic tasks

### Phase A — Preservation and executable study design

- [x] **Task 1 — Preserve and verify the inherited assets.** Inventory the transitive data/checkpoint/bank/scorer/config/custody dependencies from the accepted handoff, lock and publication. Produce a hash-bound retention/relocation manifest; verify destination copies and load compatibility on non-Confirmation support before releasing any source directory for deletion. No deletion is part of this task. Acceptance: all required bytes accounted for, missing assets explicit, archive paths mapped without editing immutable S18 records. See [retention note](../sprint18-retention-for-sprint24.md).
  - Evidence [round 2 PASS](../../artifacts/sprint-24/reviews/task-1-evidence-r2.md); [worker result](../../artifacts/sprint-24/task-1.md); [Bronze checkpoint PASS](../../artifacts/sprint-24/checkpoints/task-1-cp01.md), commit `129f363d7e9c72a22908cd07d9e0ba78338844c1`. Accepted assets remain source-retained except the verified 57-file manifest/bank archive; no source directory released for deletion. Owner approved whole-file staging of the three plan/retention documents; memory and ignored artifacts remain excluded.
- [x] **Task 2 — Audit temporal and label support.** On the preserved corpus, produce the source-ID/exposure inventory, label/episode counts and feasible chronological grouping. Record actual pretraining support and which faults/regimes can be introduced sequentially. Acceptance: a leakage-auditable support report, including unsupported scenarios, with no model fitting or performance-based sampling.
  - [Worker audit](../../artifacts/sprint-24/task-2.md), [round 1 evidence PASS](../../artifacts/sprint-24/reviews/task-2-evidence.md), [Bronze checkpoint PASS](../../artifacts/sprint-24/checkpoints/task-2-cp01.md), commit `7a2c6987830f405ea66a3d4d74f383f93c1f3f19`. Protocol design must retain history-namespaced grouping, explicit pretraining exposure and unsupported scenario limits.
- [x] **Task 3 — Freeze the prospective research protocol.** Specify the exact source-to-role views, nested budgets, feedback delays, experience schedule, method matrix, thresholds, replay policy, conditional fine-tuning trigger, compute budgets, metrics and uncertainty. Explicitly approve the new secondary-use boundary without altering S18. Obtain independent pre-execution review and owner acceptance of any practical performance/forgetting tolerances and data-role amendments. Acceptance: executable choices contain no unresolved values; negative results remain acceptable; no actual benchmark model execution before this gate.
  - Owner accepted protocol-v1 boundaries: Development adaptation/evaluation is secondary research with prior exposure; original Confirmation remains evaluation-only; no numeric success/recovery/forgetting gate. A separate 100-file validation trigger (`V-TRIG`, seed `2410`) controls bounded context-encoder fine-tuning at `mean(M3−M1)>0.05` and `mean(M1)<0.70`, never using primary evaluation or retention probes. Unsupported regime-type/duration scenarios and identity-only reserve slices remain explicit. This approves Phase A design only, not model execution.
  - [Frozen prospective protocol](../sprint24-prospective-protocol-v1.md), [worker evidence](../../artifacts/sprint-24/task-3.md), [round 2 evidence PASS](../../artifacts/sprint-24/reviews/task-3-evidence.md), [Bronze checkpoint PASS](../../artifacts/sprint-24/checkpoints/task-3-cp01.md), commit `2f7cdb012c4325cd7a7a5aa09fb886efc992cf87`. Data-only feasibility checks passed; no model comparison executed. B0/N4 asset custody, exposed-benchmark limits and unsupported scenarios remain binding.

### Phase B — Reusable evaluation machinery

- [x] **Task 4 — Implement immutable views and the label-release boundary.** Use existing data loaders with versioned view manifests; enforce ID/group/time separation, delayed labels and past-only replay eligibility. Acceptance: a tiny real fixture rejects overlapping episodes and premature label access and reproduces the same views without mutating source data.
  - Owner-approved grouping clarification/amendment: healthy files outside every event interval receive singleton `(history_id, file_id)` groups; abnormal outside-interval files retain the frozen nearest-same-robot/7-day rule. Sampling allocates groups exclusively across training, validation, trigger and retention roles before exact file-budget selection; shared episode groups across those roles are forbidden. Requires renewed source-support feasibility proof without oversampling or relaxing temporal/trajectory exclusions. No Sprint 18 data or verdict changes.
  - Owner approved source-derived per-history abnormal quotas after direct accepted-source audit found post-eligibility/withholding V-ADAPT abnormal support `90/136/133/110`, versus the old requirement of 106. Preserve exact total label budgets/slice sizes, nested training IDs, shared method support, group exclusivity and eligibility. Lock deterministic metadata-only class allocation before model outcomes, fill with eligible normals, and report realized class/episode counts; no performance-based selection or eligibility relaxation. Renew feasibility proof on all four accepted Development histories.
  - Owner approved reset-edge interval splitting with parent custody: split merged spans at maintenance boundaries but retain episode/trajectory linkage across pieces so they cannot enter conflicting adaptation/validation/trigger/retention/evaluation roles. Files overlapping maintenance remain excluded from adaptation under the unchanged eligibility predicate. Audit deterministic edge assignment and all-history feasibility; no waveform/timestamp edits.
  - Initial cycle's three-round FAIL preserved; owner-authorized linked correction cycle passed [C2 evidence review](../../artifacts/sprint-24/reviews/task-4-c2-evidence.md). [Bronze checkpoint PASS](../../artifacts/sprint-24/checkpoints/task-4-cp01.md), commit `3e002bc6d9feba6ba7e5f055f1f22bd7e371c1ec`. [Amended views contract](../sprint24-views-v2-amendment-a01.md) binds D1–D3, exact file totals, history-specific class counts and all-four-history source feasibility; full public-export smoke and focused views tests passed.
- [~] **Task 5 — Implement frozen feature extraction and score comparators.** Restore B0 checkpoints, normalization and banks as locked; retain file/patch identity and valid lengths; expose file embeddings and separate B0/N4 scores. Cache identity includes checkpoint, source support and preprocessing hashes. Acceptance: real checkpoint + real bank + tiny real dataset through the public entry point to final outputs, frozen parameters unchanged; N4 shares B0 representation and leaves `S_pop` unchanged. Smoke uses permitted non-Confirmation support.
- [ ] **Task 6 — Implement supervised transfer and scratch controls.** Add the fixed linear-head learner, random-frozen control and matched scratch-supervised learner; validation and stopping use only allowed revealed support. Acceptance: public end-to-end training/inference smoke yields auditable label usage, finite outputs and the declared trainable parameter sets; focused behavioral tests cover leakage and checkpoint restoration, not wiring.
- [ ] **Task 7 — Run the frozen label-budget comparison.** Execute the locked budgets/methods/seeds once per planned cell, preserving failed attempts and any explicitly approved operational corrections. Acceptance: complete label-efficiency tables/curves, paired contrasts, costs and uncertainty/limitations, with no evaluation-driven expansion of the search. Evaluate the predeclared fine-tuning trigger; record the resulting disposition.

### Phase C — Sequential learning

- [ ] **Task 8 — Implement the sequential feedback/replay evaluator.** Add no-update, no-replay, bounded-replay and cumulative-label references with frozen experience schedules. Acceptance: a real small sequential smoke proves predict-before-label/update ordering, no future replay access, repeatable buffer state and unchanged retention probes; checkpoint/resume preserves the next eligible update without double consumption.
- [ ] **Task 9 — Run the sequential comparison.** Execute the frozen matrix on inherited chronological views and report after every update, not only at the end. Acceptance: old/new/future performance, forgetting, alarms and cost for every planned seed/method; unsupported new-fault/regime cases remain explicit. Initial static and continually updated models share evaluation support.
- [ ] **Task 10 — Resolve the conditional fine-tuning diagnostic.** If Task 3's trigger was met in Task 7, execute only its predeclared partial/full fine-tuning comparison and carry the selected diagnostic scope through the same retention evaluation without test-based selection. Otherwise publish `NOT_TRIGGERED` with the frozen trigger and observations. Acceptance: an honest, bounded answer about whether encoder adaptation was needed; no open-ended rescue or new architecture search.

### Phase D — Scientific closeout

- [ ] **Task 11 — Publish the research conclusion and reusable package.** Answer label efficiency, sequential improvement and forgetting separately, distinguishing within-benchmark reuse from fresh generalization. Include the immutable-data lineage, source exposure, all result cells, failure/unsupported dispositions, operating-point tradeoffs and restore/run instructions. Acceptance: reusable encoder/head/scorer states and manifests survive outside temporary run roots; no positive claim without its matched evidence; Sprint 18 remains `UNRESOLVED`.
- [ ] **Task 12 — Complete the differential review and close the sprint.** Review cross-task data custody, leakage, label-budget fairness, chronological exposure, comparator identities, conditional branches and retention completeness after all task evidence/checkpoints pass. Acceptance: zero actionable findings; update the global plan with actual scientific conclusions and unresolved limitations, not a forced success verdict.

## Completion criteria

1. The Sprint 18 source corpus and historical artifacts remain unchanged and recoverable; every new view/change has explicit parent lineage.
2. Frozen-encoder label efficiency is compared with no-label scores, random-frozen features and learning from scratch under matched label support.
3. Sequential updates are evaluated for both acquisition and forgetting, including static, no-replay, bounded-replay and cumulative-label references.
4. There is no hidden use of future labels, evaluation episodes, simulator state or already-exposed Confirmation as fresh independent evidence.
5. Practical tolerances, if adopted, are prospectively agreed and reported as tradeoffs. A small old-gate miss is neither silently accepted as historical recovery nor used to reject all reuse research.
6. Evidence, checks, model states, data lineage and independent reviews support every conclusion; insufficient support or negative results are legitimate outcomes.

## Immediate dependency

Before any Sprint 18 cleanup, preserve the assets in [Sprint 18 retention for Sprint 24](../sprint18-retention-for-sprint24.md). The current documentation identifies historical dependency paths; it is not a live server inventory or permission to delete unlisted directories.
