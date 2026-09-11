# Sprint 16 — Representation Failure Localization and Component Attribution

## Sprint Goal

Identify, with stagewise measurements and controlled component substitutions, which part or parts of the representation-to-score pipeline lose or fail to exploit observable anomaly signal; do not attempt an unconstrained end-to-end redesign.

**Status:** READY — Sprint 15 benchmark recovery returned accepted `MEASURABLE` with a reviewed attribution handoff (`artifacts/sprint-15/task-22.md`). Benchmark-dependent execution may start only after the mandatory post-final evidence review passes. Sealed H-SEAL-37..40 remain forbidden to open/score. Sprint 16 has not started.

## Decision Context

Sprint 12 established that observable telemetry contains useful signal while the learned pipeline does not exploit it reliably:

- diagnostic handcrafted features: historical file AUROC 0.663 and paired-development ranking 0.948, but operational FPR 0.669;
- accepted frozen learned representation: historical file AUROC about 0.509–0.514 and paired-development ranking 0.712;
- revised target-hidden scorer: paired-development ranking 0.590;
- restored shipped scores: historical file AUROC 0.471–0.481;
- global identity-copy behavior was refuted, but active copying and encoder-versus-head causality remain unresolved.

The central question is therefore: **“Where does useful signal disappear or become unreadable between raw telemetry and the final file score?”**

Sprint 16 owns attribution, not production recovery. A later Sprint 17 may redesign only components supported by Sprint 16 causal evidence.

## Pipeline Under Attribution

```text
raw telemetry
→ conditional normalization
→ patchification
→ local patch encoder
→ sequence/context encoder and learning objective
→ patch/file embedding and pooling
→ conditional geometry/reference bank
→ scorer
→ patch-to-file aggregation
```

Candidate weaknesses include:

- normalization removing amplitude, gain, spectral, or cross-channel information;
- patch size/stride/padding or within-patch pooling diluting short/local anomalies;
- local encoder failing to retain transient, phase, coupling, or degradation signals;
- masking/prediction/contrastive objectives learning interpolation, smoothness, or nuisance shortcuts instead of physical context;
- contextual mixing diluting localized anomalies;
- mean/file pooling allowing healthy patches or duration to dominate;
- conditional geometry using inappropriate centroids, covariance, hierarchy, sparse fallback, reference banks, or distance scales;
- query-conditioned prediction, energy terms, or tail aggregation failing to read signal that remains present in latent space.

## Non-Negotiable Scientific Constraints

- Consume only accepted Sprint 15 Fit/Calibration/development/confirmation roles and accepted Sprint 12 checkpoints/artifacts. Sprint 15 sealed histories remain unopened for component selection.
- Freeze hypotheses, diagnostic metrics, component substitutions, bounded intervention budget, and attribution rules before inspecting new comparative outcomes.
- Use handcrafted features as a diagnostic oracle/ceiling, never as proof of a production-ready detector.
- Compare stages with the same support, history units, conditioning hierarchy, splits, and metrics. Do not use a favorable metric for one stage and a different metric for another.
- Keep context-mismatch and population/geometry scores separate before any fusion.
- Labels, masks, failure categories, and simulator state are post-hoc diagnostics only; they never become representation inputs.
- A low metric at stage N+1 after a high metric at stage N is localization evidence, not yet causal attribution.
- Declare a causal bottleneck only when bypassing or replacing the suspected component restores the signal across independent development/confirmation histories without unacceptable nuisance or background regression.
- Change one principal mechanism per intervention. Do not simultaneously redesign normalization, encoder, objective, geometry, and scorer.
- Linear or nonlinear probes are diagnostic readers of information content, not production success and not substitutes for the shipped scorer.
- Do not fit calibrated risk, tune production thresholds, or claim operational early warning in this sprint.

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done.

### Batch A — Attribution contract and common harness

- [ ] **Task 1 — Verify accepted inputs and runtime provenance.** Resolve the accepted Sprint 12 control/hybrid checkpoints, reference banks, handcrafted baseline, corrected target-hidden experiment, Sprint 15 development/confirmation roots, hashes, roles, and canonical runtime. Confirm Sprint 15 sealed roots are not accessible to the diagnostic workflow. Evidence: `artifacts/sprint-16/task-1.md`.
- [ ] **Task 2 — Freeze component hypotheses and attribution rules.** Predeclare the candidate components, exact bypass/replacement interventions, allowed hyperparameter budget, history/mechanism slices, stopping rules, and the evidence required for `HEALTHY`, `SUSPECT`, `BOTTLENECK`, `MULTIPLE BOTTLENECKS`, or `UNRESOLVED`. Evidence: `experiments/sprint16-attribution-protocol-v1.md` and `artifacts/sprint-16/task-2.md`.
- [ ] **Task 3 — Implement one shared diagnostic metric contract.** Build a common evaluator for paired ranking, severity ordering, nuisance false alarms, unaffected-background stability, localization, historical/static file discrimination, per-history uncertainty, and category slices. Enforce identical support, standardization provenance, conditioning, tie handling, and aggregation unless the tested component is aggregation itself. Evidence: `artifacts/sprint-16/task-3.md`.
- [ ] **Task 4 — Establish the raw and handcrafted oracle ceiling.** Reconfirm on accepted Sprint 15 development/confirmation roles that raw/handcrafted observable features contain measurable signal for each eligible failure/anomaly category. Separate physical signal absence from representation loss and preserve the FPR/scale limitations that make the oracle diagnostic-only. Evidence: `artifacts/sprint-16/task-4.md`.

### Batch B — Stagewise signal-retention localization

- [ ] **Task 5 — Measure conditional-normalization retention.** Compare pre-normalization and post-normalization observable probes for level/gain, slope/drift, spectral bands, cross-channel coupling, phase/timing, localized transients, persistent degradation, and benign nuisance controls. Hold downstream diagnostic reader and support fixed. Evidence: `artifacts/sprint-16/task-5.md`.
- [ ] **Task 6 — Measure patchification retention.** Quantify signal preservation across valid-length handling, padding, window size, stride, overlap, localized support, and within-patch summarization without changing the encoder. Determine whether short or boundary anomalies are diluted before learning. Evidence: `artifacts/sprint-16/task-6.md`.
- [ ] **Task 7 — Probe local patch-encoder information.** Evaluate frozen local latents with the shared diagnostic reader and compare them against post-patchification observables on identical patches. Measure which physical signal families remain linearly or simply nonlinearly readable. Evidence: `artifacts/sprint-16/task-7.md`.
- [ ] **Task 8 — Measure contextual-encoder retention.** Compare local and contextual latents at matched patch positions and supports. Test whether contextual mixing improves context-dependent mechanisms or dilutes localized anomaly evidence, without yet changing the objective. Evidence: `artifacts/sprint-16/task-8.md`.
- [ ] **Task 9 — Diagnose objective and masking pressure.** Using bounded frozen-checkpoint or tiny development-only interventions, measure whether masked prediction, target construction, contrastive weighting, gradient competition, or mask composition encourages interpolation/self-conditioning/nuisance invariance instead of retaining physical signals. Keep architecture fixed. Evidence: `artifacts/sprint-16/task-9.md`.
- [ ] **Task 10 — Measure file embedding and pooling retention.** Compare valid-patch distributions, mean/file embeddings, duration effects, sparse anomalies, and controlled alternative diagnostic pooling while holding latent representations fixed. Determine whether healthy-patch dominance erases localized evidence. Evidence: `artifacts/sprint-16/task-10.md`.
- [ ] **Task 11 — Measure conditional-geometry readability.** Feed both oracle features and learned features through the same regularized hierarchical healthy-only geometry. Compare simple diagnostic readers against production conditional centroids, covariance, hierarchy, fallback, reference-bank, and distance scaling. Determine whether geometry loses information already present in its inputs. Evidence: `artifacts/sprint-16/task-11.md`.
- [ ] **Task 12 — Measure scorer and aggregation readability.** Hold representations and geometry fixed while comparing the production context/population scores with predeclared simple diagnostic scorers and patch-to-file aggregations. Test residual sensitivity, energy components, tail behavior, duration dependence, and localized support without fusing independent scores. Evidence: `artifacts/sprint-16/task-12.md`.

### Batch C — Component substitution and causal attribution

- [ ] **Task 13 — Execute the component substitution matrix.** Run the predeclared cross-combinations: handcrafted features with simple reader and production geometry; local/contextual learned latents with the same reader; learned latents with production and alternative scorers; oracle features through production pooling/aggregation; and frozen encoders with target-hidden scoring. Change one boundary at a time and report per-history deltas. Evidence: `artifacts/sprint-16/task-13.md`.
- [ ] **Task 14 — Apply bounded interventions to suspect components.** For each localized suspect, execute only the frozen minimal bypass or replacement needed to test causality. Examples may include signal-preserving normalization, support-preserving patch settings, local-versus-context bypass, fixed objective/masking ablation, diagnostic pooling, alternative regularized geometry, or non-query-conditioned scoring. These are confirmation experiments, not final redesigns. Evidence: `artifacts/sprint-16/task-14.md`.
- [ ] **Task 15 — Replicate intervention recovery across histories and mechanisms.** Require recovery on independent development/confirmation histories and relevant mechanism categories. Quantify nuisance false alarms, unaffected-background stability, severity behavior, localization, and uncertainty. A single seed, pooled-only improvement, or recovery accompanied by unacceptable nuisance regression cannot establish a bottleneck. Evidence: `artifacts/sprint-16/task-15.md`.

### Batch D — Attribution verdict and closeout

- [ ] **Task 16 — Publish the component attribution matrix.** For normalization, patchification, local encoder, context encoder, objective/masking, file pooling, conditional geometry, scorer, and aggregation, report signal retention, localization evidence, intervention recovery, replication, residual uncertainty, and verdict: `HEALTHY`, `SUSPECT`, or `BOTTLENECK`. Return the overall state `IDENTIFIED`, `MULTIPLE BOTTLENECKS`, or `UNRESOLVED`. Evidence: `artifacts/sprint-16/task-16.md`.
- [ ] **Task 17 — Define the bounded Sprint 17 recovery scope.** Convert only causally supported bottlenecks into proposed implementation tasks and gates. Do not prescribe a wholesale architecture rewrite when attribution is unresolved, and do not include calibrated-risk work before useful event ranking. Evidence: `artifacts/sprint-16/task-17.md`.
- [ ] **Task 18 — Pass batch evidence gates.** Review each completed batch with an evidence reviewer. Any actionable defect returns to the same implementation owner and the affected gate repeats. Preserve all failed, corrected, and superseded records. Evidence: `artifacts/sprint-16/review-*.md`.
- [ ] **Task 19 — Pass differential review and finalize Sprint 16.** Run a sprint-wide differential deep review over accepted evidence. Finalize only with zero actionable findings and an explicit attribution state. Update `docs/PLAN.md` without claiming production recovery, independent sealed performance, early-warning utility, or calibrated risk. Evidence: `artifacts/sprint-16/task-19.md` and `artifacts/sprint-16/deep-review-final.md`.

## Component Substitution Matrix

The minimum matrix must preserve these comparisons:

| Representation/input | Geometry or diagnostic reader | Question |
|---|---|---|
| Handcrafted observable features | Simple regularized reader | Is observable signal present and what is the diagnostic ceiling? |
| Handcrafted observable features | Production conditional geometry | Does geometry erase signal that is present in oracle features? |
| Frozen local patch latents | Same simple reader | Does the local encoder retain signal? |
| Frozen contextual latents | Same simple reader | What changes after contextual mixing? |
| Frozen learned latents | Production geometry | What does the current pipeline deliver? |
| Frozen learned latents | Predeclared alternative simple scorer | Is latent signal present but unreadable by the production scorer? |
| Oracle/controlled features | Production pooling and aggregation | Does file reduction erase localized signal? |
| Frozen encoder | Target-hidden/non-query-conditioned scorer | Does scorer self-conditioning suppress sensitivity? |

## Attribution Gates

### G1 — Raw-signal confirmation

- The handcrafted/raw oracle must demonstrate measurable signal on the relevant development/confirmation categories.
- Categories without observable signal are labeled data/physics limitations, not representation failures.

### G2 — Stagewise retention

- The same metrics and supports must trace signal through normalization, patchification, local encoding, contextual encoding, pooling, geometry, scorer, and aggregation.
- A metric drop localizes a suspect boundary but does not establish causality.

### G3 — Component substitution

- Representation quality must be separated from scorer quality.
- Local encoding must be separated from contextual encoding.
- Raw/normalized/patchified inputs must be separated from learned latents.
- Geometry and file aggregation must be tested with both oracle and learned inputs.

### G4 — Causal intervention

A component is a `BOTTLENECK` only when:

1. signal is present before the component and reduced after it;
2. the frozen minimal bypass or replacement restores the signal;
3. recovery repeats across independent histories and relevant mechanisms;
4. recovery does not create unacceptable nuisance false alarms or background instability;
5. the result is not dependent on one seed, one pooled aggregate, or one favorable metric.

### G5 — Attribution verdict

The sprint must return exactly one overall state:

- **IDENTIFIED:** at least one replicated causal bottleneck is established;
- **MULTIPLE BOTTLENECKS:** multiple independent replicated bottlenecks are established;
- **UNRESOLVED:** evidence is insufficient to assign causality.

`UNRESOLVED` is an acceptable scientific result. It must not be converted into a guessed redesign.

## Acceptance Criteria

1. Accepted Sprint 15 development/confirmation data and Sprint 12 checkpoint/reference provenance are verified before execution.
2. Raw-signal availability is separated from representation retention for every eligible mechanism/category.
3. All pipeline stages are evaluated with a common diagnostic contract and comparable support.
4. The component substitution matrix separates normalization, patchification, local encoding, contextual encoding/objective, pooling, geometry, scorer, and aggregation.
5. Any bottleneck claim contains both stagewise localization and replicated intervention recovery.
6. Handcrafted and learned probes remain diagnostic; no probe metric is presented as production performance.
7. Sealed Sprint 15 histories remain unopened for component selection and attribution.
8. No calibrated-risk, probability-quality, production-threshold, or operational early-warning claim is made.
9. The final component matrix reports evidence and uncertainty for every candidate component, not only the preferred explanation.
10. Evidence and differential review gates pass with zero actionable findings.

## Scientific Stop Rules

- If Sprint 15 is not `MEASURABLE`, block benchmark-dependent attribution; do not redesign metrics inside Sprint 16.
- If the raw/handcrafted oracle cannot detect a category, do not attribute that category's failure to the learned representation.
- If stagewise probes disagree without a successful intervention, mark the component `SUSPECT` or the sprint `UNRESOLVED`.
- If an intervention improves one metric but breaks nuisance/background gates, it does not establish a usable bottleneck correction.
- If several boundaries independently lose signal, return `MULTIPLE BOTTLENECKS`; do not force a single-cause narrative.
- Do not open sealed histories to resolve ambiguous attribution.
- Do not launch a large retraining run or flexible architecture search. Targeted production recovery belongs to Sprint 17.

## Explicit Non-Goals

- Delivering a production-ready representation model.
- Maximizing AUROC through unconstrained architecture or hyperparameter search.
- Simultaneously replacing objective, encoder, geometry, and scorer.
- Treating a linear probe as evidence that the production scorer is adequate.
- Calibrated survival/risk modeling, Brier/ECE/reliability evaluation, or deployment thresholds.
- Reopening Sprint 12 blocked-final Tasks 10 or 13.
- Using Sprint 15 sealed evaluation histories for development decisions.

## Notes / Blockers

- Sprint 15 `MEASURABLE` plus zero-actionable closeout is the hard prerequisite for benchmark-dependent execution.
- Sprint 16 may finish as `IDENTIFIED`, `MULTIPLE BOTTLENECKS`, or `UNRESOLVED`; favorable model performance is not required for task acceptance.
- Sprint 17, if opened, owns targeted recovery of only the causally supported bottlenecks and must freeze new advancement gates before implementation.
