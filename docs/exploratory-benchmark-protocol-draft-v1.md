# Exploratory Benchmark Protocol Draft v1

**Status: DESIGN ONLY / NOT EXECUTABLE / NOT APPROVED FOR DATA CONTACT.** This is a decision-ready design draft, not an approved protocol, benchmark amendment, gate, or authorization. Do not bind roles, histories, seeds, roots, simulator settings, thresholds, or quotas from this document. Do not run a generator, preflight, checker, model, training, or score; do not contact new outcomes or inspect Confirmation/Sealed material; do not continue Sprint 18 or reclassify any historical candidate.

**Stop boundary:** no executable exploratory verdict is available from this draft. A later, separately authorized and independently reviewed protocol version must resolve the prerequisites in [Approval and stop boundary](#approval-and-stop-boundary) before any role/seed binding or data contact.

## 1. Decision and authority

This draft reconciles the reviewed Sprint 19 Task 1 estimand and Task 2 allowance assessment. It preserves Gate V2 and the Sprint 18 decision as they stand. It is not a retroactive interpretation of an old history and cannot produce Gate V2 `MEASURABLE`, an EG2 promotion, or any other V2 candidate verdict. No old failure may be rescued, re-scored under a changed threshold, or used to tune a future rule.

Documentary review status is tracked in the [Sprint 19 plan](sprint-plans/sprint-19.md); consult it for the latest review result, with the linked [Task 3 evidence record](../artifacts/sprint-19/task-3.md) documenting this draft's evidence history. Whatever the review state, this draft remains design-only, non-executable, and not approved for data contact.

The proposal's `H=32`, two-miss allowance, and `12.5–62.5%` cohort-mix band remain illustrative and unapproved. The reviewed Task 2 rationale supports no new soft-pass allowance. The expanded mix band is excluded: its newly permitted `12.5–<15%` and `>60–62.5%` portions contradict Gate V2's unchanged per-history hard `15–60%` mix predicate. No alternative mix exception is proposed here.

## 2. Question, independent unit, and role topology

### 2.1 Exploratory question and unit

The bounded question is whether a representation fitted using verified-healthy Fit data can produce either independent anomaly score—`S_pred` (context/prediction mismatch) or `S_pop` (population mismatch to a healthy reference)—that ranks eligible progressive (`P`) and weak-precursor (`W`) event windows above eligible same-history negative-control windows under a prospectively declared synthetic mechanism and its eligible support.

The independent accounting and resampling unit is one complete whole history. Files, events, windows, metric rows, or repeated profiles on one history/seed do not create additional independent histories. No pooled total or favorable subset can rescue a history-level failure.

### 2.2 Roles and permissions

The following is a permission topology only. Role counts, identities, seeds, roots, and source digests are deliberately unbound and unresolved. Every role must be disjoint at the whole-history, root, and seed level, with support used by a role confined to that role. The complete role roster must be frozen before any materialization or outcome contact; no role may be retried, replaced, omitted, or reused to improve an outcome.
Disjointness is necessary but not sufficient for a fresh Evaluation roster. A future Evaluation must exclude every historical candidate root that has ever been materialized or observed, every research-sandbox seed, and all histories, roots, or seeds previously contacted or assigned to any role; none may be relabeled or reused for Evaluation. Choose and freeze this fresh roster before materialization or outcome contact, using only authorized pre-existing provenance records for exclusions; never reuse old observed histories or their outcomes to select or tune gates, thresholds, quotas, status definitions, or roster membership. Do not access Confirmation or Sealed material to verify freshness or check collisions.

| Role | Permitted use in this design | Prohibited use | Binding status |
|---|---|---|---|
| **Fit** | Verified-healthy rows only to fit the representation, Fit-derived preprocessing/standardization, and any healthy reference bank required by `S_pop`. | Failure/anomaly labels or outcomes; Calibration/Evaluation/Confirmation/Sealed data; score-driven Fit-history selection. | Count, support threshold, identities, seeds, roots, and bank source are unresolved. |
| **Calibration** | Disjoint verified-healthy rows to verify declared healthy support and, only if a future protocol retains thresholded secondary diagnostics, select that diagnostic threshold under a prospectively frozen rule. The threshold-free primary AUROCs do not use Calibration. | Fitting the representation, preprocessing, or reference bank; Evaluation use; choosing the model, score formula, histories, gates, or primary result. | Count, support threshold, identities, seeds, roots, and any threshold rule are unresolved. |
| **Exploratory Evaluation** | The complete, predeclared, disjoint whole-history roster, used once for the two primary score endpoints and prospectively declared diagnostics. | Fitting/refitting; threshold selection for a primary endpoint; roster pruning; generator tuning from model results; later Confirmation reuse. | `H`, sampling regime, identities, seeds, roots, and source digests are unresolved. |

All rows and inputs remain label-isolated: labels, cohorts, failure times, simulator health/future state, and diagnostic-only metadata must not become model-visible inputs or select scores, histories, quotas, or gates. Exploratory Evaluation is not an untouched Confirmation set. Any later Confirmation or Sealed work requires a separate authorization, a separately frozen disjoint roster, and a reviewed contract; this draft grants no access to those roles.

Gate V2 records downstream observable-probe support references of at least 200 verified-healthy Fit files and 6,000 valid patches in aggregate, with fallbacks/sparse groups reported, and at least 40 eligible healthy Calibration rows. These are inherited probe-contract precedents, not evidence that they are sufficient representation-specific Fit/Calibration thresholds. Their applicability and adequacy for a later exploratory representation protocol remain unresolved; no support is declared met here.

## 3. Primary and secondary measurement semantics

### 3.1 Two independent primary endpoints

Keep `S_pred` and `S_pop` as separate score branches. Do not fuse them, compare their raw scales, select a winner, or use either score to choose histories, eligibility, or support. The exact scoring implementations, model/configuration, transformations, patch-to-file mapping, and `S_pop` bank construction/provenance are not frozen by this draft.

For each declared Evaluation history, create one score-independent eligible support set from the prospectively frozen event/window, causal, censoring, maintenance/reset, quarantine, and control predicates. Both score branches must use the same eligible `P+W` event-window keys and same-history control-window keys. Event-window score is the maximum file score over its eligible member files; control-window score is the maximum over its control-window members. Within each history and branch, compute tie-aware AUROC for `P+W` event-window scores against control-window scores; a tied event/control pair contributes one half.

For branch `s ∈ {S_pred, S_pop}` and history `h`, let `E_h` and `C_h` denote its eligible event and control windows. The per-history endpoint is:

$$
A_{s,h}^{P+W} = \frac{1}{|E_h||C_h|} \sum_{e\in E_h}\sum_{c\in C_h} [\mathbf{1}(e>c) + \tfrac{1}{2}\mathbf{1}(e=c)].
$$

For a future fixed roster of `H` histories, the two separate primary descriptive estimates are:

$$
\Theta_s^{P+W} = \frac{1}{H}\sum_{h=1}^{H} A_{s,h}^{P+W}, \qquad s\in\{S_{pred},S_{pop}\}.
$$

Each history has equal macro weight. Do not substitute pooled event-level/file-level AUROC, treat windows or pairwise terms as independent observations, or use a score to change common support. `H` is symbolic here, not a proposed or approved sample size.

### 3.2 Uncertainty and reporting

The uncertainty unit is the whole Evaluation history. Task 1 specifies a paired, deterministic history-block bootstrap convention: resample `H` whole histories with replacement, calculate the unweighted macro per replicate, use the same resampled indices for both branches, and report the 2.5th percentile as `LCB95`; its stated implementation uses `B=2000` and RNG seed `20260202`. This is a Task 1 reporting convention, not a precision guarantee or population-generalization result. Adoption and adequacy for any future roster and sampling regime must be decided prospectively; no LCB cutoff or pass condition is established here.

Report both primary estimates separately, their per-history values, all declared history/support failures, and the uncertainty summary only when the full-roster endpoints are computable under an approved future contract. Report secondary diagnostics separately: P-only and W-only AUROCs, abrupt (`A`) AUROC and support, support counts, and any separately frozen thresholded alert metrics. No secondary result replaces a primary endpoint. Calibration-thresholded recall, lead, or false-alert diagnostics are not primary AUROC criteria.

This design specifies no AUROC cutoff, effect-size target, minimum number of favorable histories, multiplicity rule, or numerical precision target. Gate V2 observable-probe cutoffs—including its EG4 thresholds—are not automatically applicable to these representation endpoints.

### 3.3 Full-roster failure and uncomputability

For each history, record every hard-predicate result and the exact observed support. A failure of any inherited hard structural predicate is `STRUCTURAL-FAIL` for that history and makes the paired primary endpoint `UNCOMPUTABLE`. Also record `UNCOMPUTABLE` if any required event/control score is missing or non-finite, the score branches differ in support keys, or either branch's per-history endpoint cannot be computed.

Do not impute a numeric value, silently drop or replace a history, retry it, or calculate the primary macro/bootstrap on only the computable/favorable subset. Preserve every declared history and predicate in the report. `STRUCTURAL-FAIL` is the reasoned structural state; `UNCOMPUTABLE` is the primary metric state. Neither is a Gate V2 candidate verdict. If all hard support passes and a Task 2 margin comparison is below its reference, the computable endpoint remains reportable; that descriptive margin comparison has no approved exploratory pass/fail effect.

## 4. Inherited hard-predicate registry and precedence

The registry below reproduces the applicable Gate V2 support and integrity constraints as a no-soft-relaxation baseline for design discussion. Gate V2 remains normative for its own benchmark and is not amended or reclassified by this document. No candidate history is being evaluated here. Exact future simulator/window semantics and proof artifacts must be fixed in an approved version before contact; this table is not an executable checker specification.

### 4.1 Role, provenance, and integrity constraints

Any applicable future protocol must preserve Gate V2's hard boundaries: whole-history independence; prospectively frozen, disjoint roles and provenance; no favorable-seed selection or retry/replacement/omission/pooling/regeneration; no learned-model influence on histories, generator settings, quotas, thresholds, or gates; and no leakage of labels or future/simulator-only information into model-visible inputs. Chronology, causal eligibility, censoring, maintenance/reset, quarantine, control-window, and anchor semantics must be frozen before outcomes.

Gate V2 EG0–EG1 specifically require, for their own versioned benchmark, a reviewed protocol/provenance freeze and 100% pass with zero leakage/integrity exceptions: exact version/digest and generator/causal/quota/role/seed/eligibility definitions; deterministic byte-identical manifest and event ledger on repeated fresh-process reloads; chronological ordering; at most one operation per robot at a time; maintenance/recommissioning reset handling exactly once; forbidden future/label/diagnostic fields kept out of model-visible rows; exact mutually exclusive role membership; applicable P/W/A physical-bound and manifestation checks; and zero seal/manifest-digest mismatch. This draft binds none of those versions, roots, identities, or seeds and claims no such checks ran.

### 4.2 Per-history evaluation structural support

The following Gate V2 floors and constraints apply per applicable history under the unchanged source contract; they are not pooled across histories.

| Predicate | Inherited hard requirement |
|---|---|
| Eligible progressive events (`P`) | At least 10. |
| Eligible weak-precursor events (`W`) | At least 10. |
| Eligible abrupt events (`A`) | At least 8. |
| All eligible positive events | At least 30. |
| Deterministic, non-overlapping, same-history/robot negative-control windows | At least 25. |
| Evaluable robot-days | At least 150. |
| Positive-contributing robots | At least 6 distinct robots. |
| Negative-control-contributing robots | At least 6 distinct robots. |
| Program representation | At least 2 programs in each of P and W. |
| Robot concentration | No robot contributes more than 35% of eligible positives or more than 40% of negative controls. |
| Program concentration | No program contributes more than 60% of either predictable cohort, P or W. |
| Cohort mix | Each of P, W, and A contributes 15–60% inclusive of all eligible positives. |
| P/W positive-lead support | In each history, at least 80% of P and at least 80% of W events have at least 3 eligible file endpoints inside the causal 7-day pre-failure horizon, at least one endpoint ending at least 24 hours before failure, no maintenance/recommissioning reset inside the evaluated window, and a clean same-history/robot baseline under the frozen predicate. |

All exclusions and definitions must be applied before counts are calculated. A pooled count cannot cure a per-history miss. These floors do not establish that a future sample size or support precision is adequate.

### 4.3 Fit and Calibration support constraints

The hard role permissions are healthy-only and disjoint as specified in §2.2. The Gate V2 values of Fit `≥200` verified-healthy files and `≥6,000` valid patches in aggregate (with every fallback/sparse conditioning group reported) and Calibration `≥40` eligible verified-healthy rows are recorded support references. Since Gate V2 describes these as downstream support floors for its observable-probe contract, this draft does not assert they suffice for representation fitting or adopt them as final exploratory thresholds. Representation-specific support values are unresolved and must be justified and frozen before any future role binding.

Gate V2 also bars Confirmation, Sealed, quarantined, censored, and maintenance-overlapping rows from Fit/Calibration. Those exclusions cannot be relaxed here.

### 4.4 Design margins are not new exploratory hard gates

For Gate V2, the following are mandatory Design promotion targets, not optional soft gates: P `≥13`, W `≥13`, A `≥10`, all positives `≥38`, controls `≥32`, and robot-days `≥188`. The distinction matters: under that historical contract a Design target miss rejects promotion even when the inherited hard floor is met. A future exploratory rule cannot silently relabel a V2 Design promotion miss.

Task 2 lists these same values only as candidate descriptive margin comparisons for a distinct future exploratory design. Their exploratory soft status is **inactive and unapproved**; no miss allowance exists. They are not permission to weaken Gate V2 or classify an exploratory candidate as passing.

## 5. Candidate soft-predicate registry (inactive)

No candidate soft predicate currently changes a hard result or creates an exploratory pass. The following registry records the only Task 2 margin comparisons that may be described in a later approved design; none is an operative acceptance rule in this draft.

| Candidate descriptive comparison | Reference value | Status in this draft |
|---|---:|---|
| P count margin | `≥13` (Gate V2 hard floor `≥10`) | Inactive / unapproved; report as a descriptive margin only if a future reviewed protocol retains it. |
| W count margin | `≥13` (hard `≥10`) | Inactive / unapproved; descriptive only. |
| A count margin | `≥10` (hard `≥8`) | Inactive / unapproved; descriptive only, not a primary predictive requirement. |
| Total-positive margin | `≥38` (hard `≥30`) | Inactive / unapproved; descriptive only. |
| Control-window margin | `≥32` (hard `≥25`) | Inactive / unapproved; descriptive only, not a construction quota. |
| Robot-day margin | `≥188` (hard `≥150`) | Inactive / unapproved; descriptive only. |
| Expanded cohort-mix band `12.5–62.5%` | Proposal example only | **Excluded.** Its expanded regions violate the inherited hard `15–60%` mix predicate; no history inside those regions can be soft-passed under this draft. |
| Aggregate soft-miss budget | Proposal example: at most two misses on two histories | **None approved.** `H`, a decision objective, the soft registry, dependence model, and prospective justification are unresolved. |

There is no workable soft-pass budget without a prospectively named history-level decision/precision objective, sampling regime, between-history variation and predicate-dependence model, and pre-outcome sensitivity evidence that does not treat files/windows/pairwise terms as independent histories. No allowance may be inferred from the illustrative `2/32` arithmetic. A hard failure, missing evidence, or uncomputable endpoint can never be counted as a soft miss.

## 6. Construction-quota disposition

Sprint 18's `48` deterministic selected negative-control windows per history remains its operative construction reserve/reject-on-shortfall rule. The `32` value is its Gate V2 Design promotion target; `25` is its inherited hard acceptance floor. Task 64/65 found neither that 48 is statistically necessary nor that reducing construction to 32 or 25 is safe. Keep the Sprint 18 reserve, frozen verdicts, blocked Tasks 6–39, and stop before Cycle 4 intact; no retrospective lower-quota requalification or historical rescue is permitted.

The construction quota for a distinct future exploratory protocol is **unresolved**. This draft does not import 48, 32, or 25 as a future construction quota, and provides no basis to reduce Sprint 18's contract. A prospective quota requires its own support/uncertainty rationale, full qualification predicates, and independent approval before any data contact.

## 7. Precedence and unavailable verdicts

### 7.1 Task 1 endpoint states for a future reviewed protocol

If an approved future protocol is eventually authorized, report structural hard failures before interpreting metric results. Any per-history hard structural failure yields `STRUCTURAL-FAIL` and makes the paired full-roster primary endpoint `UNCOMPUTABLE`. If structural support passes but required scores/support are missing, non-finite, branch-mismatched, or otherwise insufficient to calculate an endpoint, that endpoint is `UNCOMPUTABLE`. If all histories pass hard support and both endpoints are computable, report both full-roster estimates and their approved uncertainty summaries; no score threshold or favorable-history rule is supplied by this draft. Candidate margin labels remain secondary descriptions and cannot override these states.

### 7.2 Gate V2's historical candidate precedence remains unchanged

For clarity, the inherited Gate V2 precedence applies only to its own candidate workflow and is not copied into this exploratory draft as an available verdict system:

1. EG0 or EG1 failure rejects a candidate for causal/provenance defects; if roots cannot be generated or verified, the state is `UNAVAILABLE-CANDIDATE`.
2. An EG2 Design miss on any one history is `REJECTED-CANDIDATE`.
3. EG3 metric incomputability despite sufficient units is `UNAVAILABLE-CANDIDATE`.
4. EG4 observable-probe miss is `REJECTED-CANDIDATE`; favorable structural support cannot rescue it.
5. EG5 Confirmation miss on any one history is `REJECTED-CANDIDATE`.
6. EG6 Sealed miss on any one history is `REJECTED-CANDIDATE`.
7. A favorable probe never rescues a structural failure or authorizes representation, anomaly, warning, or calibrated-risk claims. EG7 is a success-only closeout after EG0–EG6 provisionally yield `MEASURABLE` and required evidence reviews pass.

This precedent cannot be shortened into an EG2-only or `MEASURABLE` result for the present design. No exploratory verdict names or decision statuses—including hypothetical pass/fail/unavailable labels—are defined or available until separately authorized in a reviewed executable version. `STRUCTURAL-FAIL` and `UNCOMPUTABLE` above describe prospective per-history/metric reporting states only; they are not an exploratory gate verdict.

## 8. Approval and stop boundary

Before any executable protocol, roster, or role can be frozen, a separate authorization and independently reviewed prospective version must, at minimum:

1. Name the simulator/generator version, configuration and source digests; freeze causal physics and P/W/A definitions, observable inputs, physical bounds, temporal views, event/control-window rules, eligibility, censoring, maintenance/reset, quarantine, and chronology semantics.
2. Justify the Evaluation sampling regime and `H`, role counts, and intended claim/precision objective. Any future Evaluation roster must be genuinely fresh: exclude every historical candidate root ever materialized or observed, every research-sandbox seed, and histories/roots/seeds previously contacted or assigned to any role; mutual disjointness alone is insufficient. Freeze the roster before materialization or outcome contact, derive collision exclusions only from authorized pre-existing provenance records, and do not access Confirmation/Sealed material to verify collisions. Do not use previously observed histories or outcomes to choose or tune gates. Bind identities or seeds only after the protocol passes review and receives the separate user authorization in item 9. Account for whole-history independence and between-history variation.
3. Fix disjoint Fit, Calibration, and Evaluation permissions and provenance; choose and justify representation-specific Fit/Calibration support, reference-bank provenance, and all permitted health-only data.
4. Freeze each score implementation and provenance (`S_pred`, `S_pop`), transformations, patch-to-file mapping, support-key generation, metric implementation, tie behavior, and missing/non-finite/error handling.
5. Decide whether to adopt the Task 1 bootstrap implementation (`B=2000`, seed `20260202`) for the justified roster, what uncertainty summary is appropriate, and whether any decision threshold is scientifically justified. Keep `S_pred` and `S_pop` as the current two primaries; if a future version adds more primaries, state and justify the multiplicity policy before outcome contact. No threshold or precision condition is supplied here.
6. Justify any exploratory construction quota against both primary score and any retained secondary support/false-alert estimands; preserve the Sprint 18 `48` contract. No old outcome or diagnostic row may tune the new quota.
7. Either leave candidate margins descriptive or prospectively justify each proposed soft predicate, its scientific consequence, dependence, and any allowable budget using a pre-outcome rationale/sensitivity plan. Explicitly preserve the hard `15–60%` cohort-mix predicate unless a separately authorized contract amendment is reviewed; the proposal's expanded mix band is not a default alternative.
8. Freeze full-roster report fields, history-level hard/metric states, precedence, no-retry/no-replacement handling, and any permitted status vocabulary before outcome contact. Decide whether and how `EXPLORATORY_USABLE` and `EXPLORATORY_USABLE_WITH_QUALIFICATIONS` may be defined and issued, including their semantics and limitations. Neither status is defined or issued by this draft or currently available, and neither is equivalent to or implies Gate V2 `MEASURABLE` or any other Gate V2 candidate verdict.
9. Before materialization or any data contact, obtain (a) a separate, explicit user decision authorizing the prospective path and (b) independent pre-outcome review with no actionable findings. These are distinct requirements; neither is granted by this draft. Any substantive change after that review requires a new prospective version and fresh review before materialization.

Until all prerequisites and approvals are satisfied, stop before generator/preflight/checker/model execution, role or seed binding, any new data/outcome contact, Confirmation/Sealed access, or Sprint 18 continuation. The draft supplies no authorization, no executable decision logic, and no historical verdict change.

## 9. Evidence basis

- [Sprint 19 plan and design-only boundary](sprint-plans/sprint-19.md).
- Reviewed [Task 1 estimands](exploratory-benchmark-estimands.md) and [Task 1 evidence](../artifacts/sprint-19/task-1.md); independent review `agent://S19Task1Review` reported PASS with zero actionable findings.
- Reviewed [Task 2 threshold rationale](exploratory-benchmark-threshold-rationale.md) and [Task 2 evidence](../artifacts/sprint-19/task-2.md); independent review `agent://S19Task2Review` reported PASS with zero actionable findings.
- [Exploratory soft-gate proposal](exploratory-benchmark-soft-gates.md), treated only as an illustrative proposal.
- [Normative Benchmark Measurability Exit Gates V2](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md), especially whole-history/independence and no-reselection boundaries, hard structural floors and downstream support, and EG0–EG7 definitions and precedence.
- [Sprint 18 Task 64 construction audit](../artifacts/sprint-18/task-64.md) and [Task 65 decision](../artifacts/sprint-18/task-65.md): 48 remains the active Sprint 18 reserve; neither its statistical necessity nor a safe lower replacement is established.
- [Sprint 15 frozen metric convention](../experiments/sprint15-observable-probe-v7.md) for event/control aggregation and history-block bootstrap, as cited by Task 1.

No generation, preflight, score, model, threshold, history/seed binding, Confirmation/Sealed access, or new outcome contact is authorized or performed by this draft.
