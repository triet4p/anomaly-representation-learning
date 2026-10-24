# Exploratory Benchmark Estimands and Evidence Surface

**Status: DESIGN ONLY / NOT EXECUTABLE.** Sprint 19 Task 1 specifies the question and measurement contract only. It does not bind roles, histories, seeds, roots, code, checkpoints, or thresholds; authorize generation, preflight, training, scoring, Confirmation/Sealed access, or any new outcome contact; change Sprint 18; or approve any proposed soft allowance. Sprint 18 remains blocked before Cycle 4 under its existing contracts.

## Decision in brief

**Question.** Conditional on a prospectively declared synthetic simulator and its eligible support, can a representation fit using verified-healthy Fit data alone produce either of its two independent anomaly scores—`S_pred` (context/prediction mismatch) or `S_pop` (population mismatch to a healthy reference)—that rank eligible progressive (`P`) and weak-precursor (`W`) failure-event windows above eligible same-history negative-control windows on untouched whole Evaluation histories?

The two primary endpoints are separate, co-primary descriptive estimates: the history-macro event AUROC for `S_pred` and the same endpoint for `S_pop`. Neither score is fused, neither raw scale is compared to the other's, and no score selects histories or eligibility. `P+W` is primary because these are the predictable cohorts in the inherited benchmark contract; abrupt (`A`) is a secondary companion with no primary predictive requirement.

This is an exploratory, simulator-conditional representation question. A favorable endpoint would support only a hypothesis about ranking under the declared synthetic mechanism and observed support. It would not establish real-world prevalence, calibrated deployment risk, fleet generalization, or independently confirmed early-warning performance. This claim boundary matches the [exploratory proposal, §Question](exploratory-benchmark-soft-gates.md#L5-L17) and the [case-control interpretation in Gate V2](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L81-L105).

## Primary measurement definitions

### Score branches and support

Keep the branches independent throughout: `S_pred` is context/prediction mismatch; `S_pop` is population mismatch against a verified-healthy reference bank. These are separate score concepts in the [project score contract](CONCEPT.md#L427-L470). Each branch must have a prospectively frozen scoring implementation and provenance, including any Fit-fitted transformation and, for `S_pop`, the bank row count and source. Score scale or threshold values are not compared across branches.

For each predeclared Evaluation history `h`, construct one score-independent eligible set under the frozen causal event, temporal-view, censoring, maintenance/reset, quarantine, and control-window predicates. It contains eligible `P` and `W` event windows and same-history/robot negative-control windows. Eligibility and the selected support keys are fixed without inspecting model scores. Both branches are scored on exactly those same event and control keys; no branch-specific support or threshold-dependent filtering is allowed.

### Event score and tie-aware event AUROC

For branch `s ∈ {S_pred, S_pop}`, use the existing benchmark's event aggregation: an event-window score is the maximum file score over its POS-eligible window files; a control-window score is the maximum file score over its control-window members. Within each history, compare eligible `P+W` event-window scores against eligible control-window scores with tie-aware AUROC. This reuses the frozen [Sprint 15 event-metric convention](../experiments/sprint15-observable-probe-v7.md#L56-L69), which defines event/control maxima, the P+W-versus-control event AUROC, and an unweighted whole-history macro; Gate V2 remains normative where a conflict exists.

Let `E_h` be all eligible `P` and `W` event-window scores in history `h`, and `C_h` its eligible negative-control-window scores for one branch. The per-history primary value is

$$
A_{s,h}^{P+W} = \frac{1}{|E_h||C_h|} \sum_{e\in E_h}\sum_{c\in C_h} \left[\mathbf{1}(e>c) + \tfrac{1}{2}\mathbf{1}(e=c)\right].
$$

Thus a tied positive/control pair contributes one half; constant scores yield `0.5`, as required by the [Gate V2 metric fixture contract](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L241-L250). Each declared Evaluation history has equal weight regardless of its number of events, control windows, or files. For the fixed roster of `H` histories, the primary endpoint for each score is

$$
\Theta_s^{P+W} = \frac{1}{H}\sum_{h=1}^{H} A_{s,h}^{P+W}, \qquad s\in\{S_{pred},S_{pop}\}.
$$

These are two estimates of the same synthetic ranking estimand, not a fused endpoint, a selected winner, or an intervention effect versus a baseline. No pooled event-level AUROC across histories, file-level AUROC, or window-as-independent uncertainty is substituted. The per-history event score and tie handling are grounded in the frozen metric contract above; exact branch-specific score construction and any patch-to-file aggregation remain to be prospectively specified.

### Primary versus diagnostic reporting

**Primary:** report `Θ_S_pred^(P+W)` and `Θ_S_pop^(P+W)` separately, their complete per-history values, and the history-block uncertainty below. Report both only on the identical, full declared history roster and support.

**Secondary diagnostics, never replacements for the primary endpoint:**

- per-history and macro `P` and `W` event AUROCs, to show cohort-specific behavior;
- `A` event AUROC and support separately, with no minimum predictive requirement and no silent exclusion;
- score-independent support counts and every hard-support or computability miss by history;
- if separately frozen with a Calibration-only threshold, `P`/`W` event recall, first-alert lead, and false-alert episodes per evaluated robot-day; warning persistence, severity, localization, and background stability may be reported only with their own frozen definitions.

Gate V2's observable-probe cutoffs (including its AUROC, recall/lead, and false-alert thresholds) are acceptance criteria for that fixed probe and role contract, not automatic pass criteria for these exploratory representation endpoints. This Task 1 design sets no ranking cutoff, effect-size gate, favorable-history count, or multiplicity-adjusted decision rule. The inherited metrics and their scope are documented in [Gate V2 EG4](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L252-L273).

## Whole-history uncertainty

The independent resampling unit is one complete, disjoint Evaluation history—not a file, event, control window, repeated profile on one seed, or metric row. This preserves the [Gate V2 whole-history/no-favorable-seed boundary](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L81-L102).

For each endpoint, apply the existing deterministic history-block bootstrap convention: resample `H` whole histories with replacement, compute the unweighted macro for each replicate, use `B=2000` replicates and bootstrap RNG seed `20260202`, and report the percentile labeled `LCB95` at 2.5% as in the [frozen probe metric contract](../experiments/sprint15-observable-probe-v7.md#L63-L69) and [Gate V2 EG4](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L259-L270). Use identical resampled history indices for the two score branches so their uncertainty summaries remain paired by history. This bootstrap seed is a metric-procedure parameter only; it does not bind or contact any data history or role seed.

The history-block bound is an uncertainty summary, not by itself a generalization guarantee or a claim of known pass probability. Its interpretation depends on a prospective history-sampling regime, independent whole-history roles, and adequate support; those are not established by choosing `H` or by resampling windows. No sample-size or precision claim follows from this design.

## Common-history support and incomputability

The paired primary comparison uses the complete predeclared Evaluation roster. For every history, both branches must have finite scores for every eligible P/W event window and every common eligible control window, with identical support keys. Structural support predicates are evaluated independently of scores. Apply inherited per-history hard floors, chronology, isolation, concentration, and eligibility rules; do not replace them with pooled totals. Gate V2's relevant inherited structural floors include `P≥10`, `W≥10`, `A≥8`, total positives `≥30`, negative controls `≥25`, robot-days `≥150`, six positive- and six negative-contributing robots, two programs in each P/W cohort, and its stated mix, concentration, and lead-support constraints ([Gate V2 §§Hard structural floors and downstream support](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L146-L182)). These are support conditions, not statistical precision claims.

If a declared history fails an inherited hard support predicate, record `STRUCTURAL-FAIL` for that history; its per-history primary endpoint is `UNCOMPUTABLE` because the required support is not met. Also record `UNCOMPUTABLE` if a required window/score is missing or non-finite, the score branches do not share the same support keys, or either branch's per-history `P+W` AUROC cannot be computed. Do not assign a numeric substitute, quietly drop that history, or publish a macro/bootstrap over only computable or better-supported histories. The paired primary result for the full roster is `UNCOMPUTABLE`; any finite branch-only/per-history values may appear only as clearly labeled secondary diagnostics. Retain the full roster and all misses in the report. This extends the proposal's no-pruning rule into explicit endpoint behavior ([proposal, all-history reporting](exploratory-benchmark-soft-gates.md#L30-L42)).

A history that meets hard floors but misses a proposed Design-style margin remains in the endpoint accounting if its frozen primary metrics are computable; whether such a miss affects a later exploratory disposition belongs to the separately reviewed soft-allowance decision, not this estimand. A structural `FAIL` and a metric `UNCOMPUTABLE` are distinct recorded states; neither is converted into an AUROC of zero.

## Fit, Calibration, and Evaluation permissions

Roles must be mutually exclusive at the whole-history/root/seed level, and all support used by one role must stay within that role. This is a permissions contract, not a binding of roster size, IDs, or seeds.

| Role | Permitted use in this design | Prohibited use |
|---|---|---|
| **Fit** | Verified-healthy rows only to fit the representation, Fit-derived preprocessing/standardization, and the frozen healthy reference bank required by `S_pop`. | Failure/anomaly labels, event outcomes, Calibration or Evaluation rows, or any later role. No score-driven selection of Fit histories. |
| **Calibration** | Disjoint verified-healthy rows only to verify the declared healthy support and, if the later protocol needs thresholded diagnostic metrics, select the one prospectively specified operating threshold. | Representation, preprocessing, or reference-bank fitting; Evaluation; choosing models, histories, score formulas, or gates. Calibration does not affect the threshold-free primary AUROC. |
| **Exploratory Evaluation** | The complete, predeclared, disjoint whole-history roster, once, for the two frozen primary score endpoints and predeclared diagnostics. | Fitting/refitting representation, preprocessing or bank; selecting thresholds for the primary endpoint; pruning histories, generator tuning, or later Confirmation reuse. |

The inherited Gate V2 downstream support references are Fit `≥200` verified-healthy files and `≥6,000` valid patches in aggregate, with fallbacks/sparse conditioning groups reported, and Calibration `≥40` verified-healthy eligible rows; it also bars confirmation, sealed, quarantined, censored, and maintenance-overlapping rows from Fit/Calibration ([Gate V2](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md#L176-L182)). Those values are recorded precedents for the observable-probe contract, not proof that they adequately size a learned representation or `S_pop` bank. Any representation-specific support minimum must be justified and frozen before outcome contact. If a thresholded diagnostic is included, its threshold rule and eligible healthy Calibration rows must also be frozen in that future protocol.

Exploratory Evaluation is never an untouched Confirmation set. Any later confirmatory question requires a separately frozen, disjoint Confirmation roster and an independently approved contract; Confirmation and Sealed are not opened or used by this task.

## Reconciliation with the exploratory proposal

This design uses the proposal's synthetic healthy-only question, separate Fit/Calibration permissions, whole-history accounting, common support across score branches, full-roster reporting, and prohibition on favorable-history pruning. It narrows the primary evidence surface to one event-ranking endpoint for each independent score; all other metrics are secondary diagnostics.

The proposal's `32` histories, allowance of at most two soft predicate×history misses, and `12.5–62.5%` mix band remain explicitly illustrative and unapproved. This document does not adopt them as sample size, operating rule, or construction quota. Likewise, `48` selected controls remains the operative Sprint 18 construction reserve; `32` is the existing Design promotion target and `25` the hard floor, not evidence that a lower construction quota is safe. The reviewed Sprint 18 record states that neither statistical necessity of 48 nor safety of a reduction to 32/25 is established ([Task 64](../artifacts/sprint-18/task-64.md#L9-L11), [Task 65](../artifacts/sprint-18/task-65.md#L9-L21)). Sprint 18 verdicts and blockers are unchanged.

## Decisions still required before any executable protocol

This estimand resolves what the two primary endpoints measure, their whole-history aggregation, the uncertainty unit, score-branch support parity, role permissions, and the handling of uncomputability. It intentionally does not fill unsupported execution parameters. A later, separately authorized and reviewed protocol must still decide and freeze at least:

1. the simulator/generator and causal eligibility/window contract, reset/censor/quarantine semantics, source digests, and observable inputs;
2. exact role counts and a justified evaluation-history sample size `H`, then disjoint history identities, seeds, roots, provenance, and the no-retry/failure registry;
3. the representation configuration and exact `S_pred` and `S_pop` scoring implementations, preprocessing, patch-to-file mapping if applicable, reference-bank construction/provenance, and metric-code identity;
4. the construction quota and its justification, without treating the proposal's 32-history/miss/mix examples as approved or weakening Sprint 18's 48-control contract;
5. whether and how to adopt the inherited bootstrap settings for the new roster, and the adequacy of its sampling/independence assumptions;
6. representation-specific Fit and Calibration support requirements, plus a threshold rule only if thresholded diagnostics are retained;
7. the full hard/soft predicate registry, any miss allowance and precedence, numerical decision thresholds if any, multiplicity policy if a later design introduces additional primaries, and the exact dispositions for structural failures versus metric uncomputability;
8. whether any later confirmatory claim is desired and, if so, its separate roles, untouched roster, endpoints, and approval boundary.

Until those choices are resolved prospectively, the definitions here are a reviewable estimand design only—not an executable protocol, data-contact authorization, Sprint 18 reclassification, or evidence of measured precision.