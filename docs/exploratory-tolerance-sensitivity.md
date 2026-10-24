# Exploratory Tolerance Sensitivity: Whole-History AUROC

**Status: DESIGN ONLY / NOT EXECUTABLE / NO DATA CONTACT.** This note addresses Sprint 20 Task 1 by bounding the mathematical influence of hypothetical support-margin misses on the reviewed whole-history `P+W` AUROC estimands. It makes no allowance recommendation, changes no hard gate or prior verdict, and authorizes no generator, preflight, history/seed or role binding, scoring, model training, Confirmation/Sealed access, or Sprint 18 continuation. The illustrative `(H,m)=(32,2)` is arithmetic only; it is not a roster, power calculation, precision claim, or probability of qualification. No prior outcomes or seed records are used or screened.

## 1. Estimand and what is being bounded

For a branch `s ∈ {S_pred, S_pop}` and complete Evaluation history `h`, let `E_h` be the eligible progressive-plus-weak-precursor (`P+W`) event-window scores and `C_h` the same-history eligible negative-control-window scores, using the common, score-independent support in the reviewed estimand. Each event/control-window score is the frozen maximum over its eligible files. With tie score `q(x,y) = 1` if `x>y`, `1/2` if `x=y`, and `0` if `x<y`, the history endpoint is

$$
A_{s,h} = \frac{1}{|E_h||C_h|}\sum_{e\in E_h}\sum_{c\in C_h}q(e,c), \qquad 0\le A_{s,h}\le 1.
$$

For a declared roster of `H` complete histories, each branch has its own equal-history macro:

$$
\Theta_s = \frac{1}{H}\sum_{h=1}^{H} A_{s,h}, \qquad s\in\{S_{pred},S_{pop}\}.
$$

The two endpoints use the same eligible event/control keys but remain separate primary estimates. No independence between their values across histories is assumed: both branches are evaluated on the same histories. Raw `S_pred` and `S_pop` scores are different score concepts and their scales are not comparable. A numerical difference between their normalized AUROCs is mathematically definable, but is not the same-branch arm contrast below and is not a primary arm effect or a basis for selecting a winning score.

For a prospective comparison of two arms using the **same score branch**, denote the per-history endpoints `A_{s,1,h}` and `A_{s,0,h}`. Only with the same predeclared histories and paired support in both arms define

$$
d_{s,h}=A_{s,1,h}-A_{s,0,h}\in[-1,1], \qquad
\Delta_s=\frac{1}{H}\sum_{h=1}^{H}d_{s,h}.
$$

This is a within-branch, paired arm difference; it is not a comparison of raw score scales. If either arm lacks a required endpoint or common support on a declared history, the full-roster paired contrast is not computable by silently dropping that history.

The factors `|E_h||C_h|` count AUROC comparisons, **not** independent observations: each event score is reused across controls and each control score across events. The independent accounting unit remains one complete whole history.

## 2. Sharp bounded-influence arithmetic for `H` histories and `m` affected histories

Let `M` be the set of distinct histories carrying at least one *margin-only* miss, let `m=|M|≤H`, and require for this calculation that each history still passes every hard predicate and has a computable endpoint on the frozen common support. A miss is a support classification, not an observed change to a score or AUROC.

### One branch's macro

The missed-stratum contribution to the full macro is

$$
K_s(M)=\frac{1}{H}\sum_{h\in M}A_{s,h}\in\left[0,\frac{m}{H}\right].
$$

Thus its full possible contribution range has width `m/H`. Replacing all `m` endpoint values by any other values in `[0,1]` can move `Θ_s` by at most `m/H`; changing one history can move it by at most `1/H`. If `m<H`, let `Θ_{s,-M}` be the mean over the remaining histories. Then

$$
\Theta_s-\Theta_{s,-M}=\frac{1}{H}\sum_{h\in M}(A_{s,h}-\Theta_{s,-M}),
\qquad |\Theta_s-\Theta_{s,-M}|\le\frac{m}{H}.
$$

This last bound describes the maximum reweighting difference between the full mean and the complement-only mean; it is **not** permission to use the complement-only mean as primary. It also does not say that a support miss changes an endpoint by `m/H`: with valid computed values, the miss does not itself change any `A_{s,h}`. Without event/control scores and support composition, counts alone cannot give a realized AUROC shift or a precision estimate.

### Same-branch paired arm difference

For the paired contrast, `d_{s,h}∈[-1,1]`, so the missed-stratum contribution satisfies

$$
K_{\Delta_s}(M)=\frac{1}{H}\sum_{h\in M}d_{s,h}\in\left[-\frac{m}{H},\frac{m}{H}\right].
$$

Its possible contribution range has width `2m/H`. Replacing the `m` paired differences can change `Δ_s` by at most `2m/H`. If `m<H`, the full-versus-complement-only contrast difference is likewise bounded by `2m/H`, because both the missed-stratum and complement means lie in `[-1,1]`. This is twice the worst-case bound for one `[0,1]` macro: a paired difference can vary across a two-unit range.

| Hypothetical case | One-branch macro | Paired same-branch arm contrast |
|---|---:|---:|
| General `H,m` | Missed contribution `[0,m/H]`; width and maximum endpoint-replacement shift `m/H` | Missed contribution `[-m/H,m/H]`; width and maximum contrast-replacement shift `2m/H` |
| `H=32`, `m=2` | Each history weight `1/32 = 0.03125`; two-history contribution `[0,2/32]`; maximum shift `2/32 = 0.0625` (6.25 percentage points) | Two-history contribution `[-2/32,2/32]`; maximum shift `4/32 = 0.125` (12.5 percentage points) |

In the illustrative case, a two-history within-branch macro's contribution can range from `0` to `0.0625`; the paired contrast's contribution can range from `-0.0625` to `+0.0625`. The latter's full range is `0.125`. These are sharp range bounds given only the metric bounds, not estimates of actual effects. If the two named predicate misses are on one history, use `m=1` for history-macro influence even though there are two predicate-by-history flags; the proposal's separate one-miss-per-history rule would nevertheless be violated. The number of correlated predicate flags is not the number of independent histories.

These formulae concern computable, hard-pass margin-miss histories. A hard-support failure, absent/non-finite score, branch-support mismatch, or otherwise uncomputable endpoint remains `STRUCTURAL-FAIL` / `UNCOMPUTABLE` under the reviewed full-roster contract. It cannot be relabeled a soft miss, assigned an invented AUROC, or dropped to calculate a primary macro or bootstrap. The bounds do not rehabilitate an invalid roster.

## 3. Conditional distribution-free uncertainty illustration

A finite declared-roster macro is an exact descriptive quantity once all its endpoints are known. A confidence statement about a wider population requires a sampling target and assumptions not established by selecting `H=32`. The following is only a distribution-free illustration, conditional on **all** of these assumptions:

- Whole Evaluation histories are sampled independently and identically from a named target population; each complete history, not its files, events, windows, robots, or pair terms, is one iid draw.
- The score, eligibility, common-support, arm-pairing, and aggregation rules are fixed before sampling/outcome contact. Every sampled history has a defined, computable endpoint under that target; histories are not selected, replaced, retried, or omitted based on support or scores.
- Each branch endpoint is bounded in `[0,1]`; for a paired arm contrast, the paired history-level `d_h` is bounded in `[-1,1]`. Dependence within a history and dependence between the two score branches are allowed; independence across the whole histories is required.

Under those assumptions, with population mean `μ_s = E[A_{s,h}]`, Hoeffding's inequality gives

$$
\Pr\left(|\Theta_s-\mu_s|\ge\varepsilon\right)\le 2\exp(-2H\varepsilon^2),
\qquad \varepsilon_{95}(H)=\sqrt{\frac{\log(40)}{2H}}.
$$

At `H=32`, the two-sided **marginal** 95% radius is `sqrt(log(40)/64) ≈ 0.2401` AUROC units (clip the interval to `[0,1]`). For an iid paired-history population mean contrast `δ_s=E[d_{s,h}]`, Hoeffding with range length 2 gives

$$
\Pr\left(|\Delta_s-\delta_s|\ge\varepsilon\right)\le 2\exp\left(-\frac{H\varepsilon^2}{2}\right),
\qquad \varepsilon_{95,\Delta}(H)=\sqrt{\frac{2\log(40)}{H}}.
$$

At `H=32`, that two-sided marginal radius is `sqrt(2 log(40)/32) ≈ 0.4802` contrast units (clip to `[-1,1]`). The contrast bound needs independent paired histories, **not** independent arms within a history. These are separate marginal statements, not simultaneous 95% coverage for both primary branches or a family of arms. The example is intentionally conservative and can be wide; it does not establish that 32 histories provide useful precision, that an iid target-population design exists, or that any history qualifies. If the roster is a fixed convenience set, histories are dependent, any hard failure prevents the endpoint from being defined, or the selection/eligibility mechanism differs, this population-mean illustration does not apply.

Retain the prior deterministic whole-history bootstrap convention as a **descriptive resampling summary**: resample `H` complete histories with replacement, use `B=2000` replicates and RNG seed `20260202`, and report the 2.5th percentile labeled `LCB95`; use the same sampled history indices for the two score branches. For a separately predeclared paired-arm contrast, preserve pairing by resampling whole histories and recomputing `d_h` within each draw. This convention neither supplies a distribution-free coverage guarantee nor proves precision at `H=32`; it does not permit window-level resampling or dropping a miss stratum.

## 4. Proposed margin-by-margin implications

The table evaluates the candidate Design-style margins in the prior analysis and the proposal's expanded mix band. A *margin-only* shortfall means the relevant inherited hard floor and every other hard condition still pass. A Design-style margin is not thereby made optional under Gate V2, nor is a soft allowance approved here.

| Candidate predicate | Hard floor → illustrative Design margin | Margin-only range | Primary `P+W` AUROC implication | Secondary implication and correlated failures |
|---|---:|---|---|---|
| Progressive (`P`) events | `10 → 13` | `10–12` | Direct support for the event side of both branch AUROCs; lower `P` support can change the within-history `P:W` composition. This is a support/possible-stability concern, not a known score shift. | Directly reduces support for `P`-only AUROC. May co-occur with the total-positive, mix, robot/program concentration, or other causal/eligibility flags. |
| Weak-precursor (`W`) events | `10 → 13` | `10–12` | Direct support for the event side of both branch AUROCs and the `P:W` composition. | Directly reduces support for `W`-only AUROC; may co-occur with total-positive, mix, concentration, or other support deficits. |
| Total positive events | `30 → 38` | `30–37` | Overall event support; direct primary support is reduced when the deficit includes `P` or `W`. If a total-count shortfall is attributable only to `A` while `P/W` support is unchanged, it does not by itself reduce the primary pair count. | Affects overall event support and the all-positive mix denominator; overlaps with P/W/A counts rather than constituting an independent check (if cohorts partition positives, the total is their sum). |
| Negative-control windows | `25 → 32` | `25–31` | Direct support for the control side of both branch AUROCs. The pair count `|E_h||C_h|` is not an independent sample size because scores are reused within history. | Can co-vary with robot/program concentration, robot-days, event/control eligibility, and the operative construction support. `48` remains the Sprint 18 construction reserve; this analysis does not reduce it or set a new exploratory quota. |
| Abrupt (`A`) events | `8 → 10` | `8–9` | Not part of the primary `P+W` event numerator and not a primary predictive requirement. It has no direct pair-count effect if P/W and controls are held fixed; its contribution to total positives and mix can indirectly change the support composition. | Directly affects the secondary `A` endpoint and the all-positive cohort-mix denominator. Can co-fail with total-positive and mix conditions. |
| Evaluable robot-days | `150 → 188` | `150–187` | Does not enter the threshold-free AUROC formula directly. It is not evidence that more/fewer histories or windows are independent. | Directly affects exposure available for any separately frozen false-alerts-per-robot-day diagnostic and therefore the scope of that secondary interpretation; may co-vary with control/event support and robot concentration. |
| Cohort mix (`P`, `W`, `A`) | Inherited hard range `15–60%`; proposal example `12.5–62.5%` | No valid soft-only region under the unchanged hard contract | The proposal's newly allowed `12.5–<15%` and `>60–62.5%` regions violate the inherited hard per-history rule and cannot be accepted as margin-only misses here. Mix shifts P/W composition within the primary event set; A changes the all-positive denominator. | P/W/A mix and their event counts are coupled; count each affected cohort as its own proposed predicate miss, but do not treat the flags as independent. A hard mix failure remains hard. |

All named hard floors remain mandatory. Robot/program contribution limits, minimum contributing robots and programs, chronology, event/lead eligibility, censoring/reset/maintenance/quarantine rules, causal validity, identity and whole-history role isolation, common support, finite scores, and other Gate V2 hard predicates are not soft candidates. A deficit below a hard floor is not a Design-margin miss. In particular, a mix excursion beyond the existing 15–60% rule, or a failed causality/identity/role predicate, cannot be rescued by any miss budget.

The original proposal counts each predicate-by-history miss separately and caps the example at two total misses on two distinct histories, with no more than one on either. This arithmetic limits only how many history terms are exposed to the worst-case bound; it does not show comparable scientific impact across predicates, independence among failures, or safety of any budget. `P`, `W`, total positives, controls, mix, A, and robot-days can co-vary within a history. In particular, an A-only or robot-day-only miss is not interchangeable with a P/W/control miss for primary support. A single underlying support pattern may produce multiple flags, and a missed margin by itself changes no computed score.

## 5. Full-roster and support-stratum sensitivity reporting

For any future, separately authorized measurement, define the support flags from frozen score-independent predicates before examining endpoint values. Report at least:

1. **Full roster first:** all `H` history IDs/roles as permitted by the future contract, every hard predicate and candidate-margin flag, observed support counts/gaps, `A_{S_pred,h}` and `A_{S_pop,h}` separately, and both complete-roster macros. If any hard predicate fails or either endpoint is uncomputable, label the paired primary roster result `UNCOMPUTABLE`; do not calculate a nominal primary/bootstrap on only computable or better-supported histories.
2. **Margin strata, secondary only:** for each branch, display counts and per-history endpoint distributions for (a) histories meeting all listed descriptive margins, (b) each named margin-miss stratum, and (c) the predeclared union of histories with any margin miss. Include `P`-only, `W`-only, and `A` secondary metrics and the relevant support for their strata. Keep empty/noncomputable strata explicit. Because flags may overlap, show both each predicate and the union rather than summing predicate counts as if they were distinct histories.
3. **Paired arms, if prospectively declared:** report every per-history same-branch `d_h` and the full-roster `Δ`; show the same support strata with the same paired history membership. A missing paired endpoint makes the full-roster contrast uncomputable, not eligible for a complete-case comparison.
4. **Dependence and bounded influence:** accompany these displays with `m/H` (one-branch macro) and `2m/H` (paired contrast) for the distinct flagged-history set, state which predicate(s) define that set, and describe any strata/contrast difference as descriptive. Do not weight, delete, replace, or select histories by score to make a favorable primary result. The margin-met-only mean is never the primary estimand.

For hard failures/uncomputable values, show the failed state and reason, not a zero, fabricated score, or favorable-subset endpoint. Support-stratum summaries cannot establish that missing support caused a score change: they compare different observed histories and may reflect correlated support, event mix, simulator composition, or other history-level factors.

## 6. Falsifiable threats to interpretation

A small numerical shortfall is a count fact, not automatically harmless or harmful. A future protocol would need to state a materiality rule tied to its intended exploratory interpretation **before** outcomes. Under such a rule, the following observable findings could falsify a claim that margin-only deficits are immaterial:

- For either branch, the prespecified P-, W-, or control-margin-miss stratum has a materially different `P+W` history-AUROC distribution from the margin-met stratum, or the full-roster interpretation crosses a prospectively named decision boundary while the margin-met-only diagnostic does not. This must be reported as support-stratum dependence, not repaired by substituting the more favorable stratum.
- For a prespecified same-branch arm contrast, missed-support histories materially change the contrast under the bounded decomposition, or the full-roster contrast reverses sign relative to a margin-met-only diagnostic. A sign/threshold disagreement is a sensitivity warning, not authorization to prefer either subset.
- A P/W support deficit coincides with an unstable or substantively different P-only versus W-only secondary pattern, showing that a favorable pooled `P+W` value may conceal cohort-specific weakness. An A or robot-day deficit cannot directly alter the primary AUROC formula, but can invalidate a broader secondary interpretation if the A endpoint or exposure-normalized false-alert diagnostic is missing or materially different.
- Margin misses repeatedly coincide with other flags (including total-positive, mix, controls, robot/program concentration, or eligibility), so the apparent “one small miss” stratum is not a separable single-predicate condition. Any failure of a hard support, mix, causality, identity, or role-isolation predicate is already a hard stop, not a falsification that a soft allowance might cure.

Conversely, similar observed stratum summaries would not prove equivalence, establish future-history representativeness, or supply a probability of qualification. The protocol has no justified numerical materiality threshold or empirical score-variance model today; this note invents neither. Falsifiability requires the later protocol to name the interpretation and margin criterion prospectively and retain the full roster.

## Evidence basis and stop boundary

- [Reviewed whole-history estimands and full-roster rules](exploratory-benchmark-estimands.md), especially the separate score branches, tie-aware AUROC, whole-history macro/bootstrap convention, common support, and `STRUCTURAL-FAIL` / `UNCOMPUTABLE` handling.
- [Reviewed margin and threshold rationale](exploratory-benchmark-threshold-rationale.md), including the hard floors, candidate Design margins, mix-band conflict, uncertainty boundary, and `48` versus `32` versus `25` construction distinction.
- [Reviewed non-executable protocol draft](exploratory-benchmark-protocol-draft-v1.md) and [historical illustrative proposal](exploratory-benchmark-soft-gates.md); neither binds a role, roster, miss allowance, or execution.
- [Sprint 20 Task 1 assignment](sprint-plans/sprint-20.md#L13) and [unchanged Sprint 19 review record](sprint-plans/sprint-19.md); Task 2 alone owns a later prospective tolerance recommendation.
- [Gate V2](BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md) remains normative. Sprint 18's 48-control construction reserve, blocked tasks, historical verdicts, and stop before Cycle 4 remain untouched.

This document supplies analytic bounds and reporting requirements only. It records no data, score, seed screen, empirical variance, or evidence of precision; makes no Gate V2 `MEASURABLE` or exploratory-use decision; and grants no authority for data contact, generator/preflight execution, roles/seeds, scoring, training, Confirmation/Sealed access, or Sprint 18 continuation.
