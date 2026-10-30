# Sprint 24 Task 7-M15 — Paired-source representation and weekly/monthly movement experiment: design

- Task / attempt / agent: `S24-7-M15` / `S24-7-M15-A01` / `S24M15GoldA01` (R2 corrections for C1/R1 findings: exact Clopper–Pearson fix, independent binomial-tail verifier, typo cleanup).
- Criterion: `docs/sprint-plans/sprint-24.md:250-257`.
- No quantitative success gate is assigned; honest design-feasibility acceptance governs. Numerical business thresholds are proposals pending explicit owner approval, never execution gates.
- What this design does NOT do: no official paired-corpus generation, no embedding capture, no model loads, no forward passes, no training, no bank refit, no remote execution, no scale, no Phase C release, no edits to accepted M11–M14 products. Cumulative observed forwards stay **77,653**.
- This report preserves M11–M14 conclusions and changes none of their bytes.

## 1. Question and decision structure

Distinguish abnormality-associated embedding movement under controlled source conditions from normal variability (Stage 1, paired sources), then determine whether it survives dilution into weekly/monthly populations (Stage 2, independent population arms).

Predeclared decision branches (bounded; no guaranteed positive or universally definitive result):

- `SCALE`: Stage-1 bar met and Stage-2 primary met with clean context controls — proceed to scale.
- `MEASURE-DILUTION`: Stage-1 bar met but Stage-2 primary missed — investigate measurement/sample-size dilution.
- `REPRESENTATION`: Stage-1 bar missed at the N=96 cap with clean context — bounded negative, change representation/objective.
- `CONTEXT`: context-only controls move — resolve confounding before any scale claim.

There is no file-accuracy replacement objective. A non-significant result is never read as absence: only the `REPRESENTATION` cell at the N=96 cap with clean context supports a bounded negative; every other non-significant cell is inconclusive.

## 2. Why a paired design (rationale from M14)

M14 measured, on the exposed cache, abnormal-versus-normal contrasts at or below the matched normal–normal band at supported sizes (branch A: 0 of 72 aggregate E1 medians above the null q95; branch B: `within_prototype_only` 0 in every cell). The limiting factor is identified as support, not metric choice: 1–10 abnormal files per `robot|program` × period stratum forced per-arm sizes of week `{2, 3}` and month `{3, 4}`. Within-stratum abnormal support cannot be grown by re-analysis; it needs a corpus where abnormal and normal variants share their source. M14 branch C orders the estimators by responsiveness to a known embedding shift (`E1` from 0.25–0.5 σᵤ, `D_LOC` from 0.5–1.0 σᵤ, `E2_K8` only at 1.0–2.0 σᵤ), which fixes the estimator hierarchy below: `E1` primary, `D_LOC` sensitivity, `E2_K8` primary occupancy.

## 3. Paired construction

Each pair starts from one pair seed. From it derive one unmodified twin and a bounded set of anomaly-injected variants (family × severity). The twin and every variant of one seed share the identical pre-noise base (causal signal, regime layout, roster, noise-seed position); only the injection and the signal-dependent realized noise differ (see §9 for the measured coupling limit).

- Source identities: new namespace `S24P-ITER-0001`, six histories `S24P-ITER-0001-PAIRED-01…06` (primary triple 01–03, replication triple 04–06). Proposed history seeds `53101–53106` (collision-checked against the certified old block `32064–32079` and the executed M8 block `42401–42406`; pair seeds derive from the `s24p-paired-v1` domain tag, never from a v2 slot stream).
- Fixed within pair: the shared base above; robot/program, time/source context, length, regime and noise-seed position by construction. Severity/family sweeps at fixed seed share the exact same base (verified: same T, regime bounds, roster across severities and across non-duration families at seed 42401).
- Varied within pair: requested family (all nine hard families in scope), requested severity (ladder 0.35/0.60/0.85 inside the planned [0.35, 0.85] band), and the realized injection outcome.
- Unavoidable downstream changes (disclosed, verified): `duration_anomaly` changes T/regimes, so it pairs within family only (owner decision D6 rules on scope); post-injection noise is signal-dependent (S1 local-variance scaling, S2 local-std-scaled walk), so twin and abnormal differ outside the mask too — the pairing controls the pre-noise base and the noise seed, not the realized noise.
- Controls: (a) independent normal–normal pairs from disjoint pair seeds (own null band); (b) low-burden business baselines with few faults in the [0.02, 0.10] band — never zero-required; pure-normal arms are diagnostic controls only. Business baselines may have few faults, not necessarily zero; pure normal is a diagnostic control.
- Separation rules: construction labels select variants only; they never enter encoder, reference, prototype, bandwidth or detector fits. Grouping key = pair seed: every variant (twin + all family/severity abnormals) of one seed stays in one split; reference-pool seeds are disjoint from pair/control seeds. Correlated variants and resampled repeats are never counted as independent evidence.

## 4. Two-stage evaluation

### Stage 1 — source-matched representation diagnostic

Unit: one pair seed (twin vs one abnormal variant). Score each pair with the frozen encoders on `E1` (primary), `D_LOC` (sensitivity), `E2_K8` (occupancy), `K ∈ {4, 8}` sensitivity as descriptive. Null calibration: the independent normal–normal pairs give the frozen q95 per estimator (same sizes, same encoders); a pair "responds" when its contrast exceeds it. Primary summary per family × severity cell: detection fraction over N independent pair seeds (N from §6) with an exact Clopper–Pearson interval. The `E1 ≥ 0.5` bar is PROPOSED (owner decision D4); the exact lower bound must clear the measured false-alarm anchor.

### Stage 2 — weekly/monthly population evaluation

Independent population arms at the inherited M14-supported sizes (week `{40, 60}`, month `{160, 250}`) with the realistic burden ladder LOW 0.05 / MID 0.15 / ELEVATED 0.275 (exact counts: week-40 → 2/6/11 abnormal; week-60 → 3/9/17; month-160 → 8/24/44; month-250 → 13/38/69; M13 rounding). Context-only controls (same roster/time mix, low burden on both arms) run beside every transition. Freeze encoders and reference fitting: one causal reference pool per new history (220 files, `end_time <= c1`), bandwidth/prototypes/standardization fitted once on it, never re-fitted per pair/period/variant or across histories; no fit touches evaluation periods, labels, families, severities or context fields. Reference seeds are disjoint from pair/control seeds; leakage checks (file-ID disjointness, reference-before-evaluation-period ordering, label-blind fit audit) are part of the execution freeze, not this design.

### Coverage (no cherry-picking)

All nine families × three severities × planned N enter Stage 1; every burden level × inherited size × horizon enters Stage 2; every planned seed counts or carries a closed abstention (`ABSTAIN-BOUNDARY`, `ABSTAIN-COVERAGE`, `ABSTAIN-REFERENCE-SHORTFALL`, `ABSTAIN-STRENGTH-GATE`). Unsupported conditions list: `duration_anomaly` cross-family pairs (T/regime change); strengths outside [0.35, 0.85]; missing-event edge-only sessions that return NORMAL on the anomaly path (counted as construction abstentions, never as normal evidence).

## 5. Estimands and endpoints

- Stage-1 primary estimand: per family × severity detection fraction (share of independent pairs above the frozen null q95). Endpoint hierarchy from M14: `E1` primary; `D_LOC` sensitivity; `E2_K8` occupancy.
- Stage-2 primary estimand: `E1` transition median minus the context-only control q95 at elevated burden, weekly and monthly; the dilution ladder must show monotone non-increasing response (descriptive).
- Null calibration: Stage-1 normal–normal pairs; Stage-2 context-only controls plus retained reference split-half bands (inherited N1/N2 roles).
- Uncertainty: paired sign test + one-sample paired-contrast test (Stage 1); history-cluster bootstrap `B = 2000`, seed 2407, descriptive at small n (Stage 2, no pair-level bootstrap — pairs inside a history are dependent).

## 6. Sample sizes and operating characteristics

Calculator: `src/representation/sprint24_paired_decision_experiment.py` (exact binomial sign-test power, Acklam normal-approximation t power, exact Clopper–Pearson intervals, exact dilution counts) with an independent reimplementation check (`_independent_calculator_check`, no shared math imports) that recomputes all 9 sign entries and the dilution ladder. Assumptions travel with every number: `p_win`/`delta` are ASSUMED sensitivity grids, not measured anomaly effects; M14 anchors calibrate responsiveness/null bands only, never power; pairs are independent by construction (grouped seeds). No new model forwards were used.

Measured anchors from M14 evidence (responsiveness only): branch-C `E1` zero-shift median detection fraction ≈ 0.06 (false-alarm anchor); `E1` response grid month-160 ≈ 0.13/0.55/1.0 and week-40 ≈ 0.07/0.17/0.54 at δ = 0.10/0.25/0.50.

Exact sign-test power (two-sided, α = 0.05 nominal; actual α ≈ 0.023–0.032):

| N pairs | p_win = 0.65 | p_win = 0.75 | p_win = 0.85 |
|---|---|---|---|
| 24 | 0.2106 | 0.6074 | 0.9428 |
| 48 | 0.4699 | 0.9296 | 0.9996 |
| 96 | 0.7989 | 0.9988 | 1.0000 |

One-sample paired-contrast power approximation (one-sided α = 0.05, ASSUMED standardized δ):

Finite grid decision (predeclared, no silent expansion): N = 24 primary, N = 48 planned; an interim futility look at N = 24 stops the family for futility when its detection fraction sits at or below the false-alarm anchor (PROPOSED cap 0.10); N = 96 is licensed ONLY inside the gray zone 0.30–0.50, and both the N = 48 and N = 96 results are reported. Bar spot check: 24/48 gives an exact interval [0.3523, 0.6477] — the lower bound clears the 0.06 anchor; 5/48 gives [0.0347, 0.2266] and stops.
Why this grid is honest: at N = 24 only large assumed win rates resolve; at N = 48 the mid-grid resolves while 0.65 stays weak; the N = 96 cap is what makes a bounded `REPRESENTATION` negative supportable (exact sign power 0.80 at assumed 0.65, ≈ 1.0 at 0.75+). No guaranteed power and no universally definitive result are claimed.

## 7. Stop/expand rules

- Futility: at the N = 24 interim look, a family at or below the false-alarm anchor stops (PROPOSED cap 0.10; owner D4). No optional stopping beyond this rule; no outcome-based metric selection.
- Gray zone: a family in [0.30, 0.50) at N = 48 may expand to N = 96 once; both results publish.
- Success bars are PROPOSED (Stage-1 `E1 ≥ 0.5` with the exact lower bound above the anchor; Stage-2 transition median above the context-only q95 at elevated burden). Paired-source uncertainty (exact intervals above) is reported separately from population uncertainty (history-cluster bootstrap).
- Distinguishability: meeting Stage 1 but missing Stage 2 routes to `MEASURE-DILUTION`, not to a representation verdict; missing Stage 1 at the cap with clean context routes to the bounded `REPRESENTATION` negative.

## 8. Decision thresholds are proposals, not gates

D4 owns every number with a threshold role (`E1 ≥ 0.5`, futility cap 0.10, gray zone, burden ladder). Until the owner approves, they are design inputs to the execution freeze, not pass/fail gates. The full M15 execution criterion accepts scientifically honest demonstrated API gaps with precise evidence — this design meets it either by feasible paired invariants (§9) or by the proved gap (GAP-PAIRED-TWIN-01).

## 9. Feasibility: what the public API can and cannot do

Verified by the tiny public-path smoke (`--mode smoke`, `SMOKE-COMPLETE`, 0 forwards, temp outputs only, deterministic replay byte-identical with waveform sha `5d5358a4…`/`2a897ff0…`):

- Feasible without new implementation: deterministic replay of every public path (byte-identical abnormal and normal); same-seed severity sweeps share the exact base (T = 450, identical regime bounds and roster at 0.35 vs 0.85); same-seed family sweeps share the exact base across non-duration families; label-free encoder inputs; realized-mask annotations; temp-only waveform writes.
- PROVED GAP `GAP-PAIRED-TWIN-01`: no public API returns the unmodified twin of an anomaly-path draw. `generate_normal(seed)` and `generate_anomaly(seed)` at equal seeds are independent base draws — measured at seed 42401: normal T = 386 roster (4, 2) vs anomaly T = 450 roster (3, 1). Reconstructing the twin replays non-public internals (normal-seed derivation, regime/envelope/causal chain, anom-seed and noise-seed positions, strength-gate loop), so the twin constructor is NEW implementation owned by the execution task. It is documented here with reproducible numbers, never silently approximated.
- Coupling limit `INV-NOISE-COUPLING-01`: post-injection noise is signal-dependent, so twin and abnormal differ outside the mask too; the pairing controls the pre-noise base and the noise seed, and realized noise differences are part of the contrast.
- Additional measured boundaries: strength-gate retries exist (e.g. seed 42447 needed 2 attempts); 8 of 300 scanned `missing_event` seeds return NORMAL on the anomaly path (construction abstentions, never normal evidence).

## 10. Owner decisions required (explicit)

- D1: approve the paired-twin constructor interface for the execution task (gap GAP-PAIRED-TWIN-01).
- D2: approve Stage-2 business burden levels (proposed LOW 0.05 / MID 0.15 / ELEVATED 0.275; baselines stay low-nonzero).
- D3: approve the Stage-1 seed grid (24 primary, 48 planned, 96 gray-zone cap) and the futility rule.
- D4: approve the proposed numerical bars as release criteria (E1 ≥ 0.5, futility cap, gray zone).
- D5: release official paired-corpus generation and frozen-encoder execution (forward budget).
- D6: rule on `duration_anomaly` scope (within-family only, or exclude).

## 11. Execution boundary (ordered, for the next task)

This design authorizes: tiny public-path smoke (temp outputs only); the deterministic calculator; the proposed contract. It forbids: official paired-corpus generation; embedding capture, model loads, forwards; training, bank refit, remote, scale, Phase C; edits to M11–M14 products. The execution task owns: the twin constructor (GAP-PAIRED-TWIN-01); the official generation freeze and execution; frozen-encoder capture and the two-stage evaluation.

## 12. Limitations and non-claims

- Design only; no paired corpus, embedding, forward, training, refit, remote or scale claim follows.
- `p_win`/`delta` grids are sensitivity assumptions, never anomaly power proof; M14 perturbations measure responsiveness only.
- The twin exists as a verified reconstruction recipe, not as a public API; the execution task must implement and freeze it.
- Business thresholds are proposals; non-significance is never absence except the bounded N = 96 `REPRESENTATION` cell.
- No file-accuracy, generalization, deployment or business-value claim.

## 13. Artifacts and reproduction

- Design module: `src/representation/sprint24_paired_decision_experiment.py`.
- CLI: `experiments/sprint24_task7m15_paired_decision_experiment.py` (`--mode smoke`, `--mode contract --out <fresh dir>`).
- Tests: `tests/representation/test_sprint24_paired_decision_experiment.py` (9 passed).
- Proposed contract: `docs/sprint24-paired-decision-experiment-contract-a01.json` (payload copy of the evidence contract; digest `b83b8520…`).
- Evidence root: `artifacts/sprint-24/evidence/s24-7-m15-a01/` (`proposed-contract.json`, `independent-calculator-check.json` PASS 9/9, `smoke.json` SMOKE-COMPLETE, `seed-plan.json` PASS, `run.json`).
- Worker record: `artifacts/sprint-24/task-7-m15-a01.md`.
- Reproduction:

```bash
python experiments/sprint24_task7m15_paired_decision_experiment.py --mode smoke
python experiments/sprint24_task7m15_paired_decision_experiment.py --mode contract --out artifacts/sprint-24/evidence/s24-7-m15-a01
python -m pytest tests/representation/test_sprint24_paired_decision_experiment.py -q
```
