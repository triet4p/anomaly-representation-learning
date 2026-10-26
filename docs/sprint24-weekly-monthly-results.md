# Sprint 24 weekly/monthly collective-movement study — results

- Study identity: `S24-BIZ-WM-v1`, freeze digest `6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd`.
- Frozen protocol: [`sprint24-weekly-monthly-protocol-v1.md`](sprint24-weekly-monthly-protocol-v1.md), digest `8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63`.
- Pre-outcome estimator conventions: [`sprint24-weekly-monthly-estimator-conventions-a01.json`](sprint24-weekly-monthly-estimator-conventions-a01.json), canonical digest `014ed05952c48196776e3292e096bfd56590f379fd08bf4a24aeb23dfcf06460`.
- Executed at source commit `31d113b4e5009cc41911023efa85ff05e92fdef5`; criterion `docs/sprint-plans/sprint-24.md:166-168`.
- Task / attempt / agent: `S24-7-M4` / `S24-7-M4-A01` / `S24M4GoldA01`.

**Result: both horizons are `INCONCLUSIVE`.** The weekly burden association leans positive but sits below both promise magnitude floors and is not retained under the capped-`n` sensitivity. The monthly association leans negative with the correct consistency, but its rolling-boundary sensitivity flips sign, so the frozen non-promise criteria do not all hold. Neither horizon is `PROMISING`; neither is `NOT-PROMISING`; no scaling study is proposed, and no specificity, detector or per-file claim is made.

## 1. What was executed

The study reused no pre-existing embedding cache because none existed; the owner authorized a bounded recapture and raised this study's cumulative forward cap from 41,000 to **54,000** ([amendment A01](sprint24-weekly-monthly-cache-budget-amendment-a01.md)).

| Step | Observed |
|---|---|
| Reviewed commit | `31d113b4e5009cc41911023efa85ff05e92fdef5` (parent `0a17ffa7…`), 9 paths, non-force push, clean remote checkout |
| Smoke | 96 forwards, `gpu-forward-smoke`, admission refused by design |
| Full capture | 26,262 forwards, 12/12 identity groups, 26,262/26,262 rows, `gpu-forward`, device `cuda`, live model/bank unchanged |
| Admission | `ADMITTED`, 192 cells pending endpoint computation |
| Cumulative forwards | 52,789 of 54,000 (96 smoke + 26,262 full), 1,211 remaining, no retry budget |
| Statistical analysis | 0 forwards, 0 model loads, 0 waveform reads |
| Determinism | two independent runs produced byte-identical data outputs |

The 13 `DEVELOPMENT-01` window-1 files that the accepted G2 capture omitted are included here; the cache-input identity for that history is therefore a new identity over the study's full required union, not a G2 match.

## 2. Pre-outcome freeze

Every estimator convention was frozen before any movement value, at canonical digest `014ed059…`, and bound in the run manifest. The frozen choices cover the E1 kernel (median pairwise distance over the **full** reference pool), the unbiased/biased estimator split, equal-file versus exposure weighting, the capped-`n` hash order and pair set, E2 prototypes (`K=8`, 4/16 sensitivity), E3/E4 definitions, N1/N2 null rules, C1/C2 association construction, seed combination (median of three run replicates, never `n=12`), sensitivity medians, bootstrap, multiplicity policy and unsupported handling. The record states that an underspecified detail was fixed rather than redefined and that a concrete protocol contradiction would stop the path; none was found.

## 3. Endpoint summaries (comparable pairs, pooled over three seeds)

| Horizon | Mode | E1 unbiased median | E1 biased median | E2 K=8 median | E3 median | E4 median | N2 half-split median |
|---|---|---:|---:|---:|---:|---:|---:|
| week | calendar | 0.005935 | 0.023461 | 0.2838 | 8.1175 | 0.030904 | 0.004796 |
| week | rolling | 0.009193 | 0.027421 | 0.3153 | 9.6309 | 0.030664 | 0.003432 |
| month | calendar | 0.001269 | 0.005679 | 0.1290 | 2.0331 | 0.012777 | 0.000999 |
| month | rolling | 0.000282 | 0.004417 | 0.1226 | 1.5373 | 0.011245 | 0.000568 |

Two descriptive observations matter for interpretation and are not decision inputs. First, the biased plug-in exceeds the unbiased value by roughly an order of magnitude at these sample sizes, which is why the frozen rule uses the U-statistic. Second, the median period-to-period movement is of the same order as the within-period half-split noise band (`E1` 0.0059 versus `N2` 0.0048 weekly; 0.00127 versus 0.00100 monthly), so the movement signal is weak relative to within-period sampling noise. The reference split-half null `q95` is `0.00069–0.00088` and the reference bandwidth `sigma` is `10.76–12.26` per history and seed, also descriptive.

## 4. Primary burden association by history

Spearman rank association within history; each cell lists the three per-seed values, their median (the decision input) and the mix-adjusted residual median. No cross-history pooling.

### Week (calendar, comparable matched pairs)

| History | Pairs | Distinct burden | Seed rhos | Seed median | Residual median |
|---|---:|---:|---|---:|---:|
| `DEVELOPMENT-01` | 12 | 5 | 0.199 / 0.296 / 0.540 | 0.296 | 0.322 |
| `DEVELOPMENT-02` | 5 | 4 | 0.667 / 0.821 / 0.667 | 0.667 | 0.667 |
| `DEVELOPMENT-03` | 6 | 4 | 0.030 / 0.516 / 0.213 | 0.213 | 0.030 |
| `DEVELOPMENT-04` | 13 | 6 | 0.190 / 0.283 / 0.285 | 0.283 | -0.168 |

### Month (calendar, comparable matched pairs)

| History | Pairs | Distinct burden | Seed rhos | Seed median | Residual median |
|---|---:|---:|---|---:|---:|
| `DEVELOPMENT-01` | 3 | 3 | -1.000 / -0.500 / -1.000 | -1.000 | -0.500 |
| `DEVELOPMENT-02` | 5 | 4 | -0.154 / -0.154 / -0.051 | -0.154 | 0.154 |
| `DEVELOPMENT-03` | 3 | 3 | +1.000 / -0.500 / -1.000 | -0.500 | 0.500 |
| `DEVELOPMENT-04` | 3 | 3 | -1.000 / -1.000 / -1.000 | -1.000 | -0.500 |

Seed stability is only partial. The weekly `DEVELOPMENT-03` seed values span `0.030–0.516`, and the monthly `DEVELOPMENT-03` values change sign across seeds (`+1.0`, `-0.5`, `-1.0`). The frozen rule uses the seed median as the run-replicate summary, but that instability is a real limitation of the three-seed representation, not a second independent test.

## 5. Sensitivity variants

Values are the median across the four estimable histories of the per-history seed medians.

| Variant | Week | Month | Primary (calendar) |
|---|---:|---:|---:|
| Primary | 0.289 | -0.750 | — |
| Rolling boundary | 0.280 | **+0.032** | sign kept weekly, flipped monthly |
| Capped-`n` exposure set | 0.105 | -0.279 | weekly retention 0.36 (< 0.5) |
| Alternate anchor (`2024-03-07Z`, `Etc/GMT-7`, Sunday) | 0.574 | -0.100 | sign kept both |

The weekly result fails the promise sensitivity criterion because the capped-`n` variant retains only 36 % of the primary magnitude (the other two variants retain the sign and at least half). The monthly result fails the non-promise sensitivity criterion because the rolling-boundary variant is positive.

## 6. Uncertainty

History-cluster bootstrap over the four histories, `B = 2000`, seed `2407`, descriptive at `n = 4`.

| Horizon | Point median | 95 % interval |
|---|---:|---|
| week | 0.289 | [0.213, 0.667] |
| month | -0.750 | [-1.000, -0.154] |

These intervals are coarse by construction and are never read as precise standard errors; adjacent pairs share a period, so no p-value is claimed.

## 7. Frozen decision traces

| Criterion | Week | Month | Detail |
|---|---|---|---|
| `A.estimability` | PASS | PASS | 4 of 4 histories reach ≥3 pairs with ≥2 burden values |
| `B.promise-magnitude` | FAIL | FAIL | medians 0.289/0.176 (week) and -0.750/-0.173 (month) versus floors 0.5/0.3 |
| `C.promise-consistency` | PASS | FAIL | 4/4 weekly histories positive; 0/4 monthly histories positive |
| `D.promise-sensitivity` | FAIL | FAIL | weekly capped-`n` retention 0.36; monthly rolling sign flip |
| `N1.nonpromise-magnitude` | FAIL | PASS | weekly median > 0; monthly median ≤ 0 and residual ≤ 0 |
| `N2.nonpromise-consistency` | FAIL | PASS | 0/4 weekly non-positive; 4/4 monthly non-positive |
| `N3.nonpromise-sensitivity` | FAIL | FAIL | weekly variants not all non-positive; monthly rolling positive |
| Specificity | not claimed | not claimed | zero annotated-clean and zero quiet-period pairs exist |

Precedence per the frozen rule: a conflict between the promise and non-promise criteria yields `INCONCLUSIVE`. That is exactly what happens at both horizons, for different reasons: the week is positive but too weak and not capped-`n` stable; the month is negative but not rolling-stable.

Resolution block: the monthly horizon is **coarse** (3–5 matched pairs per history; smallest attainable two-sided rank tail `0.0167–0.333`), so no per-history significance claim is possible there. The weekly histories have 5–13 pairs, but overlapping pairs still preclude an independence-based p-value.

## 8. Matrix and contrast accounting

- All **192** declared cells (`horizon × mode × endpoint × seed × history`) are published with an execution status: `EXECUTED` for every cell, with pair counts and median/min/max endpoint values.
- All **8** declared contrasts are accounted: `C1` and `C2` `AVAILABLE` with the associations above; `C3` quiet-period and `C4` temporal specificity `UNSUPPORTED-CONTROL-ABSENT`; `N3`/`N3b` `UNSUPPORTED-CONTROL-ABSENT`; `N1`/`N2` `DESCRIPTIVE-ONLY`.
- Pair accounting: 312 primary-calendar pairs (102 `COMPARABLE-MATCHED`, 100 `COMPARABLE-UNMATCHED-EXPOSURE`, 82 `INSUFFICIENT-PERIOD-SUPPORT`, 28 `PARTIAL-PERIOD`), plus the alternate-anchor pair table. Unsupported pairs stay in the denominator and are never silently dropped.

## 9. What this does and does not establish

- It establishes, for this exposed Development corpus and design: weekly collective movement is positively but weakly rank-associated with post-hoc failure burden, below the predeclared magnitude floors and not stable under exposure-capped support; monthly movement is negatively associated with a 4/4 consistent sign but reverses under the rolling boundary convention. Neither reaches a decisive triage disposition.
- It does **not** establish a detector, an operating point, calibrated specificity, recall, detection delay, false-alarm rate, per-file accuracy or causal effect. Movement remains a description, not an anomaly; absence of annotation remains unknown, not healthy.
- It does **not** answer the business question of whether week/month movement is promising enough to scale. The frozen rule returns `INCONCLUSIVE`; a scaling proposal would require the owner to revise the study design (more histories, denser periods, verified controls) or the disposition rule, neither of which is proposed here.
- The daily horizon remains deferred with its floor unchanged; the owner's 500–1,000 files/day deployment context is not evidence about this corpus's density.

## 10. Limitations

- Four histories, three run-replicate seeds, and overlapping adjacent pairs; monthly support is 3–5 comparable matched pairs per history, weekly 5–13. Every association is coarse.
- No annotated-clean or quiet-period pair exists at either horizon, so no specificity or normal-variation contrast is available.
- Weekly composition drift is strong and the mix-adjusted residual is load-bearing; the residual medians are themselves unstable in sign at some histories.
- The pre-outcome conventions fixed several underspecified details (full-reference bandwidth, half-split rule, capped set composition, weighted-variant form, `ddof`); these are disclosed for independent review and were not chosen after seeing outcomes.
- The corpus is exposed-benchmark exploratory research: encoder, normalization, bank and thresholds previously saw Fit/Calibration-healthy rows spanning the timeline. Sprint 18 remains `UNRESOLVED`; Confirmation was never touched.

## 11. Cost and custody

| Quantity | Value |
|---|---:|
| Forwards executed by the analysis | 0 |
| Capture forwards (smoke + full) | 96 + 26,262 = 26,358 |
| Cumulative after capture | 52,789 of 54,000 |
| Remaining authorized | 1,211 |
| Cap reset | none |
| Model / checkpoint / waveform loads in analysis | 0 / 0 / 0 |

Capture store and reports were transferred byte-identically (54 files verified). Accepted G2 raw outputs were never modified; the reviewed G2 record root stayed outside the clean checkout and was passed explicitly.

## 12. Reproduction

```bash
# Statistical analysis on the admitted store (no forwards)
python experiments/sprint24_task7m4_weekly_monthly_study.py \
  --data-root F:/ai-ml/_s24m1-local-metadata \
  --edges experiments/sprint24_task7m1_dev-edges.json \
  --g2-accepted-root artifacts/sprint-24/evidence/s24-7-m4-a01/g2-accepted \
  --cache-root artifacts/sprint-24/evidence/s24-7-m4-a01/runs/full/store \
  --expected-manifest-sha256 <history>=<sha256> ... \
  --out artifacts/sprint-24/evidence/s24-7-m4-a01/study-primary

# Focused checks
PYTHONPATH=src python -m pytest \
  tests/representation/test_sprint24_weekly_monthly_endpoints.py \
  tests/representation/test_sprint24_weekly_monthly_execution.py -q
```

## 13. Evidence inventory

All outputs are under `artifacts/sprint-24/evidence/s24-7-m4-a01/`.

| Output | Content | SHA-256 |
|---|---|---|
| `study-primary/decisions.json` | both disposition traces, medians, sensitivity values, resolution | `5c538f32548881f6568f03f3f6a6584c060da7c52c7dc0d000a3c9a95b9f3bcc` |
| `study-primary/associations.json` | per-history per-seed and seed-median associations | `ef04cf025a245c486fef3fd80e5554d1c405b3d0e885dd8ac3290b334512ff63` |
| `study-primary/sensitivities.json` | rolling, capped-`n` and alternate-anchor blocks | `367e7e5b5aa7cb773a1387e03fcc809a6c4dbb3e664254af76066e6669336694` |
| `study-primary/uncertainty.json` | history-cluster bootstrap intervals | `1d2cee826df2d96ac66f5d0ba2634fd7b4f147a1906691ef9463822fe0abb2a4` |
| `study-primary/contrasts.json` | C1–C4/N1–N3b states and outcomes | `2081ea08f791202aa562363a3abaaf2c4e560725461a716546da130df3d90b26` |
| `study-primary/execution-cells.json` | 192 cell outcomes | `d15d3b31ca05a0a8e463d436d7111edd71e6a7ba924b11983924e87f74f4156e` |
| `study-primary/execution-ledger.json` | combined cells, contrasts, decisions, uncertainty | `3c9a7c94d50b5d00cce6c0d9fe97774b223c6fb6109a789b35e38a747cbb8e8a` |
| `study-primary/pair-endpoints-primary.jsonl` | every primary-calendar pair and seed endpoint row | `c1b6745a7d4dfc11e2dc18e7b6b25c86fe5834d000235353d5c83b0c1ed87320` |
| `study-primary/pair-endpoints-alternate.jsonl` | alternate-anchor pair table | `17257050bf5fb4d63cc8183b621dbd842fd467553b5e45557db5a991da5cc530` |
| `study-primary/reference-fits.json` | bandwidth, null `q95`, reference sizes per history and seed | `086031256c20c5d7a6ddd3dc17c2c7bd69ba5320faff26f933e2d5596b89f9d6` |
| `study-primary/pre-outcome-conventions.json` | bound conventions digest | `a908fd75d72b1177670b4cda620f9d4fc148a335bbb6ae3a4f32b2e32ae6fa41` |
| `study-primary/cost.json` | zero-forward analysis cost | `eb41c3c3536346dc936d6999d2ea75864f707927f8ac09c860a79dd116476353` |
| `study-primary/custody.json` | cache and protocol custody | `d3fbf0b41e8693ce3bc2dcf644e60616609b64f3167de63cbb2fe29b1ec6e7ac` |
| `runs/full/capture-report.json` | capture proof (12 groups, 26,262 rows, `ADMITTED`) | `2d55e64813f520d925c4f2ab3a737776df86eb89486698e185c8d2c024328f15` |
| `runs/smoke/smoke-report.json` | 96-forward smoke | `6caa8b87fec6fd8ccfa87a72835e85a6adc6ab90542ad78ace2bcb2425a8405d` |
| `admission-full/run.json` | local admission proof from the transferred store | see output digest map inside |

## 14. History: preflight refusal and amendment

The first execution attempt found no persisted embedding cache and failed closed (0 of 26,262 rows; no movement). That refusal remains part of the record and is preserved in `blocked.json`, `preflight.json` and the admission evidence. The owner then authorized the bounded recapture at a 54,000 cumulative cap; the cache-budget amendment and the pre-outcome conventions above govern the run reported here. The frozen protocol, study digest, endpoints and decision rule were never changed.
