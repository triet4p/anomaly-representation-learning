# Sprint 24 Task 7-M14 — Representation signal versus population-endpoint sensitivity on the existing embedding cache

- Task / attempt / agent: `S24-7-M14` / `S24-7-M14-A02` / `S24M14GoldA02` (resumed replacement for the paused `S24-7-M14-A01`; R1 `changes_requested` consumed, R2 corrections below).
- Criterion: `docs/sprint-plans/sprint-24.md:240-248`. No quantitative success gate is assigned; a faithful negative or inconclusive result satisfies execution acceptance.
- Frozen pre-outcome specification: `docs/sprint24-representation-endpoint-diagnostic-specification-a01.json` (`S24-REPRESENTATION-ENDPOINT-DIAGNOSTIC-SPEC-A01`, sha256 `35b6ef7dd9364074360bd3c497ab27ec12df4d061f29d30dfc79a077e2d6084c`), written after the metadata-only census and before any M14 outcome. Its bytes are preserved verbatim under `artifacts/sprint-24/evidence/s24-7-m14-a01/spec/` with a custody record; it was never edited and no amendment was needed.
- Metadata-only feasibility census: `artifacts/sprint-24/evidence/s24-7-m14-a01/census/census.json` (sha256 `d3a0ce554692fd982ad383cf9154f65cea7ba6725655ca5ab821ff1bf4675bae`), `metadata_only=true`, `embeddings_read=false`, `outcomes_computed=false`.
- Cache: the admitted `S24-7-M11` embeddings preserved read-only for M12 under `artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full` (36 files, digest-of-digests `ae7aa42d7661d95d9d292400f53f4ec76128fd9618acc75cc6c6fcf581aa02b3`). **0 new forwards; cumulative observed forwards remain 77,653.**
- This is **not** a revision of `S24-7-M11`, `S24-7-M12` or `S24-7-M13`, not independent confirmation of any of them, and not a temporal, business, performance or detection claim.

## 1. What was built and verified to the fail-closed boundary

- `src/representation/sprint24_representation_endpoint_diagnostic.py` — the cache-only three-branch engine. It verifies the frozen specification digest, every pinned corpus/audit/preflight/freeze/contract digest, the canonical contract binding and every accepted M11/M12/M13 product digest **before** any statistic; admits the cache through the accepted identity-strict public admission path; fits one causal reference per `(history, encoder seed)` on the `REFERENCE` pool only; builds the metadata census and refuses any drift; then runs the frozen design and writes the machine-readable tables.
- `experiments/sprint24_task7m14_representation_endpoint_diagnostic.py` — the public entry point with `--stage census` (manifest and M9 audit only, no cache, no embeddings) and `--stage audit` (the full cache-only path). It refuses a non-empty `--out`, records the exact table digests, and re-verifies cache and prior-product bytes after the run.
- `tests/representation/test_sprint24_representation_endpoint_diagnostic.py` — 19 focused checks: capped largest-remainder apportionment, exact period and stratum quotas, duplicate-free and disjoint arm draws, equality of the implemented E1 with the accepted unbiased estimator, exact even occupancy quanta, the within-prototype diagnostic (identity zeros, tangential movement that a radial statistic would miss, empty-cluster exclusion and coverage abstention), zero and boundary-safe perturbation behaviour, census accounting/abstention, summary null-q95 semantics, E2 detection-fraction exclusion of invalid-repeat sentinels, the descriptive coverage record with `D` computed from valid repeats, and the frozen-specification byte refusal.
- `artifacts/sprint-24/evidence/s24-7-m14-a01/independent-verification.py` — an independent verifier that imports neither the M14 module nor any accepted estimator module: it rebuilds the cell/stratum tables from the manifest and M9 audit, refits the bandwidth, KMeans prototypes, residual scale and reference direction from the cache, and recomputes E1, E2 and D from the persisted arm identities before comparing every published value.
- Refusal boundaries exercised on real inputs: a tampered specification copy raises `REFUSED-SPEC-DRIFT`; a changed census status raises `REFUSED-CENSUS-DRIFT`; a changed accepted product raises `REFUSED-DIGEST-DRIFT`; a changed cache digest-of-digests raises `REFUSED-CACHE-DRIFT`; a non-empty output directory is refused. There is no recapture fallback anywhere in the code path.
- Implementation defects found during the build were corrected before the final candidate and did not change any frozen design constant: a period-matched arm split that lost its per-period quota, an accounting omission of period-matched records for pooled-abstained units, a control reassignment baseline error in the independent verifier, an E2 detection-fraction unit mismatch (raw quanta compared with a size-normalised q95, first corrected by dividing by size and in R2 further corrected to exclude the invalid-repeat sentinel from the denominator), boundary-safe records below the perturbed-mass floor still carrying computed estimator summaries into the branch-C aggregates, and a misleading unit-level coverage status that was removed in R2 in favour of the descriptive record above (the frozen per-repeat rule is unchanged; no gate was added). The full runs containing the superseded behaviour were discarded; the final candidate is a clean rerun, and the independent verifier checks the corrected control fractions, the coverage record against its repeats and the abstention-status consistency.

## 2. Lineage, cache custody and forward accounting

- Pinned digests verified before computation: corpus manifest `9606648d…`, corpus audit `5fddabc1…`, G2 preflight, canonical execution freeze, and the M10 prospective-contract canonical content; the live census/plan/custody binding was recomputed and matched the frozen contract. Every accepted M11/M12/M13 product digest matched before and after the run.
- Admission: `ADMITTED`, required 24,840 rows, available 24,840, missing 0, 18 groups. No row was captured, added or substituted.
- The cache digest-of-digests is unchanged before and after the audit, and each of the 36 files keeps its recorded digest. The prior M11 movement/run records and the M11/M12/M13 source, test and documentation files keep their accepted digests.
- Forwards: `forwards_executed_this_run = 0`, `model_loads = 0`, `checkpoint_opens = 0` (an audit hook counts every opened `.pt/.pth/.ckpt/.bin/.safetensors` path), `cuda_initialized = false`. `torch` is imported transitively by the accepted readiness/inference modules; no model, checkpoint, optimizer or GPU context is constructed. Cumulative observed forwards stay **77,653**.
- Reference fits: exactly 18 (one per history × encoder seed, never per comparison), `n_reference = 220` causal files each, `KMeans(random_state=2407, n_init=10)` for `K ∈ {4, 8, 16}` fitted on the reference pool only. Evaluation arms, labels, families, severities and context fields never enter a fit.

## 3. Frozen design

### 3.1 Specification, census and timing

The metadata-only census was written first, then the specification was frozen and pinned, and only then were any estimator values computed. The corpus, the M9 audit and the admitted embeddings were already exposed before this design existed; the record therefore makes **no pre-registration or fresh-confirmation claim**, discloses the post-exposure timing, and never rewrites an original byte. Later deviations would be separate timestamped amendment JSON records declaring `amends_spec_sha256`; none were needed.

### 3.2 Branch A — representation signal

The unit is one accepted `history × horizon` cell × one `robot|program` stratum × one supported per-arm size. Sizes are frozen from the census as week `{2, 3}` and month `{3, 4}`, because within-stratum abnormal support is only 1–10 files. For each repeat:

- the abnormal arm is `s` files drawn without replacement from the stratum's abnormal candidates;
- a single `3s` normal draw (without replacement) is split into three disjoint `s`-file arms `normal_1`, `normal_2`, `normal_3`;
- the comparison is abnormal versus `normal_1`; the matched normal-normal variability band is `normal_2` versus `normal_3` at the identical size, estimator and seed.

Labels construct and diagnose the arms only; they never enter a reference fit, prototype, bandwidth or detector. The `PERIOD-MATCHED` variant holds the per-period arm composition exactly to a capped largest-remainder quota (`min(n_abnormal_p, floor(n_normal_p / 3))` per period) and is reported only where the census supports it; the `POOLED-PERIOD` variant mixes periods and publishes the per-repeat period-mix total-variation distance as a measured confounder. Every planned stratum and size is either `COMPUTED` or carries a closed abstention code and stays in the denominator.

### 3.3 Branch B — occupancy versus within-prototype movement

The same valid arms and the same fixed causal-reference prototypes and bandwidth are scored with:

- the accepted unbiased squared MMD `E1`;
- exact-quanta occupancy `E2_K4`, `E2_K8`, `E2_K16` (`E2 = m / s`,
  $m=\tfrac12\sum_g |c_g-d_g|$);
- the predeclared bounded residual diagnostic `D`. With `r_i = x_i - c_{\ell(i)}` and
    $w_k=(n_{A,k}+n_{B,k})/(2s)$ over prototypes occupied in both arms,

    $$D_{\mathrm{LOC}}=\frac{\sum_k w_k\,\lVert \bar r_{A,k}-\bar r_{B,k}\rVert^2}{s_K^2},\qquad
    D_{\mathrm{SPREAD}}=\frac{\sum_k w_k\,\bigl|\overline{\lVert r_{A,k}\rVert^2}-\overline{\lVert r_{B,k}\rVert^2}\bigr|}{s_K^2},$$

    where $s_K^2$ is the mean squared residual norm of the causal reference rows for that
    `(history, seed, K)`. Weights are renormalised over the common prototypes and the
    un-renormalised covered mass is published as `coverage`; a repeat below 0.5 coverage is
    abstained, never silently renormalised. Each `D` summary is therefore computed from its
    valid repeats only and publishes that count as `n`; the coverage record reports
    `repeats_valid`, `repeats_total`, `valid_fraction` and a `FULL`/`PARTIAL` reliability
    label, and is descriptive — it is not a second unit-level gate. At the primary `K=8` the
    location difference is also split into radial and tangential components, because a
    radial-only statistic cannot represent every direction of movement.
- Source composition is reported as the period-mix TV; prototype-boundary proximity is
  reported as the share of each arm with margin at most `0.25·sqrt(s_K^2)` under `K=8`;
  per-file reassignment is measurable only for the paired positive-control arms of branch C
  and is never invented for disjoint real-source arms.

### 3.4 Branch C — positive-control sensitivity

For each `cell × size × encoder seed` (72 unit-seeds; sizes week `{40, 60}`, month `{160, 250}`), two disjoint equal-quota normal arms are drawn with a robot|program quota `floor(n_normal/2)` per stratum. The direction is the first right singular vector of the centred causal-reference matrix for that `(history, seed)`, its sign fixed so the largest absolute loading is positive, and `sigma_u` is the root-mean-square reference projection on it. The predeclared magnitudes are `{0, 0.1, 0.25, 0.5, 1.0, 2.0} × sigma_u`:

- `GLOBAL-SHIFT` shifts every row of the perturbed arm;
- `WITHIN-PROTOTYPE-SAFE` shifts only rows whose `K=8` boundary margin strictly exceeds
  the shift norm, so the nearest prototype cannot change by the triangle inequality; the
  perturbed count and fraction are published and a repeat below 0.5 perturbed mass is
  abstained.

`delta = 0` is a single record that is also the unchanged-versus-unchanged null band. Every record publishes the exact realised shift norm, the perturbed count, the per-file prototype reassignment count under `K=8`, and the share of repeats above the null q95 for every estimator. Artificial shifts verify **measurement responsiveness only**: not learned anomaly semantics, generalisation, significance or power guarantees, or business success.

### 3.5 Estimators and accounting

The six accounting estimators are `E1`, `E2_K4`, `E2_K8`, `E2_K16`, `D_LOC` (primary `K=8`) and `D_SPREAD` (primary `K=8`); `D_RADIAL`, `D_TANGENTIAL` and `D` at `K=4/16` are published as descriptive decomposition, not as extra endpoints. Planned grids: branch A `960 × 3 seeds × 6`; the period-matched variant the same 960 units with their own census status; branch C `72 × (11 comparison records + 1 identity record)`. No best metric, `K`, seed, size, stratum or effect magnitude is selected.

## 4. Realized support and explicit abstentions

**Planned and realized support**

| Branch | Planned | Computed | Abstained |
|---|---|---|---|
| A pooled units | 960 | 329 | 631 |
| A period-matched units | 960 | 52 | 908 |
| C unit-seeds | 72 | 72 | 0 |
| C comparison/identity records | 864 | 576 | 288 |

Branch-A abstention reasons: `{"ABSTAIN-INSUFFICIENT-ABNORMAL": 438, "ABSTAIN-INSUFFICIENT-NORMAL": 41, "ABSTAIN-NO-ABNORMAL": 152, "COMPUTED": 329}`.


Branch-A abstention reasons: `{"ABSTAIN-INSUFFICIENT-ABNORMAL": 438, "ABSTAIN-INSUFFICIENT-NORMAL": 41, "ABSTAIN-NO-ABNORMAL": 152, "COMPUTED": 329}`.


Abstention is complete and closed: the six branch-A estimator cells of every abstained unit stay in the 17,280-cell denominator, the period-matched variant keeps its own 17,280-cell denominator, and the 288 boundary-safe records that cannot perturb at least half their arm are published as `ABSTAIN-BOUNDARY-SAFE-MASS`. No planned cell is silently dropped.

## 5. Branch A — measured representation signal

**Branch A pooled-period (all cells and encoder seeds)**

| Horizon | Size | Estimator | Unit-seeds | Median > null q95 | Median contrast | Median null | Median null q95 |
|---|---|---|---|---|---|---|---|
| week | 2 | E1 | 219 | 5 (2.3%) | -0.01781 | -0.01509 | 0.2168 |
| week | 2 | E2_K4 | 219 | 8 (3.7%) | 0.5 | 0.5 | 1 |
| week | 2 | E2_K8 | 219 | 1 (0.5%) | 0.5 | 0.5 | 1 |
| week | 2 | E2_K16 | 219 | 0 (0.0%) | 1 | 1 | 1 |
| week | 2 | D_LOC | 213 | 12 (5.6%) | 1.828 | 1.78 | 2.718 |
| week | 2 | D_SPREAD | 213 | 25 (11.7%) | 0.2116 | 0.2282 | 0.5582 |
| week | 3 | E1 | 63 | 0 (0.0%) | -0.009337 | -0.01033 | 0.1431 |
| week | 3 | E2_K4 | 63 | 5 (7.9%) | 0.3333 | 0.3333 | 0.7542 |
| week | 3 | E2_K8 | 63 | 1 (1.6%) | 0.5833 | 0.6667 | 1 |
| week | 3 | E2_K16 | 63 | 0 (0.0%) | 0.6667 | 0.6667 | 1 |
| week | 3 | D_LOC | 63 | 5 (7.9%) | 1.497 | 1.46 | 2.193 |
| week | 3 | D_SPREAD | 63 | 1 (1.6%) | 0.2184 | 0.2546 | 0.4986 |
| month | 3 | E1 | 423 | 2 (0.5%) | -0.01324 | -0.01055 | 0.1398 |
| month | 3 | E2_K4 | 423 | 0 (0.0%) | 0.3333 | 0.3333 | 0.6792 |
| month | 3 | E2_K8 | 423 | 0 (0.0%) | 0.6667 | 0.6667 | 1 |
| month | 3 | E2_K16 | 423 | 0 (0.0%) | 0.6667 | 0.6667 | 1 |
| month | 3 | D_LOC | 423 | 3 (0.7%) | 1.421 | 1.516 | 2.317 |
| month | 3 | D_SPREAD | 423 | 3 (0.7%) | 0.2183 | 0.2315 | 0.526 |
| month | 4 | E1 | 282 | 1 (0.4%) | -0.01095 | -0.007408 | 0.1023 |
| month | 4 | E2_K4 | 282 | 2 (0.7%) | 0.25 | 0.25 | 0.75 |
| month | 4 | E2_K8 | 282 | 1 (0.4%) | 0.5 | 0.5 | 0.75 |
| month | 4 | E2_K16 | 282 | 2 (0.7%) | 0.625 | 0.75 | 1 |
| month | 4 | D_LOC | 282 | 3 (1.1%) | 1.321 | 1.417 | 2.146 |
| month | 4 | D_SPREAD | 282 | 3 (1.1%) | 0.2129 | 0.2231 | 0.4701 |

**Branch A exact period-matched (supported units only)**

| Horizon | Size | Estimator | Unit-seeds | Median > null q95 | Median contrast | Median null | Median null q95 |
|---|---|---|---|---|---|---|---|
| week | 2 | E1 | 3 | 0 (0.0%) | 0.01366 | -0.03931 | 0.1322 |
| week | 2 | E2_K4 | 3 | 0 (0.0%) | 0.5 | 0.5 | 1 |
| week | 2 | E2_K8 | 3 | 0 (0.0%) | 0.5 | 0.5 | 1 |
| week | 2 | E2_K16 | 3 | 0 (0.0%) | 0.5 | 0.5 | 1 |
| week | 2 | D_LOC | 3 | 0 (0.0%) | 1.192 | 2 | 2.755 |
| week | 2 | D_SPREAD | 3 | 0 (0.0%) | 0.1172 | 0.1277 | 0.2389 |
| month | 3 | E1 | 129 | 2 (1.6%) | -0.01575 | -0.01179 | 0.1282 |
| month | 3 | E2_K4 | 129 | 4 (3.1%) | 0.3333 | 0.3333 | 0.6667 |
| month | 3 | E2_K8 | 129 | 3 (2.3%) | 0.5833 | 0.6667 | 1 |
| month | 3 | E2_K16 | 129 | 4 (3.1%) | 0.6667 | 0.6667 | 1 |
| month | 3 | D_LOC | 127 | 3 (2.4%) | 1.461 | 1.531 | 2.304 |
| month | 3 | D_SPREAD | 127 | 9 (7.1%) | 0.2326 | 0.232 | 0.494 |
| month | 4 | E1 | 24 | 1 (4.2%) | 0.04366 | -0.009153 | 0.08751 |
| month | 4 | E2_K4 | 24 | 1 (4.2%) | 0.5 | 0.375 | 0.75 |
| month | 4 | E2_K8 | 24 | 0 (0.0%) | 0.5 | 0.5 | 1 |
| month | 4 | E2_K16 | 24 | 1 (4.2%) | 0.75 | 0.75 | 1 |
| month | 4 | D_LOC | 22 | 0 (0.0%) | 1.588 | 1.671 | 2.288 |
| month | 4 | D_SPREAD | 22 | 1 (4.5%) | 0.1892 | 0.2528 | 0.4661 |

The honest answer at the supported sizes is **no measured abnormality-associated contrast above the matched normal-normal variability**, not "no representation signal", and the claim is made at the grain it was measured. At the finer cell/size/seed grain, 31 of 72 aggregate `E1` medians sit above the corresponding null median while **0 of 72** sit above the null q95; at the coarser horizon/size/seed grain most aggregates are at or below the null median, with a few exceptions (for example week `N=2`, seed 181802 and month `N=3`, seed 181803), and none approaches its null q95. At the unit-seed grain only 0–2.3 % raise the `E1` median above the matched null q95 (0.7–7.9 % for `D_LOC`, 0–1.6 % for `E2_K8`). The null band itself is wide because the supported arms are tiny (2–4 files) and the files are heterogeneous, so the design cannot resolve modest effects. The pooled variant is period-confounded; the exact period-matched variant is available for only 52 of 960 units and shows the same picture. These results do not show that the embeddings carry no abnormality information; they show that this cache, inside one `robot|program` stratum and at the sizes its support allows, does not deliver a contrast larger than normal-normal sampling variability.

## 6. Branch B — occupancy versus within-prototype movement

**Branch B endpoint matrix (K=8 primary occupancy and location level)**

| Horizon | Size | Unit-seeds | E2 median exactly 0 | D_LOC above null | Within-prototype-only | Median E1 / null q95 | Median E2_K8 / null q95 | Median D_LOC / null q95 |
|---|---|---|---|---|---|---|---|---|
| month | 3 | 423 | 4 | 3 | 0 | -0.01324 / 0.1398 | 0.6667 / 1 | 1.421 / 2.317 |
| month | 4 | 282 | 1 | 3 | 0 | -0.01095 / 0.1023 | 0.5 / 0.75 | 1.321 / 2.146 |
| week | 2 | 213 | 5 | 12 | 0 | -0.01862 / 0.2153 | 0.5 / 1 | 1.828 / 2.718 |
| week | 3 | 63 | 2 | 5 | 0 | -0.009337 / 0.1431 | 0.5833 / 1 | 1.497 / 2.193 |

At `K=8` neither the occupancy endpoint nor the residual diagnostic separates the abnormal arms from the matched null at these sizes: `within_prototype_only` (E2 median exactly zero while `D_LOC` is above its null q95) is 0 in every horizon/size/seed cell, only 1–5 unit-seeds per size have an exactly-zero `E2` median, and the units with `D_LOC` above null (3–12 of 63–423) are the same order as those with `E2` above null. This is a resolution statement, not a claim that the two endpoints measure the same thing: branch C shows they respond at different magnitudes.

Coverage is reported exactly as the frozen rule operates. Of the 987 computed branch-A unit-seeds, 149 have fewer than half of their repeats above the 0.5 per-repeat coverage floor; their `D` summaries are computed from the valid repeats and publish the reduced `n` (143 of them still publish a `D_LOC` summary from at least one valid repeat). The descriptive `coverage` record marks these `PARTIAL`/`at_least_half_valid=false`; it does not suppress the estimator, does not add a unit-level gate and does not renormalise anything. `D_RADIAL` and `D_TANGENTIAL` are published per unit and per `K`; no radial-only reading is offered. Source composition is published as the period-mix TV and prototype-boundary proximity as the margin share, so within-cell distribution change is never conflated with occupancy, source composition or boundary crossings. The mechanism question is answered instead by branch C, where a known shift is applied by construction.

## 7. Branch C — positive-control sensitivity

**Branch C positive controls (unit-weighted detection fraction; `-` = no record computed)**

| Horizon | Size | delta (sigma_u) | Mode | Computed unit-seeds | E1 | E2_K8 | D_LOC | Mean of unit median reassigned | Zero-reassignment repeats |
|---|---|---|---|---|---|---|---|---|---|
| week | 40 | 0 | GLOBAL-SHIFT | 18 | 0.06 | 0.04 | 0.06 | 0.0 | 1.00 |
| week | 40 | 0.1 | GLOBAL-SHIFT | 18 | 0.07 | 0.04 | 0.05 | 1.0 | 0.32 |
| week | 40 | 0.25 | GLOBAL-SHIFT | 18 | 0.17 | 0.04 | 0.07 | 2.1 | 0.07 |
| week | 40 | 0.5 | GLOBAL-SHIFT | 18 | 0.54 | 0.06 | 0.14 | 4.8 | 0.00 |
| week | 40 | 1 | GLOBAL-SHIFT | 18 | 1.00 | 0.23 | 0.56 | 10.4 | 0.00 |
| week | 40 | 2 | GLOBAL-SHIFT | 18 | 1.00 | 1.00 | 0.89 | 27.2 | 0.00 |
| week | 40 | 0.1 | WITHIN-PROTOTYPE-SAFE | 18 | 0.06 | 0.04 | 0.06 | 0.0 | 1.00 |
| week | 40 | 0.25 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 40 | 0.5 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 40 | 1 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 40 | 2 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 60 | 0 | GLOBAL-SHIFT | 18 | 0.06 | 0.05 | 0.06 | 0.0 | 1.00 |
| week | 60 | 0.1 | GLOBAL-SHIFT | 18 | 0.08 | 0.04 | 0.07 | 1.2 | 0.17 |
| week | 60 | 0.25 | GLOBAL-SHIFT | 18 | 0.24 | 0.06 | 0.09 | 3.4 | 0.01 |
| week | 60 | 0.5 | GLOBAL-SHIFT | 18 | 0.75 | 0.12 | 0.20 | 7.5 | 0.00 |
| week | 60 | 1 | GLOBAL-SHIFT | 18 | 1.00 | 0.40 | 0.85 | 15.9 | 0.00 |
| week | 60 | 2 | GLOBAL-SHIFT | 18 | 1.00 | 1.00 | 1.00 | 40.9 | 0.00 |
| week | 60 | 0.1 | WITHIN-PROTOTYPE-SAFE | 18 | 0.07 | 0.05 | 0.06 | 0.0 | 1.00 |
| week | 60 | 0.25 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 60 | 0.5 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 60 | 1 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| week | 60 | 2 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 160 | 0 | GLOBAL-SHIFT | 18 | 0.06 | 0.06 | 0.06 | 0.0 | 1.00 |
| month | 160 | 0.1 | GLOBAL-SHIFT | 18 | 0.13 | 0.06 | 0.06 | 4.1 | 0.01 |
| month | 160 | 0.25 | GLOBAL-SHIFT | 18 | 0.55 | 0.08 | 0.13 | 9.5 | 0.00 |
| month | 160 | 0.5 | GLOBAL-SHIFT | 18 | 1.00 | 0.21 | 0.65 | 19.2 | 0.00 |
| month | 160 | 1 | GLOBAL-SHIFT | 18 | 1.00 | 0.86 | 1.00 | 40.2 | 0.00 |
| month | 160 | 2 | GLOBAL-SHIFT | 18 | 1.00 | 1.00 | 1.00 | 107.3 | 0.00 |
| month | 160 | 0.1 | WITHIN-PROTOTYPE-SAFE | 18 | 0.07 | 0.06 | 0.08 | 0.0 | 1.00 |
| month | 160 | 0.25 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 160 | 0.5 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 160 | 1 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 160 | 2 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 250 | 0 | GLOBAL-SHIFT | 18 | 0.06 | 0.06 | 0.06 | 0.0 | 1.00 |
| month | 250 | 0.1 | GLOBAL-SHIFT | 18 | 0.17 | 0.05 | 0.08 | 6.5 | 0.00 |
| month | 250 | 0.25 | GLOBAL-SHIFT | 18 | 0.75 | 0.10 | 0.22 | 14.8 | 0.00 |
| month | 250 | 0.5 | GLOBAL-SHIFT | 18 | 1.00 | 0.32 | 0.93 | 30.4 | 0.00 |
| month | 250 | 1 | GLOBAL-SHIFT | 18 | 1.00 | 0.96 | 1.00 | 63.3 | 0.00 |
| month | 250 | 2 | GLOBAL-SHIFT | 18 | 1.00 | 1.00 | 1.00 | 169.3 | 0.00 |
| month | 250 | 0.1 | WITHIN-PROTOTYPE-SAFE | 18 | 0.09 | 0.06 | 0.08 | 0.0 | 1.00 |
| month | 250 | 0.25 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 250 | 0.5 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 250 | 1 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |
| month | 250 | 2 | WITHIN-PROTOTYPE-SAFE | 0 | - | - | - | - | - |

**Within-prototype-safe feasibility**

| Horizon | Size | delta | Computed / planned unit-seeds | Median perturbed fraction |
|---|---|---|---|---|
| month | 160 | 0.1 | 18 / 18 | 0.645 |
| month | 160 | 0.25 | 0 / 18 | - |
| month | 160 | 0.5 | 0 / 18 | - |
| month | 160 | 1 | 0 / 18 | - |
| month | 160 | 2 | 0 / 18 | - |
| month | 250 | 0.1 | 18 / 18 | 0.651 |
| month | 250 | 0.25 | 0 / 18 | - |
| month | 250 | 0.5 | 0 / 18 | - |
| month | 250 | 1 | 0 / 18 | - |
| month | 250 | 2 | 0 / 18 | - |
| week | 40 | 0.1 | 18 / 18 | 0.650 |
| week | 40 | 0.25 | 0 / 18 | - |
| week | 40 | 0.5 | 0 / 18 | - |
| week | 40 | 1 | 0 / 18 | - |
| week | 40 | 2 | 0 / 18 | - |
| week | 60 | 0.1 | 18 / 18 | 0.642 |
| week | 60 | 0.25 | 0 / 18 | - |
| week | 60 | 0.5 | 0 / 18 | - |
| week | 60 | 1 | 0 / 18 | - |
| week | 60 | 2 | 0 / 18 | - |

The positive controls order the endpoints by the magnitude at which they reach the predeclared 0.5 detection threshold, and the claims use that operational threshold rather than equating a below-threshold value with no response. `E1` reaches it first (from `0.25 sigma_u` for month `160`/`250` and from `0.5 sigma_u` for week `40`/`60`); `D_LOC` follows (from `0.5–1.0 sigma_u`); occupancy `E2_K8` is the least sensitive endpoint, reaching it only at `1.0–2.0 sigma_u`. At `delta = 0.1` the median detection fractions are 0.04–0.17 and all are below the 0.5 threshold (for example month `250`, `E1` is 0.17 versus 0.06 at zero shift); that is a below-threshold measurement, not evidence of no response. At `delta = 0.25` and above the distinction becomes operational for `E1`, but not for occupancy.

In the `WITHIN-PROTOTYPE-SAFE` mode, `delta = 0.1` is computed for all 72 unit-seeds (median perturbed fraction 0.65, observed range 0.58–0.75), with zero prototype reassignment in every repeat and `E2` exactly at its null-band values. At `delta = 0.25` the perturbed fraction falls below the declared 0.5 floor in every unit-seed (the highest observed median is about 0.4), so those records — and every record at `0.5–2.0` — are explicit `ABSTAIN-BOUNDARY-SAFE-MASS`, not low-validity responses. Because the feasible within-prototype magnitude is below every estimator's resolution, this audit cannot empirically separate "E2 blind while D responds" from "both are below resolution" on a real shift; it demonstrates the structural fact (zero reassignment, occupancy unchanged) and the resolution ordering above, and it does not claim more.

## 8. Measurement zero, identity and boundary behaviour

**Measurement-zero identity checks (72 unit-seeds)**

| Check | Observed |
|---|---|
| max abs biased E1 | 0.000e+00 |
| E2_K4/K8/K16 all exactly zero | 72/72 |
| max abs D_LOC | 0.000e+00 |
| max abs D_SPREAD | 0.000e+00 |
| max reassigned | 0 |
| unbiased E1 identity value range | -1.977e-02 .. -3.003e-03 |

The identity check is exact where the estimator's own definition permits it: biased E1 is exactly zero, `E2` is exactly zero for `K=4/8/16`, `D_LOC` and `D_SPREAD` are exactly zero and no prototype is reassigned. The accepted unbiased E1 is **not** zero on identical samples by its finite-sample diagonal; its exact identity value is published instead of being claimed as zero. In the boundary-safe mode, every valid repeat has zero reassignment, which is the triangle-inequality guarantee rather than an empirical coincidence, and the occupancy contrast equals the unchanged-arm occupancy contrast repeat by repeat.

## 9. Reproducibility and independent verification

**Reproduction (audit vs audit-repro)**

| Table | Byte-identical |
|---|---|
| lineage.json | True |
| census.json | True |
| support.json | True |
| branch-a.json | True |
| branch-a-period-matched.json | True |
| branch-b.json | True |
| branch-c.json | True |
| aggregates.json | True |
| arms.json | True |
| accounting.json | True |
| repeats.npz | True |
| controls.npz | True |

The independent verifier rebuilds the reference quantities from the cache (bandwidth, KMeans prototypes, residual scale and reference direction) and recomputes, for **every repeat of every computed pooled unit**, E1, exact-quanta E2 (K=4/8/16) and D: 197,400 arm-plan records and 4,964,832 numeric comparisons. It does the same for **every repeat of every computed period-matched unit** (52 units × 3 seeds × 200 repeats) and for **every repeat of every computed control record** (504 comparison records × 50 repeats); the 288 abstained control records are checked for status consistency rather than recomputed. It additionally re-derives **every published per-seed summary** (status, `n`, median/mean/q05/q95/min/max, null median/q95, above-null flag and detection fraction) for every computed pooled and period-matched unit-seed from the persisted repeat arrays — with the invalid-repeat sentinel excluded from E2 denominators exactly as the corrected module does — and **regroups every aggregate cell** (432 pooled, 230 period-matched, 672 branch-C estimator buckets) from the published summaries. It additionally checks arm size matching, disjointness, duplicate freedom, quota exactness, integer-even occupancy quanta, identity zeros, known shift norms, boundary-safe zero reassignment, the corrected control detection fraction against valid quanta only, the descriptive coverage record against its repeats (including the no-status-gate shape), accounting totals, specification and custody digests, and the byte identity of every statistical table and repeat array between the two full runs: **PASS, 0 failures across 6,780,750 checks**. A small real-cache smoke (2 repeats) was run first with the same public command; the final candidate uses the frozen 200/50 repeats.

## 10. Limitations and non-claims

- Exploratory, cache-only and post-exposure; no pre-registration, fresh confirmation or causal claim.
- Branch A sizes are tiny (week `s ∈ {2,3}`, month `s ∈ {3,4}`) because within-stratum abnormal support is 1–10 files; the matched null band is the only comparator, and a weak or absent contrast is not proof of absent representation signal.
- Only 52 of 960 branch-A units support exact period matching; the pooled variant's period-mix confounder is measured and reported, never hidden.
- The accepted unbiased E1 has a non-zero identical-sample diagonal; the measurement-zero claim uses the biased plug-in and exact E2/D zeros.
- The within-prototype positive control is feasible only at `delta = 0.1`, where no estimator resolves the shift; the "E2 blind while D responds" separation is therefore not observed, only the structural occupancy stability.
- Positive controls say nothing about learned anomaly semantics, generalisation, significance/power or business value.
- Weekly and monthly cells, the six histories and the three encoder seeds are reported separately; no pooled representation claim is made and no family fragment is treated as an overall result.
- No duplicate file identity inflates any arm; every draw is without replacement and every arm pair is disjoint.

## 11. Conditional next-step implications

- If the within-stratum representation question is to be answered with power, the prerequisite is more abnormal files per `robot|program` × period stratum (corpus design), not a different metric on this cache. This audit does not repair or regenerate the corpus.
- If occupancy `E2` remains a primary population endpoint, this cache shows it is the least responsive of the tested endpoints to a known embedding shift; the residual diagnostic `D` responds earlier, but the audit does not promote `D` to a business endpoint.
- A within-prototype separation of occupancy blindness from estimator resolution would need either larger prototype margins or a smaller-magnitude grid fixed before outcomes; that is a new design, not a post-hoc change to this one.
- No scope change, no new data, no inference, no training and no bank refit is implied or performed.

## 12. Artifacts and reproduction

- Evidence root: `artifacts/sprint-24/evidence/s24-7-m14-a01/` (`census/`, `spec/`, `originals/`, `audit/`, `audit-repro/`, `independent-verification.py`, `independent-verification.json`).
- Worker record: `artifacts/sprint-24/task-7-m14-a02.md` (R2; the R1 record `artifacts/sprint-24/task-7-m14-a01.md` is preserved).
- Public commands:

```bash
python experiments/sprint24_task7m14_representation_endpoint_diagnostic.py --stage census \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --out artifacts/sprint-24/evidence/s24-7-m14-a01/census

python experiments/sprint24_task7m14_representation_endpoint_diagnostic.py --stage audit \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --census artifacts/sprint-24/evidence/s24-7-m14-a01/census/census.json \
  --out artifacts/sprint-24/evidence/s24-7-m14-a01/audit

python artifacts/sprint-24/evidence/s24-7-m14-a01/independent-verification.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --out artifacts/sprint-24/evidence/s24-7-m14-a01/audit \
  --repro artifacts/sprint-24/evidence/s24-7-m14-a01/audit-repro \
  --report artifacts/sprint-24/evidence/s24-7-m14-a01/independent-verification.json
```
