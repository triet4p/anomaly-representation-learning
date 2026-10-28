# Sprint 24 Task 7-M12 — E2 sample-size bias and abnormal-period movement on the existing cache

- Task / attempt / agent: `S24-7-M12` / `S24-7-M12-A01` / `S24M12GoldA01` (role Gold).
- Criterion: `docs/sprint-plans/sprint-24.md:222-229` (parts A–D of the integrated exploratory audit). No new scientific success gate is introduced, and this document is **not** a revision of `S24-7-M11` and **not** independent confirmation of it.
- Frozen pre-outcome specification: `docs/sprint24-e2-cache-diagnostic-specification-a01.json` (`S24-E2-CACHE-DIAGNOSTIC-SPEC-A01`, sha256 `cde6021ff086398cf690d76a7eb78d656274721dc55621c8e8012fae5c2d47fc`), written before any M12 outcome was computed. Its `amendments` block records, with reasons, every later change: three descriptive corrections, one real execution deviation (`A03`, floor constant 10 → pre-registered 30 after the first run), and the post-assessment R2 corrections (`A05`). §12 summarises them; nothing in that block claims a post-outcome choice was pre-registered.
- Cache: the admitted `S24-7-M11` embeddings, retrieved read-only from `trietlm@192.168.99.214:/tmp/s24-7-m11-a03-run/cache-full` into a separate local location. **0 new forwards; cumulative observed forwards remain 77,653.**
- Worker record: `artifacts/sprint-24/task-7-m12-a01.md`. Evidence: `artifacts/sprint-24/evidence/s24-7-m12-a01/` (`audit/`, `audit-repro/`, `cache-full/`, `retrieval/`, `independent-verification.json`).
- Assessment history: `S24-7-M12-C1/R1` returned `changes_requested` (five MEDIUM findings). This document is the R2 candidate; every R1 outcome it supersedes is described in §13 and preserved in the R1 record.

## 1. What was built and verified to the fail-closed boundary

- `src/representation/sprint24_e2_cache_diagnostic.py` — the cache-only audit engine: pinned-digest + canonical-freeze + contract-binding lineage gate, identity-strict cache admission through the accepted public path, one reference fit per `(history, seed)` on the causal pool only, a support-derived matched per-arm size per cell, deterministic repeated subsampling without replacement, a label-blind robot/program stratified design plus an unstratified design, exact integer occupancy quanta for every equal-`n` quantity, the baseline-variability decomposition, the K × seed sweep, the post-hoc diagnostic decomposition and complete planned/computed/abstained accounting.
- `experiments/sprint24_task7m12_e2_cache_diagnostic.py` — the public audit entry point. It refuses a non-empty `--out`, writes the machine-readable tables, and records the run.
- `tests/representation/test_sprint24_e2_cache_diagnostic.py` — 28 focused checks, including two integrated-path checks that drive an unsupported comparison through the real assembly (infeasible stratified allocation, sub-floor cell) and an association check that excludes an abstained transition without raising, plus exact-zero quanta (a mathematically equal pair is exactly `0`, never a floating tie), exact direction counts, an infeasible matched allocation that abstains without raising, a missing stratified design that yields an abstention record instead of a lookup error, the sub-floor cell abstention, the non-additivity inequality on a fixture with a known `0.4` gap, sparse/one-sided/confounded disclosure, NPY-header exposure maps and the retained-exposure construction.
- Refusal boundaries exercised on real inputs: a tampered `corpus-audit.json` copy raises `REFUSED-DIGEST-DRIFT` before any statistic; admission would refuse a missing, foreign-provenance or identity-drifted group; the CLI refuses a non-empty output directory.
- The **full-arm recomputation reproduces the published `S24-7-M11` `E2_K4`/`E2_K8`/`E2_K16` values exactly**: 540/540 cells, maximum absolute difference `0.0`. The audit therefore measures the same endpoint on the same retained arms and the same reference fit, and changes nothing in M11.

## 2. Lineage, cache custody, retained exposure and forward accounting

- Pinned digests verified before computation: corpus manifest `9606648d…`, corpus audit `5fddabc1…`, G2 preflight `26f9e45d…`, canonical execution freeze `91d4ce26…` (file `be6e1058…`), prospective contract canonical `f4f92a60…` (envelope `34031e8c…`). The live census/plan/custody binding recomputed and matched the frozen contract; the frozen 18-group identity plan was reproduced exactly.
- Cache transfer: 36 files, 15,525,540 bytes. The local sha256 list equals the remote sha256 list for all 36 files, and a remote re-checksum after transfer is byte-identical to the M11 `cache-full-before.sha256` record. Cache digest-of-digests `ae7aa42d7661d95d9d292400f53f4ec76128fd9618acc75cc6c6fcf581aa02b3`.
- Admission: `ADMITTED`, required 24,840 rows, available 24,840, missing 0, 18 groups. No row was captured, added or substituted; there is no recapture fallback in the code path.
- **Retained exposure** is recomputed per comparison with the accepted `readiness.retained_exposure` construction (retained label-blind capped-`n` matched files, manifest end-times, NPY-header durations; 8,280 header reads, no array element and no label read). Observed transition range `0.8520186854–0.9953066698` and control range `0.8852327448–0.9921450446`, which equal the accepted M10 census ranges to `1e-12`; all 48 transitions and all 12 controls are at or above the `0.8` floor. The M9 audit field `exposure_ratio_after` is the **raw pre-balancing arm ratio** (its transitions span `0.4393–0.7312`, controls `0.9358–0.9971`); it is recorded as `exposure_ratio_after_inherited_raw`, annotated, and never used as retained exposure.
- Forwards: `forwards_executed_this_run = 0`, `model_loads = 0`. Cumulative observed forwards stay **77,653** (52,789 historical + 24,864 from the M11 capture). The M11 product files, results and cache sidecars were not modified.
- Reference fits: exactly 18 (one per history × encoder seed, never per pair), `n_reference = 220` causal files each, bandwidth `10.7408–12.1219` (median `11.7546`), `KMeans(random_state=2407, n_init=10)` for `K ∈ {4, 8, 16}` fitted on the reference pool only. Evaluation periods and every label/family/severity/context field stayed outside the fit.

## 3. A — matched-size E2

Per history × horizon cell the per-arm size is `s = min over the five retained comparisons of min(n_baseline, n_contrast)`, and every comparison in the cell is evaluated at exactly `s` files per arm. Because the retained arms are already balanced within each pair, this rule equalises the *transitions and their control*, which is the comparison the sample-size question is about.

| Cell | `s` | Transition full-arm `n` | Control full-arm `n` |
|---|---|---|---|
| 01 week | 55 | 57, 56, 55, 63 | 75 |
| 02 week | 58 | 66, 58, 69, 64 | 86 |
| 03 week | 57 | 57, 59, 61, 63 | 81 |
| 04 week | 52 | 64, 57, 55, 52 | 85 |
| 05 week | 58 | 58, 61, 59, 62 | 76 |
| 06 week | 58 | 61, 65, 58, 64 | 76 |
| 01 month | 152 | 156, 154, 152, 154 | 287 |
| 02 month | 153 | 166, 167, 157, 153 | 289 |
| 03 month | 152 | 161, 163, 152, 156 | 270 |
| 04 month | 146 | 165, 165, 146, 152 | 278 |
| 05 month | 155 | 166, 163, 158, 155 | 284 |
| 06 month | 150 | 161, 158, 150, 157 | 273 |

`R = 1000` repeats per comparison; each repeat draws `s` file identities per arm **without replacement within the draw**, the same draw set is reused for every `K` and encoder seed, and the draws are derived from `sha256("S24-7-M12|…|repeat|{design}|{arm}")` under PCG64. The label-blind robot/program design was feasible in all 60 comparisons. The inherited `CAPPED-N-PER-STRATUM` balancing already equalised the robot/program mix, so the arm-to-arm **context mix TV is exactly 0.0 in all 60 comparisons**; stratification therefore acts as a variance control, not as a composition correction.

### 3.1 Matched-size levels (K=8, seed median of the three encoder seeds)

| Cell | `s` | Control | A→EA | A→EB | B→EA | B→EB |
|---|---|---|---|---|---|---|
| 01 week | 55 | 0.1636 | 0.1273 | 0.2000 | 0.1818 | 0.1273 |
| 02 week | 58 | 0.1897 | 0.1724 | 0.1552 | 0.2759 | 0.2759 |
| 03 week | 57 | 0.1579 | 0.1053 | 0.1754 | 0.1930 | 0.2105 |
| 04 week | 52 | 0.1923 | 0.2308 | 0.1731 | 0.1538 | 0.2500 |
| 05 week | 58 | 0.1724 | 0.2069 | 0.1379 | 0.1724 | 0.1552 |
| 06 week | 58 | 0.1552 | 0.1207 | 0.1379 | 0.1379 | 0.2241 |
| 01 month | 152 | 0.1184 | 0.1447 | 0.0921 | 0.1316 | 0.1184 |
| 02 month | 153 | 0.1111 | 0.1046 | 0.1111 | 0.1373 | 0.1242 |
| 03 month | 152 | 0.0987 | 0.1184 | 0.0855 | 0.1184 | 0.1053 |
| 04 month | 146 | 0.1164 | 0.1164 | 0.1233 | 0.1096 | 0.1644 |
| 05 month | 155 | 0.1161 | 0.1032 | 0.1161 | 0.0903 | 0.0903 |
| 06 month | 150 | 0.1000 | 0.1000 | 0.0867 | 0.1067 | 0.0933 |

Monthly cells label the transition arms `M4`/`M7` instead of `EA`/`EB`; the column meaning is the same (baseline A or B against the first or second elevated period group).

### 3.2 Transition-minus-control contrasts, all four published

The contrast is defined per repeat, `delta_r = E2_transition,r − E2_control,r`, on exact integer quanta, and summarised across repeats. Because both members are integers, a mathematically equal pair contributes exactly zero; the table reports the median delta, the fraction of repeats strictly above zero and the fraction exactly zero (seed medians of the per-seed values). The median of the per-repeat difference is **not** the difference of the two level medians in §3.1, and the delta distribution is the reported contrast.

| Cell | `s` | A→EA delta (pos / zero) | A→EB | B→EA | B→EB | Δ>0 / =0 / <0 |
|---|---|---|---|---|---|---|
| 01 week | 55 | +0.0000 (0.313 / 0.246) | +0.0364 (0.841 / 0.133) | +0.0182 (0.595 / 0.122) | −0.0364 (0.055 / 0.122) | 2 / 1 / 1 |
| 02 week | 58 | −0.0345 (0.047 / 0.093) | +0.0000 (0.310 / 0.157) | +0.0862 (0.986 / 0.013) | +0.0862 (0.986 / 0.013) | 2 / 1 / 1 |
| 03 week | 57 | −0.0877 (0.000 / 0.008) | −0.0351 (0.081 / 0.135) | +0.0351 (0.851 / 0.095) | +0.0526 (0.966 / 0.027) | 2 / 0 / 2 |
| 04 week | 52 | +0.0385 (0.750 / 0.146) | +0.0000 (0.440 / 0.199) | −0.0192 (0.242 / 0.112) | +0.0385 (0.790 / 0.111) | 2 / 1 / 1 |
| 05 week | 58 | +0.0345 (1.000 / 0.000) | −0.0172 (0.000 / 0.000) | +0.0172 (1.000 / 0.000) | −0.0172 (0.000 / 0.000) | 2 / 0 / 2 |
| 06 week | 58 | −0.0172 (0.161 / 0.264) | −0.0172 (0.161 / 0.261) | −0.0172 (0.177 / 0.214) | +0.1034 (1.000 / 0.000) | 1 / 0 / 3 |
| 01 month | 152 | +0.0526 (0.945 / 0.021) | −0.0263 (0.128 / 0.028) | +0.0132 (0.655 / 0.083) | +0.0066 (0.564 / 0.082) | 3 / 0 / 1 |
| 02 month | 153 | −0.0065 (0.397 / 0.061) | +0.0327 (0.847 / 0.047) | +0.0229 (0.761 / 0.066) | +0.0065 (0.525 / 0.051) | 3 / 0 / 1 |
| 03 month | 152 | +0.0066 (0.540 / 0.090) | −0.0197 (0.179 / 0.080) | +0.0197 (0.769 / 0.067) | +0.0132 (0.623 / 0.078) | 3 / 0 / 1 |
| 04 month | 146 | −0.0034 (0.425 / 0.075) | −0.0205 (0.223 / 0.058) | −0.0137 (0.325 / 0.084) | +0.0479 (0.935 / 0.030) | 1 / 0 / 3 |
| 05 month | 155 | −0.0065 (0.346 / 0.100) | +0.0065 (0.578 / 0.074) | −0.0065 (0.356 / 0.105) | −0.0129 (0.248 / 0.089) | 1 / 0 / 3 |
| 06 month | 150 | +0.0000 (0.446 / 0.087) | −0.0333 (0.065 / 0.065) | +0.0000 (0.480 / 0.074) | −0.0133 (0.265 / 0.087) | 0 / 2 / 2 |

Direction stability at `K = 8`: **22 of 48** transitions have a positive median delta, **5 are exactly zero** and **21 are negative**; at cell level 6 of 12 cells have a positive four-transition median, 2 are exactly zero and 4 are negative. Three transitions exceed zero in every one of the 1000 repeats and three stay at zero in at least 99% of repeats; the rest have mixed repeat-level direction (positive fraction 0.047–0.986). Across the whole repeat matrix (`48 transitions × 3 K × 3 seeds × 1000`) **35,773 deltas are exactly zero** (8.3%). The sign is not shared by the four transitions of a cell in any of the six weekly cells, so the four contrasts cannot be replaced by their aggregate.

### 3.3 The sample-size effect, stated descriptively

| Cell | Control `n` | `s` | Control full-arm E2 | Control matched E2 | Observed shift |
|---|---|---|---|---|---|
| 01 week | 75 | 55 | 0.1067 | 0.1636 | +0.0570 |
| 02 week | 86 | 58 | 0.1977 | 0.1897 | −0.0080 |
| 03 week | 81 | 57 | 0.1235 | 0.1579 | +0.0344 |
| 04 week | 85 | 52 | 0.1294 | 0.1923 | +0.0629 |
| 05 week | 76 | 58 | 0.1447 | 0.1724 | +0.0277 |
| 06 week | 76 | 58 | 0.1316 | 0.1552 | +0.0236 |
| 01 month | 287 | 152 | 0.0801 | 0.1184 | +0.0383 |
| 02 month | 289 | 153 | 0.0796 | 0.1111 | +0.0315 |
| 03 month | 270 | 152 | 0.0852 | 0.0987 | +0.0135 |
| 04 month | 278 | 146 | 0.0971 | 0.1164 | +0.0193 |
| 05 month | 284 | 155 | 0.0915 | 0.1161 | +0.0246 |
| 06 month | 273 | 150 | 0.0842 | 0.1000 | +0.0158 |

The control arm is the only arm whose sample size changes materially (75–86 → 52–58 weekly; 270–289 → 146–155 monthly). Its measured E2 is higher at the matched size in 11 of 12 cells (median shift `+0.026`). That direction is consistent with the finite-sample behaviour of a total-variation distance between two empirical occupancy vectors, which tends to shrink as `n` grows, but **the comparison is not a controlled experiment**: the full-arm and matched-size values also differ in conditioning (no stratification versus stratified draws), the support is fixed, and other explanations are not excluded. What the audit establishes descriptively is narrower and still useful: a transition-versus-control contrast measured on the full retained arms compares arms of materially different size, and its magnitude and sign change substantially when the sizes are equalised — positive in **12 of 12** cells at full arm (41 of 48 transitions, four-transition median `+0.0016` to `+0.0699`) versus **6 of 12** cells at matched size (22 of 48 transitions positive, 5 exactly zero, 21 negative).

### 3.4 Estimator resolution

Equal-`n` E2 is exactly `m/s` for integer `m`, verified for every stored value to `≤ 1e-12`. The resolution is therefore `1/s`: `0.0172–0.0192` weekly and `0.0065–0.0068` monthly. Differences smaller than one quantum are not resolvable, and the exact-zero entries in §3.2 are that quantization, not evidence that the underlying distributions coincide.

## 4. B — baseline variability

B1 is the observed cross-period baseline control at the A size (the retained `BASELINE-A` period group versus the retained `BASELINE-B` period group: three weekly or monthly low-burden periods each, with their real low-burden faults retained — zero-fault purity is never required). It is the same comparison as the A control (same arms, same draws, same size) and is reported once under both labels.

- B1 at `s`: weekly cell medians `0.1552–0.1923` (repeat `q05` `0.1053–0.1724`, `q95` `0.1897–0.2500`); monthly cell medians `0.0987–0.1184` (`q05` `0.0526–0.0900`, `q95` `0.1382–0.1742`).
- Baseline–baseline E2 is **not** near zero: two disjoint real low-burden period groups differ by a TV comparable to the transition values, which sets the scale any movement statement has to be read against.

B2 is the size-matched decomposition. With `s_B = min(floor(n_A/2), floor(n_B/2))` (`37–43` weekly, `135–144` monthly), the cross-period pair and two within-period disjoint halves are all measured at the same `s_B` with uniform draws, so the cross-versus-within comparison is not confounded by `n`.

| Cell | `s_B` | Cross-period | Within split A | Within split B | Difference vs A | Difference vs B |
|---|---|---|---|---|---|---|
| 01 week | 37 | 0.1892 | 0.2162 | 0.2162 | −0.0270 | −0.0270 |
| 02 week | 43 | 0.2326 | 0.2093 | 0.2326 | +0.0233 | 0.0000 |
| 03 week | 40 | 0.2000 | 0.2250 | 0.2250 | −0.0250 | −0.0250 |
| 04 week | 42 | 0.1905 | 0.2143 | 0.2143 | −0.0238 | −0.0238 |
| 05 week | 38 | 0.2105 | 0.2368 | 0.2368 | −0.0263 | −0.0263 |
| 06 week | 38 | 0.2105 | 0.2368 | 0.2368 | −0.0263 | −0.0263 |
| 01 month | 143 | 0.1189 | 0.1189 | 0.1189 | 0.0000 | 0.0000 |
| 02 month | 144 | 0.1181 | 0.1181 | 0.1181 | 0.0000 | 0.0000 |
| 03 month | 135 | 0.1185 | 0.1259 | 0.1259 | −0.0074 | −0.0074 |
| 04 month | 139 | 0.1295 | 0.1223 | 0.1151 | +0.0072 | +0.0144 |
| 05 month | 142 | 0.1197 | 0.1197 | 0.1197 | 0.0000 | 0.0000 |
| 06 month | 136 | 0.1176 | 0.1250 | 0.1250 | −0.0074 | −0.0074 |

The observed cross-period baseline difference exceeds the within-period split value in only **2 of 12** cells (02 week `+0.0233`, 04 month `+0.0072`); in the other ten it is equal to or below it, and in five monthly cells it is within about one estimator quantum. A within-period split is a resampling scale and cannot establish temporal stability, and a difference at or below that scale is **not** evidence that no drift exists — the baseline groups differ in many ways that this estimator at this size cannot resolve. What the numbers bound is only the detectable scale: no cross-period baseline difference larger than the within-period resampling variability was observed at `s_B`. Cross-comparison reuse is explicit: the control pair supplies arms for B1, B2 and four of the five A comparisons in its cell, and the baseline groups recur across transitions.

## 5. C — K and encoder-seed robustness

The identical support and sampling design was run for `K ∈ {4, 8, 16}` and all three encoder seeds. Every history × horizon × K × seed cell is published; `K = 8` is the inherited primary reporting level and `K ∈ {4, 16}` the inherited sensitivity, with no best-K or best-seed selection.

| Horizon | `K` | Matched E2 transitions, median [range] | Matched E2 controls, median [range] | Positive / zero / negative transition medians |
|---|---|---|---|---|
| week | 4 | 0.1180 [0.0517, 0.2000] | 0.1304 [0.0690, 0.1724] | 7 / 5 / 12 |
| week | 8 | 0.1727 [0.1053, 0.2759] | 0.1680 [0.1552, 0.1923] | 11 / 3 / 10 |
| week | 16 | 0.2609 [0.1552, 0.3455] | 0.2521 [0.2414, 0.3077] | 8 / 5 / 11 |
| month | 4 | 0.0771 [0.0458, 0.1176] | 0.0658 [0.0548, 0.0710] | 15 / 1 / 8 |
| month | 8 | 0.1104 [0.0855, 0.1644] | 0.1136 [0.0987, 0.1184] | 11 / 2 / 11 |
| month | 16 | 0.1572 [0.1226, 0.1986] | 0.1623 [0.1447, 0.1712] | 6 / 1 / 17 |

- The aggregated E2 levels increase with `K` in both horizons and both comparison types **in these particular fits**. Independently fitted `KMeans` partitions with different `K` are not nested, so this ordering is an observed property of the fitted prototypes, not a guaranteed consequence of increasing `K`.
- The direction of the contrast is not stable across `K`: positive counts move `7 → 11 → 8` weekly and `15 → 11 → 6` monthly for `K = 4 → 8 → 16`, and zero counts appear at every `K`. No `K` is preferred.
- Encoder seeds are run replicates and they do **not** agree on the sign: only 17 of 48 transitions carry the same non-zero sign under all three seeds at `K = 8`, and **no** history × horizon × `K` row (0 of 36) shows all-seed sign agreement. Agreement is a three-way comparison (negative, zero, positive): a unanimous-zero triple is reported through `n_all_seeds_zero` and never counted as directional agreement. On this support no transition has a unanimous-zero seed triple (`n_all_seeds_zero = 0` at every `K`), and the observed sign sets include fully mixed cases such as `{−1, 0, +1}`. The seed-median matched levels spread by a median of `0.0545` (maximum `0.2182`), the same order as the level differences under discussion. No seed is selected, and seeds are never treated as independent evidence.
- Weekly and monthly numbers are reported separately and are never pooled; the per-cell `s` differs by horizon (`52–58` weekly versus `146–155` monthly) and so does the quantum (`1/s`), so cross-horizon level comparisons are not meaningful.

## 6. D — post-hoc diagnostic decomposition

Metadata is joined by immutable file identity (`file_id` and `waveform_sha256`) only, on the retained arms. Ordinary files are carried as their own category. All nine realized families plus the ordinary category appear somewhere across the 60 contrast arms, but a single contrast arm carries a median of 9 of those 10 categories (range 3–10), so family coverage is not uniform. Labels were used **only** here: they never entered a prototype fit, a sampling draw, the sample-size rule or any A/B selection.

Denominators: across the 60 comparisons the baseline arms carry a median of 116 retained files with a median of 5 abnormal files (range 0–16), and the contrast arms a median of 116 files with a median of 17 abnormal files (range 3–44).

| Horizon | Comparison | Family mix TV | Severity-bin mix TV | Anomalous-fraction-bin mix TV | Context mix TV |
|---|---|---|---|---|---|
| week | transition | 0.2284 [0.1538, 0.3220] | 0.4333 [0.0625, 0.8462] | 0.4375 [0.1333, 0.8333] | 0.0000 |
| week | control | 0.0397 [0.0235, 0.0494] | 0.4333 [0.1667, 1.0000] | 0.3667 [0.1500, 0.6667] | 0.0000 |
| month | transition | 0.2182 [0.1849, 0.2393] | 0.2180 [0.0556, 0.4762] | 0.2050 [0.0379, 0.5078] | 0.0000 |
| month | control | 0.0274 [0.0141, 0.0346] | 0.1834 [0.0769, 0.3000] | 0.1963 [0.0769, 0.4945] | 0.0000 |

- One severity-bin and one anomalous-fraction-bin distance are **undefined** (`null`) in weekly transitions because one arm contains no abnormal file at all; an undefined distance is never written as zero.
- Sparse / one-sided / confounded disclosure: 59 of 60 comparisons contain at least one family present in only one arm, all 60 contain at least one family with fewer than five files in an arm, and 55 contain at least one severity or anomalous-fraction bin whose members come from a single family (220 such bins in total). Every family is one-sided in 19–36 of the 60 comparisons. These strata are listed, not dropped, and they are the reason no per-family E2 contrast is published as a movement estimate.
- Non-additivity of occupancy TV at `K = 8` over 180 comparison × seed decompositions: `TV_total` median `0.1234`, per-family weighted sum median `0.2329`, gap median `0.1099` (range `0.0185–0.2188`). The additive sum exceeds the total by a median factor of `1.78`, exactly the convexity overshoot `Σ_f w_f TV(p, q_f) ≥ TV(p, Σ_f w_f q_f)`; occupancy TV therefore cannot be partitioned into additive family contributions, and the gap is reported as a measured quantity.
- Descriptive association between the matched-size `K = 8` seed-median E2 and composition features, per horizon and across the retained transitions (dependent, shared arms; no p-value, no gate): family mix TV `rho = −0.195` (week) and `−0.187` (month); severity mix TV `−0.012` and `−0.208`; anomalous-fraction mix TV `−0.066` and `+0.004`; burden contrast gap `−0.250` and `−0.227`; context mix TV is degenerate (zero variance) so no association is defined. The association block reports its own denominator (24 transitions per horizon, 24 used, 0 excluded at `K = 8`); a transition whose stratified design is unavailable would be listed with its reason under `excluded_transitions` instead of being indexed. All magnitudes are small, so no composition feature is described as an explanation of the variation in E2, and no causal attribution is made.

## 7. Independent verification

`artifacts/sprint-24/evidence/s24-7-m12-a01/independent-verification.py` (kept with the evidence it produced) implements a separate code path: its own `KMeans` fit, occupancy by `numpy.unique`, TV by `1 − Σ min(p, q)`, an independent counting routine for the quanta and an independently rebuilt retained-exposure map. All nine checks pass:

| Check | Result |
|---|---|
| TV identity (module half-L1 vs independent `1 − Σ min`) | 300 real-occupancy samples, max difference `1.67e-16` |
| Equal-`n` sampling (exact size, no replacement, inside its own arm, arms size-matched) | 3,000 draws, 0 failures |
| Disjointness (retained arms; within-period split halves) | 60 pairs + 24 splits, 0 failures |
| Repeatability (two full runs) | 8 statistical tables byte-identical, repeat arrays identical, `lineage.json` identical ignoring recorded wall-clock fit durations |
| Exact E2 (independent prototypes + counting + TV vs stored integer quanta) | 5,400 values, 0 mismatching keys |
| Direction ties (exact zeros counted exactly; stored medians and fractions re-derived) | 432 transition × K × seed cells, 35,773 exact-zero deltas, 0 failures |
| Retained exposure (independently rebuilt vs stored and vs the accepted census) | 60 comparisons, 8,280 header reads, ranges equal to `1e-12` |
| Accounting | planned = computed + abstained for every table; 0 abstained on this support |
| Cache bytes | local = recorded = retrieved-remote digests, 36 files |

Additional checks: the full-arm recomputation matches the published M11 values exactly (540/540, max difference `0.0`); the focused M12 tests pass (24/24); the pre-existing M11 focused tests still pass (13/13) with no M11 file changed. The integrated-path checks exercise the unsupported-comparison branch through the same functions the audit uses, so an infeasible stratified allocation yields explicit abstention records and an excluded-transition entry rather than a lookup error.

## 8. Accounting: every planned cell accounted

| Section | Planned | Computed | Abstained |
|---|---|---|---|
| A, full arm | 540 | 540 | 0 |
| A, stratified | 540 | 540 | 0 |
| A, unstratified | 540 | 540 | 0 |
| B1 (A-size cross-period control) | 108 | 108 | 0 |
| B2 (cross + two within-period splits) | 324 | 324 | 0 |
| D, occupancy-TV decompositions | 540 | 540 | 0 |

- Cells: 12 total, 12 `COMPUTED` (`s ≥ 52`, above the inherited 30-file period floor); the same 30-file floor applies to `s_B` and never binds (`s_B ≥ 37`).
- Comparisons: 60 total, 48 transitions and 12 controls; all resolve.
- The engine records `ABSTAIN-SHORTFALL-SIZE` and `ABSTAIN-NO-COMMON-STRATA` per comparison, per design and per cell, keeps abstentions in the denominator, and reports a reason tally. On this support the tally is empty; the paths are exercised by focused tests (sub-floor cell, infeasible allocation, missing design) rather than by changing the real corpus.
- `all_planned_cells_accounted = true`, with computed + abstained equal to planned for every row of the table above. The abstention branch is implemented inside the same assembly the audit runs (not only in helper code): an unavailable design produces an explicit record with no repeats, an unsupported comparison is excluded from the descriptive association with a reason under `excluded_transitions`, and the focused tests drive that path directly.

## 9. What this does and does not establish

- Descriptively, on this fixed cache: the full-arm transition-versus-control contrast compares arms of materially different size (control 75–86 weekly and 270–289 monthly versus transition 52–63 and 146–155), and equalising the size changes both its magnitude and its sign pattern (control E2 rises in 11 of 12 cells by up to `+0.0629`; the contrast is positive in 12/12 cells at full arm and 6/12 at matched size). Under this design the matched-size endpoint shows **no consistent separation** of elevated periods from their low-burden controls. That is a bounded statement about this cache, this estimator and this support; it does not show that no difference exists, and it does not attribute the full-arm/matched-size difference to sample size alone, because the two designs also differ in conditioning.
- Occupancy TV is not additive over anomaly-family subsets, and the diagnostic strata are sparse, one-sided and confounded with family, so family-level movement statements are not supported by this cache.
- It is **exploratory and descriptive**: the author had read the published M11 values before freezing the design, so no M12 quantity is a blind replication and nothing here is independent confirmation of M11.
- Repeats are Monte Carlo integration of the sampling distribution of one fixed finite support; their quantiles are resampling variability, never confidence limits. Encoder seeds are run replicates. Comparisons sharing an arm are dependent, and arm reuse across comparisons is disclosed rather than corrected.
- Estimator resolution is `1/s`, coarse for the weekly cells (`0.0172–0.0192`); many reported differences are one quantum and 8.3% of transition deltas are exactly zero.
- No p-value, significance test, confidence interval, success gate, performance, detection-accuracy, business-density, training/refit, Confirmation, Phase C or causal claim is produced. No data was generated, no encoder inference ran, no bank or prototype was refit on evaluation data, and no M11 product, result or cache byte was changed.

## 10. Reproduction

From the repository root with the repository environment (cache already retrieved; the CLI refuses a non-empty `--out`):

```bash
python experiments/sprint24_task7m12_e2_cache_diagnostic.py \
  --corpus data/generated/sprint24_controlled_v2 \
  --audit artifacts/sprint-24/evidence/s24-7-m9-a01/corpus-audit.json \
  --contract artifacts/sprint-24/evidence/s24-7-m10-a02/prospective-contract.json \
  --cache-root artifacts/sprint-24/evidence/s24-7-m12-a01/cache-full \
  --out artifacts/sprint-24/evidence/s24-7-m12-a01/audit-FRESH
python artifacts/sprint-24/evidence/s24-7-m12-a01/independent-verification.py
python -m pytest tests/representation/test_sprint24_e2_cache_diagnostic.py -q
```

Retrieval of the cache to the separate local location (read-only, no remote edit):

```bash
ssh trietlm@192.168.99.214 'cd /tmp/s24-7-m11-a03-run/cache-full && sha256sum *' > remote-cache-sha256.txt
scp -r trietlm@192.168.99.214:/tmp/s24-7-m11-a03-run/cache-full artifacts/sprint-24/evidence/s24-7-m12-a01/
```

## 11. Runtime, storage and cost

- Audit wall time ~45–60 s per full run (`repeats = 1000`, 18 fits, 60 comparisons, 2 designs, 8,280 NPY-header reads for the exposure maps); independent verification ~31 s; focused tests ~5 s.
- Evidence: 9 JSON tables + `repeats.npz` (int64 occupancy quanta, 5 stores) per run, two runs kept (`audit/`, `audit-repro/`), the retrieved cache (36 files, 15.5 MB) and the retrieval digest lists. All under `artifacts/`, which is excluded from Git; the bulk cache is never committed.
- Forwards: **0 new**; cumulative observed cost remains 77,653. No GPU, no model load, no network compute.

## 12. Specification amendments and R2 corrections

The specification's `amendments` block records each change with its reason and separates three histories:

- **Descriptive repairs** (`A01` denominators, `A02` three declared designs, `A04` seed-key wording) — no rule and no number changes.
- **A real execution deviation** (`A03`): the pre-outcome specification said B2 abstains when `s_B` is below the inherited 30-file floor; the first two executed runs used an implementation constant of `10`, and it was set to the pre-registered `30` after the first complete run. That is a **post-outcome change to the executed code**, disclosed as such. It changed no reported number (`s_B` is 37–144 everywhere, so both constants give `COMPUTED` with identical values, and the regenerated statistical tables are byte-identical), but it is not presented as pre-registered. The exact pre-amendment bytes of the specification are not recoverable from disk (untracked file, no version-control copy); the pre-outcome wording is preserved in that amendment entry and in the recorded assessment history.
- **Post-assessment R2 corrections** (`A05`): exact occupancy quanta, retained-exposure reporting, and the completed abstention accounting. **R3 corrections** (`A06`): the integrated assembly and association path now abstain on an unsupported matching design and exclude it explicitly, and seed direction is three-way with `n_all_seeds_zero`. These follow the reviewer's R1 findings and are labelled as post-assessment, not pre-registered. The sampling design, the per-arm size rule, the estimator definition, the baseline construction, the diagnostic bins and the summary structure are unchanged from the pre-outcome specification and from R1.

Superseded results: R1 reported 24 of 48 positive transitions (float ties counted as positive) and a comparison entry that carried the raw `exposure_ratio_after` without the retained exposure; R2 still raised on an infeasible stratified allocation in the integrated association path and treated a zero as a negative for seed agreement. The numbers reported in §2 and §3 are the R3 values.

## 13. Assessment history

- `S24-7-M12-C1/R2` — reviewer `S24M12EvidenceA01` returned `changes_requested` (`advance: false`) on the candidate report `2383e0e5…`, accepting the R1 corrections as materially resolved and raising two bounded findings: the integrated association path still indexed a per-seed record that is absent for an unavailable stratified design (and no test drove the integrated path), and `direction_agreement_all_seeds` conflated zero with negative via a boolean test. Both are corrected in R3, and the R2 report remains in the record as the assessed, superseded candidate.
- `S24-7-M12-C1/R1` — reviewer `S24M12EvidenceA01` returned `changes_requested` (`advance: false`) on the candidate report `27928346…`, with five MEDIUM findings: (1) float direction ties, (2) over-claiming causal/attribution language in §3.2/§4/§9, (3) an incomplete abstention path and missing low-support accounting, (4) an insufficiently precise amendment disclosure around the B2 floor, and (5) retained exposure copied from the M9 raw ratio instead of recomputed. All five are addressed in this R2 candidate; the R1 report and its tables remain in the record as the assessed, superseded candidate.
