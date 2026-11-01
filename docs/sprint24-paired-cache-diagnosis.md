# Sprint 24 paired-cache diagnosis (Task 7-M19)

**Status: exploratory, read-only diagnosis of the accepted S24-7-M18-A04 C03 cache. No new encoder forwards, model loads, generation, training/refit or Phase C work were performed. The accepted M18 results and contracts are unchanged. Every statistic below is descriptive/exploratory: nothing here is a detection, absence, equivalence, calibrated-CI, causal, business-scale or sequential-look claim. Three M18 wording/attribution errata are documented in §9; the accepted report is not edited.**

## 1. Scope, inputs, custody and zero-forward execution

- **Cache**: `/tmp/s24-7-m18-a04-run/work/official` on `trietlm@192.168.99.214` (`di-server`), produced by attempt `S24-7-M18-A04` at reviewed commit `1788b6efa2f9bb2ae6fd164a71c7f20604302e9e`. The isolated runtime checkout `/tmp/s24-7-m18-a04-run/repo` (still at that commit), the earlier forensic root `/tmp/s24-7-m18-a02-run` and the dirty main server checkout were not modified.
- **Input hashes verified before analysis** (`evidence/s24-7-m19-a01/out/custody.json`, 25/25 checks): `official-run.json` `83a5e3a0…` (59,787,284 B), `replay.json` `ddde50a4…` (1,829,980 B), `run.json` `6f247605…` (364 B), `capture/events.jsonl` `8ae5080c…` (363,187,051 B). All 756 capture batches match their per-event SHA-256 and the journal-order digest `0af2adbb…` (223,010,304 B). Reviewed sources match `working_tree_source_sha256` (C03 contract `75e123f3…`, `sprint24_paired_official.py` `f5847664…`, estimators `cf0f48b4…`, disjoint `c5f03c5b…`, CLI `a727dae5…`).
- **Zero new work per process**: every audit script installs a meta-path guard that raises on `torch`, `representation` or `synth` imports. Recorded ledger for all passes: 0 encoder forwards, 0 model loads, 0 checkpoint loads, `torch_imported: false`. No official threshold was retuned; alternative thresholds appear only as diagnostics and are labelled as such.
- **Units of analysis**: one scored group = K = 8 paired sources (7,710 groups); one cell = history × family × severity (162 cells); one encoder = one of the three frozen seeds; one Stage-2 cluster = history (6). No patch/file pseudo-N is used. The audit reproduces the retained per-group `e1_per_seed` from the stored vectors with maximum absolute difference **5.1e-16** (0 failures at 1e-9).

## 2. What the accepted cache does and does not retain

Retained and used here: per-file pooled 128-d embeddings for every captured file, per-file patch metadata (`n_patches`, `n_valid_patches`, `n_masked_targets`, `valid_len`), per-pair source/group/attempt/strength/length/window lineage, per-group E1 for all three encoders, per-history/encoder bandwidth `sigma` and the four drawn null scores behind each q95, the within-cell source-disjoint permutation results, the 220-row reference and bandwidth pools per history, and all Stage-2 transition/control scores with mixture/control ids.

Not retained (bounded limitations, no re-capture): raw waveforms; per-patch label or mask *indices* (only counts/lengths are retained); benign/no-injection matched twins; any true business calendar (the retained time bases are planned synthetic scenario times). Consequently, waveform-level, mask-level and business-time diagnostics are unsupported and are not inferred. `strength` is the generator's measured injection strength, not the M18 severity ladder; severity is a design field.

## 3. Signal versus null/control and threshold comparability

### 3.1 Score distributions

Group E1 (the official paired U-statistic, matched cross diagonal and within-arm diagonals removed) is centred near zero for almost the whole matrix:

| Quantity | All 7,710 groups | Median across encoders |
|---|---:|---:|
| Mean E1 | +0.000265 | per-encoder means 0.000263 / 0.000310 / 0.000312 |
| SD E1 | 0.00215 | per-encoder SD 0.00230 / 0.00224 / 0.00255 |
| Negative groups | 3,491 (45.3%) | 3,764 / 3,481 / 3,479 |
| Min / max | −0.0131 / +0.0321 | max per encoder +0.0275 / +0.0321 / +0.0429 |

- `duration_anomaly` groups: mean 0.00176, SD 0.00610, median 0.00113 (n = 864).
- All other eight families pooled: mean 0.000077, SD 0.00045, median 0.000022 (n = 6,846).
- 103 of 162 cell medians are positive (range −0.000385 to +0.002613). No cell median approaches any null threshold estimate in §3.3.

For comparison, the pooled unrelated normal-source null (same estimator, same K, same frozen bandwidth; see §3.2–3.3) has mean 4.2e-5, SD 0.02549, q95 0.0458, q99 0.0720, and 54.4% negative values; the reference-pool null is nearly identical (mean 0.00039, SD 0.02591, q95 0.0466). Even the most extreme retained group (median E1 0.0321, one encoder 0.0429) lies below those null q95 values. Negative U values are expected for this unbiased estimator and are not an error; they account for roughly half of all groups.

### 3.2 What the official threshold actually is

The official exploratory threshold is the q95 of **four** twin-vs-twin group comparisons drawn from the 64-file null pool of each history (indices `[604,668)`), at the frozen bandwidth and with the paired estimator. Across the 18 history × encoder combinations it ranges **−0.00146 to 0.09933** (median 0.0260): one threshold is negative, so any positive E1 there "responds". The retained null scores and q95 recompute to 3.8e-16 / 2.8e-16 of the stored values.

### 3.3 Alternative null constructions (same K, kernel, encoders, estimator)

Five constructions were compared. **None adds independent normal sources**: all reuse the same finite retained files (8 disjoint groups per history/encoder in the 64-file null pool; 27 disjoint groups in the 216-of-220-file reference pool). The 2,000 null-pool split resamples, the 1,000 reference-pool split resamples and the 28/351 group pairings are therefore **correlated resamples of the same files, not new independent null groups**, and none of them is a calibrated ground truth.

| Construction | Range of q95 (18 history × encoder) | Median |
|---|---:|---:|
| Official 4-draw q95 (64-file null pool) | −0.0015 … 0.0993 | 0.0260 |
| Null pool, 4 disjoint group pairings | −0.0068 … 0.0612 | 0.0337 |
| Null pool, all 28 disjoint group pairings | 0.0164 … 0.1050 | 0.0426 |
| Reference pool, 13 disjoint group pairings | 0.0162 … 0.0887 | 0.0310 |
| Reference pool, 351 overlapping pairings | 0.0239 … 0.0916 | 0.0428 |

- The official 4-draw q95 lies **below** the 2,000-split re-estimate in 15 of 18 history × encoder cases, at percentile 0.52–0.998 of that distribution, with P(null score > official q95) between **0.001 and 0.472** across cases.
- The spread of q95 values across constructions (0.016–0.105) is about twice the full observed group-E1 range (−0.0131 to +0.0321). The instability is a property of having at most 8/27 disjoint normal groups per history, not of the split algorithm.
- Kernel-scale alternatives are also diagnostic only: replacing the bandwidth-pool sigma with the reference-pool sigma changes the four-draw null q95 negligibly (median 0.0261), while halving/doubling sigma gives q95 ranges −0.0027…0.1219 / −0.0007…0.0342. No official threshold was changed.

### 3.4 Threshold sensitivity on the same group values (not a signal test)

Holding the 7,710 retained group values fixed and only changing the threshold level to the §3.3 null-pool re-estimate:

- **0 of 7,710** groups exceed the median re-estimated q95; **0 of 7,710** exceed it at all three encoders. This is a statement about how the *count of exceedances* depends on the threshold, not evidence that the anomalies produce no signal, and it inherits the finite-pool dependence and non-exchangeability cautions above.
- The 33 retained "responses" are all `duration_anomaly` and all lie in the three histories whose q95 median is lowest: history 04 (q95 0.00718, 23 responses), history 02 (0.01548, 5), history 06 (0.01415, 5). Histories 01, 03 and 05 (q95 medians 0.03553, 0.08719, 0.02999) record **zero** responses even where their duration E1 medians are as high or higher (history 05 duration medians 0.00115/0.00157/0.00136 vs history 04 0.00060/−0.00039/0.00117).
- Response counts by q95 quartile: 23 (lowest quartile, 27 cells), 10 (second, 54 cells), 0, 0. Across the 18 duration cells, Spearman(response count, cell q95 median) = **−0.910**; the response groups' own median E1 is 0.0118 (range 0.0074–0.0321), i.e. below every alternative q95 in §3.3.
- Interpretation (measured sensitivity vs hypothesis): the *pattern* of the 33 exceedances is fully consistent with threshold-level variation alone. That is not a causal explanation of any individual group; see §7 for the ranked hypotheses and §3.5 for why the matched null cannot decide it.

### 3.5 Paired U versus an independent-arm null: coupling and identifiability

The same K = 8, same RBF kernel and same encoders were used to compare the official paired estimator with the independent-sample analogue (all cross terms over n², within diagonals removed):

- On **unrelated normal sources** (reference-pool pairings) the two agree: mean(paired − unpaired) = +5.3e-5 (351 overlapping pairings), +5.3e-4 (13 disjoint pairings), −2.0e-4 (null pool).
- On the **matched scored groups** they differ by **−0.0914 ± 0.0081** in kernel units (n = 7,710; 100% negative), i.e. the matched-pair coupling term is ~3.8 times the SD of the median E1, ~2–3.5 times the median alternative q95 values in §3.3, and comparable to the largest per-case q95. Matched twin/variant embeddings are far more similar than unrelated files: median matched kernel **0.9917** vs non-matched **0.6161** (duration only: 0.9065 vs 0.6132).
- Therefore an independent-arm normal null is **not exchangeable** with the paired scores: the paired estimator deliberately excludes the same-source i=j cross terms, and under a no-change null the twin and variant are the *same source*, so the relevant matched null is degenerate rather than two independent normal distributions. **No benign/no-injection twin pair was captured in this cache**, so a matched no-change null cannot be estimated without new generation/capture. The owner-approved source-disjoint permutation test is the appropriate conditional alternative because it compares twin-only vs variant-only *marginal* pools across sources.

Retained confirmatory source-disjoint test (unchanged, not recomputed here beyond reproducing its outputs): all 162 cells computed; **0 rejections after Holm** at α = 0.05 (minimum Holm-adjusted p = 0.1134); 4 cells have unadjusted p < 0.01 and 18 have p < 0.05. Those 18 are spread across families (7 duration, 3 missing_event, 2 over_regularity, 2 wrong_transition, 2 cross_channel_inconsistency, 1 freq_phase_mismatch, 1 contextual_replacement) and are descriptive only; they are not familywise evidence and do not identify a family.

### 3.6 Bandwidth, kernel scale and negative U

- Bandwidth `sigma` recomputed from the independent bandwidth pool matches the retained values exactly and ranges 10.76–12.06; bandwidth-pool vs reference-pool sigma ratio is 0.991–1.006, and Stage-1 pool sigmas are 0.991–0.995 of the reference-pool sigmas.
- Normal pools are not kernel-saturated: mean kernel 0.615, fraction k > 0.9 ≈ 0.12%, fraction k < 0.1 = 0. Scored matched pairs are near-saturated (median k 0.9917), which is the coupling issue in §3.5, not a global kernel-scale failure.
- Negative U: 45.3% of group medians and 45.1–48.8% per encoder; the null pools are 54% negative. The negative part of the distribution is intrinsic to the unbiased statistic and is not a numerical failure.

## 4. Encoder, history and duration decomposition

### 4.1 Encoders

| Encoder | Cell medians (median / range) | Groups positive | Groups negative | Median signal ÷ own q95 |
|---|---|---|---:|---:|---:|
| 181801 | +8.4e-6 / −2.9e-4 … +2.7e-3 | 51.2% | 48.8% | −0.00023 |
| 181802 | +2.1e-5 / −6.1e-4 … +3.4e-3 | 54.9% | 45.1% | +0.00065 |
| 181803 | +2.6e-5 / −6.4e-4 … +3.3e-3 | 54.9% | 45.1% | +0.00122 |

Pairwise encoder agreement across groups is high on magnitudes (Pearson 0.744–0.798) but weak on ranks/sign (Spearman 0.297–0.351; same-sign fraction 0.566–0.594), so the encoder-median summary is not three independent confirmations. Exceedances of each encoder's own official q95 are **1,221 / 79 / 47**; encoder 181801's 1,220 of 1,221 occur in history 04, whose 181801 threshold is the negative value −0.00146. That is a threshold artifact, not evidence that one encoder is more sensitive.

### 4.2 Histories

Per-history q95 medians and duration movement (all other families contribute no responses):

| History | q95 median | Duration cell medians (0.35/0.6/0.85) | Responses |
|---|---:|---|---:|
| 01 | 0.03553 | 0.00041 / 0.00063 / 0.00173 | 0 |
| 02 | 0.01548 | 0.00131 / 0.00036 / 0.00261 | 5 |
| 03 | 0.08719 | 0.00059 / 0.00113 / 0.00222 | 0 |
| 04 | 0.00718 | 0.00060 / −0.00039 / 0.00117 | 23 |
| 05 | 0.02999 | 0.00115 / 0.00157 / 0.00136 | 0 |
| 06 | 0.01415 | 0.00131 / 0.00144 / 0.00103 | 5 |

### 4.3 Within-source change versus between-source background

- Median within-source squared displacement relative to the between-source background is **0.207** for duration groups but **0.019** for all other families (10.9×). Median matched kernel 0.9065 for duration vs 0.9917 for the rest (non-matched 0.613–0.616).
- Across all families, responses are more frequent in the higher-displacement half of duration groups (5.8% vs 1.9%), but the group-level rank association between E1 and relative displacement is weak (Spearman 0.049 in duration; 0.086 in the other families).

### 4.4 Length change, patch counts and severity

- **Only `duration_anomaly` changes length**: 99.8% of its pairs have a nonzero step delta, 88.0% are longer, median |Δsteps| 91.6; the other eight families have Δsteps = 0 and Δpatches = 0 for every retained pair. Duration median step delta scales with design severity: 65.6 / 87.2 / 120.4 at 0.35 / 0.6 / 0.85 (variant longer in 88%, unchanged in 0.2%, shorter in 11.8%).
- Within duration groups, E1 increases modestly with |Δsteps|: overall Spearman **0.301** (history-cluster bootstrap over 6 histories: **0.212 … 0.387**, point 0.305); within severity 0.236 / 0.263 / 0.407. Partial rank association E1 vs |Δsteps| controlling relative displacement = **0.299**; E1 vs displacement controlling |Δsteps| = **−0.029**. Pooled across all groups the association is only 0.085.
- Duration-vs-other median E1 shift: per-history differences 0.00047–0.00147, cluster bootstrap median **0.00127 [0.00079, 0.00143]** (6 clusters, descriptive).
- Response rate rises across |Δsteps| quartiles: 1.39% / 2.78% / 4.17% / 6.94% (15/216 top-quartile; exact binomial description 3.9%–11.2%). The history-cluster description of the top-quartile rate is 2.94% [0%, 22.9%] because three histories contribute zero top-quartile responses.
- Because length change is a *design feature unique to the duration family*, this cache cannot separate "E1 tracks pooled length/patch structure" from "E1 tracks the duration anomaly semantics"; the two are perfectly confounded within this capture. The measured association is a fact; a length-shortcut mechanism is one hypothesis, not an established explanation. There is no retained non-duration family with a length perturbation to serve as a contrast.

### 4.5 Representation state and normalisation

No collapse or global kernel saturation is visible in the retained vectors:

| Encoder | Norm median (q05–q95) | Dim SD median | Dead dims (SD < 1e-3) | Participation ratio / 128 | Cos-sim median | Normal-pool mean kernel |
|---|---:|---:|---:|---:|---:|---:|
| 181801 | 8.42 (7.65–9.35) | 0.646 | 0 | 19.2 / 0.150 | 0.182 | 0.615 |
| 181802 | 8.97 (8.19–9.83) | 0.727 | 0 | 21.4 / 0.167 | 0.083 | 0.616 |
| 181803 | 8.80 (7.72–9.81) | 0.717 | 0 | 21.8 / 0.171 | 0.099 | 0.616 |

The encoders are anisotropic (effective dimension ≈ 20 of 128) but not collapsed; norms are stable; there are no dead dimensions; the normal-pool kernel is not saturated. The near-one matched kernel (§3.5) is about source identity, not representation collapse.

## 5. Full family/severity coverage, admission, retry, source reuse

- **Coverage is complete**: 162/162 cells, 9 families × 3 severities × 6 histories, no cell-wise selection. 62,208 planned pairs → 61,871 admitted (`PAIRED`) → 7,710 complete K=8 groups (61,680 grouped pairs); 191 admitted pairs did not complete a group.
- **Incomplete groups**: 36 cells have fewer than 48 groups. Their shortfall is explained by `NORMAL_RETURN` in the same cells: missing_event 300 returns, subtle_drift 22, wrong_transition 15 (total 337; zero in the other six families). Family group totals are contextual 864, cross-channel 864, duration 864, freq-phase 864, missing 816, over-regularity 864, realistic-stuck 864, subtle 858, wrong-transition 852.
- **Retry/rejection**: stage-1 paired attempts = 64,084 (61,871 accepted, 2,213 rejected; 1,740 pairs retried). Rejection reasons are `strength_above_max` 1,719 and `strength_below_min` 494. Median cell rejection rate is **7.9% for duration** vs **1.0% for all other families**; duration records 819 rejected attempts. Within duration cells, Spearman(rejection rate, response rate) = 0.30 and Spearman(strength, response rate) = −0.12, so the strong overall cross-cell association (0.35) is dominated by the family contrast, not by a within-duration retry effect.
- **No pseudo-N**: every one of the 7,710 groups contains exactly 8 distinct source ids (0 groups with a duplicate source); within-cell duplicate source ids = 0. Source reuse is across cells, not within a group.
- **Source reuse is structural**: each history reuses the same 384 planned twin sources in every one of its 27 cells (distinct twin ids per history = 384; reuse factor 26.82–26.92 including admission losses; 10,299–10,313 admitted pairs). A direct check confirms the same twin file has byte-identical embeddings across cells (max |Δ| = 0.0 for all six histories). Variants are all distinct (reuse factor 1.0). Consequently the 162 cells are not independent samples: inside a cell the units are the disjoint K=8 groups, but those same sources recur in the other 26 cells of the history, and the six histories are six dependent clusters. This matches the M18 statement that sources and calibration are shared, but the sharing is **within each history across its family × severity cells** (histories have distinct seed namespaces).
- **Reporting units**: all comparisons above use the group as the unit and are reported per cell/history; no pooled file counts are used as sample sizes.

## 6. Business population: weekly/monthly transition minus matched control

All 12 arms × 6 histories recompute exactly from retained vectors (transition max |Δ| and control max |Δ| vs stored: < 1e-15). The six-history bootstrap of the encoder-median transition reproduces the accepted values to ≤ 1e-16.

| Arm | Realized abnormal fraction | Transition − control median | Paired 6-history bootstrap | Histories +/− |
|---|---:|---:|---|---:|
| month/160/ELEVATED | 0.275 | +0.000016 | [−0.000729, +0.000855] | 3/3 |
| month/160/LOW | 0.050 | −0.000706 | [−0.001307, +0.000744] | 2/4 |
| month/160/MID | 0.150 | +0.001296 | [−0.000375, +0.001706] | 5/1 |
| month/250/ELEVATED | 0.276 | −0.000489 | [−0.000635, +0.000362] | 2/4 |
| month/250/LOW | 0.052 | −0.000195 | [−0.001191, +0.000668] | 3/3 |
| month/250/MID | 0.152 | +0.000558 | [−0.000186, +0.001652] | 5/1 |
| week/40/ELEVATED | 0.275 | +0.004876 | [−0.001705, +0.006580] | 5/1 |
| week/40/LOW | 0.050 | +0.000740 | [−0.000905, +0.006905] | 4/2 |
| week/40/MID | 0.150 | −0.000694 | [−0.003914, +0.004754] | 3/3 |
| week/60/ELEVATED | 0.283 | +0.000647 | [−0.002640, +0.002321] | 3/3 |
| week/60/LOW | 0.050 | +0.000643 | [−0.003195, +0.003105] | 4/2 |
| week/60/MID | 0.150 | +0.000489 | [−0.000977, +0.002701] | 3/3 |

- Every delta interval includes zero; the largest point contrast is week/40/ELEVATED (+0.00488) and its interval still includes zero. Per-encoder signs are near-balanced (5–15 positive of 18 depending on arm).
- Requested burdens are LOW 0.05, MID 0.15, ELEVATED 0.275; realized fractions are 0.050–0.052 / 0.150–0.152 / 0.275–0.283, the differences being integer rounding (e.g. 17/60 = 0.2833 at week/60/ELEVATED). No redraw or rebalancing occurred. All arms use `contextual_replacement` severity 0.35; none of this generalises to other families or severities.
- **Burden consistency**: no horizon/size row is monotone in burden (month/160: LOW<MID>ELEVATED; month/250: LOW>ELEVATED but MID higher; week/40: MID lowest; week/60: all similar). Size effects are also inconsistent (week/40/ELEVATED 0.00488 vs week/60/ELEVATED 0.00065; month/160/MID 0.00130 vs month/250/MID 0.00056). The arms are disjoint source slices (each arm consumes its own index band, stride 2×size), so they are **not nested** and do not form a causal burden curve; differences mix burden, sample size, source set and the shared reference.
- **Shared reference**: the transition and control scores both contain the U(R,R) term with coefficient one, so subtraction removes the reference-only term exactly; residual shared-reference dependence enters through the K(A,R)/K(O,R)/K(C,R) sample means. A 50-replicate 80% reference-row resampling per arm × history × encoder gives per-arm Corr(T,C) between −0.615 and +0.538 (150 points each; small-sample noise included) and pooled Corr = −0.037. The observed SD of deltas across arm × history rows is 0.99 of the independent-difference benchmark √(Var T + Var C), i.e. the subtraction does not hide a large common-reference variance in this cache. These are descriptive, not assumption-free inferential intervals, and 6 history clusters cannot support a business-scale claim.

## 7. Ranked explanations (facts, hypotheses, and what the cache cannot answer)

**H1 (best supported as a description of the pattern): the 33 threshold exceedances follow the level of the exploratory threshold rather than any measured excess of E1 over the retained normal references.** Measured: 15/18 official q95 values sit below a re-estimated q95 from the same finite pools, one is negative, response counts fall 23/10/0/0 across q95 quartiles, and duration cells show a −0.91 rank association between response count and threshold level. This does **not** establish that miscalibration causally explains each of the 33 groups; with a valid matched null the count could be smaller or larger, and the 33 events are not individually adjudicated.

**H2 (supported descriptively): `duration_anomaly` is the only family whose manipulation materially moves the pooled embedding, and its movement scales modestly with |ΔT|.** Measured: displacement ratio 0.207 vs 0.019, matched kernel 0.91 vs 0.99, E1 mean 0.00176 vs 0.000077, within-duration Spearman 0.30 [0.21, 0.39]. Because duration is the only family with a length/patch change, a pure length-sensitivity explanation and a duration-semantics explanation are unidentifiable in this cache.

**H3 (ruled out): representation collapse or global kernel saturation.** No dead dimensions, participation ratio ≈20/128, stable norms, non-saturated normal-pool kernel (mean 0.615, >0.9 ≈ 0.1%).

**H4 (unidentifiable with this cache): the matched no-change null.** No benign/no-injection twin pair was captured; the independent normal null differs from the paired estimator by ~0.09 kernel units on matched groups (and ~0 on unrelated normals), so it is not exchangeable. A calibrated matched-null comparison cannot be produced without new generation/capture.

**H5 (not supported): an encoder-specific artifact.** All three encoders show near-zero E1 for eight families; the apparent excess of encoder 181801 is 1,220/1,221 in the one history whose threshold is negative.

**H6 (weak/unidentifiable): a business burden response.** Deltas are small, all intervals include zero, no burden monotonicity, disjoint non-nested arms and six dependent histories.

What this cache cannot answer: whether any specific anomaly family would be detected at a controlled false-alarm rate; whether the duration effect is length- or semantics-driven; the matched-pair null distribution; any mask/patch-level mechanism (mask indices are not retained); any business-scale or real-time claim.

## 8. Concrete next experiments (proposals only; not executed here)

1. **Matched no-change calibration capture.** Re-capture the same 384 planned sources per history with a benign/no-op variant that preserves length and patch count (and, separately, a length-only no-op), keeping the same K = 8, bandwidth pool, encoders and estimator. This is the only way to estimate the paired statistic's null under its own matched dependence and to separate threshold noise from change.
2. **2 × 2 length × duration manipulation.** Generate duration variants with matched length changes but different event content, and non-duration variants with controlled padding/truncation that equalise length and patch count. Evaluate the same pooled embeddings to test whether E1 tracks length/patch structure or event semantics, without training or changing the encoders.

## 9. M18 errata and wording clarifications (cited; accepted report not edited)

All other Part-I numbers reproduce: 62,208/61,871/337; 7,710/61,680/191; family group totals; 0/162 Holm with minimum adjusted p 0.1134; 4 p < 0.01 and 18 p < 0.05; the four lowest-p rows; 33/7,710 = 0.43%; largest fraction 0.1875 with CP lower 0.0895; Stage-2 realized ranges (rounding 0.2833 → “0.283”); 220 distinct reference rows; 12 arms and 72 history-arm rows; 83,592 generated sources / 83,252 accepted / 9,110 rejected / 92,362 attempt starts; 435,378 forwards equal to the sum of persisted rows.

1. **`bar_met` value.** `docs/sprint24-paired-experiment-results.md:44` says “`bar_met` false everywhere”. The retained value is JSON **null / not computed** in all 162 cells (`summary["bar_met"] = None`). The correct statement is that no cell meets the response bar and the field was left uncomputed, not that it stores `false`. The same wording appears in the M18 worker record.
2. **Positive point medians.** `docs/sprint24-paired-experiment-results.md:67` says “Only `week / 40 / ELEVATED` has a point median above zero”. The report's own table lists **four** positive point medians: month/160/MID +0.000251, month/250/ELEVATED +9.72e-06, week/40/ELEVATED +0.004436, week/60/MID +0.000108. The table is correct; the prose should say that week/40/ELEVATED is the largest and every 95% interval includes zero.
3. **`NORMAL_RETURN` attribution.** `docs/sprint24-paired-experiment-results.md:21` says “337 `NORMAL_RETURN` abstentions in stage 1 and 3 in the stage-2 population scenarios”. The 3 additional returns belong to the **reference/bandwidth/null pools** (1 each); the Stage-2 population scenarios are all `PAIRED` (18,360/18,360). The total of 340 is correct; only the location is misstated.
4. **Wording clarification (not arithmetic)**: “cells share sources and calibration across histories” (line 44) should read “within each history, the same 384 planned sources recur across the 27 family × severity cells”; histories have distinct seeds, and the measured twin reuse factor is 26.8 within a history.
5. **Wording clarification (not arithmetic)**: `run.json` elapsed 4,959.85 s includes the in-process replay; `official-run.json`/`summary.json` elapsed 4,758.03 s is the matrix clock. Both retained; neither contradicts the other.

## 10. Verification actually performed and reproducibility

- Custody pass: 25/25 checks (all four output hashes, 756 batch hashes, journal-order digest, source hashes, run identity).
- Stage-1 reconstruction: 162/162 cells, 7,710/7,710 groups, 61,871/61,871 pair records; max |Δ| vs retained E1 5.1e-16; 0 failures; 0 Holm rejections reproduced.
- Null pass: sigma matches retained exactly; null scores/q95 reproduce to < 4e-16.
- Stage-2 pass: transition/control recompute < 1e-15; official 6-history bootstrap reproduced to ≤ 1e-16.
- Consistency pass: 41 checks, 38 pass and exactly the three genuine errata in §9 fail.
- Cross-checks/discriminating pass: all 7,710 groups have 8 distinct sources; 162/162 cells present; reported seeds are fixed (`241900`–`241906` family, 2,000/1,000/50/2,000 replicates as stated); every audit process records 0 forwards, 0 model loads, 0 checkpoint loads, no torch import.
- Scripts, outputs, commands and hashes: `artifacts/sprint-24/evidence/s24-7-m19-a01/` (`s24_m19_common.py`, `audit_00_custody.py` … `audit_08_discriminating.py`, `run_audit.sh`, `out/`, `manifest.json`). Bulk embeddings and the 59.8 MB runtime JSON remain server-side.

## 11. Limits and non-claims

- Everything is conditional on the finite retained cache: six histories, three fixed encoders, reused sources, shared exploratory calibration, 8-group null pools and 27-group reference pools. The resampled nulls are correlated resamples, not independent new data; none is a calibrated error rate or stable ground truth.
- The source-disjoint permutation test is conditional on source exchangeability and common admission; its 0/162 Holm result is not evidence of absence, equivalence or a response fraction.
- The 33 threshold exceedances are descriptive; the diagnosis identifies threshold sensitivity as the dominant measured pattern, not a proven cause of every exceedance.
- Duration/length, patch count, strength and severity are associated descriptively; no causal mechanism, length-shortcut claim or family-generalisation follows.
- Stage-2 contrasts are descriptive paired summaries over six history clusters with disjoint, non-nested source slices; no burden curve, operating characteristic or business-scale inference is claimed.
- No training, refit, fine-tuning, Phase C, model change or new capture was performed. No accepted M18 byte was edited; the errata above live only in this new report.
