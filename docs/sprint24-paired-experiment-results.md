# Sprint 24 paired experiment — results

**Status: executed fixed-N48 C03 results (S24-7-M18-A04), with the preserved A03 forensic record below.** Part I reports the owner-authorized M18 recapture at reviewed commit `1788b6efa2f9bb2ae6fd164a71c7f20604302e9e` under the C03 contract. Part II is the unchanged earlier forensic audit of the 477,792-forward `abc6f9e` capture and does not describe the current results.

The confirmatory statistic is the owner-selected source-disjoint joint-three-encoder permutation test, conditional on common admission and exchangeability. The paired K=8 group summaries, response counts and weekly/monthly burden contrasts are descriptive. No detection, absence, response-bar, sequential-look or business-scale claim follows.

## Part I — Fixed-N48 C03 recapture (S24-7-M18-A04)

### Execution and custody

- Reviewed runtime: commit `1788b6efa2f9bb2ae6fd164a71c7f20604302e9e`; all 14 frozen-snapshot path hashes equal `de3f5365…`; C03 contract `docs/sprint24-paired-execution-contract-c03.json` sha256 `75e123f3…`.
- Three accepted frozen encoder checkpoints and restored banks verified by digest against the accepted preflight sha256 `26f9e45d…`; device `cuda:0`; no random weights, no bank fit, no training or refit.
- Exact command, PID, environment, paths and asset hashes: [execution manifest](../artifacts/sprint-24/evidence/s24-7-m18-a04/execution-manifest.json); policy binding probe: [c03-binding-probe](../artifacts/sprint-24/evidence/s24-7-m18-a04/c03-binding-probe.md).
- `run.json` records attempt `S24-7-M18-A04`, source commit `1788b6e…`, 435,378 forwards, 3 loads, 162 stage-1 and 12 stage-2 keys, elapsed 4,959.85 s, status `OFFICIAL-MATRIX-COMPLETE`. Its `task` field is the immutable reviewed runner constant `S24-7-M17` (code lineage); it is disclosed and not relabeled, and the actual task/attempt/agent are bound by the sidecar manifest, the exact command line and the journal.
- The runner performed its own in-process offline replay and required it to equal the live results before writing outputs.
- Remote durable root `/tmp/s24-7-m18-a04-run/work/official`: 756 capture batches (223,010,304 bytes), `events.jsonl` 363,187,051 bytes sha256 `8ae5080c…`, `official-run.json` 59,787,284 bytes sha256 `83a5e3a0…`, `replay.json` 1,829,980 bytes sha256 `ddde50a4…`, `run.json` 364 bytes sha256 `6f247605…`; bulk hash manifest in [run-custody-manifest](../artifacts/sprint-24/evidence/s24-7-m18-a04/run-custody-manifest.json). The earlier `/tmp/s24-7-m18-a02-run` forensic root and the dirty server checkout were not touched.
- Forward ledger: 578,203 provisional completed before this attempt + 435,378 actual completed here = **1,013,581 provisional completed**; historical and in-batch failed partials remain unknown. Three model loads; no additional smoke run.

### Denominators, retries and abstentions

- 62,208 planned stage-1 pairs; **61,871 admitted (`PAIRED`)**, 337 `NORMAL_RETURN` abstentions in stage 1 and 3 in the stage-2 population scenarios; no other disposition.
- **7,710 complete K=8 groups** (61,680 pairs); 191 admitted pairs did not complete a group. Group totals by family: contextual_replacement 864, cross_channel_inconsistency 864, duration_anomaly 864, freq_phase_mismatch 864, missing_event 816, over_regularity 864, realistic_stuck 864, subtle_drift 858, wrong_transition 852.
- Across all sections 83,592 sources were generated and 83,252 admitted; the journal holds 92,362 attempt-start records, 83,252 accepted and 9,110 rejected actual injection attempts, all lifecycle-consistent with the selected pairs.
- All 162 source-disjoint tests computed (no coverage abstention) and all 12 stage-2 arms computed (no singleton-coverage abstention). Completion forwards equal the sum of persisted embedding rows.

### Stage 1 — confirmatory source-disjoint joint-three-encoder test

- Design: within each history × family × severity cell the 384 planned sources are pre-split 192/192 by the frozen `s24-c3-source-split-v1` SHA-256 key; only admitted sources enter, each contributes exactly one file (twin-only or variant-only). The statistic is the mean of the three encoder-specific unbiased unequal-N Gaussian MMD² values using the independent bandwidth pool, with 9,999 permutations and an upper-tail p-value; global Holm across all 162 cells at α = 0.05.
- Result: **0 of 162 cells reject after Holm** (minimum Holm-adjusted p = 0.1134). Unadjusted: 4 cells have p < 0.01 and 18 have p < 0.05, which is descriptive and not a calibrated claim (fewer than 8.1 such cells would be expected under a uniform null).
- Lowest unadjusted p-values:

| Cell | p | Holm-adjusted |
|---|---:|---:|
| duration_anomaly / 0.85 / 06 | 0.0007 | 0.1134 |
| wrong_transition / 0.6 / 01 | 0.0012 | 0.1932 |
| duration_anomaly / 0.85 / 02 | 0.0018 | 0.288 |
| duration_anomaly / 0.85 / 05 | 0.0042 | 0.6678 |

- Reading: under the frozen C03 conditional exchangeability and common-admission assumptions, the joint three-encoder feature distributions of twin-only and variant-only source pools are not distinguishable at familywise 0.05 after multiplicity control. This is not evidence of absence, equivalence, a response fraction of 0.5, or generalization beyond the designed corpus.

### Stage 1 — descriptive paired-group diagnostics

- 33 groups of 7,710 exceed their exploratory median null q95 (0.43%), all in `duration_anomaly` cells: history 04 9/48, 8/48, 6/48 at severity 0.85, 0.35, 0.6; history 02 2, 2, 1; history 06 2, 2, 1. The largest cell response fraction is 0.1875 (Clopper–Pearson lower bound 0.0895). Every other family records zero responses.
- These counts are descriptive. The threshold is the exploratory four-group q95 per history/encoder, the exact intervals are conditional binomial descriptions, and cells share sources and calibration across histories. No cell meets the response bar (`bar_met` false everywhere); no detection or absence claim is made.

### Stage 2 — weekly/monthly burden ladder (descriptive)

- 12 arms (week 40/60, month 160/250 × LOW/MID/ELEVATED) × 6 histories, all computed with the stratified realized-weight estimator over all 220 distinct causal-reference rows (no tiling) and an independent unequal-N control. Requested burden and realized admitted weight are recorded separately; no redraw or rebalancing.
- Realized abnormal fractions: LOW 0.050–0.052, MID 0.150–0.152, ELEVATED 0.275–0.283 (integer rounding of the frozen planned counts).
- Six-history bootstrap of the encoder-median transition (descriptive; unit = history):

| Arm | Point median | 95% bootstrap |
|---|---:|---|
| month / 160 / ELEVATED | -0.000442314 | [-0.0014357, 0.000535422] |
| month / 160 / LOW | -0.000230171 | [-0.000469842, 8.94735e-05] |
| month / 160 / MID | 0.000250646 | [-0.00107778, 0.00135956] |
| month / 250 / ELEVATED | 9.72196e-06 | [-0.000740856, 0.000354258] |
| month / 250 / LOW | -0.000865054 | [-0.00121981, 0.000292506] |
| month / 250 / MID | -0.00029828 | [-0.000590101, 0.00085044] |
| week / 40 / ELEVATED | 0.00443561 | [-4.39063e-05, 0.00604186] |
| week / 40 / LOW | -0.0001818 | [-0.00251938, 0.00481642] |
| week / 40 / MID | -0.00156667 | [-0.00286924, 0.0026625] |
| week / 60 / ELEVATED | -0.00128463 | [-0.00313457, 0.00146594] |
| week / 60 / LOW | -0.00119198 | [-0.00271012, 0.00160148] |
| week / 60 / MID | 0.000107567 | [-0.00198896, 0.000979234] |

- Only `week / 40 / ELEVATED` has a point median above zero (0.00444, bootstrap [-0.0000444, 0.00604]); every other arm interval includes zero. Controls are near zero. All arms use `contextual_replacement` severity 0.35, the first grid cell; this is not evidence that Stage 2 covers all families or severities, and no paired transition-minus-control interval, calibrated operating characteristic or business-scale inference is available.

### Verification actually performed

- Fresh-process verifier with encoder entry points replaced by raising stubs and `torch.load` forbidden: offline replay returned `REPLAY-VERIFIED`, and all **43 independent checks PASS** ([independent-verification.json](../artifacts/sprint-24/evidence/s24-7-m18-a04/independent-verification.json)).
- Covered: replay equality for all 162 stage-1 summaries, group outcomes, disjoint tests, Holm and stage-2 bootstraps; every capture batch hash, shape and finiteness; forward denominators; pre-c1 pool disjointness; bandwidth sigma taken from the independent pool; null q95 recomputation; C03 index bands; 9×3 label grid; split membership disjointness; four sampled cells recomputed across all 48 groups and three encoders; four sampled disjoint statistics with permutation p-values; Holm recomputation; sampled Stage-2 stratified/control values; all 12 arms with 220 distinct reference rows and ID separation; attempt lifecycle and attempt/disposition cross-checks; Stage-2 arm-band separation.
- Encoder-call counter was 0 during replay and `torch.load` was never called. The verifier recomputed the selected numerics with its own implementations, not the reviewed estimator functions.
- Evidence hashes: independent-verification.json `8b62de2d…`, summary.json `d156a185…`, replay.json `ddde50a4…`. Helper scripts are verification tooling; no source, contract or test file was modified.

### Limits and non-claims

- Confirmatory inference is conditional on the declared independent-source and common-admission assumptions; finite deterministic seeds do not prove iid sampling or joint-feature exchangeability, and the six histories, three encoders and reused sources are dependent structures.
- Paired group summaries, response fractions and Stage-2 contrasts are descriptive; the four-group exploratory null is not a calibrated 5% error rate, and Clopper–Pearson intervals are conditional binomial descriptions.
- Source reuse across family/severity scenarios induces dependence; calendar times are planned synthetic scenario times, not observed real-world time, and no implicit history or density expansion is claimed.
- No training, refit, fine-tuning, Phase C, business-scale or deployment claim; fixed N = 48 is not an executed G24 sequential look.
- Raw waveforms are not retained; offline replay operates on saved embeddings only.

### Reproduction

```bash
# clone the exact reviewed commit, then run with the same accepted assets
python experiments/sprint24_task7m17_paired_execution.py --mode official-run \
  --permit-official --g-level g48 --device cuda:0 \
  --attempt-id S24-7-M18-A04 --cumulative-before 578203 \
  --b0-root <accepted b0 root> --bank-root <accepted bank root> \
  --preflight <accepted preflight.json> --out <fresh dir>
# fresh-process offline replay (encoder calls forbidden)
python verify_run.py --capture-root <out>/capture --official-run <out>/official-run.json \
  --replay <out>/replay.json --out <evidence>/independent-verification.json
```

Predecessor records: [A03 worker audit](../artifacts/sprint-24/task-7-m18-a03.md), [A03 evidence](../artifacts/sprint-24/evidence/s24-7-m18-a03/), [A03 paused handoff](../artifacts/sprint-24/handoffs/s24-7-m18-a03-handoff.md). The A03 report text below is preserved verbatim from `artifacts/sprint-24/evidence/s24-7-m18-a04/predecessor-a03-report.md` (sha256 `a35db6ff…`).

---

## Part II — Preserved forensic record (A03; old 477,792-forward capture)

The following is the earlier A03 report text, preserved unchanged. It describes the `abc6f9e` capture, which remains forensic/descriptive; it does not describe the Part I results.

# Sprint 24 paired experiment: recovered capture and limitations

**Status: BLOCKED — descriptive recovery only; no scientific decision or acceptance.** The owner approved a fixed-N48 deviation after capture: the runner did not execute the planned G24 futility decision. Main reported not having viewed scientific endpoint outcomes before that amendment. This is not prospective preregistration of the deviation; no sequential guarantees, interim stop, or post-hoc G24 decision are claimed. The frozen C02 contract and raw results are unchanged. Independent verification of the fixed-N analysis and required capture lineage is not possible with the retained artifacts described below.

## Execution and custody

The completed server log reports `OFFICIAL-MATRIX-COMPLETE`, `cuda:0`, 5,512.84 seconds, 477,792 counted forwards and three model loads; the actual output contains 162 stage-1 cells, 12 stage-2 arms and nine family entries. The isolated checkout currently identifies commit `abc6f9e2589b0b813cf67646fb2f2815add4ec2f`. The completed run metadata also cites that commit but says task `S24-7-M17`, attempt `S24-7-M17-A05`; these are hard-coded by the M17 CLI, not an amended raw identity. The separate M18 A02 runtime history and isolated output/log path support the execution's custody, but the original process command and its exit code are not independently preserved in the recovered record. Checkpoint hashes in preflight and the runner's asset verification support intended three-encoder restoration; discarded embeddings preclude independent verification of each output's identity. No new inference or model loads were performed for this audit.

The raw JSON, logs, preflight, checksums, numerical checks and exact retained remote paths are documented in [A03 evidence](../artifacts/sprint-24/task-7-m18-a03.md). A02 has no saved completion handoff; its accessible runtime history ends during the run. Successful count reconciliation gives a provisional cumulative 562,153 forwards including historical 83,173 and 1,188 smoke; failed or partially executed calls are not independently ruled out. The smoke log's own cumulative field uses a stale baseline and is not the physical ledger.

## Stage 1: recorded summaries, not verified effects

There are six histories × nine families × three severities. The saved summaries record **zero responses in all 162 cells**, including every duration cell and severity 0.35, 0.6 and 0.85. This is an unverified property of the stored summaries, not evidence of absence, a general representation verdict or a successful negative test. Duration is interpretable only within-family; twin and anomaly lengths need not match. Across 62,208 planned pairs, the stored disposition totals are 61,871 `PAIRED` and 337 `NORMAL_RETURN`; 61,680 pairs entered 7,710 complete K=8 groups and 191 otherwise paired records did not complete a group. The per-cell denominator is 48 groups in 126 cells, 47 in 18, 46 in six and 45 in 12. Normal returns are abstentions, not negative examples.

Reported exact binomial intervals numerically reproduce to their four-decimal rounding, conditional on their assumptions. Individual group outcomes, per-encoder group scores, pair/retry/source manifests, actual embeddings, reference rows and null scores were not retained in the run work root. Therefore this audit cannot independently reproduce E1, verify realized pairing, leakage or calibration, assess group independence, or validate the fixed-N binomial interpretation. The runner calibrated a shared threshold from four null groups per history/encoder, rather than preserving independent cell-level null evidence. First-group sensitivity output exists for one encoder, not a complete per-group/per-encoder sensitivity audit.

The saved Holm flags reject in all 162 cells. These reflect two-sided binomial tests against response probability 0.5, including a lower-tail departure for zero responses; they are **not** tests proving detection above the null threshold or absence. Holm is applied within each family to 18 history/severity cells, not automatically across the nine family claims. Raw p-value and interval arithmetic can be reproduced, but multiplicity scope, null and dependence must be independently justified before inferential conclusions.

## Stage 2: retained descriptive aggregates

The saved data contain all 12 weekly/monthly burden arms and six histories per arm, with all three encoders' scalar transition/control E1 values and mixture/control file IDs. The code uses `contextual_replacement` at severity 0.35, the first grid cell, for all stage-2 arms; this is not evidence that stage 2 covers all nine families or severities. The saved within-arm mixture/control ID sets are disjoint in all 72 history-arm records; this does not prove independence from the reference, calibration, other arms, or upstream sources. Realized mixture and control sizes match planned N in these saved records. The bootstrap below is a descriptive six-history bootstrap of the encoder-median **transition**, not a paired transition-minus-control interval or calibrated comparison to context-only q95. No decision is implied by the numbers.

| Horizon / N | Burden | Median transition | Median control | Stored transition 95% bootstrap |
|---|---|---:|---:|---|
| week / 40 | LOW | -0.001160 | -0.003356 | [-0.002902, 0.001679] |
| week / 40 | MID | -0.003258 | -0.000302 | [-0.005657, 0.005580] |
| week / 40 | ELEVATED | 0.005666 | -0.001284 | [-0.000858, 0.010752] |
| week / 60 | LOW | -0.001793 | -0.001308 | [-0.002927, 0.000822] |
| week / 60 | MID | 0.001246 | -0.002059 | [-0.001659, 0.003530] |
| week / 60 | ELEVATED | -0.001551 | -0.001516 | [-0.003971, 0.004482] |
| month / 160 | LOW | -0.000113 | 0.000134 | [-0.000731, 0.000871] |
| month / 160 | MID | 0.000310 | -0.000568 | [-0.001057, 0.001178] |
| month / 160 | ELEVATED | 0.000123 | -0.000535 | [-0.001299, 0.001026] |
| month / 250 | LOW | -0.000557 | 0.000283 | [-0.001001, 0.000498] |
| month / 250 | MID | 0.000112 | -0.000240 | [-0.000338, 0.001304] |
| month / 250 | ELEVATED | 0.000303 | 0.000152 | [-0.000396, 0.000597] |

All 12 stored bootstrap summaries reproduce exactly from retained per-history scalars and the stated seed. For the three N=250 arms the runner tiles 220 reference embeddings to 250, duplicating 30 rows before off-diagonal unbiased MMD calculation. Repeated observations remain in off-diagonal terms and change both weighting and the expected statistic; the required independent-sample unbiased interpretation is unsupported. An illustrative no-model calculation in the evidence demonstrates a numerical difference but cannot quantify the actual correction because the reference and mixture embeddings were not retained. This particular defect does not automatically invalidate the other horizon sizes, though those remain subject to the broader custody and inference limitations.

## Required next decision

Preserve the existing raw bytes. To meet unchanged acceptance, first locate independently verifiable original per-file embeddings, source/retry/group manifests and calibration/reference records if they exist elsewhere; none are present in the examined run work/preflight root. If unavailable, an owner decision is needed before any new capture or other outcome-producing run, which the fixed-N amendment explicitly does not authorize. Any repair requires a reviewed estimator and inference plan (including non-duplicated N=250 reference handling, calibration/independence and multiplicity), preserved all-encoder per-group results and provenance, followed by an explicitly authorized reviewed/checkpointed execution and independent audit. No training, refit, business-scale or Phase C conclusion follows from this forensic recovery.
