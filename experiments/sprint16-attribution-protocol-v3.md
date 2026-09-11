# Sprint 16 Attribution Protocol v3 (prospective amendment)

## §0. Supersession and lineage (read first)

- v1 (`experiments/sprint16-attribution-protocol-v1.md`,
  sha256 `196c110b582518fa8f2beeda4fb7e1318fdec296b7c7d603668d405a9224d545`)
  is PRESERVED byte-identical as the failed-freeze record (first review FAIL:
  3 MEDIUM + 4 LOW).
- v2 (`experiments/sprint16-attribution-protocol-v2.md`,
  sha256 `cdfcadb8751696c12e7efe207c7d10a63eba06f6a45a4bc31f79f7ddf713d7fe`)
  is PRESERVED byte-identical as the superseded amendment (re-review
  NEEDS_REVIEW: 1 MEDIUM + 3 LOW, `agent://Sprint16Task2EvidenceR2`).
  Recording both digests here answers the R2 "unverifiable v1 preservation"
  note with a forward anchor: any future byte change is detectable against
  these values.
- This v3 is the live freeze for Tasks 3–16 (scope widened per R2-O2: Task 16
  applies §10 via §12). It restates the full protocol with the corrections
  below; changed paragraphs carry a `(v3: R2-0X)` tag. Untagged paragraphs
  are v2-continued.
- v3 was written before any comparative outcome was inspected. Outcomes
  observed under v1/v2 MUST NOT select v3 content — and none did (no Task 3+
  code or outputs exist in the tree).
- Delta table:

| Finding | v2 defect | v3 disposition |
|---|---|---|
| R2-01 MEDIUM | `pooled_scale` undefined in stability ratio | §4.5 freezes the denominator exactly: pooled SD of the two score sets, float64, ddof=0, floored at `STD_FLOOR` (1e-6, `src/synth/probe15.py:38`); Task 3 implements inline (no phantom helper) |
| R2-02 LOW | §4.4 hierarchy matches no implementation | §4.4 restricted to the existing global Fit-healthy Standardizer; production 3-level hierarchy evaluated only at C1/M1 with exact levels cited |
| R2-03 LOW | movement aggregation unit undefined; v1 qualifier dropped silently | §10 freezes movement = max over all (variant × history) per-history \|Δ\|; signed direction = sign at argmax (+ full signed table); deliberate departure from v1's ≥2-histories wording flagged |
| R2-04 LOW | crash-count accounting ambiguous | §8: crashed attempt consumes NO count; the single repeat consumes ONE count; at-most-once rephrased as one allocation per variant; crash-free full pass = 20, post-crash max distinct = 19 |
| R2-O1 INFO | 36 double-counts M5/C5 | §8 arithmetic: naive sum 36, distinct executions ≤34 (M5 tiny runs ARE the C5 variants); cap ≤20 unaffected |
| R2-O2 INFO | scope reads Tasks 3–15 while Task 16 bound | live scope is Tasks 3–16 throughout |
| R2-O3 INFO | G1-fail branch untabulated | new §10 row G0 maps G1-fail/data-physics limitation to component UNRESOLVED explicitly |

## 1. Authority and amendment rule

1. This protocol freezes candidate components (§3), one-boundary interventions
   (§§6–8), budget (§8), eligible slices (§5), the common metric contract
   (§§4, 4.5), stopping rules (§9), and verdict rules with precedence (§10)
   BEFORE any new comparative outcome is inspected.
2. Tasks 3–16 execute exactly what is written here. Anything not listed here is
   forbidden unless a v4 protocol freezes it first.
3. Task 3 owns implementation of the diagnostic reader; it MUST satisfy §§4,
   4.5 but has no freedom to change metrics, floors, or decision thresholds.
4. All Sprint 16 non-negotiables apply (see §11); conflicts resolve toward the
   more restrictive reading.

## 2. Accepted inputs (resolved by Task 1, re-verify before any run)

- Benchmark: Sprint 15 candidate 7, accepted `MEASURABLE`
  (`experiments/sprint15-benchmark-protocol-v7.md`).
- Roles (`artifacts/sprint-15/task-20.md` §2): Design H-DESIGN-37..40
  (1600–1603, structural reference/diagnostics only — NEVER evaluation and
  never model-fitting input); Fit H-FIT-28..30 (1604–1606, healthy rows only
  — the sole fitting input); Calibration H-CAL-10 (1607, frozen
  95th-percentile rule, threshold `138.03` — selection only, no evaluation, no
  tuning); Confirmation H-CONF-34..37 (1608–1611, the single evaluation set);
  disposable proof H-PROOF-13/14 (evidence only).
- Sealed H-SEAL-37..40 (1612–1615): FORBIDDEN to open/score/probe. Seal
  manifests verify intact before any execution (§11.1).
- Checkpoints: accepted Sprint 12 control (`8bdb845b…`) and hybrid
  (`76be843b…`); hash-re-verify on the execution host before loading.
  Reference banks are checkpoint-embedded and restored as-is; the code exposes
  an explicit opt-in — `prepare_reference_bank(bank, reference_embeddings, *,
  refit=False)` (`src/representation/inference.py:63-78`, "refit only on
  explicit opt-in") — so every Sprint 16 run MUST pass `refit=False` (or
  assert the returned provenance tag == `"restored"`) and record bank rows +
  source in each `S_pop` output. `refit=True` is forbidden without a v4
  protocol.
- Handcrafted oracle: `src/representation/handcrafted.py` (+ frozen
  dev-train-only `Standardizer`); accepted ceiling 0.663 file AUROC /
  0.948 paired ranking, FPR 0.669, diagnostic-only.
- Code bytes: Appendix A full digests. Re-verify; mismatch voids the run per
  handoff §5.
- Runtime: repo root, locked `uv` env, python 3.13.3, torch 2.14.0+cpu, AMD64.

## 3. Candidate components and pipeline boundaries

Production pipeline under attribution (`src/representation/`, `V1Config`
defaults: patch 32 / stride 16, `d_model` 128, 4 layers × 4 heads, conditional
norm on, mask ratio 0.40 with random/info/block composition, contrastive
λmax 0.1, kNN k=5):

| ID | Component | Boundary (code) | Loss hypothesis |
|---|---|---|---|
| C1 | Conditional normalization | `layers/normalization.py::ConditionalBatchNorm` in/out | removes amplitude/gain/spectral/cross-channel info |
| C2 | Patchification | `collate_variable_files` + patchifier (32/16) | dilutes short/local/boundary anomalies |
| C3 | Local patch encoder | `layers/patch_encoder.py::LocalPatchEncoder` in/out | drops transient/phase/coupling/degradation signal |
| C4 | Context encoder + objective use | `layers/sequence_encoder.py` in/out (weights frozen) | mixing dilutes localized evidence |
| C5 | Objective / masking pressure | `masking.py`, `criterion.py`, `trainer.py` (training-time only) | interpolation/smoothness/nuisance shortcuts |
| C6 | File embedding / pooling | valid-patch → file embedding reduction | healthy-patch dominance erases local evidence |
| C7 | Conditional geometry | `inference.py::NormalReferenceBank`, `embedding_extraction.py` | wrong centroids/covariance/hierarchy/bank/scale |
| C8 | Scorers | `S_pred` (context mismatch) vs `S_pop` (population), kept SEPARATE | query-conditioning/energy terms unreadable |
| C9 | Patch-to-file aggregation | patch scores → file/event score reduction | tail behavior lost in reduction |

## 4. Common diagnostic contract (Task 3 MUST implement exactly this)

1. **Primary stagewise metric:** per-history P+W macro tie-aware event AUROC
   with history-block bootstrap B=2000, seed 20260202; report point estimate +
   95% LCB (2.5th percentile). Support: identical file/event set across the
   stages of one comparison (intersection of valid support); valid-patch masks
   preserved; history unit never pooled (per-history reporting; cross-history
   macros unweighted means only).
2. **Frozen operating metrics** (from handoff §4, unchanged): directional
   count (> 0.55); per-event max aggregation for REPORTING; P recall ≥ 0.50
   with median lead ≥ 1.0 d / W recall ≥ 0.25 with median lead ≥ 0.5 d at the
   frozen threshold `138.03`; FAR ≤ 0.05 per robot-day with ≤2-day grouping +
   reset-split; abrupt companion reported separately, never decisive.
3. **Reader properties:** same support, same standardization provenance,
   same tie handling, same aggregation for every stage except when the tested
   component IS aggregation (C9). `S_pred`-family and `S_pop`-family scores are
   never fused before the verdict stage.
4. **Conditioning hierarchy and splits (v3: R2-02):** the common diagnostic
   reader standardizes EXCLUSIVELY with the existing global Fit-healthy
   transform — `standardize_fit` / `apply_standardization`
   (`src/synth/probe15.py:116-129`: single global float64 mean/std over
   Fit-healthy rows, std floored at `STD_FLOOR`) or equivalently
   `handcrafted.Standardizer` fitted on development rows only with frozen
   `apply`. No per-bucket hierarchy is constructed for the diagnostic reader.
   The production 3-level fallback hierarchy (robot+program → robot → fleet,
   `src/representation/layers/normalization.py:41-43`, selection `:202-216`,
   `min_bucket_samples` gating) is evaluated ONLY at its own boundary —
   C1/M1 and the C1 N-b variant, which mirrors the bucket rule with frozen
   Fit statistics. Splits are the role splits exclusively (Fit fit /
   Calibration threshold / Confirmation eval) — no new splits are created.
   Ties use average ranks everywhere (AUROC tie-corrected; Spearman
   average-rank). Uncertainty is history-block bootstrap (B=2000, seed
   20260202) for every metric. Aggregation is per-history first; cross-history
   summaries are unweighted means only.
5. **Candidate building blocks** (reusable, replaceable by Task 3 provided
   §§4.1–4.5 hold): `src/synth/probe15.py` (`feature_names`, `extract_features`,
   `standardize_fit`, `fit_centroid`, `score_files`, `history_block_lcb`),
   `src/representation/handcrafted.py` features.
6. Labels/masks/failure categories/simulator state: post-hoc diagnostics only,
   never representation inputs.

## 4.5. Frozen secondary evaluator outputs (v3: R2-01)

Task 3 MUST implement all three under the §4.4 rules. They are consistency
and replication inputs, never substitutes for the primary metric (§10 P4):

- **Severity ordering:** Spearman ρ between event/file scores and ordered
  severity levels, computed within P and within W separately (never pooled
  across cohorts), on the §4.1 identical-support rule; average-rank ties;
  history-block bootstrap ρ + 95% CI per history. Descriptive severity
  behavior for Task 15 = monotone non-decreasing median score across severity
  levels within cohort (no threshold).
- **Historical/static file discrimination:** standard sklearn-convention AUROC
  (tie-corrected, nan on single-class) separating event files vs control
  files on identical support; static rows = file-level scores, temporal rows
  = event-window scores; report both per history. Never substituted for event
  metrics (Sprint 12/14 rule preserved).
- **Unaffected-background stability:** support = verified-healthy files
  outside all event windows with ≤2-day grouping + reset-split,
  non-quarantined; metrics = stability ratio ≤ 0.10 AND per-history
  background FAR ≤ 0.05 at the frozen threshold, where (v3: R2-01)
  `stability ratio = |median(A) − median(B)| / pooled_scale` with
  A = per-file scores of the unaffected-background set (this history),
  B = per-file scores of the frozen Fit-healthy set, medians over files, and
  `pooled_scale = max(sqrt((var(A) + var(B)) / 2), STD_FLOOR)` with
  population variances (`numpy.std`, ddof=0, float64) and `STD_FLOOR = 1e-6`
  (`src/synth/probe15.py:38`). No pooled-scale helper exists in the tree;
  Task 3 implements this formula inline — the formula above IS the code path.
  Bootstrap uncertainty per §4.4 (resample A; B fixed as the frozen reference).

## 5. Eligible history and mechanism slices

- **Evaluation:** H-CONF-34..37 only, per-history. A claim needs ≥3 of 4
  histories (§10). "Independent histories" in plan Tasks 14–15 means these 4
  confirmation histories (+ P/W mechanism replication), NOT a separate
  development role — no other role may serve as evaluation data.
- **Fitting/standardization/banks:** Fit-healthy rows only
  (H-FIT-28..30 verified-healthy files). Design rows are NEVER fitting input.
- **Threshold:** frozen `138.03` from H-CAL-10; no re-thresholding, no tuning.
- **Structural reference:** Design roots for support/diagnostic sanity only.
- **Mechanism slices:** P1/P2, W1/W2, A1/A2, nuisance-only controls,
  unaffected background. P+W macro is primary; A1/A2 companion-only, never
  decisive alone. P and W cohort replication is individually required (§10).
- **Tiny gradient work (C5 only):** Fit-healthy rows ONLY; never Confirmation,
  Calibration, Sealed, or Design.

## 6. Stagewise localization measurements (Tasks 5–12, fixed scope)

Each measurement compares adjacent stages with the §§4–4.5 reader on identical
support and reports the upstream−downstream gap
`G = R_pre − R_post` (primary-metric points). The 8 measurements below are
mandatory, each executed once; they are NON-COUNTING procedures under the §8
budget:

- M1 (C1): pre-normalization vs post-normalization observable probes
  (level/gain, slope/drift, spectral bands, coupling, phase/timing, transients,
  degradation, nuisance controls).
- M2 (C2): pre-patchify vs post-patchify observables at production (32/16).
- M3 (C3): post-patchify observables vs frozen local latents, same patches.
- M4 (C4): frozen local vs frozen contextual latents at matched positions.
- M5 (C5): gradient/objective diagnostics on frozen checkpoints; the ≤2 tiny
  dev-only runs referenced here ARE the §8-counted C5 variants (one factor
  each: mask ratio OR contrastive weight) — not additional executions.
  (v3: R2-O1.)
- M6 (C6): valid-patch distribution vs mean/file embedding; duration/sparsity
  slices.
- M7 (C7): oracle features AND learned features through production conditional
  geometry vs the same regularized healthy-only diagnostic reader.
- M8 (C8/C9): production `S_pred` / `S_pop` (separately) vs predeclared simple
  diagnostic scorers/aggregations on fixed representations+geometry.

**Localization evidence** requires `G ≥ 0.10` with upstream LCB > 0.55
(upstream itself measurable). A gap is SUSPECT-class evidence, never causal.

## 7. Substitution matrix (Task 13 — exactly these 8 cells, fixed scope)

8 mandatory cells, each executed once, NON-COUNTING under the §8 budget. No
additions:

| Cell | Input | Reader/geometry | Question it answers |
|---|---|---|---|
| X1 | Handcrafted observables | Simple regularized reader | Observable ceiling? |
| X2 | Handcrafted observables | Production conditional geometry | Does geometry erase oracle signal? |
| X3 | Frozen local latents | Same simple reader | Local encoder retention? |
| X4 | Frozen contextual latents | Same simple reader | Contextual mixing effect? |
| X5 | Frozen learned latents | Production geometry | Current pipeline delivery? |
| X6 | Frozen learned latents | Predeclared alternative simple scorer | Signal present but unreadable? |
| X7 | Oracle/controlled features | Production pooling + aggregation | Does file reduction erase local signal? |
| X8 | Frozen encoder latents | Query-ablated (target-masked pre-mixing) scorer | Does self-conditioning suppress sensitivity? |

Coincidence note: X3/X4 measure the same quantities as M3/M4 and X6 overlaps
C7 variant G-a by construction. No double-credit: a Task 13 cell execution
does NOT satisfy a Task 14 intervention-evidence requirement and vice versa;
each task's executions are recorded and reported separately.

## 8. Bounded interventions (Task 14 — counted scope)

One principal mechanism per intervention. Frozen values; no tuning, no search.

**Counting unit (deterministic; v3: R2-04, R2-O1):** the cap counts §8
intervention-variant executions under Task 14 only. One count = one listed
variant evaluated once on the full evaluation set (all 4 confirmation
histories; both frozen encoders where §10(v) applies — histories/encoders are
mandatory replication dimensions, never separately counted). §6 measurements
and §7 cells are mandatory fixed procedures and count 0. Each listed variant
is allocated exactly one count; a crashed execution consumes NO count but
produces no evidence; ONE repeat may be granted for a recorded cause and
consumes ONE count. Crash-free arithmetic: 20 listed variants = cap 20
(exactly one full pass). After one crash+repeat, at most 19 distinct variants
can execute within the cap — the "full pass" equivalence holds crash-free
only. Naive execution sum across sections is 8 M + 8 X + ≤20 I = ≤36, but
M5's tiny runs ARE the C5 §8 variants (not additional), so distinct named
executions are ≤34; the ≤20 cap is unaffected either way.

- **C1** (3): N-a global healthy z-score (frozen Fit mean/var, per-channel);
  N-b per-(robot,program) healthy z-score frozen from Fit (global fallback
  under 32 samples, mirroring `min_bucket_samples`); N-c identity (no norm).
- **C2** (3, diagnostic reader on patchified observables — frozen encoder
  weights are incompatible across patch sizes, so no encoder reuse):
  P-a (16/16), P-b (32/32), P-c (64/32). Production (32/16) is the reference.
- **C3**: NO bypass exists without redesign → capped at SUSPECT (see §10 P3).
- **C4** (2, frozen): B-a skip sequence encoder (local latents direct to
  pooling/geometry); B-b local + fixed uniform context average.
- **C5** (≤2 tiny retrains, Fit-healthy rows ONLY per §5, ≤300 steps each):
  mask ratio 0.40→0.15 (contrastive frozen); contrastive λmax 0.1→0.0
  (masking frozen).
- **C6** (3, frozen latents): top-8 tail mean, median, max (k=8 frozen).
- **C7** (2): G-a diagnostic regularized healthy-only reader on learned
  features; G-b global unconditional healthy centroid, same distance form.
- **C8** (2, weight-free, `S_pred`/`S_pop` never fused, aggregation HELD at
  per-event max for both): S-a query-ablated context score (target masked
  pre-mixing, frozen latents); S-b patch-MSE energy score with per-event max
  aggregation (the patch-level quantity changes; aggregation does not).
- **C9** (3, diagnostic-only; reporting stays per-event max per §4.2):
  top-4 mean (k=4 frozen); event-window median; 90th percentile (q=0.9).
  Event window = the benchmark 7-day causal event horizon with the frozen
  reset/quarantine/censoring predicates (protocol-v7 §§); support = valid
  patches in-window; windows with <8 valid patches are excluded with
  per-history exclusion counts reported (exclusion removing ≥1 full history
  triggers the §9 support-failure rule); percentiles use linear interpolation
  (`numpy.percentile` default `method='linear'`) over valid patch scores;
  the median is the 50th percentile by the same method.

**Budget (hard):** ≤20 §8 variant executions (one full pass max crash-free);
frozen-latent evaluations ≤30 min wall per history on CPU; gradient work ≤2
runs / ≤300 steps / ≤8 CPU-hours total. No architecture search, no
hyperparameter tuning, no large retraining. Exhaustion → §9.

## 9. Stopping rules

1. All predeclared variants of a component execute once each — no
   outcome-driven early stopping within a component (avoids selection bias).
2. A mechanism category whose Task 4 oracle fails the §10 G1 floor is labeled
   data/physics limitation and excluded from attribution (never charged to the
   learned pipeline); components resolve UNRESOLVED on that category via §10
   row G0. (v3: R2-O3.)
3. If the primary metric is computable on <3 confirmation histories (support
   failure, incl. C9 window exclusions removing ≥1 history), affected
   components resolve to UNRESOLVED with the cause recorded.
4. Budget exhaustion stops new configurations; open components resolve to the
   verdict the existing evidence supports per the §10 table (SUSPECT or
   UNRESOLVED).
5. Never open Sealed histories to break ties or rescue an ambiguous call.

## 10. Verdict rules and decision precedence (v3: R2-03, R2-O3)

**G1 oracle gate:** a category is attributable only if the Task 4 oracle
reaches paired metric ≥ 0.65 with LCB > 0.55 on the confirmation slice.
Otherwise: data/physics limitation.

**Movement definition (v3: R2-03):** for one component, collect every
per-history primary-metric |Δ| in points on identical support over all its
executed variants × all 4 confirmation histories. Movement = the MAXIMUM of
that set (deliberate departure from v1's "on ≥2 histories" qualifier, flagged
here). Direction = the sign of the delta attaining the maximum (all tied
signs reported on ties); Task 16 additionally publishes the full signed
per-history delta table. Thresholds apply to the magnitude.

**Per-component decision table** (evaluated in row order; exactly one row
fires — total by construction):

| # | Condition | Verdict |
|---|---|---|
| G0 | Category G1-fail (data/physics limitation per §9.2) | UNRESOLVED (not a pipeline verdict; cause recorded) |
| R0 | Upstream LCB ≤ 0.55 (upstream not measurable) | UNRESOLVED (cause recorded) |
| R1 | Upstream measurable, gap < 0.10, movement < 0.05 | HEALTHY |
| R2 | Upstream measurable, gap < 0.10, movement ≥ 0.05 | SUSPECT (anomalous sensitivity; direction recorded; can never become BOTTLENECK without a measured gap ≥ 0.10) |
| R3 | Gap ≥ 0.10 upstream-measurable, and restoration + nuisance + replication + encoder conditions ALL pass | BOTTLENECK |
| R4 | Gap ≥ 0.10 upstream-measurable, any of those conditions fails | SUSPECT |

**BOTTLENECK conditions (all required):** (i) gap ≥ 0.10 upstream-measurable;
(ii) restoration `R_int − R_post ≥ 0.5 × G` with post-intervention LCB > 0.55;
(iii) nuisance veto passed — FAR ≤ 0.05/robot-day on all 4 histories AND
ΔFAR ≤ +0.02 vs the production pipeline on identical support; (iv)
replication on ≥3/4 confirmation histories AND in P and W cohorts separately;
(v) holds under BOTH frozen encoders (control + hybrid), except C1/C2
observable-level claims where no encoder is involved (stated per claim).

**Overall state:**

- `IDENTIFIED`: exactly one `BOTTLENECK`.
- `MULTIPLE BOTTLENECKS`: ≥2 `BOTTLENECK`s at independent stages (different
  pipeline stages with independent restoration — e.g. a C8 win and a C9 win
  on the same signal count once unless each restores alone with the other
  held at production).
- `UNRESOLVED`: otherwise. Insufficient evidence defaults here, never to a
  guessed bottleneck. Component-level UNRESOLVED feeds overall UNRESOLVED
  unless an independent BOTTLENECK is established.

**Precedence (numbered, conflicts resolve downward):**

- P0. Anything requiring Sealed data is rejected outright.
- P1. G1 eligibility gates every component claim on that category.
- P2. Nuisance/background regression vetoes `BOTTLENECK` (caps at `SUSPECT`)
  no matter how large the signal restoration.
- P3. Capability caps: C3 cannot exceed `SUSPECT` (no bypass exists without
  redesign); C5 `BOTTLENECK` additionally requires a replicated tiny-retrain
  win, else caps at `SUSPECT`.
- P4. Single-seed, pooled-only, or favorable-metric-only wins cannot establish
  `BOTTLENECK` (primary metric governs; secondaries are consistency only).
- P5. When two rules disagree on the overall state, the less favorable state
  wins (prefer `UNRESOLVED`).
- P6. Budget exhaustion resolves opens per the G0–R4 table on existing
  evidence; extension needs a v4 protocol.

## 11. Non-negotiable operational restatements

1. Seals verify intact (round-trip) before any execution; seal manifests only,
   never waveform/score content. Fit/Calibration/Confirmation/Design role
   permissions per §2 (handoff §§2–4) govern every task.
2. Handcrafted/learned probes are diagnostic readers, never production
   performance; no probe metric is presented as detector success.
3. No calibrated-risk, probability-quality, production-threshold, or
   operational early-warning claim anywhere in Tasks 3–16.
4. No Sprint 12 BLOCKED-FINAL task reopened; no Sprint 14/15 byte modified;
   Sprint 15 sealed histories unopened for selection AND attribution.
5. One principal mechanism per intervention; no simultaneous
   normalization/encoder/objective/geometry/scorer redesign.

## 12. Traceability (Tasks 3–16 → sections)

Task 3 → §§4, 4.5 (implement reader). Task 4 → §10 G1 (oracle ceiling +
floors). Tasks 5–12 → §6 M1–M8. Task 13 → §7 X1–X8. Task 14 → §8 variants
within §8 budget. Task 15 → §10 replication + §§4–4.5 support rules.
Task 16 → §10 (publish per-component verdicts from the G0–R4 table under
`HEALTHY`/`SUSPECT`/`BOTTLENECK`/`UNRESOLVED`, then the overall state).
All tasks → §§2, 5, 9, 10 precedence, 11.

## 13. Reviewer static checklist (no outcomes needed)

- [ ] Every cited file/hash/role/seed in §2 exists byte-identically
  (Appendix A; Task 1 §6).
- [ ] §§4–4.5, 6–8 contain no tunable values (all numbers frozen above;
      the stability denominator and percentile method are explicit formulas).
- [ ] Counting scope: §8 variant executions ≤ 20 (crashed attempts consume 0;
      repeats consume 1); §6 = 8 fixed; §7 = 8 fixed; naive sum ≤ 36,
      distinct executions ≤ 34; no double-credit across tasks.
- [ ] §10 table is total: G0–R4 cover (G1-fail × upstream-measurable × gap ×
      movement × restoration/nuisance/replication) with exactly one row firing.
- [ ] No task in §12 can consume Sealed, Calibration-eval, Fit-beyond-healthy,
      Design-as-fitting-input, or refit-True bank rows.

## Appendix A. Frozen digests (full SHA256, recomputed pre-outcome)

| File | SHA256 |
|---|---|
| `src/synth/config.py` | `55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056` |
| `src/synth/health.py` | `e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d` |
| `src/synth/events.py` | `85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d` |
| `src/synth/chronicle.py` | `b5992d6af3c4387d18df746b43be17b4bfc1fac6938e8b44be35b6fb7c65dafd` |
| `src/synth/balanced.py` | `d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43` |
| `src/synth/cli.py` | `3c264a33c7e74dd2b1ab2c9cfbb7a7e2e524f1b39a1228f734460edb0d900a37` |
| `src/synth/__init__.py` | `5c979561a2f01581e2943f23e30061788d10dabc2ec1172cbd9f25275032b168` |
| `src/synth/preflight15.py` | `19f0300e97d55609c27f51403de9cc0f7a898684df7050424972fc4e4fb21dd1` |
| `src/synth/probe15.py` | `a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b` |
| `experiments/sprint15-benchmark-protocol-v7.md` | `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6` |
| `experiments/sprint15-observable-probe-v7.md` | `4d4c2b53b3879efdf92935e362956c87d16892feba8276367f1e5aa13aa4a51a` |
| `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` | `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c` |

Lineage anchors (review-chain digests, not frozen inputs): v1
`196c110b582518fa8f2beeda4fb7e1318fdec296b7c7d603668d405a9224d545`,
v2 `cdfcadb8751696c12e7efe207c7d10a63eba06f6a45a4bc31f79f7ddf713d7fe`.
