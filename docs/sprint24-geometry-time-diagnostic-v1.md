# Sprint 24 Geometry–Time Diagnostic Protocol v1 (G1 frozen candidate, pre-execution)

- Protocol ID: `S24-GEO-TIME-v1`; status: **frozen candidate, pre-execution** (no geometry outcomes inspected under it).
- Task: `S24-7-G1`; attempt: `S24-7-G1-A01`; worker runtime agent: `S24G1GoldA01`; assigned role Gold (owner selected).
- Criterion: `docs/sprint-plans/sprint-24.md:138`.
- Base commit: local `master` `aa43ce4e88a8a360b50c1a4b78f9639f643d1066` (Task 7 Bronze checkpoint; product tree identical to `a91431d`, plan-only delta).
- Parent boundaries (unchanged, not rewritten): `docs/sprint24-prospective-protocol-v1.md` (SHA-256 `80647c63f6e1b3b3b3a8c13b3e62a31b155cbb6dc8b7ab9a1e2`) + `docs/sprint24-views-v2-amendment-a01.md`; Task 7 records and `NOT_TRIGGERED` disposition unchanged.
- Support basis: Task 2 inventory `artifacts/sprint-24/evidence/s24-t2/s24-t2-support-inventory.json` (SHA-256 `5718e827b7b2f83257f8a474bcb04248bc6da7342cba3819b3227351bd46ecfc`); T4 accepted audit + allocator proof; Task 7 preexecution lock accepted digests.

This document freezes the geometry–time diagnostic contract. Any change after owner
acceptance needs a versioned amendment (`S24-GEO-TIME-v2`) with rationale; silent edits
are forbidden. It does not authorize G2 execution, retraining, new corpus, new
Confirmation access, or Phase C execution.

## 0. Priority and non-goals

The priority is coherent movement of embedding clusters/distributions during abnormal
periods, NOT individual-file accuracy. File-level PR-AUC and the existing
above-threshold-file event summaries are preserved Task 7 context only; they are not
evidence of collective distribution movement and are not re-decided here.

Out of scope: retraining or fine-tuning of any kind, new corpus or regeneration,
architecture search, outcome-driven feature/window selection, numeric success gates,
fresh Confirmation access, and any Phase C sequential design beyond the G3 decision
input this diagnostic produces.

## 1. Source identity and immutability

The exact accepted Sprint 18 corpus is read-only (protocol-v1 §1 lineage; binding
digest `8e2705140d07d9c603989ec709de5ebb766e580a8619b37c7462f6dd2751dd3b`).
Diagnostic support uses **only the 4 Development histories** as exposed-benchmark
exploratory research; Confirmation stays the completed Task 7 secondary context
(no adaptation, tuning, checkpoint selection, or fresh access).

Per-history clock (own `tmin`, own span; absolute times never concatenated
across histories). Frozen cutoffs/edges are the Task 2 decile-bin record, reused
verbatim from the Task 7 lock (no re-derivation at G2 runtime):

| History | `c1` (bin3/bin4 edge, 40% time) | `c2` (bin6/bin7 edge, 70% time) |
|---|---|---:|
| DEVELOPMENT-01 | `12902229.909` s | `22578902.340` s |
| DEVELOPMENT-02 | `12897362.418` s | `22570384.231` s |
| DEVELOPMENT-03 | `12900372.819` s | `22575652.433` s |
| DEVELOPMENT-04 | `12900060.198` s | `22571624.329` s |

Eleven strictly increasing decile edges per history come from the Task 2
inventory (`chronology_bins`); G2 asserts exact equality with the Task 7 lock
values before building any manifest.

## 2. Representation, preprocessing, and cache identity (frozen)

Primary representation: the three accepted B0-FULL seeds `181801/181802/181803`
(identity-selected, not outcome-ranked). Fixed preprocessing and the locked Fit
reference banks are restored through the accepted public Task 5 path only:

- `representation.sprint24_features.restore_s24_b0_bundle` (strict scope, no refit),
  `capture_s24_file_features` (frozen eval forward, deterministic per-file mask
  seeds under `S24_MASK_RULE_ID = sprint18-score-mask-v2`, `S24_FEATURES_SCHEMA_ID =
  s24-frozen-features-v1`), `s24_preprocessing_digest`, `s24_feature_cache_identity`,
  `s24_live_state_digests` / `assert_s24_frozen_unchanged`;
- N4 scale only as the existing file-score event comparator (frozen
  `load_n4_scale_checkpoint` + `score_s24_n4_normalized`); N4 shares the frozen
  B0 representation and leaves `S_pop` unchanged.

Accepted digests (Task 7 lock, re-asserted before any G2 feature contact):

| Seed | Checkpoint SHA-256 | Bank tensor SHA-256 | Bank file SHA-256 |
|---|---|---|---|
| 181801 | `9292d7c5…` | `2d6e0a11…` | `546e515d…` |
| 181802 | `8ae58da9…` | `d13f0394…` | `59b95f3c…` |
| 181803 | `79edbdf3…` | `2a136229…` | `c234c0b6…` |

Full hashes, bank `k = 5`, calibration q95 values, N4 scale digests, arm-config
digests and binding digest `383c6211…` are bound in the window manifests
(§6) and re-verified at G2 preflight.

Cache identity `s24_feature_cache_identity` binds checkpoint, bank, support,
preprocessing, arm-config, data-binding, model seed, optional N4 scale digest,
**and device**. CPU and CUDA features are equivalent only within the
owner-approved T6-D1 bounded calibration rule; they are not bit-identical, so a
cache computed on one device is never silently reused as the other's.
G2 runs one declared device end-to-end (default `cpu`; `cuda` only under the
deterministic T6-D1 setting with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before
CUDA initialization). Device is recorded in every manifest and output.

Embedding reuse rule: hash-compatible embeddings from reviewed runs may be
reused only when every cache-identity input matches exactly; any substitution
fails closed and forces recomputation. Reused rows carry their source
run/lock/ledger identity.

Inherited temporal exposure (disclosed, not concealed): encoder, input
normalization, bank and thresholds saw Fit/Calibration-healthy rows spanning
the full timeline; Development waveforms were forward-exposed during Sprint 18
scoring. Results are therefore **sequential label adaptation with prior
unlabeled exposure** (protocol-v1 §10), never a strictly prospective
end-to-end temporal claim.

## 3. Label-blind construction boundary (binding)

The Main-clarified boundary is binding on G2 implementation and review:

1. Failure/degradation annotation windows (`failure_events`, degradation
   episodes, derived merged intervals) are **evaluation labels**. They are
   post-hoc assessment inputs only — never window/reference construction
   inputs, never reference fitting inputs, never score inputs, never
   threshold-selection inputs.
2. Maintenance/reset information may define observable operational context
   **only** if deployment-observable and explicitly justified in the G2
   record: each file row carries `last_reset_time` (last recommissioning end
   at or before file start, derived from manifest `maintenance_windows`), and
   the frozen quarantine/eligibility machinery marks maintenance-overlapping
   files. Episode-annotated reset *splits* of merged label intervals do not
   transfer to this diagnostic.
3. Existing T4 label/event-informed role manifests (`V-ADAPT`, `V-VAL`,
   `V-TRIG`, `V-R0`, `V-SEQ-E*`, `V-FUTURE`, budgets, groups, parent custody)
   must not silently define new "label-blind" support. Where this diagnostic
   reuses an inherited custody exclusion (quarantine, censoring, reserve
   identities, maintenance overlap — the frozen `is_eligible_for_training`
   predicate), the G2 record distinguishes that inherited exclusion from
   candidate construction and discloses the full selection lineage.
4. Windows, references, scores and any threshold support are constructed from
   **source-observable timestamps and context only**: per-file `start_time` /
   `end_time`, `robot_id`, `program_id`, `last_reset_time`, quarantine/censor
   flags, eligibility predicate outcome, and manifest maintenance windows.
   Labels (`file_label`), annotation families/severities, future targets, and
   event/episode membership are never consulted during construction.
5. Regime/type sequences and `operating_context_json` ambient/shift fields are
   **not** window/reference inputs in v1 (regime vocabulary is supported by all
   six types in Fit-healthy pretraining; §9 controls use robot/program/time
   matching instead). If G2 evidence motivates them, they enter only via a
   versioned amendment, never silently.

Observable-input audit: every window/reference manifest records the exact
observable fields consumed plus the annotation fields explicitly excluded;
a builder assertion fails closed if any annotation field is read during
construction (provenance-tested in §11).

## 4. Window design (label-blind, causal)

- Analysis span per history: `(c1, tmax]` where `tmax` is the maximum file
  `end_time` of that history. Pre-`c1` support (`end_time <= c1`) is the
  **reference pool** (§5); no reference observation is later than any
  evaluation window end. No cross-history concatenation.
- Window shape: **fleet-wide trailing windows** `(T − W, T]` with width `W =
  14 days` (`1,209,600 s`) and stride `S = 7 days` (`604,800 s`); window end
  `T` runs `c1 + S, c1 + 2S, …` while `T <= tmax`. Fleet-wide is required:
  per-robot daily yield (~0.9 files/day) cannot support distribution
  statistics at 7-day widths (1457/2007 per-robot windows below minimum
  support on the audit fixture), while 14-day fleet windows hold
  (min/median ≈ 43/106 files on the same fixture, zero windows below
  minimum at the 7-day stride).
- Eligibility: only files passing the frozen inherited predicate
  (`is_eligible_for_training`: not quarantined, not censored, no reserve
  identity, no manifest maintenance-window overlap) enter any window or
  reference. Quarantine/censor flags are deployment-observable split
  provenance; the exclusion is inherited custody (§3.3), recorded as such.
- Causal scoring: a window uses only observations with `end_time <= T`
  (all eligible files in `(T − W, T]`, every file scored, no abnormal-subset
  selection). Future files never enter the window.
- Minimum support: `n >= 30` eligible files per window; windows below minimum
  are `UNSUPPORTED` (excluded from statistics, counted in denominators).
- Maintenance-gap handling: a trailing window inevitably spans maintenance
  downtime (fleet-aggregate); quarantine/overlap exclusions thin files but do
  not break the trailing rule. Windows are calendar-anchored, never
  maintenance-resegmented in v1 (resegmentation would need annotation-adjacent
  reset splits; explicitly excluded by §3.2). Per-window eligible counts,
  quarantine-excluded counts, and overlap-excluded counts are published.
- Per-window record: `(history_id, window_end_T, window_start, n_eligible,
  n_quarantine_excluded, n_overlap_excluded, robot_mix, program_mix,
  support_sha256)`.

## 5. Reference design (label-blind, causal, matched)

- Reference pool per history: all eligible files with `end_time <= c1`
  (~1,280 files/history on the audit fixture; all 9 robots represented,
  ~133–163 files/robot). References never use evaluation-span files.
- Reference construction: **fixed reference** — one deterministic
  history-level reference from the entire reference pool (no window-matched
  resampling, no label filtering, no evaluation contact). Fixed references
  keep the comparison direction unambiguous (past → present) and auditable.
- Reference size: full pool (`n_ref ≈ 1,280`); minimum `n_ref >= 200` else the
  history is `UNSUPPORTED` for window statistics (none expected: all four
  histories clear it by ~6×).
- Temporal matching: matching is by construction — the reference is the
  immediately preceding era (`end_time <= c1`) of the same history and the
  windows extend `(c1, tmax]`; no cross-history pooling. Robot/program mix
  drift between reference and each window is published as a confounder
  (§8), not corrected by reweighting in v1.
- Reference record: `(history_id, n_ref, ref_end_max, robot_mix, program_mix,
  support_sha256, cache_identity_per_seed)`.

## 6. Embedding capture and bounded missing-feature inference

- Capture: `capture_s24_file_features` through the §2 public path, per
  accepted seed, for exactly the manifest-listed window + reference files.
  Frozen-state digests asserted unchanged before/after; device recorded.
- Missing features: none expected — every manifest file has a stored waveform
  shard. The **only** permitted bounded inference is re-running the frozen
  forward on the listed file IDs when no hash-compatible cache row exists
  (same checkpoint/bank/preprocessing/support, same device). No imputation,
  no interpolation, no synthetic files, no waveform regeneration.
- Compute cap (locked before G2): at most `4 histories × 3 seeds ×
  (n_ref + n_windows × mean_n)` frozen forwards; with audit-fixture scale
  (`n_ref ≈ 1,280`, ~31 windows × ~106 files ≈ 3,300) ≈ **13,700 file
  forwards per seed**, ≈ **41,000 total**, FP32, single-device, no search.
  G2 aborts and reports `INFEASIBLE` if manifest support exceeds 2× this
  bound instead of narrowing support to fit.
- Every output row binds `file_id → (history_id, window/ref assignment,
  cache_identity, support_sha256)`.

## 7. Original-space movement statistics (location + dispersion/distribution)

All statistics are computed in the original 128-d embedding space (no
projection). Per history × seed × window vs the fixed reference:

1. **Location**: squared Euclidean distance between window mean and reference
   mean, `‖μ_w − μ_r‖²`, plus the per-dimension standardized shift vector
   (reference SD per dimension; guarded zero-variance → `UNSUPPORTED` dim).
2. **Dispersion**: log-ratio of traces, `log(Tr(Σ_w) / Tr(Σ_r))`, and the
   distance standard-deviation ratio per dimension (median over dims).
3. **Distribution**: squared maximum-mean-discrepancy with a fixed Gaussian
   kernel (bandwidth = median reference pairwise distance, computed once per
   history × seed on the reference pool only — no evaluation contact),
   `MMD²(w, r)`; plus the 1-D Kolmogorov–Smirnov statistic distribution over
   the 128 dimensions (median, IQR, max) as a marginal-shift summary.
4. **Neighborhood stability (non-score)**: fraction of window files whose
   k=10 nearest neighbors in the pooled (reference + window) set fall in the
   reference pool, vs the same quantity for reference held-out halves
   (split-half baseline). Descriptive only; not a detection score.

No statistic uses labels, families, severities, or event membership. Formulas,
kernel bandwidth rule, guards (n < 30, zero-variance, non-finite embeddings →
`UNSUPPORTED`), and FP precision are frozen in the G2 runner; any
re-parameterization needs an amendment.

## 8. Negative controls: healthy periods and ordinary operating transitions

Labels/annotations are post-hoc assessment only (§3). Controls are matched by
observable context, selected without reading outcomes:

- **Healthy-period control**: windows whose post-hoc annotation shows zero
  abnormal files *and* zero failure/degradation membership are reported as the
  healthy baseline distribution of every statistic. No window is constructed
  or excluded using this property.
- **Ordinary-transition control**: per-window robot/program mix drift
  (total-variation distance of the window mix vs the reference mix) and
  `last_reset_time` recency (median `T − last_reset_time` over window files)
  are published alongside every statistic as the observable-transition
  covariates. Windows with large mix drift or fresh recommissioning are
  flagged `HIGH-DRIFT` (top decile per history, rule frozen) and analyzed
  as a subgroup — ordinary-regime confounding is assessed, not excluded.
- **Split-half null**: each reference pool is split once (seed `2407`) into
  halves; all §7 statistics recomputed half-vs-half give the no-movement
  null per history × seed. Window statistics are reported against this null,
  not against a tuned threshold.

## 9. Event/healthy-exposure definitions, censoring, and existing-score comparators

- Event denominator: all manifest failure events per Development history
  (Task 2 inventory: ~221–254/history); degradation-only intervals are **not**
  separate events. A window "covers" an event post-hoc iff the event's
  failure time lies in `(T − W, T]`. Events are never window inputs.
- Healthy exposure: total eligible-file days in the analysis span per history
  (sum over eligible files of file duration, in days); false-alarm episodes
  normalize per 100 healthy-exposure days where defined.
- Prospective threshold policy (§10-gated): if the calibration gate passes,
  event recall = fraction of events whose covering windows all cross the
  frozen operating point; detection delay = earliest crossing window end −
  degradation onset (clamped ≥ 0, median + IQR among recalled only;
  non-recalled events are censored — counted in recall, excluded from delay).
  False-alarm episodes = maximal runs of consecutive crossing windows with no
  covered event, per 100 healthy-exposure days. File-level FPR is never
  substituted for event false alarms. If the gate fails, no event recall is
  reported (explicit `NO-CALIBRATION` disposition instead of anfallback rule).
- Existing-score comparator: frozen B0 `S_pred` / `S_pop` and N4 `S_pred`
  event summaries (protocol-v1 §8 rule) recomputed on the **identical
  window/file support** (same files, same coverage rule). No threshold
  retuning; frozen S18 q95 values only.

## 10. Prospective operating-point policy (calibration-gated, no tuning)

No operating point is fitted on evaluation windows. Any window-level score used
for event recall must be calibrated **before** outcome inspection under this
locked rule:

1. Calibration support: the reference pool only (pre-`c1`, no evaluation
   contact). The statistic (primary: `MMD²`) is computed on reference
   split-halves (seed `2407`, ≥ 200 resamples) giving the null distribution.
2. Operating point: the `q95` of that null distribution, frozen per history ×
   seed. Evaluation windows are compared once against it; no adjustment,
   no per-window renormalization, no ROC-based selection.
3. Gate: calibration passes iff (a) the null distribution is finite and
   non-degenerate on all 4 histories × 3 seeds, and (b) at least half of
   healthy-control windows (post-hoc zero-annotation) lie below the point.
   Otherwise the disposition is `NO-CALIBRATION`: movement statistics stand,
   event recall/false alarms are not reported, and G3 receives the negative
   result — never a fallback threshold.

PCA/UMAP, if produced, are illustrations only: fitting support disclosed,
never used for separation or detection-quality claims.

## 11. Uncertainty, seed/history treatment, outputs, and missing/unsupported rules

- Units: (history, seed) is the paired unit; windows/files within a history
  are **not** independent bootstrap units (overlapping windows share files).
  Uncertainty: history-cluster bootstrap over the 4 Development histories
  (`B = 2000`, seed `2407`), reported descriptively given `n = 4`
  (small-history limitation stated; within-arm intervals never read as paired).
- Per-seed and aggregate results published; no favorable-seed or
  favorable-window/event selection. All planned histories/seeds/windows/
  events accounted for; missed/censored/unsupported cases explicit.
- Exact outputs (all under `artifacts/sprint-24/evidence/s24-7-g2/`): window
  manifests + reference manifests (CSV + JSON, §4–5 records with digests),
  per-file cache bindings, raw statistic tables (§7 per window × seed),
  null/calibration tables (§8/§10), control/drift tables, event mapping
  (post-hoc coverage only), comparator tables (§9), cost report
  (forwards, wall-time, device, trainable params = 0, pretraining vs
  adaptation compute), and a study report. Large per-file rows stay
  server-side with hashes; review-sized manifests/tables are copied locally.
- Missing/unsupported: any window with `n < 30`, non-finite embeddings, or
  empty reference halves → `UNSUPPORTED` (counted, excluded from stats);
  whole-history `n_ref < 200` → history `UNSUPPORTED`; calibration gate
  failure → `NO-CALIBRATION`; manifest mismatch or cache-identity mismatch →
  fail closed (`INFEASIBLE`, no silent narrowing).
- G2 verification (locked): metadata-only determinism scenario (this G1
  record) proves label-blind manifest reproducibility and causal reference
  support; G2 adds the frozen-forward + statistics run with the §6 cap,
  per-cell accounting, and unchanged frozen-state digests.

## 12. G2 feasibility scenario actually run (G1, metadata-only)

Public-path scenario on the checked-in T4 fixture
(`artifacts/sprint-24/evidence/s24-t4/s24-t4-dev02-fixture/`, 6,460 rows;
accepted-structure analogue, not accepted data): `build_history_views` +
frozen eligibility + §4/§5-equivalent window/reference construction from
observable fields only — `end_time`, `robot_id`, `program_id`,
`last_reset_time`, quarantine/censor flags, maintenance windows; no label,
family, severity, future-target, or episode read. Observed: reference pool
1,279 eligible files (all 9 robots); eval-span 1,697 eligible; 14-day/7-day
fleet windows min/med 43/106 (zero `UNSUPPORTED`); per-robot 7-day windows
infeasible (1,457/2,007 below minimum) — the fleet-wide choice is support-
forced, not outcome-driven. Determinism: two independent builds byte-identical
(manifest digest match). Script + record in `artifacts/sprint-24/evidence/
s24-7-g1-a01/` (hashes below). No embeddings, scores, or geometry outcomes
were computed or inspected.

## 13. Accepted risks and limitations

Small-history (`n = 4`) uncertainty is wide by construction; fleet-wide
windows trade robot specificity for distribution support; calendar-anchored
windows span maintenance downtime (thinned, not resegmented); reference is a
single past era (mix drift published, not corrected); all Development support
is exposed-benchmark reuse with prior unlabeled + scoring exposure (§2).
A negative or unsupported answer is a valid diagnostic outcome.
