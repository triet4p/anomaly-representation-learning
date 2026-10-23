# Sprint 18 supplementary Development metric supplement v1

**Identity:** prospective supplement `sprint18-supplementary-development-v1`
**Scope:** supplementary **Development** metrics only, for the seven executable
arms of `experiments/sprint18-full-training-protocol-v2.md` (SHA-256
`e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2`).
**Status:** definitions frozen **before** any supplementary value was observed.

## 0. Why this supplement exists

Protocol §8.2 requires, for every branch/arm/model seed/evaluation history:
paired deltas, event AUROC, recall/lead at the frozen q95 threshold, FAR per
robot-day, **unaffected-background stability ratio and uncertainty**,
**localization/duration accounting**, **severity ordering**, the
**P1/P2/W1/W2/A1/A2 slices**, valid support, and mechanism diagnostics.

The accepted Sprint 18 evaluation path never produced four of those families,
and the frozen closed result schema (`sprint18-full-training-result-v2`) has no
`duration_accounting` slot at all. Task 32 established this with executable
evidence (`artifacts/sprint-18/task-32-c01.md`, review
`artifacts/sprint-18/reviews/S18-T32-S18-T32R02.md`). The project user then
authorised a **bounded supplementary Development metric pass** (plan Task 32,
2026-10-24): complete the missing metrics, do not waive them, and change no
weight, normalization, bank, threshold, or support.

This supplement is that pass's scientific contract. It is **additive**: it does
not amend, relax, or reinterpret any frozen protocol clause, metric, threshold,
or historical row. The unchanged v2 validator still refuses a row that carries
`duration_accounting`, so existing v2 consumers can never silently read a
supplementary row as a historical one.

## 1. Frozen inputs (no training, new bank, or reselection)

For one `(arm_id, model_seed)` the supplementary run consumes exactly:

| Input | Frozen requirement |
|---|---|
| Checkpoint | Accepted epoch-100 checkpoint (N4: the paired B0 epoch-100 checkpoint). Exact bytes restored through the unchanged `restore_sprint18_checkpoint`; the restored model/EMA state digests must equal the accepted digests. |
| Reference state | K9: the persisted `k9_downstream.pt` restored byte-for-byte, provenance re-checked against arm/seed/checkpoint/binding. N4: the final `s18-n4-scale-checkpoint-v2` envelope restored strictly, bounds re-checked. B0/K3/K7/N2/N3: see §1.1. |
| Fit support | The complete certified Fit-healthy support (5,446 rows), fixed to the accepted membership/order, is used to restore the exact historical bank where required (§1.1) and as the stability-ratio denominator. No bank, threshold, standardizer, or weight is fitted from Development. |
| Calibration support | The complete certified Calibration-healthy support (1,877 rows). The **operative** per-branch q95 thresholds are the accepted values from custody; the Calibration re-score only *verifies* that they reproduce (≤1e-12 absolute) and aborts on mismatch. Re-scoring never reselects a threshold. |
| Development support | The complete frozen four-history Development support. Per-history manifest and common-support digests must equal the accepted digests. |
| Metric code | `src/representation/sprint18_supplementary.py`, pinned by SHA-256 in every input manifest; a mismatch aborts before any inference. |

`phase` is the literal `development` and `score_confirmation` is never
requested, so Confirmation stays closed behind the Task 33 lock. The runner
performs no optimizer step: `encoder_updates = 0` and the representation state
digest is compared before and after every pass.

### 1.1 Reference-state custody: exact accepted-digest restoration only

The selected project-user option **“Tái tạo đúng digest, fail closed”** authorizes
restoration of the exact historical Fit bank from the frozen checkpoint/EMA and
normalization state, complete qualified Fit membership, frozen order, and the
existing public `fit_fit_reference_bank` recipe. This is custody restoration,
not authorization to fit a different bank. The one non-K9 recipe uses
`sprint18-fit-bank-restore-v1` with `global-file-id-ascending-v1`; the complete
bank digest must equal the accepted historical `bank_sha256`. A mismatch aborts
before any score or bank artifact is published. There is no retry, alternate
order/recipe/options, or new scientific bank.

| Arm | `S_pred` state | `S_pop` state | Frozen behavior |
|---|---|---|---|
| `K9-FULL` | Accepted checkpoint and persisted `s18-k9-downstream-state-v1` restored as-is | Same persisted downstream state restored as-is | No Fit-bank reconstruction |
| `B0-FULL`, `K3-FULL`, `K7-FULL`, `N2-MSP`, `N3-CCRP` | Accepted checkpoint/model/EMA restored as-is | Complete qualified Fit bank reconstructed once and accepted only on the historical digest | Same Fit rows, row order and recipe; mismatch fails closed |
| `N4-CRN` | Paired B0 checkpoint plus accepted final N4 scale envelope restored strictly | Exact paired-B0 Fit bank reconstructed once and accepted only on its B0 historical digest | Preserve N4 scale bounds/state and exact B0 `S_pop` path/q95 parity |

The accepted bank provenance fixes the Fit support at 5,446 rows with source-row
digest
`447b219e0c77f6d76ed54a4de7a0bee7c2b3a9a8142f61cbb434694ed73acb94`; the
per-arm/per-seed historical bank digests and accepted-record pointers are listed
in `artifacts/sprint-18/task-32-source-c03.md`. Preflight also checks the
qualified Fit history roster, full row provenance, data binding, checkpoint,
model/EMA digests, accepted thresholds, and (for N4) the final scale roster and
paired B0 threshold.

After a successful exact digest match, the runner publishes an append-only
`bank.pt` byte artifact beside a closed `provenance.json` carrying all input
paths/digests, recipe/order, source rows, checkpoint/model/EMA state, and bank
digest. The existing K9 artifact is never rewritten; the N4 scale artifact is
never refit or rewritten. Score and metric consumers re-hash these artifacts,
and every score document requires `reference_state_unchanged = true`.

For `reference_state_modes`, `S_pred` remains `restored_from_custody`; a common
arm's `S_pop` is `reconstructed_digest_verified`, while both K9 branches and
N4's `S_pred` remain `restored_from_custody`. `bank_source` distinguishes the
same two custody modes. No accepted record or historical bank digest is
rewritten.

## 2. Reused frozen definitions (no redefinition)

These are consumed verbatim from the accepted code:

- **Event metrics, support, recall/lead, FAR**: `synth.probe15.evaluate_histories`
  and `synth.events` (POS/ANCHOR eligibility, `<= 2`-day alert-episode grouping,
  reset-spanning exclusion, tie-aware AUROC).
- **Stability ratio**: `representation.attribution_metrics.stability_ratio`,
  i.e. `|median(background) − median(Fit healthy)| / max(pooled SD, STD_FLOOR)`
  (population variances, float64, inherited `STD_FLOOR = 1e-6`).
- **Background sample**: the flattened control-window member scores of the
  history — exactly the Sprint 17 `neg` sample (`experiments/sprint17_task10_c2.py`
  `branch_metrics`).
- **History-block bootstrap**: `probe15.history_block_lcb` draw construction
  (`B = 2000`, seed `20260202`, NumPy linear quantiles); §8.2 aggregation
  arithmetic (pair within seed/history, average seeds within history, macro over
  four histories).
- **Severity ordering**: `attribution_metrics.severity_spearman`, fired only when
  at least three severity-carrying events exist for the cohort (Sprint 17 rule).
- **Slice AUROC**: `synth.events.roc_auc_tie_aware` on identical control
  negatives (Sprint 17 rule).

## 3. The five families, frozen definitions

### 3.1 Unaffected-background stability ratio (`sprint18-background-stability-v1`)

Per branch and per history: `background_stability_ratio` = §2 Frozen Sprint 17
formula applied to `(control-window member scores, Fit-healthy branch scores)`.
Reported as a finite number, or `null` when either sample is empty. It never
replaces the per-history `<= 0.10` gate.

### 3.2 Background uncertainty (`sprint18-history-block-uncertainty-v1`)

Per branch: `aggregate.background_uncertainty = [2.5th, 97.5th]` percentile of
the history-block bootstrap of the **four seed-averaged per-history stability
ratios** (`B = 2000`, seed `20260202`, NumPy linear). The four units are
resampled with replacement; seeds are not resampled. `[null, null]` when any
history ratio is undefined.

### 3.3 Localization (`sprint18-localization-v1`)

Exact Sprint 17 definition: over evaluable P+W events, take the event's
top-scoring member file; from that file's per-timestep diagnostic vector `t`
computed with the frozen forward (`masked MSE(predicted, target)` reduced over
the latent width and mapped by `Patchifier.patch_to_timestep_scores`), let
`n = |t|`, `dt = (end − start) / max(1, n)`, and `peak = start + argmax(t)·dt`.
The event is localized when `failure_time − 7 d <= peak <= failure_time`.
`localization` = localized / evaluable. Vectors that are empty, non-finite, or
constant make the event unevaluable and are counted, never scored as a miss;
`null` when nothing is evaluable. The per-file timestep vectors are pinned by a
SHA-256 digest in the score document; only the compact `argmax`/length/flags are
persisted.

The diagnostic vector uses the S_pred evidence channel for every arm, which for
the harness arms is byte-identical to `RepresentationInference.score_batch`'s
`timestep_scores`, and for K9 is the C8-A patch energy mapped through the same
frozen patchifier method.

### 3.4 Duration accounting (`sprint18-duration-accounting-v1`) — the additive slot

The frozen v2 schema cannot express this family, so it is added as a **new
closed record** (per branch, per seed, per history):

| Field | Definition |
|---|---|
| `schema_id` | `sprint18-duration-accounting-v1` |
| `evaluated_events` | evaluable P+W events at the frozen q95 threshold |
| `recalled_events` | of those, events with ≥1 file scoring `>= threshold` |
| `unrecalled_events` | `evaluated_events − recalled_events` |
| `median_declared_duration_days` | median accepted-ledger `duration_d` over evaluated events with finite `duration_d > 0`; `null` if none |
| `median_detected_alert_span_days` | median `(last − first)` flagged file end inside the event window, over recalled events, days; `null` if none recalled |
| `median_first_alert_lead_days` | median `E.lead_days(failure_time, flagged ends)` over recalled events; `null` if none |
| `total_detected_alert_days` | sum of detected alert spans in days (`0.0` if none) |
| `episode_grouping_days` | the inherited `<= 2`-day alert-episode grouping (`EPISODE_GAP_S / 86400`) |
| `events_without_positive_duration` | evaluated events whose ledger duration is missing/non-finite/`<= 0` |
| `excluded_events` | failures the canonical path excludes (reset-spanning horizon or no POS-ELIGIBLE file) |

Nothing is imputed: undefined medians are `null`, every exclusion is counted,
and no value depends on any observed supplementary outcome.

### 3.5 Severity ordering (`sprint18-severity-ordering-v1`)

Per branch and history, the within-cohort Spearman ρ of event score vs accepted
ledger `severity` for cohorts `P` and `W`; `null` below three points or on a
non-finite ρ.

### 3.6 Subtype slices (`sprint18-subtype-slices-v1`)

Per branch and history, the tie-aware event AUROC of `P1`, `P2`, `W1`, `W2`,
`A1`, `A2` events against the **same** control negatives used by 3.1; `null`
when the subtype has no evaluable event or no control exists.

## 4. Additive schema and strict consumer compatibility

**Input manifest** `sprint18-supplementary-input-manifest-v1` (closed): protocol
and supplement bytes, metric-code digest, source commit, arm, seed, scope
`experiment`, phase `development`, certified data root, Task 14 lock, certified
DATA handoff, accepted record, checkpoint, reference artifact
(`k9_downstream` | `n4_scale` | `none`), and the expected digests (arm config,
model/EMA state, DATA binding, bank, accepted thresholds, Development manifest
and support digests). Every referenced file is re-hashed and every digest
compared before inference.

**Score document** `sprint18-supplementary-scores-v1` (closed): per-file
`S_pred`/`S_pop` maps for Fit-healthy, Calibration-healthy, and the four
Development histories; compact localization diagnostics plus the timestep-vector
digest; support counts; and reproduction records for accepted/recomputed q95
thresholds, bank and provenance byte artifacts, `encoder_updates = 0`,
`representation_state_unchanged`, and `reference_state_unchanged`.

**Metric document** `sprint18-supplementary-development-v1` (closed):
`phase = development`, `arm_id`, `execution`, `inputs` (with the three seed
score documents and, for non-B0 arms, the three B0 baseline documents),
`reproduction`, `support`, `metric_definitions`, and `branches`. Each branch has:

- `thresholds_by_seed` — the three accepted q95 thresholds;
- `per_seed` — the raw 3×4 seed/history cells, each carrying the frozen v2
  metric fields **plus** `duration_accounting`;
- `per_history` — the four seed-averaged gate cells in **exact v2 shape**
  (validated directly by the unchanged `_validate_history_metric`);
- `aggregate` — exactly the frozen v2 aggregate keys, including
  `background_uncertainty`.

Compatibility is proven executable, not asserted: the per-seed row's v2
projection must validate under the unchanged v2 validator, and the complete row
(with `duration_accounting`) must still be refused by it
(`_assert_v2_refuses_supplementary_row`, run at import).

`VALID` additionally requires every family (including the duration medians and
the aggregate uncertainty) to be non-null on its fixed support, mirroring §10's
"null with an explicit status, never a fabricated number" rule.

## 5. Execution identity and publication

- Development-only supplementary execution identity, separate from the accepted
  arm-training execution: the score and metric documents carry `execution` with
  the supplementary task/attempt/agent, source commit, metric-code and
  supplement digests, device, and UTC window.
- Append-only: nothing under the accepted run directories, `artifacts/sprint-18/`
  primary records, `review-batch-d.md`, or any owner record is rewritten. New
  supplementary outputs are new digest-bound files.
- No Confirmation path, no optimizer/training, no alternative bank refit or
  recipe/options, no threshold/support/coefficient change, no scoring of accepted
  run outputs, and no significance or promotion claim.
- Fit-healthy branch-score maps are only the frozen stability-ratio denominator.
  The same accepted Fit membership also supplies the exact bank restoration in
  §1.1. Since no public Sprint 18 scoring entry point accepts FIT, the runner
  reuses the identical frozen forward/score math on rows loaded under the FIT
  `train_fit` permission; branch-score values never select or fit evaluation
  state or publish an evaluation result.

### 5.1 Exact accepted-input manifest preparation

Do not prepare normative manifests or run normative scoring until the fresh
C03 same-level source review, separate Bronze checkpoint, and isolated-server
synchronization have passed. The exact Task 14 lock, certified Task 71 handoff,
DATA root, accepted-record path map, and historical per-seed bank digests are
listed in `artifacts/sprint-18/task-32-source-c03.md`; do not infer or substitute
any of those inputs.

On the isolated synced worktree, invoke the public
`experiments/sprint18_task32_supplementary_runner.py prepare-manifest` CLI once
per accepted `(arm, seed)`, using its required `--arm`, `--model-seed`,
`--accepted-record`, `--task14-lock`, `--handoff`, `--data-root`,
`--source-commit`, and append-only `--out` arguments. The source commit must be
the exact lowercase 40-hex reviewed/synced commit selected after those gates;
C03 does not supply a future commit value. The generator derives checkpoint
paths from `primary.path` (or N4's paired
`representation_baseline.checkpoint_path`) and K9/N4 reference paths from the
accepted record. Optional path overrides are only for staged byte-identical
copies whose SHA-256 still equals accepted custody; B0/K3/K7/N2/N3 cannot name
a persisted reference artifact.

Before writing, the generator verifies the full certified Fit roster/order and
row digest, accepted bank digest/count, arm and checkpoint/model/EMA digests,
accepted q95 thresholds, Development manifest/support digests, and N4 final
scale histories/bounds plus paired-B0 `S_pop` threshold. Its output reports the
closed manifest SHA-256 and a complete `input_sha256` map of protocol,
supplement, source code, lock, handoff, accepted record, checkpoint/reference
artifacts, certified DATA binding, role manifests, and every referenced shard.
The subsequent score document and immutable bank provenance bind these exact
inputs again before/after scoring. Never hand-author or relax an input manifest.

## 6. What this supplement does not do

It does not produce Confirmation values, does not change any §8.3 gate, does not
turn a negative Development result into a recovery, does not resolve
`N1-HCC` (still STOPPED), and does not by itself make the eight-arm comparison
`VALID`. Publication still requires its own independent source review, a
separate Bronze checkpoint, isolated-server synchronization, an actual
frozen-checkpoint smoke, the bounded supplementary evaluation, fresh publication
gates, and a fresh Batch D review.
