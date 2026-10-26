# Sprint 24 weekly/monthly study — cache-budget amendment A01 (owner-authorized recapture)

- Amendment ID: `S24-WM-CACHE-AMENDMENT-A01`.
- Task / attempt / agent: `S24-7-M4` / `S24-7-M4-A01` / `S24M4GoldA01`.
- Governing study: `S24-BIZ-WM-v1`, freeze digest `6f96c2750c15552bcdf25678a51ed7b24a1f7aecdb5914bf5985d3012d1daefd`, frozen protocol [`sprint24-weekly-monthly-protocol-v1.md`](sprint24-weekly-monthly-protocol-v1.md) digest `8da4afda89dd303099843eecda546cdc337696525ece0eed59f400d1031d8b63`, base checkpoint `0a17ffa7927f0e2187cd262f7770e3dfbd7dbc48`.
- **Status: prepared for readiness review (`S24-7-M4-PRE-C1R1`). Not deployed, not executed, not an execution result.** This amendment changes only the forward-cap accounting and the capture/persistence route. It changes no endpoint, estimator, contrast, uncertainty rule, sensitivity variant, disposition rule, corpus, model or custody.

## 1. Owner decision recorded

The owner answered the fail-closed cache preflight with **"Capture lại, nâng cap lên 54.000"** (recapture; raise the cap to 54,000). Consequences recorded verbatim in this amendment:

- The inherited cumulative forward cap becomes **54,000** for this study, replacing the strict 41,000 cap **only as an owner-authorized accounting amendment**. The frozen protocol document and its digest are unchanged; the M3 study identity and every manifest digest are unchanged.
- Cumulative forwards already spent: **26,431** (accepted G2 accounting, not reset).
- Maximum additional forwards available: **27,569**.
- The authorized budget is **not** an automatic-retry budget. A failed or partial capture consumes what it executed and stops; no silent re-run is authorized.

## 2. What is authorized, and what is not

Authorized:

1. A bounded GPU capture of exactly the frozen study's required per-file B0 embeddings — 3,678 causal reference rows plus 5,076 analysis-span rows per representation seed, 26,262 rows across the three accepted seeds.
2. Persistence of those rows under the declared cache contract at the study-required cache identity, with custody hashes.
3. A bounded 96-forward smoke capture (8 files per history x 4 histories x 3 seeds) before the full capture, and an identity/coverage/admission proof before any movement value.
4. Task-owned commit/push of reviewed content to `F:/ai-ml/anomaly-representation-learning` (`master`, non-force) and an exact-commit remote capture run, per the owner's release.

Not authorized or changed:

- No endpoint, estimand, contrast, sensitivity, uncertainty, multiplicity or disposition-rule change. The frozen decision rule and its research defaults stand exactly as frozen.
- No new corpus, no retraining, no model or architecture change, no Confirmation contact, no scaled inference, no Phase C execution.
- No change to the frozen protocol document or its digest, no change to the M3 study digest, and no edit to Main-owned plans.
- No mutation of accepted G2 raw outputs, accepted manifests, frozen edges or the accepted Development corpus.
- No daily-horizon comparison and no reinterpretation of the owner's 500–1,000 files/day business context as current-corpus density.

## 3. Cache identity declaration (including the 13 `DEVELOPMENT-01` files)

The study-required cache identity remains the existing `s24_feature_cache_identity` record: checkpoint, bank tensor, support digest, preprocessing contract, arm config, data binding, model seed, scale state `none` and capture device `cuda`.

The **cache-input identity is new relative to G2** for three groups. The accepted G2 capture omitted the 13 files of its first `DEVELOPMENT-01` window because that window fell below the inherited 30-file floor. Those 13 files are inside the frozen study's required union (they are eligible files in `(c1, tmax]`), so the study's support digest for `DEVELOPMENT-01` differs from G2's captured support digest in all three seeds:

| History | Required rows | G2 captured rows | Uncaptured study rows | Required identity vs G2 |
|---|---:|---:|---:|---|
| `DEVELOPMENT-01` | 2,076 | 2,063 | 13 | differs (new cache-input identity) |
| `DEVELOPMENT-02` | 2,327 | 2,327 | 0 | equal |
| `DEVELOPMENT-03` | 2,053 | 2,053 | 0 | equal |
| `DEVELOPMENT-04` | 2,298 | 2,298 | 0 | equal |

This is declared as **exact study coverage**, not as a match to G2. The 13 files are named in `row-plan.json` and in `preflight.json.accepted_capture.per_history[...].uncaptured_study_rows`; the capture must include them and must record the support digest over the study's full 8,754-ID union per seed. A capture that reproduces G2's narrower union is refused by the admission gate.

## 4. Bounded cost plan

| Item | Forwards | Cumulative | Remaining under 54,000 |
|---|---:|---:|---:|
| Spent before this attempt | 26,431 | 26,431 | 27,569 |
| Declared smoke (8 files/history x 4 x 3 seeds) | 96 | 26,527 | 27,473 |
| Full capture (8,754 rows/seed x 3) | 26,262 | 52,789 | 1,211 |
| Full capture without a separate smoke | 26,262 | 52,693 | 1,307 |

The runner refuses any plan whose cumulative total would exceed 54,000. The 1,307 rows (or 1,211 with the smoke) are reserved for bounded checks only and are not a retry allowance. A partial capture consumes the rows it actually executed; the capture progress log preserves them.

## 5. Capture and persistence contract

- Command surface: `experiments/sprint24_task7m4_capture_runner.py` with modes `fixture` (local, no model), `smoke` (96 forwards), `full` (26,262 forwards). `full` requires a completed `smoke` record from the same execution commit.
- Store contract (shared with the admission CLI): `<history_id>__seed-<model_seed>.npz` with `file_ids` and `embeddings` (`float32 [n, 128]`) plus a JSON sidecar carrying the identity inputs, `cache_identity`, `n_rows`, `capture_provenance` and `capture_commit`.
- Provenance is enforced: only `gpu-forward` with a 40-character execution commit is admissible study input. `gpu-forward-smoke` and `fixture-no-forward` are deliberately inadmissible, so a smoke or local fixture can never satisfy the study.
- The runner restores the accepted B0 checkpoint and bank at their pinned file digests, validates the locked arm configs and binding, enables the frozen deterministic CUDA setting, asserts the live model/bank digests are unchanged after capture, and writes per-entry npz/sidecar hashes plus a capture progress log.
- No movement, contrast, uncertainty, sensitivity or disposition is computed in the capture runner.

## 6. Gates and sequence

1. **Readiness review `S24-7-M4-PRE-C1R1`** (two default rounds) of this amendment, the runner, the admission-gate changes and the local fixture/public-path smoke. No deployment before PASS.
2. **Fresh Bronze checkpoint** at the reviewed snapshot before any new code is deployed or executed remotely.
3. **Exact-commit deployment**: non-force push of reviewed content, a fresh empty remote checkout at the exact commit, clean-worktree verification.
4. **Bounded remote execution**: `smoke` first, then `full`, each with its own output root; preserve every failed or partial root and its cost.
5. **Identity and coverage proof** from the persisted store: 26,262/26,262 rows, 12/12 identity groups, device `cuda`, provenance `gpu-forward`, commit-bound hashes.
6. **Study admission** re-run through `experiments/sprint24_task7m4_weekly_monthly_execution.py --cache-root <store>`; only an `ADMITTED` result opens endpoint computation.
7. **Final evidence** `S24-7-M4-C1R1` (two default rounds) and a separate Bronze final checkpoint. No deep review.

## 7. Stop rules

- Any digest, asset, identity or device mismatch fails closed; no fallback identity, no CPU substitution, no imputation or interpolation.
- A failed capture is preserved with its executed-forward cost and stops the sequence; no automatic retry.
- Coverage shortfalls are reported exactly; the gate is never relaxed and no row is fabricated.
- Any need to change the frozen protocol, endpoint or decision rule stops the attempt for an owner decision rather than being edited in.

## 8. Limitations

- The 54,000 cap is an owner-authorized accounting amendment for this study; it is not a precedent for any other task and does not reset the inherited 26,431.
- A `cuda` identity cannot be produced on CPU, so the full capture must run on the accepted GPU host with the accepted assets. Local execution is limited to the fixture smoke and gate checks.
- The capture binds device `cuda`; CUDA results are equivalent to CPU only within the declared FP32 tolerance and are not bit-identical, so a CPU store would be refused.
- Once captured, the study remains exposed-benchmark exploratory research with no healthy/quiet controls; a later disposition still cannot claim calibrated specificity.

## 9. Reproduction

```bash
# Local fixture smoke (no model forward; refused by provenance by design)
python experiments/sprint24_task7m4_capture_runner.py --mode fixture \
  --data-root <accepted metadata root> --edges experiments/sprint24_task7m1_dev-edges.json \
  --g2-accepted-root <reviewed G2 record root> \
  --out artifacts/sprint-24/evidence/s24-7-m4-a01/capture-fixture \
  --cache-out artifacts/sprint-24/evidence/s24-7-m4-a01/capture-fixture/store \
  --expected-manifest-sha256 <history>=<sha256> ...

# Remote smoke, then full (exact commit; empty cache roots)
python experiments/sprint24_task7m4_capture_runner.py --mode smoke ... \
  --expected-execution-commit <40-hex reviewed commit>
python experiments/sprint24_task7m4_capture_runner.py --mode full ... \
  --smoke-record <smoke-report.json> --expected-execution-commit <same commit>

# Admission proof after the full capture
python experiments/sprint24_task7m4_weekly_monthly_execution.py ... \
  --cache-root <full store> --out <admission evidence root>
```
