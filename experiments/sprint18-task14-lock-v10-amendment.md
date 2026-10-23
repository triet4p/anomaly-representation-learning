# Sprint 18 Task 14 lock v10 — Task 30 Development per-history row identity/provenance contract

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A15`
- **Base:** `6ffcfc9f4ab339604ff201abb75f9d18fb12466d` (`refs/heads/master`), parent `1d9a646e9f57051fdc582e950919905f1dd70326`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v10.json`](./sprint18-task14-config-lock-v10.json), schema `s18-task14-config-lock-v10`, lock version 10, file `d0a2dc09e1048b1f857bdb326ba7360d43abb70ea1a45e98967e44338b43c5d0`, core `552991453e501d6fa48e514ae6a6adca1cafad284ce9843d17192a3afbb48396`.
- **Preserved history:** the v9 prospective lock (raw `a9990bd1…`, core `b171e500…`), the v8 prospective lock (raw `8258a439…`, core `e4482efb…`) and every earlier lock remain byte-identical through the **v9 → v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1** chain.

## Why a new lock version

The A14 full campaign completed all three frozen seeds (`181801`, `181802`, `181803`), wrote `seed_records.json` (`ed58bd99bb36a8eef513c03070ca56f337018bb728b87393c7187e8446234220`) and all 15 checkpoint artifacts, and then failed closed in the terminal paired Development report:

```
runner.py:2823 main -> run_full -> :2734 paired = _paired_development(
-> :2075 arm_histories = {row["history_id"]: row for row in arm_branch["per_history"]}
KeyError: 'history_id'
```

`_score_metrics` emitted `development.branches[*].per_history` rows carrying only the 13 frozen metric keys, while `_paired_development` indexed those rows by `row["history_id"]`. The accepted B0 rows have always carried `history_id` plus `role_manifest_sha256` / `schema_history_id` / `support_sha256`, which is why the comparison path looked valid. This is a public report-schema contract defect, not a metric-value defect: the A14 metric values are unchanged and independently reproducible.

## Exact correction

`experiments/sprint18_task30_n4_runner.py`:

- `_score_metrics` takes a **required** authenticated `roster` and emits the four identity fields on every per-history row. The roster entries carry `history_id`, `role_manifest_sha256`, `schema_history_id`, `support_sha256` and the canonical digest of the evaluated file-ID sequence; the join fails closed if that digest differs from the rows actually evaluated, if a history is duplicated, or if a field is missing.
- `_development_proof` builds the roster in the same authenticated loop that evaluates the certified Development roster: identity from the Task6 binding, support from the public scoring support just validated against the accepted B0 record, and the file-ID digest from the exact manifest rows. The accepted-B0 parity recheck now matches by `history_id` rather than by position.
- `_paired_development` indexes both arms strictly by authenticated identity (`_paired_development_roster` + `_paired_history_rows`), refuses missing, duplicated, foreign, provenance-mismatched or support-mismatched rows, accepts any row order, and adds the per-seed record/failure state plus an explicit "no paired confidence interval" aggregation block. The paired schema advances to `s18-t30-n4-paired-development-v3`.
- A new public **report/audit-only** mode (`--mode report`, `run_report`) consumes an already-executed full run. It performs **no fitting, no optimizer step and no model scoring**; it re-verifies source/lock/binding/contract/B0 plus the recorded nine-fold certified-DATA custody, the preflight/environment/progress logs, the seed records, the seed-record copies and every recorded checkpoint/score-map digest against a pinned report-auth document, joins legacy rows through the accepted B0 identity rows and the authenticated roster, emits a derived `paired_development.json` through the same `_paired_development` consumer, and writes a report record plus a manifest that separates the original scientific source from the report source. Consumed bytes are re-hashed unchanged after writing.

`experiments/sprint18_task13b_n4_capacity.py`: the versioned lock consumer moves to v10 only (no fallback), validating the complete v10 → v9 → v8 → … → v1 chain and refusing every stale schema/version.

`tests/representation/test_sprint18_task14_lock_v10.py` supersedes `tests/representation/test_sprint18_task14_lock_v9.py` (clean cutover; the v9 bytes remain pinned and unchanged). `tests/representation/test_sprint18_task30_runner.py` gains the identity/paired/report regressions.

## Reviewed implementation pins

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `d8b70a83283608ff96e250094f1b9f94f68df15933b475a27c4c3b2d3c2f8260` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at `6c3542cfda05ebc72ab98c12f3077562f9754339b79fce4b5e72f648dddb8084` (its bytes at the reviewed source commit `6ffcfc9f…` are `02e0d109…`). The superseded v9 runner pin `3af994ba…` is retained under `review.superseded_v9_implementation_pins`, alongside the preserved v8 (`a1110f77…`), v7 (`ad7fd861…`), v6 (`4bb560e1…`), v5 (`19331f57…`), v4 (`f24e825e…`) and v3 (`e9237327…`) pins.

## What does not change

- The frozen scientific recipe, the certified N4 scope (66,989 records; 6,550 final / 4,400-4,300-4,450 fold steps), the v3 mechanism amendment, the v1/v2 specifications, the final-B0/cache identities, the role rosters and permissions.
- The capacity recertification identity comparison, the certified-envelope-root derivation and the nine declared fold container digests.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3.
- The A14 scientific artifacts: `seed_records.json` `ed58bd99…`, the 15 checkpoint artifacts and the Development score maps are immutable inputs.

## Reuse boundary (for the independent review)

The A14 scientific campaign is valid at the record/checkpoint/score-map layer and was independently audited (`AUDIT_OK: true`). This delta touches only the report-schema/consumer path and adds a report-only mode; it executes no scientific code path. Whether the immutable A14 artifacts may be reused through `--mode report` instead of a fresh campaign is **the independent reviewer's decision**, not this attempt's. This attempt claims no PASS and no reuse authorization.

## Required gates before any normative run or reuse

1. Independent source-evidence review of this lock, the six pinned paths, the amendment, the correction record and the report/audit-only entrypoint.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. A fresh authorized report/audit-only run on the exact A14 artifacts (if the reviewer accepts reuse), or otherwise a fresh campaign; then the terminal independent review.

## Local proof limits

The local proof is a tiny CPU-only end-to-end report proof on tiny real checkpoint bytes, tiny seed records and tiny score maps. It drives the actual `--mode report` entrypoint, the actual `_paired_development` consumer and the actual authenticated-roster join, and it re-hashes the consumed bytes unchanged; it is **not** certified-DATA or full-campaign evidence.
