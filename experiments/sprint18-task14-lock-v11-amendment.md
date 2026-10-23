# Sprint 18 Task 14 lock v11 — Task 30 paired Development B0 support/identity join

- **Status:** prospective candidate; **no self-approval**. Independent source-evidence review, a
  separate Bronze exact-snapshot checkpoint and non-force source/sync are required before any
  normative runtime.
- **Task / attempt:** `S18-T30` / `S18-T30-A17`
- **Base:** `2b0cb38c33c0022412e5dd57f424a2c6c053aea2` (`refs/heads/master`), parent
  `6ffcfc9f4ab339604ff201abb75f9d18fb12466d`.
- **New lock:** [`experiments/sprint18-task14-config-lock-v11.json`](./sprint18-task14-config-lock-v11.json),
  schema `s18-task14-config-lock-v11`, lock version 11, file
  `ba93edda451fcf66607a49b00a88c5f9ad54e9af0f581e8ae568cf482b9bfed7`, core
  `00b293dc41943b99180e7e91ab8d80f08428ec0fcc4bacb4cfa0cf122ad34d6a`.
- **Preserved history:** the v10 prospective lock (raw `d0a2dc09…`, core `55299145…`), the v9
  prospective lock (raw `a9990bd1…`, core `b171e500…`) and every earlier lock remain byte-identical
  through the **v10 → v9 → v8 → v7 → v6 → v5 → v4 → v3 → v2 → v1** chain.

## Why a new lock version

The A16 authorized report-only run executed the reviewed v10 source against the immutable A14
artifacts, authenticated the report source, the contract files, the live lock chain, the certified
handoff, the accepted B0 ledger, the A14 seed records and every bound artifact digest, and then
failed closed in the paired consumer:

```
DRIVER_FATAL accepted B0 development support lacks role_manifest_sha256 for
S18I-ITER-0005-DEVELOPMENT-01
```

`_paired_development_roster` read `role_manifest_sha256` / `schema_history_id` from the accepted B0
`development.support.per_history[history_id]` record. The accepted ledger stores exactly three keys
there — `branch_provenance_sha256`, `file_ids`, `support_sha256` — while the certified identity lives
on the two B0 branch rows (`development.branches.S_pred/S_pop.per_history[*]`, the frozen 17-key row
schema). The v10 unit fixture had modelled the support record as an identity dict, so neither the
suite nor the tiny fixture could catch the mismatch. This is a report-schema/consumer defect only:
no metric value, threshold, support digest or scientific identity changes.

## Exact correction

`experiments/sprint18_task30_n4_runner.py`:

- `_paired_development_roster` now derives the per-history identity through two new helpers.
  `_accepted_b0_branch_rows` indexes the accepted B0 `S_pred`/`S_pop` per-history rows by
  `history_id`, requiring the frozen 17-key row schema, the exact frozen roster, and no duplicate or
  foreign history. `_accepted_b0_identity` reads `role_manifest_sha256`, `schema_history_id` and
  `support_sha256` from both branches and fails closed unless they agree exactly.
- `_check_b0_support_record` validates each accepted B0 support record as exactly
  `branch_provenance_sha256` / `file_ids` / `support_sha256`; the file-ID list must be a non-empty
  list of unique non-empty strings, the branch provenance must carry both branch names with distinct
  SHA-256 digests (the public harness contract), and the support digest must equal both the arm's
  validated per-history support and the digest on the B0 branch rows. Missing, duplicated, foreign,
  schema-drifted or support-mismatched records fail closed; row order stays irrelevant because rows
  are matched by authenticated identity.
- `_enrich_report_development` reuses `_accepted_b0_branch_rows`, so the report path refuses a
  duplicated or schema-drifted B0 branch row instead of silently collapsing it.
- `_SUPERSEDED_LOCK_SCHEMAS` now also accepts the superseded v10 lock schema.

`experiments/sprint18_task13b_n4_capacity.py`: the versioned lock consumer moves to v11 only (no
fallback), validating the complete v11 → v10 → v9 → v8 → … → v1 chain, reporting
`historical_v9_lock`, and refusing every stale schema/version.

`tests/representation/test_sprint18_task14_lock_v11.py` supersedes
`tests/representation/test_sprint18_task14_lock_v10.py` (clean cutover; the v10 bytes remain pinned
and unchanged). `tests/representation/test_sprint18_task30_runner.py` uses the actual accepted B0
support shape (three keys) plus the 17-key branch identity rows, and adds regressions for missing,
duplicated, cross-branch-disagreeing and support-drifted records plus order-permutation acceptance.

## Reviewed implementation pins

| Path | SHA-256 |
|---|---|
| `experiments/sprint18-context-residual-normalization-spec-v3.md` | `e7fe42e41832ae283614ee6a76c0623ee567068dcc540a222f5b66a9d4262da2` (unchanged) |
| `experiments/sprint18_task30_n4_runner.py` | `03cbd9239d439f5a2a090f3ec752f7d3cf73cc60df29ce5249867e2ea7143820` (corrected) |
| `src/representation/__init__.py` | `9c3c857a76acebea08f4d6eebede4af7799024cdac2f7193c2d0601fdbcbf09b` (unchanged) |
| `src/representation/crn.py` | `7902dd9b7209b2ad9f7ef0f0e6f7316eb13c7db20237bc04974336b863bb9251` (unchanged) |
| `src/representation/sprint18_checkpoint.py` | `fce263df093bc622ba0a69bfac0824941fd6a0d1c4c9ab02bcbcff62432aaef2` (unchanged) |
| `tests/representation/test_sprint18_crn.py` | `2203d3f0b74be7ddb9f1f693b1321834b9078e86d990b6764996187218809186` (unchanged) |

`runner_identity` binds `experiments/sprint18_task13b_n4_capacity.py` at
`fe433350981b6a0d8315380c4918a1b4a71937d1508d39d76f0fe390498fa4b6` (its bytes at the reviewed
source commit `2b0cb38c…` are `6c3542cf…`). The superseded v10 runner pin `d8b70a83…` is retained
under `review.superseded_v10_implementation_pins`, alongside the preserved v9 (`3af994ba…`), v8
(`a1110f77…`), v7 (`ad7fd861…`), v6 (`4bb560e1…`), v5 (`19331f57…`), v4 (`f24e825e…`) and v3
(`e9237327…`) pins.

## What does not change

- The frozen scientific recipe, the certified N4 scope (66,989 records; 6,550 final /
  4,400-4,300-4,450 fold steps), the v3 mechanism amendment, the v1/v2 specifications, the
  final-B0/cache identities, the role rosters and permissions.
- The capacity recertification identity comparison, the certified-envelope-root derivation and the
  nine declared fold container digests.
- Every arm other than N4-CRN; the historical v1 lock continues to bind B0/K3/K7/K9/N1/N2/N3.
- The A14 scientific artifacts: `seed_records.json` `ed58bd99…`, the accepted B0 ledger
  `3540ef09…`, the A14 binding `6ad4da1f…` and the 15 checkpoint artifacts are immutable inputs.

## Reuse boundary (for the independent review)

The A14 scientific campaign is valid at the record/checkpoint/score-map layer and was independently
audited (`AUDIT_OK: true`). This delta touches only the report-schema/consumer path and executes no
scientific code path. Whether the immutable A14 artifacts may be reused through `--mode report`
instead of a fresh campaign remains **the independent reviewer's decision**, not this attempt's.
This attempt claims no PASS and no reuse authorization.

## Local proof (source evidence only)

- Real-records paired consumer end to end on the exact mirrors: A14 `seed_records.json`
  `ed58bd99…`, accepted B0 ledger `3540ef09…`, A14 binding `6ad4da1f…`. Output
  `paired-development-real-a17.json` `422d5554…`: 24 seed rows, 216 metric cells verified against the
  raw records, `S_pop` delta exactly zero, identity taken from the real B0 branch rows.
- The real public `--mode report` entrypoint (actual subprocess) on the real record shape: exit 0,
  `paired_development.json` `897bd0dd…`, `report_record.json` `fe6cd770…`, `manifest.json`
  `73e56766…`, with the real preflight/environment/progress, the real accepted B0 ledger and the
  real A14 original binding. The three per-seed custody artifact bytes are absent on this
  workstation, so they are explicitly labelled TEST-ONLY temporary fixtures pinned in a TEST-ONLY
  report-auth document; no real checkpoint authority is claimed and the immutable A14 root is
  unmodified.
- No training, optimizer, scoring or profiling call was executed.

## Required gates before any normative run or reuse

1. Independent source-evidence review of this lock, the six pinned paths, the amendment and the
   correction.
2. Separate Bronze exact-snapshot checkpoint of the reviewed source.
3. Non-force source push and isolated fast-forward sync.
4. A fresh authorized report/audit-only run on the exact A14 artifacts (server root, which holds the
   15 checkpoint artifacts and score maps), or otherwise a fresh campaign; then the terminal
   independent review.
