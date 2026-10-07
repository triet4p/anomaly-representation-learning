# S18-T68-A09 durable operational proof (disposable only, no candidate contact)

**Task / attempt / agent:** `S18-T68` / `S18-T68-A09` / `S18Task68A09` (explicit `gold-task`; selector `opencode-go/muse-spark-1.3-contributor:xhigh`, `blocking: false`; provider-applied effort/quota unknown).
**Base:** `cdfc8263f1683dba7c4402709744d3cb6437ae1b` (parent `a17bfad5`).
**Machine record:** `experiments/sprint18-task68-attempt-S18-T68-A09-proof.json`.
**Normative contract:** `experiments/sprint18-iterative-data-contract-v3.md` `544de84b…` (untouched).

Roster scope for every check below: read-only local metadata (HEAD binding
bytes, A05 record SHAs, source/config digests). No role-path inspection,
existence checks, SSH, generation, or final-candidate contact.

## 1. HIGH — policy-authorized adoption of the verified old identity

- `_load_stage_state` keeps the strict fresh path (marker must equal the live
  binding fields) and, only when the marker differs, attempts adoption through
  `_adopted_old_stage_state`: the marker must equal the old fields
  (`37b14f54…` / `7002ebb6…` over the identical 16-role roster) and
  `_verify_old_state_bytes` must pass every byte check under the versioned
  policy. No generic fallback, alias, or bypass exists.
- The `materialize-nonconfirmation` dispatch runs
  `_enforce_freeze_only_resume_policy`, then appends exactly one
  `resume_adopted` event carrying old/new identities and the policy path, then
  proceeds to the genuine public CLI write. Marker and the three old ledger
  events stay immutable; the adoption is an appended operational event.
- `record-nonconfirmation-pass` (`record_nonconfirmation_result`) and
  `run_confirmation` load through the same adopted-or-fresh loader, keep the
  complete custody chain (frozen new identity, old-prefix verification, state
  permissions), and keep Confirmation gated on the Task69 PASS record plus the
  separate Main Task69 release. The pass-record verifier requires all 12
  `Design/Fit/Calibration/Development` roots; Confirmation additionally
  requires the `nonconfirmation_qualification_pass` event and the direct-child
  Task69 checkpoint.

## 2. HIGH — versioned immutable resume policy (acyclic)

- Restored `experiments/sprint18-iterative-resume-policy-v1.json`
  (`0f12ed2d…`, 15,787 bytes, LF): binds the immutable old identities
  (binding `37b14f54…`, file `f706e9ee…`, closure `7002ebb6…`, runner blob
  `1d6f9fda…`/canonical `b2012522…`, receipt `f68d3fb8…`, raw `8d6503f4…`,
  ledger `503840c7…`, marker `ac97872c…`, the three ledger-event payload
  digests, the shared 51-member old source maps, and the shared generation
  digests `97d3785d…`/`a28ef4c1…`/`4f0350fb…`/`5001e904…`/`0443b615…` plus
  common seed-independent `e933feb6…`), the same generator/config/roles/method
  equality requirement, the new base `cdfc8263`, the new runtime
  (`0.12.21`/`7af826859`, `3.12.13`, `2.5.2`/`1.18.1`/`2.14.0+cu130`), and the
  allowed operational scope (stages `materialize-nonconfirmation` +
  `record-nonconfirmation-pass`, max 12 non-Confirmation roles, Confirmation
  forbidden, explicit Main release required, append allowlist).
- Acyclic graph (no self-hash): the policy contains no digest of its own
  bytes and no digest value of the new binding (that value cannot exist before
  the policy bytes do). Canonical order: old bytes → old digests → policy
  fields → policy bytes → new closure digest → new binding digest → future
  Main receipt. The binding carries the policy only as governed content (this
  attempt keeps the 52-member closure pattern; the policy is the versioned
  sidecar the binding's `resume_rule` names). Live guards cross-check policy
  generation digests against live binding sections, policy runtime against
  live runtime, and policy base against `BASE_COMMIT` before any write.

## 3. HIGH — actual byte-hash validation of the old state

- `_verify_old_state_bytes` requires, before any stage write: marker file and
  canonical bytes `ac97872c…`; raw preflight file and canonical bytes
  `8d6503f4…`; ledger prefix `attempt_started/preflight_started/
  preflight_recorded` with the three event payload digests; ledger event-0
  equals the marker; ledger event-2 binds the raw digest; old receipt
  `f68d3fb8…` present under `artifacts/sprint-18/`; all 51 shared old source
  members equal their frozen old bytes (runner and binding are the only
  excluded entries); live binding sections equal the old generation digests.
  Any drift, dirty root overwrite, symlink, missing file, rejected/retired
  marker, partial or unexpected role refuses before any write.
- The exact old prefix is validated once; after the single `resume_adopted`
  append, subsequent loads re-verify the same prefix (adoption must
  immediately follow it) without loosening tamper detection. Retired
  candidates 1/2 stay refused; no overwrite and no accept-any-directory.
- Config/signal physics are unchanged bitwise: the live role/config/seed/
  method/collision digests equal the old ones; only the runner bytes, the uv
  pin already accepted in A08, and the new policy docs differ.
  `a17bfad5` base/parent is now correctly refused as a future release
  identity, while a corrected-base receipt passes identity. The Task69
  direct-child guard (`_validate_task69_checkpoint_parent`) is tested
  valid/invalid without Git mutation against the existing commits
  (`cdfc8263^ == a17bfad5`, so the old checkpoint cannot serve as a future
  parent). No source Git ref was edited.

## 5. Decisive disposable proof (structurally and cryptographically old state)

- Staged a disposable root with byte-copies of the verified old state
  (marker `ac97872c…`, ledger `503840c7…`, raw `8d6503f4…`, receipt
  `f68d3fb8…`, 51 shared source members): the resume gate passed, the adopted
  load returned the exact 3-event prefix, the `resume_adopted` append plus
  reload preserved the verified prefix, `_verify_preflight_pass` passed, and
  `_materialize_entries` for `S18I-ITER-0003-DESIGN-01` (seed 32032) exited
  `0` through the genuine public CLI and public reload (6,522 samples,
  manifest `9bd37fd994c5…`, ~166 s wall) with a post-write reload passing
  (6 events). Fixture coordinates are disposable; the only real-seed bytes
  used are the immutable old preflight/marker/ledger/receipt copies.
- Mutation matrix (every case refuses before any write): old marker, raw
  preflight, ledger prefix, receipt, live config/closure, dirty-root
  overwrite, stale lineage parent, missing policy, retired candidate.
  Corrected lineage parent passes identity; the Task69 direct-child guard
  passes valid and refuses skipped generations.

## 6. Regression and smoke

- `tests/synth/test_sprint18_stage_ledger_path.py` (6 tests: ledger-Path
  contract, fresh-path strictness, interrupted retire, stale-uv refusal,
  policy tamper refusals, stale-parent/direct-child guards): `6 passed`.
- `--stage validate --smoke-no-contact` on the refreshed binding: exit `0`,
  `disposable-static-no-contact` (Windows static proof only, not Linux proof).

## 7. Identities (final file state)

- Binding canonical `f019aad1d288b735f4ea7eb45cd557f2ded758131e2f2fcadfde2210cfe51c5a`
  (42,994 bytes; file `2161d6fe1b70f7310baefc25b28d52a2c9cb0e3d759a1e3bdf170e646e7efb5c`).
- Closure `53f0ffc25b84176bfaa311edfd87a7fa945d1872ac9d40bb9de8746d0f85d184`
  (52 members), runner canonical
  `55f2e8278022f2be4505cef85a9d22365c7f0c63d84068a2e0c2fc0f11ec83fd`,
  runner blob `fc73ce702f99cfa76fa992d96a801acff24987ac`, raw
  `325fb3b65…`.
- Policy `b601e8d79ac20a3014130d7e790f43a5c30515c72c4ccd1b26126913670e1fef`
  (15,787 bytes). Old identities as above. `BASE_COMMIT` /
  `required_checkpoint_parent` = `cdfc8263…`.
- No commit, no review/checkpoint/transport/Main receipt by this worker.
  Candidate 3 actual resume is NOT performed. Future Task69 success
  checkpoint must directly parent the future Task68 checkpoint; only the 12
  non-Confirmation roles run until Task69 PASS plus checkpoint plus Main
  release; no Confirmation stage until then.
