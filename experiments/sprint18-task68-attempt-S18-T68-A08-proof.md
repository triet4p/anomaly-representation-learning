# S18-T68-A08 durable operational proof (disposable only, no candidate contact)

**Task / attempt / agent:** `S18-T68` / `S18-T68-A08` / `S18Task68A08`.
**Base:** `cdfc8263f1683dba7c4402709744d3cb6437ae1b` (parent `a17bfad5`, tree `c063a438`).
**Machine record:** `experiments/sprint18-task68-attempt-S18-T68-A08-proof.json`.

## 1. HIGH — ledger Path contract (all affected stage callers)

- `materialize-nonconfirmation` dispatch, `run_confirmation`, and
  `record-nonconfirmation-pass` now take the ledger as a `Path` from
  `_candidate_paths` while the event list comes from `_load_stage_state`.
  `_fail_candidate`, `_materialize_entries`, and `run_preflight` variables were
  renamed to `ledger_path` forms; no exception suppression, no special-casing,
  no aliases. A final sweep finds no `attempt_root, ledger = _load_stage_state`,
  no `read_ledger(ledger)`, no `append_event(ledger,` and no
  `_fail_candidate(ledger,` remaining.
- Fresh-process disposable proof (temporary root only, never the bound candidate
  root): seeded a durable marker + 3-event ledger
  (`attempt_started/preflight_started/preflight_recorded`) with a crafted
  `preflight-raw.json`, then ran the corrected dispatch lines verbatim —
  ledger reread returns the exact 3 events, and the old events-list pattern
  reproduces `AttributeError: 'list' object has no attribute 'is_symlink'`.
- Genuine materialization proof in a fresh process on the disposable `93000`
  fixture block: `_materialize_entries` for the first role exited `0` through
  the real public CLI and real public loader — role root + `manifest.json`
  written (total 6463 samples), loader returns
  `S18I-ITER-SMOKE-0001-DESIGN-01` with matching `config_hash`, ledger tail
  `role_materialization_started,role_materialized` (191.48 s wall time).
  Interrupted-ledger proof retires instead of resuming (`candidate_rejected`
  appended). Confirmation/pass-record callers refuse fail-closed
  (`FileNotFoundError` on bogus record; `GuardError` with zero preflight),
  never with `AttributeError`.
- Permanent regression `tests/synth/test_sprint18_stage_ledger_path.py`
  (3 tests): consumer Path-vs-list contract, interrupted-materialization
  retirement, stale host-uv pin refusal. Focused suite observed
  `29 passed, 1 skipped in 59.49s` before the final non-behavioral doc/hash
  sync; final file state re-verified with the 3 regression tests passing.
  The 1 skip is a pre-existing conditional skip in the chronicle suite,
  unrelated to this change.

## 2. MEDIUM — host uv re-pin + fail-closed guard

- Binding re-pinned to the observed `0.12.21` / `7af826859` (platform
  `x86_64-unknown-linux-gnu`, date `2026-09-29`); stale `0.12.20` / `2274b80d6`
  appears only inside the smoke's stale-pin refusal case.
- New `_host_uv_callsign` parses the real 5-token probe shape (and the short
  4-token shape), and `_validate_host_uv_toolchain` runs on the normal
  `validate` stage (reported in `STATIC_BINDING_PASS`) and before every
  contact stage, refusing before `validate_runtime`. No install, rebuild,
  bypass, or JSON-echo proof.
- Discriminating proof on current bytes: pinned `0.12.21` probe accepts;
  stale-binding pin refuses; wrong host version refuses; nonzero probe exit
  refuses; the real local toolchain (`uv 0.9.7 (0adb44480 …)`) parses and
  refuses with `host uv toolchain differs from the bound provider`.
  Venv/runtime (`CPython 3.12.13`, `numpy 2.5.2`, `scipy 1.18.1`,
  `torch 2.14.0+cu130`, CUDA 13, RTX 4060 Ti) untouched. The real Linux normal
  guard rerun stays a Task 69 prerequisite.

## 3. STRICT — freeze-only resume of the exact existing candidate

- `_enforce_freeze_only_resume_policy` runs at the top of the
  `materialize-nonconfirmation` dispatch: it requires candidate
  `S18-ITER-0003` with seeds `32032–32047`, self-consistent live binding and
  closure digests, the corrected (non-`cdfc8263`) runner bytes, the exact
  durable 3-event preflight ledger with zero role events, and no
  rejected/retired marker. It refuses drifted bindings, dirty/interrupted
  ledgers, and any Confirmation attempt (Confirmation stays forbidden until
  a Task 69 PASS + checkpoint + Main release). Retired candidates 1/2 stay
  retired; no reseed, replacement, compat fallback, or retroactive edit.
- Proof on current bytes: exact-ledger accept, drifted-binding refusal,
  role-started-ledger refusal. The versioned sidecar file was removed to
  avoid a circular self-hash; the binding file itself carries the policy
  (old identities as code constants, new identities as live digests).
- No actual resume was performed: candidate 3 has zero contact in this
  attempt (no preflight rerun, no materialization, no `Conf`, no role-path
  inspection/existence check, no candidate-4 binding, no SSH).

## 4. Identities (current file state)

- Binding canonical `246de042482e87adbea1e25e871bd9f7120d00a1beb83422a58bb29fed8f2c0f`
  (43,276 bytes; file `1ebb73daf78c1c7c60c498a41352b33227e912e59c1b52f331187d9a45d4d972`).
- Closure `9bdf0e3579e8d5d919c36afa3aaa1d8ed7d03155cc6956de3d25597a2d8e33db`
  (52 members), runner canonical
  `c6c50a14398b80f613f3fa576dea9fed461e7f136d6a301da6b47e2837536f61`,
  runner blob `2d6347028fb2f5fd9b012f7d1aca5d49e7307593`.
- Old (refused-checkpoint) identities preserved as code constants and in the
  A08-R0 baseline: binding `37b14f545237d13010fd5a230c34782918b090123eabc7ff68814393e87cea29`,
  closure `7002ebb68889695527df95f3603ea2d64eb0c1f2d80f847d4d092b8da9a9d840`,
  runner blob `1d6f9fda0b43eccc2ebef59a456934d2bc8c2e5e`.

## 5. Files

- `experiments/sprint18_iterative_candidate_v1.py`,
  `experiments/sprint18-iterative-binding-v1.json`,
  `tests/synth/test_sprint18_stage_ledger_path.py`, `CHANGELOG.md`,
  `experiments/sprint18-task68-attempt-S18-T68-A08-proof.json` (+ this file),
  `artifacts/sprint-18/task-68.md#A08` (ignored, never checkpointed).
- No commit, no review/checkpoint/transport/Main receipt by this worker.
  Candidate 3 actual resume is NOT performed.
