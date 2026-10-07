# S18-T68-A11 durable proof — whole-fresh candidate-4 cutover (disposable proof only, no candidate contact)

**Task / attempt / agent:** `S18-T68` / `S18-T68-A11` / `S18Task68A11` (explicit `gold-task`; selector `opencode-go/muse-spark-1.3-contributor:xhigh`, `blocking: false`; provider-applied effort/quota unknown).
**Base:** `cdfc8263f1683dba7c4402709744d3cb6437ae1b` (parent `a17bfad5`).
**Machine record:** `experiments/sprint18-task68-attempt-S18-T68-A11-proof.json`.
**Normative contract:** `experiments/sprint18-iterative-data-contract-v3.md` `544de84b…` (untouched).
**Method:** `sprint18-iterative-v3` unchanged — 450-day, 225 cutoff, 2880 units, fixed-phase v3 hash, Hazard RNG, rates/count/noise/cadence/routes/maintenance, support-mix/lead/quota controls-min-48-allows-49+/hard-25/Design-32/full-ELIGIBLE-15-60/PW12A8.

Final-candidate contact in this attempt: NONE. No `32048–32063` path inspection, existence check, preflight, waveform, qualification, or probe occurred. No candidate-1–3 contact/retry. No remote work. No Git mutation.

## 1. Custody retirement (append-only amendment, no rewrite)

- Main retired WHOLE `S18-ITER-0003` for scientific custody after R08 HIGH Finding 1 (`history://S18Task68A09` turns 604/648: genuine public CLI materialization of `S18I-ITER-0003-DESIGN-01`, seed `32032`, exit `0`, 6,522 samples, manifest `9bd37fd994c5…`, under a throwaway root while claiming `contact_in_this_attempt: false`). Retirement is for custody, NOT a scientific quota/probe failure.
- New versioned owner record `experiments/sprint18-candidate3-custody-retirement-v1.json` (canonical `d07d829b…`, raw `1901c3c6…`) states the Main decision, reason, primary R08 evidence, old identities (marker `ac97872c…`, ledger `503840c7…` 3 events, raw `8d6503f4…`, receipt `f68d3fb8…`, binding `f019aad1…`, closure `53f0ffc2…`), observed time, and the no-resume/no-promotion/no-credit/no-reuse boundary. Old preflight `16/16` PASS stays honestly recorded as historical failed-attempt evidence; remote `0` roles stays true.
- Old markers/3 events/raw receipt bytes preserved; no fictional event appended at old timestamps; no old log overwritten. Historical `experiments/sprint18-iterative-resume-policy-v1.json` and A08/A09 proofs preserved byte-identical with ZERO active binding/closure/runtime references.

## 2. Whole-fresh candidate-4 binding (metadata only)

- `S18-ITER-0004`, number 4, seeds `32048–32063` (`32000+16×(N−1)`, N=4), exact 16-role order D4/F3/Cal1/Dev4/Conf4, current v3, `contact_authorized: false` until a new Main receipt in later Task 69.
- All 16 configs resolved deterministically through the live v3 factory (pure metadata, no waveforms): seed-independent digest UNCHANGED `e933feb61e94d39fb56a45fae742fa1222b321b238dd1f3608642418be279136`; shorts `32048=5531879ce27b … 32063=be7c512f87e3`.
- Collision catalog folds retired `32032–32047` into `32000–32047`: 405 unique excluded, `f1a5d0b6…`; candidate `32048–32063` intersection `[]`; proof seeds `94100/94101` are NOT excluded (disposable, never bound).
- Strict whole-fresh `attempt_policy`: `no_resume: true` + `no_resume_rule`; `resume_rule` removed; interrupted roots retire the candidate.
- 53-member closure (52 prior + retirement record): `2f939672793a37650b288df949c59406aaad2a786dc271679017b263d6a58b32`; binding `a5908b53e8567a0ed5cb9baace3415e135681c99c99e70145d2e6354dccd3dd1` (43,980 bytes; file `612899fe…`); runner canonical `f7b6aab7…`, blob `71f0cbd9…`; uv `0.12.21`/`7af826859`; `BASE_COMMIT`/`required_checkpoint_parent` = `cdfc8263…`; Task67 provenance `e576debd…`.

## 3. Runner cutover (no physics change)

- Removed ACTIVE candidate-3 resume machinery entirely: adoption loader, event path, all 9 `OLD_*` constants, `_read/_verify_resume_policy`, `_verify_old_state_bytes`, `_enforce_freeze_only_resume_policy`, `resume_adopted` branches. Strict `_load_stage_state` restored (any marker mismatch refuses).
- `_verify_preflight_pass` generalized from hardcoded `16/16` to the bound roster size (same `16/16` semantics for the final 16; fixture rosters verify at their own size; no gate loosening — pass-record still requires the exact 12-role non-Confirmation prefix).
- New minimal public `dispatch_stage(binding, root, stage)` owns the full materialize-nonconfirmation control flow (load → guard → loop → reload, current bytecode); the CLI `--stage` entry delegates to it, so both paths are identical. No new product abstraction beyond this dispatcher.
- Retained: ledger-Path contract, live uv guard (fail-closed before contact), venv `3.12.13`/`2.5.2`/`1.18.1`/`2.14.0+cu130`/CUDA13/RTX4060Ti, 53-file closure guards, `cdfc8263` ancestry, Task69-direct-child guard.

## 4. Decisive disposable proof (public dispatcher, disjoint coordinates)

- Arithmetic disjointness BEFORE any data: `{94100,94101} ∩ [32000..32063] = []` (min distance 62,037).
- Fixture: 2-role roster `S18I-PROOF-0001-DESIGN-01`/`FIT-01` at seeds `94100/94101` (never slicing final binding roles) under an isolated temp proof root (removed after durable proof).
- In-memory `run_preflight_iterative_v3` PASS `2/2`, no writes; `dispatch_stage` exit `0` (422.76 s wall): ledger `attempt_started → preflight_started → preflight_recorded → started/materialized × 2`; genuine public CLI writes (`6472` + `6451` samples; manifests `e8c5f47f…`/`7641c60d…`) and public `load_chronological` reloads with role/protocol/config-hash checks.
- Follow-up `_load_stage_state` + public reload PASS; retired-seed (`32032`) load refuses; tampered-marker load refuses. Honest accounting: disposable waveforms WERE generated (fixture only); ZERO final-candidate contact.
- Exact command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<root>/src:<root>/experiments:<root> <root>/.venv/Scripts/python -B s18-t68-a11-proof-harness.py` (throwaway harness, removed after proof).

## 5. Regression and smoke

- `tests/synth/test_sprint18_stage_ledger_path.py` (7 tests: ledger-Path contract, public-dispatcher identity, fresh-path strictness, interrupted retire, stale-uv refusal, whole-fresh candidate-4 binding/collision, stale-parent/direct-child guards): `7 passed`.
- `--stage validate --smoke-no-contact` on the frozen binding: exit `0`, `disposable-static-no-contact` (Windows static proof only, not Linux proof).
- Guard paths exercised: stale uv pin refuses, tampered marker refuses, interrupted materialization retires with no resume.

## 6. Identities (final file state)

- Binding canonical `a5908b53e8567a0ed5cb9baace3415e135681c99c99e70145d2e6354dccd3dd1` (43,980 bytes; file `612899fe496740745f7d3ef2621fad82c20897b1499012aca66656543a1bc78a`).
- Closure `2f939672793a37650b288df949c59406aaad2a786dc271679017b263d6a58b32` (53 members), runner canonical `f7b6aab70673c87f6c6965bedb1cfdab78674c04961ccb3d6e6043e6cf6167ef`, runner blob `71f0cbd91cbc4a5cd4f7fe72e3125d50323fe93f`, raw `fc6dc83a…`.
- Retirement canonical `d07d829b…` (2,783 bytes), blob `93a70721…`. Test `e292967a…` (9,538 bytes).
- `BASE_COMMIT` / `required_checkpoint_parent` = `cdfc8263…`.
- No commit, no review/checkpoint/transport/Main receipt by this worker. Future Task69 success checkpoint must directly parent the future Task68 checkpoint; only the 12 non-Confirmation roles run until Task69 PASS plus checkpoint plus Main release; no Confirmation stage until then.
