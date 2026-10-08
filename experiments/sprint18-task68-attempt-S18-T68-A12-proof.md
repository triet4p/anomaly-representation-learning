# S18-T68-A12 durable proof — public CLI metadata fix + whole-fresh candidate-5 cutover (disposable proof only, no candidate contact)

**Task / attempt / agent:** `S18-T68` / `S18-T68-A12` / `S18Task68A12` (explicit `gold-task`; selector `opencode-go/muse-spark-1.3-contributor:xhigh`, `blocking: false`; provider-applied effort/quota unknown).
**Base:** `2b1bf310d15bb682e0278ae59ae64f62bb67589c` (parent `cdfc8263`, tree `01f3f068` per assignment).
**Machine record:** `experiments/sprint18-task68-attempt-S18-T68-A12-proof.json`.
**Normative contract:** `experiments/sprint18-iterative-data-contract-v3.md` `544de84bc201a07140559058d100c991cd148b698f93058dbd37d7fe4dc3c929` (untouched).
**Method:** `sprint18-iterative-v3` unchanged — 450-day, 225 cutoff, 2880 units, fixed-phase v3 hash, Hazard RNG, rates/count/noise/cadence/routes/maintenance, support-mix/lead/quota controls-min-48-allows-49+/hard-25/Design-32/full-ELIGIBLE-15-60/PW12A8.

Final-candidate contact in this attempt: NONE. No `32064–32079` path inspection, existence check, preflight, waveform, qualification, or probe occurred. No candidate-1–4 contact/retry. No SSH. No commit/push/deploy.

## 1. Operational retirement (append-only amendment, no rewrite)

- Main retires WHOLE `S18-ITER-0004` for operations after Task 69 A06 / review R07: the eager `version(_DISTRIBUTION_NAME)` probe inside `build_parser()` refused every CLI invocation (including `--help`) on the deployed Linux worktree whose `.venv` holds third-party `dist-info`s but no application `dist-info`, before the first role write; `validate_runtime` passed exit 0 without exercising the public CLI parser. Retirement is operational/environmental, NOT a scientific quota/probe failure.
- New versioned owner record `experiments/sprint18-candidate4-operational-retirement-v1.json` states the Main decision, reason, primary R07 evidence, old identities (marker `5a4d9884…`, ledger `8da8d907…` 5 events ending `candidate_rejected: public_cli_failed:S18I-ITER-0004-DESIGN-01:1`, raw preflight `de3ef9f5…` 16/16 PASS, receipt `2b21ffb1…`, binding `a5908b53…`, closure `2f939672…`), and the no-resume/no-promotion/no-credit/no-reuse boundary. Old 16/16 preflight PASS stays honestly recorded as historical failed-attempt evidence; remote `0` roles stays true.
- Old markers/ledgers/raw receipt bytes preserved; no fictional event appended. Historical resume policy and A08/A09/A11 proofs preserved byte-identical with ZERO remaining active binding/closure/runtime references to retired candidates.

## 2. Public CLI contract fix (boring, evidence-backed, no fallback)

- `src/synth/cli.py`: new `_LazyDistributionVersionAction` resolves `version(_DISTRIBUTION_NAME)` lazily inside `__call__` only when `--version` is actually used. `build_parser()` therefore constructs (and `--help` plus all generation commands run) with no installed package metadata. On success the version prints to **stdout** exactly like the native `argparse` version action (exit 0, no side effects — original `test_cli.py` stdout assertions preserved byte-identical). When metadata is absent, `--version` reports the honest `package metadata is unavailable: ...` error on stderr (exit 2) with NO fabricated fallback version, NO hardcoded `0.1.0`, NO fake `dist-info`, NO guard downgrade.
- `experiments/sprint18_iterative_candidate_v1.py`: new `_validate_public_cli_entrypoint(binding, root)` runs the REAL same-interpreter public CLI parser in a subprocess (`python -m synth.cli --help` must exit 0 with `sprint18-iterative-v3` listed, plus a same-interpreter origin probe asserting the child imported the live checkout `src/synth/cli.py`) and `validate_runtime()` calls it before contact — a parser refusal now fails closed naming the child exit INSTEAD of retiring a candidate. The no-contact smoke also exercises this probe and reports `child_cli_origin_smoke` + `public_cli_entrypoint` identities.

## 3. Whole-fresh candidate-5 binding (metadata only)

- `S18-ITER-0005`, number 5, seeds `32064–32079` (`32000+16×(N−1)`, N=5), exact 16-role order D4/F3/Cal1/Dev4/Conf4, current v3, `contact_authorized: false` until a new Main receipt in later Task 69.
- All 16 configs resolved deterministically through the live v3 factory (pure metadata, no waveforms): seed-independent digest UNCHANGED `e933feb61e94d39fb56a45fae742fa1222b321b238dd1f3608642418be279136`; shorts `32064=aae5e52747a7 … 32079=f5ffb05669f0`.
- Collision catalog folds retired `32048–32063` into `32000–32063`: 421 unique excluded, `41045939…`; candidate `32064–32079` intersection `[]`; proof seeds `94102/94103` are NOT excluded (disposable, never bound).
- Strict whole-fresh `attempt_policy`: `no_resume: true` + `no_resume_rule`; interrupted roots retire the candidate.
- 53-member closure (candidate-4 operational record swaps the candidate-3 custody record, which stays preserved on disk as historical evidence): `73f9125c…`; binding `10e8bbbfe…` (45,046 bytes); runner canonical `0f9c4568…`, blob `5d105014…`; CLI canonical `a8a6da73…`, blob `2ad003f8…`; uv `0.12.21`/`7af826859`; `BASE_COMMIT`/`required_checkpoint_parent` = `2b1bf310…`; Task67 provenance `e576debd…`.

## 4. Decisive disposable proof (junction import-root, real source, genuine absence)

- 2-role fixture (`S18I-PROOF-A12B-DESIGN-01`/`FIT-02`, seeds `94102/94103`, never slicing final roles) under a disposable isolated venv (`numpy 2.5.2` / `scipy 1.18.1` / `torch 2.14.0+cpu`, app `dist-info` ABSENT verified by a real `PackageNotFoundError`) with directory junctions to the LIVE folders (zero byte copies, `resolve()` equality asserted): in-memory preflight PASS `2/2`; `dispatch_stage` exit `0` through the PUBLIC dispatcher with REAL public CLI children — DESIGN wrote 6,489 samples (manifest `502ae97e…`), FIT wrote 6,479 samples (`0f557211…`); ledger `attempt_started → preflight_started → preflight_recorded → started/materialized × 2` (7 events, `65fd39a5…`); both reload PASS via public `load_chronological` with role/protocol/config-hash checks. Honest accounting: an EARLIER run with a symlink-based (non-junction) proofwork layout wrote both manifests but closed `candidate_rejected public_cli_failed:1` on the second role with empty streams; the before/after difference is the fixture layout (plain symlinks + proofwork outside the symroot vs junctions + proofwork anchored under the symroot so the derived child `PYTHONPATH` resolves to the live checkout). The earlier exit-1 mechanism is UNPROVEN beyond this layout difference — no traceback was captured (children inherit stdio `check=False`) — and that run is DISCARDED with zero credit; the passing run above is the durable evidence.
- Guard paths exercised: stale uv pin refuses, tampered marker refuses, interrupted materialization retires with no resume, missing-metadata `--version` exits 2 naming the absence.

## 6. Identities (final file state)

- Binding canonical `10e8bbbfe91951296536a2e5032162d4dd3f97bdda121d5d89ca787df4f5f532` (45,046 bytes).
- Closure `73f9125c3b28f9ff7127393f2a63d8c15baca53091ac7fe450788335ae1c10a6` (53 members), runner canonical `0f9c4568…`, runner blob `5d105014…`, CLI canonical `a8a6da73…`, CLI blob `2ad003f8…`, retirement blob `74400a50…`. Test `1706106d…` (14,020 bytes).
- `BASE_COMMIT` / `required_checkpoint_parent` = `2b1bf310…`.
- No commit, no review/checkpoint/transport/Main receipt by this worker. Future Task69 success checkpoint must directly parent the future Task68 checkpoint; only the 12 non-Confirmation roles run until Task69 PASS plus checkpoint plus Main release; no Confirmation stage until then.
