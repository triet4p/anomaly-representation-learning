# Sprint 18 Isolated Feasibility Study v1

**Study ID:** `sprint18-isolated-feasibility-study-v1`  
**Status:** Prospective freeze for independent Task 58 review only. This freeze authorizes no study-seed evaluation.  
**Scope:** Bounded sandbox feasibility comparison of the unchanged Sprint 15 Candidate 7/v7 factory profile and the unchanged Sprint 18 C2 factory profile. No new algorithm, benchmark cycle, gate, or generator parameter.

## 1. Fixed design and questions

Evaluate exactly the 16 fresh research-only data seeds `918000` through `918015`, ascending. For each seed, evaluate the existing `sprint15-v7` profile first and existing `sprint18-c2` profile second. This is a fixed 16-pair, 32-evaluation design. Do not replace, reorder, omit, extend, retry, screen, or selectively rerun any seed or profile. These seeds are permanently sandbox-only and cannot become benchmark roles.

The descriptive questions are whether the existing C2 profile changes full eligible-set cohort counts and mix, subtype quota margins, controls, reset-horizon attrition, and exact allocation feasibility compared with v7 on the same seed. Report every completed profile-seed row, each complete pair and its `C2 − v7` diagnostic changes, and the descriptive all-16 conjunction for each profile and jointly. This one exploratory block is not a population pass-probability estimate, independent Confirmation, benchmark qualification, or promotion rule.

The profiles are unchanged existing factories:

- `sprint15-v7` — `synth.chronicle.sprint15_v7_history_config(seed)`; cohort-A `abrupt_rate` remains `2.2e-5`.
- `sprint18-c2` — `synth.chronicle.sprint18_c2_history_config(seed)`; the already accepted C2-M1 factory changes only cohort-A `abrupt_rate` to `3.3e-5` (`κ = 1.5`). All other configuration values, causal/generator semantics, and inherited gates remain unchanged.
- Both records retain protocol tag `sprint15-benchmark-protocol-v7`; exact allocation uses `synth.balanced.DEFAULT_QUOTA` and `method="exact"`.

No profile tuning, parameter search, algorithm changes, gate relaxation, physical-method amendment, new materialization, model execution, Sealed access, or use of a retired benchmark roster is authorized.

## 2. Isolation and stage gate

No evaluation of seeds `918000–918015` may occur until the Task 58 contract/runner handoff has an independent evidence-review PASS. Task 59 owns the single later study execution. The runner requires explicit `--study --review-passed`; this flag records acknowledgement of the independent review gate and is not a substitute for that review.

The feasibility chain runs in memory and never calls `materialize_chronological`, shard/manifest writers, seal APIs, model code, or benchmark qualification. The only persistent study files are the one-shot attempt marker and research result under `artifacts/sprint-18/isolated-feasibility-v1/`. It must not create a root under `data/generated/` or assign a role/path to a study seed. Smoke mode uses only disposable fixture seed `93058`, never any study or recorded roster seed, and writes no persistent research or generated-data files.

The attempt directory is exclusively claimed before the first study evaluation. If it already exists, the runner blocks before any evaluation. There is no resume or second attempt. The attempt marker is created before the first evaluation; raw rows are persisted after each evaluation. Proven infeasibility is an ordinary result row and does not truncate the fixed 32-evaluation budget. An infrastructure error, unavailable exact solver proof, witness breach, or unexpected audit/generation exception is recorded with all prior rows, marks the attempt blocked, and stops without retry or continuation. A partial/blocked attempt is final evidence and cannot be resumed.

## 3. Frozen audit and allocation semantics

The runner uses the existing in-memory calendar/generation and audit functions once per profile-seed evaluation:

1. Resolve the frozen profile factory and config for the fixed seed; record `SynthConfig.hash()` and check the profile's bound cohort-A rate.
2. Call `synth.chronicle.build_chronological(cfg)` once; derive maintenance windows, patch counts, manifest file rows, and the failure ledger using the same existing functions used by the no-write preflight path.
3. Call `synth.balanced.project_allocation_inputs` and `synth.balanced.eligible_anchors`; use `synth.events.positive_window_intersects_reset` for reset attrition.
4. Obtain controls with the existing `synth.events.anchor_rows` and `select_control_windows` semantics.
5. Call `synth.balanced.allocate_quotas(..., synth.balanced.DEFAULT_QUOTA, method="exact")` once. Preserve the complete returned solver/allocation record or the original machine-readable infeasibility/unavailability record.

No algorithm is substituted or reimplemented. The runner performs the same single in-memory chain used by `synth.preflight15` but exposes the broader observational diagnostics required here. It does not call preflight and then regenerate the same history: `preflight15` does not expose all eligible-set counts/mix or failure-row controls, and a second call would duplicate expensive generation.

The fixed quota values are `P=24 (P1=12, P2=12)`, `W=24 (W1=12, W2=12)`, `A=16 (A1=8, A2=8)`, `controls=48`, `robot_days=240`, `robots_pos=6`, `robots_neg=6`, `programs=2`, `spacing_s=1,209,600`, `robot_pos_cap=0.35`, `robot_neg_cap=0.40`, and `program_cap=0.60`. These are read from `DEFAULT_QUOTA` and guarded against this freeze; they are not new thresholds.

## 4. Required row and summary fields

Every completed row records the seed, profile, v7 protocol tag, config hash, A-rate, `no_write=true`, elapsed time, and exact allocation status. It captures:

- **Raw and eligible cohort counts:** raw ledger P/W/A and total; full eligible-anchor P/W/A and total.
- **Eligible mix:** P/W/A shares over the full eligible set, with the denominator retained in the row; not shares over the allocated subset.
- **Subtype quota margins:** eligible pool count, frozen quota, and `eligible − quota` for P1/P2/W1/W2/A1/A2.
- **Reset attrition:** reset-horizon rejection counts by cohort and total, raw-to-eligible attrition by cohort, non-reset ineligibility by cohort, rejection reasons, and reset-rejection fraction of raw events. Internal checks require these counts to reconcile to the existing audit output.
- **Controls:** selected control-window count, fixed quota, and margin for every row, including exact-allocation failures.
- **Exact allocation:** `FEASIBLE`, `INFEASIBLE`, or `UNAVAILABLE`/blocked; preserve all available solver status, reason, missing constraint, allocation, controls, and margins. Failure rows are first-class results, not omitted rows.

For each complete seed pair, report `C2 − v7` differences for raw/eligible totals, eligible cohort counts and shares, eligible subtype counts, subtype quota margins, reset attrition, non-reset attrition, and controls; also retain each original row and both feasibility outcomes. The all-16 conjunction is explicitly descriptive only. It must not be interpreted as a reliable population probability or an independent confirmation result.

Each row receives a SHA-256 over canonical JSON before its `record_sha256` field is added. The final raw `result.json` is hashed and its digest is written to the attempt marker.

## 5. Fixed seed isolation from existing bindings

The frozen study block is compared against existing recorded seed bindings and namespace checks before smoke or study mode:

- `experiments/sprint18-role-binding-v1.json` — Cycle 1 bound role seed records;
- `experiments/sprint18-role-binding-c2-v2.json` — Cycle 2 bound role seed records;
- `experiments/sprint18-role-binding-c3-v1.json` — Cycle 3 bound role seed records and its recorded `disjoint_seed_ranges_checked` entries;
- `experiments/sprint17-role-binding-v3.json` — Sprint 17 bound role seed records.

The C3 binding's recorded ranges cover Sprint 15 Candidate 1–7 (`1000–1017`, `1100–1117`, `1200–1217`, `1300–1317`, `1400–1417`, `1500–1517`, `1600–1617`), Sprint 17's fresh roster (`2800–2815`), Sprint 17's prior failed roster (`2700–2715`), Sprint 14's recorded bands (`742–797` and `910–933`), Cycle 1 (`31800–31815`), Cycle 2 (`31816–31831`), Sprint 18 model seeds (`181801–181803`), Sprint 17 model seeds (`171701–171703`), and disposable proof fixtures (`9000`, `9001`). The C3 binding itself binds Cycle 3's exact role seeds (`31832–31847`). The runner checks overlap against those recorded ranges/role seeds and refuses any collision; it never substitutes another seed.

No Sealed history data, Sealed paths, or Sealed seed rows are opened, listed, hashed, or probed. The namespace proof uses only the cited binding records and their recorded range summaries. Binding file digests are frozen below.

## 6. Config and source closure

Profile reference configuration hashes use the existing `SynthConfig.hash()` at reference seed `0`, solely to bind profile configuration semantics; this does not build or evaluate a history. The frozen values are `sprint15-v7=7de530a28d23` and `sprint18-c2=720e31e94681`. For every fixture/study pair, the runner checks that the two resolved configs differ in exactly one field: cohort-A `abrupt_rate` (`2.2e-5` versus `3.3e-5`). Per-evaluation config hashes are retained in their result rows.

The runner checks exact SHA-256 for all inputs below before any smoke or study evaluation. The complete synth Python tree is frozen so a transitively imported generator, audit, anomaly, physics, or schema source cannot drift unnoticed.

| Frozen input | SHA-256 |
|---|---|
| `experiments/sprint15-benchmark-protocol-v7.md` | `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6` |
| `experiments/sprint18-data-method-c2-v1.md` | `f4cc8901c2e59f615ebcaaa3509fd654773efd06f7f6fd6a55faeab6b3ba4b9b` |
| `experiments/sprint18-role-binding-v1.json` | `493c5d6a7a9b807f1fe915f00bb917be45a11183512581606eee34ee678bb228` |
| `experiments/sprint18-role-binding-c2-v2.json` | `45818410f8d483d66a5523fd4ec38169bed9dc45d3a82f16979c05f9ab91dd92` |
| `experiments/sprint18-role-binding-c3-v1.json` | `6d87b9bba4e7e76e25917e6b00a13a52dc886e8ea8052ef6e5c88cf01a863ddd` |
| `experiments/sprint17-role-binding-v3.json` | `e73fee24cf84b60d17cbe4013ecc77ddace6e9132286ca8b65155039dd47a679` |
| `pyproject.toml` | `30292a091a273d0d58958701f792ad30af9a9bbd7454480275dedd73d1421d01` |
| `uv.lock` | `4a7866878c81cdd8f5ba283cd8d0a772073bfc75941f73ec761ede5eeb2e239a` |
| `src/synth/__init__.py` | `6b98c860a7d39e152f57f829d43a6eff4446d5301a1e9be6454a8945067e80c6` |
| `src/synth/anomalies/__init__.py` | `4185ee9d09f8b3c1bcd937dac5630a33882d972b9992797a338f6088ea4db5c5` |
| `src/synth/anomalies/base.py` | `b98c8727168e7f4e6eb88d0474b2f51a26df0ed5d26bdeae1abe39ca1a5df206` |
| `src/synth/anomalies/contextual.py` | `551dad79ba4266b3524f3e261e740c43818a7eb4ba3d8462880d726e9ee050b1` |
| `src/synth/anomalies/cross_channel.py` | `f40ed7c447e85726ad17248f6d165e3dd38203ad4e2f4ea039935bf4fc3421ca` |
| `src/synth/anomalies/drift.py` | `8bece76574231c58b85e309f8c879d54d3ee87ab50f4634e6ad58c23aa6190ba` |
| `src/synth/anomalies/duration.py` | `f7b4e0a324554dd4379f951a1223c00b12c624695f8bd92a088d085b3fb129d9` |
| `src/synth/anomalies/easy_sanity.py` | `41efd72d17ee7167d8b7ef4c6fb9ebfde29e37dc6ff43de951fbfb21b97a0e01` |
| `src/synth/anomalies/freq_phase.py` | `83eb1454f9e8d3015647525baf8d3f2fc36a2a0531eebd0a12a865c4da84ebcb` |
| `src/synth/anomalies/missing_event.py` | `464fa9daed5cfdbf13186e5ab406b960a6dcb4e028c584ec0cd683937fcd5549` |
| `src/synth/anomalies/registry.py` | `7d34156b920bf9c79486e2c1610d28706e6073c73b7363608c9798efdf1561d1` |
| `src/synth/anomalies/regularity.py` | `3bb17a410df99bce933dad7cf2f0d208c23197c06b5683e4ffec499a138cce7d` |
| `src/synth/anomalies/stuck.py` | `9c7e14ac4a6c9cbb7af3cfa8df0b90240e4eaaf74df7badd54dc8c4988e399c8` |
| `src/synth/anomalies/transition.py` | `1a92b361dfeff2df80939f95de84c7bbf59ebd0ccbc3629ab326bccee87be704` |
| `src/synth/balanced.py` | `d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43` |
| `src/synth/chronicle.py` | `f932e1a3c3bbc4bf7c8123e0ae9d3125200fedf0ce586d6d7eaf637d335f2cb6` |
| `src/synth/cli.py` | `19918890c50ed121cfc6de646ed98c73fd96651bfc1d0b18f8877d05972b550d` |
| `src/synth/config.py` | `55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056` |
| `src/synth/contrastive.py` | `af1a2ded1ce5bc9cc7fef9572599d0d75443c168b58b3907660c56deb7ac9e71` |
| `src/synth/dataset.py` | `375029c48e3a19d2e1e30f986bde5339fb597474e9f0685405fd548c0bc0e662` |
| `src/synth/diagnostics.py` | `1cd87aa40893e710e3b645749eed1cb84e8a8f3f8b9a21f0e97eb006c40f8eb6` |
| `src/synth/events.py` | `85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d` |
| `src/synth/generator.py` | `32cd227943cb00c165c43aeb75835f14433a57c60ca9adca0e5c20d21e654403` |
| `src/synth/health.py` | `e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d` |
| `src/synth/masking.py` | `b73843b1fd6fd7a925d3dc6e58967bdb744de89cc1ca61927958040db0513a2c` |
| `src/synth/normal.py` | `a8203b2b9fe9f0489c249e2694dde4c17752976c6402ddc5907d0b94c565d6ac` |
| `src/synth/patchify.py` | `4b7e6876d6c43c76848d7b179a02e19f1c065aaecbfbd1eb159d448c45bc8d49` |
| `src/synth/physics/__init__.py` | `4e0d2ca9e0bec699cbd23dde004fcbfd8a422814d7a264fc11b05aa78b885a4b` |
| `src/synth/physics/causal.py` | `6e97fbe655fe0a1c55433e04eb23fb1922ce28537ec061875fda2632c6c8d183` |
| `src/synth/physics/noise.py` | `cf3f51ad9a571a3cf77c3c40aaf2e0d69b96afda860615edf9751b9f9f063d54` |
| `src/synth/preflight15.py` | `da1239b3e3d1df145389083c813ce97d99b7f80c32a9452d1d7a683b61dd5df6` |
| `src/synth/probe15.py` | `a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b` |
| `src/synth/regimes.py` | `e0287d2dc7aadf1ca50c0bf63151f1d79b11a7d7a14c68ee7eec0a21a981cafd` |
| `src/synth/scheduled.py` | `feb76b9d10c500e3dbea7fda2237e7daf8d1b41abe6cf8a7e6d38a9e73c72ce7` |
| `src/synth/scheduler.py` | `4659b94c31f3a2628f57d0fd789ea8f7d42c5f82abfc0f7dc01ffa2e7bf3e3af` |
| `src/synth/schema.py` | `6d6b001c2199933c14dbd019d9eaf8ded5c3dddc3d33997420f62175ee230203` |
| `src/synth/splits.py` | `0abace7461b2d5da9e2b616748f539e03aac8ef0e7b65fe4325f14a719fc0316` |
| `src/synth/strength.py` | `6b23db6b529e4c873e41ec8f25785412ce113184b1a29847e35571cf749c786f` |
| `src/synth/temporal.py` | `02a8ea8d7698794e5fbc0ec499e94d7a25e4fe7a3e888e1295cc13c567f36e18` |

Source map is SHA-256 hashed from canonical JSON (`sort_keys=true`, compact separators, UTF-8); the task artifact records that closure digest, the study-document SHA-256, runner SHA-256, and combined freeze digest. The combined freeze digest is SHA-256 over `b"SPRINT18-ISOLATED-FEASIBILITY-FREEZE-v1\n"`, followed by the raw 32-byte SHA-256 of the contract document, the raw 32-byte SHA-256 of the runner, then canonical JSON bytes of the sorted source closure map.

## 7. Runner and allowed commands

The isolated runner is `experiments/sprint18_task58_feasibility.py`.

- Task 58 smoke only: `uv run --locked python experiments/sprint18_task58_feasibility.py --smoke`
- Task 59 only after independent Task 58 PASS: `uv run --locked python experiments/sprint18_task58_feasibility.py --study --review-passed`

The smoke command verifies bound inputs/configs/namespaces, exercises the exclusive-attempt guard by claiming a disposable temporary marker and proving a second claim blocks, and runs each unchanged profile once on fixture seed `93058`. It asserts full eligible counts reconcile, the mix sums to one when nonempty, all requested diagnostics exist, and no persistent study output or generated-data root is created. It does not evaluate any study seed.
On the same already-generated v7 fixture only, smoke also invokes the existing exact allocator once with a disposable `P1` quota set one above the measured eligible pool. It requires a solver-proven infeasibility row while the already-captured eligible, mix, subtype, reset, and control diagnostics remain present. This is a failure-path smoke probe, not a study evaluation, bound-quota change, or reported profile result; it makes no second generator call and is discarded after smoke.

The script writes raw Task 59 evidence only inside the fixed ignored artifact directory; it never creates generated-data roots. One row is persisted after each profile evaluation. No command supports resume, partial rerun, selecting a seed/profile subset, or arbitrary output paths.
