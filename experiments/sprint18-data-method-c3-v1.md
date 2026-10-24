# Sprint 18 Data Method C3 v1 — Prospective Authorized Same-Method Repeat (Exception)

**Method ID:** `sprint18-data-method-c3-v1`
**Status:** PROSPECTIVE FREEZE FOR FRESH INDEPENDENT REVIEW — authorizes no implementation, binding, preflight, materialization, training, scoring, or Sealed access. Cycle 3 may proceed to Task 52 only after this document plus `artifacts/sprint-18/task-51.md` pass independent evidence review with zero actionable findings.
**Scope:** Exactly one prospectively declared exception: Cycle 3 reuses the reviewed Cycle 2 methodology byte-identically on the next fixed roster. No model, probe, gate, floor, threshold, role-permission, or verdict-semantics change. No new generator or preflight code.

## 1. Exception statement and user choice

Task 50 completed a Design-only diagnosis and returned a truthful BLOCKER: no distinct defensible Cycle 3 amendment exists on permitted evidence (`agent://Sprint18C3Task50Review` PASS applied to the blocker verdict — it confirmed there is no distinct method, and it is not approval to execute anything).

After reviewing that blocker, the user explicitly selected **'Cho phép lặp phương pháp C2'** (permit repeating the C2 method): Cycle 3 applies the exact C2 generation/preflight method to the next fixed 16-seed roster (31832–31847), recorded as decision `.agents/memory/decisions.md` dated 2026-09-25 ("Permit a declared same-method Cycle 3 repetition").

This document is the transparent prospective freeze of that override. It weakens the prior cross-cycle no-retry rule for Cycle 3 ONLY: each seed and each roster still has exactly one attempt and remains immutable after failure. Cycles consumed remain 2/10 (Cycle 1 seeds 31800–31815 retired after materialization; Cycle 2 seeds 31816–31831 retired at roster preflight with zero roots); at most eight cycles (3–10) remain.

## 2. No new causal method — scientific limitation

Cycle 3 introduces **no new causal-method fix**. The single bounded lever (abrupt-cohort density under reset-horizon censoring, C2-M1 κ = 1.5) was already consumed by Cycle 2. Any Cycle 3 eligibility outcome is therefore conditional on a **repeated-method candidate**, not evidence that a redesigned method solved the earlier failures:

- A Cycle 3 PASS would show only that a fresh roster draw under the identical C2 density passes all gates — it must not be reported as validation of a new methodology.
- A Cycle 3 FAIL (any preflight miss or gate miss) retires Cycle 3 in full with zero rescue, exactly as Cycles 1–2 were retired.
- Successive candidate cycles are no longer independent methodology tests; this exception knowingly weakens independent-method inference. Future cycles (4–10) require their own prospective justification and may not silently inherit this exception.

No Confirmation tally sized anything here: C2-M1's κ = 1.5 is inherited from the Design-sized C2 freeze, and no Confirmation-based tuning was performed or is permitted by this document.

## 3. Exact reused method identity

Cycle 3 reuses the following C2 artifacts identically — same bytes, same profile, same entry points, same gates:

- **Method source:** `experiments/sprint18-data-method-c2-v1.md`, SHA-256 `f4cc8901c2e59f615ebcaaa3509fd654773efd06f7f6fd6a55faeab6b3ba4b9b` (re-verified read-only from disk during Task 51).
- **Override:** for profile `sprint18-c2` only, v7 §1a literal `A abrupt_rate 2.2e-5` replaced by **`A abrupt_rate 3.3e-5` (factor κ = 1.5)**. The v7 file bytes are untouched; every other §1a literal remains in force.
- **Config factory:** `synth.chronicle.sprint18_c2_history_config(data_seed)` (additive C2 factory; post-Task42 `src/synth/chronicle.py` SHA-256 `f932e1a3c3bbc4bf7c8123e0ae9d3125200fedf0ce586d6d7eaf637d335f2cb6`).
- **Profile:** `sprint18-c2` (`S18_C2_PROFILE`); generator protocol tag remains frozen `sprint15-benchmark-protocol-v7`.
- **Preflight entries:** `synth.preflight15.preflight_seed_c2(history_seed)` and `synth.preflight15.run_preflight_c2(seeds)` (explicit fail-closed C2 entries; post-Task48 `src/synth/preflight15.py` SHA-256 `da1239b3e3d1df145389083c813ce97d99b7f80c32a9452d1d7a683b61dd5df6`; legacy v7-default entries preserved but forbidden for the C3 roster).
- **Resolved-config identity:** Task 51 re-verified on disposable non-roster seed 9001 that the v7→C2 resolved-config diff is exactly one leaf, `/health/cohorts/2/abrupt_rate 2.2e-5 → 3.3e-5`, and that repeated factory calls are deterministic (config hash `15aecaff73b9` stable across calls).
- **Inherited normative gates:** `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` SHA-256 `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c` governs on any conflict; this exception never weakens, reinterprets, or replaces it.
- **C2 binding reference:** `experiments/sprint18-role-binding-c2-v2.json` (canonical digest `9905a2eec94fbb811eaf120b35850c230129a708d21fdc0d36a0720505960ddf`) is the retired-cycle precedent; it is not reused for Cycle 3. Cycle 3 binding is a separate Task 52 freeze.

No source file was created, edited, or re-hashed by Task 51; the accepted post-review source states (chronicle post-Task42, preflight post-Task48) are inherited unchanged.

## 4. Source closure (digests re-verified read-only from disk during Task 51)

| Input | Path | SHA-256 |
|---|---|---|
| C2 method (reused identically) | `experiments/sprint18-data-method-c2-v1.md` | `f4cc8901c2e59f615ebcaaa3509fd654773efd06f7f6fd6a55faeab6b3ba4b9b` |
| C2 binding v2 (retired-cycle precedent) | `experiments/sprint18-role-binding-c2-v2.json` | file bytes `45818410f8d483d66a5523fd4ec38169bed9dc45d3a82f16979c05f9ab91dd92`; canonical digest `9905a2eec94fbb811eaf120b35850c230129a708d21fdc0d36a0720505960ddf` |
| Accepted Sprint 18 protocol v2 | `experiments/sprint18-full-training-protocol-v2.md` | `e1dcd17914e920520ad0d8f50e185874b95414e6a4fe907903619daf145622d2` |
| Inherited measurability gates v2 | `docs/BENCHMARK_MEASURABILITY_EXIT_GATES_V2.md` | `d294762331ded4fd213f6870e563e32e17e12c632d7cb11a80373478ede34e1c` |
| Candidate 7 protocol v7 | `experiments/sprint15-benchmark-protocol-v7.md` | `a1fc09db2e246ed79d0595aec953a7788fd1b47c4981fbe1d92017d944b8d7b6` |
| Frozen observable-probe spec v7 | `experiments/sprint15-observable-probe-v7.md` | `4d4c2b53b3879efdf92935e362956c87d16892feba8276367f1e5aa13aa4a51a` |
| Frozen probe implementation | `src/synth/probe15.py` | `a08b3d5fb83001e1f5c43f4c56ff536bae85e41d494db289304aeb33a339242b` |
| Generator: config | `src/synth/config.py` | `55fed2ef5b253d48c1b0c7f1dd5f7f966cdd03ad250cb86a988ac8de23d5c056` |
| Generator: health process | `src/synth/health.py` | `e5854641fb85eb9b88e05ee9d8e503c69a9432b13177d8c62ad38c3494e7fc6d` |
| Generator: event predicates | `src/synth/events.py` | `85100f5e0aca47dd2e8b01a08c58f39d32be4f1eab469cdc38e4a4b57768169d` |
| Generator: chronicle/materializer | `src/synth/chronicle.py` | `f932e1a3c3bbc4bf7c8123e0ae9d3125200fedf0ce586d6d7eaf637d335f2cb6` |
| Generator: allocator/audit | `src/synth/balanced.py` | `d62de3422bd51f870d39d18b1a8bb942f9ca73fc0044d8f23cc2d0948bd35a43` |
| Generator: CLI | `src/synth/cli.py` | `19918890c50ed121cfc6de646ed98c73fd96651bfc1d0b18f8877d05972b550d` |
| Generator: package init | `src/synth/__init__.py` | `6b98c860a7d39e152f57f829d43a6eff4446d5301a1e9be6454a8945067e80c6` |
| Preflight runner (C2-capable) | `src/synth/preflight15.py` | `da1239b3e3d1df145389083c813ce97d99b7f80c32a9452d1d7a683b61dd5df6` |

Cycle 2's rejected seeds/roots are never reused: C2 rejected seeds 31829 (W/W2 pool 10<12) and 31830 (controls 47<48) are rejection facts only, and no C2 source, gate, or accepted record is mutated by this document.

## 5. Cycle 3 namespace (specified, not consumed — binding deferred to Task 52)

- **Seed block:** exactly the disjoint fixed block **31832–31847** (16 seeds), in fixed role order Design×4, Fit×3, Calibration×1, Development×4, Confirmation×4. Task 51 verified the block is disjoint from Sprint 15 bands (1000–1617), Sprint 17 (2800–2815), Cycles 1–2 (31800–31831), and model seeds (181801–181803).
- **Identity rule:** Cycle 3 history IDs match `^H-S18C3-(DES|FIT|CAL|DEV|CONF)-[0-9]{2}$`, unique across all cycles; the exact 16 strings are frozen by Task 52, not here.
- **Directory rule:** Cycle 3 roots live under `data/generated/sprint18-full-training-c3-v1/<ROLE>/<history-id>`; Cycle 1/2 directories and roots are never written again, reused, screened, pooled, or promoted.
- **Model seeds:** unchanged (`181801, 181802, 181803`); disjointness re-verified at Task 52.
- **Zero-contact proof (Task 51):** `experiments/sprint18-data-method-c3-v1.md` (this file, newly created), `experiments/sprint18-role-binding-c3-v1.json`, and `data/generated/sprint18-full-training-c3-v1/` — the binding file and root directory are both absent; no C3 roster seed was executed or preflighted. Only disposable non-roster fixture seeds 9000/9001 were touched for factory/config identity checks.

## 6. Binding, one-shot, stage, permission, and stop rules

Inherited unchanged from the C2 contract and applied to Cycle 3:

- **Task 52 (binding):** freeze the 16 Cycle 3 identities, disjoint seeds, paths, permissions (§5), canonical binding digest, versioned source digests (§4), explicit `run_preflight_c2` entry/profile, and repeated-method status. No roots created.
- **Task 53 (roster preflight + Design materialization once):** one rejection-only `run_preflight_c2` over the full 16-seed C3 roster (`no_write`); unanimous PASS required. On unanimous pass, materialize exactly the four Design roots once (`overwrite=false`, attempt markers). Any preflight miss retires Cycle 3 in full with zero roots.
- **Tasks 54/55/56 (qualification):** Design histories under unchanged per-history structural/mix/causal and fixture gates; then one-shot Fit/Calibration/Development materialization with the frozen observable probe; then one-shot Confirmation with frozen EG0–EG5/probe. Any miss retires Cycle 3 or records `REJECTED-CANDIDATE` without rescue.
- **Role permissions (16 binding instances frozen at Task 52):** Design — structural/measurability qualification and diagnosis only; Fit — verified-healthy representation fitting only; Calibration — per-arm/per-seed q95 thresholds only; Development — complete descriptive comparison, no selection; Confirmation — one-shot locked final comparison, opened solely under Task 56 after all prior gates pass.
- **Universal prohibitions:** no screening, retry, replacement, pooling, omission, role reassignment, favorable-seed selection, or failed-Confirmation-to-Design promotion; no Sealed contact; no C1/C2 root reuse; no C2 rejected seeds/roots reuse; methodology changes only via a new prospective versioned amendment with independent review.
- **One-shot/immutable safeguards:** `no_write` preflight; `overwrite=false` materialization; per-stage attempt markers; seeds never reassigned; failed cycles preserved byte-identical and retired in full; no second run of any stage within a cycle; the 10-cycle cap is unchanged.
- **Frozen gates:** immutable 15–60% per-cohort mix on every Design/Confirmation history independently; hard floors P≥10/W≥10/A≥8/total≥30/controls≥25/robot-days≥150/robots≥6+6/programs≥2+2; concentration caps 35%/40%/60%; 80% P/W lead-support; Design promotion targets; Fit/Calibration support floors; EG3 fixtures; frozen observable-probe thresholds — all exactly as in C2 §§9–10.

## 7. Failure conditions (each retires or stops Cycle 3 as stated)

- This document or `artifacts/sprint-18/task-51.md` fails independent review → stop before Task 52; Cycle 3 unopened.
- Task 52 disjointness/digest/permission/entry mismatch → stop before preflight.
- Task 53 preflight miss on any seed → retire Cycle 3 with zero roots.
- Tasks 54/55/56 any gate miss → retire Cycle 3 or record `REJECTED-CANDIDATE`; no retry, replacement, or tuning.
- Any Sealed contact, C1/C2-root reuse, C2-rejected-seed reuse, retry/replacement/pooling, or out-of-order execution → protocol breach; affected work invalid.

## 8. Review handoff

The reviewer should verify: (a) the user-choice exception and C3-only scope (§1) against `.agents/memory/decisions.md` 2026-09-25 and the sprint plan Task 51; (b) exact method identity (§3) — factory, profile, preflight entries, abrupt rate, single-leaf diff — against disk sources; (c) all §4 digests reproduce via SHA-256 over disk bytes; (d) namespace/permission/stage rules (§§5–6) with exact binding deferred to Task 52; (e) the scientific-limitation statements (§2) are present and no future success is claimed; (f) no code change, no binding file, no roster contact, no new roots.
