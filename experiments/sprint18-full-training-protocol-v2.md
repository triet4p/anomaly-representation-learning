# Sprint 18 Full-Training Representation Protocol v2 Amendment

**Protocol ID:** `sprint18-full-training-protocol-v2`  
**Status:** PROSPECTIVE AMENDMENT FOR FRESH INDEPENDENT REVIEW — not frozen and not execution-authorized.  
**Scope:** Resolve exactly Task 2 findings F-01–F-05 from `artifacts/sprint-18/review-batch-a-contract.md`; no Sprint 18 data, model, Sealed, or outcome inspection is part of this amendment.  
**Base protocol:** `experiments/sprint18-full-training-protocol-v1.md`, SHA-256 `9ae3d43f94a6194bae602d68067958bdf7d11cf47e801dd328e87b22e7907cf6`.

## 1. Normative composition and precedence

This document is an explicit versioned amendment, not a silent modification or replacement of the frozen v1 bytes. The prospective v2 contract is the exact composition of (a) the v1 protocol identified above, (b) this v2 amendment, and (c) the four v2 mechanism specifications bound by SHA-256 in §2. All v1 common-contract clauses remain in force unless this amendment explicitly replaces their protocol/schema identity or a named v2 specification explicitly replaces a clause of its bound v1 mechanism specification.

For any conflict, apply this order:

1. This v2 amendment controls the v2 identity, schema identity mapping, specification bindings, and the five listed finding resolutions.
2. The matching v2 mechanism specification controls only the expressly replaced clauses of its bound v1 specification; all other clauses of that exact v1 specification are inherited unchanged.
3. All remaining common protocol terms are inherited from the exact v1 protocol digest above.
4. No executor may mix a v1 mechanism spec with this v2 protocol, select between competing interpretations, or use a v2 spec with v1 schema/provenance identity. The complete v2 set is one indivisible prospective contract.

The v1 protocol and v1 specs remain preserved historical inputs with their original bytes and hashes. This amendment does not retroactively alter Task 1's freeze, Task 1's evidence, or the initial Task 2 FAIL. It becomes usable only after a fresh independent review accepts the complete v2 set and the project owner explicitly advances the required gate.

## 2. Immutable source and version bindings

The four inherited mechanism specifications and their sole v2 replacements are:

| Arm | Frozen v1 specification SHA-256 | Prospective v2 specification | v2 SHA-256 |
|---|---|---|---|
| N1-HCC | `experiments/sprint18-hierarchical-conditional-contrastive-spec.md` — `bf757fa4cda85e8ea5e743928bf405294c6bc4c6f16c67d54b8edef763a69d51` | `experiments/sprint18-hierarchical-conditional-contrastive-spec-v2.md` | `0f5ede73d65441014caf22a22058c17d8ed34be735790d20ca7f36873a34d144` |
| N2-MSP | `experiments/sprint18-multiscale-latent-prediction-spec.md` — `7b92cd5c9d7f468dd95e46fa544fcd57d0a76cf0638536560a5d4e4d5439e72d` | `experiments/sprint18-multiscale-latent-prediction-spec-v2.md` | `a2deeb5479957ef8777ec4cfd66898e3ecd5e0aef80e2043f11882529ae3a9f2` |
| N3-CCRP | `experiments/sprint18-cross-channel-relational-prediction-spec.md` — `1c0f570606b665f0976bc8161f0d1dd3b2cc2d482b0f861b5dbc2975bd4c7cfd` | `experiments/sprint18-cross-channel-relational-prediction-spec-v2.md` | `eee44cebd29de2b5cbfdf0263fcca21d82fee92e0773922c0bc3209dee50443c` |
| N4-CRN | `experiments/sprint18-context-residual-normalization-spec.md` — `fd0e1722ea4f248a354086bbdf44bc486d741a9cfb018f68ef08277827d7954d` | `experiments/sprint18-context-residual-normalization-spec-v2.md` | `4d1c8097c9cc27d3fbeb2e2a6e70a29a7bc63d7899646a9c712a315b39d9ffe3` |

The v2 amendment file's exact SHA-256 is its own protocol identity and MUST be recorded as `provenance.protocol_sha256` in v2 results. For N1–N4, `provenance.mechanism_spec_sha256` MUST equal the corresponding v2 file hash in this table. All v2 file hashes are content hashes of the complete files, not hashes of selected clauses.

## 3. Finding-to-replacement map

| Finding | Sole prospective resolution |
|---|---|
| F-01 — N1 Conditional BatchNorm parity | N1 v2 replaces the ambiguous equality language and check 7: preserve B0 implementation/configuration, matched initial state/output, and fallback mapping; keep parameters trainable and do not require learned post-update state equality. |
| F-02 — N2 multi-scale mask allocation | N2 v2 replaces §3 and checks 1/4 with exact B0-rounded total count, a deterministic integer 50/25/25-nearest allocation, seeded non-overlapping group placement, exact fine-mask target, and the recorded `8 → 4 → fine` fallback. |
| F-03 — N3 relational patch sampling | N3 v2 replaces channel ordering and §4 selection with exact quarter-count rounding, per-file/epoch PRNG seed, permutation/resampling, global zero-based ordinal, and channel assignment. |
| F-04 — N4 bounds leakage | N4 v2 replaces §3 bounds: each held-out Fit fold's quantiles use only its two in-fold Fit histories; final scoring bounds are recomputed from all three Fit histories. |
| F-05 — N4 optimizer accounting | N4 v2 replaces the fit-order/step clauses with canonical record ordering, exact per-scope/per-epoch CPU-generator permutation, retained partial batches, and exact fold/final optimizer-step formulas. |

No other v1 mechanism clause is replaced.

## 4. V2 result and verdict schema identity

The v2 result and verdict JSON Schemas are the exact embedded v1 schemas from protocol v1 after these identity-only string substitutions, and no other schema-content change:

| Schema | v1 `$id` and `schema_id` constant | v2 `$id` and `schema_id` constant | v1 `protocol_id` constant | v2 `protocol_id` constant |
|---|---|---|---|---|
| Result | `sprint18-full-training-result-v1` | `sprint18-full-training-result-v2` | `sprint18-full-training-protocol-v1` | `sprint18-full-training-protocol-v2` |
| Verdict | `sprint18-verdict-v1` | `sprint18-verdict-v2` | `sprint18-full-training-protocol-v1` | `sprint18-full-training-protocol-v2` |

All other schema fields, constraints, required properties, enumerations, and gate semantics remain byte-for-byte equivalent to the v1 schemas. Every v2 result MUST use the v2 identifiers and bind this amendment/specification set by the hashes above; v1 schema/provenance identity is not valid for a v2 run.

## 5. Explicitly unchanged contract boundaries

The eight-arm order and identities, fresh-role permissions, exclusions and support rules, model seeds, common representation schedule, optimizer settings, selected coefficient grids and gates, checkpoint contract, compute measurement method and numeric envelopes, scoring/metrics, recovery decision gates, result contents other than the identity substitutions in §4, and verdict precedence remain those in the exact v1 digest. In particular, there is no role/root creation, coefficient or architecture selection, compute-limit relaxation, score fusion, threshold change, additional arm, or new execution permission in this amendment. N4's declared cross-fit and final-fit updates are all included in the unchanged common training-compute measurement and numeric envelope.

## 6. Gate and authorization

The initial Task 2 Batch A result remains **FAIL** until independently reviewed corrective evidence supersedes it; this authored amendment is not that review. Task 2 must obtain a fresh independent review of this v2 protocol plus all four bound v2 specifications before the prospective contract can be accepted. This document itself authorizes no role binding, generation, coefficient pilot, model training, scoring, Sealed inspection, Confirmation work, Task 3, or other downstream action. Sprint status remains owned by Main.