# Sprint 16 Attribution Protocol v5 (prospective C5 re-binding)

## §0. Supersession and lineage

- v1 (`experiments/sprint16-attribution-protocol-v1.md`) PRESERVED (failed
  freeze record). v2 (`...-v2.md`) PRESERVED (superseded amendment). v3
  (`...-v3.md`) PRESERVED except as amended by v4/v5. v4 (`...-v4.md`)
  PRESERVED (C1 re-binding, still live).
- This v5 re-binds ONLY C5/objective-pressure to the accepted V2
  counterfactual pipeline. Every other v3/v4 section stands unchanged.
- v5 was written before any Task 9 outcome was observed; outcomes observed
  under v1–v4 MUST NOT select v5 content (none did — no Task 9 outcome
  exists anywhere).
- Delta table:

| Item | v3 defect | v5 disposition |
|---|---|---|
| Absent objectives | v3 §3 C5 / §8 name masked prediction, EMA target construction, mask composition, contrastive weighting — none present in the accepted V2 stack | §1 declares them NOT PRESENT: never blamed, never assigned a bottleneck verdict; Task 9 audits/documents the absence executably |
| Inexecutable retrains | v3 §8 C5 variants (mask 0.40→0.15, contrastive λ→0) name nonexistent knobs | §2 replaces them with exactly two FIT-healthy-only, architecture-fixed diagnostic variants: background weight=0; covariance weight=0; all other accepted weights fixed |
| Scope | — | Frozen gradient decomposition still reports all four terms; Confirmation is evaluation only; ≤300 steps each; same budget; no outcome-selected runs |

## §1. Accepted V2 objective: what exists and what does not

1. The accepted checkpoints train and infer EXCLUSIVELY under the
   counterfactual objective (`CounterfactualCriterion`: boundary-margin,
   background, variance, covariance terms + boundary schedule) applied to
   clean/corrupt view pairs from the synthetic corruption process
   (`synthesize_corrupted_patches`; corruption masks shape the loss only
   and never enter any encoder).
2. NOT PRESENT in the accepted V2 checkpoints, config, or path (verified by
   signature/config/path inspection): masked-latent-prediction objective,
   EMA target encoder/construction, masking schedule/composition
   parameters, contrastive weighting. Task 9 MUST audit and document each
   absence (module/config/signature evidence) and MUST NOT attribute any
   bottleneck, suspicion, or loss to them.
3. The context-mixer (`context_mixer` norm over learned conditioning
   embeddings) is scorer/head territory per v4, not objective pressure; it
   is not varied in Task 9.

## §2. The exactly-two C5 diagnostic interventions (replace v3 §8 C5)

Both are FIT-healthy-only, architecture-fixed, single-factor ablations of
the REAL objective, executed as tiny retrains from the accepted checkpoint
weights:

Frozen run rules (no discretion): init EXACTLY from the accepted checkpoint
weights (hash-verified); Fit-healthy files only
(never Design/Calibration/Confirmation/Sealed as fitting input); ≤300
optimizer steps each; deterministic file order and corruption (every 4th
valid patch, severity fixed 2.0, generator seed 20260202); batch = full
valid patches of each file in order (no shuffling across files); files
whose true conditioning indices exceed the checkpoint's fitted range are
counted and skipped, never remapped (same rule as Task 9 frozen-diag).
Budget accounting (binding): the v3 ≤2-gradient-run cap covers Task 9
retrains in TOTAL, so both variants execute on CONTROL only (the normal-only
reference checkpoint) — 2 runs, cap exhausted. Hybrid is covered by the
frozen-diag gradient decomposition on both checkpoints (already planned,
no training). Consequence recorded honestly: a C5 BOTTLENECK claim would
additionally require hybrid replication of any retrain win (v3 P3/both-
encoders spirit), which is UNFUNDED under this cap — two further hybrid
runs need a new explicit ruling; until then the ceiling for C5-a/C5-b
outcomes is SUSPECT-class localization evidence, never BOTTLENECK.
Each variant executes at most once; no replacement, no rerun, no tuning.

## §3. Frozen gradient decomposition (unchanged, restated scope)

Every C5 execution (including both retrains' step-0 baselines) reports all
four term values and per-term gradient norms exactly as v3 §6 M5 / Task 9
frozen-diag defines. Dropping a term from the LOSS never drops it from the
REPORT.

## §4. Confirmation discipline

Confirmation evaluates ONLY these two predeclared variants (Task-9-style
readout of retrained local latents vs the frozen accepted baseline);
no other checkpoint, threshold, or selection touches Confirmation.
A retrained variant that fails to load, diverges (non-finite loss), or
produces degenerate latents is reported as INCONCLUSIVE for that arm —
never replaced, never rerun, never tuned.

## §5. Verdict interaction (binds Task 16 via v3 §10 P3)

C5 BOTTLENECK still requires a replicated tiny-retrain win under the v3
G4 bar; anything less caps at SUSPECT. The absent V1 objectives of §1 can
never satisfy any G4 condition and are not verdict candidates.
