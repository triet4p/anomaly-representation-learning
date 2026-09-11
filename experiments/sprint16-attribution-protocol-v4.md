# Sprint 16 Attribution Protocol v4 (prospective C1 re-binding)

## §0. Supersession and lineage

- v1 (`experiments/sprint16-attribution-protocol-v1.md`) PRESERVED (failed
  freeze record). v2 (`experiments/sprint16-attribution-protocol-v2.md`)
  PRESERVED (superseded amendment). v3
  (`experiments/sprint16-attribution-protocol-v3.md`) PRESERVED except as
  amended below — v3 remains the live freeze for every section this v4 does
  not touch.
- This v4 re-binds ONLY the C1/M1 boundary to the accepted V2
  checkpoints/path, per the Task 5 gate decision (Main: do not ratify v3's
  false C1 boundary). It was written before any Task 6+ outcome was
  observed; v1/v2/v3 outcomes MUST NOT select v4 content.
- Evidence: `artifacts/sprint-16/task5-remote-inspection.md` (bounded
  read-only remote inspection with exact commands/raw output) and the
  corrected M1 measurement (`artifacts/sprint-16/task5-normalization.json`,
  `artifacts/sprint-16/task-5.md`).

## §1. C1 boundary correction (replaces v3 §3 C1 row, §4.4 hierarchy sentence, §6 M1, §8 C1 variants)

1. The accepted V2 production path contains NO external/conditional
   input-normalization stage and NO data-dependent normalization statistics:
   both accepted checkpoint configs (28 V2Config keys) carry no norm flag;
   both state dicts (schema 2, 71 tensors) carry zero `conditional*` keys
   and zero `running_*`/`num_batches_tracked` keys; `v2_inference.py`,
   `v2_config.py`, `v2_patch.py` (except the context-mixer head norm below),
   and `synth/patchify.py` contain no input-normalization operator.
2. The only normalization parameters anywhere in the accepted path are
   parameter-only LayerNorm weights/biases: 18 tensors inside
   `SequenceContextEncoder` (layers.0..3 norm1/norm2 + final norm), 2 in
   `LocalPatchEncoder.output`, 1 in the v2 context-mixer head
   (`ContextConditionedPatchEncoder.__init__`, over learned
   robot/program/regime embeddings — scorer/head territory, never waveform
   input). These belong to C3/C4 (and the head, C8-adjacent) and are measured
   by Tasks 7/8 — never by Task 5.
3. The production stage-1 operator is therefore the verified identity.
   Pre/post C1 comparison is identity/absent by construction.
4. v3 §4.4's "production 3-level hierarchy (robot+program → robot → fleet)
   evaluated only at C1/M1" is V1-only and has no code boundary in the
   accepted path; it is STRUCK for execution (preserved as history in v3).
   No new normalization convention is introduced.

## §2. M1 disposition (Task 5, executed)

M1 compared pre- vs post-stage-1 observable probes with the Task 3 contract
reader on identical support across H-CONF-34..37: per-history/category
R_pre/R_post computed (P1 point 0.8485 reproduces the Task 4 ceiling exactly),
all 36 gaps exactly 0.0, all probe families bit-identical, positive control
sensitive. No localization evidence at C1.

## §3. C1 verdict and intervention rules (bind Tasks 14/16)

1. C1 can NEVER be a BOTTLENECK: there is no stage to bypass and no loss
   mechanism to restore.
2. At Task 16, C1 resolves UNRESOLVED with reason NOT PRESENT
   (evidence-insufficient: stage absent from the accepted path). The plan
   Task 16 menu admits this via its evidence-insufficient UNRESOLVED.
3. N-a/N-b/N-c (v3 §8) are reclassified as ADDITIVE diagnostic replacements,
   never bypasses: executing them adds a stage production lacks. They may
   still run under Task 14 within budget as diagnostic contrasts, but no
   N-variant result alone, and no N-variant result combined with the M1 zero
   gap, can establish C1 causality or convert the C1 UNRESOLVED.
4. Nothing in this amendment authorizes touching Sealed histories, refit-True
   banks, or any non-V2 checkpoint for C1 purposes.

## §4. Standing sections

Every v3 section not contradicted above (§§1–2 inputs/roles, §§4–4.5 reader
contract, §5 slices, §§6 M2–M8, §7 matrix, §§8 C2–C9 variants + budget,
§§9–11 stopping/verdicts/non-negotiables, §12 traceability, §13 checklist,
Appendix A) stands unchanged and remains executable for Tasks 6–16.
