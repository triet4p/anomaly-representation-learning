# Sprint 13 Benchmark Protocol v4.1.1 — Implementation-Conformance Correction (FROZEN, AWAITING MAIN APPROVAL)

**Status:** DRAFT CORRECTION frozen by worker before any re-audit. NOT
APPROVED — no Task 13/14 rerun, resealing, regeneration, or verdict
recomputation under v4.1.1 yet. Amends the *measurement-code* reading of
approved v4.1 (semantic SHA256
`94f9c566067df2d3e9e7a8a638bcf80d3f34a4b350e56d3b0b283a83ea367b14`, which
stays byte-identical); it is not a new data-generating protocol (no DGP)
and predeclares no new roster or seeds.
**Reason:** fresh Deep Review 2 FAIL (4 findings: temporal-view omission,
per-window/episode reset semantics, incomplete E1–E5 contract, plan
checkboxes) against `artifacts/sprint-13/deep-review-2.md`.

## 1. Reuse rationale (why no regeneration)

Generator parameters, code paths producing bulk bytes, the committed v4.1
profile, and the 500–508/600–603 roster are byte-identical to the approved
v4.1 execution: this correction changes only how the frozen measurement
semantics select and summarize already-materialized rows. Regenerating
identical bulk data would add no information while risking seed/calendar
drift; the existing manifests therefore remain immutable and are reused.
What is superseded is DERIVED verdict evidence only (v4.1 audit/proof
JSONs, seals-as-verdict-records, STRUCT states, task-12/13/14/15 texts);
fresh structural re-audit, non-sealed proof, resealing, re-verdict, and
renewed C1/D1/D2/Task16 gates follow approval.

## 2. Frozen corrections (measurement code must implement exactly this)

1. **Temporal-view membership** (`test-temporal` in `member_views`) is
   required in POS-ELIGIBLE rows, ANCHOR-ELIGIBLE rows (hence control
   members), and E5 eligible-surveillance rows. Pre-cutoff DEV rows never
   enter evaluation controls or denominators.
2. **Whole-window reset semantics**, centralized in
   `window_intersects_reset` / `positive_window_intersects_reset`:
   a positive `[T−H, T]` horizon intersecting any
   maintenance/recommission boundary is unevaluable (counted, never
   imputed); a control candidate `[e−H, e]` intersecting one is rejected;
   alert episodes split at resets (no bridging). Row-level overlap checks
   remain for file eligibility; the span rule governs units and episodes.
3. **Complete E1–E5 output contract** (non-sealed fixture proof):
   E1 tie-aware AUC value; E2 median+IQR with `SPARSE(n)` below 10 recalled
   events; E3 recall value plus exact Clopper–Pearson 95% interval; E4
   median+IQR with the same SPARSE rule; E5 point FAR plus the zero-count
   rule-of-three bound. `computable` validates the full composite output,
   never AUC alone. Sealed roles stay hard-blocked from fixture scoring;
   no file-AUROC anywhere.

## 3. Unchanged and carried forward

v4.1 §§2–5 physics, rates, roster, roles, holdouts, quarantine, horizons,
clearance, censoring, floors, §9a verdict logic, sealing format, and the
Task 14 non-sealed-only predeclared procedure. Sealed access stays
structural-only Task 13 until a future score gate; fixture scoring stays
non-sealed-only. Manifests and bulk roots are immutable.

## 4. Execution plan (post-approval — NOT R3)

After Main approval: re-run the structural audit and non-sealed fixture
proof from the approved commit; rewrite seals at Task 15 renewal;
recompute STRUCT states and the §9a verdict; renew C1, D1, D2, Task 16
with fresh reviews; then Task 17 fresh re-review. R3 delivers code +
contracts + focused test evidence only.
