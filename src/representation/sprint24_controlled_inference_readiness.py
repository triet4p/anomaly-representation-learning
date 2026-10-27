"""Sprint 24 Task 7-M10 frozen controlled-corpus inference readiness (metadata only).

Prospective, pre-outcome readiness contract for movement inference on the
official M9 controlled corpus, under the owner-approved prospective estimand
amendment ``S24-CONTROLLED-INFERENCE-AMENDMENT-A01``
(``docs/sprint-plans/sprint-24.md:205``).  This module reads only persisted
metadata (corpus manifest, realized slots, observed audit rows, NPZ array
headers for durations); it loads no model, no checkpoint, no embedding, no
score, performs no forward, and reads no waveform signal values.

It establishes the executable path and the prospectively frozen analysis
contract before any new model outcome is inspected:

- a metadata-only eligibility census for every history/horizon/pair and the
  primary/replication triples, measuring the unchanged exposure floor
  ``0.8`` on the retained label-blind capped-``n`` matched files, with the
  raw arm-level eligibility preserved as the inherited-rule
  limitation/sensitivity;
- explicit resolution of the disputed quantities: shared-arm pairs (dependent,
  disclosed, never independent), constant zero post-balancing mix drift
  (residual ``C2`` abstains corpus-wide, never duplicated as ``C1`` and never
  an independent confirmation), five-pair ranks (arithmetically supported but
  coarse, descriptive only);
- the descriptive branch: controlled-corpus movement/``C1`` versus baseline
  controls with history/seed replication; the old co-required ``C1``/``C2``
  PROMISING gate is not applicable and no replacement success gate is
  authorized;
- the causal per-new-history reference-fitting plan (bandwidth, prototypes,
  standardization on the causal pool only; no encoder or restored
  anomaly-bank refit);
- deterministic cost/resume/cache custody and the exact missing prerequisites
  (no controlled-corpus embedding cache exists; capture needs a separate
  execution release with the accepted frozen B0 checkpoints/banks).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from representation import sprint24_controlled_data_execution as execution
from representation import sprint24_controlled_data_protocol as v1
from representation.sprint24_context_audit import sha256_object

#: Versioned readiness-contract identifier frozen by this module.
READINESS_ID = "S24-CONTROLLED-INFERENCE-READINESS-v1"

#: Owner-approved prospective estimand amendment (sprint-24.md:205).
AMENDMENT_ID = "S24-CONTROLLED-INFERENCE-AMENDMENT-A01"

#: Base execution protocol this readiness builds on (bytes preserved, never edited).
BASE_EXECUTION_ID = execution.PROTOCOL_ID

#: Base design protocol (bytes preserved, never edited).
BASE_DESIGN_ID = v1.PROTOCOL_ID

#: Inherited exposure-mismatch floor, re-declared unchanged (never re-tuned).
EXPOSURE_RATIO_FLOOR = v1.EXPOSURE_RATIO_FLOOR

#: Inherited label-blind member-selection null seed, re-declared unchanged.
NULL_SEED = v1.NULL_SEED

#: Accepted frozen encoder seeds (identity-selected run replicates, never
#: independent data).  The encoder stays frozen as-is.
FROZEN_ENCODER_SEEDS = v1.FROZEN_ENCODER_SEEDS

#: Primary and replication triples (lower-bound research scale, not a power claim).
PRIMARY_TRIPLE = v1.PRIMARY_TRIPLE
REPLICATION_TRIPLE = v1.REPLICATION_TRIPLE

#: Closed per-pair eligibility states under the amended measurement point.
ELIGIBILITY_STATES = (
    "ELIGIBLE-RETAINED",
    "RAW-LIMITATION-SENSITIVITY",
    "CONTROL-EXPECTED-NULL",
    "ABSTAIN",
)

#: Exact rank reference published beside every five-pair result: the smallest
#: two-sided permutation tail of a perfect rank association over 5 pairs.
FIVE_PAIR_MIN_TWO_SIDED_TAIL = 2.0 / 120.0

#: Failure/outcome states this readiness never produces (no inference executes).
NOT_EXECUTED = "READY-NO-FORWARDS-EXECUTED"

#: Closed refusal reasons for the readiness preflight itself.
REFUSAL_REASONS = (
    "REFUSED-OVERWRITE",
    "REFUSED-MISSING-CORPUS",
    "REFUSED-DIGEST-DRIFT",
    "REFUSED-IDENTITY",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def retained_exposure(
    retained_baseline_ids: Sequence[str],
    retained_contrast_ids: Sequence[str],
    end_time_of: Mapping[str, float],
    duration_of: Mapping[str, float],
) -> float | None:
    """Return the retained min/max member file-day exposure ratio (label-blind).

    Same overlap construction as the frozen ``PeriodPair.exposure_ratio`` but
    measured on the retained label-blind capped-``n`` matched files: for each
    retained arm, the summed overlap of ``[end_time - T, end_time]`` with the
    arm span ``[min end_time, max end_time]`` in file-days; the ratio is
    ``min/max``.  Uses only end-times and durations, never labels.
    """
    import math as _math

    base = [str(fid) for fid in retained_baseline_ids]
    cont = [str(fid) for fid in retained_contrast_ids]
    if not base or not cont:
        return None

    def _arm_exposure(fids: Sequence[str]) -> float | None:
        ends: list[float] = []
        for fid in fids:
            end = end_time_of.get(fid)
            dur = duration_of.get(fid)
            if not isinstance(end, (int, float)) or not isinstance(dur, (int, float)):
                return None
            if not _math.isfinite(float(end)) or not _math.isfinite(float(dur)) or float(dur) <= 0:
                return None
            ends.append(float(end))
        if not ends:
            return None
        lo, hi = min(ends), max(ends)
        total = 0.0
        for fid in fids:
            end = float(end_time_of[fid])
            start = end - float(duration_of[fid])
            total += max(0.0, min(end, hi) - max(start, lo))
        return total / 86400.0

    exp_base = _arm_exposure(base)
    exp_cont = _arm_exposure(cont)
    if exp_base is None or exp_cont is None:
        return None
    peak = max(exp_base, exp_cont)
    if peak <= 0:
        return None
    return min(exp_base, exp_cont) / peak


def expected_retained_census() -> dict[str, object]:
    """Return the frozen expected census under the amended measurement point.

    The threshold ``0.8`` is unchanged; only the measurement point moves from
    the unequal-duration raw arms to the retained label-blind capped-``n``
    matched files:

    - 48 TRANSITION pairs: retained exposure 0.852-0.995, all at/above 0.8,
      hence ``ELIGIBLE-RETAINED`` (descriptive movement/``C1`` branch);
    - 12 CONTROL pairs: retained exposure 0.885-0.992, all at/above 0.8, but
      ``CONTROL-EXPECTED-NULL`` by design (near-zero retained contrast, no
      rank variation; versus-controls reference, never a burden rank).

    The raw arm-level eligibility (0/48 transitions, 12/12 controls) is
    preserved as the inherited-rule limitation/sensitivity.  C2 abstains
    corpus-wide (identically zero matched mix drift) and must not be
    presented as independent confirmation of ``C1``.
    """
    return {
        "readiness_id": READINESS_ID,
        "amendment_id": AMENDMENT_ID,
        "exposure_floor": EXPOSURE_RATIO_FLOOR,
        "exposure_definition": "retained label-blind capped-n matched-file min/max file-day overlap exposure (same construction as frozen PeriodPair.exposure_ratio, measured on retained files); threshold 0.8 unchanged",
        "raw_limitation": {
            "definition": "raw arm-level min/max member file-day overlap exposure (frozen PeriodPair.exposure_ratio, pre-balancing)",
            "transition_range": [0.4393119465482919, 0.7312220166992238],
            "control_range": [0.936, 0.997],
            "raw_eligible_transitions": 0,
            "raw_eligible_controls": 12,
            "role": "inherited-rule limitation/sensitivity; raw-floor primary is PREDETERMINED-INCONCLUSIVE",
        },
        "n_histories": 6,
        "n_pairs": 60,
        "transition_pairs": 48,
        "control_pairs": 12,
        "retained_exposure_transition_range": [0.8520186853520187, 0.9953066697761194],
        "retained_exposure_control_range": [0.8852327447833066, 0.992145044611403],
        "retained_eligible_transitions": 48,
        "retained_eligible_controls": 12,
        "branch": "descriptive controlled-corpus movement/C1 versus baseline controls, with history/seed replication",
        "promising_gate": "NOT-APPLICABLE: the old co-required C1/C2 PROMISING gate is not applicable and no replacement success threshold is authorized",
        "mix_drift_after": 0.0,
        "mix_drift_after_constant_pairs": 60,
        "c2_status": "ABSTAIN",
        "c2_reason": "post-balancing mix drift is exactly 0.000 on all 60 pairs: the mix-adjusted residual regressor has zero variance, so C2 abstains corpus-wide, is never duplicated as C1, and never independently confirms C1",
        "five_pair_rank_reference": {
            "min_two_sided_permutation_tail": FIVE_PAIR_MIN_TWO_SIDED_TAIL,
            "note": "arithmetically supported but coarse; descriptive only, never a precise test",
        },
        "shared_arm_pairs": "dependent by construction (NONADJACENT-SHARED-SUPPORT); disclosed, never independent samples",
    }


def _pair_eligibility(
    row: Mapping[str, object],
    end_time_of: Mapping[str, float] | None = None,
    duration_of: Mapping[str, float] | None = None,
) -> dict[str, object]:
    raw = row.get("exposure_ratio_before")
    raw_value = float(raw) if isinstance(raw, (int, float)) else None
    comparison = str(row.get("comparison_type"))
    gap = row.get("burden_contrast_gap_after")
    gap_value = float(gap) if isinstance(gap, (int, float)) else None
    mix_after = row.get("mix_drift_tv_after")
    mix_value = float(mix_after) if isinstance(mix_after, (int, float)) else None
    retained: float | None = None
    if end_time_of is not None and duration_of is not None:
        base_ids = row.get("retained_baseline_file_identities")
        cont_ids = row.get("retained_contrast_file_identities")
        if isinstance(base_ids, list) and isinstance(cont_ids, list):
            retained = retained_exposure(base_ids, cont_ids, end_time_of, duration_of)
    support_ok = bool(row.get("meets_support_minima"))
    if raw_value is None or retained is None:
        state = "ABSTAIN"
        reason = "retained or raw exposure unavailable"
    elif not support_ok:
        state = "ABSTAIN"
        reason = "retained support minima not met (independent of exposure)"
    elif comparison == "BASELINE-VS-BASELINE":
        if retained < EXPOSURE_RATIO_FLOOR:
            state = "ABSTAIN"
            reason = (
                f"control fails the retained exposure floor ({retained:.3f} < "
                f"{EXPOSURE_RATIO_FLOOR}); abstains independently of its null role"
            )
        else:
            state = "CONTROL-EXPECTED-NULL"
            reason = (
                "baseline-vs-baseline control: retained support and floor met with "
                f"near-zero retained contrast by design (retained exposure {retained:.3f}); "
                "member of the descriptive C1 association set as the low-burden pole, "
                "per the unchanged M8 C1 support rule"
            )
    elif retained >= EXPOSURE_RATIO_FLOOR:
        state = "ELIGIBLE-RETAINED"
        reason = (
            f"retained exposure {retained:.3f} meets the unchanged floor "
            f"{EXPOSURE_RATIO_FLOOR} on label-blind capped-n matched files; "
            f"raw arm-level exposure {raw_value:.3f} preserved as inherited-rule limitation"
        )
    else:
        state = "RAW-LIMITATION-SENSITIVITY"
        reason = (
            f"retained exposure {retained:.3f} below the unchanged floor "
            f"{EXPOSURE_RATIO_FLOOR}; raw exposure {raw_value:.3f} likewise recorded"
        )
    return {
        "comparison_identity": str(row.get("comparison_identity")),
        "history_id": str(row.get("history_id")),
        "horizon": str(row.get("horizon")),
        "comparison_type": comparison,
        "raw_exposure_ratio": raw_value,
        "retained_exposure_ratio": retained,
        "retained_gap_after": gap_value,
        "mix_drift_after": mix_value,
        "meets_support_minima": support_ok,
        "eligibility": state,
        "eligibility_reason": reason,
        "c1_association_member": state in ("ELIGIBLE-RETAINED", "CONTROL-EXPECTED-NULL"),
        "c1_role": (
            "low-burden pole of the 5-pair descriptive C1 set" if state == "CONTROL-EXPECTED-NULL"
            else ("burden-bearing member of the 5-pair descriptive C1 set" if state == "ELIGIBLE-RETAINED" else "not in the C1 set")
        ),
        "c2_status": "ABSTAIN",
        "c2_reason": "constant zero post-balancing mix drift corpus-wide (degenerate regressor); never duplicated as C1, never independent confirmation",
    }


def eligibility_census(
    pair_rows: Sequence[Mapping[str, object]],
    end_time_of: Mapping[str, float] | None = None,
    duration_of: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Census every observed pair under the amended measurement point."""
    pairs = [_pair_eligibility(row, end_time_of, duration_of) for row in pair_rows]
    by_history: dict[str, dict[str, int]] = {}
    history_names: dict[str, str] = {}
    for entry in pairs:
        history = str(entry["history_id"])
        history_names[history] = history
        block = by_history.setdefault(history, {"n_pairs": 0, "retained": 0, "raw_limitation": 0, "control": 0, "abstain": 0})
        block["n_pairs"] = int(block["n_pairs"]) + 1
        state = str(entry["eligibility"])
        if state == "ELIGIBLE-RETAINED":
            block["retained"] = int(block["retained"]) + 1
        elif state == "RAW-LIMITATION-SENSITIVITY":
            block["raw_limitation"] = int(block["raw_limitation"]) + 1
        elif state == "CONTROL-EXPECTED-NULL":
            block["control"] = int(block["control"]) + 1
        else:
            block["abstain"] = int(block["abstain"]) + 1
    triple = {
        "primary_triple": [str(h) for h in PRIMARY_TRIPLE],
        "replication_triple": [str(h) for h in REPLICATION_TRIPLE],
        "retained_transition_pairs": sum(
            1 for p in pairs if p["eligibility"] == "ELIGIBLE-RETAINED" and p["comparison_type"] != "BASELINE-VS-BASELINE"
        ),
        "c1_association_pairs_per_history_horizon": 5,
        "replication_note": "both triples carry 24 retained-eligible transitions each (4/history/horizon) plus 6 retained controls; descriptive 5-pair C1 sets (4 transitions + control pole) with history/seed replication, no success gate",
    }
    by_history_out: dict[str, dict[str, object]] = {
        history: {"history_id": history_names[history], **counts} for history, counts in by_history.items()
    }
    return {
        "readiness_id": READINESS_ID,
        "amendment_id": AMENDMENT_ID,
        "exposure_floor": EXPOSURE_RATIO_FLOOR,
        "exposure_definition": "retained label-blind capped-n matched-file min/max file-day overlap exposure; threshold 0.8 unchanged; raw arm-level eligibility preserved as inherited-rule limitation; exposure and retained-support applied independently",
        "n_pairs": len(pairs),
        "pairs": pairs,
        "by_history": by_history_out,
        "triples": triple,
        "branch": "descriptive controlled-corpus movement/C1 over 5-pair sets (4 retained transitions + retained control pole) versus baseline controls",
        "promising_gate": "NOT-APPLICABLE",
        "c2_status": "ABSTAIN",
    }


def reference_fitting_plan() -> dict[str, object]:
    """Return the causal per-new-history reference-fitting plan (no outcome)."""
    return {
        "scope": "per new controlled history and per representation seed, fitted once on its own causal reference pool",
        "pool": "eligible files with end_time <= c1 (220 per history), fixed before any evaluation period exists",
        "fitted_once": ["gaussian kernel bandwidth (median pairwise distance on the causal pool)", "E2 cluster prototypes (K=8; 4/16 sensitivity)", "per-dimension standardization"],
        "never_refitted": ["per period", "per pair", "per sensitivity variant", "across histories"],
        "forbidden": ["encoder refit", "restored anomaly-bank refit", "old-history bank reuse", "any fit on evaluation periods"],
        "shortfall": "reference state below 200 files refuses the history (REFUSED-SHORTFALL)",
    }


def model_input_contract() -> dict[str, object]:
    """Return the label-free model-input contract (field separation only)."""
    return {
        "encoder_visible": "waveform channels only (MODEL_INPUT_FIELD_NAMES); no file label, anomaly mask, severity, arm marker, episode, diagnostic or scheduled artifact may enter encoder inputs",
        "label_join": "labels and burden annotations join only after membership by frozen identity, never during selection",
        "admission": "cache admission and movement reuse the accepted frozen B0 encoder custody pattern; no training, no refit",
    }


def endpoint_contract() -> dict[str, object]:
    """Return the frozen endpoint/null/sensitivity/uncertainty/multiplicity contract."""
    return {
        "endpoint_hierarchy": "E1 primary; E2 secondary; E3/E4 descriptive decomposition; N1/N2 descriptive-only; C1 descriptive burden association; C2 ABSTAIN corpus-wide (degenerate regressor, never duplicated as C1, never independent confirmation)",
        "estimand_unit": "the history; descriptive within-history movement/C1 versus baseline controls over the retained-eligible pair set; encoder seeds are run replicates summarized by median, never independent data; no pooling across histories, seeds or horizons",
        "movement": "E1 unbiased squared MMD, Gaussian kernel with bandwidth from the NEW causal reference pool; E2 total-variation cluster occupancy (K=8; 4/16 sensitivity)",
        "nulls": "N1 reference split-half null (crossing necessary, never sufficient); N2 within-period half-split band; support abstentions stay in the denominator",
        "selection": "every retained-eligible pair enters; no pair selected, ranked or dropped by outcome; raw-limited pairs recorded as inherited-rule limitation",
        "multiplicity": "Holm over the declared primary contrasts at alpha=0.05 where computable; disposition does not depend on a p-value",
        "rank_reference": f"with 5 pairs the smallest two-sided permutation tail is 2/5! = {FIVE_PAIR_MIN_TWO_SIDED_TAIL:.4f}",
        "uncertainty": "history-cluster bootstrap, B=2000, seed 2407, descriptive at n=6; pairs inside a history are dependent so no pair-level bootstrap is valid",
        "dependence": "comparisons sharing an arm are NONADJACENT-SHARED-SUPPORT (dependent by construction); pair counts are support accounting, never independent samples",
        "decision": "NOT-APPLICABLE: the old co-required C1/C2 PROMISING gate is not applicable and no replacement success threshold is authorized; descriptive movement/C1 versus baseline controls with history/seed replication",
    }


def cost_custody_plan(n_files: int = 8280, n_seeds: int = 3) -> dict[str, object]:
    """Return the deterministic forward-cost envelope and cache-custody plan."""
    forwards = n_files * n_seeds
    return {
        "files_per_history": 1380,
        "n_histories": 6,
        "total_files": n_files,
        "representation_seeds": list(FROZEN_ENCODER_SEEDS),
        "planned_forwards": forwards,
        "smoke_envelope_files": 96,
        "historical_observed_forwards": 52789,
        "gate": NOT_EXECUTED,
        "gate_meaning": "readiness executes 0 forwards, loads 0 models, reads 0 waveforms/embeddings; official inference needs the separate execution release under standing owner budget authority",
        "resume": "per (history, seed) cache groups with exact cache-identity matching; any identity mismatch fails closed, never rescues",
        "cache_custody": "declared embedding-cache stores inspected under identity-strict matching; near-miss identities reported, never silently reused",
        "storage_assumption": "mean ~400 steps/file research-density planning assumption retained; observed T range 179-1404 recorded by the M9 audit",
    }
def missing_prerequisites() -> dict[str, object]:
    """Return the exact scientific prerequisites still missing (no placeholders)."""
    return {
        "embedding_cache": {
            "status": "ABSENT",
            "detail": "no controlled-corpus embedding cache exists locally; no (history, seed) cache group has been captured for S24C-ITER-0001 histories",
        },
        "capture_release": {
            "status": "REQUIRED",
            "detail": "official embedding capture needs the separate owner execution release; capture reuses the accepted frozen B0 checkpoints/banks under exact cache-identity matching",
        },
        "encoder_custody_basis": {
            "status": "PINNED-AVAILABLE",
            "detail": "accepted frozen B0 checkpoints/banks/arms/device recorded by the accepted G2 preflight (181801/181802/181803) are the custody basis for a future capture; they are cited, never loaded, by this readiness",
        },
        "exposure_gate_settled": {
            "status": "AMENDED-BEFORE-OUTCOME",
            "detail": "owner amendment S24-CONTROLLED-INFERENCE-AMENDMENT-A01: threshold 0.8 unchanged, measured on retained label-blind capped-n matched files (48/48 transitions eligible); raw arm-level eligibility (0/48 transitions) preserved as inherited-rule limitation; no floor reduction",
        },
        "c2_endpoint": {
            "status": "ABSTAIN-CORPUS-WIDE",
            "detail": "post-balancing mix drift is exactly 0.000 on all 60 pairs; C2 abstains everywhere, is never duplicated as C1, and never independently confirms C1",
        },
    }


def read_npy_shape_header(archive_path: Path, member: str = "x.npy") -> tuple[int, ...]:
    """Return an NPZ member's array shape reading only its NPY header bytes.

    Opens the ZIP member stream and reads at most the NPY header (magic +
    version + dict, typically 128 bytes), then closes the stream without
    reading any array element.  No waveform signal value is materialized.
    """
    import ast as _ast
    import zipfile as _zipfile

    with _zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            prefix = stream.read(10)
            if len(prefix) < 10 or prefix[:6] != b"\x93NUMPY":
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} is not NPY")
            header_len = int.from_bytes(prefix[8:10], "little")
            body = stream.read(header_len)
            if len(body) < header_len:
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} header truncated")
            header = _ast.literal_eval(body.decode("latin1"))
            if not isinstance(header, dict):
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} header is not a mapping")
            shape = header.get("shape")
            if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} shape missing")
            return shape


def _manifest_file_maps(corpus_manifest: Mapping[str, object]) -> tuple[dict[str, float], dict[str, float], int]:
    """Return label-blind end-time/duration maps from the corpus manifest.

    Durations come from NPZ ``x.npy`` NPY headers only (shape bytes; no array
    element is read); end-times come from the manifest.  No waveform signal
    value, label, mask, severity or arm marker is read.  The third return is
    the count of NPZ header reads performed.
    """
    from pathlib import Path as _Path

    files = corpus_manifest.get("files")
    if not isinstance(files, list):
        return {}, {}, 0
    by_id: dict[str, Mapping[str, object]] = {}
    for entry in files:
        if isinstance(entry, Mapping):
            fid = entry.get("file_id")
            if isinstance(fid, str):
                by_id[fid] = entry
    manifest_path = corpus_manifest.get("__manifest_path__")
    root = _Path(str(manifest_path)).parent if isinstance(manifest_path, str) else None
    end_time_of: dict[str, float] = {}
    duration_of: dict[str, float] = {}
    header_reads = 0
    for fid, entry in by_id.items():
        end = entry.get("end_time_rel_s")
        if isinstance(end, (int, float)):
            end_time_of[fid] = float(end)
        rel = entry.get("relpath")
        if root is not None and isinstance(rel, str):
            try:
                shape = read_npy_shape_header(root / rel)
                duration_of[fid] = float(int(shape[1]))
                header_reads += 1
            except (OSError, KeyError, ValueError, IndexError):
                continue
    return end_time_of, duration_of, header_reads


def validate_custody_inputs(g2_preflight: Mapping[str, object]) -> dict[str, object]:
    """Validate the accepted G2 custody record against pinned accepted constants.

    Checks the fields that have accepted module-level pins (protocol digest,
    binding digest, G2 source commit, cuda device identity, all three
    representation seeds) and requires well-formed 64-hex checkpoint/bank/arm
    digests for every seed.  A forged or truncated custody file fails closed
    here; no checkpoint is loaded and no forward executes.
    """
    from representation import sprint24_weekly_monthly_execution as _execution

    problems: list[str] = []
    protocol = str(g2_preflight.get("protocol_sha256"))
    if protocol != _execution.ACCEPTED_G2_PROTOCOL_SHA256:
        problems.append(f"G2 protocol digest mismatch: {protocol}")
    binding = str(g2_preflight.get("binding_sha256"))
    if binding != _execution.ACCEPTED_BINDING_SHA256:
        problems.append(f"G2 binding digest mismatch: {binding}")
    source = str(g2_preflight.get("source_commit"))
    if source != _execution.ACCEPTED_G2_SOURCE_COMMIT:
        problems.append(f"G2 source commit mismatch: {source}")
    device = g2_preflight.get("device")
    identity = device.get("identity") if isinstance(device, Mapping) else device
    if str(identity) != _execution.REQUIRED_DEVICE_IDENTITY:
        problems.append(f"G2 device identity is not {_execution.REQUIRED_DEVICE_IDENTITY}: {identity}")
    seeds = [str(seed) for seed in FROZEN_ENCODER_SEEDS]
    tables: dict[str, object] = {}
    for field in ("checkpoints", "banks", "arms"):
        table = g2_preflight.get(field)
        if not isinstance(table, Mapping) or sorted(str(key) for key in table) != seeds:
            problems.append(f"G2 {field} table does not cover the accepted seeds {seeds}")
            continue
        digests = {str(key): str(table[key]) for key in seeds}
        tables[field] = digests
        for key, digest in digests.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                problems.append(f"G2 {field}[{key}] is not a lowercase SHA-256 digest")
    if problems:
        raise ValueError(f"REFUSED-IDENTITY: accepted G2 custody invalid: {'; '.join(problems)}")
    checked = dict(tables)
    checked["protocol_sha256"] = protocol
    checked["binding_sha256"] = binding
    checked["preprocessing_sha256"] = _execution.ACCEPTED_PREPROCESSING_SHA256
    checked["device"] = _execution.REQUIRED_DEVICE_IDENTITY
    return checked


def custody_identity_plan(
    corpus_manifest: Mapping[str, object],
    custody: Mapping[str, object],
) -> dict[str, object]:
    """Derive the executable per-group cache-identity plan from accepted machinery.

    Reuses :func:`execution.cache_identity` end to end: per new history, the
    support digest is the canonical digest of that history's required file
    IDs (reference + analysis files from the corpus manifest), and one cache
    group per (history, representation seed) carries the exact study-required
    identity built from the validated custody inputs.  No row is claimed
    available; the plan is the executable capture/admission target that a
    future execution release must satisfy under identity-strict matching.
    """
    from representation import sprint24_weekly_monthly_execution as _execution

    files = corpus_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("REFUSED-IDENTITY: corpus manifest has no file table")
    per_history_ids: dict[str, list[str]] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            continue
        history = entry.get("history_id")
        fid = entry.get("file_id")
        if isinstance(history, str) and isinstance(fid, str):
            per_history_ids.setdefault(history, []).append(fid)
    inputs: dict[str, object] = {
        "checkpoints": dict(custody["checkpoints"]),  # type: ignore[index]
        "banks": dict(custody["banks"]),  # type: ignore[index]
        "arms": dict(custody["arms"]),  # type: ignore[index]
        "binding_sha256": str(custody["binding_sha256"]),
        "preprocessing_sha256": str(custody["preprocessing_sha256"]),
        "device": str(custody["device"]),
        "scale_state_sha256": None,
    }
    groups: dict[str, dict[str, object]] = {}
    for history_id in sorted(per_history_ids):
        required_ids = sorted(set(per_history_ids[history_id]))
        support_sha256 = sha256_object(required_ids)
        for seed in FROZEN_ENCODER_SEEDS:
            identity = _execution.cache_identity(inputs, model_seed=int(seed), support_sha256=support_sha256)
            groups[f"{history_id}|{seed}"] = {
                "history_id": history_id,
                "model_seed": int(seed),
                "required_rows": len(required_ids),
                "support_sha256": support_sha256,
                "cache_identity": identity,
                "device": str(custody["device"]),
                "preprocessing_sha256": str(custody["preprocessing_sha256"]),
                "data_binding_sha256": str(custody["binding_sha256"]),
                "scale_state_sha256": "none",
            }
    return {
        "derivation": "execution.cache_identity over validated accepted custody inputs and per-history required file IDs",
        "n_groups": len(groups),
        "n_histories": len(per_history_ids),
        "n_seeds": len(FROZEN_ENCODER_SEEDS),
        "groups": groups,
        "admission_rule": (
            "a cached row is reusable only when a valid cache entry carries the exact "
            "group identity; the future capture must satisfy every group before any "
            "movement endpoint is computed"
        ),
    }


def read_cache_file_ids_only(archive_path: Path, member: str = "file_ids.npy") -> list[str]:
    """Return a cache NPZ's file-ID list reading only that member's bytes.

    Opens the ZIP member stream for ``file_ids`` and parses its NPY array
    without ever opening the ``embeddings`` member.  No embedding value is
    read, loaded or materialized.
    """
    import zipfile as _zipfile

    import numpy as _np
    from numpy.lib import format as _format

    with _zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as stream:
            _format.read_magic(stream)
            shape, fortran, dtype = _format.read_array_header_1_0(stream)
            if dtype.kind not in ("U", "S"):
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} has unexpected dtype {dtype}")
            count = 1
            for dim in shape:
                count *= int(dim)
            raw = stream.read(count * int(dtype.itemsize))
            if len(raw) != count * int(dtype.itemsize):
                raise ValueError(f"REFUSED-IDENTITY: {archive_path} member {member} truncated")
            array = _np.frombuffer(raw, dtype=dtype)
            if fortran and len(shape) > 1:
                array = array.reshape(shape, order="F")
            else:
                array = array.reshape(shape)
            values = array.tolist()
            return [v.decode("ascii") if isinstance(v, bytes) else str(v) for v in values]

def inspect_cache_manifest_only(root: Path) -> dict[str, object]:
    """Inspect one cache root from directory metadata only (never embedding bytes).

    For every ``*.npz`` file: the JSON sidecar is loaded and validated
    (schema, kind, provenance/commit rule); the cache identity is re-derived
    from the sidecar's own inputs via :func:`execution.cache_identity` and
    required to match; membership comes from parsing ONLY the ``file_ids``
    member stream (central-directory sizes/CRCs plus that member's NPY
    header/array bytes — the ``embeddings`` member is never opened, read or
    hashed); ``n_rows`` must equal the member count with no duplicates.
    No whole-file byte hash is computed: hashing the file would read the
    stored embedding bytes.  Value-level validation belongs to the future
    execution release, which alone may load embedding values.
    """
    from representation import sprint24_weekly_monthly_execution as _execution

    entries: list[dict[str, object]] = []
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "status": "MISSING",
            "npz_count": 0,
            "entries": entries,
            "valid_entries": 0,
            "invalid_entries": 0,
            "total_manifest_rows": 0,
            "note": "declared root does not exist; absence is unestablished for this root",
        }
    for npz_path in sorted(root.glob("*.npz")):
        issues: list[str] = []
        sidecar_path = npz_path.with_suffix(".json")
        record: dict[str, object] = {
            "path": str(npz_path),
            "sidecar_path": str(sidecar_path),
            "root": str(root),
            "valid": False,
            "issues": issues,
            "n_rows": 0,
            "file_ids": [],
        }
        if not sidecar_path.is_file():
            issues.append("missing identity sidecar")
            entries.append(record)
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            issues.append(f"unreadable sidecar: {exc}")
            entries.append(record)
            continue
        if not isinstance(sidecar, dict):
            issues.append("sidecar is not a JSON object")
            entries.append(record)
            continue
        for field in ("history_id", "model_seed", "device", "support_sha256", "cache_identity"):
            record[field] = sidecar.get(field)
        if str(sidecar.get("schema")) != _execution.CACHE_SIDECAR_SCHEMA:
            issues.append(f"unexpected sidecar schema {sidecar.get('schema')!r}")
        if str(sidecar.get("kind")) != _execution.CACHE_SIDECAR_KIND:
            issues.append(f"unexpected sidecar kind {sidecar.get('kind')!r}")
        provenance = str(sidecar.get("capture_provenance"))
        if provenance not in (
            _execution.CACHE_PROVENANCE_GPU,
            _execution.CACHE_PROVENANCE_SMOKE,
            _execution.CACHE_PROVENANCE_FIXTURE,
        ):
            issues.append(f"unknown capture provenance {provenance!r}")
        else:
            commit = sidecar.get("capture_commit")
            if provenance == _execution.CACHE_PROVENANCE_GPU:
                if not isinstance(commit, str) or len(commit) != 40:
                    issues.append("a full GPU capture must record its 40-character execution commit")
            elif commit is not None:
                issues.append("non-GPU provenance must not record an execution commit")
        try:
            recomputed = _execution.cache_identity(
                {
                    "checkpoints": {str(sidecar.get("model_seed")): str(sidecar.get("checkpoint_sha256"))},
                    "banks": {str(sidecar.get("model_seed")): str(sidecar.get("bank_sha256"))},
                    "arms": {str(sidecar.get("model_seed")): str(sidecar.get("arm_config_sha256"))},
                    "binding_sha256": str(sidecar.get("data_binding_sha256")),
                    "preprocessing_sha256": str(sidecar.get("preprocessing_sha256")),
                    "device": str(sidecar.get("device")),
                    "scale_state_sha256": None if str(sidecar.get("scale_state_sha256")) == "none" else str(sidecar.get("scale_state_sha256")),
                },
                model_seed=int(sidecar.get("model_seed", -1)),  # type: ignore[arg-type]
                support_sha256=str(sidecar.get("support_sha256")),
            )
        except (TypeError, ValueError) as exc:
            recomputed = None
            issues.append(f"identity inputs are invalid: {exc}")
        if recomputed is not None and recomputed != str(sidecar.get("cache_identity")):
            issues.append("sidecar cache_identity does not match its own identity inputs")
        try:
            member_facts = read_cache_member_facts(npz_path)
            file_ids = read_cache_file_ids_only(npz_path)
        except (OSError, KeyError, ValueError) as exc:
            issues.append(f"unreadable cache member metadata: {exc}")
            entries.append(record)
            continue
        record["n_rows"] = len(file_ids)
        record["file_ids"] = file_ids
        record["file_ids_sha256"] = sha256_object(sorted(file_ids))
        record["member_facts"] = member_facts
        if len(set(file_ids)) != len(file_ids):
            issues.append("file_ids contain duplicates")
        try:
            declared = int(sidecar.get("n_rows", -1))
        except (TypeError, ValueError):
            declared = -1
        if declared != len(file_ids):
            issues.append(f"sidecar n_rows {sidecar.get('n_rows')} differs from manifest rows {len(file_ids)}")
        record["valid"] = not issues
        record["issues"] = issues
        entries.append(record)
    valid = [entry for entry in entries if entry["valid"]]
    invalid = [entry for entry in entries if not entry["valid"]]
    if not entries:
        status = "ABSENT"
    elif invalid and not valid:
        status = "INVALID"
    elif invalid:
        status = "PARTIAL"
    else:
        status = "NON-EMPTY"
    return {
        "root": str(root),
        "exists": True,
        "status": status,
        "npz_count": len(entries),
        "entries": entries,
        "valid_entries": len(valid),
        "invalid_entries": len(invalid),
        "total_manifest_rows": sum(int(entry["n_rows"]) for entry in valid),  # type: ignore[arg-type]
        "embedding_values_opened": 0,
        "embedding_bytes_read": 0,
    }


def read_cache_member_facts(archive_path: Path) -> dict[str, object]:
    """Return per-member central-directory facts without reading member data.

    Member names, compressed/uncompressed sizes and CRC32 come from the ZIP
    central directory at the end of the file; no member byte stream is
    opened, so no embedding byte is read — not even for hashing.
    """
    import zipfile as _zipfile

    with _zipfile.ZipFile(archive_path) as archive:
        members = [
            {
                "name": info.filename,
                "compress_size": int(info.compress_size),
                "file_size": int(info.file_size),
                "crc32": format(int(info.CRC), "08x"),
            }
            for info in archive.infolist()
        ]
    names = sorted(entry["name"] for entry in members)
    if "file_ids.npy" not in names:
        raise ValueError(f"REFUSED-IDENTITY: {archive_path} has no file_ids member")
    return {"members": members, "member_names": names}


def probe_declared_cache_roots(cache_roots: Sequence[str | Path]) -> dict[str, object]:
    """Report the declared cache roots from directory metadata only (never values).

    An empty declared list is ``NOT-CHECKED/UNKNOWN``.  A declared root that
    is missing reports ``MISSING``; a declared root with no NPZ files reports
    ``ABSENT``; a root whose entries are all invalid reports ``INVALID`` (a
    refusal, never absence); mixed roots report ``PARTIAL``; fully valid
    roots report ``NON-EMPTY``.  Near-miss identities (valid manifest rows
    whose cache identity differs from the contract plan) are listed per
    entry by the caller; this probe never consumes values.
    """
    roots = [Path(root) for root in cache_roots]
    if not roots:
        return {
            "roots": [],
            "verdict": "NOT-CHECKED/UNKNOWN",
            "detail": "no cache root declared; cache absence is unestablished, not proven",
            "stores": [],
            "npz_count": 0,
            "total_manifest_rows": 0,
            "embedding_values_opened": 0,
            "embedding_bytes_read": 0,
        }
    stores = [inspect_cache_manifest_only(root) for root in roots]
    total = sum(int(store["total_manifest_rows"]) for store in stores)  # type: ignore[arg-type]
    count = sum(int(store["npz_count"]) for store in stores)  # type: ignore[arg-type]
    statuses = sorted({str(store.get("status", "UNKNOWN")) for store in stores})
    if statuses == ["ABSENT"]:
        verdict = "ABSENT"
        detail = "declared roots exist and hold no cache files"
    elif "INVALID" in statuses or "PARTIAL" in statuses:
        verdict = "INVALID"
        detail = "declared roots hold malformed or identity-mismatched cache files; refusal, never absence"
    elif "MISSING" in statuses:
        verdict = "MISSING"
        detail = "declared roots are missing; absence is unestablished for those roots"
    elif "NON-EMPTY" in statuses:
        verdict = "NON-EMPTY"
        detail = "declared roots hold manifest rows; values never opened by this probe"
    else:
        verdict = "UNKNOWN"
        detail = "unrecognized store states; absence is unestablished"
    return {
        "roots": [str(root) for root in roots],
        "verdict": verdict,
        "detail": detail,
        "stores": stores,
        "npz_count": count,
        "total_manifest_rows": total,
        "embedding_values_opened": 0,
        "embedding_bytes_read": 0,
    }


def compare_probe_to_plan(
    probe: Mapping[str, object],
    identity_plan: Mapping[str, object],
) -> dict[str, object]:
    """Compare a manifest-only probe against the independently frozen plan.

    Emits per-entry near-miss records (valid manifest rows whose cache
    identity or support digest differs from the frozen plan group) and an
    overall verdict: ``MATCH`` (all rows match), ``NEAR-MISS`` (rows exist
    but identities differ), ``EMPTY`` (no rows), ``NOT-CHECKED`` (no roots),
    or ``REFUSED`` (missing/invalid stores).  Never consumes values.
    """
    raw_groups = identity_plan.get("groups", {}) if isinstance(identity_plan, Mapping) else {}
    plan: dict[str, object] = dict(raw_groups) if isinstance(raw_groups, dict) else {}
    stores = probe.get("stores")
    store_list = list(stores) if isinstance(stores, list) else []
    near_misses: list[dict[str, object]] = []
    n_rows = 0
    for store in store_list:
        if not isinstance(store, Mapping):
            continue
        entries = store.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("valid"):
                continue
            n_rows += 1
            key = f"{entry.get('history_id')}|{entry.get('model_seed')}"
            expected = plan.get(key)
            if not isinstance(expected, dict):
                near_misses.append({"key": key, "reason": "no frozen plan group for this key", "observed": str(entry.get("cache_identity"))})
            elif str(entry.get("cache_identity")) != str(expected.get("cache_identity")):
                near_misses.append({"key": key, "reason": "cache identity differs from frozen plan", "observed": str(entry.get("cache_identity")), "expected": str(expected.get("cache_identity"))})
            elif str(entry.get("support_sha256")) != str(expected.get("support_sha256")):
                near_misses.append({"key": key, "reason": "support digest differs from frozen plan", "observed": str(entry.get("support_sha256")), "expected": str(expected.get("support_sha256"))})
    verdict = str(probe.get("verdict", "UNKNOWN"))
    if verdict in ("INVALID", "PARTIAL", "MISSING"):
        overall = "REFUSED"
        detail = f"declared roots are {verdict}; absence is unestablished or refusal applies, never clean absence"
    elif verdict == "NOT-CHECKED/UNKNOWN":
        overall = "NOT-CHECKED"
        detail = "no cache root declared; nothing compared"
    elif near_misses:
        overall = "NEAR-MISS"
        detail = f"{len(near_misses)} manifest rows differ from the frozen plan"
    elif n_rows == 0:
        overall = "EMPTY"
        detail = "declared roots hold no manifest rows"
    else:
        overall = "MATCH"
        detail = "all manifest rows match the frozen plan identities"
    return {"verdict": overall, "detail": detail, "n_rows": n_rows, "near_misses": near_misses, "n_near_misses": len(near_misses)}


PINNED_EXECUTION_SHA256 = "91d4ce264c73f0dada20b4a5f1a7c460e5809bdb1daf67cc75823b9a1d98a3bd"
PINNED_CORPUS_MANIFEST_SHA256 = "9606648db17d64496e7268d81417e25b1806ca0ce2acdd99498a071d7a991ea3"
PINNED_CORPUS_AUDIT_SHA256 = "5fddabc1059412a589e8811d84b67e1c41856f84201f506eb48f51cedfd8eaac"
PINNED_G2_PREFLIGHT_SHA256 = "26f9e45d26af66fa17f37782278699d95ed61e24e674f5f9aed5432883e426df"


def verify_canonical_freeze(freeze: Mapping[str, object]) -> tuple[bool, str]:
    """Verify the execution freeze binds its canonical content (not a stored field).

    Recomputes the canonical digest of the freeze's own ``spec`` block and
    requires it to equal both the freeze's recorded ``protocol_sha256`` and
    the internally pinned execution digest.  A tampered spec, a forged digest
    field, or a substituted freeze file all fail closed.  No caller argument
    can confer authority: the pins live in this module.
    """
    spec = freeze.get("spec")
    recorded = freeze.get("protocol_sha256")
    if not isinstance(spec, dict) or not isinstance(recorded, str):
        return False, "execution freeze is missing its spec/digest block"
    recomputed = sha256_object(spec)
    if recomputed != recorded:
        return False, f"execution freeze digest does not bind its canonical spec: {recomputed} != {recorded}"
    if recorded != PINNED_EXECUTION_SHA256:
        return False, f"execution freeze is not the pinned M8 freeze: {recorded}"
    return True, f"canonical freeze verified: {recorded}"


def verify_pinned_artifact(path: Path, label: str, allowed: object = None) -> tuple[bool, str]:
    """Verify a file's bytes against the internally pinned digest for its role.

    The expected digest is selected by the file's role (manifest/audit/G2
    preflight resolved from its filename), never from a caller argument.  An
    explicit ``allowed`` set constrains which paths may serve the role; any
    other path fails closed even when its bytes match a pinned digest.
    """
    name = path.name
    if name == "corpus-manifest.json":
        expected = PINNED_CORPUS_MANIFEST_SHA256
    elif name == "corpus-audit.json":
        expected = PINNED_CORPUS_AUDIT_SHA256
    elif name == "preflight.json" or label == "G2 preflight":
        expected = PINNED_G2_PREFLIGHT_SHA256
    else:
        return False, f"{label}: no pinned digest for filename {name}"
    if allowed is not None and isinstance(allowed, set):
        if path.resolve() not in {Path(p).resolve() for p in allowed}:
            return False, f"{label}: path {path} is not an authorized source for this role"
    if not path.is_file():
        return False, f"{label}: missing file {path}"
    observed = sha256_file(path)
    if observed != expected:
        return False, f"{label}: digest drift {observed} != pinned {expected}"
    return True, f"{label} verified: {observed}"


def contract_binding_digest(
    census: Mapping[str, object],
    plan: Mapping[str, object],
    custody: Mapping[str, object],
) -> str:
    """Return the canonical binding digest of census + identity plan + custody.

    The digest binds the exact pair identities/eligibility, the per-group
    cache identities/support digests, and the validated custody digests.  The
    adapter recomputes it from its live inputs and requires equality with the
    expected prospective-contract digest passed by the caller; any altered or
    self-consistent other contract fails closed with ``REFUSED-IDENTITY``.
    """
    pairs = census.get("pairs")
    groups = plan.get("groups")
    payload = {
        "pairs": [
            {"comparison_identity": str(p.get("comparison_identity")), "eligibility": str(p.get("eligibility"))}
            for p in pairs  # type: ignore[union-attr]
        ] if isinstance(pairs, list) else [],
        "groups": [
            {"key": str(k), "cache_identity": str(v.get("cache_identity")), "support_sha256": str(v.get("support_sha256"))}
            for k, v in sorted(groups.items())  # type: ignore[union-attr]
        ] if isinstance(groups, dict) else [],
        "custody": {
            "checkpoints": dict(custody.get("checkpoints", {})),  # type: ignore[arg-type]
            "banks": dict(custody.get("banks", {})),  # type: ignore[arg-type]
            "arms": dict(custody.get("arms", {})),  # type: ignore[arg-type]
            "binding_sha256": str(custody.get("binding_sha256")),
            "preprocessing_sha256": str(custody.get("preprocessing_sha256")),
        },
    }
    return sha256_object(payload)

def verify_expected_contract(
    expected_contract: Mapping[str, object],
    expected_contract_sha256: str,
    corpus_manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
    custody: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the actual prospective contract BEFORE any admission or movement.

    Binds the caller-supplied contract payload three ways, all before the
    caller writes, admits, fits or measures anything:

    1. canonical content: ``sha256_object(expected_contract)`` must equal
       ``expected_contract_sha256`` (a changed term with a stale digest fails
       here);
    2. identity binding: the census and identity plan recomputed live from
       the real manifest/audit/custody must reproduce the contract's
       ``binding_digest`` and its frozen ``identity_plan`` groups exactly (a
       self-consistent other contract carrying a fresh digest fails here);
    3. analysis terms: the contract's ``endpoints``/``reference_fitting``/
       ``model_inputs`` sections must equal the module-canonical constructors
       (a contract with substituted endpoint or reference-fit terms fails
       here).

    Returns the live ``(census, plan)`` the caller must use, so execution can
    never drift from the verified contract.  Any failure raises
    ``REFUSED-IDENTITY`` before any fixture is written, admitted, fitted or
    measured.
    """
    if not isinstance(expected_contract, Mapping):
        raise ValueError("REFUSED-IDENTITY: expected prospective contract is not a mapping")
    if str(expected_contract.get("readiness_id")) != READINESS_ID:
        raise ValueError("REFUSED-IDENTITY: expected contract readiness_id does not match this module")
    if str(expected_contract.get("amendment_id")) != AMENDMENT_ID:
        raise ValueError("REFUSED-IDENTITY: expected contract amendment_id does not match this module")
    observed = sha256_object(dict(expected_contract))
    if observed != expected_contract_sha256:
        raise ValueError(
            f"REFUSED-IDENTITY: expected contract digest does not bind its canonical content: {observed} != {expected_contract_sha256}"
        )
    pair_rows = audit_record.get("pair_rows")
    rows = list(pair_rows) if isinstance(pair_rows, list) else []
    if not rows:
        raise ValueError("REFUSED-IDENTITY: audit has no pair rows")
    end_time_of, duration_of, _header_reads = _manifest_file_maps(corpus_manifest)
    census_live = eligibility_census(rows, end_time_of, duration_of)
    plan_live = custody_identity_plan(corpus_manifest, custody)
    binding_live = contract_binding_digest(census_live, plan_live, custody)
    if binding_live != str(expected_contract.get("binding_digest")):
        raise ValueError(
            f"REFUSED-IDENTITY: live census/plan/custody binding {binding_live} != expected contract binding {expected_contract.get('binding_digest')}"
        )
    frozen_executable = expected_contract.get("executable_path")
    frozen_groups: object = None
    if isinstance(frozen_executable, Mapping):
        identity_plan = frozen_executable.get("identity_plan")
        if isinstance(identity_plan, Mapping):
            frozen_groups = identity_plan.get("groups")
    live_groups = plan_live.get("groups")
    if not isinstance(frozen_groups, dict) or not isinstance(live_groups, dict):
        raise ValueError("REFUSED-IDENTITY: expected contract identity-plan groups are missing")
    if dict(frozen_groups) != dict(live_groups):
        raise ValueError("REFUSED-IDENTITY: expected contract identity-plan groups differ from the live frozen plan")
    if expected_contract.get("endpoints") != endpoint_contract():
        raise ValueError("REFUSED-IDENTITY: expected contract endpoints differ from the frozen endpoint contract")
    if expected_contract.get("reference_fitting") != reference_fitting_plan():
        raise ValueError("REFUSED-IDENTITY: expected contract reference-fitting plan differs from the frozen plan")
    if expected_contract.get("model_inputs") != model_input_contract():
        raise ValueError("REFUSED-IDENTITY: expected contract model inputs differ from the frozen contract")
    return census_live, plan_live


def controlled_adapter_probe(
    fixture_root: Path,
    corpus_manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
    custody: Mapping[str, object],
    *,
    seed: int = 2407,
    expected_contract: Mapping[str, object] | None = None,
    expected_contract_sha256: str | None = None,
) -> dict[str, object]:
    """Run the controlled-corpus adapter under the bound prospective contract.

    The caller MUST supply the actual prospective contract payload plus its
    canonical digest (``expected_contract`` + ``expected_contract_sha256``).
    Both are validated by :func:`verify_expected_contract` BEFORE any fixture
    group is written, admitted, fitted or measured: canonical content, live
    census/plan/custody binding, frozen identity-plan groups, and the frozen
    endpoint/reference-fit/model-input terms.  A contract carrying any
    altered term fails closed with ``REFUSED-IDENTITY`` before admission,
    reference fitting or endpoint computation.  No optional post-hoc binding
    comparison exists anymore: the contract gate is pre-execution and
    mandatory.

    Writes one fixture group per new history and representation seed via the
    accepted :func:`execution.write_cache_entry` (provenance
    ``fixture-no-forward``, deterministic synthetic float32 vectors keyed by
    the REAL controlled file IDs, support digests from the REAL per-history
    required file IDs), inspects the fixture with the manifest-only inspector
    (file IDs only, never embedding values), compares inspector-derived
    identities against the contract-bound frozen plan (never the inspected
    entries themselves), admits them through the accepted
    :func:`execution.cache_coverage` gate as an isolated synthetic-fixture
    pathway (the real study admission is separately proven to refuse
    fixture provenance), then computes REAL movement endpoints
    (:func:`endpoints.fit_reference` + :func:`endpoints.pair_endpoint_values`)
    on the REAL first retained transition pair with the SAME admitted vectors.
    A wrong-support-universe call and an unknown file-ID call both fail
    closed.  Everything is labeled ``SYNTHETIC-FIXTURE``; nothing here is
    corpus evidence and no model, checkpoint or real embedding is touched.
    """
    if expected_contract is None or expected_contract_sha256 is None:
        raise ValueError(
            "REFUSED-IDENTITY: controlled adapter requires the actual prospective contract payload plus its canonical digest before admission"
        )
    _contract_census, frozen_plan = verify_expected_contract(
        expected_contract, expected_contract_sha256, corpus_manifest, audit_record, custody
    )
    from types import SimpleNamespace as _NS

    import numpy as _np

    from representation import sprint24_weekly_monthly_endpoints as _endpoints
    from representation import sprint24_weekly_monthly_execution as _admission

    files = corpus_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("REFUSED-IDENTITY: corpus manifest has no file table")
    per_history_ids: dict[str, list[str]] = {}
    end_time_of: dict[str, float] = {}
    duration_of: dict[str, float] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            continue
        history = entry.get("history_id")
        fid = entry.get("file_id")
        if isinstance(history, str) and isinstance(fid, str):
            per_history_ids.setdefault(history, []).append(fid)
            end = entry.get("end_time_rel_s")
            if isinstance(end, (int, float)):
                end_time_of[fid] = float(end)
    manifest_path = corpus_manifest.get("__manifest_path__")
    root = Path(str(manifest_path)).parent if isinstance(manifest_path, str) else None
    if root is not None:
        for fid in list(end_time_of):
            rel = next(
                (str(e.get("relpath")) for e in files if isinstance(e, Mapping) and e.get("file_id") == fid),
                None,
            )
            if rel is not None:
                try:
                    shape = read_npy_shape_header(root / rel)
                    duration_of[fid] = float(int(shape[1]))
                except (OSError, KeyError, ValueError, IndexError):
                    continue
    inputs: dict[str, object] = {
        "checkpoints": dict(custody["checkpoints"]),  # type: ignore[index]
        "banks": dict(custody["banks"]),  # type: ignore[index]
        "arms": dict(custody["arms"]),  # type: ignore[index]
        "binding_sha256": str(custody["binding_sha256"]),
        "preprocessing_sha256": str(custody["preprocessing_sha256"]),
        "device": str(custody["device"]),
        "scale_state_sha256": None,
    }
    pair_rows = audit_record.get("pair_rows")
    rows = list(pair_rows) if isinstance(pair_rows, list) else []
    if not rows:
        raise ValueError("REFUSED-IDENTITY: audit has no pair rows")
    fixture_root.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(seed)
    written: dict[str, dict[str, object]] = {}
    full_ids: dict[str, list[str]] = {}
    fixture_vectors: dict[str, Any] = {}
    for history_id in sorted(per_history_ids):
        required_ids = sorted(set(per_history_ids[history_id]))
        full_ids[history_id] = required_ids
        support_sha256 = sha256_object(required_ids)
        vectors = rng.normal(0.0, 1.0, size=(len(required_ids), 128)).astype(_np.float32)
        fixture_vectors[history_id] = vectors
        for model_seed in FROZEN_ENCODER_SEEDS:
            receipt = _admission.write_cache_entry(
                fixture_root,
                history_id=history_id,
                model_seed=int(model_seed),
                file_ids=required_ids,
                embeddings=vectors,
                inputs=inputs,
                support_sha256=support_sha256,
                provenance=_admission.CACHE_PROVENANCE_FIXTURE,
                capture_commit=None,
            )
            written[f"{history_id}|{model_seed}"] = receipt
    inspected = inspect_cache_manifest_only(fixture_root)
    if int(inspected["invalid_entries"]) != 0:  # type: ignore[arg-type]
        raise ValueError(
            f"REFUSED-IDENTITY: fixture cache failed manifest-only inspection: "
            f"{inspected['invalid_entries']} invalid entries"
        )
    inspected_by_key: dict[str, dict[str, object]] = {}
    for entry in inspected["entries"]:  # type: ignore[union-attr]
        assert isinstance(entry, dict)
        inspected_by_key[f"{entry['history_id']}|{entry['model_seed']}"] = entry
    manifest_entries: list[dict[str, object]] = []
    for key in sorted(written):
        hid, seed_text = key.split("|")
        entry = inspected_by_key.get(key)
        if entry is None:
            raise ValueError(f"REFUSED-IDENTITY: fixture group {key} missing from manifest-only inspection")
        manifest_entries.append(
            {
                "path": str(entry["path"]),
                "history_id": hid,
                "model_seed": int(seed_text),
                "valid": bool(entry["valid"]),
                "file_ids": list(entry["file_ids"]),  # type: ignore[union-attr]
                "cache_identity": str(entry["cache_identity"]),
            }
        )
    probe = probe_declared_cache_roots([fixture_root])
    plan_blocks = {hid: {"required_file_ids": ids} for hid, ids in full_ids.items()}
    frozen_groups = frozen_plan.get("groups")
    assert isinstance(frozen_groups, dict)
    identity_groups: dict[str, dict[str, object]] = {}
    near_misses: list[dict[str, object]] = []
    for key in sorted(written):
        hid, seed_text = key.split("|")
        entry = inspected_by_key[key]
        expected = frozen_groups.get(key)
        assert isinstance(expected, dict)
        if str(entry["cache_identity"]) != str(expected["cache_identity"]) or str(entry["support_sha256"]) != str(expected["support_sha256"]):
            near_misses.append({"key": key, "observed": str(entry["cache_identity"]), "expected": str(expected["cache_identity"])})
            raise ValueError(f"REFUSED-IDENTITY: fixture identity for {key} differs from the independently frozen plan (near-miss, not admitted)")
        identity_groups[key] = {
            "history_id": hid,
            "model_seed": int(seed_text),
            "required_rows": len(full_ids[hid]),
            "support_sha256": str(expected["support_sha256"]),
            "cache_identity": str(expected["cache_identity"]),
        }
    coverage = _admission.cache_coverage(
        {"per_history": plan_blocks}, {"groups": identity_groups}, {"entries": manifest_entries}
    )
    n_invalid = sum(1 for entry in manifest_entries if not entry["valid"])
    plumbing = _admission.admit_execution(
        coverage, {"invalid_entries": n_invalid, "entries": manifest_entries}
    )
    study_admission: dict[str, object] = {"status": "REFUSED", "complete": False, "blockers": ["fixture provenance is not admissible study input"]}
    for entry in manifest_entries:
        sidecar_path = Path(str(entry["path"])).with_suffix(".json")
        provenance = ""
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            provenance = str(payload.get("capture_provenance")) if isinstance(payload, dict) else ""
        except OSError:
            provenance = ""
        if provenance != _admission.CACHE_PROVENANCE_GPU:
            continue
        raise ValueError("REFUSED-IDENTITY: fixture store unexpectedly carries GPU study provenance")
    admission = {"status": str(plumbing["status"]), "complete": bool(plumbing["complete"])}
    fixture_pathway = "ISOLATED-SYNTHETIC-FIXTURE (accepted gate mechanics only; never real study admission)"
    first_history = sorted(full_ids)[0]
    first_seed = int(FROZEN_ENCODER_SEEDS[0])
    admitted_ids = [str(fid) for fid in inspected_by_key[f"{first_history}|{first_seed}"]["file_ids"]]  # type: ignore[index]
    if admitted_ids != list(full_ids[first_history]):
        raise ValueError("REFUSED-IDENTITY: admitted fixture rows differ from the written fixture rows")
    matrix = _np.asarray(fixture_vectors[first_history], dtype=_np.float64)
    matrix_sha256 = hashlib.sha256(matrix.tobytes()).hexdigest()
    group = _endpoints.EmbeddingGroup(
        history_id=first_history,
        model_seed=first_seed,
        file_ids=list(admitted_ids),
        index={fid: pos for pos, fid in enumerate(admitted_ids)},
        matrix=matrix,
    )
    manifest_files = {str(e.get("file_id")): e for e in files if isinstance(e, Mapping)}
    reference_ids = sorted(
        fid
        for fid in admitted_ids
        if str(manifest_files.get(fid, {}).get("arm")) == "REFERENCE"
    )
    fit = _endpoints.fit_reference(group, reference_ids, resamples=50, seed=seed)
    first_pair = next(r for r in rows if str(r.get("history_id")) == first_history and str(r.get("comparison_type")) == "BASELINE-VS-ELEVATED")
    prev_ids = [str(fid) for fid in first_pair["retained_baseline_file_identities"]]  # type: ignore[index]
    cur_ids = [str(fid) for fid in first_pair["retained_contrast_file_identities"]]  # type: ignore[index]

    def _period_of(fids: list[str]) -> object:
        ends = sorted(end_time_of[f] for f in fids)
        starts = _np.array([end_time_of[f] - duration_of.get(f, 400.0) for f in fids])
        return _NS(
            member_starts=starts,
            member_ends=_np.array(ends),
            period=_NS(start=min(ends) - 10.0, end=max(ends)),
        )

    base_identity = str(first_pair.get("comparison_identity"))
    values = _endpoints.pair_endpoint_values(
        previous_ids=prev_ids,
        current_ids=cur_ids,
        previous_period=_period_of(prev_ids),
        current_period=_period_of(cur_ids),
        group=group,
        fit=fit,
        seed=first_seed,
        previous_identity=base_identity + "|BASELINE",
        current_identity=base_identity + "|CONTRAST",
        history_id=first_history,
    )
    prev_matrix = _endpoints.group_rows(group, prev_ids)
    cur_matrix = _endpoints.group_rows(group, cur_ids)
    e1_direct = _endpoints.mmd2_unbiased(prev_matrix, cur_matrix, fit.sigma)
    swapped = _endpoints.pair_endpoint_values(
        previous_ids=prev_ids,
        current_ids=cur_ids,
        previous_period=_period_of(prev_ids),
        current_period=_period_of(cur_ids),
        group=group,
        fit=fit,
        seed=first_seed,
        previous_identity=base_identity + "|CONTRAST",
        current_identity=base_identity + "|BASELINE",
        history_id=first_history,
    )
    swapped_n2 = swapped.get("N2_max")
    e1 = values.get("E1")
    n2 = values.get("N2_max")
    contract_sensitive = (
        isinstance(swapped_n2, float)
        and isinstance(n2, float)
        and swapped_n2 != n2
    )
    second_history = sorted(full_ids)[1]
    group_wrong = _endpoints.EmbeddingGroup(
        history_id=second_history,
        model_seed=first_seed,
        file_ids=list(full_ids[second_history]),
        index={fid: pos for pos, fid in enumerate(full_ids[second_history])},
        matrix=_np.asarray(fixture_vectors[second_history], dtype=_np.float64),
    )
    contract_refused = False
    try:
        _endpoints.group_rows(group_wrong, prev_ids)
    except RuntimeError:
        contract_refused = True
    missing_refused = False
    try:
        _endpoints.group_rows(group, ["__NOT-IN-GROUP__"])
    except RuntimeError:
        missing_refused = True
    n_prev_raw = values.get("n_previous")
    n_cur_raw = values.get("n_current")
    n_prev_count = n_prev_raw if isinstance(n_prev_raw, int) else 0
    n_cur_count = n_cur_raw if isinstance(n_cur_raw, int) else 0
    probe_rows_raw = probe.get("total_manifest_rows")
    probe_rows = probe_rows_raw if isinstance(probe_rows_raw, int) else 0
    admission_status = admission.get("status")
    admission_complete = admission.get("complete")
    binding_live = str(expected_contract.get("binding_digest"))
    binding_ok = True
    binding_verified_pre_execution = True
    numerically_bound = (
        values.get("status") == "COMPUTED"
        and isinstance(e1, float)
        and bool(_np.isfinite(e1))
        and isinstance(e1_direct, float)
        and e1 == e1_direct
    )
    checks = bool(numerically_bound and missing_refused and contract_refused and contract_sensitive and admission_complete and binding_ok)
    return {
        "kind": "SYNTHETIC-FIXTURE",
        "seed": seed,
        "n_histories": len(full_ids),
        "n_groups_written": len(written),
        "manifest_probe_verdict": str(probe["verdict"]),
        "manifest_rows": probe_rows,
        "inspector_invalid_entries": int(inspected["invalid_entries"]),  # type: ignore[arg-type]
        "near_misses": near_misses,
        "n_near_misses": len(near_misses),
        "plumbing_admission_status": str(admission_status),
        "plumbing_admission_complete": bool(admission_complete),
        "study_admission": study_admission,
        "fixture_pathway": fixture_pathway,
        "binding_live": binding_live,
        "binding_ok": bool(binding_ok),
        "binding_verified_pre_execution": bool(binding_verified_pre_execution),
        "same_data_matrix_sha256": matrix_sha256,
        "pair_identity": base_identity,
        "n_previous": int(n_prev_count),
        "n_current": int(n_cur_count),
        "e1": float(e1) if isinstance(e1, float) else None,
        "e1_direct": float(e1_direct) if isinstance(e1_direct, float) else None,
        "e1_match": bool(isinstance(e1, float) and isinstance(e1_direct, float) and e1 == e1_direct),
        "n2": float(n2) if isinstance(n2, float) else None,
        "swapped_n2": float(swapped_n2) if isinstance(swapped_n2, float) else None,
        "contract_sensitive": contract_sensitive,
        "status": str(values.get("status")),
        "missing_refused": missing_refused,
        "contract_refused": contract_refused,
        "checks": bool(checks),
    }


def prospective_contract(
    corpus_manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
    g2_preflight: Mapping[str, object],
    header_reads: int = 0,
) -> dict[str, object]:
    """Freeze the machine-readable prospective contract (metadata only)."""
    manifest_files = corpus_manifest.get("files")
    n_files = len(manifest_files) if isinstance(manifest_files, list) else 0
    pair_rows = audit_record.get("pair_rows")
    rows = list(pair_rows) if isinstance(pair_rows, list) else []
    end_time_of, duration_of, counted = _manifest_file_maps(corpus_manifest)
    custody = validate_custody_inputs(g2_preflight)
    census = eligibility_census(rows, end_time_of, duration_of)
    plan = custody_identity_plan(corpus_manifest, custody)
    plan_groups = plan.get("groups")
    assert isinstance(plan_groups, dict)
    n_identity_groups = plan.get("n_groups")
    assert isinstance(n_identity_groups, int)
    plan_group_ids = sorted(str(key) for key in plan_groups.keys())
    probe = probe_declared_cache_roots([])
    binding_expected = contract_binding_digest(census, plan, custody)
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory(prefix="s24-m10-adapter-") as _tmp:
        contract_shell: dict[str, object] = {
            "readiness_id": READINESS_ID,
            "amendment_id": AMENDMENT_ID,
            "endpoints": endpoint_contract(),
            "reference_fitting": reference_fitting_plan(),
            "model_inputs": model_input_contract(),
            "binding_digest": binding_expected,
            "executable_path": {"identity_plan": dict(plan)},
        }
        contract_shell_sha = sha256_object(dict(contract_shell))
        adapter = controlled_adapter_probe(
            Path(_tmp),
            corpus_manifest,
            audit_record,
            custody,
            expected_contract=dict(contract_shell),
            expected_contract_sha256=contract_shell_sha,
        )
        adapter_record = dict(adapter)
    contract: dict[str, object] = {
        "readiness_id": READINESS_ID,
        "amendment_id": AMENDMENT_ID,
        "base_execution_id": BASE_EXECUTION_ID,
        "base_design_id": BASE_DESIGN_ID,
        "frozen_before_outcome": True,
        "corpus_identity": {
            "namespace": "S24C-ITER-0001",
            "execution_sha256": str(audit_record.get("execution_sha256")),
            "corpus_manifest_sha256": "9606648db17d64496e7268d81417e25b1806ca0ce2acdd99498a071d7a991ea3",
            "corpus_audit_sha256": "5fddabc1059412a589e8811d84b67e1c41856f84201f506eb48f51cedfd8eaac",
            "n_files": n_files,
            "n_histories": 6,
        },
        "encoder_custody": {
            "representation_seeds": list(FROZEN_ENCODER_SEEDS),
            "checkpoints": dict(custody["checkpoints"]),  # type: ignore[index]
            "banks": dict(custody["banks"]),  # type: ignore[index]
            "arms": dict(custody["arms"]),  # type: ignore[index]
            "binding_sha256": str(custody["binding_sha256"]),
            "preprocessing_sha256": str(custody["preprocessing_sha256"]),
            "device": str(custody["device"]),
            "validated_against": "execution.ACCEPTED_G2_PROTOCOL_SHA256 / ACCEPTED_BINDING_SHA256 / ACCEPTED_G2_SOURCE_COMMIT / REQUIRED_DEVICE_IDENTITY / ACCEPTED_PREPROCESSING_SHA256",
            "note": "validated, never loaded; no checkpoint is loaded and no forward executes in readiness",
        },
        "model_inputs": model_input_contract(),
        "reference_fitting": reference_fitting_plan(),
        "endpoints": endpoint_contract(),
        "census": census,
        "expected_census": expected_retained_census(),
        "binding_digest": binding_expected,
        "executable_path": {
            "identity_plan": dict(plan),
            "identity_plan_sha256": sha256_object(dict(plan)),
            "n_identity_groups": n_identity_groups,
            "group_ids": sorted(plan_group_ids),
            "declared_cache_probe": probe,
            "controlled_adapter": adapter_record,
            "reference_fit_invocation": {
                "function": "endpoints.fit_reference(group, reference_ids)",
                "grain": "per new controlled history and representation seed on its own causal reference pool (220 files); exercised in the synthetic-fixture adapter only, never on real embeddings",
                "endpoint_invocation": "endpoints.pair_endpoint_values(previous_ids, current_ids, ..., group, fit, seed, ...) per retained-eligible pair and seed on the REAL retained pair structure; exercised in the synthetic-fixture adapter only",
            },
        },
        "cost_custody": cost_custody_plan(n_files=n_files if n_files else 8280),
        "abstentions": [
            "undefined quantities abstain, never silently become zero",
            "C2 abstains corpus-wide (degenerate regressor); never duplicated as C1, never independent confirmation",
            "descriptive C1 association set is all 5 retained pairs per history/horizon (4 transitions at/above the retained floor plus the retained control as the low-burden pole), per the unchanged M8 C1 support rule",
            "support shortfalls refuse with closed reasons (REFUSED-SHORTFALL and sibling reasons)",
        ],
        "missing_prerequisites": missing_prerequisites(),
        "reads": {
            "waveform_signal_value_reads": 0,
            "npz_npy_header_reads": counted,
            "model_loads": 0,
            "embedding_reads": 0,
            "embedding_values_opened": 0,
            "forwards": 0,
        },
        "frozen_policy": "no alternative is permitted after any outcome is read; this contract's canonical digest is bound in the first actual movement run manifest",
    }
    return {"contract_sha256": sha256_object(contract), "contract": contract}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_missing_corpus(corpus: Path) -> dict[str, str]:
    """Fail closed when the official corpus or its frozen artifacts are absent."""
    manifest = corpus / "corpus-manifest.json"
    freeze = corpus / "execution-freeze.json"
    realized = corpus / "realized-slots.json"
    missing = [str(p) for p in (manifest, freeze, realized) if not p.is_file()]
    if missing:
        raise ValueError(f"REFUSED-MISSING-CORPUS: official corpus artifacts absent: {missing}")
    waves = corpus / "waves"
    if not waves.is_dir():
        raise ValueError("REFUSED-MISSING-CORPUS: official corpus waves/ directory absent")
    return {"manifest": str(manifest), "freeze": str(freeze), "realized": str(realized), "waves": str(waves)}


def load_inputs(corpus: Path, g2_preflight_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load corpus manifest, realized slots and the accepted G2 preflight (metadata only)."""
    paths = refuse_missing_corpus(corpus)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("REFUSED-IDENTITY: corpus manifest is not a mapping")
    manifest["__manifest_path__"] = paths["manifest"]
    preflight_default = Path(__file__).resolve().parents[2] / "artifacts" / "sprint-24" / "evidence" / "s24-7-m4-a01" / "g2-accepted" / "preflight.json"
    preflight_path = g2_preflight_path or preflight_default
    if not preflight_path.is_file():
        raise ValueError(f"REFUSED-MISSING-CORPUS: accepted G2 preflight absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise ValueError("REFUSED-IDENTITY: accepted G2 preflight is not a mapping")
    audit_path = corpus / "execution-freeze.json"
    freeze = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise ValueError("REFUSED-IDENTITY: execution freeze is not a mapping")
    return manifest, freeze, preflight

__all__ = [
    "AMENDMENT_ID",
    "BASE_DESIGN_ID",
    "BASE_EXECUTION_ID",
    "ELIGIBILITY_STATES",
    "EXPOSURE_RATIO_FLOOR",
    "FIVE_PAIR_MIN_TWO_SIDED_TAIL",
    "FROZEN_ENCODER_SEEDS",
    "NOT_EXECUTED",
    "NULL_SEED",
    "PINNED_CORPUS_AUDIT_SHA256",
    "PINNED_CORPUS_MANIFEST_SHA256",
    "PINNED_EXECUTION_SHA256",
    "PINNED_G2_PREFLIGHT_SHA256",
    "PRIMARY_TRIPLE",
    "READINESS_ID",
    "REFUSAL_REASONS",
    "REPLICATION_TRIPLE",
    "controlled_adapter_probe",
    "cost_custody_plan",
    "custody_identity_plan",
    "eligibility_census",
    "endpoint_contract",
    "expected_retained_census",
    "inspect_cache_manifest_only",
    "load_inputs",
    "missing_prerequisites",
    "model_input_contract",
    "compare_probe_to_plan",
    "contract_binding_digest",
    "probe_declared_cache_roots",
    "prospective_contract",
    "read_cache_file_ids_only",
    "read_cache_member_facts",
    "read_npy_shape_header",
    "reference_fitting_plan",
    "refuse_missing_corpus",
    "retained_exposure",
    "sha256_file",
    "validate_custody_inputs",
    "verify_canonical_freeze",
    "verify_expected_contract",
    "verify_pinned_artifact",
]
