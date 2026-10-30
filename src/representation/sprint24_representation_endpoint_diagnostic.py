"""Sprint 24 Task 7-M14 representation-signal vs population-endpoint diagnostic.

Cache-only, read-only exploratory audit of the already admitted S24-7-M11
embeddings (24,840 vectors, 18 ``(history, seed)`` groups). The module never
loads a model, never runs a forward pass, never refits an encoder or an
anomaly bank, never regenerates the corpus and never rewrites an accepted
M11/M12/M13 product.

The frozen design is recorded in
``docs/sprint24-representation-endpoint-diagnostic-specification-a01.json``
(artifact ``S24-REPRESENTATION-ENDPOINT-DIAGNOSTIC-SPEC-A01``), written after a
metadata-only support census and before any M14 outcome.

Three branches, one integrated audit:

- **A - representation signal.** Within one ``history x horizon x
  robot|program`` stratum, compare a disjoint ``s``-file abnormal arm with an
  equal-size normal arm drawn from the same stratum, plus a second disjoint
  equal-size normal arm that forms the matched normal-normal variability band
  (the null). Labels construct and diagnose the arms only; they never enter a
  reference fit, a prototype, a bandwidth or a detector. A period-controlled
  variant matches the arm period composition exactly and is reported only
  where the metadata census supports it.
- **B - occupancy versus within-prototype movement.** The same valid arms and
  the same causal-reference prototypes/bandwidth are scored with the accepted
  unbiased E1, the accepted occupancy E2 for ``K in {4, 8, 16}`` and a
  predeclared bounded residual diagnostic ``D`` (location, spread, radial and
  tangential components, coverage). ``E2`` is occupancy only; ``D`` can move
  while occupancy is stable. Radial distance alone is never read as all
  directional movement: the tangential component is published beside it.
- **C - positive-control sensitivity.** A reference-derived global embedding
  perturbation (first principal direction of the causal reference pool, per
  ``(history, seed)``) of known magnitude ``delta * sigma_u`` is applied to a
  normal arm disjoint from the unchanged normal arm. The grid includes a
  zero-shift record, an unchanged-versus-unchanged null band and a
  within-prototype ("boundary-safe") case that perturbs only files whose
  prototype margin exceeds the shift, so occupancy must not move. Artificial
  shifts verify measurement responsiveness only; they say nothing about
  learned anomaly semantics, significance, power or business accuracy.

Equal-size occupancy quantities are computed in EXACT OCCUPANCY QUANTA: with
``s`` rows per arm every occupancy entry is a count divided by ``s``, so
``E2 = m / s`` with integer ``m``. No p-value, success gate or causal claim is
produced. Every planned unit either resolves or is recorded with a closed
abstention reason and stays in the denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from representation import sprint24_controlled_dose_response as m13
from representation import sprint24_controlled_inference_execution as m11
from representation import sprint24_controlled_inference_readiness as readiness
from representation import sprint24_e2_cache_diagnostic as m12
from representation import sprint24_weekly_monthly_endpoints as endpoints
from representation.sprint24_context_audit import canonical_json, sha256_object

#: Task identifier for every M14 record.
TASK_ID = "S24-7-M14"

#: Frozen pre-outcome analysis specification.
SPEC_ID = "S24-REPRESENTATION-ENDPOINT-DIAGNOSTIC-SPEC-A01"
SPEC_PATH = "docs/sprint24-representation-endpoint-diagnostic-specification-a01.json"

#: Byte digest of the frozen specification (never overwritten; amendments are separate).
SPEC_SHA256 = "35b6ef7dd9364074360bd3c497ab27ec12df4d061f29d30dfc79a077e2d6084c"

#: Repository root, used to resolve the frozen specification and prior products.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Monte Carlo repeats per branch A/B comparison (resampling variability, never evidence).
REPEATS_AB = 200

#: Monte Carlo repeats per branch C comparison.
REPEATS_C = 50

#: Supported equal per-arm sizes per horizon, frozen from the metadata census.
SIZES_A: dict[str, tuple[int, ...]] = {"week": (2, 3), "month": (3, 4)}

#: Supported per-arm normal-source sizes for the branch C controls.
SIZES_C: dict[str, tuple[int, ...]] = {"week": (40, 60), "month": (160, 250)}

#: Inherited cluster counts for the occupancy endpoint.
CLUSTER_COUNTS: tuple[int, ...] = m12.CLUSTER_COUNTS

#: Inherited primary prototype level used for the perturbation boundary rule.
PRIMARY_K = m12.INHERITED_PRIMARY_K

#: Positive-control magnitude grid in reference directional standard deviations.
DELTA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

#: Positive-control perturbation modes (zero shift is a single shared record).
MODE_GLOBAL = "GLOBAL-SHIFT"
MODE_SAFE = "WITHIN-PROTOTYPE-SAFE"
PERTURB_MODES: tuple[str, ...] = (MODE_GLOBAL, MODE_SAFE)

#: Branch A arm designs.
VARIANT_POOLED = "POOLED-PERIOD"
VARIANT_MATCHED = "PERIOD-MATCHED"

#: Accounting estimator identifiers.
ESTIMATORS: tuple[str, ...] = (
    "E1",
    "E2_K4",
    "E2_K8",
    "E2_K16",
    "D_LOC",
    "D_SPREAD",
)
CONTROL_ESTIMATORS: tuple[str, ...] = (
    "E1",
    "E2_K4",
    "E2_K8",
    "E2_K16",
    "D_LOC",
    "D_SPREAD",
    "D_RADIAL",
    "D_TANGENTIAL",
)

#: Record statuses and closed abstention reasons.
COMPUTED = "COMPUTED"
ABSTAIN_NO_ABNORMAL = "ABSTAIN-NO-ABNORMAL"
ABSTAIN_INSUFFICIENT_ABNORMAL = "ABSTAIN-INSUFFICIENT-ABNORMAL"
ABSTAIN_INSUFFICIENT_NORMAL = "ABSTAIN-INSUFFICIENT-NORMAL"
ABSTAIN_PERIOD_MATCH = "ABSTAIN-PERIOD-MATCH-INFEASIBLE"
ABSTAIN_NORMAL_QUOTA = "ABSTAIN-NORMAL-QUOTA-INFEASIBLE"
ABSTAIN_BOUNDARY_MASS = "ABSTAIN-BOUNDARY-SAFE-MASS"
ABSTAIN_COVERAGE = "ABSTAIN-INSUFFICIENT-COVERAGE"

#: The diagnostic's exact declarations.
D_MIN_COVERAGE = 0.5
BOUNDARY_SAFE_MIN_FRACTION = 0.5
BOUNDARY_PROXIMITY_FACTOR = 0.25

#: Accepted prior-product digests that this audit must leave byte-identical.
ACCEPTED_PRODUCT_DIGESTS: dict[str, str] = {
    "docs/sprint24-e2-cache-diagnostic-audit.md": (
        "423932b36052760b96000fc19d883bca9b81b3676f32909cbf40704e0c7bcad9"
    ),
    "docs/sprint24-e2-cache-diagnostic-specification-a01.json": (
        "cde6021ff086398cf690d76a7eb78d656274721dc55621c8e8012fae5c2d47fc"
    ),
    "docs/sprint24-controlled-dose-response-audit.md": (
        "a070df07360f256029fa129abdd093c566a6216d67a3bc24386e92b8d8f784ae"
    ),
    "docs/sprint24-controlled-dose-response-specification-a01.json": (
        "8a3681fc84e6187c5979f95172ab5e87773251efcca4f650c9cf7fbe2338ebe4"
    ),
    "docs/sprint24-controlled-inference-execution.md": (
        "53795c0634019d4c750ce0426bdab73fb3739ee0691497899a23161774d960c2"
    ),
    "src/representation/sprint24_e2_cache_diagnostic.py": (
        "ae7706f0eb0369d920c5d5adfdd144f2938007df0dfb9a9043e7dd12d906a128"
    ),
    "src/representation/sprint24_controlled_dose_response.py": (
        "b5d9b85ec1fec6162053ff40d98d4ba6fb5ea98e6d5e892df61ca2ec1a0e1e53"
    ),
    "experiments/sprint24_task7m12_e2_cache_diagnostic.py": (
        "9655727019f63ac702feb5fbc9ee4bf70039139fada8eb79f29c115f8357b01b"
    ),
    "experiments/sprint24_task7m13_controlled_dose_response.py": (
        "e1e8161e3bca22b587cc78e5cb635d722d98bfbdb843180205b97f93c7f92756"
    ),
    "tests/representation/test_sprint24_e2_cache_diagnostic.py": (
        "12b06aec5f8ad68d2d5f1ca9250ef5729c32d942be68b14fab9bd18947eed56f"
    ),
    "tests/representation/test_sprint24_controlled_dose_response.py": (
        "ad48d85e08ac721d421a7ecfd6ee3e06cb6cce6bca8ee5f0d32f5bb74bf02a98"
    ),
    "src/representation/sprint24_controlled_inference_execution.py": (
        "47d845c86bbb4c250cba6a4a885565283083a2c53b4077a38e5f3da82d159f7b"
    ),
    "experiments/sprint24_task7m11_controlled_inference_execution.py": (
        "e67c17af74d55c2a4f2034061fce96f81cc603be4d1ea2ec513bfc29806c27f1"
    ),
    "artifacts/sprint-24/evidence/s24-7-m11-a04/movement.json": (
        "d84747449f809eccd498806493ec570608f1b3689931395468b486e1e29515e3"
    ),
    "artifacts/sprint-24/evidence/s24-7-m11-a04/run.json": (
        "87a71f8cff50daedded969d2f50930a646efdda95953f8750baafe8010636f84"
    ),
    "artifacts/sprint-24/task-7-m13-a03.md": (
        "476a3dc5bf88370fcc8caaa4d8a5a71124fd9a3ec31583c0bafe235cf9bec7ef"
    ),
    "artifacts/sprint-24/evidence/s24-7-m13-a01/audit/run.json": (
        "2179a06c50600bb4869b2282f47a20fe047532e57590be40f4ae500f02ee2bbe"
    ),
    "artifacts/sprint-24/evidence/s24-7-m13-a01/audit/aggregates.json": (
        "5b649eb70d7261e361f38db831cd13c547571de450fd74f109a479c1491f3a78"
    ),
}

#: Admitted-cache digest-of-digests recorded by the accepted M11/M12/M13 lineage.
CACHE_DIGEST_DIGEST = "ae7aa42d7661d95d9d292400f53f4ec76128fd9618acc75cc6c6fcf581aa02b3"

#: Cumulative observed forward passes before this audit (M11 accepted accounting).
CUMULATIVE_FORWARDS = 77653

#: Default amendment directory (separate from the frozen specification bytes).
AMENDMENT_DIR = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m14-a01" / "amendments"
)

#: Diagnostic bin edges inherited unchanged from the accepted M12 audit.
SEVERITY_BINS = m12.SEVERITY_BINS
FRACTION_BINS = m12.FRACTION_BINS
ORDINARY = m12.ORDINARY

LIMITATIONS: tuple[str, ...] = (
    "Exploratory, cache-only and post-exposure: the corpus and embeddings were "
    "available before this design was frozen, so no fresh confirmation, "
    "pre-registration or causal claim is made.",
    "Branch A compares tiny equal-size arms inside one robot|program stratum; "
    "the realised support decides the sizes and no stratum below its size floor "
    "is reported as evidence. A weak or absent contrast is not proof that the "
    "representation carries no abnormality signal.",
    "The pooled-period variant mixes source periods inside one stratum; period "
    "composition is reported and only the exact period-matched variant controls "
    "it, and only where the census supports that variant.",
    "E1 is the accepted unbiased squared MMD with the accepted causal-reference "
    "bandwidth. Repeats describe resampling variability of one finite support, "
    "never confidence limits or significance.",
    "E2 is occupancy total variation against fixed causal-reference prototypes; "
    "it cannot see movement that stays inside one prototype. D is the "
    "predeclared bounded residual diagnostic that can; radial distance alone is "
    "never treated as all directional movement.",
    "Branch C perturbs embeddings along a reference-derived direction. It "
    "demonstrates measurement responsiveness only: not learned anomaly "
    "semantics, not generalisation, not power guarantees, not business accuracy.",
    "Weekly and monthly cells, the six histories and the three encoder seeds are "
    "reported separately and never pooled into one representation claim.",
    "No duplicate file identity is used to inflate any arm; every draw is "
    "without replacement and every arm pair is disjoint.",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_seed(parts: Sequence[object]) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


# --- frozen specification, amendments and prior products ------------------------


def load_specification(path: Path | None = None) -> dict[str, Any]:
    """Load and verify the frozen pre-outcome specification bytes.

    Any byte drift, identity drift or substituted rule set raises
    ``REFUSED-SPEC-DRIFT`` before any statistic. The original bytes are never
    rewritten; later deviations belong to separate amendment records.
    """
    resolved = path if path is not None else REPO_ROOT / SPEC_PATH
    _require(resolved.is_file(), f"REFUSED-SPEC-MISSING: {resolved}")
    observed = sha256_file(resolved)
    _require(
        observed == SPEC_SHA256,
        f"REFUSED-SPEC-DRIFT: {resolved} digest {observed} != frozen {SPEC_SHA256}",
    )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict) and payload.get("artifact_id") == SPEC_ID,
        "REFUSED-SPEC-DRIFT: specification artifact id mismatch",
    )
    return {
        "artifact_id": SPEC_ID,
        "path": SPEC_PATH,
        "sha256": observed,
        "state": payload.get("state"),
        "created_utc": payload.get("created_utc"),
        "payload": payload,
    }


def load_amendments(
    directory: Path | None = None, *, spec_sha256: str | None = None
) -> list[dict[str, Any]]:
    """Load separately preserved amendment records bound to the frozen spec."""
    resolved = directory if directory is not None else AMENDMENT_DIR
    if not resolved.is_dir():
        return []
    expected = spec_sha256 if spec_sha256 is not None else SPEC_SHA256
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(
            isinstance(payload, dict) and payload.get("amends_spec_sha256") == expected,
            f"REFUSED-AMENDMENT: {path.name} does not amend the frozen specification",
        )
        _require(
            isinstance(payload.get("created_utc"), str) and bool(payload["created_utc"]),
            f"REFUSED-AMENDMENT: {path.name} has no creation time",
        )
        records.append(
            {
                "path": (
                    str(path.relative_to(REPO_ROOT))
                    if path.is_relative_to(REPO_ROOT)
                    else str(path)
                ),
                "sha256": sha256_file(path),
                "payload": payload,
            }
        )
    return records


def verify_accepted_products() -> dict[str, dict[str, Any]]:
    """Verify that every accepted M11/M12/M13 product still matches its digest."""
    checks: dict[str, dict[str, Any]] = {}
    for relative, expected in ACCEPTED_PRODUCT_DIGESTS.items():
        path = REPO_ROOT / relative
        _require(path.is_file(), f"REFUSED-MISSING-PRODUCT: {relative}")
        observed = sha256_file(path)
        checks[relative] = {
            "expected": expected,
            "observed": observed,
            "match": observed == expected,
        }
        _require(
            observed == expected,
            f"REFUSED-DIGEST-DRIFT: accepted product {relative} changed: {observed}",
        )
    return checks


def verify_lineage(
    corpus: Path,
    audit_path: Path,
    contract_path: Path,
    g2_preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Verify lineage, the frozen specification and every accepted prior product."""
    spec = load_specification()
    amendments = load_amendments(spec_sha256=str(spec["sha256"]))
    lineage = m12.verify_lineage(corpus, audit_path, contract_path, g2_preflight_path)
    products = verify_accepted_products()
    return {
        "spec": {key: value for key, value in spec.items() if key != "payload"},
        "amendments": amendments,
        "prior_products": products,
        "lineage": {
            key: value
            for key, value in lineage.items()
            if key not in {"manifest", "audit_record", "contract", "custody", "identity_plan"}
        },
        "manifest": lineage["manifest"],
        "audit_record": lineage["audit_record"],
        "identity_plan": lineage["identity_plan"],
        "custody": lineage["custody"],
        "contract": lineage["contract"],
    }


# --- cells, strata and metadata support ----------------------------------------


def horizon_period_of(entry: Mapping[str, object], horizon: str) -> int | None:
    """Return the week offset or month index of one manifest file."""
    value = entry.get("week_offset") if horizon == "week" else entry.get("month_index")
    return int(value) if isinstance(value, (int, float)) else None  # type: ignore[arg-type]


def _family_of(entry: Mapping[str, Any]) -> str:
    annotation = entry.get("annotation")
    family = annotation.get("family") if isinstance(annotation, Mapping) else None
    return str(family) if isinstance(family, str) else ORDINARY


@dataclass(slots=True)
class CellTable:
    """Deterministic manifest view of one accepted ``history x horizon`` cell."""

    key: str
    history_id: str
    horizon: str
    periods: tuple[int, ...]
    entries: list[dict[str, Any]]
    ids: list[str]
    strata: np.ndarray
    stratum_names: list[str]
    abnormal: np.ndarray
    period_codes: np.ndarray
    period_numbers: list[int]

    def positions(self, stratum: str | None = None) -> np.ndarray:
        if stratum is None:
            return np.arange(len(self.ids), dtype=np.int64)
        index = self.stratum_names.index(stratum)
        return np.flatnonzero(self.strata == index).astype(np.int64)


def build_cell_table(manifest: Mapping[str, object], cell: Mapping[str, Any]) -> CellTable:
    """Build the deterministic file table of one cell from the corpus manifest."""
    files = manifest.get("files")
    _require(isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table")
    horizon = str(cell["horizon"])
    history_id = str(cell["history_id"])
    allowed = set(int(value) for value in cell["periods"])
    selected: list[Mapping[str, Any]] = []
    for entry in files:  # type: ignore[union-attr]
        if not isinstance(entry, Mapping) or entry.get("history_id") != history_id:
            continue
        if str(entry.get("arm")) == "REFERENCE":
            continue
        period = horizon_period_of(entry, horizon)
        if period is None or period not in allowed:
            continue
        selected.append(entry)
    selected.sort(key=lambda entry: str(entry["file_id"]))
    _require(bool(selected), f"REFUSED-EMPTY-POOL: {cell['cell']}")
    ids = [str(entry["file_id"]) for entry in selected]
    _require(len(set(ids)) == len(ids), f"REFUSED-IDENTITY: duplicate file id in {cell['cell']}")
    stratum_lookup: dict[str, int] = {}
    stratum_names: list[str] = []
    strata: list[int] = []
    abnormal: list[bool] = []
    period_numbers = sorted({int(horizon_period_of(entry, horizon) or 0) for entry in selected})
    period_lookup = {number: index for index, number in enumerate(period_numbers)}
    for entry in selected:
        name = m12.stratum_id(entry)
        if name not in stratum_lookup:
            stratum_lookup[name] = len(stratum_names)
            stratum_names.append(name)
        strata.append(stratum_lookup[name])
        abnormal.append(str(entry.get("file_label")) == "abnormal")
    return CellTable(
        key=str(cell["cell"]),
        history_id=history_id,
        horizon=horizon,
        periods=tuple(sorted(allowed)),
        entries=[dict(entry) for entry in selected],
        ids=ids,
        strata=np.asarray(strata, dtype=np.int64),
        stratum_names=stratum_names,
        abnormal=np.asarray(abnormal, dtype=bool),
        period_codes=np.asarray(
            [period_lookup[int(horizon_period_of(entry, horizon) or 0)] for entry in selected],
            dtype=np.int64,
        ),
        period_numbers=period_numbers,
    )


def apportion(
    weights: Mapping[str, int], caps: Mapping[str, int], total: int
) -> dict[str, int]:
    """Capped largest-remainder apportionment of ``total`` over ``weights``.

    Returns ``{}`` when the caps cannot supply ``total``. Deterministic:
    keys are processed in sorted order and the residual is granted by
    descending fractional remainder. Support-only, never outcome-dependent.
    """
    keys = sorted(weights)
    if not keys or total <= 0:
        return {}
    if sum(min(int(caps[key]), int(weights[key])) for key in keys) < total:
        return {}
    weight = float(sum(int(weights[key]) for key in keys))
    allocation = {
        key: min(int(math.floor(total * int(weights[key]) / weight)), int(caps[key]))
        for key in keys
    }
    remaining = total - sum(allocation.values())
    _require(remaining >= 0, "REFUSED-INTERNAL: apportionment overshot")
    while remaining > 0:
        order = sorted(
            keys,
            key=lambda key: (
                -(total * int(weights[key]) / weight - allocation[key]),
                str(key),
            ),
        )
        granted = False
        for key in order:
            if remaining <= 0:
                break
            if allocation[key] < int(caps[key]):
                allocation[key] += 1
                remaining -= 1
                granted = True
        if not granted:
            return {}
    return allocation


def period_quota(
    table: CellTable, stratum: str, size: int
) -> tuple[dict[int, int], str]:
    """Exact period-matched quota of one stratum and size.

    Allocation weights are the abnormal availability per period; the cap of
    each period is ``min(n_abnormal, n_normal // 3)`` because the comparison
    needs one abnormal arm and three disjoint normal arms of the same
    per-period count. Returns ``({}, reason)`` when infeasible.
    """
    positions = table.positions(stratum)
    abnormal = positions[table.abnormal[positions]]
    normal = positions[~table.abnormal[positions]]
    weights: dict[str, int] = {}
    caps: dict[str, int] = {}
    for code, number in enumerate(table.period_numbers):
        a = int(np.sum(table.period_codes[abnormal] == code))
        n = int(np.sum(table.period_codes[normal] == code))
        if a <= 0:
            continue
        weights[str(number)] = a
        caps[str(number)] = min(a, n // 3)
    allocation = apportion(weights, caps, size)
    if not allocation:
        return {}, ABSTAIN_PERIOD_MATCH
    return {int(key): int(value) for key, value in allocation.items() if value > 0}, COMPUTED


def normal_quota(
    table: CellTable, size: int, *, arms: int = 2
) -> tuple[dict[str, int], str]:
    """Robot|program quota of the branch C normal-source arms.

    Weights are the normal availability per stratum and the cap of each
    stratum is ``n_normal // arms`` so that ``arms`` disjoint equal-quota
    arms fit. Returns ``({}, reason)`` when infeasible.
    """
    positions = table.positions()
    normal = positions[~table.abnormal[positions]]
    weights: dict[str, int] = {}
    caps: dict[str, int] = {}
    for index, name in enumerate(table.stratum_names):
        count = int(np.sum(table.strata[normal] == index))
        if count <= 0:
            continue
        weights[name] = count
        caps[name] = count // arms
    allocation = apportion(weights, caps, size)
    if not allocation:
        return {}, ABSTAIN_NORMAL_QUOTA
    return allocation, COMPUTED


# --- census (metadata only) -----------------------------------------------------


def _period_counts(table: CellTable, positions: np.ndarray) -> dict[str, int]:
    return {
        str(table.period_numbers[code]): int(np.sum(table.period_codes[positions] == code))
        for code in range(len(table.period_numbers))
    }


def _bin_of(value: float, edges: Sequence[tuple[float, float]]) -> str:
    for low, high in edges:
        if low <= value < high:
            return f"[{low},{high})"
    last = edges[-1]
    return f"[{last[0]},{last[1]})"


def _quantile_edges(values: Sequence[float]) -> list[float]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return [0.0, 1.0, 2.0, 3.0, 4.0]
    return [float(np.quantile(finite, q)) for q in (0.25, 0.5, 0.75)]


def _duration_bin(value: float, edges: Sequence[float]) -> str:
    if value < edges[0]:
        return "Q1"
    if value < edges[1]:
        return "Q2"
    if value < edges[2]:
        return "Q3"
    return "Q4"


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def abnormal_composition(table: CellTable, positions: np.ndarray) -> dict[str, Any]:
    """Family/severity/fraction/duration description of an abnormal set."""
    selected = [table.entries[int(index)] for index in positions]
    family: dict[str, int] = {}
    severity: dict[str, int] = {}
    fraction: dict[str, int] = {}
    duration_values = [
        float(entry.get("annotation", {}).get("duration_steps", 0) or 0)
        for entry in selected
    ]
    edges = _quantile_edges(duration_values)
    duration: dict[str, int] = {}
    for entry, value in zip(selected, duration_values):
        name = _family_of(entry)
        family[name] = family.get(name, 0) + 1
        annotation = entry.get("annotation")
        severity_value = annotation.get("severity") if isinstance(annotation, Mapping) else None
        fraction_value = (
            annotation.get("abnormal_step_fraction")
            if isinstance(annotation, Mapping)
            else None
        )
        if isinstance(severity_value, (int, float)):
            label = _bin_of(float(severity_value), SEVERITY_BINS)
            severity[label] = severity.get(label, 0) + 1
        if isinstance(fraction_value, (int, float)):
            label = _bin_of(float(fraction_value), FRACTION_BINS)
            fraction[label] = fraction.get(label, 0) + 1
        label = _duration_bin(value, edges)
        duration[label] = duration.get(label, 0) + 1
    return {
        "family_counts": dict(sorted(family.items())),
        "severity_bin_counts": dict(sorted(severity.items())),
        "fraction_bin_counts": dict(sorted(fraction.items())),
        "duration_bin_counts": dict(sorted(duration.items())),
        "duration_edges": [float(edge) for edge in edges],
        "median_severity": _median_or_none(
            [
                float(entry["annotation"]["severity"])
                for entry in selected
                if isinstance(entry.get("annotation"), Mapping)
                and isinstance(entry["annotation"].get("severity"), (int, float))
            ]
        ),
        "median_fraction": _median_or_none(
            [
                float(entry["annotation"]["abnormal_step_fraction"])
                for entry in selected
                if isinstance(entry.get("annotation"), Mapping)
                and isinstance(entry["annotation"].get("abnormal_step_fraction"), (int, float))
            ]
        ),
        "median_duration": _median_or_none(duration_values),
    }


def build_census(
    manifest: Mapping[str, object], audit_record: Mapping[str, object]
) -> dict[str, Any]:
    """Return the metadata-only support census that the specification freezes."""
    cells = m13.build_cells(audit_record)
    units_a: list[dict[str, Any]] = []
    units_c: list[dict[str, Any]] = []
    cell_records: dict[str, dict[str, Any]] = {}
    for key, cell in sorted(cells.items()):
        table = build_cell_table(manifest, cell)
        horizon = table.horizon
        positions = table.positions()
        abnormal_positions = positions[table.abnormal[positions]]
        normal_positions = positions[~table.abnormal[positions]]
        cell_records[key] = {
            "cell": key,
            "history_id": table.history_id,
            "horizon": horizon,
            "periods": [int(value) for value in table.periods],
            "n_files": len(table.ids),
            "n_abnormal": int(abnormal_positions.size),
            "n_normal": int(normal_positions.size),
            "n_strata": len(table.stratum_names),
            "branch_a_sizes": [int(value) for value in SIZES_A[horizon]],
            "branch_c_sizes": [int(value) for value in SIZES_C[horizon]],
            "reference_files": 220,
        }
        for stratum in table.stratum_names:
            stratum_positions = table.positions(stratum)
            abnormal = stratum_positions[table.abnormal[stratum_positions]]
            normal = stratum_positions[~table.abnormal[stratum_positions]]
            n_abnormal = int(abnormal.size)
            n_normal = int(normal.size)
            for size in SIZES_A[horizon]:
                status = COMPUTED
                reason = ""
                if n_abnormal == 0:
                    status = ABSTAIN_NO_ABNORMAL
                    reason = "the stratum carries no abnormal file in the cell periods"
                elif n_abnormal < size:
                    status = ABSTAIN_INSUFFICIENT_ABNORMAL
                    reason = f"{n_abnormal} abnormal files < {size} required"
                elif n_normal < 3 * size:
                    status = ABSTAIN_INSUFFICIENT_NORMAL
                    reason = (
                        f"{n_normal} normal files < {3 * size} required for three "
                        "disjoint arms"
                    )
                quota, quota_status = period_quota(table, stratum, size)
                max_matched = sum(
                    min(
                        int(np.sum(table.period_codes[abnormal] == code)),
                        int(np.sum(table.period_codes[normal] == code)) // 3,
                    )
                    for code in range(len(table.period_numbers))
                )
                units_a.append(
                    {
                        "unit": f"{key}|{stratum}|{size}",
                        "cell": key,
                        "history_id": table.history_id,
                        "horizon": horizon,
                        "stratum": stratum,
                        "size": int(size),
                        "n_abnormal": n_abnormal,
                        "n_normal": n_normal,
                        "period_counts_abnormal": _period_counts(table, abnormal),
                        "period_counts_normal": _period_counts(table, normal),
                        "status": status,
                        "reason": reason,
                        "variant": VARIANT_POOLED,
                        "period_variant": {
                            "variant": VARIANT_MATCHED,
                            "status": quota_status,
                            "quota": {str(key2): int(value) for key2, value in sorted(quota.items())},
                            "max_matched_size": int(max_matched),
                        },
                    }
                )
        for size in SIZES_C[horizon]:
            quota, quota_status = normal_quota(table, size, arms=2)
            units_c.append(
                {
                    "unit": f"{key}|{size}",
                    "cell": key,
                    "history_id": table.history_id,
                    "horizon": horizon,
                    "size": int(size),
                    "n_normal": int(normal_positions.size),
                    "n_strata": len(table.stratum_names),
                    "quota": quota,
                    "status": quota_status,
                    "reason": (
                        ""
                        if quota_status == COMPUTED
                        else "two disjoint equal-quota normal arms do not fit"
                    ),
                }
            )
    status_counts_a: dict[str, int] = {}
    for unit in units_a:
        status_counts_a[unit["status"]] = status_counts_a.get(unit["status"], 0) + 1
    status_counts_c: dict[str, int] = {}
    for unit in units_c:
        status_counts_c[unit["status"]] = status_counts_c.get(unit["status"], 0) + 1
    return {
        "task": TASK_ID,
        "metadata_only": True,
        "embeddings_read": False,
        "outcomes_computed": False,
        "parameters": {
            "sizes_a": {key: [int(value) for value in values] for key, values in SIZES_A.items()},
            "sizes_c": {key: [int(value) for value in values] for key, values in SIZES_C.items()},
            "cluster_counts": [int(value) for value in CLUSTER_COUNTS],
            "primary_k": int(PRIMARY_K),
            "delta_grid": [float(value) for value in DELTA_GRID],
            "perturb_modes": list(PERTURB_MODES),
            "repeats_a": int(REPEATS_AB),
            "repeats_c": int(REPEATS_C),
            "encoder_seeds": [int(value) for value in readiness.FROZEN_ENCODER_SEEDS],
            "d_min_coverage": D_MIN_COVERAGE,
            "boundary_safe_min_fraction": BOUNDARY_SAFE_MIN_FRACTION,
            "boundary_proximity_factor": BOUNDARY_PROXIMITY_FACTOR,
        },
        "cells": cell_records,
        "branch_a": {
            "units": units_a,
            "n_units": len(units_a),
            "unit_status_counts": dict(sorted(status_counts_a.items())),
        },
        "branch_c": {
            "units": units_c,
            "n_units": len(units_c),
            "unit_status_counts": dict(sorted(status_counts_c.items())),
        },
        "n_cells": len(cell_records),
    }


# --- reference diagnostics ------------------------------------------------------


@dataclass(slots=True)
class ReferenceDiagnostics:
    """Per group reference-only quantities: labels, residuals, margins, direction."""

    key: str
    history_id: str
    model_seed: int
    centroids: dict[int, np.ndarray]
    ref_scale: dict[int, float]
    labels: dict[int, np.ndarray]
    residuals: dict[int, np.ndarray]
    margins: dict[int, np.ndarray]
    boundary_threshold: dict[int, float]
    direction: np.ndarray
    sigma_u: float
    n_reference: int


def build_reference_diagnostics(
    groups: Mapping[str, Any],
    manifest: Mapping[str, object],
    fits: Mapping[str, Any],
    labels: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, ReferenceDiagnostics]:
    """Derive every reference-only quantity used by branches B and C."""
    diagnostics: dict[str, ReferenceDiagnostics] = {}
    histories = sorted({key.split("|")[0] for key in groups})
    for history_id in histories:
        reference_ids = m11.reference_ids_for_history(manifest, history_id)
        reference_digest = sha256_object(reference_ids)
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            key = f"{history_id}|{model_seed}"
            group = groups[key]
            reference_positions = np.asarray(
                [group.index[fid] for fid in reference_ids], dtype=np.int64
            )
            reference_matrix = group.matrix[reference_positions]
            fit = fits[key]
            centroids: dict[int, np.ndarray] = {}
            ref_scale: dict[int, float] = {}
            residuals: dict[int, np.ndarray] = {}
            margins: dict[int, np.ndarray] = {}
            thresholds: dict[int, float] = {}
            for k in CLUSTER_COUNTS:
                centre = np.asarray(fit.kmeans[k].cluster_centers_, dtype=np.float64)
                centroids[k] = centre
                row_labels = labels[key][k]
                squared = (
                    np.einsum("ij,ij->i", group.matrix, group.matrix)[:, None]
                    + np.einsum("ij,ij->i", centre, centre)[None, :]
                    - 2.0 * (group.matrix @ centre.T)
                )
                np.maximum(squared, 0.0, out=squared)
                nearest = np.argmin(squared, axis=1)
                _require(
                    bool(np.array_equal(nearest, row_labels)),
                    f"REFUSED-INTERNAL: labels disagree with nearest centroid for {key} K={k}",
                )
                residual = group.matrix - centre[row_labels]
                residual_reference = residual[reference_positions]
                scale = float(
                    np.mean(np.einsum("ij,ij->i", residual_reference, residual_reference))
                )
                _require(scale > 0.0, "REFUSED-INTERNAL: non-positive reference residual scale")
                ref_scale[k] = scale
                residuals[k] = residual
                distances = np.sqrt(squared)
                partition = np.partition(distances, 1, axis=1)
                margin = (partition[:, 1] - partition[:, 0]) / 2.0
                margins[k] = margin
                thresholds[k] = BOUNDARY_PROXIMITY_FACTOR * math.sqrt(scale)
            centred = reference_matrix - reference_matrix.mean(axis=0)
            _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            direction = np.asarray(vt[0], dtype=np.float64)
            peak = int(np.argmax(np.abs(direction)))
            if direction[peak] < 0:
                direction = -direction
            projection = centred @ direction
            sigma_u = float(np.sqrt(np.mean(projection * projection)))
            _require(sigma_u > 0.0, "REFUSED-INTERNAL: non-positive reference directional SD")
            diagnostics[key] = ReferenceDiagnostics(
                key=key,
                history_id=history_id,
                model_seed=int(model_seed),
                centroids=centroids,
                ref_scale=ref_scale,
                labels={k: labels[key][k] for k in CLUSTER_COUNTS},
                residuals=residuals,
                margins=margins,
                boundary_threshold=thresholds,
                direction=direction,
                sigma_u=sigma_u,
                n_reference=int(reference_positions.size),
            )
            _require(
                sha256_object(reference_ids) == reference_digest,
                "REFUSED-INTERNAL: reference identity changed during the fit",
            )
    return diagnostics


# --- draws ---------------------------------------------------------------------


@dataclass(slots=True)
class ArmDraws:
    """One unit's deterministic drawing plan: disjoint equal-size arms."""

    variant: str
    unit: str
    cell: str
    stratum: str
    size: int
    abnormal: np.ndarray
    normal_1: np.ndarray
    normal_2: np.ndarray
    normal_3: np.ndarray
    quota: dict[int, int]


def draw_pooled(
    table: CellTable, stratum: str, size: int, repeats: int
) -> ArmDraws:
    """Draw one abnormal and three disjoint normal arms inside one stratum."""
    positions = table.positions(stratum)
    abnormal = positions[table.abnormal[positions]]
    normal = positions[~table.abnormal[positions]]
    _require(abnormal.size >= size, "REFUSED-SUPPORT: abnormal arm cannot be filled")
    _require(normal.size >= 3 * size, "REFUSED-SUPPORT: three normal arms cannot be filled")
    abnormal_draws = np.empty((repeats, size), dtype=np.int64)
    normal_1 = np.empty_like(abnormal_draws)
    normal_2 = np.empty_like(abnormal_draws)
    normal_3 = np.empty_like(abnormal_draws)
    for repeat in range(repeats):
        token = [TASK_ID, "A", table.key, stratum, size, repeat]
        rng = np.random.default_rng(_draw_seed(token + ["abnormal"]))
        abnormal_draws[repeat] = np.sort(rng.choice(abnormal, size=size, replace=False))
        rng = np.random.default_rng(_draw_seed(token + ["normal"]))
        picked = rng.choice(normal, size=3 * size, replace=False)
        rng.shuffle(picked)
        normal_1[repeat] = np.sort(picked[:size])
        normal_2[repeat] = np.sort(picked[size : 2 * size])
        normal_3[repeat] = np.sort(picked[2 * size :])
    return ArmDraws(
        variant=VARIANT_POOLED,
        unit=f"{table.key}|{stratum}|{size}",
        cell=table.key,
        stratum=stratum,
        size=size,
        abnormal=abnormal_draws,
        normal_1=normal_1,
        normal_2=normal_2,
        normal_3=normal_3,
        quota={},
    )


def draw_period_matched(
    table: CellTable, stratum: str, size: int, repeats: int, quota: Mapping[int, int]
) -> ArmDraws:
    """Draw arms whose per-period composition equals ``quota`` exactly."""
    positions = table.positions(stratum)
    abnormal = positions[table.abnormal[positions]]
    normal = positions[~table.abnormal[positions]]
    _require(sum(quota.values()) == size, "REFUSED-SUPPORT: period quota is not size-matched")
    abnormal_draws = np.empty((repeats, size), dtype=np.int64)
    normal_1 = np.empty_like(abnormal_draws)
    normal_2 = np.empty_like(abnormal_draws)
    normal_3 = np.empty_like(abnormal_draws)
    for repeat in range(repeats):
        token = [TASK_ID, "A-MATCHED", table.key, stratum, size, repeat]
        abnormal_parts: list[np.ndarray] = []
        normal_parts: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = (
            [],
            [],
            [],
        )
        for number, count in sorted(quota.items()):
            code = table.period_numbers.index(int(number))
            abnormal_period = abnormal[table.period_codes[abnormal] == code]
            normal_period = normal[table.period_codes[normal] == code]
            _require(
                abnormal_period.size >= count and normal_period.size >= 3 * count,
                "REFUSED-SUPPORT: period quota exceeds the period support",
            )
            rng = np.random.default_rng(_draw_seed(token + ["abnormal", number]))
            abnormal_parts.append(rng.choice(abnormal_period, size=count, replace=False))
            rng = np.random.default_rng(_draw_seed(token + ["normal", number]))
            picked = rng.choice(normal_period, size=3 * count, replace=False)
            rng.shuffle(picked)
            normal_parts[0].append(picked[:count])
            normal_parts[1].append(picked[count : 2 * count])
            normal_parts[2].append(picked[2 * count :])
        abnormal_draws[repeat] = np.sort(np.concatenate(abnormal_parts))
        normal_1[repeat] = np.sort(np.concatenate(normal_parts[0]))
        normal_2[repeat] = np.sort(np.concatenate(normal_parts[1]))
        normal_3[repeat] = np.sort(np.concatenate(normal_parts[2]))
        for arm in (normal_1[repeat], normal_2[repeat], normal_3[repeat]):
            for number, count in quota.items():
                code = table.period_numbers.index(int(number))
                _require(
                    int(np.sum(table.period_codes[arm] == code)) == count,
                    "REFUSED-INTERNAL: period-matched arm lost its quota",
                )
    return ArmDraws(
        variant=VARIANT_MATCHED,
        unit=f"{table.key}|{stratum}|{size}",
        cell=table.key,
        stratum=stratum,
        size=size,
        abnormal=abnormal_draws,
        normal_1=normal_1,
        normal_2=normal_2,
        normal_3=normal_3,
        quota={int(key): int(value) for key, value in sorted(quota.items())},
    )


@dataclass(slots=True)
class ControlDraws:
    """Branch C disjoint normal arms of one cell, size and encoder seed."""

    unit: str
    cell: str
    history_id: str
    horizon: str
    stratum_names: list[str]
    size: int
    model_seed: int
    unchanged: np.ndarray
    perturbed_base: np.ndarray
    quota: dict[str, int]


def draw_controls(
    table: CellTable,
    size: int,
    quota: Mapping[str, int],
    model_seed: int,
    repeats: int,
) -> ControlDraws:
    """Draw two disjoint equal-quota normal arms per repeat."""
    positions = table.positions()
    normal = positions[~table.abnormal[positions]]
    _require(sum(quota.values()) == size, "REFUSED-SUPPORT: normal quota is not size-matched")
    unchanged = np.empty((repeats, size), dtype=np.int64)
    perturbed = np.empty_like(unchanged)
    for repeat in range(repeats):
        token = [TASK_ID, "C", table.key, size, model_seed, repeat]
        unchanged_parts: list[np.ndarray] = []
        perturbed_parts: list[np.ndarray] = []
        for name, count in sorted(quota.items()):
            stratum_index = table.stratum_names.index(name)
            pool = normal[table.strata[normal] == stratum_index]
            _require(pool.size >= 2 * count, "REFUSED-SUPPORT: normal quota exceeds support")
            rng_a = np.random.default_rng(_draw_seed(token + ["arm-a", name]))
            arm_a = rng_a.choice(pool, size=count, replace=False)
            remaining = pool[~np.isin(pool, arm_a)]
            rng_b = np.random.default_rng(_draw_seed(token + ["arm-b", name]))
            arm_b = rng_b.choice(remaining, size=count, replace=False)
            unchanged_parts.append(arm_a)
            perturbed_parts.append(arm_b)
        unchanged[repeat] = np.sort(np.concatenate(unchanged_parts))
        perturbed[repeat] = np.sort(np.concatenate(perturbed_parts))
    return ControlDraws(
        unit=f"{table.key}|{size}|{model_seed}",
        cell=table.key,
        history_id=table.history_id,
        horizon=table.horizon,
        stratum_names=list(table.stratum_names),
        size=size,
        model_seed=int(model_seed),
        unchanged=unchanged,
        perturbed_base=perturbed,
        quota={str(key): int(value) for key, value in sorted(quota.items())},
    )


# --- estimators -----------------------------------------------------------------


def e1_repeats(
    matrix: np.ndarray, sigma: float, positions_a: np.ndarray, positions_b: np.ndarray
) -> np.ndarray:
    """Accepted unbiased squared MMD for every equal-size arm pair."""
    _require(positions_a.shape == positions_b.shape, "REFUSED-IDENTITY: arms are not size-matched")
    values = np.empty(positions_a.shape[0], dtype=np.float64)
    for repeat in range(positions_a.shape[0]):
        values[repeat] = endpoints.mmd2_unbiased(
            matrix[positions_a[repeat]], matrix[positions_b[repeat]], sigma
        )
    return values


def _nearest(
    matrix: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-prototype labels and boundary margins (half gap)."""
    squared = (
        np.einsum("ij,ij->i", matrix, matrix)[:, None]
        + np.einsum("ij,ij->i", centroids, centroids)[None, :]
        - 2.0 * (matrix @ centroids.T)
    )
    np.maximum(squared, 0.0, out=squared)
    labels = np.argmin(squared, axis=1)
    distances = np.sqrt(squared)
    partition = np.partition(distances, 1, axis=1)
    margin = (partition[:, 1] - partition[:, 0]) / 2.0
    return labels, margin


def _cluster_stats(
    residual: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(counts, mean_residual, mean_squared_norm)`` per repeat and cluster."""
    repeats, size, _dimension = residual.shape
    one_hot = np.zeros((repeats, size, k), dtype=np.float64)
    one_hot[np.arange(repeats)[:, None], np.arange(size)[None, :], labels] = 1.0
    counts = one_hot.sum(axis=1)
    sums = np.einsum("rsk,rsd->rkd", one_hot, residual)
    squared = np.einsum("rsd,rsd->rs", residual, residual)
    mean_squared = np.einsum("rsk,rs->rk", one_hot, squared)
    return counts, sums, mean_squared


def within_prototype_arrays(
    *,
    residual_a: np.ndarray,
    labels_a: np.ndarray,
    residual_b: np.ndarray,
    labels_b: np.ndarray,
    centroids: np.ndarray,
    ref_scale: float,
    k: int,
) -> dict[str, np.ndarray]:
    """Bounded within-prototype location/distribution diagnostic.

    For the prototypes occupied in both arms the diagnostic compares the mean
    residual vector (location, in every direction, not only radially), the
    mean squared residual (spread) and the radial/tangential decomposition of
    the location difference. Weights are the pooled arm mass per prototype,
    renormalised over the common prototypes; the un-renormalised covered mass
    is returned as ``coverage`` so a repeat below the declared floor is
    abstained instead of being silently renormalised.
    """
    size = residual_a.shape[1]
    counts_a, sums_a, mean_sq_a = _cluster_stats(residual_a, labels_a, k)
    counts_b, sums_b, mean_sq_b = _cluster_stats(residual_b, labels_b, k)
    mean_a = np.divide(
        sums_a, counts_a[:, :, None], out=np.zeros_like(sums_a), where=counts_a[:, :, None] > 0
    )
    mean_b = np.divide(
        sums_b, counts_b[:, :, None], out=np.zeros_like(sums_b), where=counts_b[:, :, None] > 0
    )
    spread_a = np.divide(mean_sq_a, counts_a, out=np.zeros_like(mean_sq_a), where=counts_a > 0)
    spread_b = np.divide(mean_sq_b, counts_b, out=np.zeros_like(mean_sq_b), where=counts_b > 0)
    common = (counts_a > 0) & (counts_b > 0)
    mass = np.where(common, (counts_a + counts_b) / (2.0 * size), 0.0)
    coverage = mass.sum(axis=1)
    weight = np.divide(
        mass, coverage[:, None], out=np.zeros_like(mass), where=coverage[:, None] > 0
    )
    difference = mean_a - mean_b
    norm_sq = np.einsum("rkd,rkd->rk", difference, difference)
    centroid_norm = np.linalg.norm(centroids, axis=1)
    unit = np.divide(
        centroids,
        centroid_norm[:, None],
        out=np.zeros_like(centroids),
        where=centroid_norm[:, None] > 0,
    )
    radial = np.einsum("rkd,kd->rk", difference, unit)
    radial_sq = radial * radial
    tangential_sq = np.maximum(norm_sq - radial_sq, 0.0)
    valid = coverage >= D_MIN_COVERAGE
    denominator = ref_scale * np.where(valid, 1.0, np.nan)
    return {
        "D_LOC": np.where(valid, (weight * norm_sq).sum(axis=1) / denominator, np.nan),
        "D_SPREAD": np.where(
            valid, (weight * np.abs(spread_a - spread_b)).sum(axis=1) / denominator, np.nan
        ),
        "D_RADIAL": np.where(valid, (weight * radial_sq).sum(axis=1) / denominator, np.nan),
        "D_TANGENTIAL": np.where(
            valid, (weight * tangential_sq).sum(axis=1) / denominator, np.nan
        ),
        "coverage": coverage,
        "valid": valid.astype(np.float64),
    }


def within_prototype_positions(
    *,
    residuals: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    ref_scale: float,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """Position-indexed wrapper of :func:`within_prototype_arrays`."""
    return within_prototype_arrays(
        residual_a=residuals[positions_a],
        labels_a=labels[positions_a],
        residual_b=residuals[positions_b],
        labels_b=labels[positions_b],
        centroids=centroids,
        ref_scale=ref_scale,
        k=k,
    )


def _e2_key(k: int) -> str:
    return f"E2_K{k}"


def _period_mix_tv(draws: ArmDraws, table: CellTable) -> np.ndarray:
    """Per-repeat period total-variation distance between the comparison arms."""
    period_total = len(table.period_numbers)
    codes_a = table.period_codes[draws.abnormal]
    codes_b = table.period_codes[draws.normal_1]
    counts_a = m12.prototype_counts(codes_a, period_total)
    counts_b = m12.prototype_counts(codes_b, period_total)
    return np.abs(counts_a - counts_b).sum(axis=1) / (2.0 * draws.size)


def arm_values(
    *,
    draws: ArmDraws,
    matrix: np.ndarray,
    sigma: float,
    diagnostics: ReferenceDiagnostics,
    mapping: np.ndarray,
    table: CellTable,
) -> dict[str, np.ndarray]:
    """Compute every per-repeat value of one arm plan and its matched null."""
    positions_a = mapping[draws.abnormal]
    positions_n1 = mapping[draws.normal_1]
    positions_n2 = mapping[draws.normal_2]
    positions_n3 = mapping[draws.normal_3]
    values: dict[str, np.ndarray] = {}
    values["E1_contrast"] = e1_repeats(matrix, sigma, positions_a, positions_n1)
    values["E1_null"] = e1_repeats(matrix, sigma, positions_n2, positions_n3)
    for k in CLUSTER_COUNTS:
        values[f"E2_K{k}_contrast"] = m12.e2_quanta(
            diagnostics.labels[k], positions_a, positions_n1, k
        ).astype(np.float64)
        values[f"E2_K{k}_null"] = m12.e2_quanta(
            diagnostics.labels[k], positions_n2, positions_n3, k
        ).astype(np.float64)
    for k in CLUSTER_COUNTS:
        contrast = within_prototype_positions(
            residuals=diagnostics.residuals[k],
            labels=diagnostics.labels[k],
            centroids=diagnostics.centroids[k],
            ref_scale=diagnostics.ref_scale[k],
            positions_a=positions_a,
            positions_b=positions_n1,
            k=k,
        )
        null = within_prototype_positions(
            residuals=diagnostics.residuals[k],
            labels=diagnostics.labels[k],
            centroids=diagnostics.centroids[k],
            ref_scale=diagnostics.ref_scale[k],
            positions_a=positions_n2,
            positions_b=positions_n3,
            k=k,
        )
        for name in ("D_LOC", "D_SPREAD"):
            values[f"{name}_K{k}_contrast"] = contrast[name]
            values[f"{name}_K{k}_null"] = null[name]
        values[f"coverage_K{k}_contrast"] = contrast["coverage"]
        values[f"coverage_K{k}_null"] = null["coverage"]
        if k == PRIMARY_K:
            for name in ("D_RADIAL", "D_TANGENTIAL"):
                values[f"{name}_contrast"] = contrast[name]
                values[f"{name}_null"] = null[name]
    values["period_tv"] = _period_mix_tv(draws, table)
    values["boundary_fraction_abnormal"] = np.asarray(
        [
            float(
                np.mean(
                    diagnostics.margins[PRIMARY_K][positions_a[repeat]]
                    <= diagnostics.boundary_threshold[PRIMARY_K]
                )
            )
            for repeat in range(positions_a.shape[0])
        ]
    )
    values["boundary_fraction_normal_1"] = np.asarray(
        [
            float(
                np.mean(
                    diagnostics.margins[PRIMARY_K][positions_n1[repeat]]
                    <= diagnostics.boundary_threshold[PRIMARY_K]
                )
            )
            for repeat in range(positions_n1.shape[0])
        ]
    )
    return values


# --- summaries ------------------------------------------------------------------


def float_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarise a finite float repeat vector; abstains when nothing is finite."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0, "status": ABSTAIN_COVERAGE}
    return {
        "n": int(finite.size),
        "status": COMPUTED,
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "fraction_strictly_positive": float(np.mean(finite > 0)),
        "fraction_zero": float(np.mean(finite == 0)),
        "fraction_strictly_negative": float(np.mean(finite < 0)),
    }


def quanta_summary(quanta: np.ndarray, size: int) -> dict[str, Any]:
    """Summarise an exact integer-quanta vector (M12 convention)."""
    values = np.asarray(quanta, dtype=np.int64).reshape(-1)
    values = values[values >= 0]
    if values.size == 0:
        return {"n": 0, "status": ABSTAIN_COVERAGE}
    summary = m12.summary_of_quanta(values, size)
    summary["status"] = COMPUTED
    return summary


def comparison_summary(
    contrast: np.ndarray, null: np.ndarray, kind: str, size: int
) -> dict[str, Any]:
    """Contrast and matched null summaries plus the null-q95 excess description."""
    if kind in ("E2_K4", "E2_K8", "E2_K16"):
        contrast_summary = quanta_summary(contrast, size)
        null_summary = quanta_summary(null, size)
    else:
        contrast_summary = float_summary(contrast)
        null_summary = float_summary(null)
    null_q95 = null_summary.get("q95")
    median = contrast_summary.get("median")
    status = (
        COMPUTED
        if contrast_summary.get("status") == COMPUTED
        else contrast_summary.get("status", ABSTAIN_COVERAGE)
    )
    if null_q95 is None or median is None:
        return {
            "status": ABSTAIN_COVERAGE,
            "contrast": contrast_summary,
            "null": null_summary,
            "null_q95": null_q95,
            "median_above_null_q95": False,
            "fraction_contrast_above_null_q95": 0.0,
        }
    if kind in ("E2_K4", "E2_K8", "E2_K16"):
        # null_q95 is in E2 units (quanta / size). The -1 sentinel marks a
        # repeat whose boundary-safe perturbation fell below the mass floor;
        # it is an abstention, not a zero and not a denominator entry.
        raw = np.asarray(contrast, dtype=np.float64)
        finite = raw[raw >= 0] / float(size)
    else:
        finite = np.asarray(contrast, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
    fraction = float(np.mean(finite > null_q95)) if finite.size else 0.0
    return {
        "status": status,
        "contrast": contrast_summary,
        "null": null_summary,
        "null_q95": float(null_q95),
        "median_above_null_q95": bool(median > null_q95),
        "fraction_contrast_above_null_q95": fraction,
    }


def summarise_arm_values(
    values: Mapping[str, np.ndarray], size: int
) -> dict[str, Any]:
    """Build the published estimator summaries and coverage record of one arm plan."""
    coverage_by_k = {
        k: np.asarray(values[f"coverage_K{k}_contrast"], dtype=np.float64)
        for k in CLUSTER_COUNTS
    }
    # The frozen rule is per repeat: every repeat below the 0.5 coverage floor
    # is already NaN here and is dropped by the summaries, so each D summary is
    # computed over its valid repeats and publishes that count as ``n``. The
    # coverage record below is descriptive only; it is deliberately not a
    # second unit-level gate.
    valid_fraction_by_k = {
        k: float(np.mean(np.isfinite(value) & (value >= D_MIN_COVERAGE)))
        for k, value in coverage_by_k.items()
    }
    repeats_total = int(coverage_by_k[PRIMARY_K].size)
    repeats_valid_by_k = {
        k: int(np.sum(np.isfinite(value) & (value >= D_MIN_COVERAGE)))
        for k, value in coverage_by_k.items()
    }
    estimators: dict[str, Any] = {}
    for name in ESTIMATORS:
        key = f"{name}_K{PRIMARY_K}" if name in ("D_LOC", "D_SPREAD") else name
        estimators[name] = comparison_summary(
            values[f"{key}_contrast"], values[f"{key}_null"], name, size
        )
    for name in ("D_RADIAL", "D_TANGENTIAL"):
        estimators[name] = comparison_summary(
            values[f"{name}_contrast"], values[f"{name}_null"], name, size
        )
    diagnostic_by_k = {}
    for k in CLUSTER_COUNTS:
        diagnostic_by_k[f"K{k}"] = {
            "D_LOC": comparison_summary(
                values[f"D_LOC_K{k}_contrast"], values[f"D_LOC_K{k}_null"], "D_LOC", size
            ),
            "D_SPREAD": comparison_summary(
                values[f"D_SPREAD_K{k}_contrast"],
                values[f"D_SPREAD_K{k}_null"],
                "D_SPREAD",
                size,
            ),
        }
    coverage = coverage_by_k[PRIMARY_K]
    valid_fraction = valid_fraction_by_k[PRIMARY_K]
    return {
        "estimators": estimators,
        "diagnostic_by_k": diagnostic_by_k,
        "coverage": {
            "valid_fraction": valid_fraction,
            "median": float(np.nanmedian(coverage)) if np.isfinite(coverage).any() else None,
            "repeats_valid": repeats_valid_by_k[PRIMARY_K],
            "repeats_total": repeats_total,
            "reliability": "FULL" if repeats_valid_by_k[PRIMARY_K] == repeats_total else "PARTIAL",
            "at_least_half_valid": bool(valid_fraction >= 0.5),
            "d_min_coverage": D_MIN_COVERAGE,
            "note": (
                "descriptive only: D summaries use the valid repeats and report their n; "
                "this record is not a unit-level estimator gate"
            ),
            "by_k": {
                f"K{k}": {
                    "valid_fraction": valid_fraction_by_k[k],
                    "repeats_valid": repeats_valid_by_k[k],
                    "repeats_total": repeats_total,
                    "reliability": (
                        "FULL" if repeats_valid_by_k[k] == repeats_total else "PARTIAL"
                    ),
                    "at_least_half_valid": bool(valid_fraction_by_k[k] >= 0.5),
                }
                for k in CLUSTER_COUNTS
            },
        },
        "period_mix_tv": {
            "median": float(np.median(values["period_tv"])),
            "max": float(np.max(values["period_tv"])),
        },
        "boundary_proximity_fraction": {
            "abnormal": float(np.median(values["boundary_fraction_abnormal"])),
            "normal": float(np.median(values["boundary_fraction_normal_1"])),
            "factor": BOUNDARY_PROXIMITY_FACTOR,
        },
    }


# --- positive controls -----------------------------------------------------------


def perturb(
    rows: np.ndarray,
    *,
    delta: float,
    direction: np.ndarray,
    sigma_u: float,
    margin: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the predeclared reference-derived shift.

    ``WITHIN-PROTOTYPE-SAFE`` shifts only rows whose prototype margin strictly
    exceeds the shift norm, so the nearest prototype cannot change by the
    triangle inequality. Returns the shifted rows, the perturbed mask and the
    exact realised shift norm.
    """
    shift = float(delta) * float(sigma_u) * np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(shift))
    if mode == MODE_GLOBAL:
        mask = np.ones(rows.shape[0], dtype=bool)
    elif mode == MODE_SAFE:
        mask = margin > norm
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"REFUSED-DESIGN: unknown perturbation mode {mode}")
    shifted = rows.copy()
    if mask.any():
        shifted[mask] = shifted[mask] + shift
    return shifted, mask, norm


def score_controls(
    *,
    draw: ControlDraws,
    matrix: np.ndarray,
    sigma: float,
    diagnostics: ReferenceDiagnostics,
    mapping: np.ndarray,
) -> dict[str, Any]:
    """Score the zero-shift null and every predeclared perturbed record."""
    repeats = draw.unchanged.shape[0]
    positions_a = mapping[draw.unchanged]
    positions_b = mapping[draw.perturbed_base]
    size = draw.size
    labels_primary = diagnostics.labels[PRIMARY_K]
    centroids_primary = diagnostics.centroids[PRIMARY_K]
    labels_before = labels_primary[positions_b]
    residual_a = diagnostics.residuals[PRIMARY_K][positions_a]
    shifted_batch = matrix[positions_b].copy()  # (R, size, dim)
    base_margin = diagnostics.margins[PRIMARY_K][positions_b]
    records: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    null_arrays: dict[str, np.ndarray] | None = None
    for delta in DELTA_GRID:
        modes = (MODE_GLOBAL,) if delta == 0.0 else PERTURB_MODES
        for mode in modes:
            key = f"{mode}|{delta:g}"
            perturbed_rows = np.empty_like(matrix[positions_b])
            mask = np.empty((repeats, size), dtype=bool)
            shift_norm = 0.0
            for repeat in range(repeats):
                shifted_single, mask_single, shift_norm = perturb(
                    matrix[positions_b[repeat]],
                    delta=delta,
                    direction=diagnostics.direction,
                    sigma_u=diagnostics.sigma_u,
                    margin=base_margin[repeat],
                    mode=mode,
                )
                perturbed_rows[repeat] = shifted_single
                mask[repeat] = mask_single
            fraction = mask.mean(axis=1)
            valid = fraction >= BOUNDARY_SAFE_MIN_FRACTION
            labels_after_primary, _margin_after = _nearest(perturbed_rows.reshape(-1, perturbed_rows.shape[-1]), centroids_primary)
            labels_after_primary = labels_after_primary.reshape(repeats, size)
            reassigned = np.sum(labels_after_primary != labels_before, axis=1).astype(np.int64)
            residual_b = perturbed_rows.reshape(-1, perturbed_rows.shape[-1]) - centroids_primary[
                labels_after_primary.reshape(-1)
            ]
            residual_b = residual_b.reshape(repeats, size, -1)
            diagnostic = within_prototype_arrays(
                residual_a=residual_a,
                labels_a=labels_primary[positions_a],
                residual_b=residual_b,
                labels_b=labels_after_primary,
                centroids=centroids_primary,
                ref_scale=diagnostics.ref_scale[PRIMARY_K],
                k=PRIMARY_K,
            )
            e1_values = np.empty(repeats, dtype=np.float64)
            e2_values = {k: np.empty(repeats, dtype=np.int64) for k in CLUSTER_COUNTS}
            for repeat in range(repeats):
                e1_values[repeat] = endpoints.mmd2_unbiased(
                    matrix[positions_a[repeat]], perturbed_rows[repeat], sigma
                )
                for k in CLUSTER_COUNTS:
                    centres = diagnostics.centroids[k]
                    after_k, _margin_k = _nearest(perturbed_rows[repeat], centres)
                    counts_a = np.bincount(labels_primary_at(diagnostics, k, positions_a[repeat]), minlength=k)
                    counts_b = np.bincount(after_k, minlength=k)
                    absolute = int(np.abs(counts_a - counts_b).sum())
                    _require(absolute % 2 == 0, "REFUSED-INTERNAL: occupancy difference is not even")
                    e2_values[k][repeat] = absolute // 2
            for k in CLUSTER_COUNTS:
                e2_values[k][~valid] = -1
            e1_values[~valid] = np.nan
            for name in ("D_LOC", "D_SPREAD", "D_RADIAL", "D_TANGENTIAL"):
                diagnostic[name] = np.where(valid, diagnostic[name], np.nan)
            reassigned = np.where(valid, reassigned, -1)
            record_arrays = {
                "E1": e1_values,
                "E2_K4": e2_values[4].astype(np.float64),
                "E2_K8": e2_values[8].astype(np.float64),
                "E2_K16": e2_values[16].astype(np.float64),
                "D_LOC": diagnostic["D_LOC"],
                "D_SPREAD": diagnostic["D_SPREAD"],
                "D_RADIAL": diagnostic["D_RADIAL"],
                "D_TANGENTIAL": diagnostic["D_TANGENTIAL"],
            }
            arrays.update({f"{key}|{name}": value for name, value in record_arrays.items()})
            arrays[f"{key}|reassigned"] = reassigned.astype(np.float64)
            arrays[f"{key}|perturbed"] = (mask.sum(axis=1)).astype(np.float64)
            arrays[f"{key}|fraction"] = fraction.astype(np.float64)
            arrays[f"{key}|valid"] = valid.astype(np.float64)
            if delta == 0.0:
                null_arrays = {name: value.copy() for name, value in record_arrays.items()}
            record = {
                "unit": draw.unit,
                "cell": draw.cell,
                "history_id": draw.history_id,
                "horizon": draw.horizon,
                "size": int(size),
                "model_seed": int(draw.model_seed),
                "delta": float(delta),
                "delta_units": "reference directional standard deviation",
                "mode": mode,
                "shift_norm": float(shift_norm),
                "sigma_u": float(diagnostics.sigma_u),
                "direction_sha256": sha256_object(
                    [float(value) for value in diagnostics.direction]
                ),
                "valid_repeats": int(valid.sum()),
                "status": (
                    COMPUTED
                    if valid.sum() >= repeats * BOUNDARY_SAFE_MIN_FRACTION
                    else ABSTAIN_BOUNDARY_MASS
                ),
                "n_perturbed_median": float(np.median(mask.sum(axis=1))),
                "perturb_fraction_median": float(np.median(fraction)),
                "estimators": {},
                "reassignment": {
                    "median": float(np.median(reassigned[valid])) if valid.any() else None,
                    "max": int(reassigned[valid].max()) if valid.any() else None,
                    "fraction_repeats_zero": (
                        float(np.mean(reassigned[valid] == 0)) if valid.any() else None
                    ),
                    "fraction_repeats_any": (
                        float(np.mean(reassigned[valid] > 0)) if valid.any() else None
                    ),
                },
                "repeat0": {
                    "E1": float(e1_values[0]),
                    "E2_K4": int(e2_values[4][0]),
                    "E2_K8": int(e2_values[8][0]),
                    "E2_K16": int(e2_values[16][0]),
                    "D_LOC": float(diagnostic["D_LOC"][0]),
                    "D_SPREAD": float(diagnostic["D_SPREAD"][0]),
                    "reassigned": int(reassigned[0]),
                    "perturbed": int(mask[0].sum()),
                },
            }
            records[key] = record
    _require(null_arrays is not None, "REFUSED-INTERNAL: zero-shift null record missing")
    for key, record in records.items():
        if record["status"] != COMPUTED:
            # A boundary-safe record below the perturbed-mass floor is an
            # abstention, not a low-validity response.
            for name in CONTROL_ESTIMATORS:
                record["estimators"][name] = {"status": record["status"]}
            continue
        for name in CONTROL_ESTIMATORS:
            contrast = arrays[f"{key}|{name}"]
            record["estimators"][name] = comparison_summary(
                contrast, null_arrays[name], name, size
            )
    identity = identity_check(
        draw=draw,
        matrix=matrix,
        sigma=sigma,
        diagnostics=diagnostics,
        mapping=mapping,
    )
    return {
        "record": {
            "unit": draw.unit,
            "cell": draw.cell,
            "history_id": draw.history_id,
            "horizon": draw.horizon,
            "size": int(size),
            "model_seed": int(draw.model_seed),
            "quota": draw.quota,
            "status": COMPUTED,
            "records": records,
            "zero_identity": identity,
        },
        "arrays": arrays,
        "draws": {
            "unchanged": draw.unchanged.astype(np.int16),
            "perturbed_base": draw.perturbed_base.astype(np.int16),
            "direction": diagnostics.direction,
        },
    }


def labels_primary_at(
    diagnostics: ReferenceDiagnostics, k: int, positions: np.ndarray
) -> np.ndarray:
    """Return the stored prototype labels of one arm for prototype level ``k``."""
    return diagnostics.labels[k][positions]


def identity_check(
    *,
    draw: ControlDraws,
    matrix: np.ndarray,
    sigma: float,
    diagnostics: ReferenceDiagnostics,
    mapping: np.ndarray,
) -> dict[str, Any]:
    """Zero-shift identity check on the first unchanged arm.

    ``E2``, ``D_LOC``, ``D_SPREAD`` and the reassignment count must be exactly
    zero; the accepted unbiased E1 is *not* zero on identical samples by its
    finite-sample definition, so its exact identity value is published and the
    biased plug-in E1 (which is zero on identical samples) is the zero check
    for the MMD scale.
    """
    positions = mapping[draw.unchanged[0]]
    rows = matrix[positions]
    _require(rows.shape[0] == draw.size, "REFUSED-INTERNAL: identity arm size drift")
    value_biased = endpoints.mmd2_biased(rows, rows, sigma)
    value_unbiased = endpoints.mmd2_unbiased(rows, rows, sigma)
    _require(abs(value_biased) <= 1e-12, "REFUSED-INTERNAL: biased E1 is not zero on identity")
    labels = diagnostics.labels[PRIMARY_K]
    e2 = {
        _e2_key(k): int(
            m12.e2_quanta(
                labels_of(diagnostics, k), positions[None, :], positions[None, :], k
            )[0]
        )
        for k in CLUSTER_COUNTS
    }
    _require(all(value == 0 for value in e2.values()), "REFUSED-INTERNAL: E2 identity is not zero")
    diagnostic = within_prototype_positions(
        residuals=diagnostics.residuals[PRIMARY_K],
        labels=labels,
        centroids=diagnostics.centroids[PRIMARY_K],
        ref_scale=diagnostics.ref_scale[PRIMARY_K],
        positions_a=positions[None, :],
        positions_b=positions[None, :],
        k=PRIMARY_K,
    )
    _require(
        abs(float(diagnostic["D_LOC"][0])) <= 1e-12
        and abs(float(diagnostic["D_SPREAD"][0])) <= 1e-12,
        "REFUSED-INTERNAL: D identity is not zero",
    )
    after, _margin = _nearest(rows, diagnostics.centroids[PRIMARY_K])
    reassigned = int(np.sum(after != labels[positions]))
    _require(reassigned == 0, "REFUSED-INTERNAL: identity reassigned a prototype")
    return {
        "arm_sha256": sha256_object(sorted(draw.unchanged[0].tolist())),
        "size": int(draw.size),
        "E1_biased": float(value_biased),
        "E1_unbiased": float(value_unbiased),
        "E2": e2,
        "D_LOC": float(diagnostic["D_LOC"][0]),
        "D_SPREAD": float(diagnostic["D_SPREAD"][0]),
        "D_RADIAL": float(diagnostic["D_RADIAL"][0]),
        "D_TANGENTIAL": float(diagnostic["D_TANGENTIAL"][0]),
        "reassigned": reassigned,
        "exact_zero_estimators": [
            "E1_biased",
            "E2_K4",
            "E2_K8",
            "E2_K16",
            "D_LOC",
            "D_SPREAD",
        ],
        "unbiased_E1_identity_value": (
            "not zero on identical samples by the unbiased estimator's diagonal; published"
        ),
    }


def labels_of(diagnostics: ReferenceDiagnostics, k: int) -> np.ndarray:
    return diagnostics.labels[k]


# --- run ------------------------------------------------------------------------


def _mapping_for(table: CellTable, group: Any) -> np.ndarray:
    return np.asarray([group.index[fid] for fid in table.ids], dtype=np.int64)


def _support_record(census: Mapping[str, Any]) -> dict[str, Any]:
    units = census["branch_a"]["units"]
    by_cell: dict[str, dict[str, int]] = {}
    for unit in units:
        bucket = by_cell.setdefault(unit["cell"], {"planned": 0, "computed": 0, "abstained": 0})
        bucket["planned"] += 1
        if unit["status"] == COMPUTED:
            bucket["computed"] += 1
        else:
            bucket["abstained"] += 1
    return {
        "branch_a": {
            "planned_units": len(units),
            "computed_units": sum(1 for unit in units if unit["status"] == COMPUTED),
            "abstained_units": sum(1 for unit in units if unit["status"] != COMPUTED),
            "unit_status_counts": census["branch_a"]["unit_status_counts"],
            "by_cell": dict(sorted(by_cell.items())),
            "period_matched": {
                "planned": len(units),
                "computed": sum(
                    1 for unit in units if unit["period_variant"]["status"] == COMPUTED
                ),
                "abstained": sum(
                    1 for unit in units if unit["period_variant"]["status"] != COMPUTED
                ),
            },
        },
        "branch_c": census["branch_c"],
        "cells": census["cells"],
    }


def _aggregate_a(unit_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for unit in unit_records:
        for seed, seed_record in unit["seeds"].items():
            for name, summary in seed_record["estimators"].items():
                if name not in ESTIMATORS:
                    continue
                if summary.get("status") != COMPUTED:
                    continue
                key = (
                    unit["cell"],
                    unit["horizon"],
                    unit["size"],
                    unit["variant"],
                    int(seed),
                    name,
                )
                buckets.setdefault(key, []).append(summary)
    aggregates: list[dict[str, Any]] = []
    for key, entries in sorted(buckets.items()):
        cell, horizon, size, variant, seed, name = key
        medians = [entry["contrast"]["median"] for entry in entries]
        above = sum(1 for entry in entries if entry["median_above_null_q95"])
        aggregate = {
            "cell": cell,
            "horizon": horizon,
            "size": size,
            "variant": variant,
            "model_seed": seed,
            "estimator": name,
            "n_units": len(entries),
            "n_units_median_above_null_q95": above,
            "fraction_units_median_above_null_q95": above / len(entries),
            "median_of_unit_medians": float(np.median(medians)),
            "min_of_unit_medians": float(np.min(medians)),
            "max_of_unit_medians": float(np.max(medians)),
            "median_null_median": float(
                np.median([entry["null"]["median"] for entry in entries])
            ),
            "median_null_q95": float(np.median([entry["null_q95"] for entry in entries])),
            "median_fraction_contrast_above_null_q95": float(
                np.median([entry["fraction_contrast_above_null_q95"] for entry in entries])
            ),
        }
        aggregates.append(aggregate)
    return aggregates


def _aggregate_b(unit_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, int]] = {}
    for unit in unit_records:
        for seed, seed_record in unit["seeds"].items():
            estimates = seed_record["estimators"]
            occupancy = estimates.get("E2_K8")
            location = estimates.get("D_LOC")
            spread = estimates.get("D_SPREAD")
            e1 = estimates.get("E1")
            if any(
                item is None or item.get("status") != COMPUTED
                for item in (occupancy, location, spread, e1)
            ):
                continue
            key = (unit["cell"], unit["horizon"], unit["size"], unit["variant"], int(seed))
            bucket = buckets.setdefault(
                key,
                {
                    "n_units": 0,
                    "n_occupancy_zero": 0,
                    "n_location_above_null": 0,
                    "n_within_prototype_only": 0,
                    "n_e1_above_null": 0,
                    "E1": [],
                    "E2_K8": [],
                    "D_LOC": [],
                    "D_SPREAD": [],
                    "null_E1_q95": [],
                    "null_E2_K8_q95": [],
                    "null_D_LOC_q95": [],
                    "null_D_SPREAD_q95": [],
                },
            )
            bucket["n_units"] += 1
            occupancy_zero = occupancy["contrast"]["median"] == 0.0
            location_above = bool(location["median_above_null_q95"])
            bucket["n_occupancy_zero"] += int(occupancy_zero)
            bucket["n_location_above_null"] += int(location_above)
            bucket["n_within_prototype_only"] += int(occupancy_zero and location_above)
            bucket["n_e1_above_null"] += int(e1["median_above_null_q95"])
            bucket["E1"].append(e1["contrast"]["median"])
            bucket["E2_K8"].append(occupancy["contrast"]["median"])
            bucket["D_LOC"].append(location["contrast"]["median"])
            bucket["D_SPREAD"].append(spread["contrast"]["median"])
            bucket["null_E1_q95"].append(e1["null_q95"])
            bucket["null_E2_K8_q95"].append(occupancy["null_q95"])
            bucket["null_D_LOC_q95"].append(location["null_q95"])
            bucket["null_D_SPREAD_q95"].append(spread["null_q95"])
    records = []
    for key, value in sorted(buckets.items()):
        record = {
            "cell": key[0],
            "horizon": key[1],
            "size": key[2],
            "variant": key[3],
            "model_seed": key[4],
            "n_units": value["n_units"],
            "n_occupancy_zero": value["n_occupancy_zero"],
            "n_location_above_null": value["n_location_above_null"],
            "n_within_prototype_only": value["n_within_prototype_only"],
            "n_e1_above_null": value["n_e1_above_null"],
        }
        for name in ("E1", "E2_K8", "D_LOC", "D_SPREAD"):
            record[f"median_{name}"] = float(np.median(value[name]))
            record[f"median_null_{name}_q95"] = float(np.median(value[f"null_{name}_q95"]))
        records.append(record)
    return {
        "note": (
            "K=8 is the inherited primary occupancy level; within_prototype_only counts "
            "units whose E2_K8 median is exactly zero while D_LOC_K8 is above the matched "
            "normal-normal null q95. Occupancy and within-prototype movement are reported "
            "as separate diagnostics, never as one endpoint."
        ),
        "records": records,
    }


def _aggregate_c(controls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Positive-control responsiveness and reassignment aggregates."""
    buckets: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for unit in controls:
        if unit.get("status") != COMPUTED:
            continue
        for key, record in unit["records"].items():
            if record.get("status") != COMPUTED:
                continue
            mode, token = key.split("|")
            delta = float(token)
            base = (unit["horizon"], unit["size"], int(record["model_seed"]), delta, mode)
            for name in CONTROL_ESTIMATORS:
                summary = record["estimators"][name]
                if summary.get("status") != COMPUTED:
                    continue
                bucket = buckets.setdefault(
                    base + (name,),
                    {"detection": [], "contrast": [], "null_q95": [], "above": []},
                )
                bucket["detection"].append(float(summary["fraction_contrast_above_null_q95"]))
                bucket["contrast"].append(float(summary["contrast"]["median"]))
                bucket["null_q95"].append(float(summary["null_q95"]))
                bucket["above"].append(float(summary["median_above_null_q95"]))
            reassignment = record["reassignment"]
            if reassignment["median"] is not None:
                bucket = buckets.setdefault(
                    base + ("REASSIGNMENT",),
                    {"median": [], "zero": [], "any": []},
                )
                bucket["median"].append(float(reassignment["median"]))
                bucket["zero"].append(float(reassignment["fraction_repeats_zero"]))
                bucket["any"].append(float(reassignment["fraction_repeats_any"]))
    records: list[dict[str, Any]] = []
    for key, value in sorted(buckets.items()):
        horizon, size, seed, delta, mode, name = key
        record: dict[str, Any] = {
            "horizon": horizon,
            "size": size,
            "model_seed": seed,
            "delta": delta,
            "mode": mode,
            "estimator": name,
            "n_units": len(value.get("detection", value.get("median", []))),
        }
        if name == "REASSIGNMENT":
            record["median_reassigned"] = float(np.median(value["median"]))
            record["median_fraction_repeats_zero"] = float(np.median(value["zero"]))
            record["median_fraction_repeats_any"] = float(np.median(value["any"]))
        else:
            record["n_units_median_above_null_q95"] = int(sum(value["above"]))
            record["median_detection_fraction"] = float(np.median(value["detection"]))
            record["median_contrast_median"] = float(np.median(value["contrast"]))
            record["median_null_q95"] = float(np.median(value["null_q95"]))
        records.append(record)
    return {
        "note": (
            "Per horizon x size x seed x delta x mode x estimator (and REASSIGNMENT) "
            "over the 24 cell x size unit-seeds. detection_fraction is the share of "
            "repeats above the delta=0 null q95; it measures responsiveness only."
        ),
        "records": records,
    }


def _source_reuse(units: Sequence[Mapping[str, Any]], table: CellTable) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for unit in units:
        for item in unit["candidate_abnormal"]:
            counts[item] = counts.get(item, 0) + 1
        for item in unit["candidate_normal"]:
            counts[item] = counts.get(item, 0) + 1
    values = sorted(counts.values())
    return {
        "n_files": len(table.ids),
        "n_files_reused": int(sum(1 for value in values if value > 1)),
        "max_units_per_file": int(values[-1]) if values else 0,
        "median_units_per_file": float(np.median(values)) if values else 0.0,
    }


def run_diagnostic(
    *,
    corpus: Path,
    audit_path: Path,
    contract_path: Path,
    cache_root: Path,
    census_path: Path | None = None,
    g2_preflight_path: Path | None = None,
    repeats_ab: int = REPEATS_AB,
    repeats_c: int = REPEATS_C,
) -> dict[str, Any]:
    """Run the complete cache-only M14 audit and return its tables."""
    started = time.time()
    spec = load_specification()
    amendments = load_amendments(spec_sha256=str(spec["sha256"]))
    lineage = m12.verify_lineage(corpus, audit_path, contract_path, g2_preflight_path)
    products_before = verify_accepted_products()
    manifest = lineage["manifest"]
    audit_record = lineage["audit_record"]
    identity_plan = lineage["identity_plan"]

    cache = m12.load_cache(cache_root, manifest, lineage["custody"], identity_plan)
    _require(
        cache["cache_file_digest_digest"] == CACHE_DIGEST_DIGEST,
        "REFUSED-CACHE-DRIFT: admitted cache digest-of-digests changed",
    )
    groups = cache["groups"]
    for history_id in sorted({key.split("|")[0] for key in groups}):
        orders = [
            groups[f"{history_id}|{seed}"].file_ids for seed in readiness.FROZEN_ENCODER_SEEDS
        ]
        _require(
            all(order == orders[0] for order in orders),
            f"REFUSED-IDENTITY: encoder seeds disagree on the file-ID order of {history_id}",
        )

    census = build_census(manifest, audit_record)
    census_check: dict[str, Any] = {"checked": False, "match": None, "sha256": None}
    if census_path is not None:
        _require(census_path.is_file(), f"REFUSED-CENSUS-MISSING: {census_path}")
        recorded = json.loads(census_path.read_text(encoding="utf-8"))
        recorded_census = recorded.get("census", recorded)
        recorded_units = {
            str(unit["unit"]): unit
            for unit in recorded_census.get("branch_a", {}).get("units", [])
        }
        _require(
            len(recorded_units) == len(census["branch_a"]["units"]),
            "REFUSED-CENSUS-DRIFT: recorded branch-A unit count differs",
        )
        for unit in census["branch_a"]["units"]:
            recorded_unit = recorded_units.get(unit["unit"])
            _require(
                recorded_unit is not None,
                f"REFUSED-CENSUS-DRIFT: unit {unit['unit']} not in the recorded census",
            )
            _require(
                recorded_unit["status"] == unit["status"],
                f"REFUSED-CENSUS-DRIFT: unit {unit['unit']} status differs",
            )
            _require(
                recorded_unit["period_variant"]["status"] == unit["period_variant"]["status"],
                f"REFUSED-CENSUS-DRIFT: unit {unit['unit']} period variant differs",
            )
        recorded_c = {
            str(unit["unit"]): unit
            for unit in recorded_census.get("branch_c", {}).get("units", [])
        }
        for unit in census["branch_c"]["units"]:
            recorded_unit = recorded_c.get(unit["unit"])
            _require(
                recorded_unit is not None and recorded_unit["status"] == unit["status"],
                f"REFUSED-CENSUS-DRIFT: branch-C unit {unit['unit']} differs",
            )
        census_check = {
            "checked": True,
            "match": True,
            "sha256": sha256_file(census_path),
            "path": str(census_path),
        }

    fits, fit_custody = m12.fit_references(groups, manifest)
    # Timing is not part of the evidence; keeping it out makes lineage.json
    # byte-identical between deterministic reproduction runs.
    fit_custody = {
        key: {name: value for name, value in record.items() if name != "fit_wall_s"}
        for key, record in fit_custody.items()
    }
    labels = m12.prototype_labels(groups, fits)
    diagnostics = build_reference_diagnostics(groups, manifest, fits, labels)

    cells = m13.build_cells(audit_record)
    census_by_unit = {unit["unit"]: unit for unit in census["branch_a"]["units"]}
    census_c_by_unit = {unit["unit"]: unit for unit in census["branch_c"]["units"]}

    units_a: list[dict[str, Any]] = []
    units_a_matched: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    repeats: dict[str, np.ndarray] = {}
    controls: list[dict[str, Any]] = []
    control_repeats: dict[str, np.ndarray] = {}
    reuse: dict[str, dict[str, Any]] = {}

    for key, cell in sorted(cells.items()):
        table = build_cell_table(manifest, cell)
        horizon = table.horizon
        history_id = table.history_id
        mappings = {
            int(seed): _mapping_for(table, groups[f"{history_id}|{seed}"])
            for seed in readiness.FROZEN_ENCODER_SEEDS
        }
        unit_reuse_inputs: list[dict[str, Any]] = []
        for stratum in table.stratum_names:
            positions = table.positions(stratum)
            abnormal = positions[table.abnormal[positions]]
            normal = positions[~table.abnormal[positions]]
            for size in SIZES_A[horizon]:
                unit_id = f"{key}|{stratum}|{size}"
                census_unit = census_by_unit[unit_id]
                base_record: dict[str, Any] = {
                    "unit": unit_id,
                    "cell": key,
                    "history_id": history_id,
                    "horizon": horizon,
                    "stratum": stratum,
                    "size": int(size),
                    "variant": VARIANT_POOLED,
                    "status": census_unit["status"],
                    "reason": census_unit["reason"],
                    "n_abnormal_candidates": int(abnormal.size),
                    "n_normal_candidates": int(normal.size),
                    "period_counts_abnormal": _period_counts(table, abnormal),
                    "period_counts_normal": _period_counts(table, normal),
                    "composition": abnormal_composition(table, abnormal),
                    "candidate_abnormal": [table.ids[int(index)] for index in abnormal],
                    "candidate_normal": [table.ids[int(index)] for index in normal],
                    "period_matched": census_unit["period_variant"],
                    "seeds": {},
                }
                matched_census = census_unit["period_variant"]
                matched_record = {
                    "unit": unit_id,
                    "cell": key,
                    "history_id": history_id,
                    "horizon": horizon,
                    "stratum": stratum,
                    "size": int(size),
                    "variant": VARIANT_MATCHED,
                    "status": matched_census["status"],
                    "reason": (
                        ""
                        if matched_census["status"] == COMPUTED
                        else "the exact per-period quota cannot supply the arm size"
                    ),
                    "quota": matched_census["quota"],
                    "max_matched_size": matched_census["max_matched_size"],
                    "seeds": {},
                }
                if matched_census["status"] == COMPUTED:
                    matched_draws = draw_period_matched(
                        table,
                        stratum,
                        size,
                        repeats_ab,
                        {int(k2): int(value) for k2, value in matched_census["quota"].items()},
                    )
                    for seed in readiness.FROZEN_ENCODER_SEEDS:
                        diag = diagnostics[f"{history_id}|{seed}"]
                        values = arm_values(
                            draws=matched_draws,
                            matrix=groups[f"{history_id}|{seed}"].matrix,
                            sigma=fits[f"{history_id}|{seed}"].sigma,
                            diagnostics=diag,
                            mapping=mappings[int(seed)],
                            table=table,
                        )
                        summary = summarise_arm_values(values, size)
                        matched_record["seeds"][str(int(seed))] = {
                            "status": COMPUTED,
                            "sigma": float(fits[f"{history_id}|{seed}"].sigma),
                            "estimators": summary["estimators"],
                            "diagnostic_by_k": summary["diagnostic_by_k"],
                            "coverage": summary["coverage"],
                            "period_mix_tv": summary["period_mix_tv"],
                        }
                        prefix = f"ABM|{unit_id}|{int(seed)}"
                        for name, array in values.items():
                            repeats[f"{prefix}|{name}"] = np.asarray(array)
                        repeats[f"{prefix}|draws|abnormal"] = matched_draws.abnormal.astype(np.int16)
                        repeats[f"{prefix}|draws|normal_1"] = matched_draws.normal_1.astype(np.int16)
                        repeats[f"{prefix}|draws|normal_2"] = matched_draws.normal_2.astype(np.int16)
                        repeats[f"{prefix}|draws|normal_3"] = matched_draws.normal_3.astype(np.int16)
                units_a_matched.append(matched_record)
                if census_unit["status"] != COMPUTED:
                    units_a.append(base_record)
                    continue
                draws = draw_pooled(table, stratum, size, repeats_ab)
                distinct_abnormal: set[int] = set()
                distinct_normal: set[int] = set()
                for seed in readiness.FROZEN_ENCODER_SEEDS:
                    diag = diagnostics[f"{history_id}|{seed}"]
                    mapping = mappings[int(seed)]
                    values = arm_values(
                        draws=draws,
                        matrix=groups[f"{history_id}|{seed}"].matrix,
                        sigma=fits[f"{history_id}|{seed}"].sigma,
                        diagnostics=diag,
                        mapping=mapping,
                        table=table,
                    )
                    summary = summarise_arm_values(values, size)
                    base_record["seeds"][str(int(seed))] = {
                        "status": COMPUTED,
                        "sigma": float(fits[f"{history_id}|{seed}"].sigma),
                        "direction_sha256": sha256_object(
                            [float(value) for value in diag.direction]
                        ),
                        "sigma_u": float(diag.sigma_u),
                        "estimators": summary["estimators"],
                        "diagnostic_by_k": summary["diagnostic_by_k"],
                        "coverage": summary["coverage"],
                        "period_mix_tv": summary["period_mix_tv"],
                        "boundary_proximity_fraction": summary["boundary_proximity_fraction"],
                    }
                    prefix = f"AB|{unit_id}|{int(seed)}"
                    for name, array in values.items():
                        repeats[f"{prefix}|{name}"] = np.asarray(array)
                    repeats[f"{prefix}|draws|abnormal"] = draws.abnormal.astype(np.int16)
                    repeats[f"{prefix}|draws|normal_1"] = draws.normal_1.astype(np.int16)
                    repeats[f"{prefix}|draws|normal_2"] = draws.normal_2.astype(np.int16)
                    repeats[f"{prefix}|draws|normal_3"] = draws.normal_3.astype(np.int16)
                distinct_abnormal.update(draws.abnormal.ravel().tolist())
                distinct_normal.update(draws.normal_1.ravel().tolist())
                distinct_normal.update(draws.normal_2.ravel().tolist())
                distinct_normal.update(draws.normal_3.ravel().tolist())
                base_record["source_reuse"] = {
                    "distinct_abnormal_used": len(distinct_abnormal),
                    "distinct_normal_used": len(distinct_normal),
                    "possible_abnormal_draws": int(
                        math.comb(abnormal.size, size) if abnormal.size >= size else 0
                    ),
                    "possible_normal_draws": int(
                        math.comb(normal.size, 3 * size) if normal.size >= 3 * size else 0
                    ),
                }
                arms.append(
                    {
                        "unit": unit_id,
                        "variant": VARIANT_POOLED,
                        "size": int(size),
                        "repeat0": {
                            "abnormal": [table.ids[int(index)] for index in draws.abnormal[0]],
                            "normal_1": [table.ids[int(index)] for index in draws.normal_1[0]],
                            "normal_2": [table.ids[int(index)] for index in draws.normal_2[0]],
                            "normal_3": [table.ids[int(index)] for index in draws.normal_3[0]],
                        },
                    }
                )
                units_a.append(base_record)
                unit_reuse_inputs.append(base_record)

        reuse[key] = _source_reuse(unit_reuse_inputs, table)

        # --- branch C -----------------------------------------------------------
        for size in SIZES_C[horizon]:
            unit_id = f"{key}|{size}"
            census_unit = census_c_by_unit[unit_id]
            if census_unit["status"] != COMPUTED:
                for seed in readiness.FROZEN_ENCODER_SEEDS:
                    controls.append(
                        {
                            "unit": f"{unit_id}|{int(seed)}",
                            "cell": key,
                            "history_id": history_id,
                            "horizon": horizon,
                            "size": int(size),
                            "model_seed": int(seed),
                            "status": census_unit["status"],
                            "reason": census_unit["reason"],
                            "quota": census_unit["quota"],
                            "records": {},
                            "zero_identity": {},
                        }
                    )
                continue
            quota = {name: int(value) for name, value in census_unit["quota"].items()}
            for seed in readiness.FROZEN_ENCODER_SEEDS:
                draw = draw_controls(table, size, quota, int(seed), repeats_c)
                diag = diagnostics[f"{history_id}|{seed}"]
                scored = score_controls(
                    draw=draw,
                    matrix=groups[f"{history_id}|{seed}"].matrix,
                    sigma=fits[f"{history_id}|{seed}"].sigma,
                    diagnostics=diag,
                    mapping=mappings[int(seed)],
                )
                controls.append(scored["record"])
                prefix = f"C|{unit_id}|{int(seed)}"
                for name, array in scored["arrays"].items():
                    control_repeats[f"{prefix}|values|{name}"] = np.asarray(array)
                for name, array in scored["draws"].items():
                    control_repeats[f"{prefix}|draws|{name}"] = np.asarray(array)

    accounting = _accounting(
        census, units_a, units_a_matched, controls, repeats_ab, repeats_c, len(fit_custody)
    )
    endpoint_matrix = _aggregate_b(units_a)
    endpoint_matrix_matched = _aggregate_b(units_a_matched)
    aggregates = {
        "pooled": _aggregate_a(units_a),
        "period_matched": _aggregate_a(units_a_matched),
        "branch_c": _aggregate_c(controls),
    }
    products_after = verify_accepted_products()
    return {
        "task": TASK_ID,
        "status": "AUDIT-COMPLETE" if accounting["all_planned_cells_accounted"] else "AUDIT-INCOMPLETE",
        "spec": {key2: value for key2, value in spec.items() if key2 != "payload"},
        "amendments": amendments,
        "census": census,
        "census_check": census_check,
        "support": _support_record(census),
        "lineage": {
            key2: value
            for key2, value in lineage.items()
            if key2 not in {"manifest", "audit_record", "contract", "custody", "identity_plan"}
        },
        "prior_products": products_before,
        "prior_products_after": products_after,
        "cache": {
            "cache_root": cache["cache_root"],
            "admission": cache["admission"],
            "n_groups": cache["n_groups"],
            "n_rows": cache["n_rows"],
            "n_cache_files": cache["n_cache_files"],
            "cache_file_digest_digest": cache["cache_file_digest_digest"],
        },
        "fit_custody": fit_custody,
        "units_a": units_a,
        "units_a_matched": units_a_matched,
        "controls": controls,
        "endpoint_matrix": {
            "pooled": endpoint_matrix,
            "period_matched": endpoint_matrix_matched,
        },
        "aggregates": aggregates,
        "arms": arms,
        "reuse": reuse,
        "accounting": accounting,
        "repeats": repeats,
        "control_repeats": control_repeats,
        "limitations": list(LIMITATIONS),
        "elapsed_s": time.time() - started,
    }


def _accounting(
    census: Mapping[str, Any],
    units_a: Sequence[Mapping[str, Any]],
    units_a_matched: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    repeats_ab: int,
    repeats_c: int,
    n_fits: int,
) -> dict[str, Any]:
    seeds = len(readiness.FROZEN_ENCODER_SEEDS)
    planned_a = len(census["branch_a"]["units"])
    computed_a = sum(1 for unit in census["branch_a"]["units"] if unit["status"] == COMPUTED)
    computed_matched = sum(
        1 for unit in census["branch_a"]["units"] if unit["period_variant"]["status"] == COMPUTED
    )
    planned_estimator_a = planned_a * seeds * len(ESTIMATORS)
    computed_estimator_a = 0
    for unit in units_a:
        for seed_record in unit["seeds"].values():
            for name in ESTIMATORS:
                if seed_record["estimators"].get(name, {}).get("status") == COMPUTED:
                    computed_estimator_a += 1
    planned_estimator_matched = planned_a * seeds * len(ESTIMATORS)
    computed_estimator_matched = 0
    for unit in units_a_matched:
        for seed_record in unit["seeds"].values():
            for name in ESTIMATORS:
                if seed_record["estimators"].get(name, {}).get("status") == COMPUTED:
                    computed_estimator_matched += 1

    planned_c_units = len(census["branch_c"]["units"]) * seeds
    comparison_records = len(DELTA_GRID) + (len(DELTA_GRID) - 1) * (len(PERTURB_MODES) - 1)
    planned_c_records = planned_c_units * (comparison_records + 1)
    planned_estimator_c = planned_c_units * (
        comparison_records * len(CONTROL_ESTIMATORS) + 10
    )
    computed_c_units = sum(1 for record in controls if record.get("status") == COMPUTED)
    published_c_records = sum(
        len(record.get("records", {})) + (1 if record.get("zero_identity") else 0)
        for record in controls
    )
    computed_c_records = 0
    computed_estimator_c = 0
    for record in controls:
        if record.get("status") != COMPUTED:
            continue
        for item in record["records"].values():
            if item.get("status") != COMPUTED:
                continue
            computed_c_records += 1
            for name in CONTROL_ESTIMATORS:
                if item["estimators"].get(name, {}).get("status") == COMPUTED:
                    computed_estimator_c += 1
        identity = record.get("zero_identity", {})
        if identity:
            computed_c_records += 1
            computed_estimator_c += sum(
                1
                for name in (
                    "E1_biased",
                    "E1_unbiased",
                    "D_LOC",
                    "D_SPREAD",
                    "D_RADIAL",
                    "D_TANGENTIAL",
                )
                if isinstance(identity.get(name), (int, float))
            ) + sum(1 for value in identity.get("E2", {}).values() if isinstance(value, int)) + int(
                "reassigned" in identity
            )
    all_accounted = (
        computed_a + sum(1 for unit in census["branch_a"]["units"] if unit["status"] != COMPUTED)
        == planned_a
        and computed_matched
        + sum(
            1
            for unit in census["branch_a"]["units"]
            if unit["period_variant"]["status"] != COMPUTED
        )
        == planned_a
        and len(units_a) == planned_a
        and len(units_a_matched) == planned_a
        and published_c_records == planned_c_records
        and computed_c_records <= planned_c_records
    )
    return {
        "all_planned_cells_accounted": bool(all_accounted),
        "branch_a": {
            "planned_units": planned_a,
            "computed_units": computed_a,
            "abstained_units": planned_a - computed_a,
            "planned_unit_seed_records": planned_a * seeds,
            "planned_estimator_cells": planned_estimator_a,
            "computed_estimator_cells": computed_estimator_a,
            "abstained_estimator_cells": planned_estimator_a - computed_estimator_a,
            "unit_status_counts": census["branch_a"]["unit_status_counts"],
        },
        "branch_a_period_matched": {
            "planned_units": planned_a,
            "computed_units": computed_matched,
            "abstained_units": planned_a - computed_matched,
            "planned_estimator_cells": planned_estimator_matched,
            "computed_estimator_cells": computed_estimator_matched,
            "abstained_estimator_cells": planned_estimator_matched - computed_estimator_matched,
        },
        "branch_c": {
            "planned_units": planned_c_units,
            "computed_units": computed_c_units,
            "planned_records": planned_c_records,
            "published_records": published_c_records,
            "computed_records": computed_c_records,
            "abstained_records": max(planned_c_records - computed_c_records, 0),
            "planned_estimator_cells": planned_estimator_c,
            "computed_estimator_cells": computed_estimator_c,
            "abstained_estimator_cells": max(planned_estimator_c - computed_estimator_c, 0),
        },
        "repeats_a": int(repeats_ab),
        "repeats_c": int(repeats_c),
        "n_fits": int(n_fits),
    }
