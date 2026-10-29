"""Sprint 24 Task 7-M13 controlled dose-response diagnostic on the existing cache.

Cache-only, read-only exploratory audit of the already admitted S24-7-M11
embeddings (24,840 vectors, 18 ``(history, seed)`` groups). The module never
loads a model, never runs a forward pass, never refits an encoder or an
anomaly bank, never regenerates the corpus and never rewrites an M11 or M12
product.

The frozen design is recorded in
``docs/sprint24-controlled-dose-response-specification-a01.json`` (artifact
``S24-CONTROLLED-DOSE-RESPONSE-SPEC-A01``) and its digest is pinned here. The
specification was written after a metadata-only feasibility census and before
any M13 outcome was computed; later deviations must be separate timestamped
amendment records and are never edited back into the specification bytes.

Design summary (all rules frozen before outcomes):

- Cells: the 12 accepted ``history x horizon`` cells of S24-7-M7/M11/M12. The
  source periods of a cell are read from the accepted M9 corpus audit; the
  low-burden group is the audit's baseline period set and the elevated group
  is the contrast period set of its ``BASELINE-VS-ELEVATED`` rows.
- Designs: ``POOLED-CELL-PERIODS`` draws from every period of the cell, so a
  mixture spans periods and is therefore NOT a calendar window;
  ``LOW-PERIOD-SOURCE`` draws from the low-burden periods only and is the
  period-composition-controlled sensitivity.
- Sizes: two supported equal per-arm file counts per horizon
  (``week {40, 60}``, ``month {160, 250}``), fixed from the metadata census.
- Doses: abnormal-file proportion ``POOLED-CELL-PERIODS {0.00, 0.05, 0.10,
  0.20, 0.30, 0.40}`` and ``LOW-PERIOD-SOURCE {0.00, 0.05, 0.10, 0.15}``.
  ``n_abnormal = floor(p * N + 0.5)`` exactly; every realized count is
  reported.
- Frame: the robot|program context quota is a capped largest-remainder
  apportionment of ``N`` over per-stratum capacity ``floor(ordinary/2)``, so
  the fixed baseline arm and every dose arm carry exactly the same
  robot|program mix and the baseline is disjoint from every dose arm.
- Abnormal sequence: one deterministic, family-deficit-ordered sequence per
  cell x design; the abnormal files of dose ``p`` are its first
  ``n_abnormal(p)`` entries, so the family composition of every prefix is as
  proportional to the pool's family availability as the support allows and
  families are never silently dropped. Sequence files are also constrained to
  each stratum's context headroom.
- Sampling: per repeat, files are drawn without replacement inside each
  ``(stratum, family)`` abnormal group and each ordinary stratum group; the
  fixed baseline arm is one deterministic realization of the ordinary quota
  and every dose draw excludes its identities. Arms are therefore disjoint
  and duplicate-free within a draw.
- Measurement: accepted unbiased E1 (squared MMD at the reference bandwidth of
  the accepted causal-reference fit) and E2 in exact occupancy quanta at
  ``K in {4, 8, 16}`` against the accepted causal-reference prototypes, for
  all three frozen encoder seeds. Prototype and bandwidth fitting never sees
  a label, a family, a context field or an evaluation mixture.
- Baseline: every dose is compared with the same fixed baseline distribution,
  and dose 0 is the baseline-vs-baseline draw set with identical size and
  sampling semantics. Repeats are resampling variability, never confidence
  limits; encoder seeds are run replicates, never independent data.

Labels are used only to construct and diagnose the mixtures. No p-value,
success gate, performance, detection-accuracy, business-density, causal or
generalisation claim is produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from representation import sprint24_controlled_inference_readiness as readiness
from representation import sprint24_e2_cache_diagnostic as m12
from representation import sprint24_weekly_monthly_endpoints as endpoints
from representation.sprint24_context_audit import sha256_object

#: Task identifier for every M13 record.
TASK_ID = "S24-7-M13"

#: Frozen pre-outcome specification.
SPEC_ID = "S24-CONTROLLED-DOSE-RESPONSE-SPEC-A01"
SPEC_PATH = "docs/sprint24-controlled-dose-response-specification-a01.json"

#: Byte digest of the frozen specification (never overwritten; amendments are separate).
SPEC_SHA256 = "8a3681fc84e6187c5979f95172ab5e87773251efcca4f650c9cf7fbe2338ebe4"

#: Repository root, used to resolve the frozen specification and prior products.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Monte Carlo repeats per comparison (resampling variability, never evidence).
REPEATS = 400

#: Supported equal per-arm sizes per horizon (frozen from the metadata census).
SIZES: dict[str, tuple[int, ...]] = {"week": (40, 60), "month": (160, 250)}

#: Source-pool designs.
DESIGN_POOLED = "POOLED-CELL-PERIODS"
DESIGN_LOW = "LOW-PERIOD-SOURCE"
DESIGNS: tuple[str, ...] = (DESIGN_POOLED, DESIGN_LOW)

#: Frozen burden grids (requested abnormal-file proportion).
DOSE_GRID: dict[str, tuple[float, ...]] = {
    DESIGN_POOLED: (0.00, 0.05, 0.10, 0.20, 0.30, 0.40),
    DESIGN_LOW: (0.00, 0.05, 0.10, 0.15),
}

#: Inherited cluster counts for the occupancy endpoint.
CLUSTER_COUNTS: tuple[int, ...] = m12.CLUSTER_COUNTS

#: Estimator identifiers published for every dose record.
ESTIMATORS: tuple[str, ...] = ("E1", "E2_K4", "E2_K8", "E2_K16")

#: Record statuses and closed abstention reasons.
COMPUTED = "COMPUTED"
ABSTAIN_EMPTY = "ABSTAIN-EMPTY-POOL"
ABSTAIN_ORDINARY_CAPACITY = "ABSTAIN-ORDINARY-CAPACITY"
ABSTAIN_CONTEXT_QUOTA = "ABSTAIN-CONTEXT-QUOTA"
ABSTAIN_INSUFFICIENT_ABNORMAL = "ABSTAIN-INSUFFICIENT-ABNORMAL"
ABSTAIN_ABNORMAL_HEADROOM = "ABSTAIN-ABNORMAL-HEADROOM"
ABSTAIN_REASONS: tuple[str, ...] = (
    ABSTAIN_EMPTY,
    ABSTAIN_ORDINARY_CAPACITY,
    ABSTAIN_CONTEXT_QUOTA,
    ABSTAIN_INSUFFICIENT_ABNORMAL,
    ABSTAIN_ABNORMAL_HEADROOM,
)

#: Comparison labels.
BASELINE_VS_BASELINE = "BASELINE-VS-BASELINE"
DOSE_VS_BASELINE = "DOSE-VS-FIXED-BASELINE"

#: Hash prefixes keep baseline, sequence and dose draws independent.
BASELINE_PREFIX = "S24-7-M13|baseline"
SEQUENCE_PREFIX = "S24-7-M13|sequence"
DRAW_PREFIX = "S24-7-M13|draw"

#: Accepted prior-product digests that this audit must leave byte-identical.
ACCEPTED_PRODUCT_DIGESTS: dict[str, str] = {
    "docs/sprint24-e2-cache-diagnostic-audit.md": (
        "423932b36052760b96000fc19d883bca9b81b3676f32909cbf40704e0c7bcad9"
    ),
    "docs/sprint24-e2-cache-diagnostic-specification-a01.json": (
        "cde6021ff086398cf690d76a7eb78d656274721dc55621c8e8012fae5c2d47fc"
    ),
    "docs/sprint24-controlled-inference-execution.md": (
        "53795c0634019d4c750ce0426bdab73fb3739ee0691497899a23161774d960c2"
    ),
    "src/representation/sprint24_e2_cache_diagnostic.py": (
        "ae7706f0eb0369d920c5d5adfdd144f2938007df0dfb9a9043e7dd12d906a128"
    ),
    "experiments/sprint24_task7m12_e2_cache_diagnostic.py": (
        "9655727019f63ac702feb5fbc9ee4bf70039139fada8eb79f29c115f8357b01b"
    ),
    "tests/representation/test_sprint24_e2_cache_diagnostic.py": (
        "12b06aec5f8ad68d2d5f1ca9250ef5729c32d942be68b14fab9bd18947eed56f"
    ),
    "src/representation/sprint24_controlled_inference_execution.py": (
        "47d845c86bbb4c250cba6a4a885565283083a2c53b4077a38e5f3da82d159f7b"
    ),
    "experiments/sprint24_task7m11_controlled_inference_execution.py": (
        "e67c17af74d55c2a4f2034061fce96f81cc603be4d1ea2ec513bfc29806c27f1"
    ),
    "tests/representation/test_sprint24_controlled_inference_execution.py": (
        "b4c5734da35e0cc7db0e792a8f4a93b52c67f120c5dd71e72d4ae9ad459331b7"
    ),
    "artifacts/sprint-24/evidence/s24-7-m11-a04/movement.json": (
        "d84747449f809eccd498806493ec570608f1b3689931395468b486e1e29515e3"
    ),
    "artifacts/sprint-24/evidence/s24-7-m11-a04/run.json": (
        "87a71f8cff50daedded969d2f50930a646efdda95953f8750baafe8010636f84"
    ),
}

#: Admitted-cache digest-of-digests recorded by the accepted M11/M12 lineage.
CACHE_DIGEST_DIGEST = "ae7aa42d7661d95d9d292400f53f4ec76128fd9618acc75cc6c6fcf581aa02b3"

#: Cumulative observed forward passes before this audit (M11 accepted accounting).
CUMULATIVE_FORWARDS = 77653

#: Default amendment directory (separate from the frozen specification bytes).
AMENDMENT_DIR = (
    REPO_ROOT / "artifacts" / "sprint-24" / "evidence" / "s24-7-m13-a01" / "amendments"
)

#: Diagnostic resolution floor for abnormal-only composition comparisons.
COMPOSITION_TV_QUANTUM_FLOOR = 0.05

#: Diagnostic bin edges inherited unchanged from the accepted M12 audit.
SEVERITY_BINS = m12.SEVERITY_BINS
FRACTION_BINS = m12.FRACTION_BINS
ORDINARY = m12.ORDINARY


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
    """Load separately preserved amendment records bound to the frozen spec.

    Amendments are never folded into the original specification bytes. Each
    record must declare the frozen specification digest it amends and is
    recorded with its own digest and timing.
    """
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
    """Verify that every accepted M11/M12 product still matches its digest."""
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
        "spec": spec,
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


# --- cells, source periods and pools -------------------------------------------


def _period_number(identity: str) -> int:
    token = str(identity).rsplit("|", 1)[-1]
    _require(
        token[:1] in {"W", "M"},
        f"REFUSED-IDENTITY: malformed period identity {identity}",
    )
    return int(token[1:])


def build_cells(audit_record: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    """Return the 12 accepted cells with their low-burden and elevated periods.

    Cell membership comes only from the accepted M9 audit pair rows; no period
    is added, dropped or re-selected.
    """
    pair_rows = audit_record.get("pair_rows")
    _require(
        isinstance(pair_rows, list) and bool(pair_rows),
        "REFUSED-IDENTITY: audit has no pair rows",
    )
    cells: dict[str, dict[str, Any]] = {}
    for row in pair_rows:  # type: ignore[union-attr]
        if not isinstance(row, Mapping):
            continue
        history_id = str(row["history_id"])
        horizon = str(row["horizon"])
        key = f"{history_id}|{horizon}"
        cell = cells.setdefault(
            key,
            {
                "cell": key,
                "history_id": history_id,
                "horizon": horizon,
                "periods": set(),
                "low_periods": set(),
                "elevated_periods": set(),
                "comparisons": [],
            },
        )
        for identity in row.get("baseline_period_identities", []):  # type: ignore[union-attr]
            period = _period_number(str(identity))
            cell["low_periods"].add(period)
            cell["periods"].add(period)
        if str(row.get("comparison_type")) == "BASELINE-VS-ELEVATED":
            for identity in row.get("contrast_period_identities", []):  # type: ignore[union-attr]
                period = _period_number(str(identity))
                cell["elevated_periods"].add(period)
                cell["periods"].add(period)
        cell["comparisons"].append(str(row["comparison_identity"]))
    for cell in cells.values():
        _require(
            bool(cell["low_periods"]),
            f"REFUSED-IDENTITY: cell {cell['cell']} has no low-burden period",
        )
        _require(
            bool(cell["elevated_periods"]),
            f"REFUSED-IDENTITY: cell {cell['cell']} has no elevated period",
        )
        cell["comparisons"] = sorted(set(cell["comparisons"]))
    return dict(sorted(cells.items()))


def horizon_period_of(entry: Mapping[str, object], horizon: str) -> int | None:
    """Return the week offset or month index of one manifest file."""
    value = entry.get("week_offset") if horizon == "week" else entry.get("month_index")
    return int(value) if isinstance(value, (int, float)) else None  # type: ignore[arg-type]


def _family_of(entry: Mapping[str, Any]) -> str:
    annotation = entry.get("annotation")
    family = annotation.get("family") if isinstance(annotation, Mapping) else None
    return str(family) if isinstance(family, str) else ORDINARY


@dataclass(slots=True)
class PoolTable:
    """Deterministic source pool of one ``cell x design`` with frozen indices."""

    ids: list[str]
    strata: np.ndarray
    stratum_names: list[str]
    families: np.ndarray
    family_names: list[str]
    abnormal: np.ndarray
    periods: np.ndarray
    period_numbers: list[int]
    low_periods: frozenset[int]

    def stratum_of(self, index: int) -> str:
        return self.stratum_names[int(self.strata[index])]

    def family_of(self, index: int) -> str:
        return self.family_names[int(self.families[index])]


def build_pool(
    manifest: Mapping[str, object],
    cell: Mapping[str, Any],
    design: str,
) -> PoolTable:
    """Build the deterministic source pool of one cell and design.

    The pool is every manifest file of the cell's history whose week/month is
    in the design's period set; the causal ``REFERENCE`` arm is never a source.
    ``LOW-PERIOD-SOURCE`` keeps only audit-verified low-burden periods.
    """
    _require(design in DESIGNS, f"REFUSED-DESIGN: unknown design {design}")
    files = manifest.get("files")
    _require(
        isinstance(files, list), "REFUSED-IDENTITY: corpus manifest has no file table"
    )
    horizon = str(cell["horizon"])
    history_id = str(cell["history_id"])
    allowed = (
        set(int(value) for value in cell["periods"])
        if design == DESIGN_POOLED
        else set(int(value) for value in cell["low_periods"])
    )
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
    _require(bool(selected), f"REFUSED-EMPTY-POOL: {cell['cell']} {design}")

    stratum_lookup: dict[str, int] = {}
    stratum_names: list[str] = []
    family_lookup: dict[str, int] = {}
    family_names: list[str] = []
    strata: list[int] = []
    families: list[int] = []
    abnormal: list[bool] = []
    periods: list[int] = []
    period_numbers = sorted(
        {int(horizon_period_of(entry, horizon) or 0) for entry in selected}
    )
    period_lookup = {number: index for index, number in enumerate(period_numbers)}
    for entry in selected:
        name = m12.stratum_id(entry)
        if name not in stratum_lookup:
            stratum_lookup[name] = len(stratum_names)
            stratum_names.append(name)
        family = _family_of(entry)
        if family not in family_lookup:
            family_lookup[family] = len(family_names)
            family_names.append(family)
        strata.append(stratum_lookup[name])
        families.append(family_lookup[family])
        abnormal.append(str(entry.get("file_label")) == "abnormal")
        periods.append(period_lookup[int(horizon_period_of(entry, horizon) or 0)])
    return PoolTable(
        ids=[str(entry["file_id"]) for entry in selected],
        strata=np.asarray(strata, dtype=np.int64),
        stratum_names=stratum_names,
        families=np.asarray(families, dtype=np.int64),
        family_names=family_names,
        abnormal=np.asarray(abnormal, dtype=bool),
        periods=np.asarray(periods, dtype=np.int64),
        period_numbers=period_numbers,
        low_periods=frozenset(int(value) for value in cell["low_periods"]),
    )


# --- allocation and dose construction ------------------------------------------


def apportion(
    weights: Mapping[Any, int], caps: Mapping[Any, int], total: int
) -> dict[Any, int]:
    """Capped largest-remainder apportionment of ``total`` over ``weights``.

    Returns ``{}`` when the caps cannot supply ``total``. Deterministic: keys
    are processed in sorted order and the residual is granted by descending
    fractional remainder. Support-only, never outcome-dependent.
    """
    keys = sorted(weights)
    if not keys or total <= 0:
        return {}
    if sum(min(int(caps[key]), int(weights[key])) for key in keys) < total:
        return {}
    weight = float(sum(int(weights[key]) for key in keys))
    allocation = {
        key: min(
            int(math.floor(total * int(weights[key]) / weight)), int(caps[key])
        )
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


def abnormal_count(dose: float, size: int) -> int:
    """Frozen count rounding: ``floor(p * N + 0.5)``."""
    return int(math.floor(float(dose) * size + 0.5))


@dataclass(slots=True)
class Frame:
    """Frozen composition frame of one ``cell x size x design`` unit."""

    status: str
    reason: str
    size: int
    quota: dict[str, int]
    ordinary_by_stratum: dict[str, int]
    capacity: dict[str, int]
    abnormal_by_stratum: dict[str, int]
    sequence: list[int]
    n_max_abnormal: int
    pool: PoolTable


def build_frame(pool: PoolTable, size: int, n_max_abnormal: int) -> Frame:
    """Build the context quota and the abnormal sequence of one unit.

    Feasibility is decided from metadata only:

    1. each stratum may contribute at most ``floor(ordinary / 2)`` files, so
       the fixed baseline arm and the zero-abnormal draw remain disjoint;
    2. the abnormal sequence respects the same per-stratum headroom.
    """
    ordinary_counts: dict[str, int] = {}
    abnormal_counts: dict[str, int] = {}
    for index in range(len(pool.ids)):
        name = pool.stratum_of(index)
        if bool(pool.abnormal[index]):
            abnormal_counts[name] = abnormal_counts.get(name, 0) + 1
        else:
            ordinary_counts[name] = ordinary_counts.get(name, 0) + 1
    capacity = {name: count // 2 for name, count in ordinary_counts.items()}
    if sum(capacity.values()) < size:
        return Frame(
            ABSTAIN_ORDINARY_CAPACITY,
            "the low-burden source pool cannot supply two disjoint arms of this size",
            size,
            {},
            ordinary_counts,
            capacity,
            abnormal_counts,
            [],
            n_max_abnormal,
            pool,
        )
    quota = apportion(capacity, capacity, size)
    if not quota:
        return Frame(
            ABSTAIN_CONTEXT_QUOTA,
            "no capped largest-remainder context quota exists for this size",
            size,
            {},
            ordinary_counts,
            capacity,
            abnormal_counts,
            [],
            n_max_abnormal,
            pool,
        )
    total_abnormal = int(pool.abnormal.sum())
    if n_max_abnormal > total_abnormal:
        return Frame(
            ABSTAIN_INSUFFICIENT_ABNORMAL,
            f"max dose needs {n_max_abnormal} abnormal files, pool holds {total_abnormal}",
            size,
            quota,
            ordinary_counts,
            capacity,
            abnormal_counts,
            [],
            n_max_abnormal,
            pool,
        )
    sequence = build_abnormal_sequence(pool, n_max_abnormal, quota)
    if sequence is None:
        return Frame(
            ABSTAIN_ABNORMAL_HEADROOM,
            "the abnormal pool cannot fill the max dose inside the context headroom",
            size,
            quota,
            ordinary_counts,
            capacity,
            abnormal_counts,
            [],
            n_max_abnormal,
            pool,
        )
    return Frame(
        COMPUTED,
        "",
        size,
        quota,
        ordinary_counts,
        capacity,
        abnormal_counts,
        sequence,
        n_max_abnormal,
        pool,
    )


def build_abnormal_sequence(
    pool: PoolTable,
    n_max: int,
    quota: Mapping[str, int],
) -> list[int] | None:
    """Return the family-deficit-ordered abnormal sequence of one unit.

    At every step the family with the largest deficit against its proportional
    availability is served next, in a frozen hash order inside the family, and
    only into a stratum whose context headroom still allows it. Motivated only
    by the metadata census; never by an outcome.
    """
    if n_max <= 0:
        return []
    positions = np.flatnonzero(pool.abnormal)
    by_family: dict[int, list[int]] = {}
    for index in positions.tolist():
        by_family.setdefault(int(pool.families[index]), []).append(index)
    for family in by_family:
        by_family[family].sort(
            key=lambda index: hashlib.sha256(
                f"{SEQUENCE_PREFIX}|{pool.ids[index]}".encode("utf-8")
            ).hexdigest()
        )
    total = len(positions)
    rate = {family: len(values) / total for family, values in by_family.items()}
    used = {family: 0 for family in by_family}
    cursor = {family: 0 for family in by_family}
    headroom: dict[str, int] = {}
    for family_values in by_family.values():
        for index in family_values:
            name = pool.stratum_of(index)
            headroom[name] = headroom.get(name, 0) + 1
    for name in list(headroom):
        headroom[name] = min(headroom[name], int(quota.get(name, 0)))
    used_stratum: dict[str, int] = {}
    sequence: list[int] = []
    for step in range(n_max):
        order = sorted(
            by_family,
            key=lambda family: (
                -(rate[family] * (step + 1) - used[family]),
                pool.family_names[family],
            ),
        )
        chosen: int | None = None
        for family in order:
            values = by_family[family]
            while cursor[family] < len(values):
                index = values[cursor[family]]
                name = pool.stratum_of(index)
                if used_stratum.get(name, 0) < headroom.get(name, 0):
                    chosen = index
                    cursor[family] += 1
                    break
                cursor[family] += 1
            if chosen is not None:
                used[family] += 1
                name = pool.stratum_of(chosen)
                used_stratum[name] = used_stratum.get(name, 0) + 1
                sequence.append(chosen)
                break
        if chosen is None:
            return None
    return sequence


def dose_plan(frame: Frame, dose: float) -> dict[str, Any]:
    """Return the exact abnormal/ordinary allocation of one dose.

    Counts are deterministic and shared by every repeat; only the file
    identities inside each group vary.
    """
    size = frame.size
    n_abnormal = abnormal_count(dose, size)
    prefix = frame.sequence[:n_abnormal]
    abnormal_by_pair: dict[tuple[str, str], int] = {}
    abnormal_by_stratum: dict[str, int] = {}
    for index in prefix:
        pair = (frame.pool.stratum_of(index), frame.pool.family_of(index))
        abnormal_by_pair[pair] = abnormal_by_pair.get(pair, 0) + 1
        abnormal_by_stratum[pair[0]] = abnormal_by_stratum.get(pair[0], 0) + 1
    ordinary_by_stratum: dict[str, int] = {}
    for name, quota in frame.quota.items():
        remainder = int(quota) - abnormal_by_stratum.get(name, 0)
        _require(
            remainder >= 0,
            f"REFUSED-INTERNAL: dose plan exceeds context quota in {name}",
        )
        ordinary_by_stratum[name] = remainder
    _require(
        sum(abnormal_by_pair.values()) + sum(ordinary_by_stratum.values()) == size,
        "REFUSED-INTERNAL: dose plan does not sum to the arm size",
    )
    return {
        "dose": float(dose),
        "n_abnormal": n_abnormal,
        "n_ordinary": size - n_abnormal,
        "abnormal_by_pair": abnormal_by_pair,
        "abnormal_by_stratum": abnormal_by_stratum,
        "ordinary_by_stratum": ordinary_by_stratum,
    }


def baseline_indices(frame: Frame, *, cell: str, design: str) -> np.ndarray:
    """Return the fixed baseline arm as deterministic hash-ordered indices.

    The baseline arm is all-ordinary and carries exactly the frame's context
    quota, so the baseline distribution is fixed and every dose arm can be
    disjoint from it.
    """
    selected: list[int] = []
    by_stratum: dict[str, list[int]] = {}
    for index in range(len(frame.pool.ids)):
        if bool(frame.pool.abnormal[index]):
            continue
        by_stratum.setdefault(frame.pool.stratum_of(index), []).append(index)
    for name in sorted(frame.quota):
        count = int(frame.quota[name])
        values = by_stratum.get(name, [])
        _require(
            len(values) >= count,
            f"REFUSED-INTERNAL: baseline arm exceeds ordinary supply in {name}",
        )
        ordered = sorted(
            values,
            key=lambda index: hashlib.sha256(
                f"{BASELINE_PREFIX}|{cell}|{design}|{frame.size}|{name}|"
                f"{frame.pool.ids[index]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:count])
    indices = np.asarray(selected, dtype=np.int64)
    _require(indices.size == frame.size, "REFUSED-INTERNAL: baseline arm size mismatch")
    _require(
        np.unique(indices).size == indices.size,
        "REFUSED-INTERNAL: duplicate baseline file",
    )
    return indices


def _sample_without_replacement(
    candidates: np.ndarray,
    count: int,
    repeats: int,
    parts: Sequence[object],
) -> np.ndarray:
    """Draw ``count`` distinct candidates per repeat from ``candidates``."""
    if count <= 0:
        return np.empty((repeats, 0), dtype=np.int64)
    _require(
        candidates.size >= count,
        "REFUSED-INTERNAL: sampling group smaller than its allocation",
    )
    rng = np.random.default_rng(_draw_seed(parts))
    order = np.argsort(rng.random((repeats, candidates.size)), axis=1, kind="stable")
    return np.asarray(candidates[order[:, :count]], dtype=np.int64)


def draw_dose_arms(
    frame: Frame,
    plan: Mapping[str, Any],
    baseline: np.ndarray,
    *,
    cell: str,
    design: str,
    repeats: int,
) -> np.ndarray:
    """Return the ``(repeats, size)`` dose-arm pool indices.

    Files are drawn without replacement inside each ``(stratum, family)``
    abnormal group and each ordinary stratum group, and never reuse a fixed
    baseline identity. The construction is label-blind given the frozen plan.
    """
    pool = frame.pool
    baseline_set = set(int(value) for value in baseline.tolist())
    pair_positions: dict[tuple[str, str], list[int]] = {}
    ordinary_positions: dict[str, list[int]] = {}
    for index in range(len(pool.ids)):
        name = pool.stratum_of(index)
        if bool(pool.abnormal[index]):
            pair_positions.setdefault((name, pool.family_of(index)), []).append(index)
        elif index not in baseline_set:
            ordinary_positions.setdefault(name, []).append(index)
    chunks: list[np.ndarray] = []
    base_parts = [
        DRAW_PREFIX,
        cell,
        frame.size,
        design,
        f"{float(plan['dose']):.4f}",
    ]
    for pair, count in sorted(plan["abnormal_by_pair"].items()):
        candidates = np.asarray(sorted(pair_positions.get(pair, [])), dtype=np.int64)
        chunks.append(
            _sample_without_replacement(
                candidates,
                int(count),
                repeats,
                base_parts + ["abnormal", pair[0], pair[1]],
            )
        )
    for name, count in sorted(plan["ordinary_by_stratum"].items()):
        candidates = np.asarray(sorted(ordinary_positions.get(name, [])), dtype=np.int64)
        chunks.append(
            _sample_without_replacement(
                candidates,
                int(count),
                repeats,
                base_parts + ["ordinary", name],
            )
        )
    matrix = np.concatenate(chunks, axis=1)
    _require(
        matrix.shape == (repeats, frame.size),
        "REFUSED-INTERNAL: dose arm size mismatch",
    )
    sorted_rows = np.sort(matrix, axis=1)
    _require(
        bool(np.all(np.diff(sorted_rows, axis=1) > 0)),
        "REFUSED-INTERNAL: duplicate file identity inside a draw",
    )
    _require(
        not bool(np.isin(sorted_rows, baseline).any()),
        "REFUSED-INTERNAL: baseline and comparison arms are not disjoint",
    )
    repeats_count, size = sorted_rows.shape
    offsets = np.arange(repeats_count, dtype=np.int64)[:, None]
    stratum_codes = pool.strata[sorted_rows]
    flat = (stratum_codes + offsets * len(pool.stratum_names)).reshape(-1)
    counts = np.bincount(
        flat, minlength=repeats_count * len(pool.stratum_names)
    ).reshape(repeats_count, len(pool.stratum_names))
    expected = np.asarray(
        [int(frame.quota.get(name, 0)) for name in pool.stratum_names], dtype=np.int64
    )
    _require(
        bool(np.all(counts == expected[None, :])),
        "REFUSED-INTERNAL: realized context counts differ from the frozen quota",
    )
    family_codes = pool.families[sorted_rows]
    family_flat = (family_codes + offsets * len(pool.family_names)).reshape(-1)
    family_counts = np.bincount(
        family_flat, minlength=repeats_count * len(pool.family_names)
    ).reshape(repeats_count, len(pool.family_names))
    expected_family = np.zeros(len(pool.family_names), dtype=np.int64)
    for index in frame.sequence[: int(plan["n_abnormal"])]:
        expected_family[int(pool.families[index])] += 1
    expected_family[pool.family_names.index(ORDINARY)] += int(plan["n_ordinary"])
    _require(
        bool(np.all(family_counts == expected_family[None, :])),
        "REFUSED-INTERNAL: realized abnormal-family counts differ from the frozen plan",
    )
    return sorted_rows


# --- measurement ---------------------------------------------------------------


def e1_repeats(
    group: Any,
    baseline_positions: np.ndarray,
    dose_positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Accepted unbiased E1 for every repeat (squared MMD at ``sigma``).

    Uses the accepted public estimator unchanged, so the value is the same
    quantity the M11/M12 endpoints report.
    """
    baseline = group.matrix[baseline_positions]
    values = np.empty(dose_positions.shape[0], dtype=np.float64)
    for repeat in range(dose_positions.shape[0]):
        values[repeat] = endpoints.mmd2_unbiased(
            group.matrix[dose_positions[repeat]], baseline, sigma
        )
    return values


def e2_quanta_repeats(
    labels: np.ndarray,
    baseline_positions: np.ndarray,
    dose_positions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact E2 occupancy quanta (``E2 = m / size``) for every repeat."""
    return m12.e2_quanta(
        labels,
        np.asarray(baseline_positions, dtype=np.int64)[None, :],
        dose_positions,
        k,
    )


def float_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarise a float repeat vector; quantiles are resampling variability."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"n": 0, "status": ABSTAIN_EMPTY}
    return {
        "n": int(array.size),
        "status": COMPUTED,
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "fraction_strictly_positive": float(np.mean(array > 0.0)),
        "fraction_zero": float(np.mean(array == 0.0)),
        "fraction_strictly_negative": float(np.mean(array < 0.0)),
    }


def quanta_summary(quanta: np.ndarray, size: int) -> dict[str, Any]:
    """Summarise exact integer occupancy quanta at one arm size."""
    return m12.summary_of_quanta(np.asarray(quanta, dtype=np.int64), int(size))


def movement_summary(
    values: np.ndarray,
    null_values: np.ndarray,
    *,
    size: int | None,
    exact_quanta: bool,
) -> dict[str, Any]:
    """Describe a dose distribution against the baseline-vs-baseline band.

    With ``exact_quanta`` the inputs are integer numerators compared exactly in
    quanta and reported in E2 units (``quanta / size``); otherwise the inputs
    are already in estimator units.
    """
    if values.size == 0 or null_values.size == 0:
        return {"status": ABSTAIN_EMPTY}
    scale = float(size) if exact_quanta and size else 1.0
    null_q95_native = float(np.quantile(null_values, 0.95))
    null_median_native = float(np.median(null_values))
    median_native = float(np.median(values))
    record: dict[str, Any] = {
        "status": COMPUTED,
        "null_median": null_median_native / scale,
        "null_q95": null_q95_native / scale,
        "median": median_native / scale,
        "median_excess": (median_native - null_median_native) / scale,
        "fraction_above_null_q95": float(np.mean(values > null_q95_native)),
        "median_above_null_q95": bool(median_native > null_q95_native),
        "max_above_null_q95": bool(float(values.max()) > null_q95_native),
    }
    if exact_quanta and size is not None:
        record["median_excess_quanta"] = median_native - null_median_native
        record["null_q95_quanta"] = null_q95_native
    return record


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    position = 0
    while position < array.size:
        end = position
        while end + 1 < array.size and array[order[end + 1]] == array[order[position]]:
            end += 1
        rank = (position + end) / 2.0 + 1.0
        for cursor in range(position, end + 1):
            ranks[order[cursor]] = rank
        position = end + 1
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    """Tie-aware Spearman rank correlation; ``None`` when undefined."""
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denominator <= 0.0:
        return None
    return float((rx * ry).sum() / denominator)


def trend_record(
    doses: Sequence[float], medians: Sequence[float | None], null_q95: float | None
) -> dict[str, Any]:
    """Describe the dose curve of one unit, seed and estimator."""
    usable = [
        (float(dose), float(median))
        for dose, median in zip(doses, medians, strict=False)
        if median is not None
    ]
    if len(usable) < 3:
        return {
            "status": ABSTAIN_EMPTY,
            "n_doses": len(usable),
            "rho": None,
            "steps_increasing": None,
            "steps_equal": None,
            "steps_decreasing": None,
            "monotone_nondecreasing": None,
            "n_doses_above_null_q95": None,
            "dose_curve": [],
        }
    x = [item[0] for item in usable]
    y = [item[1] for item in usable]
    steps = np.diff(np.asarray(y, dtype=np.float64))
    above = (
        int(sum(1 for value in y if value > float(null_q95)))
        if null_q95 is not None
        else None
    )
    return {
        "status": COMPUTED,
        "n_doses": len(usable),
        "rho": spearman(x, y),
        "steps_increasing": int(np.sum(steps > 0.0)),
        "steps_equal": int(np.sum(steps == 0.0)),
        "steps_decreasing": int(np.sum(steps < 0.0)),
        "monotone_nondecreasing": bool(np.all(steps >= 0.0) and np.any(steps > 0.0)),
        "n_doses_above_null_q95": above,
        "dose_curve": [
            {"dose": float(dose), "median": float(median)} for dose, median in usable
        ],
    }


# --- composition and support diagnostics ---------------------------------------


def _bin_label(edges: tuple[float, float]) -> str:
    return m12.bin_label(edges)


def _bin_of(value: float, edges: Sequence[tuple[float, float]]) -> str:
    for edge in edges:
        if edge[0] <= value < edge[1]:
            return _bin_label(edge)
    return _bin_label(edges[-1])


def _mix_tv(first: Mapping[str, int], second: Mapping[str, int]) -> float | None:
    keys = set(first) | set(second)
    if not keys:
        return None
    total_first = sum(first.values())
    total_second = sum(second.values())
    if total_first == 0 or total_second == 0:
        return None
    return 0.5 * sum(
        abs(first.get(key, 0) / total_first - second.get(key, 0) / total_second)
        for key in keys
    )


def family_availability(frame: Frame) -> dict[str, int]:
    """Abnormal availability per family in the unit's source pool."""
    availability: dict[str, int] = {}
    for index in np.flatnonzero(frame.pool.abnormal).tolist():
        family = frame.pool.family_of(index)
        availability[family] = availability.get(family, 0) + 1
    return dict(sorted(availability.items()))


def family_target(frame: Frame, n_abnormal: int) -> dict[str, int]:
    """Proportional abnormal-family target of one dose (pool availability)."""
    availability = family_availability(frame)
    if n_abnormal <= 0 or not availability:
        return {}
    return apportion(availability, availability, n_abnormal)


def abnormal_family_counts(frame: Frame, n_abnormal: int) -> dict[str, int]:
    """Abnormal-only family counts of one dose."""
    counts: dict[str, int] = {}
    for index in frame.sequence[: int(n_abnormal)]:
        family = frame.pool.family_of(index)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def plan_family_counts(frame: Frame, n_abnormal: int) -> dict[str, int]:
    """Plan-level family counts of one dose, ordinary included."""
    counts = abnormal_family_counts(frame, n_abnormal)
    counts[ORDINARY] = frame.size - int(n_abnormal)
    return dict(sorted(counts.items()))


def plan_composition(
    frame: Frame,
    n_abnormal: int,
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Exact plan-level composition of one dose (families, severity, fraction)."""
    family_counts = plan_family_counts(frame, n_abnormal)
    severity_counts: dict[str, int] = {
        _bin_label(edges): 0 for edges in SEVERITY_BINS
    }
    fraction_counts: dict[str, int] = {
        _bin_label(edges): 0 for edges in FRACTION_BINS
    }
    severity_counts[ORDINARY] = int(family_counts.get(ORDINARY, 0))
    fraction_counts[ORDINARY] = int(family_counts.get(ORDINARY, 0))
    durations: list[int] = []
    for index in frame.sequence[: int(n_abnormal)]:
        family = frame.pool.family_of(index)
        entry = entries.get(frame.pool.ids[index], {})
        annotation = entry.get("annotation") if isinstance(entry, Mapping) else None
        annotation = annotation if isinstance(annotation, Mapping) else {}
        severity = annotation.get("severity")
        fraction = annotation.get("abnormal_step_fraction")
        severity_bin = (
            _bin_of(float(severity), SEVERITY_BINS)
            if family != ORDINARY and isinstance(severity, (int, float))
            else ORDINARY
        )
        fraction_bin = (
            _bin_of(float(fraction), FRACTION_BINS)
            if family != ORDINARY and isinstance(fraction, (int, float))
            else ORDINARY
        )
        severity_counts[severity_bin] = severity_counts.get(severity_bin, 0) + 1
        fraction_counts[fraction_bin] = fraction_counts.get(fraction_bin, 0) + 1
        duration = annotation.get("duration_steps")
        if isinstance(duration, (int, float)):
            durations.append(int(duration))
    return {
        "family_counts": family_counts,
        "severity_counts": dict(sorted(severity_counts.items())),
        "fraction_counts": dict(sorted(fraction_counts.items())),
        "abnormal_duration_steps": {
            "n": len(durations),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "median": float(np.median(durations)) if durations else None,
        },
    }


def _bin_index(value: float, edges: Sequence[tuple[float, float]]) -> int:
    for index, edge in enumerate(edges):
        if edge[0] <= value < edge[1]:
            return index
    return len(edges) - 1


def composition_codes(
    frame: Frame, entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    """Per-pool-index severity/fraction/duration codes for realized accounting."""
    pool = frame.pool
    severity = np.full(len(pool.ids), len(SEVERITY_BINS), dtype=np.int64)
    fraction = np.full(len(pool.ids), len(FRACTION_BINS), dtype=np.int64)
    duration = np.full(len(pool.ids), -1, dtype=np.int64)
    for index, fid in enumerate(pool.ids):
        if frame.pool.family_of(index) == ORDINARY:
            continue
        entry = entries.get(fid, {})
        annotation = entry.get("annotation") if isinstance(entry, Mapping) else None
        annotation = annotation if isinstance(annotation, Mapping) else {}
        value = annotation.get("severity")
        if isinstance(value, (int, float)):
            severity[index] = _bin_index(float(value), SEVERITY_BINS)
        value = annotation.get("abnormal_step_fraction")
        if isinstance(value, (int, float)):
            fraction[index] = _bin_index(float(value), FRACTION_BINS)
        value = annotation.get("duration_steps")
        if isinstance(value, (int, float)):
            duration[index] = int(value)
    return {"severity": severity, "fraction": fraction, "duration": duration}


def _code_counts(codes: np.ndarray, n_codes: int, repeats: int) -> np.ndarray:
    offsets = np.arange(repeats, dtype=np.int64)[:, None] * n_codes
    flat = (np.asarray(codes, dtype=np.int64) + offsets).reshape(-1)
    return np.bincount(flat, minlength=repeats * n_codes).reshape(repeats, n_codes)


def realized_composition(
    draws: np.ndarray,
    codes: Mapping[str, np.ndarray],
    planned: Mapping[str, Any],
    *,
    size: int,
) -> dict[str, Any]:
    """Composition of the ACTUAL sampled arms per repeat.

    The context and abnormal-family counts are exact by construction; severity,
    anomalous fraction and duration are attributes of the sampled identities and
    therefore vary. This block measures that variation instead of reporting the
    sequence-prefix plan as if it were the realized arm.
    """
    repeats = draws.shape[0]
    severity_labels = [_bin_label(edges) for edges in SEVERITY_BINS] + [ORDINARY]
    fraction_labels = [_bin_label(edges) for edges in FRACTION_BINS] + [ORDINARY]
    severity_counts = _code_counts(codes["severity"][draws], len(severity_labels), repeats)
    fraction_counts = _code_counts(codes["fraction"][draws], len(fraction_labels), repeats)
    planned_severity = np.asarray(
        [planned["severity_counts"].get(label, 0) for label in severity_labels],
        dtype=np.float64,
    )
    planned_fraction = np.asarray(
        [planned["fraction_counts"].get(label, 0) for label in fraction_labels],
        dtype=np.float64,
    )
    severity_tv = 0.5 * np.abs(severity_counts / float(size) - planned_severity / float(size)).sum(axis=1)
    fraction_tv = 0.5 * np.abs(fraction_counts / float(size) - planned_fraction / float(size)).sum(axis=1)
    duration_codes = codes["duration"][draws]
    positive = duration_codes > 0
    per_repeat_median: list[float] = []
    total_abnormal = 0
    minimum: int | None = None
    maximum: int | None = None
    for repeat in range(repeats):
        values = duration_codes[repeat][positive[repeat]]
        if values.size == 0:
            continue
        total_abnormal += int(values.size)
        per_repeat_median.append(float(np.median(values)))
        row_min = int(values.min())
        row_max = int(values.max())
        minimum = row_min if minimum is None else min(minimum, row_min)
        maximum = row_max if maximum is None else max(maximum, row_max)
    return {
        "description": (
            "composition of the actual sampled arms per repeat; context and abnormal "
            "family counts are exact, severity/fraction/duration are sampled"
        ),
        "severity_counts_repeat_0": {
            label: int(severity_counts[0, index])
            for index, label in enumerate(severity_labels)
        },
        "severity_counts_mean": {
            label: float(severity_counts[:, index].mean())
            for index, label in enumerate(severity_labels)
        },
        "fraction_counts_repeat_0": {
            label: int(fraction_counts[0, index])
            for index, label in enumerate(fraction_labels)
        },
        "fraction_counts_mean": {
            label: float(fraction_counts[:, index].mean())
            for index, label in enumerate(fraction_labels)
        },
        "severity_mix_tv_vs_planned_sequence_prefix": {
            "median": float(np.median(severity_tv)),
            "q05": float(np.quantile(severity_tv, 0.05)),
            "q95": float(np.quantile(severity_tv, 0.95)),
        },
        "fraction_mix_tv_vs_planned_sequence_prefix": {
            "median": float(np.median(fraction_tv)),
            "q05": float(np.quantile(fraction_tv, 0.05)),
            "q95": float(np.quantile(fraction_tv, 0.95)),
        },
        "abnormal_duration": {
            "mean_abnormal_files_per_repeat": total_abnormal / float(repeats),
            "median_of_repeat_medians": (
                float(np.median(per_repeat_median)) if per_repeat_median else None
            ),
            "min": minimum,
            "max": maximum,
        },
        "severity_controlled": False,
        "fraction_controlled": False,
        "duration_controlled": False,
        "ordinary_count_exact": int(size)
        - int(
            sum(
                value
                for label, value in planned["family_counts"].items()
                if label != ORDINARY
            )
        ),
    }


def duration_bin_edges(frame: Frame, codes: Mapping[str, np.ndarray]) -> list[float]:
    """Per-unit abnormal-duration quartile edges (diagnostic bins, metadata only)."""
    values = codes["duration"][np.flatnonzero(frame.pool.abnormal)]
    values = values[values > 0]
    if values.size < 4:
        return [30.0, 60.0, 120.0]
    edges: list[float] = []
    for value in np.quantile(values, [0.25, 0.5, 0.75]).tolist():
        candidate = float(value)
        if not edges or candidate > edges[-1]:
            edges.append(candidate)
    while len(edges) < 3:
        edges.append(edges[-1] + 1.0 if edges else 60.0)
    return edges


def _duration_bin(codes: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(
        np.asarray(edges, dtype=np.float64),
        np.asarray(codes, dtype=np.float64),
        side="right",
    )


def _masked_code_counts(
    codes: np.ndarray, mask: np.ndarray, n_codes: int, repeats: int
) -> np.ndarray:
    matrix = np.zeros((repeats, n_codes), dtype=np.int64)
    for repeat in range(repeats):
        row = codes[repeat][mask[repeat]]
        if row.size:
            matrix[repeat] = np.bincount(row, minlength=n_codes)[:n_codes]
    return matrix


def _rows_tv(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(first - second).sum(axis=1)


def _tv_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _comparison(
    current: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    *,
    same_repeats: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("severity", "fraction", "duration"):
        if same_repeats:
            values = _rows_tv(current[metric], target[metric])
        else:
            values = 0.5 * np.abs(
                current[metric] - target[metric][None, :]
            ).sum(axis=1)
        out[metric] = _tv_summary(values)
    return out


def abnormal_only_composition(
    frame: Frame,
    draws_by_dose: Mapping[float, np.ndarray],
    grid: Sequence[float],
    codes: Mapping[str, np.ndarray],
    plans: Mapping[float, Mapping[str, Any]],
) -> dict[str, Any]:
    """Abnormal-only severity/fraction/duration composition per dose.

    Normalised inside the actual abnormal files of each arm (never diluted by
    the ordinary part), compared with the planned abnormal-only prefix mix, the
    maximum-dose arm and the adjacent lower dose. Each adjacent-dose comparison
    carries its empirical TV quantum ``0.5 * (1/n_current + 1/n_previous)`` and
    a declared support flag at ``COMPOSITION_TV_QUANTUM_FLOOR``.
    """
    pool = frame.pool
    repeats = next(iter(draws_by_dose.values())).shape[0]
    edges = duration_bin_edges(frame, codes)
    severity_labels = [_bin_label(edge) for edge in SEVERITY_BINS]
    fraction_labels = [_bin_label(edge) for edge in FRACTION_BINS]
    duration_labels = [f"bin{index}" for index in range(4)]
    probabilities: dict[float, dict[str, np.ndarray]] = {}
    planned: dict[float, dict[str, np.ndarray]] = {}
    records: dict[str, Any] = {}
    for dose in grid:
        n_abnormal = int(plans[float(dose)]["n_abnormal"])
        if n_abnormal == 0:
            probabilities[float(dose)] = {}
            planned[float(dose)] = {}
            records[f"{float(dose):.2f}"] = {
                "n_abnormal": 0,
                "comparison_supported": False,
                "reason": "no abnormal files",
            }
            continue
        prefix = frame.sequence[:n_abnormal]
        planned[float(dose)] = {
            "severity": np.bincount(
                codes["severity"][prefix], minlength=len(SEVERITY_BINS)
            )
            / float(n_abnormal),
            "fraction": np.bincount(
                codes["fraction"][prefix], minlength=len(FRACTION_BINS)
            )
            / float(n_abnormal),
            "duration": np.bincount(
                _duration_bin(codes["duration"][prefix], edges), minlength=4
            )
            / float(n_abnormal),
        }
        draws = draws_by_dose[float(dose)]
        mask = pool.abnormal[draws]
        severity = _masked_code_counts(
            codes["severity"][draws], mask, len(SEVERITY_BINS), repeats
        )
        fraction = _masked_code_counts(
            codes["fraction"][draws], mask, len(FRACTION_BINS), repeats
        )
        duration = _masked_code_counts(
            _duration_bin(codes["duration"][draws], edges), mask, 4, repeats
        )
        probabilities[float(dose)] = {
            "severity": severity / float(n_abnormal),
            "fraction": fraction / float(n_abnormal),
            "duration": duration / float(n_abnormal),
        }
        records[f"{float(dose):.2f}"] = {
            "n_abnormal": n_abnormal,
            "severity_counts_mean": {
                label: float(severity[:, index].mean())
                for index, label in enumerate(severity_labels)
            },
            "severity_probs_repeat_0": {
                label: float(probabilities[float(dose)]["severity"][0, index])
                for index, label in enumerate(severity_labels)
            },
            "severity_probs_mean": {
                label: float(probabilities[float(dose)]["severity"][:, index].mean())
                for index, label in enumerate(severity_labels)
            },
            "fraction_counts_mean": {
                label: float(fraction[:, index].mean())
                for index, label in enumerate(fraction_labels)
            },
            "fraction_probs_repeat_0": {
                label: float(probabilities[float(dose)]["fraction"][0, index])
                for index, label in enumerate(fraction_labels)
            },
            "fraction_probs_mean": {
                label: float(probabilities[float(dose)]["fraction"][:, index].mean())
                for index, label in enumerate(fraction_labels)
            },
            "duration_bin_edges": edges,
            "duration_counts_mean": {
                label: float(duration[:, index].mean())
                for index, label in enumerate(duration_labels)
            },
            "duration_probs_repeat_0": {
                label: float(probabilities[float(dose)]["duration"][0, index])
                for index, label in enumerate(duration_labels)
            },
            "duration_probs_mean": {
                label: float(probabilities[float(dose)]["duration"][:, index].mean())
                for index, label in enumerate(duration_labels)
            },
        }
    maximum = float(grid[-1])
    positive = [
        float(dose) for dose in grid if int(plans[float(dose)]["n_abnormal"]) > 0
    ]
    for dose in positive:
        current = probabilities[float(dose)]
        record = records[f"{float(dose):.2f}"]
        record["vs_planned_abnormal_only"] = _comparison(
            current, planned[float(dose)], same_repeats=False
        )
        record["vs_max_dose"] = _comparison(
            current, probabilities[maximum], same_repeats=True
        )
        lower = [value for value in positive if value < dose]
        if lower:
            previous = lower[-1]
            n_current = int(plans[float(dose)]["n_abnormal"])
            n_previous = int(plans[float(previous)]["n_abnormal"])
            quantum = 0.5 * (1.0 / n_current + 1.0 / n_previous)
            supported = quantum <= COMPOSITION_TV_QUANTUM_FLOOR
            record["previous_dose"] = previous
            record["tv_quantum_vs_previous"] = quantum
            record["comparison_supported_vs_previous"] = supported
            record["vs_previous_dose"] = {
                metric: {
                    **_tv_summary(
                        _rows_tv(current[metric], probabilities[previous][metric])
                    ),
                    "tv_quantum": quantum,
                    "supported": supported,
                }
                for metric in ("severity", "fraction", "duration")
            }
    records["duration_bin_edges"] = edges
    return records


def abnormal_composition_aggregates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate supported/unsupported abnormal-only adjacent-dose comparisons."""
    groups: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    counts: dict[tuple[str, int, str], dict[str, int]] = {}
    for unit in records:
        key = (str(unit["horizon"]), int(unit["size"]), str(unit["design"]))
        bucket = groups.setdefault(
            key, {"severity": [], "fraction": [], "duration": [], "quantum": []}
        )
        tally = counts.setdefault(key, {"supported": 0, "unsupported": 0})
        for dose_key, dose_record in unit["doses"].items():  # type: ignore[union-attr]
            if dose_key == "duration_bin_edges":
                continue
            if "comparison_supported_vs_previous" not in dose_record:
                continue
            if dose_record["comparison_supported_vs_previous"]:
                tally["supported"] += 1
                for metric in ("severity", "fraction", "duration"):
                    bucket[metric].append(
                        float(dose_record["vs_previous_dose"][metric]["median"])
                    )
                bucket["quantum"].append(float(dose_record["tv_quantum_vs_previous"]))
            else:
                tally["unsupported"] += 1
    aggregates: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = groups[key]
        aggregates.append(
            {
                "horizon": key[0],
                "size": key[1],
                "design": key[2],
                "supported_adjacent_transitions": counts[key]["supported"],
                "unsupported_adjacent_transitions": counts[key]["unsupported"],
                "median_severity_tv": (
                    float(np.median(bucket["severity"]))
                    if bucket["severity"]
                    else None
                ),
                "median_fraction_tv": (
                    float(np.median(bucket["fraction"]))
                    if bucket["fraction"]
                    else None
                ),
                "median_duration_tv": (
                    float(np.median(bucket["duration"]))
                    if bucket["duration"]
                    else None
                ),
                "median_tv_quantum": (
                    float(np.median(bucket["quantum"]))
                    if bucket["quantum"]
                    else None
                ),
            }
        )
    return aggregates


def _abnormal_only_probs(
    frame: Frame,
    draws: np.ndarray,
    codes: Mapping[str, np.ndarray],
    edges: Sequence[float],
    n_abnormal: int,
) -> dict[str, np.ndarray]:
    mask = frame.pool.abnormal[draws]
    repeats = draws.shape[0]
    return {
        "severity": _masked_code_counts(
            codes["severity"][draws], mask, len(SEVERITY_BINS), repeats
        )
        / float(n_abnormal),
        "fraction": _masked_code_counts(
            codes["fraction"][draws], mask, len(FRACTION_BINS), repeats
        )
        / float(n_abnormal),
        "duration": _masked_code_counts(
            _duration_bin(codes["duration"][draws], edges), mask, 4, repeats
        )
        / float(n_abnormal),
    }


def nested_prefix_arms(
    frame: Frame,
    plans: Mapping[float, Mapping[str, Any]],
    draws_by_dose: Mapping[float, np.ndarray],
    grid: Sequence[float],
    baseline: np.ndarray,
) -> dict[float, np.ndarray]:
    """Sensitivity construction: the declared original nesting rule.

    Abnormal identities are prefixes of the maximum-dose arm taken group-wise
    (stratum x family) in frozen hash order, so the context quota and abnormal
    family counts stay exact; the ordinary slots are the implemented dose arm's
    ordinary identities. This is a sensitivity arm, never the main design.
    """
    pool = frame.pool
    maximum = float(grid[-1])
    max_draws = draws_by_dose[maximum]
    repeats, size = max_draws.shape
    pairs: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(pool.abnormal).tolist():
        pairs.setdefault((pool.stratum_of(index), pool.family_of(index)), []).append(
            index
        )
    for pair in pairs:
        pairs[pair].sort(
            key=lambda index: hashlib.sha256(
                f"{SEQUENCE_PREFIX}|{pool.ids[index]}".encode("utf-8")
            ).hexdigest()
        )
    big = np.iinfo(np.int64).max
    sorted_ranks: dict[tuple[str, str], np.ndarray] = {}
    for pair, ids in pairs.items():
        inverse = np.full(len(pool.ids), -1, dtype=np.int64)
        inverse[np.asarray(ids, dtype=np.int64)] = np.arange(len(ids), dtype=np.int64)
        ranks = inverse[max_draws]
        sorted_ranks[pair] = np.sort(np.where(ranks >= 0, ranks, big), axis=1)
    result: dict[float, np.ndarray] = {}
    for dose in grid:
        n_abnormal = int(plans[float(dose)]["n_abnormal"])
        main = draws_by_dose[float(dose)]
        ordinary = np.sort(
            np.where(pool.abnormal[main], big, main), axis=1
        )[:, : size - n_abnormal]
        chunks: list[np.ndarray] = [ordinary]
        for pair, ids in pairs.items():
            count = int(plans[float(dose)]["abnormal_by_pair"].get(pair, 0))
            if count == 0:
                continue
            chunks.append(
                np.asarray(ids, dtype=np.int64)[sorted_ranks[pair][:, :count]]
            )
        nested = np.sort(np.concatenate(chunks, axis=1), axis=1)
        _require(
            nested.shape == (repeats, size),
            "REFUSED-INTERNAL: nested sensitivity arm size mismatch",
        )
        _require(
            bool(np.all(np.diff(nested, axis=1) > 0)),
            "REFUSED-INTERNAL: duplicate identity in the nested sensitivity arm",
        )
        _require(
            not bool(np.isin(nested, baseline).any()),
            "REFUSED-INTERNAL: nested sensitivity arm shares baseline identities",
        )
        offsets = np.arange(repeats, dtype=np.int64)[:, None]
        codes = pool.strata[nested]
        counts = np.bincount(
            (codes + offsets * len(pool.stratum_names)).reshape(-1),
            minlength=repeats * len(pool.stratum_names),
        ).reshape(repeats, len(pool.stratum_names))
        expected = np.asarray(
            [int(frame.quota.get(name, 0)) for name in pool.stratum_names],
            dtype=np.int64,
        )
        _require(
            bool(np.all(counts == expected[None, :])),
            "REFUSED-INTERNAL: nested sensitivity context differs from the quota",
        )
        result[float(dose)] = nested
    return result


def sensitivity_aggregates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Median/max absolute E1 and E2_K8 change of the nested sensitivity arms."""
    groups: dict[tuple[str, int, str, str], list[float]] = {}
    for record in records:
        if record.get("anchor_dose"):
            continue
        for estimator, key in (("E1", "e1_delta"), ("E2_K8", "e2_k8_delta")):
            group = (
                str(record["horizon"]),
                int(record["size"]),
                str(record["design"]),
                estimator,
            )
            groups.setdefault(group, []).append(abs(float(record[key])))
    aggregates: list[dict[str, Any]] = []
    for group in sorted(groups):
        values = groups[group]
        aggregates.append(
            {
                "horizon": group[0],
                "size": group[1],
                "design": group[2],
                "estimator": group[3],
                "n": len(values),
                "median_abs_delta": float(np.median(values)),
                "max_abs_delta": float(max(values)),
            }
        )
    return aggregates


def shared_sequence_infeasibility(
    manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
) -> dict[str, Any]:
    """Metadata-only check of the declared shared-sequence reading.

    The original sentence asks for one sequence per ``cell x design`` reused at
    every size. The frozen headroom rule needs the size-specific context quota,
    so a largest-size sequence can exceed a smaller size's per-stratum quota.
    This reports every such violation with the exact strata and excess; it does
    not materialise any arm.
    """
    cells = build_cells(audit_record)
    rows: list[dict[str, Any]] = []
    for key, cell in sorted(cells.items()):
        horizon = str(cell["horizon"])
        sizes = SIZES[horizon]
        largest = max(sizes)
        for design in DESIGNS:
            pool = build_pool(manifest, cell, design)
            grid = DOSE_GRID[design]
            n_large = max(abnormal_count(dose, largest) for dose in grid)
            frame_large = build_frame(pool, largest, n_large)
            for small in sorted(sizes):
                if small == largest:
                    continue
                n_small = max(abnormal_count(dose, small) for dose in grid)
                frame_small = build_frame(pool, small, n_small)
                if frame_large.status != COMPUTED or frame_small.status != COMPUTED:
                    rows.append(
                        {
                            "cell": key,
                            "design": design,
                            "small_size": int(small),
                            "feasible": None,
                            "large_status": frame_large.status,
                            "small_status": frame_small.status,
                        }
                    )
                    continue
                for dose in grid:
                    n_abnormal = abnormal_count(dose, small)
                    prefix = frame_large.sequence[:n_abnormal]
                    counts = Counter(pool.stratum_of(index) for index in prefix)
                    excess = {
                        name: int(counts.get(name, 0))
                        - int(frame_small.quota.get(name, 0))
                        for name in frame_small.quota
                        if int(counts.get(name, 0))
                        > int(frame_small.quota.get(name, 0))
                    }
                    rows.append(
                        {
                            "cell": key,
                            "design": design,
                            "small_size": int(small),
                            "dose": float(dose),
                            "n_abnormal": int(n_abnormal),
                            "feasible": not excess,
                            "excess_total": int(sum(excess.values())),
                            "excess_strata": excess,
                        }
                    )
    summary = {
        "checks": len(rows),
        "feasible": sum(1 for row in rows if row.get("feasible") is True),
        "infeasible": sum(1 for row in rows if row.get("feasible") is False),
        "unavailable": sum(1 for row in rows if row.get("feasible") is None),
        "max_excess_total": max(
            (int(row.get("excess_total", 0)) for row in rows), default=0
        ),
        "rule": (
            "original abnormal_sequence.scope: one sequence per cell x design, "
            "independent of size; checked against the frozen size-specific quota"
        ),
    }
    return {"summary": summary, "rows": rows}


def period_mix_summary(
    frame: Frame,
    draws: np.ndarray,
    baseline_period_counts: Mapping[str, int],
    baseline_size: int,
) -> dict[str, Any]:
    """Period-composition of a realized arm and its distance from the baseline.

    Every draw carries the same context quota; the source-period mix varies
    because the abnormal prefix is concentrated in the corpus's elevated
    periods. The distance is reported per repeat, never treated as an effect.
    """
    pool = frame.pool
    n_periods = len(pool.period_numbers)
    codes = pool.periods[draws]
    repeats, size = codes.shape
    counts = np.empty((repeats, n_periods), dtype=np.float64)
    for code in range(n_periods):
        counts[:, code] = (codes == code).sum(axis=1)
    baseline = np.asarray(
        [baseline_period_counts.get(str(number), 0) for number in pool.period_numbers],
        dtype=np.float64,
    )
    if baseline.sum() != baseline_size:
        baseline = np.zeros(n_periods, dtype=np.float64)
    baseline = baseline / max(float(baseline_size), 1.0)
    probabilities = counts / float(size)
    distances = 0.5 * np.abs(probabilities - baseline[None, :]).sum(axis=1)
    low_codes = [
        index
        for index, number in enumerate(pool.period_numbers)
        if number in pool.low_periods
    ]
    low_share = (
        counts[:, low_codes].sum(axis=1) / float(size) if low_codes else np.zeros(repeats)
    )
    return {
        "period_share_median": {
            str(number): float(np.median(probabilities[:, index]))
            for index, number in enumerate(pool.period_numbers)
        },
        "period_mix_tv_vs_baseline": {
            "median": float(np.median(distances)),
            "q05": float(np.quantile(distances, 0.05)),
            "q95": float(np.quantile(distances, 0.95)),
        },
        "low_period_share": {
            "median": float(np.median(low_share)),
            "q05": float(np.quantile(low_share, 0.05)),
            "q95": float(np.quantile(low_share, 0.95)),
        },
        "context_counts_exact": True,
    }


# --- census (metadata only) ----------------------------------------------------


def build_census(
    manifest: Mapping[str, object],
    audit_record: Mapping[str, object],
) -> dict[str, Any]:
    """Metadata-only feasibility census; no embedding, cache or outcome is read."""
    cells = build_cells(audit_record)
    units: list[dict[str, Any]] = []
    for key, cell in sorted(cells.items()):
        horizon = str(cell["horizon"])
        for design in DESIGNS:
            pool = build_pool(manifest, cell, design)
            ordinary = int((~pool.abnormal).sum())
            abnormal = int(pool.abnormal.sum())
            family_counts: dict[str, int] = {}
            for index in np.flatnonzero(pool.abnormal).tolist():
                name = pool.family_of(index)
                family_counts[name] = family_counts.get(name, 0) + 1
            period_counts: dict[str, dict[str, int]] = {}
            for index in range(len(pool.ids)):
                period = str(pool.period_numbers[int(pool.periods[index])])
                bucket = period_counts.setdefault(period, {"ordinary": 0, "abnormal": 0})
                bucket["abnormal" if bool(pool.abnormal[index]) else "ordinary"] += 1
            for size in SIZES[horizon]:
                grid = DOSE_GRID[design]
                n_max = max(abnormal_count(dose, size) for dose in grid)
                frame = build_frame(pool, size, n_max)
                units.append(
                    {
                        "cell": key,
                        "history_id": cell["history_id"],
                        "horizon": horizon,
                        "design": design,
                        "size": int(size),
                        "periods": (
                            [int(value) for value in sorted(cell["periods"])]
                            if design == DESIGN_POOLED
                            else [int(value) for value in sorted(cell["low_periods"])]
                        ),
                        "n_pool": len(pool.ids),
                        "n_ordinary": ordinary,
                        "n_abnormal": abnormal,
                        "family_counts": dict(sorted(family_counts.items())),
                        "period_counts": dict(sorted(period_counts.items())),
                        "n_strata": len(pool.stratum_names),
                        "status": frame.status,
                        "reason": frame.reason,
                        "n_max_abnormal": int(n_max),
                        "quota": {
                            name: int(value)
                            for name, value in sorted(frame.quota.items())
                        },
                        "doses": [
                            {
                                "dose": float(dose),
                                "n_abnormal": abnormal_count(dose, size),
                                "n_ordinary": int(size) - abnormal_count(dose, size),
                            }
                            for dose in grid
                        ],
                    }
                )
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit["status"]] = counts.get(unit["status"], 0) + 1
    return {
        "metadata_only": True,
        "embeddings_read": False,
        "cache_read": False,
        "outcomes_computed": False,
        "task": TASK_ID,
        "parameters": {
            "sizes": {horizon: list(values) for horizon, values in SIZES.items()},
            "designs": list(DESIGNS),
            "dose_grid": {design: list(values) for design, values in DOSE_GRID.items()},
            "repeats": REPEATS,
            "encoder_seeds": list(readiness.FROZEN_ENCODER_SEEDS),
            "cluster_counts": list(CLUSTER_COUNTS),
            "count_rounding": "floor(p * N + 0.5)",
            "context_quota": (
                "capped largest-remainder over floor(ordinary/2) per robot|program stratum"
            ),
            "abnormal_sequence": (
                "family-deficit proportional order within per-stratum context headroom"
            ),
        },
        "cells": {
            key: {
                "history_id": cell["history_id"],
                "horizon": cell["horizon"],
                "low_periods": [int(value) for value in sorted(cell["low_periods"])],
                "elevated_periods": [
                    int(value) for value in sorted(cell["elevated_periods"])
                ],
                "comparisons": cell["comparisons"],
            }
            for key, cell in sorted(cells.items())
        },
        "n_cells": len(cells),
        "n_units": len(units),
        "unit_status_counts": dict(sorted(counts.items())),
        "units": units,
    }


# --- audit orchestration -------------------------------------------------------


def _pool_positions(pool: PoolTable, group: Any) -> np.ndarray:
    return np.asarray([group.index[fid] for fid in pool.ids], dtype=np.int64)


def run_dose_response(
    *,
    corpus: Path,
    audit_path: Path,
    contract_path: Path,
    cache_root: Path,
    census_path: Path | None = None,
    g2_preflight_path: Path | None = None,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """Run the complete cache-only M13 audit and return its tables."""
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
            groups[f"{history_id}|{seed}"].file_ids
            for seed in readiness.FROZEN_ENCODER_SEEDS
        ]
        _require(
            all(order == orders[0] for order in orders),
            f"REFUSED-IDENTITY: encoder seeds disagree on the file-ID order of {history_id}",
        )
    entries: dict[str, Mapping[str, Any]] = {
        str(entry["file_id"]): entry
        for entry in manifest["files"]  # type: ignore[index,union-attr]
        if isinstance(entry, Mapping) and isinstance(entry.get("file_id"), str)
    }
    fits, fit_custody = m12.fit_references(groups, manifest)
    labels = m12.prototype_labels(groups, fits)

    census = build_census(manifest, audit_record)
    census_check: dict[str, Any] = {"checked": False, "match": None, "sha256": None}
    if census_path is not None:
        _require(census_path.is_file(), f"REFUSED-CENSUS-MISSING: {census_path}")
        recorded = json.loads(census_path.read_text(encoding="utf-8"))
        recorded_census = recorded.get("census", recorded)
        recorded_units = {
            (str(unit["cell"]), int(unit["size"]), str(unit["design"])): unit
            for unit in recorded_census.get("units", [])  # type: ignore[union-attr]
        }
        for unit in census["units"]:
            key = (str(unit["cell"]), int(unit["size"]), str(unit["design"]))
            _require(
                key in recorded_units,
                f"REFUSED-CENSUS-DRIFT: unit {key} not in the recorded census",
            )
            _require(
                recorded_units[key]["status"] == unit["status"],
                f"REFUSED-CENSUS-DRIFT: unit {key} status {unit['status']} != "
                f"recorded {recorded_units[key]['status']}",
            )
        census_check = {
            "checked": True,
            "match": True,
            "sha256": sha256_file(census_path),
            "path": str(census_path),
        }

    cells = build_cells(audit_record)
    units: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    baseline_table: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    abnormal_composition: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    arms_evidence: list[dict[str, Any]] = []
    repeats_store: dict[str, np.ndarray] = {}

    for key, cell in sorted(cells.items()):
        horizon = str(cell["horizon"])
        for design in DESIGNS:
            grid = DOSE_GRID[design]
            pool = build_pool(manifest, cell, design)
            for size in SIZES[horizon]:
                n_max = max(abnormal_count(dose, size) for dose in grid)
                frame = build_frame(pool, size, n_max)
                unit_id = f"{key}|{size}|{design}"
                unit_record: dict[str, Any] = {
                    "unit": unit_id,
                    "cell": key,
                    "history_id": cell["history_id"],
                    "horizon": horizon,
                    "design": design,
                    "size": int(size),
                    "periods": [
                        int(period)
                        for period in (
                            sorted(cell["periods"])
                            if design == DESIGN_POOLED
                            else sorted(cell["low_periods"])
                        )
                    ],
                    "n_pool": len(pool.ids),
                    "n_ordinary_pool": int((~pool.abnormal).sum()),
                    "n_abnormal_pool": int(pool.abnormal.sum()),
                    "n_strata": len(pool.stratum_names),
                    "status": frame.status,
                    "reason": frame.reason,
                    "doses": [float(dose) for dose in grid],
                    "quota": {
                        name: int(value) for name, value in sorted(frame.quota.items())
                    },
                    "quota_total": int(sum(frame.quota.values())),
                    "n_max_abnormal": int(n_max),
                    "sequence_sha256": sha256_object(
                        [pool.ids[index] for index in frame.sequence]
                    ),
                    "abnormal_sequence_length": len(frame.sequence),
                    "family_availability": (
                        family_availability(frame) if frame.status == COMPUTED else {}
                    ),
                }
                units.append(unit_record)
                if frame.status != COMPUTED:
                    for dose in grid:
                        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                            curves.append(
                                {
                                    "unit": unit_id,
                                    "cell": key,
                                    "horizon": horizon,
                                    "design": design,
                                    "size": int(size),
                                    "dose": float(dose),
                                    "model_seed": int(model_seed),
                                    "status": frame.status,
                                    "reason": frame.reason,
                                    "comparison": DOSE_VS_BASELINE,
                                    "estimators": {},
                                }
                            )
                    continue

                baseline = baseline_indices(frame, cell=key, design=design)
                composition_catalog = composition_codes(frame, entries)
                baseline_positions = _pool_positions(pool, groups[f"{cell['history_id']}|{readiness.FROZEN_ENCODER_SEEDS[0]}"])[baseline]
                baseline_period_counts = {
                    str(pool.period_numbers[code]): int(
                        np.sum(pool.periods[baseline] == code)
                    )
                    for code in range(len(pool.period_numbers))
                }
                baseline_table.append(
                    {
                        "unit": unit_id,
                        "size": int(size),
                        "sha256": sha256_object(
                            sorted(pool.ids[index] for index in baseline.tolist())
                        ),
                        "n": int(baseline.size),
                        "period_counts": baseline_period_counts,
                        "family_counts": {ORDINARY: int(baseline.size)},
                        "stratum_counts": {
                            name: int(value)
                            for name, value in sorted(frame.quota.items())
                        },
                        "dose_zero_is_separate_draw": True,
                        "not_a_calendar_window": design == DESIGN_POOLED,
                    }
                )

                unit_draws: dict[float, np.ndarray] = {}
                unit_plans: dict[float, dict[str, Any]] = {}
                for dose in grid:
                    plan = dose_plan(frame, dose)
                    draws = draw_dose_arms(
                        frame,
                        plan,
                        baseline,
                        cell=key,
                        design=design,
                        repeats=repeats,
                    )
                    unit_draws[float(dose)] = draws
                    unit_plans[float(dose)] = plan
                    if float(dose) == 0.0:
                        _require(
                            plan["n_abnormal"] == 0,
                            "REFUSED-INTERNAL: zero dose carries abnormal files",
                        )
                    planned_composition = plan_composition(
                        frame, int(plan["n_abnormal"]), entries
                    )
                    composition.append(
                        {
                            "unit": unit_id,
                            "dose": float(dose),
                            "requested_dose": float(dose),
                            "realized_dose": plan["n_abnormal"] / float(size),
                            "n_abnormal": int(plan["n_abnormal"]),
                            "n_ordinary": int(plan["n_ordinary"]),
                            "family_counts": plan_family_counts(
                                frame, int(plan["n_abnormal"])
                            ),
                            "abnormal_family_counts": abnormal_family_counts(
                                frame, int(plan["n_abnormal"])
                            ),
                            "family_target": family_target(
                                frame, int(plan["n_abnormal"])
                            ),
                            "family_mix_tv_target": _mix_tv(
                                abnormal_family_counts(frame, int(plan["n_abnormal"])),
                                family_target(frame, int(plan["n_abnormal"])),
                            ),
                            "planned_sequence_prefix": planned_composition,
                            "realized": realized_composition(
                                draws,
                                composition_catalog,
                                planned_composition,
                                size=size,
                            ),
                            "composition_control": {
                                "context_exact": True,
                                "family_exact": True,
                                "severity_controlled": False,
                                "fraction_controlled": False,
                                "duration_controlled": False,
                                "note": (
                                    "the sequence prefix fixes the abnormal family counts; "
                                    "severity, anomalous fraction and duration are attributes "
                                    "of the sampled identities and are measured, not fixed"
                                ),
                            },
                            "context_tv_vs_baseline": 0.0,
                            "context_exact_match": True,
                            "planned_prefix_nested": True,
                            "realized_sets_nested": False,
                            "shared_support_with_baseline": 0,
                            "period_mix": period_mix_summary(
                                frame,
                                draws,
                                baseline_period_counts,
                                int(baseline.size),
                            ),
                            "comparison": (
                                BASELINE_VS_BASELINE
                                if float(dose) == 0.0
                                else DOSE_VS_BASELINE
                            ),
                        }
                    )

                abnormal_records = abnormal_only_composition(
                    frame, unit_draws, grid, composition_catalog, unit_plans
                )
                unit_record["abnormal_only_composition"] = abnormal_records
                abnormal_composition.append(
                    {
                        "unit": unit_id,
                        "cell": key,
                        "horizon": horizon,
                        "design": design,
                        "size": int(size),
                        "doses": abnormal_records,
                    }
                )

                zero_draws = unit_draws[float(grid[0])]
                null_values: dict[str, dict[str, np.ndarray]] = {}
                null_summaries: dict[str, dict[str, Any]] = {}
                for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                    group_key = f"{cell['history_id']}|{model_seed}"
                    group = groups[group_key]
                    sigma = float(fits[group_key].sigma)
                    zero_positions = _pool_positions(pool, group)[zero_draws]
                    null_values[str(model_seed)] = {
                        "E1": e1_repeats(group, baseline_positions, zero_positions, sigma)
                    }
                    for k in CLUSTER_COUNTS:
                        null_values[str(model_seed)][f"E2_K{k}"] = e2_quanta_repeats(
                            labels[group_key][k], baseline_positions, zero_positions, k
                        )
                    null_summaries[str(model_seed)] = {
                        "E1": float_summary(null_values[str(model_seed)]["E1"]),
                        **{
                            f"E2_K{k}": quanta_summary(
                                null_values[str(model_seed)][f"E2_K{k}"], size
                            )
                            for k in CLUSTER_COUNTS
                        },
                    }
                    repeats_store[f"{unit_id}|0.00|{model_seed}|E1"] = null_values[
                        str(model_seed)
                    ]["E1"]
                    for k in CLUSTER_COUNTS:
                        repeats_store[
                            f"{unit_id}|0.00|{model_seed}|E2_K{k}"
                        ] = null_values[str(model_seed)][f"E2_K{k}"]

                main_medians: dict[tuple[float, int, str], float] = {}
                for dose in grid:
                    draws = unit_draws[float(dose)]
                    for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                        group_key = f"{cell['history_id']}|{model_seed}"
                        group = groups[group_key]
                        sigma = float(fits[group_key].sigma)
                        positions = _pool_positions(pool, group)[draws]
                        estimators: dict[str, Any] = {}
                        if float(dose) == 0.0:
                            for estimator in ESTIMATORS:
                                estimators[estimator] = {
                                    **null_summaries[str(model_seed)][estimator],
                                    "movement": None,
                                    "null_band_reference": BASELINE_VS_BASELINE,
                                }
                        else:
                            e1_values = e1_repeats(
                                group, baseline_positions, positions, sigma
                            )
                            e1_summary = float_summary(e1_values)
                            e1_summary["movement"] = movement_summary(
                                e1_values,
                                null_values[str(model_seed)]["E1"],
                                size=size,
                                exact_quanta=False,
                            )
                            estimators["E1"] = e1_summary
                            repeats_store[
                                f"{unit_id}|{dose:.2f}|{model_seed}|E1"
                            ] = e1_values
                            for k in CLUSTER_COUNTS:
                                quanta = e2_quanta_repeats(
                                    labels[group_key][k],
                                    baseline_positions,
                                    positions,
                                    k,
                                )
                                summary = quanta_summary(quanta, size)
                                summary["movement"] = movement_summary(
                                    quanta.astype(np.float64),
                                    null_values[str(model_seed)][f"E2_K{k}"],
                                    size=size,
                                    exact_quanta=True,
                                )
                                estimators[f"E2_K{k}"] = summary
                                repeats_store[
                                    f"{unit_id}|{dose:.2f}|{model_seed}|E2_K{k}"
                                ] = quanta
                        curves.append(
                            {
                                "unit": unit_id,
                                "cell": key,
                                "horizon": horizon,
                                "design": design,
                                "size": int(size),
                                "dose": float(dose),
                                "model_seed": int(model_seed),
                                "status": COMPUTED,
                                "reason": "",
                                "comparison": (
                                    BASELINE_VS_BASELINE
                                    if float(dose) == 0.0
                                    else DOSE_VS_BASELINE
                                ),
                                "estimators": estimators,
                            }
                        )
                        main_medians[(float(dose), int(model_seed), "E1")] = float(
                            estimators["E1"]["median"]
                        )
                        main_medians[(float(dose), int(model_seed), "E2_K8")] = float(
                            estimators["E2_K8"]["median"]
                        )

                if design == DESIGN_POOLED and int(size) == max(SIZES[horizon]):
                    nested = nested_prefix_arms(
                        frame, unit_plans, unit_draws, grid, baseline
                    )
                    nested_edges = duration_bin_edges(frame, composition_catalog)
                    for dose in grid:
                        if float(dose) <= 0.0:
                            continue
                        n_abn = int(unit_plans[float(dose)]["n_abnormal"])
                        main_probs = _abnormal_only_probs(
                            frame,
                            unit_draws[float(dose)],
                            composition_catalog,
                            nested_edges,
                            n_abn,
                        )
                        nested_probs = _abnormal_only_probs(
                            frame,
                            nested[float(dose)],
                            composition_catalog,
                            nested_edges,
                            n_abn,
                        )
                        composition_tv = {
                            metric: float(
                                np.median(_rows_tv(main_probs[metric], nested_probs[metric]))
                            )
                            for metric in ("severity", "fraction", "duration")
                        }
                        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                            group_key = f"{cell['history_id']}|{model_seed}"
                            group = groups[group_key]
                            sigma = float(fits[group_key].sigma)
                            positions_nested = _pool_positions(pool, group)[
                                nested[float(dose)]
                            ]
                            e1_nested_values = e1_repeats(
                                group, baseline_positions, positions_nested, sigma
                            )
                            e1_nested = float(np.median(e1_nested_values))
                            quanta_nested = e2_quanta_repeats(
                                labels[group_key][8],
                                baseline_positions,
                                positions_nested,
                                8,
                            )
                            e2_nested = float(np.median(quanta_nested)) / float(size)
                            repeats_store[
                                f"{unit_id}|{float(dose):.2f}|{model_seed}|E1_nested"
                            ] = e1_nested_values
                            repeats_store[
                                f"{unit_id}|{float(dose):.2f}|{model_seed}|E2_K8_nested"
                            ] = quanta_nested
                            nested_ids = [
                                pool.ids[index] for index in nested[float(dose)][0].tolist()
                            ]
                            sensitivity.append(
                                {
                                    "unit": unit_id,
                                    "cell": key,
                                    "horizon": horizon,
                                    "design": design,
                                    "size": int(size),
                                    "dose": float(dose),
                                    "model_seed": int(model_seed),
                                    "e1_main": main_medians[
                                        (float(dose), int(model_seed), "E1")
                                    ],
                                    "e1_nested": e1_nested,
                                    "e1_delta": e1_nested
                                    - main_medians[(float(dose), int(model_seed), "E1")],
                                    "e2_k8_main": main_medians[
                                        (float(dose), int(model_seed), "E2_K8")
                                    ],
                                    "e2_k8_nested": e2_nested,
                                    "e2_k8_delta": e2_nested
                                    - main_medians[
                                        (float(dose), int(model_seed), "E2_K8")
                                    ],
                                    "nested_composition_tv": composition_tv,
                                    "nested_ids_repeat_0": nested_ids,
                                    "nested_sha256": sha256_object(sorted(nested_ids)),
                                    "anchor_dose": float(dose) == float(grid[-1]),
                                    "sensitivity_rule": (
                                        "declared original nested-prefix identities "
                                        "(group-wise hash prefix of the maximum-dose arm) "
                                        "vs implemented independent per-dose sampling"
                                    ),
                                }
                            )

                unit_record["zero_dose_arm_sha256"] = sha256_object(
                    sorted(pool.ids[index] for index in zero_draws[0].tolist())
                )
                unit_record["source_reuse"] = _source_reuse(
                    pool, unit_draws, grid, repeats, size, key
                )
                arms_evidence.append(
                    {
                        "unit": unit_id,
                        "cell": key,
                        "horizon": horizon,
                        "design": design,
                        "size": int(size),
                        "baseline": [pool.ids[index] for index in baseline.tolist()],
                        "baseline_sha256": sha256_object(
                            sorted(pool.ids[index] for index in baseline.tolist())
                        ),
                        "context_counts": {
                            name: int(value)
                            for name, value in sorted(frame.quota.items())
                        },
                        "reference_repeat": 0,
                        "doses": {
                            f"{float(dose):.2f}": {
                                "requested_dose": float(dose),
                                "ids": [
                                    pool.ids[index]
                                    for index in unit_draws[float(dose)][
                                        0
                                    ].tolist()
                                ],
                                "sha256": sha256_object(
                                    sorted(
                                        pool.ids[index]
                                        for index in unit_draws[float(dose)][
                                            0
                                        ].tolist()
                                    )
                                ),
                                "n_abnormal": int(
                                    unit_plans[float(dose)]["n_abnormal"]
                                ),
                                "abnormal_family_counts": abnormal_family_counts(
                                    frame,
                                    int(unit_plans[float(dose)]["n_abnormal"]),
                                ),
                            }
                            for dose in grid
                        },
                    }
                )

    accounting = _accounting(
        units, curves, composition, repeats, n_fits=len(fit_custody)
    )
    trends = _trends(curves, units)
    aggregates = _aggregates(trends)
    return {
        "task": TASK_ID,
        "status": (
            "AUDIT-COMPLETE"
            if accounting["all_planned_cells_accounted"]
            else "AUDIT-INCOMPLETE"
        ),
        "spec": spec,
        "amendments": amendments,
        "census_check": census_check,
        "lineage": {
            inner_key: value
            for inner_key, value in lineage.items()
            if inner_key
            not in {"manifest", "audit_record", "identity_plan", "custody", "contract"}
        },
        "prior_products": products_before,
        "cache": {
            "cache_root": cache["cache_root"],
            "admission": cache["admission"],
            "n_groups": cache["n_groups"],
            "n_rows": cache["n_rows"],
            "n_cache_files": cache["n_cache_files"],
            "cache_file_digest_digest": cache["cache_file_digest_digest"],
            "cache_file_digests": cache["cache_file_digests"],
        },
        "fit_custody": fit_custody,
        "census": census,
        "support": {
            "sizes": {horizon: list(values) for horizon, values in SIZES.items()},
            "designs": list(DESIGNS),
            "dose_grid": {design: list(values) for design, values in DOSE_GRID.items()},
            "count_rounding": "floor(p * N + 0.5)",
            "context_quota": (
                "capped largest-remainder over floor(ordinary/2) per robot|program stratum"
            ),
            "units": units,
        },
        "curves": curves,
        "baseline": {
            "description": (
                "fixed all-ordinary baseline arm at the frame context quota; every "
                "dose arm is disjoint from it and dose 0 is the "
                "baseline-vs-baseline draw set"
            ),
            "records": baseline_table,
        },
        "composition": composition,
        "abnormal_composition": abnormal_composition,
        "abnormal_composition_aggregates": abnormal_composition_aggregates(
            abnormal_composition
        ),
        "sensitivity": sensitivity,
        "sensitivity_aggregates": sensitivity_aggregates(sensitivity),
        "shared_sequence_infeasibility": shared_sequence_infeasibility(
            manifest, audit_record
        ),
        "arms": arms_evidence,
        "trends": trends,
        "aggregates": aggregates,
        "accounting": accounting,
        "repeats": repeats_store,
        "forwards_executed_this_run": 0,
        "model_loads": 0,
        "cumulative_forwards_after_run": CUMULATIVE_FORWARDS,
        "elapsed_s": time.time() - started,
        "limitations": [
            "cache-only exploratory audit of one fixed admitted cache; no capture, no forward pass and no independent confirmation of S24-7-M11 or S24-7-M12",
            "constructed mixtures are label-informed and are NOT calendar windows; the pooled design deliberately spans periods, so source-period composition is a measured confounder",
            "the fixed baseline arm is one realized all-ordinary sample at the frozen context quota; every dose arm is disjoint from it by construction",
            "repeats resample file identities inside fixed composition groups; they are Monte Carlo variability of one finite support, never confidence limits",
            "encoder seeds are run replicates, never independent data; every dose shares the same fixed baseline and the abnormal sets are nested, so dose comparisons are dependent by design",
            "the abnormal sequence is one deterministic family-deficit ordering; its support-conditional selection is disclosed and no dose is selected on an outcome",
            "no p-value, significance test, success gate, performance, detection-accuracy, business-density, causal or generalisation claim is produced",
        ],
    }


def _membership_matrix(
    draws: np.ndarray, column_of: np.ndarray, n_columns: int
) -> np.ndarray:
    """Boolean membership of each repeat over a pool partition."""
    repeats, size = draws.shape
    rows = np.repeat(np.arange(repeats, dtype=np.int64)[:, None], size, axis=1)
    columns = column_of[draws]
    valid = columns >= 0
    matrix = np.zeros((repeats, n_columns), dtype=bool)
    matrix[rows[valid], columns[valid]] = True
    return matrix


def _source_reuse(
    pool: PoolTable,
    draws_by_dose: Mapping[float, np.ndarray],
    grid: Sequence[float],
    repeats: int,
    size: int,
    cell: str,
) -> dict[str, Any]:
    """True cross-dose and cross-repeat source reuse of one unit.

    The planned abnormal counts come from nested sequence prefixes, but the
    realized arms resample identities per dose. This block reports the actual
    identity overlap (abnormal and ordinary parts separately, over all repeats)
    and measures how often the nested-prefix property fails at identity level.
    """
    abnormal_indices = np.flatnonzero(pool.abnormal)
    ordinary_indices = np.flatnonzero(~pool.abnormal)
    abnormal_column = np.full(len(pool.ids), -1, dtype=np.int64)
    abnormal_column[abnormal_indices] = np.arange(abnormal_indices.size, dtype=np.int64)
    ordinary_column = np.full(len(pool.ids), -1, dtype=np.int64)
    ordinary_column[ordinary_indices] = np.arange(ordinary_indices.size, dtype=np.int64)
    membership: dict[float, dict[str, np.ndarray]] = {}
    abnormal_counts: dict[float, int] = {}
    ordinary_counts: dict[float, int] = {}
    for dose in grid:
        draws = draws_by_dose[float(dose)]
        membership[float(dose)] = {
            "abnormal": _membership_matrix(draws, abnormal_column, abnormal_indices.size),
            "ordinary": _membership_matrix(draws, ordinary_column, ordinary_indices.size),
        }
        first_repeat = membership[float(dose)]
        abnormal_counts[float(dose)] = int(first_repeat["abnormal"][0].sum())
        ordinary_counts[float(dose)] = int(first_repeat["ordinary"][0].sum())
    reference_repeat = 0
    zero_dose = float(grid[0])
    # Repeat-0 exact counts, as before.
    zero_ids = set(int(value) for value in draws_by_dose[zero_dose][reference_repeat].tolist())
    overlap: dict[str, Any] = {}
    for dose in grid:
        ids = set(
            int(value) for value in draws_by_dose[float(dose)][reference_repeat].tolist()
        )
        shared = ids & zero_ids
        overlap[f"{float(dose):.2f}"] = {
            "overlap_with_zero_dose_arm": len(shared),
            "overlap_fraction_with_zero_dose_arm": len(shared) / float(size),
            "abnormal_overlap_with_zero_dose_arm": len(
                {
                    int(value)
                    for value in draws_by_dose[float(dose)][reference_repeat].tolist()
                    if bool(pool.abnormal[value])
                }
                & {
                    int(value)
                    for value in draws_by_dose[zero_dose][reference_repeat].tolist()
                    if bool(pool.abnormal[value])
                }
            ),
        }
    for previous, current in zip(grid, grid[1:], strict=False):
        first = set(
            int(value)
            for value in draws_by_dose[float(previous)][reference_repeat].tolist()
        )
        second = set(
            int(value)
            for value in draws_by_dose[float(current)][reference_repeat].tolist()
        )
        overlap[f"{float(current):.2f}"]["overlap_with_previous_dose"] = len(
            first & second
        )
        overlap[f"{float(current):.2f}"]["overlap_fraction_with_previous_dose"] = (
            len(first & second) / float(size)
        )
        overlap[f"{float(current):.2f}"]["abnormal_overlap_with_previous_dose"] = len(
            {
                int(value)
                for value in draws_by_dose[float(previous)][
                    reference_repeat
                ].tolist()
                if bool(pool.abnormal[value])
            }
            & {
                int(value)
                for value in draws_by_dose[float(current)][reference_repeat].tolist()
                if bool(pool.abnormal[value])
            }
        )
    # All-repeat overlap summaries over the abnormal and ordinary partitions.
    all_repeat: dict[str, Any] = {}
    maximum = float(grid[-1])
    for dose in grid:
        if float(dose) == zero_dose:
            continue
        current_abnormal = int(abnormal_counts[float(dose)])
        current_ordinary = int(ordinary_counts[float(dose)])
        record: dict[str, Any] = {
            "n_abnormal": current_abnormal,
            "n_ordinary": current_ordinary,
        }
        for label, other in (
            ("zero_dose", zero_dose),
            ("previous_dose", None),
            ("max_dose", maximum),
        ):
            if label == "previous_dose":
                lower = [
                    float(value) for value in grid if float(value) < float(dose)
                ]
                if not lower:
                    continue
                other = lower[-1]
            if other is None:
                continue
            if other == dose:
                continue
            abnormal_shared = (
                membership[float(dose)]["abnormal"]
                & membership[float(other)]["abnormal"]
            ).sum(axis=1)
            ordinary_shared = (
                membership[float(dose)]["ordinary"]
                & membership[float(other)]["ordinary"]
            ).sum(axis=1)
            record[f"abnormal_overlap_fraction_with_{label}"] = {
                "mean": float(abnormal_shared.mean() / max(current_abnormal, 1)),
                "median": float(np.median(abnormal_shared) / max(current_abnormal, 1)),
            }
            record[f"ordinary_overlap_fraction_with_{label}"] = {
                "mean": float(ordinary_shared.mean() / max(current_ordinary, 1)),
                "median": float(np.median(ordinary_shared) / max(current_ordinary, 1)),
            }
        all_repeat[f"{float(dose):.2f}"] = record
    # Identity-level nesting of successive positive-dose arms.
    nesting_checked = 0
    nesting_failures_reference = 0
    nesting_failures_all_repeats = 0
    examples: list[dict[str, Any]] = []
    for previous, current in zip(grid, grid[1:], strict=False):
        if current <= 0.0 or previous == current:
            continue
        lower_abnormal = membership[float(previous)]["abnormal"]
        upper_abnormal = membership[float(current)]["abnormal"]
        missing = (lower_abnormal & ~upper_abnormal).sum(axis=1)
        nesting_checked += 1
        nesting_failures_reference += int(missing[reference_repeat] > 0)
        nesting_failures_all_repeats += int(np.sum(missing > 0))
        if len(examples) < 3 and missing[reference_repeat] > 0:
            examples.append(
                {
                    "from_dose": float(previous),
                    "to_dose": float(current),
                    "missing_at_reference_repeat": int(missing[reference_repeat]),
                }
            )
    distinct: set[int] = set()
    for values in draws_by_dose.values():
        distinct.update(int(value) for value in np.unique(values).tolist())
    return {
        "cell": cell,
        "n_distinct_source_files_used": len(distinct),
        "n_pool_files": len(pool.ids),
        "n_repeat_slot_draws": int(repeats * size * len(grid)),
        "overlap_reference_repeat": reference_repeat,
        "overlap": overlap,
        "overlap_across_repeats": all_repeat,
        "nesting": {
            "planned_prefix_nested": True,
            "realized_identity_sets_nested": nesting_failures_all_repeats == 0,
            "positive_transitions_checked": nesting_checked,
            "reference_repeat_failures": nesting_failures_reference,
            "all_repeat_failures": nesting_failures_all_repeats,
            "examples": examples,
        },
    }


def _trends(
    curves: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per unit, seed and estimator dose curves and trend descriptions."""
    computed_units = [unit for unit in units if unit["status"] == COMPUTED]
    by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in curves:
        by_key.setdefault(
            (str(record["unit"]), int(record["model_seed"])), []
        ).append(record)
    results: list[dict[str, Any]] = []
    for unit in computed_units:
        unit_id = str(unit["unit"])
        grid = [float(value) for value in unit["doses"]]  # type: ignore[arg-type]
        for model_seed in readiness.FROZEN_ENCODER_SEEDS:
            records = {
                float(record["dose"]): record
                for record in by_key.get((unit_id, int(model_seed)), [])
            }
            entry: dict[str, Any] = {
                "unit": unit_id,
                "cell": unit["cell"],
                "horizon": unit["horizon"],
                "design": unit["design"],
                "size": unit["size"],
                "model_seed": int(model_seed),
                "trends": {},
            }
            for estimator in ESTIMATORS:
                medians: list[float | None] = []
                null_q95: float | None = None
                for dose in grid:
                    record = records.get(float(dose))
                    estimator_record = (
                        record["estimators"].get(estimator) if record else None
                    )
                    if not estimator_record or estimator_record.get("status") != COMPUTED:
                        medians.append(None)
                        continue
                    medians.append(float(estimator_record["median"]))
                    if float(dose) == 0.0 and null_q95 is None:
                        null_q95 = float(estimator_record["q95"])
                entry["trends"][estimator] = trend_record(grid, medians, null_q95)
            results.append(entry)
    return results


def _aggregates(trends: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive aggregates per horizon, size, design and estimator.

    Horizons, sizes and designs are never pooled. Counts are descriptive; no
    seed or cell is selected as best.
    """
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for entry in trends:
        key = (str(entry["horizon"]), int(entry["size"]), str(entry["design"]))
        groups.setdefault(key, []).append(entry)
    aggregates: list[dict[str, Any]] = []
    for key in sorted(groups):
        entries = groups[key]
        cells = sorted({str(entry["cell"]) for entry in entries})
        for estimator in ESTIMATORS:
            per_seed: dict[str, dict[str, int]] = {}
            for model_seed in readiness.FROZEN_ENCODER_SEEDS:
                counts = {"positive": 0, "zero": 0, "negative": 0, "undefined": 0}
                for entry in entries:
                    if int(entry["model_seed"]) != int(model_seed):
                        continue
                    rho = entry["trends"][estimator]["rho"]
                    if rho is None:
                        counts["undefined"] += 1
                    elif rho > 0.0:
                        counts["positive"] += 1
                    elif rho < 0.0:
                        counts["negative"] += 1
                    else:
                        counts["zero"] += 1
                per_seed[str(model_seed)] = counts
            seed_median_rho: list[float | None] = []
            all_positive = 0
            all_negative = 0
            all_zero = 0
            for cell in cells:
                rhos = [
                    entry["trends"][estimator]["rho"]
                    for entry in entries
                    if entry["cell"] == cell
                ]
                defined = [float(value) for value in rhos if value is not None]
                seed_median_rho.append(
                    float(np.median(defined)) if defined else None
                )
                if len(defined) == len(readiness.FROZEN_ENCODER_SEEDS):
                    if all(value > 0.0 for value in defined):
                        all_positive += 1
                    elif all(value < 0.0 for value in defined):
                        all_negative += 1
                    elif all(value == 0.0 for value in defined):
                        all_zero += 1
            clean = [value for value in seed_median_rho if value is not None]
            aggregates.append(
                {
                    "horizon": key[0],
                    "size": key[1],
                    "design": key[2],
                    "estimator": estimator,
                    "n_cells": len(cells),
                    "n_cells_with_defined_seed_median": len(clean),
                    "median_seed_median_rho": (
                        float(np.median(clean)) if clean else None
                    ),
                    "seed_median_rho_positive": int(sum(1 for v in clean if v > 0.0)),
                    "seed_median_rho_zero": int(sum(1 for v in clean if v == 0.0)),
                    "seed_median_rho_negative": int(sum(1 for v in clean if v < 0.0)),
                    "cells_all_seeds_positive": all_positive,
                    "cells_all_seeds_negative": all_negative,
                    "cells_all_seeds_zero": all_zero,
                    "per_seed_counts": per_seed,
                }
            )
    return aggregates


def _accounting(
    units: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    composition: Sequence[Mapping[str, Any]],
    repeats: int,
    *,
    n_fits: int,
) -> dict[str, Any]:
    """Account for every planned unit, dose record and estimator cell."""
    planned_units = len(units)
    computed_units = [unit for unit in units if unit["status"] == COMPUTED]
    abstained_units = [unit for unit in units if unit["status"] != COMPUTED]
    planned_dose_records = 0
    for unit in units:
        planned_dose_records += len(unit["doses"]) * len(  # type: ignore[arg-type]
            readiness.FROZEN_ENCODER_SEEDS
        )
    computed_dose_records = sum(
        1 for record in curves if record.get("status") == COMPUTED
    )
    abstained_dose_records = sum(
        1 for record in curves if record.get("status") != COMPUTED
    )
    planned_estimator_cells = planned_dose_records * len(ESTIMATORS)
    computed_estimator_cells = 0
    for record in curves:
        if record.get("status") != COMPUTED:
            continue
        computed_estimator_cells += sum(
            1
            for estimator in ESTIMATORS
            if record["estimators"].get(estimator, {}).get("status") == COMPUTED
        )
    reasoned: dict[str, int] = {}
    for unit in abstained_units:
        status = str(unit["status"])
        reasoned[status] = reasoned.get(status, 0) + 1
    planned_composition = sum(len(unit["doses"]) for unit in computed_units)  # type: ignore[arg-type]
    return {
        "planned_units": planned_units,
        "computed_units": len(computed_units),
        "abstained_units": len(abstained_units),
        "abstained_unit_reasons": dict(sorted(reasoned.items())),
        "planned_dose_records": planned_dose_records,
        "computed_dose_records": computed_dose_records,
        "abstained_dose_records": abstained_dose_records,
        "planned_estimator_cells": planned_estimator_cells,
        "computed_estimator_cells": computed_estimator_cells,
        "abstained_estimator_cells": planned_estimator_cells
        - computed_estimator_cells,
        "planned_composition_records": planned_composition,
        "computed_composition_records": len(composition),
        "repeats": repeats,
        "n_fits": n_fits,
        "all_planned_cells_accounted": bool(
            len(computed_units) + len(abstained_units) == planned_units
            and computed_dose_records + abstained_dose_records
            == planned_dose_records
            and len(composition) == planned_composition
        ),
        "records_checked": len(curves),
    }
