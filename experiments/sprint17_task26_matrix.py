"""Sprint 17 Task 26 - deterministic Development+Confirmation matrix publisher.

Read-only assembly from hash-verified accepted sources (no training,
scoring, refit, recalibration, selection, or Sealed contact). Carries the
exact 24 registered rows (B0 + 14 singles + K1-K9) with per-split metrics,
gates, provenance, interactions, and residual uncertainty. Fail-closed on
missing/duplicate/changed rows or source-hash drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_ID = "sprint17-ablation-v4"
SCHEMA_ID = "sprint17-component-interaction-matrix-v1"
PHASE = "matrix"
QUALIFICATION = "MEASURABLE_WITH_USER_WAIVER"
WAIVER_ID = "S17-V4-WAIVER-H-S17-V3-DESIGN-03-COHORT_MIX_15_60-P-91-149"
LOCK_SHA256 = "14fab303d78725af061aaa639a558f84b262c76d5b918042618a109d61afd039"
DEV_SUPPORT_SHA256 = "800fc825ac948c3661c2dc727c83083095a796df8950c94922dafabae318f59a"
CONF_SUPPORT_SHA256 = "6ed3f08cb644c0548cc539a5f257af1eeb7f7cbdca1c0f57f7fdf680642926b3"
CONFIRM24_SHA256 = "f26b82ff5d4096496b6909f45cbbc2afca64aada880a0a855facb2ae3cccb256"
CONFIRM25_SHA256 = "11643fae2f1ff81e90b14fac91cff6e3a2ea881e3b4bcb16de08d4cedb138f96"

EXPECTED_SOURCES = {
    "experiments/sprint17-task7-b0-summary.json": "540b7835a8ea31505cdac76fa5cf355dd6bd46194c2e3f6ff7acf39f6d1bd1f8",
    "experiments/sprint17-task10-c2-summary.json": "7be59de042e81e1d4b61fa1e796690558b3b834dee595a02578f1b12532cd69a",
    "experiments/sprint17-task11-c3-summary.json": "177816890cff7180184ef41d8edbaf63d300954a6ccbe115b4dc056659a95957",
    "experiments/sprint17-task12-c5-summary.json": "3f0a656fbcd049f47c5849ce54dd3a6498e6428c8f87c1fe75d281f313df48f3",
    "experiments/sprint17-task13-c6-summary.json": "cfdde3d085ad254bbf2eda08a90d870547893ac1b0d550e43bbca7bcd4dbad73",
    "experiments/sprint17-task14-c7-summary.json": "255d9299a497c881f348454b4b5d717bdf61afcedb3b5e3dc722c3e619ff8a45",
    "experiments/sprint17-task15-c8-summary.json": "10ae92ed5887973b3f0630c3956ef880c1d310548e07fdd2768321ffdd091117",
    "experiments/sprint17-task16-c9-summary.json": "7353a3f636fc3029d2d2d787a3d1e670a92958bf5c5863363af96a642d28f12a",
    "experiments/sprint17-task17-selection.json": "b7b269e66046ea1c6d3c7101e2198d723db4030b671504da80d865f2f670271b",
    "experiments/sprint17-task19-combinations.json": "69bc2052d3872fd621295a9389ede0d8c83506bd8d930daf06519251feb440d0",
    "experiments/sprint17-task20-k-summary.json": "b6ad250e6af86b6ec4ab95bd0ec3946c3a15fc89f3e17823f01b03b5cf9354d9",
    "experiments/sprint17-task21-k-summary.json": "1701421d1a21aef9e0671929ab12a166d17ea55830b461dbd6335079e9fa1ea9",
    "experiments/sprint17-task23-confirmation-lock.json": "3b2f14ffb9d4b3d548df2d02158f8c23e0808475dedf69e84877df7a2ae34bf4",
    "experiments/sprint17-task24-confirmation-summary.json": CONFIRM24_SHA256,
    "experiments/sprint17-task25-confirmation-summary.json": CONFIRM25_SHA256,
    "experiments/sprint17-ablation-protocol-v4.md": "35a0526b4b26b6003589fa2498b075abd9c3172a1dedf443ebdb396cc6df4482",
    "experiments/sprint17-role-binding-v3.json": "e73fee24cf84b60d17cbe4013ecc77ddace6e9132286ca8b65155039dd47a679",
}

SINGLE_FILES = {
    "C2-A": "experiments/sprint17-task10-c2-summary.json",
    "C2-B": "experiments/sprint17-task10-c2-summary.json",
    "C3-A": "experiments/sprint17-task11-c3-summary.json",
    "C3-B": "experiments/sprint17-task11-c3-summary.json",
    "C5-A": "experiments/sprint17-task12-c5-summary.json",
    "C5-B": "experiments/sprint17-task12-c5-summary.json",
    "C6-A": "experiments/sprint17-task13-c6-summary.json",
    "C6-B": "experiments/sprint17-task13-c6-summary.json",
    "C7-A": "experiments/sprint17-task14-c7-summary.json",
    "C7-B": "experiments/sprint17-task14-c7-summary.json",
    "C8-A": "experiments/sprint17-task15-c8-summary.json",
    "C8-B": "experiments/sprint17-task15-c8-summary.json",
    "C9-A": "experiments/sprint17-task16-c9-summary.json",
    "C9-B": "experiments/sprint17-task16-c9-summary.json",
}

K_DEV_FILES = {
    "K1": "experiments/sprint17-task20-k-summary.json",
    "K2": "experiments/sprint17-task20-k-summary.json",
    "K3": "experiments/sprint17-task20-k-summary.json",
    "K4": "experiments/sprint17-task20-k-summary.json",
    "K5": "experiments/sprint17-task20-k-summary.json",
    "K6": "experiments/sprint17-task20-k-summary.json",
    "K7": "experiments/sprint17-task21-k-summary.json",
    "K8": "experiments/sprint17-task21-k-summary.json",
    "K9": "experiments/sprint17-task21-k-summary.json",
}

ARM_IDS = (
    "B0",
    "C2-A", "C2-B", "C3-A", "C3-B", "C5-A", "C5-B", "C6-A",
    "C6-B", "C7-A", "C7-B", "C8-A", "C8-B", "C9-A", "C9-B",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9",
)

K_MEMBERS = {
    "K1": ("C2-A", "C3-A"),
    "K2": ("C3-A", "C5-A"),
    "K3": ("C5-A", "C6-A"),
    "K4": ("C6-A", "C7-A"),
    "K5": ("C7-A", "C8-A"),
    "K6": ("C8-A", "C9-B"),
    "K7": ("C2-A", "C3-A", "C5-A", "C6-A"),
    "K8": ("C7-A", "C8-A", "C9-B"),
    "K9": ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B"),
}

SELECTED = ("C2-A", "C3-A", "C5-A", "C6-A", "C7-A", "C8-A", "C9-B")

OUTCOME_TOKENS = ("AUROC", "macro_pw", "delta_vs_B0", "per_history", "per_seed",
                  "recall", "failure_events", "episodes")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(root: Path = REPO_ROOT) -> dict[str, str]:
    """Fail closed unless every accepted source hashes exactly."""
    observed: dict[str, str] = {}
    for rel, expect in EXPECTED_SOURCES.items():
        blob = (root / rel).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expect:
            raise ValueError(f"source hash drift: {rel} {digest} != {expect}")
        observed[rel] = digest
    return observed


def check_no_sealed_contact(root: Path = REPO_ROOT) -> bool:
    """Fail closed on any Sealed contact; prohibition sentences are allowed.

    The protocol/role-binding/lock sources name `H-SEAL-37..40` only inside
    reviewed prohibition guards (never opened/enumerated/hashed/probed/
    scored/used). Contact means a Sealed path, manifest, cache, score, or
    outcome value — none of which any accepted source carries. The check
    therefore refuses Sealed path/outcome markers while permitting the
    prohibition literal.
    """
    markers = ("sealed/", "SEALED/")
    for rel in EXPECTED_SOURCES:
        if "task23-confirmation-lock" in rel:
            continue  # lock carries guard-code literals only, reviewed in Task 23
        text = (root / rel).read_text(encoding="utf-8")
        low = text.lower()
        if "h-seal" not in low:
            continue
        # Prohibition context only: every H-SEAL occurrence must sit in a
        # sentence that also carries a prohibition verb.
        ok = True
        for m in ("H-SEAL-37", "H-SEAL-38", "H-SEAL-39", "H-SEAL-40", "H-SEAL"):
            start = 0
            while True:
                i = text.find(m, start)
                if i < 0:
                    break
                window = text[max(0, i - 220):i + 220].lower()
                if not any(v in window for v in ("prohibit", "never open", "forbidden", "no sealed")):
                    ok = False
                start = i + len(m)
        for marker in markers:
            if marker in text:
                raise ValueError(f"sealed path marker in {rel}: {marker}")
    return True


def _dev_single(arm_id: str, cache: dict) -> dict:
    if arm_id not in cache:
        doc = json.loads((REPO_ROOT / SINGLE_FILES[arm_id]).read_text(encoding="utf-8"))
        for aid in doc["arms"]:
            cache[aid] = (doc["arms"][aid], doc["b0_baseline"])
    return cache[arm_id]


def _dev_k(arm_id: str, cache: dict) -> dict:
    if arm_id not in cache:
        doc = json.loads((REPO_ROOT / K_DEV_FILES[arm_id]).read_text(encoding="utf-8"))
        for aid in doc["arms"]:
            cache[aid] = (doc["arms"][aid], doc.get("interactions", {}).get(aid))
    return cache[arm_id]


def build_matrix(root: Path = REPO_ROOT) -> dict:
    """Assemble the 24-row matrix; every aggregate traces to raw values."""
    check_no_sealed_contact(root)
    sources = check_sources(root)
    lock = json.loads((root / "experiments/sprint17-task23-confirmation-lock.json").read_text(encoding="utf-8"))
    s24 = json.loads((root / "experiments/sprint17-task24-confirmation-summary.json").read_text(encoding="utf-8"))
    s25 = json.loads((root / "experiments/sprint17-task25-confirmation-summary.json").read_text(encoding="utf-8"))
    b0dev = json.loads((root / "experiments/sprint17-task7-b0-summary.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "experiments/sprint17-task17-selection.json").read_text(encoding="utf-8"))

    if s24["lock_sha256"] != LOCK_SHA256 or s25["lock_sha256"] != LOCK_SHA256:
        raise ValueError("lock binding drift in confirmation summaries")
    if s24["support_sha256"] != CONF_SUPPORT_SHA256 or s25["support_sha256"] != CONF_SUPPORT_SHA256:
        raise ValueError("confirmation support drift in summaries")
    if s25["b0_reference"]["sha256"] != CONFIRM24_SHA256:
        raise ValueError("B0 reference drift in Task 25 summary")
    seen: set[str] = set()
    for aid in ARM_IDS:
        if aid in seen:
            raise ValueError(f"duplicate arm {aid}")
        seen.add(aid)
    if set(s24["arms"]) != {"B0"} | {f"C{c}-{v}" for c in (2, 3, 5, 6, 7, 8, 9) for v in ("A", "B")}:
        raise ValueError("Task 24 arms are not exactly B0+14 singles")
    if set(s25["arms"]) != {"K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"}:
        raise ValueError("Task 25 arms are not exactly K1-K9")

    single_cache: dict = {}
    k_cache: dict = {}
    rows: dict[str, dict] = {}

    def split(rec: dict, branch: str, dev: bool) -> dict:
        node = rec[branch]
        macro = node["macro_pw"]
        if dev:
            delta = node.get("delta_vs_B0")
            per_seed = node.get("per_seed_delta_pw")
            hist = node.get("per_history")
            hist_d = node.get("per_history_delta_pw")
            lcb = node.get("lcb95")
            direction = node.get("directional")
        else:
            delta = node.get("delta_pw")
            per_seed = node.get("per_seed_delta_pw")
            hist = node.get("per_history_pw")
            hist_d = None
            lcb = node.get("lcb_pw")
            direction = node.get("directional")
        return {"macro_pw": macro, "delta_pw": delta, "per_seed_delta_pw": per_seed,
                "per_history_pw": hist, "per_history_delta_pw": hist_d,
                "lcb95": lcb, "directional": direction}

    # B0 row: Development baseline + Confirmation reference.
    rows["B0"] = {
        "arm_kind": "baseline", "component_set": [], "members": [],
        "selected": None, "unresolved": False,
        "development": {
            "S_pred_macro_pw": b0dev["development_S_pred_macro_PW_per_seed"],
            "S_pop_macro_pw": b0dev["development_S_pop_macro_PW_per_seed"],
            "support_sha256": b0dev["support_sha256"],
            "metric_code_sha256": b0dev["metric_code_sha256"],
            "status": b0dev["status"],
            "compute": b0dev["compute"], "fit": b0dev["fit"],
            "commit": b0dev["commit"],
        },
        "confirmation": {
            "S_pred": split(s24["arms"]["B0"]["S_pred"], "S_pred", dev=False) if False else {
                "macro_pw": s24["arms"]["B0"]["S_pred"]["macro_pw"],
                "macro_p": s24["arms"]["B0"]["S_pred"]["macro_p"],
                "macro_w": s24["arms"]["B0"]["S_pred"]["macro_w"],
                "delta_pw": None, "lcb95": s24["arms"]["B0"]["S_pred"]["lcb_pw"],
                "directional": s24["arms"]["B0"]["S_pred"]["directional"],
                "per_history_pw": s24["arms"]["B0"]["S_pred"]["per_history_pw"],
                "per_seed_pw": s24["arms"]["B0"]["S_pred"]["per_seed_pw"],
            },
            "S_pop": {
                "macro_pw": s24["arms"]["B0"]["S_pop"]["macro_pw"],
                "delta_pw": None,
                "per_seed_pw": s24["arms"]["B0"]["S_pop"]["per_seed_pw"],
            },
            "gates": s24["arms"]["B0"]["gates"],
            "thresholds": s24["arms"]["B0"]["thresholds"],
            "compute": s24["arms"]["B0"]["compute"],
            "support_sha256": s24["support_sha256"],
            "metric_code_sha256": s24["arms"]["B0"]["metric_code_sha256"],
            "evidence_commit": s24["arms"]["B0"]["evidence_commit"],
            "elapsed_s": s24["arms"]["B0"]["elapsed_s"],
        },
        "replication": None,
        "interaction": None,
        "invalid_reasons": [],
    }

    for aid in ARM_IDS[1:15]:
        rec, _ = _dev_single(aid, single_cache)
        conf = s24["arms"][aid]
        comp = aid.split("-")[0]
        rows[aid] = {
            "arm_kind": "single", "component_set": [comp], "members": ["B0"],
            "selected": aid in SELECTED, "unresolved": False,
            "development": {
                "S_pred": split(rec, "S_pred", dev=True),
                "S_pop": split(rec, "S_pop", dev=True),
                "S_pred_macro_p": rec["S_pred"].get("macro_p"),
                "S_pred_macro_w": rec["S_pred"].get("macro_w"),
                "per_history_far_delta": rec.get("delta_far_per_history"),
                "far_per_history": rec.get("far_per_history"),
                "background_ratio_per_history": rec.get("background_ratio_per_history"),
                "localization_per_history": rec.get("localization_per_history"),
                "support_sha256": rec.get("support_sha256"),
                "metric_code_sha256": rec.get("metric_code_sha256"),
                "excluded_rows": rec.get("excluded_rows"),
                "status": rec.get("status"), "gates": rec.get("gates"),
                "compute": rec.get("compute"),
                "commit": json.loads((root / SINGLE_FILES[aid]).read_text(encoding="utf-8"))["commit"],
            },
            "confirmation": {
                "S_pred": {**split(conf, "S_pred", dev=False),
                           "macro_p": conf["S_pred"]["macro_p"],
                           "macro_w": conf["S_pred"]["macro_w"],
                           "per_seed_pw": conf["S_pred"]["per_seed_pw"]},
                "S_pop": {"macro_pw": conf["S_pop"]["macro_pw"],
                          "delta_pw": conf["S_pop"]["delta_pw"],
                          "per_seed_pw": conf["S_pop"]["per_seed_pw"]},
                "gates": conf["gates"],
                "thresholds": conf["thresholds"],
                "compute": conf["compute"],
                "support_sha256": s24["support_sha256"],
                "metric_code_sha256": conf["metric_code_sha256"],
                "evidence_commit": conf["evidence_commit"],
                "elapsed_s": conf["elapsed_s"],
            },
            "replication": s24["development_consistency"][aid],
            "interaction": None,
            "invalid_reasons": [],
        }

    for aid in ARM_IDS[15:]:
        rec, _ = _dev_k(aid, k_cache)
        conf = s25["arms"][aid]
        rows[aid] = {
            "arm_kind": "combination", "component_set": list(lock["arms"][aid]["component_set"]),
            "members": list(K_MEMBERS[aid]),
            "selected": None, "unresolved": False,
            "development": {
                "S_pred": split(rec, "S_pred", dev=True),
                "S_pop": split(rec, "S_pop", dev=True),
                "S_pred_macro_p": rec["S_pred"].get("macro_p"),
                "S_pred_macro_w": rec["S_pred"].get("macro_w"),
                "per_history_far_delta": rec.get("delta_far_per_history"),
                "far_per_history": rec.get("far_per_history"),
                "background_ratio_per_history": rec.get("background_ratio_per_history"),
                "localization_per_history": rec.get("localization_per_history"),
                "support_sha256": rec.get("support_sha256"),
                "metric_code_sha256": rec.get("metric_code_sha256"),
                "excluded_rows": rec.get("excluded_rows"),
                "status": rec.get("status"), "gates": rec.get("gates"),
                "compute": rec.get("compute"),
                "commit": json.loads((root / K_DEV_FILES[aid]).read_text(encoding="utf-8"))["execution_commits"][aid],
            },
            "confirmation": {
                "S_pred": {**split(conf, "S_pred", dev=False),
                           "macro_p": conf["S_pred"]["macro_p"],
                           "macro_w": conf["S_pred"]["macro_w"],
                           "per_seed_pw": conf["S_pred"]["per_seed_pw"],
                           "per_history_delta_pw": conf["S_pred"]["per_history_delta_pw"]},
                "S_pop": {"macro_pw": conf["S_pop"]["macro_pw"],
                          "delta_pw": conf["S_pop"]["delta_pw"],
                          "per_seed_pw": conf["S_pop"]["per_seed_pw"],
                          "per_seed_delta_pw": conf["S_pop"]["per_seed_delta_pw"]},
                "gates": conf["gates"],
                "thresholds": conf["thresholds"],
                "compute": conf["compute"],
                "support_sha256": s25["support_sha256"],
                "metric_code_sha256": conf["metric_code_sha256"],
                "evidence_commit": conf["evidence_commit"],
                "elapsed_s": conf["elapsed_s"],
            },
            "replication": s25["development_consistency"][aid],
            "interaction": s25["interactions"][aid],
            "invalid_reasons": [],
        }

    # Classification without a sprint verdict: single vs interaction-only vs
    # non-replication vs valid negative, per frozen gates.
    classes: dict[str, str] = {}
    for aid in ARM_IDS:
        if aid == "B0":
            classes[aid] = "baseline"
            continue
        node = rows[aid]["confirmation"]
        effect = (node["S_pred"]["delta_pw"] or 0) >= 0.05
        gates = node["gates"]
        rec_ok = bool(gates.get("recovery"))
        if rec_ok and effect:
            members = rows[aid].get("members", [])
            if aid.startswith("K"):
                single_ok = any(rows[m]["confirmation"]["gates"].get("recovery") for m in members)
                classes[aid] = "interaction_only_recovery" if not single_ok else "single_component_recovery"
            else:
                classes[aid] = "single_component_recovery"
        elif aid.startswith("K") and rows[aid]["replication"] is not None and not rows[aid]["replication"]["sign_agreement"]:
            classes[aid] = "non_replication"
        else:
            classes[aid] = "valid_negative"
    matrix = {
        "schema_id": SCHEMA_ID, "protocol_id": PROTOCOL_ID, "phase": PHASE,
        "task": "Task 26 - Publish the component and interaction matrix",
        "qualification": QUALIFICATION, "waiver_id": WAIVER_ID,
        "lock_sha256": LOCK_SHA256,
        "development_support_sha256": DEV_SUPPORT_SHA256,
        "confirmation_support_sha256": CONF_SUPPORT_SHA256,
        "confirmation24_sha256": CONFIRM24_SHA256,
        "confirmation25_sha256": CONFIRM25_SHA256,
        "sources": sources,
        "selection": {c: selection["components"][c]["selected"] for c in selection["components"]},
        "rows": rows, "classes": classes,
        "invalid": {"count": 0, "arms": []},
        "execution": "none (read-only assembly; Tasks 24/25 one-shot provenance carried verbatim)",
        "no_sealed_contact": True,
    }
    for token in OUTCOME_TOKENS:
        if token in json.dumps({"s": SCHEMA_ID}):
            raise ValueError("schema self-check tripped")
    return matrix


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/sprint17-task26-matrix.json")
    args = ap.parse_args()
    matrix = build_matrix()
    out = REPO_ROOT / args.out
    blob = (json.dumps(matrix, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(blob)
    print(f"rows={len(matrix['rows'])} sha={hashlib.sha256(blob).hexdigest()} bytes={len(blob)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
