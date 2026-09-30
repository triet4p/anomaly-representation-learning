"""Focused Task 3 contract tests: toy fixture, fail-closed gates (no checkpoints).

Covers the consumer-visible validation surface of
``experiments/sprint21_disposable_scorer_proof.py`` without loading any
historical checkpoint, contacting any history/root, or requiring the frozen
CUDA runtime. The real-checkpoint toy smoke remains the separate,
review-authorized proof step.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import sprint21_disposable_scorer_proof as proof


def test_toy_arrays_match_frozen_tensor_digests() -> None:
    roster = proof.build_toy_roster()
    assert [r.file_id for r in roster] == ["S21T2-TOY-A", "S21T2-TOY-B"]
    for record, spec in zip(roster, proof.TOY_SPECS):
        assert record.robot_idx == spec["robot_idx"]
        assert record.program_idx == spec["program_idx"]
        assert proof.tensor_digest(record.x) == spec["tensor_sha256"]
        assert proof.file_seed(record.file_id) == spec["file_seed"]


def _static_profile_config() -> SimpleNamespace:
    robot_ids = [
        "robot-01", "robot-02", "robot-03", "robot-04", "robot-05",
        "robot-06", "robot-07", "robot-08", "robot-09",
    ]
    program_ids = [
        "program-01", "program-02", "program-03", "program-04",
        "program-05", "program-06", "program-07", "program-08",
    ]
    stages = [
        SimpleNamespace(robot_id=robot_id, program_id=program_id)
        for robot_id, program_id in zip(robot_ids, [*program_ids, "program-01"])
    ]
    return SimpleNamespace(
        scheduler=SimpleNamespace(routes=[SimpleNamespace(stages=stages)]),
        fleet=SimpleNamespace(n_robots=len(robot_ids), n_programs=len(program_ids)),
    )


def test_conditioning_mapping_is_derived_from_profile_and_rejects_drift() -> None:
    config = _static_profile_config()
    mapping, digest = proof.conditioning_mapping_from_config(config)
    assert mapping["robot_id_to_idx"]["robot-09"] == 8
    assert mapping["program_id_to_idx"]["program-08"] == 7
    assert digest == proof.canonical_hash(mapping)

    config.scheduler.routes[0].stages.pop()
    with pytest.raises(proof.ProofFailure) as exc:
        proof.conditioning_mapping_from_config(config)
    assert exc.value.disposition == "CONDITIONING_MISMATCH"


def test_uv_lock_digest_gate_rejects_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"pinned lock bytes")
    expected = proof.sha256_file(lock)
    monkeypatch.setattr(proof, "EXPECTED_UV_LOCK_SHA256", expected)
    assert proof.verify_uv_lock_digest(tmp_path) == expected

    lock.write_bytes(b"changed lock bytes")
    with pytest.raises(proof.ProofFailure) as exc:
        proof.verify_uv_lock_digest(tmp_path)
    assert exc.value.disposition == "RESTORE_ONLY_OR_ISOLATION_FAILURE"


def test_metric_code_gate_rejects_member_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_path = "src/synth/probe15.py"
    metric = tmp_path / relative_path
    metric.parent.mkdir(parents=True)
    metric.write_bytes(b"frozen metric")
    expected = proof.canonical_hash([[relative_path, proof.sha256_file(metric)]])
    monkeypatch.setattr(proof, "METRIC_FILES", (relative_path,))
    monkeypatch.setattr(proof, "METRIC_DIGEST", expected)
    assert proof.verify_metric_code(tmp_path)["digest"] == expected

    metric.write_bytes(b"changed metric")
    with pytest.raises(proof.ProofFailure) as exc:
        proof.verify_metric_code(tmp_path)
    assert exc.value.disposition == "RESTORE_ONLY_OR_ISOLATION_FAILURE"


def test_role_binding_gate_rejects_canonical_payload_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_path = Path("experiments/sprint17-role-binding-v3.json")
    binding_path = tmp_path / relative_path
    binding_path.parent.mkdir(parents=True)
    payload = {"profile_id": "sprint15-v7", "roles": ["FIT", "CAL", "DEV"]}
    digest = proof.canonical_hash(payload)
    document = {**payload, "binding_sha256": digest}
    binding_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(proof, "BINDING_DIGEST", digest)
    observed = proof.verify_role_binding(tmp_path)
    assert observed["digest"] == digest
    assert observed["file_sha256"] == proof.sha256_file(binding_path)
    assert observed["profile_id"] == "sprint15-v7"

    document["roles"] = ["FIT", "CAL", "CONF"]
    binding_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(proof.ProofFailure) as exc:
        proof.verify_role_binding(tmp_path)
    assert exc.value.disposition == "RESTORE_ONLY_OR_ISOLATION_FAILURE"


def test_full_b0_config_digest_requires_all_seeds_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {
        str(seed): {"model_seed": seed, "architecture": "frozen"}
        for seed in proof.MODEL_SEEDS
    }
    expected = proof.canonical_hash(configs)
    monkeypatch.setattr(proof, "CONFIG_SHA256", expected)
    assert proof.validate_b0_config_map(configs) == expected

    with pytest.raises(proof.ProofFailure) as missing:
        proof.validate_b0_config_map({str(proof.MODEL_SEEDS[0]): configs[str(proof.MODEL_SEEDS[0])]})
    assert missing.value.disposition == "PROVENANCE_MISMATCH"

    drifted = {seed: dict(config) for seed, config in configs.items()}
    drifted[str(proof.MODEL_SEEDS[-1])]["architecture"] = "drifted"
    with pytest.raises(proof.ProofFailure) as changed:
        proof.validate_b0_config_map(drifted)
    assert changed.value.disposition == "PROVENANCE_MISMATCH"



def test_b0_config_loader_uses_pinned_file_not_cached_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "experiments" / "sprint17_task7_b0.py"
    module_path.parent.mkdir()
    module_path.write_text(
        "def b0_config_dict(model_seed):\n"
        "    return {'model_seed': model_seed, 'architecture': 'pinned'}\n",
        encoding="utf-8",
    )
    expected_configs = {
        str(seed): {"model_seed": seed, "architecture": "pinned"}
        for seed in proof.MODEL_SEEDS
    }
    expected_digest = proof.canonical_hash(expected_configs)
    monkeypatch.setattr(proof, "CONFIG_SHA256", expected_digest)

    cached_package = ModuleType("experiments")
    cached_package.__path__ = []
    cached_task7 = ModuleType("experiments.sprint17_task7_b0")
    cached_task7.__file__ = str(tmp_path / "untrusted" / "sprint17_task7_b0.py")
    cached_task7.b0_config_dict = lambda seed: {
        "model_seed": seed,
        "architecture": "untrusted",
    }
    monkeypatch.setitem(sys.modules, "experiments", cached_package)
    monkeypatch.setitem(sys.modules, "experiments.sprint17_task7_b0", cached_task7)

    configs, digest = proof.load_b0_config_map(tmp_path)
    assert configs == expected_configs
    assert digest == expected_digest


def test_restored_seed_config_rejects_wrong_seed_or_field() -> None:
    from representation.config import V1Config

    seed = proof.MODEL_SEEDS[0]
    expected = V1Config(
        n_channels=6, patch_size=32, stride=16, d_model=128, seed=seed
    ).to_dict()
    assert proof.validate_restored_config(expected, expected, seed) == proof.canonical_hash(expected)

    wrong_seed = dict(expected)
    wrong_seed["seed"] = proof.MODEL_SEEDS[1]
    with pytest.raises(proof.ProofFailure) as exc:
        proof.validate_restored_config(wrong_seed, expected, seed)
    assert exc.value.disposition == "PROVENANCE_MISMATCH"

    changed_field = dict(expected)
    changed_field["patch_size"] = 16
    with pytest.raises(proof.ProofFailure) as changed:
        proof.validate_restored_config(changed_field, expected, seed)
    assert changed.value.disposition == "PROVENANCE_MISMATCH"


def test_conditioning_mismatch_fails_closed() -> None:
    mapping, _ = proof.conditioning_mapping_from_config(_static_profile_config())
    roster = proof.build_toy_roster()
    base = roster[0]
    with pytest.raises(proof.ProofFailure) as exc:
        proof.validate_conditioning(
            proof.ToyRecord(base.file_id, "robot-99", 99, base.program_id, base.program_idx, base.x),
            mapping,
        )
    assert exc.value.disposition == "CONDITIONING_MISMATCH"
    with pytest.raises(proof.ProofFailure) as exc2:
        proof.validate_conditioning(
            proof.ToyRecord(base.file_id, base.robot_id, 5, base.program_id, base.program_idx, base.x),
            mapping,
        )
    assert exc2.value.disposition == "CONDITIONING_MISMATCH"


def test_input_contract_failures_are_fail_closed() -> None:
    roster = proof.build_toy_roster()
    base = roster[0]
    bad_channels = proof.ToyRecord(
        base.file_id, base.robot_id, base.robot_idx, base.program_id, base.program_idx,
        np.zeros((5, 64), dtype=np.float32),
    )
    with pytest.raises(proof.ProofFailure) as exc:
        proof.validate_input_array(bad_channels)
    assert exc.value.disposition == "INPUT_CONTRACT_FAILURE"
    poisoned = np.array(base.x, dtype=np.float32, copy=True)
    poisoned[0, 0] = np.inf
    bad_finite = proof.ToyRecord(
        base.file_id, base.robot_id, base.robot_idx, base.program_id, base.program_idx, poisoned
    )
    with pytest.raises(proof.ProofFailure) as exc2:
        proof.validate_input_array(bad_finite)
    assert exc2.value.disposition == "INPUT_CONTRACT_FAILURE"


def test_full_roster_invariant_rejects_partial_macro_rescue() -> None:
    roster = proof.build_toy_roster()
    ids = [r.file_id for r in roster]
    with pytest.raises(proof.ProofFailure) as exc:
        proof.check_full_roster(ids, {ids[0]: {"S_pred": 0.1, "S_pop": 0.2}})
    assert exc.value.disposition == "ROSTER_OR_SCORE_INCOMPLETE"
    with pytest.raises(proof.ProofFailure) as exc2:
        proof.check_full_roster(
            [ids[0], ids[0]], {ids[0]: {"S_pred": 0.1, "S_pop": 0.2}}
        )
    assert exc2.value.disposition == "ROSTER_OR_SCORE_INCOMPLETE"
    with pytest.raises(proof.ProofFailure) as exc3:
        proof.check_full_roster(ids, {fid: {"S_pred": 0.1} for fid in ids})
    assert exc3.value.disposition == "ROSTER_OR_SCORE_INCOMPLETE"
    # Complete roster passes.
    proof.check_full_roster(ids, {fid: {"S_pred": 0.1, "S_pop": 0.2} for fid in ids})


def test_production_collator_rejects_wrong_seed_and_replays_canonical_mask() -> None:
    from representation.config import V1Config
    from synth.config import PatchConfig
    from synth.patchify import Patchifier

    roster = proof.build_toy_roster()
    mapping, _ = proof.conditioning_mapping_from_config(_static_profile_config())
    patchifier = Patchifier(PatchConfig(patch_size=32, stride=16, pad_end=True))
    cfg = V1Config(n_channels=6, patch_size=32, stride=16, d_model=128)
    for record in roster:
        batch, seed, bits = proof.collate_toy_record(record, patchifier, cfg, mapping)
        assert seed == proof.file_seed(record.file_id)
        assert batch["patch_valid_mask"][0].numpy().astype(bool).tolist() == [True, True, True]
        assert sum(bits) == 1
        replay, _, replay_bits = proof.collate_toy_record(record, patchifier, cfg, mapping)
        assert replay_bits == bits
        assert (replay["mask"][0].numpy() == batch["mask"][0].numpy()).all()

    with pytest.raises(proof.ProofFailure) as exc:
        proof.collate_toy_record(
            roster[0],
            patchifier,
            cfg,
            mapping,
            masking_seed=proof.file_seed(roster[0].file_id) + 1,
        )
    assert exc.value.disposition == "MASK_CONTRACT_FAILURE"
