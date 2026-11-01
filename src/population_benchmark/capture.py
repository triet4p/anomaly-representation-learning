"""Real generation and frozen-encoder capture for the S24-7-M20 benchmark.

This is the only module that imports torch, ``representation`` and ``synth``
generation. It reuses the accepted public paths:

- paired sources come from ``SessionGenerator.generate_paired_twin`` (M16);
- the frozen encoder forward is the same collate/mask/co-locate path used by
  ``capture_s24_file_features`` (deterministic Sprint18 mask seed, frozen
  model/bank state digests verified before and after every chunk);
- raw readout features come from ``representation.handcrafted.patch_features``
  applied to the whole valid file;
- pre-pooling items come from ``context_latents`` restricted to
  ``patch_valid_mask`` with a frozen random-Fourier-feature map.

Labels, anomaly masks and diagnostic metadata never enter the encoder batch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from population_benchmark import spec
from population_benchmark.evaluation import LENGTH_KEY, RAW_KEY, readout_key
from representation.sprint24_features import (
    assert_s24_frozen_unchanged,
    restore_s24_b0_bundle,
    s24_live_state_digests,
)

MIN_STEPS = 64
EXPECTED_CHANNELS = 6


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


# --- frozen encoder bundles ----------------------------------------------------


def restore_bundles(
    *,
    repo: Path,
    b0_root: Path,
    bank_root: Path,
    preflight_path: Path,
    device: str = "cpu",
) -> dict[int, Any]:
    """Restore the three accepted frozen B0 bundles (bytes verified before load)."""
    from representation.sprint18_harness import (
        Sprint18ArmConfig,
        arm_config_sha256,
        load_task6_binding,
        validate_arm_config,
    )

    if not Path(preflight_path).is_file():
        raise _refused("REFUSED-MISSING-PREREQUISITE", f"preflight absent: {preflight_path}")
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise _refused("REFUSED-IDENTITY", "preflight is not a mapping")
    binding = load_task6_binding(
        Path(repo) / "experiments/sprint18-task71-certified-data-handoff-S18-ITER-0005-v1.json"
    )
    lock_doc = json.loads(
        (Path(repo) / "experiments/sprint18-task14-config-lock.json").read_text(encoding="utf-8")
    )
    bundles: dict[int, Any] = {}
    for seed in spec.ENCODER_SEEDS:
        checkpoint_path = Path(b0_root) / f"seed-{seed}" / "checkpoints" / f"b0-seed{seed}-epoch100.pt"
        bank_path = Path(bank_root) / f"B0-FULL-{seed}.json.fit-bank" / "bank.pt"
        expected_checkpoint = str(preflight["checkpoints"][str(seed)])
        expected_bank_file = str(preflight["bank_files"][str(seed)])
        expected_bank = str(preflight["banks"][str(seed)])
        for path, digest in ((checkpoint_path, expected_checkpoint), (bank_path, expected_bank_file)):
            if not path.is_file():
                raise _refused("REFUSED-MISSING-PREREQUISITE", f"required accepted asset absent: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise _refused("REFUSED-IDENTITY", f"accepted asset digest mismatch: {path}")
        arm = validate_arm_config(
            Sprint18ArmConfig(**lock_doc["locked_configs"][f"B0-FULL/seed-{seed}"]),
            require_locked=True,
        )
        if int(arm.model_seed) != int(seed):
            raise _refused("REFUSED-IDENTITY", f"accepted B0 arm seed mismatch: {seed}")
        if arm_config_sha256(arm) != str(preflight["arms"][str(seed)]):
            raise _refused("REFUSED-IDENTITY", f"accepted arm-config digest mismatch: {seed}")
        bundles[int(seed)] = restore_s24_b0_bundle(
            binding=binding,
            arm=arm,
            checkpoint_path=checkpoint_path,
            bank_path=bank_path,
            expected_checkpoint_sha256=expected_checkpoint,
            expected_bank_sha256=expected_bank,
            device=device,
        )
    return bundles


# --- readout item construction -------------------------------------------------


def raw_item(sample: Any) -> np.ndarray:
    """Deterministic whole-file raw-signal descriptor (label-blind)."""
    from representation.handcrafted import patch_features

    values = np.asarray(sample.x, dtype=np.float64)
    pad = np.zeros(values.shape[1], dtype=bool)
    item = patch_features(values, pad)
    return np.asarray(item, dtype=np.float32)


def length_item(sample: Any, n_patches: int, n_valid: int) -> np.ndarray:
    return np.asarray([float(sample.T), float(n_patches), float(n_valid)], dtype=np.float32)


@dataclass(frozen=True)
class RFFMap:
    """Frozen Gaussian random-Fourier-feature map for the pre-pooling readout."""

    dim: int
    seed: int
    omega: np.ndarray = field(compare=False)
    phase: np.ndarray = field(compare=False)
    sigma: float
    latent_dim: int

    @classmethod
    def create(cls, latent_dim: int, sigma: float, dim: int = spec.RFF_DIM, seed: int = spec.RFF_SEED) -> "RFFMap":
        if latent_dim <= 0 or dim <= 0:
            raise _refused("REFUSED-UNKNOWN-DESIGN", "RFF dimensions must be positive")
        if not math.isfinite(sigma) or sigma <= 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", "RFF bandwidth must be finite and positive")
        rng = np.random.default_rng(int(seed))
        omega = rng.standard_normal((int(latent_dim), int(dim)))
        phase = rng.uniform(0.0, 2.0 * math.pi, size=int(dim))
        return cls(dim=int(dim), seed=int(seed), omega=omega, phase=phase, sigma=float(sigma), latent_dim=int(latent_dim))

    def features(self, latents: np.ndarray) -> np.ndarray:
        values = np.asarray(latents, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.latent_dim:
            raise _refused("REFUSED-ERROR-BOUNDARY", "latent matrix does not match the RFF map")
        return math.sqrt(2.0 / self.dim) * np.cos((values / self.sigma) @ self.omega + self.phase)

    def item(self, latents: np.ndarray, valid: np.ndarray) -> np.ndarray:
        mask = np.asarray(valid, dtype=bool)
        values = np.asarray(latents, dtype=np.float64)[mask]
        if values.shape[0] == 0:
            raise _refused("REFUSED-ERROR-BOUNDARY", "file has no valid patch latents")
        phi = self.features(values)
        mean = phi.mean(axis=0)
        std = phi.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float32)

    def spec(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "seed": self.seed,
            "sigma": self.sigma,
            "latent_dim": self.latent_dim,
            "item": "concat(mean_d phi_d, std_d phi_d)",
            "feature": "sqrt(2/D) * cos((z/sigma) @ omega0 + phase), omega0 ~ N(0,1)",
        }


def fit_prepool_sigma(latents_per_file: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Median pairwise distance on a deterministic bandwidth-latent subsample."""
    from population_benchmark.evaluation import median_pairwise_distance

    collected: list[np.ndarray] = []
    for latents, valid in latents_per_file:
        mask = np.asarray(valid, dtype=bool)
        values = np.asarray(latents, dtype=np.float64)[mask]
        for row in values:
            collected.append(row)
            if len(collected) >= spec.RFF_BANDWIDTH_LATENTS:
                break
        if len(collected) >= spec.RFF_BANDWIDTH_LATENTS:
            break
    if len(collected) < 2:
        raise _refused("REFUSED-ERROR-BOUNDARY", "not enough bandwidth patch latents")
    return median_pairwise_distance(np.stack(collected))


def build_prepool_items(latents_per_file: Sequence[tuple[np.ndarray, np.ndarray]], maps: Mapping[int, RFFMap]) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for seed, rff in maps.items():
        output[int(seed)] = np.stack([rff.item(latents, valid) for latents, valid in latents_per_file])
    return output


# --- admission and episode construction ----------------------------------------


@dataclass
class SourcePlan:
    slot: int
    block: int
    source_index: int
    source_seed: int
    disposition: str
    admitted: bool
    reason: str
    strength: float | None
    n_attempts: int
    twin_file_id: str | None
    abnormal_file_id: str | None
    rank: int | None
    robot_idx: int | None = None
    program_idx: int | None = None
    normal_path_diverges: bool = False
    config_hash: str = ""
    generator_version: str = ""
    attempts: tuple[dict[str, Any], ...] = ()


def admit_file(sample: Any, config_hash: str) -> str:
    values = np.asarray(sample.x)
    if values.ndim != 2:
        return "ABSTAIN-CHANNEL"
    if values.shape[0] != EXPECTED_CHANNELS:
        return "ABSTAIN-CHANNEL"
    if values.shape[1] < MIN_STEPS:
        return "ABSTAIN-SHORT"
    if not np.isfinite(values).all():
        return "ABSTAIN-NONFINITE"
    if str(sample.config_hash) != str(config_hash):
        return "ABSTAIN-LINEAGE"
    return "ADMITTED"


def admit_pair(pair: Any, config_hash: str) -> tuple[bool, str]:
    """Common admission policy, applied before membership; labels never enter."""
    if str(pair.disposition) != "PAIRED":
        return False, "ABSTAIN-NONPAIRED"
    for sample in (pair.twin, pair.abnormal):
        reason = admit_file(sample, config_hash)
        if reason != "ADMITTED":
            return False, reason
    return True, "ADMITTED"


@dataclass
class EpisodeBuild:
    episode: spec.Episode
    sources: list[SourcePlan]
    samples: list[Any]
    file_records: list[dict[str, Any]]
    windows: list[dict[str, Any]]


def build_episode(
    *,
    generator: Any,
    episode: spec.Episode,
    plan: spec.Plan,
    max_level: int,
    on_attempt: Callable[[dict[str, object]], None] | None = None,
) -> EpisodeBuild:
    """Generate every planned source once and derive the nested window rosters."""
    from synth.schema import AnomalyFamily

    family = AnomalyFamily(str(episode.family))
    config_hash = str(generator.cfg.hash())
    sources: list[SourcePlan] = []
    pairs: dict[int, Any] = {}
    for block in range(episode.n_blocks):
        for slot in episode.block_slots(block):
            seed = episode.source_seed(slot)
            pair = generator.generate_paired_twin(
                seed, family, float(episode.severity), on_attempt=on_attempt
            )
            pairs[int(slot)] = pair
            admitted, reason = admit_pair(pair, config_hash)
            sources.append(
                SourcePlan(
                    slot=int(slot),
                    block=int(block),
                    source_index=int(episode.source_index(slot)),
                    source_seed=int(seed),
                    disposition=str(pair.disposition),
                    admitted=bool(admitted),
                    reason=str(reason),
                    strength=float(pair.strength) if pair.strength is not None else None,
                    n_attempts=int(pair.n_attempts),
                    twin_file_id=str(pair.twin.file_id),
                    abnormal_file_id=str(pair.abnormal.file_id) if admitted else None,
                    rank=None,
                    robot_idx=int(pair.twin.robot_idx),
                    program_idx=int(pair.twin.program_idx),
                    normal_path_diverges=bool(pair.normal_path_diverges),
                    config_hash=str(pair.twin.config_hash),
                    generator_version=str(pair.twin.generator_version),
                    attempts=tuple(
                        {
                            "attempt": int(value.attempt),
                            "attempt_seed": int(value.attempt_seed),
                            "status": str(value.status),
                            "strength": (float(value.strength) if value.strength is not None else None),
                            "reason": str(value.reason),
                        }
                        for value in pair.attempts
                    ),
                )
            )

    # Membership is assigned only among admitted sources, in the frozen hash
    # order of each block; admission never removes or reorders outcomes to
    # improve a result.
    by_slot = {source.slot: source for source in sources}
    for block in range(episode.n_blocks):
        allowed = {source.slot for source in sources if source.block == block and source.admitted}
        ordered = [
            slot
            for slot in spec.rank_order(episode.history_id, episode.episode_id, block, plan.block_files)
            if slot in allowed
        ]
        for rank, slot in enumerate(ordered):
            by_slot[slot].rank = rank

    admitted_per_block = {
        block: sum(1 for source in sources if source.block == block and source.admitted)
        for block in range(episode.n_blocks)
    }
    samples: list[Any] = []
    file_records: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: item.slot):
        if not source.admitted or source.rank is None:
            continue
        # A block with fewer than eight admitted sources cannot represent the
        # three strictly nested burden levels; its windows abstain in
        # ``spec.episode_windows`` and no variants are materialized.
        if admitted_per_block[source.block] < 8:
            continue
        twin, abnormal = spec.required_variants(
            source.rank, admitted_per_block[source.block], max_level
        )
        pair = pairs[source.slot]
        if twin:
            samples.append(pair.twin)
            file_records.append({
                "file_id": str(pair.twin.file_id), "role": "normal",
                "slot": int(source.slot), "block": int(source.block),
                "source_seed": int(source.source_seed), "rank": int(source.rank),
            })
        if abnormal:
            samples.append(pair.abnormal)
            file_records.append({
                "file_id": str(pair.abnormal.file_id), "role": "abnormal",
                "slot": int(source.slot), "block": int(source.block),
                "source_seed": int(source.source_seed), "rank": int(source.rank),
            })

    windows = _window_records(episode, plan, sources, spec.block_fault_counts(plan.block_files), max_level)
    return EpisodeBuild(
        episode=episode,
        sources=sources,
        samples=samples,
        file_records=file_records,
        windows=windows,
    )


def _window_records(
    episode: spec.Episode,
    plan: spec.Plan,
    sources: Sequence[SourcePlan],
    counts: tuple[int, int, int],
    max_level: int,
) -> list[dict[str, Any]]:
    del counts  # counts are re-derived from the frozen rule in spec.episode_windows
    admitted_ranks = {source.slot: source.rank for source in sources}
    twin_ids = {source.slot: source.twin_file_id for source in sources}
    abnormal_ids = {source.slot: source.abnormal_file_id for source in sources}
    return spec.episode_windows(
        episode=episode,
        plan=plan,
        admitted_ranks=admitted_ranks,
        twin_ids=twin_ids,
        abnormal_ids=abnormal_ids,
        max_level=max_level,
    )


@dataclass
class Captured:
    items: dict[str, np.ndarray]
    file_records: list[dict[str, Any]]
    latents: dict[int, list[tuple[np.ndarray, np.ndarray]]]
    prepool_maps: dict[int, RFFMap]

    @property
    def files(self) -> int:
        return len(self.file_records)


def capture_samples(
    *,
    bundles: Mapping[int, Any],
    samples: Sequence[Any],
    history_id: str,
    prepool_maps: Mapping[int, RFFMap] | None,
    keep_latents: bool = False,
    device: str | None = None,
    on_file: Callable[[int], None] | None = None,
) -> Captured:
    """Capture one chunk through the frozen public encoder path."""
    import torch

    from representation.data import collate_variable_files
    from representation.inference import RepresentationInference
    from representation.sprint18_harness import _encoder_batch, deterministic_mask_seed

    materialized = list(samples)
    if not materialized:
        raise _refused("REFUSED-ERROR-BOUNDARY", "capture needs at least one file")
    file_ids = [str(sample.file_id) for sample in materialized]
    if len(set(file_ids)) != len(file_ids):
        raise _refused("REFUSED-IDENTITY", "capture file IDs are not unique")

    raw = np.stack([raw_item(sample) for sample in materialized]).astype(np.float32)
    length = np.zeros((len(materialized), 3), dtype=np.float32)
    records: list[dict[str, Any]] = [
        {
            "file_id": file_id,
            "role": "normal" if str(sample.file_label.value) == "normal" else "abnormal",
            "T": int(sample.T),
            "C": int(sample.C),
            "robot_idx": int(sample.robot_idx),
            "program_idx": int(sample.program_idx),
            "n_patches": 0,
            "n_valid_patches": 0,
            "valid_timesteps": int(sample.T),
        }
        for file_id, sample in zip(file_ids, materialized)
    ]
    pooled: dict[int, np.ndarray] = {}
    prepool: dict[int, np.ndarray] = {}
    kept: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        int(seed): [] for seed in bundles
    } if keep_latents else {}
    maps: dict[int, RFFMap] = dict(prepool_maps or {})

    for seed in spec.ENCODER_SEEDS:
        bundle = bundles[int(seed)]
        model = bundle.model  # type: ignore[attr-defined]
        bank = bundle.bank  # type: ignore[attr-defined]
        inference = RepresentationInference(model, bank.bank, model.patchifier, masking_config=model.config)
        model.eval()
        before = s24_live_state_digests(model, bank)
        pooled_rows: list[np.ndarray] = []
        prepool_rows: list[np.ndarray] = []
        with torch.no_grad():
            for index, sample in enumerate(materialized):
                mask_seed = deterministic_mask_seed(int(seed), str(history_id), str(sample.file_id))
                batch = collate_variable_files(
                    [sample],
                    model.patchifier,
                    masking_config=model.config,
                    masking_seed=int(mask_seed),
                )
                clean = _encoder_batch(
                    {key: value for key, value in batch.items() if key != "file_samples"}
                )
                moved = inference.co_locate_batch(clean)  # type: ignore[arg-type]
                output = model(moved)
                embedding = output["file_embedding"]
                latents = output["context_latents"]
                valid = output["patch_valid_mask"]
                if (
                    not isinstance(embedding, torch.Tensor)
                    or not isinstance(latents, torch.Tensor)
                    or not isinstance(valid, torch.Tensor)
                ):
                    raise _refused("REFUSED-ERROR-BOUNDARY", "encoder output is missing required tensors")
                if embedding.shape != (1, model.config.d_model):
                    raise _refused("REFUSED-IDENTITY", "pooled embedding shape drifted")
                vector = embedding[0].detach().cpu().float().numpy()
                latent = latents[0].detach().cpu().float().numpy()
                mask = valid[0].detach().cpu().numpy().astype(bool)
                if not np.isfinite(vector).all() or not np.isfinite(latent).all():
                    raise _refused("REFUSED-ERROR-BOUNDARY", "capture produced non-finite values")
                if not mask.any():
                    raise _refused("REFUSED-ERROR-BOUNDARY", "file has no valid patches")
                pooled_rows.append(np.asarray(vector, dtype=np.float32))
                if int(seed) in maps:
                    prepool_rows.append(maps[int(seed)].item(latent, mask))
                if keep_latents:
                    kept[int(seed)].append((latent, mask))
                if int(seed) == spec.ENCODER_SEEDS[0]:
                    n_patches = int(batch["patches"].shape[1])
                    records[index]["n_patches"] = n_patches
                    records[index]["n_valid_patches"] = int(mask.sum())
                    length[index] = length_item(sample, n_patches, int(mask.sum()))
                if on_file is not None:
                    on_file(1)
        after = s24_live_state_digests(model, bank)
        assert_s24_frozen_unchanged(before, after)
        pooled[int(seed)] = np.stack(pooled_rows).astype(np.float32)
        if int(seed) in maps:
            prepool[int(seed)] = np.stack(prepool_rows).astype(np.float32)

    items: dict[str, np.ndarray] = {RAW_KEY: raw, LENGTH_KEY: length}
    for seed, values in pooled.items():
        items[readout_key("POOLED", seed)] = values
    for seed, values in prepool.items():
        items[readout_key("PREPOOL", seed)] = values
    return Captured(items=items, file_records=records, latents=kept, prepool_maps=maps)


__all__ = [
    "Captured",
    "EpisodeBuild",
    "EXPECTED_CHANNELS",
    "MIN_STEPS",
    "RFFMap",
    "SourcePlan",
    "admit_file",
    "admit_pair",
    "build_episode",
    "build_prepool_items",
    "capture_samples",
    "fit_prepool_sigma",
    "length_item",
    "raw_item",
    "restore_bundles",
]
