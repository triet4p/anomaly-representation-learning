"""Durable, hash-verified custody for the S24-7-M20 population benchmark.

Storage is deliberately boring: one ``items.npz`` (the exact per-file
sufficient statistics for every readout), one ``waveforms.npz`` (actual
signal custody), and one JSON sidecar per chunk. Items and waveforms are
separate so a fresh offline replay can skip waveforms entirely. Every write
is atomic (temp file, fsync, replace) and every read verifies the recorded
SHA-256, so a corrupted or partially written chunk fails closed instead of
silently feeding a replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from population_benchmark import spec
from population_benchmark.allocation import PAIRED_ROLES
from population_benchmark.evaluation import Chunk

KINDS = ("reference", "bandwidth", "calibration", "test")


def _refused(reason: str, detail: str) -> ValueError:
    return ValueError(f"{reason}: {detail}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            handle = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(handle, path)
        handle = None
    finally:
        if handle is not None:
            try:
                os.unlink(handle)
            except FileNotFoundError:
                pass


def write_json(path: Path, payload: object) -> str:
    text = spec.canonical_json(payload) + "\n"
    _atomic_write(Path(path), text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = __import__("io").BytesIO()
    np.savez(output, **{str(key): np.asarray(value) for key, value in sorted(arrays.items())})
    return output.getvalue()


@dataclass
class RunStore:
    """Chunk store plus append-only event journal under one explicit root."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @classmethod
    def create(cls, root: Path, *, fresh: bool = True) -> "RunStore":
        target = Path(root)
        if fresh and target.exists() and any(target.iterdir()):
            raise _refused("REFUSED-OVERWRITE", f"{target} is non-empty")
        target.mkdir(parents=True, exist_ok=True)
        (target / "chunks").mkdir(exist_ok=True)
        store = cls(root=target)
        if not store.events_path.exists():
            _atomic_write(store.events_path, b"")
        return store

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    def append_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        line = json.dumps({"kind": str(kind), **dict(payload)}, sort_keys=True, allow_nan=False)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        return [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line]

    def _dir(self, history_id: str, kind: str) -> Path:
        if kind not in KINDS:
            raise _refused("REFUSED-UNKNOWN-DESIGN", f"unknown chunk kind {kind!r}")
        return self.root / "chunks" / str(history_id) / str(kind)

    def chunk_exists(self, history_id: str, kind: str, chunk_id: str) -> bool:
        return (self._dir(history_id, kind) / f"{chunk_id}.json").is_file()

    def write_chunk(
        self,
        *,
        history_id: str,
        kind: str,
        chunk_id: str,
        items: Mapping[str, np.ndarray],
        waveforms: Sequence[np.ndarray],
        meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        directory = self._dir(history_id, kind)
        directory.mkdir(parents=True, exist_ok=True)
        n_files = len(waveforms)
        if not n_files:
            raise _refused("REFUSED-ERROR-BOUNDARY", "chunk needs at least one file")
        if len(meta.get("files", [])) != n_files:
            raise _refused("REFUSED-IDENTITY", "chunk file order and waveform count disagree")
        for key, values in items.items():
            array = np.asarray(values)
            if array.ndim < 1:
                raise _refused("REFUSED-ERROR-BOUNDARY", f"item array {key} is scalar")
            if not np.isfinite(array).all():
                raise _refused("REFUSED-ERROR-BOUNDARY", f"item array {key} is non-finite")
        waveform_arrays = {
            f"wave_{index:06d}": np.asarray(values, dtype=np.float32)
            for index, values in enumerate(waveforms)
        }
        for key, values in waveform_arrays.items():
            if not np.isfinite(values).all():
                raise _refused("REFUSED-ERROR-BOUNDARY", f"waveform {key} is non-finite")
        items_bytes = _npz_bytes(items)
        waveforms_bytes = _npz_bytes(waveform_arrays)
        items_path = directory / f"{chunk_id}.items.npz"
        waveforms_path = directory / f"{chunk_id}.waveforms.npz"
        _atomic_write(items_path, items_bytes)
        _atomic_write(waveforms_path, waveforms_bytes)
        record = {
            **dict(meta),
            "kind": str(kind),
            "history_id": str(history_id),
            "chunk_id": str(chunk_id),
            "files": list(meta["files"]),
            "sha256": {
                "items": sha256_bytes(items_bytes),
                "waveforms": sha256_bytes(waveforms_bytes),
            },
            "bytes": {
                "items": len(items_bytes),
                "waveforms": len(waveforms_bytes),
            },
            "items_keys": sorted(items),
        }
        write_json(directory / f"{chunk_id}.json", record)
        return record

    def read_meta(self, history_id: str, kind: str, chunk_id: str) -> dict[str, Any]:
        path = self._dir(history_id, kind) / f"{chunk_id}.json"
        if not path.is_file():
            raise _refused("REFUSED-MISSING-PREREQUISITE", f"chunk sidecar absent: {path}")
        return read_json(path)

    def read_chunk(self, history_id: str, kind: str, chunk_id: str) -> Chunk:
        directory = self._dir(history_id, kind)
        sidecar = directory / f"{chunk_id}.json"
        if not sidecar.is_file():
            raise _refused("REFUSED-MISSING-PREREQUISITE", f"chunk sidecar absent: {sidecar}")
        meta = read_json(sidecar)
        items_path = directory / f"{chunk_id}.items.npz"
        waveforms_path = directory / f"{chunk_id}.waveforms.npz"
        for path, key in ((items_path, "items"), (waveforms_path, "waveforms")):
            if not path.is_file():
                raise _refused("REFUSED-MISSING-PREREQUISITE", f"chunk member absent: {path}")
            if sha256_file(path) != meta["sha256"][key]:
                raise _refused("REFUSED-HASH-MISMATCH", f"chunk hash differs: {path}")
        with np.load(items_path, allow_pickle=False) as archive:
            items = {key: np.asarray(archive[key]) for key in archive.files}
        for key, values in items.items():
            if values.shape[0] != len(meta["files"]):
                raise _refused("REFUSED-IDENTITY", f"item {key} row count drifted in {chunk_id}")
        return Chunk(kind=kind, history_id=history_id, chunk_id=chunk_id, meta=meta, items=items)

    def read_waveforms(self, history_id: str, kind: str, chunk_id: str) -> list[np.ndarray]:
        directory = self._dir(history_id, kind)
        meta = read_json(directory / f"{chunk_id}.json")
        path = directory / f"{chunk_id}.waveforms.npz"
        if sha256_file(path) != meta["sha256"]["waveforms"]:
            raise _refused("REFUSED-HASH-MISMATCH", f"waveform hash differs: {path}")
        with np.load(path, allow_pickle=False) as archive:
            return [np.asarray(archive[key]) for key in sorted(archive.files)]

    def sidecars(self, kind: str | None = None) -> Iterable[Path]:
        root = self.root / "chunks"
        if not root.exists():
            return []
        paths = sorted(root.rglob("*.json"))
        if kind is None:
            return paths
        return [path for path in paths if path.parent.name == kind]

    def chunk_ids(self, history_id: str, kind: str) -> list[str]:
        directory = self._dir(history_id, kind)
        if not directory.exists():
            return []
        return sorted(path.name[: -len(".json")] for path in directory.glob("*.json"))

    def load_chunks(self, kinds: Sequence[str] = KINDS) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in self.sidecars():
            if path.parent.name not in kinds:
                continue
            meta = read_json(path)
            chunk_id = str(meta["chunk_id"])
            history_id = str(meta["history_id"])
            chunks.append(self.read_chunk(history_id, path.parent.name, chunk_id))
        return chunks

    def verify(self) -> dict[str, Any]:
        """Verify every chunk hash and item row count without any model call."""
        checks: list[dict[str, Any]] = []
        for path in self.sidecars():
            meta = read_json(path)
            kind = str(meta["kind"])
            history_id = str(meta["history_id"])
            chunk_id = str(meta["chunk_id"])
            try:
                chunk = self.read_chunk(history_id, kind, chunk_id)
            except ValueError as exc:
                checks.append({"chunk": f"{history_id}/{kind}/{chunk_id}", "pass": False, "error": str(exc)})
                continue
            checks.append({
                "chunk": f"{history_id}/{kind}/{chunk_id}",
                "pass": True,
                "files": len(chunk.meta["files"]),
                "items": sorted(chunk.items),
            })
        return {
            "checks": len(checks),
            "failures": [item for item in checks if not item["pass"]],
            "pass": all(item["pass"] for item in checks),
        }

    def verify_source_identity(self, allocation: Any, kinds: Sequence[str] = KINDS) -> dict[str, Any]:
        """Bind every persisted source/file row to one frozen allocation.

        Fails closed on a row whose recorded seed or physical-source key differs
        from the frozen allocation, on one logical source persisted under two
        chunks, on a physical key or generator seed shared by distinct logical
        sources, and on a repeated file id.
        """
        history_index = {str(name): index for index, name in enumerate(allocation.history_ids)}
        row_owner: dict[tuple[str, int, int], str] = {}
        key_owner: dict[int, tuple[str, int, int]] = {}
        seed_owner: dict[tuple[str, int], tuple[str, int, int]] = {}
        file_owner: dict[str, str] = {}
        failures: list[dict[str, Any]] = []
        chunks = 0
        for path in self.sidecars():
            kind = str(path.parent.name)
            if kind not in kinds:
                continue
            meta = read_json(path)
            history_id = str(meta["history_id"])
            label = f"{history_id}/{kind}/{meta['chunk_id']}"
            chunks += 1
            if history_id not in history_index:
                failures.append({"chunk": label, "error": "history is not in the frozen allocation"})
                continue
            history = history_index[history_id]
            for record in meta.get("files", []):
                file_id = str(record.get("file_id"))
                if "source_index" not in record:
                    failures.append({"chunk": label, "file_id": file_id, "error": "file row carries no frozen source identity"})
                    continue
                source_index = int(record["source_index"])
                try:
                    expected_seed = int(allocation.seed_for(kind, history, source_index))
                    expected_key = int(allocation.key_for(kind, history, source_index))
                except ValueError as exc:
                    failures.append({"chunk": label, "file_id": file_id, "error": str(exc)})
                    continue
                row = (kind, history, source_index)
                if int(record.get("source_seed", -1)) != expected_seed:
                    failures.append({
                        "chunk": label, "file_id": file_id,
                        "error": f"seed {record.get('source_seed')} != frozen {expected_seed}",
                    })
                if int(record.get("source_key", -1)) != expected_key:
                    failures.append({
                        "chunk": label, "file_id": file_id,
                        "error": f"source key {record.get('source_key')} != frozen {expected_key}",
                    })
                previous_chunk = row_owner.setdefault(row, label)
                if previous_chunk != label:
                    failures.append({"chunk": label, "row": list(row), "error": f"logical source persisted in {previous_chunk} too"})
                key_holder = key_owner.setdefault(expected_key, row)
                if key_holder != row:
                    failures.append({"chunk": label, "row": list(row), "error": f"physical source shared with {key_holder}"})
                domain = "paired" if kind in PAIRED_ROLES else "normal"
                seed_holder = seed_owner.setdefault((domain, expected_seed), row)
                if seed_holder != row:
                    failures.append({"chunk": label, "row": list(row), "error": f"generator seed shared with {seed_holder}"})
                holder = file_owner.setdefault(file_id, label)
                if holder != label:
                    failures.append({"chunk": label, "file_id": file_id, "error": f"file id persisted in {holder} too"})
        payload = {
            "pass": not failures,
            "chunks": int(chunks),
            "source_rows": int(len(row_owner)),
            "expected_sources": int(allocation.total),
            "complete_census": bool(len(row_owner) == allocation.total),
            "allocation_digest": str(allocation.digest),
            "failures": failures[:20],
        }
        if failures:
            raise _refused("REFUSED-IDENTITY", json.dumps(payload, sort_keys=True)[:1500])
        return payload

    def require_disk(self, budget_bytes: int = spec.DISK_BUDGET_BYTES) -> dict[str, Any]:
        usage = shutil.disk_usage(self.root)
        return {
            "root": str(self.root),
            "free_bytes": int(usage.free),
            "required_bytes": int(budget_bytes),
            "pass": int(usage.free) >= int(budget_bytes),
        }


__all__ = [
    "KINDS",
    "RunStore",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "write_json",
]
