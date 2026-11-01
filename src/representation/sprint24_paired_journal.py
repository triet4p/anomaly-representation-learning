"""Durable batch journal for paired-experiment capture and offline inspection."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


class CaptureJournal:
    """Append-only capture records; every completed batch is flushed to disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.events = self.root / "events.jsonl"
        self._next_batch = 0

    def record(self, kind: str, data: dict[str, Any]) -> None:
        line = json.dumps({"kind": kind, **data}, sort_keys=True, allow_nan=False)
        with self.events.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def capture(self, seed: int, history_id: str, captured: list[Any]) -> None:
        """Save the exact CPU float vectors and file identities for one encoder."""
        name = f"batch-{self._next_batch:08d}.npy"
        self._next_batch += 1
        matrix = np.stack([row.embedding.detach().cpu().float().numpy() for row in captured])
        target = self.root / name
        with target.open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        self.record("capture", {
            "path": name, "sha256": digest, "history_id": history_id,
            "model_seed": int(seed), "shape": list(matrix.shape),
            "files": [{"file_id": str(row.file_id), "mask_seed": int(row.mask_seed),
                       "n_patches": int(row.n_patches), "n_valid_patches": int(row.n_valid_patches),
                       "n_masked_targets": int(row.n_masked_targets),
                       "valid_len": list(row.valid_len), "starts": list(row.starts)}
                      for row in captured],
        })


def record_terminal_failure(root: Path, exc: BaseException, completed_forwards: int) -> None:
    """Record a Python-level failure after a journal exists; abrupt kills remain unknown."""
    events = Path(root) / "events.jsonl"
    if not events.is_file():
        return
    line = json.dumps({"kind": "terminal_failure", "exception": repr(exc),
                       "completed_batch_forwards": int(completed_forwards),
                       "partial_batch_forwards": "unknown", "ended_unix": time.time()}, sort_keys=True)
    with events.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_capture_batches(root: Path) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Reload and verify capture bytes without loading or invoking an encoder."""
    root = Path(root)
    batches: list[tuple[dict[str, Any], np.ndarray]] = []
    with (root / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] != "capture":
                continue
            name = event["path"]
            if Path(name).name != name:
                raise ValueError("REFUSED-IDENTITY: invalid capture path")
            path = root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != event["sha256"]:
                raise ValueError("REFUSED-IDENTITY: capture hash mismatch")
            with path.open("rb") as handle:
                matrix = np.load(handle, allow_pickle=False)
            if list(matrix.shape) != event["shape"] or len(event["files"]) != len(matrix):
                raise ValueError("REFUSED-IDENTITY: capture dimensions disagree")
            if not np.isfinite(matrix).all():
                raise ValueError("REFUSED-IDENTITY: capture contains non-finite values")
            batches.append((event, matrix))
    return batches
