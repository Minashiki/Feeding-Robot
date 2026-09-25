"""Write a case file only when an acceptance run sets M2_EVIDENCE_DIR."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def evidence_dir() -> Path | None:
    raw = os.environ.get("M2_EVIDENCE_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_json(case_id: str, payload: dict) -> None:
    root = evidence_dir()
    if root is None:
        return
    body = {"case_id": case_id, **_plain(payload)}
    (root / f"{case_id}.json").write_text(json.dumps(body))


def write_npz(case_id: str, **arrays) -> None:
    root = evidence_dir()
    if root is None:
        return
    np.savez(root / f"{case_id}.npz", **arrays)
