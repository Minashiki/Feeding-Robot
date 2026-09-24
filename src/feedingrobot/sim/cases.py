"""Optional per-case records for the acceptance report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def save_case(
    case_id: str,
    metrics: dict,
    seeds=None,
    status: str = "passed",
    failure_reason=None,
    first_failure_tick=None,
    log=None,
    effective_config=None,
    thresholds=None,
    duration_s=None,
) -> None:
    directory = os.environ.get("M1_CASE_DIR")
    if not directory:
        return
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    log_paths = []
    if log:
        npz = out / f"{case_id}.npz"
        np.savez_compressed(npz, **{key: np.asarray(value) for key, value in log.items()})
        log_paths.append(str(npz))
    payload = {
        "case_id": case_id,
        "status": status,
        "seeds": list(seeds or []),
        "metrics": _jsonify(metrics),
        "effective_config": _jsonify(effective_config),
        "thresholds": _jsonify(thresholds),
        "failure_reason": failure_reason,
        "first_failure_tick": first_failure_tick,
        "log_paths": log_paths,
        "duration_s": duration_s,
    }
    (out / f"{case_id}.json").write_text(json.dumps(payload, indent=2))


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
