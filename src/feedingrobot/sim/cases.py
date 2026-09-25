"""Case records. Writes are atomic so a crash cannot leave a half file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2


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
    if thresholds is None:
        thresholds = {"recorded_by": case_id}
    if effective_config is None:
        effective_config = {"config": os.environ.get("M1_CONFIG_PATH", "configs/m1_scene.json")}
    if duration_s is None:
        duration_s = 0.0
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
    payload["schema_version"] = SCHEMA_VERSION
    payload["run_id"] = os.environ.get("M1_RUN_ID")
    _atomic_text(out / f"{case_id}.json", json.dumps(payload, indent=2))


def save_dynamic(case_id: str, metrics: dict, arrays: dict, *, seed, solver, effective_config, thresholds, duration_s, initial_hash, status="passed", failure_reason=None, first_failure_tick=None) -> dict:
    directory = os.environ.get("M1_CASE_DIR")
    if not directory:
        return {"log_paths": []}
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{case_id}.npz"
    tmp = out / f".{case_id}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz_path)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    time_s = np.asarray(arrays["time_s"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": os.environ.get("M1_RUN_ID"),
        "case_id": case_id,
        "seed": int(seed),
        "solver_setting": solver,
        "status": status,
        "initial_state_sha256": initial_hash,
        "effective_config": _jsonify(effective_config),
        "thresholds": _jsonify(thresholds),
        "metrics": _jsonify(metrics),
        "failure_reason": failure_reason,
        "first_failure_tick": first_failure_tick,
        "duration_s": float(duration_s),
        "log_paths": [npz_path.name],
        "log_sha256": digest,
        "n_samples": int(time_s.shape[0]),
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
    }
    _atomic_text(out / f"{case_id}.json", json.dumps(payload, indent=2))
    return payload


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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
