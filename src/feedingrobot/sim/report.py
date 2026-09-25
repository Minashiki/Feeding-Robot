"""Evidence check. Booleans in the report are outputs, not inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
KINDS = ("plate", "spoon", "tilt", "accel", "lip")
SEEDS = (0, 1, 2, 3, 4, 5)
SOLVERS = ("A", "B", "C")
TRACE_FIELDS = (
    "time_s", "input_interval_s", "q", "dq", "qacc",
    "tau_requested", "tau_command", "tau_actual",
    "tcp_pos", "tcp_rot", "mouth_pos", "mouth_rot",
    "food_pos", "food_quat", "food_vel", "raw_wrench",
    "contact_force_group", "single_contact_peak", "min_contact_dist", "warning_count",
    "velocity_fault", "forbidden_contact", "event_candidate", "event_confirmed",
    "head_target", "head_actual", "head_command", "contact_flat",
)
PROCESS_IDS = ("T08_tilt", "T08_accel", "T10_motion", "HOLD")
STATIC_IDS = (
    "T01", "T02", "T03", "T04", "T05", "T06", "T07",
    "T08_fault", "T09", "T10_negative", "T11", "T12", "T13_T14",
)
RECOMPUTE_ATOL = 1e-9
RECOMPUTE_RTOL = 1e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _formal_ids():
    ids = [f"{kind}_seed{seed}_{solver}" for seed in SEEDS for kind in KINDS for solver in SOLVERS]
    ids.append("tilt_seed4_D")
    return ids


def assess_run(run_dir: Path) -> dict:
    reasons = []
    run_dir = Path(run_dir)
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        reasons.append("missing run.json")
        return _verdict(reasons, incomplete=True)
    run = json.loads(run_path.read_text())
    if run.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema_version")
    if run.get("status") != "completed" or run.get("incomplete") is True:
        reasons.append("incomplete")
    execution_path = run_dir / "execution.json"
    if not execution_path.is_file():
        reasons.append("missing execution.json")
    else:
        execution = json.loads(execution_path.read_text())
        if execution.get("run_id") != run.get("run_id"):
            reasons.append("execution run_id mismatch")
        outcomes = {}
        for row in execution.get("reports", []):
            if row.get("when") != "call":
                continue
            outcomes.setdefault(row["nodeid"], []).append(row.get("outcome"))
        if any(item != "passed" for values in outcomes.values() for item in values):
            reasons.append("pytest outcome is not passed")
        collected = execution.get("collected")
        matrix_node = "tests/test_m1_long.py::test_t13_t14_contact_matrix"
        if collected is not None and matrix_node not in collected:
            reasons.append("matrix test was not collected")
        skipped = [row["nodeid"] for row in execution.get("reports", []) if row.get("outcome") in ("skipped", "xfailed")]
        if skipped:
            reasons.append("skipped " + skipped[0])
    before = run_dir / "input_hash_before.json"
    after = run_dir / "input_hash_after.json"
    if not before.is_file() or not after.is_file():
        reasons.append("missing input hashes")
    else:
        b = json.loads(before.read_text())["aggregate_sha256"]
        a = json.loads(after.read_text())["aggregate_sha256"]
        if b != a:
            reasons.append("input hash changed during the run")
    case_dir = run_dir / "cases"
    for case_id in _formal_ids():
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            reasons.append(f"missing {case_id}")
            continue
        _check_dynamic(case_dir, path, run.get("run_id"), reasons)
    for case_id in PROCESS_IDS:
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            reasons.append(f"missing {case_id}")
            continue
        _check_dynamic(case_dir, path, run.get("run_id"), reasons)
    for case_id in STATIC_IDS:
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            reasons.append(f"missing {case_id}")
            continue
        row = json.loads(path.read_text())
        if row.get("run_id") != run.get("run_id"):
            reasons.append(f"{case_id} run_id mismatch")
        if not row.get("metrics") or not _finite_numbers(row["metrics"]):
            reasons.append(f"{case_id} metrics")
        if not row.get("thresholds") or not row.get("effective_config") or row.get("duration_s") is None:
            reasons.append(f"{case_id} missing thresholds or effective_config")
        if "n_rows" in row.get("metrics", {}) and len(row["metrics"]) == 1:
            reasons.append(f"{case_id} only n_rows")
    comparisons = run_dir / "comparisons.json"
    if comparisons.is_file():
        rows = json.loads(comparisons.read_text())
        pairs = {(row.get("kind"), row.get("seed"), row.get("pair")) for row in rows}
        for seed in SEEDS:
            for kind in KINDS:
                for pair in ("A/B", "A/C"):
                    if (kind, seed, pair) not in pairs:
                        reasons.append(f"missing comparison {kind} seed {seed} {pair}")
        for pair in ("A/D", "B/D"):
            if ("tilt", 4, pair) not in pairs:
                reasons.append(f"missing comparison tilt seed 4 {pair}")
        for row in rows:
            if "delta_event_s" not in row or "thresholds" not in row or "peak_a" not in row:
                reasons.append(f"comparison fields {row.get('kind')} seed {row.get('seed')}")
                continue
            if row.get("event_a") is not None and row.get("event_b") is not None:
                if abs(row["event_a"] - row["event_b"]) > 0.02 + 1e-9:
                    reasons.append(f"event delta {row['kind']} seed {row['seed']}")
            if not all(row.get(key) for key in ("steady_ok", "peak_ok", "vector_impulse_ok", "scalar_impulse_ok")):
                reasons.append(f"comparison failed {row.get('kind')} seed {row.get('seed')}")
    else:
        reasons.append("missing comparisons.json")
    frames = run_dir / "frames.json"
    if not frames.is_file():
        reasons.append("missing frames")
    else:
        for row in json.loads(frames.read_text()):
            image = run_dir / row["file"]
            if not image.is_file() or image.stat().st_size < 100:
                reasons.append(f"missing image {row['file']}")
    return _verdict(reasons, incomplete=run.get("incomplete") is True or run.get("status") != "completed")


def _check_dynamic(case_dir: Path, path: Path, run_id, reasons: list) -> None:
    row = json.loads(path.read_text())
    case_id = row.get("case_id", path.stem)
    if row.get("run_id") != run_id:
        reasons.append(f"{case_id} run_id mismatch")
    metrics = row.get("metrics") or {}
    if not metrics or not _finite_numbers(metrics):
        reasons.append(f"{case_id} metrics")
        return
    for key in ("peak_force_n", "scalar_impulse_ns", "vector_impulse_ns", "min_contact_dist_m", "max_joint_speed_rad_s", "n_samples"):
        if key not in metrics or metrics[key] is None:
            reasons.append(f"{case_id} missing {key}")
    if row.get("thresholds") in (None, {}) or row.get("effective_config") in (None, {}):
        reasons.append(f"{case_id} missing thresholds or effective_config")
    logs = row.get("log_paths") or []
    if len(logs) != 1:
        reasons.append(f"{case_id} log")
        return
    npz_path = case_dir / logs[0]
    if not npz_path.is_file():
        reasons.append(f"{case_id} missing log file")
        return
    if _sha256(npz_path) != row.get("log_sha256"):
        reasons.append(f"{case_id} hash")
        return
    with np.load(npz_path) as data:
        names = set(data.files)
        if "n_rows" in names and "time_s" not in names:
            reasons.append(f"{case_id} summary npz")
            return
        missing = [key for key in TRACE_FIELDS if key not in names]
        if missing:
            reasons.append(f"{case_id} missing array {missing[0]}")
            return
        time_s = np.asarray(data["time_s"], dtype=float)
        if time_s.size < 2 or np.any(np.diff(time_s) <= 0) or not np.all(np.isfinite(time_s)):
            reasons.append(f"{case_id} time")
            return
        if abs(float(time_s[-1]) - float(row["duration_s"])) > 0.002:
            reasons.append(f"{case_id} duration")
        force = np.asarray(data["contact_force_group"], dtype=float)
        dist = np.asarray(data["min_contact_dist"], dtype=float)
        speed = np.asarray(data["dq"], dtype=float)
        if force.shape[0] != time_s.shape[0] or dist.shape[0] != time_s.shape[0] or speed.shape[0] != time_s.shape[0]:
            reasons.append(f"{case_id} ragged")
            return
        if not (np.all(np.isfinite(force)) and np.all(np.isfinite(dist)) and np.all(np.isfinite(speed))):
            reasons.append(f"{case_id} nonfinite")
            return
        dt = float(np.median(np.diff(time_s)))
        peak = float(np.max(np.linalg.norm(force, axis=1)))
        scalar = float(np.sum(np.linalg.norm(force, axis=1)) * dt)
        vector = np.sum(force, axis=0) * dt
        deepest = float(np.min(dist))
        max_speed = float(np.max(np.abs(speed)))
        if not _close(peak, float(metrics["peak_force_n"])):
            reasons.append(f"{case_id} peak mismatch")
        if not _close(scalar, float(metrics["scalar_impulse_ns"])):
            reasons.append(f"{case_id} scalar impulse mismatch")
        if not np.allclose(vector, np.asarray(metrics["vector_impulse_ns"], dtype=float), atol=RECOMPUTE_ATOL, rtol=RECOMPUTE_RTOL):
            reasons.append(f"{case_id} vector impulse mismatch")
        if not _close(deepest, float(metrics["min_contact_dist_m"])):
            reasons.append(f"{case_id} penetration mismatch")
        if not _close(max_speed, float(metrics["max_joint_speed_rad_s"])):
            reasons.append(f"{case_id} speed mismatch")
        mark_name = "mouth_contact" if case_id.startswith("lip_") else "event_confirmed"
        event = _first_time(time_s, data[mark_name]) if mark_name in data.files else None
        claimed = metrics.get("event_s")
        if claimed is None and event is not None:
            reasons.append(f"{case_id} event missing in summary")
        elif claimed is not None and (event is None or not _close(float(claimed), event)):
            reasons.append(f"{case_id} event mismatch")
        if deepest < -0.001 - 1e-9:
            reasons.append(f"{case_id} penetration")
        if int(np.max(data["velocity_fault"])) != 0 and row.get("status") == "passed":
            reasons.append(f"{case_id} velocity fault in trace")


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= RECOMPUTE_ATOL + RECOMPUTE_RTOL * abs(right)


def _first_time(time_s: np.ndarray, mark) -> float | None:
    hits = np.flatnonzero(np.asarray(mark))
    if hits.size == 0:
        return None
    return float(time_s[hits[0]])


def _verdict(reasons: list, incomplete: bool) -> dict:
    ready = len(reasons) == 0
    if ready:
        status = "passed"
    elif incomplete:
        status = "incomplete"
    else:
        status = "failed"
    return {
        "physics_passed": ready,
        "provenance_passed": ready and not any("hash" in item for item in reasons),
        "visual_status": "passed" if ready or not any("image" in item or item == "missing frames" for item in reasons) else "failed",
        "m2_ready": ready,
        "overall_status": status,
        "reasons": reasons,
    }
