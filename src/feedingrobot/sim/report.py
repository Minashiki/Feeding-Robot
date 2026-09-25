"""Evidence check. Booleans in the report are outputs, not inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.sim.contacts import contact_pair, geom_group
from feedingrobot.sim.events import Departure
from feedingrobot.sim.trial import FORBIDDEN, is_forbidden_contact

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
PAIR_THRESHOLDS = {
    "event_s": 0.02,
    "steady_force_n": {"floor": 0.001, "frac": 0.1},
    "peak_force_n": {"floor": 0.02, "frac": 0.1},
    "vector_impulse_ns": {"floor": 1e-4, "frac": 0.1},
    "scalar_impulse_ns": {"floor": 1e-4, "frac": 0.1},
    "steady_food_pos_m": 0.002,
    "min_contact_dist_m": -0.001,
}
SOLVER_SPEC = {
    "A": {"timestep": 0.001, "iterations": 50, "tolerance": 1e-8},
    "B": {"timestep": 0.0005, "iterations": 50, "tolerance": 1e-8},
    "C": {"timestep": 0.001, "iterations": 100, "tolerance": 1e-9},
    "D": {"timestep": 0.00025, "iterations": 50, "tolerance": 1e-8},
}
SUPPORT = {"half_xy_m": 0.02, "z_min_m": -0.004, "z_max_m": 0.012, "confirm_s": 0.1}


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
    verified = {}
    for case_id in list(_formal_ids()) + list(PROCESS_IDS):
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            reasons.append(f"missing {case_id}")
            continue
        verified[case_id] = _check_dynamic(case_dir, path, run.get("run_id"), reasons)
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
        _check_comparisons(json.loads(comparisons.read_text()), verified, reasons)
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
        dts = np.asarray(data["input_interval_s"], dtype=float)
        dt_i = dts[:, 1] - dts[:, 0]
        if dt_i.shape[0] != time_s.shape[0] or np.any(dt_i <= 0):
            reasons.append(f"{case_id} input interval")
            return None
        vector = np.sum(force * dt_i[:, None], axis=0)
        scalar = float(np.sum(np.linalg.norm(force, axis=1) * dt_i))
        window = max(1, int(round(0.2 / float(np.median(dt_i)))))
        steady = np.mean(force[-window:], axis=0)
        food = np.mean(np.asarray(data["food_pos"], dtype=float)[-window:], axis=0)
        event = _event_from_trace(case_id, data, time_s, dt_i)
        flagged = _flagged_event(case_id, data, time_s)
        if not _same_optional(event, flagged):
            reasons.append(f"{case_id} event flag inconsistent")
        claimed = metrics.get("event_s")
        if claimed is None and event is not None:
            reasons.append(f"{case_id} event missing in summary")
        elif claimed is not None and (event is None or not _close(float(claimed), event)):
            reasons.append(f"{case_id} event mismatch")
        if "steady_force_n" in metrics and not np.allclose(steady, np.asarray(metrics["steady_force_n"], dtype=float), atol=RECOMPUTE_ATOL, rtol=RECOMPUTE_RTOL):
            reasons.append(f"{case_id} steady force mismatch")
        if "steady_food_pos_m" in metrics and not np.allclose(food, np.asarray(metrics["steady_food_pos_m"], dtype=float), atol=RECOMPUTE_ATOL, rtol=RECOMPUTE_RTOL):
            reasons.append(f"{case_id} food position mismatch")
        if deepest < -0.001 - 1e-9:
            reasons.append(f"{case_id} penetration")
        if int(np.max(data["velocity_fault"])) != 0 and row.get("status") == "passed":
            reasons.append(f"{case_id} velocity fault in trace")
        rebuilt = _forbidden_from_contacts(data, time_s.shape[0])
        logged = np.asarray(data["forbidden_contact"])
        if rebuilt.shape != logged.shape or np.any(rebuilt != logged):
            reasons.append(f"{case_id} forbidden flag inconsistent")
        if int(np.max(rebuilt)) != 0 and row.get("status") == "passed":
            reasons.append(f"{case_id} forbidden contact")
        parts = _formal_parts(case_id)
        if parts is not None:
            kind, seed, solver = parts
            if path.stem != case_id or row.get("seed") != seed or row.get("solver_setting") != solver:
                reasons.append(f"{case_id} identity")
            spec = SOLVER_SPEC[solver]
            cfg = row.get("effective_config") or {}
            if not _close(float(cfg.get("timestep", -1)), spec["timestep"]) or int(cfg.get("iterations", -1)) != spec["iterations"] or not _close(float(cfg.get("tolerance", -1)), spec["tolerance"]):
                reasons.append(f"{case_id} solver spec")
            if not row.get("initial_state_sha256"):
                reasons.append(f"{case_id} missing initial state")
        if reasons and reasons[-1].startswith(case_id):
            return None
        return {
            "case_id": case_id,
            "seed": row.get("seed"),
            "solver": row.get("solver_setting"),
            "initial_state_sha256": row.get("initial_state_sha256"),
            "duration_s": float(row["duration_s"]),
            "metrics": {
                "peak_force_n": peak,
                "scalar_impulse_ns": scalar,
                "vector_impulse_ns": vector,
                "steady_force_n": steady,
                "steady_food_pos_m": food,
                "event_s": event,
                "min_contact_dist_m": deepest,
                "max_joint_speed_rad_s": max_speed,
            },
        }


def _formal_parts(case_id: str):
    for kind in KINDS:
        prefix = f"{kind}_seed"
        if case_id.startswith(prefix):
            seed_s, solver = case_id[len(prefix) :].split("_", 1)
            return kind, int(seed_s), solver
    return None


def _event_from_trace(case_id: str, data, time_s: np.ndarray, dt_i: np.ndarray):
    kind = _formal_parts(case_id)[0] if _formal_parts(case_id) else ""
    if case_id.startswith("lip_") or kind == "lip":
        return _first_mouth(data, time_s)
    if kind in ("plate", "spoon", "tilt", "accel") or case_id in ("T08_tilt", "T08_accel"):
        return _rebuild_departure(data, time_s, dt_i)
    return None


def _flagged_event(case_id: str, data, time_s: np.ndarray):
    kind = _formal_parts(case_id)[0] if _formal_parts(case_id) else ""
    if case_id.startswith("lip_") or kind == "lip":
        return _first_time(time_s, data["mouth_contact"])
    return _first_time(time_s, data["event_confirmed"])


def _first_mouth(data, time_s: np.ndarray):
    names = _geom_names(data)
    for row in np.asarray(data["contact_flat"]):
        pair = {names.get(int(row[1]), ""), names.get(int(row[2]), "")}
        if "mouth_upper" in pair:
            return float(time_s[int(row[0])])
    return None


def _rebuild_departure(data, time_s: np.ndarray, dt_i: np.ndarray):
    names = _geom_names(data)
    groups = {gid: geom_group(name) for gid, name in names.items()}
    touching = np.zeros(time_s.shape[0], dtype=bool)
    for row in np.asarray(data["contact_flat"]):
        pair = contact_pair(groups.get(int(row[1]), "arm"), groups.get(int(row[2]), "arm"))
        if pair == contact_pair("food", "spoon"):
            touching[int(row[0])] = True
    food = np.asarray(data["food_pos"], dtype=float)
    tcp = np.asarray(data["tcp_pos"], dtype=float)
    rot = np.asarray(data["tcp_rot"], dtype=float)
    depart = Departure(SUPPORT["confirm_s"])
    for i in range(time_s.shape[0]):
        local = rot[i].reshape(3, 3).T @ (food[i] - tcp[i])
        xy = float(np.linalg.norm(local[:2]))
        outside = xy > SUPPORT["half_xy_m"] or float(local[2]) > SUPPORT["z_max_m"] or float(local[2]) < SUPPORT["z_min_m"]
        depart.update(float(time_s[i]), float(dt_i[i]), outside, bool(touching[i]))
    return depart.confirmed_s


def _geom_names(data) -> dict:
    ids = np.asarray(data["geom_ids"])
    raw = np.asarray(data["geom_name_bytes"])
    out = {}
    for gid, name in zip(ids, raw):
        text = name.decode() if isinstance(name, bytes) else str(name)
        out[int(gid)] = text.rstrip("\x00")
    return out


def _forbidden_from_contacts(data, n: int) -> np.ndarray:
    names = _geom_names(data)
    groups = {gid: geom_group(name) for gid, name in names.items()}
    flags = np.zeros(n, dtype=np.int8)
    for row in np.asarray(data["contact_flat"]):
        if is_forbidden_contact(groups.get(int(row[1]), "arm"), groups.get(int(row[2]), "arm"), FORBIDDEN):
            flags[int(row[0])] = 1
    return flags


def _expected_comparison_keys():
    keys = [(kind, seed, pair) for seed in SEEDS for kind in KINDS for pair in ("A/B", "A/C")]
    keys.extend([("tilt", 4, "A/D"), ("tilt", 4, "B/D")])
    return keys


def _band(left: float, right: float, floor: float, frac: float) -> bool:
    return abs(left - right) <= max(floor, frac * max(abs(left), abs(right)))


def _check_comparisons(rows: list, verified: dict, reasons: list) -> None:
    expected = _expected_comparison_keys()
    found = []
    for row in rows:
        if not _finite_numbers(row):
            reasons.append(f"comparison nonfinite {row.get('kind')} seed {row.get('seed')} {row.get('pair')}")
            continue
        found.append((row.get("kind"), row.get("seed"), row.get("pair")))
    if len(found) != len(set(found)):
        reasons.append("duplicate comparison " + str(found))
    for key in expected:
        if key not in found:
            reasons.append(f"missing comparison {key[0]} seed {key[1]} {key[2]}")
    for key in found:
        if key not in expected:
            reasons.append(f"unknown comparison {key}")
    for row in rows:
        kind, seed, pair = row.get("kind"), row.get("seed"), row.get("pair")
        if (kind, seed, pair) not in expected or not isinstance(pair, str) or "/" not in pair:
            continue
        left, right = pair.split("/")
        case_a = f"{kind}_seed{seed}_{left}"
        case_b = f"{kind}_seed{seed}_{right}"
        if row.get("case_a") != case_a or row.get("case_b") != case_b:
            reasons.append(f"{kind} seed {seed} {pair}: identity")
            continue
        if row.get("thresholds") != PAIR_THRESHOLDS:
            reasons.append(f"{kind} seed {seed} {pair}: threshold definition")
            continue
        side_a = verified.get(case_a)
        side_b = verified.get(case_b)
        if not side_a or not side_b:
            reasons.append(f"{kind} seed {seed} {pair}: missing verified trace")
            continue
        if side_a["initial_state_sha256"] != side_b["initial_state_sha256"]:
            reasons.append(f"{kind} seed {seed} {pair}: initial state")
        if not _close(side_a["duration_s"], side_b["duration_s"]):
            reasons.append(f"{kind} seed {seed} {pair}: duration")
        _match_summary(kind, seed, pair, row, side_a["metrics"], side_b["metrics"], reasons)


def _match_summary(kind, seed, pair, row, left: dict, right: dict, reasons: list) -> None:
    label = f"{kind} seed {seed} {pair}"
    for name, key in (("peak_a", "peak_force_n"), ("peak_b", "peak_force_n"), ("scalar_a", "scalar_impulse_ns"), ("scalar_b", "scalar_impulse_ns")):
        side = left if name.endswith("_a") else right
        if name not in row or isinstance(row[name], bool) or not isinstance(row[name], (int, float)):
            reasons.append(f"{label}: {name} summary shape")
            return
        if not _close(float(row[name]), float(side[key])):
            reasons.append(f"{label}: {name} summary={row[name]}, trace={side[key]}; comparison inconsistent")
    for name, key in (("steady_a", "steady_force_n"), ("steady_b", "steady_force_n"), ("vector_a", "vector_impulse_ns"), ("vector_b", "vector_impulse_ns")):
        side = left if name.endswith("_a") else right
        value = np.asarray(row.get(name), dtype=float)
        if value.shape != (3,):
            reasons.append(f"{label}: {name} summary shape")
            return
        if not np.allclose(value, side[key], atol=RECOMPUTE_ATOL, rtol=RECOMPUTE_RTOL):
            reasons.append(f"{label}: {name} summary={row.get(name)}, trace={np.asarray(side[key]).tolist()}; comparison inconsistent")
    event_a, event_b = left["event_s"], right["event_s"]
    if kind in ("tilt", "accel", "lip"):
        event_ok = event_a is not None and event_b is not None and abs(event_a - event_b) <= PAIR_THRESHOLDS["event_s"]
        delta = None if event_a is None or event_b is None else abs(event_a - event_b)
    else:
        event_ok = "not_applicable"
        delta = None if event_a is None or event_b is None else abs(event_a - event_b)
    if row.get("event_a") != event_a and not _same_optional(row.get("event_a"), event_a):
        reasons.append(f"{label}: event_a summary={row.get('event_a')}, trace={event_a}; comparison inconsistent")
    if row.get("event_b") != event_b and not _same_optional(row.get("event_b"), event_b):
        reasons.append(f"{label}: event_b summary={row.get('event_b')}, trace={event_b}; comparison inconsistent")
    if not _same_optional(row.get("delta_event_s"), delta):
        reasons.append(f"{label}: delta_event_s summary={row.get('delta_event_s')}, trace={delta}; comparison inconsistent")
    if kind in ("plate", "spoon"):
        distance = float(np.linalg.norm(left["steady_food_pos_m"] - right["steady_food_pos_m"]))
        position_ok = distance <= PAIR_THRESHOLDS["steady_food_pos_m"]
    else:
        distance = None
        position_ok = "not_applicable"
    if not _same_optional(row.get("position_delta_m"), distance):
        reasons.append(f"{label}: position_delta_m summary={row.get('position_delta_m')}, trace={distance}; comparison inconsistent")
    steady_ok = _band(float(np.linalg.norm(left["steady_force_n"])), float(np.linalg.norm(right["steady_force_n"])), 0.001, 0.1)
    peak_ok = _band(left["peak_force_n"], right["peak_force_n"], 0.02, 0.1)
    vector_ok = _band(float(np.linalg.norm(left["vector_impulse_ns"])), float(np.linalg.norm(right["vector_impulse_ns"])), 1e-4, 0.1)
    scalar_ok = _band(left["scalar_impulse_ns"], right["scalar_impulse_ns"], 1e-4, 0.1)
    for name, got in (("steady_ok", steady_ok), ("peak_ok", peak_ok), ("vector_impulse_ok", vector_ok), ("scalar_impulse_ok", scalar_ok), ("event_ok", event_ok), ("position_ok", position_ok)):
        if row.get(name) != got:
            reasons.append(f"{label}: {name} summary={row.get(name)}, trace={got}; comparison inconsistent")
    passed = steady_ok and peak_ok and vector_ok and scalar_ok and event_ok is not False and position_ok is not False
    if row.get("conclusion") != ("passed" if passed else "failed"):
        reasons.append(f"{label}: conclusion")
    if not passed:
        reasons.append(f"{label}: gate failed")


def _same_optional(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _close(float(left), float(right))


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
