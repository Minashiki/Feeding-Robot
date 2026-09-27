"""V3 evidence checks. Thresholds live here; trajectory files cannot relax them."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from feedingrobot.controllers.guard import contact_allowed
from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.sim.report import assess_run

F6_POS_M = 0.02
F6_ROT_RAD = 0.1
SPEED_LIMIT = 0.5
TAU_RATE = 2000.0
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
F4_SPEED = 0.01
GEAR_POS = {"FREE": 0.02, "ACQUIRE": 0.01, "MOUTH": 0.005, "STOP": 0.02}
GEAR_GAIN = {
    "FREE": np.array([300.0, 8.0, 49.0, 0.8]),
    "ACQUIRE": np.array([150.0, 5.0, 35.0, 0.63]),
    "MOUTH": np.array([100.0, 3.0, 28.0, 0.49]),
}
WRENCH_FORCE = {"FREE": 8.0, "ACQUIRE": 4.0, "MOUTH": 2.0, "STOP": 8.0}
BOWL = {"bowl_bottom", "bowl_back", "bowl_left", "bowl_right", "bowl_front"}
GEAR_NORMAL = np.array([0.0, 1.0, 0.0])
RELEASE_DIRECTION = np.array([1.0, 0.0, 0.0])
RELEASE_FORCE_N = 3.0
RELEASE_RAMP_S = 1.2
COMMAND_LIN_ZERO = 1e-4
COMMAND_ANG_ZERO = 1e-4
FORCE_ZERO = 1e-6
TORQUE_ZERO = 1e-6
RELEASE_SCORING_VERSION = "v6-staged"
RAMP_FORCE_ATOL = 1e-6
PHYSICAL = {
    "pause": {"duration": 0.3, "pose": "seed"},
    "release": {"duration": 2.5, "pose": 0},
    "release_fast": {"duration": 2.5, "pose": 0},
    "gear": {"duration": 2.0, "pose": 0},
    "wrench": {"duration": 0.3, "pose": 0},
    "takeover": {"duration": 0.3, "pose": 0},
}
VARIANTS = {
    "A": {"dt": 0.001, "iterations": 50, "tolerance": 1e-8},
    "B": {"dt": 0.0005, "iterations": 50, "tolerance": 1e-8},
    "C": {"dt": 0.001, "iterations": 100, "tolerance": 1e-9},
}
STRUCTURAL = {
    "wrong_type",
    "missing_field",
    "missing_case",
    "missing_file",
    "evidence_manifest",
    "hash_mismatch",
    "sample_count",
    "bad_time",
    "event_missing",
    "run_incomplete",
    "input_hash",
    "input_set",
    "m1_failed",
    "nodeid",
    "teardown",
    "exitstatus",
    "unsupported_schema",
    "duplicate_path",
    "unknown_file",
    "nonfinite",
    "skipped",
    "manifest",
    "solver_mismatch",
    "unreadable",
    "scope_mismatch",
    "summary_mismatch",
    "initial_state",
    "invalid_rotation",
    "bad_offsets",
    "invalid_enum",
    "bad_shape",
    "bad_dtype",
    "event_mismatch",
    "io_error",
}
V6_R3_EXPECTED = {
    "v6_tau_limit": {"torque_rate"},
    "v6_gain": {"gain_path_mismatch"},
    "v6_tau_nan": {"nonfinite"},
    "v6_flag_nan": {"bad_dtype"},
    "v6_shape": {"bad_shape"},
    "v6_tick_dtype": {"bad_dtype"},
    "v6_normal_fault": {"unexpected_fault"},
    "v6_history": {"stimulus"},
    "v6_empty_grid": {"stimulus"},
    "v6_grid_boundary": {"stimulus"},
    "v6_grid_dt": {"stimulus"},
    "v6_early_unload": {"stimulus"},
    "v6_load_direction": {"stimulus"},
    "v6_load_time": {"stimulus"},
    "v6_delayed_cross": {"convergence"},
    "v6_initial_state": {"initial_state"},
    "v6_missing_c": {"missing_case", "missing_comparison"},
    "v6_comparison_drop": {"summary_mismatch"},
}

ADVERSARIAL_EXPECTED = {
    'auto_resume': {'auto_resume'},
    'bad_dtype_tick': {'bad_dtype'},
    'bad_offsets': {'bad_offsets'},
    'bad_rotation': {'invalid_rotation'},
    'bad_shape_k': {'bad_shape'},
    'comparisons_lie': {'summary_mismatch'},
    'f6_limit': {'reference_error', 'threshold_mismatch'},
    'f6_missing_rot': {'missing_field'},
    'f6_nan': {'nonfinite'},
    'failed_m1': {'m1_failed'},
    'fast_ramp': {'unload_duration'},
    'fast_skip': {'event_mismatch'},
    'fast_speed': {'release_speed'},
    'fault_tick': {'fault_tick'},
    'gain_huge': {'gain_path_mismatch'},
    'gear_fake_free': {'phase_gear_mismatch'},
    'hash_stale': {'hash_mismatch'},
    'input_changed': {'input_hash'},
    'joint_projection': {'joint_projection'},
    'k_nan': {'nonfinite'},
    'manifest_drop': {'evidence_manifest'},
    'manifest_dup': {'duplicate_path'},
    'manifest_empty': {'evidence_manifest'},
    'missing_A': {'missing_comparison'},
    'mode_aborted': {'unexpected_mode'},
    'mode_unknown': {'invalid_enum'},
    'node_missing': {'nodeid'},
    'pause_integral': {'pause_motion'},
    'pause_vref': {'reference_speed'},
    'q_100': {'joint_range'},
    'release_fake_time': {'event_missing'},
    'release_gap': {'bad_time'},
    'release_json': {'wrong_type'},
    'release_reload': {'external_reloaded'},
    'release_speed': {'release_speed'},
    'release_two_frames': {'event_missing', 'sample_count'},
    'run_incomplete': {'run_incomplete'},
    'singularity_stop': {'singularity_stop'},
    'slow_ramp_late': {'release_speed'},
    'slow_spike': {'release_speed'},
    'solver_label': {'solver_mismatch'},
    'stop_speed': {'stop_speed'},
    'stop_torque': {'stop_torque'},
    'teardown_fail': {'teardown'},
    **V6_R3_EXPECTED,
}
ADVERSARIAL = tuple(sorted(ADVERSARIAL_EXPECTED))


def reason(code, case_id=None, field=None, observed=None, expected=None, text=""):
    row = {"code": code, "text": text or code}
    if case_id is not None:
        row["case_id"] = case_id
    if field is not None:
        row["field"] = field
    if observed is not None and np.size(observed) == 1:
        value = float(np.asarray(observed))
        row["observed"] = None if not np.isfinite(value) else value
    if expected is not None and np.size(expected) == 1:
        value = float(np.asarray(expected))
        row["expected"] = None if not np.isfinite(value) else value
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    def _reject(token):
        raise ValueError(token)

    return json.loads(Path(path).read_text(), parse_constant=_reject)


def _finite(arr) -> bool:
    return isinstance(arr, np.ndarray) and arr.size > 0 and bool(np.all(np.isfinite(arr)))


def _nums(value):
    if isinstance(value, (bool, str)):
        raise TypeError("scalar type")
    arr = np.asarray(value, dtype=float)
    return arr


def case_kind(case_id: str) -> str:
    if case_id.startswith("V6-zero-j"):
        return "v6zero"
    if case_id.startswith("V6-zero-phys-") or case_id.startswith("V6-stop-"):
        return "v6trace"
    family = case_id.split("-", 1)[0]
    if family in PHYSICAL and case_id.count("-") == 2:
        return "physical"
    if case_id in {"F4-speed", "F6-pause1000"}:
        return "array"
    return "json"


def evidence_paths() -> list[str]:
    from feedingrobot.controllers.acceptance import required_case_ids

    rows = [
        "execution.json",
        "environment.json",
        "pytest.txt",
        "input_hash_before.json",
        "input_hash_after.json",
        "controller_config.json",
        "acceptance_config.json",
        "case_manifest.json",
        "baseline.json",
        "m1_regression_binding.json",
        "calibration.md",
    ]
    for case_id in required_case_ids():
        kind = case_kind(case_id)
        if kind in {"physical", "v6trace"}:
            rows.extend((f"cases/{case_id}.npz", f"cases/{case_id}.json"))
        elif kind == "array":
            rows.append(f"cases/{case_id}.npz")
        else:
            rows.append(f"cases/{case_id}.json")
    return rows


def _safe(run: Path, rel: str) -> Path | None:
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        return None
    path = (run / rel).resolve()
    root = run.resolve()
    if path != root and root not in path.parents:
        return None
    return path


def _add(reasons, code, case_id=None, field=None, observed=None, expected=None, text=""):
    reasons.append(reason(code, case_id, field, observed, expected, text))


def _same(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
        return bool(np.isclose(float(left), float(right), atol=1e-9, rtol=1e-7))
    return left == right


def compare_scalar(field, left, right, frac, floor):
    if not (np.isfinite(left) and np.isfinite(right)):
        return reason("nonfinite", field=field)
    allow = max(frac * max(float(left), float(right)), floor)
    delta = abs(float(left) - float(right))
    if delta > allow + 1e-12:
        return reason("convergence", field=field, observed=delta, expected=allow)
    return None


def compare_recorded_peaks(left: dict, right: dict) -> list[dict]:
    """Section 2.1 numbers must fail. Equal values and exact-floor deltas pass."""
    rows = []
    for field, frac, floor in (
        ("contact_force", 0.1, 0.02),
        ("tcp_force", 0.1, 0.02),
        ("tcp_torque", 0.1, 0.002),
    ):
        item = compare_scalar(field, left[field], right[field], frac, floor)
        if item is not None:
            rows.append(item)
    return rows


def _check_f6_pause(path: Path, reasons: list) -> None:
    case_id = "F6-pause1000"
    data = np.load(path, allow_pickle=False)
    if "limit" in data.files:
        limit = np.asarray(data["limit"]).reshape(-1)
        if limit.size != 1 or not np.isfinite(limit[0]) or abs(float(limit[0]) - F6_POS_M) > 1e-12:
            _add(reasons, "threshold_mismatch", case_id, "limit", None if limit.size != 1 else limit[0], F6_POS_M)
    for key in ("pos", "p_ref", "v_ref", "integral"):
        if key not in data.files:
            _add(reasons, "missing_field", case_id, key)
            return
    pos = np.asarray(data["pos"], dtype=float)
    pref = np.asarray(data["p_ref"], dtype=float)
    vref = np.asarray(data["v_ref"], dtype=float)
    integral = np.asarray(data["integral"], dtype=float)
    if not all(np.all(np.isfinite(item)) for item in (pos, pref, vref, integral)):
        _add(reasons, "nonfinite", case_id, "pos")
        return
    if pos.shape != (1000, 3) or pref.shape != (1000, 3) or vref.shape[0] != 1000 or integral.shape != (1000, 3):
        _add(reasons, "sample_count", case_id, "pos", pos.shape[0], 1000)
        return
    err = np.linalg.norm(pref - pos, axis=1)
    if np.any(err > F6_POS_M + 1e-9):
        _add(reasons, "reference_error", case_id, "position", float(np.max(err)), F6_POS_M)
    if np.any(np.linalg.norm(vref, axis=1) > 1e-9) or np.any(np.linalg.norm(integral, axis=1) > 1e-12):
        _add(reasons, "reference_error", case_id, "v_ref", float(np.max(np.linalg.norm(vref, axis=1))), 0.0)


def _check_f6_json(case_id: str, row: dict, reasons: list) -> None:
    try:
        pos = _nums(row["pos"])
        pref = _nums(row["p_ref"])
        vref = _nums(row["v_ref"])
    except (KeyError, TypeError, ValueError):
        _add(reasons, "missing_field", case_id, "pos")
        return
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(pref)) and np.all(np.isfinite(vref))):
        _add(reasons, "nonfinite", case_id, "pos")
        return
    if "limit" in row and abs(float(row["limit"]) - F6_POS_M) > 1e-12 and case_id != "F6-rot":
        _add(reasons, "threshold_mismatch", case_id, "limit", row["limit"], F6_POS_M)
    if case_id == "F6-rot" and "limit" in row and abs(float(row["limit"]) - F6_ROT_RAD) > 1e-12:
        _add(reasons, "threshold_mismatch", case_id, "limit", row["limit"], F6_ROT_RAD)
    perr = float(np.linalg.norm(pref.reshape(3) - pos.reshape(3)))
    if case_id == "F6-trans":
        if np.max(np.abs(pref.reshape(3) - np.array([0.01, 0.0, 0.0]))) > 1e-9:
            _add(reasons, "reference_error", case_id, "p_ref", perr, F6_POS_M)
        corr = _nums(row["correction"]) if "correction" in row else None
        if corr is None or np.max(np.abs(corr.reshape(3) - np.array([0.01, 0.0, 0.0]))) > 1e-9:
            _add(reasons, "reference_error", case_id, "correction")
    elif case_id == "F6-within":
        if np.max(np.abs(pref.reshape(3))) > 1e-12 or abs(perr - 0.005) > 1e-9:
            _add(reasons, "reference_error", case_id, "p_ref", perr, 0.005)
    elif perr > F6_POS_M + 1e-9:
        _add(reasons, "reference_error", case_id, "position", perr, F6_POS_M)
    if float(np.linalg.norm(vref)) > 1e-9:
        _add(reasons, "reference_error", case_id, "v_ref", float(np.linalg.norm(vref)), 0.0)
    if case_id in {"F6-rot", "F6-both"}:
        if "r_ref" not in row or "rot" not in row:
            _add(reasons, "missing_field", case_id, "r_ref")
            return
        r_ref = _nums(row["r_ref"]).reshape(3, 3)
        rot = _nums(row["rot"]).reshape(3, 3)
        if not (np.all(np.isfinite(r_ref)) and np.all(np.isfinite(rot))):
            _add(reasons, "nonfinite", case_id, "r_ref")
            return
        ang = float(np.linalg.norm(orientation_error(r_ref, rot)))
        if ang > F6_ROT_RAD + 1e-9:
            _add(reasons, "reference_error", case_id, "rotation", ang, F6_ROT_RAD)
    if case_id in {"F6-prohibit", "F6-stop", "F6-power-on"} and row.get("execution") != case_id.split("-", 1)[1].replace("-", "_"):
        expected = {"F6-prohibit": "prohibit", "F6-stop": "stop", "F6-power-on": "power_on"}[case_id]
        if row.get("execution") != expected:
            _add(reasons, "missing_field", case_id, "execution")


def _check_unit(case_id: str, row: dict, reasons: list, controller_version=None) -> None:
    if not isinstance(row, dict) or row.get("case_id") != case_id:
        _add(reasons, "missing_field", case_id, "case_id")
        return
    if case_id.startswith("F6-"):
        _check_f6_json(case_id, row, reasons)
        return
    if case_id.startswith("F4-j") or case_id in {"F4-pair", "F4-recover"}:
        outward = float(row["outward_speed"])
        if not np.isfinite(outward):
            _add(reasons, "nonfinite", case_id, "outward_speed")
        elif outward > 1e-9:
            _add(reasons, "reference_error", case_id, "outward_speed", outward, 1e-9)
        if case_id.endswith("-in") or case_id == "F4-recover":
            inward = float(row["inward_speed"])
            if not np.isfinite(inward) or inward <= 1e-6:
                _add(reasons, "reference_error", case_id, "inward_speed", inward, 1e-6)
        return
    if case_id.startswith("F4-scale-"):
        expected = {"F4-scale-0": 0.0, "F4-scale-0.1": 0.005, "F4-scale-1": 0.05}[case_id]
        limit = float(row["speed_limit"])
        actual = float(row["actual_speed"])
        if not (np.isfinite(limit) and np.isfinite(actual)):
            _add(reasons, "nonfinite", case_id, "actual_speed")
            return
        if abs(limit - expected) > 1e-12:
            _add(reasons, "threshold_mismatch", case_id, "speed_limit", limit, expected)
        if actual > expected + 1e-9:
            _add(reasons, "speed_limit", case_id, "actual_speed", actual, expected)
        return
    if case_id.startswith("F5-") and case_id not in {"F5-repeat", "F5-return-limits"}:
        _tag, src, dst, _ms, dt_s = case_id.split("-")
        dt = float(dt_s)
        gains = _nums(row["gains"])
        start = _nums(row["k_start"])
        if not (np.all(np.isfinite(gains)) and np.all(np.isfinite(start))):
            _add(reasons, "nonfinite", case_id, "gains")
            return
        if abs(float(row["dt"]) - dt) > 1e-15:
            _add(reasons, "threshold_mismatch", case_id, "dt", row["dt"], dt)
        gains_by_gear = {gear: value.copy() for gear, value in GEAR_GAIN.items()}
        if controller_version == "m2-full-v5-tracking5":
            gains_by_gear["FREE"][0] = 330.0
            gains_by_gear["FREE"][2] = 40.0
        target = gains_by_gear[src]
        other = gains_by_gear[dst]
        if "k_target" in row and np.max(np.abs(_nums(row["k_target"]) - target)) > 1e-9:
            _add(reasons, "threshold_mismatch", case_id, "k_target")
        allow = np.abs(target - other) * dt / 0.2 + 1e-9
        if np.any(np.abs(np.diff(gains, axis=0)) > allow + 1e-12):
            _add(reasons, "gain_jump", case_id, "gains")
        return
    if case_id == "F5-return-limits":
        during = float(row["limit_during"])
        after = float(row["limit_after"])
        if not (np.isfinite(during) and np.isfinite(after)):
            _add(reasons, "nonfinite", case_id, "limit_during")
        elif during > 0.01 + 1e-12 or abs(after - 0.05) > 1e-12:
            _add(reasons, "gain_jump", case_id, "limit")
        return
    if case_id == "F5-repeat":
        if row.get("finished") is not True:
            _add(reasons, "gain_jump", case_id, "finished")
        return
    if case_id.startswith("F2-com-"):
        vel = float(row["vel_error"])
        acc = float(row["acc_error"])
        if not (np.isfinite(vel) and np.isfinite(acc)):
            _add(reasons, "nonfinite", case_id, "vel_error")
        elif vel < 0 or acc < 0 or vel > 1e-8 or acc > 1e-5:
            _add(reasons, "reference_error", case_id, "vel_error", vel, 1e-8)
        return
    if case_id.startswith("F3-pulse-"):
        if row.get("reason") != "wrench" or int(row["fault_tick"]) != int(row["sample_tick"]) or row.get("next_mode") != "STOPPING" or int(row["filter_delta"]) != 2:
            _add(reasons, "event_missing", case_id, "fault_tick")
        return
    if case_id == "F7-commands":
        for key in ("nonfinite", "stale", "conflict", "idempotent"):
            if row.get(key) is not True:
                _add(reasons, "event_missing", case_id, key)
                return
        return
    if case_id == "reset-100":
        if row.get("clean") is not True or int(row.get("n", 0)) != 100:
            _add(reasons, "event_missing", case_id, "n")
        return
    if case_id.startswith("replay-"):
        q_err = float(row["q_error"])
        tcp = float(row["tcp_error"])
        wrench = float(row["wrench_error"])
        if not all(np.isfinite(v) for v in (q_err, tcp, wrench)):
            _add(reasons, "nonfinite", case_id, "q_error")
        elif q_err > 1e-9 or tcp > 1e-9 or wrench > 1e-7:
            _add(reasons, "reference_error", case_id, "q_error", q_err, 1e-9)


def _check_speed(path: Path, reasons: list) -> None:
    case_id = "F4-speed"
    data = np.load(path, allow_pickle=False)
    if "v_hist" not in data.files or "t" not in data.files or "p_ref" not in data.files:
        _add(reasons, "missing_field", case_id, "v_hist")
        return
    speed = np.asarray(data["v_hist"], dtype=float)
    times = np.asarray(data["t"], dtype=float)
    pref = np.asarray(data["p_ref"], dtype=float)
    if speed.size == 0 or times.size == 0:
        _add(reasons, "sample_count", case_id, "v_hist", 0, 2)
        return
    if not (np.all(np.isfinite(speed)) and np.all(np.isfinite(times)) and np.all(np.isfinite(pref))):
        _add(reasons, "nonfinite", case_id, "v_hist")
        return
    if "limit" in data.files and abs(float(np.asarray(data["limit"]).reshape(-1)[0]) - F4_SPEED) > 1e-15:
        _add(reasons, "threshold_mismatch", case_id, "limit", float(data["limit"]), F4_SPEED)
    if speed.ndim != 2 or speed.shape[1] != 6 or len(speed) < 2:
        _add(reasons, "sample_count", case_id, "v_hist", len(speed), 2)
        return
    if np.any(np.diff(times) <= 1e-12):
        _add(reasons, "bad_time", case_id, "t")
        return
    linear = np.linalg.norm(speed[:, :3], axis=1)
    if np.any(linear > F4_SPEED + 1e-9):
        _add(reasons, "speed_limit", case_id, "v_hist", float(np.max(linear)), F4_SPEED)
    step = pref[1:] - pref[:-1]
    dt = np.diff(times)
    if not np.allclose(step, speed[:-1, :3] * dt[:, None], atol=1e-9, rtol=1e-7):
        _add(reasons, "bad_time", case_id, "integration")


def _gear_of_phase(phase: str) -> str:
    if phase in {"SELECT", "TRANSPORT", "WAIT_READY"}:
        return "FREE"
    if phase == "ACQUIRE":
        return "ACQUIRE"
    if phase in {"APPROACH", "TRANSFER", "RETRACT"}:
        return "MOUTH"
    return "STOP"


def _target_force(geom1, geom2, force_on_1):
    names = {str(geom1), str(geom2)}
    if "spring_pad" not in names or not (names & BOWL):
        return None
    if str(geom1) in BOWL:
        return np.asarray(force_on_1, dtype=float)
    return -np.asarray(force_on_1, dtype=float)


def _rebuild_blocked(command, force_used, pref, tcp, dt):
    rebuilt = np.zeros(len(command), dtype=bool)
    timer = 0.0
    previous = np.asarray(tcp[0], dtype=float)
    for k in range(len(command)):
        cmd = np.asarray(command[k, :3], dtype=float)
        scale = float(np.linalg.norm(cmd))
        if scale > 1e-6:
            direction = cmd / scale
            oppose = -float(np.dot(np.asarray(force_used[k], dtype=float), direction))
            lag = float(np.dot(previous - tcp[k], direction))
            timer = timer + dt if oppose > 0.3 and lag > 0.002 else 0.0
            rebuilt[k] = timer >= 0.01 - 1e-15
        else:
            timer = 0.0
        previous = np.asarray(pref[k], dtype=float)
    return rebuilt


def _check_physical(case_id: str, run: Path, reasons: list) -> dict | None:
    family, seed_s, variant = case_id.split("-")
    seed = int(seed_s[1:])
    spec = PHYSICAL[family]
    var = VARIANTS[variant]
    n = int(round(spec["duration"] / var["dt"]))
    npz_path = run / "cases" / f"{case_id}.npz"
    js_path = run / "cases" / f"{case_id}.json"
    if js_path.is_file() and not npz_path.is_file():
        _add(reasons, "wrong_type", case_id, "npz")
        return None
    if not npz_path.is_file() or not js_path.is_file():
        _add(reasons, "missing_case", case_id, "npz")
        return None
    try:
        meta = _load(js_path)
        data = np.load(npz_path, allow_pickle=False)
    except ValueError:
        _add(reasons, "nonfinite", case_id, "json")
        return None
    from feedingrobot.controllers.v4fields import screen_arrays, semantic_arrays, hard_arrays

    found = screen_arrays(data, case_id, n, family, var["dt"])
    if found:
        reasons.extend(found)
        if family in {"release", "release_fast"} and any(item["code"] == "sample_count" for item in found):
            _add(reasons, "event_missing", case_id, "release_window")
        return None
    reasons.extend(hard_arrays(data, case_id, family, var["dt"]))
    config_path = run / "controller_config.json"
    config = _load(config_path) if config_path.is_file() else {}
    controller_version = config.get("version")
    reasons.extend(semantic_arrays(data, case_id, family, var["dt"], controller_version))
    needed = (
        "t_state",
        "tick",
        "q",
        "dq",
        "tcp_pos",
        "tcp_rot",
        "tcp_twist",
        "tau_applied",
        "tau_raw",
        "qpos0",
        "tau_before",
        "command",
        "p_ref",
        "r_ref",
        "v_ref",
        "K",
        "D",
        "wrench_compensated",
        "external_wrench",
        "mode",
        "phase",
        "contact_geom1",
        "contact_geom2",
        "contact_point",
        "contact_torque",
        "force_used",
        "pause_applied",
        "saturation_flags",
        "blocked_now",
        "transition_active",
        "contact_force",
        "contact_dist",
        "contact_offsets",
        "warnings",
    )
    for key in needed:
        if key not in data.files:
            _add(reasons, "missing_field", case_id, key)
            return None
    t_state = np.asarray(data["t_state"], dtype=float)
    tau = np.asarray(data["tau_applied"], dtype=float)
    tau_before = np.asarray(data["tau_before"], dtype=float).reshape(-1)
    if t_state.ndim != 1 or t_state.size < 2 or np.any(np.abs(np.diff(t_state) - var["dt"]) > 1e-9):
        _add(reasons, "bad_time", case_id, "dt")
    if t_state.shape != (n + 1,) or tau.shape != (n, 7) or tau_before.shape != (7,):
        _add(reasons, "sample_count", case_id, "t_state", int(t_state.shape[0]), n + 1)
        if family == "release":
            _add(reasons, "event_missing", case_id, "release_window")
        return None
    arrays = [t_state, tau, tau_before, np.asarray(data["q"], dtype=float), np.asarray(data["dq"], dtype=float), np.asarray(data["tcp_pos"], dtype=float), np.asarray(data["wrench_compensated"], dtype=float)]
    if any(not np.all(np.isfinite(item)) for item in arrays):
        _add(reasons, "nonfinite", case_id, "t_state")
        return None
    if np.any(np.abs(np.diff(t_state) - var["dt"]) > 1e-9):
        _add(reasons, "bad_time", case_id, "dt")
    tick = np.asarray(data["tick"], dtype=int)
    if tick.shape != (n + 1,) or np.any(np.diff(tick) != 1):
        _add(reasons, "bad_time", case_id, "tick")
    if abs(float(meta.get("dt", -1)) - var["dt"]) > 1e-15 or int(meta.get("iterations", -1)) != var["iterations"] or abs(float(meta.get("tolerance", -1)) - var["tolerance"]) > 1e-18:
        _add(reasons, "solver_mismatch", case_id, "dt")
    pose_expected = seed if spec["pose"] == "seed" else int(spec["pose"])
    if int(meta.get("pose_id", -1)) != pose_expected or int(meta.get("seed", -1)) != seed:
        _add(reasons, "initial_state", case_id, "pose_id", meta.get("pose_id"), pose_expected)
    q = np.asarray(data["q"], dtype=float)
    dq = np.asarray(data["dq"], dtype=float)
    if np.max(np.abs(dq)) > SPEED_LIMIT + 1e-9:
        _add(reasons, "joint_speed", case_id, "dq", float(np.max(np.abs(dq))), SPEED_LIMIT)
    if np.max(np.abs(tau[0] - tau_before)) > TAU_RATE * var["dt"] + 1e-9 or np.max(np.abs(np.diff(tau, axis=0))) > TAU_RATE * var["dt"] + 1e-9:
        _add(reasons, "torque_rate", case_id, "tau_applied")
    if np.any(np.abs(tau) > TAU_MAX + 1e-9):
        _add(reasons, "torque_rate", case_id, "tau_max")
    tcp = np.asarray(data["tcp_pos"], dtype=float)
    pref = np.asarray(data["p_ref"], dtype=float)
    phase = np.asarray(data["phase"]).astype(str)
    active = np.asarray(data["active_gear"]).astype(str) if "active_gear" in data.files else phase
    target = np.asarray(data["target_gear"]).astype(str) if "target_gear" in data.files else phase
    trans = np.asarray(data["transition_active"], dtype=float)
    pre_err = np.linalg.norm(pref - tcp[:-1], axis=1)
    for k in range(n):
        gears = [_gear_of_phase(phase[k])]
        if trans[k] > 0.5:
            gears = [_gear_of_phase(str(active[k])), _gear_of_phase(str(target[k]))]
        limit = min(GEAR_POS.get(item, F6_POS_M) for item in gears)
        if pre_err[k] > limit + 1e-9:
            _add(reasons, "reference_error", case_id, "position", float(pre_err[k]), limit)
            break
    warnings = np.asarray(data["warnings"], dtype=float)
    if np.any(warnings != 0):
        _add(reasons, "event_missing", case_id, "warnings")
    offsets = np.asarray(data["contact_offsets"], dtype=int)
    forces = np.asarray(data["contact_force"], dtype=float).reshape(-1, 3)
    dists = np.asarray(data["contact_dist"], dtype=float)
    geom1 = np.asarray(data["contact_geom1"]).astype(str)
    geom2 = np.asarray(data["contact_geom2"]).astype(str)
    if offsets.shape != (n + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != len(forces):
        _add(reasons, "missing_field", case_id, "contact_offsets")
        return None
    if forces.size and not np.all(np.isfinite(forces)):
        _add(reasons, "nonfinite", case_id, "contact_force")
        return None
    press = family == "gear"
    target_norm = np.zeros(n)
    target_sum = np.zeros(n)
    target_normal = np.zeros(n)
    scene_sum = np.zeros(n)
    for k in range(n):
        a, b = int(offsets[k]), int(offsets[k + 1])
        total = np.zeros(3)
        for j in range(a, b):
            if dists[j] < -0.001:
                _add(reasons, "reference_error", case_id, "contact_dist", float(dists[j]), -0.001)
            if not contact_allowed(str(phase[k]), str(geom1[j]), str(geom2[j]), {"press_test": press, "bowl_geoms": tuple(BOWL)}):
                _add(reasons, "event_missing", case_id, "contact_whitelist")
                break
            piece = _target_force(geom1[j], geom2[j], forces[j])
            scene_sum[k] += float(np.linalg.norm(forces[j]))
            if piece is not None:
                total += piece
                target_sum[k] += float(np.linalg.norm(piece))
        target_norm[k] = float(np.linalg.norm(total))
        target_normal[k] = float(np.dot(total, GEAR_NORMAL))
    post_err = np.linalg.norm(pref - tcp[1:], axis=1)
    wrench = np.asarray(data["wrench_compensated"], dtype=float)
    force_peak = float(np.max(np.linalg.norm(wrench[:, :3], axis=1)))
    torque_peak = float(np.max(np.linalg.norm(wrench[:, 3:], axis=1)))
    impulse = float(np.sum(np.linalg.norm(wrench[:, :3], axis=1) * var["dt"]))
    r_ref = np.asarray(data["r_ref"], dtype=float).reshape(n, 3, 3)
    tcp_rot = np.asarray(data["tcp_rot"], dtype=float).reshape(n + 1, 3, 3)
    ang = np.array([float(np.linalg.norm(orientation_error(r_ref[k], tcp_rot[k + 1]))) for k in range(n)])
    twist = np.asarray(data["tcp_twist"], dtype=float)
    metrics = {
        "pos_peak": float(np.max(post_err)),
        "pos_rms": float(np.sqrt(np.mean(post_err**2))),
        "rot_peak": float(np.max(ang)),
        "rot_rms": float(np.sqrt(np.mean(ang**2))),
        "contact_force": float(np.max(target_norm)),
        "contact_sum": float(np.max(scene_sum)),
        "tcp_force": force_peak,
        "tcp_torque": torque_peak,
        "impulse": impulse,
        "tau_peak": np.max(np.abs(tau), axis=0),
        "q0": np.asarray(data["q"], dtype=float)[0].copy(),
        "qpos0": np.asarray(data["qpos0"], dtype=float).reshape(-1).copy(),
        "qvel0": np.asarray(data["qvel0"], dtype=float).reshape(-1).copy(),
        "events": {},
        "times": {
            "pos_peak": float(t_state[int(np.argmax(post_err)) + 1]),
            "pos_rms": float(t_state[1]),
            "rot_peak": float(t_state[int(np.argmax(ang)) + 1]),
            "rot_rms": float(t_state[1]),
            "contact_force": float(t_state[int(np.argmax(target_norm)) + 1]),
            "contact_sum": float(t_state[int(np.argmax(scene_sum)) + 1]),
            "tcp_force": float(t_state[int(np.argmax(np.linalg.norm(wrench[:, :3], axis=1))) + 1]),
            "tcp_torque": float(t_state[int(np.argmax(np.linalg.norm(wrench[:, 3:], axis=1))) + 1]),
            "impulse": float(t_state[-1]),
        },
    }
    pause = np.asarray(data["pause_applied"]).astype(int) == 1
    blocked = np.asarray(data["blocked_now"], dtype=float) > 0.5
    external = np.asarray(data["external_wrench"], dtype=float)
    command = np.asarray(data["command"], dtype=float)
    if family == "pause":
        idx = np.flatnonzero(pause)
        if idx.size == 0:
            _add(reasons, "event_missing", case_id, "pause")
        else:
            metrics["events"]["pause"] = float(t_state[int(idx[0])])
        if not np.any((t_state[:-1] >= 0.05) & (t_state[:-1] < 0.08) & pause):
            _add(reasons, "event_missing", case_id, "pause_window")
    if family == "gear":
        idx = np.flatnonzero(target_norm > 0.0)
        switched = np.flatnonzero(phase == "ACQUIRE")
        if idx.size == 0 or switched.size == 0:
            _add(reasons, "event_missing", case_id, "contact")
        else:
            contact_t = float(t_state[int(idx[0]) + 1])
            switch_t = float(t_state[int(switched[0])])
            metrics["events"]["contact"] = contact_t
            metrics["events"]["switch"] = switch_t
            if switch_t < contact_t + 0.02 - 1e-9:
                _add(reasons, "event_missing", case_id, "contact_before_switch")
            window = (t_state[:-1] >= switch_t) & (t_state[:-1] < switch_t + 0.2)
            if not np.any(window) or np.any(target_normal[window] < 0.1 - 1e-9):
                _add(reasons, "event_missing", case_id, "hold_force")
            finished = (t_state[:-1] >= switch_t + 0.2 - 1e-9) & (phase == "ACQUIRE") & (trans < 0.5)
            if not np.any(finished):
                _add(reasons, "event_missing", case_id, "blend")
            switch_speed = float(np.linalg.norm(twist[int(switched[0]), :3]))
            if switch_speed > 0.02 + 1e-9:
                _add(reasons, "event_missing", case_id, "switch_speed", switch_speed, 0.02)
        from feedingrobot.controllers.v4fields import rebuild_gain_path

        built = rebuild_gain_path(phase, var["dt"], controller_version=controller_version)
        done = built.get("acquire_done")
        logged_k = np.asarray(data["K"], dtype=float)
        logged_d = np.asarray(data["D"], dtype=float)
        if done is not None and np.allclose(logged_k[done], built["K"][done], atol=1e-9, rtol=1e-7) and np.allclose(logged_d[done], built["D"][done], atol=1e-9, rtol=1e-7):
            metrics["events"]["gain_complete"] = float(t_state[int(done)])
    if family != "gear" and np.any(target_norm > 0):
        _add(reasons, "event_missing", case_id, "unexpected_contact")
    if family == "wrench":
        loaded = np.linalg.norm(external[:, :3], axis=1) > 0.5
        if not np.any(loaded):
            _add(reasons, "event_missing", case_id, "load")
        else:
            metrics["events"]["load"] = float(t_state[int(np.flatnonzero(loaded)[0])])
            if float(np.max(np.abs(wrench[loaded, 0]))) < 0.5:
                _add(reasons, "reference_error", case_id, "load_residual")
    if family == "takeover":
        if np.max(np.abs(tau_before)) > 1e-12:
            _add(reasons, "initial_state", case_id, "tau_before")
        mode = np.asarray(data["mode"]).astype(str)
        if mode.size == 0 or str(mode[0]) != "POWER_ON":
            _add(reasons, "event_missing", case_id, "power_on")
    force_used = np.asarray(data["force_used"], dtype=float).reshape(n, 3)
    if controller_version in {"m2-full-v5", "m2-full-v5-tracking5"}:
        from feedingrobot.validation.m2.legacy import rebuild_progress_events
        rebuilt, progress_errors = rebuild_progress_events(data, config, var["dt"])
        for field in progress_errors:
            _add(reasons, "event_mismatch", case_id, field)
    else:
        rebuilt = _rebuild_blocked(command, force_used, pref, tcp, var["dt"])
    if np.any(rebuilt & ~blocked):
        _add(reasons, "event_missing", case_id, "blocked_rebuild")
    if family in {"release", "release_fast"}:
        _score_release(reasons, metrics, case_id, family, rebuilt, command, external, twist, tcp, t_state, var["dt"], meta)
    return metrics


def _zero_masks(command, external):
    linear = np.linalg.norm(command[:, :3], axis=1)
    angular = np.linalg.norm(command[:, 3:], axis=1)
    force = np.linalg.norm(external[:, :3], axis=1)
    torque = np.linalg.norm(external[:, 3:], axis=1)
    return (linear <= COMMAND_LIN_ZERO) & (angular <= COMMAND_ANG_ZERO), (force <= FORCE_ZERO) & (torque <= TORQUE_ZERO)


def check_release_tail(reasons, case_id, command, external, t_state, unload_k, zero_k) -> None:
    """Zero command from unload_start and zero applied wrench from force_zero through the last interval."""
    n = int(len(command))
    if not (0 <= int(unload_k) <= int(zero_k) < n):
        _add(reasons, "event_mismatch", case_id, "release_tail")
        return
    unload_k = int(unload_k)
    zero_k = int(zero_k)
    linear = np.linalg.norm(command[:, :3], axis=1)
    angular = np.linalg.norm(command[:, 3:], axis=1)
    force = np.linalg.norm(external[:, :3], axis=1)
    torque = np.linalg.norm(external[:, 3:], axis=1)
    checks = (
        (linear[unload_k:] > COMMAND_LIN_ZERO, unload_k, "command_resumed", "command", COMMAND_LIN_ZERO, linear),
        (angular[unload_k:] > COMMAND_ANG_ZERO, unload_k, "angular_command_resumed", "command", COMMAND_ANG_ZERO, angular),
        (force[zero_k:] > FORCE_ZERO, zero_k, "external_reloaded", "external_wrench", FORCE_ZERO, force),
        (torque[zero_k:] > TORQUE_ZERO, zero_k, "external_torque_reloaded", "external_wrench", TORQUE_ZERO, torque),
    )
    for mask, origin, code, field, limit, values in checks:
        bad = np.flatnonzero(mask)
        if bad.size == 0:
            continue
        index = int(bad[0] + origin)
        _add(reasons, code, case_id, f"{field}[{index}]", float(values[index]), limit, text=f"{code} t={float(t_state[index])}")


def _step_count(dt, seconds):
    dt = float(dt)
    steps = int(round(float(seconds) / dt))
    if steps < 0 or abs(steps * dt - float(seconds)) > 1e-9:
        return None
    return steps


def _window_fits(length, start, steps) -> bool:
    return steps is not None and start >= 0 and start + steps < length


def _first_over(values, limit):
    bad = np.flatnonzero(np.asarray(values, dtype=float) > float(limit) + 1e-9)
    if bad.size == 0:
        return None
    return int(bad[0])


def _release_events(reasons, case_id, rebuilt, command, external):
    idx = np.flatnonzero(rebuilt)
    if idx.size == 0:
        _add(reasons, "event_missing", case_id, "blocked_first")
        return None
    blocked_k = int(idx[0])
    command_zero, wrench_zero = _zero_masks(command, external)
    if blocked_k + 1 >= len(command) or not bool(command_zero[blocked_k + 1]):
        _add(reasons, "event_mismatch", case_id, "unload_start")
        return None
    unload_k = blocked_k + 1
    if unload_k == 0 or bool(wrench_zero[unload_k - 1]):
        _add(reasons, "event_mismatch", case_id, "early_unload")
        return None
    zero = np.flatnonzero((np.arange(len(command)) >= unload_k) & wrench_zero)
    if zero.size == 0:
        _add(reasons, "event_mismatch", case_id, "force_zero")
        return None
    return blocked_k, unload_k, int(zero[0])


def _check_transient(reasons, case_id, label, tcp, twist, t_state, dt, k0):
    half = _step_count(dt, 0.5)
    if not _window_fits(len(tcp), k0, half) or twist.shape[0] < len(tcp):
        _add(reasons, "sample_count", case_id, label + "_window")
        return None
    speed = np.linalg.norm(twist[k0 : k0 + half + 1, :3], axis=1)
    fd = np.linalg.norm(np.diff(tcp[k0 : k0 + half + 1], axis=0), axis=1) / float(dt)
    extra = np.maximum(0.0, (tcp[k0 : k0 + half + 1] - tcp[k0]) @ RELEASE_DIRECTION)
    if speed.size != half + 1 or fd.size != half:
        _add(reasons, "sample_count", case_id, label + "_window")
        return None
    hit = _first_over(speed, 0.03)
    if hit is not None:
        index = k0 + hit
        _add(reasons, "release_speed", case_id, label + "_speed", float(speed[hit]), 0.03, text=f"release_speed {label}_speed t={float(t_state[index])} index={index}")
    else:
        hit_fd = _first_over(fd, 0.03)
        if hit_fd is not None:
            index = k0 + hit_fd
            _add(reasons, "release_speed", case_id, label + "_fd", float(fd[hit_fd]), 0.03, text=f"release_speed {label}_fd t={float(t_state[index])} index={index}")
    hit_x = _first_over(extra, 0.005)
    if hit_x is not None:
        index = k0 + hit_x
        _add(reasons, "release_speed", case_id, label + "_displacement", float(extra[hit_x]), 0.005, text=f"release_speed {label}_displacement t={float(t_state[index])} index={index}")
    return {"speed": float(np.max(speed)), "fd": float(np.max(fd)) if fd.size else 0.0, "disp": float(np.max(extra))}


def _check_stable(reasons, case_id, label, tcp, twist, t_state, dt, k0):
    start = _step_count(dt, 0.8)
    end = _step_count(dt, 1.0)
    if not _window_fits(len(tcp), k0, end) or start is None or twist.shape[0] < len(tcp):
        _add(reasons, "sample_count", case_id, label + "_stable_window")
        return None
    speed = np.linalg.norm(twist[k0 + start : k0 + end + 1, :3], axis=1)
    omega = np.linalg.norm(twist[k0 + start : k0 + end + 1, 3:], axis=1)
    fd = np.linalg.norm(np.diff(tcp[k0 + start : k0 + end + 1], axis=0), axis=1) / float(dt)
    expect = end - start + 1
    if speed.size != expect or omega.size != expect or fd.size != expect - 1:
        _add(reasons, "sample_count", case_id, label + "_stable_window")
        return None
    hit = _first_over(speed, 0.005)
    if hit is not None:
        index = k0 + start + hit
        _add(reasons, "release_speed", case_id, label + "_stable", float(speed[hit]), 0.005, text=f"release_speed {label}_stable t={float(t_state[index])} index={index}")
    else:
        hit_fd = _first_over(fd, 0.005)
        if hit_fd is not None:
            index = k0 + start + hit_fd
            _add(reasons, "release_speed", case_id, label + "_fd", float(fd[hit_fd]), 0.005, text=f"release_speed {label}_fd t={float(t_state[index])} index={index}")
    hit_w = _first_over(omega, 0.05)
    if hit_w is not None:
        index = k0 + start + hit_w
        _add(reasons, "release_speed", case_id, label + "_omega", float(omega[hit_w]), 0.05, text=f"release_speed {label}_omega t={float(t_state[index])} index={index}")
    return {"v": float(np.max(speed)), "w": float(np.max(omega))}


def _check_ramp_speed(reasons, case_id, tcp, twist, t_state, dt, unload_k, force_k):
    if force_k < unload_k or force_k >= len(tcp) or twist.shape[0] < len(tcp):
        _add(reasons, "sample_count", case_id, "ramp_window")
        return None
    speed = np.linalg.norm(twist[unload_k : force_k + 1, :3], axis=1)
    fd = np.linalg.norm(np.diff(tcp[unload_k : force_k + 1], axis=0), axis=1) / float(dt)
    if speed.size != force_k - unload_k + 1 or fd.size != force_k - unload_k:
        _add(reasons, "sample_count", case_id, "ramp_window")
        return None
    hit = _first_over(speed, 0.03)
    if hit is not None:
        index = unload_k + hit
        _add(reasons, "release_speed", case_id, "ramp_speed", float(speed[hit]), 0.03, text=f"release_speed ramp_speed t={float(t_state[index])} index={index}")
    elif fd.size:
        hit_fd = _first_over(fd, 0.03)
        if hit_fd is not None:
            index = unload_k + hit_fd
            _add(reasons, "release_speed", case_id, "ramp_fd", float(fd[hit_fd]), 0.03, text=f"release_speed ramp_fd t={float(t_state[index])} index={index}")
    return float(np.max(speed)) if speed.size else 0.0


def _check_slow_stimulus(reasons, case_id, external, t_state, dt, unload_k, force_k):
    t0 = float(t_state[unload_k])
    for k in range(unload_k, force_k):
        remain = max(1.0 - (float(t_state[k]) - t0) / RELEASE_RAMP_S, 0.0)
        expected = -RELEASE_FORCE_N * remain
        force = np.asarray(external[k], dtype=float)
        if abs(float(force[0]) - expected) > RAMP_FORCE_ATOL or float(np.linalg.norm(force[1:3])) > FORCE_ZERO or float(np.linalg.norm(force[3:])) > TORQUE_ZERO:
            _add(reasons, "event_mismatch", case_id, "ramp_force", float(force[0]), expected, text=f"event_mismatch ramp_force t={float(t_state[k])} index={k}")
            break
    if abs(float(t_state[force_k]) - (t0 + RELEASE_RAMP_S)) > float(dt) + 1e-9:
        _add(reasons, "unload_duration", case_id, "force_zero", float(t_state[force_k] - t0), RELEASE_RAMP_S)


def check_release_stimulus(reasons, case_id, command, external, dt, unload_k):
    # hook runs before compute and observes the previous compute timestamp.
    # At 0.04 s the corresponding physical interval is tick (0.04/dt + 1).
    load_k = _step_count(dt, 0.04) + 1
    push_k = _step_count(dt, 0.05)
    expected_force = np.zeros_like(external[:unload_k])
    expected_force[load_k:, 0] = -RELEASE_FORCE_N
    expected_command = np.zeros_like(command[:unload_k])
    expected_command[push_k:, 0] = 0.05
    if unload_k <= push_k:
        _add(reasons, "stimulus", case_id, "push_duration")
    for field, actual, expected, tolerance in (("hold_force", external[:unload_k], expected_force, RAMP_FORCE_ATOL), ("push_command", command[:unload_k], expected_command, 1e-9)):
        bad = np.flatnonzero(np.any(np.abs(actual - expected) > tolerance, axis=1))
        if bad.size:
            k = int(bad[0])
            _add(reasons, "stimulus", case_id, field, text=f"stimulus {field} index={k} t={k * dt:.6f}")


def _score_release(reasons, metrics, case_id, family, rebuilt, command, external, twist, tcp, t_state, dt, meta):
    found = _release_events(reasons, case_id, rebuilt, command, external)
    if found is None:
        return
    blocked_k, unload_k, force_k = found
    metrics["events"]["blocked_first"] = float(t_state[blocked_k])
    metrics["events"]["unload_start"] = float(t_state[unload_k])
    metrics["events"]["force_zero"] = float(t_state[force_k])
    if family == "release":
        metrics["events"]["blocked"] = float(t_state[blocked_k])
        metrics["events"]["release"] = float(t_state[force_k])
    check_release_stimulus(reasons, case_id, command, external, dt, unload_k)
    full = _step_count(dt, 1.0)
    if family == "release_fast":
        gap = float(t_state[force_k] - t_state[unload_k])
        if gap < -1e-15 or gap > float(dt) + 1e-9:
            _add(reasons, "unload_duration", case_id, "force_zero", gap, dt)
        if not _window_fits(len(tcp), unload_k, full) or not _window_fits(len(tcp), force_k, full):
            _add(reasons, "sample_count", case_id, "release_window")
    else:
        _check_slow_stimulus(reasons, case_id, external, t_state, dt, unload_k, force_k)
        if not _window_fits(len(tcp), unload_k, full) or not _window_fits(len(tcp), force_k, full):
            _add(reasons, "sample_count", case_id, "release_window")
        if "release_time" in meta and abs(float(meta["release_time"]) - float(t_state[force_k])) > 1e-9:
            _add(reasons, "event_missing", case_id, "release_time", meta.get("release_time"), float(t_state[force_k]))
    check_release_tail(reasons, case_id, command, external, t_state, unload_k, force_k)
    unload_peak = _check_transient(reasons, case_id, "unload", tcp, twist, t_state, dt, unload_k)
    if unload_peak is not None:
        metrics["unload_speed_peak"] = unload_peak["speed"]
        metrics["times"]["unload_speed_peak"] = float(t_state[unload_k])
    if family == "release":
        ramp_peak = _check_ramp_speed(reasons, case_id, tcp, twist, t_state, dt, unload_k, force_k)
        if ramp_peak is not None:
            metrics["ramp_speed_peak"] = ramp_peak
            metrics["times"]["ramp_speed_peak"] = float(t_state[unload_k])
        start = _step_count(dt, 0.8)
        end = _step_count(dt, 1.0)
        if start is not None and _window_fits(len(tcp), unload_k, end):
            speed = np.linalg.norm(twist[unload_k + start : unload_k + end + 1, :3], axis=1)
            omega = np.linalg.norm(twist[unload_k + start : unload_k + end + 1, 3:], axis=1)
            if speed.size:
                metrics["diagnostic_ramp_v"] = float(np.max(speed))
                metrics["diagnostic_ramp_w"] = float(np.max(omega))
                metrics["times"]["diagnostic_ramp_v"] = float(t_state[unload_k + start])
    else:
        held = _check_stable(reasons, case_id, "unload", tcp, twist, t_state, dt, unload_k)
        if held is not None:
            metrics["unload_stable_v"] = held["v"]
            metrics["times"]["unload_stable_v"] = float(t_state[unload_k])
    force_peak = _check_transient(reasons, case_id, "force_zero", tcp, twist, t_state, dt, force_k)
    if force_peak is not None:
        metrics["force_speed_peak"] = force_peak["speed"]
        metrics["times"]["force_speed_peak"] = float(t_state[force_k])
    settled = _check_stable(reasons, case_id, "force_zero", tcp, twist, t_state, dt, force_k)
    if settled is not None:
        metrics["force_stable_v"] = settled["v"]
        metrics["force_stable_w"] = settled["w"]
        metrics["times"]["force_stable_v"] = float(t_state[force_k])


def _abc(metrics: dict, reasons: list) -> list[dict]:
    rows = []
    pairs = (
        ("pos_peak", 0.1, 1e-4, "m"),
        ("pos_rms", 0.1, 1e-4, "m"),
        ("rot_peak", 0.1, float(np.deg2rad(0.05)), "rad"),
        ("rot_rms", 0.1, float(np.deg2rad(0.05)), "rad"),
        ("contact_force", 0.1, 0.02, "N"),
        ("contact_sum", 0.1, 0.02, "N"),
        ("tcp_force", 0.1, 0.02, "N"),
        ("tcp_torque", 0.1, 0.002, "N*m"),
        ("impulse", 0.1, 1e-4, "N*s"),
    )
    families = [(family, (0, 1, 2)) for family in PHYSICAL]
    families.extend((family, (None,)) for family in ("V6-zero-phys-upper", "V6-zero-phys-lower", "V6-stop-rest", "V6-stop-cross"))
    for family, seeds in families:
        for seed in seeds:
            prefix = f"{family}-s{seed}" if seed is not None else family
            base_id = f"{prefix}-A"
            base = metrics.get(base_id)
            if not base:
                _add(reasons, "missing_comparison", base_id, "A")
                continue
            for variant in ("B", "C"):
                other_id = f"{prefix}-{variant}"
                other = metrics.get(other_id)
                if not other:
                    _add(reasons, "missing_comparison", other_id, variant)
                    continue
                if base["qpos0"].shape != other["qpos0"].shape or np.max(np.abs(base["qpos0"] - other["qpos0"])) > 1e-8:
                    _add(reasons, "initial_state", other_id, "qpos0")
                if base["qvel0"].shape != other["qvel0"].shape or np.max(np.abs(base["qvel0"] - other["qvel0"])) > 1e-8:
                    _add(reasons, "initial_state", other_id, "qvel0")
                for key, frac, floor, unit in pairs:
                    item = compare_scalar(key, base[key], other[key], frac, floor)
                    passed = item is None
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "pair": f"A/{variant}",
                            "metric": key,
                            "unit": unit,
                            "a": float(base[key]),
                            "b": float(other[key]),
                            "a_time": base["times"][key],
                            "b_time": other["times"][key],
                            "delta": abs(float(base[key]) - float(other[key])),
                            "allow": max(frac * max(float(base[key]), float(other[key])), floor),
                            "passed": passed,
                        }
                    )
                    if item is not None:
                        item["case_id"] = other_id
                        reasons.append(item)
                for joint in range(7):
                    left = float(base["tau_peak"][joint])
                    right = float(other["tau_peak"][joint])
                    item = compare_scalar("tau_axis", left, right, 0.1, 0.05)
                    allow = max(0.1 * max(left, right), 0.05)
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "pair": f"A/{variant}",
                            "metric": f"tau_{joint}",
                            "unit": "N*m",
                            "a": left,
                            "b": right,
                            "a_time": None,
                            "b_time": None,
                            "delta": abs(left - right),
                            "allow": allow,
                            "passed": item is None,
                        }
                    )
                    if item is not None:
                        item["case_id"] = other_id
                        item["field"] = f"tau_{joint}"
                        reasons.append(item)
                events = set(base["events"]) | set(other["events"])
                if family.startswith("V6-stop-"):
                    events.update(("fault", "stop_speed", "stop_confirm"))
                elif family.startswith("V6-zero-"):
                    rows.append({"family": family, "seed": seed, "pair": f"A/{variant}", "metric": "fault", "role": "not_applicable", "passed": None})
                if family == "gear":
                    events.update(("contact", "switch", "gain_complete"))
                if family == "release":
                    events.update(("blocked", "blocked_first", "unload_start", "force_zero", "release"))
                if family == "release_fast":
                    events.update(("blocked_first", "unload_start", "force_zero"))
                release_pairs = []
                if family in {"release", "release_fast"}:
                    release_pairs.append(("unload_speed_peak", 0.1, 1e-4, "m/s"))
                    release_pairs.append(("force_stable_v", 0.1, 1e-4, "m/s"))
                if family == "release":
                    release_pairs.append(("ramp_speed_peak", 0.1, 1e-4, "m/s"))
                    release_pairs.append(("diagnostic_ramp_v", 0.1, 1e-4, "m/s"))
                for key, frac, floor, unit in release_pairs:
                    if key not in base or key not in other:
                        _add(reasons, "missing_comparison", other_id, key)
                        continue
                    item = compare_scalar(key, base[key], other[key], frac, floor)
                    passed = None if key == "diagnostic_ramp_v" else item is None
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "pair": f"A/{variant}",
                            "metric": key,
                            "unit": unit,
                            "role": "diagnostic_during_ramp" if key == "diagnostic_ramp_v" else "scored",
                            "a": float(base[key]),
                            "b": float(other[key]),
                            "a_time": base["times"].get(key),
                            "b_time": other["times"].get(key),
                            "delta": abs(float(base[key]) - float(other[key])),
                            "allow": max(frac * max(float(base[key]), float(other[key])), floor),
                            "passed": passed,
                        }
                    )
                    if key != "diagnostic_ramp_v" and item is not None:
                        item["case_id"] = other_id
                        reasons.append(item)
                if family == "pause":
                    events.add("pause")
                if family == "wrench":
                    events.add("load")
                for name in sorted(events):
                    ta = base["events"].get(name)
                    tb = other["events"].get(name)
                    if ta is None or tb is None:
                        _add(reasons, "event_missing", other_id, name)
                        continue
                    allow = 0.02
                    delta = abs(ta - tb)
                    rows.append({"family": family, "seed": seed, "pair": f"A/{variant}", "metric": name, "unit": "s", "a": ta, "b": tb, "a_time": ta, "b_time": tb, "delta": delta, "allow": allow, "passed": delta <= allow})
                    if delta > allow:
                        _add(reasons, "convergence", other_id, name, delta, allow)
    return rows


def _execution(execution: dict, reasons: list) -> None:
    from feedingrobot.controllers.acceptance import required_nodeids

    if int(execution.get("exitstatus", 1)) != 0:
        _add(reasons, "exitstatus")
    reports = execution.get("reports") or []
    if any(row.get("outcome") in {"skipped", "xfailed"} for row in reports):
        _add(reasons, "skipped")
    if any(row.get("when") == "teardown" and row.get("outcome") != "passed" for row in reports):
        _add(reasons, "teardown")
    collected = list(execution.get("collected") or [])
    calls, setups, teardowns = {}, {}, {}
    for row in reports:
        bucket = {"call": calls, "setup": setups, "teardown": teardowns}.get(row.get("when"))
        if bucket is not None:
            bucket.setdefault(row.get("nodeid"), []).append(row.get("outcome"))
    for node in required_nodeids():
        if node not in collected or calls.get(node) != ["passed"] or setups.get(node) != ["passed"] or teardowns.get(node) != ["passed"]:
            _add(reasons, "nodeid", field=node)


def _manifest_ok(run: Path, reasons: list) -> None:
    path = run / "evidence_manifest.json"
    if not path.is_file():
        _add(reasons, "evidence_manifest", field="evidence_manifest.json")
        return
    try:
        rows = _load(path).get("files") or []
    except ValueError:
        _add(reasons, "nonfinite", field="evidence_manifest.json")
        return
    seen = []
    for row in rows:
        rel = row.get("path")
        if rel in seen:
            _add(reasons, "duplicate_path", field=rel)
            return
        seen.append(rel)
        target = _safe(run, rel or "")
        if target is None or not target.is_file():
            _add(reasons, "evidence_manifest", field=rel)
            return
        if _sha256(target) != row.get("sha256"):
            _add(reasons, "hash_mismatch", field=rel)
            return
    expected = set(evidence_paths())
    extra = [item for item in seen if item not in expected]
    if any(item not in {"adversarial_execution.json", "adversarial_results.json"} for item in extra) or not expected.issubset(set(seen)):
        _add(reasons, "evidence_manifest", field="set")


def evaluate_evidence(run, check_workspace: bool = False) -> dict:
    from feedingrobot.controllers.acceptance import INPUT_PREFIXES, REMAINING_FULL_M2, collect_inputs, required_case_ids

    run = Path(run)
    reasons = []
    for name in (
        "report.json",
        "run.json",
        "execution.json",
        "case_manifest.json",
        "evidence_manifest.json",
        "input_hash_before.json",
        "input_hash_after.json",
        "controller_config.json",
        "m1_regression_binding.json",
        "baseline.json",
    ):
        if not (run / name).is_file():
            _add(reasons, "missing_file", field=name)
    expected = required_case_ids()
    if (run / "case_manifest.json").is_file():
        try:
            listed = list(_load(run / "case_manifest.json").get("cases") or [])
        except ValueError:
            listed = []
            _add(reasons, "nonfinite", field="case_manifest.json")
        if listed != expected:
            _add(reasons, "manifest", field="case_manifest.json")
    _manifest_ok(run, reasons)
    from feedingrobot.controllers.v6checks import CheckContext

    try:
        controller_version = _load(run / "controller_config.json").get("version") if (run / "controller_config.json").is_file() else None
    except ValueError:
        controller_version = None  # The configuration check below reports the invalid JSON.
    context = CheckContext(controller_version)
    metrics = {}
    if (run / "cases").is_dir():
        known = set(expected)
        for path in (run / "cases").iterdir():
            if path.stem not in known:
                _add(reasons, "unknown_file", field=path.name)
        for case_id in expected:
            kind = case_kind(case_id)
            try:
                if kind == "physical":
                    metrics[case_id] = _check_physical(case_id, run, reasons)
                elif kind == "v6zero":
                    from feedingrobot.controllers.v6checks import check_zero_grid

                    target = run / "cases" / f"{case_id}.json"
                    if not target.is_file():
                        _add(reasons, "missing_case", case_id, "json")
                    else:
                        reasons.extend(check_zero_grid(case_id, _load(target), context))
                elif kind == "v6trace":
                    from feedingrobot.controllers.v6checks import check_trace

                    npz_path = run / "cases" / f"{case_id}.npz"
                    js_path = run / "cases" / f"{case_id}.json"
                    if not npz_path.is_file() or not js_path.is_file():
                        _add(reasons, "missing_case", case_id, "npz")
                    else:
                        derived = {}
                        with np.load(npz_path, allow_pickle=False) as data:
                            reasons.extend(check_trace(case_id, data, _load(js_path), context, derived))
                        metrics[case_id] = derived
                elif kind == "array" and case_id == "F6-pause1000":
                    target = run / "cases" / f"{case_id}.npz"
                    if not target.is_file():
                        _add(reasons, "missing_case", case_id, "npz")
                    else:
                        _check_f6_pause(target, reasons)
                elif kind == "array":
                    target = run / "cases" / f"{case_id}.npz"
                    if not target.is_file():
                        _add(reasons, "missing_case", case_id, "npz")
                    else:
                        _check_speed(target, reasons)
                else:
                    target = run / "cases" / f"{case_id}.json"
                    if (run / "cases" / f"{case_id}.npz").is_file():
                        _add(reasons, "wrong_type", case_id, "json")
                    if not target.is_file():
                        _add(reasons, "missing_case", case_id, "json")
                    else:
                        _check_unit(case_id, _load(target), reasons, controller_version)
            except ValueError:
                _add(reasons, "nonfinite", case_id, "parse")
            except (OSError, KeyError, TypeError):
                _add(reasons, "missing_field", case_id, "parse")
    comparisons = _abc({key: value for key, value in metrics.items() if value}, reasons)
    if (run / "run.json").is_file():
        try:
            if _load(run / "run.json").get("incomplete") is True:
                _add(reasons, "run_incomplete")
        except ValueError:
            _add(reasons, "nonfinite", field="run.json")
    if (run / "execution.json").is_file():
        try:
            _execution(_load(run / "execution.json"), reasons)
        except ValueError:
            _add(reasons, "nonfinite", field="execution.json")
    before = after = None
    if (run / "input_hash_before.json").is_file() and (run / "input_hash_after.json").is_file():
        try:
            before = _load(run / "input_hash_before.json")
            after = _load(run / "input_hash_after.json")
        except ValueError:
            before = after = None
            _add(reasons, "nonfinite", field="input_hash")
        if before is not None and after is not None:
            if before.get("files") != after.get("files") or before.get("aggregate_sha256") != after.get("aggregate_sha256"):
                _add(reasons, "input_hash")
            recorded = {row["path"]: row["sha256"] for row in after.get("files") or []}
            if not recorded or any(not any(path == prefix or path.startswith(prefix) for path in recorded) for prefix in INPUT_PREFIXES):
                _add(reasons, "input_set")
            if check_workspace:
                try:
                    current = {row["path"]: row["sha256"] for row in collect_inputs()["files"]}
                except RuntimeError:
                    current = {}
                    _add(reasons, "input_set")
                if current != recorded:
                    _add(reasons, "input_set")
    binding = {}
    if (run / "m1_regression_binding.json").is_file():
        try:
            binding = _load(run / "m1_regression_binding.json")
        except ValueError:
            _add(reasons, "nonfinite", field="m1_regression_binding.json")
    regression = Path(binding.get("regression_dir", ""))
    if not regression.is_dir() or not assess_run(regression).get("m2_ready"):
        _add(reasons, "m1_failed", field="regression")
    elif after is not None:
        shared_path = regression / "input_hash_after.json"
        if shared_path.is_file():
            shared = {row["path"]: row["sha256"] for row in _load(shared_path).get("files") or []}
            recorded = {row["path"]: row["sha256"] for row in after.get("files") or []}
            for path, digest in shared.items():
                if path in recorded and recorded[path] != digest:
                    _add(reasons, "input_hash", field=path)
                    break
    baseline = Path(binding.get("baseline_dir", ""))
    if not baseline.is_dir() or not assess_run(baseline).get("m2_ready"):
        _add(reasons, "m1_failed", field="baseline")
    if (run / "controller_config.json").is_file():
        try:
            cfg = _load(run / "controller_config.json")
        except ValueError:
            cfg = {}
            _add(reasons, "nonfinite", field="controller_config.json")
        if cfg.get("hybrid_normal_force", {}).get("enabled") is not False or float(cfg.get("tau_rate_nm_s", 0)) != TAU_RATE:
            _add(reasons, "threshold_mismatch", field="controller_config")
        if after is not None:
            expected_hash = next((row['sha256'] for row in after.get('files', [])
                                  if row['path'] == 'configs/m2_controller.json'), None)
            if expected_hash != hashlib.sha256((run / 'controller_config.json').read_bytes()).hexdigest():
                _add(reasons, "input_hash", field="controller_config.json")
    if (run / "comparisons.json").is_file():
        try:
            stored = _load(run / "comparisons.json")
        except ValueError:
            stored = None
            _add(reasons, "nonfinite", field="comparisons.json")
        if stored is not None and not _same(stored, comparisons):
            _add(reasons, "summary_mismatch", field="comparisons.json")
    structural = [item for item in reasons if item["code"] in STRUCTURAL]
    metric = [item for item in reasons if item["code"] not in STRUCTURAL and item["code"] != "full_incomplete"]
    return {
        "evidence_valid": not structural,
        "tests_passed": not any(item["code"] in {"nodeid", "teardown", "exitstatus", "skipped"} for item in reasons),
        "fixes_passed": not reasons,
        "m3_ready": False,
        "hybrid_force_status": "disabled",
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "reasons": reasons,
        "comparisons": comparisons,
        "metric_failed": bool(metric),
    }


def _unlink_write(path: Path, text: str) -> None:
    if path.exists():
        path.unlink()
    path.write_text(text)


def _rehash(run: Path, rel: str) -> None:
    manifest_path = run / "evidence_manifest.json"
    manifest = _load(manifest_path)
    digest = _sha256(run / rel)
    found = False
    for row in manifest["files"]:
        if row["path"] == rel:
            row["sha256"] = digest
            found = True
    if not found:
        manifest["files"].append({"path": rel, "sha256": digest})
    _unlink_write(manifest_path, json.dumps(manifest))


def _save_npz(path: Path, data) -> None:
    payload = {key: data[key] for key in data.files}
    if path.exists():
        path.unlink()
    np.savez(path, **payload)


def apply_tamper(name: str, run: Path) -> None:
    if name in V6_R3_EXPECTED:
        apply_v6_r3_tamper(name, run)
    elif name == "f6_limit":
        path = run / "cases" / "F6-pause1000.npz"
        data = np.load(path, allow_pickle=False)
        payload = {key: data[key] for key in data.files}
        payload["pos"] = np.zeros((1000, 3))
        payload["p_ref"] = np.zeros((1000, 3))
        payload["p_ref"][:, 0] = 5.0
        payload["limit"] = np.array(10.0)
        if path.exists():
            path.unlink()
        np.savez(path, **payload)
        _rehash(run, "cases/F6-pause1000.npz")
    elif name == "f6_nan":
        path = run / "cases" / "F6-pause1000.npz"
        data = np.load(path, allow_pickle=False)
        payload = {key: np.array(data[key]) for key in data.files}
        payload["pos"] = np.array(payload["pos"])
        payload["pos"][0, 0] = np.nan
        if path.exists():
            path.unlink()
        np.savez(path, **payload)
        _rehash(run, "cases/F6-pause1000.npz")
    elif name == "f6_missing_rot":
        path = run / "cases" / "F6-rot.json"
        row = _load(path)
        row.pop("r_ref", None)
        row.pop("rot", None)
        row["passed"] = True
        _unlink_write(path, json.dumps(row))
        _rehash(run, "cases/F6-rot.json")
    elif name == "manifest_empty":
        _unlink_write(run / "evidence_manifest.json", json.dumps({"files": []}))
    elif name == "manifest_drop":
        manifest = _load(run / "evidence_manifest.json")
        manifest["files"] = manifest["files"][1:]
        _unlink_write(run / "evidence_manifest.json", json.dumps(manifest))
    elif name == "manifest_dup":
        manifest = _load(run / "evidence_manifest.json")
        manifest["files"].append(dict(manifest["files"][0]))
        _unlink_write(run / "evidence_manifest.json", json.dumps(manifest))
    elif name == "hash_stale":
        path = run / "cases" / "F4-speed.npz"
        blob = path.read_bytes()
        if path.exists():
            path.unlink()
        path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    elif name == "release_json":
        npz = run / "cases" / "release-s0-A.npz"
        if npz.exists():
            npz.unlink()
        _unlink_write(run / "cases" / "release-s0-A.json", json.dumps({"case_id": "release-s0-A"}))
        manifest = _load(run / "evidence_manifest.json")
        files = []
        for row in manifest["files"]:
            if row["path"] == "cases/release-s0-A.npz":
                files.append({"path": "cases/release-s0-A.json", "sha256": _sha256(run / "cases" / "release-s0-A.json")})
            else:
                files.append(row)
        _unlink_write(run / "evidence_manifest.json", json.dumps({"files": files}))
    elif name == "release_two_frames":
        for path in (run / "cases").glob("release-*.npz"):
            _slice_trace(run, path.stem, 1)
            _rehash(run, f"cases/{path.name}")
    elif name == "release_gap":
        _drop_index(run, "release-s0-A", 10)
        _rehash(run, "cases/release-s0-A.npz")
    elif name == "release_fake_time":
        path = run / "cases" / "release-s0-A.npz"
        data = np.load(path, allow_pickle=False)
        payload = {key: np.array(data[key]) for key in data.files}
        payload["external_wrench"] = np.zeros_like(payload["external_wrench"])
        payload["blocked_now"] = np.zeros_like(payload["blocked_now"])
        if path.exists():
            path.unlink()
        np.savez(path, **payload)
        meta = _load(run / "cases" / "release-s0-A.json")
        meta["release_time"] = 0.05
        _unlink_write(run / "cases" / "release-s0-A.json", json.dumps(meta))
        _rehash(run, "cases/release-s0-A.npz")
        _rehash(run, "cases/release-s0-A.json")
    elif name == "release_speed":
        path = run / "cases" / "release-s0-A.npz"
        data = np.load(path, allow_pickle=False)
        payload = {key: np.array(data[key]) for key in data.files}
        payload["tcp_pos"] = np.array(payload["tcp_pos"])
        payload["tcp_twist"] = np.array(payload["tcp_twist"])
        # After blocked (0.148 s). Index 100 is still in the push and erases the rebuilt blocked event.
        start = int(np.argmin(np.abs(np.asarray(payload["t_state"], dtype=float) - 0.20)))
        payload["tcp_pos"][start:, 0] += 0.2
        payload["tcp_twist"][start:, 0] = 0.2
        if path.exists():
            path.unlink()
        np.savez(path, **payload)
        _rehash(run, "cases/release-s0-A.npz")
    elif name == "missing_A":
        for suffix in (".npz", ".json"):
            path = run / "cases" / f"pause-s0-A{suffix}"
            if path.exists():
                path.unlink()
        manifest = _load(run / "evidence_manifest.json")
        manifest["files"] = [row for row in manifest["files"] if not row["path"].startswith("cases/pause-s0-A")]
        _unlink_write(run / "evidence_manifest.json", json.dumps(manifest))
    elif name == "comparisons_lie":
        path = run / "comparisons.json"
        rows = _load(path)
        if rows:
            rows[0]["a"] = 0.0
            rows[0]["passed"] = True
        _unlink_write(path, json.dumps(rows))
    elif name == "solver_label":
        path = run / "cases" / "gear-s0-A.json"
        meta = _load(path)
        meta["dt"] = 0.01
        _unlink_write(path, json.dumps(meta))
        _rehash(run, "cases/gear-s0-A.json")
    elif name == "node_missing":
        path = run / "execution.json"
        row = _load(path)
        dropped = "tests/test_m2_math.py::test_t01_so3_and_quaternion_roundtrip"
        row["collected"] = [node for node in row.get("collected") or [] if node != dropped]
        row["reports"] = [item for item in row.get("reports") or [] if item.get("nodeid") != dropped]
        _unlink_write(path, json.dumps(row))
    elif name == "teardown_fail":
        path = run / "execution.json"
        row = _load(path)
        row["exitstatus"] = 1
        for item in row.get("reports") or []:
            if item.get("when") == "teardown":
                item["outcome"] = "failed"
                break
        _unlink_write(path, json.dumps(row))
    elif name == "run_incomplete":
        path = run / "run.json"
        row = _load(path)
        row["incomplete"] = True
        _unlink_write(path, json.dumps(row))
    elif name == "input_changed":
        path = run / "input_hash_after.json"
        row = _load(path)
        row["aggregate_sha256"] = "0" * 64
        _unlink_write(path, json.dumps(row))
    elif name == "failed_m1":
        bad = run / "_bad_m1"
        bad.mkdir(exist_ok=True)
        (bad / "run.json").write_text(json.dumps({"schema_version": "nope", "status": "failed", "incomplete": True}))
        path = run / "m1_regression_binding.json"
        row = _load(path)
        row["regression_dir"] = str(bad)
        _unlink_write(path, json.dumps(row))
    elif name == "k_nan":
        _edit_case(run, "pause-s0-A", lambda payload: payload["K"].__setitem__((0, 0), np.nan))
    elif name == "gain_huge":
        _edit_case(run, "pause-s0-A", lambda payload: payload.__setitem__("K", np.full_like(payload["K"], 1.0e6)))
    elif name == "pause_vref":
        def _vref(payload):
            mask = np.array(payload["pause_applied"]).astype(int) == 1
            payload["v_ref"][mask, 0] = 5.0
        _edit_case(run, "pause-s0-A", _vref)
    elif name == "pause_integral":
        def _task(payload):
            mask = np.array(payload["pause_applied"]).astype(int) == 1
            payload["p_ref"][mask, 0] += 0.01
        _edit_case(run, "pause-s0-A", _task)
    elif name == "q_100":
        _edit_case(run, "pause-s0-A", lambda payload: payload["q"].__setitem__((10, 0), 100.0))
    elif name == "mode_aborted":
        _edit_case(run, "pause-s0-A", lambda payload: payload.__setitem__("mode", np.array(["ABORTED"] * len(payload["mode"]))))
    elif name == "mode_unknown":
        _edit_case(run, "wrench-s0-A", lambda payload: payload.__setitem__("mode", np.array(["NOPE"] * len(payload["mode"]))))
    elif name == "bad_shape_k":
        _edit_case(run, "pause-s0-A", lambda payload: payload.__setitem__("K", np.concatenate([payload["K"], payload["K"][:, :1]], axis=1)))
    elif name == "bad_dtype_tick":
        _edit_case(run, "pause-s0-A", lambda payload: payload.__setitem__("tick", np.array(payload["tick"], dtype=float)))
    elif name == "bad_offsets":
        _edit_case(run, "pause-s0-A", lambda payload: payload["contact_offsets"].__setitem__(-1, -1))
    elif name == "bad_rotation":
        _edit_case(run, "pause-s0-A", lambda payload: payload["r_ref"].__setitem__((0, 0, 0), 2.0))
    elif name == "fast_ramp":
        def _ramp(payload):
            ext = np.array(payload["external_wrench"])
            zero = np.flatnonzero(np.linalg.norm(ext[:, :3], axis=1) < 1e-6)
            if zero.size:
                start = int(zero[0])
                ext[start : start + 1200, 0] = -3.0
            payload["external_wrench"] = ext
        _edit_case(run, "release_fast-s0-A", _ramp)
    elif name == "fast_skip":
        def _skip(payload):
            payload["command"] = np.array(payload["command"])
            payload["command"][:, 0] = 0.05
        _edit_case(run, "release_fast-s0-A", _skip)
    elif name == "fast_speed":
        def _speed(payload):
            payload["tcp_pos"] = np.array(payload["tcp_pos"])
            payload["tcp_twist"] = np.array(payload["tcp_twist"])
            payload["tcp_pos"][400:, 0] += 0.2
            payload["tcp_twist"][400:, 0] = 0.2
        _edit_case(run, "release_fast-s0-A", _speed)
    elif name == "gear_fake_free":
        version = _load(run / "controller_config.json").get("version")
        free_kp = 330.0 if version == "m2-full-v5-tracking5" else 300.0
        free_dp = 40.0 if version == "m2-full-v5-tracking5" else 49.0
        def _fake_free(payload):
            payload["K"][:] = np.array([free_kp, free_kp, free_kp, 8.0, 8.0, 8.0], dtype=payload["K"].dtype)
            payload["D"][:] = np.array([free_dp, free_dp, free_dp, 0.8, 0.8, 0.8], dtype=payload["D"].dtype)
            payload["transition_active"][:] = 0
            payload["active_gear"][:] = "FREE"
            payload["target_gear"][:] = "FREE"
        _edit_case(run, "gear-s0-A", _fake_free)
    elif name == "release_reload":
        def _reload(payload):
            times = np.array(payload["t_state"][:-1], dtype=float)
            mask = (times >= 0.5) & (times < 0.6)
            payload["external_wrench"][mask, 0] = -3.0
        _edit_case(run, "release_fast-s0-A", _reload)
    elif name == "slow_spike":
        def _spike(payload):
            times = np.asarray(payload["t_state"], dtype=float)
            index = int(np.argmin(np.abs(times - 0.3)))
            payload["tcp_twist"] = np.array(payload["tcp_twist"])
            payload["tcp_twist"][index, 0] = 0.5
        _edit_case(run, "release-s0-A", _spike)
    elif name == "slow_ramp_late":
        def _late(payload):
            times = np.asarray(payload["t_state"], dtype=float)
            index = int(np.argmin(np.abs(times - 1.1)))
            payload["tcp_twist"] = np.array(payload["tcp_twist"])
            payload["tcp_twist"][index, 0] = 0.5
        _edit_case(run, "release-s0-A", _late)
    elif name == "joint_projection":
        path = run / "cases" / "V6-zero-j0-upper-linear-A.json"
        row = _load(path)
        row["beta"] = 0.0
        row["final"] = list(row["candidate"])
        row["p_after"] = (np.asarray(row["p_before"], dtype=float) + np.asarray(row["candidate"], dtype=float)[:3] * float(row["dt"])).tolist()
        row["correction"] = [0.0, 0.0, 0.0]
        _unlink_write(path, json.dumps(row))
        _rehash(run, "cases/V6-zero-j0-upper-linear-A.json")
    elif name == "singularity_stop":
        def _running(payload):
            payload["mode"] = np.array(["RUNNING"] * len(payload["mode"]))
            payload["guard_status"] = np.array(["RUNNING"] * len(payload["guard_status"]))
            payload["execution"] = np.array(["run"] * len(payload["execution"]))
        _edit_case(run, "V6-stop-rest-A", _running)
    elif name == "stop_torque":
        def _task(payload):
            payload["tau_task"] = np.array(payload["tau_task"])
            payload["tau_task"][:, 0] = 1.0
        _edit_case(run, "V6-stop-rest-A", _task)
    elif name == "fault_tick":
        def _tick(payload):
            payload["fault_tick"] = np.int64(int(np.asarray(payload["fault_tick"]).reshape(-1)[0]) + 50)
        _edit_case(run, "V6-stop-rest-A", _tick)
    elif name == "auto_resume":
        def _resume(payload):
            payload["command"] = np.array(payload["command"])
            payload["command"][-1, 0] = 0.05
        _edit_case(run, "V6-stop-rest-A", _resume)
    elif name == "stop_speed":
        def _fast(payload):
            payload["dq"] = np.array(payload["dq"])
            payload["dq"][:, 1] = 0.1
            payload["stopped_ok"] = np.int8(1)
        _edit_case(run, "V6-stop-rest-A", _fast)
    else:
        raise KeyError(name)


def apply_v6_r3_tamper(name, run):
    zero = "V6-zero-phys-upper-A"
    if name in {"v6_empty_grid", "v6_grid_boundary", "v6_grid_dt"}:
        rel = "cases/V6-zero-j0-upper-linear-A.json"
        row = _load(run / rel)
        if name == "v6_empty_grid":
            row.update(jbar=np.zeros((7, 6)).tolist(), candidate=[0.] * 6, final=[0.] * 6, p_after=row["p_before"])
        elif name == "v6_grid_boundary":
            row["q"][0] -= 0.5
        else:
            row["dt"] = 0.0005
        _unlink_write(run / rel, json.dumps(row))
        _rehash(run, rel)
        return
    if name == "v6_missing_c":
        (run / "cases/V6-stop-cross-C.npz").unlink()
        return
    if name == "v6_comparison_drop":
        path = run / "comparisons.json"
        rows = [row for row in _load(path) if not str(row.get("family", "")).startswith("V6-")]
        _unlink_write(path, json.dumps(rows))
        return
    case = "release_fast-s0-A" if name in {"v6_early_unload", "v6_load_direction", "v6_load_time"} else ("V6-stop-cross-B" if name in {"v6_delayed_cross", "v6_initial_state"} else zero)
    def edit(payload):
        if name == "v6_tau_limit":
            payload["tau_applied"][:, 6] = np.minimum(np.arange(len(payload["tau_applied"])) + 1, 50)
        elif name == "v6_gain":
            payload["K"][:] = 1e9
        elif name == "v6_tau_nan":
            payload["tau_before"][0] = np.nan
        elif name == "v6_flag_nan":
            payload["transition_active"] = np.full(len(payload["transition_active"]), np.nan)
        elif name == "v6_shape":
            payload["dq"] = payload["dq"][:-1]
        elif name == "v6_tick_dtype":
            payload["tick"] = payload["tick"].astype(float)
        elif name == "v6_normal_fault":
            payload["fault_tick"] = np.int64(100)
            payload["first_fault_tick"] = np.int64(100)
        elif name == "v6_history":
            payload["history_twist0"][:] = 0
            payload["candidate"][:] = 0
            payload["final_twist"][:] = 0
        elif name == "v6_early_unload":
            mask = (payload["t_state"][:-1] >= 0.05) & (payload["t_state"][:-1] < 0.149)
            payload["external_wrench"][mask, 0] = -0.001
        elif name == "v6_load_direction":
            payload["external_wrench"][60, 1] = 3.0
        elif name == "v6_load_time":
            payload["external_wrench"][42, 0] = 0.0
        elif name == "v6_initial_state":
            payload["qpos0"][0] += 0.01
        elif name == "v6_delayed_cross":
            n = len(payload["command"])
            shift = 200
            for key, arr in list(payload.items()):
                if key in {"tick", "t_state"} or key.startswith("contact_"):
                    continue
                if arr.ndim and arr.shape[0] in {n, n + 1}:
                    payload[key] = np.concatenate([np.repeat(arr[:1], shift, axis=0), arr[:-shift]], axis=0)
            payload["fault_tick"] += shift
            payload["first_fault_tick"] += shift
        else:
            raise KeyError(name)
    _edit_case(run, case, edit)


def _edit_case(run: Path, case_id: str, edit) -> None:
    path = run / "cases" / f"{case_id}.npz"
    data = np.load(path, allow_pickle=False)
    payload = {key: np.array(data[key]) for key in data.files}
    edit(payload)
    if path.exists():
        path.unlink()
    np.savez(path, **payload)
    _rehash(run, f"cases/{case_id}.npz")


def _tree_id(run: Path) -> dict:
    rows = {}
    for path in sorted(run.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows[str(path.relative_to(run))] = _sha256(path)
    return rows


def run_adversarial(run, fail_after: str | None = None) -> list[dict]:
    run = Path(run)
    source_id = _tree_id(run)
    results = []
    names = tuple(sorted(ADVERSARIAL))
    # One clean evaluation plus byte-identical copies is equivalent to repeatedly
    # solving the same clean evidence. Every modified copy is still evaluated afresh.
    try:
        with tempfile.TemporaryDirectory(prefix="m2-v4-adversarial-") as work:
            if fail_after is None:
                clean = evaluate_evidence(run, check_workspace=True)
                if clean["reasons"]:
                    return [{"name": name, "fixes_passed": False, "evidence_valid": clean["evidence_valid"], "codes": ["clean_copy", *[item["code"] for item in clean["reasons"]]]} for name in names]
            for name in names:
                with tempfile.TemporaryDirectory(prefix="case-", dir=work) as case_work:
                    dest = Path(case_work) / "run"
                    shutil.copytree(run, dest, copy_function=shutil.copy2)
                    if fail_after == name:
                        raise RuntimeError(name)
                    if _tree_id(dest) != source_id:
                        results.append({"name": name, "fixes_passed": False, "evidence_valid": False, "codes": ["clean_copy", "io_error"]})
                        continue
                    apply_tamper(name, dest)
                    verdict = evaluate_evidence(dest, check_workspace=True)
                    results.append({"name": name, "fixes_passed": verdict["fixes_passed"], "evidence_valid": verdict["evidence_valid"], "codes": sorted({item["code"] for item in verdict["reasons"]})})
    except OSError:
        return [{"name": "copy", "fixes_passed": False, "evidence_valid": False, "codes": ["io_error"]}]
    if _tree_id(run) != source_id:
        results.append({"name": "source_mutated", "fixes_passed": False, "evidence_valid": False, "codes": ["io_error"]})
    return results


def _slice_trace(run: Path, case_id: str, n_keep: int) -> None:
    path = run / "cases" / f"{case_id}.npz"
    data = np.load(path, allow_pickle=False)
    payload = {}
    for key in data.files:
        arr = np.array(data[key])
        if arr.ndim >= 1 and arr.shape[0] > n_keep + 1 and key not in {"tau_before"}:
            width = n_keep + 1 if arr.shape[0] == int(np.asarray(data["t_state"]).shape[0]) else n_keep
            payload[key] = arr[:width]
        else:
            payload[key] = arr
    if "contact_offsets" in payload:
        payload["contact_offsets"] = np.zeros(n_keep + 1, dtype=int)
        payload["contact_force"] = np.zeros((0, 3))
        payload["contact_dist"] = np.zeros(0)
        payload["contact_geom1"] = np.zeros(0, dtype="<U16")
        payload["contact_geom2"] = np.zeros(0, dtype="<U16")
    if path.exists():
        path.unlink()
    np.savez(path, **payload)


def _drop_index(run: Path, case_id: str, index: int) -> None:
    path = run / "cases" / f"{case_id}.npz"
    data = np.load(path, allow_pickle=False)
    n_state = int(np.asarray(data["t_state"]).shape[0])
    payload = {}
    for key in data.files:
        arr = np.array(data[key])
        if arr.ndim >= 1 and arr.shape[0] == n_state:
            payload[key] = np.delete(arr, index, axis=0)
        elif arr.ndim >= 1 and arr.shape[0] == n_state - 1 and key != "tau_before":
            payload[key] = np.delete(arr, min(index, arr.shape[0] - 1), axis=0)
        else:
            payload[key] = arr
    if path.exists():
        path.unlink()
    np.savez(path, **payload)


def check_second_stage(execution, reasons):
    from feedingrobot.controllers.acceptance import V6_SECOND_STAGE_NODEIDS

    reports = execution.get("reports") or []
    collected = execution.get("collected") or []
    if int(execution.get("exitstatus", 1)) != 0:
        _add(reasons, "exitstatus", field="adversarial_execution.json")
    for node in V6_SECOND_STAGE_NODEIDS:
        phases = {when: [item.get("outcome") for item in reports if item.get("nodeid") == node and item.get("when") == when] for when in ("setup", "call", "teardown")}
        if collected.count(node) != 1 or any(values != ["passed"] for values in phases.values()):
            _add(reasons, "nodeid", field=node)


def verify_finished_report(run, scope: str, check_workspace: bool, *, rerun_adversarial=True) -> dict:
    from feedingrobot.controllers.acceptance import REMAINING_FULL_M2, SCHEMA_VERSION, SCOPE_NAME

    if scope in {"fixes-v1", "fixes-v2", "fixes-v3", "fixes-v4", "fixes-v5"}:
        return {
            "fixes_passed": False,
            "tests_passed": False,
            "m3_ready": False,
            "evidence_valid": False,
            "scope_passed": False,
            "hybrid_force_status": "disabled",
            "reasons": [reason("unsupported_schema", text=f"{scope} is not a V6 certification")],
            "scope": scope,
            "remaining_m2_requirements": list(REMAINING_FULL_M2),
        }
    if scope not in {"fixes-v6", "full"}:
        return {
            "fixes_passed": False,
            "m3_ready": False,
            "evidence_valid": False,
            "scope_passed": False,
            "hybrid_force_status": "disabled",
            "reasons": [reason("unsupported_schema", text=f"unknown scope {scope}")],
            "scope": scope,
            "remaining_m2_requirements": list(REMAINING_FULL_M2),
        }
    run = Path(run)
    verdict = evaluate_evidence(run, check_workspace=check_workspace)
    reasons = list(verdict["reasons"])
    report = {}
    if (run / "report.json").is_file():
        try:
            report = _load(run / "report.json")
        except ValueError:
            _add(reasons, "unreadable", field="report.json")
    if report.get("release_scoring_version") != RELEASE_SCORING_VERSION:
        _add(reasons, "summary_mismatch", field="release_scoring_version")
    if report.get("schema_version") != SCHEMA_VERSION:
        _add(reasons, "unsupported_schema", field="schema_version", expected=None)
        reasons[-1]["text"] = "unsupported schema " + str(report.get("schema_version"))
    if report.get("scope") != SCOPE_NAME:
        _add(reasons, "scope_mismatch")
    if report.get("hybrid_force_status") != "disabled":
        _add(reasons, "threshold_mismatch", field="hybrid_force_status")
    if report.get("m3_ready") is True:
        _add(reasons, "summary_mismatch", field="m3_ready")
    core_fixes = not verdict["reasons"]
    if report.get("fixes_passed") is not core_fixes or report.get("evidence_valid") is not verdict["evidence_valid"]:
        _add(reasons, "summary_mismatch", field="report.json")
    if scope == "fixes-v6":
        exec_path = run / "adversarial_execution.json"
        if not exec_path.is_file():
            _add(reasons, "missing_file", field="adversarial_execution.json")
        else:
            execution = _load(exec_path)
            if int(execution.get("exitstatus", 1)) != 0:
                _add(reasons, "exitstatus", field="adversarial_execution.json")
            check_second_stage(execution, reasons)
    adv_path = run / "adversarial_results.json"
    if scope == "fixes-v6" and not reasons:
        fresh = run_adversarial(run) if rerun_adversarial else None
        if not adv_path.is_file():
            _add(reasons, "missing_file", field="adversarial_results.json")
        else:
            stored = _load(adv_path)
            if fresh is None:
                fresh = stored
            if not isinstance(stored, list) or any(not isinstance(row, dict) for row in stored) or [row.get('name') for row in stored] != sorted(ADVERSARIAL):
                _add(reasons, "summary_mismatch", field="adversarial_results.json")
                fresh = []
            elif not _same(stored, fresh):
                _add(reasons, "summary_mismatch", field="adversarial_results.json")
            for row in fresh:
                if row["fixes_passed"] or not row["codes"] or "clean_copy" in row["codes"] or not ADVERSARIAL_EXPECTED.get(row["name"], set()).issubset(row["codes"]):
                    _add(reasons, "summary_mismatch", field=row["name"])
    if scope == "full":
        _add(reasons, "full_incomplete", text="full M2 still missing: " + "; ".join(REMAINING_FULL_M2))
        reasons[-1]["text"] = "full M2 still missing: " + "; ".join(REMAINING_FULL_M2)
    structural = [item for item in reasons if item.get("code") in STRUCTURAL]
    metric = [item for item in reasons if item.get("code") not in STRUCTURAL and item.get("code") != "full_incomplete"]
    fixes_passed = not structural and not metric
    scope_passed = fixes_passed if scope == "fixes-v6" else False
    return {
        "fixes_passed": fixes_passed,
        "tests_passed": bool(verdict["tests_passed"]) and not structural,
        "m3_ready": False,
        "evidence_valid": not structural,
        "scope_passed": scope_passed,
        "hybrid_force_status": "disabled",
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "reasons": reasons,
        "scope": scope,
        "comparisons": verdict.get("comparisons", []),
    }
