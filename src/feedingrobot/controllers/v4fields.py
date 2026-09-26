"""Field shape, dtype, and semantic checks. Thresholds are not read from the trace."""

from __future__ import annotations

import numpy as np

from feedingrobot.controllers.guard import GEAR_OF, PHASES

BLEND_S = 0.2
BLEND_TIME_TOL_S = 1e-12

GEAR_K = {
    "FREE": np.array([300.0, 300.0, 300.0, 8.0, 8.0, 8.0]),
    "ACQUIRE": np.array([150.0, 150.0, 150.0, 5.0, 5.0, 5.0]),
    "MOUTH": np.array([100.0, 100.0, 100.0, 3.0, 3.0, 3.0]),
    "STOP": np.zeros(6),
}
GEAR_D = {
    "FREE": np.array([49.0, 49.0, 49.0, 0.8, 0.8, 0.8]),
    "ACQUIRE": np.array([35.0, 35.0, 35.0, 0.63, 0.63, 0.63]),
    "MOUTH": np.array([28.0, 28.0, 28.0, 0.49, 0.49, 0.49]),
    "STOP": np.zeros(6),
}
GEAR_VW = {"FREE": (0.05, 0.3), "ACQUIRE": (0.02, 0.15), "MOUTH": (0.01, 0.1), "STOP": (0.0, 0.0)}
MODES = {"RUNNING", "POWER_ON", "STOPPING", "ABORTED"}
EXECUTIONS = {"run", "zero", "stop", "prohibit", "power_on"}
GEARS = set(GEAR_K)
_MODEL = {}


def _reason(code, case_id, field, observed=None, expected=None, index=None):
    row = {"code": code, "text": code, "case_id": case_id, "field": field}
    if index is not None:
        row["index"] = int(index)
    if observed is not None and np.size(observed) == 1:
        value = float(np.asarray(observed).reshape(-1)[0])
        row["observed"] = None if not np.isfinite(value) else value
    if expected is not None and np.size(expected) == 1:
        value = float(np.asarray(expected).reshape(-1)[0])
        row["expected"] = None if not np.isfinite(value) else value
    return row


def model_layout(family: str):
    key = "gear" if family == "gear" else "base"
    if key not in _MODEL:
        import json

        from feedingrobot.sim.scene import FeedingScene

        with open("configs/m1_scene.json") as handle:
            cfg = json.load(handle)
        if key == "gear":
            cfg["model"] = "assets/tests/m2_spring_surface.xml"
        scene = FeedingScene(cfg)
        joints = scene.index.arm_joint_ids
        _MODEL[key] = {
            "nq": int(scene.model.nq),
            "nv": int(scene.model.nv),
            "range": np.array(scene.model.jnt_range[joints], dtype=float),
        }
    return _MODEL[key]


def _shape_of(spec, n, c, nq, nv):
    out = []
    for item in spec:
        out.append({"N1": n + 1, "N": n, "C": c, "nq": nq, "nv": nv, "7": 7, "3": 3, "6": 6, "9": 9}[item])
    return tuple(out)


def _float_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def _int_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.bool_)


def _flag_dtype(arr) -> bool:
    if arr.dtype == np.bool_:
        return True
    if _int_dtype(arr.dtype):
        values = np.unique(arr)
        return values.size == 0 or set(values.tolist()) <= {0, 1}
    return False


def _text_dtype(dtype) -> bool:
    return dtype.kind in {"U", "S"}


def screen_arrays(data, case_id: str, n: int, family: str, dt: float, extra_specs=None) -> list[dict]:
    """Reject bad dtype or shape before any cast. Returns reasons; empty means the arrays may be used."""
    layout = model_layout(family)
    reasons = []
    names = (
        "t_state",
        "tick",
        "q",
        "dq",
        "tcp_pos",
        "tcp_twist",
        "tcp_rot",
        "tau_applied",
        "tau_raw",
        "tau_before",
        "qpos0",
        "qvel0",
        "command",
        "v_ref",
        "K",
        "D",
        "p_ref",
        "p_ref_before",
        "r_ref",
        "r_ref_before",
        "reference_correction_pos",
        "wrench_raw",
        "wrench_compensated",
        "external_wrench",
        "force_used",
        "phase",
        "mode",
        "execution",
        "active_gear",
        "target_gear",
        "guard_status",
        "pause_applied",
        "saturation_flags",
        "blocked_now",
        "transition_active",
        "warnings",
        "first_fault_tick",
        "contact_offsets",
        "contact_force",
        "contact_torque",
        "contact_point",
        "contact_dist",
        "contact_geom1",
        "contact_geom2",
        "contact_frame",
    )
    for name in names:
        if name not in data:
            reasons.append(_reason("missing_field", case_id, name))
    if "t_state" in data:
        times = data["t_state"]
        if times.ndim == 1 and _float_dtype(times.dtype) and times.size >= 2 and np.any(np.abs(np.diff(np.array(times, dtype=float)) - dt) > 1e-9):
            reasons.append(_reason("bad_time", case_id, "dt"))
    if reasons:
        return reasons
    force = data["contact_force"]
    if force.ndim != 2 or force.shape[1:] != (3,) or not _float_dtype(force.dtype):
        reasons.append(_reason("bad_shape" if force.ndim != 2 or force.shape[1:] != (3,) else "bad_dtype", case_id, "contact_force"))
        return reasons
    count = int(force.shape[0])
    specs = {
        "t_state": (("N1",), "float"),
        "tick": (("N1",), "int"),
        "q": (("N1", "7"), "float"),
        "dq": (("N1", "7"), "float"),
        "tcp_pos": (("N1", "3"), "float"),
        "tcp_twist": (("N1", "6"), "float"),
        "tcp_rot": (("N1", "3", "3"), "float"),
        "tau_applied": (("N", "7"), "float"),
        "tau_raw": (("N", "7"), "float"),
        "tau_before": (("7",), "float"),
        "qpos0": (("nq",), "float"),
        "qvel0": (("nv",), "float"),
        "command": (("N", "6"), "float"),
        "v_ref": (("N", "6"), "float"),
        "K": (("N", "6"), "float"),
        "D": (("N", "6"), "float"),
        "p_ref": (("N", "3"), "float"),
        "p_ref_before": (("N", "3"), "float"),
        "r_ref": (("N", "3", "3"), "float"),
        "r_ref_before": (("N", "3", "3"), "float"),
        "reference_correction_pos": (("N", "3"), "float"),
        "wrench_raw": (("N", "6"), "float"),
        "wrench_compensated": (("N", "6"), "float"),
        "external_wrench": (("N", "6"), "float"),
        "force_used": (("N", "3"), "float"),
        "phase": (("N",), "text"),
        "mode": (("N",), "text"),
        "execution": (("N",), "text"),
        "active_gear": (("N",), "text"),
        "target_gear": (("N",), "text"),
        "guard_status": (("N",), "text"),
        "pause_applied": (("N",), "flag"),
        "saturation_flags": (("N",), "flag"),
        "blocked_now": (("N",), "flag"),
        "transition_active": (("N",), "flag"),
        "warnings": (("N",), "int"),
        "first_fault_tick": ((), "int"),
        "contact_offsets": (("N1",), "int"),
        "contact_force": (("C", "3"), "float"),
        "contact_torque": (("C", "3"), "float"),
        "contact_point": (("C", "3"), "float"),
        "contact_dist": (("C",), "float"),
        "contact_geom1": (("C",), "text"),
        "contact_geom2": (("C",), "text"),
        "contact_frame": (("C", "9"), "float"),
    }
    specs.update(extra_specs or {})
    for name, (shape, kind) in specs.items():
        if name not in data:
            reasons.append(_reason("missing_field", case_id, name))
            continue
        arr = np.asarray(data[name])
        expected = _shape_of(shape, n, count, layout["nq"], layout["nv"])
        if arr.shape != expected:
            code = "sample_count" if name == "t_state" and arr.ndim == 1 else "bad_shape"
            reasons.append(_reason(code, case_id, name, arr.shape[0] if arr.ndim else 0, expected[0] if expected else 0))
            continue
        ok = {"float": _float_dtype(arr.dtype), "int": _int_dtype(arr.dtype), "text": _text_dtype(arr.dtype), "flag": _flag_dtype(arr)}[kind]
        if not ok:
            reasons.append(_reason("bad_dtype", case_id, name))
            continue
        if kind == "float" and arr.size and not np.all(np.isfinite(arr)):
            flat = np.argwhere(~np.isfinite(arr))
            reasons.append(_reason("nonfinite", case_id, name, index=int(flat[0][0]) if flat.size else 0))
    if reasons:
        return reasons
    offsets = np.array(data["contact_offsets"])
    if int(offsets[0]) != 0 or int(offsets[-1]) != count or np.any(np.diff(offsets) < 0) or np.any(offsets < 0) or np.any(offsets > count):
        reasons.append(_reason("bad_offsets", case_id, "contact_offsets"))
    for name in ("tcp_rot", "r_ref", "r_ref_before"):
        mats = np.array(data[name], dtype=float)
        if not _rotations_ok(mats):
            reasons.append(_reason("invalid_rotation", case_id, name))
    return reasons


def _rotations_ok(mats) -> bool:
    if mats.ndim == 2:
        mats = mats.reshape(1, 3, 3)
    if mats.size == 0:
        return True
    gram = np.swapaxes(mats, -1, -2) @ mats
    eye = np.eye(3)
    err = np.max(np.abs(gram - eye), axis=(-2, -1))
    det = np.linalg.det(mats)
    return bool(np.all(err <= 1e-6) and np.all(np.abs(det - 1.0) <= 1e-6))


def hard_arrays(data, case_id: str, family: str, dt: float) -> list[dict]:
    """Unconditional physical constraints; expected stops receive no exemption."""
    from feedingrobot.controllers.guard import contact_allowed
    from feedingrobot.controllers.v3spec import BOWL, GEAR_POS, SPEED_LIMIT, TAU_MAX, TAU_RATE
    from feedingrobot.controllers.so3 import orientation_error

    data = {key: np.asarray(data[key]) for key in ("command", "tick", "t_state", "q", "dq", "tau_applied", "tau_before", "phase", "mode", "guard_status", "execution", "active_gear", "target_gear", "warnings", "contact_offsets", "contact_dist", "contact_geom1", "contact_geom2", "wrench_compensated", "p_ref", "r_ref", "tcp_pos", "tcp_rot")}
    reasons = []
    n = len(data["command"])
    if not np.array_equal(data["tick"], np.arange(n + 1)) or not np.allclose(data["t_state"], np.arange(n + 1) * dt, atol=1e-9, rtol=0):
        reasons.append(_reason("bad_time", case_id, "tick"))
    q = data["q"]
    ranges = model_layout(family)["range"]
    if np.any(q < ranges[:, 0] - 1e-6) or np.any(q > ranges[:, 1] + 1e-6):
        reasons.append(_reason("joint_range", case_id, "q"))
    if np.max(np.abs(data["dq"])) > SPEED_LIMIT + 1e-9:
        reasons.append(_reason("joint_speed", case_id, "dq"))
    tau = data["tau_applied"]
    if np.any(np.abs(tau) > TAU_MAX + 1e-9):
        reasons.append(_reason("torque_rate", case_id, "tau_max"))
    if np.max(np.abs(np.diff(np.vstack([data["tau_before"], tau]), axis=0))) > TAU_RATE * dt + 1e-9:
        reasons.append(_reason("torque_rate", case_id, "tau_applied"))
    enums = {"phase": set(PHASES), "mode": MODES | {"STOPPED"}, "guard_status": {"RUNNING", "STOPPING", "STOPPED", "ABORTED"}, "execution": EXECUTIONS, "active_gear": GEARS, "target_gear": GEARS}
    for name, allowed in enums.items():
        if any(str(value) not in allowed for value in data[name]):
            reasons.append(_reason("invalid_enum", case_id, name))
    if np.any(data["warnings"] != 0):
        reasons.append(_reason("unexpected_fault", case_id, "warnings"))
    offsets = data["contact_offsets"]
    limits = {"FREE": (8.0, 0.8), "ACQUIRE": (4.0, 0.4), "MOUTH": (2.0, 0.2), "STOP": (8.0, 0.8)}
    for k in range(n):
        phase = str(data["phase"][k])
        if phase not in GEAR_OF:
            continue
        gear = GEAR_OF[phase]
        if np.linalg.norm(data["p_ref"][k] - data["tcp_pos"][k]) > GEAR_POS[gear] + 1e-9:
            reasons.append(_reason("reference_error", case_id, "position", index=k))
            break
        rot_limit = {"FREE": 0.13962634, "ACQUIRE": 0.08726646, "MOUTH": 0.05235988, "STOP": 0.13962634}[gear]
        if np.linalg.norm(orientation_error(data["r_ref"][k], data["tcp_rot"][k])) > rot_limit + 1e-9:
            reasons.append(_reason("reference_error", case_id, "orientation", index=k))
            break
        force_lim, torque_lim = limits[gear]
        wrench = data["wrench_compensated"][k]
        if np.linalg.norm(wrench[:3]) > force_lim + 1e-9 or np.linalg.norm(wrench[3:]) > torque_lim + 1e-9:
            reasons.append(_reason("wrench_limit", case_id, "wrench_compensated", index=k))
            break
        for j in range(int(offsets[k]), int(offsets[k + 1])):
            if data["contact_dist"][j] < -0.001:
                reasons.append(_reason("reference_error", case_id, "contact_dist", index=k))
                break
            if not contact_allowed(phase, str(data["contact_geom1"][j]), str(data["contact_geom2"][j]), {"press_test": family == "gear", "bowl_geoms": tuple(BOWL)}):
                reasons.append(_reason("event_missing", case_id, "contact_whitelist", index=k))
                break
    return reasons


def semantic_arrays(data, case_id: str, family: str, dt: float) -> list[dict]:
    reasons = []
    layout = model_layout(family)
    q = np.array(data["q"], dtype=float)
    low, high = layout["range"][:, 0], layout["range"][:, 1]
    if np.any(q < low - 1e-6) or np.any(q > high + 1e-6):
        bad = np.argwhere((q < low - 1e-6) | (q > high + 1e-6))[0]
        joint = int(bad[1])
        reasons.append(_reason("joint_range", case_id, f"q[{joint}]", q[tuple(bad)], high[joint] if q[tuple(bad)] > 0 else low[joint], index=int(bad[0])))
    K = np.array(data["K"], dtype=float)
    D = np.array(data["D"], dtype=float)
    if np.any(K < -1e-9) or np.any(D < -1e-9):
        reasons.append(_reason("gain_mismatch", case_id, "K"))
    trans = np.array(data["transition_active"]).astype(int)
    active = np.array(data["active_gear"]).astype(str)
    target = np.array(data["target_gear"]).astype(str)
    phase = np.array(data["phase"]).astype(str)
    mode = np.array(data["mode"]).astype(str)
    execution = np.array(data["execution"]).astype(str)
    for name, values, allowed in (("phase", phase, set(PHASES)), ("active_gear", active, GEARS), ("target_gear", target, GEARS), ("execution", execution, EXECUTIONS)):
        unknown = [item for item in values if item not in allowed]
        if unknown:
            reasons.append(_reason("invalid_enum", case_id, name))
            break
    unknown_mode = [item for item in mode if item not in MODES]
    if unknown_mode:
        reasons.append(_reason("invalid_enum", case_id, "mode"))
    elif family == "takeover":
        if str(mode[0]) != "POWER_ON" or not np.any(mode == "RUNNING") or np.any(np.isin(mode, ["ABORTED", "STOPPING"])):
            reasons.append(_reason("unexpected_mode", case_id, "mode"))
    elif np.any(mode != "RUNNING"):
        reasons.append(_reason("unexpected_mode" if np.any(np.isin(mode, ["ABORTED", "STOPPING"])) else "invalid_enum", case_id, "mode"))
    guard = np.array(data["guard_status"]).astype(str)
    if np.any(guard != "RUNNING"):
        reasons.append(_reason("unexpected_fault", case_id, "guard_status"))
    fault = int(np.array(data["first_fault_tick"]).reshape(-1)[0])
    warnings = np.array(data["warnings"])
    if fault != -1 or np.any(warnings != 0):
        reasons.append(_reason("unexpected_fault", case_id, "first_fault_tick", fault, -1))
    built = check_logged_gains(K, D, trans, active, target, phase, dt, case_id, reasons, require_acquire=family == "gear")
    v_ref = np.array(data["v_ref"], dtype=float)
    pause = np.array(data["pause_applied"]).astype(int) == 1
    held = pause | np.isin(execution, ["power_on", "stop", "prohibit"])
    if np.any(held) and np.any(np.linalg.norm(v_ref[held], axis=1) > 1e-9):
        reasons.append(_reason("reference_speed", case_id, "v_ref", float(np.max(np.linalg.norm(v_ref[held], axis=1))), 0.0))
    task = np.array(data["p_ref"], dtype=float) - np.array(data["p_ref_before"], dtype=float) - np.array(data["reference_correction_pos"], dtype=float)
    if np.any(held) and np.any(np.linalg.norm(task[held], axis=1) > 1e-9):
        reasons.append(_reason("pause_motion", case_id, "task_integral", float(np.max(np.linalg.norm(task[held], axis=1))), 0.0))
    if built is not None:
        for k in range(len(v_ref)):
            if float(np.linalg.norm(v_ref[k, :3])) > built["v_lim"][k] + 1e-9 or float(np.linalg.norm(v_ref[k, 3:])) > built["w_lim"][k] + 1e-9:
                reasons.append(_reason("reference_speed", case_id, "v_ref", float(np.linalg.norm(v_ref[k, :3])), built["v_lim"][k], index=k))
                break
    return reasons


def rebuild_gain_path(phase, dt: float, initial_phase: str = "TRANSPORT") -> dict:
    """Expected gains from the case's initial phase and the logged phase. Logged gears are not inputs."""
    phase = np.asarray(phase).astype(str)
    n = int(phase.shape[0])
    init = GEAR_OF[str(initial_phase)]
    expected_k = GEAR_K[init].copy()
    expected_d = GEAR_D[init].copy()
    completed = init
    desired_prev = init
    in_transition = False
    start_k = expected_k.copy()
    start_d = expected_d.copy()
    elapsed = 0.0
    out_k = np.zeros((n, 6))
    out_d = np.zeros((n, 6))
    out_active = np.empty(n, dtype=object)
    out_target = np.empty(n, dtype=object)
    out_trans = np.zeros(n, dtype=int)
    v_lim = np.zeros(n)
    w_lim = np.zeros(n)
    acquire_done = None
    for k in range(n):
        name = str(phase[k])
        if name not in GEAR_OF:
            return {"error": name, "index": k}
        desired = GEAR_OF[name]
        if desired == "STOP":
            expected_k = np.zeros(6)
            expected_d = np.zeros(6)
            completed = "STOP"
            desired_prev = "STOP"
            in_transition = False
            elapsed = BLEND_S
            out_k[k] = expected_k
            out_d[k] = expected_d
            out_active[k] = completed
            out_target[k] = desired
            out_trans[k] = 0
            v_lim[k], w_lim[k] = GEAR_VW["STOP"]
            continue
        if desired != desired_prev:
            start_k = expected_k.copy()
            start_d = expected_d.copy()
            elapsed = 0.0
            in_transition = True
        if in_transition:
            elapsed = min(BLEND_S, elapsed + float(dt))
            if elapsed >= BLEND_S - BLEND_TIME_TOL_S:
                expected_k = GEAR_K[desired].copy()
                expected_d = GEAR_D[desired].copy()
                completed = desired
                in_transition = False
                if desired == "ACQUIRE" and acquire_done is None:
                    acquire_done = k
            else:
                alpha = elapsed / BLEND_S
                expected_k = (1.0 - alpha) * start_k + alpha * GEAR_K[desired]
                expected_d = (1.0 - alpha) * start_d + alpha * GEAR_D[desired]
        else:
            expected_k = GEAR_K[desired].copy()
            expected_d = GEAR_D[desired].copy()
            completed = desired
        out_k[k] = expected_k
        out_d[k] = expected_d
        out_active[k] = completed
        out_target[k] = desired
        out_trans[k] = 1 if in_transition else 0
        left = GEAR_VW[completed]
        right = GEAR_VW[desired]
        v_lim[k] = min(left[0], right[0]) if in_transition else left[0]
        w_lim[k] = min(left[1], right[1]) if in_transition else left[1]
        desired_prev = desired
    return {
        "K": out_k,
        "D": out_d,
        "active": out_active,
        "target": out_target,
        "trans": out_trans,
        "v_lim": v_lim,
        "w_lim": w_lim,
        "acquire_done": acquire_done,
    }


def check_logged_gains(K, D, trans, active, target, phase, dt, case_id, reasons, initial_phase="TRANSPORT", require_acquire=False):
    built = rebuild_gain_path(phase, dt, initial_phase)
    if built.get("error"):
        reasons.append(_reason("invalid_enum", case_id, "phase", index=built["index"]))
        return None
    n = len(K)
    if n == 0:
        reasons.append(_reason("bad_shape", case_id, "K", 0, 1))
        return None
    for k in range(n):
        if str(target[k]) != built["target"][k]:
            reasons.append(_reason("phase_gear_mismatch", case_id, "target_gear", index=k))
            break
        if not np.allclose(K[k], built["K"][k], atol=1e-9, rtol=1e-7) or not np.allclose(D[k], built["D"][k], atol=1e-9, rtol=1e-7):
            reasons.append(_reason("gain_path_mismatch", case_id, "K", K[k, 0], built["K"][k, 0], index=k))
            break
        logged_trans = int(trans[k])
        if str(active[k]) != built["active"][k] or logged_trans != int(built["trans"][k]):
            reasons.append(_reason("transition_state_mismatch", case_id, "transition_active", logged_trans, int(built["trans"][k]), index=k))
            break
    done = built["acquire_done"]
    if require_acquire and (done is None or done >= n - 1):
        reasons.append(_reason("transition_incomplete", case_id, "K", None if done is None else done, n - 1))
    return built
