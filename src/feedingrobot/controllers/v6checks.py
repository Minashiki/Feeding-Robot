"""V6 zero-command and singularity-stop evidence checks.

Thresholds live here. A trace cannot relax them, and a logged beta or stopped_ok flag is not proof.
"""

from __future__ import annotations

import numpy as np

from feedingrobot.controllers.cartesian_impedance import clip_torque, damped_nullspace, singularity_gains
LENGTH = 0.1
SPEED_PREDICT = 0.4
MARGIN_SLOW = 0.10
MARGIN_STOP = 0.05
RHO_STOP = 0.01
RHO_NORMAL = 0.05
EPS_NORMAL = 1e-8
EPS_SINGULAR = 1e-2
A_FLOOR = 1e-4
STOP_DAMPING = 1.0
TAU_RATE = 2000.0
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
SPEED_LIMIT = 0.5
STOP_SPEED = 0.02
STOP_WITHIN_S = 0.5
STOP_HOLD_S = 0.2
TRACE_AFTER_S = 0.7

SINGULAR_Q = np.array(
    [
        -1.8729622292175738,
        1.2953719117084952,
        0.2528651848336265,
        -0.3794389338812425,
        -0.13147233171490358,
        1.608251115930711,
        1.6627727807805979,
    ]
)
REST_DQ = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
CROSS_Q = np.array(
    [
        -1.4863286076289621,
        1.0898983356544798,
        0.2006664908819923,
        -0.6107551753121649,
        -0.10433263666036569,
        1.689119425733349,
        1.4433857142668987,
    ]
)
CROSS_DQ = np.array(
    [
        -0.05615934743398632,
        0.02984546946204058,
        0.007581970179378971,
        0.033599171009150726,
        -0.003942097838149128,
        -0.01174629223383445,
        0.031866433240447405,
    ]
)
ZERO_DURATION_S = 0.3
STOP_DURATION_S = 1.0

class CheckContext:
    """Owned by one evaluation; never shared between concurrent calls."""

    def __init__(self, controller_version=None):
        import mujoco
        from feedingrobot.sim.scene import FeedingScene

        self.scene = FeedingScene("configs/m1_scene.json")
        self.mujoco = mujoco
        self.controller_version = controller_version

    @property
    def ranges(self):
        return np.array(self.scene.model.jnt_range[self.scene.index.arm_joint_ids], dtype=float)



def _reason(code, case_id, field, observed=None, expected=None, text=""):
    row = {"code": code, "text": text or code, "case_id": case_id, "field": field}
    if observed is not None and np.size(observed) == 1:
        value = float(np.asarray(observed).reshape(-1)[0])
        row["observed"] = None if not np.isfinite(value) else value
    if expected is not None and np.size(expected) == 1:
        value = float(np.asarray(expected).reshape(-1)[0])
        row["expected"] = None if not np.isfinite(value) else value
    return row


def joint_beta(jbar, twist, q, ranges) -> float:
    """Uniform scale. This is the checker's copy of the boundary rule, not a call into compute()."""
    scale = np.diag([1.0, 1.0, 1.0, LENGTH, LENGTH, LENGTH])
    pred = np.asarray(jbar, dtype=float) @ (scale @ np.asarray(twist, dtype=float).reshape(6))
    beta = 1.0
    q = np.asarray(q, dtype=float).reshape(7)
    for i in range(7):
        vel = float(pred[i])
        if abs(vel) <= 1e-12:
            continue
        dist = float(ranges[i, 1] - q[i]) if vel > 0.0 else float(q[i] - ranges[i, 0])
        if dist >= MARGIN_SLOW:
            allow = SPEED_PREDICT
        elif dist <= MARGIN_STOP:
            allow = 0.0
        else:
            allow = SPEED_PREDICT * (dist - MARGIN_STOP) / (MARGIN_SLOW - MARGIN_STOP)
        beta = min(beta, allow / abs(vel))
    return float(np.clip(beta, 0.0, 1.0))


def _arm_state(q, dq, context):
    scene, mujoco = context.scene, context.mujoco
    scene.data.qpos[scene.index.arm_qpos_adr] = np.asarray(q, dtype=float).reshape(7)
    scene.data.qvel[scene.index.arm_dof_adr] = np.asarray(dq, dtype=float).reshape(7)
    mujoco.mj_forward(scene.model, scene.data)
    jacp = np.zeros((3, scene.model.nv))
    jacr = np.zeros((3, scene.model.nv))
    mujoco.mj_jacSite(scene.model, scene.data, jacp, jacr, scene.index.site_ids["tcp"])
    cols = scene.index.arm_dof_adr
    jacobian = np.vstack([jacp[:, cols], jacr[:, cols]])
    full = np.empty((scene.model.nv, scene.model.nv))
    mujoco.mj_fullM(scene.model, scene.data, full)
    mass = full[np.ix_(cols, cols)].copy()
    bias = np.array(scene.data.qfrc_bias[cols], dtype=float).copy()
    return scene, jacobian, mass, bias


def _rho(jacobian) -> float:
    scale = np.diag([1.0, 1.0, 1.0, LENGTH, LENGTH, LENGTH])
    singular = np.linalg.svd(scale @ jacobian, compute_uv=False)
    if singular[0] <= 0.0 or not np.isfinite(singular[0]):
        return 0.0
    return float(singular[-1] / singular[0])


def check_zero_grid(case_id: str, row: dict, context=None) -> list[dict]:
    reasons = []
    context = context or CheckContext()
    need = ("q", "jbar", "candidate", "final", "p_before", "p_after", "correction", "dt")
    if any(key not in row for key in need):
        return [_reason("missing_field", case_id, "grid")]
    shapes = {"q": (7,), "jbar": (7, 6), "candidate": (6,), "final": (6,), "p_before": (3,), "p_after": (3,), "correction": (3,), "dt": ()}
    for field, shape in shapes.items():
        arr = np.asarray(row[field])
        if arr.shape != shape:
            return [_reason("bad_shape", case_id, field)]
        if arr.dtype.kind not in "fi":
            return [_reason("bad_dtype", case_id, field)]
        if not np.all(np.isfinite(arr)):
            return [_reason("nonfinite", case_id, field)]
    q = np.asarray(row["q"], dtype=float).reshape(7)
    jbar = np.asarray(row["jbar"], dtype=float).reshape(7, 6)
    candidate = np.asarray(row["candidate"], dtype=float).reshape(6)
    final = np.asarray(row["final"], dtype=float).reshape(6)
    correction = np.asarray(row["correction"], dtype=float).reshape(3)
    before = np.asarray(row["p_before"], dtype=float).reshape(3)
    after = np.asarray(row["p_after"], dtype=float).reshape(3)
    dt = float(row["dt"])
    if not all(np.all(np.isfinite(item)) for item in (q, jbar, candidate, final, correction, before, after)):
        return [_reason("nonfinite", case_id, "grid")]
    parts = case_id.split("-")
    from feedingrobot.controllers.acceptance import v6_zero_grid_ids
    if case_id not in v6_zero_grid_ids():
        return [_reason("stimulus", case_id, "case_id")]
    joint, side, motion, variant = int(parts[2][1:]), parts[3], parts[4], parts[5]
    expected_q = np.array(context.scene.config["q_torque_poses"][0], dtype=float)
    expected_q[joint] = context.ranges[joint, 1] - 0.04 if side == "upper" else context.ranges[joint, 0] + 0.04
    expected_jbar = np.zeros((7, 6))
    expected_candidate = np.zeros(6)
    sign = 1.0 if side == "upper" else -1.0
    if motion in {"linear", "mixed"}:
        expected_jbar[joint, 0] = sign
        expected_candidate[0] = 0.05 if motion == "linear" else 0.02
    if motion in {"angular", "mixed"}:
        expected_jbar[joint, 3] = sign
        expected_candidate[3] = 0.5 if motion == "angular" else 0.2
    expected_dt = 0.001 if variant == "A" else 0.0005
    for field, actual, expected_input in (("q", q, expected_q), ("jbar", jbar, expected_jbar), ("candidate", candidate, expected_candidate), ("dt", dt, expected_dt), ("p_before", before, np.zeros(3)), ("correction", correction, np.zeros(3))):
        if not np.allclose(actual, expected_input, atol=1e-12, rtol=0):
            reasons.append(_reason("stimulus", case_id, field))
    beta = joint_beta(jbar, candidate, q, context.ranges)
    expected = np.zeros(6) if beta <= 1e-15 else candidate * beta
    if not np.allclose(final, expected, atol=1e-9, rtol=0.0):
        reasons.append(_reason("joint_projection", case_id, "final", float(np.linalg.norm(final)), float(np.linalg.norm(expected))))
    step = after - before - correction
    if not np.allclose(step, expected[:3] * dt, atol=1e-9, rtol=0.0):
        reasons.append(_reason("joint_projection", case_id, "integral", float(np.linalg.norm(step)), float(np.linalg.norm(expected[:3] * dt))))
    scale = np.diag([1.0, 1.0, 1.0, LENGTH, LENGTH, LENGTH])
    pred = jbar @ (scale @ expected)
    joint = int(case_id.split("-")[2][1:])
    outward = pred[joint]
    if "lower" in case_id:
        outward = -outward
    if outward > 1e-9:
        reasons.append(_reason("joint_projection", case_id, "qdot_pred", float(outward), 0.0))
    return reasons


def _variant(case_id: str) -> str:
    return case_id.rsplit("-", 1)[-1]


def _as_str(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def check_trace(case_id: str, data, meta: dict, context=None, metrics=None) -> list[dict]:
    from feedingrobot.controllers.v3spec import VARIANTS
    from feedingrobot.controllers.v4fields import screen_arrays, hard_arrays, semantic_arrays

    context = context or CheckContext()
    reasons = []
    variant = _variant(case_id)
    if variant not in VARIANTS:
        return [_reason("solver_mismatch", case_id, "variant")]
    spec = VARIANTS[variant]
    dt = float(spec["dt"])
    if abs(float(meta.get("dt", -1.0)) - dt) > 1e-15:
        reasons.append(_reason("solver_mismatch", case_id, "dt", meta.get("dt"), dt))
    if int(meta.get("iterations", -1)) != int(spec["iterations"]) or abs(float(meta.get("tolerance", -1.0)) - float(spec["tolerance"])) > 1e-18:
        reasons.append(_reason("solver_mismatch", case_id, "iterations"))
    duration = STOP_DURATION_S if case_id.startswith("V6-stop-") else ZERO_DURATION_S
    n = int(round(duration / dt))
    extras = {name: (("N", "7"), "float") for name in ("tau_task", "tau_null", "tau_bias")}
    extras.update({name: (("N", "6"), "float") for name in ("candidate", "final_twist")})
    extras.update({"beta": (("N",), "float"), "history_twist0": (("6",), "float"), "fault_tick": ((), "int"), "fault_reason": ((), "text"), "stopped_ok": ((), "flag")})
    found = screen_arrays(data, case_id, n, "base", dt, extra_specs=extras)
    if found:
        return reasons + found
    data = {key: np.asarray(data[key]) for key in data}
    reasons.extend(hard_arrays(data, case_id, "base", dt))
    if int(data["fault_tick"]) != int(data["first_fault_tick"]):
        reasons.append(_reason("fault_tick", case_id, "first_fault_tick"))
    scene = context.scene
    # Rebuild the frozen complete initial snapshot, not just the seven arm coordinates.
    scene.reset(seed=0, preset="food_on_plate", settle_steps=0)
    name = case_id.split("-")[-2]
    initial_q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    initial_dq = np.zeros(7)
    if name in {"upper", "lower"}:
        initial_q[0] = context.ranges[0, 1] - 0.04 if name == "upper" else context.ranges[0, 0] + 0.04
    else:
        initial_q = SINGULAR_Q if name == "rest" else CROSS_Q
        initial_dq = REST_DQ if name == "rest" else CROSS_DQ
    scene.data.qpos[scene.index.arm_qpos_adr] = initial_q
    scene.data.qvel[scene.index.arm_dof_adr] = initial_dq
    for field, expected_initial in (("qpos0", scene.data.qpos), ("qvel0", scene.data.qvel)):
        recorded_meta = np.asarray(meta.get(field, []), dtype=float)
        if recorded_meta.shape != expected_initial.shape or not np.allclose(recorded_meta, expected_initial, atol=1e-8, rtol=0) or not np.allclose(data[field], expected_initial, atol=1e-8, rtol=0):
            reasons.append(_reason("initial_state", case_id, field))
    if not np.allclose(data["q"][0], initial_q, atol=1e-8, rtol=0) or not np.allclose(data["dq"][0], initial_dq, atol=1e-8, rtol=0):
        reasons.append(_reason("initial_state", case_id, "first_frame"))
    context.mujoco.mj_forward(scene.model, scene.data)
    initial = scene.snapshot()
    for field, actual, expected_initial in (("tcp_pos", data["tcp_pos"][0], initial["tcp_pos"]), ("tcp_rot", data["tcp_rot"][0], np.asarray(initial["tcp_mat"]).reshape(3, 3)), ("p_ref_before", data["p_ref_before"][0], initial["tcp_pos"]), ("r_ref_before", data["r_ref_before"][0], np.asarray(initial["tcp_mat"]).reshape(3, 3))):
        if not np.allclose(actual, expected_initial, atol=1e-8, rtol=0):
            reasons.append(_reason("initial_state", case_id, field))
    if not np.allclose(data["tau_before"], scene.data.qfrc_bias[scene.index.arm_dof_adr], atol=1e-7, rtol=0):
        reasons.append(_reason("initial_state", case_id, "tau_before"))
    if np.any(np.abs(data["external_wrench"]) > 1e-9) or np.any(data["phase"] != "TRANSPORT"):
        reasons.append(_reason("stimulus", case_id, "external_or_phase"))
    try:
        t_state = np.asarray(data["t_state"], dtype=float)
        q = np.asarray(data["q"], dtype=float)
        dq = np.asarray(data["dq"], dtype=float)
        command = np.asarray(data["command"], dtype=float)
        execution = _as_str(data["execution"])
        mode = _as_str(data["mode"])
        guard_status = _as_str(data["guard_status"])
        phase = _as_str(data["phase"])
        stiffness = np.asarray(data["K"], dtype=float)
        damping = np.asarray(data["D"], dtype=float)
        transition = np.asarray(data["transition_active"]).astype(int)
        tau_task = np.asarray(data["tau_task"], dtype=float)
        tau_null = np.asarray(data["tau_null"], dtype=float)
        tau_raw = np.asarray(data["tau_raw"], dtype=float)
        tau_applied = np.asarray(data["tau_applied"], dtype=float)
        tau_before = np.asarray(data["tau_before"], dtype=float).reshape(7)
        v_ref = np.asarray(data["v_ref"], dtype=float)
    except KeyError as exc:
        return [_reason("missing_field", case_id, str(exc).strip("'"))]
    arrays = (t_state, q, dq, command, stiffness, damping, tau_task, tau_null, tau_raw, tau_applied, v_ref)
    if t_state.shape != (n + 1,) or q.shape != (n + 1, 7) or command.shape != (n, 6) or tau_applied.shape != (n, 7):
        reasons.append(_reason("sample_count", case_id, "t_state", int(t_state.shape[0]) if t_state.ndim else 0, n + 1))
        return reasons
    if any(not np.all(np.isfinite(item)) for item in arrays):
        return [_reason("nonfinite", case_id, "trace")]
    if np.any(np.abs(np.diff(t_state) - dt) > 1e-9) or abs(float(t_state[0])) > 1e-12:
        reasons.append(_reason("bad_time", case_id, "dt"))
    if np.max(np.abs(dq)) > SPEED_LIMIT + 1e-9:
        reasons.append(_reason("joint_speed", case_id, "dq", float(np.max(np.abs(dq))), SPEED_LIMIT))
    ranges = context.ranges
    if np.any(q < ranges[:, 0] - 1e-6) or np.any(q > ranges[:, 1] + 1e-6):
        reasons.append(_reason("joint_range", case_id, "q"))
    events = {}
    if case_id.startswith("V6-zero-phys-"):
        reasons.extend(semantic_arrays(data, case_id, "base", dt, context.controller_version))
        if int(data["fault_tick"]) != -1 or str(data["fault_reason"]) != "" or int(data["stopped_ok"]) != 0:
            reasons.append(_reason("unexpected_fault", case_id, "fault_tick"))
        reasons.extend(_check_zero_trace(case_id, data, q, dq, command, execution, guard_status, v_ref, tau_applied, tau_before, dt, n, context))
    else:
        reasons.extend(
            _check_stop_trace(
                case_id,
                data,
                t_state,
                q,
                dq,
                command,
                execution,
                mode,
                guard_status,
                phase,
                stiffness,
                damping,
                transition,
                tau_task,
                tau_null,
                tau_raw,
                tau_applied,
                tau_before,
                v_ref,
                dt,
                n,
                context,
                events,
            )
        )
    if metrics is not None:
        metrics.update(trace_metrics(case_id, data, dt, events))
    return reasons


def _check_zero_trace(case_id, data, q, dq, command, execution, guard_status, v_ref, tau_applied, tau_before, dt, n, context):
    reasons = []
    try:
        candidate = np.asarray(data["candidate"], dtype=float)
        final = np.asarray(data["final_twist"], dtype=float)
        p_ref = np.asarray(data["p_ref"], dtype=float)
        p_before = np.asarray(data["p_ref_before"], dtype=float)
        correction = np.asarray(data["reference_correction_pos"], dtype=float)
    except KeyError as exc:
        return [_reason("missing_field", case_id, str(exc).strip("'"))]
    if candidate.shape != (n, 6) or final.shape != (n, 6) or p_ref.shape != (n, 3):
        return [_reason("bad_shape", case_id, "candidate")]
    if np.any(guard_status != "RUNNING") or np.any(execution != "zero"):
        reasons.append(_reason("unexpected_fault", case_id, "guard_status"))
    home = np.array(context.scene.config["q_torque_poses"][0], dtype=float)
    side = 1.0 if "-upper-" in case_id else -1.0
    home[0] = context.ranges[0, 1] - 0.04 if side > 0 else context.ranges[0, 0] + 0.04
    if np.max(np.abs(q[0] - home)) > 1e-8:
        reasons.append(_reason("initial_state", case_id, "q", float(q[0, 0]), float(home[0])))
    if np.any(np.abs(command) > 1e-12):
        reasons.append(_reason("stimulus", case_id, "command"))
    _sc, jac0, mass0, _bias0 = _arm_state(q[0], dq[0], context)
    scale_matrix = np.diag([1., 1., 1., LENGTH, LENGTH, LENGTH])
    _null0, jbar0, _lam0 = damped_nullspace(scale_matrix @ jac0, mass0, EPS_NORMAL, A_FLOOR)
    history = np.zeros(6)
    history[:3] = side * 0.01 * jbar0[0, :3] / np.linalg.norm(jbar0[0, :3])
    if not np.allclose(data["history_twist0"], history, atol=1e-9, rtol=0):
        reasons.append(_reason("stimulus", case_id, "history_twist0"))
    outward_hit = None
    for k in range(n):
        speed = np.linalg.norm(history[:3])
        expected_candidate = history.copy()
        expected_candidate[:3] *= max(0.0, 1.0 - 0.25 * dt / speed) if speed > 0 else 0.0
        if not np.allclose(candidate[k], expected_candidate, atol=1e-9, rtol=0):
            reasons.append(_reason("stimulus", case_id, "candidate", text=f"stimulus candidate index={k}"))
            break
        history = final[k].copy()
        if execution[k] not in {"zero", "run"}:
            continue
        _scene_k, jacobian, mass, _bias = _arm_state(q[k], dq[k], context)
        rho = _rho(jacobian)
        _scale, eps = singularity_gains(rho, EPS_NORMAL, EPS_SINGULAR, RHO_NORMAL, RHO_STOP)
        _null, jbar, _lam = damped_nullspace(np.diag([1.0, 1.0, 1.0, LENGTH, LENGTH, LENGTH]) @ jacobian, mass, eps, A_FLOOR)
        beta = joint_beta(jbar, candidate[k], q[k], context.ranges)
        expected = np.zeros(6) if beta <= 1e-15 else candidate[k] * beta
        if not np.allclose(final[k], expected, atol=1e-8, rtol=0.0):
            reasons.append(_reason("joint_projection", case_id, "final_twist", float(np.linalg.norm(final[k])), float(np.linalg.norm(expected)), text=f"joint_projection t={float(data['t_state'][k])}"))
            break
        delta = p_ref[k] - p_before[k] - correction[k]
        if not np.allclose(delta, expected[:3] * dt, atol=1e-8, rtol=0.0):
            reasons.append(_reason("joint_projection", case_id, "integral", float(np.linalg.norm(delta)), float(np.linalg.norm(expected[:3] * dt))))
            break
        pred = float((jbar @ (np.diag([1.0, 1.0, 1.0, LENGTH, LENGTH, LENGTH]) @ expected))[0])
        outward = pred if "-upper-" in case_id else -pred
        if outward > 1e-9 and outward_hit is None:
            outward_hit = (k, outward)
    if outward_hit is not None:
        reasons.append(_reason("joint_projection", case_id, "qdot_pred", outward_hit[1], 0.0, text=f"joint_projection index={outward_hit[0]}"))
    if np.max(np.abs(tau_applied[0] - tau_before)) > TAU_RATE * dt + 1e-6 or np.max(np.abs(np.diff(tau_applied, axis=0))) > TAU_RATE * dt + 1e-6:
        reasons.append(_reason("torque_rate", case_id, "tau_applied"))
    if np.any(np.linalg.norm(v_ref[:, :3], axis=1) > 0.05 + 1e-6) or np.any(np.linalg.norm(v_ref[:, 3:], axis=1) > 0.3 + 1e-6):
        reasons.append(_reason("reference_speed", case_id, "v_ref"))
    del command
    return reasons


def _check_stop_trace(
    case_id,
    data,
    t_state,
    q,
    dq,
    command,
    execution,
    mode,
    guard_status,
    phase,
    stiffness,
    damping,
    transition,
    tau_task,
    tau_null,
    tau_raw,
    tau_applied,
    tau_before,
    v_ref,
    dt,
    n,
    context,
    events,
):
    reasons = []
    target_q = SINGULAR_Q if case_id.startswith("V6-stop-rest-") else CROSS_Q
    target_dq = REST_DQ if case_id.startswith("V6-stop-rest-") else CROSS_DQ
    if np.max(np.abs(q[0] - target_q)) > 1e-8 or np.max(np.abs(dq[0] - target_dq)) > 1e-8:
        reasons.append(_reason("initial_state", case_id, "q"))
    if np.any(np.abs(data["history_twist0"]) > 1e-12):
        reasons.append(_reason("stimulus", case_id, "history_twist0"))
    fault_k = None
    rhos = np.zeros(n + 1)
    biases = np.zeros((n, 7))
    for k in range(n + 1):
        scene, jacobian, _mass, bias = _arm_state(q[k], dq[k], context)
        rhos[k] = _rho(jacobian)
        if k < n:
            biases[k] = bias
        if fault_k is None and rhos[k] <= RHO_STOP:
            fault_k = k
        for i in range(int(scene.data.ncon)):
            if float(scene.data.contact[i].dist) < -0.001:
                reasons.append(_reason("reference_error", case_id, "contact_dist", float(scene.data.contact[i].dist), -0.001))
                break
    if fault_k is not None and fault_k >= n:
        return reasons + [_reason("sample_count", case_id, "stop_window")]
    if fault_k is None:
        return reasons + [_reason("singularity_stop", case_id, "rho", float(np.min(rhos)), RHO_STOP)]
    if case_id.startswith("V6-stop-rest-") and fault_k != 0:
        reasons.append(_reason("singularity_stop", case_id, "rho", float(rhos[0]), RHO_STOP))
    if case_id.startswith("V6-stop-cross-") and fault_k < 1:
        reasons.append(_reason("singularity_stop", case_id, "cross", float(rhos[0]), RHO_STOP))
    if float(t_state[-1] - t_state[fault_k]) + 1e-12 < TRACE_AFTER_S:
        reasons.append(_reason("sample_count", case_id, "stop_window", float(t_state[-1] - t_state[fault_k]), TRACE_AFTER_S))
    try:
        logged_tick = int(np.asarray(data["fault_tick"]).reshape(-1)[0])
        logged_reason = str(np.asarray(data["fault_reason"]).reshape(-1)[0])
        ticks = np.asarray(data["tick"]).astype(int)
    except KeyError as exc:
        return reasons + [_reason("missing_field", case_id, str(exc).strip("'"))]
    if logged_reason != "singularity":
        reasons.append(_reason("singularity_stop", case_id, "fault_reason", text="singularity_stop fault_reason"))
    if ticks.shape != (n + 1,) or logged_tick != int(ticks[fault_k]):
        reasons.append(_reason("fault_tick", case_id, "fault_tick", logged_tick, int(ticks[fault_k]) if ticks.shape == (n + 1,) else -1, text=f"fault_tick logged={logged_tick}"))
    if np.any(phase != "TRANSPORT"):
        reasons.append(_reason("singularity_stop", case_id, "phase"))
    expected_command = np.zeros((fault_k, 6))
    if case_id.startswith("V6-stop-cross-"):
        expected_command[:, 0] = 0.02
    if not np.allclose(command[:fault_k], expected_command, atol=1e-12, rtol=0):
        reasons.append(_reason("stimulus", case_id, "command"))
    for k in range(fault_k):
        if mode[k] != "RUNNING" or guard_status[k] != "RUNNING" or execution[k] == "stop":
            reasons.append(_reason("singularity_stop", case_id, "pre_stop", text=f"singularity_stop pre_stop index={k}"))
            break
    if mode[fault_k] != "STOPPING" or guard_status[fault_k] != "STOPPING" or execution[fault_k] != "stop":
        reasons.append(_reason("singularity_stop", case_id, "mode", text=f"singularity_stop mode={mode[fault_k]} execution={execution[fault_k]}"))
    for k in range(fault_k, n):
        if execution[k] != "stop" or mode[k] not in {"STOPPING", "STOPPED"} or guard_status[k] != mode[k]:
            reasons.append(_reason("singularity_stop", case_id, "mode", text=f"singularity_stop index={k}"))
            break
        if np.max(np.abs(stiffness[k])) > 1e-9 or np.max(np.abs(damping[k])) > 1e-9 or int(transition[k]) != 0:
            reasons.append(_reason("singularity_stop", case_id, "K"))
            break
        if np.linalg.norm(tau_task[k]) > 1e-8 or np.linalg.norm(tau_null[k]) > 1e-8:
            reasons.append(_reason("stop_torque", case_id, "tau_task", float(np.linalg.norm(tau_task[k])), 0.0, text=f"stop_torque index={k}"))
            break
        if np.linalg.norm(v_ref[k, :3]) > 1e-8 or np.linalg.norm(v_ref[k, 3:]) > 1e-8:
            reasons.append(_reason("singularity_stop", case_id, "v_ref"))
            break
        if np.linalg.norm(command[k]) > 1e-8:
            reasons.append(_reason("auto_resume", case_id, "command", float(np.linalg.norm(command[k])), 0.0, text=f"auto_resume t={float(t_state[k])}"))
            break
        expected_raw = biases[k] - STOP_DAMPING * dq[k]
        if not np.allclose(tau_raw[k], expected_raw, atol=1e-5, rtol=0.0):
            reasons.append(_reason("stop_torque", case_id, "tau_raw", float(np.linalg.norm(tau_raw[k] - expected_raw)), 0.0))
            break
        prev = tau_before if k == 0 else tau_applied[k - 1]
        expected_applied = clip_torque(tau_raw[k], prev, TAU_MAX, TAU_RATE, dt)
        if not np.allclose(tau_applied[k], expected_applied, atol=1e-6, rtol=0.0):
            reasons.append(_reason("torque_rate", case_id, "tau_applied"))
            break
    events["fault"] = float(t_state[fault_k])
    held = _stop_speed(reasons, case_id, t_state, dq, fault_k)
    if held is not None:
        expected_mode = np.where(t_state[fault_k:n] >= held[1] - 1e-12, "STOPPED", "STOPPING")
        if np.any(mode[fault_k:] != expected_mode):
            reasons.append(_reason("singularity_stop", case_id, "stop_confirm_mode"))
        events["stop_speed"] = held[0]
        events["stop_confirm"] = held[1]
    if int(data["stopped_ok"]) != 1:
        reasons.append(_reason("stop_speed", case_id, "stopped_ok"))
    if fault_k > 0:
        from feedingrobot.controllers.v4fields import check_logged_gains

        prefix = slice(0, fault_k)
        built = check_logged_gains(
            stiffness[prefix],
            damping[prefix],
            transition[prefix],
            _as_str(data["active_gear"])[prefix],
            _as_str(data["target_gear"])[prefix],
            phase[prefix],
            dt,
            case_id,
            reasons,
            controller_version=context.controller_version,
        )
        if built is not None and (np.any(np.linalg.norm(v_ref[prefix, :3], axis=1) > built["v_lim"] + 1e-9) or np.any(np.linalg.norm(v_ref[prefix, 3:], axis=1) > built["w_lim"] + 1e-9)):
            reasons.append(_reason("reference_speed", case_id, "v_ref"))
    return reasons


def _stop_speed(reasons, case_id, t_state, dq, fault_k):
    speed = np.max(np.abs(dq), axis=1)
    t_fault = float(t_state[fault_k])
    reached = None
    for k in range(fault_k, len(speed)):
        if float(t_state[k] - t_fault) > STOP_WITHIN_S + 1e-9:
            break
        if speed[k] <= STOP_SPEED + 1e-12:
            reached = k
            break
    if reached is None:
        reasons.append(_reason("stop_speed", case_id, "dq", float(np.min(speed[fault_k:])), STOP_SPEED, text="stop_speed within"))
        return
    hold_end = float(t_state[reached] + STOP_HOLD_S)
    if float(t_state[-1]) + 1e-12 < hold_end:
        reasons.append(_reason("sample_count", case_id, "stop_hold", float(t_state[-1] - t_state[reached]), STOP_HOLD_S))
        return
    window = (t_state >= float(t_state[reached]) - 1e-12) & (t_state <= hold_end + 1e-12)
    if int(np.count_nonzero(window)) < 2 or np.any(speed[window] > STOP_SPEED + 1e-9):
        reasons.append(_reason("stop_speed", case_id, "dq", float(np.max(speed[window])) if np.any(window) else None, STOP_SPEED, text="stop_speed hold"))

    else:
        return float(t_state[reached]), hold_end


def trace_metrics(case_id, data, dt, events):
    from feedingrobot.controllers.so3 import orientation_error

    times = data["t_state"][1:]
    pos = np.linalg.norm(data["p_ref"] - data["tcp_pos"][1:], axis=1)
    rot = np.array([np.linalg.norm(orientation_error(ref, actual)) for ref, actual in zip(data["r_ref"], data["tcp_rot"][1:])])
    wrench = data["wrench_compensated"]
    force = np.linalg.norm(wrench[:, :3], axis=1)
    torque = np.linalg.norm(wrench[:, 3:], axis=1)
    contact_peak = np.zeros(len(times))
    contact_sum = np.zeros(len(times))
    offsets = data["contact_offsets"]
    for k in range(len(times)):
        contact = data["contact_force"][offsets[k]:offsets[k + 1]]
        if len(contact):
            contact_peak[k] = np.max(np.linalg.norm(contact, axis=1))
            contact_sum[k] = np.sum(np.linalg.norm(contact, axis=1))
    series = {"pos_peak": pos, "rot_peak": rot, "tcp_force": force, "tcp_torque": torque, "contact_force": contact_peak, "contact_sum": contact_sum}
    metrics = {key: float(np.max(value)) for key, value in series.items()}
    metrics["times"] = {key: float(times[np.argmax(value)]) for key, value in series.items()}
    metrics.update({"pos_rms": float(np.sqrt(np.mean(pos**2))), "rot_rms": float(np.sqrt(np.mean(rot**2))), "impulse": float(np.sum(force) * dt), "tau_peak": np.max(np.abs(data["tau_applied"]), axis=0), "qpos0": data["qpos0"], "qvel0": data["qvel0"], "events": events})
    metrics["times"].update({"pos_rms": float(times[0]), "rot_rms": float(times[0]), "impulse": float(times[-1])})
    return metrics
