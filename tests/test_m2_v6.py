"""V6: zero-command projection, singularity stop, and staged release windows."""

import os
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from feedingrobot.controllers.acceptance import v6_trace_ids, v6_zero_grid_ids
from feedingrobot.controllers.cartesian_impedance import control_step, damped_nullspace, enters_singularity_stop, singularity_gains
from feedingrobot.controllers.v3spec import RELEASE_SCORING_VERSION, _score_release, run_adversarial
from feedingrobot.controllers.v6checks import (
    CROSS_DQ,
    CROSS_Q,
    REST_DQ,
    SINGULAR_Q,
    STOP_DURATION_S,
    ZERO_DURATION_S,
    joint_beta,
)
from feedingrobot.sim.scene import FeedingScene
from tests.m2_evidence import write_json, write_npz
from tests.m2_support import load_m2_config, place_arm, start_controller
from tests.test_m2_v2 import _apply_variant, _record_physical, _tcp_twist


def _codes(reasons):
    return {row["code"] for row in reasons}


def _fields(reasons):
    return {row.get("field") for row in reasons}


def _outward(scene, q):
    state = place_arm(scene, q)
    cfg = load_m2_config()
    ctl, guard = start_controller(scene, cfg, state, tau=scene.bias_tau())
    ctl.power_on = False
    jac, _ = ctl._jacobian()
    scaled = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) @ jac
    _null, jbar, _lam = damped_nullspace(scaled, ctl._mass(), ctl.eps_normal, ctl.a_floor)
    direction = jbar[0, :3] / np.linalg.norm(jbar[0, :3])
    return state, ctl, guard, direction


@pytest.mark.parametrize("dt", [0.001, 0.0005])
def test_zero_command_still_projects_outward_history(dt):
    scene = FeedingScene("configs/m1_scene.json")
    scene.model.opt.timestep = dt
    scene.dt = dt
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    q[0] = scene.model.jnt_range[scene.index.arm_joint_ids[0], 1] - 0.04
    state, ctl, guard, direction = _outward(scene, q)
    ctl.reference.v_lin = direction * 0.01
    assert ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["accepted"]
    _tau, info = ctl.compute(state, dt)
    assert guard.failure is None
    assert info["execution"] == "zero"
    assert info["qdot_pred"][0] <= 1e-9
    assert info["beta"] <= 1e-12


@pytest.mark.parametrize("dt", [0.001, 0.0005])
def test_singular_region_enters_latched_controlled_stop(dt):
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    scene.model.opt.timestep = dt
    scene.dt = dt
    state = place_arm(scene, SINGULAR_Q)
    assert not state["contacts"]
    ctl, guard = start_controller(scene, cfg, state, tau=scene.bias_tau())
    ctl.power_on = False
    ctl.reference.p_ref[0] += 0.002
    assert ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["accepted"]
    _tau, info = ctl.compute(state, dt)
    assert info["rho"] < cfg["rho_stop"]
    assert guard.status == "STOPPING" and guard.failure["reason"] == "singularity"
    assert guard.failure["tick"] == int(state["tick"])
    assert abs(float(guard.failure["time"]) - float(state["episode_time"])) <= 1e-12
    assert abs(float(guard._stop_since) - float(state["episode_time"])) <= 1e-12
    assert np.linalg.norm(info["tau_task"]) <= 1e-9
    assert np.linalg.norm(info["tau_null"]) <= 1e-9
    assert np.linalg.norm(info["v_ref"][:3]) <= 1e-9 and np.linalg.norm(info["v_ref"][3:]) <= 1e-9
    assert info["execution"] == "stop" and info["control_mode"] == "STOPPING"
    assert ctl._command is None
    assert ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["rejected"] == "fault"
    _nxt, second = control_step(ctl, guard)
    _third_state, third = control_step(ctl, guard)
    assert guard.status in {"STOPPING", "STOPPED"} and guard.failure["reason"] == "singularity"
    assert np.linalg.norm(second["tau_task"]) <= 1e-9 and np.linalg.norm(third["tau_task"]) <= 1e-9


def test_joint_allow_boundaries():
    scene = FeedingScene("configs/m1_scene.json")
    q0 = np.array(scene.config["q_torque_poses"][0], dtype=float)
    state = place_arm(scene, q0)
    ctl, _guard = start_controller(scene, load_m2_config(), state)
    upper = float(ctl.ranges[0, 1])
    jbar = np.zeros((7, 6))
    jbar[0, 0] = 1.0
    twist = np.array([1.0, 0, 0, 0, 0, 0])
    expected = {
        0.05: 0.0,
        0.05 - 1e-6: 0.0,
        0.075: 0.2,
        0.10: 0.4,
        0.10 + 1e-4: 0.4,
    }
    expected[0.05 + 1e-4] = 0.4 * (1e-4 / 0.05)
    expected[0.10 - 1e-4] = 0.4 * ((0.05 - 1e-4) / 0.05)
    for dist, allow in expected.items():
        q = q0.copy()
        q[0] = upper - dist
        beta = ctl._joint_scale(jbar, twist, q)
        assert abs(beta - allow) <= 1e-8, (dist, beta, allow)


def test_zero_far_from_limit_decelerates():
    scene = FeedingScene("configs/m1_scene.json")
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    state = place_arm(scene, q)
    ctl, guard = start_controller(scene, load_m2_config(), state, tau=scene.bias_tau())
    ctl.power_on = False
    ctl.reference.v_lin[:] = [0.01, 0.0, 0.0]
    before = ctl.reference.p_ref.copy()
    ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    _tau, info = ctl.compute(state, 0.001)
    assert info["execution"] == "zero" and guard.failure is None
    speed = float(np.linalg.norm(info["v_ref"][:3]))
    assert abs(speed - 0.00975) <= 1e-6
    assert speed > 1e-4
    assert np.linalg.norm(info["p_ref"] - before) > 1e-6


def test_inward_history_is_not_frozen():
    scene = FeedingScene("configs/m1_scene.json")
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    q[0] = scene.model.jnt_range[scene.index.arm_joint_ids[0], 1] - 0.04
    state, ctl, guard, direction = _outward(scene, q)
    ctl.reference.v_lin = -direction * 0.01
    ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    _tau, info = ctl.compute(state, 0.001)
    assert guard.failure is None and info["execution"] == "zero"
    assert info["qdot_pred"][0] <= 1e-9
    assert np.linalg.norm(info["v_ref"][:3]) > 1e-4


def test_multi_axis_uses_strictest_beta():
    scene = FeedingScene("configs/m1_scene.json")
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    state = place_arm(scene, q)
    ctl, _guard = start_controller(scene, load_m2_config(), state)
    q = q.copy()
    q[0] = ctl.ranges[0, 1] - 0.075
    q[1] = ctl.ranges[1, 1] - 0.10
    jbar = np.zeros((7, 6))
    jbar[0, 0] = 1.0
    jbar[1, 3] = 1.0
    candidate = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    beta = ctl._joint_scale(jbar, candidate, q)
    # Joint 0 allow is 0.2 at pred 1. Joint 1 pred is 0.1, allow 0.4, ratio 4, so the scalar is 0.2.
    assert abs(beta - 0.2) <= 1e-8
    final = candidate * beta
    assert np.allclose(final / np.linalg.norm(final), candidate / np.linalg.norm(candidate))
    ranges = ctl.ranges
    assert abs(joint_beta(jbar, candidate, q, ranges) - beta) <= 1e-12


def test_stop_power_on_and_prohibit_do_not_integrate():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, SINGULAR_Q)
    ctl, guard = start_controller(scene, cfg, state, tau=scene.bias_tau())
    ctl.power_on = False
    ctl.set_command(np.array([0.02, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    before = ctl.reference.p_ref.copy()
    _tau, info = ctl.compute(state, 0.001)
    assert info["execution"] == "stop"
    assert np.linalg.norm(info["p_ref"] - before) <= 1e-9
    assert np.linalg.norm(info["v_ref"]) <= 1e-9
    normal = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl.reset(normal, scene.bias_tau())
    ctl.power_on = True
    ctl.set_command(np.array([0.05, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    before = ctl.reference.p_ref.copy()
    _tau, info = ctl.compute(normal, 0.001)
    assert info["execution"] == "power_on" and guard.failure is None
    assert np.linalg.norm(info["v_ref"]) <= 1e-9
    assert np.linalg.norm(info["p_ref"] - before) <= 1e-8
    shaper = ctl.reference
    shaper.v_lin[:] = [0.01, 0, 0]
    held = shaper.p_ref.copy()
    out = shaper.shape(np.array([0.05, 0, 0, 0, 0, 0]), normal["tcp_pos"], normal["tcp_mat"], np.zeros(3), 0.001, {"v": 0.05, "w": 0.3, "a": 0.25, "alpha": 1.5, "pos_dev": 0.02, "rot_dev": 0.14}, 1.0, 1.0, True, False, "prohibit", project=lambda twist: 1.0)
    assert np.linalg.norm(out["v_hist"]) <= 1e-12
    assert np.allclose(out["p_ref"], held)


def test_zero_history_does_not_rewind():
    scene = FeedingScene("configs/m1_scene.json")
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    q[0] = scene.model.jnt_range[scene.index.arm_joint_ids[0], 1] - 0.04
    state, ctl, guard, direction = _outward(scene, q)
    ctl.reference.v_lin = direction * 0.01
    ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    state, blocked = control_step(ctl, guard)
    assert blocked["beta"] <= 1e-12
    assert np.linalg.norm(ctl.reference.v_lin) <= 1e-12
    assert ctl.set_command(-np.concatenate([direction * 0.05, np.zeros(3)]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["accepted"]
    _state, nxt = control_step(ctl, guard)
    assert guard.failure is None
    assert float(np.linalg.norm(nxt["v_ref"][:3])) < 0.002
    tau = np.asarray(nxt["tau_cmd"])
    assert np.max(np.abs(tau - nxt["tau_before"])) <= 2000.0 * 0.001 + 1e-6


def test_rho_threshold_edges():
    for rho, stopped, scale in (
        (0.01, True, 0.0),
        (0.01 - 1e-6, True, 0.0),
        (0.01 + 1e-4, False, None),
        (0.05 - 1e-4, False, None),
        (0.05, False, 1.0),
        (0.05 + 1e-4, False, 1.0),
    ):
        assert enters_singularity_stop(rho, 0.01) is stopped
        gained, _eps = singularity_gains(rho, 1e-8, 1e-2, 0.05, 0.01)
        if scale is not None:
            assert abs(gained - scale) <= 1e-12
        else:
            assert 0.0 < gained < 1.0


def test_first_fault_is_kept():
    scene = FeedingScene("configs/m1_scene.json")
    state = place_arm(scene, SINGULAR_Q)
    ctl, guard = start_controller(scene, load_m2_config(), state, tau=scene.bias_tau())
    ctl.power_on = False
    guard.latch({"tick": 3, "time": 0.0, "reason": "saturation", "phase": "TRANSPORT", "pair": None, "geoms": None, "measured": 0.02, "limit": 0.02})
    ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    _tau, info = ctl.compute(state, 0.001)
    assert guard.failure["reason"] == "saturation" and guard.failure["tick"] == 3
    assert info["execution"] == "stop" and np.linalg.norm(info["tau_task"]) <= 1e-9
    guard.status = "ABORTED"
    aborted, info = ctl.compute(state, 0.001)
    assert info.get("apply") is False and np.allclose(aborted, 0.0)


def test_power_on_blend_yields_to_singularity():
    scene = FeedingScene("configs/m1_scene.json")
    state = place_arm(scene, SINGULAR_Q)
    ctl, guard = start_controller(scene, load_m2_config(), state, tau=scene.bias_tau())
    ctl.power_on = True
    ctl.transition_active = True
    ctl.transition_elapsed = 0.05
    ctl.phase = "ACQUIRE"
    ctl._k[:] = 200.0
    ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "ACQUIRE")
    _tau, info = ctl.compute(state, 0.001)
    assert guard.failure["reason"] == "singularity"
    assert info["execution"] == "stop" and info["control_mode"] == "STOPPING"
    assert not info["power_on"] and not info["transition_active"]
    assert np.allclose(info["k"], 0.0) and np.allclose(info["d"], 0.0)
    assert np.linalg.norm(info["tau_task"]) <= 1e-9
    assert info["phase_k"] == "ACQUIRE"


def test_rho_recovery_stays_latched():
    scene = FeedingScene("configs/m1_scene.json")
    state = place_arm(scene, SINGULAR_Q)
    ctl, guard = start_controller(scene, load_m2_config(), state, tau=scene.bias_tau())
    ctl.power_on = False
    ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    ctl.compute(state, 0.001)
    assert guard.failure["reason"] == "singularity"
    home = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl._tcp_pos = np.asarray(home["tcp_pos"], dtype=float)
    ctl._tcp_rot = np.asarray(home["tcp_mat"], dtype=float)
    _tau, info = ctl.compute(home, 0.001)
    assert info["rho"] > 0.01
    assert guard.failure["reason"] == "singularity"
    assert info["execution"] == "stop"
    assert ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["rejected"] == "fault"


def test_reset_clears_or_relatches_singularity():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, SINGULAR_Q)
    ctl, guard = start_controller(scene, cfg, state, tau=scene.bias_tau())
    ctl.power_on = False
    ctl.compute(state, 0.001)
    home = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl.reset(home, scene.bias_tau())
    ctl.power_on = False
    assert guard.failure is None and guard.status == "RUNNING"
    assert ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.1, "TRANSPORT")["accepted"]
    singular = place_arm(scene, SINGULAR_Q)
    ctl.reset(singular, scene.bias_tau())
    ctl.power_on = False
    assert guard.failure is None
    ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), ctl.now(), ctl.now() + 0.1, "TRANSPORT")
    _tau, info = ctl.compute(singular, 0.001)
    assert guard.failure["reason"] == "singularity" and info["execution"] == "stop"


def _release_arrays(n=2500, dt=0.001, blocked=149, ramp=1.2):
    t_state = np.arange(n + 1) * dt
    command = np.zeros((n, 6))
    external = np.zeros((n, 6))
    command[int(round(0.05 / dt)): blocked + 1, 0] = 0.05
    external[int(round(0.04 / dt)) + 1: blocked + 1, 0] = -3.0
    unload = blocked + 1
    force = unload + int(round(ramp / dt))
    for k in range(unload, force):
        external[k, 0] = -3.0 * max(1.0 - (t_state[k] - t_state[unload]) / ramp, 0.0)
    twist = np.zeros((n + 1, 6))
    twist[:, 0] = 0.001
    tcp = np.cumsum(np.concatenate([[0.0], np.full(n, 0.001 * dt)])).reshape(-1, 1) * np.array([[1.0, 0.0, 0.0]])
    tcp = np.repeat(tcp, 1, axis=0) if False else np.column_stack([np.cumsum(np.concatenate([[0.0], np.full(n, 0.001 * dt)])), np.zeros(n + 1), np.zeros(n + 1)])
    rebuilt = np.zeros(n, dtype=bool)
    rebuilt[blocked] = True
    return command, external, twist, tcp, t_state, rebuilt, unload, force


def test_slow_release_scores_from_unload():
    command, external, twist, tcp, t_state, rebuilt, _unload, _force = _release_arrays()
    twist[950:1151, 0] = 0.009
    metrics = {"events": {}, "times": {}}
    reasons = []
    _score_release(reasons, metrics, "release-s0-A", "release", rebuilt, command, external, twist, tcp, t_state, 0.001, {})
    assert reasons == [], reasons
    assert metrics["diagnostic_ramp_v"] > 0.005
    spiked = twist.copy()
    spiked[300, 0] = 0.5
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, spiked, tcp, t_state, 0.001, {})
    assert "release_speed" in _codes(reasons)
    assert {"unload_speed", "ramp_speed"} <= _fields(reasons)
    late = twist.copy()
    late[1100, 0] = 0.5
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, late, tcp, t_state, 0.001, {})
    assert "ramp_speed" in _fields(reasons)
    assert "unload_speed" not in _fields(reasons)
    moved = tcp.copy()
    moved[200:, 0] += 0.04
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, twist, moved, t_state, 0.001, {})
    assert "unload_fd" in _fields(reasons)
    bumped = tcp.copy()
    bumped[300:400, 0] += 0.006
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, twist, bumped, t_state, 0.001, {})
    assert "unload_displacement" in _fields(reasons)


def test_release_boundaries_and_fast_stable_window():
    command, external, twist, tcp, t_state, rebuilt, unload, force = _release_arrays()
    quiet = twist.copy()
    quiet[:, 0] = 0.03
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, quiet, tcp, t_state, 0.001, {})
    assert "unload_speed" not in _fields(reasons)
    quiet[unload + 10, 0] = 0.030001
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, quiet, tcp, t_state, 0.001, {})
    assert "unload_speed" in _fields(reasons)
    stable = twist.copy()
    stable[force + 900, 0] = 0.01
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, stable, tcp, t_state, 0.001, {})
    assert "force_zero_stable" in _fields(reasons)
    stable = twist.copy()
    stable[force + 900, 3] = 0.08
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release-s0-A", "release", rebuilt, command, external, stable, tcp, t_state, 0.001, {})
    assert "force_zero_omega" in _fields(reasons)
    fast_external = np.zeros_like(external)
    fast_external[: unload, 0] = -3.0
    fast_twist = twist.copy()
    fast_twist[unload + 900, 0] = 0.01
    reasons = []
    _score_release(reasons, {"events": {}, "times": {}}, "release_fast-s0-A", "release_fast", rebuilt, command, fast_external, fast_twist, tcp, t_state, 0.001, {})
    assert "unload_stable" in _fields(reasons)
    assert RELEASE_SCORING_VERSION == "v6-staged"


def test_adversarial_io_error_is_reported(tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("feedingrobot.controllers.v3spec.tempfile.TemporaryDirectory", _boom)
    rows = run_adversarial(tmp_path)
    assert rows == [{"name": "copy", "fixes_passed": False, "evidence_valid": False, "codes": ["io_error"]}]


@pytest.mark.parametrize("case_id", v6_zero_grid_ids())
def test_zero_projection_grid(case_id):
    _tag, _zero, joint_s, side, motion, variant = case_id.split("-")
    joint = int(joint_s[1:])
    dt = 0.001 if variant == "A" else 0.0005
    scene = FeedingScene("configs/m1_scene.json")
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    state = place_arm(scene, q)
    ctl, _guard = start_controller(scene, load_m2_config(), state)
    outward = 1.0 if side == "upper" else -1.0
    q = q.copy()
    q[joint] = ctl.ranges[joint, 1] - 0.04 if side == "upper" else ctl.ranges[joint, 0] + 0.04
    jbar = np.zeros((7, 6))
    candidate = np.zeros(6)
    if motion == "linear":
        jbar[joint, 0] = outward
        candidate[0] = 0.05
    elif motion == "angular":
        jbar[joint, 3] = outward
        candidate[3] = 0.5
    else:
        jbar[joint, 0] = outward
        jbar[joint, 3] = outward
        candidate[0] = 0.02
        candidate[3] = 0.2
    beta = ctl._joint_scale(jbar, candidate, q)
    final = np.zeros(6) if beta <= 1e-15 else candidate * beta
    scale = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    pred = float((jbar @ (scale @ final))[joint])
    assert (pred if side == "upper" else -pred) <= 1e-9
    write_json(
        case_id,
        {
            "dt": dt,
            "q": q,
            "jbar": jbar,
            "candidate": candidate,
            "final": final,
            "beta": beta,
            "p_before": np.zeros(3),
            "p_after": final[:3] * dt,
            "correction": np.zeros(3),
        },
    )


def _trace(case_id):
    kind, name, variant = case_id.split("-")[-3:]
    del kind
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    # These files exercise the archived V6 evidence format, whose stop law is frozen at 1.
    cfg["stop"]["damping_nm_s_per_rad"] = 1.0
    cfg["power_on_damping_scale"] = 1.0
    if name in {"upper", "lower"}:
        q = np.array(scene.config["q_torque_poses"][0], dtype=float)
        q[0] = scene.model.jnt_range[scene.index.arm_joint_ids[0], 1] - 0.04 if name == "upper" else scene.model.jnt_range[scene.index.arm_joint_ids[0], 0] + 0.04
        dq0 = np.zeros(7)
        duration = ZERO_DURATION_S
        twist = np.zeros(6)
    elif name == "rest":
        q = SINGULAR_Q.copy()
        dq0 = REST_DQ.copy()
        duration = STOP_DURATION_S
        twist = np.zeros(6)
    else:
        q = CROSS_Q.copy()
        dq0 = CROSS_DQ.copy()
        duration = STOP_DURATION_S
        twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    place_arm(scene, q)
    scene.data.qvel[scene.index.arm_dof_adr] = dq0
    mujoco.mj_forward(scene.model, scene.data)
    qpos = scene.data.qpos.copy()
    qvel = scene.data.qvel.copy()
    _apply_variant(scene, variant)
    scene.data.qpos[:] = qpos
    scene.data.qvel[:] = qvel
    mujoco.mj_forward(scene.model, scene.data)
    scene.tick = 0
    scene.time_offset = float(scene.data.time)
    state = scene.snapshot()
    bias = np.array(scene.data.qfrc_bias[scene.index.arm_dof_adr], dtype=float)
    ctl, guard = start_controller(scene, cfg, state, tau=bias)
    ctl.power_on = False
    if name in {"upper", "lower"}:
        jac, _bias_j = ctl._jacobian()
        scaled = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) @ jac
        _null, jbar, _lam = damped_nullspace(scaled, ctl._mass(), ctl.eps_normal, ctl.a_floor)
        direction = jbar[0, :3] / np.linalg.norm(jbar[0, :3])
        ctl.reference.v_lin = direction * (0.01 if name == "upper" else -0.01)
    steps = int(round(duration / scene.dt))
    publish = max(int(round(0.05 / scene.dt)), 1)
    states = [scene.snapshot()]
    rows = []
    inputs = {key: [] for key in ("command", "external", "force_used", "pause", "saturation", "blocked", "p_ref_before", "r_ref_before", "guard_status")}
    twists = [_tcp_twist(scene)]
    history = np.concatenate([ctl.reference.v_lin, ctl.reference.v_ang]).copy()
    for i in range(steps):
        inputs["p_ref_before"].append(ctl.reference.p_ref.copy())
        inputs["r_ref_before"].append(ctl.reference.r_ref.copy())
        cached = ctl.wrench._cache
        inputs["force_used"].append(np.zeros(3) if cached is None else np.array(cached["compensated_wrench_tcp"][:3]))
        inputs["pause"].append(int(ctl._pause_advance))
        inputs["external"].append(np.zeros(6))
        if i % publish == 0 and guard.failure is None:
            ctl.set_command(twist, ctl.sim_time, ctl.sim_time + 0.1, ctl.phase)
        state, info = control_step(ctl, guard)
        rows.append(info)
        states.append(state)
        twists.append(_tcp_twist(scene))
        inputs["command"].append(np.asarray(info["twist_command"]))
        inputs["saturation"].append(int(bool(info.get("saturated_latched"))))
        inputs["blocked"].append(int(bool(info.get("blocked_now"))))
        inputs["guard_status"].append(str(info["status"]))
    assert len(rows) == steps
    if name in {"upper", "lower"}:
        assert guard.failure is None
        assert rows[0]["execution"] == "zero" and rows[0]["beta"] <= 1e-9
        outward = rows[0]["qdot_pred"][0] if name == "upper" else -rows[0]["qdot_pred"][0]
        assert outward <= 1e-9
    else:
        assert guard.failure is not None and guard.failure["reason"] == "singularity"
        assert guard.stopped_ok is True
        assert float(np.max(np.abs(np.vstack([row["dq"] for row in states])))) <= 0.5 + 1e-9
        if name == "cross":
            assert int(guard.failure["tick"]) > 0
    meta = {
        "dt": float(scene.model.opt.timestep),
        "iterations": int(scene.model.opt.iterations),
        "tolerance": float(scene.model.opt.tolerance),
        "variant": variant,
        "qpos0": qpos,
        "qvel0": qvel,
    }
    meta["tau_before"] = bias
    meta["first_fault_tick"] = None if guard.failure is None else guard.failure["tick"]
    _record_physical(case_id, scene, states, twists, rows, inputs, meta, extra={
        "tau_task": np.vstack([info["tau_task"] for info in rows]),
        "tau_null": np.vstack([info["tau_null"] for info in rows]),
        "tau_bias": np.vstack([info["tau_bias"] for info in rows]),
        "candidate": np.vstack([info["candidate_twist"] for info in rows]),
        "final_twist": np.vstack([info["twist_limited"] for info in rows]),
        "beta": np.array([float(info["beta"]) for info in rows]),
        "history_twist0": history,
        "fault_tick": np.int64(-1 if guard.failure is None else guard.failure["tick"]),
        "fault_reason": np.array("" if guard.failure is None else str(guard.failure["reason"])),
        "stopped_ok": np.int8(1 if guard.stopped_ok else 0),
    })
    return rows


@pytest.mark.parametrize("case_id", v6_trace_ids())
def test_v6_trace(case_id):
    _trace(case_id)


if os.environ.get("M2_V6_PACKAGE"):

    def test_v6_package_tampers_are_rejected():
        import json

        from feedingrobot.controllers.v3spec import ADVERSARIAL, evaluate_evidence

        package = Path(os.environ["M2_V6_PACKAGE"])
        before = evaluate_evidence(package, check_workspace=True)
        assert before["fixes_passed"] is True
        rows = run_adversarial(package)
        from feedingrobot.controllers.v3spec import ADVERSARIAL_EXPECTED

        expected = ADVERSARIAL_EXPECTED
        assert [row["name"] for row in rows] == list(ADVERSARIAL)
        for row in rows:
            assert row["fixes_passed"] is False
            assert expected[row["name"]] <= set(row["codes"])
            assert "clean_copy" not in row["codes"]
        again = evaluate_evidence(package, check_workspace=True)
        assert again["fixes_passed"] is True and again["reasons"] == []
        path = os.environ.get("M2_V6_RESULTS")
        if path:
            Path(path).write_text(json.dumps(rows, indent=2))

    def test_v6_copies_do_not_use_hardlinks(monkeypatch):
        from feedingrobot.controllers import v3spec

        package = Path(os.environ["M2_V6_PACKAGE"])
        before = v3spec._tree_id(package)
        original_copytree = shutil.copytree
        copies = []

        def inspect_copy(source, destination, *args, **kwargs):
            result = original_copytree(source, destination, *args, **kwargs)
            if Path(source).resolve() == package.resolve():
                dest = Path(destination)
                copies.append(dest)
                assert v3spec._tree_id(dest) == before
                for path in package.rglob("*"):
                    if path.is_file():
                        copied = dest / path.relative_to(package)
                        assert (path.stat().st_dev, path.stat().st_ino) != (copied.stat().st_dev, copied.stat().st_ino)
                # Exercise the real executor's copy, before its injected exception.
                (dest / "run.json").write_text("modified private copy")
                assert v3spec._tree_id(package) == before
            return result

        monkeypatch.setattr(v3spec.shutil, "copytree", inspect_copy)
        with pytest.raises(RuntimeError, match="^auto_resume$"):
            v3spec.run_adversarial(package, fail_after="auto_resume")
        assert len(copies) == 1 and not copies[0].exists()
        assert v3spec._tree_id(package) == before

    def test_v6_exception_leaves_the_source_package():
        package = Path(os.environ["M2_V6_PACKAGE"])
        before = {path.relative_to(package).as_posix(): path.read_bytes()[:0] or path.stat().st_mtime_ns for path in package.rglob("*") if path.is_file()}
        digest_before = __import__("hashlib").sha256()
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            digest_before.update(path.read_bytes())
        try:
            run_adversarial(package, fail_after="auto_resume")
        except RuntimeError as exc:
            assert str(exc) == "auto_resume"
        else:
            raise AssertionError("tamper injection did not raise")
        digest_after = __import__("hashlib").sha256()
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            digest_after.update(path.read_bytes())
        assert digest_before.digest() == digest_after.digest()
        del before

    def test_v6_concurrent_copies_stay_isolated():
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from feedingrobot.controllers.v3spec import _tree_id, apply_tamper, evaluate_evidence

        package = Path(os.environ["M2_V6_PACKAGE"])
        before = _tree_id(package)
        serial = evaluate_evidence(package, check_workspace=False)
        assert serial["fixes_passed"]
        for tamper_one in (False, True):
            barrier = threading.Barrier(2, timeout=60)
            def once(index):
                with tempfile.TemporaryDirectory(prefix="m2-v6-concurrent-") as work:
                    dest = Path(work) / "run"
                    shutil.copytree(package, dest, copy_function=shutil.copy2)
                    if tamper_one and index == 0:
                        apply_tamper("v6_tau_limit", dest)
                    barrier.wait()
                    return evaluate_evidence(dest, check_workspace=False)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(once, (0, 1)))
            assert results[1] == serial
            if tamper_one:
                assert results[0]["fixes_passed"] is False
                assert "torque_rate" in _codes(results[0]["reasons"])
            else:
                assert results[0] == serial
        assert _tree_id(package) == before
