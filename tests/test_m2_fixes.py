"""Targeted checks for the seven M2 fix items."""

import math

import mujoco
import numpy as np

from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.controllers.wrench import body_spatial
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, run_commands, start_controller


def _limits(v=0.05, w=0.3):
    return {"v": v, "w": w, "a": 0.25, "alpha": 1.5, "pos_dev": 0.02, "rot_dev": 0.14}


def test_f4_new_speed_bound_projects_history():
    cfg = load_m2_config()
    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    shaper.v_lin[:] = [0.05, 0.0, 0.0]
    out = shaper.shape(np.array([0.05, 0, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(0.01, 0.1), 1.0, 1.0, True, False, "run")
    assert np.linalg.norm(out["v_hist"][:3]) <= 0.01 + 1e-12
    assert out["constraint_reset"] == "linear_speed"
    shaper.v_ang[:] = [0.2, 0.2, 0.0]
    out = shaper.shape(np.array([0, 0, 0, 0.2, 0.2, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(0.05, 0.1), 1.0, 1.0, True, False, "run")
    assert np.linalg.norm(out["v_hist"][3:]) <= 0.1 + 1e-12


def test_f4_prohibit_clears_reference_velocity():
    cfg = load_m2_config()
    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    shaper.v_lin[:] = [0.05, 0, 0]
    out = shaper.shape(np.ones(6), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), 0.0, 1.0, True, False, "run")
    assert np.linalg.norm(out["v_hist"]) == 0.0
    assert np.linalg.norm(shaper.p_ref) <= 1e-15


def test_f5_interrupted_blend_stays_continuous():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, _guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    ctl.phase = "APPROACH"
    ctl._tcp_pos = state["tcp_pos"]
    ctl._tcp_rot = state["tcp_mat"]
    ctl._update_gains(0.05)
    mid = ctl._k.copy()
    assert abs(mid[0] - (0.75 * 300 + 0.25 * 100)) < 1e-9
    ctl.phase = "ACQUIRE"
    before = ctl._k.copy()
    ctl._update_gains(0.001)
    step = np.max(np.abs(ctl._k - before))
    span = np.max(np.abs(ctl._vector_gain("ACQUIRE") - before))
    assert step <= span * 0.001 / 0.2 + 1e-9
    ctl.phase = "TRANSPORT"
    for _ in range(250):
        ctl._update_gains(0.001)
    assert abs(ctl.effective_limits()["v"] - 0.05) < 1e-12
    assert abs(ctl._k[0] - 300) < 1e-9


def test_f5_repeat_command_does_not_restart():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, _guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    ctl.phase = "APPROACH"
    ctl._tcp_pos = state["tcp_pos"]
    ctl._tcp_rot = state["tcp_mat"]
    ctl._update_gains(0.0)
    for _ in range(40):
        ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.05, "APPROACH")
        ctl._update_gains(0.005)
    assert ctl.transition_elapsed >= 0.2 - 1e-12
    assert ctl.active_gear == "MOUTH"


def test_f7_rejects_nonfinite_and_keeps_the_live_command():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    ok = ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + 0.05, "TRANSPORT")
    assert ok["accepted"]
    for bad in (math.nan, math.inf, -math.inf):
        fresh = start_controller(scene, cfg, state)[0]
        fresh.guard = guard
        guard.reset()
        rejected = fresh.set_command(np.zeros(6), bad, 0.05, "TRANSPORT")
        assert rejected["accepted"] is False
        assert fresh.guard.status == "ABORTED"
        assert fresh._command is None
    ctl2, guard2 = start_controller(scene, cfg, state)
    now = ctl2.now()
    ctl2.set_command(np.array([0.01, 0, 0, 0, 0, 0]), now, now + 0.05, "TRANSPORT")
    stale = ctl2.set_command(np.zeros(6), now - 0.01, now + 0.04, "TRANSPORT")
    assert stale["rejected"] == "stale"
    clash = ctl2.set_command(np.ones(6), now, now + 0.05, "ACQUIRE")
    assert clash["rejected"] == "conflict"
    assert ctl2._command["phase"] == "TRANSPORT"
    again = ctl2.set_command(np.array([0.01, 0, 0, 0, 0, 0]), now, now + 0.05, "TRANSPORT")
    assert again.get("idempotent") is True


def test_f2_com_velocity_matches_jacobian():
    scene = FeedingScene("configs/m1_scene.json")
    body = scene.index.tool_body_id
    jacp = np.zeros((3, scene.model.nv))
    jacr = np.zeros((3, scene.model.nv))
    for q in scene.config["q_torque_poses"]:
        place_arm(scene, q)
        dq = np.linspace(-0.3, 0.4, scene.model.nv)
        scene.data.qvel[:] = dq
        mujoco.mj_forward(scene.model, scene.data)
        mujoco.mj_jacBodyCom(scene.model, scene.data, jacp, jacr, body)
        omega, v_com, _alpha, _a = body_spatial(scene.model, scene.data, body)
        assert np.max(np.abs(v_com - jacp @ dq)) <= 1e-8
        assert np.max(np.abs(omega - jacr @ dq)) <= 1e-8


def test_f3_force_pulse_latches_on_the_post_step_sample():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    run_commands(ctl, guard, 0.4, lambda t, c: np.zeros(6))
    assert guard.failure is None
    point = scene.snapshot()["tcp_pos"]
    updates = ctl.wrench.filter_updates
    scene.set_external_wrench([9.0, 0.0, 0.0], [0.0, 0.0, 0.0], point)
    from feedingrobot.controllers.cartesian_impedance import control_step

    nxt, info = control_step(ctl, guard)
    assert guard.failure is not None
    assert guard.failure["reason"] == "wrench"
    assert guard.failure["tick"] == nxt["tick"]
    assert info["ft_sample_tick"] == nxt["tick"]
    scene.clear_external_wrench()
    _nxt2, info2 = control_step(ctl, guard)
    assert info2["control_mode"] == "STOPPING"
    assert guard.failure["tick"] == nxt["tick"]
    assert ctl.wrench.filter_updates == updates + 2


def test_v2_f5_return_to_free_is_continuous():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, _guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    ctl.phase = "APPROACH"
    ctl._tcp_pos = state["tcp_pos"]
    ctl._tcp_rot = state["tcp_mat"]
    ctl._update_gains(0.05)
    assert abs(ctl._k[0] - 250.0) < 1e-9
    ctl.phase = "TRANSPORT"
    ctl._update_gains(0.001)
    assert abs(ctl._k[0] - 250.25) < 1e-9
    ctl2, _g2 = start_controller(scene, cfg, state)
    ctl2.power_on = False
    ctl2.phase = "APPROACH"
    ctl2._tcp_pos = state["tcp_pos"]
    ctl2._tcp_rot = state["tcp_mat"]
    ctl2._update_gains(0.05)
    ctl2.phase = "TRANSPORT"
    ctl2._update_gains(0.0005)
    assert abs(ctl2._k[0] - 250.125) < 1e-9


def test_v2_f6_pause_still_limits_deviation():
    cfg = load_m2_config()
    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    limits = _limits()
    limits["pos_dev"] = 0.02
    out = shaper.shape(np.zeros(6), np.array([0.03, 0.0, 0.0]), np.eye(3), np.zeros(3), 0.001, limits, 1.0, 1.0, True, True, "run")
    assert np.max(np.abs(out["p_ref"] - np.array([0.01, 0.0, 0.0]))) <= 1e-9
    assert np.linalg.norm(out["p_ref"] - np.array([0.03, 0.0, 0.0])) <= 0.02 + 1e-9
    assert np.max(np.abs(out["reference_correction_pos"] - np.array([0.01, 0.0, 0.0]))) <= 1e-9
    assert np.linalg.norm(out["v_ref"]) == 0.0
    assert np.linalg.norm(out["v_hist"]) == 0.0
    held = ReferenceShaper(cfg)
    held.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    small = held.shape(np.zeros(6), np.array([0.005, 0, 0]), np.eye(3), np.zeros(3), 0.001, limits, 1.0, 1.0, True, True, "run")
    assert np.linalg.norm(small["p_ref"]) <= 1e-12


def test_v2_f4_final_candidate_respects_joint_limit():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    q = np.array(scene.config["q_torque_poses"][0], dtype=float)
    q[0] = float(scene.model.jnt_range[scene.index.arm_joint_ids[0], 1] - 0.04)
    state = place_arm(scene, q)
    ctl, _guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    jac, _bias = ctl._jacobian()
    mass = ctl._mass()
    scale = np.diag([1, 1, 1, 0.1, 0.1, 0.1])
    _null, jbar, _lam = __import__("feedingrobot.controllers.cartesian_impedance", fromlist=["damped_nullspace"]).damped_nullspace(scale @ jac, mass, 1e-8, 1e-4)
    outward = np.zeros(6)
    inward = np.zeros(6)
    for trial in (np.eye(3)[i] for i in range(3)):
        for sign in (1.0, -1.0):
            twist = np.zeros(6)
            twist[:3] = sign * trial
            pred = jbar @ (scale @ twist)
            if pred[0] > 0.01:
                outward = twist
            if pred[0] < -0.01:
                inward = twist
    assert outward[0] != 0.0 or np.any(outward)
    shaper = ctl.reference
    shaper.v_lin[:] = outward[:3] * 0.05
    raw = jbar @ (scale @ np.concatenate([shaper.v_lin, shaper.v_ang]))
    assert raw[0] > 1e-6
    out = shaper.shape(inward, state["tcp_pos"], state["tcp_mat"], np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, False, "run", project=lambda v: ctl._joint_scale(jbar, v, q))
    final = jbar @ (scale @ out["v_hist"])
    assert final[0] <= 1e-9


def test_f6_commit_uses_applied_torque_and_pauses():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    ctl._last_tau_raw = np.full(7, 50.0)
    ctl._tau_before = np.zeros(7)
    ctl._tcp_pos = state["tcp_pos"]
    ctl._tcp_rot = state["tcp_mat"]
    before = ctl.tau_prev.copy()
    ctl.commit_applied(np.full(7, 2.0), 0.001, was_power_on=False)
    assert np.allclose(ctl.tau_prev, 2.0)
    assert ctl._pause_advance is True
    assert not np.allclose(ctl.tau_prev, before)
    ctl._pause_advance = False
    ctl._sat_union = 0.0
    preview = ctl.tau_prev.copy()
    ctl.compute(scene.snapshot(), 0.001)
    assert np.allclose(ctl.tau_prev, preview)
