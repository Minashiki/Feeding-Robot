"""V2 case grid. The same calls write evidence when M2_EVIDENCE_DIR is set."""

import json
import math

import mujoco
import numpy as np
import pytest

from feedingrobot.controllers.acceptance import f4_joint_case_ids, f5_case_ids, physical_case_ids
from feedingrobot.controllers.so3 import integrate_rotation, orientation_error
from feedingrobot.controllers.wrench import body_spatial
from feedingrobot.sim.contacts import min_distance
from feedingrobot.sim.scene import FeedingScene
from tests.m2_evidence import write_json, write_npz
from tests.m2_support import load_m2_config, place_arm, start_controller

PHASE = {"FREE": "TRANSPORT", "ACQUIRE": "ACQUIRE", "MOUTH": "APPROACH"}


def _limits(v=0.05, w=0.3, pos_dev=0.02, rot_dev=0.14):
    return {"v": v, "w": w, "a": 0.25, "alpha": 1.5, "pos_dev": pos_dev, "rot_dev": rot_dev}


@pytest.fixture(scope="module")
def rig():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    return scene, cfg, state, ctl, guard


def _fresh(rig):
    scene, cfg, state, ctl, guard = rig
    ctl.reset(state, np.zeros(7), "TRANSPORT")
    ctl.power_on = False
    ctl._tcp_pos = np.asarray(state["tcp_pos"], dtype=float)
    ctl._tcp_rot = np.asarray(state["tcp_mat"], dtype=float)
    guard.reset()
    return ctl


def _gains(ctl):
    return [float(ctl._k[0]), float(ctl._k[3]), float(ctl._d[0]), float(ctl._d[3])]


@pytest.mark.parametrize("case_id", f4_joint_case_ids())
def test_f4_joint_cell(case_id, rig):
    ctl = _fresh(rig)
    _tag, joint_s, side, motion = case_id.split("-")
    joint = int(joint_s[1:])
    q = np.array(rig[2]["q"], dtype=float)
    outward = 1.0 if side == "upper" else -1.0
    q[joint] = ctl.ranges[joint, 1] - 0.04 if side == "upper" else ctl.ranges[joint, 0] + 0.04
    jbar = np.zeros((7, 6))
    jbar[joint, 0] = 1.0
    shaper = ctl.reference
    shaper.anchor_pose(np.zeros(3), np.eye(3), q)
    if motion == "hist":
        shaper.v_lin[:] = np.array([outward * 0.05, 0.0, 0.0])
        command = np.array([-outward * 0.05, 0, 0, 0, 0, 0])
    elif motion == "in":
        command = np.array([-outward * 0.05, 0, 0, 0, 0, 0])
    else:
        command = np.array([outward * 0.05, 0, 0, 0, 0, 0])
    out = shaper.shape(command, np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, False, "run", project=lambda v: ctl._joint_scale(jbar, v, q))
    pred = float(out["v_hist"][0])
    outward_speed = max(0.0, pred * outward)
    inward_speed = max(0.0, -pred * outward)
    assert outward_speed <= 1e-9
    if motion == "in":
        assert inward_speed > 1e-6
    write_json(case_id, {"outward_speed": outward_speed, "inward_speed": inward_speed, "nodeid": f"tests/test_m2_v2.py::test_f4_joint_cell[{case_id}]"})


@pytest.mark.parametrize("case_id", ["F4-speed", "F4-pair", "F4-recover", "F4-scale-0", "F4-scale-0.1", "F4-scale-1"])
def test_f4_case(case_id, rig):
    cfg = rig[1]
    if case_id == "F4-speed":
        from feedingrobot.controllers.reference import ReferenceShaper

        shaper = ReferenceShaper(cfg)
        shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
        shaper.v_lin[:] = [0.05, 0.0, 0.0]
        rows = []
        for _ in range(2):
            rows.append(shaper.shape(np.array([0.05, 0, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(0.01, 0.1), 1.0, 1.0, True, False, "run"))
        assert all(np.linalg.norm(row["v_hist"][:3]) <= 0.01 + 1e-12 for row in rows)
        write_npz(case_id, t=np.array([0.001, 0.002]), v_hist=np.vstack([row["v_hist"] for row in rows]), p_ref=np.vstack([row["p_ref"] for row in rows]), limit=np.array(0.01))
        return
    ctl = _fresh(rig)
    if case_id == "F4-pair":
        q = np.array(rig[2]["q"], dtype=float)
        q[0] = ctl.ranges[0, 1] - 0.04
        q[1] = ctl.ranges[1, 1] - 0.04
        jbar = np.zeros((7, 6))
        jbar[0, 0] = 1.0
        jbar[1, 1] = 1.0
        shaper = ctl.reference
        shaper.anchor_pose(np.zeros(3), np.eye(3), q)
        out = shaper.shape(np.array([0.2, 0.2, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, False, "run", project=lambda v: ctl._joint_scale(jbar, v, q))
        pred = jbar @ out["v_hist"]
        outward = float(np.max(np.maximum(pred, 0.0)))
        assert outward <= 1e-9
        write_json(case_id, {"outward_speed": outward, "inward_speed": 0.0})
        return
    if case_id == "F4-recover":
        q = np.array(rig[2]["q"], dtype=float)
        q[0] = ctl.ranges[0, 1] - 0.04
        jbar = np.zeros((7, 6))
        jbar[0, 0] = 1.0
        shaper = ctl.reference
        shaper.anchor_pose(np.zeros(3), np.eye(3), q)
        shaper.v_lin[:] = [0.05, 0, 0]
        blocked = shaper.shape(np.array([-0.05, 0, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, False, "run", project=lambda v: ctl._joint_scale(jbar, v, q))
        assert blocked["v_hist"][0] <= 1e-9
        recovered = shaper.shape(np.array([-0.05, 0, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, False, "run", project=lambda v: ctl._joint_scale(jbar, v, q))
        inward = max(0.0, -float(recovered["v_hist"][0]))
        assert inward > 1e-6
        write_json(case_id, {"outward_speed": max(0.0, float(recovered["v_hist"][0])), "inward_speed": inward})
        return
    scale = float(case_id.split("-")[-1])
    from feedingrobot.controllers.reference import ReferenceShaper

    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    shaper.v_lin[:] = [0.05, 0, 0]
    out = shaper.shape(np.array([0.05, 0, 0, 0, 0, 0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, _limits(), scale, 1.0, True, False, "run")
    actual = float(np.linalg.norm(out["v_hist"][:3]))
    expected = 0.05 * scale
    assert actual <= expected + 1e-9
    write_json(case_id, {"speed_limit": expected, "actual_speed": actual})


@pytest.mark.parametrize("case_id", f5_case_ids())
def test_f5_blend(case_id, rig):
    _tag, src, dst, ms_s, dt_s = case_id.split("-")
    ms = int(ms_s)
    dt = float(dt_s)
    ctl = _fresh(rig)
    if src != "FREE":
        ctl.phase = PHASE[src]
        ctl._update_gains(0.2)
    ctl.phase = PHASE[dst]
    steps = max(int(round((ms / 1000.0) / dt)), 1)
    for _ in range(steps):
        ctl._update_gains(dt)
    start = np.array(_gains(ctl), dtype=float)
    ctl.phase = PHASE[src]
    target = np.array([ctl.gears[src]["kp"], ctl.gears[src]["kr"], ctl.gears[src]["dp"], ctl.gears[src]["dr"]], dtype=float)
    gains = [start]
    for _ in range(max(int(round(0.2 / dt)), 1)):
        ctl._update_gains(dt)
        gains.append(np.array(_gains(ctl), dtype=float))
    stacked = np.vstack(gains)
    allow = np.max(np.abs(target - start)) * dt / 0.2 + 1e-9
    assert np.max(np.abs(np.diff(stacked, axis=0))) <= allow + 1e-12
    write_json(case_id, {"gains": stacked, "k_start": start, "k_target": target, "dt": dt})


@pytest.mark.parametrize("case_id", ["F5-repeat"])
def test_f5_repeat(case_id, rig):
    ctl = _fresh(rig)
    ctl.phase = "APPROACH"
    ctl._update_gains(0.0)
    for i in range(40):
        ctl.set_command(np.zeros(6), ctl.now() + i * 0.005, ctl.now() + i * 0.005 + 0.05, "APPROACH")
        ctl._update_gains(0.005)
    assert ctl.transition_elapsed >= 0.2 - 1e-12
    assert ctl.active_gear == "MOUTH"
    assert ctl.transition_active is False
    write_json(case_id, {"finished": True})


@pytest.mark.parametrize("case_id", ["F5-return-limits"])
def test_f5_return_limits(case_id, rig):
    ctl = _fresh(rig)
    ctl.phase = "APPROACH"
    ctl._update_gains(0.05)
    ctl.phase = "TRANSPORT"
    ctl._update_gains(0.001)
    during = float(ctl.effective_limits()["v"])
    assert during <= 0.01 + 1e-12
    for _ in range(200):
        ctl._update_gains(0.001)
    after = float(ctl.effective_limits()["v"])
    assert abs(after - 0.05) < 1e-12
    assert abs(ctl._k[0] - 300.0) < 1e-9
    write_json(case_id, {"limit_during": during, "limit_after": after})


def _shape_deviation(cfg, pos, rot, execution, saturated=False, limits=None):
    from feedingrobot.controllers.reference import ReferenceShaper

    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    limits = _limits() if limits is None else limits
    return shaper.shape(np.zeros(6), pos, rot, np.zeros(3), 0.001, limits, 1.0, 1.0, True, saturated, execution)


@pytest.mark.parametrize("case_id", ["F6-trans", "F6-rot", "F6-both", "F6-within", "F6-pause1000", "F6-prohibit", "F6-stop", "F6-power-on"])
def test_f6_case(case_id, rig):
    cfg = rig[1]
    if case_id == "F6-trans":
        out = _shape_deviation(cfg, np.array([0.03, 0.0, 0.0]), np.eye(3), "run", True)
        assert np.max(np.abs(out["p_ref"] - np.array([0.01, 0, 0]))) <= 1e-9
        assert np.linalg.norm(out["v_ref"]) == 0.0
        write_json(case_id, {"p_ref": out["p_ref"], "correction": out["reference_correction_pos"], "v_ref": 0.0})
        return
    if case_id == "F6-within":
        out = _shape_deviation(cfg, np.array([0.005, 0, 0]), np.eye(3), "run", True)
        assert np.linalg.norm(out["p_ref"]) <= 1e-12
        write_json(case_id, {"error": 0.005, "limit": 0.02, "p_ref": out["p_ref"]})
        return
    if case_id == "F6-pause1000":
        from feedingrobot.controllers.reference import ReferenceShaper

        shaper = ReferenceShaper(cfg)
        shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
        worst = 0.0
        for i in range(1000):
            pos = np.array([0.03 * math.sin(i / 50.0), 0.0, 0.0])
            out = shaper.shape(np.zeros(6), pos, np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, True, "run")
            worst = max(worst, float(np.linalg.norm(out["p_ref"] - pos)))
            assert np.linalg.norm(out["v_ref"]) == 0.0
        assert worst <= 0.02 + 1e-9
        write_json(case_id, {"error": worst, "limit": 0.02})
        return
    if case_id in {"F6-rot", "F6-both"}:
        rot_ref = integrate_rotation(np.eye(3), np.array([1.0, 0.0, 0.0]), 0.4)
        from feedingrobot.controllers.reference import ReferenceShaper

        shaper = ReferenceShaper(cfg)
        shaper.anchor_pose(np.array([0.03, 0, 0]) if case_id == "F6-both" else np.zeros(3), rot_ref, np.zeros(7))
        limits = _limits(rot_dev=0.1)
        pos = np.array([0.03, 0, 0]) if case_id == "F6-rot" else np.zeros(3)
        out = shaper.shape(np.zeros(6), pos, np.eye(3), np.zeros(3), 0.001, limits, 1.0, 1.0, True, True, "run")
        ang = float(np.linalg.norm(orientation_error(out["r_ref"], np.eye(3))))
        assert ang <= 0.1 + 1e-9
        perr = float(np.linalg.norm(out["p_ref"] - pos))
        assert perr <= 0.02 + 1e-9
        if case_id == "F6-rot":
            write_json(case_id, {"error": ang, "limit": 0.1})
        else:
            write_json(case_id, {"error": perr, "limit": 0.02, "rot_error": ang, "rot_limit": 0.1})
        return
    execution = {"F6-prohibit": "prohibit", "F6-stop": "stop", "F6-power-on": "power_on"}[case_id]
    out = _shape_deviation(cfg, np.array([0.03, 0, 0]), np.eye(3), execution)
    err = float(np.linalg.norm(out["p_ref"] - np.array([0.03, 0, 0])))
    assert err <= 0.02 + 1e-9
    assert np.linalg.norm(out["v_ref"]) == 0.0
    write_json(case_id, {"error": err, "limit": 0.02})


@pytest.mark.parametrize("case_id", ["F2-com-0", "F2-com-1", "F2-com-2"])
def test_f2_pose(case_id, rig):
    scene = rig[0]
    pose = int(case_id[-1])
    place_arm(scene, scene.config["q_torque_poses"][pose])
    body = scene.index.tool_body_id
    dq = np.linspace(-0.2, 0.3, scene.model.nv)
    scene.data.qvel[:] = dq
    mujoco.mj_forward(scene.model, scene.data)
    jacp = np.zeros((3, scene.model.nv))
    jacr = np.zeros((3, scene.model.nv))
    mujoco.mj_jacBodyCom(scene.model, scene.data, jacp, jacr, body)
    omega, v_com, _alpha, a_com = body_spatial(scene.model, scene.data, body)
    vel_error = float(max(np.max(np.abs(v_com - jacp @ dq)), np.max(np.abs(omega - jacr @ dq))))
    qpos = scene.data.qpos.copy()
    eps = 1e-6
    mujoco.mj_integratePos(scene.model, scene.data.qpos, scene.data.qvel, eps)
    mujoco.mj_forward(scene.model, scene.data)
    _o2, v2, _a2, _acc2 = body_spatial(scene.model, scene.data, body)
    scene.data.qpos[:] = qpos
    mujoco.mj_forward(scene.model, scene.data)
    convective = (v2 - v_com) / eps
    qacc = np.array(scene.data.qacc, dtype=float)
    predicted = jacp @ qacc + convective
    acc_error = float(np.max(np.abs(predicted - a_com)))
    assert vel_error <= 1e-8
    assert acc_error <= 1e-5
    write_json(case_id, {"vel_error": vel_error, "acc_error": acc_error})


@pytest.mark.parametrize("case_id", ["F3-pulse-A", "F3-pulse-B"])
def test_f3_pulse(case_id, rig):
    from feedingrobot.controllers.cartesian_impedance import control_step

    scene, cfg, _state, _ctl, _guard = rig
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    if case_id.endswith("B"):
        scene.model.opt.timestep = 0.0005
        scene.dt = 0.0005
    ctl, guard = start_controller(scene, cfg, state)
    ctl.power_on = False
    for _ in range(20):
        ctl.set_command(np.zeros(6), ctl.sim_time, ctl.sim_time + 0.1, "TRANSPORT")
        control_step(ctl, guard)
    assert guard.failure is None
    updates = ctl.wrench.filter_updates
    scene.set_external_wrench([9.0, 0, 0], [0, 0, 0], scene.snapshot()["tcp_pos"])
    nxt, info = control_step(ctl, guard)
    assert guard.failure is not None and guard.failure["reason"] == "wrench"
    assert guard.failure["tick"] == nxt["tick"] == info["ft_sample_tick"]
    scene.clear_external_wrench()
    _nxt2, info2 = control_step(ctl, guard)
    assert info2["control_mode"] == "STOPPING"
    assert ctl.wrench.filter_updates == updates + 2
    write_json(case_id, {"reason": "wrench", "fault_tick": int(nxt["tick"]), "sample_tick": int(info["ft_sample_tick"]), "next_mode": "STOPPING", "filter_delta": 2})
    scene.restore_model_options()


@pytest.mark.parametrize("case_id", ["F7-commands"])
def test_f7_commands(case_id, rig):
    scene, cfg, state, _ctl, _guard = rig
    ctl, guard = start_controller(scene, cfg, state)
    flags = {"nonfinite": True, "stale": False, "conflict": False, "idempotent": False}
    for bad in (math.nan, math.inf, -math.inf):
        fresh, fresh_guard = start_controller(scene, cfg, state)
        rejected = fresh.set_command(np.zeros(6), bad, 0.05, "TRANSPORT")
        flags["nonfinite"] = flags["nonfinite"] and rejected["accepted"] is False and fresh_guard.status == "ABORTED" and fresh._command is None
    now = ctl.now()
    ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), now, now + 0.05, "TRANSPORT")
    flags["stale"] = ctl.set_command(np.zeros(6), now - 0.01, now + 0.04, "TRANSPORT")["rejected"] == "stale"
    flags["conflict"] = ctl.set_command(np.ones(6), now, now + 0.05, "ACQUIRE")["rejected"] == "conflict"
    flags["idempotent"] = ctl.set_command(np.array([0.01, 0, 0, 0, 0, 0]), now, now + 0.05, "TRANSPORT").get("idempotent") is True
    assert all(flags.values())
    write_json(case_id, flags)


@pytest.mark.parametrize("case_id", ["reset-100"])
def test_reset_pollution(case_id, rig):
    _scene, _cfg, state, ctl, guard = rig
    for _ in range(100):
        ctl.transition_active = True
        ctl.transition_elapsed = 0.01
        ctl._k[:] = 1.0
        ctl.reference.v_lin[:] = 0.04
        ctl.reference.reference_correction_pos[:] = 0.01
        ctl._pause_advance = True
        ctl.wrench._cache = {"compensated_wrench_tcp": np.ones(6)}
        ctl.reset(state, np.zeros(7), "TRANSPORT")
        guard.reset()
        assert ctl.transition_active is False
        assert np.linalg.norm(ctl.reference.v_lin) == 0.0
        assert ctl._pause_advance is False
        assert np.linalg.norm(ctl.reference.reference_correction_pos) == 0.0
        assert not np.allclose(ctl.wrench._cache["compensated_wrench_tcp"], np.ones(6))
    write_json(case_id, {"clean": True, "n": 100})


@pytest.mark.parametrize("case_id", ["replay-s0", "replay-s1", "replay-s2"])
def test_replay(case_id, rig):
    from feedingrobot.controllers.cartesian_impedance import control_step

    scene, cfg, _state, _ctl, _guard = rig
    seed = int(case_id[-1])
    scene.restore_model_options()

    def once():
        state = place_arm(scene, scene.config["q_torque_poses"][seed], seed=seed)
        ctl, guard = start_controller(scene, cfg, state)
        ctl.power_on = False
        last = state
        for i in range(50):
            if i % 25 == 0:
                ctl.set_command(np.zeros(6), ctl.sim_time, ctl.sim_time + 0.1, "TRANSPORT")
            last, info = control_step(ctl, guard)
        wrench = np.asarray(info.get("ft_compensated_wrench_tcp", np.zeros(6)), dtype=float)
        return last["q"], last["tcp_pos"], wrench

    q1, p1, w1 = once()
    q2, p2, w2 = once()
    q_error = float(np.max(np.abs(q1 - q2)))
    tcp_error = float(np.max(np.abs(p1 - p2)))
    wrench_error = float(np.max(np.abs(w1 - w2)))
    assert q_error <= 1e-9 and tcp_error <= 1e-9 and wrench_error <= 1e-7
    write_json(case_id, {"q_error": q_error, "tcp_error": tcp_error, "wrench_error": wrench_error})


def json_scene_spring():
    with open("configs/m1_scene.json") as handle:
        cfg = json.load(handle)
    cfg["model"] = "assets/tests/m2_spring_surface.xml"
    return cfg


def _apply_variant(scene, variant):
    scene.restore_model_options()
    if variant == "B":
        scene.model.opt.timestep = 0.0005
        scene.dt = 0.0005
    elif variant == "C":
        scene.model.opt.iterations = int(scene._opt_baseline["iterations"] * 2)
        scene.model.opt.tolerance = float(scene._opt_baseline["tolerance"] * 0.1)


def _record_physical(case_id, samples, dt):
    state_rows = [row[0] for row in samples]
    info_rows = [row[1] for row in samples]
    t = np.array([row["episode_time"] for row in state_rows], dtype=float)
    dq = np.vstack([row["dq"] for row in state_rows])
    tau = np.vstack([row["tau_command"] for row in state_rows])
    tcp = np.vstack([row["tcp_pos"] for row in state_rows])
    pref = np.vstack([np.asarray(info["p_ref"], dtype=float) for info in info_rows])
    vref = np.vstack([np.asarray(info["v_ref"], dtype=float) for info in info_rows])
    wrench = np.vstack([np.asarray(info.get("ft_compensated_wrench_tcp", np.zeros(6)), dtype=float) for info in info_rows])
    contact = np.array([sum(float(np.linalg.norm(item["force_contact"])) for item in row["contacts"]) for row in state_rows])
    dist = np.array([min_distance(row["contacts"]) for row in state_rows])
    gains = np.vstack([np.asarray(info["k"], dtype=float) for info in info_rows])
    write_npz(case_id, t=t, dq=dq, tau=tau, tcp=tcp, p_ref=pref, v_ref=vref, wrench=wrench, contact=contact, min_dist=dist, k=gains, dt=np.array(dt), tau0=np.asarray(info_rows[0].get("tau_before", np.zeros(7)), dtype=float))
    return t, dq, tau, tcp, pref


@pytest.mark.parametrize("case_id", physical_case_ids())
def test_physical(case_id):
    family, seed_s, variant = case_id.split("-")
    seed = int(seed_s[1:])
    cfg = load_m2_config()
    if family == "gear":
        scene_cfg = json_scene_spring()
        cfg = dict(cfg)
        cfg["press_test"] = True
    else:
        scene_cfg = "configs/m1_scene.json"
    scene = FeedingScene(scene_cfg)
    pose = seed if family == "pause" else 0
    state = place_arm(scene, scene.config["q_torque_poses"][pose], seed=seed)
    _apply_variant(scene, variant)
    bias = np.array(scene.data.qfrc_bias[scene.index.arm_dof_adr], dtype=float)
    ctl, guard = start_controller(scene, cfg, state, tau=np.zeros(7) if family == "takeover" else bias)
    ctl.power_on = family == "takeover"
    origin = state["tcp_pos"].copy()
    release_at = None
    release_pos = None

    def twist(t, controller):
        if family == "release" and 0.02 <= t < 0.08:
            return np.array([0.05, 0, 0, 0, 0, 0])
        if family == "gear":
            mode = getattr(controller, "_gear_phase", "approach")
            if mode == "backoff":
                return np.array([0.0, 0.01, 0, 0, 0, 0])
            if mode == "switched":
                return np.zeros(6)
            return np.array([0.0, -0.02, 0, 0, 0, 0])
        return np.zeros(6)

    def hook(i, t, controller):
        nonlocal release_at, release_pos
        scene.clear_external_wrench()
        if family == "pause" and 0.05 <= t < 0.08:
            controller._pause_advance = True
            scene.set_external_wrench([4.0, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
        elif family == "release" and 0.04 <= t < 0.1:
            scene.set_external_wrench([-5.0, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
        elif family == "release" and t >= 0.1 and release_at is None:
            release_at = t
            release_pos = controller.scene.snapshot()["tcp_pos"].copy()
        elif family == "wrench" and 0.05 <= t < 0.15:
            scene.set_external_wrench([1.0, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
        elif family == "gear":
            contacted = any("spring_pad" in {row.get("geom1"), row.get("geom2")} for row in controller.scene.snapshot()["contacts"])
            if contacted and getattr(controller, "_gear_phase", "approach") == "approach":
                controller._gear_phase = "backoff"
                controller._gear_t = t
            if getattr(controller, "_gear_phase", "") == "backoff" and t >= controller._gear_t + 0.08:
                controller._gear_phase = "switched"
                controller.phase = "ACQUIRE"
                controller.set_command(np.zeros(6), controller.sim_time, controller.sim_time + 0.1, "ACQUIRE")

    duration = {"pause": 0.3, "release": 1.2, "gear": 1.2, "wrench": 0.3, "takeover": 0.3}[family]
    samples = []
    dt = scene.dt
    steps = int(round(duration / dt))
    publish = max(int(round(0.05 / dt)), 1)
    from feedingrobot.controllers.cartesian_impedance import control_step

    for i in range(steps):
        hook(i, ctl.sim_time, ctl)
        if i % publish == 0 and guard.status == "RUNNING" and not ctl.power_on:
            ctl.set_command(twist(ctl.sim_time, ctl), ctl.sim_time, ctl.sim_time + 0.1, ctl.phase)
        state, info = control_step(ctl, guard)
        if "p_ref" not in info:
            break
        samples.append((state, info))
        if guard.status == "ABORTED":
            break
    assert guard.failure is None, guard.failure
    assert len(samples) >= 2
    t, dq, tau, tcp, pref = _record_physical(case_id, samples, dt)
    assert np.max(np.abs(dq)) <= 0.5 + 1e-9
    assert np.max(np.abs(np.diff(tau, axis=0))) <= 2000.0 * dt + 1e-9
    assert np.max(np.linalg.norm(pref - tcp, axis=1)) <= 0.02 + 1e-9
    if family == "gear":
        assert any("spring_pad" in {c["geom1"], c["geom2"]} for row in samples for c in row[0]["contacts"])
    if family == "release":
        assert release_at is not None
        speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.diff(t)
        speed_t = t[1:]
        window = (speed_t >= release_at) & (speed_t <= release_at + 0.5)
        assert np.max(speed[window]) <= 0.03 + 1e-9
        extra = tcp[(t >= release_at) & (t <= release_at + 0.5), 0] - release_pos[0]
        assert np.max(extra) <= 0.005 + 1e-9
        stable = (speed_t >= release_at + 0.8) & (speed_t <= release_at + 1.0)
        assert np.max(speed[stable]) <= 0.03 + 1e-9
    del origin
