"""V2 case grid. The same calls write evidence when M2_EVIDENCE_DIR is set."""

import json
import math

import mujoco
import numpy as np
import pytest

from feedingrobot.controllers.acceptance import f4_joint_case_ids, f5_case_ids, physical_case_ids
from feedingrobot.controllers.so3 import integrate_rotation, orientation_error
from feedingrobot.controllers.wrench import body_spatial
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
    assert abs(ctl._k[0] - 330.0) < 1e-9
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
        write_json(
            case_id,
            {
                "pos": [0.03, 0.0, 0.0],
                "p_ref": out["p_ref"],
                "correction": out["reference_correction_pos"],
                "v_ref": out["v_ref"],
                "limit": 0.02,
            },
        )
        return
    if case_id == "F6-within":
        pos = np.array([0.005, 0.0, 0.0])
        out = _shape_deviation(cfg, pos, np.eye(3), "run", True)
        assert np.linalg.norm(out["p_ref"]) <= 1e-12
        write_json(case_id, {"pos": pos, "p_ref": out["p_ref"], "v_ref": out["v_ref"], "limit": 0.02})
        return
    if case_id == "F6-pause1000":
        from feedingrobot.controllers.reference import ReferenceShaper

        shaper = ReferenceShaper(cfg)
        shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
        pos_rows = []
        ref_rows = []
        vel_rows = []
        for i in range(1000):
            pos = np.array([0.03 * math.sin(i / 50.0), 0.0, 0.0])
            out = shaper.shape(np.zeros(6), pos, np.eye(3), np.zeros(3), 0.001, _limits(), 1.0, 1.0, True, True, "run")
            err = float(np.linalg.norm(out["p_ref"] - pos))
            assert err <= 0.02 + 1e-9
            assert np.linalg.norm(out["v_ref"]) == 0.0
            pos_rows.append(pos)
            ref_rows.append(out["p_ref"])
            vel_rows.append(out["v_ref"])
        write_npz(
            case_id,
            pos=np.vstack(pos_rows),
            p_ref=np.vstack(ref_rows),
            v_ref=np.vstack(vel_rows),
            integral=np.zeros((1000, 3)),
            limit=np.array(0.02),
        )
        return
    if case_id in {"F6-rot", "F6-both"}:
        rot_ref = integrate_rotation(np.eye(3), np.array([1.0, 0.0, 0.0]), 0.4)
        from feedingrobot.controllers.reference import ReferenceShaper

        anchor = np.array([0.03, 0, 0]) if case_id == "F6-both" else np.zeros(3)
        shaper = ReferenceShaper(cfg)
        shaper.anchor_pose(anchor, rot_ref, np.zeros(7))
        limits = _limits(rot_dev=0.1)
        pos = np.array([0.03, 0, 0]) if case_id == "F6-rot" else np.zeros(3)
        out = shaper.shape(np.zeros(6), pos, np.eye(3), np.zeros(3), 0.001, limits, 1.0, 1.0, True, True, "run")
        ang = float(np.linalg.norm(orientation_error(out["r_ref"], np.eye(3))))
        assert ang <= 0.1 + 1e-9
        perr = float(np.linalg.norm(out["p_ref"] - pos))
        assert perr <= 0.02 + 1e-9
        write_json(
            case_id,
            {
                "pos": pos,
                "p_ref": out["p_ref"],
                "rot": np.eye(3),
                "r_ref": out["r_ref"],
                "v_ref": out["v_ref"],
                "limit": 0.1 if case_id == "F6-rot" else 0.02,
            },
        )
        return
    execution = {"F6-prohibit": "prohibit", "F6-stop": "stop", "F6-power-on": "power_on"}[case_id]
    pos = np.array([0.03, 0, 0])
    out = _shape_deviation(cfg, pos, np.eye(3), execution)
    err = float(np.linalg.norm(out["p_ref"] - pos))
    assert err <= 0.02 + 1e-9
    assert np.linalg.norm(out["v_ref"]) == 0.0
    write_json(case_id, {"pos": pos, "p_ref": out["p_ref"], "v_ref": out["v_ref"], "execution": execution, "limit": 0.02, "integral": [0.0, 0.0, 0.0]})


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


def _tcp_twist(scene):
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(scene.model, scene.data, mujoco.mjtObj.mjOBJ_SITE, int(scene.index.site_ids["tcp"]), vel, 0)
    return np.concatenate([vel[3:], vel[:3]])


def _bowl_force(contacts):
    total = np.zeros(3)
    bowls = {"bowl_bottom", "bowl_back", "bowl_left", "bowl_right", "bowl_front"}
    for row in contacts:
        names = {row.get("geom1"), row.get("geom2")}
        if "spring_pad" not in names or not (names & bowls):
            continue
        if row.get("geom1") in bowls:
            total += np.asarray(row["force_on_geom1_world"], dtype=float)
        else:
            total += np.asarray(row["force_on_geom2_world"], dtype=float)
    return total


def _record_physical(case_id, scene, states, twists, infos, inputs, meta, extra=None):
    n = len(infos)
    forces = []
    torques = []
    points = []
    dists = []
    geom1 = []
    geom2 = []
    frames = []
    offsets = [0]
    for state, info in zip(states[1:], infos):
        for row in state["contacts"]:
            force = np.asarray(row["force_on_geom1_world"], dtype=float)
            point = np.asarray(row["pos"], dtype=float)
            geom1.append(str(row["geom1"]))
            geom2.append(str(row["geom2"]))
            forces.append(force)
            points.append(point)
            torques.append(np.cross(point - state["tcp_pos"], force))
            dists.append(float(row["dist"]))
            frames.append(np.asarray(row["frame"], dtype=float).reshape(-1))
        offsets.append(len(forces))
    empty3 = np.zeros((0, 3))
    write_npz(
        case_id,
        t_state=np.array([row["episode_time"] for row in states], dtype=float),
        tick=np.array([row["tick"] for row in states], dtype=int),
        q=np.vstack([row["q"] for row in states]),
        dq=np.vstack([row["dq"] for row in states]),
        tcp_pos=np.vstack([row["tcp_pos"] for row in states]),
        tcp_rot=np.stack([np.asarray(row["tcp_mat"], dtype=float).reshape(3, 3) for row in states]),
        tcp_twist=np.vstack(twists),
        tau_applied=np.vstack([row["tau_command"] for row in states[1:]]),
        tau_raw=np.vstack([np.asarray(info["tau_raw"], dtype=float) for info in infos]),
        tau_before=np.asarray(meta["tau_before"], dtype=float),
        qpos0=np.asarray(meta["qpos0"], dtype=float),
        qvel0=np.asarray(meta["qvel0"], dtype=float),
        command=np.vstack(inputs["command"]),
        p_ref=np.vstack([np.asarray(info["p_ref"], dtype=float) for info in infos]),
        p_ref_before=np.vstack(inputs["p_ref_before"]),
        r_ref=np.stack([np.asarray(info["r_ref"], dtype=float).reshape(3, 3) for info in infos]),
        r_ref_before=np.stack(inputs["r_ref_before"]),
        reference_correction_pos=np.vstack([np.asarray(info["reference_correction_pos"], dtype=float) for info in infos]),
        v_ref=np.vstack([np.asarray(info["v_ref"], dtype=float) for info in infos]),
        K=np.vstack([np.asarray(info["k"], dtype=float) for info in infos]),
        D=np.vstack([np.asarray(info["d"], dtype=float) for info in infos]),
        phase=np.array([str(info["phase_k"]) for info in infos]),
        mode=np.array([str(info["control_mode"]) for info in infos]),
        execution=np.array([str(info["execution"]) for info in infos]),
        active_gear=np.array([str(info["active_gear"]) for info in infos]),
        target_gear=np.array([str(info["target_gear"]) for info in infos]),
        guard_status=np.array(inputs["guard_status"]),
        wrench_raw=np.vstack([np.asarray(info["ft_raw_wrench_sensor"], dtype=float) for info in infos]),
        wrench_compensated=np.vstack([np.asarray(info["ft_compensated_wrench_tcp"], dtype=float) for info in infos]),
        external_wrench=np.vstack(inputs["external"]),
        force_used=np.vstack(inputs["force_used"]),
        pause_applied=np.asarray(inputs["pause"], dtype=np.int8),
        saturation_flags=np.asarray(inputs["saturation"], dtype=np.int8),
        blocked_now=np.asarray(inputs["blocked"], dtype=np.int8),
        block_progress=np.array([info["block_progress"] for info in infos]),
        block_correction_pos=np.array([info["block_correction_pos"] for info in infos]),
        transition_active=np.asarray([1 if info["transition_active"] else 0 for info in infos], dtype=np.int8),
        warnings=np.asarray([int(np.sum(row["warnings"])) for row in states[1:]], dtype=int),
        first_fault_tick=np.int64(-1 if meta.get("first_fault_tick") is None else meta["first_fault_tick"]),
        contact_force=np.vstack(forces) if forces else empty3,
        contact_torque=np.vstack(torques) if torques else empty3,
        contact_point=np.vstack(points) if points else empty3,
        contact_dist=np.asarray(dists, dtype=float),
        contact_geom1=np.asarray(geom1, dtype=str),
        contact_geom2=np.asarray(geom2, dtype=str),
        contact_frame=np.vstack(frames) if frames else np.zeros((0, 9)),
        contact_offsets=np.asarray(offsets, dtype=int),
        **(extra or {}),
    )
    write_json(case_id, meta)
    del scene, n
    return np.vstack([row["tcp_pos"] for row in states]), np.array([row["episode_time"] for row in states])


_GEAR_PEAKS = {}


@pytest.mark.parametrize("case_id", physical_case_ids())
def test_physical(case_id):
    from feedingrobot.controllers.cartesian_impedance import control_step
    from feedingrobot.controllers.v3spec import PHYSICAL, compare_scalar

    family, seed_s, variant = case_id.split("-")
    seed = int(seed_s[1:])
    cfg = load_m2_config()
    # Historical evidence-format regression. Full-v1 has its own physical matrix/profile.
    cfg["stop"]["damping_nm_s_per_rad"] = 1.0
    cfg["power_on_damping_scale"] = 1.0
    if family == "gear":
        scene_cfg = json_scene_spring()
        cfg = dict(cfg)
        cfg["press_test"] = True
    else:
        scene_cfg = "configs/m1_scene.json"
    scene = FeedingScene(scene_cfg)
    pose = seed if family == "pause" else 0
    place_arm(scene, scene.config["q_torque_poses"][pose], seed=seed)
    hold = np.array(scene.data.qfrc_bias[scene.index.arm_dof_adr], dtype=float)
    for _ in range(200):
        scene.step_physics(hold)
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
    ctl, guard = start_controller(scene, cfg, state, tau=np.zeros(7) if family == "takeover" else bias)
    ctl.power_on = family == "takeover"
    release_at = None

    def twist(controller):
        if family in {"release", "release_fast"} and getattr(controller, "_rel_phase", "push") == "push" and controller.sim_time >= 0.02:
            return np.array([0.05, 0, 0, 0, 0, 0])
        if family == "gear":
            mode = getattr(controller, "_gear_phase", "approach")
            if mode in {"press", "blend"}:
                return np.array([0.0, -0.002, 0, 0, 0, 0])
            if mode == "approach":
                return np.array([0.0, -0.008, 0, 0, 0, 0])
        return np.zeros(6)

    def hook(t, controller):
        nonlocal release_at
        scene.clear_external_wrench()
        if family == "pause" and 0.05 <= t < 0.08:
            controller._pause_advance = True
            scene.set_external_wrench([4.0, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
        elif family in {"release", "release_fast"}:
            from feedingrobot.controllers.v3spec import RELEASE_FORCE_N, RELEASE_RAMP_S

            phase = getattr(controller, "_rel_phase", "push")
            if phase == "push" and controller.reference.blocked:
                controller.set_command(np.zeros(6), controller.sim_time, controller.sim_time + 0.1, controller.phase)
                if family == "release_fast":
                    controller._rel_phase = "free"
                    if release_at is None:
                        release_at = t
                else:
                    controller._rel_phase = "ramp"
                    controller._rel_block = t
                phase = controller._rel_phase
            if phase == "push" and t >= 0.04:
                scene.set_external_wrench([-RELEASE_FORCE_N, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
            elif phase == "ramp":
                alpha = min(1.0, (t - controller._rel_block) / RELEASE_RAMP_S)
                if alpha < 1.0:
                    scene.set_external_wrench([-RELEASE_FORCE_N * (1.0 - alpha), 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
                elif release_at is None:
                    controller._rel_phase = "free"
                    release_at = t
        elif family == "wrench" and 0.05 <= t < 0.15:
            scene.set_external_wrench([1.0, 0, 0], [0, 0, 0], controller.scene.snapshot()["tcp_pos"])
        elif family == "gear":
            normal = float(_bowl_force(controller.scene.snapshot()["contacts"])[1])
            mode = getattr(controller, "_gear_phase", "approach")
            if mode == "approach" and normal > 0.05:
                controller._gear_phase = "press"
                controller._press_t = t
                controller.set_command(np.array([0.0, -0.002, 0, 0, 0, 0]), controller.sim_time, controller.sim_time + 0.1, controller.phase)
            elif mode == "press" and t >= controller._press_t + 0.08 and normal >= 0.15:
                controller._gear_phase = "blend"
                controller.phase = "ACQUIRE"
                controller.set_command(np.array([0.0, -0.002, 0, 0, 0, 0]), controller.sim_time, controller.sim_time + 0.1, "ACQUIRE")

    duration = PHYSICAL[family]["duration"]
    dt = scene.dt
    steps = int(round(duration / dt))
    publish = max(int(round(0.05 / dt)), 1)
    states = [scene.snapshot()]
    twists = [_tcp_twist(scene)]
    infos = []
    inputs = {"command": [], "external": [], "force_used": [], "pause": [], "saturation": [], "blocked": [], "p_ref_before": [], "r_ref_before": [], "guard_status": []}
    for i in range(steps):
        cached = ctl.wrench._cache
        inputs["force_used"].append(np.zeros(3) if cached is None else np.asarray(cached["compensated_wrench_tcp"][:3], dtype=float))
        hook(ctl.sim_time, ctl)
        inputs["pause"].append(1 if ctl._pause_advance else 0)
        inputs["p_ref_before"].append(np.array(ctl.reference.p_ref, dtype=float).copy())
        inputs["r_ref_before"].append(np.array(ctl.reference.r_ref, dtype=float).reshape(3, 3).copy())
        applied = np.zeros(6)
        if scene.applied_wrench is not None:
            applied = np.concatenate([np.asarray(scene.applied_wrench["force"], dtype=float), np.asarray(scene.applied_wrench["torque"], dtype=float)])
        inputs["external"].append(applied)
        if i % publish == 0 and guard.status == "RUNNING" and not ctl.power_on:
            ctl.set_command(twist(ctl), ctl.sim_time, ctl.sim_time + 0.1, ctl.phase)
        state, info = control_step(ctl, guard)
        inputs["command"].append(np.asarray(info.get("twist_command", np.zeros(6)), dtype=float))
        inputs["saturation"].append(1 if info.get("saturated_latched") else 0)
        inputs["blocked"].append(1 if info.get("blocked_now") else 0)
        inputs["guard_status"].append(str(guard.status))
        states.append(state)
        twists.append(_tcp_twist(scene))
        infos.append(info)
    assert guard.failure is None, guard.failure
    assert len(infos) == steps
    meta = {
        "dt": float(scene.model.opt.timestep),
        "iterations": int(scene.model.opt.iterations),
        "tolerance": float(scene.model.opt.tolerance),
        "seed": seed,
        "pose_id": pose,
        "variant": variant,
        "tau_before": np.zeros(7) if family == "takeover" else bias,
        "qpos0": qpos,
        "qvel0": qvel,
        "unload": "fast" if family == "release_fast" else ("slow" if family == "release" else None),
    }
    if family in {"release", "release_fast"}:
        meta["blocked_seen"] = any(flag > 0.5 for flag in inputs["blocked"])
    tcp, times = _record_physical(case_id, scene, states, twists, infos, inputs, meta)
    assert np.max(np.abs(np.vstack([row["dq"] for row in states]))) <= 0.5 + 1e-9
    tau = np.vstack([row["tau_command"] for row in states[1:]])
    assert np.max(np.abs(tau[0] - meta["tau_before"])) <= 2000.0 * dt + 1e-6
    assert np.max(np.abs(np.diff(tau, axis=0))) <= 2000.0 * dt + 1e-6
    pref = np.vstack([np.asarray(info["p_ref"], dtype=float) for info in infos])
    assert np.max(np.linalg.norm(pref - tcp[:-1], axis=1)) <= 0.02 + 1e-9
    if family == "gear":
        normals = []
        for row in states[1:]:
            normals.append(float(_bowl_force(row["contacts"])[1]))
        peak = float(np.max(normals)) if normals else 0.0
        assert peak > 0.1
        _GEAR_PEAKS[(seed, variant)] = peak
        other = _GEAR_PEAKS.get((seed, "A"))
        if variant in {"B", "C"} and other is not None:
            assert compare_scalar("contact_force", other, peak, 0.1, 0.02) is None
        assert any(str(info["phase_k"]) == "ACQUIRE" for info in infos)
    if family in {"release", "release_fast"}:
        assert any(flag > 0.5 for flag in inputs["blocked"])
        assert release_at is not None
        speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.diff(times)
        speed_t = times[1:]
        window = (speed_t >= release_at) & (speed_t <= release_at + 0.5)
        assert np.max(speed[window]) <= 0.03 + 1e-9
        stable = (speed_t >= release_at + 0.8) & (speed_t <= release_at + 1.0)
        assert np.max(speed[stable]) <= 0.005 + 1e-9
    if family == "wrench":
        loaded = [info for info, row in zip(infos, inputs["external"]) if abs(row[0]) > 0.5]
        assert loaded
        assert max(abs(float(np.asarray(info["ft_compensated_wrench_tcp"])[0])) for info in loaded) > 0.5
