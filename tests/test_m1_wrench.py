"""T06: known wrench, sign, and moment arm."""

import mujoco
import numpy as np

from feedingrobot.sim.cases import save_case
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.sim.sensors import read_raw_wrench, site_position, site_rotation, world_and_tcp_wrench


def _apply(model, data, body_id, force, torque, point):
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    com = np.array(data.xipos[body_id], dtype=float)
    data.xfrc_applied[body_id, :3] = force
    data.xfrc_applied[body_id, 3:] = torque + np.cross(point - com, force)
    mujoco.mj_forward(model, data)


def _read(model, data, sign):
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
    raw = read_raw_wrench(model, data, int(model.sensor_adr[fid]), int(model.sensor_adr[tid]))
    wrench, _ = world_and_tcp_wrench(raw, site_rotation(data, sid), site_position(data, sid), site_position(data, sid), sign)
    return wrench


def _check_delta(model, data, body, sign, point):
    forces = [np.eye(3)[i] for i in range(3)] + [-np.eye(3)[i] for i in range(3)]
    _apply(model, data, body, np.zeros(3), np.zeros(3), point)
    base = _read(model, data, sign)
    for force in forces:
        _apply(model, data, body, force, np.zeros(3), point)
        got = _read(model, data, sign) - base
        r = point - site_position(data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site"))
        expect_t = np.cross(r, force)
        assert np.max(np.abs(got[:3] - force)) <= max(0.02, 0.02 * np.max(np.abs(force)))
        assert np.max(np.abs(got[3:] - expect_t)) <= max(0.002, 0.02 * max(np.max(np.abs(expect_t)), 1e-9))
    torque = np.array([0.0, 0.2, 0.0])
    _apply(model, data, body, np.zeros(3), torque, point)
    got = _read(model, data, sign) - base
    assert np.max(np.abs(got[:3])) <= 0.02
    assert np.max(np.abs(got[3:] - torque)) <= max(0.002, 0.02 * 0.2)


def test_t06_fixture_and_rotated_tool():
    path = repo_root() / "assets" / "scenes" / "spoon_fixture.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_mount")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
    mujoco.mj_forward(model, data)
    point = site_position(data, sid) + np.array([0.05, 0.01, 0.02])
    sign = -1.0
    _check_delta(model, data, body, sign, point)
    # 90 deg about world X by rotating the parent body quat
    model.body_quat[body] = [0.70710678118, 0.70710678118, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    point = site_position(data, sid) + np.array([0.0, 0.04, 0.03])
    _check_delta(model, data, body, sign, point)


def test_t06_full_panda_static(m1_config):
    scene = FeedingScene(m1_config)
    q = np.array(scene.config["q_torque_poses"][2], dtype=float)
    scene.data.qpos[scene.index.arm_qpos_adr] = q
    scene.data.qvel[:] = 0
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 7] = [2, 2, 0.5, 1, 0, 0, 0]
    q_hold = q.copy()
    for _ in range(200):
        scene.step_physics(scene.diagnostic_pd_tau(q_hold, kp=80, kd=16), hold_driver=True)
    point = scene.snapshot()["ft_pos"] + np.array([0.04, 0.0, 0.02])
    samples = []
    for axis in range(3):
        force = np.zeros(3)
        force[axis] = 1.0
        scene.set_external_wrench(force, np.zeros(3), point)
        window = []
        sites = []
        for _ in range(150):
            state = scene.step_physics(scene.diagnostic_pd_tau(q_hold, kp=80, kd=16), hold_driver=True)
            window.append(state["wrench_world"].copy())
            sites.append(state["ft_pos"].copy())
        samples.append((force, np.mean(window[-30:], axis=0), np.mean(sites[-30:], axis=0)))
        scene.clear_external_wrench()
        for _ in range(80):
            scene.step_physics(scene.diagnostic_pd_tau(q_hold, kp=80, kd=16), hold_driver=True)
    scene.clear_external_wrench()
    base_window = []
    for _ in range(80):
        base_window.append(scene.step_physics(scene.diagnostic_pd_tau(q_hold, kp=80, kd=16), hold_driver=True)["wrench_world"])
    base = np.mean(base_window[-30:], axis=0)
    for force, mean, site in samples:
        delta = mean - base
        assert np.max(np.abs(delta[:3] - force)) <= 0.02
        r = point - site
        expect_t = np.cross(r, force)
        assert np.max(np.abs(delta[3:] - expect_t)) <= max(0.002, 0.02 * max(np.max(np.abs(expect_t)), 1e-9))
    save_case("T06", {"max_force_error_n": float(np.max(np.abs(delta[:3] - force)))})
