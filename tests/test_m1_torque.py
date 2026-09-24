"""T02, T03, T04: torque units, no hidden servo, gravity compensation."""

import mujoco
import numpy as np

from feedingrobot.sim.cases import save_case
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene


def _scene(m1_config):
    return FeedingScene(m1_config)


def _park_food(scene):
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 7] = [2.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    scene.data.qvel[scene.index.food_dof_adr : scene.index.food_dof_adr + 6] = 0


def _set_pose(scene, q, dq=None):
    scene.data.qpos[scene.index.arm_qpos_adr] = q
    scene.data.qvel[scene.index.arm_dof_adr] = 0 if dq is None else dq
    scene.data.qpos[scene.index.head_qpos_adr] = 0
    scene.data.qvel[scene.index.head_dof_adr] = 0
    scene.data.qpos[scene.index.jaw_qpos_adr] = 0.1
    scene.data.qvel[scene.index.jaw_dof_adr] = 0
    _park_food(scene)
    scene.data.ctrl[:] = 0
    mujoco.mj_forward(scene.model, scene.data)


def test_t02_unit_torque_three_poses(m1_config):
    scene = _scene(m1_config)
    poses = scene.config["q_torque_poses"]
    max_err = 0.0
    for q in poses:
        for joint in range(7):
            for tau in (0.0, 1.0, -1.0):
                _set_pose(scene, q)
                cmd = np.zeros(7)
                cmd[joint] = tau
                scene.data.ctrl[scene.index.arm_actuator_ids] = cmd
                mujoco.mj_forward(scene.model, scene.data)
                got = scene.data.qfrc_actuator[scene.index.arm_dof_adr]
                err = np.max(np.abs(got - cmd))
                assert err <= 1e-8, (joint, tau, got, err)
                max_err = max(max_err, float(err))
    save_case("T02", {"max_abs_error_nm": max_err})


def test_t02_ctrl_saturates_at_force_limit(m1_config):
    scene = _scene(m1_config)
    _set_pose(scene, scene.config["q_torque_poses"][0])
    scene.data.ctrl[scene.index.arm_actuator_ids[0]] = 200.0
    scene.data.ctrl[scene.index.arm_actuator_ids[4]] = -50.0
    mujoco.mj_forward(scene.model, scene.data)
    assert abs(scene.data.actuator_force[scene.index.arm_actuator_ids[0]] - 87.0) <= 1e-6
    assert abs(scene.data.actuator_force[scene.index.arm_actuator_ids[4]] + 12.0) <= 1e-6
    state = scene.step_physics(np.array([200.0, 0, 0, 0, -50.0, 0, 0]))
    assert state["software_limited"][0] and state["software_limited"][4]
    assert abs(state["tau_command"][0] - 87.0) <= 1e-9
    assert abs(state["tau_command"][4] + 12.0) <= 1e-9


def test_t03_zero_ctrl_has_no_actuator_force(m1_config):
    scene = _scene(m1_config)
    q = np.array(scene.config["q_torque_poses"][1], dtype=float)
    for dq in (np.zeros(7), np.array([0.2, -0.1, 0.05, 0.0, -0.2, 0.1, 0.0])):
        _set_pose(scene, q, dq)
        scene.data.ctrl[:] = 0
        mujoco.mj_forward(scene.model, scene.data)
        got = scene.data.qfrc_actuator[scene.index.arm_dof_adr]
        assert np.max(np.abs(got)) <= 1e-8
    save_case("T03", {"max_abs_actuator_nm": float(np.max(np.abs(got)))})


def test_t04_gravity_and_compensation(m1_config):
    scene = _scene(m1_config)
    q0 = np.array(scene.config["q_torque_poses"][0], dtype=float)
    _set_pose(scene, q0)
    q_start = scene.data.qpos[scene.index.arm_qpos_adr].copy()
    for _ in range(200):
        scene.step_physics(np.zeros(7), hold_driver=True)
    q_end = scene.data.qpos[scene.index.arm_qpos_adr]
    assert np.max(np.abs(q_end - q_start)) > 1e-3
    pairs = {(c["group1"], c["group2"]) for c in scene.snapshot()["contacts"]}
    assert ("arm", "table") not in pairs and ("spoon", "table") not in pairs

    for q in scene.config["q_torque_poses"]:
        _set_pose(scene, q)
        scene.model.opt.gravity[:] = 0
        q_start = scene.data.qpos[scene.index.arm_qpos_adr].copy()
        for _ in range(200):
            scene.step_physics(np.zeros(7), hold_driver=True)
        assert np.max(np.abs(scene.data.qpos[scene.index.arm_qpos_adr] - q_start)) < 1e-4
        scene.restore_model_options()

    _set_pose(scene, q0)
    scene.model.opt.gravity[:] = scene._opt_baseline["gravity"]
    q_start = scene.data.qpos[scene.index.arm_qpos_adr].copy()
    for _ in range(2000):
        scene.step_physics(scene.bias_tau(), hold_driver=True)
    drift = np.max(np.abs(scene.data.qpos[scene.index.arm_qpos_adr] - q_start))
    assert drift < 1e-3, drift
    scene.restore_model_options()
    save_case("T04", {"hold_drift_rad": float(drift)})
