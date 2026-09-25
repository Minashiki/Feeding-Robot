"""T01–T03: Jacobian, SO(3), mass matrix, torque and command limits."""

import json

import mujoco
import numpy as np
import pytest

from feedingrobot.controllers.cartesian_impedance import (
    CartesianImpedance,
    clip_torque,
    damped_nullspace,
    singularity_gains,
)
from feedingrobot.controllers.guard import Guard
from feedingrobot.controllers.so3 import (
    mat_to_quat_wxyz,
    orientation_error,
    quat_wxyz_to_mat,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    so3_exp,
    so3_log,
)
from feedingrobot.sim.scene import FeedingScene


def _cfg():
    with open("configs/m2_controller.json") as handle:
        return json.load(handle)


def _arm_at(scene, q):
    scene.reset(seed=0, preset="food_on_plate", settle_steps=0)
    scene.data.qpos[scene.index.arm_qpos_adr] = q
    scene.data.qvel[scene.index.arm_dof_adr] = 0
    scene.data.ctrl[:] = 0
    mujoco.mj_forward(scene.model, scene.data)
    return scene.snapshot()


def test_t01_so3_and_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        quat = rng.normal(size=4)
        quat = quat / np.linalg.norm(quat)
        mat = quat_wxyz_to_mat(quat)
        back = mat_to_quat_wxyz(mat)
        if np.dot(back, quat) < 0:
            back = -back
        assert np.max(np.abs(back - quat)) <= 1e-10
        xyzw = quat_wxyz_to_xyzw(quat)
        assert np.max(np.abs(quat_xyzw_to_wxyz(xyzw) - quat)) <= 1e-10
    small = so3_exp(np.array([0.0, 0.0, 1e-4]))
    err = orientation_error(small, np.eye(3))
    assert err[2] > 0
    near = so3_exp(np.array([0.0, 1.0, 0.0]) * (np.pi - 1e-6))
    vec = so3_log(near)
    assert np.all(np.isfinite(vec))
    assert abs(np.linalg.norm(vec) - (np.pi - 1e-6)) < 1e-5
    flipped = so3_log(so3_exp(-vec))
    assert np.dot(vec, -flipped) > 0


def test_t01_jacobian_matches_central_difference():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = _cfg()
    h = 1e-6
    for q in scene.config["q_torque_poses"]:
        state = _arm_at(scene, q)
        ctl = CartesianImpedance(scene, cfg)
        ctl.guard = Guard(cfg, 0.5)
        ctl.reset(state, np.zeros(7), "TRANSPORT")
        jac, _bias = ctl._jacobian()
        qpos = scene.data.qpos.copy()
        adr = scene.index.arm_qpos_adr
        tcp = scene.index.site_ids["tcp"]
        for axis, sign in ((0, 1.0), (1, -1.0), (3, 1.0), (5, -1.0)):
            for column in range(7):
                plus = qpos.copy()
                minus = qpos.copy()
                plus[adr[column]] += h
                minus[adr[column]] -= h
                scene.data.qpos[:] = plus
                mujoco.mj_forward(scene.model, scene.data)
                p_plus = scene.data.site_xpos[tcp].copy()
                r_plus = scene.data.site_xmat[tcp].reshape(3, 3).copy()
                scene.data.qpos[:] = minus
                mujoco.mj_forward(scene.model, scene.data)
                p_minus = scene.data.site_xpos[tcp].copy()
                r_minus = scene.data.site_xmat[tcp].reshape(3, 3).copy()
                scene.data.qpos[:] = qpos
                mujoco.mj_forward(scene.model, scene.data)
                dp = (p_plus - p_minus) / (2 * h)
                assert np.max(np.abs(dp - jac[0:3, column])) <= 1e-6
                w = so3_log(r_plus @ r_minus.T) / (2 * h)
                assert np.max(np.abs(w - jac[3:6, column])) <= 1e-6
            del axis, sign
        dq = np.linspace(-0.2, 0.2, 7)
        scene.data.qvel[scene.index.arm_dof_adr] = dq
        mujoco.mj_forward(scene.model, scene.data)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(scene.model, scene.data, int(mujoco.mjtObj.mjOBJ_SITE), tcp, vel, 0)
        twist = np.concatenate([vel[3:], vel[:3]])
        assert np.max(np.abs(jac @ dq - twist)) <= 1e-8


def test_t01_small_rotation_restores():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = _cfg()
    state = _arm_at(scene, scene.config["q_torque_poses"][0])
    ctl = CartesianImpedance(scene, cfg)
    ctl.guard = Guard(cfg, 0.5)
    ctl.reset(state, scene.bias_tau(), "TRANSPORT")
    ctl.power_on = False
    ctl.tau_prev = scene.bias_tau()
    delta = so3_exp(np.array([0.0, 0.0, 0.02]))
    ctl.reference.r_ref = delta @ state["tcp_mat"]
    _tau, info = ctl.compute(scene.snapshot(), scene.dt)
    assert info["e"][5] > 0
    assert np.dot(info["tau_task"], info["tau_task"]) > 0


def test_t02_mass_spd_and_leakage():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = _cfg()
    for q in scene.config["q_torque_poses"]:
        state = _arm_at(scene, q)
        ctl = CartesianImpedance(scene, cfg)
        mass = ctl._mass()
        assert np.max(np.abs(mass - mass.T)) <= 1e-10
        np.linalg.cholesky(mass)
        jac, bias = ctl._jacobian()
        assert np.all(np.isfinite(bias))
        scale = np.diag([1, 1, 1, 0.1, 0.1, 0.1])
        j_s = scale @ jac
        sv = np.linalg.svd(j_s, compute_uv=False)
        rho = float(sv[-1] / sv[0])
        assert rho > 0.01
        null, _jbar, _lam = damped_nullspace(j_s, mass, 1e-8, 1e-4)
        posture = np.ones(7)
        leak = jac @ np.linalg.solve(mass, null.T @ posture)
        assert np.linalg.norm(leak[:3]) <= 0.01
        assert np.linalg.norm(leak[3:]) <= 0.01
        state["rho"] = rho
    low, eps_low = singularity_gains(0.005, 1e-8, 1e-2, 0.05, 0.01)
    mid, _eps_mid = singularity_gains(0.03, 1e-8, 1e-2, 0.05, 0.01)
    high, eps_high = singularity_gains(0.2, 1e-8, 1e-2, 0.05, 0.01)
    assert low == 0.0 and eps_low == 1e-2
    assert 0.0 < mid < 1.0
    assert high == 1.0 and eps_high == 1e-8


def test_t03_torque_and_command_rejection():
    prev = np.zeros(7)
    limit = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
    out = clip_torque(np.full(7, 1000.0), prev, limit, 2000.0, 0.001)
    assert np.all(out <= limit)
    assert np.max(np.abs(out - prev)) <= 2000.0 * 0.001 + 1e-12
    scene = FeedingScene("configs/m1_scene.json")
    cfg = _cfg()
    state = _arm_at(scene, scene.config["q_torque_poses"][0])
    q_before = scene.data.qpos.copy()
    ctl = CartesianImpedance(scene, cfg)
    guard = Guard(cfg, 0.5)
    ctl.guard = guard
    ctl.reset(state, np.zeros(7), "TRANSPORT")
    bad = ctl.set_command([np.nan, 0, 0, 0, 0, 0], 0.0, 0.1, "TRANSPORT")
    assert bad["accepted"] is False
    assert guard.status == "ABORTED"
    tau, info = ctl.compute(scene.snapshot(), 0.001)
    assert info["apply"] is False
    assert np.allclose(scene.data.qpos, q_before)
    future = CartesianImpedance(scene, cfg)
    future.guard = Guard(cfg, 0.5)
    future.reset(state, np.zeros(7), "TRANSPORT")
    assert future.set_command(np.ones(6), 10.0, 10.1, "TRANSPORT")["rejected"] == "future"
    assert future.set_command(np.zeros(6), 0.0, 0.1, "NOT_A_PHASE")["rejected"] == "phase"
