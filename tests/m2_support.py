"""Shared M2 episode helper. Commands are refreshed every 50 ms for 100 ms."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from feedingrobot.controllers.cartesian_impedance import CartesianImpedance, control_step
from feedingrobot.controllers.guard import Guard
from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.sim.scene import FeedingScene


def load_m2_config():
    with open("configs/m2_controller.json") as handle:
        return json.load(handle)


def place_arm(scene: FeedingScene, q, seed: int = 0) -> dict:
    scene.reset(seed=seed, preset="food_on_plate", settle_steps=0)
    scene.data.qpos[scene.index.arm_qpos_adr] = np.asarray(q, dtype=float)
    scene.data.qvel[scene.index.arm_dof_adr] = 0
    scene.data.ctrl[:] = 0
    mujoco.mj_forward(scene.model, scene.data)
    return scene.snapshot()


def start_controller(scene, cfg, state, phase="TRANSPORT", tau=None):
    ctl = CartesianImpedance(scene, cfg)
    guard = Guard(cfg, cfg["joint_speed_limit_rad_s"])
    ctl.guard = guard
    if tau is None:
        tau = np.zeros(7)
    ctl.reset(state, tau, phase)
    return ctl, guard


def run_commands(ctl, guard, duration_s: float, twist_fn, phase: str = "TRANSPORT"):
    """twist_fn(t_after_start, ctl, state) -> 6-vector. t is episode time at publish."""
    samples = []
    steps = int(round(duration_s / ctl.scene.dt))
    publish = max(int(round(0.05 / ctl.scene.dt)), 1)
    for i in range(steps):
        if i % publish == 0 and guard.status == "RUNNING":
            twist = np.asarray(twist_fn(ctl.sim_time, ctl), dtype=float).reshape(6)
            ctl.set_command(twist, ctl.sim_time, ctl.sim_time + 0.1, phase)
        state, info = control_step(ctl, guard, None)
        samples.append((state, info))
        if guard.status == "ABORTED":
            break
    return samples
