"""One contact-matrix run from a saved physical initial state."""

from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np

from feedingrobot.sim.events import Departure, food_on_spoon, outside_support
from feedingrobot.sim.trace import Trace
from feedingrobot.sim.trajectory import quintic, smooth_pulse
from feedingrobot.sim.trial import FORBIDDEN, Trial


def state_digest(data) -> str:
    blob = np.concatenate([np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)])
    return hashlib.sha256(blob.tobytes()).hexdigest()


def capture_state(scene) -> dict:
    return {
        "qpos": np.array(scene.data.qpos, dtype=np.float64).copy(),
        "qvel": np.array(scene.data.qvel, dtype=np.float64).copy(),
    }


def restore_state(scene, snap: dict) -> None:
    mujoco.mj_resetData(scene.model, scene.data)
    scene.data.qpos[:] = snap["qpos"]
    scene.data.qvel[:] = snap["qvel"]
    scene.data.ctrl[:] = 0
    scene.data.qfrc_applied[:] = 0
    scene.data.xfrc_applied[:] = 0
    mujoco.mj_forward(scene.model, scene.data)
    scene.tick = 0
    scene.time_offset = float(scene.data.time)
    scene.velocity_fault = False
    scene._hold_latched = False
    scene._hold_head_target = None


def _group_force(contacts, group, partner) -> tuple[np.ndarray, float]:
    total = np.zeros(3)
    peak = 0.0
    for row in contacts:
        if {row["group1"], row["group2"]} != {group, partner}:
            continue
        force = row["force_on_geom1_world"] if row["group1"] == group else row["force_on_geom2_world"]
        total = total + force
        peak = max(peak, float(np.linalg.norm(force)))
    return total, peak


def partner_for(kind: str) -> tuple[str, str]:
    if kind == "plate":
        return "food", "plate"
    if kind in ("spoon", "tilt", "accel"):
        return "food", "spoon"
    return "spoon", "mouth"


def run_case(scene, snap, kind: str, duration_s: float) -> dict:
    restore_state(scene, snap)
    cfg = scene.config
    support = cfg["spoon_support"]
    accel = cfg["diagnostic_accel"]
    trial = Trial(scene.speed_limit)
    trace = Trace()
    depart = Departure(float(support["confirm_s"]))
    q0 = scene.data.qpos[scene.index.arm_qpos_adr].copy()
    dt = float(scene.model.opt.timestep)
    steps = int(round(duration_s / dt))
    lip_event = None
    for _ in range(steps):
        t_ref = float(scene.data.time - scene.time_offset)
        if kind == "tilt":
            q_ref, dq_ref = quintic(q0, np.array(cfg["q_tilt"]), t_ref, float(cfg["tilt"]["duration_s"]))
            tau = scene.diagnostic_pd_tau(q_ref, float(cfg["tilt"]["kp"]), float(cfg["tilt"]["kd"]), dq_ref)
        elif kind == "accel":
            tau = scene.bias_tau()
            tau[int(accel["joint"])] += smooth_pulse(t_ref, float(accel["start_s"]), float(accel["duration_s"]), float(accel["amplitude_nm"]))
        elif kind == "lip":
            q_ref, dq_ref = quintic(np.array(cfg["q_mouth_out"]), np.array(cfg["q_mouth_lip"]), t_ref, float(cfg["matrix"]["lip_move_s"]))
            tau = scene.diagnostic_pd_tau(q_ref, float(cfg["mouth_move"]["kp"]), float(cfg["mouth_move"]["kd"]), dq_ref)
        else:
            tau = scene.bias_tau()
        state = scene.step_physics(tau, hold_driver=True)
        banned = False
        for row in state["contacts"]:
            pair = tuple(sorted((row["group1"], row["group2"])))
            if pair in FORBIDDEN:
                banned = True
        state["forbidden_contact"] = banned
        trial.observe(state)
        group, other = partner_for(kind)
        force, peak = _group_force(state["contacts"], group, other)
        local = state["tcp_mat"].T @ (state["food_pos"] - state["tcp_pos"])
        touching = food_on_spoon(state["contacts"])
        outside = outside_support(local, support)
        depart.update(state["episode_time"], dt, outside, touching)
        if kind == "lip" and lip_event is None and any("mouth_upper" in (c["geom1"], c["geom2"]) for c in state["contacts"]):
            lip_event = float(state["episode_time"])
        trace.add(state, tau, state["tau_command"], force, peak, outside, depart)
    arrays = trace.arrays()
    force = arrays["contact_force_group"]
    window = max(1, int(round(0.2 / dt)))
    steady = force[-window:]
    vector_impulse = np.sum(force, axis=0) * dt
    scalar_impulse = float(np.sum(np.linalg.norm(force, axis=1)) * dt)
    event = lip_event if kind == "lip" else depart.confirmed_s
    metrics = {
        "departure_start_s": depart.start_s,
        "departure_confirmed_s": depart.confirmed_s,
        "event_s": event,
        "steady_force_n": np.mean(steady, axis=0).tolist(),
        "peak_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        "vector_impulse_ns": vector_impulse.tolist(),
        "scalar_impulse_ns": scalar_impulse,
        "steady_food_pos_m": np.mean(arrays["food_pos"][-window:], axis=0).tolist(),
        "min_contact_dist_m": float(np.min(arrays["min_contact_dist"])),
        "max_joint_speed_rad_s": float(np.max(np.abs(arrays["dq"]))),
        "ok": trial.ok,
        "failure_reason": None if trial.ok else trial.failure["reason"],
        "first_failure_tick": None if trial.ok else int(trial.failure["tick"]),
        "n_samples": int(force.shape[0]),
    }
    return {"metrics": metrics, "arrays": arrays, "trial": trial}
