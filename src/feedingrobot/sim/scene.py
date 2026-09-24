"""Reset and one physical step of the P0 feeding scene."""

from __future__ import annotations

import mujoco
import numpy as np

from feedingrobot.sim.contacts import min_distance, read_contacts
from feedingrobot.sim.model import ModelIndex, load_config, load_model, resolve_path
from feedingrobot.sim.sensors import read_raw_wrench, site_position, site_rotation, world_and_tcp_wrench

_ARM_TABLE_GROUPS = {"arm", "spoon"}
_FORBIDDEN_RESET = {("arm", "table"), ("arm", "plate"), ("arm", "mouth"), ("spoon", "table"), ("spoon", "mouth")}


def _as_vec(value, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(n)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite vector")
    return arr


class FeedingScene:
    def __init__(self, config: dict | str):
        if isinstance(config, str):
            config = load_config(config)
        self.config = config
        self.model = load_model(config["model"])
        self.data = mujoco.MjData(self.model)
        self.index = ModelIndex(self.model)
        self.index.assert_arm_motors()
        self.dt = float(self.model.opt.timestep)
        self.wrench_sign = float(config["wrench_sign"])
        self.rng = np.random.default_rng()
        self._opt_baseline = {
            "gravity": np.array(self.model.opt.gravity, dtype=float).copy(),
            "timestep": float(self.model.opt.timestep),
            "iterations": int(self.model.opt.iterations),
            "tolerance": float(self.model.opt.tolerance),
        }
        self.applied_wrench = None
        self._clear_python_state()

    def restore_model_options(self) -> None:
        self.model.opt.gravity[:] = self._opt_baseline["gravity"]
        self.model.opt.timestep = self._opt_baseline["timestep"]
        self.model.opt.iterations = self._opt_baseline["iterations"]
        self.model.opt.tolerance = self._opt_baseline["tolerance"]
        self.dt = float(self.model.opt.timestep)

    def _clear_python_state(self) -> None:
        self.tick = 0
        self.time_offset = 0.0
        self.contact_events: list = []
        self.peak_contact_force = 0.0
        self.last_tau = np.zeros(7)
        self.velocity_fault = False
        self.applied_wrench = None
        self.settle_steps = 0

    def set_external_wrench(self, force, torque, point, body_name: str = "tool_mount") -> None:
        """Wrench applied on the next steps. Torque is about `point` in world."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.applied_wrench = {
            "body_id": int(body_id),
            "force": _as_vec(force, 3),
            "torque": _as_vec(torque, 3),
            "point": _as_vec(point, 3),
        }

    def clear_external_wrench(self) -> None:
        self.applied_wrench = None

    def _write_external(self) -> None:
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        spec = self.applied_wrench
        if spec is None:
            return
        mujoco.mj_forward(self.model, self.data)
        body = spec["body_id"]
        com = np.array(self.data.xipos[body], dtype=float)
        force = spec["force"]
        torque_com = spec["torque"] + np.cross(spec["point"] - com, force)
        self.data.xfrc_applied[body, :3] = force
        self.data.xfrc_applied[body, 3:] = torque_com

    def _head_targets(self, t: float, hold: bool):
        cfg = self.config["head"]
        q = np.zeros(5)
        q[:4] = self.data.qpos[self.index.head_qpos_adr]
        q[4] = self.data.qpos[self.index.jaw_qpos_adr]
        if hold:
            return q, np.zeros(5)
        w = 2.0 * np.pi * float(cfg["freq_hz"])
        amp = float(cfg["amp_m"])
        yaw = float(cfg["yaw_amp_rad"])
        jaw0 = float(cfg["jaw_center_rad"])
        jaw = float(cfg["jaw_amp_rad"])
        des = np.array(
            [
                amp * np.sin(w * t),
                amp * np.sin(w * t + 1.2),
                0.0,
                yaw * np.sin(w * t),
                jaw0 + jaw * np.sin(w * t),
            ]
        )
        vel = np.array(
            [
                amp * w * np.cos(w * t),
                amp * w * np.cos(w * t + 1.2),
                0.0,
                yaw * w * np.cos(w * t),
                jaw * w * np.cos(w * t),
            ]
        )
        return des, vel

    def _write_head(self, hold: bool) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        t = self.tick * self.dt
        des, vel = self._head_targets(t, hold)
        q = np.zeros(5)
        dq = np.zeros(5)
        q[:4] = self.data.qpos[self.index.head_qpos_adr]
        q[4] = self.data.qpos[self.index.jaw_qpos_adr]
        dq[:4] = self.data.qvel[self.index.head_dof_adr]
        dq[4] = self.data.qvel[self.index.jaw_dof_adr]
        bias = np.zeros(5)
        bias[:4] = self.data.qfrc_bias[self.index.head_dof_adr]
        bias[4] = self.data.qfrc_bias[self.index.jaw_dof_adr]
        kp = np.asarray(self.config["head"]["kp"], dtype=float)
        kd = np.asarray(self.config["head"]["kd"], dtype=float)
        tau = bias + kp * (des - q) + kd * (vel - dq)
        ids = self.index.scene_actuator_ids
        lo = self.model.actuator_ctrlrange[ids, 0]
        hi = self.model.actuator_ctrlrange[ids, 1]
        cmd = np.clip(tau, lo, hi)
        self.data.ctrl[ids] = cmd
        self._head_gravcomp = bias.copy()
        self._head_tau = cmd.copy()
        return cmd

    def bias_tau(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.qfrc_bias[self.index.arm_dof_adr], dtype=float).copy()

    def diagnostic_pd_tau(self, q_des, kp: float = 40.0, kd: float = 8.0) -> np.ndarray:
        """Explicit joint PD plus bias. Test fixture only; not the M2 controller."""
        q = np.array(self.data.qpos[self.index.arm_qpos_adr], dtype=float)
        dq = np.array(self.data.qvel[self.index.arm_dof_adr], dtype=float)
        tau = self.bias_tau() + kp * (np.asarray(q_des, dtype=float) - q) - kd * dq
        return tau

    def step_physics(self, tau_arm, hold_driver: bool = False) -> dict:
        tau = _as_vec(tau_arm, 7)
        ids = self.index.arm_actuator_ids
        lo = self.model.actuator_ctrlrange[ids, 0]
        hi = self.model.actuator_ctrlrange[ids, 1]
        cmd = np.clip(tau, lo, hi)
        limited = np.abs(tau - cmd) > 1e-12
        self._write_external()
        self._write_head(hold_driver)
        self.data.ctrl[ids] = cmd
        t_before = float(self.data.time)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tick += 1
        self.last_tau = cmd.copy()
        dq = np.array(self.data.qvel[self.index.arm_dof_adr], dtype=float)
        limit = float(self.config["velocity_soft_limit_rad_s"])
        if np.any(np.abs(dq) > limit):
            self.velocity_fault = True
        state = self.snapshot()
        state["tau_requested"] = tau.copy()
        state["tau_command"] = cmd.copy()
        state["software_limited"] = limited.copy()
        state["input_interval"] = (t_before, float(self.data.time))
        state["head_gravcomp"] = self._head_gravcomp.copy()
        state["head_tau"] = self._head_tau.copy()
        peak = 0.0
        for row in state["contacts"]:
            peak = max(peak, float(np.linalg.norm(row["force_on_geom2_world"])))
        self.peak_contact_force = max(self.peak_contact_force, peak)
        self.contact_events.append(
            {"tick": self.tick, "ncon": len(state["contacts"]), "min_dist": min_distance(state["contacts"])}
        )
        return state

    def snapshot(self) -> dict:
        idx = self.index
        raw = read_raw_wrench(self.model, self.data, idx.force_adr, idx.torque_adr)
        rot = site_rotation(self.data, idx.site_ids["ft_site"])
        p_site = site_position(self.data, idx.site_ids["ft_site"])
        p_tcp = site_position(self.data, idx.site_ids["tcp"])
        wrench_w, wrench_tcp = world_and_tcp_wrench(raw, rot, p_site, p_tcp, self.wrench_sign)
        food_adr = idx.food_qpos_adr
        contacts = read_contacts(self.model, self.data)
        warnings = [int(self.data.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))]
        q = np.array(self.data.qpos[idx.arm_qpos_adr], dtype=float).copy()
        dq = np.array(self.data.qvel[idx.arm_dof_adr], dtype=float).copy()
        return {
            "tick": int(self.tick),
            "sim_time": float(self.data.time),
            "episode_time": float(self.data.time - self.time_offset),
            "q": q,
            "dq": dq,
            "tcp_pos": p_tcp,
            "tcp_mat": site_rotation(self.data, idx.site_ids["tcp"]),
            "ft_pos": p_site,
            "ft_mat": rot,
            "food_pos": np.array(self.data.qpos[food_adr : food_adr + 3], dtype=float).copy(),
            "food_quat": np.array(self.data.qpos[food_adr + 3 : food_adr + 7], dtype=float).copy(),
            "food_vel": np.array(self.data.qvel[idx.food_dof_adr : idx.food_dof_adr + 3], dtype=float).copy(),
            "head_q": np.array(self.data.qpos[idx.head_qpos_adr], dtype=float).copy(),
            "head_dq": np.array(self.data.qvel[idx.head_dof_adr], dtype=float).copy(),
            "jaw_q": float(self.data.qpos[idx.jaw_qpos_adr]),
            "jaw_dq": float(self.data.qvel[idx.jaw_dof_adr]),
            "raw_wrench_sensor": raw,
            "wrench_world": wrench_w,
            "tcp_wrench_world": wrench_tcp,
            "compensated_wrench": None,
            "contacts": contacts,
            "warnings": warnings,
            "qfrc_actuator": np.array(self.data.qfrc_actuator[idx.arm_dof_adr], dtype=float).copy(),
            "actuator_force": np.array(self.data.actuator_force[idx.arm_actuator_ids], dtype=float).copy(),
            "velocity_fault": bool(self.velocity_fault),
            "finite": bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))),
        }

    def _set_arm(self, q) -> None:
        q = _as_vec(q, 7)
        ranges = self.model.jnt_range[self.index.arm_joint_ids]
        if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
            raise ValueError(f"arm qpos out of range: {q}")
        self.data.qpos[self.index.arm_qpos_adr] = q
        self.data.qvel[self.index.arm_dof_adr] = 0

    def _set_food(self, pos, quat) -> None:
        adr = self.index.food_qpos_adr
        self.data.qpos[adr : adr + 3] = pos
        quat = np.asarray(quat, dtype=float)
        quat = quat / np.linalg.norm(quat)
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[self.index.food_dof_adr : self.index.food_dof_adr + 6] = 0

    def _zero_head(self) -> None:
        self.data.qpos[self.index.head_qpos_adr] = 0
        self.data.qvel[self.index.head_dof_adr] = 0
        self.data.qpos[self.index.jaw_qpos_adr] = float(self.config["head"]["jaw_center_rad"])
        self.data.qvel[self.index.jaw_dof_adr] = 0

    def _illegal(self) -> bool:
        contacts = read_contacts(self.model, self.data)
        if min_distance(contacts) < -1e-3:
            return True
        for row in contacts:
            pair = tuple(sorted((row["group1"], row["group2"])))
            if pair in _FORBIDDEN_RESET or (
                row["group1"] in _ARM_TABLE_GROUPS and row["group2"] in {"table", "plate", "mouth"}
            ):
                return True
        return False

    def reset(self, seed: int | None = None, preset: str = "food_on_plate") -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.restore_model_options()
        mujoco.mj_resetData(self.model, self.data)
        self._clear_python_state()
        spec = self.config["presets"][preset]
        self._set_arm(spec["qpos"])
        self._zero_head()
        mujoco.mj_forward(self.model, self.data)
        placed = False
        last_pos = None
        for _ in range(8):
            pos, quat = self._sample_food(preset)
            last_pos = pos
            self._set_food(pos, quat)
            mujoco.mj_forward(self.model, self.data)
            if not self._illegal():
                placed = True
                break
        if not placed:
            raise RuntimeError(f"reset failed for preset {preset}, last food pos {last_pos}")
        settle = int(self.config["settle_steps"])
        for _ in range(settle):
            self.step_physics(self.bias_tau(), hold_driver=True)
        self.settle_steps = settle
        self.contact_events.clear()
        self.peak_contact_force = 0.0
        self.velocity_fault = False
        self.tick = 0
        self.time_offset = float(self.data.time)
        self.data.ctrl[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        self.applied_wrench = None
        self.last_tau[:] = 0
        mujoco.mj_forward(self.model, self.data)
        state = self.snapshot()
        state["settle_steps"] = settle
        state["preset"] = preset
        state["seed"] = seed
        return state

    def _sample_food(self, preset: str):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if preset == "food_on_spoon":
            tcp = site_position(self.data, self.index.site_ids["tcp"])
            rot = site_rotation(self.data, self.index.site_ids["tcp"])
            jitter = self.rng.uniform(-0.002, 0.002, size=2)
            pos = tcp + rot @ np.array([jitter[0], jitter[1], 0.007])
            return pos, quat
        if preset == "near_mouth":
            return np.array([0.45, -0.18, 0.03]), quat
        jitter = self.rng.uniform(-0.025, 0.025, size=2)
        pos = np.array([0.45 + jitter[0], -0.18 + jitter[1], 0.03])
        return pos, quat
