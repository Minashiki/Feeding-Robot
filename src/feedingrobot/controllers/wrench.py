"""Tool-load compensation, causal filtering, and a fixed delivery delay."""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from feedingrobot.sim.sensors import rotate_wrench, shift_torque


def _principal_world(model, data, body_id: int) -> np.ndarray:
    rot = np.array(data.ximat[body_id], dtype=float).reshape(3, 3)
    return rot @ np.diag(np.array(model.body_inertia[body_id], dtype=float)) @ rot.T


def body_spatial(model, data, body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """COM twist and coordinate acceleration for one body.

    On MuJoCo 3.14.0, mj_objectVelocity/Acceleration for mjOBJ_BODY already
    report the center of mass. The linear acceleration uses the specific-force
    convention, so gravity is added to obtain coordinate acceleration.
    """
    if hasattr(mujoco, "mj_rnePostConstraint"):
        mujoco.mj_rnePostConstraint(model, data)
    vel = np.zeros(6)
    acc = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body_id), vel, 0)
    mujoco.mj_objectAcceleration(model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body_id), acc, 0)
    omega = vel[:3].copy()
    v_com = vel[3:].copy()
    alpha = acc[:3].copy()
    a_com = acc[3:].copy() + np.array(model.opt.gravity, dtype=float)
    return omega, v_com, alpha, a_com


def tool_body_ids(model, root_id: int) -> list[int]:
    ids = []
    for body_id in range(model.nbody):
        cursor = body_id
        while cursor != 0 and cursor != root_id:
            cursor = int(model.body_parentid[cursor])
        if cursor == root_id:
            ids.append(body_id)
    return ids


def predict_tool_wrench(model, data, body_ids: list[int], p_tcp: np.ndarray) -> np.ndarray:
    gravity = np.array(model.opt.gravity, dtype=float)
    force = np.zeros(3)
    moment = np.zeros(3)
    for body_id in body_ids:
        mass = float(model.body_mass[body_id])
        if mass <= 0:
            continue
        com = np.array(data.xipos[body_id], dtype=float)
        omega, _v, alpha, a_com = body_spatial(model, data, body_id)
        specific = gravity - a_com
        f = mass * specific
        inertia = _principal_world(model, data, body_id)
        moment = moment + np.cross(com - p_tcp, f) - inertia @ alpha - np.cross(omega, inertia @ omega)
        force = force + f
    return np.concatenate([force, moment])


class WrenchPipeline:
    def __init__(self, model, config: dict, tool_body_id: int):
        self.model = model
        self.body_ids = tool_body_ids(model, int(tool_body_id))
        self.fc = float(config["filter_hz"])
        self.delay_s = float(config["delay_s"])
        noise = config.get("noise", {})
        self.noise_on = bool(noise.get("enable", False))
        self.force_std = float(noise.get("force_std", 0.02))
        self.torque_std = float(noise.get("torque_std", 0.002))
        self.seed = int(noise.get("seed", 0))
        self.bias_sensor = np.zeros(6)
        self.bias_ready = False
        self._bias_buf: list[np.ndarray] = []
        self._stable = 0.0
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        self.generation += 1
        self.filtered = np.zeros(6)
        self.filter_ready = False
        self._filter_time = 0.0
        self.filter_updates = 0
        self._buffer: deque[tuple[float, np.ndarray, bool]] = deque()
        self.rng = np.random.default_rng(self.seed)
        self._prev_rot = None
        self._prev_omega = None
        self.bias_sensor = np.zeros(6)
        self.bias_ready = False
        self._bias_buf = []
        self._stable = 0.0
        self._cache_key = None
        self._cache = None
        self._kinematics_prev = None
        self._kinematics_time = None

    def measure(self, model, data, raw, rot_ws, p_site, p_tcp, sign: float) -> dict:
        """Pure compensation. Does not touch the filter, noise, or delay queue."""
        from feedingrobot.sim.sensors import world_and_tcp_wrench

        if np.all(np.isfinite(raw)):
            wrench_w, wrench_tcp = world_and_tcp_wrench(raw, rot_ws, p_site, p_tcp, sign)
        else:
            wrench_w, wrench_tcp = np.full(6, np.nan), np.full(6, np.nan)
        predicted = predict_tool_wrench(model, data, self.body_ids, p_tcp)
        bias_f, bias_m = rotate_wrench(self.bias_sensor[:3], self.bias_sensor[3:], rot_ws, sign)
        bias_tcp = np.concatenate([bias_f, shift_torque(bias_m, bias_f, p_site, p_tcp)])
        compensated = wrench_tcp - predicted - bias_tcp
        velocities = []
        for body_id in self.body_ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            velocities.append(velocity)
        return {
            "raw_wrench_sensor": np.asarray(raw, dtype=float).copy(),
            "wrench_world_at_ft": wrench_w.copy(),
            "tcp_wrench_world": wrench_tcp.copy(),
            "tool_load_predicted": predicted.copy(),
            "compensated_wrench_tcp": compensated.copy(),
            "exact_model": compensated.copy(),
            "finite": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(compensated))),
            "bias_wrench_tcp": bias_tcp,
            "tool_velocity": np.asarray(velocities),
            "tool_com": np.array(data.xipos[self.body_ids], dtype=float),
            "tool_inertia": np.array([_principal_world(model, data, body_id) for body_id in self.body_ids]),
            "tcp_position": np.asarray(p_tcp, dtype=float).copy(),
        }

    def update_sample(self, measurement: dict, tick: int, sample_time: float, dt: float) -> dict:
        """Advance the observation path once for a new (generation, tick, time)."""
        key = (int(self.generation), int(tick), float(sample_time))
        if key == self._cache_key:
            return self._cache
        compensated = np.asarray(measurement["compensated_wrench_tcp"], dtype=float)
        finite = bool(measurement["finite"] and np.all(np.isfinite(compensated))
                      and np.all(np.isfinite(measurement.get("raw_wrench_sensor", []))))
        if not finite:
            result = {**measurement, "finite": False, "filtered_wrench_tcp": self.filtered.copy(),
                      "delivered_wrench_tcp": np.full(6, np.nan), "sample_time": float(sample_time),
                      "sample_tick": int(tick), "deliver_time": None, "age_s": None, "valid": False,
                      "warmup": not self.filter_ready, "bias_ready": self.bias_ready,
                      "filter_updates": self.filter_updates, "estimated_valid": False,
                      "estimated_kinematics": np.full(6, np.nan)}
            self._cache_key, self._cache = key, result
            return result
        if self.filter_updates == 0:
            self.filtered = compensated.copy()
        else:
            alpha = 1.0 - np.exp(-2.0 * np.pi * self.fc * float(dt))
            self.filtered = self.filtered + alpha * (compensated - self.filtered)
            self._filter_time += float(dt)
        self.filter_updates += 1
        self.filter_ready = self.filter_updates > 1 and self._filter_time >= 1.0 / max(self.fc, 1e-6)
        observed = self.filtered.copy()
        if self.noise_on and self.filter_ready:
            observed[:3] += self.rng.normal(0.0, self.force_std, size=3)
            observed[3:] += self.rng.normal(0.0, self.torque_std, size=3)
        self._buffer.append((float(sample_time), observed.copy(), self.filter_ready))
        deliver_t = float(sample_time) - self.delay_s
        delivered = None
        sample_t = None
        sample_ready = False
        while self._buffer and self._buffer[0][0] <= deliver_t + 1e-12:
            sample_t, delivered, sample_ready = self._buffer.popleft()
        valid = delivered is not None and sample_ready and bool(np.all(np.isfinite(delivered)))
        if delivered is None:
            delivered = np.full(6, np.nan)
            age = None
        else:
            age = float(sample_time) - float(sample_t)
        estimated, estimated_valid = self._estimated(measurement, float(sample_time))
        result = {
            **measurement,
            "filtered_wrench_tcp": self.filtered.copy(),
            "delivered_wrench_tcp": np.asarray(delivered, dtype=float).copy(),
            "sample_time": float(sample_time),
            "sample_tick": int(tick),
            "deliver_time": float(sample_time) if valid else None,
            "age_s": age,
            "valid": bool(valid),
            "warmup": not self.filter_ready,
            "bias_ready": self.bias_ready,
            "filter_updates": int(self.filter_updates),
            "estimated_valid": estimated_valid,
            "estimated_kinematics": estimated,
        }
        self._cache_key = key
        self._cache = result
        return result

    def _estimated(self, measurement: dict, t: float) -> tuple[np.ndarray, bool]:
        """Differentiate each body's measured world velocity; never read qacc."""
        if "tool_velocity" not in measurement:
            return np.full(6, np.nan), False
        velocity = np.asarray(measurement["tool_velocity"], dtype=float)
        previous, previous_time = self._kinematics_prev, self._kinematics_time
        self._kinematics_prev, self._kinematics_time = velocity.copy(), t
        if previous is None or previous_time is None or t <= previous_time:
            return np.full(6, np.nan), False
        acceleration = (velocity - previous) / (t - previous_time)
        force, moment = np.zeros(3), np.zeros(3)
        for index, body_id in enumerate(self.body_ids):
            mass = float(self.model.body_mass[body_id])
            omega, alpha = velocity[index, :3], acceleration[index, :3]
            part = mass * (self.model.opt.gravity - acceleration[index, 3:])
            inertia = measurement["tool_inertia"][index]
            force += part
            moment += (np.cross(measurement["tool_com"][index] - measurement["tcp_position"], part)
                       - inertia @ alpha - np.cross(omega, inertia @ omega))
        estimated = measurement["tcp_wrench_world"] - measurement["bias_wrench_tcp"] - np.concatenate([force, moment])
        return estimated, bool(np.all(np.isfinite(estimated)))

    def update(self, model, data, raw, rot_ws, p_site, p_tcp, sign: float, t: float, dt: float, calibrate: bool = False, stable: bool = False) -> dict:
        del calibrate, stable
        measured = self.measure(model, data, raw, rot_ws, p_site, p_tcp, sign)
        return self.update_sample(measured, tick=0, sample_time=float(t), dt=float(dt))


def so3_omega(rot: np.ndarray, prev: np.ndarray, dt: float) -> np.ndarray:
    from feedingrobot.controllers.so3 import so3_log

    if prev is None:
        return np.zeros(3)
    return so3_log(rot @ prev.T) / dt
