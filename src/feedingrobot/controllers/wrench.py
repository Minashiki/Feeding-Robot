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
    """World angular velocity, COM linear velocity, angular acc, COM linear acc.

    mj_objectAcceleration returns classical rot:lin of the body frame. The COM
    acceleration adds the rigid offset from that frame origin.
    """
    vel = np.zeros(6)
    acc = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body_id), vel, 0)
    mujoco.mj_objectAcceleration(model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body_id), acc, 0)
    omega = vel[:3].copy()
    origin = np.array(data.xpos[body_id], dtype=float)
    com = np.array(data.xipos[body_id], dtype=float)
    offset = com - origin
    alpha = acc[:3].copy()
    # At rest this binding returns +9.81 m/s^2 upward, i.e. specific force.
    # Coordinate acceleration is that reading plus gravity (0 when the body is still).
    a_origin = acc[3:].copy() + np.array(model.opt.gravity, dtype=float)
    a_com = a_origin + np.cross(alpha, offset) + np.cross(omega, np.cross(omega, offset))
    v_com = vel[3:].copy() + np.cross(omega, offset)
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
        self.reset()

    def reset(self) -> None:
        self.filtered = np.zeros(6)
        self.filter_ready = False
        self._filter_time = 0.0
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self.rng = np.random.default_rng(self.seed)
        self._prev_com = None
        self._prev_rot = None
        self._prev_omega = None
        self._prev_v = None
        self._prev_time = None
        self.bias_sensor = np.zeros(6)
        self.bias_ready = False
        self._bias_buf = []
        self._stable = 0.0

    def _estimated(self, data, p_tcp: np.ndarray, t: float) -> np.ndarray:
        """Causal COM finite difference. Inertia terms stay on the exact-model path."""
        gravity = np.array(self.model.opt.gravity, dtype=float)
        force = np.zeros(3)
        moment = np.zeros(3)
        coms = []
        for body_id in self.body_ids:
            mass = float(self.model.body_mass[body_id])
            if mass <= 0:
                continue
            coms.append((body_id, mass, np.array(data.xipos[body_id], dtype=float)))
        key = tuple(np.round(np.concatenate([row[2] for row in coms]), 9)) if coms else ()
        accel = np.zeros(3)
        if self._prev_com is not None and self._prev_time is not None:
            dt = max(float(t - self._prev_time), 1e-9)
            vel = (coms[0][2] - self._prev_com) / dt if coms else np.zeros(3)
            if self._prev_v is not None:
                accel = (vel - self._prev_v) / dt
            self._prev_v = vel
        if coms:
            self._prev_com = coms[0][2].copy()
        self._prev_time = float(t)
        for _body_id, mass, com in coms:
            part = mass * (gravity - accel)
            force = force + part
            moment = moment + np.cross(com - p_tcp, part)
        del key
        return np.concatenate([force, moment])

    def update(self, model, data, raw: np.ndarray, rot_ws: np.ndarray, p_site: np.ndarray, p_tcp: np.ndarray, sign: float, t: float, dt: float, calibrate: bool, stable: bool) -> dict:
        from feedingrobot.sim.sensors import world_and_tcp_wrench

        wrench_w, wrench_tcp = world_and_tcp_wrench(raw, rot_ws, p_site, p_tcp, sign)
        predicted = predict_tool_wrench(model, data, self.body_ids, p_tcp)
        estimated = self._estimated(data, p_tcp, t)
        if calibrate and not self.bias_ready:
            if stable:
                self._stable += dt
                if self._stable >= 0.5:
                    self._bias_buf.append(np.asarray(raw, dtype=float).copy())
                    if self._stable >= 1.5:
                        mean_raw = np.mean(self._bias_buf, axis=0)
                        # Residual in the sensor frame after removing the known tool load.
                        load_s_force = rot_ws.T @ predicted[:3] / sign if sign != 0 else predicted[:3]
                        # raw maps by sign*R, so sensor residual = raw - R.T @ (sign-inverse of predicted).
                        pred_sensor_f = rot_ws.T @ (predicted[:3] / sign)
                        pred_sensor_m = rot_ws.T @ (predicted[3:] / sign)
                        self.bias_sensor = mean_raw - np.concatenate([pred_sensor_f, pred_sensor_m])
                        self.bias_ready = True
            else:
                self._stable = 0.0
                self._bias_buf = []
        bias_f, bias_m = rotate_wrench(self.bias_sensor[:3], self.bias_sensor[3:], rot_ws, sign)
        bias_tcp_m = shift_torque(bias_m, bias_f, p_site, p_tcp)
        bias_tcp = np.concatenate([bias_f, bias_tcp_m])
        compensated = wrench_tcp - predicted - bias_tcp
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.fc * dt)
        self.filtered = self.filtered + alpha * (compensated - self.filtered)
        self._filter_time += dt
        self.filter_ready = self._filter_time >= 1.0 / max(self.fc, 1e-6)
        observed = self.filtered.copy()
        if self.noise_on and self.filter_ready:
            observed[:3] += self.rng.normal(0.0, self.force_std, size=3)
            observed[3:] += self.rng.normal(0.0, self.torque_std, size=3)
        self._buffer.append((t, observed.copy()))
        deliver_t = t - self.delay_s
        delivered = None
        sample_t = None
        while self._buffer and self._buffer[0][0] <= deliver_t + 1e-12:
            sample_t, delivered = self._buffer.popleft()
        valid = delivered is not None and self.filter_ready
        if delivered is None:
            delivered = np.full(6, np.nan)
            age = np.nan
        else:
            age = t - sample_t
        finite = bool(np.all(np.isfinite(compensated)))
        return {
            "raw_wrench_sensor": np.asarray(raw, dtype=float).copy(),
            "wrench_world_at_ft": wrench_w.copy(),
            "tcp_wrench_world": wrench_tcp.copy(),
            "tool_load_predicted": predicted.copy(),
            "compensated_wrench_tcp": compensated.copy(),
            "filtered_wrench_tcp": self.filtered.copy(),
            "delivered_wrench_tcp": np.asarray(delivered, dtype=float).copy(),
            "sample_time": None if sample_t is None else float(sample_t),
            "deliver_time": float(t) if valid else None,
            "age_s": None if sample_t is None else float(age),
            "valid": bool(valid and finite),
            "warmup": not self.filter_ready,
            "bias_ready": self.bias_ready,
            "exact_model": compensated.copy(),
            "estimated_kinematics": (wrench_tcp - estimated - bias_tcp).copy(),
        }


def so3_omega(rot: np.ndarray, prev: np.ndarray, dt: float) -> np.ndarray:
    from feedingrobot.controllers.so3 import so3_log

    if prev is None:
        return np.zeros(3)
    return so3_log(rot @ prev.T) / dt
