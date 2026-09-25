"""Twist limits, reference integration, and blocked-direction anti-windup."""

from __future__ import annotations

import numpy as np

from feedingrobot.controllers.so3 import integrate_rotation, orientation_error, so3_log


def _limit_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= limit or n < 1e-15:
        return v.copy()
    return v * (limit / n)


def _accel_limit(prev: np.ndarray, target: np.ndarray, accel: float, dt: float) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(prev, dtype=float)
    return np.asarray(prev, dtype=float) + _limit_norm(delta, float(accel) * float(dt))


class ReferenceShaper:
    def __init__(self, config: dict):
        self.config = config
        self.feature_length = float(config["feature_length_m"])
        blocked = config["blocked"]
        self.block_force = float(blocked["force_n"])
        self.block_error = float(blocked["error_m"])
        self.block_confirm = float(blocked["confirm_s"])
        self.sat_confirm = float(blocked["sat_s"])
        self.dev_confirm = float(blocked["dev_s"])
        self.box = float(config["workspace_box_m"])
        self.reset_motion()

    def reset_motion(self) -> None:
        self.p_ref = None
        self.r_ref = None
        self.v_lin = np.zeros(3)
        self.v_ang = np.zeros(3)
        self.anchor = None
        self.q0 = None
        self._block_time = 0.0
        self._sat_time = 0.0
        self._dev_time = 0.0
        self.blocked = False
        self.reanchored = False
        self.clipped_offset = np.zeros(3)

    def anchor_pose(self, pos: np.ndarray, rot: np.ndarray, q: np.ndarray) -> None:
        self.p_ref = np.asarray(pos, dtype=float).copy()
        self.r_ref = np.asarray(rot, dtype=float).reshape(3, 3).copy()
        self.anchor = self.p_ref.copy()
        self.q0 = np.asarray(q, dtype=float).copy()
        self.v_lin[:] = 0
        self.v_ang[:] = 0
        self._block_time = 0.0
        self._sat_time = 0.0
        self._dev_time = 0.0
        self.blocked = False
        self.reanchored = False
        self.clipped_offset[:] = 0

    def reanchor(self, pos: np.ndarray, rot: np.ndarray) -> None:
        jump = np.asarray(pos, dtype=float) - self.p_ref
        self.clipped_offset = jump.copy()
        self.p_ref = np.asarray(pos, dtype=float).copy()
        self.r_ref = np.asarray(rot, dtype=float).reshape(3, 3).copy()
        self.v_lin[:] = 0
        self.v_ang[:] = 0
        self.reanchored = True
        self._block_time = 0.0
        self._dev_time = 0.0

    def shape(
        self,
        twist: np.ndarray,
        pos: np.ndarray,
        rot: np.ndarray,
        force: np.ndarray,
        dt: float,
        limits: dict,
        scale: float,
        joint_scale: float,
        active: bool,
        saturated: bool,
    ) -> dict:
        """Integrate one step. `active` is false during faults, power-on, and expired commands."""
        self.reanchored = False
        twist = np.asarray(twist, dtype=float).reshape(6)
        if not active:
            twist = np.zeros(6)
        scale = float(np.clip(scale, 0.0, 1.0)) * float(np.clip(joint_scale, 0.0, 1.0))
        lin = _limit_norm(twist[:3], limits["v"]) * scale
        ang = _limit_norm(twist[3:], limits["w"]) * scale
        lin = _accel_limit(self.v_lin, lin, limits["a"], dt)
        ang = _accel_limit(self.v_ang, ang, limits["alpha"], dt)
        lin, ang, blocked_now = self._suppress(lin, ang, pos, rot, force, dt, limits, saturated, active)
        self.v_lin = lin
        self.v_ang = ang
        prev_p = self.p_ref.copy()
        prev_r = self.r_ref.copy()
        if self.anchor is not None and self.box > 0:
            trial = prev_p + lin * dt
            delta = trial - self.anchor
            excess = np.abs(delta) - self.box
            if np.any(excess > 0):
                for i in range(3):
                    if excess[i] > 0 and abs(lin[i]) > 1e-15:
                        room = self.box - abs(prev_p[i] - self.anchor[i])
                        lin[i] = np.sign(lin[i]) * max(room, 0.0) / dt
                self.v_lin = lin
        self.p_ref = prev_p + lin * dt
        self.r_ref = integrate_rotation(prev_r, ang, dt)
        self._clip_deviation(pos, rot, limits, dt)
        if self.reanchored:
            v_ref = np.zeros(6)
        else:
            v_ref = np.concatenate([(self.p_ref - prev_p) / dt, so3_log(self.r_ref @ prev_r.T) / dt])
        return {
            "twist_limited": np.concatenate([lin, ang]),
            "p_ref": self.p_ref.copy(),
            "r_ref": self.r_ref.copy(),
            "v_ref": v_ref,
            "blocked": self.blocked,
            "blocked_now": blocked_now,
            "reanchored": self.reanchored,
            "clipped_offset": self.clipped_offset.copy(),
            "scale": scale,
        }

    def _suppress(self, lin, ang, pos, rot, force, dt, limits, saturated, active):
        err = self.p_ref - np.asarray(pos, dtype=float)
        n = float(np.linalg.norm(lin))
        blocked_now = False
        if active and n > 1e-6:
            direction = lin / n
            oppose = -float(np.dot(np.asarray(force, dtype=float).reshape(3), direction))
            lag = float(np.dot(err, direction))
            if oppose > self.block_force and lag > self.block_error:
                self._block_time += dt
            else:
                self._block_time = 0.0
            if self._block_time >= self.block_confirm - 1e-15:
                lin = lin - direction * float(np.dot(lin, direction))
                blocked_now = True
                self.blocked = True
        else:
            self._block_time = 0.0
        if saturated:
            self._sat_time += dt
        else:
            self._sat_time = 0.0
        dev = float(np.linalg.norm(err))
        ori = float(np.linalg.norm(orientation_error(self.r_ref, rot)))
        at_cap = dev >= limits["pos_dev"] - 1e-6 or ori >= limits["rot_dev"] - 1e-5
        if at_cap:
            self._dev_time += dt
        else:
            self._dev_time = 0.0
        if self._sat_time >= self.sat_confirm - 1e-15 or self._dev_time >= self.dev_confirm - 1e-15:
            self.reanchor(pos, rot)
            lin = np.zeros(3)
            ang = np.zeros(3)
            blocked_now = True
            self.blocked = True
        return lin, ang, blocked_now

    def _clip_deviation(self, pos, rot, limits, dt):
        err = self.p_ref - np.asarray(pos, dtype=float)
        n = float(np.linalg.norm(err))
        if n > limits["pos_dev"] and n > 0:
            self.clipped_offset = err * (1.0 - limits["pos_dev"] / n)
            self.p_ref = np.asarray(pos, dtype=float) + err * (limits["pos_dev"] / n)
            if n - limits["pos_dev"] > 1e-4:
                self.reanchored = True
                self.v_lin[:] = 0
                self.v_ang[:] = 0
        e_r = orientation_error(self.r_ref, rot)
        ang = float(np.linalg.norm(e_r))
        if ang > limits["rot_dev"] > 0:
            self.r_ref = integrate_rotation(rot, e_r, limits["rot_dev"] / ang)
            self.reanchored = True
            self.v_lin[:] = 0
            self.v_ang[:] = 0
            del dt
