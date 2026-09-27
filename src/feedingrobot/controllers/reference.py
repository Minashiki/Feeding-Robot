"""Twist limits, reference integration, and blocked-direction anti-windup."""

from __future__ import annotations

from collections import deque

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
        self.progress_window = float(blocked["progress_window_s"])
        self.progress_ratio = float(blocked["progress_ratio"])
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
        self._block_deficit = 0.0
        self._sat_time = 0.0
        self._dev_time = 0.0
        self.blocked = False
        self.reanchored = False
        self.clipped_offset = np.zeros(3)
        self.constraint_reset = None
        self.reference_correction_pos = np.zeros(3)
        self.reference_correction_rot = 0.0
        self.correction_reason = None
        self.block_correction_pos = np.zeros(3)
        self._sat_time = 0.0
        self.reset_progress()

    def reset_progress(self) -> None:
        self._positions = deque()
        self._progress_time = 0.0
        self._progress_direction = None
        self._progress_velocity = np.zeros(3)
        self._progress_span = 0.0
        self._block_time = 0.0
        self._block_deficit = 0.0

    def _measure_progress(self, pos, twist, dt, enabled):
        speed = float(np.linalg.norm(twist[:3]))
        if not enabled or speed <= 1e-6:
            self.reset_progress()
            return
        direction = twist[:3] / speed
        if self._progress_direction is not None and direction @ self._progress_direction <= 1e-12:
            self.reset_progress()
        self._progress_direction = direction.copy()
        self._progress_time += dt
        self._positions.append((self._progress_time, np.asarray(pos, dtype=float).copy()))
        start = self._progress_time - self.progress_window
        while len(self._positions) > 1 and self._positions[1][0] <= start + 1e-12:
            self._positions.popleft()
        time, previous = self._positions[0]
        self._progress_span = self._progress_time - time
        self._progress_velocity = ((np.asarray(pos) - previous) / self._progress_span
                                   if self._progress_span > 1e-12 else np.zeros(3))

    def anchor_pose(self, pos: np.ndarray, rot: np.ndarray, q: np.ndarray) -> None:
        self.reset_progress()
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
        self.constraint_reset = None

    def reanchor(self, pos: np.ndarray, rot: np.ndarray) -> None:
        self.reset_progress()
        jump = np.asarray(pos, dtype=float) - self.p_ref
        self.clipped_offset = jump.copy()
        self.p_ref = np.asarray(pos, dtype=float).copy()
        self.r_ref = np.asarray(rot, dtype=float).reshape(3, 3).copy()
        self.v_lin[:] = 0
        self.v_ang[:] = 0
        self.reanchored = True
        self._block_time = 0.0
        self._dev_time = 0.0

    def _project_history(self, limits: dict) -> str | None:
        reason = None
        if float(np.linalg.norm(self.v_lin)) > float(limits["v"]) + 1e-12:
            self.v_lin = _limit_norm(self.v_lin, float(limits["v"]))
            reason = "linear_speed"
        if float(np.linalg.norm(self.v_ang)) > float(limits["w"]) + 1e-12:
            self.v_ang = _limit_norm(self.v_ang, float(limits["w"]))
            reason = "angular_speed" if reason is None else reason + "+angular_speed"
        return reason

    def _pack(self, lin, ang, scale: float, blocked_now: bool) -> dict:
        return {
            "twist_limited": np.concatenate([np.asarray(lin, dtype=float).reshape(3), np.asarray(ang, dtype=float).reshape(3)]),
            "p_ref": None if self.p_ref is None else self.p_ref.copy(),
            "r_ref": None if self.r_ref is None else self.r_ref.copy(),
            "v_ref": np.zeros(6),
            "blocked": self.blocked,
            "blocked_now": blocked_now,
            "reanchored": self.reanchored,
            "clipped_offset": self.clipped_offset.copy(),
            "scale": float(scale),
            "constraint_reset": self.constraint_reset,
            "v_hist": np.concatenate([self.v_lin, self.v_ang]),
        }

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
        execution: str = "run",
        project=None,
    ) -> dict:
        """Integrate one step.

        `execution` separates a normal zero command from a hard stop.
        A tighter speed bound projects the stored reference velocity first.
        """
        self.reanchored = False
        self.constraint_reset = None
        self.reference_correction_pos = np.zeros(3)
        self.reference_correction_rot = 0.0
        self.correction_reason = None
        self.block_correction_pos = np.zeros(3)
        beta = 1.0
        twist = np.asarray(twist, dtype=float).reshape(6)
        self._measure_progress(pos, twist, dt, active and execution in {"run", "zero"}
                               and not saturated and float(scale) > 0.0)
        lin = np.zeros(3)
        ang = np.zeros(3)
        candidate = np.zeros(6)
        blocked_now = False
        integrate = False
        envelope = dict(limits)
        if execution in {"power_on", "stop", "prohibit"} or (not active and execution != "zero") or saturated or float(scale) <= 0.0:
            self.v_lin[:] = 0.0
            self.v_ang[:] = 0.0
            if saturated:
                self.constraint_reset = "saturation_pause"
            elif execution == "prohibit" or float(scale) <= 0.0:
                self.constraint_reset = "prohibit"
        else:
            factor = float(np.clip(scale, 0.0, 1.0)) * float(np.clip(joint_scale, 0.0, 1.0))
            envelope["v"] = float(limits["v"]) * factor
            envelope["w"] = float(limits["w"]) * factor
            if not active:
                twist = np.zeros(6)
            self.constraint_reset = self._project_history(envelope)
            lin = _limit_norm(twist[:3], envelope["v"])
            ang = _limit_norm(twist[3:], envelope["w"])
            lin = _accel_limit(self.v_lin, lin, limits["a"], dt)
            ang = _accel_limit(self.v_ang, ang, limits["alpha"], dt)
            lin = _limit_norm(lin, envelope["v"])
            ang = _limit_norm(ang, envelope["w"])
            lin, ang, blocked_now = self._suppress(lin, ang, pos, rot, force, dt, limits, saturated, active, twist[:3])
            if self.anchor is not None and self.box > 0:
                trial = self.p_ref + lin * dt
                excess = np.abs(trial - self.anchor) - self.box
                for i in range(3):
                    if excess[i] > 0 and abs(lin[i]) > 1e-15:
                        room = self.box - abs(self.p_ref[i] - self.anchor[i])
                        lin[i] = np.sign(lin[i]) * max(room, 0.0) / dt
            candidate = np.concatenate([lin, ang])
            if project is not None:
                beta = float(np.clip(project(candidate), 0.0, 1.0))
                lin = lin * beta
                ang = ang * beta
                if beta <= 1e-15:
                    lin = np.zeros(3)
                    ang = np.zeros(3)
                    self.constraint_reset = "joint_limit"
            integrate = True
        prev_p = None if self.p_ref is None else self.p_ref.copy()
        prev_r = None if self.r_ref is None else self.r_ref.copy()
        if integrate and prev_p is not None:
            self.p_ref = prev_p + lin * dt
            self.r_ref = integrate_rotation(prev_r, ang, dt)
        self._clip_deviation(pos, rot, limits, dt)
        if float(np.linalg.norm(self.reference_correction_pos)) > 1e-15 or self.reference_correction_rot > 1e-15:
            self.reset_progress()
            self.correction_reason = "saturation_deviation"
            self.reanchored = True
        if self.reanchored or not integrate:
            self.v_lin[:] = 0.0
            self.v_ang[:] = 0.0
            v_ref = np.zeros(6)
        else:
            self.v_lin = lin
            self.v_ang = ang
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
            "constraint_reset": self.constraint_reset,
            "v_hist": np.concatenate([self.v_lin, self.v_ang]),
            "reference_correction_pos": np.array(self.reference_correction_pos, dtype=float).copy(),
            "reference_correction_rot": float(self.reference_correction_rot),
            "correction_reason": self.correction_reason,
            "beta": float(beta),
            "candidate": np.asarray(candidate, dtype=float).copy(),
            "block_progress": np.r_[self._progress_velocity, self._progress_span, self._block_time, self._block_deficit],
            "block_correction_pos": self.block_correction_pos.copy(),
        }

    def _suppress(self, lin, ang, pos, rot, force, dt, limits, saturated, active, command):
        err = self.p_ref - np.asarray(pos, dtype=float)
        n = float(np.linalg.norm(command))
        blocked_now = False
        if active and n > 1e-6:
            direction = command / n
            oppose = -float(np.dot(np.asarray(force, dtype=float).reshape(3), direction))
            lag = float(np.dot(err, direction))
            normal = -np.asarray(force, dtype=float) / max(float(np.linalg.norm(force)), 1e-30)
            requested = float(command @ normal)
            progress = float(self._progress_velocity @ normal)
            stalled = (self._progress_span >= self.progress_window - 1e-12
                       and requested > 1e-6
                       and progress <= self.progress_ratio * requested)
            if oppose > self.block_force and lag > self.block_error and stalled:
                self._block_time += dt
                self._block_deficit += (requested - progress) * dt
            else:
                self._block_time = self._block_deficit = 0.0
            deficit_confirmed = (self._block_time >= self.config["blocked"]["deficit_confirm_s"] - 1e-15
                                 and self._block_deficit >= self.config["blocked"]["deficit_m"] - 1e-15)
            if self._block_time >= self.block_confirm - 1e-15 or deficit_confirmed:
                lin = lin - normal * max(float(lin @ normal), 0.0)
                self.block_correction_pos = -normal * max(float(err @ normal) - self.block_error, 0.0)
                self.p_ref += self.block_correction_pos
                if np.linalg.norm(self.block_correction_pos) > 1e-15:
                    self.reanchored = True
                    self.correction_reason = "blocked_progress"
                    self.reset_progress()
                blocked_now = True
                self.blocked = True
        else:
            self._block_time = self._block_deficit = 0.0
        if saturated:
            self._sat_time += dt
        else:
            self._sat_time = 0.0
        dev = float(np.linalg.norm(self.p_ref - np.asarray(pos, dtype=float)))
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
        del dt
        if self.p_ref is None or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            return
        old = self.p_ref.copy()
        err = old - np.asarray(pos, dtype=float)
        n = float(np.linalg.norm(err))
        if n > float(limits["pos_dev"]) + 1e-12:
            self.p_ref = np.asarray(pos, dtype=float) + err * (float(limits["pos_dev"]) / n)
            self.reference_correction_pos = self.p_ref - old
            self.clipped_offset = old - self.p_ref
            self.reanchored = True
        e_r = orientation_error(self.r_ref, rot)
        ang = float(np.linalg.norm(e_r))
        if ang > float(limits["rot_dev"]) + 1e-12:
            self.r_ref = integrate_rotation(rot, e_r, float(limits["rot_dev"]) / ang)
            self.reference_correction_rot = ang - float(limits["rot_dev"])
            self.reanchored = True
