"""Cartesian impedance: J^T (K e + D e_dot) + bias + damped nullspace."""

from __future__ import annotations

import mujoco
import numpy as np

from feedingrobot.controllers.guard import GEAR_OF, PHASES, gear_of
from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.controllers.wrench import WrenchPipeline, predict_tool_wrench

_ARM = 7


def singularity_gains(rho: float, eps_normal: float, eps_singular: float, rho_normal: float, rho_stop: float) -> tuple[float, float]:
    """Return (command scale, epsilon). Scale falls from 1 to 0 across the singular band."""
    if rho >= rho_normal:
        return 1.0, eps_normal
    if rho <= rho_stop:
        return 0.0, eps_singular
    span = rho_normal - rho_stop
    frac = (rho - rho_stop) / span
    log_eps = np.log(eps_normal) * frac + np.log(eps_singular) * (1.0 - frac)
    return float(frac), float(np.exp(log_eps))


def damped_nullspace(j_s: np.ndarray, mass: np.ndarray, eps: float, a_floor: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the damped, inertia-weighted nullspace without forming M inverse."""
    minv_js_t = np.linalg.solve(mass, j_s.T)
    operational = j_s @ minv_js_t
    lam2 = eps * max(float(np.trace(operational)) / 6.0, a_floor)
    reg = operational + lam2 * np.eye(6)
    jbar = np.linalg.solve(reg, minv_js_t.T).T
    null = np.eye(j_s.shape[1]) - jbar @ j_s
    return null, jbar, lam2


def clip_torque(tau: np.ndarray, tau_prev: np.ndarray, tau_max: np.ndarray, rate: float, dt: float) -> np.ndarray:
    step = float(rate) * float(dt)
    lo = np.maximum(-tau_max, tau_prev - step)
    hi = np.minimum(tau_max, tau_prev + step)
    return np.clip(tau, lo, hi)


class CartesianImpedance:
    def __init__(self, scene, config: dict):
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.index = scene.index
        self.config = config
        self.gears = config["gears"]
        self.length = float(config["feature_length_m"])
        self.eps_normal = float(config["epsilon_normal"])
        self.eps_singular = float(config["epsilon_singular"])
        self.a_floor = float(config["a_floor"])
        self.rho_normal = float(config["rho_normal"])
        self.rho_stop = float(config["rho_stop"])
        self.kq = float(config["posture"]["kq"])
        self.dq_gain = float(config["posture"]["dq"])
        self.posture_limit = float(config["posture"]["tau_limit"])
        self.rate = float(config["tau_rate_nm_s"])
        self.speed_predict = float(config["joint_speed_predict_rad_s"])
        self.margin_slow = float(config["joint_margin_slow_rad"])
        self.margin_stop = float(config["joint_margin_stop_rad"])
        self.power_on_limit = float(config["power_on_s"])
        self.blend_s = float(config["gain_blend_s"])
        self.hybrid = dict(config.get("hybrid_normal_force", {"enabled": False}))
        ids = self.index.arm_actuator_ids
        self.tau_max = np.maximum(
            np.abs(self.model.actuator_ctrlrange[ids, 0]),
            np.abs(self.model.actuator_ctrlrange[ids, 1]),
        )
        self.ranges = np.array(self.model.jnt_range[self.index.arm_joint_ids], dtype=float)
        self.reference = ReferenceShaper(config)
        self.wrench = WrenchPipeline(self.model, config, self.index.tool_body_id)
        self.guard = None
        self._command = None
        self.reset_flags()

    def reset_flags(self) -> None:
        self.tau_prev = np.zeros(_ARM)
        self.power_on = True
        self.power_time = 0.0
        self.takeover_failed = False
        self.phase = "TRANSPORT"
        self._gear_now = "FREE"
        self._gear_next = "FREE"
        self._blend = 1.0
        self._k = self._vector_gain("FREE")
        self._d = self._vector_damp("FREE")
        self.sim_time = 0.0
        self._command = None
        self._sat_time = 0.0
        self._stop_anchored = False

    def reset(self, state: dict, tau_applied, phase: str = "TRANSPORT") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase}")
        self.reset_flags()
        self.wrench.reset()
        self.reference.reset_motion()
        self.reference.anchor_pose(state["tcp_pos"], state["tcp_mat"], state["q"])
        self.tau_prev = np.asarray(tau_applied, dtype=float).reshape(_ARM).copy()
        self.phase = phase
        self._gear_now = gear_of(phase)
        self._gear_next = self._gear_now
        self._k = self._vector_gain(self._gear_now)
        self._d = self._vector_damp(self._gear_now)
        self.sim_time = float(state.get("episode_time", 0.0))
        if self.guard is not None:
            self.guard.reset()

    def set_command(self, twist_world, command_time: float, valid_until: float, phase: str, frame: str = "world") -> dict:
        twist = np.asarray(twist_world, dtype=float).reshape(6)
        rejected = None
        if frame != "world":
            rejected = "frame"
        elif phase not in PHASES:
            rejected = "phase"
        elif not np.all(np.isfinite(twist)):
            rejected = "non-finite-command"
            if self.guard is not None:
                self.guard.latch(
                    {
                        "tick": int(self.scene.tick),
                        "time": float(command_time),
                        "reason": "non-finite",
                        "phase": self.phase,
                        "pair": None,
                        "geoms": None,
                        "measured": None,
                        "limit": None,
                    }
                )
                self.guard.status = "ABORTED"
        elif float(command_time) > self.sim_time + 1e-9:
            rejected = "future"
        fault = self.guard.failure if self.guard is not None else None
        if rejected is None and fault is None:
            self._command = {
                "twist": twist.copy(),
                "command_time": float(command_time),
                "valid_until": float(valid_until),
                "phase": phase,
                "frame": frame,
            }
            self.phase = phase
        return {"accepted": rejected is None and fault is None, "rejected": rejected}

    def _vector_gain(self, gear: str) -> np.ndarray:
        spec = self.gears[gear]
        return np.array([spec["kp"]] * 3 + [spec["kr"]] * 3, dtype=float)

    def _vector_damp(self, gear: str) -> np.ndarray:
        spec = self.gears[gear]
        return np.array([spec["dp"]] * 3 + [spec["dr"]] * 3, dtype=float)

    def _limits(self, gear: str) -> dict:
        spec = self.gears[gear]
        return {
            "v": float(spec["v"]),
            "w": float(spec["w"]),
            "a": float(spec["a"]),
            "alpha": float(spec["alpha"]),
            "pos_dev": float(spec["pos_dev_m"]),
            "rot_dev": float(spec["rot_dev_rad"]),
        }

    def _strict_limits(self, gear_a: str, gear_b: str) -> dict:
        a = self._limits(gear_a)
        b = self._limits(gear_b)
        return {key: min(a[key], b[key]) for key in a}

    def _update_gains(self, dt: float) -> None:
        target = gear_of(self.phase) if self.phase in GEAR_OF else "STOP"
        if target != self._gear_next:
            if target in {"STOP", "RECOVER"}:
                self._gear_now = "STOP"
                self._gear_next = "STOP"
                self._blend = 1.0
            else:
                self._gear_now = self._gear_next if self._blend >= 1.0 else self._gear_now
                self._gear_next = target
                self._blend = 0.0
                new_limits = self._limits(target)
                err = self.reference.p_ref - self._tcp_pos
                if float(np.linalg.norm(err)) > new_limits["pos_dev"] or float(np.linalg.norm(orientation_error(self.reference.r_ref, self._tcp_rot))) > new_limits["rot_dev"]:
                    self.reference.reanchor(self._tcp_pos, self._tcp_rot)
        if self._gear_next == "STOP":
            self._k = np.zeros(6)
            self._d = self._vector_damp("FREE") * 0.0
            return
        self._blend = min(1.0, self._blend + dt / self.blend_s)
        k0 = self._vector_gain(self._gear_now)
        k1 = self._vector_gain(self._gear_next)
        d0 = self._vector_damp(self._gear_now)
        d1 = self._vector_damp(self._gear_next)
        self._k = (1.0 - self._blend) * k0 + self._blend * k1
        self._d = (1.0 - self._blend) * d0 + self._blend * d1

    def _jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.index.site_ids["tcp"])
        cols = self.index.arm_dof_adr
        jacobian = np.vstack([jacp[:, cols], jacr[:, cols]])
        return jacobian, np.array(self.data.qfrc_bias[cols], dtype=float).copy()

    def _mass(self) -> np.ndarray:
        full = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, full)
        cols = self.index.arm_dof_adr
        return full[np.ix_(cols, cols)].copy()

    def _joint_scale(self, jbar: np.ndarray, twist: np.ndarray, q: np.ndarray) -> float:
        scale_mat = np.diag([1, 1, 1, self.length, self.length, self.length])
        pred = jbar @ (scale_mat @ twist)
        scale = 1.0
        peak = float(np.max(np.abs(pred))) if pred.size else 0.0
        if peak > self.speed_predict:
            scale = self.speed_predict / peak
        pred = pred * scale
        for i in range(_ARM):
            vel = float(pred[i])
            if vel > 0:
                dist = float(self.ranges[i, 1] - q[i])
            elif vel < 0:
                dist = float(q[i] - self.ranges[i, 0])
            else:
                continue
            if dist <= self.margin_stop:
                return 0.0
            if dist < self.margin_slow:
                allow = self.speed_predict * (dist - self.margin_stop) / (self.margin_slow - self.margin_stop)
                if abs(vel) > allow > 0:
                    scale *= allow / abs(vel)
                    pred *= allow / abs(vel)
                elif allow <= 0:
                    return 0.0
        return float(np.clip(scale, 0.0, 1.0))

    def compute(self, state: dict, dt: float) -> tuple[np.ndarray, dict]:
        dt = float(dt)
        self.sim_time = float(state.get("episode_time", state.get("sim_time", self.sim_time)))
        self._tcp_pos = np.asarray(state["tcp_pos"], dtype=float)
        self._tcp_rot = np.asarray(state["tcp_mat"], dtype=float).reshape(3, 3)
        q = np.asarray(state["q"], dtype=float).reshape(_ARM)
        dq = np.asarray(state["dq"], dtype=float).reshape(_ARM)
        if self.guard is not None and self.guard.status == "ABORTED":
            return np.zeros(_ARM), {"apply": False, "status": "ABORTED", "phase_k": self.phase, "gear": "STOP"}
        mujoco.mj_forward(self.model, self.data)
        jacobian, bias = self._jacobian()
        mass = self._mass()
        scale_mat = np.diag([1, 1, 1, self.length, self.length, self.length])
        j_s = scale_mat @ jacobian
        singular = np.linalg.svd(j_s, compute_uv=False)
        rho = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
        cmd_scale, eps = singularity_gains(rho, self.eps_normal, self.eps_singular, self.rho_normal, self.rho_stop)
        null, jbar, lam2 = damped_nullspace(j_s, mass, eps, self.a_floor)
        stopping = self.guard is not None and self.guard.status in {"STOPPING", "STOPPED"}
        if stopping and not self._stop_anchored:
            self.reference.reanchor(self._tcp_pos, self._tcp_rot)
            self._stop_anchored = True
            self._command = None
        active_cmd = self._command
        expired = False
        twist = np.zeros(6)
        if active_cmd is not None:
            if self.sim_time > active_cmd["valid_until"] + 1e-12:
                expired = not self.power_on
            else:
                twist = active_cmd["twist"].copy()
                if active_cmd["phase"] != self.phase and self.guard is not None and self.guard.failure is None:
                    self.phase = active_cmd["phase"]
        if self.power_on or stopping or (self.guard is not None and self.guard.failure is not None):
            twist = np.zeros(6)
        self._update_gains(dt)
        limit_gear = self._gear_next if self._gear_next in self.gears else "FREE"
        other = self._gear_now if self._gear_now in self.gears else limit_gear
        limits = self._strict_limits(limit_gear, other) if not stopping else self._limits("MOUTH")
        if stopping:
            for key in ("v", "w", "a", "alpha"):
                limits[key] = 0.0
        joint_scale = 0.0 if cmd_scale <= 0 else self._joint_scale(jbar, twist, q)
        wrench_info = self._sense(state, dt)
        force = np.asarray(wrench_info["compensated_wrench_tcp"][:3], dtype=float)
        shaped = self.reference.shape(
            twist,
            self._tcp_pos,
            self._tcp_rot,
            force,
            dt,
            limits,
            cmd_scale,
            joint_scale,
            active=not (self.power_on or stopping or expired or cmd_scale <= 0),
            saturated=False,
        )
        e = np.concatenate(
            [
                shaped["p_ref"] - self._tcp_pos,
                orientation_error(shaped["r_ref"], self._tcp_rot),
            ]
        )
        v_actual = jacobian @ dq
        if stopping:
            wrench_c = np.zeros(6)
            tau_posture = np.zeros(_ARM)
            null_term = np.zeros(_ARM)
        else:
            wrench_c = self._k * e + self._d * (shaped["v_ref"] - v_actual)
            if self.hybrid.get("enabled"):
                wrench_c = self._hybrid(wrench_c, state)
            tau_posture = np.clip(self.kq * (self.reference.q0 - q) - self.dq_gain * dq, -self.posture_limit, self.posture_limit)
            if cmd_scale < 1.0:
                tau_posture *= cmd_scale
            null_term = null.T @ tau_posture
        if stopping:
            tau_raw = bias - self.dq_gain * dq
            null_term = np.zeros(_ARM)
            tau_posture = np.zeros(_ARM)
            wrench_c = np.zeros(6)
        else:
            tau_task = jacobian.T @ wrench_c
            tau_raw = tau_task + bias + null_term
        tau_cmd = clip_torque(tau_raw, self.tau_prev, self.tau_max, self.rate, dt)
        magnitude_sat = bool(np.any(np.abs(tau_raw) > self.tau_max + 1e-8))
        if magnitude_sat:
            self._sat_time += dt
        else:
            self._sat_time = 0.0
        self.tau_prev = tau_cmd.copy()
        if self.power_on:
            self.power_time += dt
            err = float(np.linalg.norm(self.reference.p_ref - self._tcp_pos))
            speed = float(np.max(np.abs(dq)))
            caught = float(np.max(np.abs(tau_cmd - (bias - self.dq_gain * dq)))) <= self.rate * dt + 1e-6
            held = err <= limits["pos_dev"] and speed <= float(self.config["joint_speed_limit_rad_s"])
            if self.power_time >= 0.05 and caught and held and speed < 0.05:
                self.power_on = False
            elif self.power_time >= self.power_on_limit:
                self.power_on = False
                if not held and self.guard is not None:
                    self.takeover_failed = True
                    self.guard.latch(
                        {
                            "tick": int(state["tick"]),
                            "time": self.sim_time,
                            "reason": "takeover",
                            "phase": self.phase,
                            "pair": None,
                            "geoms": None,
                            "measured": err,
                            "limit": limits["pos_dev"],
                        }
                    )
        info = {
            "apply": True,
            "phase_k": self.phase,
            "gear": "STOP" if stopping else (self._gear_next if self._blend > 0 else self._gear_now),
            "status": "POWER_ON" if self.power_on else (self.guard.status if self.guard is not None else "RUNNING"),
            "q": q.copy(),
            "dq": dq.copy(),
            "p": self._tcp_pos.copy(),
            "r": self._tcp_rot.copy(),
            "v": v_actual.copy(),
            "twist_command": twist.copy(),
            "twist_limited": shaped["twist_limited"],
            "p_ref": shaped["p_ref"],
            "r_ref": shaped["r_ref"],
            "v_ref": shaped["v_ref"],
            "k": self._k.copy(),
            "d": self._d.copy(),
            "rho": rho,
            "lam2": lam2,
            "scale": shaped["scale"],
            "blocked": shaped["blocked"],
            "reanchored": shaped["reanchored"],
            "tau_task": jacobian.T @ wrench_c,
            "tau_bias": bias.copy(),
            "tau_null": null_term.copy(),
            "tau_raw": tau_raw.copy(),
            "tau_cmd": tau_cmd.copy(),
            "joint_ranges": self.ranges.copy(),
            "command_expired": expired,
            "saturated_latched": self._sat_time >= float(self.config["blocked"]["sat_s"]),
            "sat_time_s": self._sat_time,
            "sat_limit_s": float(self.config["blocked"]["sat_s"]),
            "power_on": self.power_on,
            "guard_wrench": wrench_info["compensated_wrench_tcp"],
            "observation_invalid": not np.all(np.isfinite(wrench_info["compensated_wrench_tcp"])),
            "e": e.copy(),
            **{f"ft_{key}": value for key, value in wrench_info.items()},
        }
        return tau_cmd, info

    def _sense(self, state: dict, dt: float) -> dict:
        idx = self.index
        raw = np.asarray(state["raw_wrench_sensor"], dtype=float)
        return self.wrench.update(
            self.model,
            self.data,
            raw,
            np.asarray(state["ft_mat"], dtype=float),
            np.asarray(state["ft_pos"], dtype=float),
            np.asarray(state["tcp_pos"], dtype=float),
            float(self.scene.wrench_sign),
            float(state.get("episode_time", 0.0)),
            dt,
            calibrate=bool(self.config.get("calibrate_bias", False)),
            stable=float(np.max(np.abs(state["dq"]))) < 0.01,
        )

    def _hybrid(self, wrench_c: np.ndarray, state: dict) -> np.ndarray:
        normal = np.asarray(self.hybrid.get("normal", [0, 0, 1]), dtype=float)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        selector = np.zeros((6, 6))
        selector[:3, :3] = np.outer(normal, normal)
        pose = np.eye(6) - selector
        measured = float(normal @ np.asarray(state.get("guard_wrench", np.zeros(6))[:3], dtype=float))
        desired = float(self.hybrid.get("f_d", 0.5))
        force = np.clip(desired + float(self.hybrid.get("kf", 0.2)) * (desired - max(measured, 0.0)), 0.0, float(self.hybrid.get("u_max", 2.0)))
        return pose @ wrench_c + np.concatenate([-normal * force, np.zeros(3)])


def control_step(controller: CartesianImpedance, guard, phase: str | None = None):
    """One control interval: compute from the current state, then step physics, then observe."""
    state = controller.scene.snapshot()
    if phase is not None and (guard.failure is None) and controller.guard.status == "RUNNING" and not controller.power_on:
        controller.phase = phase
    tau, info = controller.compute(state, controller.scene.dt)
    if not info.get("apply", True):
        return state, info
    nxt = controller.scene.step_physics(tau)
    used = info["phase_k"]
    guard.observe(nxt, used, info)
    return nxt, info
