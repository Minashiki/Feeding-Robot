"""Cartesian impedance: J^T (K e + D e_dot) + bias + damped nullspace."""

from __future__ import annotations

import mujoco
import numpy as np

from feedingrobot.controllers.guard import GEAR_OF, PHASES, gear_of
from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.controllers.wrench import WrenchPipeline, predict_tool_wrench

_ARM = 7


def enters_singularity_stop(rho: float, rho_stop: float) -> bool:
    """True at and below the stop threshold. The slowdown band above it is not a stop."""
    return bool(np.isfinite(rho) and float(rho) <= float(rho_stop))


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
        self.active_gear = "FREE"
        self.target_gear = "FREE"
        self._gear_now = "FREE"
        self._gear_next = "FREE"
        self._blend = 1.0
        self.transition_elapsed = float(self.blend_s)
        self.transition_active = False
        self._k = self._vector_gain("FREE")
        self._d = self._vector_damp("FREE")
        self.k_start = self._k.copy()
        self.d_start = self._d.copy()
        self.limits_start = self._limits("FREE")
        self.sim_time = 0.0
        self._command = None
        self._last_stamp = None
        self._sat_time = 0.0
        self._sat_union = 0.0
        self._pause_advance = False
        self._last_tau_raw = np.zeros(_ARM)
        self._tau_before = np.zeros(_ARM)
        self._stop_anchored = False
        self.command_max_valid = float(self.config.get("command_valid_max_s", 0.1))

    def reset(self, state: dict, tau_applied, phase: str = "TRANSPORT") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase}")
        self.reset_flags()
        self.wrench.reset()
        self.reference.reset_motion()
        self.reference.anchor_pose(state["tcp_pos"], state["tcp_mat"], state["q"])
        self.tau_prev = np.asarray(tau_applied, dtype=float).reshape(_ARM).copy()
        self.phase = phase
        self.active_gear = gear_of(phase)
        self.target_gear = self.active_gear
        self._gear_now = self.active_gear
        self._gear_next = self.active_gear
        self._k = self._vector_gain(self.active_gear)
        self._d = self._vector_damp(self.active_gear)
        self.k_start = self._k.copy()
        self.d_start = self._d.copy()
        self.limits_start = self._limits(self.active_gear)
        self.transition_elapsed = float(self.blend_s)
        self.transition_active = False
        self.sim_time = float(state.get("episode_time", 0.0))
        if self.guard is not None:
            self.guard.reset()
        self._seed_wrench(state)

    def now(self) -> float:
        return float(self.data.time - self.scene.time_offset)

    def _abort_input(self, reason: str, when: float) -> None:
        self._command = None
        if self.guard is None:
            return
        self.guard.latch(
            {
                "tick": int(self.scene.tick),
                "time": float(when) if np.isfinite(when) else self.now(),
                "reason": "non-finite",
                "phase": self.phase,
                "pair": None,
                "geoms": None,
                "measured": reason,
                "limit": None,
            }
        )
        self.guard.status = "ABORTED"

    def set_command(self, twist_world, command_time: float, valid_until: float, phase: str, frame: str = "world") -> dict:
        twist = np.asarray(twist_world, dtype=float).reshape(-1)
        now = self.now()
        rejected = None
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            rejected = "non-finite-command"
        elif not np.isfinite(command_time) or not np.isfinite(valid_until):
            rejected = "non-finite-time"
        elif frame != "world":
            rejected = "frame"
        elif phase not in PHASES:
            rejected = "phase"
        elif float(command_time) > now + 1e-12:
            rejected = "future"
        elif not (float(valid_until) > float(command_time) + 1e-12):
            rejected = "invalid-horizon"
        elif float(valid_until) - float(command_time) > self.command_max_valid + 1e-12:
            rejected = "horizon-too-long"
        elif float(valid_until) <= now + 1e-12:
            rejected = "already-expired"
        if rejected in {"non-finite-command", "non-finite-time"}:
            self._abort_input(rejected, now)
            return {"accepted": False, "rejected": rejected}
        fault = self.guard.failure if self.guard is not None else None
        if fault is not None:
            return {"accepted": False, "rejected": "fault"}
        if rejected is not None:
            return {"accepted": False, "rejected": rejected}
        stamp = float(command_time)
        payload = {
            "twist": twist.copy(),
            "command_time": stamp,
            "valid_until": float(valid_until),
            "phase": phase,
            "frame": frame,
        }
        if self._last_stamp is not None and stamp < self._last_stamp - 1e-12:
            return {"accepted": False, "rejected": "stale"}
        if self._last_stamp is not None and abs(stamp - self._last_stamp) <= 1e-12:
            same = self._command is not None and self._command["phase"] == phase and abs(self._command["valid_until"] - payload["valid_until"]) <= 1e-12 and np.allclose(self._command["twist"], twist)
            if same:
                return {"accepted": True, "rejected": None, "idempotent": True}
            return {"accepted": False, "rejected": "conflict"}
        self._command = payload
        self._last_stamp = stamp
        self.phase = phase
        return {"accepted": True, "rejected": None, "idempotent": False}

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
        if target in {"STOP"} or self.phase in {"STOP", "RECOVER"}:
            self.active_gear = "STOP"
            self.target_gear = "STOP"
            self._k = np.zeros(6)
            self._d = np.zeros(6)
            self._blend = 1.0
            self.transition_elapsed = self.blend_s
            return
        if target != self.target_gear:
            self.k_start = self._k.copy()
            self.d_start = self._d.copy()
            self.limits_start = self.effective_limits()
            self.target_gear = target
            self.transition_elapsed = 0.0
            self.transition_active = True
            new_limits = self._limits(target)
            err = self.reference.p_ref - self._tcp_pos
            ori = float(np.linalg.norm(orientation_error(self.reference.r_ref, self._tcp_rot)))
            if float(np.linalg.norm(err)) > new_limits["pos_dev"] or ori > new_limits["rot_dev"]:
                self.reference.reanchor(self._tcp_pos, self._tcp_rot)
        if self.transition_active:
            self.transition_elapsed = min(self.blend_s, self.transition_elapsed + dt)
            span = max(self.blend_s, 1e-9)
            mix = min(self.transition_elapsed / span, 1.0)
            self._k = (1.0 - mix) * self.k_start + mix * self._vector_gain(self.target_gear)
            self._d = (1.0 - mix) * self.d_start + mix * self._vector_damp(self.target_gear)
            self._blend = mix
            if mix >= 1.0 - 1e-15:
                self.transition_active = False
                self.active_gear = self.target_gear
                self.limits_start = self._limits(self.active_gear)
                self._k = self._vector_gain(self.active_gear)
                self._d = self._vector_damp(self.active_gear)
        self._gear_now = self.active_gear
        self._gear_next = self.target_gear

    def effective_limits(self) -> dict:
        if not self.transition_active or self.target_gear not in self.gears:
            gear = self.active_gear if self.active_gear in self.gears else "FREE"
            return self._limits(gear)
        return {key: min(self.limits_start[key], self._limits(self.target_gear)[key]) for key in self.limits_start}

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
        """Uniform scale of one candidate twist. 0.4 rad/s, softer inside 0.10 rad, zero inside 0.05 rad."""
        scale_mat = np.diag([1.0, 1.0, 1.0, self.length, self.length, self.length])
        pred = np.asarray(jbar, dtype=float) @ (scale_mat @ np.asarray(twist, dtype=float).reshape(6))
        beta = 1.0
        for i in range(_ARM):
            vel = float(pred[i])
            if abs(vel) <= 1e-12:
                continue
            dist = float(self.ranges[i, 1] - q[i]) if vel > 0 else float(q[i] - self.ranges[i, 0])
            if dist >= self.margin_slow:
                allow = self.speed_predict
            elif dist <= self.margin_stop:
                allow = 0.0
            else:
                allow = self.speed_predict * (dist - self.margin_stop) / (self.margin_slow - self.margin_stop)
            beta = min(beta, allow / abs(vel))
        return float(np.clip(beta, 0.0, 1.0))

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
        finite_model = bool(np.all(np.isfinite(jacobian)) and np.all(np.isfinite(mass)) and np.all(np.isfinite(q)) and np.all(np.isfinite(dq)))
        if not finite_model:
            rho = float("nan")
        else:
            try:
                singular = np.linalg.svd(j_s, compute_uv=False)
                rho = float(singular[-1] / singular[0]) if singular[0] > 0.0 and np.isfinite(singular[0]) else 0.0
            except np.linalg.LinAlgError:
                rho = float("nan")
        if not np.isfinite(rho):
            if self.guard is not None:
                self.guard.latch(
                    {
                        "tick": int(state["tick"]),
                        "time": self.sim_time,
                        "reason": "non-finite",
                        "phase": self.phase,
                        "pair": None,
                        "geoms": None,
                        "measured": "jacobian",
                        "limit": None,
                    }
                )
                self.guard.status = "ABORTED"
            return np.zeros(_ARM), {"apply": False, "status": "ABORTED", "phase_k": self.phase, "gear": "STOP"}
        cmd_scale, eps = singularity_gains(rho, self.eps_normal, self.eps_singular, self.rho_normal, self.rho_stop)
        null, jbar, lam2 = damped_nullspace(j_s, mass, eps, self.a_floor)
        stopping = self.guard is not None and self.guard.status in {"STOPPING", "STOPPED"}
        if self.guard is not None and self.guard.status != "ABORTED" and enters_singularity_stop(rho, self.rho_stop):
            if self.guard.failure is None:
                self.guard.begin_stop(
                    {
                        "tick": int(state["tick"]),
                        "time": self.sim_time,
                        "reason": "singularity",
                        "phase": self.phase,
                        "pair": None,
                        "geoms": None,
                        "measured": rho,
                        "limit": self.rho_stop,
                    }
                )
            if self.guard.status == "RUNNING":
                self.guard.status = "STOPPING"
            self.power_on = False
            stopping = self.guard.status in {"STOPPING", "STOPPED"}
        if stopping and not self._stop_anchored:
            self.reference.reanchor(self._tcp_pos, self._tcp_rot)
            self._stop_anchored = True
            self._command = None
        active_cmd = self._command
        expired = False
        twist = np.zeros(6)
        if active_cmd is not None and (not self.power_on) and self.sim_time >= active_cmd["valid_until"] - 1e-12:
            expired = True
            self._command = None
            if self.guard is not None and self.guard.failure is None:
                self.guard.latch(
                    {
                        "tick": int(state["tick"]),
                        "time": self.sim_time,
                        "reason": "command_expired",
                        "phase": self.phase,
                        "pair": None,
                        "geoms": None,
                        "measured": self.sim_time,
                        "limit": active_cmd["valid_until"],
                    }
                )
            stopping = True
        elif active_cmd is not None and self.sim_time < active_cmd["valid_until"] - 1e-12:
            twist = active_cmd["twist"].copy()
        if self.power_on or stopping or (self.guard is not None and self.guard.failure is not None):
            twist = np.zeros(6)
        if stopping:
            self.transition_active = False
            self.transition_elapsed = self.blend_s
            self._k = np.zeros(6)
            self._d = np.zeros(6)
            self._blend = 1.0
        else:
            self._update_gains(dt)
        limits = self.effective_limits()
        if stopping:
            for key in ("v", "w", "a", "alpha"):
                limits[key] = 0.0
        if stopping or expired:
            execution = "stop"
        elif self.power_on:
            execution = "power_on"
        elif cmd_scale <= 0.0:
            execution = "prohibit"
        elif float(np.linalg.norm(twist)) == 0.0:
            execution = "zero"
        else:
            execution = "run"
        cached = self.wrench._cache
        force = np.zeros(3) if cached is None else np.asarray(cached["compensated_wrench_tcp"][:3], dtype=float)

        def project(candidate):
            return self._joint_scale(jbar, candidate, q)

        shaped = self.reference.shape(
            twist,
            self._tcp_pos,
            self._tcp_rot,
            force,
            dt,
            limits,
            cmd_scale,
            1.0,
            active=execution in {"run", "zero"},
            saturated=bool(self._pause_advance) and not self.power_on,
            execution=execution,
            project=project if execution in {"run", "zero"} else None,
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
        tau_before = self.tau_prev.copy()
        tau_cmd = clip_torque(tau_raw, tau_before, self.tau_max, self.rate, dt)
        self._last_tau_raw = tau_raw.copy()
        self._tau_before = tau_before
        if self.power_on and not stopping:
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
            "gear": "STOP" if stopping else self.target_gear,
            "wrench_gear": "STOP" if stopping else (gear_of(self.phase) if self.phase in GEAR_OF else "FREE"),
            "control_mode": (self.guard.status if self.guard is not None else "STOPPING") if stopping else ("POWER_ON" if self.power_on else "RUNNING"),
            "status": (self.guard.status if self.guard is not None else "STOPPING") if stopping else ("POWER_ON" if self.power_on else (self.guard.status if self.guard is not None else "RUNNING")),
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
            "blocked_now": bool(shaped.get("blocked_now", False)),
            "reanchored": shaped["reanchored"],
            "tau_task": jacobian.T @ wrench_c,
            "tau_bias": bias.copy(),
            "tau_null": null_term.copy(),
            "tau_raw": tau_raw.copy(),
            "tau_cmd": tau_cmd.copy(),
            "joint_ranges": self.ranges.copy(),
            "command_expired": expired,
            "constraint_reset": shaped.get("constraint_reset"),
            "saturated_latched": self._sat_union >= float(self.config["blocked"]["sat_s"]) - 1e-12,
            "sat_time_s": self._sat_union,
            "sat_limit_s": float(self.config["blocked"]["sat_s"]),
            "power_on": self.power_on,
            "k_start": self.k_start.copy(),
            "d_start": self.d_start.copy(),
            "transition_elapsed": float(self.transition_elapsed),
            "active_gear": self.active_gear,
            "target_gear": self.target_gear,
            "limits": dict(limits),
            "tau_before": tau_before.copy(),
            "e": e.copy(),
            "execution": execution,
            "candidate_twist": np.asarray(shaped.get("candidate", np.zeros(6)), dtype=float),
            "beta": float(shaped.get("beta", 1.0)),
            "v_hist": np.asarray(shaped.get("v_hist", np.zeros(6)), dtype=float),
            "qdot_pred": jbar @ (scale_mat @ np.asarray(shaped.get("v_hist", np.zeros(6)), dtype=float)),
            "reference_correction_pos": np.asarray(shaped.get("reference_correction_pos", np.zeros(3)), dtype=float),
            "reference_correction_rot": float(shaped.get("reference_correction_rot", 0.0)),
            "correction_reason": shaped.get("correction_reason"),
            "transition_active": bool(self.transition_active),
        }
        return tau_cmd, info

    def _seed_wrench(self, state: dict) -> None:
        measured = self.wrench.measure(
            self.model,
            self.data,
            np.asarray(state["raw_wrench_sensor"], dtype=float),
            np.asarray(state["ft_mat"], dtype=float),
            np.asarray(state["ft_pos"], dtype=float),
            np.asarray(state["tcp_pos"], dtype=float),
            float(self.scene.wrench_sign),
        )
        self.wrench.update_sample(measured, int(state["tick"]), float(state.get("episode_time", 0.0)), 0.0)

    def commit_applied(self, applied, dt: float, was_power_on: bool) -> None:
        applied = np.asarray(applied, dtype=float).reshape(_ARM)
        raw = self._last_tau_raw
        before = self._tau_before
        step = self.rate * float(dt)
        magnitude = np.abs(raw) > self.tau_max + 1e-8
        rate_hit = (np.abs(applied - before) >= step - 1e-9) & (np.abs(raw - applied) > 1e-8)
        self.tau_prev = applied.copy()
        if was_power_on:
            return
        if np.any(magnitude | rate_hit):
            self._sat_union += float(dt)
            self._pause_advance = True
        else:
            self._sat_union = 0.0
            self._pause_advance = False
        if self._sat_union >= float(self.config["blocked"]["sat_s"]) - 1e-12 and self.guard is not None and self.guard.failure is None:
            self.reference.reanchor(self._tcp_pos, self._tcp_rot)
            self.guard.latch(
                {
                    "tick": int(self.scene.tick),
                    "time": self.sim_time,
                    "reason": "saturation",
                    "phase": self.phase,
                    "pair": None,
                    "geoms": None,
                    "measured": float(self._sat_union),
                    "limit": float(self.config["blocked"]["sat_s"]),
                }
            )

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
    was_power_on = bool(controller.power_on)
    tau, info = controller.compute(state, controller.scene.dt)
    if not info.get("apply", True):
        return state, info
    nxt = controller.scene.step_physics(tau)
    measured = controller.wrench.measure(
        controller.model,
        controller.data,
        np.asarray(nxt["raw_wrench_sensor"], dtype=float),
        np.asarray(nxt["ft_mat"], dtype=float),
        np.asarray(nxt["ft_pos"], dtype=float),
        np.asarray(nxt["tcp_pos"], dtype=float),
        float(controller.scene.wrench_sign),
    )
    sample = controller.wrench.update_sample(measured, int(nxt["tick"]), float(nxt["episode_time"]), float(controller.scene.dt))
    controller.commit_applied(nxt["tau_command"], controller.scene.dt, was_power_on)
    for key, value in sample.items():
        info[f"ft_{key}"] = value
    guard.observe(nxt, info["phase_k"], info, measurement=sample)
    return nxt, info
