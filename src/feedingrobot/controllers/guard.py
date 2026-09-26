"""Phase contact rules and latched faults. The first fault is never overwritten."""

from __future__ import annotations

import numpy as np

from feedingrobot.sim.contacts import contact_pair, geom_group

PHASES = (
    "SELECT",
    "ACQUIRE",
    "TRANSPORT",
    "WAIT_READY",
    "APPROACH",
    "TRANSFER",
    "RETRACT",
    "RECOVER",
    "STOP",
)
FREE_PHASES = ("SELECT", "TRANSPORT", "WAIT_READY")
MOUTH_PHASES = ("APPROACH", "TRANSFER", "RETRACT")
GEAR_OF = {name: "FREE" for name in FREE_PHASES}
GEAR_OF["ACQUIRE"] = "ACQUIRE"
for _name in MOUTH_PHASES:
    GEAR_OF[_name] = "MOUTH"
GEAR_OF["RECOVER"] = "STOP"
GEAR_OF["STOP"] = "STOP"

_BOWL_DEFAULT = ("bowl_bottom", "bowl_back", "bowl_left", "bowl_right", "bowl_front")
_PLATE_DEFAULT = ("plate_bottom",)
_MOUTH_DEFAULT = ("jaw_lip",)


def gear_of(phase: str) -> str:
    if phase not in GEAR_OF:
        raise KeyError(phase)
    return GEAR_OF[phase]


def contact_allowed(phase: str, geom_a: str, geom_b: str, context: dict | None = None) -> bool:
    """Pure predicate. Unknown robot/tool pairs are rejected. Order does not matter."""
    ctx = context or {}
    bowl = set(ctx.get("bowl_geoms", _BOWL_DEFAULT))
    plate = set(ctx.get("plate_work_geoms", _PLATE_DEFAULT))
    mouth = set(ctx.get("mouth_geoms", _MOUTH_DEFAULT))
    a = geom_a or ""
    b = geom_b or ""
    names = {a, b}
    ga, gb = geom_group(a), geom_group(b)
    groups = {ga, gb}
    if names & {"spring_pad"} or any(name.startswith("spring_") or name.startswith("test_") for name in names):
        if not ctx.get("press_test"):
            return False
        return bool(names & bowl) and "spring_pad" in names
    if groups == {"food", "spoon"} or groups == {"food", "plate"}:
        return True
    if groups == {"food", "mouth"}:
        return phase in MOUTH_PHASES or bool(ctx.get("mouth_case"))
    if "food" in groups and groups <= {"food", "table", "floor", "plate"}:
        return True
    robot = groups & {"arm", "spoon"}
    if not robot:
        return True
    if groups <= {"arm", "spoon"}:
        return True
    if "arm" in groups and groups & {"table", "plate", "mouth", "floor"}:
        return False
    if groups == {"spoon", "table"}:
        return False
    if groups == {"spoon", "plate"}:
        return phase == "ACQUIRE" and bool(names & bowl) and bool(names & plate)
    if groups == {"spoon", "mouth"}:
        return phase in MOUTH_PHASES and bool(names & bowl) and bool(names & mouth)
    return False


class Guard:
    def __init__(self, config: dict, speed_limit: float):
        self.config = config
        self.speed_limit = float(speed_limit)
        self.context = {
            "bowl_geoms": tuple(config.get("contact", {}).get("bowl_geoms", _BOWL_DEFAULT)),
            "plate_work_geoms": tuple(config.get("contact", {}).get("plate_work_geoms", _PLATE_DEFAULT)),
            "mouth_geoms": tuple(config.get("contact", {}).get("mouth_geoms", _MOUTH_DEFAULT)),
            "press_test": bool(config.get("press_test", False)),
            "mouth_case": bool(config.get("mouth_case", False)),
        }
        limits = config["wrench_limits"]
        self.wrench_limits = {key: (float(val[0]), float(val[1])) for key, val in limits.items()}
        stop = config["stop"]
        self.stop_speed = float(stop["speed_rad_s"])
        self.stop_within_s = float(stop["within_s"])
        self.stop_hold_s = float(stop["hold_s"])
        self.reset()

    def reset(self) -> None:
        self.status = "RUNNING"
        self.failure = None
        self.events: list[dict] = []
        self._stop_since = None
        self._slow_since = None
        self.stopped_ok = False
        self.phase = "TRANSPORT"
        self._limit_hit_tick = None

    def allowed(self, phase: str, geom_a: str, geom_b: str) -> bool:
        return contact_allowed(phase, geom_a, geom_b, self.context)

    def latch(self, fault: dict) -> None:
        if self.failure is None:
            self.failure = dict(fault)
            self.events.append({"kind": "fault", **self.failure})
        if fault.get("reason") in {"non-finite", "warning"} or str(fault.get("reason", "")).startswith("non-finite"):
            self.status = "ABORTED"
        elif self.status == "RUNNING":
            self.status = "STOPPING"

    def begin_stop(self, fault: dict) -> None:
        """Latch a pre-step stop and start the hold clock at the interval start."""
        self.latch(fault)
        if self.status in {"STOPPING", "STOPPED"} and self._stop_since is None:
            self._stop_since = float(fault["time"])

    def request_stop(self, tick: int, time: float, phase: str) -> None:
        """An explicit stop is an event, not a fabricated sensor failure."""
        if self.status != "RUNNING":
            return
        self.status = "STOPPING"
        self._stop_since = float(time)
        self.events.append({"kind": "stop_request", "tick": int(tick), "time": float(time), "phase": phase})

    def observe(self, state: dict, phase: str, control_info: dict | None = None, measurement: dict | None = None) -> None:
        self.phase = phase
        info = control_info or {}
        if self.status == "ABORTED":
            return
        reason = None
        measured = None
        limit = None
        pair = None
        geoms = None
        if not state.get("finite", True) or not np.all(np.isfinite(state.get("q", [0]))) or not np.all(np.isfinite(state.get("dq", [0]))):
            reason = "non-finite"
            self.status = "ABORTED"
        elif measurement is not None and (
            not measurement.get("finite", False)
            or not np.all(np.isfinite(measurement.get("raw_wrench_sensor", [])))
            or not np.all(np.isfinite(measurement.get("compensated_wrench_tcp", [])))
        ):
            reason = "non-finite-wrench"
            measured = "wrench"
            self.status = "ABORTED"
        elif sum(state.get("warnings") or []) > 0:
            reason = "warning"
            measured = state.get("warnings")
            self.status = "ABORTED"
        else:
            speed = float(np.max(np.abs(state["dq"]))) if len(state.get("dq", [])) else 0.0
            if speed > self.speed_limit + 1e-9 or state.get("velocity_fault"):
                reason = "speed"
                measured = speed
                limit = self.speed_limit
            else:
                dists = [float(c["dist"]) for c in state.get("contacts") or []]
                if dists and min(dists) < -1e-3:
                    reason = "penetration"
                    measured = min(dists)
                    limit = -1e-3
                else:
                    ranges = info.get("joint_ranges")
                    q = np.asarray(state["q"], dtype=float)
                    if ranges is not None and (np.any(q < ranges[:, 0] - 1e-8) or np.any(q > ranges[:, 1] + 1e-8)):
                        reason = "joint_limit"
                        measured = q.tolist()
                    if reason is None:
                        for row in state.get("contacts") or []:
                            if not self.allowed(phase, row.get("geom1"), row.get("geom2")):
                                if geom_group(row.get("geom1")) in {"arm", "spoon"} or geom_group(row.get("geom2")) in {"arm", "spoon"}:
                                    reason = "forbidden_contact"
                                    pair = contact_pair(row["group1"], row["group2"])
                                    geoms = (row.get("geom1"), row.get("geom2"))
                                    measured = float(row["dist"])
                                    break
                    if reason is None and info.get("command_expired"):
                        reason = "command_expired"
                    if reason is None and info.get("observation_invalid"):
                        reason = "observation_invalid"
                    if reason is None:
                        wrench = None if measurement is None else measurement.get("compensated_wrench_tcp")
                        gear = info.get("wrench_gear") or (gear_of(phase) if phase in GEAR_OF else "FREE")
                        if wrench is not None and gear in self.wrench_limits:
                            force_lim, moment_lim = self.wrench_limits[gear]
                            force = float(np.linalg.norm(np.asarray(wrench[:3], dtype=float)))
                            moment = float(np.linalg.norm(np.asarray(wrench[3:], dtype=float)))
                            if force > force_lim or moment > moment_lim:
                                reason = "wrench"
                                measured = [force, moment]
                                limit = [force_lim, moment_lim]
                    if reason is None and info.get("saturated_latched"):
                        reason = "saturation"
                        measured = info.get("sat_time_s")
                        limit = info.get("sat_limit_s")
        if reason is not None and self.failure is None:
            self.latch(
                {
                    "tick": int(state["tick"]),
                    "time": float(state.get("episode_time", state.get("sim_time", 0.0))),
                    "reason": reason,
                    "phase": phase,
                    "pair": pair,
                    "geoms": geoms,
                    "measured": measured,
                    "limit": limit,
                }
            )
        self._update_stop(state)

    def _update_stop(self, state: dict) -> None:
        if self.status not in {"STOPPING", "STOPPED"}:
            return
        if self.status == "ABORTED":
            return
        speed = float(np.max(np.abs(state["dq"])))
        t = float(state.get("episode_time", 0.0))
        if self._stop_since is None:
            self._stop_since = t
        if speed <= self.stop_speed + 1e-12:
            if self._slow_since is None:
                self._slow_since = t
            if t - self._slow_since >= self.stop_hold_s - 1e-12:
                self.status = "STOPPED"
                if self._stop_since is not None and self._slow_since - self._stop_since <= self.stop_within_s + 1e-9:
                    self.stopped_ok = True
        else:
            self._slow_since = None
            self.status = "STOPPING"
