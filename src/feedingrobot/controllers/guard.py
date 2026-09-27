"""Phase contact rules and latched faults. The first fault is never overwritten."""

from __future__ import annotations

import numpy as np
import mujoco

from feedingrobot.sim.contacts import contact_pair, geom_group

from feedingrobot.controllers.contracts import (
    PHASES, FREE_PHASES, MOUTH_PHASES, GEAR_OF,
    BOWL_GEOMS as _BOWL_DEFAULT, PLATE_GEOMS as _PLATE_DEFAULT, MOUTH_GEOMS as _MOUTH_DEFAULT,
)


def geometry_registry(model):
    """Resolve unnamed robot geoms by body ownership, never by an unknown name."""
    rows = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        ancestors = []
        cursor = body
        while cursor:
            ancestors.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cursor))
            cursor = int(model.body_parentid[cursor])
        if 'tool_mount' in ancestors:
            group = 'spoon'
        elif body_name in {f'link{i}' for i in range(8)}:
            group = 'arm'
        elif 'food' in ancestors:
            group = 'food'
        elif 'plate' in ancestors:
            group = 'plate'
        elif 'head' in ancestors:
            group = 'mouth'
        elif body == 0 and name in {'table', 'floor'}:
            group = name
        elif body_name == 'spring_slider' and name == 'spring_pad':
            group = 'fixture'
        else:
            group = 'unknown'
        rows.append({'id': gid, 'name': name, 'body': body_name, 'group': group})
    return rows


def recorded_contact_allowed(phase, row, context, registry):
    groups = []
    for side in (1, 2):
        gid = row.get(f'geom{side}_id')
        if not isinstance(gid, (int, np.integer)) or not 0 <= gid < len(registry):
            return False
        item = registry[int(gid)]
        if row.get(f'geom{side}') != item['name']:
            return False
        groups.append(item['group'])
    return contact_allowed(phase, row.get('geom1'), row.get('geom2'), {**context, 'groups': groups})


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
    ga, gb = ctx.get('groups', tuple('unknown' if geom_group(n) == 'arm' else geom_group(n) for n in (a, b)))
    groups = {ga, gb}
    if names & {"spring_pad"} or any(name.startswith("spring_") or name.startswith("test_") for name in names):
        if not ctx.get("press_test"):
            return False
        return bool(names & bowl) and "spring_pad" in names
    if 'unknown' in groups:
        return False
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
        return False
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
        self.registry = None
        self.reset()

    def bind_model(self, model):
        self.registry = geometry_registry(model)

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
                            allowed = (recorded_contact_allowed(phase, row, self.context, self.registry)
                                       if self.registry is not None else self.allowed(phase, row.get("geom1"), row.get("geom2")))
                            if not allowed:
                                reason = "forbidden_contact"
                                pair = contact_pair(row.get("group1", "unknown"), row.get("group2", "unknown"))
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
