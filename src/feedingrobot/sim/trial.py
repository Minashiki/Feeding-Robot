"""Sticky per-step failure record. A later success does not clear the first fault."""

from __future__ import annotations

import numpy as np

FORBIDDEN = {
    ("arm", "table"),
    ("arm", "plate"),
    ("arm", "mouth"),
    ("spoon", "table"),
    ("spoon", "plate"),
}


class Trial:
    def __init__(self, speed_limit: float):
        self.speed_limit = float(speed_limit)
        self.failure = None
        self.max_speed = 0.0
        self.min_dist = 0.0
        self.peak_force = 0.0

    def observe(self, state: dict, extra_forbidden: set | None = None) -> None:
        speed = float(np.max(np.abs(state["dq"])))
        self.max_speed = max(self.max_speed, speed)
        dists = [c["dist"] for c in state["contacts"]]
        if dists:
            self.min_dist = min(self.min_dist, min(dists))
        for row in state["contacts"]:
            self.peak_force = max(self.peak_force, float(np.linalg.norm(row["force_on_geom2_world"])))
        if self.failure is not None:
            return
        reason = None
        if speed > self.speed_limit + 1e-9 or state.get("velocity_fault"):
            reason = f"speed {speed:.4f} > {self.speed_limit}"
        elif not state["finite"]:
            reason = "non-finite state"
        elif sum(state["warnings"]) > 0:
            reason = f"warning {state['warnings']}"
        elif dists and min(dists) < -1e-3:
            reason = f"penetration {min(dists):.5f}"
        else:
            banned = FORBIDDEN | (extra_forbidden or set())
            for row in state["contacts"]:
                pair = tuple(sorted((row["group1"], row["group2"])))
                if pair in banned:
                    reason = f"forbidden contact {pair}"
                    break
        if reason is not None:
            self.failure = {"tick": int(state["tick"]), "reason": reason, "speed": speed}

    @property
    def ok(self) -> bool:
        return self.failure is None
