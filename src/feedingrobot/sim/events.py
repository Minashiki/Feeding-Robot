"""Shared spoon-departure test. Durations are seconds, not step counts."""

from __future__ import annotations

import numpy as np


def outside_support(local: np.ndarray, support: dict) -> bool:
    xy = float(np.linalg.norm(local[:2]))
    z = float(local[2])
    return xy > float(support["half_xy_m"]) or z > float(support["z_max_m"]) or z < float(support["z_min_m"])


def food_on_spoon(contacts) -> bool:
    return any({row["group1"], row["group2"]} == {"food", "spoon"} for row in contacts)


class Departure:
    """Tracks the first continuous off-spoon interval."""

    def __init__(self, confirm_s: float):
        self.confirm_s = float(confirm_s)
        self.start_s = None
        self.confirmed_s = None
        self._accum = 0.0

    def update(self, episode_time: float, dt: float, outside: bool, touching: bool) -> None:
        if self.confirmed_s is not None:
            return
        if outside and not touching:
            if self._accum == 0.0:
                self.start_s = float(episode_time)
            self._accum += float(dt)
            if self._accum + 1e-12 >= self.confirm_s:
                self.confirmed_s = float(episode_time)
        else:
            self.start_s = None
            self._accum = 0.0

    @property
    def candidate(self) -> bool:
        return self._accum > 0.0 and self.confirmed_s is None
