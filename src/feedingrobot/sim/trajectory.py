"""Quintic joint reference. Position and velocity are continuous and zero at the ends."""

from __future__ import annotations

import numpy as np


def quintic(q0, q1, t: float, duration: float):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if duration <= 0:
        raise ValueError("duration must be positive")
    s = float(np.clip(t / duration, 0.0, 1.0))
    a = 10 * s**3 - 15 * s**4 + 6 * s**5
    adot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    return q0 + a * (q1 - q0), adot * (q1 - q0)


def smooth_pulse(t: float, t0: float, duration: float, amplitude: float) -> float:
    """Torque pulse that starts and ends at zero with zero slope."""
    if t < t0 or t > t0 + duration:
        return 0.0
    s = (t - t0) / duration
    return float(amplitude * np.sin(np.pi * s) ** 2)
