"""Mouth-frame passage checks. Local +X points into the mouth."""

from __future__ import annotations

import numpy as np

# Bowl corners in the TCP frame: x toward the tip, z out of the bowl.
BOWL_KEYPOINTS = np.array(
    [[x, y, z] for x in (-0.022, 0.022) for y in (-0.015, 0.015) for z in (0.0, 0.008)],
    dtype=float,
)


def opening_bounds(jaw_q: float) -> dict:
    """Opening of the rigid mouth. The lower edge follows the jaw hinge."""
    jaw_q = float(jaw_q)
    c, s = np.cos(jaw_q), np.sin(jaw_q)
    lip = np.array([0.001, 0.0, 0.0])
    rotated_z = -s * lip[0] + c * lip[2]
    z_lower = -0.015 + rotated_z
    return {"half_width": 0.023, "z_lower": float(z_lower), "z_upper": 0.015, "back_x": 0.038}


def to_mouth(points_world: np.ndarray, mouth_pos: np.ndarray, mouth_rot: np.ndarray) -> np.ndarray:
    return (mouth_rot.T @ (np.asarray(points_world, dtype=float) - mouth_pos).T).T


def bowl_points_world(tcp_pos: np.ndarray, tcp_rot: np.ndarray) -> np.ndarray:
    return tcp_pos + (tcp_rot @ BOWL_KEYPOINTS.T).T


def passage(tcp_pos, tcp_rot, mouth_pos, mouth_rot, jaw_q: float) -> dict:
    bounds = opening_bounds(jaw_q)
    tcp_local = to_mouth(np.asarray(tcp_pos, dtype=float).reshape(1, 3), mouth_pos, mouth_rot)[0]
    points = to_mouth(bowl_points_world(tcp_pos, tcp_rot), mouth_pos, mouth_rot)
    inside_yz = np.all(
        (np.abs(points[:, 1]) <= bounds["half_width"])
        & (points[:, 2] >= bounds["z_lower"] + 0.001)
        & (points[:, 2] <= bounds["z_upper"] - 0.001)
    )
    crossed_back = bool(np.any(points[:, 0] > bounds["back_x"]))
    entered = bool(tcp_local[0] >= 0.005 and inside_yz and not crossed_back)
    exited = bool(np.all(points[:, 0] < -0.005))
    return {
        "tcp_local": tcp_local,
        "points": points,
        "inside_yz": bool(inside_yz),
        "crossed_back": crossed_back,
        "entered": entered,
        "exited": exited,
        "bounds": bounds,
    }
