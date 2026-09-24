"""Raw wrist F/T and frame transforms.

MuJoCo force/torque sensors report the parent-on-child interaction in the site
frame. This project publishes the wrench of the outside world on the tool.
The fixed sign s is not chosen from the current sample.
"""

from __future__ import annotations

import numpy as np

# Sensor parent-on-child is opposite the external wrench on the tool.
WRENCH_SIGN_EXTERNAL_ON_TOOL = -1.0


def site_rotation(data, site_id: int) -> np.ndarray:
    """Rotation that maps site-frame vectors into world."""
    return np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()


def site_position(data, site_id: int) -> np.ndarray:
    return np.array(data.site_xpos[site_id], dtype=float).copy()


def read_raw_wrench(model, data, force_adr: int, torque_adr: int) -> np.ndarray:
    force = np.array(data.sensordata[force_adr : force_adr + 3], dtype=float)
    torque = np.array(data.sensordata[torque_adr : torque_adr + 3], dtype=float)
    return np.concatenate([force, torque])


def rotate_wrench(force_s: np.ndarray, torque_s: np.ndarray, rot_ws: np.ndarray, sign: float):
    """Map a site wrench into world. Moment stays at the site origin."""
    sign = float(sign)
    force_w = sign * (rot_ws @ np.asarray(force_s, dtype=float))
    torque_w = sign * (rot_ws @ np.asarray(torque_s, dtype=float))
    return force_w, torque_w


def shift_torque(torque_w: np.ndarray, force_w: np.ndarray, p_from: np.ndarray, p_to: np.ndarray):
    """Move the moment origin from p_from to p_to. Force is unchanged."""
    return np.asarray(torque_w, dtype=float) + np.cross(
        np.asarray(p_from, dtype=float) - np.asarray(p_to, dtype=float),
        np.asarray(force_w, dtype=float),
    )


def world_and_tcp_wrench(raw_sensor: np.ndarray, rot_ws: np.ndarray, p_site: np.ndarray, p_tcp: np.ndarray, sign: float):
    force_w, torque_w = rotate_wrench(raw_sensor[:3], raw_sensor[3:], rot_ws, sign)
    torque_tcp = shift_torque(torque_w, force_w, p_site, p_tcp)
    return np.concatenate([force_w, torque_w]), np.concatenate([force_w, torque_tcp])
