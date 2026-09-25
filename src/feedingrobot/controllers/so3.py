"""SO(3) maps and quaternion conversions. MuJoCo quaternions are wxyz."""

from __future__ import annotations

import mujoco
import numpy as np

_PI = np.pi


def skew(w: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    w = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def so3_log(rot: np.ndarray) -> np.ndarray:
    """Rotation vector of `rot`. Near pi the axis is taken from the diagonal and signed stably."""
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if theta > _PI - 1e-4:
        sym = 0.5 * (r + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(sym), 0.0))
        if axis[0] > 1e-8:
            axis[1] = np.copysign(axis[1], sym[0, 1])
            axis[2] = np.copysign(axis[2], sym[0, 2])
        elif axis[1] > 1e-8:
            axis[0] = np.copysign(axis[0], sym[0, 1])
            axis[2] = np.copysign(axis[2], sym[1, 2])
        else:
            axis[0] = np.copysign(axis[0], sym[0, 2])
            axis[1] = np.copysign(axis[1], sym[1, 2])
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3)
        axis = axis / n
        if float(np.dot(axis, vee)) < 0.0:
            axis = -axis
        return axis * theta
    return (theta / (2.0 * np.sin(theta))) * vee


def orientation_error(rot_ref: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """World rotation vector taking the current frame to the reference."""
    return so3_log(np.asarray(rot_ref, dtype=float).reshape(3, 3) @ np.asarray(rot, dtype=float).reshape(3, 3).T)


def integrate_rotation(rot: np.ndarray, omega_world: np.ndarray, dt: float) -> np.ndarray:
    return so3_exp(np.asarray(omega_world, dtype=float).reshape(3) * float(dt)) @ np.asarray(rot, dtype=float).reshape(3, 3)


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float).reshape(4))
    return mat.reshape(3, 3)


def mat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
    return quat


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]])
