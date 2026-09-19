"""Shared MuJoCo math helpers (single source of truth).

Consolidates the `quat_to_projected_gravity` copies previously in
`g1_stand_onnx.py` and `record_getup.py`, plus the fallen-pose sampler
previously only in `record_getup.py`.
"""

from __future__ import annotations

import math

import numpy as np


def quat_to_projected_gravity(quat) -> np.ndarray:
    """Body-frame gravity for MuJoCo framequat sensor (w, x, y, z).

    projected_gravity = R(body->world)^T @ [0, 0, -1].
    Identity orientation -> [0, 0, -1].
    """
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    gx = 2.0 * (w * y - x * z)
    gy = -2.0 * (w * x + y * z)
    gz = 2.0 * (x * x + y * y) - 1.0
    return np.array([gx, gy, gz], dtype=np.float32)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX euler (rad) -> MuJoCo (w, x, y, z) quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def tilt_from_quat(quat) -> float:
    """Upright-ness: norm of projected-gravity XY (0 = upright)."""
    g = quat_to_projected_gravity(quat)
    return float(np.linalg.norm(g[:2]))
