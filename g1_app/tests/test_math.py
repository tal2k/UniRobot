import math

import numpy as np

from core.math import euler_to_quat, quat_to_projected_gravity, tilt_from_quat


def test_identity_gravity():
    g = quat_to_projected_gravity(np.array([1.0, 0, 0, 0]))
    assert np.allclose(g, [0, 0, -1]), g


def test_pitch_90_gravity():
    # 90° pitch about Y: down vector rotates to [-1? +1?] — check norm + z≈0.
    q = euler_to_quat(0.0, math.pi / 2, 0.0)
    g = quat_to_projected_gravity(q)
    assert abs(float(np.linalg.norm(g)) - 1.0) < 1e-5
    assert abs(float(g[2])) < 1e-5


def test_tilt_zero_upright():
    assert tilt_from_quat(np.array([1.0, 0, 0, 0])) < 1e-6
