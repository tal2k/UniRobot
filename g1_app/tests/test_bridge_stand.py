import os

import mujoco
import numpy as np
import pytest

from core.bridge import (
    DEFAULT_LOCAL_POLICY,
    StandStillPolicy,
    WalkStandBridge,
    find_latest_stand_policy,
    reset_standing,
)
from core.config import G1_MODEL_DIR


def _stand_policy_path():
    return find_latest_stand_policy()


STAND_POLICY = _stand_policy_path() or find_latest_stand_policy()

needs_stand_policy = pytest.mark.skipif(
    STAND_POLICY is None, reason="no trained g1_stand snapshot yet")


def _sim():
    model = mujoco.MjModel.from_xml_path(os.path.join(G1_MODEL_DIR, "scene_29dof.xml"))
    model.opt.timestep = 0.005
    return model, mujoco.MjData(model)


@needs_stand_policy
def test_find_latest_stand_policy():
    assert STAND_POLICY is not None and os.path.isfile(STAND_POLICY)


def test_curated_models_policy_wins(monkeypatch, tmp_path):
    import core.bridge as bridge_mod

    curated = tmp_path / "g1_stand_policy.onnx"
    curated.write_bytes(b"fake")
    snap_dir = tmp_path / "logs" / "run"
    snap_dir.mkdir(parents=True)
    snap = snap_dir / "policy.onnx"
    snap.write_bytes(b"fake")
    # Make the snapshot newer so only the curated-first rule can win.
    import time

    old, new = time.time() - 100, time.time()
    os.utime(curated, (old, old))
    os.utime(snap, (new, new))
    monkeypatch.setattr(bridge_mod, "STAND_POLICY_PATH", str(curated))
    monkeypatch.setattr(bridge_mod, "WORKSPACE", str(tmp_path))
    assert find_latest_stand_policy() == str(curated)


@needs_stand_policy
def test_stand_policy_loads_94dof():
    model, data = _sim()
    pol = StandStillPolicy(model, data, STAND_POLICY)
    assert pol.obs_dim == 94
    assert pol.observe().shape == (94,)
    assert len(pol.joint_names) == 29


@needs_stand_policy
def test_stand_policy_headless_steps():
    model, data = _sim()
    pol = StandStillPolicy(model, data, STAND_POLICY)
    reset_standing(model, data, pol.default_pos)
    for _ in range(200):
        mujoco.mj_step(model, data)
        pol.step_sim()
    assert np.all(np.isfinite(data.qpos))
    assert float(data.qpos[2]) > 0.5


@needs_stand_policy
def test_walk_stand_auto_switch():
    model, data = _sim()
    bridge = WalkStandBridge(model, data, DEFAULT_LOCAL_POLICY, STAND_POLICY,
                             mode="auto")
    reset_standing(model, data, bridge.default_pos)
    bridge.set_command(0.4, 0.0, 0.0)
    for _ in range(400):
        mujoco.mj_step(model, data)
        bridge.step_sim()
    assert bridge.active == "walk"
    bridge.set_command(0.0, 0.0, 0.0)
    for _ in range(600):
        mujoco.mj_step(model, data)
        bridge.step_sim()
    assert bridge.active == "stand"
    assert bridge._blend == 1.0
    assert np.all(np.isfinite(data.qpos))
    assert float(data.qpos[2]) > 0.45
