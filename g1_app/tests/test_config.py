import os

from core.config import DEPLOY_YAML, get_local_cfg, load_local_cfg
from core.terrains import TERRAINS


def test_deploy_yaml_loads_29dof():
    cfg = load_local_cfg(DEPLOY_YAML)
    for k in ("stiffness", "damping", "default_pos", "action_scale"):
        assert len(cfg[k]) == 29, (k, len(cfg[k]))
    assert abs(cfg["step_dt"] - 0.02) < 1e-9
    assert abs(cfg["gait_period"] - 0.6) < 1e-9


def test_fallback_matches_yaml_shapes():
    cfg = get_local_cfg()
    for k in ("stiffness", "damping", "default_pos", "action_scale"):
        assert len(cfg[k]) == 29


def test_terrains_exist():
    assert set(TERRAINS) == {"flat", "rough", "slope", "steps", "obstacles",
                             "apartment"}
    for name, path in TERRAINS.items():
        assert os.path.isfile(path), f"{name}: {path}"
