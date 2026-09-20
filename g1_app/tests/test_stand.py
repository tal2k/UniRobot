import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "unitree_rl_mjlab",
))

from training.stand.stand_env_cfg import make_stand_env_cfg


def test_stand_env_cfg_builds():
    cfg = make_stand_env_cfg()
    assert "stand_still" in cfg.rewards
    assert "stand_success" in cfg.rewards
    assert "push_robot" in cfg.events
    assert "fell_over" in cfg.terminations
    assert "time_out" in cfg.terminations
    assert abs(cfg.episode_length_s - 10.0) < 1e-9
    assert cfg.decimation == 4


def test_stand_push_is_strong_and_frequent():
    cfg = make_stand_env_cfg()
    push = cfg.events["push_robot"]
    assert push.interval_range_s is not None
    assert push.interval_range_s[1] <= 4.0
    vr = push.params["velocity_range"]
    assert abs(vr["x"][1]) >= 1.0 and abs(vr["y"][1]) >= 1.0


def test_stand_task_registers():
    from mjlab.tasks.registry import list_tasks

    import training.stand.config.g1  # noqa: F401  (registers Unitree-G1-Stand)

    assert "Unitree-G1-Stand" in list_tasks()
