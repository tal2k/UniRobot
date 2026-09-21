import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_WORKSPACE, "unitree_rl_mjlab"))

from core.getup_stages import (  # noqa: E402
    DONE,
    IDLE,
    SUPINE_TARGET,
    A,
    B,
    C,
    StageGate,
    StageSwitcher,
    gate_joint_indices,
    is_fallen,
    mean_abs_deviation,
    rms_pose_error,
)
from training.getup.getup_env_cfg import make_getup_env_cfg  # noqa: E402
from training.getup.reposition_env_cfg import make_reposition_env_cfg  # noqa: E402
from training.getup.situp_env_cfg import make_situp_env_cfg  # noqa: E402

ACTOR_TERMS = ["base_ang_vel", "projected_gravity", "base_height",
               "joint_pos", "joint_vel", "actions"]


@pytest.mark.parametrize("factory,episode,expected", [
    (make_reposition_env_cfg, 12.0,
     {"supine_success", "supine_pose", "torso_horizontal"}),
    (make_situp_env_cfg, 10.0,
     {"pelvis_rising", "stand_height", "stand_on_feet", "feet_force",
      "bad_support", "no_head_contact"}),
    (make_getup_env_cfg, 8.0,
     {"stand_success", "stand_on_feet", "bad_support", "no_head_contact"}),
])
def test_stage_env_cfgs_build(factory, episode, expected):
    cfg = factory()
    assert expected <= set(cfg.rewards)
    assert abs(cfg.episode_length_s - episode) < 1e-9
    assert cfg.decimation == 4
    assert set(cfg.terminations) == {"time_out"}
    assert list(cfg.observations["actor"].terms) == ACTOR_TERMS
    assert list(cfg.observations["critic"].terms) == ACTOR_TERMS + ["base_lin_vel"]
    assert "joint_pos" in cfg.actions
    assert "foot_friction" in cfg.events and "base_com" in cfg.events


def test_stage_a_has_no_balance_only_terms():
    cfg = make_reposition_env_cfg()
    assert "stand_on_feet" not in cfg.rewards
    assert "bad_support" not in cfg.rewards
    # Reposition needs only self-collision feedback.
    assert [s.name for s in cfg.scene.sensors] == ["self_collision"]


def test_stage_b_lying_reset():
    cfg = make_situp_env_cfg()
    assert cfg.events["reset_lying"].func.__name__ == "reset_lying_pose"


def test_stage_tasks_register():
    from mjlab.tasks.registry import list_tasks

    import training.getup.config.g1  # noqa: F401  (registers the tasks)

    tasks = list_tasks()
    for task_id in ("Unitree-G1-Getup", "Unitree-G1-Getup-Reposition",
                    "Unitree-G1-Getup-SitUp", "Unitree-G1-Getup-Rise"):
        assert task_id in tasks


def test_stage_tasks_load_full_config():
    import training.getup.config.g1  # noqa: F401

    path = os.path.join(_WORKSPACE, "unitree_rl_mjlab", "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("rl_train_entry_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rl_train_entry_test"] = mod
    spec.loader.exec_module(mod)
    for task_id, experiment in (
        ("Unitree-G1-Getup-Reposition", "g1_getup_reposition"),
        ("Unitree-G1-Getup-SitUp", "g1_getup_situp"),
        ("Unitree-G1-Getup-Rise", "g1_getup_rise"),
    ):
        cfg = mod.TrainConfig.from_task(task_id)
        assert cfg.agent.experiment_name == experiment
        assert cfg.env.scene.entities["robot"] is not None


def test_train_stage_mapping():
    from lab.train import STAGE_TASK_IDS, TASK_IDS

    assert STAGE_TASK_IDS["A"] == "Unitree-G1-Getup-Reposition"
    assert STAGE_TASK_IDS["B"] == "Unitree-G1-Getup-SitUp"
    assert STAGE_TASK_IDS["C"] == "Unitree-G1-Getup-Rise"
    assert TASK_IDS["getup"] == "Unitree-G1-Getup"


def test_curriculum_reexports():
    from training.getup import curriculum

    assert curriculum.A == A and curriculum.DONE == DONE
    assert curriculum.STAGE_SEQUENCE == (A, B, C)
    assert set(curriculum.STAGE_GATES) == {A, B, C}


def _switcher(min_ticks=3):
    return StageSwitcher(min_ticks=min_ticks, dt=0.02)


def test_stage_switcher_full_sequence():
    sw = _switcher()
    assert sw.state == IDLE
    # fallen -> engage; lying + supine-matching -> B ...
    assert sw.update(0.15, 0.90, 0.0, 0.50, 0.10) == A
    for _ in range(3):
        state = sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert state == B
    for _ in range(3):
        state = sw.update(0.60, 0.50, 0.0, 0.40)  # crouch
    assert state == C
    for _ in range(3):
        state = sw.update(0.76, 0.20, 0.05, 0.05)  # standing, slow
    assert state == DONE


def test_stage_a_gate_uses_pose_not_extension():
    # Lying + supine-matching joints advance even with low limb extension.
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(3):
        state = sw.update(0.15, 0.90, 0.0, 0.05, 0.10)
    assert state == B
    # ... while lying + sprawled joints (high extension, bad pose) do not.
    sw2 = _switcher()
    sw2.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(5):
        state = sw2.update(0.15, 0.90, 0.0, 0.80, 0.60)
    assert state == A


def test_stage_switcher_standing_shortcut():
    # Standing still while in A jumps straight to DONE (no A-gate stall).
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert sw.state == A
    assert sw.update(0.78, 0.10, 0.05, 0.05, 0.80) == DONE
    # Same from B: no detour through C.
    sw2 = _switcher()
    sw2.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(3):
        sw2.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert sw2.state == B
    assert sw2.update(0.78, 0.10, 0.05, 0.05, 0.80) == DONE
    # ... but a fast-moving standing robot does NOT shortcut (not handoff-safe).
    sw3 = _switcher()
    sw3.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert sw3.update(0.78, 0.10, 0.50, 0.05, 0.80) == A


def test_stage_switcher_hysteresis_resets_hold():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)  # 2 of 3 ticks
    sw.update(0.15, 0.90, 0.0, 0.50, 0.60)  # pose lost -> hold reset
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert sw.update(0.15, 0.90, 0.0, 0.50, 0.10) == B


def test_stage_switcher_reverts_c_to_b():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(3):
        sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(3):
        sw.update(0.60, 0.50, 0.0, 0.40)
    assert sw.state == C
    for _ in range(3):
        state = sw.update(0.40, 0.50, 0.0, 0.40)  # sank below the crouch
    assert state == B


def test_stage_switcher_timeout_restarts_at_a():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    for _ in range(3):
        sw.update(0.15, 0.90, 0.0, 0.50, 0.10)
    assert sw.state == B
    ticks = int(6.0 / sw.dt) + 1
    for _ in range(ticks):
        state = sw.update(0.30, 0.90, 0.0, 0.10, 0.50)  # never reaches crouch
    assert state == A
    assert sw.elapsed == 0.0


def test_stage_switcher_reengages_after_done():
    sw = _switcher()
    sw._enter(DONE)
    assert sw.update(0.20, 0.90, 0.0, 0.0) == A  # fell again -> recover


def test_is_fallen():
    assert is_fallen(0.20, 0.10)
    assert is_fallen(0.70, 0.80)
    assert not is_fallen(0.70, 0.20)


def test_gate_joint_indices_and_deviation():
    names = ("left_shoulder_pitch_joint", "left_elbow_joint", "left_knee_joint",
             "waist_pitch_joint", "left_hip_pitch_joint")
    idx = gate_joint_indices(names)
    assert idx == [0, 1, 2]
    q = [0.5, 0.0, 0.0, 0.9, 0.0]
    q0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert abs(mean_abs_deviation(q, q0, idx) - 0.5 / 3) < 1e-9
    assert mean_abs_deviation(q, q0, []) == 0.0


def test_gate_satisfied_none_means_ignore():
    gate = StageGate(min_height=0.5)
    assert gate.satisfied(0.6, 1.0, 5.0, 0.0)
    assert not gate.satisfied(0.4, 0.0, 0.0, 0.0)


def test_rms_pose_error():
    assert rms_pose_error([0.0] * 29, SUPINE_TARGET) == 0.0
    assert rms_pose_error([], []) == 0.0
    q = [0.3] * 29
    assert abs(rms_pose_error(q, SUPINE_TARGET) - 0.3) < 1e-9


def test_supine_target_shared_with_training():
    import training.getup.mdp as mdp

    assert len(SUPINE_TARGET) == 29
    assert list(mdp.SUPINE_TARGET) == list(SUPINE_TARGET)
    cfg_target = make_reposition_env_cfg().rewards["supine_success"].params["target_pos"]
    assert list(cfg_target) == list(SUPINE_TARGET)


def test_stage_metric_names_exist_in_cfgs():
    # The dashboard hint printed by `g1 train -- --stage X` must name a
    # reward the stage actually logs (regression: Stage A pointed at a
    # reward that was never defined).
    from lab.train import STAGE_METRICS

    assert STAGE_METRICS["A"] in make_reposition_env_cfg().rewards
    assert STAGE_METRICS["B"] in make_situp_env_cfg().rewards
    assert STAGE_METRICS["C"] in make_getup_env_cfg().rewards
