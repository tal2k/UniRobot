import importlib.util
import os
import sys

import pytest

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.getup_stages import (  # noqa: E402
    DONE,
    GETUP,
    IDLE,
    ROLL,
    SUPINE_FACING,
    SUPINE_TARGET,
    StageGate,
    StageSwitcher,
    rms_pose_error,
)
from training.getup.roll_env_cfg import make_roll_env_cfg  # noqa: E402
from training.getup.standup_env_cfg import make_standup_env_cfg  # noqa: E402

ACTOR_TERMS = ["base_ang_vel", "projected_gravity", "base_height",
               "joint_pos", "joint_vel", "actions"]


@pytest.mark.parametrize("factory,episode,expected", [
    (make_roll_env_cfg, 12.0,
     {"roll_success", "face_up", "supine_pose", "torso_horizontal"}),
    (make_standup_env_cfg, 12.0,
     {"stand_success", "stand_on_feet", "pelvis_rising", "feet_force",
      "bad_support", "no_head_contact"}),
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


def test_roll_cfg_minimal_sensors():
    cfg = make_roll_env_cfg()
    # Rolling needs orientation + pose feedback only (no feet/head terms).
    assert [s.name for s in cfg.scene.sensors] == ["self_collision"]
    assert "stand_on_feet" not in cfg.rewards
    assert "feet_force" not in cfg.rewards
    assert cfg.rewards["roll_success"].params["max_facing"] == SUPINE_FACING


def test_standup_cfg_merged():
    cfg = make_standup_env_cfg()
    # Mixed starts (standing + crouch + lying) break the discovery plateau.
    assert cfg.events["reset_mixed"].func.__name__ == "reset_mixed_starts"
    assert cfg.events["reset_mixed"].params["stand_prob"] == 0.15
    # Pull-assist with within-episode fade (HoST-style).
    assert cfg.events["lift_assist"].func.__name__ == "lift_assist"
    assert cfg.events["lift_assist"].params["max_force"] == 180.0
    assert "wrist_pose_l2" in cfg.rewards
    # Crouch-loading gate loosened vs the Stage-C recipe so the sit-up
    # phase earns signal; stand_success pays for the full rise.
    assert cfg.rewards["stand_on_feet"].params["min_height"] == 0.55
    assert cfg.rewards["stand_success"].params["min_height"] == 0.70


def test_stage_tasks_register():
    from mjlab.tasks.registry import list_tasks

    import training.getup.config.g1  # noqa: F401  (registers the tasks)

    tasks = list_tasks()
    for task_id in ("Unitree-G1-Getup-Roll", "Unitree-G1-Getup-StandUp"):
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
        ("Unitree-G1-Getup-Roll", "g1_getup_roll"),
        ("Unitree-G1-Getup-StandUp", "g1_getup_standup"),
    ):
        cfg = mod.TrainConfig.from_task(task_id)
        assert cfg.agent.experiment_name == experiment
        assert cfg.env.scene.entities["robot"] is not None


def test_train_stage_mapping():
    from lab.train import STAGE_TASK_IDS, TASK_IDS

    assert STAGE_TASK_IDS["roll"] == "Unitree-G1-Getup-Roll"
    assert STAGE_TASK_IDS["standup"] == "Unitree-G1-Getup-StandUp"
    assert TASK_IDS["stand"] == "Unitree-G1-Stand"


def test_curriculum_reexports():
    from training.getup import curriculum

    assert curriculum.ROLL == ROLL and curriculum.DONE == DONE
    assert curriculum.GETUP == GETUP
    assert curriculum.STAGE_SEQUENCE == (ROLL, GETUP)
    assert set(curriculum.STAGE_GATES) == {ROLL, GETUP}


def _switcher(min_ticks=3):
    return StageSwitcher(min_ticks=min_ticks, dt=0.02)


def test_switcher_engages_roll_when_prone():
    sw = _switcher()
    assert sw.state == IDLE
    # Fallen face-down -> ROLL.
    assert sw.update(0.15, 0.90, 0.0, 0.90, 0.60) == ROLL


def test_switcher_skips_roll_when_already_supine():
    sw = _switcher()
    # Fallen but already face-up -> straight to GETUP.
    assert sw.update(0.15, 0.90, 0.0, -0.90, 0.10) == GETUP


def test_switcher_stays_idle_when_tall():
    sw = _switcher()
    # Tall but tilted: still standing business, not recovery.
    assert sw.update(0.70, 0.80, 0.0, 0.50, 0.50) == IDLE


def test_switcher_full_sequence():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    assert sw.state == ROLL
    # Rolled to supine, joints near target -> GETUP after hold ticks.
    for _ in range(3):
        state = sw.update(0.15, 0.90, 0.0, -0.80, 0.10)
    assert state == GETUP
    # Standing still -> DONE immediately (standing shortcut).
    assert sw.update(0.76, 0.20, 0.05, -0.10, 0.80) == DONE


def test_roll_gate_needs_pose_not_just_facing():
    # Face-up but tucked: handoff would leave GETUP's start distribution.
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    for _ in range(5):
        state = sw.update(0.15, 0.90, 0.0, -0.80, 0.60)
    assert state == ROLL


def test_switcher_hysteresis_resets_hold():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    sw.update(0.15, 0.90, 0.0, -0.80, 0.10)  # 1 of 3 ticks
    sw.update(0.15, 0.90, 0.0, -0.80, 0.10)  # 2 of 3 ticks
    sw.update(0.15, 0.90, 0.0, -0.80, 0.60)  # pose lost -> hold reset
    sw.update(0.15, 0.90, 0.0, -0.80, 0.10)
    sw.update(0.15, 0.90, 0.0, -0.80, 0.10)
    assert sw.update(0.15, 0.90, 0.0, -0.80, 0.10) == GETUP


def test_switcher_reverts_getup_to_roll():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, -0.90, 0.10)
    assert sw.state == GETUP
    # Rolled back to prone mid-rise -> re-roll after hold ticks.
    for _ in range(3):
        state = sw.update(0.40, 0.50, 0.0, 0.50, 0.40)
    assert state == ROLL


def test_switcher_timeout_rerolls():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, -0.90, 0.10)
    assert sw.state == GETUP
    ticks = int(12.0 / sw.dt) + 1
    for _ in range(ticks):
        state = sw.update(0.60, 0.50, 0.0, -0.60, 0.50)  # never stands
    assert state == ROLL
    assert sw.elapsed == 0.0


def test_switcher_roll_timeout_retries():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    assert sw.state == ROLL
    ticks = int(8.0 / sw.dt) + 1
    for _ in range(ticks):
        state = sw.update(0.15, 0.90, 0.0, 0.90, 0.60)  # never rolls
    assert state == ROLL
    assert sw.elapsed == 0.0


def test_switcher_standing_shortcut_from_roll():
    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    assert sw.state == ROLL
    # Somehow standing while rolling -> straight to DONE.
    assert sw.update(0.78, 0.10, 0.05, 0.0, 0.80) == DONE
    # ... but a fast-moving tall robot does NOT shortcut.
    sw2 = _switcher()
    sw2.update(0.15, 0.90, 0.0, 0.90, 0.60)
    assert sw2.update(0.78, 0.10, 0.50, 0.0, 0.80) == ROLL


def test_switcher_force_override():
    import pytest

    sw = _switcher()
    sw.update(0.15, 0.90, 0.0, 0.90, 0.60)
    assert sw.state == ROLL
    sw.force(GETUP)  # GUI manual override
    assert sw.state == GETUP
    assert sw.elapsed == 0.0
    sw.force(IDLE)  # AUTO resumes via reset/engage
    assert sw.update(0.15, 0.90, 0.0, 0.90, 0.60) == ROLL
    with pytest.raises(AssertionError):
        sw.force("walk")


def test_switcher_reengages_after_done():
    sw = _switcher()
    sw._enter(DONE)
    assert sw.update(0.20, 0.90, 0.0, 0.80, 0.50) == ROLL  # prone again
    sw._enter(DONE)
    assert sw.update(0.20, 0.90, 0.0, -0.80, 0.10) == GETUP  # supine


def test_gate_facing_thresholds():
    gate = StageGate(max_facing=SUPINE_FACING)
    assert gate.satisfied(0.15, 0.90, 0.0, -0.80, 0.10)
    assert not gate.satisfied(0.15, 0.90, 0.0, 0.50, 0.10)
    revert = StageGate(min_facing=0.0)
    assert revert.satisfied(0.40, 0.50, 0.0, 0.50, 0.40)
    assert not revert.satisfied(0.40, 0.50, 0.0, -0.50, 0.40)


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
    cfg_target = make_roll_env_cfg().rewards["supine_pose"].params["target_pos"]
    assert list(cfg_target) == list(SUPINE_TARGET)


def test_stage_metric_names_exist_in_cfgs():
    # The dashboard hint printed by `g1 train -- --stage X` must name a
    # reward the stage actually logs.
    from lab.train import STAGE_METRICS

    assert STAGE_METRICS["roll"] in make_roll_env_cfg().rewards
    assert STAGE_METRICS["standup"] in make_standup_env_cfg().rewards
