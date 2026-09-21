from core.brace import (  # noqa: E402
    BRACING,
    HOLD,
    NOMINAL,
    BraceGuard,
    fall_trigger,
)
from training.getup.brace_env_cfg import make_brace_env_cfg  # noqa: E402


def test_fall_trigger_calm_standing():
  assert not fall_trigger(0.78, 0.05)
  assert not fall_trigger(0.78, 0.20, vz=-0.3, angvel=0.5)


def test_fall_trigger_tip_drop_spin():
  assert fall_trigger(0.78, 0.80)                      # tipping over
  assert fall_trigger(0.78, 0.40, vz=-1.5)              # dropping fast
  assert fall_trigger(0.78, 0.20, angvel=2.5)           # spinning
  assert not fall_trigger(0.30, 0.90, vz=-2.0)          # low: already down


def test_guard_stays_nominal_when_calm():
  g = BraceGuard()
  for _ in range(30):
    assert g.update(0.78, 0.05) == NOMINAL


def test_guard_braces_then_holds_separated():
  g = BraceGuard()
  assert g.update(0.78, 0.80, 0.5, vz=-0.2) == BRACING
  # Lands and settles -> HOLD (never a get-up stage).
  for _ in range(30):
    g.update(0.20, 0.90, 0.0)
  assert g.state == HOLD
  # Latch: stays put, no chaining.
  assert g.update(0.15, 0.90, 0.0, -0.9, 0.1) == HOLD
  g.reset()
  assert g.state == NOMINAL


def test_guard_timeout_holds():
  g = BraceGuard(dt=0.5)
  assert g.update(0.78, 0.80) == BRACING
  for _ in range(10):  # never settles: budget exhausts
    g.update(0.60, 0.80, 1.0)
  assert g.state == HOLD


def test_guard_stumble_back_to_nominal():
  g = BraceGuard()
  assert g.update(0.78, 0.80) == BRACING
  assert g.update(0.78, 0.10, 0.05) == NOMINAL  # caught itself: nominal


def test_brace_cfg_build():
  cfg = make_brace_env_cfg()
  assert {"torso_impact", "arm_impact", "leg_impact", "pelvis_impact",
          "brace_success", "landed_face_up"} <= set(cfg.rewards)
  assert abs(cfg.episode_length_s - 3.0) < 1e-9
  assert cfg.decimation == 4
  assert set(cfg.terminations) == {"time_out"}
  names = [s.name for s in cfg.scene.sensors]
  assert "torso_impact" in names and "arm_impact" in names
  assert cfg.events["reset_falling"].func.__name__ == "reset_falling"
  # Same 94-dim actor contract as every other stage.
  assert list(cfg.observations["actor"].terms) == [
    "base_ang_vel", "projected_gravity", "base_height",
    "joint_pos", "joint_vel", "actions"]
  # No wrist lock: arms are the brace tool. No rise shaping: falling is
  # the job.
  assert "wrist_pose_l2" not in cfg.rewards
  assert "com_vel_z" not in cfg.rewards
  assert "lift_assist" not in cfg.events


def test_brace_task_registered():
  from mjlab.tasks.registry import list_tasks

  import training.getup.config.g1  # noqa: F401 (registers the tasks)

  assert "Unitree-G1-Getup-Brace" in list_tasks()


def test_train_stage_map_has_brace():
  from lab.train import STAGE_METRICS, STAGE_TASK_IDS

  assert STAGE_TASK_IDS["brace"] == "Unitree-G1-Getup-Brace"
  assert STAGE_METRICS["brace"] == "brace_success"
