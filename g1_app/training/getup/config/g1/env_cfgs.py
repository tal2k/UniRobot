"""Unitree G1 fall-recovery (get-up) environment configurations.

v2 pipeline (see g1_app/docs/getup_staged_policies.md §14):
  Unitree-G1-Getup-Roll      Roll: any fall -> supine (roll_env_cfg)
  Unitree-G1-Getup-StandUp   StandUp: lying family -> stand, merged (standup_env_cfg)
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)

from training.getup.roll_env_cfg import make_roll_env_cfg
from training.getup.standup_env_cfg import make_standup_env_cfg


def _finalize_g1_cfg(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  """Attach the G1 robot and its per-robot parameters to a stage cfg."""
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def unitree_g1_getup_roll_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 Roll (v2 funnel stage) configuration."""
  return _finalize_g1_cfg(make_roll_env_cfg(), play)


def unitree_g1_getup_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 StandUp (v2 merged get-up) configuration."""
  return _finalize_g1_cfg(make_standup_env_cfg(), play)
