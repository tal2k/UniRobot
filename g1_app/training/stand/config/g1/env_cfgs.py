"""Unitree G1 stand-still (in-place balance) environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)

from training.stand.stand_env_cfg import make_stand_env_cfg


def unitree_g1_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 stand-still configuration (flat ground)."""
  cfg = make_stand_env_cfg()

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
  cfg.rewards["ang_vel_damp"].params["asset_cfg"].body_names = ("torso_link",)

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
