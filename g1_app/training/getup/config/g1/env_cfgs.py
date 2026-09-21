"""Unitree G1 fall-recovery (get-up) environment configurations.

Three staged tasks + the legacy single-policy task:
  Unitree-G1-Getup-Reposition  Stage A (reposition_env_cfg)
  Unitree-G1-Getup-SitUp       Stage B (situp_env_cfg)
  Unitree-G1-Getup-Rise        Stage C (getup_env_cfg, refinement recipe)
  Unitree-G1-Getup             legacy single-policy task (Stage C recipe)
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)

from training.getup.getup_env_cfg import make_getup_env_cfg
from training.getup.reposition_env_cfg import make_reposition_env_cfg
from training.getup.situp_env_cfg import make_situp_env_cfg


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


def unitree_g1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 get-up configuration (flat ground)."""
  return _finalize_g1_cfg(make_getup_env_cfg(), play)


def unitree_g1_getup_reposition_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 Reposition (Stage A) configuration."""
  return _finalize_g1_cfg(make_reposition_env_cfg(), play)


def unitree_g1_getup_situp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 SitUp (Stage B) configuration."""
  return _finalize_g1_cfg(make_situp_env_cfg(), play)


def unitree_g1_getup_rise_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 Rise (Stage C) configuration."""
  return _finalize_g1_cfg(make_getup_env_cfg(), play)
