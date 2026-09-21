"""SitUp (Stage B) task configuration — staged get-up phase 2.

Episodes start from the lying family Stage A ends in: supine, prone or on
the side, near the ground, joints offset up to 0.4 rad. The goal is a
*crouch*: pelvis lifted (target 0.55), torso mostly upright, both feet
loaded. Rewards reuse the final-rise terms with stage-B targets (wide
height kernel, crouch-level support gates).

Advance gate (deployment, see `core/getup_stages.py`): height > 0.55 and
|projected gravity xy| < 0.70, held 10 policy ticks.

Run: `g1 train -- --task getup --stage B`.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg

import training.getup.mdp as mdp
from training.getup import stage_common as sc


def make_situp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the SitUp (Stage B) task configuration."""

  sensors = sc.make_sensors()
  left_foot_cfg = sensors["left_foot_contact"]
  right_foot_cfg = sensors["right_foot_contact"]
  upper_touch_cfg = sensors["upper_body_touch"]
  self_collision_cfg = sensors["self_collision"]

  ##
  # Events
  ##

  events = {
    # Lying start family (Stage A output distribution).
    "reset_lying": EventTermCfg(
      func=mdp.reset_lying_pose,
      mode="reset",
      params={},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_to_target,
      mode="reset",
      params={
        "target_pos": mdp.SUPINE_TARGET,
        "position_range": (-0.2, 0.2),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    **sc.make_drift_events(),
  }

  ##
  # Rewards (existing rise terms, stage-B targets)
  ##

  rewards = {
    # --- task: rise from the floor onto loaded feet ---
    "pelvis_rising": RewardTermCfg(
      func=mdp.pelvis_rising,
      weight=1.5,
      params={"std": 0.35, "target_height": 0.55},
    ),
    "stand_height": RewardTermCfg(
      func=mdp.stand_height,
      weight=1.0,
      params={"target_height": 0.55, "std": 0.35},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_bonus,
      weight=1.0,
      params={"std": 0.5},
    ),
    "feet_force": RewardTermCfg(
      func=mdp.feet_force,
      weight=1.0,
      params={
        "left_sensor": left_foot_cfg.name,
        "right_sensor": right_foot_cfg.name,
        "target_force": 250.0,
      },
    ),
    "stand_on_feet": RewardTermCfg(
      func=mdp.stand_on_feet,
      weight=2.0,
      params={
        "left_sensor": left_foot_cfg.name,
        "right_sensor": right_foot_cfg.name,
        "min_height": 0.50,
        "max_tilt": 0.70,
      },
    ),
    "stand_pose": RewardTermCfg(
      func=mdp.stand_pose,
      weight=0.5,
      params={
        "std": 0.7,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "bad_support": RewardTermCfg(
      func=mdp.bad_support,
      weight=-3.0,
      params={
        "sensor_name": upper_touch_cfg.name,
        "min_height": 0.45,
      },
    ),
    "no_head_contact": RewardTermCfg(
      func=mdp.no_head_contact,
      weight=-2.0,
      params={"min_height": 0.40},
    ),
    "com_vel_z": RewardTermCfg(
      func=mdp.com_vel_z,
      weight=2.0,
      params={"std": 0.8},
    ),
    "symmetry": RewardTermCfg(
      func=mdp.mirror_symmetry,
      weight=-0.02,
      params={},
    ),
    # --- gentleness: smooth, low-effort, hardware-safe motion ---
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.5,
      params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    ),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-3.0e-6),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-3.0e-4),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Terminations (timeout only — lying down is the start state)
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      sensors=tuple(sensors.values()),
      num_envs=1,
      extent=2.0,
    ),
    observations=sc.make_observations(),
    actions=sc.make_actions(),
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=sc.make_metrics(),
    viewer=sc.make_viewer_cfg(),
    sim=sc.make_sim_cfg(),
    decimation=4,
    episode_length_s=10.0,
  )
