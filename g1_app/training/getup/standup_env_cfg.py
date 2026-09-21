"""StandUp (v2 merged get-up) task configuration — lying family -> stand.

Episodes start mixed: standing (15%), crouch key-state (25%) or the lying
family the ROLL stage ends in (60%). Pure lying starts starve discovery of
stand reward (sit-up-without-standing plateau); tasting the full chain early
plus an upward pelvis assist that fades within each episode (HoST-style)
breaks it. Wrists are locked by reward (they add dims, not leverage).

Rewards merge the old SitUp + Rise phases height-banded HoST-style: the
crouch-loading terms pay off at 0.55 m, the stand terms at 0.78 m, so a
single policy learns sit-up *and* rise without a deployment switch between
them. Most support gates already self-band by height (stand_on_feet,
bad_support, no_head_contact).

Train discovery first (weak regularization, this recipe), then refine with
--resume under stronger regularization (HUMANUP-style two-phase, same as the
legacy Stage-II recipe this derives from).

Run: `g1 train -- --task getup --stage standup`.
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


def make_standup_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the merged supine-to-stand task configuration."""

  sensors = sc.make_sensors()
  left_foot_cfg = sensors["left_foot_contact"]
  right_foot_cfg = sensors["right_foot_contact"]
  upper_touch_cfg = sensors["upper_body_touch"]
  self_collision_cfg = sensors["self_collision"]

  ##
  # Events
  ##

  events = {
    # Mixed starts: standing + crouch key-state + lying family. Pure lying
    # starts starve discovery of stand reward; tasting the full chain early
    # breaks the sit-up-without-standing plateau.
    "reset_mixed": EventTermCfg(
      func=mdp.reset_mixed_starts,
      mode="reset",
      params={
        "stand_prob": 0.15,
        "crouch_prob": 0.25,
        "stand_noise": 0.1,
        "crouch_noise": 0.15,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Upward pelvis assist with a within-episode fade (HoST-style): full
    # help early, must finish unassisted. Training-only; deployment never
    # applies it. ~180 N is ~half G1 body weight.
    "lift_assist": EventTermCfg(
      func=mdp.lift_assist,
      mode="interval",
      interval_range_s=(0.2, 0.5),
      params={
        "max_force": 180.0,
        "full_until": 0.3,
        "zero_after": 0.6,
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),
    **sc.make_drift_events(),
  }

  ##
  # Rewards (crouch terms pay at 0.55 m, stand terms at 0.78 m)
  ##

  rewards = {
    # --- task: rise from the floor onto loaded feet, then stand tall ---
    "pelvis_rising": RewardTermCfg(
      func=mdp.pelvis_rising,
      weight=1.5,
      params={"std": 0.35, "target_height": 0.78},
    ),
    "stand_height": RewardTermCfg(
      func=mdp.stand_height,
      weight=1.0,
      params={"target_height": 0.78, "std": 0.35},
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
      weight=2.5,
      params={
        "left_sensor": left_foot_cfg.name,
        "right_sensor": right_foot_cfg.name,
        # Loosened to the crouch so the sit-up phase earns signal; the
        # stand_success bonus below pays for the full rise.
        "min_height": 0.55,
        "max_tilt": 0.50,
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
    "stand_success": RewardTermCfg(
      func=mdp.stand_success,
      weight=3.0,
      params={"min_height": 0.70, "max_tilt": 0.3},
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
    "wrist_pose_l2": RewardTermCfg(
      func=mdp.wrist_pose_l2,
      weight=-1.0,
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
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-6),
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
    episode_length_s=12.0,
  )
