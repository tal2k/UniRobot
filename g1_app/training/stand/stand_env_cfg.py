"""In-place balance (stand-still) task configuration.

Every episode starts from the nominal standing pose — upright, root at
standing height, joints at default plus small noise (the "walking zero"
position). Strong shoves arrive at random intervals; stepping to catch
balance is allowed, falling ends the episode.

Run: `g1 train -- --task stand` (or `g1 train-stand`).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import training.stand.mdp as mdp


def make_stand_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base stand-still balance task configuration."""

  ##
  # Sensors
  ##

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Per-foot ground contact (for feet-loading rewards).
  left_foot_cfg = ContactSensorCfg(
    name="left_foot_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^left_ankle_roll_link$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  right_foot_cfg = ContactSensorCfg(
    name="right_foot_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^right_ankle_roll_link$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  # Upper-body ground contact — penalized only while the body is high.
  upper_touch_cfg = ContactSensorCfg(
    name="upper_body_touch",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(pelvis|torso_link|waist_.*|.*_shoulder_.*|.*_elbow.*|.*_wrist.*)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  head_contact_cfg = ContactSensorCfg(
    name="head_contact",
    primary=ContactMatch(
      mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  ##
  # Observations (actor 3+3+1+29+29+29 = 94 for the 29-DoF G1)
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_height": ObservationTermCfg(
      func=mdp.base_height,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    # Standing start: nominal pose plus small noise (not a fallen pose).
    "reset_root": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.76, 0.80),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.05, 0.05),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Strong shoves at random intervals — stepping to recover is allowed.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
          "z": (-0.1, 0.1),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-1.0, 1.0),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # --- task: stay tall, upright, supported by both feet ---
    "stand_height": RewardTermCfg(
      func=mdp.stand_height,
      weight=1.0,
      params={"target_height": 0.78, "std": 0.15},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright_bonus,
      weight=1.5,
      params={"std": 0.3},
    ),
    "feet_force": RewardTermCfg(
      func=mdp.feet_force,
      weight=0.5,
      params={
        "left_sensor": left_foot_cfg.name,
        "right_sensor": right_foot_cfg.name,
        "target_force": 300.0,
      },
    ),
    "stand_on_feet": RewardTermCfg(
      func=mdp.stand_on_feet,
      weight=3.0,
      params={
        "left_sensor": left_foot_cfg.name,
        "right_sensor": right_foot_cfg.name,
        "min_height": 0.70,
        "max_tilt": 0.3,
      },
    ),
    "stand_pose": RewardTermCfg(
      func=mdp.stand_pose,
      weight=1.0,
      params={
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "stand_success": RewardTermCfg(
      func=mdp.stand_success,
      weight=5.0,
      params={"min_height": 0.70, "max_tilt": 0.3},
    ),
    # --- stillness: quiet root, recovery steps allowed ---
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=1.5,
      params={"lin_std": 0.3, "ang_std": 0.5},
    ),
    "bad_support": RewardTermCfg(
      func=mdp.bad_support,
      weight=-3.0,
      params={
        "sensor_name": upper_touch_cfg.name,
        "min_height": 0.55,
      },
    ),
    "no_head_contact": RewardTermCfg(
      func=mdp.no_head_contact,
      weight=-2.0,
      params={"min_height": 0.45},
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
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-6),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Terminations (timeout + fall — unlike get-up, falling ends the episode)
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
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
      sensors=(self_collision_cfg, left_foot_cfg, right_foot_cfg,
               upper_touch_cfg, head_contact_cfg),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
