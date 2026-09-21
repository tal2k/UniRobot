"""Roll (v2 funnel stage) task configuration — any fall -> supine.

Episodes start from a random sprawl on the ground (supine, prone, side, any
yaw, low height, joints offset up to 0.6 rad). The goal is the canonical
supine neutral pose (flat on back, extended limbs) that the GETUP policy
starts from. Shaping follows HumanUP: match the face-up gravity vector;
the sparse gate mirrors the deployment ROLL gate (low + face-up).

Rolling over is the easy sub-problem (~98% in HumanUP) that makes the hard
one (standing up) start-family-independent. No feet/head terms: rolling
needs only orientation + pose feedback plus self-collision guarding.

Run: `g1 train -- --task getup --stage roll`.
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

# All 29 joints for pose tracking (shared canonical target).
_ALL_JOINTS = (".*",)


def make_roll_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the Roll (any fall -> supine) task configuration."""

  # Rolling needs only self-collision feedback (no feet/head terms).
  self_collision_cfg = sc.make_sensors()["self_collision"]

  ##
  # Events
  ##

  events = {
    # Random sprawl, low to the ground: supine, prone and side starts.
    # Upper z bound stays at 0.30 so a seated start cannot earn the lying
    # reward by deliberately collapsing.
    "reset_fallen": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (0.06, 0.30),
          "roll": (-3.14, 3.14),
          "pitch": (-1.57, 1.57),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.6, 0.6),
        "velocity_range": (-0.2, 0.2),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    **sc.make_drift_events(),
  }

  ##
  # Rewards
  ##

  rewards = {
    # --- task: end up face-up, low, near the supine pose ---
    "roll_success": RewardTermCfg(
      func=mdp.roll_success,
      weight=3.0,
      params={
        "max_height": 0.35,
        "max_facing": -0.5,
      },
    ),
    "face_up": RewardTermCfg(
      func=mdp.face_up_gravity,
      weight=1.5,
      params={"std": 0.5},
    ),
    "supine_pose": RewardTermCfg(
      func=mdp.target_pose,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINTS),
        "std": 0.5,
        "target_pos": mdp.SUPINE_TARGET,
      },
    ),
    "torso_horizontal": RewardTermCfg(
      func=mdp.torso_horizontal,
      weight=0.5,
      params={"std": 0.35},
    ),
    "symmetry": RewardTermCfg(
      func=mdp.mirror_symmetry,
      weight=-0.02,
      params={},
    ),
    # --- discovery gentleness: weak regularization ---
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
  # Terminations (timeout only — the fall is the start state)
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
      sensors=(self_collision_cfg,),
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
