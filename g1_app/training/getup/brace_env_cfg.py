"""Brace (independent pre-impact stage) task configuration.

Episodes start from doomed falls — standing throws (a push the balance
policy cannot catch) or mid-air tumbles — matching the deployment
``fall_trigger`` domain. The policy has ~0.5 s before touchdown to
reconfigure the body, then rides out the landing. Short episodes (3 s):
fall + touchdown + settle, so impact penalties propagate back to the
pre-impact actions (SafeFall-style credit assignment).

Rewards price DAMAGE, never pose (Shi et al. 2025: a direct "arms out"
reward hacks into stiff straight arms that would snap). The same impact
force costs 10x on the torso but is cheap on the arms, so protective
contact placement emerges from the price gradient. No wrist lock (arms are
the tool here), no lift assist, no rise shaping — falling is the job.

Model honesty: the G1 XML exposes no separate head link, so the torso
sensor doubles as the vulnerable-core proxy (head + chest slam together).
A faceplant and a chest-slam both spike it; a distributed back landing
does not.

This stage is SEPARATED from the get-up chain: it never transitions to
ROLL/GETUP in deployment (exits to a passive IDLE hold), and its training
never touches the roll/standup tasks.

Run: `g1 train -- --task getup --stage brace`.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

import training.getup.mdp as mdp
from training.getup import stage_common as sc


def _impact_sensor(name: str, pattern: str, mode: str = "subtree") -> ContactSensorCfg:
  """Ground-impact force sensor for one body group (BRACE damage pricing)."""
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode=mode, pattern=pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def make_brace_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the Brace (fall damage minimization) task configuration."""

  shared = sc.make_sensors()
  sensors = {
    **{k: shared[k] for k in ("self_collision", "left_foot_contact",
                              "right_foot_contact")},
    # Vulnerable core (torso carries the head: no separate head link).
    "torso_impact": _impact_sensor("torso_impact", r"^torso_link$", mode="body"),
    # Cheap absorbers: upper arms/forearms/hands take the fall.
    "arm_impact": _impact_sensor(
      "arm_impact", r"^(.*_shoulder_.*|.*_elbow.*|.*_wrist.*)$"),
    "leg_impact": _impact_sensor(
      "leg_impact", r"^(.*_hip_.*|.*_knee.*|.*_ankle_.*)$"),
    "pelvis_impact": _impact_sensor("pelvis_impact", r"^pelvis$", mode="body"),
  }

  ##
  # Events
  ##

  events = {
    # Doomed falls: standing throws + mid-air tumbles (trigger domain).
    "reset_falling": EventTermCfg(
      func=mdp.reset_falling,
      mode="reset",
      params={
        "throw_prob": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.15, 0.15),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    **sc.make_drift_events(),
  }

  ##
  # Rewards: price the damage per body group, bonus the safe ending
  ##

  rewards = {
    # --- damage pricing: same force, different price per group ---
    "torso_impact": RewardTermCfg(
      func=mdp.impact_force,
      weight=-3.0,
      params={"sensor_name": "torso_impact", "max_force": 3000.0},
    ),
    "pelvis_impact": RewardTermCfg(
      func=mdp.impact_force,
      weight=-1.0,
      params={"sensor_name": "pelvis_impact", "max_force": 2500.0},
    ),
    "leg_impact": RewardTermCfg(
      func=mdp.impact_force,
      weight=-0.8,
      params={"sensor_name": "leg_impact", "max_force": 2000.0},
    ),
    "arm_impact": RewardTermCfg(
      func=mdp.impact_force,
      weight=-0.3,
      params={"sensor_name": "arm_impact", "max_force": 1500.0},
    ),
    # --- safe ending: settled on the ground, or stumbled back to stand ---
    "brace_success": RewardTermCfg(
      func=mdp.brace_success,
      weight=2.0,
      params={
        "max_height": 0.35,
        "max_speed": 0.30,
        "min_height": 0.70,
        "max_tilt": 0.30,
      },
    ),
    # --- shaping: back-landings are head-safe; feet loading is ideal ---
    "landed_face_up": RewardTermCfg(
      func=mdp.roll_success,
      weight=0.5,
      params={"max_height": 0.35, "max_facing": -0.5},
    ),
    "face_up": RewardTermCfg(
      func=mdp.face_up_gravity,
      weight=0.3,
      params={"std": 0.5},
    ),
    "feet_force": RewardTermCfg(
      func=mdp.feet_force,
      weight=0.5,
      params={
        "left_sensor": "left_foot_contact",
        "right_sensor": "right_foot_contact",
        "target_force": 250.0,
      },
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
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
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
    episode_length_s=3.0,
  )
