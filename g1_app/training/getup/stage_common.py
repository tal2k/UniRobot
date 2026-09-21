"""Shared building blocks for the staged get-up env cfgs.

One 94-dim actor contract, one sensor set, one sim/viewer config: the three
stage factories (Reposition, SitUp, Rise) differ only in reset distribution,
rewards and episode length. Keeping the plumbing here prevents the policy
obs/action contract from drifting between stages.
"""

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import training.getup.mdp as mdp


def make_sensors() -> dict[str, ContactSensorCfg]:
  """The five get-up contact sensors, keyed by term name."""
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
  # Upper-body ground contact (pelvis/torso/arms — legs and feet excluded).
  # Penalized only while the body is high: lying flat is free, propping your
  # head/elbows with a raised pelvis is not.
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
  # Head/neck ground contact — blocks head-bridging exploits (the robot must
  # not push its head/neck into the floor to recover).  The G1 model does not
  # expose a separate `head_link`; this sensor matches the torso which carries
  # the head, and works together with `bad_support` / `stand_on_feet`.
  head_contact_cfg = ContactSensorCfg(
    name="head_contact",
    primary=ContactMatch(
      mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  return {
    "self_collision": self_collision_cfg,
    "left_foot_contact": left_foot_cfg,
    "right_foot_contact": right_foot_cfg,
    "upper_body_touch": upper_touch_cfg,
    "head_contact": head_contact_cfg,
  }


def make_observations() -> dict[str, ObservationGroupCfg]:
  """Actor 3+3+1+29+29+29 = 94 for the 29-DoF G1 (same for every stage)."""
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
  return {
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


def make_metrics() -> dict[str, MetricsTermCfg]:
  return {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }


def make_actions() -> dict[str, ActionTermCfg]:
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }


def make_drift_events() -> dict[str, EventTermCfg]:
  """Startup domain randomization shared by every get-up stage."""
  return {
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


def make_sim_cfg() -> SimulationCfg:
  return SimulationCfg(
    nconmax=35,
    njmax=300,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )


def make_viewer_cfg() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="",  # Set per-robot.
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )
