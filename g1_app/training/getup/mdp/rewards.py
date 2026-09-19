from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _robot(env: ManagerBasedRlEnv) -> Entity:

  asset: Entity = env.scene["robot"]
  return asset


def stand_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
) -> torch.Tensor:
  """Reward root height approaching standing height (exp kernel)."""
  h = _robot(env).data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square((target_height - h) / std))


def upright_bonus(
  env: ManagerBasedRlEnv,
  std: float,
) -> torch.Tensor:
  """Reward torso being upright (projected gravity close to [0, 0, -1])."""
  g_xy = _robot(env).data.projected_gravity_b[:, :2]
  tilt_sq = torch.sum(torch.square(g_xy), dim=1)
  return torch.exp(-tilt_sq / std**2)


def stand_pose(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward joints matching the nominal standing pose (exp kernel on MSE)."""
  asset = _robot(env)
  ids = asset_cfg.joint_ids
  q = asset.data.joint_pos if ids is None else asset.data.joint_pos[:, ids]
  q0 = torch.as_tensor(asset.data.default_joint_pos, device=q.device, dtype=q.dtype)
  q0 = q0.flatten().unsqueeze(0)  # [1, num_joints], broadcasts over envs
  q0 = q0 if ids is None else q0[:, ids]
  mse = torch.mean(torch.square(q - q0), dim=1)
  return torch.exp(-mse / std**2)


def stand_success(
  env: ManagerBasedRlEnv,
  min_height: float,
  max_tilt: float,
) -> torch.Tensor:
  """Sparse bonus: 1 when standing tall and upright, else 0."""
  h = _robot(env).data.root_link_pos_w[:, 2]
  tilt = torch.norm(_robot(env).data.projected_gravity_b[:, :2], dim=1)
  return ((h > min_height) & (tilt < max_tilt)).float()


def _contact_found(sensor) -> torch.Tensor:
  """Any-contact boolean per env, robust to [B,N] / [B,N,1] layouts."""
  assert sensor.data.found is not None
  return sensor.data.found.reshape(sensor.data.found.shape[0], -1).any(dim=-1)


def _contact_force(sensor) -> torch.Tensor:
  """Total contact-force magnitude per env."""
  from mjlab.sensor import ContactSensor

  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  f = sensor.data.force.reshape(sensor.data.force.shape[0], -1, 3)
  return torch.norm(f, dim=-1).sum(dim=-1)


def feet_force(
  env: ManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  target_force: float,
) -> torch.Tensor:
  """Dense support signal: clipped total foot-ground force (HUMANUP-style).

  Rewards pushing through the feet continuously — the gradient that leads
  out of head-bridging exploits and into real standing.
  """
  from mjlab.sensor import ContactSensor

  fl = _contact_force(env.scene[left_sensor])
  fr = _contact_force(env.scene[right_sensor])
  assert isinstance(env.scene[left_sensor], ContactSensor)
  return torch.clamp((fl + fr) / target_force, 0.0, 1.0)


def stand_on_feet(
  env: ManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  min_height: float,
  max_tilt: float,
) -> torch.Tensor:
  """Bonus 1 when tall, upright AND supported by both feet (not head/hands)."""
  h = _robot(env).data.root_link_pos_w[:, 2]
  tilt = torch.norm(_robot(env).data.projected_gravity_b[:, :2], dim=1)
  bl = _contact_found(env.scene[left_sensor])
  br = _contact_found(env.scene[right_sensor])
  return (bl & br & (h > min_height) & (tilt < max_tilt)).float()


def bad_support(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_height: float,
) -> torch.Tensor:
  """Penalize pelvis/torso/arm ground contact while the body is high.

  Lying flat is free (height gate); propping head/elbows with a raised
  pelvis — the observed exploit — scores nothing and costs here.
  """
  h = _robot(env).data.root_link_pos_w[:, 2]
  touching = _contact_found(env.scene[sensor_name])
  return (touching & (h > min_height)).float()


def no_head_contact(
  env: ManagerBasedRlEnv,
  min_height: float,
) -> torch.Tensor:
  """Penalize head/neck touching ground when the body is not flat.

  The robot must not use its head to bridge to the floor — this blocks the
  common exploit of pushing the head down to gain contact while the torso
  remains raised.
  """
  from mjlab.sensor import ContactSensor

  head_sensor = env.scene["head_contact"]
  assert isinstance(head_sensor, ContactSensor)
  h = _robot(env).data.root_link_pos_w[:, 2]
  touching = _contact_found(head_sensor)
  return (touching & (h > min_height)).float()


def pelvis_rising(
  env: ManagerBasedRlEnv,
  std: float,
) -> torch.Tensor:
  """Reward pelvis rising toward standing height (exp kernel).

  Shapes the early-to-mid rise so the agent learns a smooth upward
  trajectory rather than a head-down / butt-up collapse.
  """
  pelvis_pos = _robot(env).data.root_link_pos_w[:, 2]
  target = 0.78
  return torch.exp(-torch.square((target - pelvis_pos) / std))


def com_vel_z(
  env: ManagerBasedRlEnv,
  std: float,
) -> torch.Tensor:
  """Reward positive vertical CoM velocity (robot is rising).

  Directly shapes the rising motion — the agent gets immediate gradient
  when its center of mass moves up, blocking head-down / butt-up shortcuts.
  """

  asset: Entity = env.scene["robot"]
  # root link is typically the pelvis/base; its z-velocity tells us rising speed
  vz = asset.data.root_link_vel_w[:, 2]
  return torch.exp(-torch.square(vz / std))


def _mirror_pairs(joint_names: tuple[str, ...]) -> list[tuple[int, int, float]]:
  """Left/right joint index pairs + mirror sign.

  Pitch-like joints (axis y) take the same sign on both sides; roll/yaw
  joints mirror with opposite sign. Waist stays unpaired.
  """
  idx = {n: i for i, n in enumerate(joint_names)}
  pairs = []
  for name, i in idx.items():
    if not name.startswith("left_"):
      continue
    other = "right_" + name[len("left_"):]
    if other not in idx:
      continue
    # Pitch-like joints take the same sign on both sides; roll/yaw mirror.
    if "roll" in name or "yaw" in name:
      sign = -1.0
    else:
      sign = 1.0
    pairs.append((i, idx[other], sign))
  return pairs


def mirror_symmetry(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Soft bilateral symmetry cost on the latest actions (HUMANUP-style).

  Keeps the weight tiny: it halves the search space and strangles
  one-sided hacks without forbidding genuinely asymmetric recoveries.
  """
  asset = _robot(env)
  a = env.action_manager.action
  total = torch.zeros(a.shape[0], device=a.device, dtype=a.dtype)
  for i, j, sign in _mirror_pairs(tuple(asset.joint_names)):
    total = total + torch.square(a[:, i] - sign * a[:, j])
  return total / max(1, len(_mirror_pairs(tuple(asset.joint_names))))
