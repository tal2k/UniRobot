from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# Supine neutral target, canonical in `core/getup_stages.py` (shared with the
# deployment gate + recorder so the three can never drift apart).
# Order: 6 L leg, 6 R leg, 3 waist, 7 L arm, 7 R arm (G1 29-DoF).
try:
  from core.getup_stages import SUPINE_TARGET  # noqa: F401 (re-exported via mdp)
except ImportError:  # installed as g1_app.training.*
  from g1_app.core.getup_stages import SUPINE_TARGET  # noqa: F401 (re-exported via mdp)


def reset_lying_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  z_range: tuple[float, float] = (0.06, 0.20),
  mode_probs: tuple[float, float, float] = (0.45, 0.35, 0.20),
  angle_noise: float = 0.4,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset the root to a lying pose: supine, prone, or on the side.

  A single box over roll/pitch would also include upside-down and
  balanced-on-head poses; this samples the three lying families the SitUp
  stage is meant to handle (the Reposition end distribution). ``mode_probs``
  is (supine, prone, side) and must sum to <= 1.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()
  n = len(env_ids)
  device = env.device

  z = sample_uniform(z_range[0], z_range[1], (n,), device=device)
  offsets = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
  positions = root_states[:, 0:3] + offsets + env.scene.env_origins[env_ids]

  half_pi = math.pi / 2
  u = torch.rand(n, device=device)
  p_supine = mode_probs[0]
  p_prone = mode_probs[0] + mode_probs[1]
  zero = torch.zeros_like(u)
  pitch = torch.where(
    u < p_supine,
    torch.full_like(u, -half_pi),  # supine (belly up)
    torch.where(u < p_prone, torch.full_like(u, half_pi), zero),  # prone
  )
  side_sign = torch.where(torch.rand(n, device=device) < 0.5, -half_pi, half_pi)
  roll = torch.where(u < p_prone, zero, side_sign)  # side
  noise = sample_uniform(-angle_noise, angle_noise, (n, 2), device=device)
  roll = roll + noise[:, 0]
  pitch = pitch + noise[:, 1]
  yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
  orientations = quat_mul(
    root_states[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))

  velocities = torch.zeros((n, 6), device=device)
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_to_target(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  target_pos: list[float],
  position_range: tuple[float, float] = (-0.2, 0.2),
  velocity_range: tuple[float, float] = (-0.1, 0.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints to a specific target pose + noise.

  Unlike reset_joints_by_offset (which offsets from standing default),
  this sets joints directly to target_pos, matching the Stage A output.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None

  n = len(env_ids)
  device = env.device

  target = torch.as_tensor(target_pos, device=device, dtype=torch.float32)
  target = target.flatten().unsqueeze(0).repeat(n, 1)
  joint_pos = target[:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, device)

  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=device)

  asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids, env_ids=env_ids)
