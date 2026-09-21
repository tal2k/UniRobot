from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

from core.getup_stages import SUPINE_TARGET  # noqa: F401 (re-exported via mdp)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_PELVIS_CFG = SceneEntityCfg("robot", body_names=("pelvis",))

# Crouch key-state for mixed starts (29-DoF training joint order: 6 L leg,
# 6 R leg, 3 waist, 7 L arm, 7 R arm). Deep knees, hips over feet — the pose
# the sit-up must pass through. Root is placed at ~0.55 m by the reset.
CROUCH_POSE = [
  -0.5, 0.0, 0.0, 0.9, -0.4, 0.0,   # L leg
  -0.5, 0.0, 0.0, 0.9, -0.4, 0.0,   # R leg
  0.0, 0.0, 0.0,                    # waist
  0.2, 0.15, 0.0, 0.4, 0.0, 0.0, 0.0,    # L arm (mild bend)
  0.2, -0.15, 0.0, 0.4, 0.0, 0.0, 0.0,   # R arm (mirrored)
]


# Supine neutral target is canonical in `core/getup_stages.py` (shared with
# the deployment gate + recorder so the three can never drift apart; see
# the import above). Order: 6 L leg, 6 R leg, 3 waist, 7 L arm, 7 R arm.


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


def reset_mixed_starts(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stand_prob: float = 0.15,
  crouch_prob: float = 0.25,
  stand_noise: float = 0.1,
  crouch_noise: float = 0.15,
  crouch_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Mixed episode starts: standing, crouch key-state, or lying family.

  Pure lying starts starve the policy of stand reward early (the discovery
  plateau); mixing in standing poses (HumanUP) and a crouch key-state
  (HiFAR-style KSI) lets it taste the full reward chain from iteration 0.
  The lying fraction reuses the ROLL-output distribution
  (``reset_lying_pose`` + joints near supine).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  device = env.device
  u = torch.rand(len(env_ids), device=device)
  is_stand = u < stand_prob
  is_crouch = (~is_stand) & (u < stand_prob + crouch_prob)
  stand_ids = env_ids[is_stand]
  crouch_ids = env_ids[is_crouch]
  lying_ids = env_ids[(~is_stand) & (~is_crouch)]

  asset = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  # Per-env joint rows (MJLab stores (num_envs, num_joints); a flatten
  # would mix envs and crash downstream).
  default_q = torch.as_tensor(
    asset.data.default_joint_pos, device=device, dtype=torch.float32)
  if default_q.dim() == 1:
    default_q = default_q.unsqueeze(0)
  if default_q.shape[0] == 1:
    default_q = default_q.expand(len(env_ids), -1)
  else:
    default_q = default_q[env_ids]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  def _upright_quat(n: int) -> torch.Tensor:
    base = default_root_state[:1, 3:7].repeat(n, 1)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    rp = sample_uniform(-0.1, 0.1, (n, 2), device=device)
    return quat_mul(base, quat_from_euler_xyz(rp[:, 0], rp[:, 1], yaw))

  order = torch.arange(len(env_ids), device=device)
  crouch_pose = torch.as_tensor(CROUCH_POSE, device=device, dtype=torch.float32)
  for ids, rows, z, q0, noise, per_env in (
    (stand_ids, order[is_stand], None, default_q, stand_noise, True),
    (crouch_ids, order[is_crouch], crouch_height, crouch_pose, crouch_noise, False),
  ):
    if len(ids) == 0:
      continue
    root = default_root_state[env_ids][rows].clone()
    root[:, 0:2] += sample_uniform(-0.1, 0.1, (len(ids), 2), device=device)
    if z is not None:
      root[:, 2] = z + sample_uniform(-0.02, 0.02, (len(ids),), device=device)
    quat = _upright_quat(len(ids))
    positions = root[:, 0:3] + env.scene.env_origins[ids]
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, quat], dim=-1), env_ids=ids)
    asset.write_root_link_velocity_to_sim(
      torch.zeros((len(ids), 6), device=device), env_ids=ids)
    # Per-env default rows for standing, shared key pose for crouch.
    q = q0[rows].clone() if per_env else q0.unsqueeze(0).repeat(len(ids), 1)
    q += sample_uniform(-noise, noise, q.shape, device)
    limits = soft_joint_pos_limits[ids]
    q = q.clamp_(limits[..., 0], limits[..., 1])
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=device)
    asset.write_joint_state_to_sim(
      q, torch.zeros_like(q), joint_ids, env_ids=ids)

  if len(lying_ids) > 0:
    reset_lying_pose(env, lying_ids)
    reset_joints_to_target(
      env, lying_ids, target_pos=SUPINE_TARGET,
      position_range=(-0.2, 0.2), velocity_range=(-0.1, 0.1),
      asset_cfg=asset_cfg)


def reset_falling(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  throw_prob: float = 0.5,
  throw_speed: tuple[float, float] = (1.0, 2.5),
  throw_spin: float = 1.5,
  tumble_height: tuple[float, float] = (0.35, 0.85),
  tumble_speed: float = 1.5,
  tumble_spin: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Doomed-fall episode starts for the BRACE stage (independent of get-up).

  ``throw`` fraction: standing pose, knocked with a random planar velocity +
  spin (a push the balance policy cannot catch). The rest: mid-air tumbles
  (random height/orientation/velocity, already falling). Both match the
  deployment ``fall_trigger`` domain (high + tilted/dropping/spinning), so
  the trained policy starts where the switcher engages it. Joints stay near
  standing — set them with ``reset_joints_by_offset`` as a chained event.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  device = env.device
  asset = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None

  u = torch.rand(len(env_ids), device=device)
  is_throw = u < throw_prob
  throw_ids = env_ids[is_throw]
  tumble_ids = env_ids[~is_throw]

  if len(throw_ids) > 0:
    n = len(throw_ids)
    root = default_root_state[throw_ids].clone()
    root[:, 0:2] += sample_uniform(-0.1, 0.1, (n, 2), device=device)
    base = root[:, 3:7]
    rp = sample_uniform(-0.3, 0.3, (n, 2), device=device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    quat = quat_mul(base, quat_from_euler_xyz(rp[:, 0], rp[:, 1], yaw))
    positions = root[:, 0:3] + env.scene.env_origins[throw_ids]
    heading = sample_uniform(-math.pi, math.pi, (n,), device=device)
    speed = sample_uniform(throw_speed[0], throw_speed[1], (n,), device=device)
    lin = torch.stack([speed * torch.cos(heading),
                       speed * torch.sin(heading),
                       sample_uniform(-0.2, 0.2, (n,), device=device)], dim=-1)
    ang = sample_uniform(-throw_spin, throw_spin, (n, 3), device=device)
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, quat], dim=-1), env_ids=throw_ids)
    asset.write_root_link_velocity_to_sim(
      torch.cat([lin, ang], dim=-1), env_ids=throw_ids)

  if len(tumble_ids) > 0:
    n = len(tumble_ids)
    root = default_root_state[tumble_ids].clone()
    root[:, 0:2] += sample_uniform(-0.2, 0.2, (n, 2), device=device)
    root[:, 2] = sample_uniform(
      tumble_height[0], tumble_height[1], (n,), device=device)
    roll = sample_uniform(-math.pi, math.pi, (n,), device=device)
    pitch = sample_uniform(-math.pi / 2, math.pi / 2, (n,), device=device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    quat = quat_mul(root[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))
    positions = root[:, 0:3] + env.scene.env_origins[tumble_ids]
    lin = sample_uniform(-tumble_speed, tumble_speed, (n, 3), device=device)
    lin[:, 2] = sample_uniform(-1.0, 0.0, (n,), device=device)
    ang = sample_uniform(-tumble_spin, tumble_spin, (n, 3), device=device)
    asset.write_root_link_pose_to_sim(
      torch.cat([positions, quat], dim=-1), env_ids=tumble_ids)
    asset.write_root_link_velocity_to_sim(
      torch.cat([lin, ang], dim=-1), env_ids=tumble_ids)


def lift_assist(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  max_force: float = 180.0,
  full_until: float = 0.3,
  zero_after: float = 0.6,
  asset_cfg: SceneEntityCfg = _PELVIS_CFG,
) -> None:
  """Upward pelvis assist with a within-episode fade (HoST-style).

  Full ``max_force`` for the first ``full_until`` fraction of the episode,
  linearly faded to zero by ``zero_after`` — the policy gets discovery help
  early and must finish unassisted. Register with ``mode="interval"`` so the
  latched MuJoCo external force tracks the fade. ~180 N is ~half G1 body
  weight (HoST used ~60%). Training-only event: deployment never applies it.
  """
  asset = env.scene[asset_cfg.name]
  n = len(env_ids)
  device = env.device
  buf = getattr(env, "episode_length_buf", None)
  total = getattr(env, "max_episode_length", None)
  if buf is None or total is None or total <= 0:
    scale = torch.ones((n,), device=device)
  else:
    progress = buf[env_ids].float() / float(total)
    scale = torch.clamp(
      (zero_after - progress) / max(zero_after - full_until, 1e-6), 0.0, 1.0)
  forces = torch.zeros((n, 1, 3), device=device)
  forces[:, 0, 2] = max_force * scale
  torques = torch.zeros((n, 1, 3), device=device)
  body_ids = asset_cfg.body_ids
  assert isinstance(body_ids, list) and len(body_ids) == 1, \
    "lift_assist needs one body (pelvis)"
  asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids, body_ids=body_ids)
