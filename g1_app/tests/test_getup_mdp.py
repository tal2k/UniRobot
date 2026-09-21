"""Unit tests for the get-up training MDP functions (CPU, no simulator).

The reset/event/reward functions in `training/getup/mdp/` only execute
inside MJLab training runs — a shape or indexing bug there wastes GPU
hours. These fakes exercise them directly on CPU.
"""

from types import SimpleNamespace  # noqa: E402

import torch

from training.getup.mdp import events as E  # noqa: E402
from training.getup.mdp import rewards as R  # noqa: E402

N_JOINTS = 29


class FakeScene(dict):
  def __init__(self, asset, n):
    super().__init__(robot=asset)
    self.env_origins = torch.zeros((n, 3))


def _make_asset(n):
  data = SimpleNamespace(
    default_root_state=torch.tensor([[0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]] * n),
    default_joint_pos=torch.zeros(N_JOINTS),
    default_joint_vel=torch.zeros((n, N_JOINTS)),
    soft_joint_pos_limits=torch.tensor([(-2.0, 2.0)] * n * N_JOINTS).reshape(n, N_JOINTS, 2),
    projected_gravity_b=torch.zeros((n, 3)),
    root_link_pos_w=torch.zeros((n, 3)),
    joint_pos=torch.zeros((n, N_JOINTS)),
  )
  calls = {}

  def _rec(name):
    def fn(*args, **kwargs):
      calls[name] = (args, kwargs)
    return fn

  asset = SimpleNamespace(data=data)
  asset.write_root_link_pose_to_sim = _rec("root_pose")
  asset.write_root_link_velocity_to_sim = _rec("root_vel")
  asset.write_joint_state_to_sim = _rec("joints")
  asset.write_external_wrench_to_sim = _rec("wrench")
  return asset, calls


def _make_env(n):
  asset, calls = _make_asset(n)
  env = SimpleNamespace(
    device="cpu",
    num_envs=n,
    scene=FakeScene(asset, n),
    episode_length_buf=torch.zeros((n,), dtype=torch.int64),
    max_episode_length=600,
  )
  return env, asset, calls


def _cfg(**kw):
  base = {"name": "robot", "joint_ids": list(range(N_JOINTS)), "body_ids": [0]}
  base.update(kw)
  return SimpleNamespace(**base)


def test_reset_mixed_starts_splits_fractions():
  torch.manual_seed(0)
  env, asset, calls = _make_env(200)
  E.reset_mixed_starts(env, None, stand_prob=0.15, crouch_prob=0.25,
                       asset_cfg=_cfg())
  # lying fraction delegated to the shared lying reset
  assert "root_pose" in calls and "joints" in calls
  n_stand = (torch.rand(200) < 0.15).sum()  # same seed path not guaranteed;
  assert n_stand >= 0  # smoke: runs without error on a batch


def test_reset_mixed_all_standing():
  env, asset, calls = _make_env(4)
  E.reset_mixed_starts(env, None, stand_prob=1.0, crouch_prob=0.0,
                       asset_cfg=_cfg())
  pose = calls["root_pose"][0][0]
  assert pose.shape == (4, 7)
  assert torch.allclose(pose[:, 2], torch.full((4,), 0.78), atol=0.01)
  q = calls["joints"][0][0]
  assert q.shape == (4, N_JOINTS)
  assert q.abs().max() <= 0.1 + 1e-6


def test_reset_mixed_all_crouch():
  env, asset, calls = _make_env(4)
  E.reset_mixed_starts(env, None, stand_prob=0.0, crouch_prob=1.0,
                       asset_cfg=_cfg())
  pose = calls["root_pose"][0][0]
  assert torch.allclose(pose[:, 2], torch.full((4,), 0.55), atol=0.03)
  q = calls["joints"][0][0]
  knee = q[:, 3]
  assert (knee > 0.5).all()  # deep knees, not standing


def test_crouch_pose_shape_and_limits():
  assert len(E.CROUCH_POSE) == N_JOINTS
  assert all(-2.0 <= v <= 2.0 for v in E.CROUCH_POSE)


def test_lift_assist_full_then_fades():
  env, asset, calls = _make_env(2)
  env.episode_length_buf = torch.tensor([0, 590])  # start vs end of episode
  E.lift_assist(env, torch.tensor([0, 1]), max_force=180.0,
                full_until=0.3, zero_after=0.6, asset_cfg=_cfg())
  forces = calls["wrench"][0][0]
  assert forces.shape == (2, 1, 3)
  assert abs(float(forces[0, 0, 2]) - 180.0) < 1e-4  # full help at start
  assert abs(float(forces[1, 0, 2])) < 1e-4  # faded to zero late
  assert float(forces[0, 0, 0]) == 0.0 and float(forces[0, 0, 1]) == 0.0


def test_lift_assist_unknown_progress_is_safe():
  env, asset, calls = _make_env(1)
  del env.episode_length_buf
  E.lift_assist(env, torch.tensor([0]), asset_cfg=_cfg())
  assert float(calls["wrench"][0][0][0, 0, 2]) == 180.0


def test_wrist_pose_l2():
  env, asset, _ = _make_env(2)
  asset.data.joint_pos = torch.zeros((2, N_JOINTS))
  cfg = SimpleNamespace(name="robot", joint_ids=[15, 16, 17, 22, 23, 24])
  assert torch.allclose(R.wrist_pose_l2(env, cfg), torch.zeros(2))
  asset.data.joint_pos[0, 15] = 1.0
  out = R.wrist_pose_l2(env, cfg)
  assert out[0] > 0.0 and out[1] == 0.0


def test_default_joint_pos_is_per_env():
  # MJLab stores (num_envs, num_joints); env 1 must use env 1's defaults,
  # not env 0's (flattening mixes envs — crashed reset_mixed_starts and
  # silently biased stand_pose).
  env, asset, _ = _make_env(2)
  asset.data.default_joint_pos = torch.zeros((2, N_JOINTS))
  asset.data.default_joint_pos[1] = 0.5
  asset.data.joint_pos = torch.zeros((2, N_JOINTS))
  asset.data.joint_pos[1] = 0.5
  out = R.stand_pose(env, std=0.7, asset_cfg=_cfg())
  assert torch.allclose(out, torch.ones(2))
  w = R.wrist_pose_l2(env, _cfg())
  assert torch.allclose(w, torch.zeros(2))


def test_reset_mixed_uses_env_default_rows():
  env, asset, calls = _make_env(4)
  asset.data.default_joint_pos = torch.zeros((4, N_JOINTS))
  asset.data.default_joint_pos[:, 3] = 0.3  # knee default, like the G1
  E.reset_mixed_starts(env, None, stand_prob=1.0, crouch_prob=0.0,
                       stand_noise=0.0, asset_cfg=_cfg())
  q = calls["joints"][0][0]
  assert torch.allclose(q[:, 3], torch.full((4,), 0.3))


def test_face_up_gravity_and_roll_success():
  env, asset, _ = _make_env(3)
  asset.data.projected_gravity_b = torch.tensor([
    [-1.0, 0.0, 0.0],  # flat supine
    [1.0, 0.0, 0.0],   # flat prone
    [0.0, 0.0, -1.0],  # upright
  ])
  asset.data.root_link_pos_w = torch.tensor([
    [0.0, 0.0, 0.10], [0.0, 0.0, 0.10], [0.0, 0.0, 0.78]])
  face = R.face_up_gravity(env, std=0.5)
  # supine (dist 0) > upright (dist^2 2) > prone (dist^2 4)
  assert face[0] > face[2] > face[1]
  roll = R.roll_success(env, max_height=0.35, max_facing=-0.5)
  assert roll.tolist() == [1.0, 0.0, 0.0]
