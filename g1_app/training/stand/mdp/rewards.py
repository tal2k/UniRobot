from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def stand_still(
  env: ManagerBasedRlEnv,
  lin_std: float = 0.3,
  ang_std: float = 0.5,
) -> torch.Tensor:
  """Reward the root staying still (small lin/ang velocity, exp kernel).

  Unlike frozen-PD holding, stepping to catch balance is allowed: only the
  root velocity is penalized, not foot movement. Push recovery via a step
  costs a brief dip here but earns back the upright/height/success terms.
  """
  from mjlab.entity import Entity

  asset: Entity = env.scene["robot"]
  lin = asset.data.root_link_vel_w[:, :2]
  ang = asset.data.root_link_ang_vel_w
  lin_err = torch.sum(torch.square(lin / lin_std), dim=1)
  ang_err = torch.sum(torch.square(ang / ang_std), dim=1)
  return torch.exp(-(lin_err + ang_err))


def feet_width(
  env: ManagerBasedRlEnv,
  left_body: str = "left_ankle_roll_link",
  right_body: str = "right_ankle_roll_link",
  target_width: float = 0.20,
  std: float = 0.10,
  min_height: float = 0.70,
) -> torch.Tensor:
  """Cost for standing wider than the hips (returns >= 0, use a negative weight).

  Wide stances are the optimal push-rejection machine, so without this term
  the policy spreads its legs to farm the stand_success bonus. Only the
  *excess* beyond hip width is penalized, and only while standing tall
  (height gate): recovery steps taken from a crouch — when the robot dips
  below min_height mid-stumble — are free, so catching balance with a step
  is never punished.
  """
  from mjlab.entity import Entity

  asset: Entity = env.scene["robot"]
  left_ids, _ = asset.find_bodies(left_body)
  right_ids, _ = asset.find_bodies(right_body)
  pos = asset.data.body_link_pos_w
  lateral = torch.abs(pos[:, left_ids[0], 1] - pos[:, right_ids[0], 1])
  excess = torch.clamp(lateral - target_width, min=0.0)
  tall = (asset.data.root_link_pos_w[:, 2] > min_height).float()
  return torch.square(excess / std) * tall
