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
