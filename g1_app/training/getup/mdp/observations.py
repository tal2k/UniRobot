from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Root-link height above ground, shape (num_envs, 1)."""
  from mjlab.entity import Entity

  asset: Entity = env.scene["robot"]
  return asset.data.root_link_pos_w[:, 2:3]
