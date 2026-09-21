from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_getup_env_cfg,
  unitree_g1_getup_reposition_env_cfg,
  unitree_g1_getup_rise_env_cfg,
  unitree_g1_getup_situp_env_cfg,
)
from .rl_cfg import (
  unitree_g1_getup_ppo_runner_cfg,
  unitree_g1_getup_reposition_ppo_runner_cfg,
  unitree_g1_getup_rise_ppo_runner_cfg,
  unitree_g1_getup_situp_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup",
  env_cfg=unitree_g1_getup_env_cfg(),
  play_env_cfg=unitree_g1_getup_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-Reposition",
  env_cfg=unitree_g1_getup_reposition_env_cfg(),
  play_env_cfg=unitree_g1_getup_reposition_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_reposition_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-SitUp",
  env_cfg=unitree_g1_getup_situp_env_cfg(),
  play_env_cfg=unitree_g1_getup_situp_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_situp_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-Rise",
  env_cfg=unitree_g1_getup_rise_env_cfg(),
  play_env_cfg=unitree_g1_getup_rise_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_rise_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
