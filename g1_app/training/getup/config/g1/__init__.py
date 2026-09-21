from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_getup_brace_env_cfg,
  unitree_g1_getup_roll_env_cfg,
  unitree_g1_getup_standup_env_cfg,
)
from .rl_cfg import (
  unitree_g1_getup_brace_ppo_runner_cfg,
  unitree_g1_getup_roll_ppo_runner_cfg,
  unitree_g1_getup_standup_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-Roll",
  env_cfg=unitree_g1_getup_roll_env_cfg(),
  play_env_cfg=unitree_g1_getup_roll_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_roll_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-StandUp",
  env_cfg=unitree_g1_getup_standup_env_cfg(),
  play_env_cfg=unitree_g1_getup_standup_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_standup_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Getup-Brace",
  env_cfg=unitree_g1_getup_brace_env_cfg(),
  play_env_cfg=unitree_g1_getup_brace_env_cfg(play=True),
  rl_cfg=unitree_g1_getup_brace_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
