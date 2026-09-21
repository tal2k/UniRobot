"""RL configuration for the staged Unitree G1 get-up tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _runner_cfg(experiment_name: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
  """Shared runner config: discovery-friendly PPO (same net for every stage)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )


def unitree_g1_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Legacy single-policy get-up runner (experiment g1_getup)."""
  return _runner_cfg("g1_getup", 5001)


def unitree_g1_getup_reposition_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Reposition (Stage A) runner."""
  return _runner_cfg("g1_getup_reposition", 3000)


def unitree_g1_getup_situp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """SitUp (Stage B) runner."""
  return _runner_cfg("g1_getup_situp", 3000)


def unitree_g1_getup_rise_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Rise (Stage C) runner."""
  return _runner_cfg("g1_getup_rise", 5001)
