"""G1 app package.

New code lives in subpackages:

- `core/`    shared sim math, config, policy bridge, terrains
- `apps/`     interactive entrypoints: GUI control, env check
- `lab/`      training loop, dashboard, headless recorder
- `tools/`    one-off generators (test terrains)
- `training/` fall-recovery task definition (env, rewards, configs)
- `tests/`    pytest suite (`g1 verify` runs the fast subset)

Unified CLI: `python -m g1_app.cli <command>` (installed as `g1`).
Flat workspace imports (`core.*`, `lab.*`, …); see docs/architecture.md.
"""

__version__ = "0.2.0"
