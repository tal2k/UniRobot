"""Training code owned by the G1 app (task definitions, launchers).

Everything here is *our* code. It builds on two third-party pieces that stay
untouched in their own repos/folders:
  * `mjlab` (pip package) — simulation + RL framework,
  * `unitree_rl_mjlab` (sibling folder, added to sys.path at runtime) —
    robot assets and reusable task building blocks.
"""
