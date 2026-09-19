# Third-party pins

Vendored upstreams are used read-only. `g1_app/training/` only *imports*
them; runtime adds them to `sys.path`. Do not edit them in place — fork
upstream if needed.

Recorded 2026-09-19 (`git rev-parse --short HEAD`):

| Repo | SHA | Subject |
|---|---|---|
| `unitree_mujoco` | `673e44a` | Default simulator to G1 robot, joystick off |
| `unitree_rl_mjlab` | `1425b15` | Fix the warnings during rough-terrain training. |
| `unitree_sdk2_python` | `65691c8` | Correct the order and spelling errors in the H2 joint index. |
| `g1_app` (ours) | `c678784` + uncommitted Stage-II v2 (head/no_head, pelvis_rising, com_vel_z) | Stage-II v2 getup recipe |

Full SHAs: resolve with `git -C <dir> rev-parse HEAD`.

Update procedure:
1. `git -C <dir> fetch && git -C <dir> checkout <new-sha>`
2. Run `g1 verify` + `g1 stand --headless --seconds 5`
3. Update this file (old → new SHA + reason).
