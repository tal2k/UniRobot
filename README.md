# UniRobot workspace

Research workspace for Unitree G1 humanoid: balanced stand, interactive control,
test terrains, and fall-recovery (get-up) training in MuJoCo.

## Layout (what goes where)

| Path | What | Edit? |
|---|---|---|
| `g1_app/` | **Our code.** The only folder you edit day-to-day. Own git repo, installable package, unified `g1` CLI. | ✅ yes |
| `unitree_mujoco/` | Upstream G1 MuJoCo simulator + robot descriptions (pinned submodule-style vendor). | ❌ read-only |
| `unitree_rl_mjlab/` | Upstream MJLab RL framework + reusable task blocks (pinned vendor). | ❌ read-only, never commit `logs/` |
| `unitree_sdk2_python/` | Upstream Unitree SDK2 Python bindings (pinned vendor). | ❌ read-only |
| `thirdparty/` | Prebuilt DDS/install prefix used by SDK deploys. | ❌ generated |
| `thirdparty_src/` | Sources for the above (`cyclonedds`, `iceoryx`, `unitree_sdk2`). `build/` dirs are local only. | ❌ build-only |
| `.venv/` | Local Python env (not versioned). | ❌ |
| `outputs/` | Local run artifacts (videos, logs). Created on demand, never committed. | ❌ |

## Quickstart (from this folder)

```bash
# 1. env (once)
python3 -m venv .venv && .venv/bin/pip install -e ./g1_app

# 2. stand + control
g1 check                 # 10 s GUI self-test (needs display)
g1 stand --seconds 30    # viewer demo
g1 gui                   # 3D view + control panel
g1 stand --headless --seconds 12   # servers / CI

# 3. get-up training
g1 train                 # training + dashboard on :6006
g1 dashboard             # dashboard only
g1 record --episodes 3   # headless CPU video of latest policy
g1 terrains              # regenerate test scenes
g1 verify                # model sha256 + deploy.yaml consistency
```

Legacy entrypoints still work during migration:
`.venv/bin/python g1_app/g1_gui_control.py`, `run_g1_stand.py`, etc. —
each prints where it moved.

## Pins

Vendored upstreams are pinned; see `THIRDPARTY_PINS.md`.
To update one: note old+new SHA, run its upstream tests, update the pin file.

## Artifacts

- Training logs/checkpoints: `unitree_rl_mjlab/logs/` (local) → prefer `outputs/rl_logs/` symlink; never commit.
- Videos: `g1_app/outputs/videos/` (new) with fallback to legacy `g1_app/videos/`.
- `MUJOCO_LOG.TXT`, `__pycache__/`, `*.mp4/*.json` under videos: git-ignored.
