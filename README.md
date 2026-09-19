# UniRobot — Unitree G1 Humanoid Research Workspace

Balanced stand, interactive control, test terrains, and fall-recovery (get-up) training in MuJoCo.

## Quickstart

```bash
# 1. Clone with submodules + LFS
git clone --recurse-submodules <url>
cd UniRobot
git lfs pull

# 2. Python env (once)
python3 -m venv .venv && .venv/bin/pip install -e ./g1_app

# 3. Run
g1 check                 # 10 s GUI self-test (needs display)
g1 stand --seconds 30    # viewer demo
g1 gui                   # 3D view + control panel
g1 stand --headless --seconds 12   # servers / CI
g1 train                 # get-up training + dashboard on :6006
g1 dashboard             # dashboard only
g1 record --episodes 3   # headless CPU video of latest policy
g1 terrains              # regenerate test scenes
g1 verify                # model sha256 + config + math checks
```

Or use `just` shortcuts from workspace root:
```bash
just stand --seconds 30
just gui
just train -- --num-envs 1024
just verify
```

## Repository Layout

| Path | What | Edit? |
|---|---|---|
| `g1_app/` | **Our code** — installable package, unified `g1` CLI, tests, docs | ✅ yes |
| `unitree_mujoco/` | Upstream MuJoCo sim + G1 robot XML (git submodule, pinned) | ❌ read-only |
| `unitree_rl_mjlab/` | Upstream MJLab RL framework + task blocks (git submodule, pinned) | ❌ read-only |
| `unitree_sdk2_python/` | Upstream SDK2 Python bindings (git submodule, pinned) | ❌ read-only |
| `thirdparty/` | Prebuilt DDS install prefix (gitignored, generated) | ❌ generated |
| `thirdparty_src/` | CMake sources for thirdparty (gitignored, build trees ~500MB) | ❌ build-only |
| `.venv/` | Local Python env (gitignored) | ❌ |
| `outputs/` | Local run artifacts — videos, logs (gitignored) | ❌ |

## g1_app Package Structure

```
g1_app/
├── cli.py                 # unified `g1` CLI entrypoint
├── core/                  # shared library (single source of truth)
│   ├── math.py            # quat_to_projected_gravity, euler_to_quat
│   ├── config.py          # paths, load_deploy_yaml (29-DoF), resolve_videos_dir
│   ├── terrains.py        # TERRAINS dict + resolve_scene()
│   └── bridge.py          # G1StandPolicy, reset_standing, run_stand, telemetry()
├── apps/                  # interactive entrypoints
│   ├── stand.py           # viewer/headless balanced stand
│   ├── gui.py             # 3D + tkinter control panel
│   └── check.py           # tkinter env self-test
├── lab/                   # training & analysis
│   ├── train.py           # one-command get-up training + dashboard
│   ├── dashboard.py       # friendly live dashboard (:6006)
│   └── record.py          # headless CPU policy video
├── tools/terrains.py      # test scene generator
├── training/getup/        # fall-recovery task (env, rewards, PPO config)
├── tests/                 # pytest: math convention, config shapes, terrain files
├── scripts/verify_model.py # `g1 verify` - model + config + math checks
├── models/                # curated assets (LFS-tracked)
│   ├── g1_policy.onnx     # pretrained 29-DoF velocity policy
│   ├── deploy.yaml        # gains, pose, scales, rates (authority)
│   └── g1/                # MuJoCo scenes + 60 STL meshes
├── outputs/videos/        # local artifacts (gitignored)
└── docs/                  # split documentation
    ├── architecture.md
    ├── how_it_works.md
    ├── terrains.md
    ├── training.md
    └── troubleshooting.md
```

## Vendored Upstreams (Git Submodules)

Pinned commits recorded in `THIRDPARTY_PINS.md`:

| Repo | SHA | Purpose |
|---|---|---|
| `unitree_mujoco` | `673e44a` | G1 MuJoCo sim + robot descriptions |
| `unitree_rl_mjlab` | `1425b15` | MJLab RL framework, PPO runner, task blocks |
| `unitree_sdk2_python` | `65691c8` | Unitree SDK2 Python bindings (real robot) |

Update procedure:
```bash
cd unitree_rl_mjlab && git fetch && git checkout <new-sha>
cd .. && git add unitree_rl_mjlab && git commit -m "update unitree_rl_mjlab to <sha>"
# update THIRDPARTY_PINS.md
```

## Large Files (Git LFS)

Tracked via `.gitattributes`:
- `g1_app/models/g1_policy.onnx` (~858 KB)
- `g1_app/models/g1/meshes/*.STL` (~33 MB total)
- `*.mp4`, `*.png` (terrain heightfield, recordings)

`git lfs pull` fetches them after clone.

## Ignored (never committed)

`.venv/`, `outputs/`, `unitree_rl_mjlab/logs/`, `thirdparty/`, `thirdparty_src/*/build/`, `__pycache__/`, `MUJOCO_LOG.TXT`, `*.tfevents*`.

## Documentation

- `g1_app/docs/architecture.md` — package map, owned vs. vendored boundary
- `g1_app/docs/how_it_works.md` — policy bridge, observation/action, standstill states
- `g1_app/docs/terrains.md` — 5 test scenes + regeneration
- `g1_app/docs/training.md` — get-up task, reward recipe, dashboard, videos
- `g1_app/docs/troubleshooting.md` — GUI/display, falls, common errors

## Testing & Quality

```bash
.venv/bin/pip install -e ./g1_app[dev]
pytest g1_app/tests -q       # 6 tests: math, config, terrains
ruff check g1_app            # lint
g1 verify                    # model sha256 + deploy.yaml + math + terrains
```

## License

Apache-2.0 (inherited from Unitree upstreams). See individual submodule LICENSE files.