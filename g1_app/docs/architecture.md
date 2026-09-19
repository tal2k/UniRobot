# Architecture

```
g1_app/
├── cli.py            # `g1` unified CLI (stand|gui|check|train|dashboard|record|terrains|verify)
├── core/             # shared lib: math, config (deploy.yaml), terrains, bridge
│   ├── math.py       # quat_to_projected_gravity, euler_to_quat (single copy)
│   ├── config.py     # paths, load_deploy_yaml, videos/outputs resolution
│   ├── terrains.py   # TERRAINS dict + resolve_scene()
│   └── bridge.py     # G1StandPolicy, reset_standing, run_stand
├── apps/             # interactive: stand.py, gui.py (telemetry), check.py
├── lab/              # train.py, dashboard.py, record.py (CPU headless video)
├── tools/terrains.py # scene generator (`g1 terrains`)
├── training/getup/   # fall-recovery task: env cfg, mdp rewards/obs, PPO cfg
├── models/           # curated: g1_policy.onnx + deploy.yaml + g1/*.xml + meshes
├── tests/            # pytest: math convention, config shapes, terrain files
├── scripts/verify_model.py  # `g1 verify`
└── outputs/videos/   # local artifacts (git-ignored); legacy videos/ still read
```

Legacy flat scripts (`g1_stand_onnx.py`, `g1_gui_control.py`, `run_g1_stand.py`,
`check_gui.py`, `make_terrains.py`, `train_getup.py`, `dashboard.py`,
`record_getup.py`) remain as deprecation shims that re-export the new homes.

## Owned vs. vendored

Ours (`g1_app/`): everything above. Upstream is **used, never edited**:
`unitree_mujoco` (sim + robot XML), `unitree_rl_mjlab` + pip `mjlab`
(framework, PPO runner, task blocks), `unitree_sdk2_python` (real-robot SDK).
`lab/train.py` adds both to `sys.path` at runtime and `chdir`s to the RL repo
so `logs/` land in one place. Pins: workspace `THIRDPARTY_PINS.md`.
