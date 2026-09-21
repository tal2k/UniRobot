# Architecture

Single entry point: `g1` (`python -m g1_app.cli`). Flat workspace imports
(`core.*`, `lab.*`, `training.*`); `cli.py` + `lab/train.py` own the only
`sys.path` bootstraps. No other invocation is supported.

```
g1_app/
├── cli.py            # `g1` commands: stand|gui|check|recover|train|train-stand|
│                     #   dashboard|record|terrains|export|verify
├── core/             # shared lib (no training imports here, ever)
│   ├── math.py       # quat/gravity convention (single copy)
│   ├── config.py     # paths, deploy.yaml parsing, videos/outputs resolution
│   ├── terrains.py   # TERRAINS dict + resolve_scene()
│   ├── getup_stages.py  # StageSwitcher + gates (pure stdlib, v2: ROLL→GETUP→DONE)
│   └── bridge.py     # policies + runners:
│                     #   G1StandPolicy (walk), StandStillPolicy (94-dim balance/getup),
│                     #   WalkStandBridge (walk↔stand), GetUpBridge (roll↔getup),
│                     #   run_stand / run_recover, reset_standing / reset_fallen
├── apps/             # gui.py (viewer + panel, walk|stand|auto|recover), check.py
├── lab/              # train.py, dashboard.py, record.py (CPU headless video),
│                     #   export.py (.pt → .onnx + metadata + parity check)
├── tools/terrains.py # scene generator (`g1 terrains`)
├── training/         # MJLab task definitions (register on import)
│   ├── getup/        #   Roll (any fall → supine) + StandUp (lying → stand)
│   └── stand/        #   stand-still balance + push curriculum
├── models/           # curated policies + deploy.yaml + g1 scenes/meshes
├── tests/            # pytest (conftest.py owns the single path bootstrap)
├── scripts/verify_model.py  # `g1 verify`
└── outputs/videos/   # local artifacts (git-ignored)
```

## Data flow

Training: `g1 train` → `lab/train.py` (chdir to `unitree_rl_mjlab/`) →
task registry → MJLab PPO → `logs/rsl_rl/<experiment>/` (checkpoints +
`policy.onnx` every 100 iters) + dashboard :6006.
Deployment: ONNX → bridge policy (50 Hz inference, 200 Hz PD, 0.4 s
cross-fade on switches) → `g1 stand|gui|recover`, `g1 record` for video.
`deploy.yaml` is the authority for gains/pose; exports embed it as ONNX
metadata so policies are self-describing.

## Owned vs. vendored

Ours (`g1_app/`): everything above. Upstream is **used, never edited**:
`unitree_mujoco` (sim + robot XML), `unitree_rl_mjlab` + pip `mjlab`
(framework, PPO runner, task blocks), `unitree_sdk2_python` (real-robot SDK).
Pins: workspace `THIRDPARTY_PINS.md`.

## Where to add X

- New sim command → `cli.py` + `core/bridge.py` runner, status via `telemetry()`.
- New recovery phase → `core/getup_stages.py` gate + `training/getup/<name>_env_cfg.py`
  + registration in `training/getup/config/g1/` + `STAGE_TASK_IDS` in `lab/train.py`.
- New reward/reset → `training/getup/mdp/rewards.py|events.py` + unit test in
  `tests/test_getup_mdp.py` (CPU fakes — never debug training code on GPU time).
- New dashboard chart → `CHART_GROUPS` in `lab/dashboard.py` (page is generated).
