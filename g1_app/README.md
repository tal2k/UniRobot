# G1 App — Balanced Stand + Interactive Control (MuJoCo + ONNX Policy)

A standalone application that makes a simulated Unitree G1 humanoid stand
balanced and walk around in MuJoCo, driven by a pretrained neural-network
controller. It ships everything it needs in `models/` and depends on nothing
outside this folder except a Python environment with MuJoCo.

## Quickstart (Unified CLI)

```bash
# From workspace root (after git clone + git lfs pull + pip install -e ./g1_app)
g1 check                 # 10 s GUI self-test (needs display)
g1 stand --seconds 30    # viewer demo
g1 gui                   # 3D view + control panel
g1 stand --headless --seconds 12   # servers / CI
g1 stand --terrain slope # pick terrain: flat|rough|slope|steps|obstacles
g1 train                 # get-up training + dashboard on :6006
g1 dashboard             # dashboard only
g1 record --episodes 3   # headless CPU video of latest policy
g1 terrains              # regenerate test scenes
g1 verify                # model sha256 + deploy.yaml + math + terrains
```

Requirements: `mujoco>=3.5`, `onnxruntime>=1.20`, `numpy>=1.26`, `tkinter`.
Viewer/GUI need a display; headless commands do not.

---

## 1. Where the Model Came From

The brain is `models/g1_policy.onnx` (~860 KB, LFS-tracked). It is the **official
pretrained G1 velocity-tracking policy** exported from Unitree's RL training
repo (`unitree_rl_mjlab`), copied verbatim (sha256
`2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`) from:

```
unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx
```

Training/deployment parameters travel with it in `models/deploy.yaml` (authority):
PD stiffness/damping per joint, nominal standing pose, per-joint action scales,
50 Hz control rate (`step_dt: 0.02`), gait period (0.6 s), observation scales.
The app reads constants from this YAML so policy and parameters can never disagree.

The robot body (`models/g1/`) is Unitree's G1 MuJoCo description from
`unitree_mujoco/unitree_robots/g1/`: `scene_29dof.xml` (flat floor + `g1_29dof.xml`
with 29 torque-controlled motors) plus `meshes/` (60 STL files, LFS-tracked).

---

## 2. How It Works

Classic learned-locomotion stack: ONNX policy proposes joint targets at 50 Hz,
PD spring-dampers realize them every sim step (200 Hz).

```
every sim step (200 Hz):   read joints → PD torque → write motor commands
every 4th sim step (50 Hz): read IMU+joints → build 98-number observation
                             → ONNX inference → 29 target joint angles
```

**Observation (98).** Body angular velocity (3) + projected gravity (3) +
velocity command (3) + gait clock sin/cos (2) + joint angles relative to nominal
pose (29) + joint velocities (29) + previous action (29). Scales from `deploy.yaml`.

**Action (29).** Target angle = nominal pose + correction × per-joint scale.
Motor torque = `kp × (target − angle) − kd × velocity`, clamped to actuator limits.
Legs, waist, arms, wrists — all driven by the network, nothing hand-scripted.

**Standstill (default).** Zero command → state machine:
- `walk` — you command motion; passes through untouched.
- `hold` — released while moving: anchor = stop spot, gentle corrective
  command (≤ 0.3 m/s) steps back. Drift ≈ 2 cm.
- `frozen` — smoothed body speed < 0.08 m/s for 1 s: gait clock stops,
  network holds static balanced pose. Drift ≈ 1 cm.
  Push (> 0.30 m/s) → back to `hold`.

Use `set_command()` so logic sees intent. `--no-standstill` restores raw march-in-place.

Two fixed bugs worth knowing: gravity sign was inverted; PD ran at 50 Hz instead
of every sim step. Both fixed in `core/bridge.py`.

---

## 3. Using the Control Panel (`g1 gui`)

Opens 3D viewer + **G1 Control** panel:

- **Sliders** — Forward (−0.5…1.0 m/s), Sideways (−0.5…0.5 m/s),
  Turn (−1.0…1.0 rad/s). Exact training ranges; stay inside them.
- **Buttons** — Stand / Walk / Turn left / Stop (presets), Reset (re-drop),
  Quit (close all).
- **Keyboard** — arrows move, `A`/`D` turn, `Space` = stand.
- **Checkbox** — "Stand still (lock position…)" toggles anchor logic.
- **Terrain dropdown + Load** — switch flat/rough/slope/steps/obstacles
  without restart. Robot respawns at start pad; sliders preserved.
- **Status line** — sim time, torso height (~0.78 m), tilt (~0 upright),
  command vector, state: `WALKING`, `STANDING (settling…)`,
  `STANDING STILL`, or `FELL` (press Reset).

---

## 4. Test Terrains

Five scenes in `models/g1/` (regenerate: `g1 terrains`, needs `numpy` +
`imageio`). Every scene keeps a flat 2×2 m start pad at origin; features at
x ≥ 1.2 m.

| Terrain | File | Verified (headless) |
|---|---|---|
| `flat` | `scene_29dof.xml` | 0.5 m/s → 5.7 m, no fall |
| `rough` | `scene_rough.xml` + `meshes/rough.png` | 0.5 m/s → 8.7 m in 20 s |
| `slope` | `scene_slope.xml` | 0.4 m/s → climbs 7° ramp to platform |
| `steps` | `scene_steps.xml` | 0.25 m/s → 5×6 cm steps up, over, down |
| `obstacles` | `scene_obstacles.xml` | 0.25 m/s → clears bars, corridor, pillar |

Policy trained for walking, not parkour — drive slowly, steer via GUI.

---

## 5. Verification (All Headless, Passing)

| Test | Result |
|---|---|
| Stand 15–30 s from rest | No fall, height 0.778–0.785 m, tilt < 0.04, drift ~1 cm |
| Walk 5 s → release → 12–18 s stand | Settles to static pose, drift from stop ~2 cm |
| Walk + turn commands | Upright (height 0.776 m over 5 s) |
| Identity | Model sha256 matches training export exactly |
| Portability | Same result from `/tmp` — no workspace dependency |
| Get-up smoke (120 iters) | Trains, exports 94-obs `policy.onnx`, no fall errors |

Run: `g1 verify` (model sha256 + config 29-DoF + gravity convention + all 5 terrains).

---

## 6. Training Fall Recovery (Get-Up Policy)

All training code in `training/getup/` — third-party folders used, never modified.
Task `Unitree-G1-Getup`: random fallen starts (any orientation, low height,
sprawled joints), 8 s episodes, no failure termination. Stage I (discovery,
weak regularization) → Stage II (this recipe, resume via `--resume`).

**Reward recipe v2 (HUMANUP-style):**

*Correct way up:*
- `stand_height` (exp kernel → 0.78 m)
- `upright` (exp kernel → projected gravity [0,0,-1])
- `feet_force` (dense foot-ground force, target 250 N)
- `stand_on_feet` (bonus: tall + upright + both feet contact)
- `stand_pose` (exp kernel → nominal pose)
- `stand_success` (sparse: height > 0.70 + tilt < 0.3)
- `pelvis_rising` (exp kernel → pelvis height)
- `com_vel_z` (exp kernel → positive vertical CoM velocity)
- `symmetry` (soft bilateral action symmetry)

*Gentle:*
- `bad_support` (penalize torso/arm contact while body high)
- `no_head_contact` (penalize head/neck contact while body not flat)
- `joint_torques`, `joint_vel`, `joint_acc_l2`, `joint_pos_limits`,
  `action_rate_l2` (effort/smoothness/limit penalties)
- `self_collisions` (force-threshold penalty)

```bash
g1 train                              # defaults: 2048 envs, 5001 iters
g1 train -- --num-envs 1024 --max-iterations 2000
g1 train -- --run-name overnight --resume   # resume from Stage I
```

Dashboard: `g1 dashboard` → http://localhost:6006 (auto-updates every 3 s).
Watch **Stand-up success → 3.0**. Raw TensorBoard: `g1 train -- --tensorboard`
(port 6007). Checkpoints + `policy.onnx` every 100 iters:
`unitree_rl_mjlab/logs/rsl_rl/g1_getup/`. Ctrl+C stops cleanly.
Lower `--num-envs` (e.g. 512) on smaller GPUs.

---

## 7. Troubleshooting

- **No control panel?** `g1 check` first. If that fails → display issue (SSH
  without X). Else: ran `g1 stand` (no panel) instead of `g1 gui`; look behind
  3D window / taskbar (forced on top 3 s at startup).
- **Traceback?** App prints model+scene before windows — paste full output.
- **`libdecor` / Wayland warnings?** Harmless.
- **`OpenGL 0x502` in headless logs?** Benign.
- **Falls at extreme sliders?** Press Reset; outside
  vx∈[−0.5,1.0], vy∈[−0.5,0.5], wz∈[−1,1] stability not guaranteed.

---

## 8. Package Layout

```
g1_app/
├── cli.py                 # unified `g1` CLI
├── core/                  # shared lib (single source of truth)
│   ├── math.py            # quat_to_projected_gravity, euler_to_quat
│   ├── config.py          # paths, load_deploy_yaml, resolve_videos_dir
│   ├── terrains.py        # TERRAINS dict + resolve_scene()
│   └── bridge.py          # G1StandPolicy, reset_standing, run_stand, telemetry()
├── apps/                  # interactive
│   ├── stand.py           # viewer/headless balanced stand
│   ├── gui.py             # 3D + tkinter control panel
│   └── check.py           # tkinter self-test
├── lab/                   # training & analysis
│   ├── train.py           # get-up training + dashboard
│   ├── dashboard.py       # friendly live dashboard (:6006)
│   └── record.py          # headless CPU policy video
├── tools/terrains.py      # test scene generator (`g1 terrains`)
├── training/getup/        # fall-recovery task (env, rewards, PPO)
├── tests/                 # pytest: math, config, terrains
├── scripts/verify_model.py # `g1 verify`
├── models/                # curated assets (LFS)
│   ├── g1_policy.onnx     # pretrained 29-DoF velocity policy
│   ├── deploy.yaml        # gains, pose, scales, rates
│   └── g1/                # scenes + 60 STL meshes
├── outputs/videos/        # local artifacts (gitignored)
└── docs/                  # split documentation
    ├── architecture.md
    ├── how_it_works.md
    ├── terrains.md
    ├── training.md
    └── troubleshooting.md
```

---

## 9. Development

```bash
.venv/bin/pip install -e ./g1_app[dev]
pytest g1_app/tests -q       # 6 tests
ruff check g1_app            # lint
g1 verify                    # smoke test
```