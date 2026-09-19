# G1 App — balanced stand + interactive control (MuJoCo + ONNX policy)

A standalone application that makes a simulated Unitree G1 humanoid stand
balanced and walk around in MuJoCo, driven by a pretrained neural-network
controller. It ships everything it needs in `models/` and depends on nothing
outside this folder except a Python environment with MuJoCo.

Three ways to run it:

| Script | What you get |
|---|---|
| `g1_gui_control.py` | 3D view + control panel (sliders, buttons, keyboard) — the full app |
| `run_g1_stand.py` | 3D view only, robot standing in place — the demo |
| `g1_stand_onnx.py --headless` | No windows, terminal log only — for testing servers |
| `check_gui.py` | 10 s window test — use when the control panel won't appear |

All commands run from the workspace root (`UniRobot/`) with its `.venv`:

```bash
.venv/bin/python g1_app/g1_gui_control.py
.venv/bin/python g1_app/run_g1_stand.py --seconds 30
.venv/bin/python g1_app/g1_stand_onnx.py --headless --seconds 12
.venv/bin/python g1_app/check_gui.py
```

Requirements: `mujoco`, `onnxruntime`, `numpy`, `tkinter`. The viewer/GUI
scripts need a display; the headless script does not.

---

## 1. Where the model came from

The brain of the robot is `models/g1_policy.onnx` (~860 KB). It was **not**
trained here — it is the official pretrained G1 velocity-tracking policy
exported from Unitree's RL training repo (`unitree_rl_mjlab`), copied
verbatim (sha256 `2a66ca63…09d3fc28`) from:

```
unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx
```

Its training/deployment parameters travel with it in `models/deploy.yaml`:
PD stiffness/damping per joint, the nominal standing pose, the per-joint
action scales, the 50 Hz control rate (`step_dt: 0.02`), the gait period
(0.6 s) and the observation scales. The app reads its constants from that
same configuration, so policy and parameters can never disagree.

The robot body (`models/g1/`) is Unitree's G1 MuJoCo description, copied
from `unitree_mujoco/unitree_robots/g1/`: `scene_29dof.xml` (flat floor +
`g1_29dof.xml` with 29 torque-controlled motors) plus `meshes/`. Scores of
megabytes, but it is what you see in the viewer.

A note on the [ioai-tech `onnx_policy` zoo](https://github.com/ioai-tech/onnx_policy):
that repo advertises ready-made G1 policies, including a dedicated
`controller_policies/stand_loop_policy` and `walking_policy`. We investigated
all three G1 binaries — every Git-LFS object returns **404 (not on the
server)**, and both controller files are absolute-path symlinks to
`/opt/onnx_policy/...`, broken outside the author's machine. So nobody can
download them today. The code already auto-detects the zoo's 47-observation
legs-only format (with LSTM state), so the day upstream is fixed you just run
`--policy <g1_policy.onnx>` with no code change. Until then, the bundled
policy below does the job.

## 2. How it works

The app is a classic learned-locomotion stack: a neural network decides
*where the joints should go*, and simple spring-damper motors (PD
controllers) pull them there. It runs at two speeds:

```
every sim step (200 Hz):   read joints → PD torque → write motor commands
every 4th sim step (50 Hz): read IMU+joints → build 98-number observation
                             → ONNX inference → 29 target joint angles
```

**Observation (98 numbers).** Body angular velocity (3) + projected gravity
(3, i.e. "which way is down in the robot's own frame") + your velocity
command (3: forward, sideways, turn) + gait clock sin/cos (2) + joint angles
relative to the nominal pose (29) + joint velocities (29) + the network's own
previous output (29). Scales come straight from `deploy.yaml`.

**Action (29 numbers).** The network outputs small corrections; target angle
= nominal pose + correction × per-joint scale. Each motor then applies
`torque = kp × (target − angle) − kd × velocity`, clamped to the actuator
limits from the MuJoCo model. Legs, waist, arms and wrists are all driven by
the network — nothing is hand-scripted.

**Standing still.** A velocity policy with a zero command still marches in
place and wanders (~0.34 m in 20 s). With the default `standstill=True` the
bridge (`G1StandPolicy`) instead runs a small state machine on top:

- `walk` — you are commanding motion; the command passes through untouched.
- `hold` — you released the command while moving: the spot where you stopped
  becomes an anchor and a gentle corrective command (≤ 0.3 m/s) steps the
  robot back to it. Drift ≈ 2 cm.
- `frozen` — once the smoothed body speed stays below 0.08 m/s for a second,
  the gait clock stops and the network holds a static balanced pose.
  Drift ≈ 1 cm. A real push (speed > 0.30 m/s) drops back to `hold`
  automatically.

Always change the command through `set_command()` so this logic sees your
intent. `--no-standstill` (or the GUI checkbox) restores raw march-in-place.

Two bugs from earlier attempts are fixed in here and worth knowing about:
the body's "down" vector was computed upside-down, and motor torques were
recomputed only at the slow 50 Hz instead of every simulation step.

## 3. Using the control panel

`g1_gui_control.py` opens the 3D viewer plus a **G1 Control** panel:

- **Sliders** — Forward (−0.5…1.0 m/s), Sideways (−0.5…0.5 m/s),
  Turn (−1.0…1.0 rad/s). These are the exact ranges the policy was trained on;
  stay inside them.
- **Buttons** — Stand / Walk / Turn left / Stop set preset commands, Reset
  re-drops the robot in its nominal pose, Quit closes everything.
- **Keyboard** — arrows move, `A`/`D` turn, `Space` stands still.
- **Checkbox** — "Stand still (lock position…)" toggles the anchor logic above.
- **Terrain dropdown + Load** — switch between flat / rough / slope / steps /
  obstacles without restarting (see section 4). The robot respawns at the
  start pad; your slider positions are kept.
- **Status line** — sim time, torso height (~0.78 m standing), tilt (near 0
  upright), your command, and state: `WALKING`, `STANDING (settling…)` or
  `STANDING STILL`. If it ever says `FELL`, press Reset.

## 4. Test terrains

Five scenes ship in `models/g1/` (regenerate with `make_terrains.py`, which
needs only `numpy` + `imageio`). Every scene keeps a flat 2×2 m start pad at
the origin; features start at x ≥ 1.2 m ahead of the robot. Pick one with the
GUI dropdown or `--terrain <name>` (`--scene <path>` still overrides):

| Terrain | What it is | Verified headless result |
|---|---|---|
| `flat` | Empty floor (`scene_29dof.xml`) | 0.5 m/s → 5.7 m, no fall |
| `rough` | Choppy bumps 2–17 cm over a 12×12 m field (`rough.png` heightfield) | 0.5 m/s → 8.7 m in 20 s, no fall |
| `slope` | 7° ramp, 3 m wide, up to a 0.33 m platform | 0.4 m/s → climbs onto platform, no fall |
| `steps` | 5× 6 cm steps up, platform, steps back down (3 m wide) | 0.25 m/s → climbs up and over, no fall |
| `obstacles` | 5–6 cm step-over bars + blocks forming a 1.4 m corridor + pillar | 0.25 m/s → clears the bars, no fall |

Two honest caveats from testing: the policy was trained for walking, not
parkour — take features **slowly** (blind full-speed charges trip on step
edges and faceplant into blocks), and steer with the GUI rather than walking
blind; drift will eventually carry an unguided robot into something. The
scenes are deliberately gentle (wide features, low risers, solid blocks with
no floating edges) so careful driving succeeds.

## 5. Verification (all headless, all passing)

| Test | Result |
|---|---|
| Stand 15–30 s from rest | No fall, height 0.778–0.785 m, tilt < 0.04, drift ~1 cm |
| Walk 5 s → release → 12–18 s stand | No fall, settles to static pose, drift from stop ~2 cm |
| Walk + turn commands | Stays upright (height 0.776 m over 5 s) |
| Identity | Bundled model sha256 matches the training export exactly |
| Portability | Same result launched from `/tmp` — no workspace dependency |
| Get-up task smoke (120 iters) | Trains, exports 94-obs `policy.onnx`, no fall-related errors |

## 6. Training fall recovery (get-up policy)

All training code lives here in `training/` — the third-party folders are
only used, never modified. The task `Unitree-G1-Getup` starts every episode
with the robot in a random fallen pose (any orientation, low height,
sprawled joints) on flat ground. No failure termination: episodes run the
full 8 s. Design details match the velocity task's PPO setup (same network,
same auto-ONNX-export runner).

Reward recipe (v2, HUMANUP-style two-stage refinement):
- *Correct way up*: height → 0.78 m, upright torso, **pushing through the
  feet** (dense foot-force signal), **standing on both feet** bonus, nominal
  pose shaping, stand-success bonus — plus a penalty for torso/arm ground
  contact while the body is high (head-bridging scores nothing) and soft
  left/right symmetry.
- *Gentle*: torque², joint-velocity², action-rate², joint-acceleration²,
  joint-limit and self-collision penalties — smooth, low-effort motion that
  won't hurt people or the robot.

Stage I (discovery, weak regularization) → Stage II (this recipe, resumed
from Stage I's final checkpoint via `--run-name <stage1> --resume`).

One command runs everything (dashboard + training), from anywhere:

```bash
.venv/bin/python g1_app/train_getup.py
.venv/bin/python g1_app/train_getup.py --num-envs 1024 --max-iterations 2000
.venv/bin/python g1_app/train_getup.py --run-name overnight --resume
```

Dashboard: http://localhost:6006 — a plain-language page served by
`dashboard.py`. It auto-updates every 3 seconds (no refresh needed) and shows:
live/idle status, iteration / max, elapsed time, training speed, ETA, and
charts with explanations — **Stand-up success** (the money chart, climbs to
3.0 when recovery is learned), shaping rewards (height → upright → pose),
and total reward. A run picker lists every training run, newest first.
The **Watch latest policy** card plays a recorded rollout of the newest
saved policy (3 episodes from random falls, with per-episode results);
press **Record fresh video** any time for a new one — recording runs
headlessly on CPU (`record_getup.py`, safe to use while training continues)
and the page picks it up automatically. Raw TensorBoard is still available
via `--tensorboard` (port 6007) for deep
dives. Checkpoints and `policy.onnx` snapshots land in
`unitree_rl_mjlab/logs/rsl_rl/g1_getup/` every 100 iterations (~1.5 h for the
default 5001 iterations with 2048 envs at ~45k steps/s on an RTX 3070).
Ctrl+C stops everything cleanly. Lower `--num-envs` (e.g. 512) on smaller
GPUs.

## 7. Troubleshooting

- **No control panel?** Run `g1_app/check_gui.py` first. If even that shows
  nothing, the problem is your desktop setup (e.g. SSH without X forwarding),
  not this app. If the check passes: make sure you launched
  `g1_gui_control.py` (not `run_g1_stand.py`, which has no panel), look
  behind the 3D window and in the taskbar — the panel forces itself on top
  for 3 seconds at startup.
- **Terminal shows a traceback?** Paste the whole output — the app prints
  which model/scene it loaded before opening windows, which pinpoints it.
- **`libdecor`/`GLFWError Wayland` warnings?** Harmless fallbacks; the
  windows still work.
- **`OpenGL error 0x502` in old simulator logs?** Benign headless noise,
  unrelated to this app.
- **Robot falls after you push sliders to extremes?** Press Reset. Commands
  outside the trained ranges are not guaranteed stable.

## 8. Repository layout

```
g1_app/
├── README.md            # this file
├── g1_stand_onnx.py     # G1StandPolicy bridge + headless/optional-viewer runner
├── g1_gui_control.py    # interactive 3D + tkinter control panel (+ terrain switch)
├── run_g1_stand.py      # viewer-only standing demo
├── check_gui.py         # tkinter environment self-test
├── make_terrains.py     # regenerates the test scenes
├── train_getup.py       # one-command get-up training + dashboard
├── dashboard.py         # friendly live training dashboard (port 6006)
├── record_getup.py      # headless CPU recorder: latest policy -> mp4 + results
│                        # (videos/ output is git-ignored)
├── training/            # our training code (third-party stays untouched)
│   └── getup/
│       ├── getup_env_cfg.py     # fall-recovery env: fallen starts, shaping rewards
│       ├── mdp/                 # base_height obs + stand_* rewards
│       └── config/g1/           # G1 env wiring, PPO config, task registration
└── models/ below
└── models/
    ├── g1_policy.onnx   # pretrained 29-DoF velocity policy (the brain)
    ├── deploy.yaml      # its gains, nominal pose, scales, rates
    └── g1/              # G1 MuJoCo description: scene + meshes (the body)
        ├── scene_29dof.xml      # flat (default)
        ├── scene_rough.xml + meshes/rough.png
        ├── scene_slope.xml
        ├── scene_steps.xml
        └── scene_obstacles.xml
```
