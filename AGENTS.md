# AGENTS.md — UniRobot (G1 humanoid, MuJoCo)

## Scope: what to edit / not edit
- Our code: `g1_app/` only (CLI `g1_app/cli.py`, lib `g1_app/core/`, apps `g1_app/apps/`, training `g1_app/lab/` + `g1_app/training/getup/`).
- Read-only vendored submodules, never edit in place: `unitree_mujoco/`, `unitree_rl_mjlab/`, `unitree_sdk2_python/`. Pins + update procedure in `THIRDPARTY_PINS.md` (update = checkout SHA, run `g1 verify` + `g1 stand --headless --seconds 5`, update pins file).
- Generated / gitignored, never commit: `.venv/`, `outputs/`, `g1_app/outputs/`, `g1_app/videos/`, `unitree_rl_mjlab/logs/`, `thirdparty/`, `thirdparty_src/`, `__pycache__/`, `MUJOCO_LOG.TXT`, `*.tfevents*`.
- LFS-tracked: `*.onnx`, `*.STL/.stl`, `*.mp4`, `*.png`. After clone run `git lfs pull`. If `g1_app/models/g1_policy.onnx` is <10KB it is an LFS pointer, not a model — `g1 verify` and `G1StandPolicy` fail on this by design.

## Run commands (venv-aware)
- Prefer from workspace root: `just <cmd>` (uses `.venv/bin/python` if present, else `python3`), or `.venv/bin/pip install -e ./g1_app` once then `g1 <cmd>`. Fallback: `python3 -m g1_app.cli <cmd>`.
- `just stand --seconds 30` / `g1 stand --headless --seconds 12` (headless for CI/SSH), `just gui`, `just terrains`, `just verify`, `just test`, `just lint`.
- `train` args forward after `--`: `g1 train -- --num-envs 1024 --max-iterations 2000` (same with `just train -- --num-envs …`). Defaults: 2048 envs, 5001 iters; use `--num-envs 512` on small GPUs.
- Verify order for changes: `pytest g1_app/tests -q` + `ruff check g1_app` (quick) → `g1 verify` (model sha256 + deploy.yaml + math + terrains) → `g1 stand --headless --seconds 5` for sim/bridge changes. Note root `just lint` appends `|| true`, so check ruff output manually.
- Tests: `pytest g1_app/tests -q` (config: `pyproject.toml` `testpaths = ["tests", "g1_app/tests"]`). Only tests are math convention, deploy.yaml 29-DoF shapes, terrain files — no sim/policy tests in pytest; sim coverage is `g1 verify` + headless stand.
- Extras: `[dev]` = pytest+ruff; `[train]` = `mjlab==1.2.0` + `mujoco-warp==3.5.0` (GPU training only). Runtime deps: mujoco, onnxruntime, numpy, imageio(-ffmpeg), tensorboard. Python >=3.10.

## Architecture gotchas
- `models/deploy.yaml` is the authority for gains/pose/scales/rates (29-DoF, `step_dt 0.02`). `core/config.py` parses it with regex (no PyYAML dependency) and validates every list is length 29; `FALLBACK_LOCAL_CFG` is warning-path only.
- Control rates: PD torque **every sim step** (200 Hz, `sim_dt 0.005`) in `bridge.py:apply_pd`; ONNX inference every 4th step (50 Hz, decimation). Do not move PD into the policy tick.
- Math convention (single copy `core/math.py`): MuJoCo quat order `(w,x,y,z)`, identity → projected gravity `[0,0,-1]`.
- Standstill: zero user command anchors position via `walk → hold → frozen` state machine. Always use `set_command()` so logic sees intent; `--no-standstill` restores raw march-in-place. GUI/CLI status must use `telemetry()`, not `_base_state()`.
- Walk/stand switch (`core/bridge.py`): `StandStillPolicy` runs the trained 94-dim `g1_stand` policy (metadata-driven gains/joint order, joints mapped by name); `WalkStandBridge` holds both policies, `--mode walk|stand|auto` (auto brakes via walk to <0.12 m/s root speed, then hands a near-standing state to the balance policy — never mid-stride), cross-fades targets over ~0.4 s. `--stand-policy` overrides the snapshot path, default is `find_latest_stand_policy()` (curated `models/g1_stand_policy.onnx` first, logs snapshot fallback).
- Staged get-up (`g1 recover`): `GetUpBridge` sequences Reposition→SitUp→Rise 94-dim policies via `core/getup_stages.py::StageSwitcher` (pure stdlib, shared with training through `training/getup/curriculum.py`). A-gate mirrors the `supine_success` reward exactly (h<0.35, tilt>0.55, RMS supine err<0.25); supine target is canonical in `core/getup_stages.py`. Standing shortcut jumps A/B→DONE when the Rise handoff gate already holds. Train with `g1 train -- --task getup --stage A|B|C`; dashboard metric per stage is `STAGE_METRICS` in `lab/train.py` (must name a real reward).
- Import style is intentional: `cli.py` / `lab/*` bootstrap `sys.path` (workspace root + app dir) and `lab/train.py` `chdir`s to `unitree_rl_mjlab/` so `logs/` land there. Both `g1_app.core.*` and `core.*` imports exist; ruff `E402` per-file-ignores cover this — do not "fix" import order.
- Terrains: `core/terrains.py` `TERRAINS` dict + `resolve_scene()`; explicit `--scene` overrides `--terrain`. All scenes keep a flat 2×2 m start pad, features at x ≥ 1.2 m. Regen: `g1 terrains` (needs numpy + imageio).
- Training tasks live in `g1_app/training/<getup|stand>/` and register via import side effect (`lab/train.py` imports both; `--task getup|stand` selects, default `getup`). Checkpoints + `policy.onnx` every 100 iters → `unitree_rl_mjlab/logs/rsl_rl/g1_getup/`. Dashboard (friendly, :6006) auto-starts with train; raw TensorBoard on :6007 via `--tensorboard`. `g1 record` replays the latest snapshot headlessly on CPU (sets `MUJOCO_GL=egl` itself, safe during training); videos → `g1_app/outputs/videos/` (legacy `videos/` still read via `resolve_videos_dir()`).

## Display / stability notes
- `stand` (viewer), `gui`, `check` need a display; always add `--headless` on servers/CI/SSH. `g1 check` diagnoses display vs. app issues. `libdecor`/Wayland warnings and headless `OpenGL 0x502` are benign.
- Command ranges outside vx∈[−0.5,1.0], vy∈[−0.5,0.5], wz∈[−1,1] are untested — falls expected, press Reset.
