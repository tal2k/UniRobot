# UniRobot

**Teaching a Unitree G1 humanoid to stand its ground, get back up, and feel at home.**

![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![MuJoCo](https://img.shields.io/badge/sim-MuJoCo-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

UniRobot is a simulation platform for training balance and locomotion policies for the Unitree G1 humanoid.

## Highlights

- **Two-policy locomotion with graceful handover.** A velocity walking policy for
  getting around, a dedicated stand-still balance policy for staying put, and an
  `auto` mode that brakes to a near-stop *before* handing over — never mid-stride —
  with a 0.4 s cross-fade so torques never jump.
- **Fall recovery, trained from scratch.** A full get-up pipeline on MJLab/RSL-RL
  (PPO, 2048 parallel envs) with a live web dashboard, plus a shove-aware
  stand-still task with a push curriculum for the refinement round.
- **A digital-twin apartment.** An 18×11 m, 11-zone apartment — living room,
  kitchen, bedroom, bathroom with grab bars, walker, rugs, doors, clutter — alongside
  5 outdoor test tracks, all regenerable with one command.
- **One-command UX.** A unified `g1` CLI (`stand`, `gui`, `train`, `record`,
  `terrains`, `verify`) with `just` shortcuts; headless mode for servers and CI.
- **Train anywhere.** Local GPU runs or free Colab T4 sessions with Drive-backed
  checkpoints, resume support, and TensorBoard — same commands, same configs.
- **Reproducible by construction.** Pinned upstream submodules, LFS-versioned models,
  `deploy.yaml` as the single authority for gains and poses, and `g1 verify` to prove
  a checkout is good.

## See it in action

```bash
g1 stand --seconds 30          # balanced stand in the viewer
g1 gui --mode auto             # drive it with WASD: walk, release, watch it settle
g1 stand --terrain apartment   # stand inside the elderly apartment
g1 record --policy g1_app/models/g1_stand_policy.onnx --stand  # headless policy video
```

No display (SSH/CI)? Append `--headless` to any sim command.

## Quickstart

```bash
# 1. Clone with submodules + large files
git clone --recurse-submodules https://github.com/tal2k/UniRobot.git
cd UniRobot
git lfs pull

# 2. Python env (once)
python3 -m venv .venv && .venv/bin/pip install -e ./g1_app

# 3. Prove it works
g1 verify                      # model checksums + config + math + terrains
g1 check                       # 10 s GUI self-test (needs display)
g1 stand --headless --seconds 12
```

`just` shortcuts work from the workspace root (`just stand --seconds 30`,
`just train -- --num-envs 1024`, `just verify`).

## Under the hood

| Piece | What it is |
|---|---|
| `g1_app/core/bridge.py` | 50 Hz ONNX inference over 200 Hz PD torque, walk/stand switching, telemetry |
| `g1_app/training/` | `getup` (fall recovery) and `stand` (balance) MJLab task definitions |
| `g1_app/models/` | Curated policies (`g1_policy`, `g1_stand_policy`), `deploy.yaml`, G1 scenes |
| `g1_app/lab/` | Training launcher, live dashboard (:6006), headless video recorder, `.pt`→`.onnx` exporter |
| `unitree_*` | Upstream sim / RL / SDK submodules — pinned, read-only (see `THIRDPARTY_PINS.md`) |

The full story lives in `g1_app/docs/`: `how_it_works.md` (control loop, observations,
standstill logic), `training.md` (reward recipes, push curriculum, Colab guide),
`terrains.md` (all six scenes), `architecture.md`, `troubleshooting.md`.

## Train your own

```bash
g1 train-stand -- --num-envs 512 --max-iterations 2000   # balance (512 envs fits small GPUs)
g1 train -- --num-envs 1024 --max-iterations 2000        # get-up
g1 export --ckpt <run>/model_iter1000.pt --out my_policy.onnx
```

Checkpoints land in `unitree_rl_mjlab/logs/` every 100 iterations with TensorBoard
events; the friendly dashboard auto-starts on :6006. On Colab, open
`colab_train.ipynb` — it handles drivers, patches, and Drive-backed logging.

## Where this is going

- [x] Balanced standing + walk/stand auto-switching
- [x] Get-up training pipeline with dashboard and video replay
- [x] Indoor apartment world for assistive scenarios
- [ ] Shove-proof refinement (push curriculum, gentler regularization) — spec'd in `docs/training.md`
- [ ] Everyday apartment tasks: doorway passing, cluttered-floor robustness, gentle contact
- [ ] Sim-to-real via the SDK2 bindings when the policies earn it

## Repository layout

| Path | What | Edit? |
|---|---|---|
| `g1_app/` | **Our code** — CLI, library, training tasks, tests, docs | ✅ yes |
| `unitree_mujoco/` | Upstream G1 MuJoCo sim + robot XML (pinned submodule) | ❌ read-only |
| `unitree_rl_mjlab/` | Upstream MJLab RL framework (pinned submodule) | ❌ read-only |
| `unitree_sdk2_python/` | Upstream real-robot SDK bindings (pinned submodule) | ❌ read-only |
| `colab_train.ipynb` | Colab training notebook (GPU + Drive) | ✅ yes |
| `.venv/`, `outputs/`, `thirdparty*/` | Local env and generated artifacts (gitignored) | ❌ generated |

Pinned commits and the update procedure live in `THIRDPARTY_PINS.md`.
LFS-tracked: `*.onnx`, `*.STL`, `*.mp4`, `*.png` — `git lfs pull` fetches them.

## Testing & quality

```bash
.venv/bin/pip install -e ./g1_app[dev]
pytest g1_app/tests -q       # unit tests: math convention, 29-DoF shapes, terrains, bridge
ruff check g1_app            # lint
g1 verify                    # end-to-end checkout health
g1 stand --headless --seconds 5   # sim smoke test after bridge changes
```

## License

Apache-2.0, inherited from the Unitree upstreams. See the individual submodule
LICENSE files.
