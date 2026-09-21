# Get-up as staged multi-policy training — plan + status

Status: **v2 implemented in the working tree (uncommitted)** — ROLL→GETUP
deployment with facing gates, Roll + merged-StandUp training tasks; see
§14. The v1 A/B/C deployment switcher below is superseded (its training
cfgs stay as curriculum/warm-start sources).
Created: 2026-09-20, updated: 2026-09-21
Scope: train the G1 to recover from any fall with **three specialist policies**
that hand off in sequence — Reposition → SitUp → Rise — instead of one
monolithic get-up policy.

---

## 0. TL;DR

- Each get-up phase is its own policy (own RL run, own ONNX), switched by a
  small state machine on root height/tilt, with the same 0.4 s target
  cross-fade the walk/stand bridge already uses.
- All policies share one contract: 94-dim obs, 29 joint-position actions,
  50 Hz policy over 200 Hz PD, gains/pose from `deploy.yaml`.
- Train sequentially (`--stage A|B|C`), never co-train; evaluate between
  stages before advancing.
- Deployment adds `GetUpBridge` next to `WalkStandBridge`; walk/stand/auto
  behavior is untouched.

---

## 1. Why staged policies (evidence)

- **HumanUP** (RSS 2025): two-stage discovery→refinement for G1 get-up; notes
  the manufacturer's own G1 get-up trajectory is **3 phases** — reset (1 s),
  lying→squat (6 s), squat→stand (4 s). Learned continuous motion replaced it.
- **"From Rolling Over to Walking"** (2023): graph curriculum
  roll→kneel→crouch→stand; a bad early stage produced "gorilla-like" motion —
  stage gating and clean per-stage signals matter more than one big reward.
- **BAT** (ICRA 2026): online switching between specialist whole-body policies
  beats single monolithic policies on long-horizon humanoid tasks; validated
  on G1 (walk → dance → stand).
- **Move-Then-Operate** (2026): hard-switched phase experts introduce a
  "structural inductive bias" that reduces optimization interference — each
  expert specializes in its phase dynamics.
- **HoST** (RSS 2025): posture-adaptive standing-up from scratch on G1;
  multi-critic + smoothness regularization + implicit motion-speed bound for
  deployable hardware behavior.
- **Atlas (Boston Dynamics)**: deliberately repositions limbs before pushing
  up — explicitly to avoid stepping on its own arm — and uses the reposition
  as a sensor sanity check before committing to the rise.
- **In-repo precedent**: `core/bridge.py::WalkStandBridge` already implements
  specialist switching with braking + 0.4 s cross-fade. Extend that pattern.

---

## 2. Architecture

```
fallen (h < 0.55 or tilt > 0.6, zero user command)
   │
   ▼
Policy A: Reposition ──(h<0.35, tilt>0.55, RMS joint err to supine<0.25)──▶ Policy B: SitUp
                                              ──(h>0.55, tilt<0.7)──▶ Policy C: Rise
                                                                      ──(h>0.72, tilt<0.3, speed<0.12)──▶
                                                                      handoff to stand/walk policy
```

Stage A ends when the robot has reached the canonical supine pose SitUp
starts from (same thresholds as the `supine_success` training reward, so a
policy that earns the reward also triggers the switch). Standing shortcut:
if the robot already satisfies the Rise handoff gate while in A/B, the
switcher jumps straight to DONE — the lying A gate could never fire while
standing (see `core/getup_stages.py::StageSwitcher`).

- **Policy contract** (same as the trained stand policy):
  - obs 94 = gyro 3 + projected gravity 3 + base height 1 + (q−q0) 29 +
    qvel 29 + last action 29
  - action 29 joint-position targets; PD torque every sim step (200 Hz),
    inference every 4th step (50 Hz)
  - gains / nominal pose / action scale from `models/deploy.yaml` metadata
- **Switching**: state machine v1 (no learned gating). Thresholds in §3;
  implemented in `training/getup/curriculum.py` (pure logic, reusable by the
  bridge) and consumed by `core/bridge.py::GetUpBridge`.
- **Hysteresis** to prevent chatter: require N consecutive policy ticks
  (e.g. 10) above threshold and a small margin before advancing; revert if
  the state regresses (e.g. B→A when h < 0.30 for 10 ticks).
- **Safety fallbacks**: per-stage timeout (A 8 s, B 6 s, C 5 s) or
  height decreasing → reset sequence to A after a cooldown; GUI Reset re-arms.
- **Handoff**: when C reaches standing-equivalent state (h > 0.75,
  tilt < 0.3, root speed < 0.12 m/s) hand off to the existing stand/walk
  bridge, matching its own handover rule (`stop_speed=0.12`).
- v2 option: learned switch policy (BAT-style) or distillation of the three
  policies into one ONNX — only if the state machine proves insufficient.

---

## 3. Per-stage specification

All stages: no failure termination, timeout only; foot friction and encoder
bias / COM randomization from the existing get-up cfg; falls are the start
state, not an error.

### Stage A — Reposition

| | |
|---|---|
| Start | random sprawled: x/y ±0.3, z 0.06–0.3, roll ±π, pitch ±π/2, yaw ±π, joints ±0.6 rad |
| Goal | canonical supine neutral (flat on back, extended limbs) — the SitUp start pose |
| Episode | 12 s |
| Advance when | `h < 0.35`, `tilt > 0.55` and RMS joint error to supine `< 0.25` for N ticks (= the `supine_success` reward condition) |

| Reward term | Weight | Notes |
|---|---|---|
| `supine_success` | +3.0 | sparse gate: h<0.35, tilt>0.55, pose err<0.25 |
| `supine_pose` | +1.5 | exp kernel on MSE to supine target |
| `torso_horizontal` | +1.0 | exp kernel on projected-gravity xy |
| `symmetry` | −0.02 | action mirror (reuse `mirror_symmetry`) |
| `self_collisions` | −0.5 | force > 10 N |
| `joint_torques` | −3e-6 | weak reg (discovery stage) |
| `joint_vel` | −3e-4 | weak reg |
| `joint_pos_limits` | −5.0 | hardware safety |
| `action_rate_l2` | −0.05 | weak reg |

### Stage B — SitUp

| | |
|---|---|
| Start | horizontal pose family: canonical supine + prone (roll-mix), joint noise |
| Goal | crouch/squat: h > 0.55, tilt < 0.7, both feet touching, pelvis rising |
| Episode | 10 s |
| Advance when | `h > 0.55` and `tilt < 0.7` for N ticks |

Reuse/extend existing `training/getup/mdp/rewards.py` terms with
**stage-B targets** (the current ones assume final standing at 0.78 m — they
must be parameterized before reuse):

| Reward term | Weight | Notes |
|---|---|---|
| `pelvis_rising` | +1.5 | target 0.55 for this stage |
| `stand_height` | +1.0 | target 0.55, wide std |
| `upright` | +1.0 | |
| `feet_force` | +1.0 | target 250 N total |
| `stand_on_feet` | +2.0 | gated at min_height 0.5 |
| `bad_support` | −3.0 | min_height 0.45 |
| `no_head_contact` | −2.0 | min_height 0.40 |
| `com_vel_z` | +2.0 | rising reward |
| `symmetry` | −0.02 | |
| `self_collisions` | −0.5 | |
| `joint_torques` / `joint_vel` / `action_rate_l2` | −3e-6 / −3e-4 / −0.05 | |

### Stage C — Rise (push to stand)

| | |
|---|---|
| Start | crouch family; fallback: existing get-up fallen randomization |
| Goal | standing: h > 0.78, tilt < 0.3, both feet loaded |
| Episode | 8 s |
| Advance when | `stand_success > 0.5` aggregate, then composed chain test |

Rewards: **existing `training/getup/getup_env_cfg.py` recipe unchanged**
(`stand_height` 1.0, `upright` 1.0, `feet_force` 1.0, `stand_on_feet` 3.0,
`stand_pose` 0.5, `stand_success` 3.0, `bad_support` −3.0, `no_head_contact`
−2.0, `pelvis_rising` 1.5, `com_vel_z` 2.0, `symmetry` −0.02, gentleness
penalties). Optional push curriculum later (Phase 1 none → Phase 2 ±0.5 m/s
root shoves), following the `stand` task pattern in `docs/training.md`.

---

## 4. Training protocol

Sequential. Evaluate each stage before starting the next. Never co-train.

```bash
# Stage A (discovery: weak regularization)
g1 train -- --task getup --stage A --num-envs 2048 --max-iterations 3000

# Stage B
g1 train -- --task getup --stage B --num-envs 2048 --max-iterations 3000

# Stage C (existing refinement recipe)
g1 train -- --task getup --stage C --num-envs 2048 --max-iterations 5001

# Optional refinement pass (resume, never restart)
g1 train -- --task getup --stage A --resume --max-iterations 1000

# Small GPUs: --num-envs 512
```

- Logs: `unitree_rl_mjlab/logs/rsl_rl/g1_getup_reposition/`,
  `.../g1_getup_situp/`, `.../g1_getup_rise/` — checkpoints + `policy.onnx`
  every 100 iters; dashboard `:6006` auto-starts.
- Weight transfer A→B→C: obs/action spaces are identical, but rsl_rl resume
  is scoped to one experiment folder. v1 = independent runs; v2 = manual
  checkpoint warm-start into the next experiment (same architecture). See §9.
- Advance criteria (from dashboard metrics + `g1 record` video):
  - A: `supine_success` ≥ 0.8, self-collision rate < 5 %
  - B: crouch gate (h>0.55, tilt<0.7) ≥ 0.8, both-feet contact most steps
  - C: `stand_success` ≥ 0.8, then composed fall→stand ≥ 70 % on flat

---

## 5. Deployment / switching design

New `core/bridge.py::GetUpBridge`:

- state: `IDLE` (not fallen) | `A` | `B` | `C` | `DONE`
- `step_sim()`: pick active policy with hysteresis (§2), cross-fade joint
  targets over 0.4 s on every switch, PD every sim step, policy every 4th.
- `telemetry()`: returns `(height, tilt, user_command, state)` for GUI/CLI —
  same shape as `WalkStandBridge.telemetry()`.
- Terminal: `DONE` hands off to `WalkStandBridge` / `g1_stand_policy`; if no
  stand policy is available, hold the standing pose.
- CLI: `g1 recover` runs the staged rollout from a fall (implemented in
  `core/bridge.py::run_recover`; `walk|stand|auto` behavior untouched).
- Export: verify `g1 export` embeds ONNX metadata (`joint_names`,
  `default_joint_pos`, `joint_stiffness`, `joint_damping`, `action_scale`)
  like the existing stand policy; `StandStillPolicy`-style loading depends on
  it. Extend the exporter if metadata is missing.

---

## 6. File plan

Preferred layout — flat, single `mdp` namespace, no duplicate packages:

```
g1_app/training/getup/
  getup_env_cfg.py          # existing Stage C recipe (keep as-is)
  reposition_env_cfg.py     # NEW: Stage A factory
  situp_env_cfg.py          # NEW: Stage B factory
  curriculum.py             # NEW: switch thresholds + helpers (pure logic)
  mdp/
    rewards.py              # extend: parameterize height targets; add A terms
    observations.py         # unchanged (94-dim layout)
  config/g1/
    __init__.py             # register 3 tasks with per-stage rl_cfg
    env_cfgs.py             # unitree_g1_getup_{reposition,situp,rise}_env_cfg
    rl_cfg.py               # per-stage runner cfgs (experiment_name per stage)
g1_app/core/bridge.py       # add GetUpBridge (+ telemetry); optional --mode getup
g1_app/lab/train.py         # --stage A|B|C -> task_id/experiment mapping
g1_app/tests/test_getup_stages.py  # NEW: config shape + registration tests
```

- Task ids: `Unitree-G1-Getup-Reposition`, `Unitree-G1-Getup-SitUp`,
  `Unitree-G1-Getup-Rise` (avoids clashing with the `Unitree-G1-Stand`
  balance task).
- Rejected alternative: nested `training/getup/reposition/{config,mdp}/`
  packages (duplicate mdp trees, star-import shadowing, more import surface).

---

## 7. Known pitfalls (from the partial attempt — read before coding)

1. **mjlab namespaces.** `from mjlab.envs.mdp import dr` imports the `dr`
   submodule (events/DR helpers). Observation and penalty functions
   (`builtin_sensor`, `projected_gravity`, `joint_pos_rel`, `action_rate_l2`,
   `time_out`, `mean_action_acc`, …) live in `mjlab.envs.mdp` and are
   re-exported by `training.getup.mdp`. Follow the existing cfg pattern:
   `import training.getup.mdp as mdp` then `mdp.<fn>`. Calling
   `dr.builtin_sensor` raised
   `AttributeError: module 'mjlab.envs.mdp.dr' has no attribute 'builtin_sensor'`.
2. **Name sync.** The env cfg referenced `symmetry` while the function is
   `mirror_symmetry` — latent `NameError`. Add an import smoke test (§6).
3. **No duplicate star-imports / duplicate mdp packages** — shadowing bugs.
4. **`train.py` chdirs** to `unitree_rl_mjlab/` before importing task modules;
   `training.*` resolves via `APP_DIR` on `sys.path`. Keep imports after
   `sys.path` setup, as today.
5. **`experiment_name` lives in the per-stage `rl_cfg.py`**, not in
   `train.py` — that is what determines the log dir.
6. **ONNX metadata** is required by the deploy loader; verify before wiring
   `GetUpBridge` (see §5).
7. Keep the `self_collision` contact sensor (subtree `pelvis`) — Stage A/B
   rewards need it; foot contact sensors are needed for B/C only.
8. **MJLab default tensors are per-env** (`default_joint_pos` is
   `(num_envs, J)`, verified in `mjlab/entity/entity.py` + upstream
   `default_joint_pos[env_ids]` indexing). Never `.flatten()` them: in an
   env factory that *crashes* past 1 env; in a reward it silently
   broadcasts env 0's pose to all envs. Use row-indexed helpers
   (`rewards._default_joint_pos`, per-env rows in resets) and cover both
   shapes in `tests/test_getup_mdp.py` with non-identical per-env fakes.

---

## 8. State of the working tree

Implemented 2026-09-21 (uncommitted) per the §6 layout, with one design
change: Stage A targets the canonical supine pose (`supine_success` /
`target_pose` rewards, pose-error deployment gate) instead of the
extension-based `reposition_success` / `limb_clearance` sketch in §3 above
— the extension proxy sat only 0.04 above its threshold for a perfect
supine, while the pose gate mirrors the training reward exactly. The
switcher additionally jumps straight to DONE when the robot already
satisfies the Rise handoff gate while in A/B (no stall when a timeout
restarts a standing robot at A). The supine target lives in
`core/getup_stages.py` and is shared by training, deployment and `g1 record`.

Next: train Stage A (`--stage A`, watch `Episode_Reward/supine_success`),
evaluate on video, then B, then C; promote exports to
`models/g1_getup_{reposition,situp,rise}_policy.onnx` so `g1 recover`
resolves them.

---

## 9. Open questions (decide when resuming)

1. Warm-start between stages (A→B→C) — worth the checkpoint plumbing or
   independent training?
2. Stage B starts: from arbitrary lying poses (like existing get-up) or only
   from Stage-A end states? → Decided (2026-09-21): mixed lying family
   (supine/prone/side) — strict-supine over-constrains vs biomechanics (§13).
3. Stage C push curriculum — in scope for the first milestone or later?
4. State machine vs learned switch (BAT-style) — v1 state machine; revisit
   only if chatter/edge cases show up. → Confirmed v1 (2026-09-21): BAT shows
   learned switching needs offline supervision; heuristic first (§13).
5. Names: Reposition / SitUp / Rise — confirm (vs. Roll/Crouch/PushUp).
6. Ship the switchboard (3 ONNX) vs distill to one unified ONNX later.

---

## 10. Acceptance / verification

```bash
pytest g1_app/tests -q          # incl. new stage config-shape tests
ruff check g1_app
g1 verify                       # model sha256 + config + math + terrains
g1 stand --headless --seconds 5 # walk policy regression
g1 stand --mode auto --headless --seconds 10  # walk/stand regression
```

- Per-stage videos via `g1 record` (watch for self-collisions, head-bridging,
  violent motions).
- Composed: start fallen, end standing ≥ 70 % of runs on flat; bounded tilt;
  no failure terminations.
- Walk/stand/auto behavior unchanged (no regressions in telemetry or CLI).

---

## 11. References

- HumanUP (RSS 2025): https://humanoid-getup.github.io/ , arXiv 2502.12152 ,
  code `RunpeiDong/HumanUP`
- HoST (RSS 2025): arXiv 2502.08378 , code `OpenRobotLab/HoST`
- HiFAR multi-stage curriculum: arXiv 2502.20061
- From Rolling Over to Walking: arXiv 2303.02581
- BAT online policy switching: alphaXiv 2604.01064 (ICRA 2026)
- Move-Then-Operate: arXiv 2604.23620
- FIRM unified fall-safety: arXiv 2511.07407
- Balance-informed recovery (capture point/CoM): arXiv 2603.08619
- Gait-conditioned multi-phase curriculum: arXiv 2505.20619
- In-repo: `docs/how_it_works.md`, `docs/training.md`,
  `core/bridge.py::WalkStandBridge`, `training/getup/getup_env_cfg.py`
- FIRM (2025): unified fall mitigation + recovery, single G1 policy from
  demos + RL + diffusion memory — https://firm2025.github.io/ ,
  arXiv 2511.07407
- HiFAR (2025): multi-stage curriculum (2D sagittal → full 3D,
  relaxed → realistic limits) + key-state initialization, real robot —
  arXiv 2502.20061
- HumanoidBench (RSS 2024): flat end-to-end RL fails on high-dim humanoids,
  structured/hierarchical wins — https://humanoid-bench.github.io/ ,
  arXiv 2403.10506
- DeepMimic (2018) / AMP (2021) / PHC (2023): reference-motion priors and
  multi-skill distillation line behind demo-seeding (§13)
- NimbRo (Behnke, RoboCup): classical split prone vs supine routines; prone
  is kinematically harder (knees don't bend backward, arms trapped)
- Human floor-rise (VanSant 1988; Burton et al. 2023): paths go through
  side/prone/quadruped; direct symmetric sit-up is rare (~25% of adults)

---

## 12. Command cheatsheet

```bash
# Train (sequential)
g1 train -- --task getup --stage A --num-envs 2048 --max-iterations 3000
g1 train -- --task getup --stage B --num-envs 2048 --max-iterations 3000
g1 train -- --task getup --stage C --num-envs 2048 --max-iterations 5001

# Inspect / record
g1 dashboard
g1 record --episodes 3
g1 record -- --experiment g1_getup_reposition --episodes 3

# Verify
pytest g1_app/tests -q && ruff check g1_app && g1 verify

---

## 13. v2 roadmap — wider literature review (2026-09-21)

Beyond §11, three more papers change the evaluation: **FIRM** (unified
fall-safety, real G1), **HiFAR** (multi-stage curriculum, real robot) and
**HumanoidBench** (why flat end-to-end fails), plus the DeepMimic/AMP/PHC
imitation line and human floor-rise biomechanics.

### 13.1 Verdict per design question

| Paper | Staged training? | Multi-policy deployment? | Supine as hub? |
|---|---|---|---|
| HumanUP (real G1) | Yes — discovery→refine essential | No — ONE get-up policy; 2nd policy splits by *start family* (prone roll-over), not phase | **Yes, explicitly**: prone = roll→supine→stand |
| HoST (real G1) | Yes — stage rewards + multi-critic; single critic = 0% | No — single policy; joint supine+prone still open/hard | Supports funnel (joint training is the hard part) |
| Rolling→Walking (sim) | Yes — gated stage rewards required | No — single policy; warns linear chains go "gorilla" | Differs: path via prone/kneel (humanlike) |
| BAT (real G1) | n/a (loco-manipulation) | Switching works, but needs *learned* switch + blending | n/a |
| FIRM (real G1) | Yes + demo priors | No — unified single policy; from-scratch RL "unstable", multimodality needs funnel or diffusion | Funnel supported (multimodality argument) |
| HiFAR (real robot) | Yes — 2D→3D dimensionality + KSI | Single policy | n/a |
| HumanoidBench | Structure beats flat end-to-end | Hierarchical > monolithic | n/a |
| Biomechanics | n/a | n/a | Paths via side/prone/quadruped; direct sit-up rare — don't over-constrain starts |

Takeaway: *staged training* is consensus; *phase-switched deployment* is our
own (reasonable, BAT-adjacent) choice; *supine-as-hub* is the best-supported
canonical intermediate. The switcher/handoff is the riskiest part — validate
it explicitly (Stage B success from Stage A's *actual* outputs).

### 13.2 Target architecture (v2)

Deployment: `ROLL → GETUP → DONE → stand` (2 recovery policies, HumanUP's
split). `facing = projected_gravity[0]` (yaw-invariant: supine ≈ −1,
prone ≈ +1, side ≈ 0 — verify sign in sim) drives the roll gate; today's C
gate drives the stand handoff. Training: `Roll` run (any-sprawl → supine,
face-up gravity shaping à la HumanUP `r_roll` + `supine_success`) and one
merged `GetUp` run (lying family → stand, B+C rewards height-banded
HoST-style), each discovery→refine via `--resume`.

### 13.3 Phased plan

**Now (cheap, no pipeline retrain):**
1. Drop-and-settle reset dataset (HumanUP): bake ~10k collision-free lying
   poses once (randomize → drop 0.5 m → simulate 10 s), sample resets from
   npz. Fixes interpenetrating uniform-sprawl starts.
2. Freeze wrists (HumanUP): exclude wrist actuators in stage action cfgs,
   hold at nominal. 29→~23 action dims, zero capability lost.
3. Standing + key-state start mixture (HumanUP/HiFAR-KSI): fraction of
   episodes from standing or hand-set crouch. Reset-mixture change only.
4. Eval protocol + targets: `g1 eval-recover` matrix (starts
   supine/prone/side × seeds) reporting success, time-to-stand, peak torque,
   self-collision events, arm/leg torque share. Targets: 78.3% get-up,
   98.3% roll-over (HumanUP real-world).

**Next (v2 core):**
5. Facing metric + ROLL state + Roll env cfg (§13.2).
6. Merged GetUp run (B+C, height-banded rewards); keep A/B/C cfgs as
   curriculum/warm-start sources (same 94-dim arch).
7. Discovery→refine per policy (weak-reg run, then `--resume` strong-reg).
8. Pull-assist curriculum (HoST): decaying upward pelvis force via custom
   MJLab event — biggest exploration lever for the sit-up.
9. Demo-seeding (FIRM/DeepMimic insurance against from-scratch instability):
   one hand/mocap get-up trajectory as a tracking reward term alongside task
   rewards, not full imitation.

**Later:**
10. Fall safely (FIRM): controlled descent when a shove is unsurvivable,
    then recover. Needs impact-force sensing/rewards.
11. Distill to one ONNX (PHC-style DAGGER over composed rollouts) — only
    once all specialists work.
12. Learned switcher (BAT-style) — only if heuristic gates chatter.

### 13.4 What NOT to do

- Multi-critic runner (needs custom RSL-RL algorithm; stage-banded rewards
  get most of it).
- Joint supine+prone single-policy training before the funnel exists (HoST
  lists it unfinished; FIRM shows why it's unstable).
- Tightening supine tolerances below RMS 0.25 / noise ±0.2 (sim-convenience
  target; hardware needs the slack).

---

## 14. v2 implementation — ROLL → GETUP (shipped 2026-09-21, uncommitted)

v1 (A/B/C deployment switcher, §2–§8) is superseded and its training tasks
(`Reposition`/`SitUp`/`Rise`, legacy `Getup`) were removed from the codebase
(recoverable from git history); §2–§8 stay as design history. v2 splits by
*start family* with supine as the funnel and trains **two** runs.

### 14.1 Deployment: one recovery switch

`core/getup_stages.py::StageSwitcher` states are now
`IDLE | ROLL | GETUP | DONE`. New metric `facing` = body-x projected
gravity (yaw-invariant: supine ≈ −1, prone ≈ +1, side ≈ 0 — verify the sign
in sim if the model ever changes):

| State | Runs when | Advances when (10 ticks) |
|---|---|---|
| ROLL | fallen + not face-up (`facing > −0.5`) | supine-ish (`facing < −0.5`, h<0.35, slow) + joints near supine (pose_err<0.30, keeps the handoff inside GETUP's start distribution) → GETUP |
| GETUP | face-up fall (skips ROLL entirely) | standing gate (h>0.72, tilt<0.3, slow) → DONE |
| DONE | — | balance policy; falls re-engage ROLL/GETUP by facing |

Kept from v1: hysteresis, per-stage timeouts (ROLL 8 s, GETUP 12 s —
timeouts re-roll instead of dead-ending), GETUP→ROLL revert when supine is
lost (`facing > 0` sustained), standing shortcut to DONE, 0.4 s target
cross-fade on every switch, PD every sim step / inference every 4th.
`GetUpBridge` holds `{"roll", "getup"}` policies; `g1 recover` takes
`--policy-roll/--policy-standup` (curated `models/g1_getup_roll_policy.onnx`
/ `models/g1_getup_standup_policy.onnx` first, else newest
`g1_getup_roll` / `g1_getup_standup` snapshot).

### 14.2 Training: two runs

- **Roll** (`--stage roll`, exp `g1_getup_roll`): any-sprawl resets;
  `roll_success` (low + face-up, w 3.0) + `face_up` gravity shaping (w 1.5,
  HumanUP-style) + `supine_pose`/`torso_horizontal` + weak reg. Watch
  `Episode_Reward/roll_success` (target ≥ 0.8, cf. HumanUP 98.3%).
- **StandUp** (`--stage standup`, exp `g1_getup_standup`): lying-family
  resets; merged B+C rewards height-banded HoST-style — `stand_on_feet`
  loosened to the crouch (min_h 0.55) so sit-up earns signal,
  `stand_success` (0.70/0.3) pays the full rise; 12 s episodes. Discovery
  aids shipped after the first plateau: mixed starts (15% standing, 25%
  crouch key-state, 60% lying), `lift_assist` (upward pelvis force fading
  within each episode, HoST-style), `wrist_pose_l2` lock (wrists add dims,
  not leverage). Train discovery, then `--resume` refine. Watch
  `Episode_Reward/stand_success` (target ≥ 0.8, cf. HumanUP 78.3%).
  Retrain with these aids (2026-09-21) lifted both money metrics off zero
  by iter ~2000 where the unaided run sat at 0.0 through iter 3600.
- Legacy `--stage A|B|C` tasks stay registered as curriculum/warm-start
  sources (same 94-dim arch) but are not deployed.

### 14.3 Eval

`g1 record --start prone` replays roll-over; `--start supine` the stand-up.
Composed target (flat): fall→stand ≥ 70% across supine/prone/side starts —
to be tightened to the §13.3 protocol (`g1 eval-recover` matrix) once both
policies exist.
```