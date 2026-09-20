# Fall-recovery (get-up) training

Task `Unitree-G1-Getup` (`training/getup/`): random fallen starts, 8 s
episodes, no failure termination. Stage I (discovery) → Stage II
(refinement, current recipe: feet force + both-feet bonus + no-head +
gentleness penalties, resume via `g1 train -- --run-name <stage1> --resume`).

```bash
g1 train                                # dashboard :6006 + training
g1 train -- --num-envs 1024 --max-iterations 2000
g1 dashboard --port 6006                # dashboard only
g1 record --episodes 3                  # CPU headless video (safe during train)
tensorboard: g1 train -- --tensorboard  # raw TB on :6007
```

Watch **Stand-up success → 3.0** on the dashboard. Checkpoints +
`policy.onnx` every 100 iters: `unitree_rl_mjlab/logs/rsl_rl/g1_getup/`.
Videos: `g1_app/outputs/videos/` (legacy `videos/` still read).
Lower `--num-envs` (e.g. 512) on small GPUs. Ctrl+C stops cleanly.

# Stand-still (in-place balance) training

Task `Unitree-G1-Stand` (`training/stand/`): every episode starts from the
nominal standing pose (upright, root ~0.78 m, joints at default + small
noise — the "walking zero" position). Shoves kick the root on a two-phase
curriculum (`PUSH_PHASE` in `stand_env_cfg.py`); stepping to catch balance
is allowed, falling (>70° tilt) ends the episode (10 s timeout otherwise).
Rewards reuse the get-up stand terms (tighter height/upright kernels,
`stand_success` × 5.0) plus `stand_still` (quiet root velocity).

Stage II refinement (fixes sway + spread legs — same idea as get-up
Stage II: discover first, regularize after):

- **Sway:** `entropy_coef` 0.01 → 0.002 and LR 1e-3 → 3e-4 (stop paying the
  policy to dither), `action_rate_l2` −0.05 → −0.15 plus new
  `action_acc_l2` −0.05 (punish direction changes), new `ang_vel_damp`
  −0.05 on torso pitch/roll rate, `joint_vel` −3e-4 → −1e-3.
- **Spread legs:** the old single `stand_pose` (1.0, loose, all 29 joints)
  never stood a chance against `stand_success` × 5 — wide is stable, so the
  policy farmed it. Now split: `leg_pose` 2.5 with tight std 0.35 on
  hips/knees/ankles, `upper_pose` 1.0 loose on waist/arms (arms must stay
  free to windmill during recovery), plus `feet_width` −1.0 penalizing
  ankle distance beyond 0.20 m — but only while tall (>0.70 m), so
  crouched recovery steps are free.
- **Curriculum:** Phase 1 = gentle shoves (±0.5 m/s, every 4–6 s) to learn
  tall narrow standing; Phase 2 = full shoves (±1.0 m/s, every 2–4 s).
  Flip `PUSH_PHASE` to 2 once videos show quiet standing, then resume —
  never restart from scratch.

```bash
g1 train-stand                         # or: g1 train -- --task stand
g1 train-stand -- --num-envs 1024 --max-iterations 2000
g1 train-stand -- --resume             # continue Stage II from latest ckpt
g1 record -- --experiment g1_stand --stand --episodes 3
```

Checkpoints + `policy.onnx` every 100 iters:
`unitree_rl_mjlab/logs/rsl_rl/g1_stand/`. The exported policy has the same
94-dim actor observation layout as get-up, so `record` replays it directly
(`--stand` starts episodes standing instead of fallen). Walk/stand switching
in `bridge.py`/`gui.py` lands once this policy is trained.
