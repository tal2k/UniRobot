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
noise — the "walking zero" position). Strong shoves kick the root every
2–4 s (`push_robot`: xy ±1.0 m/s, yaw ±1.0); stepping to catch balance is
allowed, falling (>70° tilt) ends the episode (10 s timeout otherwise).
Rewards reuse the get-up stand terms (tighter height/upright kernels,
`stand_success` × 5.0) plus `stand_still` (quiet root velocity).

```bash
g1 train-stand                         # or: g1 train -- --task stand
g1 train-stand -- --num-envs 1024 --max-iterations 2000
g1 record -- --experiment g1_stand --stand --episodes 3
```

Checkpoints + `policy.onnx` every 100 iters:
`unitree_rl_mjlab/logs/rsl_rl/g1_stand/`. The exported policy has the same
94-dim actor observation layout as get-up, so `record` replays it directly
(`--stand` starts episodes standing instead of fallen). Walk/stand switching
in `bridge.py`/`gui.py` lands once this policy is trained.
