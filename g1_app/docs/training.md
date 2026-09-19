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
