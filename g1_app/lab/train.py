"""One-command G1 training + live dashboard. Owned by the G1 app.

Canonical home (moved from legacy train_getup.py). Prefer:

    g1 train                        # get-up policy (default)
    g1 train -- --task stand        # stand-still balance policy
    g1 train -- --task getup --stage A   # staged get-up: Reposition
    g1 train -- --task getup --stage B   # staged get-up: SitUp
    g1 train -- --task getup --stage C   # staged get-up: Rise
    g1 train-stand -- --num-envs 1024
    g1 train -- --num-envs 1024 --max-iterations 2000
"""

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(LAB_DIR)
WORKSPACE = os.path.dirname(APP_DIR)
RL_MJLAB_DIR = os.path.join(WORKSPACE, "unitree_rl_mjlab")
TASK_IDS = {
  "getup": "Unitree-G1-Getup",
  "stand": "Unitree-G1-Stand",
}
# Staged get-up: one task (and experiment folder) per phase. See
# g1_app/docs/getup_staged_policies.md.
STAGE_TASK_IDS = {
  "A": "Unitree-G1-Getup-Reposition",
  "B": "Unitree-G1-Getup-SitUp",
  "C": "Unitree-G1-Getup-Rise",
}
STAGE_METRICS = {
  "A": "supine_success",
  "B": "stand_on_feet",
  "C": "stand_success",
}


def port_in_use(port: int) -> bool:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(0.5)
    return s.connect_ex(("127.0.0.1", port)) == 0


def load_train_module():
  """Import unitree_rl_mjlab's train script without touching that repo."""
  path = os.path.join(RL_MJLAB_DIR, "scripts", "train.py")
  spec = importlib.util.spec_from_file_location("rl_train_entry", path)
  assert spec is not None and spec.loader is not None, f"cannot load {path}"
  mod = importlib.util.module_from_spec(spec)
  sys.modules["rl_train_entry"] = mod
  spec.loader.exec_module(mod)
  return mod


def main() -> int:
  ap = argparse.ArgumentParser(description="Train G1 get-up/stand policy + dashboard")
  ap.add_argument("--task", choices=sorted(TASK_IDS), default="getup",
                  help="Which policy to train (default: getup)")
  ap.add_argument("--stage", choices=sorted(STAGE_TASK_IDS), default=None,
                  help="Staged get-up phase A|B|C (requires --task getup); "
                       "default trains the legacy single-policy getup task")
  ap.add_argument("--num-envs", type=int, default=2048,
                  help="Parallel sim environments (fewer for small GPUs)")
  ap.add_argument("--max-iterations", type=int, default=None,
                  help="Default: 5001 from the task config")
  ap.add_argument("--run-name", type=str, default=None)
  ap.add_argument("--resume", action="store_true",
                  help="Resume from the latest checkpoint of --run-name")
  ap.add_argument("--no-dashboard", action="store_true")
  ap.add_argument("--tensorboard", action="store_true",
                  help="Also launch raw TensorBoard (port+1) for deep dives")
  ap.add_argument("--port", type=int, default=6006)
  args = ap.parse_args()

  if args.stage is not None and args.task != "getup":
    ap.error("--stage is only valid with --task getup")

  sys.path.insert(0, APP_DIR)
  sys.path.insert(0, RL_MJLAB_DIR)
  # Run rooted in unitree_rl_mjlab so logs/, checkpoints and the dashboard
  # logdir always land in the same place no matter where this is invoked from.
  os.chdir(RL_MJLAB_DIR)

  dashboard_proc: subprocess.Popen | None = None
  tensorboard_proc: subprocess.Popen | None = None
  try:
    if not args.no_dashboard:
      if port_in_use(args.port):
        print(f"[all] Dashboard already up: http://localhost:{args.port}/")
      else:
        print(f"[all] Starting dashboard: http://localhost:{args.port}/")
        dashboard_proc = subprocess.Popen(
          [sys.executable, "-m", "g1_app.cli", "dashboard",
           "--port", str(args.port)],
          cwd=WORKSPACE,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
          if port_in_use(args.port):
            break
          time.sleep(1)
    if args.tensorboard and not port_in_use(args.port + 1):
      print(f"[all] Starting raw TensorBoard: http://localhost:{args.port + 1}/")
      tensorboard_proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main",
         "--logdir", "logs", "--port", str(args.port + 1)],
        cwd=RL_MJLAB_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      )

    # Populate the task registry: framework tasks + ours (import side effect).
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    import training.getup.config.g1  # noqa: F401  (registers Unitree-G1-Getup)
    import training.stand.config.g1  # noqa: F401  (registers Unitree-G1-Stand)

    train_mod = load_train_module()
    if args.stage is not None:
      task_id = STAGE_TASK_IDS[args.stage]
      metric = STAGE_METRICS[args.stage]
    else:
      task_id = TASK_IDS[args.task]
      metric = "stand_success"
    cfg = train_mod.TrainConfig.from_task(task_id)
    cfg.agent.logger = "tensorboard"
    cfg.env.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
      cfg.agent.max_iterations = args.max_iterations
    if args.run_name is not None:
      cfg.agent.run_name = args.run_name
    if args.resume:
      cfg.agent.resume = True
      # load_run is a regex over run folders (default ".*" = latest).
      if args.run_name is not None:
        cfg.agent.load_run = f".*{args.run_name}.*"

    print(f"[all] Training {task_id}: {args.num_envs} envs, "
          f"{cfg.agent.max_iterations} iterations", flush=True)
    print(f"[all] Watch Episode_Reward/{metric} at "
          f"http://localhost:{args.port}/", flush=True)
    train_mod.launch_training(task_id, cfg)
    print("[all] Training finished.")
    return 0
  except KeyboardInterrupt:
    print("\n[all] Interrupted, shutting down...")
    return 130
  finally:
    # Only stop servers we started ourselves; a pre-existing dashboard
    # (e.g. serving another training run) is left alone.
    for p in (dashboard_proc, tensorboard_proc):
      if p is not None and p.poll() is None:
        p.terminate()
    for p in (dashboard_proc, tensorboard_proc):
      if p is not None:
        try:
          p.wait(timeout=15)
        except subprocess.TimeoutExpired:
          p.kill()


if __name__ == "__main__":
  raise SystemExit(main())
