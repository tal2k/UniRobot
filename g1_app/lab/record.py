"""Record the latest get-up policy acting, headlessly on CPU.

Canonical home (moved from legacy record_getup.py). Prefer:

    g1 record --episodes 3
"""
import argparse
import glob
import json
import math
import os
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from core.bridge import StandStillPolicy
from core.config import G1_MODEL_DIR, resolve_videos_dir
from core.getup_stages import SUPINE_TARGET
from core.math import euler_to_quat, quat_to_projected_gravity

DEFAULT_SCENE = os.path.join(G1_MODEL_DIR, "scene_29dof.xml")
VIDEOS_DIR = resolve_videos_dir()
# g1_app/lab/record.py -> APP_DIR=g1_app/, WORKSPACE=parent.
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(APP_DIR)
SIM_DT = 0.005


class GetupRollout:
  """Headless policy rollout: episode resets + renderer loop.

  Policy loading, observations and PD live in `core.bridge.StandStillPolicy`
  (single implementation) — this class only adds resets and video.
  """

  def __init__(self, policy_path, scene=DEFAULT_SCENE, seed=0):
    self.model = mujoco.MjModel.from_xml_path(scene)
    self.model.opt.timestep = SIM_DT
    self.data = mujoco.MjData(self.model)
    self.pol = StandStillPolicy(self.model, self.data, policy_path,
                                sim_dt=SIM_DT)
    self.rng = np.random.default_rng(seed)

  def reset_fallen(self, data):
    data.qpos[0:2] = self.rng.uniform(-0.3, 0.3, 2)
    data.qpos[2] = float(self.rng.uniform(0.1, 0.5))
    data.qpos[3:7] = euler_to_quat(
      float(self.rng.uniform(-math.pi, math.pi)),
      float(self.rng.uniform(-math.pi / 2, math.pi / 2)),
      float(self.rng.uniform(-math.pi, math.pi)))
    data.qpos[7:][self.pol.muj_q] = (
      self.pol.default_pos + self.rng.uniform(-0.6, 0.6, len(self.pol.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)

  def reset_supine(self, data):
    """Reset to supine neutral (matching Stage B training reset)."""
    # Lying flat on back, height ~0.15
    data.qpos[0:2] = self.rng.uniform(-0.1, 0.1, 2)
    data.qpos[2] = float(self.rng.uniform(0.06, 0.20))
    # Supine: pitch = -pi/2, small roll/yaw noise
    roll = float(self.rng.uniform(-0.4, 0.4))
    pitch = -math.pi / 2 + float(self.rng.uniform(-0.4, 0.4))
    yaw = float(self.rng.uniform(-math.pi, math.pi))
    data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)
    # Joints at the shared supine target (flat on back, extended limbs) + noise.
    data.qpos[7:][self.pol.muj_q] = (
      np.asarray(SUPINE_TARGET, dtype=np.float32)
      + self.rng.uniform(-0.2, 0.2, len(self.pol.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)

  def reset_prone(self, data):
    """Reset to prone (matching Roll training resets, for roll-over eval)."""
    # Lying flat face-down, height ~0.15
    data.qpos[0:2] = self.rng.uniform(-0.1, 0.1, 2)
    data.qpos[2] = float(self.rng.uniform(0.06, 0.20))
    # Prone: pitch = +pi/2, small roll/yaw noise
    roll = float(self.rng.uniform(-0.4, 0.4))
    pitch = math.pi / 2 + float(self.rng.uniform(-0.4, 0.4))
    yaw = float(self.rng.uniform(-math.pi, math.pi))
    data.qpos[3:7] = euler_to_quat(roll, pitch, yaw)
    # Joints sprawled (Roll trains from ±0.6 offsets, not the supine pose).
    data.qpos[7:][self.pol.muj_q] = (
      np.asarray(SUPINE_TARGET, dtype=np.float32)
      + self.rng.uniform(-0.6, 0.6, len(self.pol.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)

  def reset_standing(self, data):
    data.qpos[0:2] = self.rng.uniform(-0.05, 0.05, 2)
    data.qpos[2] = float(self.rng.uniform(0.76, 0.80))
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:][self.pol.muj_q] = (
      self.pol.default_pos + self.rng.uniform(-0.05, 0.05, len(self.pol.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)


def latest_policy_onnx(run_dir):
  p = os.path.join(run_dir, "policy.onnx")
  if not os.path.isfile(p):
    raise FileNotFoundError(f"no policy.onnx snapshot in {run_dir} yet")
  return p


def find_latest_run(experiment="g1_getup"):
  cands = sorted(glob.glob(os.path.join(
    WORKSPACE, "unitree_rl_mjlab", "logs", "rsl_rl", experiment, "*")),
    key=os.path.getmtime)
  cands = [c for c in cands if os.path.isfile(os.path.join(c, "policy.onnx"))]
  if not cands:
    raise FileNotFoundError(f"no trained snapshots found for {experiment} yet")
  return cands[-1]


def record(policy_path, episodes=3, seconds=8.0, fps=20, width=480, height=360,
           out_path=None, seed=0, standing=False, start="fallen"):
  os.environ.setdefault("MUJOCO_GL", "egl")
  roller = GetupRollout(policy_path, seed=seed)
  data = roller.data
  pol = roller.pol
  renderer = mujoco.Renderer(roller.model, height=height, width=width)
  n_steps = int(seconds / SIM_DT)
  every = max(1, int(round(1.0 / fps / SIM_DT)))
  ckpt_mtime = os.path.getmtime(policy_path)
  results = {"policy": policy_path, "checkpoint_mtime": ckpt_mtime,
             "recorded_at": time.time(), "episodes": []}
  if out_path is None:
    tag = f"{os.path.basename(os.path.dirname(policy_path))}_{int(ckpt_mtime)}"
    prefix = "stand" if standing else "getup"
    out_path = os.path.join(VIDEOS_DIR, f"{prefix}_{tag}.mp4")
  os.makedirs(VIDEOS_DIR, exist_ok=True)

  writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7)
  try:
    for ep in range(episodes):
      if standing:
        roller.reset_standing(data)
      elif start == "supine":
        roller.reset_supine(data)
      elif start == "prone":
        roller.reset_prone(data)
      else:
        roller.reset_fallen(data)
      pol.last_action = np.zeros_like(pol.last_action)
      min_h, max_h, tilt_end = 1e9, 0.0, 1.0
      tilts = []
      for step in range(n_steps):
        mujoco.mj_step(roller.model, data)
        if step % pol.decimation == 0:
          pol.update_policy()
        pol.apply_pd()
        h = float(data.qpos[2])
        min_h, max_h = min(min_h, h), max(max_h, h)
        if step % every == 0:
          renderer.update_scene(data)
          writer.append_data(renderer.render())
        if step >= n_steps - int(1.0 / SIM_DT):
          q = data.sensordata[pol.imu_quat_adr:pol.imu_quat_adr + 4]
          tilts.append(float(np.linalg.norm(quat_to_projected_gravity(q)[:2])))
      tilt_end = float(np.mean(tilts)) if tilts else 1.0
      success = bool(min_h > 0.35 and max_h > 0.70 and tilt_end < 0.3)
      results["episodes"].append(
        {"success": success, "min_h": round(min_h, 3),
         "max_h": round(max_h, 3), "tilt_end": round(tilt_end, 3)})
      print(f"episode {ep + 1}/{episodes}: success={success} "
            f"min_h={min_h:.2f} max_h={max_h:.2f} tilt={tilt_end:.2f}", flush=True)
  finally:
    writer.close()
    renderer.close()
  results["stood_up"] = f"{sum(e['success'] for e in results['episodes'])}/{episodes}"
  with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
    json.dump(results, f, indent=1)
  print(f"saved {out_path} ({results['stood_up']} stood up)")
  return out_path, results


def main():
    ap = argparse.ArgumentParser(description="Record latest policy (CPU, headless)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--policy", default=None,
                    help="policy.onnx path directly (e.g. models/g1_stand_policy.onnx); "
                         "overrides --run-dir/--experiment lookup")
    ap.add_argument("--experiment", default="g1_getup_standup",
                    help="Experiment folder under logs/rsl_rl "
                         "(g1_getup_standup|g1_getup_roll|g1_stand)")
    ap.add_argument("--stand", action="store_true",
                    help="Start episodes standing (for stand policy) not fallen")
    ap.add_argument("--start", choices=["fallen", "supine", "prone"], default="fallen",
                    help="Start state: fallen (random sprawl), supine (flat on "
                         "back, extended) or prone (face-down, for roll-over eval)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    policy_path = args.policy
    if policy_path is None:
        run_dir = args.run_dir or find_latest_run(args.experiment)
        print(f"run: {run_dir}")
        policy_path = latest_policy_onnx(run_dir)
    record(policy_path, episodes=args.episodes, seconds=args.seconds,
           fps=args.fps, out_path=args.out, seed=args.seed, standing=args.stand,
           start=args.start)


if __name__ == "__main__":
  main()
