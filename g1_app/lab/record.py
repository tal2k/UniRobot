"""Record the latest get-up policy acting, headlessly on CPU.

Canonical home (moved from legacy record_getup.py). Prefer:

    g1 record --episodes 3
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

try:
    from g1_app.core.config import G1_MODEL_DIR, resolve_videos_dir
    from g1_app.core.math import euler_to_quat, quat_to_projected_gravity
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.config import G1_MODEL_DIR, resolve_videos_dir
        from core.math import euler_to_quat, quat_to_projected_gravity
    except ImportError:  # legacy flat layout
        from core.math import euler_to_quat, quat_to_projected_gravity  # type: ignore

        G1_MODEL_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "g1")

        def resolve_videos_dir():  # type: ignore
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            new = os.path.join(base, "outputs", "videos")
            old = os.path.join(base, "videos")
            return new if os.path.isdir(new) else old

DEFAULT_SCENE = os.path.join(G1_MODEL_DIR, "scene_29dof.xml")
VIDEOS_DIR = resolve_videos_dir()
# g1_app/lab/record.py -> APP_DIR=g1_app/, WORKSPACE=parent.
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(APP_DIR)
SIM_DT = 0.005
POLICY_DT = 0.02  # 50 Hz, matches training decimation


def _meta(session, key):
  for p in session.get_modelmeta().custom_metadata_map:
    if p == key:
      return session.get_modelmeta().custom_metadata_map[p]
  props = {}
  try:
    import onnx
    m = onnx.load(session._model_path if hasattr(session, "_model_path") else "")
    for pr in m.metadata_props:
      props[pr.key] = pr.value
  except Exception:
    pass
  return props.get(key, "")


def _csv(session, key):
  return [x for x in _meta(session, key).split(",") if x != ""]


class GetupRollout:
  def __init__(self, policy_path, scene=DEFAULT_SCENE, seed=0):
    self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    self.input_name = self.session.get_inputs()[0].name
    self.joint_names = _csv(self.session, "joint_names")
    self.default_pos = np.array([float(v) for v in _csv(self.session, "default_joint_pos")],
                                dtype=np.float32)
    self.kp = np.array([float(v) for v in _csv(self.session, "joint_stiffness")],
                       dtype=np.float32)
    self.kd = np.array([float(v) for v in _csv(self.session, "joint_damping")],
                       dtype=np.float32)
    self.action_scale = np.array([float(v) for v in _csv(self.session, "action_scale")],
                                 dtype=np.float32)
    n = len(self.joint_names)
    assert len(self.default_pos) == n, "metadata inconsistent"

    self.model = mujoco.MjModel.from_xml_path(scene)
    self.model.opt.timestep = SIM_DT
    assert self.model.nu == n, f"scene has {self.model.nu} motors, policy wants {n}"
    # map policy joint order -> mujoco qpos/qvel indices (by name, not position)
    self.qpos_idx, self.qvel_idx = [], []
    for jn in self.joint_names:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
      assert jid >= 0, f"joint {jn} not in scene"
      self.qpos_idx.append(self.model.jnt_qposadr[jid])
      self.qvel_idx.append(self.model.jnt_dofadr[jid])
    # policy joint i lives at these offsets inside qpos[7:] / qvel[6:]
    self.muj_q = np.array(self.qpos_idx) - 7
    self.muj_v = np.array(self.qvel_idx) - 6

    self.imu_gyro_adr = self.model.sensor_adr[
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
    self.imu_quat_adr = self.model.sensor_adr[
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]
    self.rng = np.random.default_rng(seed)
    self.last_action = np.zeros(n, dtype=np.float32)
    self.decimation = max(1, int(round(POLICY_DT / SIM_DT)))

  def reset_fallen(self, data):
    data.qpos[0:2] = self.rng.uniform(-0.3, 0.3, 2)
    data.qpos[2] = float(self.rng.uniform(0.1, 0.5))
    data.qpos[3:7] = euler_to_quat(
      float(self.rng.uniform(-math.pi, math.pi)),
      float(self.rng.uniform(-math.pi / 2, math.pi / 2)),
      float(self.rng.uniform(-math.pi, math.pi)))
    data.qpos[7:][self.muj_q] = (
      self.default_pos + self.rng.uniform(-0.6, 0.6, len(self.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)

  def reset_standing(self, data):
    data.qpos[0:2] = self.rng.uniform(-0.05, 0.05, 2)
    data.qpos[2] = float(self.rng.uniform(0.76, 0.80))
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:][self.muj_q] = (
      self.default_pos + self.rng.uniform(-0.05, 0.05, len(self.default_pos)))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, data)

  def observe(self, data):
    gyro = np.array(data.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3], dtype=np.float32)
    quat = np.array(data.sensordata[self.imu_quat_adr:self.imu_quat_adr + 4], dtype=np.float32)
    grav = quat_to_projected_gravity(quat)
    q = np.array(data.qpos[7:][self.muj_q], dtype=np.float32)
    v = np.array(data.qvel[6:][self.muj_v], dtype=np.float32)
    h = np.array([float(data.qpos[2])], dtype=np.float32)
    return np.concatenate([gyro, grav, h, q - self.default_pos, v,
                           self.last_action]).astype(np.float32)

  def act(self, data):
    obs = self.observe(data).reshape(1, -1)
    action = self.session.run(None, {self.input_name: obs})[0].flatten().astype(np.float32)
    self.last_action = action
    return self.default_pos + action * self.action_scale

  def apply_pd(self, data, target):
    q = np.array(data.qpos[7:][self.muj_q], dtype=np.float32)
    v = np.array(data.qvel[6:][self.muj_v], dtype=np.float32)
    torque = self.kp * (target - q) - self.kd * v
    full = np.zeros(self.model.nu, dtype=np.float32)
    full[self.muj_q] = torque
    for i in range(self.model.nu):
      lo, hi = self.model.actuator_ctrlrange[i]
      data.ctrl[i] = float(np.clip(full[i], lo, hi))


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
           out_path=None, seed=0, standing=False):
  os.environ.setdefault("MUJOCO_GL", "egl")
  roller = GetupRollout(policy_path, seed=seed)
  data = mujoco.MjData(roller.model)
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
      else:
        roller.reset_fallen(data)
      roller.last_action = np.zeros_like(roller.last_action)
      target = roller.default_pos.copy()
      min_h, max_h, tilt_end = 1e9, 0.0, 1.0
      tilts = []
      for step in range(n_steps):
        mujoco.mj_step(roller.model, data)
        if step % roller.decimation == 0:
          target = roller.act(data)
        roller.apply_pd(data, target)
        h = float(data.qpos[2])
        min_h, max_h = min(min_h, h), max(max_h, h)
        if step % every == 0:
          renderer.update_scene(data)
          writer.append_data(renderer.render())
        if step >= n_steps - int(1.0 / SIM_DT):
          q = data.sensordata[roller.imu_quat_adr:roller.imu_quat_adr + 4]
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
  ap.add_argument("--experiment", default="g1_getup",
                  help="Experiment folder under logs/rsl_rl (g1_getup|g1_stand)")
  ap.add_argument("--stand", action="store_true",
                  help="Start episodes standing (for stand policy) not fallen")
  ap.add_argument("--episodes", type=int, default=3)
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--fps", type=int, default=20)
  ap.add_argument("--out", default=None)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  run_dir = args.run_dir or find_latest_run(args.experiment)
  print(f"run: {run_dir}")
  record(latest_policy_onnx(run_dir), episodes=args.episodes, seconds=args.seconds,
         fps=args.fps, out_path=args.out, seed=args.seed, standing=args.stand)


if __name__ == "__main__":
  main()
