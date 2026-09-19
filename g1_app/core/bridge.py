"""G1 policy bridge: ONNX inference + PD torque + standstill logic.

Canonical home of G1StandPolicy (moved from legacy g1_stand_onnx.py).
New code: `from core.bridge import G1StandPolicy, reset_standing, run_stand`
(or `from g1_app.core.bridge import ...` from the workspace root).
"""
import argparse
import os
import time

import mujoco
import numpy as np
import onnxruntime as ort

try:  # package-relative (pip install / python -m g1_app.cli)
    from .config import DEFAULT_LOCAL_POLICY, G1_MODEL_DIR, get_local_cfg
    from .math import quat_to_projected_gravity
    from .terrains import TERRAINS, resolve_scene
except ImportError:  # legacy flat sys.path (APP_DIR on sys.path)
    from core.config import DEFAULT_LOCAL_POLICY, G1_MODEL_DIR, get_local_cfg
    from core.math import quat_to_projected_gravity
    from core.terrains import TERRAINS, resolve_scene

DEFAULT_SCENE = os.path.join(G1_MODEL_DIR, "scene_29dof.xml")

# Local 29-DoF config: deploy.yaml is the authority, fallback is built in.
LOCAL_CFG = get_local_cfg()

# ---------------------------------------------------------------- zoo 12-DoF metadata (g1_policy_metadata.json)
ZOO_CFG = {
    "step_dt": 0.02,  # standard Unitree locomotion policy rate (50 Hz)
    "num_obs": 47,
    "num_act": 12,
    "action_scale": 0.25,
    "default_pos": [-0.1, 0, 0.0, 0.3, -0.2, 0, -0.1, 0, 0.0, 0.3, -0.2, 0],
    "kp": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
    "kd": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
    "ang_vel_scale": 0.25,
    "dof_pos_scale": 1.0,
    "dof_vel_scale": 0.05,
    "cmd_scale": [2.0, 2.0, 0.25],
    "gait_period": 0.6,
}
# Upper-body hold pose/gains used with the zoo legs-only policy (legs = policy,
# waist+arms = fixed PD). Upper defaults mirror the local training defaults.
ZOO_UPPER_DEFAULT = [0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0]
ZOO_UPPER_KP = [40.2, 28.5, 28.5, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
                14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8]
ZOO_UPPER_KD = [2.6, 1.8, 1.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
                0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1]


class G1StandPolicy:
    def __init__(self, mj_model, mj_data, policy_path, sim_dt=0.005,
                 standstill=True):
        """standstill: with a zero user command, lock the robot in place:
        while still moving, a small corrective command holds the anchor
        position (marching in place, ~2 cm drift); once settled, the gait
        phase is frozen and the policy holds a static pose (~1 cm drift).
        Set False for the raw march-in-place behaviour (~30 cm drift/min).
        Use set_command() so the standstill logic sees the user intent."""
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.num_motor = mj_model.nu
        assert self.num_motor == 29, f"expected 29 G1 motors, got {self.num_motor}"

        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"policy not found: {policy_path}")
        if os.path.getsize(policy_path) < 10000:
            # LFS pointer file, not a real model
            with open(policy_path) as f:
                head = f.read(200)
            raise ValueError(
                f"{policy_path} looks like a Git-LFS pointer, not a model:\n{head}\n"
                "The onnx_policy zoo LFS objects are currently missing server-side (404)."
            )
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.obs_dim = self.session.get_inputs()[0].shape[1]
        self.is_lstm = len(self.input_names) > 1
        print(f"ONNX inputs: {[(i.name, i.shape) for i in self.session.get_inputs()]}")
        print(f"ONNX outputs: {[(o.name, o.shape) for o in self.session.get_outputs()]}")

        if self.obs_dim == 98:
            self.mode = "local29"
            cfg = LOCAL_CFG
            self.default_pos = np.array(cfg["default_pos"], dtype=np.float32)
            self.action_scale = np.array(cfg["action_scale"], dtype=np.float32)
            self.kp = np.array(cfg["stiffness"], dtype=np.float32)
            self.kd = np.array(cfg["damping"], dtype=np.float32)
            self.step_dt = cfg["step_dt"]
            self.gait_period = cfg["gait_period"]
            print("Mode: local 29-DoF velocity policy (stand with cmd=0)")
        elif self.obs_dim == 47:
            self.mode = "zoo12"
            self.default_legs = np.array(ZOO_CFG["default_pos"], dtype=np.float32)
            self.default_pos = np.concatenate(
                [self.default_legs, np.array(ZOO_UPPER_DEFAULT, dtype=np.float32)]
            ).astype(np.float32)
            self.kp = np.array(ZOO_CFG["kp"] + ZOO_UPPER_KP, dtype=np.float32)
            self.kd = np.array(ZOO_CFG["kd"] + ZOO_UPPER_KD, dtype=np.float32)
            self.step_dt = ZOO_CFG["step_dt"]
            self.gait_period = ZOO_CFG["gait_period"]
            # LSTM hidden state (num_layers=1, hidden=64)
            self.h = np.zeros((1, 1, 64), dtype=np.float32)
            self.c = np.zeros((1, 1, 64), dtype=np.float32)
            print("Mode: zoo 12-DoF locomotion policy (legs via LSTM, upper body PD hold)")
        else:
            raise ValueError(f"unsupported obs dim {self.obs_dim} (expected 98 or 47)")

        n_act = 29 if self.mode == "local29" else 12
        self.last_action = np.zeros(n_act, dtype=np.float32)
        self.target_pos = self.default_pos.copy()
        self.phase = 0.0
        self.user_command = np.zeros(3, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # effective cmd fed to policy
        self.standstill = standstill
        self.stand_state = "walk"  # walk | hold | frozen
        self._anchor = np.zeros(2, dtype=np.float32)
        self._slow_steps = 0
        self._speed_ema = 0.0
        self.hold_gain = 2.0       # position-hold P gain (m/s per m)
        self.hold_max_cmd = 0.3    # max corrective command (m/s)
        self.settle_ema = 0.08     # EMA(root speed) below this counts as settled
        self.settle_steps = 50     # ... for this many policy steps -> frozen
        self.disturb_speed = 0.30  # above this in frozen -> back to hold

        self.sim_dt = sim_dt
        self.decimation = max(1, int(round(self.step_dt / self.sim_dt)))
        self._sim_steps = 0
        print(f"Policy {1.0 / self.step_dt:.0f} Hz, sim {1.0 / self.sim_dt:.0f} Hz, "
              f"decimation {self.decimation}")

        m = mj_model
        self.imu_gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        self.imu_quat_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]

    # -- observations -----------------------------------------------------
    def _base_state(self):
        gyro = np.array(self.mj_data.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3],
                        dtype=np.float32)
        quat = np.array(self.mj_data.sensordata[self.imu_quat_adr:self.imu_quat_adr + 4],
                        dtype=np.float32)
        grav = quat_to_projected_gravity(quat)
        qpos = np.array(self.mj_data.qpos[7:7 + self.num_motor], dtype=np.float32)
        qvel = np.array(self.mj_data.qvel[6:6 + self.num_motor], dtype=np.float32)
        return gyro, grav, qpos, qvel

    def _obs_local29(self, gyro, grav, qpos, qvel):
        phase_obs = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        return np.concatenate([
            gyro, grav, self.command, phase_obs,
            (qpos - self.default_pos).astype(np.float32), qvel,
            self.last_action.astype(np.float32),
        ]).astype(np.float32)

    def _obs_zoo12(self, gyro, grav, qpos, qvel):
        q_legs, v_legs = qpos[:12], qvel[:12]
        phase_obs = np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32)
        cmd = self.command * np.array(ZOO_CFG["cmd_scale"], dtype=np.float32)
        return np.concatenate([
            (gyro * ZOO_CFG["ang_vel_scale"]).astype(np.float32),
            grav.astype(np.float32),
            cmd.astype(np.float32),
            ((q_legs - self.default_legs) * ZOO_CFG["dof_pos_scale"]).astype(np.float32),
            (v_legs * ZOO_CFG["dof_vel_scale"]).astype(np.float32),
            self.last_action.astype(np.float32),
            phase_obs,
        ]).astype(np.float32)

    def set_command(self, vx, vy, wz):
        self.user_command = np.array([vx, vy, wz], dtype=np.float32)

    def _standstill_update(self):
        """Decide effective command + whether to advance phase. Returns True
        if the gait phase should advance this policy step."""
        if not self.standstill or not np.allclose(self.user_command, 0.0):
            self.stand_state = "walk"
            self.command = self.user_command.copy()
            return True
        root_xy = np.array(self.mj_data.qpos[:2], dtype=np.float32)
        speed = float(np.linalg.norm(self.mj_data.qvel[:3]))
        if self.stand_state == "walk":  # just released -> anchor here
            self._anchor = root_xy.copy()
            self.stand_state = "hold"
            self._slow_steps = 0
            self._speed_ema = speed
        if self.stand_state == "hold":
            err = self._anchor - root_xy
            self.command = np.append(
                np.clip(self.hold_gain * err, -self.hold_max_cmd,
                        self.hold_max_cmd), 0.0).astype(np.float32)
            self._speed_ema = 0.95 * self._speed_ema + 0.05 * speed
            if self._speed_ema < self.settle_ema:
                self._slow_steps += 1
            else:
                self._slow_steps = 0
            if self._slow_steps >= self.settle_steps:  # settled -> static pose
                self.stand_state = "frozen"
            return True
        # frozen: static pose, phase held
        if speed > self.disturb_speed:  # pushed/disturbed -> back to active hold
            self.stand_state = "hold"
            self._slow_steps = 0
            return True
        self.command = np.zeros(3, dtype=np.float32)
        return False

    # -- policy update (50 Hz) ---------------------------------------------
    def update_policy(self):
        if self._standstill_update():
            self.phase += 2.0 * np.pi / self.gait_period * self.step_dt
        gyro, grav, qpos, qvel = self._base_state()
        if self.mode == "local29":
            obs = self._obs_local29(gyro, grav, qpos, qvel).reshape(1, -1)
            action = self.session.run(None, {self.input_names[0]: obs})[0].flatten()
            self.last_action = action.astype(np.float32)
            self.target_pos = self.default_pos + self.last_action * self.action_scale
        else:
            obs = self._obs_zoo12(gyro, grav, qpos, qvel).reshape(1, -1)
            feed = {self.input_names[0]: obs}
            # map hidden states by name heuristic
            for nm, val in (("h", self.h), ("c", self.c), ("h_in", self.h), ("c_in", self.c),
                            ("lstm_h_in", self.h), ("lstm_c_in", self.c)):
                if nm in feed or nm in self.input_names:
                    feed[nm] = val
            # fill any remaining inputs with hidden states in order
            outs = self.session.get_outputs()
            res = self.session.run([o.name for o in outs], feed)
            action = np.asarray(res[0]).flatten().astype(np.float32)
            # try to pick up updated hidden states if the model returns them
            if len(res) >= 3:
                try:
                    self.h = np.asarray(res[1], dtype=np.float32).reshape(self.h.shape)
                    self.c = np.asarray(res[2], dtype=np.float32).reshape(self.c.shape)
                except Exception:
                    pass
            self.last_action = action
            leg_target = self.default_legs + self.last_action * ZOO_CFG["action_scale"]
            self.target_pos = np.concatenate([leg_target, self.default_pos[12:]]).astype(np.float32)
        return self.last_action

    # -- torque every sim step ----------------------------------------------
    def apply_pd(self):
        qpos = np.array(self.mj_data.qpos[7:7 + self.num_motor], dtype=np.float32)
        qvel = np.array(self.mj_data.qvel[6:6 + self.num_motor], dtype=np.float32)
        torque = self.kp * (self.target_pos - qpos) - self.kd * qvel
        # clamp to actuator limits for safety
        for i in range(self.num_motor):
            lo, hi = self.mj_model.actuator_ctrlrange[i]
            self.mj_data.ctrl[i] = float(np.clip(torque[i], lo, hi))

    def step_sim(self):
        do_policy = (self._sim_steps % self.decimation == 0)
        if do_policy:
            action = self.update_policy()
        else:
            action = None
        self.apply_pd()
        self._sim_steps += 1
        return action

    # -- public telemetry (GUI/CLI should use this, not _base_state) ---------
    def telemetry(self):
        """Return (height, tilt, user_command, status_hint) for status lines."""
        h = float(self.mj_data.qpos[2])
        _, grav, _, _ = self._base_state()
        tilt = float(np.linalg.norm(grav[:2]))
        return h, tilt, self.user_command, self.stand_state


def reset_standing(mj_model, mj_data, default_pos, height=0.78):
    mj_data.qpos[0] = 0.0
    mj_data.qpos[1] = 0.0
    mj_data.qpos[2] = height
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qpos[7:7 + len(default_pos)] = default_pos
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


def run_stand(policy_path=DEFAULT_LOCAL_POLICY, scene=DEFAULT_SCENE,
              seconds=10.0, sim_dt=0.005, headless=False, log_hz=2.0,
              standstill=True):
    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)

    bridge = G1StandPolicy(mj_model, mj_data, policy_path, sim_dt=sim_dt,
                           standstill=standstill)
    reset_standing(mj_model, mj_data, bridge.default_pos)

    viewer = None
    if not headless:
        try:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        except Exception as e:
            print(f"viewer unavailable ({e}), continuing headless")
            headless = True

    n_steps = int(seconds / sim_dt)
    log_every = max(1, int((1.0 / log_hz) / sim_dt))
    t0 = time.perf_counter()
    min_h, max_tilt, fell = 1e9, 0.0, False
    for step in range(n_steps):
        mujoco.mj_step(mj_model, mj_data)
        bridge.step_sim()
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
        h = float(mj_data.qpos[2])
        min_h = min(min_h, h)
        gyro, grav, _, _ = bridge._base_state()
        tilt = float(np.linalg.norm(grav[:2]))
        max_tilt = max(max_tilt, tilt)
        if not np.all(np.isfinite(mj_data.qpos)) or h < 0.35 or h > 1.5 or tilt > 0.9:
            print(f"FALL/unstable at t={step * sim_dt:.2f}s h={h:.3f} tilt={tilt:.3f}")
            fell = True
            break
        if step % log_every == 0:
            print(f"t={step * sim_dt:5.2f}s h={h:.3f} tilt={tilt:.3f} "
                  f"act[:4]={np.round(bridge.last_action[:4], 3)}")
        if not headless and viewer is not None:
            time_until = sim_dt - (time.perf_counter() - t0 - step * sim_dt)
            if time_until > 0:
                time.sleep(time_until)
    print(f"done: fell={fell} min_h={min_h:.3f} max_tilt={max_tilt:.3f} "
          f"final_h={float(mj_data.qpos[2]):.3f}")
    if viewer is not None:
        viewer.close()
    return not fell


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="G1 balanced stand with pretrained ONNX policy")
    ap.add_argument("--policy", default=DEFAULT_LOCAL_POLICY)
    ap.add_argument("--scene", default=None,
                    help="MuJoCo scene XML (overrides --terrain)")
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="flat",
                    help="Bundled test terrain (default: flat)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-standstill", action="store_true",
                    help="Disable position lock: zero command marches in place")
    args = ap.parse_args()
    scene = resolve_scene(args.terrain, args.scene)
    ok = run_stand(args.policy, scene, args.seconds, args.sim_dt,
                   args.headless, standstill=not args.no_standstill)
    raise SystemExit(0 if ok else 1)
