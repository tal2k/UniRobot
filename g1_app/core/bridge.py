"""G1 policy bridge: ONNX inference + PD torque + standstill logic.

Canonical home of G1StandPolicy (moved from legacy g1_stand_onnx.py).
New code: `from core.bridge import G1StandPolicy, reset_standing, run_stand`
(or `from g1_app.core.bridge import ...` from the workspace root).
"""
import os
import time

import mujoco
import numpy as np
import onnxruntime as ort

from .config import DEFAULT_LOCAL_POLICY, G1_MODEL_DIR, MODELS_DIR, WORKSPACE, get_local_cfg
from .getup_stages import (
    DONE,
    GETUP,
    IDLE,
    ROLL,
    STAGE_SEQUENCE,
    SUPINE_TARGET,
    StageSwitcher,
    rms_pose_error,
)
from .math import euler_to_quat, quat_to_projected_gravity

DEFAULT_SCENE = os.path.join(G1_MODEL_DIR, "scene_29dof.xml")

# Curated stand-still policy: download/Colab exports land here, next to the
# walking policy (LFS-tracked). Training snapshots stay under logs/.
STAND_POLICY_PATH = os.path.join(MODELS_DIR, "g1_stand_policy.onnx")

# Staged get-up policies (v2): curated exports first (models/), newest
# training snapshot under logs/rsl_rl/<experiment>/ as fallback — same rule
# as stand. Kind is "roll" (any fall -> supine), "getup" (supine -> stand)
# or "brace" (doomed fall -> safe landing, independent pre-impact stage).
RECOVERY_POLICY = {
    "roll": ("g1_getup_roll_policy.onnx", "g1_getup_roll"),
    "getup": ("g1_getup_standup_policy.onnx", "g1_getup_standup"),
    "brace": ("g1_brace_policy.onnx", "g1_getup_brace"),
}

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


def find_latest_stand_policy():
    """Curated stand-still policy if present, else newest training snapshot.

    `g1_app/models/g1_stand_policy.onnx` is the curated home (lives next to
    the walking policy, versioned via LFS); the logs glob is the scratch
    fallback for fresh training runs not yet promoted.
    """
    import glob

    if os.path.isfile(STAND_POLICY_PATH):
        return STAND_POLICY_PATH
    cands = sorted(glob.glob(os.path.join(
        WORKSPACE, "unitree_rl_mjlab", "logs", "rsl_rl", "g1_stand", "*",
        "policy.onnx")), key=os.path.getmtime)
    return cands[-1] if cands else None


def find_latest_recovery_policy(kind: str):
    """Curated recovery policy if present, else newest training snapshot.

    Kind is "roll" (any fall -> supine), "getup" (supine -> stand) or
    "brace" (doomed fall -> safe landing, independent stage). Same lookup
    rule as `find_latest_stand_policy`; returns None when nothing trained.
    """
    import glob

    curated_name, experiment = RECOVERY_POLICY[kind]
    curated = os.path.join(MODELS_DIR, curated_name)
    if os.path.isfile(curated):
        return curated
    cands = sorted(glob.glob(os.path.join(
        WORKSPACE, "unitree_rl_mjlab", "logs", "rsl_rl", experiment, "*",
        "policy.onnx")), key=os.path.getmtime)
    return cands[-1] if cands else None


def _csv_meta(session, key):
    for k, v in session.get_modelmeta().custom_metadata_map.items():
        if k == key:
            return [x for x in v.split(",") if x != ""]
    return []


class StandStillPolicy:
    """94-dim in-place balance policy (trained by `g1 train-stand`).

    Observation layout (actor): gyro 3 + projected gravity 3 + base height 1
    + (q - q0) 29 + qvel 29 + last action 29. No velocity command, no gait
    phase — balance is reactive, not steered or rhythmic. Gains, nominal
    pose and joint order come from the ONNX metadata (self-describing),
    joints are mapped by name so policy/MuJoCo order need not match.
    """

    STEP_DT = 0.02  # 50 Hz, matches training decimation

    def __init__(self, mj_model, mj_data, policy_path, sim_dt=0.005):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.num_motor = mj_model.nu
        assert self.num_motor == 29, f"expected 29 G1 motors, got {self.num_motor}"

        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"stand policy not found: {policy_path}")
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.obs_dim = self.session.get_inputs()[0].shape[1]
        if self.obs_dim != 94:
            raise ValueError(f"stand policy obs dim {self.obs_dim}, expected 94")

        meta = self.session.get_modelmeta().custom_metadata_map
        _ = meta  # metadata access goes through _csv_meta for missing-key safety
        self.joint_names = _csv_meta(self.session, "joint_names")
        assert len(self.joint_names) == 29, "stand policy metadata lacks joint_names"
        self.default_pos = np.array(
            [float(v) for v in _csv_meta(self.session, "default_joint_pos")],
            dtype=np.float32)
        self.kp = np.array(
            [float(v) for v in _csv_meta(self.session, "joint_stiffness")],
            dtype=np.float32)
        self.kd = np.array(
            [float(v) for v in _csv_meta(self.session, "joint_damping")],
            dtype=np.float32)
        self.action_scale = np.array(
            [float(v) for v in _csv_meta(self.session, "action_scale")],
            dtype=np.float32)
        assert len(self.default_pos) == 29 and len(self.kp) == 29

        # Policy joint i lives at these offsets inside qpos[7:] / qvel[6:].
        self.qpos_idx, self.qvel_idx = [], []
        for jn in self.joint_names:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            assert jid >= 0, f"joint {jn} not in scene"
            self.qpos_idx.append(self.mj_model.jnt_qposadr[jid])
            self.qvel_idx.append(self.mj_model.jnt_dofadr[jid])
        self.muj_q = np.array(self.qpos_idx) - 7
        self.muj_v = np.array(self.qvel_idx) - 6

        m = mj_model
        self.imu_gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        self.imu_quat_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]

        self.last_action = np.zeros(29, dtype=np.float32)
        self.target_pos = self.default_pos.copy()
        self.user_command = np.zeros(3, dtype=np.float32)
        self.mode = "stand94"
        self.sim_dt = sim_dt
        self.decimation = max(1, int(round(self.STEP_DT / self.sim_dt)))
        self._sim_steps = 0
        print("Mode: stand-still 94-dim balance policy "
              f"(50 Hz, sim {1.0 / self.sim_dt:.0f} Hz, decimation {self.decimation})")

    def set_command(self, vx, vy, wz):
        # Stored for telemetry/switch decisions; the policy itself takes none.
        self.user_command = np.array([vx, vy, wz], dtype=np.float32)

    def _base_state(self):
        gyro = np.array(self.mj_data.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3],
                        dtype=np.float32)
        quat = np.array(self.mj_data.sensordata[self.imu_quat_adr:self.imu_quat_adr + 4],
                        dtype=np.float32)
        grav = quat_to_projected_gravity(quat)
        qpos = np.array(self.mj_data.qpos[7:7 + self.num_motor], dtype=np.float32)
        qvel = np.array(self.mj_data.qvel[6:6 + self.num_motor], dtype=np.float32)
        return gyro, grav, qpos, qvel

    def observe(self):
        gyro, grav, _, _ = self._base_state()
        q = np.array(self.mj_data.qpos[7:][self.muj_q], dtype=np.float32)
        v = np.array(self.mj_data.qvel[6:][self.muj_v], dtype=np.float32)
        h = np.array([float(self.mj_data.qpos[2])], dtype=np.float32)
        return np.concatenate([gyro, grav, h, q - self.default_pos, v,
                               self.last_action]).astype(np.float32)

    def update_policy(self):
        obs = self.observe().reshape(1, -1)
        action = self.session.run(None, {self.input_names[0]: obs})[0].flatten()
        self.last_action = action.astype(np.float32)
        self.target_pos = self.default_pos + self.last_action * self.action_scale
        return self.last_action

    def apply_pd(self):
        qpos = np.array(self.mj_data.qpos[7:7 + self.num_motor], dtype=np.float32)
        qvel = np.array(self.mj_data.qvel[6:6 + self.num_motor], dtype=np.float32)
        torque = self.kp * (self.target_pos - qpos) - self.kd * qvel
        for i in range(self.num_motor):
            lo, hi = self.mj_model.actuator_ctrlrange[i]
            self.mj_data.ctrl[i] = float(np.clip(torque[i], lo, hi))

    def step_sim(self):
        do_policy = (self._sim_steps % self.decimation == 0)
        action = self.update_policy() if do_policy else None
        self.apply_pd()
        self._sim_steps += 1
        return action

    def telemetry(self):
        h = float(self.mj_data.qpos[2])
        _, grav, _, _ = self._base_state()
        tilt = float(np.linalg.norm(grav[:2]))
        return h, tilt, self.user_command, "stand"


class WalkStandBridge:
    """Velocity walking + stand-still balance with cross-faded switching.

    mode: "walk" (velocity policy only, legacy behaviour), "stand"
    (balance policy only) or "auto" (nonzero user command walks, zero
    command balances — replaces the anchor/frozen standstill hack).
    A zero command does NOT hand over mid-stride: the walk policy brakes
    to ~standstill first (it owns deceleration), and the balance policy
    only takes over a slow root — the state distribution it trained on.
    On every switch the joint targets cross-fade over blend_s seconds so
    torques never jump.
    """

    def __init__(self, mj_model, mj_data, walk_policy_path, stand_policy_path,
                 sim_dt=0.005, mode="auto", standstill=True, blend_s=0.4,
                 stop_speed=0.12):
        assert mode in ("walk", "stand", "auto")
        self.walk = G1StandPolicy(mj_model, mj_data, walk_policy_path,
                                  sim_dt=sim_dt, standstill=standstill)
        self.stand = StandStillPolicy(mj_model, mj_data, stand_policy_path,
                                      sim_dt=sim_dt)
        self._standstill = standstill
        self.mode_sel = mode
        self.active = "stand" if mode == "stand" else "walk"
        self.blend_s = blend_s
        self._blend = 1.0
        self._prev_target = self._active().target_pos.copy()
        self.default_pos = self.walk.default_pos.copy()
        self.user_command = np.zeros(3, dtype=np.float32)
        self.stop_speed = float(stop_speed)
        self._sim_steps = 0
        print(f"Mode: walk/stand switch (selected={mode}, active={self.active})")

    def _active(self):
        return self.stand if self.active == "stand" else self.walk

    @property
    def mode(self):
        return f"walk+stand({self.active})"

    @property
    def last_action(self):
        return self._active().last_action

    @property
    def phase(self):
        return self.walk.phase

    @phase.setter
    def phase(self, value):
        self.walk.phase = value

    @property
    def standstill(self):
        return self._standstill

    @standstill.setter
    def standstill(self, value):
        self._standstill = bool(value)
        self.walk.standstill = bool(value)

    def _base_state(self):
        return self.walk._base_state()

    def set_command(self, vx, vy, wz):
        self.user_command = np.array([vx, vy, wz], dtype=np.float32)
        self.walk.set_command(vx, vy, wz)
        self.stand.set_command(vx, vy, wz)

    def set_mode(self, mode):
        assert mode in ("walk", "stand", "auto")
        self.mode_sel = mode

    def _wanted(self):
        if self.mode_sel in ("walk", "stand"):
            return self.mode_sel
        if not np.allclose(self.user_command, 0.0):
            return "walk"
        # Zero command: stay on the walk policy until the root is slow.
        # Handing a mid-stride state to the balance policy (trained from
        # standing starts) is a guaranteed fall; the walker owns braking.
        if self.active == "walk" and self._root_speed() > self.stop_speed:
            return "walk"
        return "stand"

    def _root_speed(self):
        return float(np.linalg.norm(self.walk.mj_data.qvel[0:2]))

    def step_sim(self):
        wanted = self._wanted()
        if wanted != self.active:
            self._prev_target = self._blended_target().copy()
            self.active = wanted
            self._blend = 0.0
        policy = self._active()
        action = policy.step_sim()
        if self._blend < 1.0:
            n = max(1, int(round(self.blend_s / policy.sim_dt)))
            self._blend = min(1.0, self._blend + 1.0 / n)
            policy.target_pos = self._blended_target()
            # Re-apply PD with the blended target: policy.step_sim() above
            # already ran PD once with the unblended target, and the blend
            # only converges by overwriting ctrl on this same sim step.
            policy.apply_pd()
        self._sim_steps += 1
        return action

    def _blended_target(self):
        a = self._blend
        return ((1.0 - a) * self._prev_target + a * self._active().target_pos
                ).astype(np.float32)

    def apply_pd(self):
        self._active().apply_pd()

    def telemetry(self):
        h = float(self.walk.mj_data.qpos[2])
        _, grav, _, _ = self.walk._base_state()
        tilt = float(np.linalg.norm(grav[:2]))
        if self.active == "stand":
            return h, tilt, self.user_command, "stand"
        return h, tilt, self.user_command, self.walk.stand_state


class GetUpBridge:
    """Staged get-up v2: ROLL (any fall -> supine) -> GETUP (supine -> stand).

    Holds one 94-dim policy per phase (same obs/action contract as
    StandStillPolicy) plus an optional balance policy for the standing
    handoff. `core.getup_stages.StageSwitcher` decides transitions from
    height/tilt/speed/facing/supine-pose-error with hysteresis and per-stage
    timeouts; every switch cross-fades joint targets over blend_s (same rule
    as WalkStandBridge). PD runs every sim step, inference every 4th (50 Hz).

    Facing (body-x projected gravity, yaw-invariant) tells face-up (~-1)
    from face-down (~+1) from the IMU alone, so prone/side falls roll first
    and already-supine falls skip straight to GETUP. Velocity commands are
    ignored in v2: recovery is a zero-command mode. `telemetry()` reports
    (height, tilt, user_command, state) with state in {idle, roll, getup,
    done} for CLI/GUI status lines.
    """

    def __init__(self, mj_model, mj_data, stage_policies, sim_dt=0.005,
                 blend_s=0.4, stand_policy_path=None):
        missing = [s for s in STAGE_SEQUENCE if s not in stage_policies]
        assert not missing, f"missing recovery policies: {missing}"
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.stage_policies = {
            s: StandStillPolicy(mj_model, mj_data, stage_policies[s], sim_dt=sim_dt)
            for s in STAGE_SEQUENCE
        }
        self.stand = None
        if stand_policy_path is not None:
            self.stand = StandStillPolicy(mj_model, mj_data, stand_policy_path,
                                          sim_dt=sim_dt)
        ref = self.stage_policies[ROLL]
        self.switcher = StageSwitcher(dt=ref.STEP_DT)
        self.active = IDLE
        self.blend_s = blend_s
        self._blend = 1.0
        self._prev_target = ref.default_pos.copy()
        self.sim_dt = sim_dt
        self._sim_steps = 0
        self.default_pos = ref.default_pos.copy()
        self.user_command = np.zeros(3, dtype=np.float32)
        print(f"Mode: staged get-up ({' -> '.join(STAGE_SEQUENCE)})"
              + (", balance handoff" if self.stand is not None
                 else ", hold pose on done"))

    def _active_policy(self):
        if self.active == IDLE:
            return self.stand if self.stand is not None else self.stage_policies[ROLL]
        if self.active == DONE:
            return self.stand if self.stand is not None else self.stage_policies[GETUP]
        return self.stage_policies[self.active]

    @property
    def mode(self):
        return f"getup({self.active})"

    @property
    def last_action(self):
        return self._active_policy().last_action

    def set_command(self, vx, vy, wz):
        # Stored for telemetry/switch decisions; recovery ignores commands.
        self.user_command = np.array([vx, vy, wz], dtype=np.float32)
        for p in self.stage_policies.values():
            p.set_command(vx, vy, wz)
        if self.stand is not None:
            self.stand.set_command(vx, vy, wz)

    def _base_state(self):
        return self.stage_policies[ROLL]._base_state()

    def _height_tilt_facing(self):
        h = float(self.mj_data.qpos[2])
        _, grav, _, _ = self._base_state()
        tilt = float(np.linalg.norm(grav[:2]))
        # Facing: body-x projected gravity (yaw-invariant). Supine ~-1
        # (down toward the back), prone ~+1 (down into the chest), side ~0.
        return h, tilt, float(grav[0])

    def _metrics(self):
        """(height, tilt, root speed, facing, supine pose error)."""
        h, tilt, facing = self._height_tilt_facing()
        speed = float(np.linalg.norm(self.mj_data.qvel[0:2]))
        ref = self.stage_policies[ROLL]
        q = np.array(self.mj_data.qpos[7:][ref.muj_q], dtype=np.float32)
        # q is in policy joint order; SUPINE_TARGET is all zeros, so the RMS
        # is order-independent. Keeps the ROLL handoff inside GETUP's start
        # distribution (joints near supine).
        pose_err = rms_pose_error(q, SUPINE_TARGET)
        return h, tilt, speed, facing, pose_err

    def step_sim(self):
        policy = self._active_policy()
        if self._sim_steps % policy.decimation == 0:
            h, tilt, speed, facing, pose_err = self._metrics()
            wanted = self.switcher.update(h, tilt, speed, facing, pose_err)
            if wanted != self.active:
                self._prev_target = self._blended_target().copy()
                self.active = wanted
                self._blend = 0.0
                print(f"getup stage -> {wanted} (h={h:.2f} tilt={tilt:.2f} "
                       f"speed={speed:.2f} facing={facing:+.2f} "
                       f"pose_err={pose_err:.2f})")
                policy = self._active_policy()
            # Idle without a balance policy: hold the nominal pose.
            if self.active != IDLE or self.stand is not None:
                action = policy.update_policy()
            else:
                action = None
        else:
            action = None
        if self._blend < 1.0:
            n = max(1, int(round(self.blend_s / self.sim_dt)))
            self._blend = min(1.0, self._blend + 1.0 / n)
            policy.target_pos = self._blended_target()
        policy.apply_pd()
        self._sim_steps += 1
        return action

    def _blended_target(self):
        a = self._blend
        return ((1.0 - a) * self._prev_target
                + a * self._active_policy().target_pos).astype(np.float32)

    def apply_pd(self):
        self._active_policy().apply_pd()

    def telemetry(self):
        h, tilt, _ = self._height_tilt_facing()
        return h, tilt, self.user_command, self.active


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
              standstill=True, stand_policy_path=None, mode="walk"):
    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)

    if mode in ("stand", "auto"):
        if stand_policy_path is None:
            stand_policy_path = find_latest_stand_policy()
        if stand_policy_path is None:
            raise FileNotFoundError(
                "no stand policy: pass --stand-policy or train one "
                "(`g1 train-stand`)")
        bridge = WalkStandBridge(mj_model, mj_data, policy_path,
                                 stand_policy_path, sim_dt=sim_dt, mode=mode,
                                 standstill=standstill)
    else:
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


def reset_fallen(mj_model, mj_data, default_pos, muj_q=None, seed=0,
                 z_range=(0.06, 0.30)):
    """Random sprawl matching the Reposition training reset (seeded).

    muj_q maps policy joint order -> offsets inside qpos[7:] (see
    StandStillPolicy); None assumes the orders already match.
    """
    import math

    rng = np.random.default_rng(seed)
    mj_data.qpos[0:2] = rng.uniform(-0.3, 0.3, 2)
    mj_data.qpos[2] = float(rng.uniform(*z_range))
    mj_data.qpos[3:7] = euler_to_quat(
        float(rng.uniform(-math.pi, math.pi)),
        float(rng.uniform(-math.pi / 2, math.pi / 2)),
        float(rng.uniform(-math.pi, math.pi)))
    if muj_q is None:
        muj_q = np.arange(len(default_pos))
    mj_data.qpos[7:][muj_q] = (
        np.asarray(default_pos, dtype=np.float64)
        + rng.uniform(-0.6, 0.6, len(default_pos)))
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


def run_recover(stage_policies=None, scene=DEFAULT_SCENE, seconds=15.0,
                sim_dt=0.005, headless=False, log_hz=2.0,
                stand_policy_path=None, start="fallen", seed=0):
    """Staged get-up rollout (viewer or headless) from a fallen start.

    stage_policies maps "roll"/"getup" to ONNX paths; None resolves each with
    `find_latest_recovery_policy`. Hands off to the balance policy at DONE when
    one is available. Returns True when the robot ends standing.
    """
    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)

    if stage_policies is None:
        stage_policies = {}
        for s in STAGE_SEQUENCE:
            path = find_latest_recovery_policy(s)
            if path is None:
                train_stage = "roll" if s == ROLL else "standup"
                raise FileNotFoundError(
                    f"no {s} policy: train it with "
                    f"`g1 train -- --task getup --stage {train_stage}` or pass "
                    f"--policy-{s}")
            stage_policies[s] = path
    if stand_policy_path is None:
        stand_policy_path = find_latest_stand_policy()

    bridge = GetUpBridge(mj_model, mj_data, stage_policies, sim_dt=sim_dt,
                         stand_policy_path=stand_policy_path)
    if start == "fallen":
        reset_fallen(mj_model, mj_data, bridge.default_pos,
                     muj_q=bridge.stage_policies[ROLL].muj_q, seed=seed)
    else:
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
    for step in range(n_steps):
        mujoco.mj_step(mj_model, mj_data)
        bridge.step_sim()
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
        if not np.all(np.isfinite(mj_data.qpos)):
            print(f"simulation diverged at t={step * sim_dt:.2f}s")
            if viewer is not None:
                viewer.close()
            return False
        if step % log_every == 0:
            h, tilt, _, state = bridge.telemetry()
            print(f"t={step * sim_dt:5.2f}s state={state:4s} h={h:.3f} tilt={tilt:.3f}")
        if not headless and viewer is not None:
            time_until = sim_dt - (time.perf_counter() - t0 - step * sim_dt)
            if time_until > 0:
                time.sleep(time_until)
    h, tilt, _, state = bridge.telemetry()
    # Same definition of "standing" as the Rise handoff gate (C).
    stood = bool(h > 0.72 and tilt < 0.30)
    print(f"done: state={state} final_h={h:.3f} tilt={tilt:.3f} stood={stood}")
    if viewer is not None:
        viewer.close()
    return stood
