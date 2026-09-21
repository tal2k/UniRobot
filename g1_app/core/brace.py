"""Independent fall-brace: catch the body when a fall is unrecoverable.

This module has NO connection to the get-up procedure (`getup_stages.py` /
`GetUpBridge`): no shared state, no transitions into ROLL/GETUP, no shared
gates. It is a standalone guard + policy runner:

  NOMINAL (stand policy or passive hold) --[fall_trigger]--> BRACING
      --[touchdown + settle / timeout]--> HOLD (passive, stays until reset)

The fall trigger is the analytic v1 "point of no return": the robot is high
and either tilted past ~40 deg, dropping fast, or spinning. Calm standing
sway never fires it. A learned predictor (SafeFall-style GRU) can replace
``fall_trigger`` later without touching the guard.

The brace policy itself trains independently:
`g1 train -- --task getup --stage brace`.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from .bridge import DEFAULT_SCENE, StandStillPolicy

__all__ = [
  "BRACE_ANGVEL",
  "BRACE_ARM_HEIGHT",
  "BRACE_MIN_TICKS",
  "BRACE_SETTLE_V",
  "BRACE_TIMEOUT",
  "BRACE_TILT",
  "BRACE_TOUCHDOWN_H",
  "BRACE_VZ",
  "BRACING",
  "HOLD",
  "NOMINAL",
  "BraceBridge",
  "BraceGuard",
  "fall_trigger",
  "run_brace",
  "topple_robot",
]

NOMINAL = "nominal"
BRACING = "bracing"
HOLD = "hold"

# Trigger: high + (tilted | dropping | spinning). Conservative on purpose —
# normal standing sway (tilt < 0.3, |vz| < 0.3) never fires it.
BRACE_ARM_HEIGHT = 0.55
BRACE_TILT = 0.65
BRACE_VZ = -1.0
BRACE_ANGVEL = 2.0

# Release: down + settling (touchdown), held, or budget exhausted.
BRACE_TOUCHDOWN_H = 0.45
BRACE_SETTLE_V = 0.30
BRACE_TIMEOUT = 3.0
BRACE_MIN_TICKS = 10  # consecutive 50 Hz ticks before release (0.2 s)

# Stumbled back to a standing hold while bracing -> nominal takes over.
BRACE_STAND_H = 0.70
BRACE_STAND_TILT = 0.30


def fall_trigger(height: float, tilt: float, vz: float = 0.0,
                 angvel: float = 0.0) -> bool:
  """True when a high robot is falling beyond recovery (engage BRACE)."""
  return (
    height > BRACE_ARM_HEIGHT
    and (tilt > BRACE_TILT or vz < BRACE_VZ or angvel > BRACE_ANGVEL)
  )


class BraceGuard:
  """Standalone brace state machine: NOMINAL -> BRACING -> HOLD.

  Pure logic (no mujoco), unit-testable. ``update`` runs once per policy
  tick; HOLD latches until ``reset()`` re-arms — a braced landing stays put
  for inspection instead of chaining into anything else.
  """

  def __init__(self, min_ticks: int = BRACE_MIN_TICKS, dt: float = 0.02):
    self.min_ticks = int(min_ticks)
    self.dt = float(dt)
    self.state = NOMINAL
    self._hold = 0
    self._elapsed = 0.0

  def reset(self) -> None:
    """Re-arm to NOMINAL (clears the HOLD latch)."""
    self.state = NOMINAL
    self._hold = 0
    self._elapsed = 0.0

  @property
  def elapsed(self) -> float:
    return self._elapsed

  def update(self, height: float, tilt: float, speed: float = 0.0,
             vz: float = 0.0, angvel: float = 0.0) -> str:
    if self.state == NOMINAL:
      if fall_trigger(height, tilt, vz, angvel):
        self.state = BRACING
        self._hold = 0
        self._elapsed = 0.0
      return self.state
    if self.state == HOLD:
      return self.state
    # BRACING: release on touchdown + settle, on standing again, or timeout.
    self._elapsed += self.dt
    stood = (height > BRACE_STAND_H and tilt < BRACE_STAND_TILT
             and speed < BRACE_SETTLE_V)
    if stood:
      self.state = NOMINAL
      self._hold = 0
      self._elapsed = 0.0
      return self.state
    landed = height < BRACE_TOUCHDOWN_H and speed < BRACE_SETTLE_V
    self._hold = self._hold + 1 if landed else 0
    if self._hold >= self.min_ticks or self._elapsed >= BRACE_TIMEOUT:
      self.state = HOLD
      self._hold = 0
      self._elapsed = 0.0
    return self.state


def topple_robot(mj_model, mj_data, default_pos, seed=0,
                 kick: tuple[float, float] = (1.2, 2.2)):
  """Stand the robot up and shove it (brace demo start, seeded).

  A planar kick the balance policy cannot catch: the guard must trigger
  mid-fall and the brace policy must ride out the landing.
  """
  rng = np.random.default_rng(seed)
  mj_data.qpos[0:2] = rng.uniform(-0.05, 0.05, 2)
  mj_data.qpos[2] = 0.78
  mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  mj_data.qpos[7:7 + len(default_pos)] = default_pos
  heading = float(rng.uniform(-np.pi, np.pi))
  speed = float(rng.uniform(kick[0], kick[1]))
  mj_data.qvel[:] = 0.0
  mj_data.qvel[0] = speed * np.cos(heading)
  mj_data.qvel[1] = speed * np.sin(heading)
  mj_data.qvel[3:6] = rng.uniform(-0.5, 0.5, 3)
  mj_data.ctrl[:] = 0.0
  mujoco.mj_forward(mj_model, mj_data)


class BraceBridge:
  """Standalone brace runner: nominal control + guard + brace policy.

  Same execution idiom as the other bridges (PD every sim step, inference
  every 4th = 50 Hz, cross-faded policy swaps over blend_s) but a world of
  its own: NOMINAL (stand policy when given, else passive PD hold) ->
  BRACING (brace policy) -> HOLD (passive, until reset()). No get-up chain,
  no handoffs, no shared state with any other bridge.
  """

  def __init__(self, mj_model, mj_data, brace_policy_path, sim_dt=0.005,
               blend_s=0.4, stand_policy_path=None):
    self.mj_model = mj_model
    self.mj_data = mj_data
    self.brace = StandStillPolicy(mj_model, mj_data, brace_policy_path,
                                  sim_dt=sim_dt)
    self.nominal = None
    if stand_policy_path is not None:
      self.nominal = StandStillPolicy(mj_model, mj_data, stand_policy_path,
                                      sim_dt=sim_dt)
    self.guard = BraceGuard(dt=self.brace.STEP_DT)
    self.blend_s = blend_s
    self._blend = 1.0
    self._prev_target = self.brace.default_pos.copy()
    self.sim_dt = sim_dt
    self._sim_steps = 0
    self.default_pos = self.brace.default_pos.copy()
    self.user_command = np.zeros(3, dtype=np.float32)
    print("Mode: brace (nominal -> bracing -> hold)"
          + (", stand nominal" if self.nominal is not None
             else ", passive nominal"))

  @property
  def mode(self):
    return f"brace({self.guard.state})"

  @property
  def last_action(self):
    return self._active_policy().last_action

  def set_command(self, vx, vy, wz):
    # Stored for telemetry; brace ignores commands.
    self.user_command = np.array([vx, vy, wz], dtype=np.float32)

  def _active_policy(self):
    if self.guard.state == BRACING:
      return self.brace
    if self.nominal is not None:
      return self.nominal
    return self.brace

  def _metrics(self):
    """(height, tilt, planar speed, drop rate, angular speed)."""
    h = float(self.mj_data.qpos[2])
    _, grav, _, _ = self.brace._base_state()
    tilt = float(np.linalg.norm(grav[:2]))
    speed = float(np.linalg.norm(self.mj_data.qvel[0:2]))
    return h, tilt, speed, float(self.mj_data.qvel[2]), float(
      np.linalg.norm(self.mj_data.qvel[3:6]))

  def topple(self, seed=0):
    """Re-stand, shove, and re-arm the guard (demo reset)."""
    topple_robot(self.mj_model, self.mj_data, self.default_pos, seed=seed)
    self.guard.reset()

  def step_sim(self):
    policy = self._active_policy()
    if self._sim_steps % policy.decimation == 0:
      before = self.guard.state
      self.guard.update(*self._metrics())
      if self.guard.state != before:
        self._prev_target = self._blended_target().copy()
        print(f"brace -> {self.guard.state} "
              f"(h={self.mj_data.qpos[2]:.2f})")
        self._blend = 0.0
        policy = self._active_policy()
      # HOLD is passive (PD holds last targets); otherwise run inference.
      action = None if self.guard.state == HOLD else policy.update_policy()
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
    h, tilt, _, _, _ = self._metrics()
    return h, tilt, self.user_command, self.guard.state


def run_brace(brace_policy_path, scene=DEFAULT_SCENE, seconds=8.0,
              sim_dt=0.005, headless=False, log_hz=5.0,
              stand_policy_path=None, seed=0):
  """Standalone brace rollout (viewer or headless) from a topple.

  No get-up chain involved: topple -> guard triggers -> brace rides the
  landing -> HOLD. Returns True when the guard engaged and settled to HOLD.
  """
  mj_model = mujoco.MjModel.from_xml_path(scene)
  mj_model.opt.timestep = sim_dt
  mj_data = mujoco.MjData(mj_model)

  bridge = BraceBridge(mj_model, mj_data, brace_policy_path, sim_dt=sim_dt,
                       stand_policy_path=stand_policy_path)
  bridge.topple(seed=seed)

  viewer = None
  if not headless:
    try:
      viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    except Exception as e:
      print(f"viewer unavailable ({e}), continuing headless")
      headless = True

  engaged = False
  n_steps = int(seconds / sim_dt)
  log_every = max(1, int((1.0 / log_hz) / sim_dt))
  t0 = time.perf_counter()
  for step in range(n_steps):
    mujoco.mj_step(mj_model, mj_data)
    bridge.step_sim()
    if bridge.guard.state == BRACING:
      engaged = True
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
      print(f"t={step * sim_dt:5.2f}s state={state:7s} h={h:.3f} tilt={tilt:.3f}")
    if not headless and viewer is not None:
      time_until = sim_dt - (time.perf_counter() - t0 - step * sim_dt)
      if time_until > 0:
        time.sleep(time_until)
  h, tilt, _, state = bridge.telemetry()
  ok = bool(engaged and state == HOLD)
  print(f"done: state={state} engaged={engaged} final_h={h:.3f} "
        f"tilt={tilt:.3f} ok={ok}")
  if viewer is not None:
    viewer.close()
  return ok
