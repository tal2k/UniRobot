"""Stage gates + switching state machine for the staged get-up (v2).

Pure-stdlib logic shared by training (re-exported by
`training/getup/curriculum.py`) and deployment
(`core/bridge.py::GetUpBridge`). It lives under `core/` so the runtime sim
never has to import the training package.

v2 design (HumanUP-style split by start family, supine as the funnel):
ROLL (any fall -> supine) -> GETUP (supine -> stand) -> DONE (stand handoff).
One recovery switch instead of three; the merged GetUp policy covers the old
SitUp+Rise phases in a single run with height-banded rewards.

Stage metrics, sampled once per policy tick (50 Hz):

* ``height``   -- root (pelvis) height: ~0.10 lying, ~0.78 standing
* ``tilt``     -- |projected gravity xy|: 0 upright, ~1 torso horizontal
* ``speed``    -- |root xy velocity| [m/s]
* ``facing``   -- body-x projected gravity: supine ~-1, prone ~+1, side ~0.
  Yaw-invariant (yaw rotations never move the gravity vector), so this
  distinguishes face-up from face-down from the IMU alone. Assumes the
  G1 x-forward torso convention — verify the sign in sim (print gravity
  while supine vs prone) if the model ever changes.
* ``pose_err`` -- RMS joint error to the supine neutral target (keeps the
  ROLL handoff inside the GETUP policy's start distribution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
  "DONE", "FACE_UP_GRAVITY", "FALLEN_HEIGHT", "GETUP", "GETUP_GATE",
  "IDLE", "MIN_TICKS", "REVERT_GATES", "ROLL", "ROLL_GATE",
  "ROLL_TIMEOUT", "GETUP_TIMEOUT", "STAGE_GATES", "STAGE_SEQUENCE",
  "STAGE_TIMEOUTS", "SUPINE_FACING", "SUPINE_TARGET", "StageGate",
  "StageSwitcher", "rms_pose_error",
]

IDLE = "idle"
ROLL = "roll"    # any fall -> canonical supine (funnel stage)
GETUP = "getup"  # supine -> crouch -> standing (merged SitUp+Rise)
DONE = "done"

STAGE_SEQUENCE = (ROLL, GETUP)
_NEXT = {ROLL: GETUP, GETUP: DONE}

# Engagement: below this height the robot is truly down (a tall-but-tilted
# robot stays with the balance policy, not recovery).
FALLEN_HEIGHT = 0.55

# Facing thresholds (body-x projected gravity).
SUPINE_FACING = -0.5  # below: confidently face-up -> GETUP can take over
REVERT_FACING = 0.0   # above, sustained: lost supine -> re-roll

# Face-up gravity target (exact flat supine after pitching back).
FACE_UP_GRAVITY = (-1.0, 0.0, 0.0)

# Consecutive satisfied policy ticks before a switch (0.2 s at 50 Hz).
MIN_TICKS = 10

# Per-stage budget [s]; on timeout the sequence re-rolls (ROLL handles any
# pose, so there is no dead-end restart state).
ROLL_TIMEOUT = 8.0
GETUP_TIMEOUT = 12.0

# Supine neutral target (ROLL goal / GETUP start): lying flat on back, legs
# extended, arms at sides. Canonical home of the constant — training
# (`training/getup/mdp/events.py`, stage cfgs) and the recorder import it
# from here so the deployment gate and the training reward can never drift
# apart. All zeros, hence independent of joint order.
# Order: 6 L leg, 6 R leg, 3 waist, 7 L arm, 7 R arm (G1 29-DoF).
SUPINE_TARGET = [0.0] * 29


@dataclass(frozen=True)
class StageGate:
  """Thresholds a metric vector must satisfy to advance (None = ignore)."""

  min_height: float | None = None
  max_height: float | None = None
  min_tilt: float | None = None
  max_tilt: float | None = None
  max_speed: float | None = None
  max_facing: float | None = None
  min_facing: float | None = None
  max_pose_err: float | None = None

  def satisfied(self, height: float, tilt: float, speed: float = 0.0,
                facing: float = 0.0, pose_err: float = 0.0) -> bool:
    return (
      (self.min_height is None or height > self.min_height)
      and (self.max_height is None or height < self.max_height)
      and (self.min_tilt is None or tilt > self.min_tilt)
      and (self.max_tilt is None or tilt < self.max_tilt)
      and (self.max_speed is None or speed < self.max_speed)
      and (self.max_facing is None or facing < self.max_facing)
      and (self.min_facing is None or facing > self.min_facing)
      and (self.max_pose_err is None or pose_err < self.max_pose_err)
    )


# ROLL ends in canonical supine: face-up, low, slow, joints near the target
# the GETUP policy trains from (pose_err keeps the handoff inside GETUP's
# start distribution). GETUP ends standing and slow (balance handoff).
ROLL_GATE = StageGate(max_height=0.35, max_speed=0.15,
                      max_facing=SUPINE_FACING, max_pose_err=0.30)
GETUP_GATE = StageGate(min_height=0.72, max_tilt=0.30, max_speed=0.12)

STAGE_GATES = {ROLL: ROLL_GATE, GETUP: GETUP_GATE}

# Regressions that send GETUP back to ROLL (held MIN_TICKS ticks): rolled
# back to prone/side mid-rise. ROLL has no revert — any pose is its domain.
REVERT_GATES = {
  GETUP: (ROLL, StageGate(min_facing=REVERT_FACING)),
}

STAGE_TIMEOUTS = {ROLL: ROLL_TIMEOUT, GETUP: GETUP_TIMEOUT}


def rms_pose_error(q, q0) -> float:
  """RMS joint error to a target pose (matches the training reward).

  Same quantity the ROLL gate thresholds at ``max_pose_err``: with the
  all-zero ``SUPINE_TARGET`` this is simply the RMS joint angle.
  """
  n = len(q0)
  if n == 0:
    return 0.0
  return math.sqrt(sum((float(q[i]) - float(q0[i])) ** 2 for i in range(n)) / n)


class StageSwitcher:
  """Hysteretic, timeout-guarded stage sequencer (v2: ROLL -> GETUP -> DONE).

  ``update`` is called once per policy tick with the current metrics and
  returns the stage the bridge should run. Truly-down robots engage ROLL,
  unless already supine (skips straight to GETUP); stages advance after
  ``min_ticks`` consecutive gate hits; GETUP reverts to ROLL when supine is
  lost; budgets re-roll instead of dead-ending; and an already-standing
  robot jumps straight to DONE (the lying gates could never fire).
  """

  def __init__(self, gates=None, revert_gates=None, timeouts=None,
               min_ticks: int = MIN_TICKS, dt: float = 0.02):
    self.gates = dict(STAGE_GATES if gates is None else gates)
    self.revert_gates = dict(REVERT_GATES if revert_gates is None else revert_gates)
    self.timeouts = dict(STAGE_TIMEOUTS if timeouts is None else timeouts)
    self.min_ticks = int(min_ticks)
    self.dt = float(dt)
    self.state = IDLE
    self._hold = 0
    self._revert_hold = 0
    self._elapsed = 0.0

  def reset(self) -> None:
    """Re-arm: the next update engages ROLL/GETUP if the robot is down."""
    self.state = IDLE
    self._hold = 0
    self._revert_hold = 0
    self._elapsed = 0.0

  def force(self, state: str) -> None:
    """Manually enter a stage (GUI override); AUTO resumes via reset()."""
    assert state in (IDLE, ROLL, GETUP, DONE), f"unknown stage {state!r}"
    self._enter(state)

  @property
  def elapsed(self) -> float:
    """Seconds spent in the current stage (0 while idle/done)."""
    return self._elapsed

  def _enter(self, state: str) -> None:
    self.state = state
    self._hold = 0
    self._revert_hold = 0
    self._elapsed = 0.0

  def update(self, height: float, tilt: float, speed: float = 0.0,
             facing: float = 0.0, pose_err: float = 0.0) -> str:
    if self.state in (IDLE, DONE):
      if height < FALLEN_HEIGHT:
        # Already supine? Skip the roll, go straight to get-up.
        self._enter(GETUP if facing < SUPINE_FACING else ROLL)
      return self.state

    self._elapsed += self.dt
    if self._elapsed >= self.timeouts.get(self.state, math.inf):
      self._enter(ROLL)  # budget exhausted: re-roll (handles any pose)
      return self.state

    if self.state in STAGE_SEQUENCE:
      # Standing shortcut: already in the handoff state while recovering
      # (e.g. stood up during ROLL) — jump to DONE, the lying gates could
      # never fire while standing.
      handoff = self.gates.get(GETUP)
      if (handoff is not None
              and handoff.satisfied(height, tilt, speed, facing, pose_err)):
        self._enter(DONE)
        return self.state

    revert = self.revert_gates.get(self.state)
    if revert is not None:
      prev, gate = revert
      if gate.satisfied(height, tilt, speed, facing, pose_err):
        self._revert_hold += 1
        if self._revert_hold >= self.min_ticks:
          self._enter(prev)
          return self.state
      else:
        self._revert_hold = 0

    gate = self.gates.get(self.state)
    if gate is not None and gate.satisfied(height, tilt, speed, facing, pose_err):
      self._hold += 1
      if self._hold >= self.min_ticks:
        self._enter(_NEXT[self.state])
    else:
      self._hold = 0
    return self.state
