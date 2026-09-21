"""Stage gates + switching state machine for the staged get-up.

Pure-stdlib logic shared by training (re-exported by
`training/getup/curriculum.py`) and deployment
(`core/bridge.py::GetUpBridge`). It lives under `core/` so the runtime sim
never has to import the training package.

Stage metrics, sampled once per policy tick (50 Hz):

* ``height``    -- root (pelvis) height: ~0.10 lying, ~0.78 standing
* ``tilt``      -- |projected gravity xy|: 0 upright, ~1 torso horizontal
* ``speed``     -- |root xy velocity| [m/s]
* ``extension`` -- mean |q - q0| over shoulder/elbow/knee joints (limbs clear)
* ``pose_err``  -- RMS joint error to the supine neutral target (Stage A gate)

Sequence: Reposition (A) -> SitUp (B) -> Rise (C) -> DONE (stand handoff).
The A advance gate mirrors the training reward exactly
(``supine_success``: height < 0.35, tilt > 0.55, pose_err < 0.25) so a
policy that earns the reward also triggers the deployment switch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
  "A", "A_GATE_JOINT_PATTERNS", "B", "C", "DONE", "FALLEN_HEIGHT",
  "FALLEN_TILT", "IDLE", "MIN_TICKS", "REVERT_GATES", "STAGE_GATES",
  "STAGE_SEQUENCE", "STAGE_TIMEOUTS", "SUPINE_TARGET", "StageGate",
  "StageSwitcher", "gate_joint_indices", "is_fallen", "mean_abs_deviation",
  "rms_pose_error",
]

IDLE = "idle"
A = "A"  # Reposition: sprawled -> canonical lying, limbs clear
B = "B"  # SitUp: lying -> crouch
C = "C"  # Rise: crouch -> standing
DONE = "done"

STAGE_SEQUENCE = (A, B, C)
_NEXT = {A: B, B: C, C: DONE}

# Engagement: below this height or above this tilt the robot is not standing
# (a crouched robot at h=0.5 still counts as fallen and gets recovered).
FALLEN_HEIGHT = 0.55
FALLEN_TILT = 0.60

# Consecutive satisfied policy ticks before a switch (0.2 s at 50 Hz).
MIN_TICKS = 10

# Limbs-clear gate: joints averaged for the mean |q - q0| extension metric.
A_GATE_JOINT_PATTERNS = ("shoulder", "elbow", "knee")

# Supine neutral target (Stage A goal): lying flat on back, legs extended,
# arms at sides. Canonical home of the constant — training
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
  min_extension: float | None = None
  max_pose_err: float | None = None

  def satisfied(self, height: float, tilt: float, speed: float = 0.0,
                extension: float = 0.0, pose_err: float = 0.0) -> bool:
    return (
      (self.min_height is None or height > self.min_height)
      and (self.max_height is None or height < self.max_height)
      and (self.min_tilt is None or tilt > self.min_tilt)
      and (self.max_tilt is None or tilt < self.max_tilt)
      and (self.max_speed is None or speed < self.max_speed)
      and (self.min_extension is None or extension > self.min_extension)
      and (self.max_pose_err is None or pose_err < self.max_pose_err)
    )


# Advance gates. A ends in the canonical supine pose the SitUp stage starts
# from — the gate mirrors the training `supine_success` reward exactly
# (max_height 0.35, min_tilt 0.55, max_pose_err 0.25). B ends crouched, C
# ends standing and slow enough for the balance handoff.
STAGE_GATES = {
  A: StageGate(max_height=0.35, min_tilt=0.55, max_pose_err=0.25),
  B: StageGate(min_height=0.55, max_tilt=0.70),
  C: StageGate(min_height=0.72, max_tilt=0.30, max_speed=0.12),
}

# Regressions that send the sequence back one stage (held MIN_TICKS ticks).
# Only C -> B: a rise that sinks back below the crouch is re-attempted.
# A/B have no revert gates because their own start states satisfy them.
REVERT_GATES = {
  C: (B, StageGate(max_height=0.45)),
}

# Per-stage budget [s]; on timeout the sequence restarts at Reposition.
STAGE_TIMEOUTS = {A: 8.0, B: 6.0, C: 5.0}


def is_fallen(height: float, tilt: float) -> bool:
  """Engagement condition: not standing and not merely crouched."""
  return height < FALLEN_HEIGHT or tilt > FALLEN_TILT


def mean_abs_deviation(q, q0, indices) -> float:
  """Mean |q - q0| over ``indices`` (pure Python; ~12 joints at 50 Hz)."""
  if not indices:
    return 0.0
  return sum(abs(float(q[i]) - float(q0[i])) for i in indices) / len(indices)


def rms_pose_error(q, q0) -> float:
  """RMS joint error to a target pose (matches the training reward).

  Same quantity ``supine_success`` thresholds at ``max_pose_err``: with the
  all-zero ``SUPINE_TARGET`` this is simply the RMS joint angle.
  """
  n = len(q0)
  if n == 0:
    return 0.0
  return math.sqrt(sum((float(q[i]) - float(q0[i])) ** 2 for i in range(n)) / n)


def gate_joint_indices(joint_names) -> list[int]:
  """Indices of shoulder/elbow/knee joints in the policy joint order."""
  return [i for i, n in enumerate(joint_names)
          if any(p in n for p in A_GATE_JOINT_PATTERNS)]


class StageSwitcher:
  """Hysteretic, timeout-guarded stage sequencer.

  ``update`` is called once per policy tick with the current metrics and
  returns the stage the bridge should run. It engages Reposition when the
  robot is fallen, advances after ``min_ticks`` consecutive gate hits,
  reverts C -> B when the crouch is lost, restarts the sequence at
  Reposition when a stage budget runs out, and jumps straight to DONE when
  the robot already satisfies the Rise handoff gate while in A/B (standing
  shortcut — the lying A gate could never fire then).
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
    """Re-arm: the next update engages Reposition if the robot is fallen."""
    self.state = IDLE
    self._hold = 0
    self._revert_hold = 0
    self._elapsed = 0.0

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
             extension: float = 0.0, pose_err: float = 0.0) -> str:
    if self.state in (IDLE, DONE):
      if is_fallen(height, tilt):
        self._enter(A)
      return self.state

    self._elapsed += self.dt
    if self._elapsed >= self.timeouts.get(self.state, math.inf):
      self._enter(A)  # budget exhausted: restart the sequence
      return self.state

    if self.state in (A, B):
      # Standing shortcut: the robot already satisfies the Rise handoff gate
      # (tall, upright, slow) — e.g. it stood up during Reposition, or a
      # timeout restarted a standing robot at A. Without this the switcher
      # deadlocks: the A gate (lying) can never fire while standing, so the
      # timeout would loop in A forever driving a standing robot with the
      # lying-stage policy.
      handoff = self.gates.get(C)
      if (handoff is not None
              and handoff.satisfied(height, tilt, speed, extension, pose_err)):
        self._enter(DONE)
        return self.state

    revert = self.revert_gates.get(self.state)
    if revert is not None:
      prev, gate = revert
      if gate.satisfied(height, tilt, speed, extension, pose_err):
        self._revert_hold += 1
        if self._revert_hold >= self.min_ticks:
          self._enter(prev)
          return self.state
      else:
        self._revert_hold = 0

    gate = self.gates.get(self.state)
    if gate is not None and gate.satisfied(height, tilt, speed, extension, pose_err):
      self._hold += 1
      if self._hold >= self.min_ticks:
        self._enter(_NEXT[self.state])
    else:
      self._hold = 0
    return self.state
