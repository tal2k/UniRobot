# How it works (policy bridge)

Classic learned-locomotion stack: ONNX policy proposes joint targets at
50 Hz, PD spring-dampers realize them every sim step (200 Hz).

- Observation (98): ang-vel 3 + projected gravity 3 + command 3 +
  gait sin/cos 2 + (q - q0) 29 + qvel 29 + last action 29.
- Action (29): target = q0 + action * scale; torque = kp*(target-q) - kd*v,
  clamped to actuator limits.
- Standstill (default): `walk` (pass command) → `hold` (anchor + ≤0.3 m/s
  correction, ~2 cm drift) → `frozen` (phase held, ~1 cm drift).
  Disturbance >0.30 m/s returns to `hold`. Use `set_command()`.
  `--no-standstill` restores raw march-in-place.

Fixed bugs worth knowing: gravity sign was inverted; PD ran at policy rate.
Both fixed in `core/bridge.py`. Telemetry for GUIs: `bridge.telemetry()`.

# Walk/stand switching (trained balance policy)

`core/bridge.py` `StandStillPolicy` runs the 94-dim policy from
`g1 train-stand`: gyro 3 + gravity 3 + height 1 + (q-q0) 29 + qvel 29 +
last action 29. No command, no gait phase. Gains/pose/joint order come
from the ONNX metadata; joints map by name.

`WalkStandBridge` holds both policies: `--mode walk` (legacy velocity
only), `--mode stand` (balance only), `--mode auto` (nonzero command
walks, zero command balances — no anchor needed). Switches cross-fade
joint targets over ~0.4 s. `--stand-policy` overrides the path; default
is the latest `g1_stand` snapshot (`find_latest_stand_policy()`).

# Staged recovery (trained Roll + GetUp policies)

`GetUpBridge` sequences ROLL (any fall → supine) → GETUP (supine → stand)
→ DONE (balance handoff), all 94-dim like the stand policy. Switching is
decided by `core/getup_stages.py::StageSwitcher` from height/tilt/speed +
`facing` (body-x gravity: supine ≈ −1, prone ≈ +1, yaw-invariant), with
hysteresis, timeouts and 0.4 s cross-fades. Run it with `g1 recover`, watch
it with `g1 gui --mode recover` (Roll/GetUp/Auto/Drop buttons). Full spec:
`getup_staged_policies.md` §14.
