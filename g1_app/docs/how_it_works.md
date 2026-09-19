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
