"""Visible G1 balanced-stand demo: simulator + pretrained ONNX policy.

Canonical home (moved from legacy run_g1_stand.py).
Prefer the unified CLI:

    g1 stand --seconds 60
    .venv/bin/python -m g1_app.cli stand --seconds 60
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Robust imports: work from workspace root (g1_app.*) and from g1_app/ (core.*).
try:
    from g1_app.core.bridge import DEFAULT_LOCAL_POLICY, G1StandPolicy, reset_standing
    from g1_app.core.terrains import TERRAINS, resolve_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.bridge import DEFAULT_LOCAL_POLICY, G1StandPolicy, reset_standing
        from core.terrains import TERRAINS, resolve_scene
    except ImportError:  # legacy flat shim on path
        from g1_stand_onnx import DEFAULT_LOCAL_POLICY, TERRAINS, G1StandPolicy, reset_standing

        def resolve_scene(terrain="flat", scene=None):
            return scene or TERRAINS[terrain]


def main():
    ap = argparse.ArgumentParser(description="Show G1 standing balanced in MuJoCo viewer")
    ap.add_argument("--policy", default=DEFAULT_LOCAL_POLICY)
    ap.add_argument("--scene", default=None, help="MuJoCo scene XML (overrides --terrain)")
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="flat")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="0 = run until window closed")
    ap.add_argument("--sim-dt", type=float, default=0.005)
    args = ap.parse_args()
    args.scene = resolve_scene(args.terrain, args.scene)

    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_model.opt.timestep = args.sim_dt
    mj_data = mujoco.MjData(mj_model)

    bridge = G1StandPolicy(mj_model, mj_data, args.policy, sim_dt=args.sim_dt)
    reset_standing(mj_model, mj_data, bridge.default_pos, height=0.78)

    print(f"Policy: {args.policy} (mode={bridge.mode})")
    print(f"Scene:  {args.scene}")
    print("Opening MuJoCo viewer — close the window to stop.")

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    # Frame the robot nicely.
    viewer.cam.azimuth = -130
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.6]

    run_forever = args.seconds <= 0
    n_steps = int(args.seconds / args.sim_dt) if not run_forever else None
    step = 0
    t_wall = time.perf_counter()
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        bridge.step_sim()
        viewer.sync()

        if step % 400 == 0:
            h = float(mj_data.qpos[2])
            print(f"t={step * args.sim_dt:6.2f}s height={h:.3f} "
                  f"act[:4]={np.round(bridge.last_action[:4], 3)}")
        step += 1
        if n_steps is not None and step >= n_steps:
            print("Time limit reached, closing.")
            break
        # Pace to real time.
        elapsed = time.perf_counter() - t_wall
        target = step * args.sim_dt
        if target > elapsed:
            time.sleep(target - elapsed)

    viewer.close()
    print("Done.")


if __name__ == "__main__":
    main()
