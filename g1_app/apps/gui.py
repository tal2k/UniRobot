"""G1 interactive control: MuJoCo 3D view + tkinter control panel.

Canonical home (moved from legacy g1_gui_control.py).
Prefer the unified CLI (needs a display):

    g1 gui
    g1 gui --terrain slope
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    from g1_app.core.bridge import DEFAULT_LOCAL_POLICY, G1StandPolicy, reset_standing
    from g1_app.core.terrains import TERRAINS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.bridge import DEFAULT_LOCAL_POLICY, G1StandPolicy, reset_standing
        from core.terrains import TERRAINS
    except ImportError:
        from g1_stand_onnx import DEFAULT_LOCAL_POLICY, TERRAINS, G1StandPolicy, reset_standing

import tkinter as tk

SIM_DT = 0.005
CMD_RANGES = {"vx": (-0.5, 1.0), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0)}


class G1Gui:
    def __init__(self, policy=DEFAULT_LOCAL_POLICY, scene=None, terrain="flat"):
        self.policy_path = policy
        self.viewer = None
        self._load_scene(scene or TERRAINS.get(terrain, TERRAINS["flat"]))

        self.sim_steps = 0
        self.t_wall = time.perf_counter()
        self.fell = False
        self._build_panel()
        self._tick()

    def _load_scene(self, scene_path):
        """(Re)build simulator + policy on a new scene file."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_model.opt.timestep = SIM_DT
        self.mj_data = mujoco.MjData(self.mj_model)
        self.bridge = G1StandPolicy(self.mj_model, self.mj_data, self.policy_path,
                                    sim_dt=SIM_DT)
        self.bridge.standstill = getattr(self, "standstill_var", None) is None or \
            self.standstill_var.get()
        reset_standing(self.mj_model, self.mj_data, self.bridge.default_pos, height=0.78)
        self.scene_path = scene_path

        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer.cam.azimuth = -130
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 3.5
        self.viewer.cam.lookat[:] = [0, 0, 0.6]
        self.sim_steps = 0
        self.t_wall = time.perf_counter()

    # -- panel -----------------------------------------------------------
    def _build_panel(self):
        self.root = tk.Tk()
        self.root.title("G1 Control")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        # Make sure the panel is visible above the viewer on launch.
        self.root.geometry("+40+40")
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(3000, lambda: self.root.attributes("-topmost", False))

        self.sliders = {}
        for key, label in (("vx", "Forward (m/s)"), ("vy", "Sideways (m/s)"), ("wz", "Turn (rad/s)")):
            lo, hi = CMD_RANGES[key]
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, width=14, anchor="w").pack(side="left")
            s = tk.Scale(row, from_=lo, to=hi, resolution=0.05, orient="horizontal",
                         length=220, command=lambda _v, k=key: self._push_cmd())
            s.set(0.0)
            s.pack(side="left")
            self.sliders[key] = s

        btns = tk.Frame(self.root)
        btns.pack(pady=6)
        for text, cmd in (("Stand", (0, 0, 0)), ("Walk", (0.5, 0, 0)),
                          ("Turn left", (0, 0, 0.5)), ("Stop", (0, 0, 0))):
            tk.Button(btns, text=text, width=9,
                      command=lambda c=cmd: self._preset(c)).pack(side="left", padx=2)
        tk.Button(btns, text="Reset", width=7, command=self._reset).pack(side="left", padx=2)
        tk.Button(btns, text="Quit", width=7, command=self._quit).pack(side="left", padx=2)

        self.standstill_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Stand still (lock position on zero command)",
                       variable=self.standstill_var,
                       command=lambda: setattr(self.bridge, "standstill",
                                               self.standstill_var.get())).pack(pady=(0, 4))

        trow = tk.Frame(self.root)
        trow.pack(fill="x", padx=8, pady=2)
        tk.Label(trow, text="Terrain", width=14, anchor="w").pack(side="left")
        self.terrain_var = tk.StringVar(value="flat")
        tk.OptionMenu(trow, self.terrain_var, *sorted(TERRAINS)).pack(side="left")
        tk.Button(trow, text="Load",
                  command=lambda: self._load_scene(TERRAINS[self.terrain_var.get()])
                  ).pack(side="left", padx=6)

        self.status = tk.Label(self.root, text="starting…", anchor="w", justify="left",
                               font=("monospace", 10))
        self.status.pack(fill="x", padx=8, pady=4)
        tk.Label(self.root, text="Keys: arrows = move, A/D = turn, Space = stand",
                 fg="gray").pack(pady=(0, 6))

        self.root.bind("<Up>", lambda _e: self._nudge("vx", +0.1))
        self.root.bind("<Down>", lambda _e: self._nudge("vx", -0.1))
        self.root.bind("<Left>", lambda _e: self._nudge("vy", +0.1))
        self.root.bind("<Right>", lambda _e: self._nudge("vy", -0.1))
        self.root.bind("a", lambda _e: self._nudge("wz", +0.1))
        self.root.bind("A", lambda _e: self._nudge("wz", +0.1))
        self.root.bind("d", lambda _e: self._nudge("wz", -0.1))
        self.root.bind("D", lambda _e: self._nudge("wz", -0.1))
        self.root.bind("<space>", lambda _e: self._preset((0, 0, 0)))

    # -- command helpers ---------------------------------------------------
    def _push_cmd(self):
        self.bridge.set_command(self.sliders["vx"].get(),
                                self.sliders["vy"].get(),
                                self.sliders["wz"].get())

    def _preset(self, cmd):
        for k, v in zip(("vx", "vy", "wz"), cmd):
            self.sliders[k].set(v)
        self._push_cmd()

    def _nudge(self, key, delta):
        lo, hi = CMD_RANGES[key]
        self.sliders[key].set(min(hi, max(lo, self.sliders[key].get() + delta)))
        self._push_cmd()

    def _reset(self):
        reset_standing(self.mj_model, self.mj_data, self.bridge.default_pos, height=0.78)
        self.bridge.phase = 0.0
        self.fell = False

    def _quit(self):
        try:
            self.viewer.close()
        except Exception:
            pass
        self.root.destroy()

    # -- sim loop (tkinter timer keeps everything in one thread) ------------
    def _tick(self):
        if not self.viewer.is_running():
            self._quit()
            return
        # Advance sim to real time (10 ms per tick).
        target_steps = int((time.perf_counter() - self.t_wall) / SIM_DT)
        for _ in range(min(max(1, target_steps - self.sim_steps), 20)):
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.bridge.step_sim()
            self.sim_steps += 1
        self.viewer.sync()

        h, tilt, cmd, stand_state = self.bridge.telemetry()
        if h < 0.4 or tilt > 0.7:
            state = "FELL — press Reset"
        elif not np.allclose(cmd, 0):
            state = "WALKING"
        else:
            state = {"hold": "STANDING (settling…)", "frozen": "STANDING STILL",
                     "walk": "STANDING"}.get(stand_state, "STANDING")
        self.fell = state.startswith("FELL")
        self.status.config(
            text=f"t={self.sim_steps * SIM_DT:6.1f}s  height={h:.2f}m  tilt={tilt:.2f}\n"
                 f"cmd=[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]  {state}")
        self.root.after(10, self._tick)

    def run(self):
        print(f"Policy: {self.bridge.mode}, sliders set walk command in m/s.")
        print("Two windows should now be visible:")
        print("  1. MuJoCo viewer (3D robot)")
        print("  2. 'G1 Control' panel (sliders/buttons) — forced on top for 3 s;")
        print("     if you lose it, look behind the viewer or in the taskbar/dock.")
        self.root.mainloop()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="G1 interactive GUI control")
    ap.add_argument("--policy", default=DEFAULT_LOCAL_POLICY)
    ap.add_argument("--scene", default=None, help="MuJoCo scene XML (overrides --terrain)")
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="flat")
    G1Gui(**vars(ap.parse_args())).run()
