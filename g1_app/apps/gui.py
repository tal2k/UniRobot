"""G1 interactive control: MuJoCo 3D view + tkinter control panel.

Canonical home (moved from legacy g1_gui_control.py).
Prefer the unified CLI (needs a display):

    g1 gui
    g1 gui --terrain slope
"""
import time
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np

from core.brace import BRACING, BraceBridge
from core.bridge import (
    DEFAULT_LOCAL_POLICY,
    G1StandPolicy,
    GetUpBridge,
    WalkStandBridge,
    find_latest_recovery_policy,
    find_latest_stand_policy,
    reset_fallen,
    reset_standing,
)
from core.getup_stages import GETUP, ROLL
from core.terrains import TERRAINS

SIM_DT = 0.005
CMD_RANGES = {"vx": (-0.5, 1.0), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0)}


class G1Gui:
    def __init__(self, policy=DEFAULT_LOCAL_POLICY, scene=None, terrain="flat",
                 stand_policy=None, mode="walk", policy_roll=None,
                 policy_standup=None, policy_brace=None, seed=0):
        self.policy_path = policy
        self.stand_policy_path = stand_policy or find_latest_stand_policy()
        self.policy_mode = mode
        self.policy_roll = policy_roll
        self.policy_standup = policy_standup
        self.policy_brace = policy_brace
        self._seed = seed
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
        if self.policy_mode == "brace" and BraceBridge is not None:
            # Independent fall-brace (no get-up chain): stand + shove, the
            # guard auto-engages the brace policy mid-fall, then holds.
            path = self.policy_brace or find_latest_recovery_policy("brace")
            if path is None:
                raise FileNotFoundError(
                    "no brace policy: train it with "
                    "`g1 train -- --task getup --stage brace`")
            self.bridge = BraceBridge(
                self.mj_model, self.mj_data, path, sim_dt=SIM_DT,
                stand_policy_path=self.stand_policy_path)
            self._topple()
        elif self.policy_mode == "recover" and GetUpBridge is not None:
            stage_policies = {}
            for kind, given in (("roll", self.policy_roll),
                                ("getup", self.policy_standup)):
                path = given or find_latest_recovery_policy(kind)
                if path is None:
                    train_stage = "roll" if kind == "roll" else "standup"
                    raise FileNotFoundError(
                        f"no {kind} policy: train it with "
                        f"`g1 train -- --task getup --stage {train_stage}`")
                stage_policies[kind] = path
            self.bridge = GetUpBridge(
                self.mj_model, self.mj_data, stage_policies, sim_dt=SIM_DT,
                stand_policy_path=self.stand_policy_path)
            self._drop()
        elif self.stand_policy_path is not None and WalkStandBridge is not None:
            self.bridge = WalkStandBridge(
                self.mj_model, self.mj_data, self.policy_path,
                self.stand_policy_path, sim_dt=SIM_DT, mode=self.policy_mode)
        else:
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
        if self.policy_mode == "brace" and BraceBridge is not None and \
                isinstance(self.bridge, BraceBridge):
            tk.Button(btns, text="Topple", width=9,
                      command=self._topple).pack(side="left", padx=2)
        elif self.policy_mode == "recover" and GetUpBridge is not None and \
                isinstance(self.bridge, GetUpBridge):
            for text, cmd in (("Roll", ROLL), ("GetUp", GETUP), ("Auto", None)):
                tk.Button(btns, text=text, width=9,
                          command=lambda c=cmd: self._stage(c)).pack(side="left", padx=2)
        else:
            for text, cmd in (("Stand", (0, 0, 0)), ("Walk", (0.5, 0, 0)),
                              ("Turn left", (0, 0, 0.5)), ("Stop", (0, 0, 0))):
                tk.Button(btns, text=text, width=9,
                          command=lambda c=cmd: self._preset(c)).pack(side="left", padx=2)
        tk.Button(btns, text="Drop" if self.policy_mode == "recover" else "Reset",
                  width=7, command=self._reset).pack(side="left", padx=2)
        tk.Button(btns, text="Quit", width=7, command=self._quit).pack(side="left", padx=2)

        if GetUpBridge is None or not isinstance(self.bridge, GetUpBridge):
            self.standstill_var = tk.BooleanVar(value=True)
            tk.Checkbutton(self.root, text="Stand still (lock position on zero command)",
                           variable=self.standstill_var,
                           command=lambda: setattr(self.bridge, "standstill",
                                                   self.standstill_var.get())).pack(pady=(0, 4))

        if WalkStandBridge is not None and isinstance(self.bridge, WalkStandBridge):
            mrow = tk.Frame(self.root)
            mrow.pack(fill="x", padx=8, pady=2)
            tk.Label(mrow, text="Controller", width=14, anchor="w").pack(side="left")
            self.mode_var = tk.StringVar(value=self.bridge.mode_sel)
            tk.OptionMenu(mrow, self.mode_var, "walk", "stand", "auto",
                          command=lambda m: self.bridge.set_mode(m)).pack(side="left")
            tk.Label(mrow, text="(auto: zero cmd → stand policy)",
                     fg="gray").pack(side="left", padx=6)

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
        if self.policy_mode == "recover":
            hint = "Roll/GetUp force a stage, Auto resumes the switcher, Drop = new fall"
        elif self.policy_mode == "brace":
            hint = "Topple = stand + shove (guard auto-braces, then holds; sliders inactive)"
        else:
            hint = "Keys: arrows = move, A/D = turn, Space = stand"
        tk.Label(self.root, text=hint, fg="gray").pack(pady=(0, 6))

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
        if BraceBridge is not None and isinstance(self.bridge, BraceBridge):
            # Calm re-stand + re-arm (no shove); Topple demos the brace.
            reset_standing(self.mj_model, self.mj_data,
                           self.bridge.default_pos, height=0.78)
            self.bridge.guard.reset()
            self.fell = False
            return
        if GetUpBridge is not None and isinstance(self.bridge, GetUpBridge):
            self._drop()
            return
        reset_standing(self.mj_model, self.mj_data, self.bridge.default_pos, height=0.78)
        self.bridge.phase = 0.0
        self.fell = False

    def _topple(self):
        """Brace mode: stand + shove (seeded) + re-arm the guard."""
        self._seed += 1
        self.bridge.topple(seed=self._seed)
        self.fell = False

    def _drop(self):
        """Recover mode: new random fall (seeded) + re-arm the switcher."""
        self._seed += 1
        muj_q = self.bridge.stage_policies[ROLL].muj_q
        reset_fallen(self.mj_model, self.mj_data, self.bridge.default_pos,
                     muj_q=muj_q, seed=self._seed)
        self.bridge.switcher.reset()
        self.fell = False

    def _stage(self, stage):
        """Recover mode: force ROLL/GETUP, or None for AUTO (switcher decides)."""
        if stage is None:
            self.bridge.switcher.reset()
        else:
            self.bridge.switcher.force(stage)

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
        is_brace = BraceBridge is not None and isinstance(self.bridge, BraceBridge)
        is_getup = GetUpBridge is not None and isinstance(self.bridge, GetUpBridge)
        extra = ""
        if is_brace:
            state = f"BRACE [{stand_state.upper()}]"
            self.fell = stand_state == BRACING
        elif is_getup:
            try:
                _, _, _, facing, _ = self.bridge._metrics()
                extra = f" facing={facing:+.2f}"
            except Exception:
                pass
            state = f"GETUP [{stand_state.upper()}]"
            self.fell = stand_state in ("roll", "getup")
        elif h < 0.4 or tilt > 0.7:
            state = "FELL — press Reset"
            self.fell = True
        elif not np.allclose(cmd, 0):
            state = "WALKING"
            self.fell = False
        else:
            state = {"hold": "STANDING (settling…)", "frozen": "STANDING STILL",
                      "walk": "STANDING",
                      "stand": "STANDING (balance policy)"}.get(stand_state, "STANDING")
            self.fell = False
        self.status.config(
            text=f"t={self.sim_steps * SIM_DT:6.1f}s  height={h:.2f}m  tilt={tilt:.2f}{extra}\n"
                  f"cmd=[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]  {state}")
        self.root.after(10, self._tick)

    def run(self):
        print(f"Policy: {self.bridge.mode}, sliders set walk command in m/s.")
        print("Two windows should now be visible:")
        print("  1. MuJoCo viewer (3D robot)")
        print("  2. 'G1 Control' panel (sliders/buttons) — forced on top for 3 s;")
        print("     if you lose it, look behind the viewer or in the taskbar/dock.")
        self.root.mainloop()
