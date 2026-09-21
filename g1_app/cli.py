"""Unified G1 CLI: one entrypoint for every app/lab/tool.

Usage:
    python -m g1_app.cli <command> [args]
    g1 <command> [args]              (after `pip install -e ./g1_app`)

Commands:
    stand      viewer-only balanced stand demo
    gui        3D viewer + tkinter control panel
    check      10 s tkinter self-test
    recover    staged get-up v2 (Roll -> GetUp) from a fall
    train      training + dashboard (--task getup|stand, default getup)
    train-stand stand-still balance training + dashboard (shortcut)
    dashboard  friendly live dashboard (:6006)
    record     headless CPU rollout video of latest policy
    terrains   regenerate test scenes
    verify     model sha256 + deploy.yaml consistency + math smoke
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure imports work from workspace root AND from g1_app/ directly.
_CLI_DIR = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_CLI_DIR)
for _p in (_WS, _CLI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _add_common_stand_args(ap: argparse.ArgumentParser,
                           modes=("walk", "stand", "auto")):
    from core.bridge import DEFAULT_LOCAL_POLICY
    from core.terrains import TERRAINS

    ap.add_argument("--policy", default=DEFAULT_LOCAL_POLICY)
    ap.add_argument("--scene", default=None, help="MuJoCo scene XML (overrides --terrain)")
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="flat")
    ap.add_argument("--stand-policy", default=None,
                    help="Stand-still policy (default: models/g1_stand_policy.onnx, "
                         "else latest g1_stand snapshot)")
    ap.add_argument("--mode", choices=modes, default="walk",
                    help="walk = velocity policy, stand = balance policy, "
                         "auto = switch on zero command")
    return ap


def cmd_stand(args: argparse.Namespace) -> int:
    from core.bridge import run_stand
    from core.terrains import resolve_scene

    scene = resolve_scene(args.terrain, args.scene)
    ok = run_stand(args.policy, scene, args.seconds, args.sim_dt,
                   args.headless, standstill=not args.no_standstill,
                   stand_policy_path=args.stand_policy, mode=args.mode)
    return 0 if ok else 1


def cmd_gui(args: argparse.Namespace) -> int:
    from apps.gui import G1Gui

    G1Gui(policy=args.policy, scene=args.scene, terrain=args.terrain,
          stand_policy=args.stand_policy, mode=args.mode,
          policy_roll=args.policy_roll, policy_standup=args.policy_standup,
          seed=args.seed).run()
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    from apps.check import main

    try:
        main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    from core.bridge import run_recover
    from core.terrains import resolve_scene

    scene = resolve_scene(args.terrain, args.scene)
    stage_policies = {
        "roll": args.policy_roll,
        "getup": args.policy_standup,
    }
    stage_policies = {k: v for k, v in stage_policies.items() if v is not None}
    ok = run_recover(stage_policies or None, scene, args.seconds, args.sim_dt,
                     args.headless, stand_policy_path=args.stand_policy,
                     start=args.start, seed=args.seed)
    return 0 if ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    # lab.train has its own argparse; forward sys.argv style.
    sys.argv = ["g1 train"] + args.forward
    from lab.train import main

    return main()


def cmd_train_stand(args: argparse.Namespace) -> int:
    # Shortcut for `g1 train -- --task stand`.
    sys.argv = ["g1 train-stand", "--task", "stand"] + args.forward
    from lab.train import main

    return main()


def cmd_dashboard(args: argparse.Namespace) -> int:
    from lab.dashboard import main as dash_main

    sys.argv = ["g1 dashboard", "--port", str(args.port)]
    return dash_main()


def cmd_record(args: argparse.Namespace) -> int:
    from lab.record import find_latest_run, latest_policy_onnx, record

    if args.policy is not None:
        policy_path = args.policy
    else:
        run_dir = args.run_dir or find_latest_run(args.experiment)
        print(f"run: {run_dir}")
        policy_path = latest_policy_onnx(run_dir)
    record(policy_path, episodes=args.episodes,
           seconds=args.seconds, fps=args.fps, out_path=args.out, seed=args.seed,
           standing=args.stand)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from lab.export import export_ckpt

    out = args.out or os.path.splitext(args.ckpt)[0] + ".onnx"
    export_ckpt(args.ckpt, out)
    return 0


def cmd_terrains(_args: argparse.Namespace) -> int:
    from tools.terrains import main

    main()
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    from scripts.verify_model import main

    return main()


def build_parser() -> argparse.ArgumentParser:
    from core.terrains import TERRAINS

    ap = argparse.ArgumentParser(prog="g1", description="G1 humanoid app CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stand", help="viewer/headless balanced stand")
    _add_common_stand_args(p)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--sim-dt", type=float, default=0.005)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-standstill", action="store_true")
    p.set_defaults(func=cmd_stand)

    p = sub.add_parser("gui", help="interactive 3D + control panel")
    _add_common_stand_args(p, modes=("walk", "stand", "auto", "recover"))
    p.add_argument("--policy-roll", default=None,
                   help="Roll-to-supine ONNX for --mode recover")
    p.add_argument("--policy-standup", default=None,
                   help="Supine-to-stand ONNX for --mode recover")
    p.add_argument("--seed", type=int, default=0,
                   help="Fall seed for --mode recover drops")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("check", help="tkinter env self-test")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("recover", help="staged get-up from a fall")
    p.add_argument("--scene", default=None, help="MuJoCo scene XML (overrides --terrain)")
    p.add_argument("--terrain", choices=sorted(TERRAINS), default="flat")
    p.add_argument("--policy-roll", default=None,
                   help="Roll-to-supine ONNX (default: curated, else latest "
                        "g1_getup_roll snapshot)")
    p.add_argument("--policy-standup", default=None,
                   help="Supine-to-stand ONNX (default: curated, else latest "
                        "g1_getup_standup snapshot)")
    p.add_argument("--stand-policy", default=None,
                   help="Balance policy for the DONE handoff (default: "
                        "models/g1_stand_policy.onnx, else latest snapshot)")
    p.add_argument("--start", choices=("fallen", "standing"), default="fallen")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--sim-dt", type=float, default=0.005)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("train", help="get-up/stand training + dashboard")
    p.add_argument("forward", nargs=argparse.REMAINDER,
                   help="args forwarded to lab.train (e.g. -- --task stand)")
    # Consume optional `--` separator.
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-stand", help="stand-still training + dashboard")
    p.add_argument("forward", nargs=argparse.REMAINDER,
                   help="args forwarded to lab.train (e.g. -- --num-envs 1024)")
    p.set_defaults(func=cmd_train_stand)

    p = sub.add_parser("dashboard", help="friendly live dashboard")
    p.add_argument("--port", type=int, default=6006)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("record", help="headless CPU policy video")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--policy", default=None,
                   help="policy.onnx directly (e.g. models/g1_stand_policy.onnx); "
                        "overrides --run-dir/--experiment lookup")
    p.add_argument("--experiment", default="g1_getup_standup",
                   help="experiment folder under logs/rsl_rl "
                        "(g1_getup_standup|g1_getup_roll|g1_stand)")
    p.add_argument("--stand", action="store_true",
                   help="start episodes standing (for stand policy) not fallen")
    p.add_argument("--start", choices=["fallen", "supine", "prone"], default="fallen",
                   help="start state: fallen (random sprawl), supine (flat on "
                        "back, extended) or prone (face-down, for roll-over eval)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("terrains", help="regenerate test scenes")
    p.set_defaults(func=cmd_terrains)

    p = sub.add_parser("export", help="RSL-RL .pt checkpoint -> policy.onnx")
    p.add_argument("--ckpt", required=True, help="training checkpoint (model_*.pt)")
    p.add_argument("--out", default=None,
                   help="output .onnx (default: <ckpt-stem>.onnx next to ckpt)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="model + config + math checks")
    p.set_defaults(func=cmd_verify)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `g1 train -- --num-envs 8` -> strip the `--`.
    if getattr(args, "forward", None) and args.forward[:1] == ["--"]:
        args.forward = args.forward[1:]
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
