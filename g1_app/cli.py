"""Unified G1 CLI: one entrypoint for every app/lab/tool.

Usage:
    python -m g1_app.cli <command> [args]
    g1 <command> [args]              (after `pip install -e ./g1_app`)

Commands:
    stand      viewer-only balanced stand demo
    gui        3D viewer + tkinter control panel
    check      10 s tkinter self-test
    train      get-up training + dashboard
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


def _add_common_stand_args(ap: argparse.ArgumentParser):
    from g1_app.core.bridge import DEFAULT_LOCAL_POLICY
    from g1_app.core.terrains import TERRAINS

    ap.add_argument("--policy", default=DEFAULT_LOCAL_POLICY)
    ap.add_argument("--scene", default=None, help="MuJoCo scene XML (overrides --terrain)")
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="flat")
    return ap


def cmd_stand(args: argparse.Namespace) -> int:
    from g1_app.core.bridge import run_stand
    from g1_app.core.terrains import resolve_scene

    scene = resolve_scene(args.terrain, args.scene)
    ok = run_stand(args.policy, scene, args.seconds, args.sim_dt,
                   args.headless, standstill=not args.no_standstill)
    return 0 if ok else 1


def cmd_gui(args: argparse.Namespace) -> int:
    from g1_app.apps.gui import G1Gui

    G1Gui(policy=args.policy, scene=args.scene, terrain=args.terrain).run()
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    from g1_app.apps.check import main

    try:
        main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # lab.train has its own argparse; forward sys.argv style.
    sys.argv = ["g1 train"] + args.forward
    from g1_app.lab.train import main

    return main()


def cmd_dashboard(args: argparse.Namespace) -> int:
    from g1_app.lab.dashboard import main as dash_main

    sys.argv = ["g1 dashboard", "--port", str(args.port)]
    return dash_main()


def cmd_record(args: argparse.Namespace) -> int:
    from g1_app.lab.record import find_latest_run, latest_policy_onnx, record

    run_dir = args.run_dir or find_latest_run()
    print(f"run: {run_dir}")
    record(latest_policy_onnx(run_dir), episodes=args.episodes,
           seconds=args.seconds, fps=args.fps, out_path=args.out, seed=args.seed)
    return 0


def cmd_terrains(_args: argparse.Namespace) -> int:
    from g1_app.tools.terrains import main

    main()
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    from g1_app.scripts.verify_model import main

    return main()


def build_parser() -> argparse.ArgumentParser:
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
    _add_common_stand_args(p)
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("check", help="tkinter env self-test")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("train", help="get-up training + dashboard")
    p.add_argument("forward", nargs=argparse.REMAINDER,
                   help="args forwarded to lab.train (e.g. -- --num-envs 1024)")
    # Consume optional `--` separator.
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("dashboard", help="friendly live dashboard")
    p.add_argument("--port", type=int, default=6006)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("record", help="headless CPU policy video")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("terrains", help="regenerate test scenes")
    p.set_defaults(func=cmd_terrains)

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
