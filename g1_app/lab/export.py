"""Convert an RSL-RL training checkpoint (model_*.pt) to sim-ready policy.onnx.

Training already exports `policy.onnx` alongside every checkpoint, so prefer
that file straight from the run folder. Use this when only the .pt survived
(e.g. a checkpoint downloaded from Drive):

    g1 export --ckpt g1_app/models/g1_stand_policy.pt
    g1 export --ckpt model_1500.pt --out stand1500.onnx

The actor MLP (with its observation normalizer baked in, exactly like the
runner's own export) plus the metadata the bridge needs (joint order from
the G1 XML; gains/pose/scales from deploy.yaml) land in the .onnx, which
`g1 record` and `g1 stand --stand-policy` accept directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import torch

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(LAB_DIR)
WORKSPACE = os.path.dirname(APP_DIR)
for _p in (APP_DIR, WORKSPACE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def joint_names_in_order(xml_path: str) -> list[str]:
    """29 named hinge joints in document (= training) order."""
    tree = ET.parse(xml_path)
    names = [j.get("name") for j in tree.getroot().iter("joint")
             if j.get("type", "hinge") == "hinge" and j.get("name")]
    assert len(names) == 29, f"expected 29 joints, got {len(names)}"
    return names


class StandActor(torch.nn.Module):
    """Deterministic actor: normalize -> MLP/ELU -> mean action."""

    def __init__(self, state: dict):
        super().__init__()
        w0 = state["mlp.0.weight"]
        obs_dim, act_dim = w0.shape[1], state["mlp.6.weight"].shape[0]
        n_hidden = 3  # 512/256/128 layout: mlp.{0,2,4} hidden, mlp.6 out
        assert set(state) >= {f"mlp.{2 * i}.weight" for i in range(n_hidden + 1)}
        self.register_buffer("_mean", state["obs_normalizer._mean"].clone())
        self.register_buffer("_std", state["obs_normalizer._std"].clone())
        self.eps = 1.0e-2  # matches rsl_rl EmpiricalNormalization
        layers: list = []
        dims = [obs_dim] + [state[f"mlp.{2 * i}.weight"].shape[0]
                            for i in range(n_hidden)] + [act_dim]
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(torch.nn.Linear(din, dout))
            if i < n_hidden:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)
        # Copy weights (state uses mlp.{0,2,4,6}.* for the 4 linears, which
        # sit at sequential positions 0,2,4,6 between the ELUs).
        for i in range(n_hidden + 1):
            for kind in ("weight", "bias"):
                src, dst = state[f"mlp.{2 * i}.{kind}"], getattr(
                    self.mlp[2 * i], kind)
                assert dst.shape == src.shape, (i, kind, dst.shape, src.shape)
                with torch.no_grad():
                    dst.copy_(src)
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def forward(self, x):
        x = (x - self._mean) / (self._std + self.eps)
        return self.mlp(x)


def export_ckpt(ckpt_path: str, out_path: str) -> str:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    from core.config import DEPLOY_YAML, G1_MODEL_DIR, load_local_cfg

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "actor_state_dict" in ckpt, f"not an RSL-RL checkpoint: {ckpt_path}"
    actor = StandActor(ckpt["actor_state_dict"]).eval()
    print(f"[export] iter {ckpt.get('iter')}: "
          f"actor {actor.obs_dim}d -> {actor.act_dim}d")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros(1, actor.obs_dim)
    torch.onnx.export(
      actor, dummy, out_path, export_params=True, opset_version=18,
      input_names=["obs"], output_names=["action"], dynamic_axes={},
      dynamo=False,
    )

    cfg = load_local_cfg(DEPLOY_YAML)
    for k in ("stiffness", "damping", "default_pos"):
        assert len(cfg[k]) == 29, (k, len(cfg[k]))
    xml_path = os.path.join(G1_MODEL_DIR, "g1_29dof.xml")
    metadata = {
      "run_path": f"local-export:{os.path.basename(ckpt_path)}"
                  f":iter{ckpt.get('iter')}",
      "joint_names": joint_names_in_order(xml_path),
      "joint_stiffness": [float(v) for v in cfg["stiffness"]],
      "joint_damping": [float(v) for v in cfg["damping"]],
      "default_joint_pos": [float(v) for v in cfg["default_pos"]],
      "command_names": [],
      "observation_names": ["base_ang_vel", "projected_gravity",
                            "base_height", "joint_pos", "joint_vel",
                            "actions"],
      "action_scale": cfg["action_scale"],
    }
    attach_metadata_to_onnx(out_path, metadata)

    # Parity check: onnxruntime output must match the torch actor exactly.
    # (Batch is fixed to 1 like the runner's export, so check rows singly.)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    errs = []
    with torch.no_grad():
        for r in range(4):
            x = rng.normal(size=(1, actor.obs_dim)).astype(np.float32)
            ref = actor(torch.from_numpy(x)).numpy()
            got = sess.run(None, {"obs": x})[0]
            errs.append(float(np.abs(ref - got).max()))
    err = max(errs)
    print(f"[export] onnx parity max-abs-err: {err:.2e}")
    assert err < 1e-5, f"onnx/torch mismatch: {err}"
    print(f"[export] wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RSL-RL .pt checkpoint -> policy.onnx")
    ap.add_argument("--ckpt", required=True, help="training checkpoint (model_*.pt)")
    ap.add_argument("--out", default=None,
                    help="output .onnx (default: <ckpt-stem>.onnx next to ckpt)")
    args = ap.parse_args(argv)
    out = args.out or os.path.splitext(args.ckpt)[0] + ".onnx"
    export_ckpt(args.ckpt, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
