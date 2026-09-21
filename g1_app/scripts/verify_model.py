"""Verify bundled model + config consistency (used by `g1 verify`).

Checks:
1. models/g1_policy.onnx exists, >10KB (not an LFS pointer), sha256 logged.
2. models/deploy.yaml parses to 29-entry gains/pose/scales.
3. core.math projected-gravity convention: identity -> [0,0,-1].
4. All TERRAINS scenes exist on disk.
"""

from __future__ import annotations

import hashlib
import os
import sys

_CLI_DIR = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_CLI_DIR)
_WS = os.path.dirname(_APP)
for _p in (_WS, _APP):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    import numpy as np

    from core.bridge import DEFAULT_LOCAL_POLICY
    from core.config import DEPLOY_YAML, get_local_cfg
    from core.math import quat_to_projected_gravity
    from core.terrains import TERRAINS

    ok = True

    # 1. model
    if not os.path.isfile(DEFAULT_LOCAL_POLICY):
        print(f"FAIL: missing policy {DEFAULT_LOCAL_POLICY}")
        ok = False
    elif os.path.getsize(DEFAULT_LOCAL_POLICY) < 10000:
        print(f"FAIL: policy looks like LFS pointer: {DEFAULT_LOCAL_POLICY}")
        ok = False
    else:
        h = hashlib.sha256()
        with open(DEFAULT_LOCAL_POLICY, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print(f"policy: {DEFAULT_LOCAL_POLICY} ({os.path.getsize(DEFAULT_LOCAL_POLICY)/1024:.0f} KB)")
        print(f"sha256: {h.hexdigest()}")
        print("expected prefix (training export): 2a66ca63… (full value in README)")

    # 2. config
    try:
        cfg = get_local_cfg()
        assert len(cfg["stiffness"]) == 29
        print(f"config: {DEPLOY_YAML} OK (29-DoF, step_dt={cfg['step_dt']})")
    except Exception as e:
        print(f"FAIL: deploy.yaml: {e}")
        ok = False

    # 3. math convention
    g = quat_to_projected_gravity(np.array([1.0, 0, 0, 0]))
    if np.allclose(g, [0, 0, -1]):
        print("math: projected gravity identity -> [0,0,-1] OK")
    else:
        print(f"FAIL: projected gravity identity -> {g}")
        ok = False

    # 4. scenes
    for name, path in TERRAINS.items():
        if os.path.isfile(path):
            print(f"terrain {name}: OK")
        else:
            print(f"FAIL: terrain {name} missing: {path}")
            ok = False

    print("verify: PASS" if ok else "verify: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
