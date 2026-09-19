"""Single source of truth for app paths + policy config.

`models/deploy.yaml` is the authority for gains/pose/scales/rates.
`LOCAL_CFG` fallback keeps headless runs working if the yaml is missing,
but `load_local_cfg()` prefers the yaml and validates shapes.
"""

from __future__ import annotations

import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# When installed (`pip install -e .`), __file__ is .../g1_app/config.py so
# dirname x1 == g1_app/. When run as core/config.py, dirname x2 == g1_app/.
# Normalise: if basename is 'core', go one up.
if os.path.basename(APP_DIR) == "core":
    APP_DIR = os.path.dirname(APP_DIR)
WORKSPACE = os.path.dirname(APP_DIR)

MODELS_DIR = os.path.join(APP_DIR, "models")
DEFAULT_LOCAL_POLICY = os.path.join(MODELS_DIR, "g1_policy.onnx")
DEPLOY_YAML = os.path.join(MODELS_DIR, "deploy.yaml")
G1_MODEL_DIR = os.path.join(MODELS_DIR, "g1")

# Run artifacts: prefer outputs/videos/, fall back to legacy videos/.
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")
LEGACY_VIDEOS_DIR = os.path.join(APP_DIR, "videos")


def resolve_videos_dir() -> str:
    """Where to read/write rollout videos (new location preferred)."""
    if os.path.isdir(VIDEOS_DIR):
        return VIDEOS_DIR
    if os.path.isdir(LEGACY_VIDEOS_DIR) and not os.path.isdir(OUTPUTS_DIR):
        return LEGACY_VIDEOS_DIR
    return VIDEOS_DIR


def _parse_float_list(text: str) -> list[float]:
    import re

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return [float(n) for n in nums]


def load_local_cfg(path: str = DEPLOY_YAML) -> dict:
    """Load 29-DoF velocity-policy config from deploy.yaml (no PyYAML needed).

    Returns dict with step_dt, stiffness, damping, default_pos,
    action_scale, gait_period. Raises FileNotFoundError if missing.
    """
    with open(path) as f:
        text = f.read()

    def section_list(key: str) -> list[float]:
        import re

        # Allow leading indentation (deploy.yaml nests scale/offset).
        # Match `key: [...]` possibly spanning lines until the closing ']'.
        m = re.search(rf"^[ \t]*{key}:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            return _parse_float_list(m.group(1))
        # multiline flow style without opening '[' on the key line
        m = re.search(rf"^[ \t]*{key}:\s*(.*?)\n(?=^\w|\Z)", text, re.M | re.S)
        if m:
            vals = _parse_float_list(m.group(0))
            if vals:
                return vals
        raise KeyError(f"{key} not found in {path}")

    import re

    m_dt = re.search(r"step_dt:\s*([0-9.]+)", text)
    m_gait = re.search(r"period:\s*([0-9.]+)", text)
    cfg = {
        "step_dt": float(m_dt.group(1)) if m_dt else 0.02,
        "stiffness": section_list("stiffness"),
        "damping": section_list("damping"),
        "default_pos": section_list("default_joint_pos")
        if "default_joint_pos" in text
        else section_list("offset"),
        "action_scale": section_list("scale") if "actions:" in text else [],
        "gait_period": float(m_gait.group(1)) if m_gait else 0.6,
    }
    # action scale lives under actions: -> scale: [...] (second occurrence);
    # fall back to explicit parse of that block.
    if not cfg["action_scale"]:
        m = re.search(r"scale:\s*\[(.*?)\]", text, re.S)
        cfg["action_scale"] = _parse_float_list(m.group(1)) if m else []
    for k in ("stiffness", "damping", "default_pos", "action_scale"):
        if len(cfg[k]) != 29:
            raise ValueError(f"{k} has {len(cfg[k])} entries, expected 29 in {path}")
    return cfg


# Hardcoded fallback (mirrors deploy.yaml at time of writing) — used only
# when the yaml is unreadable, with a warning. Keeps CI/headless robust.
FALLBACK_LOCAL_CFG = {
    "step_dt": 0.02,
    "stiffness": [40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
                  40.2, 28.5, 28.5, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
                  14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8],
    "damping": [2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
                2.6, 1.8, 1.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
                0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1],
    "default_pos": [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
                    0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0, 0,
                    0.35, -0.18, 0, 0.87, 0, 0, 0],
    "action_scale": [0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                     0.55, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
                     0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07],
    "gait_period": 0.6,
}


def get_local_cfg() -> dict:
    """Best-effort yaml load with fallback (warns on fallback)."""
    try:
        return load_local_cfg()
    except Exception as e:  # noqa: BLE001 - fallback path is intentional
        print(f"[config] WARNING: using fallback LOCAL_CFG ({e})")
        return dict(FALLBACK_LOCAL_CFG)
