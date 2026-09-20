"""Bundled test terrains (single source of truth).

Every scene keeps a flat 2x2 m start pad at the origin; features start at
x >= 1.2 m. Regenerate with `g1 terrains` (tools/terrains.py).
"""

from __future__ import annotations

import os

from .config import G1_MODEL_DIR

TERRAINS = {
    "flat": os.path.join(G1_MODEL_DIR, "scene_29dof.xml"),
    "rough": os.path.join(G1_MODEL_DIR, "scene_rough.xml"),
    "slope": os.path.join(G1_MODEL_DIR, "scene_slope.xml"),
    "steps": os.path.join(G1_MODEL_DIR, "scene_steps.xml"),
    "obstacles": os.path.join(G1_MODEL_DIR, "scene_obstacles.xml"),
    "apartment": os.path.join(G1_MODEL_DIR, "scene_apartment.xml"),
}


def resolve_scene(terrain: str = "flat", scene: str | None = None) -> str:
    """Return scene XML path; explicit --scene overrides --terrain."""
    if scene:
        return scene
    if terrain not in TERRAINS:
        raise ValueError(f"unknown terrain {terrain!r}, choose from {sorted(TERRAINS)}")
    return TERRAINS[terrain]
