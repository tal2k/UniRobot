"""Generate test-terrain scenes for the G1 app (run once, scenes are committed).

Canonical home (moved from legacy make_terrains.py). Prefer:

    g1 terrains

Reads models/g1/scene_29dof.xml (flat floor) and derives rough/slope/steps/
obstacles. Every scene keeps a flat 2x2 m start pad around the origin.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import imageio
import numpy as np

try:
    from g1_app.core.config import G1_MODEL_DIR
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from core.config import G1_MODEL_DIR
    except ImportError:
        HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        G1_MODEL_DIR = os.path.join(HERE, "models", "g1")

G1_DIR = G1_MODEL_DIR
BASE_SCENE = os.path.join(G1_DIR, "scene_29dof.xml")


def quat_y(theta):
    """MuJoCo (w,x,y,z) quaternion for a rotation about Y (pitch)."""
    return [math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0]


def s(v):
    return " ".join(f"{x:.4f}" for x in np.asarray(v, dtype=float).flat)


def load_base():
    tree = ET.parse(BASE_SCENE)
    return tree, tree.getroot().find("asset"), tree.getroot().find("worldbody")


def add_box(worldbody, pos, size_full, euler_quat=None, geom_type="box"):
    """size_full = full extents (MuJoCo wants half sizes)."""
    g = ET.SubElement(worldbody, "geom")
    g.set("pos", s(pos))
    g.set("type", geom_type)
    g.set("size", s(0.5 * np.asarray(size_full, dtype=float)))
    if euler_quat is not None:
        g.set("quat", s(euler_quat))
    return g


def save(tree, name):
    tree.write(os.path.join(G1_DIR, name))
    print("wrote", name)


# ---------------------------------------------------------------- value noise
def fbm(n=256, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    amp, freq, total = 1.0, 12, 0.0
    for _ in range(5):  # 5 octaves of coarse random grids, bilinear upsampled
        gx = rng.random((freq + 1, freq + 1))
        xs = np.linspace(0, freq, n)
        x0 = np.clip(xs.astype(int), 0, freq - 1)
        fx = (xs - x0)[None, :]                       # (1, n)
        rows = gx[:, x0] * (1 - fx) + gx[:, x0 + 1] * fx  # (freq+1, n)
        fy = fx.T                                     # (n, 1)
        img += amp * (rows[x0, :] * (1 - fy) + rows[x0 + 1, :] * fy)
        total += amp
        amp *= 0.5
        freq *= 2
    img = (img - img.min()) / (img.max() - img.min())
    # Fade edges to 0 so the field meets the flat floor with no lip.
    n = img.shape[0]
    ramp = np.minimum(np.arange(n), np.arange(n)[::-1]) / 10.0
    ramp = np.clip(ramp, 0, 1)
    img = img * ramp[None, :] * ramp[:, None]
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------- scenes
def make_rough():
    tree, asset, world = load_base()
    png = "rough.png"
    # NOTE: MuJoCo resolves asset files relative to the compiler meshdir,
    # so the heightfield image must live in meshes/ next to the STLs.
    imageio.imwrite(os.path.join(G1_DIR, "meshes", png), fbm())
    h = ET.SubElement(asset, "hfield")
    h.set("name", "rough_hfield")
    h.set("size", s([6.0, 6.0, 0.15, 0.02]))  # surface in [2, 17] cm over the floor
    h.set("file", png)
    g = ET.SubElement(world, "geom")
    g.set("type", "hfield")
    g.set("hfield", "rough_hfield")
    g.set("pos", s([7.2, 0.0, 0.0]))  # covers x in [1.2, 13.2], y in [-6, 6]
    save(tree, "scene_rough.xml")


def make_slope():
    tree, _, world = load_base()
    a, L, t, W = 0.12, 3.0, 0.12, 3.0  # angle, slope length, thickness, width
    # Top surface runs from (1.5, 0) up to (1.5+L*cos a, L*sin a); bury it a touch.
    cx = 1.5 + 0.5 * L * math.cos(a)
    cz = 0.5 * L * math.sin(a) - 0.5 * t - 0.03
    ramp = add_box(world, [cx, 0.0, cz], [L, W, t], quat_y(-a))
    ramp.set("pos", s([cx, 0.0, cz]))
    top_x = 1.5 + L * math.cos(a)
    top_z = L * math.sin(a)
    add_box(world, [top_x + 1.0, 0.0, top_z - 0.09], [2.0, W, 0.12])
    print(f"slope: ramp top edge at x={top_x:.2f} z={top_z:.2f}")
    save(tree, "scene_slope.xml")


def make_steps():
    tree, _, world = load_base()
    w, h, n, y = 0.32, 0.06, 5, 3.0  # tread depth, riser, count, width
    x0 = 1.5
    for i in range(1, n + 1):  # solid blocks: no floating edges
        add_box(world, [x0 + (i - 0.5) * w, 0.0, 0.5 * i * h], [w, y, i * h])
    plat_x0 = x0 + n * w
    add_box(world, [plat_x0 + 0.6, 0.0, 0.5 * n * h], [1.2, y, n * h])
    down_x0 = plat_x0 + 1.2
    for j in range(1, n + 1):
        top = (n - j) * h
        if top <= 0:
            continue
        add_box(world, [down_x0 + (j - 0.5) * w, 0.0, 0.5 * top], [w, y, top])
    print(f"steps: {n}x{h} m up from x={x0}, platform top z={n*h:.2f}")
    save(tree, "scene_steps.xml")


def make_obstacles():
    tree, _, world = load_base()
    add_box(world, [2.5, 0.0, 0.03], [0.3, 2.4, 0.06])      # step-over bar, 6 cm
    add_box(world, [4.0, 1.25, 0.25], [0.5, 1.1, 0.5])       # block right...
    add_box(world, [4.0, -1.25, 0.25], [0.5, 1.1, 0.5])      # ...block left: 1.4 m corridor
    cyl = add_box(world, [5.8, 0.7, 0.25], [0.25, 0.25], geom_type="cylinder")
    cyl.set("size", s([0.25, 0.25]))                          # r=0.25, h=0.5 pillar
    add_box(world, [7.0, 0.0, 0.025], [0.3, 2.4, 0.05])      # step-over bar, 5 cm
    save(tree, "scene_obstacles.xml")


def make_all():
    make_rough()
    make_slope()
    make_steps()
    make_obstacles()
    print("done — scenes reference g1_29dof.xml + meshes in the same folder")


def main():
    make_all()


if __name__ == "__main__":
    main()
