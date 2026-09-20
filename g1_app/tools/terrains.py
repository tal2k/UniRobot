"""Generate test-terrain scenes for the G1 app (run once, scenes are committed).

Canonical home (moved from legacy make_terrains.py). Prefer:

    g1 terrains

Reads models/g1/scene_29dof.xml (flat floor) and derives rough/slope/steps/
obstacles/apartment. Every scene keeps a flat 2x2 m start pad around the origin.
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


def add_box(worldbody, pos, size_full, euler_quat=None, geom_type="box",
            rgba=None, material=None):
    """size_full = full extents (MuJoCo wants half sizes)."""
    g = ET.SubElement(worldbody, "geom")
    g.set("pos", s(pos))
    g.set("type", geom_type)
    g.set("size", s(0.5 * np.asarray(size_full, dtype=float)))
    if euler_quat is not None:
        g.set("quat", s(euler_quat))
    if rgba is not None:
        g.set("rgba", s(rgba))
    if material is not None:
        g.set("material", material)
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


# ---------------------------------------------------------------- apartment
# A realistic elderly apartment: living room (robot spawn), kitchen,
# bedroom, bathroom, with walls, 0.9 m doorways, furniture, and everyday
# trip hazards (rug lips, thresholds, clutter, a power cord). The 2x2 m
# spawn pad around the origin stays flat and empty.
RUG_RED = [0.65, 0.22, 0.22, 1.0]
TILE_BLUE = [0.30, 0.50, 0.78, 1.0]
WOOD = [0.50, 0.35, 0.20, 1.0]
FABRIC = [0.25, 0.45, 0.35, 1.0]
WHITE = [0.90, 0.90, 0.90, 1.0]
GREEN = [0.30, 0.60, 0.35, 1.0]
DARK = [0.15, 0.15, 0.15, 1.0]
WALL_C = [0.82, 0.80, 0.75, 1.0]

WALL_T, WALL_H = 0.15, 2.4


def quat_x(theta):
    """MuJoCo (w,x,y,z) quaternion for a rotation about X."""
    return [math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0]


def wall_x(world, x1, x2, y):
    """Wall segment along X from x1 to x2 at height y."""
    cx = 0.5 * (x1 + x2)
    add_box(world, [cx, y, 0.5 * WALL_H], [x2 - x1, WALL_T, WALL_H],
            rgba=WALL_C)


def wall_y(world, y1, y2, x):
    """Wall segment along Y from y1 to y2 at height x."""
    cy = 0.5 * (y1 + y2)
    add_box(world, [x, cy, 0.5 * WALL_H], [WALL_T, y2 - y1, WALL_H],
            rgba=WALL_C)


def table(world, x, y, w, d, top_z=0.72, rgba=WOOD):
    """Table: thin top + 4 legs (feet fit underneath)."""
    add_box(world, [x, y, top_z - 0.03], [w, d, 0.06], rgba=rgba)
    for dx in (-0.5 * w + 0.05, 0.5 * w - 0.05):
        for dy in (-0.5 * d + 0.05, 0.5 * d - 0.05):
            add_box(world, [x + dx, y + dy, 0.5 * (top_z - 0.06)],
                    [0.05, 0.05, top_z - 0.06], rgba=rgba)


def chair(world, x, y, rgba=WOOD, rot_z=0.0):
    """Dining chair: seat + backrest + 4 legs (rot_z turns it)."""
    q = quat_z(rot_z) if rot_z else None
    add_box(world, [x, y, 0.45], [0.45, 0.45, 0.06], euler_quat=q, rgba=rgba)
    bx, by = x - 0.2 * math.cos(rot_z), y - 0.2 * math.sin(rot_z)
    add_box(world, [bx, by, 0.75], [0.06, 0.45, 0.55], euler_quat=q, rgba=rgba)
    for dx in (-0.18, 0.18):
        for dy in (-0.18, 0.18):
            add_box(world, [x + dx, y + dy, 0.21], [0.05, 0.05, 0.42],
                    rgba=rgba)


# ------------------------------------------------------- realism helpers
CREAM = [0.92, 0.88, 0.80, 1.0]
TERRA = [0.75, 0.40, 0.25, 1.0]
MUSTARD = [0.85, 0.65, 0.20, 1.0]
STEEL = [0.70, 0.72, 0.75, 1.0]
BLACK = [0.10, 0.10, 0.12, 1.0]
BOOK_COLORS = [
    [0.70, 0.20, 0.20, 1.0], [0.20, 0.40, 0.70, 1.0],
    [0.20, 0.55, 0.30, 1.0], [0.80, 0.60, 0.15, 1.0],
    [0.55, 0.30, 0.60, 1.0], [0.85, 0.45, 0.20, 1.0],
]

_rng = np.random.default_rng(11)


def ensure_materials(asset):
    """Emissive materials (lamps, screens, daylight glass)."""
    for name, emission, rgba in (
        ("bulb", "1.0", [1.0, 1.0, 0.9, 1.0]),
        ("screen", "0.8", [0.45, 0.65, 1.0, 1.0]),
        ("sky", "0.55", [0.72, 0.86, 1.0, 1.0]),
    ):
        m = ET.SubElement(asset, "material")
        m.set("name", name)
        m.set("emission", emission)
        m.set("rgba", s(rgba))


def quat_z(theta):
    """MuJoCo (w,x,y,z) quaternion for a rotation about Z (yaw)."""
    return [math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)]


def swing_door(world, hinge, closed_center, angle_deg, width=0.9, rgba=WOOD):
    """Open door panel: rotate the closed pose about the hinge point."""
    a = math.radians(angle_deg)
    dx, dy = closed_center[0] - hinge[0], closed_center[1] - hinge[1]
    ca, sa = math.cos(a), math.sin(a)
    cx = hinge[0] + dx * ca - dy * sa
    cy = hinge[1] + dx * sa + dy * ca
    dims = [0.05, width, 2.0] if abs(dx) < 1e-9 else [width, 0.05, 2.0]
    add_box(world, [cx, cy, 1.0], dims, euler_quat=quat_z(a), rgba=rgba)


def window(world, cx, cy, cz, w=1.6, h=1.2, orient="x", curtains=TERRA,
           glass="sky"):
    """Fake window mounted on a wall face: frame + glowing glass + curtains.

    orient "x": wall runs along X (glass plane faces +/-Y).
    orient "y": wall runs along Y (glass plane faces +/-X).
    curtains=None skips the side drapes (sidelights, frosted glass).
    """
    if orient == "x":
        dims = lambda a, b, c: [a, b, c]  # noqa: E731
    else:
        dims = lambda a, b, c: [b, a, c]  # noqa: E731
    add_box(world, [cx, cy, cz], dims(w, 0.04, h), material=glass)
    add_box(world, [cx, cy, cz + h / 2 + 0.05], dims(w + 0.24, 0.08, 0.1),
            rgba=WHITE)
    add_box(world, [cx, cy, cz - h / 2 - 0.05], dims(w + 0.24, 0.08, 0.1),
            rgba=WHITE)
    add_box(world, [cx - w / 2 - 0.06 if orient == "x" else cx,
                    cy - w / 2 - 0.06 if orient == "y" else cy, cz],
            dims(0.12, 0.08, h + 0.2), rgba=WHITE)
    add_box(world, [cx + w / 2 + 0.06 if orient == "x" else cx,
                    cy + w / 2 + 0.06 if orient == "y" else cy, cz],
            dims(0.12, 0.08, h + 0.2), rgba=WHITE)
    add_box(world, [cx, cy, cz - h / 2 - 0.16], dims(w + 0.4, 0.14, 0.06),
            rgba=WHITE)  # sill
    if curtains is None:
        return
    for side in (-1, 1):  # curtains
        add_box(world, [cx + side * (w / 2 + 0.28) if orient == "x" else cx,
                        cy + side * (w / 2 + 0.28) if orient == "y" else cy,
                        cz + 0.05],
                dims(0.35, 0.07, h + 0.3), rgba=curtains)


def mug(world, x, y, z, rgba=MUSTARD):
    """Coffee mug: squat cylinder."""
    m = add_box(world, [x, y, z + 0.05], [0.04, 0.04], geom_type="cylinder",
                rgba=rgba)
    m.set("size", s([0.04, 0.10]))


def ceiling_lamp(world, x, y, shade_z=2.0):
    """Hanging lamp: cord + shade + glowing bulb (no ceiling modeled)."""
    cord = add_box(world, [x, y, (2.4 + shade_z) / 2 + 0.09], [0.024, 0.024],
                   geom_type="cylinder")
    cord.set("size", s([0.012, 2.4 - shade_z]))
    shade = add_box(world, [x, y, shade_z], [0.16, 0.16],
                    geom_type="cylinder", rgba=CREAM)
    shade.set("size", s([0.16, 0.18]))
    add_box(world, [x, y, shade_z - 0.10], [0.12], geom_type="sphere",
            material="bulb")


def book_row(world, x, y, z, n, along="x", seed=None):
    """Row of colorful books on a shelf."""
    rng = np.random.default_rng(seed) if seed is not None else _rng
    for i in range(n):
        w = 0.03 + rng.random() * 0.025
        hh = 0.20 + rng.random() * 0.08
        c = BOOK_COLORS[i % len(BOOK_COLORS)]
        if along == "x":
            add_box(world, [x + i * 0.05, y, z + hh / 2], [w, 0.14, hh],
                    rgba=c)
        else:
            add_box(world, [x, y + i * 0.05, z + hh / 2], [0.14, w, hh],
                    rgba=c)


def bordered_rug(world, cx, cy, w, d, base=TERRA, inset=CREAM):
    """Two-tone rug with border (thin trip lip)."""
    add_box(world, [cx, cy, 0.01], [w, d, 0.02], rgba=base)
    add_box(world, [cx, cy, 0.016], [w - 0.3, d - 0.3, 0.012], rgba=inset)


def plant(world, x, y, scale=1.0):
    """Potted plant: pot + foliage spheres."""
    pot = add_box(world, [x, y, 0.15 * scale], [0.16 * scale, 0.16 * scale],
                  geom_type="cylinder", rgba=TERRA)
    pot.set("size", s([0.16 * scale, 0.30 * scale]))
    for dx, dy, dz, r in ((0, 0, 0.55, 0.22), (0.15, 0.1, 0.42, 0.15),
                          (-0.14, 0.08, 0.44, 0.16)):
        add_box(world, [x + dx * scale, y + dy * scale, dz * scale],
                [2 * r * scale], geom_type="sphere", rgba=GREEN)


def make_apartment():
    tree, asset, world = load_base()
    ensure_materials(asset)

    # --- shell: x in [-6, 12], y in [-5, 6] ---
    wall_x(world, -6, 12, -5)
    wall_x(world, -6, 12, 6)
    wall_y(world, -5, 6, -6)
    wall_y(world, -5, 6, 12)

    # --- interior walls (0.9 m doorways, wide living|foyer opening) ---
    wall_x(world, -2, 1, -1)       # living | foyer
    wall_x(world, 3.5, 5, -1)
    wall_y(world, -5, -3.5, 5)     # foyer|utility, foyer|kitchen, living|kitchen
    wall_y(world, -2.6, -1.8, 5)
    wall_y(world, -0.9, 0.5, 5)
    wall_y(world, 1.4, 4, 5)
    wall_y(world, -5, -0.5, -2)    # bathroom|foyer, bedroom|living
    wall_y(world, 0.5, 6, -2)
    wall_x(world, -6, -4.5, 0.5)   # bedroom | hall
    wall_x(world, -3.6, -2, 0.5)
    wall_x(world, -6, -4.5, -0.5)  # bathroom | hall
    wall_x(world, -3.6, -2, -0.5)
    wall_x(world, 5, 6.2, 4)       # kitchen | balcony
    wall_x(world, 7.1, 8.5, 4)
    add_box(world, [6.65, 4, 2.2], [1.1, 0.15, 0.2], rgba=WOOD)  # beam
    wall_y(world, -5, -4, 8.5)     # utility|storage, kitchen|office, balcony|office
    wall_y(world, -3.1, -2, 8.5)
    wall_y(world, -2, -1.0, 8.5)     # office door gap y in [-1.0, -0.1]
    wall_y(world, -0.1, 4, 8.5)
    wall_y(world, 4, 6, 8.5)
    wall_x(world, 5, 8.5, -2)      # kitchen | utility
    wall_x(world, 8.5, 12, -1)     # storage | office

    # --- open interior doors (ajar panels) ---
    swing_door(world, (-3.6, 0.5), (-4.05, 0.5), -65)    # bedroom
    swing_door(world, (-3.6, -0.5), (-4.05, -0.5), 65)   # bathroom
    swing_door(world, (5, 1.4), (5, 0.95), -60)          # kitchen|living
    swing_door(world, (8.5, -0.1), (8.5, -0.55), -60)     # office
    swing_door(world, (5, -2.6), (5, -3.05), -60)        # utility
    swing_door(world, (8.5, -3.1), (8.5, -3.55), -60)    # storage
    # Balcony glazed door, half open.
    add_box(world, [6.81, 3.66, 1.0], [0.9, 0.05, 2.0],
            euler_quat=quat_z(math.radians(50)), rgba=WOOD)
    add_box(world, [6.81, 3.66, 1.4], [0.6, 0.02, 1.0],
            euler_quat=quat_z(math.radians(50)), material="sky")
    # Bathroom threshold strip (hall door).
    add_box(world, [-4.05, -0.5, 0.015], [0.9, 0.15, 0.03], rgba=WOOD)
    # Front door (closed) + frame + knob + mat + sidelight.
    add_box(world, [1.0, -4.88, 1.05], [1.05, 0.1, 2.1], rgba=WOOD)
    add_box(world, [0.42, -4.88, 1.12], [0.12, 0.14, 2.25], rgba=WHITE)
    add_box(world, [1.58, -4.88, 1.12], [0.12, 0.14, 2.25], rgba=WHITE)
    add_box(world, [1.0, -4.88, 2.22], [1.3, 0.14, 0.15], rgba=WHITE)
    add_box(world, [1.35, -4.80, 1.05], [0.07], geom_type="sphere",
            rgba=STEEL)
    add_box(world, [1.0, -4.2, 0.008], [1.3, 0.8, 0.015], rgba=GREEN)
    window(world, 2.4, -4.90, 1.5, w=0.5, h=1.4, orient="x", curtains=None)

    # --- living room (spawn at origin; 2x2 m around it stays empty) ---
    bordered_rug(world, 2.9, -2.7, 2.8, 2.0)
    # L-couch: base, backrest, arms, cushions, throw on the chaise.
    add_box(world, [-0.2, 3.2, 0.21], [2.4, 0.95, 0.42], rgba=FABRIC)
    add_box(world, [-0.2, 3.8, 0.62], [2.4, 0.28, 0.85], rgba=FABRIC)
    add_box(world, [-1.26, 3.2, 0.50], [0.28, 0.95, 0.60], rgba=FABRIC)
    add_box(world, [0.86, 3.2, 0.50], [0.28, 0.95, 0.60], rgba=FABRIC)
    add_box(world, [-0.95, 3.15, 0.50], [0.68, 0.75, 0.16], rgba=CREAM)
    add_box(world, [-0.2, 3.15, 0.50], [0.68, 0.75, 0.16], rgba=FABRIC)
    add_box(world, [0.55, 3.15, 0.50], [0.68, 0.75, 0.16], rgba=CREAM)
    add_box(world, [0.75, 2.35, 0.21], [0.95, 1.5, 0.42], rgba=FABRIC)
    add_box(world, [0.75, 2.35, 0.44], [0.9, 1.4, 0.04], rgba=MUSTARD)
    # Coffee table + book + mug.
    table(world, -0.5, 1.6, 1.2, 0.65, top_z=0.40)
    add_box(world, [-0.7, 1.6, 0.42], [0.3, 0.22, 0.04], rgba=BOOK_COLORS[0])
    mug(world, -0.2, 1.7, 0.40)
    # TV wall (west): bench, panel, glowing screen, speakers.
    add_box(world, [-1.62, 1.8, 0.225], [0.45, 1.8, 0.45], rgba=WOOD)
    add_box(world, [-1.60, 1.8, 1.15], [0.08, 1.5, 0.85], rgba=BLACK)
    add_box(world, [-1.55, 1.8, 1.15], [0.02, 1.4, 0.75], material="screen")
    add_box(world, [-1.62, 0.75, 0.65], [0.18, 0.18, 0.45], rgba=BLACK)
    add_box(world, [-1.62, 2.85, 0.65], [0.18, 0.18, 0.45], rgba=BLACK)
    # Bookshelf (east) with shelf boards, book rows, decor.
    add_box(world, [4.70, 3.0, 0.95], [0.35, 1.6, 1.9], rgba=WOOD)
    add_box(world, [4.68, 3.0, 0.53], [0.3, 1.6, 0.04], rgba=WOOD)
    add_box(world, [4.68, 3.0, 1.03], [0.3, 1.6, 0.04], rgba=WOOD)
    book_row(world, 4.50, 2.30, 0.55, 10, along="y", seed=3)
    book_row(world, 4.50, 2.30, 1.05, 8, along="y", seed=4)
    add_box(world, [4.55, 3.30, 0.65], [0.25, 0.25, 0.20], rgba=MUSTARD)
    vase = add_box(world, [4.55, 2.45, 1.70], [0.09, 0.09],
                   geom_type="cylinder", rgba=TERRA)
    vase.set("size", s([0.09, 0.30]))
    # Reading corner: armchair, side table, floor lamp.
    add_box(world, [3.8, 3.9, 0.21], [0.75, 0.75, 0.42], rgba=FABRIC)
    add_box(world, [3.8, 4.2, 0.60], [0.75, 0.20, 0.70], rgba=FABRIC)
    add_box(world, [3.8, 3.85, 0.50], [0.60, 0.60, 0.14], rgba=CREAM)
    side = add_box(world, [3.0, 4.2, 0.25], [0.25, 0.25],
                   geom_type="cylinder", rgba=WOOD)
    side.set("size", s([0.25, 0.50]))
    mug(world, 3.0, 4.2, 0.50)
    pole = add_box(world, [4.4, 4.3, 0.80], [0.03, 0.03],
                   geom_type="cylinder")
    pole.set("size", s([0.03, 1.6]))
    fshade = add_box(world, [4.4, 4.3, 1.62], [0.18, 0.18],
                     geom_type="cylinder", rgba=CREAM)
    fshade.set("size", s([0.18, 0.22]))
    add_box(world, [4.4, 4.3, 1.50], [0.12], geom_type="sphere",
            material="bulb")
    # Window, art, clock, ceiling lamp, plant.
    window(world, 1.0, 5.90, 1.5, w=2.2, h=1.3, orient="x")
    add_box(world, [-1.5, 5.90, 1.7], [1.0, 0.06, 0.8], rgba=WOOD)
    add_box(world, [-1.5, 5.885, 1.7], [0.86, 0.03, 0.66], rgba=MUSTARD)
    face = add_box(world, [-1.88, 0.2, 1.7], [0.18, 0.18],
                   geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                   rgba=WHITE)
    face.set("size", s([0.18, 0.04]))
    add_box(world, [-1.85, 0.2, 1.73], [0.015, 0.02, 0.10], rgba=BLACK)
    add_box(world, [-1.85, 0.2, 1.70], [0.015, 0.09, 0.02], rgba=BLACK)
    ceiling_lamp(world, 0.5, 2.0)
    plant(world, 4.2, -0.4)

    # --- kitchen (x in [5, 8.5], y in [-2, 4]) ---
    add_box(world, [6.75, 1.0, 0.006], [3.5, 6.0, 0.012], rgba=TILE_BLUE)
    # South counter run + uppers + light strip.
    add_box(world, [6.6, -1.65, 0.45], [3.0, 0.6, 0.9], rgba=WHITE)
    add_box(world, [6.6, -1.65, 0.925], [3.1, 0.67, 0.05], rgba=STEEL)
    add_box(world, [6.6, -1.72, 1.75], [2.4, 0.4, 0.8], rgba=WHITE)
    add_box(world, [6.6, -1.68, 1.33], [2.4, 0.3, 0.02], material="bulb")
    # Stove + oven + hood + chimney.
    add_box(world, [5.7, -1.65, 0.45], [0.7, 0.6, 0.9], rgba=DARK)
    for bx in (5.52, 5.88):
        for by in (-1.81, -1.49):
            b = add_box(world, [bx, by, 0.955], [0.09, 0.09],
                        geom_type="cylinder", rgba=BLACK)
            b.set("size", s([0.09, 0.02]))
    add_box(world, [5.7, -1.34, 0.45], [0.6, 0.02, 0.5], rgba=BLACK)
    for kx in (5.48, 5.63, 5.77, 5.92):
        k = add_box(world, [kx, -1.34, 0.78], [0.02, 0.02],
                    geom_type="cylinder", euler_quat=quat_x(math.pi / 2),
                    rgba=STEEL)
        k.set("size", s([0.02, 0.04]))
    add_box(world, [5.7, -1.65, 1.65], [0.75, 0.6, 0.15], rgba=STEEL)
    add_box(world, [5.7, -1.65, 2.05], [0.3, 0.3, 0.7], rgba=STEEL)
    # East counter run + sink + faucet.
    add_box(world, [8.1, 1.6, 0.45], [0.65, 2.2, 0.9], rgba=WHITE)
    add_box(world, [8.1, 1.6, 0.925], [0.72, 2.3, 0.05], rgba=STEEL)
    add_box(world, [8.1, 1.6, 0.93], [0.45, 0.7, 0.08], rgba=BLACK)
    f1 = add_box(world, [8.32, 1.6, 1.10], [0.025, 0.025],
                 geom_type="cylinder", rgba=STEEL)
    f1.set("size", s([0.025, 0.32]))
    f2 = add_box(world, [8.18, 1.6, 1.24], [0.02, 0.02],
                 geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                 rgba=STEEL)
    f2.set("size", s([0.02, 0.30]))
    # Microwave, kettle, fruit bowl, cutting board + tomato.
    add_box(world, [7.3, -1.65, 1.11], [0.55, 0.42, 0.32], rgba=STEEL)
    add_box(world, [7.3, -1.43, 1.11], [0.45, 0.02, 0.22], rgba=BLACK)
    ket = add_box(world, [7.8, 0.6, 1.06], [0.09, 0.09],
                  geom_type="cylinder", rgba=BLACK)
    ket.set("size", s([0.09, 0.22]))
    bowl = add_box(world, [8.1, 2.6, 0.99], [0.16, 0.16],
                   geom_type="cylinder", rgba=WHITE)
    bowl.set("size", s([0.16, 0.08]))
    for i, c in enumerate((TERRA, MUSTARD, GREEN)):
        add_box(world, [8.02 + 0.09 * i, 2.6, 1.06], [0.10],
                geom_type="sphere", rgba=c)
    add_box(world, [7.6, 2.6, 0.96], [0.4, 0.25, 0.02], rgba=WOOD)
    add_box(world, [7.6, 2.6, 1.0], [0.12], geom_type="sphere", rgba=RUG_RED)
    # Fridge + handles.
    add_box(world, [5.6, 3.4, 0.95], [0.9, 0.9, 1.9], rgba=WHITE)
    add_box(world, [5.25, 3.4, 1.10], [0.05, 0.05, 0.5], rgba=STEEL)
    add_box(world, [5.25, 3.4, 0.45], [0.05, 0.05, 0.4], rgba=STEEL)
    # Dining set for three + pendant lamp + bin + spill.
    table(world, 6.9, 1.6, 1.5, 0.95)
    chair(world, 6.9, 2.5, rot_z=-math.pi / 2)
    chair(world, 6.9, 0.7, rot_z=math.pi / 2)
    chair(world, 5.85, 1.6)
    ceiling_lamp(world, 6.9, 1.6, shade_z=1.6)
    tbin = add_box(world, [8.2, 3.6, 0.25], [0.18, 0.18],
                   geom_type="cylinder", rgba=STEEL)
    tbin.set("size", s([0.18, 0.5]))
    spill = add_box(world, [7.5, 2.5, 0.014], [0.3, 0.3],
                    geom_type="cylinder", rgba=TILE_BLUE)
    spill.set("size", s([0.3, 0.006]))

    # --- bedroom (x in [-6, -2], y in [0.5, 6]) ---
    add_box(world, [-4.55, 3.4, 0.15], [2.3, 1.8, 0.3], rgba=WOOD)    # frame
    add_box(world, [-5.85, 3.4, 0.55], [0.12, 1.8, 1.1], rgba=WOOD)   # headboard
    add_box(world, [-4.55, 3.4, 0.425], [2.2, 1.7, 0.25], rgba=WHITE)  # mattress
    add_box(world, [-3.95, 3.4, 0.59], [1.1, 1.74, 0.08], rgba=TERRA)  # blanket
    add_box(world, [-5.20, 2.95, 0.62], [0.55, 0.65, 0.15], rgba=WHITE)  # pillows
    add_box(world, [-5.20, 3.85, 0.62], [0.55, 0.65, 0.15], rgba=WHITE)
    # Nightstand + lamp + book + clock.
    add_box(world, [-2.90, 2.6, 0.275], [0.5, 0.5, 0.55], rgba=WOOD)
    nlamp = add_box(world, [-2.90, 2.6, 0.70], [0.11, 0.11],
                    geom_type="cylinder", rgba=CREAM)
    nlamp.set("size", s([0.11, 0.22]))
    add_box(world, [-2.90, 2.6, 0.86], [0.07], geom_type="sphere",
            material="bulb")
    add_box(world, [-2.90, 2.45, 0.58], [0.25, 0.18, 0.04],
            rgba=BOOK_COLORS[1])
    add_box(world, [-2.90, 2.80, 0.58], [0.20, 0.12, 0.07], rgba=BLACK)
    # Wardrobe + mirror door.
    add_box(world, [-3.4, 5.6, 1.05], [1.8, 0.6, 2.1], rgba=WOOD)
    add_box(world, [-3.8, 5.28, 1.10], [0.04, 0.03, 0.5], rgba=STEEL)
    add_box(world, [-3.0, 5.28, 1.10], [0.04, 0.03, 0.5], rgba=STEEL)
    add_box(world, [-3.85, 5.28, 1.20], [0.7, 0.02, 1.4],
            rgba=[0.85, 0.9, 0.95, 1.0])
    bordered_rug(world, -4.0, 2.0, 2.4, 1.8)
    # Walker, slippers, book, laundry basket.
    for dx in (-0.25, 0.25):
        for dy in (-0.25, 0.25):
            add_box(world, [-2.7 + dx, 4.6 + dy, 0.40], [0.04, 0.04, 0.80])
    add_box(world, [-2.7, 4.6, 0.82], [0.55, 0.55, 0.04])
    add_box(world, [-3.3, 1.6, 0.03], [0.28, 0.12, 0.06])
    add_box(world, [-3.2, 1.85, 0.03], [0.28, 0.12, 0.06])
    add_box(world, [-4.8, 1.4, 0.015], [0.30, 0.20, 0.03],
            rgba=BOOK_COLORS[3])
    bask = add_box(world, [-5.5, 1.1, 0.25], [0.28, 0.28],
                   geom_type="cylinder", rgba=CREAM)
    bask.set("size", s([0.28, 0.5]))
    window(world, -4.0, 5.90, 1.5, w=2.0, h=1.3, orient="x")
    window(world, -5.90, 3.0, 1.5, w=1.2, h=1.1, orient="y")
    add_box(world, [-2.07, 4.5, 1.6], [0.06, 1.0, 0.8], rgba=WOOD)    # art
    add_box(world, [-2.03, 4.5, 1.6], [0.02, 0.86, 0.66], rgba=TERRA)
    ceiling_lamp(world, -4.0, 3.2)

    # --- bathroom (x in [-6, -2], y in [-5, -0.5]) ---
    add_box(world, [-4.0, -2.75, 0.006], [4.0, 4.5, 0.012], rgba=TILE_BLUE)
    # Tub: solid body + dark water + faucet + shower pole, head, curtain.
    add_box(world, [-5.4, -3.2, 0.30], [0.9, 1.9, 0.60], rgba=WHITE)
    add_box(world, [-5.4, -3.2, 0.605], [0.7, 1.7, 0.02],
            rgba=[0.2, 0.35, 0.55, 1.0])
    vf = add_box(world, [-5.4, -2.32, 1.0], [0.03, 0.03],
                 geom_type="cylinder", rgba=STEEL)
    vf.set("size", s([0.03, 0.3]))
    vs = add_box(world, [-5.28, -2.32, 1.14], [0.02, 0.02],
                 geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                 rgba=STEEL)
    vs.set("size", s([0.02, 0.25]))
    spad = add_box(world, [-5.7, -2.4, 0.95], [0.03, 0.03],
                   geom_type="cylinder", rgba=STEEL)
    spad.set("size", s([0.03, 1.9]))
    shead = add_box(world, [-5.7, -2.4, 1.95], [0.09, 0.09],
                    geom_type="cylinder", rgba=STEEL)
    shead.set("size", s([0.09, 0.04]))
    rod = add_box(world, [-4.9, -3.2, 1.95], [0.02, 0.02],
                  geom_type="cylinder", euler_quat=quat_x(math.pi / 2),
                  rgba=STEEL)
    rod.set("size", s([0.02, 1.9]))
    add_box(world, [-4.9, -2.75, 1.10], [0.03, 0.9, 1.7], rgba=CREAM)
    # Toilet + tank + seat.
    add_box(world, [-3.2, -4.2, 0.20], [0.5, 0.62, 0.40], rgba=WHITE)
    add_box(world, [-3.2, -4.2, 0.43], [0.52, 0.60, 0.06], rgba=WHITE)
    add_box(world, [-3.2, -4.62, 0.65], [0.5, 0.2, 0.55], rgba=WHITE)
    # Sink + faucet + mirror (east wall).
    sped = add_box(world, [-2.35, -2.2, 0.375], [0.22, 0.22],
                   geom_type="cylinder", rgba=WHITE)
    sped.set("size", s([0.22, 0.75]))
    sbas = add_box(world, [-2.35, -2.2, 0.81], [0.3, 0.3],
                   geom_type="cylinder", rgba=WHITE)
    sbas.set("size", s([0.3, 0.12]))
    add_box(world, [-2.06, -2.2, 1.55], [0.03, 0.7, 0.9], rgba=WOOD)
    add_box(world, [-2.045, -2.2, 1.55], [0.02, 0.6, 0.8],
            rgba=[0.85, 0.9, 0.95, 1.0])
    # Grab bars: tub wall + toilet wall.
    for gx, gy in ((-5.88, -3.2), (-2.10, -3.4)):
        gb = add_box(world, [gx, gy, 0.90], [0.025, 0.025],
                     geom_type="cylinder", euler_quat=quat_x(math.pi / 2),
                     rgba=STEEL)
        gb.set("size", s([0.025, 0.8]))
    # Towel rail + towels, mat, TP, scale, frosted window, lamp.
    rail = add_box(world, [-4.0, -4.88, 1.20], [0.02, 0.02],
                   geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                   rgba=STEEL)
    rail.set("size", s([0.02, 0.7]))
    add_box(world, [-4.2, -4.86, 0.85], [0.35, 0.04, 0.6], rgba=GREEN)
    add_box(world, [-3.8, -4.86, 0.85], [0.35, 0.04, 0.6], rgba=TERRA)
    add_box(world, [-3.5, -2.6, 0.008], [0.9, 0.6, 0.015], rgba=GREEN)
    tp = add_box(world, [-2.10, -4.0, 0.80], [0.06, 0.06],
                 geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                 rgba=WHITE)
    tp.set("size", s([0.06, 0.12]))
    add_box(world, [-2.6, -1.2, 0.015], [0.35, 0.35, 0.03], rgba=BLACK)
    window(world, -5.90, -2.0, 1.6, w=1.0, h=0.9, orient="y", curtains=None,
           glass="bulb")
    ceiling_lamp(world, -4.0, -2.75)

    # --- foyer (x in [-2, 5], y in [-5, -1]) ---
    add_box(world, [1.5, -3.0, 0.006], [7.0, 4.0, 0.012], rgba=TERRA)
    # Shoe rack + shoes.
    add_box(world, [-1.5, -4.55, 0.25], [0.9, 0.35, 0.05], rgba=WOOD)
    add_box(world, [-1.5, -4.55, 0.60], [0.9, 0.35, 0.05], rgba=WOOD)
    add_box(world, [-1.9, -4.55, 0.45], [0.05, 0.35, 0.9], rgba=WOOD)
    add_box(world, [-1.1, -4.55, 0.45], [0.05, 0.35, 0.9], rgba=WOOD)
    for i, sx in enumerate((-1.7, -1.3)):
        add_box(world, [sx, -4.55, 0.325], [0.28, 0.12, 0.10], rgba=DARK)
        add_box(world, [sx, -4.55, 0.675], [0.28, 0.12, 0.10],
                rgba=BOOK_COLORS[i % len(BOOK_COLORS)])
    # Coat rack + bench + mirror.
    cbase = add_box(world, [3.6, -4.4, 0.02], [0.2, 0.2],
                    geom_type="cylinder", rgba=WOOD)
    cbase.set("size", s([0.2, 0.04]))
    cpole = add_box(world, [3.6, -4.4, 0.89], [0.025, 0.025],
                    geom_type="cylinder", rgba=WOOD)
    cpole.set("size", s([0.025, 1.7]))
    for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        add_box(world, [3.6 + 0.15 * math.cos(a), -4.4 + 0.15 * math.sin(a),
                        1.62], [0.16, 0.04, 0.04], euler_quat=quat_z(a),
                rgba=WOOD)
    add_box(world, [3.0, -2.2, 0.21], [1.2, 0.4, 0.42], rgba=WOOD)
    add_box(world, [3.0, -2.2, 0.47], [1.1, 0.35, 0.10], rgba=FABRIC)
    add_box(world, [4.90, -4.0, 1.50], [0.04, 0.9, 1.1], rgba=WOOD)
    add_box(world, [4.875, -4.0, 1.50], [0.02, 0.76, 0.96],
            rgba=[0.85, 0.9, 0.95, 1.0])
    ceiling_lamp(world, 1.5, -3.0)

    # --- hallway (x in [-6, 0.5], y in [-0.5, 0.5]) ---
    add_box(world, [-3.4, 0.0, 0.008], [4.6, 0.8, 0.015], rgba=RUG_RED)
    table(world, -4.5, 0.0, 1.0, 0.3, top_z=0.80)
    add_box(world, [-4.5, 0.0, 0.86], [0.35, 0.2, 0.06], rgba=BOOK_COLORS[2])
    vv = add_box(world, [-4.2, 0.0, 0.95], [0.07, 0.07],
                 geom_type="cylinder", rgba=TERRA)
    vv.set("size", s([0.07, 0.24]))
    add_box(world, [-3.0, -0.44, 1.60], [0.7, 0.04, 0.55], rgba=WOOD)
    add_box(world, [-3.0, -0.425, 1.60], [0.6, 0.02, 0.45], rgba=MUSTARD)
    add_box(world, [-5.0, -0.44, 1.60], [0.7, 0.04, 0.55], rgba=WOOD)
    add_box(world, [-5.0, -0.425, 1.60], [0.6, 0.02, 0.45], rgba=TERRA)
    ceiling_lamp(world, -2.75, 0.0, shade_z=2.1)

    # --- utility (x in [5, 8.5], y in [-5, -2]) ---
    add_box(world, [6.75, -3.5, 0.006], [3.5, 3.0, 0.012], rgba=STEEL)
    for wx in (6.0, 6.8):
        add_box(world, [wx, -4.5, 0.45], [0.7, 0.7, 0.9], rgba=WHITE)
        wdoor = add_box(world, [wx, -4.13, 0.45], [0.25, 0.25],
                        geom_type="cylinder", euler_quat=quat_x(math.pi / 2),
                        rgba=BLACK)
        wdoor.set("size", s([0.25, 0.03]))
        add_box(world, [wx + 0.2, -4.13, 0.70], [0.06],
                geom_type="sphere", rgba=STEEL)
    add_box(world, [7.8, -4.6, 0.80], [0.6, 0.5, 0.25], rgba=WHITE)
    add_box(world, [8.3, -3.0, 1.40], [0.3, 1.6, 0.05], rgba=WOOD)
    add_box(world, [8.3, -3.0, 1.80], [0.3, 1.6, 0.05], rgba=WOOD)
    for i, c in enumerate(BOOK_COLORS[:4]):
        add_box(world, [8.3, -3.5 + 0.4 * i, 1.55], [0.22, 0.25, 0.25],
                rgba=c)
    add_box(world, [8.3, -2.6, 1.95], [0.24, 0.5, 0.2], rgba=CREAM)
    mop = add_box(world, [5.3, -2.4, 0.70], [0.02, 0.02],
                  geom_type="cylinder", rgba=WOOD)
    mop.set("size", s([0.02, 1.4]))
    add_box(world, [5.3, -2.4, 0.05], [0.2, 0.06, 0.08], rgba=GREEN)
    ceiling_lamp(world, 6.75, -3.5)

    # --- storage (x in [8.5, 12], y in [-5, -1]) ---
    add_box(world, [9.5, -1.3, 1.0], [0.08, 0.4, 2.0], rgba=STEEL)
    add_box(world, [11.5, -1.3, 1.0], [0.08, 0.4, 2.0], rgba=STEEL)
    for sz in (0.4, 1.0, 1.6):
        add_box(world, [10.5, -1.3, sz], [2.4, 0.4, 0.06], rgba=STEEL)
    for i, c in enumerate(BOOK_COLORS):
        add_box(world, [9.6 + 0.45 * (i % 4), -1.3, 0.65 + 0.6 * (i // 4)],
                [0.4, 0.32, 0.45], rgba=c)
    for rz in (0.12, 0.34):
        rug = add_box(world, [11.3, -3.5, rz], [0.12, 0.12],
                      geom_type="cylinder", euler_quat=quat_y(math.pi / 2),
                      rgba=TERRA)
        rug.set("size", s([0.12, 1.6]))
    add_box(world, [9.2, -4.4, 0.25], [0.7, 0.25, 0.5], rgba=BOOK_COLORS[4])
    add_box(world, [9.2, -4.4, 0.62], [0.6, 0.22, 0.25], rgba=BOOK_COLORS[1])
    add_box(world, [10.6, -4.4, 0.125], [0.5, 0.25, 0.25], rgba=RUG_RED)
    cord = add_box(world, [10.2, -3.0, 2.2], [0.012, 0.012],
                   geom_type="cylinder")
    cord.set("size", s([0.012, 0.4]))
    add_box(world, [10.2, -3.0, 1.95], [0.12], geom_type="sphere",
            material="bulb")

    # --- office (x in [8.5, 12], y in [-1, 6]) ---
    bordered_rug(world, 10.2, 2.0, 2.6, 2.0)
    table(world, 11.5, 2.2, 0.7, 1.7)                                # desk
    add_box(world, [11.32, 2.2, 1.10], [0.04, 0.9, 0.55], rgba=BLACK)
    add_box(world, [11.28, 2.2, 1.07], [0.02, 0.8, 0.45], material="screen")
    add_box(world, [11.42, 2.2, 0.76], [0.12, 0.2, 0.08], rgba=BLACK)
    add_box(world, [11.05, 2.2, 0.735], [0.15, 0.45, 0.03], rgba=BLACK)
    mug(world, 11.45, 2.9, 0.72)
    # Office chair: seat + back + pole + disc.
    add_box(world, [10.6, 2.2, 0.50], [0.5, 0.5, 0.07], rgba=BLACK)
    add_box(world, [10.36, 2.2, 0.85], [0.07, 0.5, 0.60], rgba=BLACK)
    och = add_box(world, [10.6, 2.2, 0.25], [0.03, 0.03],
                  geom_type="cylinder", rgba=STEEL)
    och.set("size", s([0.03, 0.45]))
    obase = add_box(world, [10.6, 2.2, 0.03], [0.28, 0.28],
                    geom_type="cylinder", rgba=BLACK)
    obase.set("size", s([0.28, 0.06]))
    # Bookshelf + books + boxes.
    add_box(world, [10.0, 5.6, 0.95], [1.6, 0.35, 1.9], rgba=WOOD)
    add_box(world, [10.0, 5.5, 1.13], [1.6, 0.3, 0.04], rgba=WOOD)
    add_box(world, [10.0, 5.5, 0.63], [1.6, 0.3, 0.04], rgba=WOOD)
    book_row(world, 9.35, 5.40, 1.15, 10, along="x", seed=5)
    book_row(world, 9.35, 5.40, 0.65, 8, along="x", seed=6)
    add_box(world, [10.6, 5.40, 0.30], [0.4, 0.25, 0.5], rgba=MUSTARD)
    # Reading nook + windows + plant + art + lamp + bin.
    add_box(world, [9.2, 4.8, 0.21], [0.75, 0.75, 0.42], rgba=FABRIC)
    add_box(world, [9.2, 5.1, 0.60], [0.75, 0.20, 0.70], rgba=FABRIC)
    add_box(world, [9.2, 4.75, 0.50], [0.60, 0.60, 0.14], rgba=CREAM)
    npole = add_box(world, [8.8, 5.2, 0.80], [0.03, 0.03],
                    geom_type="cylinder")
    npole.set("size", s([0.03, 1.6]))
    nshade = add_box(world, [8.8, 5.2, 1.62], [0.18, 0.18],
                     geom_type="cylinder", rgba=CREAM)
    nshade.set("size", s([0.18, 0.22]))
    add_box(world, [8.8, 5.2, 1.50], [0.12], geom_type="sphere",
            material="bulb")
    table(world, 9.9, 4.5, 0.5, 0.5, top_z=0.5)
    add_box(world, [9.9, 4.5, 0.55], [0.3, 0.2, 0.04], rgba=BOOK_COLORS[0])
    window(world, 11.90, 2.5, 1.5, w=2.4, h=1.3, orient="y")
    window(world, 10.0, 5.90, 1.5, w=1.8, h=1.2, orient="x")
    plant(world, 11.5, 5.3)
    add_box(world, [8.60, 1.0, 1.6], [0.06, 1.0, 0.8], rgba=WOOD)
    add_box(world, [8.635, 1.0, 1.6], [0.02, 0.86, 0.66], rgba=TERRA)
    ceiling_lamp(world, 10.2, 2.5)
    waste = add_box(world, [11.6, 1.2, 0.15], [0.14, 0.14],
                    geom_type="cylinder", rgba=STEEL)
    waste.set("size", s([0.14, 0.3]))

    # --- balcony (x in [5, 8.5], y in [4, 6]) ---
    add_box(world, [6.75, 5.0, 0.006], [3.5, 2.0, 0.012],
            rgba=[0.55, 0.55, 0.58, 1.0])
    for rx in (5.0, 8.5):  # railings: low wall + top rail + balusters
        add_box(world, [rx, 5.0, 0.50], [0.1, 2.0, 1.0], rgba=WHITE)
        add_box(world, [rx, 5.0, 1.03], [0.14, 2.0, 0.06], rgba=WOOD)
        for i in range(6):
            add_box(world, [rx, 4.15 + 0.34 * i, 0.50], [0.05, 0.05, 0.9],
                    rgba=WHITE)
    # Bistro set.
    btop = add_box(world, [7.6, 5.2, 0.70], [0.35, 0.35],
                   geom_type="cylinder", rgba=WHITE)
    btop.set("size", s([0.35, 0.05]))
    bpole = add_box(world, [7.6, 5.2, 0.35], [0.03, 0.03],
                    geom_type="cylinder", rgba=STEEL)
    bpole.set("size", s([0.03, 0.7]))
    for sx, sy in ((7.0, 5.5), (8.2, 5.4)):
        seat = add_box(world, [sx, sy, 0.45], [0.2, 0.2],
                       geom_type="cylinder", rgba=WOOD)
        seat.set("size", s([0.2, 0.06]))
        spole = add_box(world, [sx, sy, 0.22], [0.025, 0.025],
                        geom_type="cylinder", rgba=STEEL)
        spole.set("size", s([0.025, 0.42]))
    # Planters + outdoor lamp.
    add_box(world, [5.5, 5.7, 0.15], [0.8, 0.3, 0.3], rgba=TERRA)
    add_box(world, [7.5, 5.7, 0.15], [0.8, 0.3, 0.3], rgba=TERRA)
    for px, py in ((5.3, 5.7), (5.7, 5.7), (7.3, 5.7), (7.7, 5.7)):
        add_box(world, [px, py, 0.45], [0.30], geom_type="sphere",
                rgba=GREEN)
    add_box(world, [7.6, 5.92, 1.80], [0.25, 0.06, 0.15], material="bulb")

    print("apartment: 11 zones incl. office, utility, storage, balcony, hall")
    save(tree, "scene_apartment.xml")


def make_all():
    make_rough()
    make_slope()
    make_steps()
    make_obstacles()
    make_apartment()
    print("done — scenes reference g1_29dof.xml + meshes in the same folder")


def main():
    make_all()


if __name__ == "__main__":
    main()
