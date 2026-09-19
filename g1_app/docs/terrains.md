# Test terrains

Five scenes in `models/g1/` (flat 2×2 m start pad at origin, features at x≥1.2 m).
Regenerate: `g1 terrains` (needs `numpy` + `imageio`).

| Terrain | File | Notes |
|---|---|---|
| flat | scene_29dof.xml | default |
| rough | scene_rough.xml + meshes/rough.png | 2–17 cm bumps, 12×12 m |
| slope | scene_slope.xml | 7° ramp to 0.33 m platform |
| steps | scene_steps.xml | 5×6 cm steps up, platform, down |
| obstacles | scene_obstacles.xml | bars + 1.4 m corridor + pillar |

Drive slowly; the policy was trained for walking, not parkour. GUI dropdown
or `g1 gui --terrain slope` / `g1 stand --terrain steps`.
