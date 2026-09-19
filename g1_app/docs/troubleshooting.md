# Troubleshooting

- **No control panel?** `g1 check` first. If that shows nothing, it's your
  display (SSH without X), not the app. Else: you ran `g1 stand` (no panel)
  instead of `g1 gui`; look behind the 3D window / taskbar (forced on top 3 s).
- **Traceback?** The app prints model+scene before windows — paste all output.
- **`libdecor` / Wayland warnings?** Harmless.
- **`OpenGL 0x502` in headless logs?** Benign.
- **Falls at extreme sliders?** Press Reset; outside
  vx[-0.5,1.0] vy[-0.5,0.5] wz[-1,1] stability isn't guaranteed.
- **Import errors after reorg?** `pip install -e ./g1_app` or run from the
  workspace root; legacy `g1_app/*.py` shims still work.
