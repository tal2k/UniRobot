"""10-second GUI environment check: tkinter only, no MuJoCo, no policy.

Canonical home (moved from legacy check_gui.py). Prefer:

    g1 check
"""
import sys
import tkinter as tk


def main():
    print("tkinter version:", tk.TkVersion, flush=True)
    root = tk.Tk()
    root.title("GUI check")
    root.geometry("320x120+40+40")
    tk.Label(root, text="If you see this window,\ntkinter works on your machine.").pack(
        expand=True)
    root.deiconify()
    root.lift()
    print("window created, entering mainloop (auto-close in 10 s)...", flush=True)
    root.after(10000, root.destroy)
    root.mainloop()
    print("OK: mainloop ran and exited cleanly", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
