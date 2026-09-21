"""Single import bootstrap for the test suite (replaces per-file sys.path blocks).

Tests import the flat workspace packages (`core.*`, `lab.*`, `training.*`)
plus the vendored MJLab framework. Run via `pytest g1_app/tests` (or
`just test`); do not re-add sys.path boilerplate to test modules.
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_TESTS_DIR)
_WORKSPACE = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, os.path.join(_WORKSPACE, "unitree_rl_mjlab")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
