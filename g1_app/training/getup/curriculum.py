"""Training-side access to the stage gates (see ``core.getup_stages``).

The canonical implementation lives in ``core/getup_stages.py`` (pure stdlib)
so the deployment bridge never imports the training package. Training env
cfgs and tests import it from here.
"""

from __future__ import annotations

from core.getup_stages import *  # noqa: F401, F403
from core.getup_stages import __all__ as _stage_exports

__all__ = list(_stage_exports)
