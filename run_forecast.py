#!/usr/bin/env python3
"""Local dev wrapper: run scripts/forecast_daily.py from the repo root.

Until ABL-340 this file existed to work around the broken import graph: it
aliased `sys.modules['db'] = src.db` and `sys.modules['forecaster'] =
src.forecaster` so that `forecast_daily.py`'s flat `from db import ...` would
resolve. That aliasing is now actively harmful — it makes each module reachable
under two names, which is the bug ABL-340 removed — so the wrapper just runs the
script, which imports `src` as a package on its own.

`python scripts/forecast_daily.py` works directly and is the documented form.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "scripts" / "forecast_daily.py"), run_name="__main__")
