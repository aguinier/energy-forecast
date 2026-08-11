#!/usr/bin/env python3
"""Local dev wrapper to run forecast_daily.py with correct imports."""
import sys, os
from pathlib import Path

os.chdir(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

# Patch the script's imports to use package-style
import importlib.util
spec = importlib.util.spec_from_file_location("forecast_daily", "scripts/forecast_daily.py",
    submodule_search_locations=[])

# Monkey-patch: make 'from forecaster import' resolve to 'from src.forecaster import'
import src.forecaster as forecaster_mod
import src.db as db_mod
sys.modules['forecaster'] = forecaster_mod
sys.modules['db'] = db_mod

# Now exec the script
exec(open("scripts/forecast_daily.py").read())
