> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Quick Start

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models for all countries
python scripts/train.py --countries all --types all

# Generate D+2 forecasts
python scripts/forecast_daily.py

# Setup daily cron job (18:00)
bash scripts/scheduler_setup.sh
```
