> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Module Overview

## Module Overview

D+2 energy forecasting module for European electricity markets. Generates 24-hour forecasts for the day after tomorrow.

`scripts/scheduler_setup.sh` installs `forecast_daily.py` at 18:00, but that is
not the only job: every Chronos-2 net-position run in the database was generated
at **~06:00 UTC** (8 runs at 06:00, 1 at 07:00 as of 2026-08-04), scheduled
elsewhere. `RUN_HOUR` in `compare_experiments.py` tracks that measured time,
since backtest `as_of` bounds depend on it — check it against real `generated_at`
values before trusting a backtest, rather than against this file.

**Forecast Types:**
- **Load** - Electricity demand (MW)
- **Price** - Day-ahead prices (EUR/MWh)
- **Renewable** - Total renewable generation (MW)
- **Individual Renewable Types:**
  - Solar - Solar PV generation (MW)
  - Wind Onshore - Onshore wind generation (MW)
  - Wind Offshore - Offshore wind generation (MW)
  - Hydro Total - Combined run-of-river and reservoir hydro (MW)
  - Biomass - Biomass generation (MW)
- **Net Position** - Cross-border import/export balance (MW) [Chronos-2 only]

**Coverage:** 24 European countries with complete data
