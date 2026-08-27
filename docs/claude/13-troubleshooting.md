> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Troubleshooting

## Troubleshooting

**"Model not found"**
- Run training first: `python scripts/train.py --countries <code> --types <type>`

**"Database error"**
- Check database path exists
- Set `ENERGY_DB_PATH` environment variable if needed

**Low accuracy**
- Ensure sufficient training data (minimum 1 year recommended)
- Check for data quality issues in source tables
- Consider retraining with more recent data
