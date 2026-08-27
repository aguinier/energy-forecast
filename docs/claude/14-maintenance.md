> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Maintenance

## Maintenance

**Weekly:** Retrain models with latest data
```bash
python scripts/train.py --countries all --types all
```

**Monitor logs:**
```bash
tail -f logs/daily_*.log
```

**Check cron job:**
```bash
crontab -l | grep forecast
```
