> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Evaluation Queries

## Evaluation Queries

```sql
-- Compare forecasts vs actuals for load
SELECT
    f.target_timestamp_utc,
    f.forecast_value AS predicted,
    l.load_mw AS actual,
    ABS(f.forecast_value - l.load_mw) AS error
FROM forecasts f
JOIN energy_load l
    ON f.country_code = l.country_code
    AND f.target_timestamp_utc = l.timestamp_utc
WHERE f.forecast_type = 'load'
    AND f.country_code = 'DE'
ORDER BY f.target_timestamp_utc DESC
LIMIT 24;

-- Forecast accuracy summary by country
SELECT
    f.country_code,
    f.forecast_type,
    COUNT(*) as forecasts,
    AVG(ABS(f.forecast_value - l.load_mw)) as avg_mae
FROM forecasts f
JOIN energy_load l
    ON f.country_code = l.country_code
    AND f.target_timestamp_utc = l.timestamp_utc
WHERE f.forecast_type = 'load'
GROUP BY f.country_code, f.forecast_type;
```
