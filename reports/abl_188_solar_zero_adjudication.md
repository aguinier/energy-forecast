# ABL-188 — adjudication of the 5,096 zero-filled DE solar actuals

**Disposition:** adjudicated. The DE `energy_renewable.solar_mw` zeros are
**not** published measurements — they are missingness silently encoded as
0.0 by `energy-data-gathering`'s renewable-column mapper. Root cause traced
to `entsoe_client.py`. No database write executed; a training-data
invariant now guards the boundary where this table's data becomes a
training target. All reads below used the SQLite read-only URI
(`file:...?mode=ro` / `get_connection(readonly=True)`) against the live
replica at `C:\Code\able\data\energy_dashboard.db` (5.90 GB). No write of
any kind touched the replica or any sidecar database.

## Verdict

**Zero-filled missingness, not a measured zero.** Four independent pieces
of read-only evidence, strongest first:

1. **The sibling, NaN-preserving table has the real numbers for the same
   fetch, same country, same timestamps.** `energy_generation` and
   `energy_renewable` are derived from one A75 ENTSO-E fetch per
   country/window (`fetch_renewable.py:56-77`, `entsoe_client.py:1339-1462`
   — the "no second upstream fetch" invariant holds). For DE at
   `2025-09-15 11:00:00`, `energy_generation.solar_mw = 30,589.5 MW`; at
   `2025-10-13 11:30:00`, `19,869.8 MW`; at `2025-11-01 12:00:00`,
   `10,463.1 MW`. `energy_renewable.solar_mw` at the identical
   country/timestamps is `0.0`. Across the whole window, `energy_generation`
   has **0 NULLs and 0 zeros** in DE solar_mw (n=6,432, max 43.4 GW);
   `energy_renewable` has **6,408 of 6,408 exactly 0.0**. The data ENTSO-E
   published was captured correctly by one mapper and lost by the other, in
   the same fetch.
2. **Solar is physically incapable of reading exactly 0.0 at every solar
   noon for 68 straight days.** Germany's solar fleet was well over 90 GW of
   capacity in this period; the sampled midday points above prove real
   generation was tens of GW throughout September–November.
3. **The batch signature is a single backfill, not live ingestion.** Every
   one of the 6,408 DE rows shares one `fetched_at = 2025-11-20 09:24:12` —
   a single write, roughly 6–68 days after the timestamps it covers.
   Genuine near-real-time ingestion for this table shows `fetched_at`
   advancing every few hours (confirmed on the rows immediately
   after: `2025-11-21 15:51:55`, `16:45:44`, `17:45:44`, …).
4. **Only `solar_mw` is zero; the row's other columns are not.** Within the
   same backfill batch, `wind_onshore_mw`, `hydro_reservoir_mw`, and
   `biomass_mw` are **never** 0 across all 6,408 rows (real, varying values
   — e.g. `wind_onshore_mw = 15,294.0` and `422.5` at the two sampled
   timestamps above). This rules out "the whole row is a missing-fetch
   placeholder"; the defect is specific to how the renewable mapper handles
   the `Solar` production type.

## Scope correction: the window is larger than 5,096 rows

The parent diagnosis (ABL-185) named 5,096 rows, 2025-09-08 22:00 through
2025-10-31 23:45 UTC. Read-only census against the current replica shows the
**same exact-zero, same-`fetched_at` run actually extends to 2025-11-14
15:45 UTC — 6,408 quarter-hour rows**, not 5,096. (My first pass at this
census undercounted at 5,000/5,096 rows due to a `T`- vs space-separated
timestamp comparison bug in my own query against `energy_renewable`'s
`T`-separated `timestamp_utc` strings; corrected and re-verified below.)
5,096 was very likely a correct count against an *earlier* replica state
before 2025-11-01..11-14 had synced — this report supersedes it with the
current, full extent. The first real (non-zero, differently-`fetched_at`)
DE solar row after the run is `2025-11-14 16:00:00` = `0.02 MW`,
`fetched_at = 2025-11-21 15:51:55`.

## Before/after read-only census evidence

All queries below ran read-only against
`C:\Code\able\data\energy_dashboard.db` via `sqlite3 "file:...?mode=ro"` or
`energy_forecast.src.db.get_connection(readonly=True)`.

### Before — raw `energy_renewable` (no invariant)

```
SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc),
       SUM(solar_mw = 0.0), SUM(solar_mw IS NULL)
FROM energy_renewable
WHERE country_code='DE'
  AND timestamp_utc >= '2025-09-08T22:00:00'
  AND timestamp_utc <= '2025-11-14T15:45:00';

  -> 6408 rows | 2025-09-08T22:00:00 | 2025-11-14T15:45:00 | zero=6408 | null=0
```

```
SELECT fetched_at, COUNT(*) FROM energy_renewable
WHERE country_code='DE' AND timestamp_utc BETWEEN
  '2025-09-08T22:00:00' AND '2025-11-14T15:45:00'
GROUP BY fetched_at;

  -> 2025-11-20 09:24:12 | 6408   (single batch, no other fetched_at value)
```

```
SELECT COUNT(*), SUM(solar_mw IS NULL), SUM(solar_mw=0), MAX(solar_mw)
FROM energy_generation
WHERE country_code='DE' AND timestamp_utc BETWEEN
  '2025-09-08T22:00:00' AND '2025-11-14T15:45:00';

  -> 6432 rows | null=0 | zero=0 | max=43408.53 MW   (the sibling table, same fetch)
```

Live end-to-end read through the (pre-fix) loader,
`load_renewable_type_data('DE', 'solar', '2025-09-01', '2025-12-01')` /
`SELECT timestamp_utc, solar_mw FROM energy_renewable WHERE country_code='DE' ...`
(raw, bypassing the new invariant): **9,176 rows, 7,982 of them exactly
0.0** (the 6,408-row anomaly plus ordinary nighttime zeros elsewhere in the
window).

### After — same window, through the fixed loader

```
$ ENERGY_DB_PATH="C:\Code\able\data\energy_dashboard.db" .venv/Scripts/python.exe -c "
from src.db import load_renewable_type_data
df = load_renewable_type_data('DE', 'solar', '2025-09-01', '2025-12-01')
print(len(df), df['target_value'].isna().sum(), df['target_value'].notna().sum())
"

WARNING training-data invariant (ABL-188): excluding suspect constant run
  target_value=0 from 2025-09-08 22:00:00 to 2025-11-14 15:45:00
  (6408 rows, 1601.8h) -- held one exact value too long to be a real
  measurement; treated as unadjudicated-missing, not zero. [DE/solar]
WARNING Excluded 6408 suspect-constant solar rows for DE

9176 rows total | 6408 nulled (excluded) | 2768 non-null retained
```

The excluded count (6,408) matches the independently-measured anomaly
exactly. `load_training_data`'s existing hourly resample + `.dropna()`
(`db.py:559-562`) then drops these nulled rows from training the same way
it already drops any other genuinely-missing interval — no downstream
change was needed to make the exclusion take effect.

### Spot checks — no false positives on healthy data

| country | window | rows | excluded by invariant |
|---|---|---:|---:|
| DE | 2025-12-01 – 2026-01-01 (post-anomaly) | 2,976 | 0 |
| FR | 2025-09-01 – 2025-12-01 | 10,991 | 0 |
| BE | 2025-09-01 – 2025-12-01 | 2,460 | **15** (separate, smaller finding below) |

BE also has a secondary, much smaller run flagged: `2025-11-08 17:00` to
`2025-11-10 06:00` (15 rows, 37h) — inside the *same* 2025-11-20 09:24:12
backfill batch that touched AT/BE/DE/FR, but for BE it is a short,
plausible ingestion gap rather than a multi-week fabrication. Not in
ABL-188's stated scope (DE); flagging here since the invariant surfaced it
for free and it should get the same adjudication treatment before any BE
solar retraining. No artifact currently depends on it — BE's CatBoost
solar artifact predates this window (ABL-185: saved 2026-02-01).

## Root cause: `entsoe_client.py`, the renewable-column mapper

`query_generation_and_renewable_with_metadata` (`entsoe_client.py:1339`)
fetches the A75 document once and derives two frames from it:

- `_map_generation_columns` → `energy_generation.solar_mw`: **never**
  `fillna(0)`s; a production type absent from the response stays `NaN` →
  SQL `NULL` (documented at `entsoe_client.py:1663-1671`, and it is
  faithful to that in this window — see census above).
- `_map_renewable_columns` → `energy_renewable.solar_mw`
  (`entsoe_client.py:1607-1655`): **initialises every renewable column to
  0.0 up front** (`entsoe_client.py:1621-1624`, `result[col] = 0.0`) before
  checking whether the source frame even contains that production type,
  and uses `.fillna(0)` on whatever it finds (`entsoe_client.py:1648,1650`).
  If `'Solar'` is absent from — or thin/NaN-heavy in — the per-window
  flattened frame `_flatten_prefer_aggregated` produces, the column simply
  stays at its 0.0 initial value for every row in that window, with no
  signal that anything was missing. `data_quality` is still written as
  `'actual'` (confirmed: all 6,408 rows say `data_quality='actual'`).

This is the same class of bug CLAUDE.md's rule 1 (`NULL is not 0`) already
names for `energy_generation` — except `_map_renewable_columns` is exactly
the code path that rule was written to rule out, and it doesn't have the
protection its sibling does. `energy_renewable` is frozen (cannot be
retired or re-derived — CEO approval required per repo boundaries), so this
report does not propose changing `_map_renewable_columns`; see remediation
below for what is proposed instead.

I was not able to determine, from the stored data alone, *why* ENTSO-E's
response lacked a usable `Solar`/`Actual Aggregated` series for DE specifically
across this window while `energy_generation`'s netted flatten still saw it —
both flattens read the same fetched `df`, so the divergence lives inside
`_flatten_prefer_aggregated` vs `_net_generation_consumption`'s handling of
that particular MultiIndex shape, or in some property of the 2025-11-20
backfill run itself (e.g. chunking, retry, or partial-response handling
specific to that script). This needs an `energy-data-gathering` owner with
access to the original backfill run's logs/request history; flagging as an
open question rather than guessing further.

## Remediation proposal for the 6,408 stored rows (not executed)

No production or replica write is authorized by this report. Proposed,
for a separate approved change:

1. **Do not overwrite `energy_renewable.solar_mw` in place.** The table has
   `solar_mw REAL DEFAULT 0` and no way to represent "unknown" without a
   schema change (out of bounds for this issue — would need CEO approval).
   Overwriting 0.0 with a re-fetched or interpolated value would also violate
   "never extrapolate."
2. **Preferred: re-fetch DE solar for 2025-09-08 22:00–2025-11-14 15:45 UTC
   from ENTSO-E** through the existing, already-reviewed
   `fetch_renewable_data` path (`fetch_renewable.py`) — a normal supplemental
   backfill for a known-bad window, not a second upstream fetch for an
   already-good one (does not violate "one A75 fetch per window" — this
   window's one fetch produced a defect and needs replacing, not doubling).
   This is an `energy-data-gathering`-owned action; propose it there as a
   scoped backfill re-run, with the resulting `fetched_at` distinct from
   `2025-11-20 09:24:12` serving as its own before/after proof.
3. **If re-fetch is unavailable or ENTSO-E has no data for part of the
   window** (e.g. a genuine publication gap), the correct value is `NULL`,
   not a guess — which requires the schema change in point 1, or, short of
   that, a `data_quality` value other than `'actual'` (e.g. `'suspect'`) so
   readers can filter it out without a schema change. Either option needs
   sign-off beyond this issue.
4. **Until either fix lands, do not retrain the DE solar artifact.** The
   invariant below already blocks this specific 6,408-row run from being
   silently used, so this is a process point, not a code gap: retraining
   before point 2/3 lands would produce an artifact trained on a smaller,
   correctly-gapped dataset, not a wrong one — acceptable, but the CEO
   should decide whether to wait for the re-fetch first, since more data is
   strictly better for a level-fit failure this severe (ABL-185: DE's
   stored mean forecast is 38.6% of mean actual).

## Training-data invariant / provenance witness (implemented, this issue)

`energy-forecast/src/data_quality.py` (new): `find_suspect_constant_runs` /
`exclude_suspect_constant_runs`. A live, weather- or dispatch-driven
generation series does not hold one bit-identical float for a full day;
any run that does (0.0 included — the check is not solar- or
zero-specific, so it also catches a future defect that zero-fills a
*different* constant) is treated as unadjudicated-missing and nulled out
before it can enter a resample/train step. Wired into
`load_renewable_type_data` (`db.py:280`), the single choke point all five
individual-renewable-type training targets (`solar`, `wind_onshore`,
`wind_offshore`, `hydro_total`, `biomass`) pass through — this covers the
whole `energy_renewable`-sourced training surface `_map_renewable_columns`
can affect, not just solar. Nulled rows are dropped by
`load_training_data`'s existing `.dropna()` after the hourly resample
(`db.py:559-562`), so a run like this now produces a correctly *smaller*
training set instead of a corrupted one, and a loud `logger.warning` names
the exact excluded window every time it fires — the provenance witness this
issue asked for. Colocated tests: `tests/test_data_quality.py` (7 cases:
normal diurnal solar passes untouched, the exact ABL-188 shape is caught,
an ordinary single night is not a false positive, sub-threshold runs pass
through, pre-existing NaN isn't double-counted, a non-zero constant is
caught too, and only the in-run rows are nulled — surrounding real values
survive). `pytest tests/` — **156 passed** (149 pre-existing + 7 new), full
suite, no regressions.

This is a training-data-boundary guard, not an artifact-metadata fix.
ABL-185's separate finding that CatBoost artifacts store no training
window/data-digest (so exact inclusion of these rows in the current DE
artifact is still unprovable) is unchanged by this issue and remains
owned by the Forecasting Scientist step ABL-185 already named.

## Recommended next steps

1. **Founding Engineer / energy-data-gathering owner:** investigate why
   `_flatten_prefer_aggregated` lost the `Solar` series for DE across this
   specific window while `_net_generation_consumption` kept it, and why the
   2025-11-20 09:24:12 backfill run wrote `data_quality='actual'` for a
   result it never validated against the sibling `energy_generation` write
   in the same call. Not attempted here — needs the backfill script's own
   run history/logs, which this issue's read-only DB access doesn't cover.
2. **energy-data-gathering owner (separate approval):** execute the
   preferred re-fetch in "Remediation proposal" point 2 for DE 2025-09-08
   22:00–2025-11-14 15:45 UTC, and the BE 2025-11-08 17:00–11-10 06:00 window
   found above.
3. **CEO:** decide whether DE solar retraining should wait for that
   re-fetch, given the severity of the existing level failure (ABL-185).
4. **Forecasting Scientist (per ABL-185, unchanged):** build the fresh
   solar artifact with training-window/data-digest provenance only after
   1–2 above (and ABL-183's serve-faithful feature builder) land.
