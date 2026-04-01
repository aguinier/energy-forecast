# Phase 2: ENTSO-E Cross-Border Flow & Net Position Data Collection

**Date:** 2026-04-01
**Status:** Draft
**Scope:** Extend energy-data-gathering to collect cross-border physical flows and net position
**Target repo:** `C:\Code\energy-data-gathering\`

## Context

Phase 1 added the Chronos-2 forecasting engine to the dashboard. Phase 3 will use it to forecast net position. But the database currently has no cross-border flow or net position data. This spec adds the data collection pipeline needed to feed Phase 3.

**What we're collecting:**
1. **Cross-border physical flows** (ENTSO-E document type A11) — bilateral MW between interconnected countries
2. **Realized net position** (ENTSO-E document type A25) — aggregated import/export balance per country

**Constraints:**
- Only ENTSO-E data (no Meteologica, no VPS/commercial schedules)
- Backfill from 2023-01-01 to present
- All interconnections between our 24 supported countries (including flows to/from non-supported neighbors)

## Database Schema

### `crossborder_flows` — Bilateral physical flows

```sql
CREATE TABLE crossborder_flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_from TEXT NOT NULL,
    country_to TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    flow_mw REAL NOT NULL,
    data_quality TEXT DEFAULT 'actual',
    publication_timestamp_utc TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(country_from, country_to, timestamp_utc)
);

CREATE INDEX idx_cbf_from ON crossborder_flows(country_from, timestamp_utc);
CREATE INDEX idx_cbf_to ON crossborder_flows(country_to, timestamp_utc);
CREATE INDEX idx_cbf_pair ON crossborder_flows(country_from, country_to, timestamp_utc);
```

**Storage convention:** For each of the 24 countries, we store exports from that country's perspective using `query_physical_crossborder_allborders(country, export=True)`. This means for border DE-FR, both `(DE, FR, +X)` and `(FR, DE, -X)` are stored as separate rows — no information loss, and useful for data quality cross-checks.

### `net_position` — Aggregated net position per country

```sql
CREATE TABLE net_position (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    net_position_mw REAL NOT NULL,
    data_quality TEXT DEFAULT 'actual',
    publication_timestamp_utc TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(country_code, timestamp_utc)
);

CREATE INDEX idx_np_lookup ON net_position(country_code, timestamp_utc);
```

**Sign convention:** Positive = net exporter, Negative = net importer. Matches ENTSO-E convention.

## Fetcher Architecture

### `src/fetch_crossborder_flows.py`

Uses `query_physical_crossborder_allborders(country, export=True)` from entsoe-py. This function internally queries each neighbor via `query_crossborder_flows(country, neighbor)` and returns a DataFrame with columns = neighbor names + 'sum'.

```
Pipeline flow:
  query_physical_crossborder_allborders(country, export=True)
    → DataFrame (wide: columns = neighbor names)
    → melt to long format (country_from, country_to, flow_mw)
    → resample to hourly
    → upsert to crossborder_flows table
```

Key behaviors:
- Handles `ENTSOENoDataError` silently (some borders have no data for certain periods)
- Drops the 'sum' column (we store individual bilateral flows, sum can be recomputed)
- Maps entsoe-py neighbor names (which may use bidding zone codes like 'DE_AT_LU') back to our 2-letter country codes where possible
- Per-country isolation: failure for one country doesn't block others

### `src/fetch_net_position.py`

Uses `query_net_position(country, dayahead=True)` from entsoe-py.

```
Pipeline flow:
  query_net_position(country, dayahead=True)
    → pd.Series (timestamp → MW)
    → convert to DataFrame (timestamp_utc, net_position_mw)
    → resample to hourly
    → upsert to net_position table
```

### entsoe_client.py additions

Add two wrapper methods to `ENTSOEClient`:

```python
def query_crossborder_all(self, country_code, start, end, export=True):
    """Wrapper for query_physical_crossborder_allborders with rate limiting."""
    self._rate_limit()
    return self.client.query_physical_crossborder_allborders(
        country_code, start, end, export=export
    )

def query_net_position(self, country_code, start, end, dayahead=True):
    """Wrapper for query_net_position with rate limiting."""
    self._rate_limit()
    return self.client.query_net_position(country_code, start, end, dayahead=dayahead)
```

## Backfill Strategy

### Chunked monthly processing

The backfill script processes data month-by-month with progress checkpointing:

```
for each month in [2023-01, 2023-02, ..., 2026-03]:
    for each country in SUPPORTED_COUNTRIES:
        fetch_crossborder_flows(country, month_start, month_end)
        fetch_net_position(country, month_start, month_end)
    save checkpoint(month)
```

### Resume capability

A JSON checkpoint file tracks progress:
```json
{
  "crossborder_flows": {
    "last_completed_month": "2024-06",
    "countries_completed": ["AT", "BE", ...],
    "started_at": "2026-04-01T10:00:00",
    "total_records": 145230
  },
  "net_position": {
    "last_completed_month": "2024-06",
    ...
  }
}
```

If interrupted, the script resumes from the last incomplete month.

### Estimated API volume

| Data type | Calculation | API calls |
|-----------|------------|-----------|
| Cross-border flows | 24 countries × ~5 neighbors × 39 months | ~4,680 |
| Net position | 24 countries × 39 months | 936 |
| **Total** | | **~5,616** |

At 300 requests/minute: ~19 min theoretical, ~45-60 min realistic (retries, no-data, processing).

### Daily updates

After backfill, daily updates fetch the last 7 days (same `UPDATE_DAYS_BACK` as other types). Idempotent via `INSERT OR REPLACE` on UNIQUE constraints.

## Configuration Changes

### `config.py` additions

```python
ENTSOE_API_CONFIG['crossborder_flows'] = {
    'document_type': 'A11',
    'table': 'crossborder_flows',
    'entsoe_method': 'query_physical_crossborder_allborders',
    'backfill_start': '2023-01-01',
}

ENTSOE_API_CONFIG['net_position'] = {
    'document_type': 'A25',
    'table': 'net_position',
    'entsoe_method': 'query_net_position',
    'backfill_start': '2023-01-01',
}
```

## Bidding Zone Mapping

entsoe-py uses a `NEIGHBOURS` mapping that references bidding zone names (e.g., `DE_AT_LU`, `IT_NORD`, `DK_1`). These need to be mapped back to our 2-letter country codes.

Mapping table:

| Bidding zone | Country code |
|---|---|
| DE_AT_LU, DE_LU | DE |
| IT_NORD, IT_CNOR, IT_CSUD, IT_SUD, IT_SARD, IT_SICI, IT_NORD_AT | IT |
| DK_1, DK_2 | DK (not in our 24, stored as-is) |
| SE_1, SE_2, SE_3, SE_4 | SE |
| NO_1, NO_2, NO_3, NO_4, NO_5 | NO |

**Multi-zone countries:** For countries with multiple bidding zones (IT, NO, SE, DK), flows to individual zones are aggregated to country level during the normalization step. E.g., flows to IT_NORD, IT_CSUD, IT_SUD are all stored as `country_to='IT'` with their values summed per timestamp.

**External countries:** Countries not in our 24 (e.g., GB, RS, BA, ME, MK, UA) are stored with their 2-letter code in `country_to` when identifiable, or their ENTSO-E bidding zone code otherwise. These flows affect our countries' net position even though we don't forecast them.

## Files to Create/Modify

### In `energy-data-gathering`:

| File | Action |
|------|--------|
| `src/fetch_crossborder_flows.py` | **CREATE** — fetcher for bilateral physical flows |
| `src/fetch_net_position.py` | **CREATE** — fetcher for aggregated net position |
| `src/entsoe_client.py` | **MODIFY** — add wrapper methods for cross-border queries |
| `src/db.py` | **MODIFY** — add table creation + upsert functions |
| `config.py` | **MODIFY** — add new data types to ENTSOE_API_CONFIG |
| `src/pipeline.py` | **MODIFY** — wire new fetchers into update/backfill loops |
| `scripts/backfill_crossborder.py` | **CREATE** — standalone backfill with chunking + progress |

### Database operations to add to `db.py`:

- `create_crossborder_flows_table()`
- `create_net_position_table()`
- `upsert_crossborder_flows(df, country_from)` — idempotent bilateral flow upsert
- `upsert_net_position(df, country_code)` — idempotent net position upsert

## Verification

1. Run backfill for a single country + single month:
   ```bash
   python scripts/backfill_crossborder.py --countries DE --months 2024-01
   ```
2. Check `crossborder_flows` table has rows for DE → each neighbor
3. Check `net_position` table has hourly rows for DE
4. Cross-check: `SUM(flow_mw) WHERE country_from='DE'` should approximate `net_position_mw WHERE country_code='DE'`
5. Run daily update and verify idempotency (no duplicates)
6. Verify data quality: plot DE flows over a week, check for gaps/outliers
