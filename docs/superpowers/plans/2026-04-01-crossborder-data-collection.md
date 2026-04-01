# Cross-Border Flow & Net Position Data Collection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend energy-data-gathering to collect ENTSO-E cross-border physical flows and realized net position for all 24 supported countries.

**Architecture:** Two new fetchers (`fetch_crossborder_flows.py`, `fetch_net_position.py`) follow the existing fetch pattern (client → normalize → upsert). A standalone backfill script handles the 2023-01 to present backfill with monthly chunking and resume. Both data types integrate into the existing pipeline for daily updates.

**Tech Stack:** Python, entsoe-py (already installed), SQLite, pandas

**Target repo:** `C:\Code\energy-data-gathering\`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/db.py` | MODIFY | Add table creation + upsert for crossborder_flows and net_position |
| `src/entsoe_client.py` | MODIFY | Add wrapper methods for cross-border and net position queries |
| `src/fetch_crossborder_flows.py` | CREATE | Fetcher for bilateral physical flows via all-borders aggregation |
| `src/fetch_net_position.py` | CREATE | Fetcher for realized net position per country |
| `config.py` | MODIFY | Add new data types to ENTSOE_API_CONFIG and BACKFILL_DEFAULTS |
| `src/pipeline.py` | MODIFY | Wire new fetchers into _fetch_data_chunk dispatch |
| `scripts/backfill_crossborder.py` | CREATE | Standalone monthly-chunked backfill with progress tracking |

---

### Task 1: Database schema — table creation functions

**Files:**
- Modify: `C:\Code\energy-data-gathering\src\db.py`

- [ ] **Step 1: Add `create_crossborder_flows_table()` to db.py**

Add after the existing `create_*_table()` functions (around line 150):

```python
def create_crossborder_flows_table():
    """Create crossborder_flows table for bilateral physical flow data."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crossborder_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_from TEXT NOT NULL,
                country_to TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                flow_mw REAL NOT NULL,
                data_quality TEXT DEFAULT 'actual',
                publication_timestamp_utc TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(country_from, country_to, timestamp_utc)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cbf_from
            ON crossborder_flows(country_from, timestamp_utc)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cbf_to
            ON crossborder_flows(country_to, timestamp_utc)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cbf_pair
            ON crossborder_flows(country_from, country_to, timestamp_utc)
        """)
        conn.commit()
    logger.info("crossborder_flows table created/verified")
```

- [ ] **Step 2: Add `create_net_position_table()` to db.py**

Add directly after `create_crossborder_flows_table()`:

```python
def create_net_position_table():
    """Create net_position table for aggregated net position data."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS net_position (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                net_position_mw REAL NOT NULL,
                data_quality TEXT DEFAULT 'actual',
                publication_timestamp_utc TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(country_code, timestamp_utc)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_np_lookup
            ON net_position(country_code, timestamp_utc)
        """)
        conn.commit()
    logger.info("net_position table created/verified")
```

- [ ] **Step 3: Verify tables can be created**

Run from the energy-data-gathering directory:
```bash
cd C:/Code/energy-data-gathering
python -c "from src import db; db.create_crossborder_flows_table(); db.create_net_position_table(); print('OK')"
```
Expected: `OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/db.py
git commit -m "feat: add crossborder_flows and net_position table creation"
```

---

### Task 2: Database upsert functions

**Files:**
- Modify: `C:\Code\energy-data-gathering\src\db.py`

- [ ] **Step 1: Add `upsert_crossborder_flows()` to db.py**

Add after the existing upsert functions:

```python
def upsert_crossborder_flows(
    df: pd.DataFrame,
    country_from: str,
) -> Tuple[int, int]:
    """
    Insert or update cross-border flow data.

    Args:
        df: DataFrame with columns: country_to, timestamp_utc, flow_mw
        country_from: ISO 2-letter country code (exporting country)

    Returns:
        Tuple of (records_affected, 0)
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for crossborder flows from {country_from}")
        return 0, 0

    records_affected = 0

    with get_connection() as conn:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            ts_str = utils.format_timestamp_for_db(row["timestamp_utc"]) if pd.notna(row["timestamp_utc"]) else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO crossborder_flows
                (country_from, country_to, timestamp_utc, flow_mw,
                 data_quality, fetched_at)
                VALUES (?, ?, ?, ?, 'actual', CURRENT_TIMESTAMP)
                """,
                (
                    country_from,
                    row["country_to"],
                    ts_str,
                    float(row["flow_mw"]) if pd.notna(row["flow_mw"]) else None,
                ),
            )
            records_affected += cursor.rowcount

    logger.info(f"Upserted {records_affected} crossborder flow records from {country_from}")
    return records_affected, 0
```

- [ ] **Step 2: Add `upsert_net_position()` to db.py**

```python
def upsert_net_position(
    df: pd.DataFrame,
    country_code: str,
) -> Tuple[int, int]:
    """
    Insert or update net position data.

    Args:
        df: DataFrame with columns: timestamp_utc, net_position_mw
        country_code: ISO 2-letter country code

    Returns:
        Tuple of (records_affected, 0)
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for net position, country {country_code}")
        return 0, 0

    records_affected = 0

    with get_connection() as conn:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            ts_str = utils.format_timestamp_for_db(row["timestamp_utc"]) if pd.notna(row["timestamp_utc"]) else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO net_position
                (country_code, timestamp_utc, net_position_mw,
                 data_quality, fetched_at)
                VALUES (?, ?, ?, 'actual', CURRENT_TIMESTAMP)
                """,
                (
                    country_code,
                    ts_str,
                    float(row["net_position_mw"]) if pd.notna(row["net_position_mw"]) else None,
                ),
            )
            records_affected += cursor.rowcount

    logger.info(f"Upserted {records_affected} net position records for {country_code}")
    return records_affected, 0
```

- [ ] **Step 3: Commit**

```bash
git add src/db.py
git commit -m "feat: add upsert functions for crossborder flows and net position"
```

---

### Task 3: ENTSO-E client wrapper methods

**Files:**
- Modify: `C:\Code\energy-data-gathering\src\entsoe_client.py`

- [ ] **Step 1: Add bidding zone → country code mapping constant**

Add near the top of the file, after imports:

```python
# Bidding zone to country code mapping for multi-zone countries
# Used to normalize entsoe-py neighbor names to 2-letter codes
BIDDING_ZONE_TO_COUNTRY = {
    "DE_AT_LU": "DE", "DE_LU": "DE",
    "IT_NORD": "IT", "IT_CNOR": "IT", "IT_CSUD": "IT", "IT_SUD": "IT",
    "IT_SARD": "IT", "IT_SICI": "IT", "IT_NORD_AT": "IT",
    "IT_NORD_FR": "IT", "IT_NORD_SI": "IT", "IT_NORD_CH": "IT",
    "IT_BRNN": "IT", "IT_FOGN": "IT", "IT_GR": "IT",
    "IT_PRGP": "IT", "IT_ROSN": "IT", "IT_CALA": "IT",
    "DK_1": "DK", "DK_2": "DK",
    "SE_1": "SE", "SE_2": "SE", "SE_3": "SE", "SE_4": "SE",
    "NO_1": "NO", "NO_2": "NO", "NO_3": "NO", "NO_4": "NO", "NO_5": "NO",
    "GB": "GB", "GB_NIR": "GB",
    "IE_SEM": "IE",
}


def normalize_zone_to_country(zone_name: str) -> str:
    """Map an entsoe-py bidding zone name to a 2-letter country code.

    Args:
        zone_name: Bidding zone name (e.g., 'DE_AT_LU', 'IT_NORD', 'FR')

    Returns:
        2-letter country code. If no mapping found, returns the zone name as-is
        (for external zones like 'RS', 'BA', etc. which are already 2-letter).
    """
    if zone_name in BIDDING_ZONE_TO_COUNTRY:
        return BIDDING_ZONE_TO_COUNTRY[zone_name]
    # If it's already a 2-letter code, return as-is
    if len(zone_name) == 2 and zone_name.isalpha():
        return zone_name
    # Unknown zone — return as-is for traceability
    return zone_name
```

- [ ] **Step 2: Add `query_crossborder_all()` method to ENTSOEClient**

Add as a new method on the `ENTSOEClient` class:

```python
def query_crossborder_all(
    self,
    country_code: str,
    start: datetime,
    end: datetime,
    export: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Query all cross-border physical flows for a country.

    Uses entsoe-py's query_physical_crossborder_allborders which
    internally queries each neighbor and returns a wide DataFrame.

    Args:
        country_code: ISO 2-letter country code
        start: Start datetime (UTC)
        end: End datetime (UTC)
        export: True for exports from country, False for imports

    Returns:
        DataFrame with columns = neighbor zone names + 'sum',
        index = timestamps. None if no data.
    """
    try:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        self._rate_limit()
        df = self.client.query_physical_crossborder_allborders(
            country_code, start=start_ts, end=end_ts, export=export
        )

        if df is None or df.empty:
            logger.warning(f"No crossborder flow data for {country_code}")
            return None

        logger.info(
            f"Retrieved crossborder flows for {country_code} "
            f"({'export' if export else 'import'}): "
            f"{len(df)} rows, {len(df.columns)} borders"
        )
        return df

    except Exception as e:
        if "No matching data" in str(e) or "NoMatchingDataError" in type(e).__name__:
            logger.warning(f"No crossborder data for {country_code}: {e}")
            return None
        raise
```

- [ ] **Step 3: Add `query_net_position_data()` method to ENTSOEClient**

```python
def query_net_position_data(
    self,
    country_code: str,
    start: datetime,
    end: datetime,
    dayahead: bool = True,
) -> Optional[pd.Series]:
    """
    Query realized net position for a country.

    Args:
        country_code: ISO 2-letter country code
        start: Start datetime (UTC)
        end: End datetime (UTC)
        dayahead: True for day-ahead, False for intraday

    Returns:
        pd.Series with timestamp index and MW values. None if no data.
    """
    try:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        self._rate_limit()
        series = self.client.query_net_position(
            country_code, start=start_ts, end=end_ts, dayahead=dayahead
        )

        if series is None or series.empty:
            logger.warning(f"No net position data for {country_code}")
            return None

        logger.info(f"Retrieved {len(series)} net position records for {country_code}")
        return series

    except Exception as e:
        if "No matching data" in str(e) or "NoMatchingDataError" in type(e).__name__:
            logger.warning(f"No net position data for {country_code}: {e}")
            return None
        raise
```

- [ ] **Step 4: Commit**

```bash
git add src/entsoe_client.py
git commit -m "feat: add crossborder flow and net position query methods to ENTSO-E client"
```

---

### Task 4: Cross-border flows fetcher

**Files:**
- Create: `C:\Code\energy-data-gathering\src\fetch_crossborder_flows.py`

- [ ] **Step 1: Create the fetcher file**

```python
"""
Fetch cross-border physical flow data from ENTSO-E.

Uses query_physical_crossborder_allborders() which returns a wide DataFrame
with one column per neighbor. Normalizes to long format (country_from,
country_to, flow_mw) for storage.
"""

import logging
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd

from src import db, utils
from src.entsoe_client import ENTSOEClient, normalize_zone_to_country

logger = logging.getLogger("energy_data_gathering.fetch_crossborder_flows")


def _normalize_wide_to_long(
    wide_df: pd.DataFrame,
    country_from: str,
) -> pd.DataFrame:
    """Convert wide DataFrame (columns=neighbors) to long format.

    Args:
        wide_df: DataFrame from query_physical_crossborder_allborders,
                 index=timestamps, columns=neighbor zone names + 'sum'
        country_from: ISO 2-letter code of the exporting country

    Returns:
        DataFrame with columns: timestamp_utc, country_to, flow_mw
    """
    # Drop the 'sum' column if present — we store individual bilateral flows
    cols_to_melt = [c for c in wide_df.columns if c != "sum"]

    if not cols_to_melt:
        return pd.DataFrame(columns=["timestamp_utc", "country_to", "flow_mw"])

    # Reset index to get timestamps as a column
    df = wide_df[cols_to_melt].copy()
    df.index.name = "timestamp_utc"
    df = df.reset_index()

    # Ensure timestamps are tz-aware UTC
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Melt: wide → long
    long_df = df.melt(
        id_vars=["timestamp_utc"],
        var_name="zone_name",
        value_name="flow_mw",
    )

    # Drop NaN flow values
    long_df = long_df.dropna(subset=["flow_mw"])

    # Map bidding zone names to 2-letter country codes
    long_df["country_to"] = long_df["zone_name"].apply(normalize_zone_to_country)

    # Aggregate multiple zones for same country (e.g., IT_NORD + IT_CSUD → IT)
    long_df = (
        long_df.groupby(["timestamp_utc", "country_to"], as_index=False)["flow_mw"]
        .sum()
    )

    # Resample to hourly (some data may be 15-min or 30-min)
    if not long_df.empty:
        result_frames = []
        for country_to, group in long_df.groupby("country_to"):
            hourly = group.set_index("timestamp_utc")["flow_mw"].resample("h").mean()
            hourly_df = hourly.reset_index()
            hourly_df["country_to"] = country_to
            result_frames.append(hourly_df)
        long_df = pd.concat(result_frames, ignore_index=True)

    return long_df[["timestamp_utc", "country_to", "flow_mw"]]


def fetch_crossborder_flows_data(
    client: ENTSOEClient,
    country_code: str,
    start: datetime,
    end: datetime,
    log_id: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Fetch and store cross-border physical flows for a country (exports).

    Args:
        client: ENTSO-E client instance
        country_code: ISO 2-letter country code
        start: Start datetime (UTC)
        end: End datetime (UTC)
        log_id: Optional ingestion log ID

    Returns:
        Tuple of (records_inserted, records_updated, records_failed)
    """
    logger.info(f"Fetching crossborder flows for {country_code}: {start.date()} to {end.date()}")

    try:
        # Ensure tables exist
        db.create_crossborder_flows_table()

        # Query all borders (exports from this country)
        wide_df = client.query_crossborder_all(country_code, start, end, export=True)

        if wide_df is None or wide_df.empty:
            logger.warning(f"No crossborder flow data for {country_code}")
            return 0, 0, 0

        # Normalize wide → long format
        long_df = _normalize_wide_to_long(wide_df, country_code)

        if long_df.empty:
            logger.warning(f"No valid flow data after normalization for {country_code}")
            return 0, 0, 0

        # Upsert to database
        records_inserted, _ = db.upsert_crossborder_flows(long_df, country_code)

        logger.info(
            f"Stored {records_inserted} crossborder flow records for {country_code} "
            f"({long_df['country_to'].nunique()} neighbors)"
        )
        return records_inserted, 0, 0

    except Exception as e:
        logger.error(f"Error fetching crossborder flows for {country_code}: {e}")
        if log_id:
            db.log_ingestion_complete(log_id, records_failed=1, error_message=str(e))
        return 0, 0, 1
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/fetch_crossborder_flows.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/fetch_crossborder_flows.py
git commit -m "feat: add cross-border physical flow fetcher"
```

---

### Task 5: Net position fetcher

**Files:**
- Create: `C:\Code\energy-data-gathering\src\fetch_net_position.py`

- [ ] **Step 1: Create the fetcher file**

```python
"""
Fetch realized net position data from ENTSO-E.

Net position = aggregated import/export balance per country.
Positive = net exporter, Negative = net importer.
"""

import logging
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd

from src import db, utils
from src.entsoe_client import ENTSOEClient

logger = logging.getLogger("energy_data_gathering.fetch_net_position")


def fetch_net_position_data(
    client: ENTSOEClient,
    country_code: str,
    start: datetime,
    end: datetime,
    log_id: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Fetch and store net position data for a country.

    Args:
        client: ENTSO-E client instance
        country_code: ISO 2-letter country code
        start: Start datetime (UTC)
        end: End datetime (UTC)
        log_id: Optional ingestion log ID

    Returns:
        Tuple of (records_inserted, records_updated, records_failed)
    """
    logger.info(f"Fetching net position for {country_code}: {start.date()} to {end.date()}")

    try:
        # Ensure table exists
        db.create_net_position_table()

        # Query net position
        series = client.query_net_position_data(country_code, start, end, dayahead=True)

        if series is None or series.empty:
            logger.warning(f"No net position data for {country_code}")
            return 0, 0, 0

        # Convert Series to DataFrame
        df = series.to_frame(name="net_position_mw")
        df.index.name = "timestamp_utc"
        df = df.reset_index()

        # Ensure timestamps are tz-aware UTC
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Resample to hourly
        df = df.set_index("timestamp_utc")
        df = df["net_position_mw"].resample("h").mean().reset_index()

        # Drop NaN
        df = df.dropna(subset=["net_position_mw"])

        if df.empty:
            logger.warning(f"No valid net position data after resampling for {country_code}")
            return 0, 0, 0

        # Upsert to database
        records_inserted, _ = db.upsert_net_position(df, country_code)

        logger.info(f"Stored {records_inserted} net position records for {country_code}")
        return records_inserted, 0, 0

    except Exception as e:
        logger.error(f"Error fetching net position for {country_code}: {e}")
        if log_id:
            db.log_ingestion_complete(log_id, records_failed=1, error_message=str(e))
        return 0, 0, 1
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/fetch_net_position.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/fetch_net_position.py
git commit -m "feat: add net position fetcher"
```

---

### Task 6: Configuration updates

**Files:**
- Modify: `C:\Code\energy-data-gathering\config.py`

- [ ] **Step 1: Add new data types to ENTSOE_API_CONFIG**

Add after the existing entries in the `ENTSOE_API_CONFIG` dict:

```python
    'crossborder_flows': {
        'name': 'Cross-Border Physical Flows',
        'document_type': 'A11',
        'process_type': None,
        'table': 'crossborder_flows',
        'value_column': 'flow_mw',
        'entsoe_method': 'query_physical_crossborder_allborders',
        'description': 'Physical electricity flows between interconnected countries (MW)',
    },
    'net_position': {
        'name': 'Realized Net Position',
        'document_type': 'A25',
        'process_type': None,
        'table': 'net_position',
        'value_column': 'net_position_mw',
        'entsoe_method': 'query_net_position',
        'description': 'Aggregated import/export balance per country (MW). Positive = exporter.',
    },
```

- [ ] **Step 2: Add to BACKFILL_DEFAULTS**

Add to the `BACKFILL_DEFAULTS` dict:

```python
    'crossborder_flows': '2023-01-01',
    'net_position': '2023-01-01',
```

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add crossborder flows and net position to config"
```

---

### Task 7: Pipeline integration

**Files:**
- Modify: `C:\Code\energy-data-gathering\src\pipeline.py`

- [ ] **Step 1: Add import at top of pipeline.py**

```python
from src import fetch_crossborder_flows, fetch_net_position
```

- [ ] **Step 2: Add dispatch branches in `_fetch_data_chunk()`**

Find the `_fetch_data_chunk()` method and add two new `elif` branches after the existing ones:

```python
        elif data_type == 'crossborder_flows':
            inserted, updated, failed = fetch_crossborder_flows.fetch_crossborder_flows_data(
                self.client, country_code, start, end
            )
        elif data_type == 'net_position':
            inserted, updated, failed = fetch_net_position.fetch_net_position_data(
                self.client, country_code, start, end
            )
```

- [ ] **Step 3: Verify the import doesn't break**

```bash
python -c "from src.pipeline import ENTSOEPipeline; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: wire crossborder flows and net position into pipeline dispatch"
```

---

### Task 8: Standalone backfill script

**Files:**
- Create: `C:\Code\energy-data-gathering\scripts\backfill_crossborder.py`

- [ ] **Step 1: Create the backfill script**

```python
#!/usr/bin/env python3
"""
Backfill cross-border flows and net position data from ENTSO-E.

Processes data month-by-month with progress checkpointing.
Can be interrupted and resumed — picks up from last completed month.

Usage:
    # Full backfill (2023-01 to present)
    python scripts/backfill_crossborder.py

    # Single country, single month (for testing)
    python scripts/backfill_crossborder.py --countries DE --start-month 2024-01 --end-month 2024-02

    # Resume interrupted backfill
    python scripts/backfill_crossborder.py --resume

    # Only net position (skip cross-border flows)
    python scripts/backfill_crossborder.py --types net_position
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.entsoe_client import ENTSOEClient
from src import fetch_crossborder_flows, fetch_net_position, db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backfill_crossborder")

CHECKPOINT_FILE = Path(__file__).parent.parent / "backfill_crossborder_progress.json"


def load_checkpoint() -> dict:
    """Load progress checkpoint from disk."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(progress: dict):
    """Save progress checkpoint to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(progress, f, indent=2, default=str)


def get_months(start_month: str, end_month: str) -> list[tuple[datetime, datetime]]:
    """Generate list of (month_start, month_end) tuples."""
    months = []
    current = pd.Timestamp(start_month + "-01")
    end = pd.Timestamp(end_month + "-01")

    while current <= end:
        month_start = current.to_pydatetime()
        month_end = (current + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)).to_pydatetime()
        months.append((month_start, month_end))
        current += pd.offsets.MonthBegin(1)

    return months


def main():
    parser = argparse.ArgumentParser(description="Backfill cross-border and net position data")
    parser.add_argument("--countries", default="all", help="Comma-separated country codes or 'all'")
    parser.add_argument("--start-month", default="2023-01", help="Start month YYYY-MM (default: 2023-01)")
    parser.add_argument("--end-month", default=None, help="End month YYYY-MM (default: current month)")
    parser.add_argument("--types", default="all", help="Comma-separated: crossborder_flows,net_position or 'all'")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Parse arguments
    if args.countries == "all":
        countries = config.SUPPORTED_COUNTRIES
    else:
        countries = [c.strip().upper() for c in args.countries.split(",")]

    if args.end_month is None:
        args.end_month = datetime.now().strftime("%Y-%m")

    if args.types == "all":
        data_types = ["crossborder_flows", "net_position"]
    else:
        data_types = [t.strip() for t in args.types.split(",")]

    months = get_months(args.start_month, args.end_month)

    # Load checkpoint for resume
    progress = load_checkpoint() if args.resume else {}
    last_completed = progress.get("last_completed_month", "")

    # Ensure tables exist
    db.create_crossborder_flows_table()
    db.create_net_position_table()

    # Initialize client
    client = ENTSOEClient()

    total_records = progress.get("total_records", 0)
    total_errors = 0

    logger.info(f"=== Cross-Border Data Backfill ===")
    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Data types: {data_types}")
    logger.info(f"Months: {args.start_month} to {args.end_month} ({len(months)} months)")
    if last_completed:
        logger.info(f"Resuming from: {last_completed}")

    for month_start, month_end in months:
        month_key = month_start.strftime("%Y-%m")

        # Skip already completed months (resume mode)
        if args.resume and month_key <= last_completed:
            logger.info(f"Skipping {month_key} (already completed)")
            continue

        logger.info(f"\n--- {month_key} ---")

        for country in countries:
            for data_type in data_types:
                try:
                    if data_type == "crossborder_flows":
                        inserted, _, failed = fetch_crossborder_flows.fetch_crossborder_flows_data(
                            client, country, month_start, month_end
                        )
                    elif data_type == "net_position":
                        inserted, _, failed = fetch_net_position.fetch_net_position_data(
                            client, country, month_start, month_end
                        )
                    else:
                        continue

                    total_records += inserted
                    total_errors += failed

                except Exception as e:
                    logger.error(f"  {country}/{data_type}: {e}")
                    total_errors += 1

        # Save checkpoint after each month
        progress["last_completed_month"] = month_key
        progress["total_records"] = total_records
        progress["total_errors"] = total_errors
        progress["updated_at"] = datetime.now().isoformat()
        save_checkpoint(progress)

        logger.info(f"  Checkpoint saved: {month_key} ({total_records} total records)")

    logger.info(f"\n=== Backfill Complete ===")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Total errors: {total_errors}")

    # Clean up checkpoint on successful completion
    if total_errors == 0 and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed (clean completion)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/backfill_crossborder.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_crossborder.py
git commit -m "feat: add chunked backfill script for cross-border data with resume"
```

---

### Task 9: Smoke test — single country, single month

- [ ] **Step 1: Test net position for DE, January 2024**

```bash
cd C:/Code/energy-data-gathering
python -c "
from datetime import datetime
from src.entsoe_client import ENTSOEClient
from src import fetch_net_position, db

db.create_net_position_table()
client = ENTSOEClient()
start = datetime(2024, 1, 15)
end = datetime(2024, 1, 16)
inserted, updated, failed = fetch_net_position.fetch_net_position_data(client, 'DE', start, end)
print(f'Net position: inserted={inserted}, failed={failed}')
"
```
Expected: `Net position: inserted=24, failed=0` (approximately 24 hourly records)

- [ ] **Step 2: Test crossborder flows for DE, January 2024**

```bash
python -c "
from datetime import datetime
from src.entsoe_client import ENTSOEClient
from src import fetch_crossborder_flows, db

db.create_crossborder_flows_table()
client = ENTSOEClient()
start = datetime(2024, 1, 15)
end = datetime(2024, 1, 16)
inserted, updated, failed = fetch_crossborder_flows.fetch_crossborder_flows_data(client, 'DE', start, end)
print(f'Crossborder flows: inserted={inserted}, failed={failed}')
"
```
Expected: ~120-180 records (24 hours × 5-8 neighbors)

- [ ] **Step 3: Verify data in database**

```bash
python -c "
import sqlite3
conn = sqlite3.connect(str(__import__('config').DATABASE_PATH))
print('=== Net Position ===')
for row in conn.execute('SELECT country_code, COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM net_position GROUP BY country_code'):
    print(row)
print()
print('=== Crossborder Flows ===')
for row in conn.execute('SELECT country_from, country_to, COUNT(*) FROM crossborder_flows GROUP BY country_from, country_to ORDER BY country_from, country_to'):
    print(row)
conn.close()
"
```

- [ ] **Step 4: Test backfill script with single month**

```bash
python scripts/backfill_crossborder.py --countries DE --start-month 2024-01 --end-month 2024-01
```
Expected: Completes without errors, prints record counts.

- [ ] **Step 5: Commit any fixes found during smoke testing**

```bash
git add -A
git commit -m "fix: adjustments from smoke testing crossborder data pipeline"
```

---

### Task 10: Cross-check data quality

- [ ] **Step 1: Verify flow symmetry**

```bash
python -c "
import sqlite3, config
conn = sqlite3.connect(str(config.DATABASE_PATH))

# DE→FR exports should approximately equal -(FR→DE exports)
de_fr = conn.execute('''
    SELECT timestamp_utc, flow_mw FROM crossborder_flows
    WHERE country_from='DE' AND country_to='FR'
    ORDER BY timestamp_utc LIMIT 5
''').fetchall()

fr_de = conn.execute('''
    SELECT timestamp_utc, flow_mw FROM crossborder_flows
    WHERE country_from='FR' AND country_to='DE'
    ORDER BY timestamp_utc LIMIT 5
''').fetchall()

print('DE→FR:', de_fr)
print('FR→DE:', fr_de)
print('(Values should be approximately opposite in sign)')
conn.close()
"
```

- [ ] **Step 2: Verify net position approximates sum of flows**

```bash
python -c "
import sqlite3, config
conn = sqlite3.connect(str(config.DATABASE_PATH))

# Sum of exports from DE should approximate DE net position
flow_sum = conn.execute('''
    SELECT SUM(flow_mw) FROM crossborder_flows
    WHERE country_from='DE' AND timestamp_utc LIKE '2024-01-15%'
''').fetchone()[0]

net_pos = conn.execute('''
    SELECT SUM(net_position_mw) FROM net_position
    WHERE country_code='DE' AND timestamp_utc LIKE '2024-01-15%'
''').fetchone()[0]

print(f'Sum of DE exports: {flow_sum:.0f} MW')
print(f'DE net position:   {net_pos:.0f} MW')
print(f'Difference:        {abs(flow_sum - net_pos):.0f} MW')
conn.close()
"
```

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: cross-border flow and net position data collection complete"
```
