#!/usr/bin/env python
"""Push the latest net-position forecast vintage to the dashboard API.

The Chronos net-position run happens on the workstation because it needs the
GPU, and writes to the local sidecar DB. The dashboard reads the canonical
database, so the forecast has to be shipped there to be visible anywhere.

The endpoint is idempotent per (country, forecast_type, model, generated_at),
so re-running this after a failure replaces the vintage rather than duplicating
it. Only the newest generated_at is sent; older vintages stay in the sidecar.

Environment:
    FORECAST_OUTPUT_DB      sidecar written by forecast_chronos2.py
    DASHBOARD_API_URL       e.g. http://192.168.86.36:3001
    DASHBOARD_WRITE_TOKEN   matches HELIO_WRITE_TOKEN on the server

Exit codes: 0 ok, 1 failure, 2 nothing to push.
"""
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

FORECAST_TYPE = "net_position"
TIMEOUT_S = 60


def latest_vintage(conn):
    row = conn.execute(
        "SELECT generated_at, model_name, model_version FROM forecasts "
        "WHERE forecast_type = ? ORDER BY generated_at DESC LIMIT 1",
        (FORECAST_TYPE,),
    ).fetchone()
    return row


def build_payload(conn, generated_at, model_name, model_version):
    points = conn.execute(
        "SELECT country_code, target_timestamp_utc, horizon_hours, forecast_value "
        "FROM forecasts WHERE forecast_type = ? AND generated_at = ? "
        "ORDER BY country_code, target_timestamp_utc",
        (FORECAST_TYPE, generated_at),
    ).fetchall()

    bands = {}
    try:
        for cc, ts, q, val in conn.execute(
            "SELECT country_code, target_timestamp_utc, quantile, forecast_value "
            "FROM forecast_quantiles WHERE forecast_type = ? AND generated_at = ?",
            (FORECAST_TYPE, generated_at),
        ):
            bands.setdefault((cc, ts), {})[str(q)] = val
    except sqlite3.OperationalError:
        # No quantiles stored for this run; the API accepts median-only rows.
        pass

    rows = []
    for cc, ts, horizon, value in points:
        row = {
            "country_code": cc,
            "target_timestamp_utc": ts,
            "horizon_hours": horizon,
            "forecast_value": value,
        }
        band = bands.get((cc, ts))
        if band:
            row["quantiles"] = band
        rows.append(row)

    return {
        "model": {"name": model_name, "version": model_version},
        "generated_at": generated_at,
        "rows": rows,
    }


def main():
    db_path = os.getenv("FORECAST_OUTPUT_DB")
    api_url = os.getenv("DASHBOARD_API_URL")
    token = os.getenv("DASHBOARD_WRITE_TOKEN")

    missing = [
        name
        for name, val in (
            ("FORECAST_OUTPUT_DB", db_path),
            ("DASHBOARD_API_URL", api_url),
            ("DASHBOARD_WRITE_TOKEN", token),
        )
        if not val
    ]
    if missing:
        print(f"ERROR: missing environment: {', '.join(missing)}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        vintage = latest_vintage(conn)
        if vintage is None:
            print("Nothing to push: no net_position forecasts in the sidecar.")
            return 2
        generated_at, model_name, model_version = vintage
        payload = build_payload(conn, generated_at, model_name, model_version)
    finally:
        conn.close()

    if not payload["rows"]:
        print("Nothing to push: latest vintage has no rows.")
        return 2

    quantile_rows = sum(len(r.get("quantiles", {})) for r in payload["rows"])
    print(
        f"Pushing {len(payload['rows'])} points "
        f"({quantile_rows} quantiles) from {generated_at} [{model_name}]"
    )

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/forecasts/net-position",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"ERROR: HTTP {exc.code} from dashboard: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ERROR: could not reach {api_url}: {exc.reason}", file=sys.stderr)
        return 1

    data = body.get("data", {})
    print(
        f"OK: stored {data.get('points')} points, {data.get('quantiles')} quantiles"
        f"{' (replaced an existing copy of this run)' if data.get('replaced') else ''}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
