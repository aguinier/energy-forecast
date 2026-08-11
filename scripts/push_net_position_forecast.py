#!/usr/bin/env python
"""Push every registered net-position model's latest vintage to the dashboard.

The Chronos net-position run happens on the workstation because it needs the
GPU, and writes to the local sidecar DB. The dashboard reads the canonical
database, so a forecast has to be shipped there to be visible anywhere - and,
since ABL-70, to accrue as a scored vintage the promotion gate can count.

The endpoint is idempotent per (country, forecast_type, model, generated_at),
so re-running this after a failure replaces a vintage rather than duplicating
it. For each model, only its newest generated_at is sent; older vintages stay
in the sidecar.

**Every registered model is pushed, each under its own name, independently**
(ABL-175). Before this, only `CHAMPION_MODEL_NAME` was ever sent - the sidecar
has held challenger vintages since ABL-68, but nothing shipped them, so
challenger accrual in prod was permanently zero and ABL-137's backfill was the
only data the promotion gate could ever see. Every query is still scoped to
one `model_name` at a time; without that scoping "the newest generated_at"
would be whichever model wrote last (challengers run after the champion in the
same job, so they would win) and a vintage's row set would mix models under
one label. Models are pushed one at a time and a failure - no vintage yet,
an HTTP error, a network error - is isolated to that model: it is reported and
the remaining models still get their own attempt. Optional quantiles ride
along per model where the run emitted them (V016's correction layer stores a
p10-p90 band; V012's baseline and V014's XGBoost store none) and their absence
is not a failure - `build_payload` already treats "no quantile rows for this
model_name/generated_at" as ordinary, median-only rows.

Environment:
    FORECAST_OUTPUT_DB      sidecar written by forecast_chronos2.py /
                             forecast_challengers.py
    CHAMPION_MODEL_NAME     the champion's model_name (default chronos-2-V010)
    DASHBOARD_API_URL       e.g. http://192.168.86.36:3001
    DASHBOARD_WRITE_TOKEN   matches HELIO_WRITE_TOKEN on the server

Exit codes:
    0  every model pushed its latest vintage successfully
    1  at least one model's push failed (HTTP or network error)
    2  no push failed, but at least one model had nothing to push
"""
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.challengers.registry import CHALLENGERS  # noqa: E402

FORECAST_TYPE = "net_position"
DEFAULT_CHAMPION_MODEL = "chronos-2-V010"
TIMEOUT_S = 60


def champion_model_name() -> str:
    """The champion's model_name. Never inferred from data."""
    return os.getenv("CHAMPION_MODEL_NAME") or DEFAULT_CHAMPION_MODEL


def models_to_push() -> list:
    """Champion first, then every registered challenger - order-preserving,
    de-duplicated in case CHAMPION_MODEL_NAME is overridden to a name that is
    also a registered challenger's."""
    names = [champion_model_name()] + [spec.model_name for spec in CHALLENGERS.values()]
    return list(dict.fromkeys(names))


def latest_vintage(conn, model_name):
    row = conn.execute(
        "SELECT generated_at, model_name, model_version FROM forecasts "
        "WHERE forecast_type = ? AND model_name = ? "
        "ORDER BY generated_at DESC LIMIT 1",
        (FORECAST_TYPE, model_name),
    ).fetchone()
    return row


def build_payload(conn, generated_at, model_name, model_version):
    points = conn.execute(
        "SELECT country_code, target_timestamp_utc, horizon_hours, forecast_value "
        "FROM forecasts WHERE forecast_type = ? AND generated_at = ? AND model_name = ? "
        "ORDER BY country_code, target_timestamp_utc",
        (FORECAST_TYPE, generated_at, model_name),
    ).fetchall()

    bands = {}
    try:
        for cc, ts, q, val in conn.execute(
            "SELECT country_code, target_timestamp_utc, quantile, forecast_value "
            "FROM forecast_quantiles "
            "WHERE forecast_type = ? AND generated_at = ? AND model_name = ?",
            (FORECAST_TYPE, generated_at, model_name),
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


def push_model(conn, api_url, token, model_name):
    """Push one model's latest vintage. Never raises - every failure mode is
    caught and reported so one model's problem cannot cost another its push.

    Returns (status, detail) with status one of "ok" / "no_data" / "failed".
    """
    try:
        vintage = latest_vintage(conn, model_name)
        if vintage is None:
            return "no_data", f"no {model_name} net_position forecasts in the sidecar"

        generated_at, resolved_name, model_version = vintage
        payload = build_payload(conn, generated_at, resolved_name, model_version)
        if not payload["rows"]:
            return "no_data", f"latest {model_name} vintage ({generated_at}) has no rows"

        quantile_rows = sum(len(r.get("quantiles", {})) for r in payload["rows"])
        print(
            f"{model_name}: pushing {len(payload['rows'])} points "
            f"({quantile_rows} quantiles) from {generated_at}"
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
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        return "failed", f"HTTP {exc.code} from dashboard: {detail}"
    except urllib.error.URLError as exc:
        return "failed", f"could not reach {api_url}: {exc.reason}"
    except Exception as exc:  # a single model's failure must never take down the rest
        return "failed", f"unexpected error: {exc!r}"

    data = body.get("data", {})
    replaced = " (replaced an existing copy of this run)" if data.get("replaced") else ""
    return "ok", f"stored {data.get('points')} points, {data.get('quantiles')} quantiles{replaced}"


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

    models = models_to_push()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        results = {}
        for model_name in models:
            status, detail = push_model(conn, api_url, token, model_name)
            results[model_name] = status
            stream = sys.stderr if status == "failed" else sys.stdout
            print(f"{model_name}: {status.upper()} - {detail}", file=stream)
    finally:
        conn.close()

    print(
        "Summary: "
        + ", ".join(f"{name}={status}" for name, status in results.items())
    )

    if any(status == "failed" for status in results.values()):
        return 1
    if any(status == "no_data" for status in results.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
