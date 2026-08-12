#!/usr/bin/env python3
"""
ABL-335 / ABL-337: measure solar forecasts against physical reality.

Three read-only measurements, kept because each one is the evidence behind a
decision that will be revisited:

  --stored-forecasts   The ABL-335 repro. Counts negative rows and non-zero
                       night rows in stored solar forecasts. Point it at the
                       replica to reproduce the defect, or at a sidecar a fresh
                       run just wrote to verify the ABL-337 clamp.

  --check-actuals      The ABL-337 threshold validation. For a range of night
                       thresholds, counts the hours that ever recorded non-zero
                       actual solar which the mask would zero. This is what
                       NIGHT_ELEVATION_THRESHOLD_DEG was chosen from, so rerun
                       it before changing that constant.

  --print-points       Regenerate SOLAR_REPRESENTATIVE_POINTS from
                       `weather_location`, for when the cluster table changes.

Nothing here writes. Every query opens SQLite read-only.

    python scripts/abl335_solar_night_probe.py --stored-forecasts
    python scripts/abl335_solar_night_probe.py --stored-forecasts --db C:/tmp/run.db
    python scripts/abl335_solar_night_probe.py --check-actuals
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.solar_geometry import (
    NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_REPRESENTATIVE_POINTS,
    max_sun_elevation_over_hour,
)

DEFAULT_COUNTRIES = ['AT', 'BE', 'DE', 'FR']
CANDIDATE_THRESHOLDS = [0.0, -0.833, -3.0, -6.0, -8.0, -10.0, -12.0]

# ABL-335 counted these UTC hours as "night" by hand. Kept so the repro's output
# is comparable to the numbers on that issue — the clamp itself uses sun
# elevation, not a fixed hour list, because hour 3 is dawn in August and hour 3
# is deep night in December.
ABL335_NIGHT_HOURS = [0, 1, 2, 3, 22, 23]


def connect_readonly(db_path) -> sqlite3.Connection:
    target = Path(db_path).resolve().as_posix()
    return sqlite3.connect(f"file:{target}?mode=ro", uri=True)


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse the three timestamp spellings this database carries (ABL-211/324)."""
    parsed = pd.to_datetime(series, format='mixed', utc=True)
    return parsed.dt.tz_convert('UTC').dt.tz_localize(None)


def stored_forecasts(db_path, countries, model_name=None, since=None):
    """Negative and night-time solar rows in stored forecasts (the ABL-335 table)."""
    print(f"Stored solar forecasts in {db_path}")
    if since:
        print(f"generated_at >= {since}")
    print()
    header = (f"{'country':<8} {'rows':>8} {'negative':>9} {'night>1MW':>11} "
              f"{'night rows':>11} {'min MW':>9} {'night max MW':>13} {'mask-night>0':>13}")
    print(header)
    print("-" * len(header))

    con = connect_readonly(db_path)
    try:
        totals = np.zeros(5, dtype=float)
        for country in countries:
            sql = ("SELECT target_timestamp_utc, forecast_value, model_name FROM forecasts "
                   "WHERE country_code = ? AND (renewable_type = 'solar' "
                   "  OR (renewable_type IS NULL AND forecast_type = 'solar'))")
            params = [country]
            if model_name:
                sql += " AND model_name = ?"
                params.append(model_name)
            if since:
                sql += " AND generated_at >= ?"
                params.append(since)
            df = pd.read_sql_query(sql, con, params=params)
            if df.empty:
                print(f"{country:<8} {'no rows':>8}")
                continue

            target = parse_utc(df['target_timestamp_utc'])
            value = df['forecast_value'].astype(float)
            clock_night = target.dt.hour.isin(ABL335_NIGHT_HOURS)
            # The clamp's own definition, which is what a clamped run must satisfy.
            mask_night = np.asarray(
                max_sun_elevation_over_hour(country, target.dt.floor('h'))
            ) < NIGHT_ELEVATION_THRESHOLD_DEG

            n_negative = int((value < 0).sum())
            n_night_gt1 = int((clock_night & (value > 1)).sum())
            n_mask_night_nonzero = int(((value != 0) & mask_night).sum())
            night_max = float(value[clock_night].max()) if clock_night.any() else float('nan')

            print(f"{country:<8} {len(df):>8} {n_negative:>9} {n_night_gt1:>11} "
                  f"{int(clock_night.sum()):>11} {value.min():>9.1f} {night_max:>13.1f} "
                  f"{n_mask_night_nonzero:>13}")
            totals += [len(df), n_negative, n_night_gt1, int(clock_night.sum()),
                       n_mask_night_nonzero]
    finally:
        con.close()

    print("-" * len(header))
    print(f"{'total':<8} {int(totals[0]):>8} {int(totals[1]):>9} {int(totals[2]):>11} "
          f"{int(totals[3]):>11} {'':>9} {'':>13} {int(totals[4]):>13}")
    print()
    print("negative      : rows below zero — physically impossible")
    print("night>1MW     : rows above 1 MW at UTC hours "
          f"{ABL335_NIGHT_HOURS} (ABL-335's fixed-hour definition)")
    print("mask-night>0  : non-zero rows at hours the ABL-337 night mask covers "
          f"(sun below {NIGHT_ELEVATION_THRESHOLD_DEG:g} deg all hour).")
    print("                Both this and 'negative' must be 0 for a clamped run.")


def check_actuals(db_path, countries):
    """Would the night mask ever zero an hour that really generated?

    Reads every `energy_renewable.solar_mw` row above zero, collapses to the
    containing hour, and reports how many of those hours each candidate
    threshold would mask — with the largest actual it would discard.
    """
    print(f"Never-zero-a-real-actual check against {db_path}")
    print(f"Shipping threshold: {NIGHT_ELEVATION_THRESHOLD_DEG:g} deg\n")

    con = connect_readonly(db_path)
    try:
        for country in countries:
            df = pd.read_sql_query(
                "SELECT timestamp_utc, solar_mw FROM energy_renewable "
                "WHERE country_code = ? AND solar_mw IS NOT NULL AND solar_mw > 0",
                con, params=(country,))
            if df.empty:
                print(f"{country}: no non-zero actual solar on record\n")
                continue

            hour = parse_utc(df['timestamp_utc']).dt.floor('h')
            hourly = df.assign(hour=hour).groupby('hour')['solar_mw'].max().reset_index()
            hourly['peak_elevation_deg'] = max_sun_elevation_over_hour(country, hourly['hour'])

            print(f"{country}: {len(hourly)} distinct hours with non-zero actual solar "
                  f"({hourly['hour'].min()} .. {hourly['hour'].max()})")
            print(f"     lowest peak elevation over any of them: "
                  f"{hourly['peak_elevation_deg'].min():.2f} deg")
            for threshold in CANDIDATE_THRESHOLDS:
                zeroed = hourly[hourly['peak_elevation_deg'] < threshold]
                marker = '  <-- shipping' if threshold == NIGHT_ELEVATION_THRESHOLD_DEG else ''
                largest = f", largest {zeroed['solar_mw'].max():.1f} MW" if len(zeroed) else ""
                print(f"     {threshold:>7.3f} deg: would zero {len(zeroed):>5} of them"
                      f"{largest}{marker}")
            worst = hourly.nsmallest(3, 'peak_elevation_deg')
            print("     deepest-night hours claiming generation:")
            for _, row in worst.iterrows():
                print(f"       {row['hour']}  elevation {row['peak_elevation_deg']:7.2f} deg  "
                      f"{row['solar_mw']:.3f} MW")
            print()
    finally:
        con.close()

    print("An hour appearing here at the shipping threshold is generation the clamp")
    print("would delete. Judge it by the MW, not the count: single-digit MW on a")
    print("multi-GW fleet is metering dust at twilight; hundreds of MW at -20 deg is")
    print("an actuals defect, and the threshold is not the thing to change for it.")


def print_points(db_path):
    """Recompute the capacity-weighted solar centroid per country."""
    con = connect_readonly(db_path)
    try:
        zones = pd.read_sql_query(
            "SELECT country_code, lat, lon, weight, capacity_mw FROM weather_location "
            "WHERE zone_type = 'solar'", con)
    finally:
        con.close()

    print("# Regenerated from weather_location (zone_type='solar'), "
          "capacity-weighted centroid.")
    print("SOLAR_REPRESENTATIVE_POINTS = {")
    for country in config.SUPPORTED_COUNTRIES:
        group = zones[zones['country_code'] == country]
        if group.empty:
            print(f"    # {country}: no solar clusters in weather_location")
            continue
        weight = group['weight'].fillna(0.0)
        if weight.sum() <= 0:
            weight = pd.Series(np.ones(len(group)), index=group.index)
        lat = float((group['lat'] * weight).sum() / weight.sum())
        lon = float((group['lon'] * weight).sum() / weight.sum())
        capacity = float(group['capacity_mw'].fillna(0.0).sum())
        current = SOLAR_REPRESENTATIVE_POINTS.get(country)
        drift = ""
        if current and (abs(current[0] - lat) > 0.01 or abs(current[1] - lon) > 0.01):
            drift = f"  # CHANGED from {current}"
        print(f"    '{country}': ({lat:.3f}, {lon:.3f}),   # {capacity:.0f} MW{drift}")
    print("}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stored-forecasts', action='store_true',
                        help='count negative and night-time rows in stored solar forecasts')
    parser.add_argument('--check-actuals', action='store_true',
                        help='validate the night threshold against actual solar generation')
    parser.add_argument('--print-points', action='store_true',
                        help='regenerate SOLAR_REPRESENTATIVE_POINTS from weather_location')
    parser.add_argument('--db', default=None,
                        help='database to read (default: ENERGY_DB_PATH via config)')
    parser.add_argument('--countries', default=','.join(DEFAULT_COUNTRIES),
                        help=f'comma-separated country codes (default: {",".join(DEFAULT_COUNTRIES)})')
    parser.add_argument('--model', default=None,
                        help='restrict --stored-forecasts to one model_name')
    parser.add_argument('--since', default=None,
                        help='restrict --stored-forecasts to generated_at >= this ISO timestamp')
    args = parser.parse_args()

    if not (args.stored_forecasts or args.check_actuals or args.print_points):
        parser.error('pick at least one of --stored-forecasts, --check-actuals, --print-points')

    db_path = args.db or config.DATABASE_PATH
    if not Path(db_path).exists():
        parser.error(f"database not found: {db_path}")
    countries = [c.strip().upper() for c in args.countries.split(',') if c.strip()]

    if args.stored_forecasts:
        stored_forecasts(db_path, countries, model_name=args.model, since=args.since)
        print()
    if args.check_actuals:
        check_actuals(db_path, countries)
        print()
    if args.print_points:
        print_points(db_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
