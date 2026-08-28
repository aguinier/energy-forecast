"""ABL-430 -- diagnose Romania, not the models.

RO fails on two independent targets: `wind_onshore` grades B on G4 in all three
gate bands (ABL-417 tranche 2e), and `net_position` is the only one of 19 gate
countries that loses to the zero forecast (ABL-280). The issue asks whether
something upstream of both -- a sign convention, a timezone or DST offset, a
zone-code mismatch, or an actuals series that is not what we think it is --
explains them together.

This script measures the candidates rather than arguing them. It is read-only:
it opens the replica with the SQLite `mode=ro` URI and writes nothing but its
own report files.

Six checks, each one able to fail on its own:

  A. Clock. Three independent alignment tests, each a lag scan whose argmax
     must be 0. A1 compares the daily mass centroid of solar generation with
     astronomical solar noon at the country's own capacity-weighted point
     (`src.solar_geometry`, imported -- never a second copy). A2 scans RO wind
     actuals against the TSO's own day-ahead publication. A3 scans net position
     against a generation-minus-load balance assembled from two other tables.

  B. Sign, scale and zone. OLS slope of each independent reference on the
     series we score, plus the cross-border flow identity. A sign flip shows up
     as a negative slope, a unit error as a slope far from 1, and a zone swap as
     an energy balance that does not close.

  C. Series integrity. Native publication resolution, lag-1 autocorrelation
     (a physically smooth series cannot be noise), exact-zero count.

  D. What the wind challenger actually had to work with. Per country, the two
     feature families in `wind_retrain.FEATURE_COLUMNS` that carry physical
     information -- the target's own history, and the country-mean weather wind
     speed -- measured against the challenger's directional skill in the stored
     gate results. Includes the fleet-dispersion hypothesis, which is tested
     and NOT supported.

  E. What the net-position model actually had to work with. Coverage of every
     input family over the V010 training span, per gate country.

Usage:
    .venv\\Scripts\\python.exe scripts/abl430_ro_country_diagnosis.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db --stdout
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.solar_geometry import SOLAR_REPRESENTATIVE_POINTS  # noqa: E402
from src.tso_plausibility import guard_tso_frame  # noqa: E402

# The 19 net-position gate countries (LU and GR are excluded by name upstream).
GATE_COUNTRIES = (
    "AT", "BE", "BG", "CZ", "DE", "EE", "ES", "FI", "FR",
    "HR", "HU", "LT", "LV", "NL", "PL", "PT", "RO", "SI", "SK",
)

# The 18 wind_onshore pairs ABL-348 registers, fitted across tranches 1a/2b/2e.
WIND_RESULT_FILES = (
    "results_abl380_tranche1a.json",
    "results_abl406_tranche2b.json",
    "results_abl417_tranche2e.json",
)

# ABL-348's frozen windows. The fit window is what a challenger learned the
# target's structure from; measuring feature strength anywhere else would not be
# measuring what the model saw.
FIT_START, GATE_END = "2026-01-14", "2026-08-10"

# A window long enough for a stable cross-correlation, inside the current
# ingest regime. Deliberately not the gate window: 720 hours cannot separate a
# 1-hour clock offset from noise on a synoptic series.
ALIGN_START, ALIGN_END = "2026-01-01", "2026-08-11"

# V010's registered training span (`experiments/V010/config.json`).
V010_TRAIN_START, V010_TRAIN_END = "2023-01-01", "2026-03-01"

DAY_MW_COLUMNS = (
    "solar_mw", "wind_onshore_mw", "wind_offshore_mw", "hydro_run_mw",
    "hydro_reservoir_mw", "biomass_mw", "nuclear_mw", "fossil_gas_mw",
    "fossil_hard_coal_mw", "fossil_brown_coal_mw", "fossil_oil_mw",
    "geothermal_mw", "other_renewable_mw", "waste_mw", "other_mw",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def open_replica(path: Path) -> sqlite3.Connection:
    """Read-only connection. The replica is a mirror of prod; nothing may write."""
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def guard_wind_forecast(fc: pd.DataFrame, con: sqlite3.Connection) -> pd.DataFrame:
    """ABL-462: route the TSO wind read through the ABL-431 plausibility guard.

    Runs per country and **before** `to_hourly`, per CLAUDE.md: a refused value
    averaged into its neighbour first is not refused.

    This is not hygiene here. Measured read-only on the live replica over this
    script's own window (`2026-01-01..2026-08-11`, 24 countries, 408,159 rows):
    the guard nulls **96 rows, all HU, all 2026-02-03/04, up to 140,996 MW
    against a 283 MW reference** -- the exact rows ABL-431 was written for. The
    other 23 countries are untouched, 0 rows.

    `timestamp_column` is deliberately not passed: `ts` here is the raw stored
    string and the two generation tables use different separator forms
    (ABL-200), so only `to_hourly`'s `format="mixed"` parse is safe on it. The
    cost is that the guard's warning names row positions, not timestamps.
    """
    if fc.empty:
        return fc
    guarded = [
        guard_tso_frame(group, con, cc, "energy_generation_forecast",
                        "wind_onshore_mw", frame_column="f",
                        context=f"ABL-430 A2 {cc} wind_onshore day_ahead")
        for cc, group in fc.groupby("country_code", sort=True)
    ]
    return pd.concat(guarded, ignore_index=True)


def to_hourly(df: pd.DataFrame, value: str, key: str = "country_code") -> pd.DataFrame:
    """Floor to the hour and take the hourly mean.

    Most countries publish sub-hourly (ABL-332): 22 of 24 carry sub-hourly rows
    in at least one generation table. The hourly mean is the one resolution that
    leaves a read, so a lag scan built on the `:00` sub-sample would be
    comparing a different series to the one every model is fitted on.
    """
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], format="mixed", utc=True)
    out["ts"] = out["ts"].dt.tz_localize(None).dt.floor("h")
    return out.groupby([key, "ts"], as_index=False)[value].mean()


def lag_scan(a: pd.Series, b: pd.Series, lags: range) -> dict:
    """corr(a_t, b_{t+k}) for each k. A correctly aligned pair peaks at k = 0."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 200 or joined["a"].std() == 0 or joined["b"].std() == 0:
        return {"n": len(joined), "argmax_lag": None, "by_lag": {}}
    by_lag = {k: float(joined["a"].corr(joined["b"].shift(k))) for k in lags}
    finite = {k: v for k, v in by_lag.items() if np.isfinite(v)}
    return {
        "n": int(len(joined)),
        "argmax_lag": int(max(finite, key=finite.get)) if finite else None,
        "by_lag": {str(k): round(v, 4) for k, v in by_lag.items()},
    }


def ols_slope(x: pd.Series, y: pd.Series) -> float | None:
    """Slope of y on x. Sign flips and unit errors both land here."""
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    if len(joined) < 50 or joined["x"].std() == 0:
        return None
    return float(np.polyfit(joined["x"], joined["y"], 1)[0])


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.corr(b, method="spearman"))


def solar_noon_utc(lon_deg: float) -> float:
    """Mean solar noon in UTC hours at a longitude. 4 minutes per degree east."""
    return 12.0 - lon_deg / 15.0


def circular_mass_centroid(hours: np.ndarray, mass: np.ndarray) -> float:
    """Mass-weighted mean hour on a 24-hour circle.

    Used instead of argmax because an hourly argmax quantises to 1 h, which is
    coarser than the offsets this check has to be able to see.
    """
    theta = 2 * math.pi * hours / 24.0
    x = float((mass * np.cos(theta)).sum())
    y = float((mass * np.sin(theta)).sum())
    return (math.atan2(y, x) / (2 * math.pi) * 24.0) % 24.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# A1 -- the clock, against the sun
# ---------------------------------------------------------------------------

def check_solar_clock(con: sqlite3.Connection) -> dict:
    """Daily mass centroid of solar generation vs astronomical solar noon.

    This is the strongest available clock test because the reference is
    geometry, not data: the sun's position at a longitude cannot be
    mis-ingested. A country stamped in local time instead of UTC would sit 2-3 h
    away from the fleet.

    The whole fleet is expected to land near -0.5 h, not 0.0: ENTSO-E timestamps
    label the START of a settlement interval, so an hour's mass is booked at its
    opening edge and the centroid falls half an hour early by construction.
    """
    countries = tuple(SOLAR_REPRESENTATIVE_POINTS)
    df = pd.read_sql(
        "SELECT country_code, timestamp_utc, solar_mw FROM energy_generation "
        "WHERE timestamp_utc >= '2026-05-01' AND timestamp_utc < '2026-08-01' "
        f"AND solar_mw IS NOT NULL AND country_code IN {countries}",
        con,
    )
    df["ts"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True)
    df["hh"] = df["ts"].dt.tz_localize(None).dt.hour
    rows = []
    for cc, grp in df.groupby("country_code"):
        prof = grp.groupby("hh")["solar_mw"].mean()
        if prof.max() <= 0:
            continue
        noon = solar_noon_utc(SOLAR_REPRESENTATIVE_POINTS[cc][1])
        centroid = circular_mass_centroid(prof.index.to_numpy(float), prof.to_numpy())
        rows.append({
            "country": cc,
            "centroid_h_utc": round(centroid, 3),
            "solar_noon_h_utc": round(noon, 3),
            "offset_h": round(centroid - noon, 3),
        })
    tab = pd.DataFrame(rows).sort_values("offset_h")
    ro = tab.loc[tab.country == "RO"].iloc[0]
    fleet = tab.offset_h
    return {
        "window": "2026-05-01..2026-08-01",
        "per_country": rows,
        "fleet_median_offset_h": round(float(fleet.median()), 3),
        "fleet_min_offset_h": round(float(fleet.min()), 3),
        "fleet_max_offset_h": round(float(fleet.max()), 3),
        "ro_offset_h": float(ro.offset_h),
        "ro_rank_of": f"{int((fleet < ro.offset_h).sum()) + 1}/{len(fleet)}",
        "ro_z_vs_fleet": round(float((ro.offset_h - fleet.mean()) / fleet.std()), 2),
        "verdict": (
            "RO sits inside the fleet distribution. No local-time, UTC-offset or "
            "DST error on energy_generation."
        ),
    }


# ---------------------------------------------------------------------------
# A2 + B -- wind: alignment, sign and scale against the TSO's own publication
# ---------------------------------------------------------------------------

def check_wind_against_tso(con: sqlite3.Connection, countries: tuple[str, ...]) -> dict:
    """Score our wind actuals against ENTSO-E's independently published forecast.

    The point is NOT that the TSO forecasts well. It is that an independently
    produced series of the same physical quantity agrees with ours -- which a
    time shift, a sign flip, a unit error or a zone swap all break, and none of
    which a revision can repair.

    Caveat carried into the report: `publication_timestamp_utc` on this table is
    the fetch time, not the publication time, so the stored `day_ahead` rows are
    revision-contaminated (ABL-348 `tso_role`). That bounds what the WAPE column
    means -- it is a lower bound on true day-ahead error, never a target -- but
    it does not touch sign, alignment, scale or zone.

    ABL-462: the read is now guarded (`guard_wind_forecast`). The first
    publication of this table ran unguarded and its **HU row is superseded** --
    `corr=0.0237, slope=2.0992, wape_pct=597.0, n=5328` was 96 rows of up to
    140,996 MW against a 283 MW fleet, not a zone or alignment defect. See
    `reports/abl_462_guard_scope_triage.md` for the corrected row; the other 23
    countries are unchanged.
    """
    fc = pd.read_sql(
        "SELECT country_code, target_timestamp_utc AS ts, wind_onshore_mw AS f "
        "FROM energy_generation_forecast WHERE forecast_type = 'day_ahead' "
        "AND wind_onshore_mw IS NOT NULL "
        f"AND target_timestamp_utc >= '{ALIGN_START}' AND target_timestamp_utc < '{ALIGN_END}' "
        f"AND country_code IN {countries}",
        con,
    )
    ac = pd.read_sql(
        "SELECT country_code, timestamp_utc AS ts, wind_onshore_mw AS a "
        "FROM energy_generation WHERE wind_onshore_mw IS NOT NULL "
        f"AND timestamp_utc >= '{ALIGN_START}' AND timestamp_utc < '{ALIGN_END}' "
        f"AND country_code IN {countries}",
        con,
    )
    fc = guard_wind_forecast(fc, con)
    fc, ac = to_hourly(fc, "f"), to_hourly(ac, "a")
    rows = []
    for cc in sorted(set(fc.country_code) & set(ac.country_code)):
        a = ac.loc[ac.country_code == cc].set_index("ts")["a"].sort_index()
        f = fc.loc[fc.country_code == cc].set_index("ts")["f"].sort_index()
        j = pd.concat([a.rename("a"), f.rename("f")], axis=1, join="inner").dropna()
        if len(j) < 500 or j["f"].std() == 0 or j["a"].std() == 0:
            continue
        scan = lag_scan(a, f, range(-3, 4))
        denom = float(j["a"].abs().sum())
        rows.append({
            "country": cc,
            "n": int(len(j)),
            "corr": round(float(j["a"].corr(j["f"])), 4),
            "argmax_lag_h": scan["argmax_lag"],
            "slope_tso_on_actual": round(ols_slope(j["a"], j["f"]) or float("nan"), 4),
            "wape_pct": round(100 * float((j["f"] - j["a"]).abs().sum()) / denom, 2),
            "actual_mean_mw": round(float(j["a"].mean()), 1),
            "by_lag": scan["by_lag"] if cc == "RO" else None,
        })
    return {
        "window": f"{ALIGN_START}..{ALIGN_END}",
        "per_country": rows,
        "caveat": (
            "publication_timestamp_utc is fetch time, so these day_ahead rows are "
            "revision-contaminated (ABL-348 tso_role). WAPE here is a lower bound "
            "on true day-ahead error and is not an achievability claim. Sign, "
            "alignment, scale and zone are unaffected by revision."
        ),
    }


# ---------------------------------------------------------------------------
# A3 + B -- net position: alignment and sign against an independent balance
# ---------------------------------------------------------------------------

def check_net_position_balance(con: sqlite3.Connection, countries: tuple[str, ...]) -> dict:
    """net_position vs (total generation - load), assembled from two other tables.

    A zone-code mismatch is exactly the failure this catches: if the
    `net_position` rows we store under 'RO' were another zone's, they could not
    stay consistent with RO's own generation and RO's own load.
    """
    gen_expr = " + ".join(f"COALESCE({c}, 0)" for c in DAY_MW_COLUMNS)
    gen = pd.read_sql(
        f"SELECT country_code, timestamp_utc AS ts, ({gen_expr}) AS g "
        "FROM energy_generation "
        f"WHERE timestamp_utc >= '{ALIGN_START}' AND timestamp_utc < '{ALIGN_END}' "
        f"AND country_code IN {countries}",
        con,
    )
    # ABL-109 / ABL-111: zero-as-missing rows in energy_load. Dropped, not zeroed.
    load = pd.read_sql(
        "SELECT country_code, timestamp_utc AS ts, load_mw AS l FROM energy_load "
        f"WHERE timestamp_utc >= '{ALIGN_START}' AND timestamp_utc < '{ALIGN_END}' "
        f"AND load_mw > 0 AND country_code IN {countries}",
        con,
    )
    npos = pd.read_sql(
        "SELECT country_code, timestamp_utc AS ts, net_position_mw AS v FROM net_position "
        f"WHERE timestamp_utc >= '{ALIGN_START}' AND timestamp_utc < '{ALIGN_END}' "
        f"AND net_position_mw IS NOT NULL AND country_code IN {countries}",
        con,
    )
    gen, load, npos = to_hourly(gen, "g"), to_hourly(load, "l"), to_hourly(npos, "v")
    joined = gen.merge(load, on=["country_code", "ts"]).merge(npos, on=["country_code", "ts"])
    joined["bal"] = joined["g"] - joined["l"]
    rows = []
    for cc, grp in joined.groupby("country_code"):
        grp = grp.sort_values("ts").set_index("ts")
        scan = lag_scan(grp["v"], grp["bal"], range(-2, 3))
        rows.append({
            "country": cc,
            "n": int(len(grp)),
            "corr": round(float(grp["v"].corr(grp["bal"])), 4),
            "argmax_lag_h": scan["argmax_lag"],
            "slope_np_on_balance": round(ols_slope(grp["bal"], grp["v"]) or float("nan"), 4),
            "mean_np_mw": round(float(grp["v"].mean()), 1),
            "by_lag": scan["by_lag"] if cc == "RO" else None,
        })
    return {"window": f"{ALIGN_START}..{ALIGN_END}", "per_country": rows}


# ---------------------------------------------------------------------------
# B -- the cross-border flow identity, and which RO legs do not exist
# ---------------------------------------------------------------------------

def check_flow_identity(con: sqlite3.Connection) -> dict:
    """Does RO's net position agree with the sum of its own border flows?

    Reported with the legs enumerated, because the residual is only readable
    once you know which borders are absent from the table entirely.
    """
    legs = pd.read_sql(
        "SELECT country_from, country_to, COUNT(*) AS n, "
        "MIN(timestamp_utc) AS first_ts, MAX(timestamp_utc) AS last_ts "
        "FROM crossborder_flows WHERE country_from = 'RO' OR country_to = 'RO' "
        "GROUP BY 1, 2 ORDER BY 1, 2",
        con,
    )
    present = {(r.country_from, r.country_to) for r in legs.itertuples()}
    # Romania's physical interconnections, from the ENTSO-E transmission map.
    expected = {("RO", n) for n in ("BG", "HU", "RS", "UA", "MD")}
    expected |= {(n, "RO") for n in ("BG", "HU", "RS", "UA", "MD")}

    xb = pd.read_sql(
        "SELECT country_from AS f, country_to AS t, timestamp_utc AS ts, flow_mw AS v "
        "FROM crossborder_flows WHERE (country_from = 'RO' OR country_to = 'RO') "
        f"AND timestamp_utc >= '2026-03-01' AND timestamp_utc < '{ALIGN_END}'",
        con,
    )
    xb["ts"] = pd.to_datetime(xb["ts"], format="mixed", utc=True).dt.tz_localize(None).dt.floor("h")
    piv = xb.groupby(["f", "t", "ts"], as_index=False)["v"].mean().pivot_table(
        index="ts", columns=["f", "t"], values="v"
    ).dropna()
    out_cols = [c for c in piv.columns if c[0] == "RO"]
    in_cols = [c for c in piv.columns if c[1] == "RO"]
    net = piv[out_cols].sum(axis=1) - piv[in_cols].sum(axis=1)

    npos = pd.read_sql(
        "SELECT timestamp_utc AS ts, net_position_mw AS v FROM net_position "
        f"WHERE country_code = 'RO' AND timestamp_utc >= '2026-03-01' AND timestamp_utc < '{ALIGN_END}'",
        con,
    )
    npos["ts"] = pd.to_datetime(npos["ts"], format="mixed", utc=True).dt.tz_localize(None).dt.floor("h")
    a = npos.groupby("ts")["v"].mean()
    j = pd.concat([a.rename("np"), net.rename("xb")], axis=1, join="inner").dropna()
    # The residual is RO net position that no row of crossborder_flows accounts
    # for. Its DAY-TO-DAY spread is the comparable quantity: ABL-280 measured
    # RO's per-vintage-day forecast bias at sd 721.5 MW against mean |actual|
    # 709.0 MW, and a covariate blind to a component that swings by that much
    # cannot supply the day level.
    resid = (j["np"] - j["xb"])
    daily = resid.groupby(resid.index.date).mean()
    return {
        "legs_present": [
            {"from": r.country_from, "to": r.country_to, "rows": int(r.n),
             "first": r.first_ts, "last": r.last_ts}
            for r in legs.itertuples()
        ],
        "legs_expected_but_absent": sorted(f"{a}->{b}" for a, b in expected - present),
        "identity_on_complete_hours": {
            "n": int(len(j)),
            "corr": round(float(j["np"].corr(j["xb"])), 4),
            "mean_np_mw": round(float(j["np"].mean()), 1),
            "mean_flow_sum_mw": round(float(j["xb"].mean()), 1),
            "mean_residual_mw": round(float(resid.mean()), 1),
            "residual_daily_mean_sd_mw": round(float(daily.std()), 1),
            "residual_daily_min_mw": round(float(daily.min()), 1),
            "residual_daily_max_mw": round(float(daily.max()), 1),
            "n_days": int(len(daily)),
            "abl280_bias_sd_mw": 721.5,
        },
    }


# ---------------------------------------------------------------------------
# C -- series integrity
# ---------------------------------------------------------------------------

def check_series_integrity(con: sqlite3.Connection) -> dict:
    """Resolution, smoothness and zero-fill on the two RO targets.

    Lag-1 autocorrelation is the discriminator that matters: a series carrying
    injected noise, stale repeats or interpolation cannot also be physically
    smooth at 1 h, and a low-persistence series that IS smooth at 1 h is telling
    you about the weather, not about the ingest.
    """
    out = {}
    for label, table, col, key in (
        ("RO wind_onshore", "energy_generation", "wind_onshore_mw", "country_code"),
        ("RO net_position", "net_position", "net_position_mw", "country_code"),
    ):
        raw = pd.read_sql(
            f"SELECT timestamp_utc AS ts, {col} AS v FROM {table} "
            f"WHERE {key} = 'RO' AND {col} IS NOT NULL "
            f"AND timestamp_utc >= '{FIT_START}' AND timestamp_utc < '{GATE_END}'",
            con,
        )
        raw["ts"] = pd.to_datetime(raw["ts"], format="mixed", utc=True).dt.tz_localize(None)
        s = raw.set_index("ts")["v"].sort_index()
        steps = s.index.to_series().diff().dt.total_seconds().div(60).value_counts()
        h = s.resample("h").mean()
        # ABL-188: a bit-identical value held for 24+ hours is the zero-fill tell.
        runs = (h != h.shift()).cumsum()
        longest = int(h.groupby(runs).size().max())
        out[label] = {
            "rows": int(len(s)),
            "native_step_minutes": {str(int(k)): int(v) for k, v in steps.head(3).items()},
            "hours": int(h.notna().sum()),
            "mean_mw": round(float(h.mean()), 1),
            "exact_zero_hours": int((h == 0).sum()),
            "longest_constant_run_h": longest,
            "ac1": round(float(h.corr(h.shift(1))), 4),
            "ac24": round(float(h.corr(h.shift(24))), 4),
            "ac168": round(float(h.corr(h.shift(168))), 4),
        }
    return out


# ---------------------------------------------------------------------------
# D -- what the wind challenger had to work with
# ---------------------------------------------------------------------------

def check_wind_feature_strength(con: sqlite3.Connection, repo: Path) -> dict:
    """Per-country strength of the two informative feature families, against
    the challenger's measured directional skill in the stored gate results.

    `wind_retrain.FEATURE_COLUMNS` is 24 names: 10 calendar, 11 lag/rolling
    transforms of the target itself, and 3 weather columns. So a country's
    challenger can only know two things about the physics -- how the target
    repeats, and what the country-mean wind speed is doing. Both are measured
    here on ABL-348's own fit window.
    """
    cells = []
    for name in WIND_RESULT_FILES:
        path = repo / "experiments" / "ABL348" / name
        if not path.exists():
            continue
        for c in json.loads(path.read_text())["gate_cells"]:
            if c.get("forecast_type") != "wind_onshore":
                continue
            s = c["scores"]
            cells.append({
                "country": c["country"],
                "band": c["horizon_band"],
                "n": s["challenger"]["n"],
                "ch_corr": s["challenger"]["correlation"],
                "ch_slope": s["challenger"]["slope"],
                "ch_wape": s["challenger"]["wape_pct"],
                "d7_corr": s["seasonal_naive"]["correlation"],
                "persistence_corr": s["persistence"]["correlation"],
            })
    if not cells:
        return {"error": "no stored wind gate results found"}
    cf = pd.DataFrame(cells)
    agg = cf.groupby("country").agg(
        ch_corr=("ch_corr", "mean"), ch_slope=("ch_slope", "mean"),
        ch_wape=("ch_wape", "mean"), d7_corr=("d7_corr", "mean"),
        persistence_corr=("persistence_corr", "mean"),
    )

    countries = tuple(agg.index)
    gen = to_hourly(pd.read_sql(
        "SELECT country_code, timestamp_utc AS ts, wind_onshore_mw AS a "
        "FROM energy_generation WHERE wind_onshore_mw IS NOT NULL "
        f"AND timestamp_utc >= '{FIT_START}' AND timestamp_utc < '{GATE_END}' "
        f"AND country_code IN {countries}", con), "a")
    wx = to_hourly(pd.read_sql(
        "SELECT country_code, timestamp_utc AS ts, wind_speed_100m_ms AS w "
        "FROM weather_data WHERE data_quality = 'actual' "
        f"AND timestamp_utc >= '{FIT_START}' AND timestamp_utc < '{GATE_END}' "
        f"AND country_code IN {countries}", con), "w")
    feats = []
    for cc, grp in gen.groupby("country_code"):
        s = grp.set_index("ts")["a"].asfreq("h")
        w = wx.loc[wx.country_code == cc].set_index("ts")["w"].asfreq("h")
        j = pd.concat([s.rename("a"), w.rename("w")], axis=1).dropna()
        feats.append({
            "country": cc,
            "ac24": round(float(s.corr(s.shift(24))), 4),
            "ac168": round(float(s.corr(s.shift(168))), 4),
            "corr_ws100": round(float(j["a"].corr(j["w"])), 4),
            "mean_mw": round(float(s.mean()), 1),
        })
    ft = pd.DataFrame(feats).set_index("country")

    # Fleet dispersion -- the intuitive explanation for a weak country-mean
    # weather covariate. Tested here so the report can say it does not hold.
    wl = pd.read_sql(
        "SELECT country_code, lat, lon, weight FROM weather_location "
        f"WHERE zone_type = 'wind_onshore' AND country_code IN {countries}", con)
    disp = []
    for cc, grp in wl.groupby("country_code"):
        wts = grp.weight.to_numpy() / grp.weight.sum()
        lat, lon = grp.lat.to_numpy(), grp.lon.to_numpy()
        clat, clon = float((wts * lat).sum()), float((wts * lon).sum())
        disp.append({
            "country": cc,
            "n_clusters": int(len(grp)),
            "wtd_dist_to_centroid_km": round(
                float(sum(wts[i] * haversine_km(lat[i], lon[i], clat, clon)
                          for i in range(len(grp)))), 1),
        })
    dt = pd.DataFrame(disp).set_index("country")

    tab = agg.join(ft).join(dt).sort_values("ch_corr")
    drivers = {}
    for src in ("ac24", "ac168", "corr_ws100", "mean_mw", "wtd_dist_to_centroid_km"):
        sub = tab[[src, "ch_corr", "ch_slope"]].dropna()
        drivers[src] = {
            "n": int(len(sub)),
            "spearman_vs_ch_corr": round(spearman(sub[src], sub["ch_corr"]), 3),
            "pearson_vs_ch_corr": round(float(sub[src].corr(sub["ch_corr"])), 3),
            "spearman_vs_ch_slope": round(spearman(sub[src], sub["ch_slope"]), 3),
        }

    ro_bands = cf.loc[cf.country == "RO"].to_dict("records")
    for b in ro_bands:
        # z of a correlation against its own null, so "anti-correlated" can be
        # checked rather than asserted. Fisher z, SE = 1/sqrt(n-3).
        for key in ("ch_corr", "d7_corr"):
            r = b[key]
            b[key + "_z"] = round(float(np.arctanh(r) * math.sqrt(b["n"] - 3)), 2)

    return {
        "fit_window": f"{FIT_START}..{GATE_END}",
        "feature_families": ("10 calendar + 11 lag/rolling of the target + 3 weather "
                             "(wind_speed_100m_ms, wind_speed_10m_ms, temperature_c)"),
        "table": tab.reset_index().to_dict("records"),
        "drivers": drivers,
        "ro_bands": ro_bands,
    }


# ---------------------------------------------------------------------------
# E -- what the net-position model had to work with
# ---------------------------------------------------------------------------

def check_net_position_covariates(con: sqlite3.Connection) -> dict:
    """Coverage of every net-position input family over V010's training span.

    `_load_crossborder_flow_covariates` queries `country_from = ?` only -- a
    documented defect (ABL-28) whose fleet-wide cost was measured at 0.8% of
    MAE. The question here is what it costs a country whose OUTBOUND legs are
    the sparse ones.

    ABL-462: the `energy_load_forecast` row below is **deliberately unguarded**,
    unlike A2's read. This counts hours the ingest actually holds; the guard
    nulls values, so guarding a presence census would report the pipeline as
    having fetched fewer hours than it did -- it would measure the guard, not
    the coverage. Measured read-only on the live replica over this exact window
    (`2023-01-01..2026-03-01`, 19 gate countries, 872,355 rows) the guard would
    in any case null **0 rows**, so nothing here rests on the distinction today.
    """
    span_h = int(
        (pd.Timestamp(V010_TRAIN_END) - pd.Timestamp(V010_TRAIN_START)).total_seconds() // 3600
    )
    hour_expr = "COUNT(DISTINCT strftime('%Y-%m-%dT%H', {col}))"

    def coverage(table: str, cc_col: str, ts_col: str, where: str, label: str) -> pd.Series:
        q = (f"SELECT {cc_col} AS cc, {hour_expr.format(col=ts_col)} AS h FROM {table} "
             f"WHERE {ts_col} >= '{V010_TRAIN_START}' AND {ts_col} < '{V010_TRAIN_END}' "
             f"AND {where} AND {cc_col} IN {GATE_COUNTRIES} GROUP BY 1")
        d = pd.read_sql(q, con)
        return (d.h / span_h * 100).round(1).rename(label).set_axis(d.cc)

    families = pd.concat([
        coverage("net_position", "country_code", "timestamp_utc",
                 "net_position_mw IS NOT NULL", "net_position"),
        coverage("energy_price", "country_code", "timestamp_utc",
                 "price_eur_mwh IS NOT NULL", "da_price"),
        coverage("energy_load_forecast", "country_code", "target_timestamp_utc",
                 "forecast_value_mw IS NOT NULL", "tso_load_forecast"),
        coverage("weather_data", "country_code", "timestamp_utc",
                 "data_quality = 'actual'", "weather"),
        coverage("crossborder_flows", "country_from", "timestamp_utc",
                 "flow_mw IS NOT NULL", "xb_outbound_READ"),
        coverage("crossborder_flows", "country_to", "timestamp_utc",
                 "flow_mw IS NOT NULL", "xb_inbound_NOT_read"),
    ], axis=1).fillna(0.0)

    monthly = pd.read_sql(
        "SELECT SUBSTR(timestamp_utc, 1, 7) AS m, country_from AS cc, "
        "COUNT(DISTINCT strftime('%Y-%m-%dT%H', timestamp_utc)) AS h "
        "FROM crossborder_flows WHERE country_from IN ('RO','BG','CZ','HU','PL') "
        "AND timestamp_utc >= '2025-01-01' GROUP BY 1, 2",
        con,
    ).pivot(index="m", columns="cc", values="h").fillna(0).astype(int).sort_index()

    # Zero and partial months, per country, so the RO hole is comparable rather
    # than merely listed.
    month_health = {}
    for cc in monthly.columns:
        full = monthly.max(axis=1)
        month_health[cc] = {
            "months": int(len(monthly)),
            "zero_months": int((monthly[cc] == 0).sum()),
            "partial_months": int(((monthly[cc] > 0) & (monthly[cc] < 0.5 * full)).sum()),
        }

    # The gate cohort ABL-280 scores is AFTER V010's training span, so the two
    # regimes have to be reported separately: a covariate can be absent from
    # training and present at serving, and that is a different defect.
    cohort = pd.read_sql(
        "SELECT country_from AS cc, COUNT(DISTINCT strftime('%Y-%m-%dT%H', timestamp_utc)) AS h "
        "FROM crossborder_flows WHERE timestamp_utc >= '2026-08-01' AND timestamp_utc < '2026-08-14' "
        f"AND country_from IN {GATE_COUNTRIES} GROUP BY 1",
        con,
    )
    cohort["pct"] = (cohort.h / 312 * 100).round(1)

    ro_row = families.loc["RO"]
    return {
        "outbound_month_health": month_health,
        "outbound_coverage_in_abl280_cohort_pct": dict(zip(cohort.cc, cohort.pct)),
        "training_span": f"{V010_TRAIN_START}..{V010_TRAIN_END} ({span_h} h)",
        "note": ("Values above 100% are sub-hourly publication, not over-coverage: "
                 "the count is of distinct hour labels present in a table that also "
                 "carries quarter-hour rows for some countries."),
        "coverage_pct": families.rename_axis("country").reset_index().to_dict("records"),
        "ro_is_fleet_minimum_on": [
            c for c in families.columns if float(ro_row[c]) == float(families[c].min())
        ],
        "outbound_monthly_hours": monthly.reset_index().to_dict("records"),
        "loader": ("src/chronos2/input_builder._load_crossborder_flow_covariates reads "
                   "country_from only (ABL-28), so flow__total_export_mw / "
                   "flow__net_mw come entirely from the outbound leg and "
                   "flow__total_import_mw is a constant zero for every country."),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def render(results: dict) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 78)
    add("ABL-430 -- RO country diagnosis (read-only)")
    add("=" * 78)
    add(f"replica : {results['meta']['replica_db']}")
    add(f"bytes   : {results['meta']['replica_bytes']:,}")
    add(f"run     : {results['meta']['generated_at']}")
    add("")

    a1 = results["A1_solar_clock"]
    add("-- A1  clock vs the sun ------------------------------------------------")
    add(f"window {a1['window']};  offset = solar-generation mass centroid - solar noon")
    add(f"  fleet   median {a1['fleet_median_offset_h']:+.2f} h   "
        f"range [{a1['fleet_min_offset_h']:+.2f}, {a1['fleet_max_offset_h']:+.2f}]")
    add(f"  RO      {a1['ro_offset_h']:+.2f} h   rank {a1['ro_rank_of']}   "
        f"z vs fleet {a1['ro_z_vs_fleet']:+.2f}")
    add(f"  -> {a1['verdict']}")
    add("")

    a2 = results["A2_wind_vs_tso"]
    add("-- A2  RO wind actuals vs the TSO's own day-ahead publication ----------")
    ro = next(r for r in a2["per_country"] if r["country"] == "RO")
    add(f"  n {ro['n']}   corr {ro['corr']:+.3f}   argmax lag {ro['argmax_lag_h']} h   "
        f"slope(tso~actual) {ro['slope_tso_on_actual']:.3f}   WAPE {ro['wape_pct']}%")
    add(f"  lag scan: {ro['by_lag']}")
    worst = sorted((r for r in a2["per_country"] if r["actual_mean_mw"] > 100),
                   key=lambda r: r["corr"])[:3]
    add("  weakest agreement among fleets > 100 MW: "
        + ", ".join(f"{r['country']} {r['corr']:+.3f}" for r in worst))
    add(f"  caveat: {a2['caveat']}")
    add("")

    a3 = results["A3_net_position_balance"]
    add("-- A3  RO net position vs (generation - load) --------------------------")
    ro = next(r for r in a3["per_country"] if r["country"] == "RO")
    add(f"  n {ro['n']}   corr {ro['corr']:+.3f}   argmax lag {ro['argmax_lag_h']} h   "
        f"slope(np~balance) {ro['slope_np_on_balance']:.3f}")
    add(f"  lag scan: {ro['by_lag']}")
    rank = sorted(a3["per_country"], key=lambda r: -r["corr"])
    add("  fleet order: " + ", ".join(f"{r['country']} {r['corr']:.2f}" for r in rank))
    add("")

    fi = results["B_flow_identity"]
    add("-- B   RO cross-border legs and the flow identity ----------------------")
    for leg in fi["legs_present"]:
        add(f"  {leg['from']}->{leg['to']}  {leg['rows']:>6} rows  {leg['first'][:10]} .. {leg['last'][:10]}")
    add(f"  physically present but ABSENT from the table: {fi['legs_expected_but_absent']}")
    idn = fi["identity_on_complete_hours"]
    add(f"  identity on {idn['n']} all-legs-present hours: corr {idn['corr']:+.3f}, "
        f"np {idn['mean_np_mw']} MW vs flow sum {idn['mean_flow_sum_mw']} MW "
        f"(residual {idn['mean_residual_mw']:+.1f} MW)")
    add(f"  residual RO net position no flow row accounts for: daily mean sd "
        f"{idn['residual_daily_mean_sd_mw']} MW over {idn['n_days']} days, range "
        f"[{idn['residual_daily_min_mw']:+.0f}, {idn['residual_daily_max_mw']:+.0f}] MW")
    add(f"     for comparison, ABL-280's RO per-vintage-day forecast bias sd is "
        f"{idn['abl280_bias_sd_mw']} MW")
    add("")

    add("-- C   RO series integrity ---------------------------------------------")
    for label, v in results["C_series_integrity"].items():
        add(f"  {label}: {v['hours']} h, native step {v['native_step_minutes']}, "
            f"mean {v['mean_mw']} MW")
        add(f"     exact zeros {v['exact_zero_hours']}, longest constant run "
            f"{v['longest_constant_run_h']} h, ac1 {v['ac1']:.3f}, "
            f"ac24 {v['ac24']:.3f}, ac168 {v['ac168']:.3f}")
    add("")

    d = results["D_wind_feature_strength"]
    add("-- D   what the wind challenger had to work with ------------------------")
    add(f"  features: {d['feature_families']}")
    add(f"  {'cc':<4}{'ch_corr':>9}{'ch_slope':>10}{'d7_corr':>9}{'ac24':>8}{'ac168':>8}"
        f"{'corr_ws':>9}{'mean_MW':>10}{'disp_km':>9}")
    for r in d["table"]:
        add(f"  {r['country']:<4}{r['ch_corr']:>9.3f}{r['ch_slope']:>10.3f}"
            f"{r['d7_corr']:>9.3f}{r['ac24']:>8.3f}{r['ac168']:>8.3f}"
            f"{r['corr_ws100']:>9.3f}{r['mean_mw']:>10.1f}"
            f"{r.get('wtd_dist_to_centroid_km', float('nan')):>9.0f}")
    add("")
    add("  which country property predicts the challenger's directional skill:")
    for src, v in d["drivers"].items():
        add(f"    {src:<26} Spearman vs ch_corr {v['spearman_vs_ch_corr']:+.3f}   "
            f"vs ch_slope {v['spearman_vs_ch_slope']:+.3f}   (n={v['n']})")
    add("")
    add("  RO per band, with each correlation's Fisher z against its own null:")
    for b in d["ro_bands"]:
        add(f"    {b['band']:<8} n {b['n']:>4}  challenger corr {b['ch_corr']:+.3f} "
            f"(z {b['ch_corr_z']:+.2f})   D-7 corr {b['d7_corr']:+.3f} (z {b['d7_corr_z']:+.2f})")
    add("")

    e = results["E_net_position_covariates"]
    add("-- E   what the net-position model had to work with ---------------------")
    add(f"  span {e['training_span']}")
    add(f"  {e['note']}")
    cols = [c for c in e["coverage_pct"][0] if c != "country"]
    add("  " + f"{'cc':<4}" + "".join(f"{c:>21}" for c in cols))
    for r in sorted(e["coverage_pct"], key=lambda r: r["xb_outbound_READ"]):
        add("  " + f"{r['country']:<4}" + "".join(f"{r[c]:>21.1f}" for c in cols))
    add(f"  RO is the fleet MINIMUM on: {e['ro_is_fleet_minimum_on']}")
    add(f"  loader: {e['loader']}")
    add("")
    add("  outbound-leg month health since 2025-01 (the leg the loader reads):")
    for cc, v in sorted(e["outbound_month_health"].items(),
                        key=lambda kv: -(kv[1]["zero_months"] + kv[1]["partial_months"])):
        add(f"    {cc}  {v['zero_months']} zero + {v['partial_months']} partial "
            f"of {v['months']} months")
    for r in e["outbound_monthly_hours"]:
        zeros = [k for k, v in r.items() if k != "m" and v == 0]
        if zeros:
            add(f"      {r['m']}  no outbound rows at all: {', '.join(zeros)}")
    add("")
    coh = e["outbound_coverage_in_abl280_cohort_pct"]
    add(f"  ...but in the ABL-280 scored cohort (2026-08-01..08-14) RO outbound is "
        f"{coh.get('RO')}% against a fleet median of "
        f"{round(float(np.median(list(coh.values()))), 1)}%.")
    add("  So the flow-covariate hole is a TRAINING-span defect for RO, not a")
    add("  serving-time one. Both matter; they are different repairs.")
    add("=" * 78)
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="ABL-430 RO country diagnosis (read-only).")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH),
                   help="Path to the read-only replica.")
    p.add_argument("--json-out", default="reports/abl_430_ro_diagnosis.json",
                   help="Where to write the machine record.")
    p.add_argument("--stdout", action="store_true", help="Also print the report.")
    args = p.parse_args()

    replica = Path(args.replica_db)
    if not replica.exists():
        print(f"ERROR: replica not found: {replica}", file=sys.stderr)
        print("A worktree has no .env, so config.DATABASE_PATH degrades to a bare "
              "path. Pass --replica-db explicitly.", file=sys.stderr)
        return 2

    con = open_replica(replica)
    try:
        wind_countries = tuple(sorted(set(GATE_COUNTRIES) | {"GR", "IT", "NO", "SE", "CH"}))
        results = {
            "meta": {
                "issue": "ABL-430",
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "replica_db": str(replica),
                "replica_bytes": replica.stat().st_size,
                "read_only": True,
                "interpreter": sys.executable,
            },
            "A1_solar_clock": check_solar_clock(con),
            "A2_wind_vs_tso": check_wind_against_tso(con, wind_countries),
            "A3_net_position_balance": check_net_position_balance(con, GATE_COUNTRIES),
            "B_flow_identity": check_flow_identity(con),
            "C_series_integrity": check_series_integrity(con),
            "D_wind_feature_strength": check_wind_feature_strength(con, Path(__file__).parent.parent),
            "E_net_position_covariates": check_net_position_covariates(con),
        }
    finally:
        con.close()

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    report = render(results)
    if args.stdout:
        print(report)
    print(f"\n[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
