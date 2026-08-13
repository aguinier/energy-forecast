#!/usr/bin/env python3
"""ABL-411: settle whether Red Electrica's *solar termica* accounts for ES's
overnight solar MW, or falsifies the concentrated-solar-power (CSP) reading.

Why this probe exists
---------------------
The ABL-396 screen (`reports/abl_396_solar_night_floor_screen.md` section 3)
measured ES booking 385-992 MW of `solar_mw` at sun elevations down to -28.7
degrees, 1.35% of all its solar energy, on 100% of gate-window night hours.
That screen argued the output is genuine CSP dispatch from thermal storage
rather than contamination, from five aggregate properties: a within-month
detrended day-to-night correlation of r = +0.515 over 585 days, a 42 MW
(December) to 599 MW (July) seasonal swing, a monotone eight-hour discharge
curve, a magnitude consistent with Spain's ~2.3 GW CSP fleet, and a separately
reported `energy_storage_mw` averaging 1.6 MW.

Every one of those is inference from an aggregate. ENTSO-E folds CSP and PV into
a single production type (B16), so `solar_mw` cannot distinguish them. The
screen named its own falsifier: Red Electrica publishes *solar termica*
separately from *solar fotovoltaica*. This probe fetches that series and asks
the question directly.

**The falsifier is the point.** If `solTer` does not account for ES's overnight
MW, the CSP reading is wrong and ABL-396's second-ranked country is contaminated
after all. That outcome is reported as loudly as the confirmation.

Sources
-------
`demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula`
serves peninsular Spain at 5-minute resolution with `solFot` (fotovoltaica),
`solTer` (termica) and `sol` (their sum) as separate fields. Timestamps are
Europe/Madrid local: the response pads the requested local day by three hours on
each side (361 rows x 5 min = 30 h), and PV peaks at 14:00 local, i.e. 12:00 UTC.

That archive serves 2021-01-01 through 2025-12-14. From 2025-12-15 onward it
answers `curva DEMANDAQH no valida` while the sibling `demandaInstantanea`
endpoint still returns a current timestamp -- an archive gap, not an outage. The
ABL-348 registered windows (fit 2026-01-14 -> 2026-07-11, gate 2026-07-11 ->
2026-08-10) therefore cannot be covered at hourly resolution from this source;
`--daily-check` runs the coarser energy-budget test over them using
`apidatos.ree.es` `estructura-generacion` at `time_trunc=day`, which does cover
2026.

Read-only. Fetches a public series over HTTP, reads the replica with `mode=ro`,
writes nothing but its own JSON and its HTTP cache. No ingest path is added:
the cache is scratch, and nothing here writes to either database.

Scope note
----------
REE's peninsular perimeter and the ENTSO-E `ES` bidding zone are both peninsular
Spain, and `solar_geometry.SOLAR_REPRESENTATIVE_POINTS['ES']` is (39.125,
-4.129), central peninsula. The Canary and Balearic systems are outside both.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.solar_features import night_mask  # noqa: E402
from src.solar_geometry import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_REPRESENTATIVE_POINTS,
    sun_elevation_deg,
)

QH_URL = (
    "https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/"
    "demandaGeneracionPeninsula?curva=DEMANDAQH&fecha={day}"
)
APIDATOS_URL = (
    "https://apidatos.ree.es/en/datos/generacion/estructura-generacion"
    "?start_date={start}T00:00&end_date={end}T23:59&time_trunc=day"
)
MADRID = "Europe/Madrid"

#: REE technology titles in the apidatos daily response.
REE_PV_TITLE = "Solar photovoltaic"
REE_CSP_TITLE = "Thermal solar"

#: 1 MW, ABL-338's threshold, so counts stay comparable to ABL-396.
NIGHT_THRESHOLD_MW = 1.0

_JSONP = re.compile(r"^[A-Za-z_0-9$]*\((.*)\);?\s*$", re.DOTALL)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def _http_get(url: str, retries: int = 3, pause: float = 1.5) -> str:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "able-energy-forecast/abl411"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover - network
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last})")


def fetch_qh_day(day: date, cache_dir: Path) -> Optional[List[dict]]:
    """5-minute peninsular generation for one Europe/Madrid calendar day.

    Returns None when REE has no archive for that day (the post-2025-12-14 gap),
    so callers can distinguish 'no data' from 'zero generation'.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"qh_{day.isoformat()}.json"
    if path.exists():
        raw = path.read_text(encoding="utf-8")
    else:
        raw = _http_get(QH_URL.format(day=day.isoformat()))
        path.write_text(raw, encoding="utf-8")

    raw = raw.strip()
    if "no valida" in raw or not raw.startswith(("null(", "{", "JSON")):
        match = _JSONP.match(raw)
        if match is None:
            return None
    match = _JSONP.match(raw)
    body = match.group(1) if match else raw
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    rows = payload.get("valoresHorariosGeneracion")
    if not rows:
        return None
    # Drop the +/-3h padding: keep only rows whose local date is the requested one.
    stamp = day.isoformat()
    return [r for r in rows if str(r.get("ts", "")).startswith(stamp)]


def fetch_qh_range(start: date, end: date, cache_dir: Path, workers: int) -> Tuple[pd.DataFrame, List[str]]:
    """Hourly-UTC frame of `solFot` / `solTer` / `sol` over a local-day range."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    missing: List[str] = []
    collected: List[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for day, rows in zip(days, pool.map(lambda d: fetch_qh_day(d, cache_dir), days)):
            if rows is None:
                missing.append(day.isoformat())
                continue
            collected.extend(rows)

    if not collected:
        return pd.DataFrame(columns=["solFot", "solTer", "sol"]), missing

    frame = pd.DataFrame(collected)
    # REE disambiguates the autumn fold in the label itself: on the transition
    # day the repeated local hour is written "2A:mm" (first pass, still CEST)
    # and "2B:mm" (second pass, CET). Nothing has to be inferred from ordering.
    raw_ts = frame["ts"].astype(str)
    is_dst = ~raw_ts.str.contains(r" \d[B]:", regex=True)
    frame["ts"] = pd.to_datetime(
        raw_ts.str.replace(r" (\d)[AB]:", r" 0\1:", regex=True), format="%Y-%m-%d %H:%M"
    )
    frame["_is_dst"] = is_dst.to_numpy()
    frame = frame.sort_values(["ts", "_is_dst"], ascending=[True, False])
    frame = frame.drop_duplicates(subset=["ts", "_is_dst"], keep="first")
    local = pd.DatetimeIndex(frame["ts"])
    # A spring-forward gap simply has no rows, so `nonexistent` should never fire.
    localised = local.tz_localize(
        MADRID, ambiguous=frame["_is_dst"].to_numpy(), nonexistent="raise"
    )
    frame.index = localised.tz_convert("UTC").tz_localize(None)
    frame = frame.drop(columns=["_is_dst"])
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()

    keep = [c for c in ("solFot", "solTer", "sol") if c in frame.columns]
    numeric = frame[keep].apply(pd.to_numeric, errors="coerce")
    hourly = numeric.resample("h").mean()
    counts = numeric["solTer"].resample("h").count() if "solTer" in numeric else None
    if counts is not None:
        hourly = hourly[counts >= 6]  # at least half an hour of 5-minute samples
    return hourly, missing


def fetch_daily_structure(start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """Daily REE generation energy (MWh) per technology, from apidatos."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"apidatos_{start.isoformat()}_{end.isoformat()}.json"
    if path.exists():
        raw = path.read_text(encoding="utf-8")
    else:
        raw = _http_get(APIDATOS_URL.format(start=start.isoformat(), end=end.isoformat()))
        path.write_text(raw, encoding="utf-8")

    payload = json.loads(raw)
    if "included" not in payload:
        raise RuntimeError(f"apidatos returned no data for {start}..{end}: {raw[:200]}")

    series: Dict[str, pd.Series] = {}
    for item in payload["included"]:
        title = item.get("type")
        values = item.get("attributes", {}).get("values") or []
        if not values:
            continue
        idx = pd.to_datetime([v["datetime"] for v in values], utc=True, format="ISO8601")
        # apidatos stamps local midnight; the calendar day is the local date.
        local_day = idx.tz_convert(MADRID).date
        series[title] = pd.Series([v["value"] for v in values], index=pd.Index(local_day, name="day"))
    return pd.DataFrame(series)


# --------------------------------------------------------------------------
# replica
# --------------------------------------------------------------------------
def read_replica_solar(db_path: str, table: str, start: date, end: date) -> pd.DataFrame:
    """Hourly-UTC ES `solar_mw` from one source table.

    Date-only bounds keep the mixed timestamp format off the index (a normalising
    expression in the WHERE clause full-scans the 9.4 GB replica).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = (
            f"SELECT timestamp_utc, solar_mw FROM {table} "
            "WHERE country_code = 'ES' AND solar_mw IS NOT NULL "
            "AND timestamp_utc >= ? AND timestamp_utc < ? ORDER BY timestamp_utc"
        )
        frame = pd.read_sql_query(
            sql, conn, params=(start.isoformat(), (end + timedelta(days=1)).isoformat())
        )
    finally:
        conn.close()

    if frame.empty:
        return pd.DataFrame(columns=["solar_mw"])
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    raw = pd.Series(frame["solar_mw"].to_numpy(dtype=float), index=pd.DatetimeIndex(stamps))
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    hourly = raw.resample("h").mean().to_frame("solar_mw")
    return hourly.dropna()


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _safe_corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
    }


def lag_scan(replica: pd.Series, ree: pd.Series, max_lag: int = 3) -> Dict[str, object]:
    """Confirm the two series are aligned before any per-hour claim is made."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = ree.shift(lag)
        joined = pd.concat([replica, shifted], axis=1).dropna()
        if len(joined) < 24:
            continue
        out[str(lag)] = {
            "n": int(len(joined)),
            "corr": _safe_corr(joined.iloc[:, 0], joined.iloc[:, 1]),
            "mae_mw": float(np.mean(np.abs(joined.iloc[:, 0] - joined.iloc[:, 1]))),
        }
    best = min(out.items(), key=lambda kv: kv[1]["mae_mw"]) if out else (None, None)
    return {"by_lag": out, "best_lag_hours": None if best[0] is None else int(best[0])}


def analyse(
    replica: pd.DataFrame,
    ree: pd.DataFrame,
    table: str,
) -> Dict[str, object]:
    joined = replica.join(ree, how="inner").dropna(subset=["solar_mw", "solTer", "solFot"])
    if joined.empty:
        return {"source_table": table, "error": "no overlapping hours"}

    hours = pd.DatetimeIndex(joined.index)
    night = night_mask("ES", hours)
    elevation = np.asarray(sun_elevation_deg("ES", hours + pd.Timedelta(minutes=30)), dtype=float)
    joined = joined.assign(is_night=night, sun_elevation_deg=elevation)

    # The defensible total is the two components summed. REE's own `sol` field
    # agrees to rounding except on 2025-04-28, the Iberian blackout, where it
    # disagrees by up to 3 GW -- so `sol` is reported as a check, never used.
    total = joined["solFot"] + joined["solTer"]
    sol_gap = (joined["sol"] - total).abs() if "sol" in joined else pd.Series(dtype=float)

    result: Dict[str, object] = {
        "source_table": table,
        "window_utc": [str(joined.index[0]), str(joined.index[-1])],
        "n_hours": int(len(joined)),
        "n_night_hours": int(night.sum()),
        "alignment": {
            "ree_sol_field_vs_components": {
                "note": "REE's own `sol` field against solFot+solTer; unused, reported as a check",
                "max_abs_gap_mw": float(sol_gap.max()) if len(sol_gap) else None,
                "hours_gap_above_5mw": int((sol_gap > 5).sum()) if len(sol_gap) else None,
                "worst_day": str(sol_gap.idxmax().date()) if len(sol_gap) and sol_gap.max() > 0 else None,
            },
            "all_hours": {
                "corr_replica_vs_ree_total": _safe_corr(joined["solar_mw"], total),
                "mae_mw": float(np.mean(np.abs(joined["solar_mw"] - total))),
                "mean_replica_mw": float(joined["solar_mw"].mean()),
                "mean_ree_total_mw": float(total.mean()),
                "mean_signed_gap_mw": float((joined["solar_mw"] - total).mean()),
            },
            "lag_scan": lag_scan(joined["solar_mw"], total),
        },
    }

    # ---- the question: what is ES booking when the sun is down? ----
    nightf = joined[joined["is_night"]]
    dayf = joined[~joined["is_night"]]

    def band(frame: pd.DataFrame) -> Dict[str, object]:
        if frame.empty:
            return {"n": 0}
        ree_total = frame["solFot"] + frame["solTer"]
        rep = frame["solar_mw"]
        return {
            "n": int(len(frame)),
            "replica_solar_mw": _stats(rep.to_numpy()),
            "ree_csp_solTer_mw": _stats(frame["solTer"].to_numpy()),
            "ree_pv_solFot_mw": _stats(frame["solFot"].to_numpy()),
            "share_of_replica_explained_by_csp": float(frame["solTer"].sum() / rep.sum())
            if rep.sum() > 0
            else None,
            "share_of_replica_explained_by_pv": float(frame["solFot"].sum() / rep.sum())
            if rep.sum() > 0
            else None,
            "corr_replica_vs_csp": _safe_corr(rep, frame["solTer"]),
            "corr_replica_vs_pv": _safe_corr(rep, frame["solFot"]),
            "residual_replica_minus_ree_total_mw": _stats((rep - ree_total).to_numpy()),
            "hours_above_threshold": {
                "replica": int((rep > NIGHT_THRESHOLD_MW).sum()),
                "ree_csp": int((frame["solTer"] > NIGHT_THRESHOLD_MW).sum()),
                "ree_pv": int((frame["solFot"] > NIGHT_THRESHOLD_MW).sum()),
            },
        }

    result["night"] = band(nightf)
    result["not_night"] = band(dayf)

    # The single number the falsifier turns on: how much of the replica's night
    # MW does REE's own PV + CSP split account for, hour by hour?
    if not nightf.empty:
        resid = nightf["solar_mw"] - (nightf["solFot"] + nightf["solTer"])
        result["night_residual"] = {
            "pct_of_replica_night_energy_explained_by_ree": float(
                100.0 * (nightf["solFot"] + nightf["solTer"]).sum() / nightf["solar_mw"].sum()
            ),
            "mae_mw": float(resid.abs().mean()),
            "median_abs_mw": float(resid.abs().median()),
            "p95_abs_mw": float(resid.abs().quantile(0.95)),
            "mean_replica_mw": float(nightf["solar_mw"].mean()),
        }

        # ABL-396 read a monotone discharge curve off the aggregate. A stuck or
        # forward-filled value would also look flat, so check the flattest
        # nights against REE: if REE is flat at the same level, the flatness is
        # the TSO's, not the replica's.
        by_night = nightf.assign(day=nightf.index.date).groupby("day")
        per_night = by_night.apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "rep_range": g["solar_mw"].max() - g["solar_mw"].min(),
                    "rep_mean": g["solar_mw"].mean(),
                    "ree_range": (g["solFot"] + g["solTer"]).max() - (g["solFot"] + g["solTer"]).min(),
                    "ree_mean": (g["solFot"] + g["solTer"]).mean(),
                }
            ),
            include_groups=False,
        )
        flat = per_night[(per_night["n"] >= 6) & (per_night["rep_range"] <= 4.0)]
        result["flat_night_check"] = {
            "note": "nights where the replica holds one value (range <= 4 MW, the ES quantum)",
            "n_nights_total": int(len(per_night)),
            "n_nights_flat": int(len(flat)),
            "on_flat_nights": {
                "mean_replica_mw": float(flat["rep_mean"].mean()) if len(flat) else None,
                "mean_ree_mw": float(flat["ree_mean"].mean()) if len(flat) else None,
                "mean_ree_range_mw": float(flat["ree_range"].mean()) if len(flat) else None,
            },
        }

    # Energy shares, the ABL-396 rankable quantity, computed on both series.
    tot_energy = joined["solar_mw"].sum()
    result["energy"] = {
        "pct_of_replica_energy_at_night": float(100.0 * nightf["solar_mw"].sum() / tot_energy)
        if tot_energy > 0
        else None,
        "pct_of_ree_csp_energy_at_night": float(100.0 * nightf["solTer"].sum() / joined["solTer"].sum())
        if joined["solTer"].sum() > 0
        else None,
        "pct_of_ree_pv_energy_at_night": float(100.0 * nightf["solFot"].sum() / joined["solFot"].sum())
        if joined["solFot"].sum() > 0
        else None,
        "night_replica_mwh": float(nightf["solar_mw"].sum()),
        "night_ree_csp_mwh": float(nightf["solTer"].sum()),
        "night_ree_pv_mwh": float(nightf["solFot"].sum()),
    }

    # Monthly night means: the seasonality ABL-396 read off the aggregate,
    # now split into its two components.
    result["monthly_night_mean_mw"] = []
    for month, g in nightf.groupby(nightf.index.to_period("M").astype(str)):
        rep_sum = g["solar_mw"].sum()
        result["monthly_night_mean_mw"].append(
            {
                "month": month,
                "n_hours": int(len(g)),
                "replica_mw": float(g["solar_mw"].mean()),
                "ree_csp_mw": float(g["solTer"].mean()),
                "ree_pv_mw": float(g["solFot"].mean()),
                "csp_share_pct": float(100.0 * g["solTer"].sum() / rep_sum) if rep_sum else None,
                "pv_share_pct": float(100.0 * g["solFot"].sum() / rep_sum) if rep_sum else None,
                "residual_mean_mw": float(
                    (g["solar_mw"] - g["solTer"] - g["solFot"]).mean()
                ),
                "corr_replica_vs_csp": _safe_corr(g["solar_mw"], g["solTer"]),
            }
        )

    # Hour-of-day discharge profile on night hours only.
    prof = (
        nightf.assign(hour=nightf.index.hour)
        .groupby("hour")[["solar_mw", "solTer", "solFot", "sun_elevation_deg"]]
        .mean()
        .reset_index()
    )
    result["night_hour_profile"] = [
        {
            "hour_utc": int(row["hour"]),
            "replica_mw": float(row["solar_mw"]),
            "ree_csp_mw": float(row["solTer"]),
            "ree_pv_mw": float(row["solFot"]),
            "mean_sun_elevation_deg": float(row["sun_elevation_deg"]),
        }
        for _, row in prof.iterrows()
    ]

    # ABL-396's charge-coupling statistic, recomputed on the CSP series itself.
    daily = joined.assign(day=joined.index.date)
    day_energy = daily[~daily["is_night"]].groupby("day")[["solar_mw", "solFot", "solTer"]].sum()
    night_energy = daily[daily["is_night"]].groupby("day")[["solar_mw", "solTer"]].sum()
    charge = pd.DataFrame(
        {
            "daylight_replica": day_energy["solar_mw"],
            "daylight_pv": day_energy["solFot"],
            "night_replica": night_energy["solar_mw"],
            "night_csp": night_energy["solTer"],
        }
    ).dropna()
    if len(charge) >= 30:
        months = pd.PeriodIndex(pd.to_datetime(charge.index), freq="M")
        detr = charge.groupby(months).transform(lambda s: s - s.mean())
        result["charge_coupling_detrended"] = {
            "n_days": int(len(charge)),
            "corr_daylight_replica_vs_night_replica": _safe_corr(
                detr["daylight_replica"], detr["night_replica"]
            ),
            "corr_daylight_pv_vs_night_csp": _safe_corr(detr["daylight_pv"], detr["night_csp"]),
            "corr_night_replica_vs_night_csp": _safe_corr(detr["night_replica"], detr["night_csp"]),
        }
    return result


def daily_source_calibration(ree_hourly: pd.DataFrame, daily: pd.DataFrame) -> Dict[str, object]:
    """Calibrate the daily `apidatos` CSP series against the 5-minute archive.

    The two are different REE products. Where they overlap they agree closely on
    high-output days, but `apidatos` under-reports CSP badly on low-output days,
    which is exactly the regime the winter half of any `night <= CSP` test falls
    in. Quoting the daily check without this calibration would manufacture
    contamination out of a product difference.
    """
    if ree_hourly.empty or daily.empty or REE_CSP_TITLE not in daily:
        return {"error": "no overlap between the two REE products"}
    qh = ree_hourly.assign(day=ree_hourly.index.date).groupby("day")["solTer"].sum()
    joined = pd.DataFrame({"qh_csp": qh, "ap_csp": daily[REE_CSP_TITLE]}).dropna()
    if joined.empty:
        return {"error": "no overlapping days"}
    bins = [-1.0, 200.0, 1000.0, 3000.0, 10000.0, 30000.0, float("inf")]
    joined["bin"] = pd.cut(joined["ap_csp"], bins)
    rows = []
    for label, g in joined.groupby("bin", observed=True):
        ratio = g["qh_csp"] / g["ap_csp"].replace(0.0, np.nan)
        rows.append(
            {
                "apidatos_daily_csp_mwh_band": str(label),
                "n_days": int(len(g)),
                "mean_apidatos_mwh": float(g["ap_csp"].mean()),
                "mean_5min_archive_mwh": float(g["qh_csp"].mean()),
                "median_ratio_archive_over_apidatos": float(ratio.median()),
            }
        )
    return {
        "n_days": int(len(joined)),
        "corr": _safe_corr(joined["qh_csp"], joined["ap_csp"]),
        "by_output_band": rows,
    }


def daily_budget_check(
    replica: pd.DataFrame, daily: pd.DataFrame, label: str
) -> Dict[str, object]:
    """Coarse energy-budget test for windows the 5-minute archive cannot reach.

    Asks whether the replica's *nightly* solar energy fits inside REE's *daily*
    CSP energy, and whether the two move together day to day. It cannot place
    CSP output in a particular hour -- that is what the hourly test is for.
    """
    if replica.empty or daily.empty:
        return {"label": label, "error": "no data"}

    hours = pd.DatetimeIndex(replica.index)
    night = night_mask("ES", hours)
    frame = replica.assign(is_night=night, day=hours.date)
    night_mwh = frame[frame["is_night"]].groupby("day")["solar_mw"].sum()
    total_mwh = frame.groupby("day")["solar_mw"].sum()
    full_days = frame.groupby("day")["solar_mw"].count()
    keep = full_days[full_days == 24].index
    night_mwh = night_mwh.reindex(keep).dropna()
    total_mwh = total_mwh.reindex(keep).dropna()

    csp = daily.get(REE_CSP_TITLE)
    pv = daily.get(REE_PV_TITLE)
    if csp is None or pv is None:
        return {"label": label, "error": "apidatos response lacks the solar split"}

    joined = pd.DataFrame(
        {"night_mwh": night_mwh, "total_mwh": total_mwh, "ree_csp_mwh": csp, "ree_pv_mwh": pv}
    ).dropna()
    if joined.empty:
        return {"label": label, "error": "no overlapping days"}

    ratio = joined["night_mwh"] / joined["ree_csp_mwh"]
    months = pd.PeriodIndex(pd.to_datetime(joined.index), freq="M")
    detr = joined.groupby(months).transform(lambda s: s - s.mean())
    return {
        "label": label,
        "window": [str(joined.index.min()), str(joined.index.max())],
        "n_days": int(len(joined)),
        "aggregation_identity": {
            "note": "replica daily solar energy vs REE (PV + CSP) daily energy",
            "corr": _safe_corr(joined["total_mwh"], joined["ree_pv_mwh"] + joined["ree_csp_mwh"]),
            "mean_replica_mwh": float(joined["total_mwh"].mean()),
            "mean_ree_pv_plus_csp_mwh": float((joined["ree_pv_mwh"] + joined["ree_csp_mwh"]).mean()),
            "mean_pct_gap": float(
                100.0
                * (
                    (joined["total_mwh"] - (joined["ree_pv_mwh"] + joined["ree_csp_mwh"]))
                    / (joined["ree_pv_mwh"] + joined["ree_csp_mwh"])
                ).mean()
            ),
        },
        "night_fits_inside_csp_budget": {
            "days_night_le_csp": int((joined["night_mwh"] <= joined["ree_csp_mwh"]).sum()),
            "days_total": int(len(joined)),
            "ratio_night_over_csp": _stats(ratio.to_numpy()),
        },
        "corr_night_vs_csp_raw": _safe_corr(joined["night_mwh"], joined["ree_csp_mwh"]),
        "corr_night_vs_pv_raw": _safe_corr(joined["night_mwh"], joined["ree_pv_mwh"]),
        "corr_night_vs_csp_detrended": _safe_corr(detr["night_mwh"], detr["ree_csp_mwh"]),
        "corr_night_vs_pv_detrended": _safe_corr(detr["night_mwh"], detr["ree_pv_mwh"]),
        "monthly_mean_mwh": [
            {
                "month": str(m),
                "night_replica_mwh": float(g["night_mwh"].mean()),
                "ree_csp_mwh": float(g["ree_csp_mwh"].mean()),
                "ree_pv_mwh": float(g["ree_pv_mwh"].mean()),
            }
            for m, g in joined.groupby(months)
        ],
    }


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Red Electrica solar termica (CSP) and test whether it accounts for "
            "the ES overnight solar_mw that ABL-396 measured. Read-only."
        )
    )
    parser.add_argument("--start", default="2025-01-01", help="first UTC day of the hourly test")
    parser.add_argument("--end", default="2025-12-13", help="last UTC day of the hourly test")
    parser.add_argument(
        "--daily-start", default="2026-01-14", help="first day of the daily energy-budget check"
    )
    parser.add_argument(
        "--daily-end", default="2026-08-10", help="last day of the daily energy-budget check"
    )
    parser.add_argument("--skip-daily", action="store_true", help="hourly test only")
    parser.add_argument(
        "--replica-db",
        default=os.environ.get("ENERGY_DB_PATH", r"C:\Code\able\data\energy_dashboard.db"),
    )
    parser.add_argument(
        "--tables",
        default="energy_renewable,energy_generation",
        help="comma-separated replica source tables to test against",
    )
    parser.add_argument("--cache-dir", default=None, help="HTTP cache directory (scratch)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="reports/abl_411_es_csp_probe.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    cache_dir = Path(
        args.cache_dir
        or os.environ.get("PAPERCLIP_RUN_SCRATCH_DIR")
        or os.environ.get("TEMP", ".")
    ) / "abl411_ree_cache"

    db_path = args.replica_db
    db_bytes = os.path.getsize(db_path)
    print(f"replica: {db_path} ({db_bytes} bytes, mode=ro)")
    print(f"ES point: {SOLAR_REPRESENTATIVE_POINTS['ES']}, night <= {NIGHT_ELEVATION_THRESHOLD_DEG} deg")

    print(f"fetching REE 5-minute peninsular generation {start} .. {end} ...")
    # One extra local day on each side so the UTC hours at the edges are complete.
    ree, missing = fetch_qh_range(start - timedelta(days=1), end + timedelta(days=1), cache_dir, args.workers)
    print(f"  REE hourly rows: {len(ree)}; local days with no archive: {len(missing)}")

    payload: Dict[str, object] = {
        "issue": "ABL-411",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replica": {"path": db_path, "bytes": db_bytes, "mode": "ro"},
        "ree_source": {
            "hourly": QH_URL.format(day="YYYY-MM-DD"),
            "daily": APIDATOS_URL.format(start="YYYY-MM-DD", end="YYYY-MM-DD"),
            "timezone_of_ts": MADRID,
            "perimeter": "peninsular Spain",
        },
        "night_definition": {
            "predicate": "src.solar_features.night_mask (the ABL-337 clamp's own predicate)",
            "threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
            "point": list(SOLAR_REPRESENTATIVE_POINTS["ES"]),
        },
        "hourly_window_requested": [args.start, args.end],
        "ree_days_without_archive": missing,
        "results": [],
    }

    for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
        print(f"reading replica {table} ...")
        replica = read_replica_solar(db_path, table, start, end)
        print(f"  {table}: {len(replica)} hourly rows")
        if replica.empty or ree.empty:
            payload["results"].append({"source_table": table, "error": "no data"})
            continue
        payload["results"].append(analyse(replica, ree, table))

    if not args.skip_daily:
        d_start = datetime.strptime(args.daily_start, "%Y-%m-%d").date()
        d_end = datetime.strptime(args.daily_end, "%Y-%m-%d").date()
        print(f"daily energy-budget check {d_start} .. {d_end} ...")
        daily = fetch_daily_structure(d_start, d_end, cache_dir)
        # Calibrate the daily product where the 5-minute archive still reaches.
        overlap_daily = fetch_daily_structure(start, end, cache_dir)
        payload["daily_source_calibration"] = daily_source_calibration(ree, overlap_daily)
        payload["daily_check"] = []
        for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
            replica = read_replica_solar(db_path, table, d_start, d_end)
            payload["daily_check"].append(
                daily_budget_check(replica, daily, f"{table} {d_start}..{d_end}")
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
