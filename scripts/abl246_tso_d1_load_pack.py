"""ABL-246 -- D+1 load-serving evidence pack: TSO vs our ML, per country.

Scores the TSO day-ahead load forecast as it was *actually issued* against our
production D+2 ML forecast and a seasonal-naive D-7 baseline, on the vintages
accrued in `forecast_vintage_archive` (ABL-184) since it went live.

Why this issue exists at all
----------------------------
Before the archive, `energy_load_forecast` retained exactly one TSO vintage per
target -- the last one, after every revision. Any TSO-vs-ML comparison built on
that table gives the TSO a forecast it did not have at D+1 gate closure. The
archive is the first record that can separate the two, so this script scores
three TSO arms and reports the spread between them as the size of that
optimism.

Arms (all scored on one identical (country, target-hour) intersection)
---------------------------------------------------------------------
  tso_d1_first  TSO day-ahead, *earliest* vintage seen before the target's
                local market day opened.
  tso_d1_last   TSO day-ahead, *latest* vintage seen before that market day
                opened -- the freshest information a genuine D+1 user holds.
                This is the honest D+1 arm.
  tso_final     TSO day-ahead, last vintage ever seen, revisions included.
                Not available at D+1; stands in for the pre-archive
                `energy_load_forecast` read, i.e. the optimistic number.
  ml_d2         Our production forecast in the scorecard's registered D+2
                horizon band (24-64h), latest leak-free vintage.
  d7_naive      Seasonal naive: the actual load 168h before the target.

Why the D+1 cut is a market day, not a UTC day and not a lead
-------------------------------------------------------------
Two rules were tried and both are wrong, in opposite directions:

*A UTC-day cut* (`first_seen < target's UTC midnight`) quietly punishes the TSO
by timezone. ENTSO-E publishes for the *local market day*: in August DE's runs
22:00 UTC to 22:00 UTC, so UTC hours 22:00-23:00 of any target day belong to
the next market day and are published a day later. Measured cost: 2 hours/day
lost for every CEST country, 1 for PT, 3 for the EET fleet -- invisible in a
per-country total.

*A flat `lead >= 24h` cut* is worse. A day-ahead forecast published at 11:00 on
D-1 leads target hour D 00:00 by thirteen hours, not twenty-four; "day-ahead"
names the delivery day, never a fixed lead. Requiring 24h therefore deletes the
early hours of every delivery day and keeps only the evening -- measured, DE
fell from 22 scored hours/day to 8, all of them late -- turning a per-country
WAPE into an hour-of-day-biased statistic with a different hour set per country.

The correct condition is the product's own: **issued before the target's local
market day began**. It keeps all 24 hours and is exact rather than approximate.

Leak-freeness
-------------
The archive stamps `first_seen_at` (when our poller saw the value), never the
TSO's own publication time. Every lead quoted here is therefore a **lower
bound** on how early the forecast really existed.

The two arms are not horizon-matched, and cannot be: TSO day-ahead leads its
delivery day by roughly 10-34h, our D+2 product by 24-64h. That asymmetry is
the product difference the issue is about, not a flaw in the comparison -- and
it runs *against* the TSO, which is the safe direction for a result that
favours it. The pack reports the median lead of each arm so it stays visible.

Reads are read-only on the replica and pass through the ABL-431/458 TSO
plausibility guard, which the archive `load` read is registered for.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SUPPORTED_COUNTRIES
from src.tso_plausibility import VINTAGE_ARCHIVE_TABLE, guard_tso_frame

#: The archive's go-live backfill is stamped 2026-08-11 and carries retained
#: post-revision values for target days back to 2018. It is not a record of
#: what was issued when, so every read here starts the day after.
GENUINE_VINTAGE_FLOOR = "2026-08-12"

DAY_AHEAD_MODEL = "tso-day_ahead"

#: The market timezone each country's day-ahead product is published against.
#: There is deliberately **no default**: an unregistered country raises rather
#: than silently borrowing UTC, which would reintroduce the boundary error this
#: table exists to remove (the same reason `NIGHT_GENERATION_POSSIBLE` has no
#: default). DST is handled by `zoneinfo`, not by a fixed offset -- the window
#: happens to be all-summer, but a fixed +2 would rot in October.
MARKET_TIMEZONE = {
    "AT": "Europe/Vienna",    "BE": "Europe/Brussels", "BG": "Europe/Sofia",
    "CH": "Europe/Zurich",    "CZ": "Europe/Prague",   "DE": "Europe/Berlin",
    "EE": "Europe/Tallinn",   "ES": "Europe/Madrid",   "FI": "Europe/Helsinki",
    "FR": "Europe/Paris",     "GR": "Europe/Athens",   "HR": "Europe/Zagreb",
    "HU": "Europe/Budapest",  "IT": "Europe/Rome",     "LT": "Europe/Vilnius",
    "LV": "Europe/Riga",      "NL": "Europe/Amsterdam", "NO": "Europe/Oslo",
    "PL": "Europe/Warsaw",    "PT": "Europe/Lisbon",   "RO": "Europe/Bucharest",
    "SE": "Europe/Stockholm", "SI": "Europe/Ljubljana", "SK": "Europe/Bratislava",
}


class UnregisteredMarketTimezoneError(KeyError):
    """A country reached the D+1 cut with no declared market timezone."""


def market_day_start_utc(target: pd.Series, country_code: str) -> pd.Series:
    """UTC instant at which each target's *local market day* began."""
    try:
        tz = MARKET_TIMEZONE[country_code]
    except KeyError:
        raise UnregisteredMarketTimezoneError(
            f"{country_code} has no entry in MARKET_TIMEZONE; register its "
            f"market timezone rather than defaulting to UTC"
        ) from None
    local = target.dt.tz_localize("UTC").dt.tz_convert(tz)
    midnight = local.dt.normalize()
    return midnight.dt.tz_convert("UTC").dt.tz_localize(None)

#: The scorecard's registered D+2 horizon band (`evaluate_scorecard.py`), used
#: here so the ML arm is the same slice the rest of the programme scores.
D2_HORIZON_BAND = (24, 64)

SEASONAL_NAIVE_LAG_HOURS = 168

#: Countries whose truth series and forecast series are on incompatible bases
#: over this window, established by the orphan-hour screen in `main` rather
#: than asserted here. Scored and reported, but held out of the recommendation.
NOT_EVALUABLE_REASONS = {
    "NL": (
        "energy_load is net of behind-the-meter solar (midday trough falls to "
        "0.17x the country median and deepens Jan->Aug with the solar year); "
        "the TSO day-ahead forecast, our ML forecast and the D-7 baseline all "
        "sit ~9-10 GW through that trough. Truth and forecasts are on "
        "different bases, so no arm is measurable against it."
    ),
}


def connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_archive(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every genuine (non-backfill) load vintage, both sources."""
    sql = """
        SELECT source, country_code, target_timestamp_utc, model_name,
               horizon_hours, forecast_value, first_seen_at
        FROM forecast_vintage_archive
        WHERE forecast_type = 'load'
          AND first_seen_at >= ?
    """
    df = pd.read_sql_query(sql, conn, params=(GENUINE_VINTAGE_FLOOR,))
    # The archive tracks 34 countries; our served fleet is the 24 in
    # config.SUPPORTED_COUNTRIES. Scoring the other ten would report on models
    # that do not exist.
    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)]
    # The two sources spell the same column differently -- `ml` writes an ISO
    # `T` separator, `tso` writes a space. Joining on the raw text silently
    # matches nothing; normalise before anything else touches it.
    df["target"] = pd.to_datetime(df["target_timestamp_utc"], format="mixed")
    df["first_seen"] = pd.to_datetime(
        df["first_seen_at"], format="mixed", utc=True
    ).dt.tz_localize(None)
    df["target_day"] = df["target"].dt.normalize()
    return df


def load_actuals(conn: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    """Hourly-mean actual load.

    `energy_load` is mixed-cadence -- roughly half the fleet is quarter-hourly
    and half hourly over this window -- so a mean over raw rows would be
    cadence-weighted. Aggregate to the hour first (the ABL-332 contract).
    """
    sql = """
        SELECT country_code, timestamp_utc, load_mw
        FROM energy_load
        WHERE timestamp_utc >= ? AND timestamp_utc < ?
    """
    df = pd.read_sql_query(sql, conn, params=(start, end))
    df["ts"] = pd.to_datetime(df["timestamp_utc"], format="mixed")
    # ABL-111/ABL-109: a 0.0 in this table encodes "missing", not zero demand.
    n_zero = int((df["load_mw"] == 0).sum())
    df = df[df["load_mw"] != 0]
    df = df.dropna(subset=["load_mw"])
    hourly = (
        df.set_index("ts")
        .groupby("country_code")["load_mw"]
        .resample("h")
        .mean()
        .reset_index()
        .rename(columns={"ts": "target", "load_mw": "actual"})
        .dropna(subset=["actual"])
    )
    hourly.attrs["zero_rows_dropped"] = n_zero
    return hourly


def guard_tso(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Apply the ABL-431/458 plausibility guard per country, then drop refusals."""
    out = []
    for country, grp in df.groupby("country_code", sort=True):
        guarded = guard_tso_frame(
            grp.rename(columns={"target": "timestamp_utc"}),
            conn,
            country_code=country,
            table=VINTAGE_ARCHIVE_TABLE,
            column="load",
            frame_column="forecast_value",
            timestamp_column="timestamp_utc",
            context=f"ABL-246 {country} load vintages",
        ).rename(columns={"timestamp_utc": "target"})
        out.append(guarded)
    guarded_all = pd.concat(out, ignore_index=True)
    refused = int(guarded_all["forecast_value"].isna().sum())
    guarded_all = guarded_all.dropna(subset=["forecast_value"])
    guarded_all.attrs["guard_refusals"] = refused
    return guarded_all


def build_arms(archive: pd.DataFrame, conn: sqlite3.Connection) -> tuple:
    """Collapse the vintage stack into one value per (country, target) per arm."""
    tso = archive[
        (archive["source"] == "tso") & (archive["model_name"] == DAY_AHEAD_MODEL)
    ].copy()
    tso = guard_tso(tso, conn)
    guard_refusals = tso.attrs.get("guard_refusals", 0)

    # A D+1 product is one that exists before its delivery day opens.
    tso = tso.copy()
    tso["market_day_start"] = pd.concat([
        market_day_start_utc(grp["target"], cc)
        for cc, grp in tso.groupby("country_code", sort=False)
    ]).sort_index()
    tso_pre = tso[tso["first_seen"] < tso["market_day_start"]]
    key = ["country_code", "target"]

    first = (
        tso_pre.sort_values("first_seen")
        .groupby(key, as_index=False)
        .first()[key + ["forecast_value", "first_seen"]]
        .rename(columns={"forecast_value": "tso_d1_first",
                         "first_seen": "tso_d1_first_seen"})
    )
    last = (
        tso_pre.sort_values("first_seen")
        .groupby(key, as_index=False)
        .last()[key + ["forecast_value", "first_seen"]]
        .rename(columns={"forecast_value": "tso_d1_last",
                         "first_seen": "tso_d1_last_seen"})
    )
    # Revisions included -- deliberately NOT leak-free, that is the point.
    final = (
        tso.sort_values("first_seen")
        .groupby(key, as_index=False)
        .last()[key + ["forecast_value"]]
        .rename(columns={"forecast_value": "tso_final"})
    )
    n_vintages = (
        tso_pre.groupby(key, as_index=False)["forecast_value"]
        .count()
        .rename(columns={"forecast_value": "n_pre_day_vintages"})
    )

    # Our D+2 product, taken on the scorecard's registered horizon band rather
    # than on a run-date proxy: the band is what the rest of the programme
    # scores, and a run-date rule silently follows whichever hour the daily job
    # happened to fire at (measured here: a 07:00 UTC run and a 20:00 UTC run,
    # spanning leads of 28-64h between them).
    ml = archive[archive["source"] == "ml"].copy()
    lo_h, hi_h = D2_HORIZON_BAND
    ml_d2 = ml[
        ml["horizon_hours"].between(lo_h, hi_h) & (ml["first_seen"] < ml["target"])
    ]
    ml_d2 = (
        ml_d2.sort_values("first_seen")
        .groupby(key, as_index=False)
        .last()[key + ["forecast_value", "model_name", "horizon_hours", "first_seen"]]
        .rename(columns={"forecast_value": "ml_d2",
                         "model_name": "ml_model",
                         "first_seen": "ml_d2_seen"})
    )

    merged = (
        first.merge(last, on=key)
        .merge(final, on=key)
        .merge(n_vintages, on=key)
        .merge(ml_d2, on=key)
    )
    return merged, guard_refusals


def wape(err: np.ndarray, actual: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(err).sum() / denom * 100) if denom else float("nan")


ARMS = ["tso_d1_first", "tso_d1_last", "tso_final", "ml_d2", "d7_naive"]


def score(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, grp in panel.groupby("country_code", sort=True):
        rec = {
            "country": country,
            "n": len(grp),
            "days": grp["target"].dt.normalize().nunique(),
            "ml_model": grp["ml_model"].mode().iat[0],
            "mean_load_mw": float(grp["actual"].mean()),
            "median_pre_day_vintages": float(grp["n_pre_day_vintages"].median()),
            "median_tso_d1_lead_h": float(
                (grp["target"] - grp["tso_d1_last_seen"])
                .dt.total_seconds().div(3600).median()
            ),
            "median_ml_d2_lead_h": float(
                (grp["target"] - grp["ml_d2_seen"])
                .dt.total_seconds().div(3600).median()
            ),
            "evaluable": country not in NOT_EVALUABLE_REASONS,
        }
        for arm in ARMS:
            err = (grp[arm] - grp["actual"]).to_numpy()
            rec[f"wape_{arm}"] = wape(err, grp["actual"].to_numpy())
            rec[f"mae_{arm}"] = float(np.abs(err).mean())
            rec[f"bias_{arm}"] = float(err.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def debias(panel: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per-country bias correction on `arm`, in two forms.

    LT and EE are the only countries where the TSO loses, and both look like
    pure level errors (-13.4% and +6.5% of mean load) rather than shape
    failures. Whether that is recoverable decides whether P5 should treat them
    as "TSO unusable" or "TSO usable behind a calibration", so it is worth a
    number rather than an assertion.

    `_insample` subtracts the mean bias over the very window it is then scored
    on -- an upper bound on the achievable gain, never a forecastable result.
    `_causal` subtracts the mean bias over *prior days only*, which is what a
    live correction could actually have known; its first day has no prior and
    drops out, so it carries its own n.
    """
    out = []
    for country, grp in panel.groupby("country_code", sort=True):
        grp = grp.sort_values("target").copy()
        err = grp[arm] - grp["actual"]
        insample = wape((err - err.mean()).to_numpy(), grp["actual"].to_numpy())

        day = grp["target"].dt.normalize()
        daily_err = err.groupby(day).mean()
        prior = daily_err.expanding().mean().shift(1)
        corrected = grp[arm] - day.map(prior)
        ok = corrected.notna()
        causal = (
            wape((corrected[ok] - grp.loc[ok, "actual"]).to_numpy(),
                 grp.loc[ok, "actual"].to_numpy()) if ok.any() else float("nan")
        )
        out.append({
            "country": country,
            f"wape_{arm}": wape(err.to_numpy(), grp["actual"].to_numpy()),
            f"wape_{arm}_debiased_insample": insample,
            f"wape_{arm}_debiased_causal": causal,
            "n_causal": int(ok.sum()),
        })
    return pd.DataFrame(out)


def paired_daily(panel: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    """Per-country paired t-interval on the daily WAPE difference (a - b).

    A point estimate on 14-odd days is not a result on its own; this says
    whether the gap survives day-to-day variation.
    """
    rows = []
    panel = panel.copy()
    panel["day"] = panel["target"].dt.normalize()
    for country, grp in panel.groupby("country_code", sort=True):
        diffs = []
        for _, day in grp.groupby("day"):
            a = wape((day[arm_a] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            b = wape((day[arm_b] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            diffs.append(a - b)
        d = np.array(diffs, dtype=float)
        k = len(d)
        mean = float(d.mean())
        if k > 1:
            se = float(d.std(ddof=1) / np.sqrt(k))
            # t(0.975) for small k; table beats a scipy dependency here.
            tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                     7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                     12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
                     16: 2.131, 17: 2.120}.get(k, 2.093)
            lo, hi = mean - tcrit * se, mean + tcrit * se
        else:
            lo = hi = float("nan")
        rows.append({
            "country": country, "k_days": k,
            "mean_daily_wape_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "readable": bool(k > 1 and (lo > 0 or hi < 0)),
            "days_arm_a_better": int((d < 0).sum()),
        })
    return pd.DataFrame(rows)


#: ABL-283's own decomposition hours, reused verbatim so its table and this one
#: are comparable rather than merely similar.
NIGHT_HOURS = [0, 1, 2, 3, 22, 23]
MIDDAY_HOURS = [9, 10, 11, 12, 13, 14]


def diurnal_bias(panel: pd.DataFrame) -> pd.DataFrame:
    """Night-vs-midday relative bias per arm (the ABL-277/283 signature).

    A basis divergence (behind-the-meter solar netted out of the realized
    series but not the forecast) shows up as a clean midday-only skew with a
    quiet night. A plain calibration error does not. The two need different
    fixes, so the swing is worth separating from the daily mean bias.
    """
    def relbias(g: pd.DataFrame, arm: str) -> float:
        denom = g["actual"].sum()
        return float((g[arm] - g["actual"]).sum() / denom * 100) if denom else float("nan")

    hour = panel["target"].dt.hour
    rows = []
    for country, g in panel.groupby("country_code", sort=True):
        h = g["target"].dt.hour
        night, midday = g[h.isin(NIGHT_HOURS)], g[h.isin(MIDDAY_HOURS)]
        rec = {"country": country}
        for arm in ("tso_d1_last", "ml_d2"):
            rec[f"{arm}_night_pct"] = relbias(night, arm)
            rec[f"{arm}_midday_pct"] = relbias(midday, arm)
            rec[f"{arm}_all_pct"] = relbias(g, arm)
            rec[f"{arm}_diurnal_swing_pp"] = (
                rec[f"{arm}_midday_pct"] - rec[f"{arm}_night_pct"])
        rows.append(rec)
    return pd.DataFrame(rows)


def recommend(table: pd.DataFrame, vs_ml: pd.DataFrame,
              vs_d7: pd.DataFrame) -> pd.DataFrame:
    """Per-country D+1 serving recommendation, derived from the paired reads.

    Deliberately keyed on the paired daily interval rather than on the WAPE
    point estimate: over 15 target days a 0.3pp gap is not a result, and this
    pack is upstream of a Board decision that should not rest on one.

    A recommendation is about the D+1 slot only. Nothing here bears on D+2,
    where the TSO has no product to offer.
    """
    m = vs_ml.set_index("country")
    d = vs_d7.set_index("country")
    rows = []
    for _, r in table.iterrows():
        cc = r["country"]
        if not r["evaluable"]:
            verdict, why = "NOT EVALUABLE", NOT_EVALUABLE_REASONS[cc]
        else:
            beats_ml = m.at[cc, "readable"] and m.at[cc, "mean_daily_wape_diff"] < 0
            loses_ml = m.at[cc, "readable"] and m.at[cc, "mean_daily_wape_diff"] > 0
            beats_d7 = d.at[cc, "readable"] and d.at[cc, "mean_daily_wape_diff"] < 0
            loses_d7 = d.at[cc, "readable"] and d.at[cc, "mean_daily_wape_diff"] > 0
            if loses_ml:
                verdict = "DO NOT SERVE TSO"
                why = "TSO D+1 readably worse than our ML D+2"
            elif beats_ml and not loses_d7:
                verdict = "SERVE TSO AT D+1"
                why = ("TSO D+1 readably better than ML D+2"
                       + ("; also readably better than D-7" if beats_d7
                          else "; not readable against D-7"))
            elif beats_ml and loses_d7:
                verdict = "SERVE TSO AT D+1 (D-7 CAVEAT)"
                why = ("TSO D+1 readably better than ML D+2 but readably worse "
                       "than a D-7 seasonal naive -- fix the baseline first")
            else:
                verdict = "HOLD"
                why = "TSO-vs-ML difference not readable over this window"
        rows.append({"country": cc, "recommendation": verdict, "rationale": why,
                     "wape_tso_d1_last": r["wape_tso_d1_last"],
                     "wape_ml_d2": r["wape_ml_d2"],
                     "wape_d7_naive": r["wape_d7_naive"]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replica-db", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    conn = connect_ro(args.replica_db)
    archive = load_archive(conn)
    arms, guard_refusals = build_arms(archive, conn)

    # Reach back a full seasonal-naive lag before the first target, or the D-7
    # join silently truncates the scored window by a week.
    lo = (
        arms["target"].min() - pd.Timedelta(hours=SEASONAL_NAIVE_LAG_HOURS)
    ).strftime("%Y-%m-%d")
    hi = (arms["target"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    actuals = load_actuals(conn, lo, hi)
    zero_rows = actuals.attrs.get("zero_rows_dropped", 0)

    panel = arms.merge(actuals, on=["country_code", "target"], how="inner")

    naive = actuals.rename(columns={"actual": "d7_naive"}).copy()
    naive["target"] = naive["target"] + pd.Timedelta(hours=SEASONAL_NAIVE_LAG_HOURS)
    panel = panel.merge(naive, on=["country_code", "target"], how="inner")

    panel = panel.dropna(subset=ARMS + ["actual"])

    # Evaluability screen. When the TSO forecast, our ML forecast and the D-7
    # baseline -- three independent predictors -- all sit far above the actual,
    # the suspect is the truth series, not three models at once. Measured, so
    # that NOT_EVALUABLE_REASONS is a disposition of a number rather than an
    # assertion about a country.
    consensus = panel[["tso_d1_last", "ml_d2", "d7_naive"]].min(axis=1)
    panel["orphan_hour"] = panel["actual"] < 0.5 * consensus
    orphan = (
        panel.groupby("country_code")["orphan_hour"]
        .agg(orphan_hours="sum", n="size")
        .reset_index()
    )
    orphan["orphan_pct"] = 100 * orphan["orphan_hours"] / orphan["n"]

    table = score(panel)
    tso_vs_ml = paired_daily(panel, "tso_d1_last", "ml_d2")
    tso_vs_d7 = paired_daily(panel, "tso_d1_last", "d7_naive")
    ml_vs_d7 = paired_daily(panel, "ml_d2", "d7_naive")

    meta = {
        "issue": "ABL-246",
        "replica_db": args.replica_db,
        "generated_from_archive_floor": GENUINE_VINTAGE_FLOOR,
        "target_window_utc": [
            panel["target"].min().isoformat(),
            panel["target"].max().isoformat(),
        ],
        "target_days": int(panel["target"].dt.normalize().nunique()),
        "countries": int(panel["country_code"].nunique()),
        "pairs_scored": int(len(panel)),
        "tso_plausibility_refusals": int(guard_refusals),
        "actual_zero_rows_dropped_abl111": int(zero_rows),
        "archive_max_first_seen": archive["first_seen"].max().isoformat(),
        "d1_rule": "first_seen < local market day start (MARKET_TIMEZONE)",
        "d2_horizon_band": list(D2_HORIZON_BAND),
        "not_evaluable": NOT_EVALUABLE_REASONS,
    }

    tso_debias = debias(panel, "tso_d1_last")

    out = {
        "meta": meta,
        "evaluability_screen": json.loads(orphan.to_json(orient="records")),
        "per_country": json.loads(table.to_json(orient="records")),
        "tso_d1_bias_correction": json.loads(tso_debias.to_json(orient="records")),
        "diurnal_bias": json.loads(diurnal_bias(panel).to_json(orient="records")),
        "recommendation": json.loads(
            recommend(table, tso_vs_ml, tso_vs_d7).to_json(orient="records")),
        "paired_tso_d1_last_vs_ml_d2": json.loads(tso_vs_ml.to_json(orient="records")),
        "paired_tso_d1_last_vs_d7": json.loads(tso_vs_d7.to_json(orient="records")),
        "paired_ml_d2_vs_d7": json.loads(ml_vs_d7.to_json(orient="records")),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    table.to_csv(args.csv_out, index=False)

    print(json.dumps(meta, indent=2))
    print("\n-- evaluability screen (orphan hours: truth below every predictor) --")
    print(orphan.sort_values("orphan_pct", ascending=False).head(5)
          .round(2).to_string(index=False))
    cols = ["country", "n", "days", "ml_model", "evaluable",
            "median_tso_d1_lead_h", "median_ml_d2_lead_h",
            "wape_tso_d1_last", "wape_tso_final", "wape_ml_d2", "wape_d7_naive"]
    print("\n-- per country (WAPE %) --")
    print(table[cols].round(2).to_string(index=False))
    ev = table[table["evaluable"]]
    print(f"\nevaluable countries: {len(ev)} of {len(table)}")
    print("TSO D+1 beats ML D+2 in %d of %d" % (
        int((ev["wape_tso_d1_last"] < ev["wape_ml_d2"]).sum()), len(ev)))
    print("TSO D+1 beats D-7      in %d of %d" % (
        int((ev["wape_tso_d1_last"] < ev["wape_d7_naive"]).sum()), len(ev)))
    print("ML D+2  beats D-7      in %d of %d" % (
        int((ev["wape_ml_d2"] < ev["wape_d7_naive"]).sum()), len(ev)))
    print("\n-- TSO D+1 bias correction (LT/EE question) --")
    print(tso_debias[tso_debias["country"].isin(ev["country"])]
          .round(2).to_string(index=False))
    rec = recommend(table, tso_vs_ml, tso_vs_d7)
    print("\n-- recommendation --")
    print(rec[["country", "recommendation"]].to_string(index=False))
    print("\n" + rec["recommendation"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
