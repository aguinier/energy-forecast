"""ABL-607 -- why our D+2 load model loses to a D-7 seasonal naive.

ABL-246 section 4.1 measured the loss: our production D+2 load forecast readably
loses to a D-7 seasonal naive in 10 of 23 evaluable countries and readably beats
it in 1. This script diagnoses the cause. It fits nothing, promotes nothing and
writes nothing outside its own report files; the replica is opened read-only.

That "10" is the count as ABL-246 published it, and this script's own re-read
does not reproduce it -- see section 3.1 of `reports/abl_607_d2_load_diagnosis.md`
and `section_a_reproduction` in `reports/abl_607_d2_load_diagnosis_reread.json`.
**9 of 23 is the current count**; DE is the difference, and it moves out of
readability rather than out of loss. The mechanism below is unaffected: it is
about which row the model anchors on, not about how many intervals clear zero.

The candidate mechanism, and how each section tests it
-----------------------------------------------------
Load reaches inference through `Forecaster.predict_d2`'s **proxy-row** branch
(`src/forecaster.py`), not through the serve-faithful builder -- that builder
covers wind and solar only (`SERVE_FAITHFUL_FORECAST_TYPES`). The proxy-row
branch builds one feature frame out of *history*, then for each target hour h
takes `same_hour_data.iloc[-1:]`, the most recent historical row with that
hour-of-day, and overrides only the calendar and weather columns to point at the
target. Every history feature -- the three lags and the eight rolling
statistics, 11 of the artifact's 26 columns -- keeps the value it had for the
**last observed day**, not for the target day.

So the model is served a history block anchored g days early, where g is the
distance from the last ingested actual to the target. Its `target_value_lag_7d`
is not "same hour, seven days before the target"; it is "same hour, seven days
before the anchor day" -- a different day of the week. That is the feature a
D-7 seasonal naive uses correctly, which is why "our horizon is longer" does not
excuse the loss: the horizon does not remove the D-7 value, the serving path
does.

  section A  reproduce ABL-246 section 4.1 as a control
  section B  split the registered 24-64h band by run-day offset g, so the cost
             of one extra day of anchor staleness is measured rather than
             assumed (both arms are in the archive: every run emits 48 hours)
  section C  each ML arm against D-7 and against the plain lag ladder D-1..D-14
  section D  anchor identification -- which day's actual does our forecast
             actually track? argmin over k of WAPE(ML, actual(target - k days))
  section E  the weekday signature the mechanism predicts: an anchor g days
             early misreads weekday level, so the loss must concentrate on the
             weekday/weekend transitions and vanish midweek
  section F  cross-country -- the loss should scale with weekly amplitude, the
             thing a misaligned weekday anchor gets wrong
  section G  level vs shape, and a leak-free trailing debias: is this a
             calibration problem (fixable cheaply) or a structural one
  section H  hour-of-day error profile per arm
  section I  algorithm split (AT/BE/FR serve xgboost, the other 21 catboost)
             and the artifact feature-importance audit: how much of each
             model's decision weight sits on the mis-anchored block
  section J  direct reconstruction of the served feature vector -- what the
             proxy row actually carried, against what a target-aligned block
             would have carried
  section K  day completeness (ABL-639) -- how many hours each country-day was
             actually scored on, against how many the window allows it, plus
             the complete-days-only arm of every paired interval above

Protocol, inherited from ABL-246 so the two packs are comparable
---------------------------------------------------------------
Vintages are first-seen rows from `forecast_vintage_archive` (ABL-184) at
`first_seen_at >= 2026-08-12`; the 2026-08-11 bucket is the go-live backfill and
is excluded. The archive read is **deliberately unfiltered** and registered in
the ABL-462 sweep's `ML_SLICE_ONLY_EXEMPT` (ABL-611, recategorised on ABL-617
so the slice is machine-checked rather than claimed): it selects `source = 'ml'`
only, so it reads no TSO row, and the ABL-431 guard is one-sided -- filtering
here could only delete our own largest over-forecasts, which are the errors this
pack exists to measure. The guard's reference is still computed over exactly
these rows and written to section 0 (`plausibility_census`) by every run of this
script, so the count it *would* have refused is measured rather than assumed:
**0 of 67,008 rows, 24/24 countries evaluable** (ABL-619, published in
`reports/abl_607_d2_load_diagnosis_reread.json`; the first run predates the
census). `tests/test_abl607_guarded_read.py` holds this paragraph against that
artifact in both directions, so it cannot drift from it.
Truth is `energy_load` aggregated to hourly means (ABL-332), with
0.0 rows dropped -- ABL-111/ABL-109 encode missing as zero. Every arm is scored
on one identical (country, target-hour) intersection. NL is scored but held out
of every conclusion: its realized series is net of behind-the-meter solar while
our forecast is gross (ABL-277 / ABL-505 / ABL-506), so scoring it measures a
basis mismatch rather than skill.

Every paired interval weights a target day equally, so it also depends on each
day having been scored on a comparable number of hours. Section K measures that
and `--min-day-completeness` screens on it; the default is 0.0, which screens
nothing, so the pack's published protocol is unchanged unless asked (ABL-639).

Out-of-sample throughout. The one in-sample number, in section G, is labelled.

Usage:
  .venv\\Scripts\\python.exe scripts/abl607_d2_load_diagnosis.py
      --replica-db C:\\Code\\able\\data\\energy_dashboard.db
      --json-out reports/abl_607_d2_load_diagnosis.json
      --models-dir models
      [--min-day-completeness 0.0]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import SUPPORTED_COUNTRIES
from src.tso_plausibility import (
    PLAUSIBILITY_TOLERANCE,
    VINTAGE_ARCHIVE_TABLE,
    implausible_mask,
    reference_scale,
)

#: The archive's go-live backfill is stamped 2026-08-11 and carries retained
#: post-revision values for target days back to 2018. ABL-246's floor, reused.
GENUINE_VINTAGE_FLOOR = "2026-08-12"

#: ABL-246's scored window ended at target hour 2026-08-28 00:00 inclusive, so
#: the exclusive bound is one hour past it. Defaulted so section A reproduces
#: that pack exactly rather than approximately; raise it for a later re-read.
DEFAULT_MAX_TARGET = "2026-08-28 01:00:00"

#: The scorecard's registered D+2 horizon band (`evaluate_scorecard.py`).
D2_HORIZON_BAND = (24, 64)

#: NL only. Reason and owners in the module docstring; kept as data so the
#: report states the holdout rather than silently applying it.
NOT_EVALUABLE = {"NL"}

#: ABL-246 section 4.1's ten losers and its one winner, quoted here only so
#: section A can report whether the reproduction agrees rather than leaving a
#: reader to diff two tables by eye.
ABL246_LOSERS = ["SI", "AT", "CZ", "PL", "SK", "ES", "DE", "LV", "PT", "SE"]
ABL246_WINNERS = ["GR"]

#: The history block: every artifact column whose value is a function of the
#: target's own recent past. These are the columns the proxy row mis-anchors.
HISTORY_FEATURE_PREFIXES = ("target_value_lag_", "target_value_roll_")

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)

#: t(0.975) for small k; a table beats a scipy dependency for this one use.
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
          8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
          14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
          20: 2.093}

#: The day-completeness threshold reported beside the primary as a sensitivity
#: arm (ABL-639): complete days only. Fixed rather than exposed, because the
#: point of the arm is that a reader sees the *same* alternative every run.
SENSITIVITY_MIN_DAY_COMPLETENESS = 1.0


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def plausibility_census(df: pd.DataFrame,
                        conn: sqlite3.Connection) -> pd.DataFrame:
    """Measure what the ABL-431 guard *would* refuse here. Delete nothing.

    **This read is deliberately raw, and the exemption is structural.** The
    ABL-462 sweep is a substring match on three table names; here it has found
    a table *name*, not a TSO read. `load_archive` selects
    `forecast_type = 'load' AND source = 'ml'` -- our own forecasts, and no TSO
    row anywhere in this file. The guard's archive reference is the exact
    complement: `forecast_read` bounds it on
    `source = 'tso' AND model_name = 'tso-day_ahead'`, because (the module's
    own docstring) "`source = 'ml'` rows are our own forecasts ... not a
    published TSO series". Registered on ABL-611, and on `ML_SLICE_ONLY_EXEMPT`
    since ABL-617 -- the category whose membership condition is that claim,
    checked over this file's SQL rather than reviewed once.

    **Why filtering here would be a scoring defect, not hygiene.** The guard is
    one-sided: it refuses values *above* `3 x` the fleet reference and nothing
    below. Applied to the arm under test, the only rows it can remove are our
    own largest over-forecasts -- which are exactly the errors this pack
    measures. Every arm scores on one shared (country, target-hour)
    intersection, so dropping such a row also drops the D-7 comparator's cell
    for that hour, in the country-hour where our model was worst. The effect is
    one-directional and in our own favour: it can only shrink the D+2-vs-D-7
    gap, never widen it. A guard that deletes the arm-under-test's worst errors
    is not a guard on this read; it is a thumb on the scale.

    That is the opposite of the ABL-431 case, where the implausible value was
    an *input* about to be fitted on. Here it would be an *output* being
    graded. Same table, same rows, opposite role.

    **So the guard runs as a measurement instead.** `implausible_mask` is the
    guard's own predicate with none of its filtering, so the pack can still
    answer "how many rows would it refuse over this window" without any row
    leaving the panel. A non-zero count here is a **finding about our own
    model** -- a published load forecast above three times a country's fleet
    peak is a defect worth its own issue -- and it must be reported and scored,
    not silently deleted. The per-country reference, threshold and the ratio
    our arm cleared by are all recorded: a bare "0 refused" is not evidence
    without its headroom, and `evaluable` is carried because a cell that was
    never evaluated is not a cell that passed.

    No `as_of` is passed, matching ABL-246: the reference is the whole history
    of a slowly-varying fleet property, not a target-correlated signal.
    """
    census = []
    n_flagged_total = 0
    for cc, grp in df.groupby("country_code", sort=True):
        ref = reference_scale(conn, cc, VINTAGE_ARCHIVE_TABLE, "load")
        observed = pd.to_numeric(grp["forecast_value"], errors="coerce")
        mx = float(observed.max()) if observed.notna().any() else None
        mask = implausible_mask(grp["forecast_value"], ref)
        n_flagged = int(mask.sum())
        n_flagged_total += n_flagged
        census.append({
            "country": cc,
            "n_rows": int(len(grp)),
            # `forecast_value` is declared NOT NULL, so this should be 0.
            # Counted anyway: without it a pre-existing null could be read as a
            # flag, and the flag count is the number this section exists for.
            "n_null_in_read": int(observed.isna().sum()),
            "evaluable": bool(ref.evaluable),
            "reference_mw": ref.reference_mw,
            "threshold_mw": ref.threshold_mw,
            "n_tso_day_ahead_rows": int(ref.n_forecast),
            "n_actual_rows": int(ref.n_actual),
            "ml_max_mw": mx,
            "ml_min_mw": float(observed.min()) if observed.notna().any() else None,
            "max_over_threshold": (mx / ref.threshold_mw
                                   if mx is not None and ref.evaluable else None),
            "n_would_be_refused": n_flagged,
            "reason": ref.reason,
        })

    out = df.copy()
    out.attrs["guard_would_refuse"] = n_flagged_total
    out.attrs["guard_rows_read"] = int(len(df))
    out.attrs["guard_rows_dropped"] = 0
    out.attrs["guard_census"] = census
    return out


def load_archive(conn: sqlite3.Connection, max_target: str) -> pd.DataFrame:
    """Our own ML load vintages, genuine (non-backfill) rows only.

    `generated_at` is recovered as `target - horizon_hours` rather than read:
    the archive stamps `first_seen_at`, when our poller saw the row, which lags
    generation by a variable few minutes. The horizon is stamped by the runner
    at generation, so the subtraction is exact and the poller lag drops out.

    The values are measured against the ABL-431 plausibility reference before
    any arm is built (`plausibility_census`), which reports and removes
    nothing -- see that function for why filtering this particular read would
    bias the comparison in our own model's favour.
    """
    sql = """
        SELECT country_code, target_timestamp_utc, model_name,
               horizon_hours, forecast_value, first_seen_at
        FROM forecast_vintage_archive
        WHERE forecast_type = 'load' AND source = 'ml'
          AND first_seen_at >= ?
    """
    df = pd.read_sql_query(sql, conn, params=(GENUINE_VINTAGE_FLOOR,))
    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)]
    # The two sources spell the timestamp differently -- `ml` writes an ISO `T`
    # separator, `tso` a space. Normalise before anything joins on it.
    df["target"] = pd.to_datetime(df["target_timestamp_utc"], format="mixed")
    df["first_seen"] = pd.to_datetime(
        df["first_seen_at"], format="mixed", utc=True).dt.tz_localize(None)
    df = df[df["target"] < pd.Timestamp(max_target)]
    # Measured on exactly the rows this diagnosis scores -- after the window and
    # fleet filters, before any arm is built. Reports; removes nothing.
    df = plausibility_census(df, conn)
    df["generated_at"] = df["target"] - pd.to_timedelta(df["horizon_hours"], unit="h")
    df["run_day"] = df["generated_at"].dt.normalize()
    df["target_day"] = df["target"].dt.normalize()
    # Run-day offset: how many calendar days before the target the run fired.
    # This is the quantity the proxy-row anchor is a function of, and it is what
    # the registered lead band only approximates.
    df["g"] = ((df["target_day"] - df["run_day"]) / DAY).astype(int)
    return df


def load_actuals(conn: sqlite3.Connection, start: str, end: str,
                 quality_filter: bool = False) -> pd.DataFrame:
    """Hourly-mean actual load.

    `energy_load` is mixed-cadence over this window -- roughly half the fleet is
    quarter-hourly -- so a mean over raw rows would be cadence-weighted.
    Aggregate to the hour first (the ABL-332 contract).

    `quality_filter` mirrors `src/db.py:load_energy_data`, which restricts the
    *training and serving* frame to `data_quality = 'actual'`. The truth series
    ABL-246 scored against does not apply that filter, so section J takes the
    filtered frame (what serving saw) and every scoring section takes the
    unfiltered one (what ABL-246 scored), and the two are never mixed.
    """
    q = "AND data_quality = 'actual'" if quality_filter else ""
    sql = f"""
        SELECT country_code, timestamp_utc, load_mw
        FROM energy_load
        WHERE timestamp_utc >= ? AND timestamp_utc < ? {q}
    """
    df = pd.read_sql_query(sql, conn, params=(start, end))
    # `energy_load` tracks 34 countries; our served fleet is the 24 in
    # config.SUPPORTED_COUNTRIES. The extra ten have no model to diagnose.
    df = df[df["country_code"].isin(SUPPORTED_COUNTRIES)]
    df["ts"] = pd.to_datetime(df["timestamp_utc"], format="mixed")
    # ABL-111/ABL-109: a 0.0 in this table encodes "missing", not zero demand.
    n_zero = int((df["load_mw"] == 0).sum())
    df = df[df["load_mw"] != 0].dropna(subset=["load_mw"])
    hourly = (
        df.set_index("ts").groupby("country_code")["load_mw"]
        .resample("h").mean().reset_index()
        .rename(columns={"ts": "target", "load_mw": "actual"})
        .dropna(subset=["actual"])
    )
    hourly.attrs["zero_rows_dropped"] = n_zero
    return hourly


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def build_ml_arms(archive: pd.DataFrame) -> pd.DataFrame:
    """One value per (country, target) per ML arm.

    Three arms, deliberately:

    `ml_band` is ABL-246's arm verbatim -- latest leak-free vintage inside the
    registered 24-64h band. It is the control, and it is *not* a clean D+2
    product: every run emits 48 target hours, so for a target's late hours the
    band also admits rows from the T-1 run (target T 23:00 generated T-1 20:00
    leads 27h). Whether that matters is measured in section B rather than
    argued.

    `ml_g1` / `ml_g2` split the same rows by **run-day offset** -- the run that
    fired on the target's own eve, and the one that fired the day before that.
    That is the axis the proxy-row anchor moves along, so it is the axis a
    staleness claim has to be tested on.

    They are joined **left** onto the band arm, so requiring both offsets does
    not silently narrow the window section A reproduces ABL-246 on. The two
    panels are separated in `main` instead, and each section says which it used.
    """
    key = ["country_code", "target"]
    leak_free = archive[archive["first_seen"] < archive["target"]]

    lo, hi = D2_HORIZON_BAND
    band = leak_free[leak_free["horizon_hours"].between(lo, hi)]
    ml_band = (
        band.sort_values("first_seen").groupby(key, as_index=False).last()
        [key + ["forecast_value", "model_name", "horizon_hours", "first_seen", "g"]]
        .rename(columns={"forecast_value": "ml_band", "model_name": "ml_model",
                         "first_seen": "ml_band_seen", "horizon_hours": "ml_band_lead",
                         "g": "ml_band_g"})
    )

    out = ml_band
    for g in (1, 2):
        arm = (
            leak_free[leak_free["g"] == g]
            .sort_values("first_seen").groupby(key, as_index=False).last()
            [key + ["forecast_value", "horizon_hours"]]
            .rename(columns={"forecast_value": f"ml_g{g}",
                             "horizon_hours": f"ml_g{g}_lead"})
        )
        out = out.merge(arm, on=key, how="left")
    return out


def attach_lag_arms(panel: pd.DataFrame, actuals: pd.DataFrame,
                    lags_days: List[int]) -> pd.DataFrame:
    """Join actual(target - k days) for each k as its own arm."""
    for k in lags_days:
        lagged = actuals.rename(columns={"actual": f"d{k}_naive"}).copy()
        lagged["target"] = lagged["target"] + pd.Timedelta(days=k)
        panel = panel.merge(lagged, on=["country_code", "target"], how="inner")
    return panel


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def wape(err: np.ndarray, actual: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(err).sum() / denom * 100) if denom else float("nan")


def hours_expected_per_day(panel: pd.DataFrame) -> pd.Series:
    """Target hours each calendar day *could* contribute, given the window.

    Indexed by normalised day, over the panel's own inclusive target span. This
    is the denominator of the ABL-639 completeness ratio, and the reason that
    ratio is not `hours / 24`.

    The window's two end days are legitimately partial and must not be read as
    defective. ABL-607's panel runs `2026-08-13 08:00` -> `2026-08-28 00:00`,
    so its first day expects 16 hours and its last expects **1** -- and that
    terminal hour is the single largest difference the pack recorded between
    its two reads. A constant-24 screen would drop both ends for every country
    at once, silently, which is a worse defect than the one being screened for.

    Computed fleet-wide, from one span shared by every country, on purpose. A
    per-country span would be defined by the very truncation this measures, so
    every country would score 100% complete by construction -- the vacuous
    form of the check.
    """
    span = pd.date_range(panel["target"].min(), panel["target"].max(), freq="h")
    return pd.Series(1, index=span).groupby(span.normalize()).sum()


def day_completeness(panel: pd.DataFrame) -> pd.DataFrame:
    """Scored hours per (country, target day) against what the window allows.

    `hours_present` counts rows that survived into `panel`, so it is one number
    short of nothing: a missing vintage, a missing D-7 comparator and a missing
    hour of truth all reduce it identically. That is wider than the
    truth-side cause ABL-639 was filed on (`load_actuals` resamples with no
    `min_count`, so an hour holding one quarter-hourly slot still yields a
    mean), and wider is what the paired interval actually needs: the weight a
    day carries in `paired_daily` is set by the hours it was *scored* on, not
    by the hours that existed somewhere upstream.

    Emitted unconditionally, whether or not a screen is applied.
    """
    expected = hours_expected_per_day(panel)
    day = panel["target"].dt.normalize()
    present = panel.groupby([panel["country_code"], day]).size()
    rows = []
    for (country, d), n in present.items():
        exp = int(expected.loc[d])
        rows.append({"country": country, "day": str(d.date()),
                     "hours_expected": exp, "hours_present": int(n),
                     "completeness": float(n) / exp if exp else float("nan"),
                     "is_short": bool(n < exp)})
    return pd.DataFrame(rows).sort_values(["country", "day"], ignore_index=True)


def paired_daily(panel: pd.DataFrame, arm_a: str, arm_b: str,
                 min_day_completeness: float = 0.0) -> pd.DataFrame:
    """Per-country paired t-interval on the daily WAPE difference (a - b).

    ABL-246's function. A point estimate on 15-odd days is not a result on its
    own; this says whether the gap survives day-to-day variation. Positive mean
    favours arm b.

    `min_day_completeness` (ABL-639) drops a country-day whose scored hours
    fall below that fraction of the hours its day could have carried. Each day
    contributes exactly one `a - b` regardless of how many hours it was
    computed on, so without a screen a country-day holding 2 surviving hours is
    weighted like a 24-hour one, and `k = len(d)` -- which keys `T_CRIT` --
    counts it as a full observation. The truncation is per-country, so the
    countries are not even scored over the same window.

    **The default 0.0 is a provable no-op**: a group yielded by `groupby` is
    never empty, so `hours_present >= 0.0 * hours_expected` holds for every
    day. The screen is off unless it is asked for, and the arm order and
    float summation order are unchanged, so the interval is bit-identical to
    the unscreened one. `tests/test_abl639_day_completeness.py` pins that
    against a verbatim copy of the pre-ABL-639 function.

    Screening trades bias for variance and the trade is reported rather than
    assumed: dropping days lowers `k`, `T_CRIT` is keyed on `k`, so a screened
    interval is *wider*. `k_days_screened_out` carries the cost per country.
    """
    rows = []
    panel = panel.copy()
    panel["day"] = panel["target"].dt.normalize()
    expected = hours_expected_per_day(panel)
    for country, grp in panel.groupby("country_code", sort=True):
        diffs = []
        n_short, n_dropped = 0, 0
        for day_ts, day in grp.groupby("day"):
            hours_expected = int(expected.loc[day_ts])
            hours_present = len(day)
            n_short += hours_present < hours_expected
            if hours_present < min_day_completeness * hours_expected:
                n_dropped += 1
                continue
            a = wape((day[arm_a] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            b = wape((day[arm_b] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            diffs.append(a - b)
        d = np.array(diffs, dtype=float)
        k = len(d)
        mean = float(d.mean()) if k else float("nan")
        if k > 1:
            se = float(d.std(ddof=1) / np.sqrt(k))
            tcrit = T_CRIT.get(k, 2.086)
            lo, hi = mean - tcrit * se, mean + tcrit * se
        else:
            lo = hi = float("nan")
        rows.append({"country": country, "k_days": k, "mean_daily_wape_diff": mean,
                     "ci_lo": lo, "ci_hi": hi,
                     "readable": bool(k > 1 and (lo > 0 or hi < 0)),
                     "days_arm_a_better": int((d < 0).sum()),
                     "k_days_short": int(n_short),
                     "k_days_screened_out": int(n_dropped)})
    return pd.DataFrame(rows)


def readable_cells(paired: pd.DataFrame) -> Dict[str, List[str]]:
    """The readable losers and winners of a paired frame, NL held out."""
    ev = paired[~paired["country"].isin(NOT_EVALUABLE) & paired["readable"]]
    return {
        "readable_losers": sorted(ev[ev["mean_daily_wape_diff"] > 0]["country"]),
        "readable_winners": sorted(ev[ev["mean_daily_wape_diff"] < 0]["country"]),
    }


def score(panel: pd.DataFrame, arms: List[str]) -> pd.DataFrame:
    rows = []
    for country, grp in panel.groupby("country_code", sort=True):
        rec = {"country": country, "n": len(grp),
               "days": int(grp["target"].dt.normalize().nunique()),
               "ml_model": grp["ml_model"].mode().iat[0],
               "mean_load_mw": float(grp["actual"].mean()),
               "evaluable": country not in NOT_EVALUABLE}
        for arm in arms:
            err = (grp[arm] - grp["actual"]).to_numpy()
            rec[f"wape_{arm}"] = wape(err, grp["actual"].to_numpy())
            rec[f"relbias_{arm}"] = float(
                err.sum() / grp["actual"].sum() * 100)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# section D -- anchor identification
# --------------------------------------------------------------------------

def anchor_identification(panel: pd.DataFrame, actuals: pd.DataFrame,
                          arm: str, max_k: int = 8) -> pd.DataFrame:
    """Which day's actual does our forecast actually track?

    Scores the forecast for target T against `actual(T - k days)` for k = 0..7
    and reports the k that minimises WAPE. k = 0 is the honest reading: the
    forecast is trying to predict its own target. Any other argmin says the
    served value is a better description of a different day, which is the
    signature a mis-anchored history block leaves.

    Reported alongside is the daily-level correlation at each k, because WAPE
    alone conflates level and shape.
    """
    rows = []
    for country, grp in panel.groupby("country_code", sort=True):
        act = actuals[actuals["country_code"] == country].set_index("target")["actual"]
        rec = {"country": country}
        best_k, best_w = None, np.inf
        for k in range(max_k):
            shifted = grp["target"].map(lambda t, k=k: act.get(t - k * DAY, np.nan))
            ok = shifted.notna()
            if ok.sum() < 24:
                rec[f"wape_vs_actual_lag{k}d"] = float("nan")
                continue
            w = wape((grp.loc[ok, arm] - shifted[ok]).to_numpy(),
                     shifted[ok].to_numpy())
            rec[f"wape_vs_actual_lag{k}d"] = w
            if w < best_w:
                best_k, best_w = k, w
        rec["argmin_k"] = best_k
        rec["wape_at_argmin"] = best_w
        rec["wape_at_k0"] = rec.get("wape_vs_actual_lag0d")
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# section E/F -- the weekday signature
# --------------------------------------------------------------------------

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekday_profile(panel: pd.DataFrame, arms: List[str]) -> pd.DataFrame:
    """Relative bias and WAPE by target day-of-week, per arm, pooled per country.

    The mechanism's prediction is specific and falsifiable: a history block
    anchored g days early carries the level of a *different weekday*. So the ML
    arm must over-predict when the target is a low-load day whose anchor was a
    high-load one (Sat/Sun) and under-predict on the reverse (Mon), while a
    weekday-aligned D-7 lag shows no such pattern. A flat profile refutes it.
    """
    rows = []
    p = panel.copy()
    p["dow"] = p["target"].dt.dayofweek
    for (country, dow), grp in p.groupby(["country_code", "dow"], sort=True):
        rec = {"country": country, "dow": int(dow), "dow_name": WEEKDAY_NAMES[dow],
               "n": len(grp), "days": int(grp["target"].dt.normalize().nunique())}
        for arm in arms:
            err = (grp[arm] - grp["actual"]).to_numpy()
            rec[f"relbias_{arm}"] = float(err.sum() / grp["actual"].sum() * 100)
            rec[f"wape_{arm}"] = wape(err, grp["actual"].to_numpy())
        rows.append(rec)
    return pd.DataFrame(rows)


def weekly_amplitude(actuals: pd.DataFrame, lo: pd.Timestamp,
                     hi: pd.Timestamp) -> pd.DataFrame:
    """The two axes a lag baseline trades off, measured on the truth series.

    `weekly_amplitude_pct` is (mean weekday load - mean weekend load) / mean
    load. This is the quantity a weekday-misaligned anchor gets wrong, so if
    the mechanism is real the ML - D-7 gap should scale with it across
    countries.

    `week_drift_pct` is the mean absolute week-on-week change in the hourly
    level, mean |actual(t) - actual(t - 168h)| / mean load. This is what a D-7
    lag pays and a fresher anchor does not, so it is the axis on which our
    model can still win despite being misaligned. The two together should
    account for the sign of every readable cell, and they are measured on truth
    alone -- neither reads a forecast, so neither can be fitted to the answer.
    """
    a = actuals[(actuals["target"] >= lo) & (actuals["target"] < hi)].copy()
    a["is_weekend"] = a["target"].dt.dayofweek >= 5
    rows = []
    for country, grp in a.groupby("country_code", sort=True):
        wd = grp.loc[~grp["is_weekend"], "actual"].mean()
        we = grp.loc[grp["is_weekend"], "actual"].mean()
        mean = grp["actual"].mean()
        series = (actuals[actuals["country_code"] == country]
                  .set_index("target")["actual"].sort_index())
        lagged = series.reindex(grp["target"] - pd.Timedelta(hours=168))
        drift = np.abs(grp["actual"].to_numpy() - lagged.to_numpy())
        rows.append({"country": country,
                     "weekday_mean_mw": float(wd), "weekend_mean_mw": float(we),
                     "weekly_amplitude_pct": float((wd - we) / mean * 100),
                     "week_drift_pct": float(np.nanmean(drift) / mean * 100)})
    return pd.DataFrame(rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# section G -- level vs shape, and a leak-free debias
# --------------------------------------------------------------------------

def level_shape_split(panel: pd.DataFrame, arms: List[str]) -> pd.DataFrame:
    """Split each arm's mean absolute error into a daily-level and a shape part.

    For each target day, the level part is the absolute error of the day's mean
    (one number per day); the shape part is what remains after removing that
    day's mean error from every hour. A model whose anchor carries the wrong
    day's *level* loses on the level term; one whose diurnal profile is wrong
    loses on the shape term. The two need different fixes, so they are worth
    separating before anyone proposes one.
    """
    rows = []
    p = panel.copy()
    p["day"] = p["target"].dt.normalize()
    for country, grp in p.groupby("country_code", sort=True):
        rec = {"country": country, "mean_load_mw": float(grp["actual"].mean())}
        for arm in arms:
            err = grp[arm] - grp["actual"]
            day_mean_err = err.groupby(grp["day"]).transform("mean")
            level = float(day_mean_err.abs().mean())
            shape = float((err - day_mean_err).abs().mean())
            total = float(err.abs().mean())
            rec[f"mae_{arm}"] = total
            rec[f"level_mae_{arm}"] = level
            rec[f"shape_mae_{arm}"] = shape
            rec[f"level_share_{arm}"] = level / total if total else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def debias(panel: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per-country bias correction on `arm`, in two forms (ABL-246's function).

    `_causal` subtracts the mean daily bias over *prior days only* -- what a
    live correction could have known; its first day has no prior and drops out.
    `_insample` subtracts the mean bias over the very window it is then scored
    on: an upper bound on the achievable gain, never a forecastable result.
    """
    out = []
    for country, grp in panel.groupby("country_code", sort=True):
        grp = grp.sort_values("target").copy()
        err = grp[arm] - grp["actual"]
        insample = wape((err - err.mean()).to_numpy(), grp["actual"].to_numpy())
        day = grp["target"].dt.normalize()
        prior = err.groupby(day).mean().expanding().mean().shift(1)
        corrected = grp[arm] - day.map(prior)
        ok = corrected.notna()
        causal = (wape((corrected[ok] - grp.loc[ok, "actual"]).to_numpy(),
                       grp.loc[ok, "actual"].to_numpy()) if ok.any() else float("nan"))
        out.append({"country": country,
                    f"wape_{arm}": wape(err.to_numpy(), grp["actual"].to_numpy()),
                    f"wape_{arm}_debiased_causal": causal,
                    f"wape_{arm}_debiased_insample": insample,
                    "n_causal": int(ok.sum())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# section H -- hour-of-day profile
# --------------------------------------------------------------------------

def hourly_profile(panel: pd.DataFrame, arms: List[str]) -> pd.DataFrame:
    rows = []
    p = panel.copy()
    p["hour"] = p["target"].dt.hour
    for hour, grp in p.groupby("hour", sort=True):
        ev = grp[~grp["country_code"].isin(NOT_EVALUABLE)]
        rec = {"hour": int(hour), "n": len(ev)}
        for arm in arms:
            err = (ev[arm] - ev["actual"]).to_numpy()
            rec[f"wape_{arm}"] = wape(err, ev["actual"].to_numpy())
            rec[f"relbias_{arm}"] = float(err.sum() / ev["actual"].sum() * 100)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# section I -- artifact audit
# --------------------------------------------------------------------------

def artifact_audit(models_dir: Path) -> List[Dict]:
    """What each served load artifact depends on, and how old it is.

    The number this section exists for is `history_importance_share`: the
    fraction of the model's decision weight that sits on the lag and rolling
    columns -- exactly the block the proxy-row path anchors on the wrong day.
    A model that spends most of its weight there cannot be indifferent to
    having it mis-anchored.

    Read with `joblib` under the rail interpreter. An xgboost 3.3.0 pickle
    opened under 2.1.4 keeps its trees and silently resets the fitted
    intercept, so this must run under `.venv\\Scripts\\python.exe`; the version
    actually in use is recorded in the output.
    """
    import joblib

    out = []
    for cc in sorted(SUPPORTED_COUNTRIES):
        path = models_dir / cc / "load" / "model.joblib"
        if not path.exists():
            out.append({"country": cc, "artifact_present": False})
            continue
        data = joblib.load(path)
        feats = list(data.get("feature_columns") or [])
        rec = {
            "country": cc, "artifact_present": True,
            "algorithm": data.get("algorithm"),
            "saved_at": data.get("saved_at"),
            "model_version": data.get("model_version"),
            "n_features": len(feats),
            "history_features": [f for f in feats
                                 if f.startswith(HISTORY_FEATURE_PREFIXES)],
            "training_mape": (data.get("training_metrics") or {}).get("mape"),
            "training_mase": (data.get("training_metrics") or {}).get("mase"),
        }
        rec["n_history_features"] = len(rec["history_features"])
        try:
            imp = np.asarray(data["model"].feature_importances_, dtype=float)
            total = float(imp.sum()) or 1.0
            share = {f: float(v) / total for f, v in zip(feats, imp)}
            rec["history_importance_share"] = round(
                sum(v for f, v in share.items()
                    if f.startswith(HISTORY_FEATURE_PREFIXES)), 4)
            for name in ("target_value_lag_1d", "target_value_lag_7d",
                         "target_value_lag_14d"):
                rec[f"imp_{name}"] = round(share.get(name, 0.0), 4)
            rec["imp_rolling"] = round(
                sum(v for f, v in share.items()
                    if f.startswith("target_value_roll_")), 4)
            rec["top_features"] = [
                [f, round(v, 4)] for f, v in
                sorted(share.items(), key=lambda kv: -kv[1])[:5]]
        except Exception as exc:               # pragma: no cover - artifact shape
            rec["importance_error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# section J -- what the proxy row actually carried
# --------------------------------------------------------------------------

def build_history_frame(act: pd.Series) -> pd.DataFrame:
    """Mirror `create_all_features`' history block on one country's hourly series.

    Deliberately positional, because `src/features.py` is: `create_lag_features`
    and `create_rolling_features` call `.shift(n)` on the frame, which counts
    **rows**, not hours. `load_training_data` drops NaN hours before that, so a
    gap in `energy_load` silently makes a "24-hour" lag something other than 24
    hours. Reproducing the row semantics is the only way to measure how far the
    served value sat from the value its column name promises.
    """
    df = pd.DataFrame({"timestamp_utc": act.index, "target_value": act.to_numpy()})
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    for days in config.LAG_DAYS:
        df[f"target_value_lag_{days}d"] = df["target_value"].shift(days * 24)
    for window in config.ROLLING_WINDOWS:
        base = df["target_value"].shift(1).rolling(window=window, min_periods=1)
        df[f"target_value_roll_{window}h_mean"] = base.mean()
    essential = [c for c in df.columns if "_lag_" in c]
    df = df[df[essential].notna().all(axis=1)].reset_index(drop=True)
    df["hour"] = df["timestamp_utc"].dt.hour
    return df


def proxy_row_audit(archive: pd.DataFrame, served_actuals: pd.DataFrame,
                    truth: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the served history block and compare it to a target-aligned one.

    For each (country, run day, target hour) in the D+2 arm, this repeats what
    `predict_d2` does: build the history frame over the same lookback window the
    serving path uses, take the most recent row carrying the target's
    hour-of-day, and read its lag columns. Against that it puts the value the
    same column name would have had if it were anchored on the target --
    `actual(target - 7 days)` for `lag_7d`, which a D+2 forecast *can* compute,
    since 168h is well past the 64h horizon.

    The anchor gap reported is a **lower bound**. It assumes every actual up to
    the serving path's own date bound had already been ingested when the run
    fired; ABL-71 (prod ingest stale) can only push the last observed day
    further back, never forward, so the true gap is at least this large.
    """
    lookback_days = max(config.LAG_DAYS) + 7          # forecaster.py's own value
    rows = []
    for country, arch_c in archive.groupby("country_code", sort=True):
        served = served_actuals[served_actuals["country_code"] == country]
        served = served.set_index("target")["actual"].sort_index()
        truth_c = truth[truth["country_code"] == country].set_index("target")["actual"]
        if served.empty:
            continue
        for run_day, arch_r in arch_c[arch_c["g"] == 2].groupby("run_day"):
            # `predict_d2`: start = ref - 21d, end = ref + 1d, end exclusive.
            start = run_day - pd.Timedelta(days=lookback_days)
            end = run_day + DAY
            window = served[(served.index >= start) & (served.index < end)]
            if len(window) < 24 * (max(config.LAG_DAYS) + 1):
                continue
            hist = build_history_frame(window)
            if hist.empty:
                continue
            for target in sorted(arch_r["target"].unique()):
                target = pd.Timestamp(target)
                same_hour = hist[hist["hour"] == target.hour]
                if same_hour.empty:
                    continue
                proxy = same_hour.iloc[-1]
                proxy_ts = proxy["timestamp_utc"]
                aligned_lag7 = truth_c.get(target - 7 * DAY, np.nan)
                aligned_lag1 = truth_c.get(target - DAY, np.nan)
                actual = truth_c.get(target, np.nan)
                rows.append({
                    "country": country, "run_day": run_day, "target": target,
                    "proxy_timestamp": proxy_ts,
                    "anchor_gap_days": float((target.normalize() - proxy_ts.normalize()) / DAY),
                    "proxy_dow": int(proxy_ts.dayofweek),
                    "target_dow": int(target.dayofweek),
                    "served_lag_7d": float(proxy["target_value_lag_7d"]),
                    "aligned_lag_7d": float(aligned_lag7) if pd.notna(aligned_lag7) else np.nan,
                    "served_lag_1d": float(proxy["target_value_lag_1d"]),
                    "aligned_lag_1d": float(aligned_lag1) if pd.notna(aligned_lag1) else np.nan,
                    # What the column name promises, taken by clock rather than
                    # by row count -- the positional-shift check.
                    "clock_lag_7d_at_proxy": float(served.get(proxy_ts - 7 * DAY, np.nan)),
                    "actual": float(actual) if pd.notna(actual) else np.nan,
                })
    return pd.DataFrame(rows)


def summarise_proxy_audit(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, grp in audit.groupby("country", sort=True):
        mean_load = grp["actual"].mean()
        served, aligned = grp["served_lag_7d"], grp["aligned_lag_7d"]
        ok = served.notna() & aligned.notna()
        pos = grp["served_lag_7d"] - grp["clock_lag_7d_at_proxy"]
        rows.append({
            "country": country, "n": len(grp),
            "median_anchor_gap_days": float(grp["anchor_gap_days"].median()),
            "max_anchor_gap_days": float(grp["anchor_gap_days"].max()),
            "pct_target_dow_mismatch_lag7": float(
                ((grp["proxy_dow"] != grp["target_dow"]).mean()) * 100),
            "lag7_misalignment_mape": float(
                (served[ok] - aligned[ok]).abs().sum() / aligned[ok].abs().sum() * 100)
            if ok.any() else float("nan"),
            "lag7_misalignment_pct_of_mean_load": float(
                (served[ok] - aligned[ok]).abs().mean() / mean_load * 100)
            if ok.any() and mean_load else float("nan"),
            "positional_shift_hours_pct": float((pos.abs() > 1e-9).mean() * 100),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def fmt(df: pd.DataFrame, cols: Optional[List[str]] = None, n: int = 60) -> str:
    view = df[cols] if cols else df
    return view.head(n).to_string(index=False, float_format=lambda v: f"{v:8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replica-db", default=None,
                    help="read-only replica path (default: ENERGY_DB_PATH/config)")
    ap.add_argument("--json-out", required=True, help="machine record path")
    ap.add_argument("--models-dir", default=None,
                    help="artifact root for the section I audit (default: config.MODELS_DIR)")
    ap.add_argument("--max-target", default=DEFAULT_MAX_TARGET,
                    help="exclusive upper bound on target hours (default: ABL-246's window end)")
    ap.add_argument("--min-day-completeness", type=float, default=0.0,
                    help="ABL-639: drop a country-day from the paired intervals "
                         "when its scored hours fall below this fraction of the "
                         "hours the window allows that day. Default 0.0 keeps "
                         "every day, which is what the pack published; the "
                         f"{SENSITIVITY_MIN_DAY_COMPLETENESS} arm is reported "
                         "beside it either way")
    args = ap.parse_args()
    if not 0.0 <= args.min_day_completeness <= 1.0:
        ap.error("--min-day-completeness must be in [0.0, 1.0]")

    replica = args.replica_db or str(config.DATABASE_PATH)
    models_dir = Path(args.models_dir) if args.models_dir else Path(config.MODELS_DIR)
    conn = connect_ro(replica)

    archive = load_archive(conn, args.max_target)
    guard_census = pd.DataFrame(archive.attrs.get("guard_census", []))
    guard_would_refuse = int(archive.attrs.get("guard_would_refuse", 0))
    guard_rows_read = int(archive.attrs.get("guard_rows_read", 0))
    guard_rows_dropped = int(archive.attrs.get("guard_rows_dropped", 0))
    ml = build_ml_arms(archive)

    # Reach back a full seasonal-naive lag *and* the serving lookback before the
    # first target, or the D-7 join silently truncates the scored window by a
    # week and section J has no history to rebuild.
    lo = (ml["target"].min() - pd.Timedelta(days=max(config.LAG_DAYS) + 9)).strftime("%Y-%m-%d")
    hi = (ml["target"].max() + DAY).strftime("%Y-%m-%d")
    truth = load_actuals(conn, lo, hi)
    zero_rows = truth.attrs.get("zero_rows_dropped", 0)
    served_actuals = load_actuals(conn, lo, hi, quality_filter=True)

    lag_days = [1, 2, 3, 7, 14]
    band_arms = ["ml_band"] + [f"d{k}_naive" for k in lag_days]
    arms = ["ml_band", "ml_g1", "ml_g2"] + [f"d{k}_naive" for k in lag_days]

    # Three panels, kept apart on purpose, because each added arm narrows the
    # intersection and ABL-246's most marginal cell (LV, ci_lo = +0.03) is
    # decided by a handful of rows.
    #
    #   panel_a  the registered band arm, the D-7 lag and truth -- ABL-246's
    #            own arm requirement, so section A reproduces on like for like.
    #            It is still not *identical*: ABL-246 additionally required its
    #            three TSO arms, so this intersection is the wider one.
    #   panel    + the whole lag ladder (D-1..D-14), for the descriptive table.
    #   panel_g  + both run-day offsets for the same target hour, which costs
    #            the window's first target day. Everything that compares g=1
    #            against g=2 runs on it, and says so.
    panel_a = ml.merge(truth, on=["country_code", "target"], how="inner")
    panel_a = attach_lag_arms(panel_a, truth, [7])
    panel_a = panel_a.dropna(subset=["ml_band", "d7_naive", "actual"])

    panel = attach_lag_arms(
        panel_a, truth, [k for k in lag_days if k != 7]
    ).dropna(subset=band_arms + ["actual"])
    panel_g = panel.dropna(subset=["ml_g1", "ml_g2"])

    table = score(panel, band_arms)
    table_g = score(panel_g, arms)
    ev = table[table["evaluable"]]
    ev_g = table_g[table_g["evaluable"]]

    # Every paired interval below is screened at the same threshold, so the
    # primary read is one protocol rather than a mixture. `mdc` is 0.0 unless
    # asked for, and 0.0 drops nothing.
    mdc = args.min_day_completeness
    sens = SENSITIVITY_MIN_DAY_COMPLETENESS

    # ---- A: reproduce ABL-246 section 4.1 (panel_a) -----------------------
    a_band_vs_d7 = paired_daily(panel_a, "ml_band", "d7_naive", mdc)
    table_a = score(panel_a, ["ml_band", "d7_naive"])
    a_readable = readable_cells(a_band_vs_d7)
    losers, winners = a_readable["readable_losers"], a_readable["readable_winners"]

    # ---- B: the run-day offset split (panel_g) ----------------------------
    b_g2_vs_g1 = paired_daily(panel_g, "ml_g2", "ml_g1", mdc)

    # ---- C: each ML arm against the lag ladder (panel_g) ------------------
    c_g2_vs_d7 = paired_daily(panel_g, "ml_g2", "d7_naive", mdc)
    c_g1_vs_d7 = paired_daily(panel_g, "ml_g1", "d7_naive", mdc)

    # ---- K: day completeness, and the complete-days-only arm (ABL-639) ----
    # The diagnostic is emitted whatever the threshold; the sensitivity arm is
    # recomputed at 1.0 beside the primary so both intervals are on the page.
    k_completeness = {
        name: {
            "window_start": str(p["target"].min()),
            "window_end": str(p["target"].max()),
            "hours_expected_per_day": {str(d.date()): int(n) for d, n
                                       in hours_expected_per_day(p).items()},
            "n_country_days": int(len(tbl)),
            "n_short_country_days": int(tbl["is_short"].sum()),
            "per_country_day": json.loads(tbl.to_json(orient="records")),
            "short_country_days": json.loads(
                tbl[tbl["is_short"]].to_json(orient="records")),
        }
        for name, p, tbl in (("panel_a", panel_a, day_completeness(panel_a)),
                             ("panel_g", panel_g, day_completeness(panel_g)))
    }
    k_sens = {
        "section_a_ml_band_vs_d7": paired_daily(panel_a, "ml_band", "d7_naive", sens),
        "section_b_ml_g2_vs_ml_g1": paired_daily(panel_g, "ml_g2", "ml_g1", sens),
        "section_c_ml_g2_vs_d7": paired_daily(panel_g, "ml_g2", "d7_naive", sens),
        "section_c_ml_g1_vs_d7": paired_daily(panel_g, "ml_g1", "d7_naive", sens),
    }
    #: Primary beside sensitivity, per country, so the k trade is visible in
    #: the record rather than left to a reader to recompute from two tables.
    #: `k_days_short` counts the same days in both arms -- it is a property of
    #: the panel, not of the threshold -- so it is carried once, unsuffixed,
    #: rather than as two always-equal columns.
    k_compare = {
        name: primary.merge(k_sens[name].drop(columns=["k_days_short"]),
                            on="country", suffixes=("_primary", "_screened"))
        for name, primary in (("section_a_ml_band_vs_d7", a_band_vs_d7),
                              ("section_b_ml_g2_vs_ml_g1", b_g2_vs_g1),
                              ("section_c_ml_g2_vs_d7", c_g2_vs_d7),
                              ("section_c_ml_g1_vs_d7", c_g1_vs_d7))
    }

    # ---- D: anchor identification (panel_g) -------------------------------
    d_anchor_g2 = anchor_identification(panel_g, truth, "ml_g2")
    d_anchor_d7 = anchor_identification(panel_g, truth, "d7_naive")

    # ---- E/F: weekday signature (panel_g) ---------------------------------
    e_weekday = weekday_profile(panel_g, ["ml_g2", "ml_band", "d7_naive"])
    f_amp = weekly_amplitude(truth, panel["target"].min(), panel["target"].max() + HOUR)
    gap = paired_daily(panel_g, "ml_g2", "d7_naive", mdc).merge(f_amp, on="country")
    gap = gap[~gap["country"].isin(NOT_EVALUABLE)]
    rho_amp = spearman(gap["weekly_amplitude_pct"].to_numpy(),
                       gap["mean_daily_wape_diff"].to_numpy())
    rho_drift = spearman(gap["week_drift_pct"].to_numpy(),
                         gap["mean_daily_wape_diff"].to_numpy())
    # The two axes together: amplitude is what our anchor misreads, drift is
    # what the D-7 lag pays. Their ratio should order the readable cells.
    gap["amplitude_over_drift"] = gap["weekly_amplitude_pct"] / gap["week_drift_pct"]
    rho_ratio = spearman(gap["amplitude_over_drift"].to_numpy(),
                         gap["mean_daily_wape_diff"].to_numpy())

    # ---- G: level vs shape (panel_g), causal debias (panel) ---------------
    g_split = level_shape_split(panel_g, ["ml_g2", "ml_band", "d7_naive"])
    g_debias = debias(panel, "ml_band")

    # ---- H: hour-of-day (panel_g) -----------------------------------------
    h_hourly = hourly_profile(panel_g, ["ml_band", "ml_g2", "d7_naive"])

    # ---- I: artifacts ------------------------------------------------------
    i_artifacts = artifact_audit(models_dir)
    served_algo = (panel.groupby("country_code")["ml_model"]
                   .agg(lambda s: s.mode().iat[0]).to_dict())
    algo_mismatch = sorted(
        a["country"] for a in i_artifacts
        if a.get("artifact_present") and served_algo.get(a["country"])
        and a.get("algorithm") != served_algo[a["country"]])

    # ---- J: the served feature block --------------------------------------
    j_audit = proxy_row_audit(archive, served_actuals, truth)
    j_summary = summarise_proxy_audit(j_audit) if not j_audit.empty else pd.DataFrame()

    # ---- I/F join: does importance share track the loss? -------------------
    imp = pd.DataFrame([{"country": a["country"],
                         "history_importance_share": a.get("history_importance_share")}
                        for a in i_artifacts if a.get("artifact_present")])
    gap_imp = gap.merge(imp, on="country", how="left")
    rho_imp = spearman(gap_imp["history_importance_share"].to_numpy(dtype=float),
                       gap_imp["mean_daily_wape_diff"].to_numpy(dtype=float))

    # ---- report ------------------------------------------------------------
    def medians(frame: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
        return {c: float(frame[c].median()) for c in cols}

    import xgboost
    meta = {
        "issue": "ABL-607",
        "parent_pack": "ABL-246 reports/abl_246_tso_d1_load_pack.md section 4.1",
        "window_start": str(panel["target"].min()),
        "window_end": str(panel["target"].max()),
        "target_days": int(panel["target"].dt.normalize().nunique()),
        "n_scored_pairs": int(len(panel)),
        "panel_a_n_scored_pairs": int(len(panel_a)),
        "panel_a_target_days": int(panel_a["target"].dt.normalize().nunique()),
        "abl246_n_scored_pairs": 8436,
        "panel_g_window_start": str(panel_g["target"].min()),
        "panel_g_window_end": str(panel_g["target"].max()),
        "panel_g_target_days": int(panel_g["target"].dt.normalize().nunique()),
        "panel_g_n_scored_pairs": int(len(panel_g)),
        "panel_note": (
            "section A uses panel_a (band arm + D-7 + truth, ABL-246's arm "
            "requirement minus its three TSO arms, so slightly wider); the "
            "descriptive table and the G debias use panel (+ the D-1..D-14 "
            "ladder); sections B/C/D/E/F/G-split/H use panel_g (+ both run-day "
            "offsets for the same target hour), which costs the first target day"),
        "countries_scored": int(panel["country_code"].nunique()),
        "countries_evaluable": int(len(ev)),
        "not_evaluable": sorted(NOT_EVALUABLE),
        "basis": "out-of-sample except the labelled in-sample debias column",
        "truth": "energy_load hourly means, 0.0 rows dropped (ABL-111/ABL-109)",
        "zero_rows_dropped": zero_rows,
        # ABL-639. Deliberately not prefixed `guard`: the ABL-619 pin in
        # tests/test_abl607_guarded_read.py reads every `meta` key starting
        # with that word out of this literal and demands the *committed*
        # report carry it, so a new one there would go red on an artifact this
        # change is forbidden to rewrite.
        "min_day_completeness": mdc,
        "sensitivity_min_day_completeness": sens,
        "day_completeness_note": (
            "ABL-639: each country-day contributes one daily WAPE difference to "
            "the paired interval regardless of the hours it was scored on, so a "
            "truncated day is weighted like a whole one and counts as a full "
            "observation in the k that keys T_CRIT. hours_expected is the hours "
            "the fleet-wide window allows that day, not a constant 24 -- the "
            "window's end days are legitimately partial (16 and 1 here). The "
            "primary threshold above screens nothing at 0.0; the complete-days-"
            "only arm is in section_k_day_completeness.sensitivity either way"),
        # Names ABL-611's category, not ABL-617's, on purpose: see the
        # `disposition` key below.
        "guard": ("ABL-431/458 plausibility reference measured over the archive "
                  "read, report-only; the read is EXEMPT_READS (ABL-611)"),
        "guard_rows_read": guard_rows_read,
        "guard_would_refuse": guard_would_refuse,
        "guard_rows_dropped": guard_rows_dropped,
        "replica": replica,
        "models_dir": str(models_dir),
        "python": sys.version.split()[0],
        "xgboost": xgboost.__version__,
        "arms": arms,
        "d2_horizon_band": list(D2_HORIZON_BAND),
    }

    record = {
        "meta": meta,
        "section_0_plausibility_census": {
            "table": VINTAGE_ARCHIVE_TABLE,
            "column": "load",
            "tolerance": PLAUSIBILITY_TOLERANCE,
            "rows_read": guard_rows_read,
            "rows_would_be_refused": guard_would_refuse,
            "rows_dropped": guard_rows_dropped,
            # ABL-617 moved this file to `ML_SLICE_ONLY_EXEMPT` and deliberately
            # did **not** touch this string. A report is the record of a run,
            # and at the run that produced the committed
            # `abl_607_d2_load_diagnosis_reread.json` the disposition was
            # `EXEMPT_READS`. Rewriting it here would leave the code claiming
            # one thing and the published artifact carrying another, with no
            # re-run to reconcile them -- which is ABL-619 exactly. The current
            # category is in `plausibility_census`'s docstring, which is prose
            # and lands in no artifact. Change this only in a commit that also
            # regenerates the report.
            "disposition": "EXEMPT_READS (ABL-611): report-only, filters nothing",
            "as_of": None,
            "note": (
                "reference = 3 x max(p99.5 TSO day-ahead load vintages, p99.5 "
                "energy_load) per country; source='ml' rows are excluded from "
                "setting it by TSO_FORECAST_SOURCES/forecast_read. One-sided, "
                "so the only rows it could remove are our own largest "
                "over-forecasts -- the errors this pack measures. Reported "
                "rather than filtered for that reason; a non-zero count is a "
                "finding about our model, not a cleaning step"),
            "per_country": json.loads(guard_census.to_json(orient="records"))
            if not guard_census.empty else [],
        },
        "per_country": json.loads(table.to_json(orient="records")),
        "per_country_panel_g": json.loads(table_g.to_json(orient="records")),
        "fleet_medians": medians(ev, [f"wape_{a}" for a in band_arms]),
        "fleet_medians_panel_g": medians(ev_g, [f"wape_{a}" for a in arms]),
        "section_a_reproduction": {
            "per_country": json.loads(table_a.to_json(orient="records")),
            "paired_ml_band_vs_d7": json.loads(a_band_vs_d7.to_json(orient="records")),
            "readable_losers": losers, "readable_winners": winners,
            "abl246_losers": ABL246_LOSERS, "abl246_winners": ABL246_WINNERS,
            "losers_match": sorted(losers) == sorted(ABL246_LOSERS),
            "winners_match": sorted(winners) == sorted(ABL246_WINNERS),
        },
        "section_b_run_offset": {
            "paired_ml_g2_vs_ml_g1": json.loads(b_g2_vs_g1.to_json(orient="records")),
            "band_g_composition": json.loads(
                panel.groupby("country_code")["ml_band_g"]
                .value_counts().unstack(fill_value=0).reset_index()
                .to_json(orient="records")),
        },
        "section_c_vs_lag_ladder": {
            "paired_ml_g2_vs_d7": json.loads(c_g2_vs_d7.to_json(orient="records")),
            "paired_ml_g1_vs_d7": json.loads(c_g1_vs_d7.to_json(orient="records")),
        },
        "section_d_anchor": {
            "ml_g2": json.loads(d_anchor_g2.to_json(orient="records")),
            "d7_naive": json.loads(d_anchor_d7.to_json(orient="records")),
        },
        "section_e_weekday": json.loads(e_weekday.to_json(orient="records")),
        "section_f_amplitude": {
            "per_country": json.loads(f_amp.to_json(orient="records")),
            "gap_vs_axes": json.loads(gap.to_json(orient="records")),
            "gap_arm": "ml_g2 - d7_naive, paired daily, panel_g",
            "spearman_amplitude_vs_gap": rho_amp,
            "spearman_week_drift_vs_gap": rho_drift,
            "spearman_amplitude_over_drift_vs_gap": rho_ratio,
            "spearman_history_importance_vs_gap": rho_imp,
        },
        "section_g_level_shape": {
            "split": json.loads(g_split.to_json(orient="records")),
            "debias_ml_band": json.loads(g_debias.to_json(orient="records")),
        },
        "section_h_hourly": json.loads(h_hourly.to_json(orient="records")),
        "section_i_artifacts": {
            "artifacts": i_artifacts,
            "served_algorithm": served_algo,
            "artifact_vs_served_algorithm_mismatch": algo_mismatch,
        },
        "section_j_proxy_row": {
            "summary": json.loads(j_summary.to_json(orient="records"))
            if not j_summary.empty else [],
        },
        "section_k_day_completeness": {
            "min_day_completeness": mdc,
            "sensitivity_min_day_completeness": sens,
            "screen_applied_to": [
                "section_a_reproduction.paired_ml_band_vs_d7",
                "section_b_run_offset.paired_ml_g2_vs_ml_g1",
                "section_c_vs_lag_ladder.paired_ml_g2_vs_d7",
                "section_c_vs_lag_ladder.paired_ml_g1_vs_d7",
                "section_f_amplitude.gap_vs_axes",
            ],
            "not_screened": (
                "the pooled WAPE tables (per_country, fleet_medians, sections "
                "E/G/H) weight by hour, so a short day already contributes "
                "proportionally there and needs no screen; only the daily "
                "paired intervals give every day equal weight"),
            "panels": k_completeness,
            "sensitivity": {
                name: json.loads(frame.to_json(orient="records"))
                for name, frame in k_sens.items()
            },
            "primary_vs_sensitivity": {
                name: json.loads(frame.to_json(orient="records"))
                for name, frame in k_compare.items()
            },
            "sensitivity_section_a_readable": readable_cells(
                k_sens["section_a_ml_band_vs_d7"]),
            "primary_section_a_readable": a_readable,
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print("== 0: ABL-431 plausibility reference over the archive read "
          "(report-only) ==")
    print(f"  rows read {guard_rows_read} | would be refused "
          f"{guard_would_refuse} | dropped {guard_rows_dropped} | "
          f"tolerance {PLAUSIBILITY_TOLERANCE}x")
    if not guard_census.empty:
        print(fmt(guard_census.sort_values("max_over_threshold", ascending=False),
                  ["country", "n_rows", "evaluable", "reference_mw",
                   "threshold_mw", "ml_max_mw", "max_over_threshold",
                   "n_null_in_read", "n_would_be_refused"]))
    print()
    print(f"panel   {meta['window_start']} -> {meta['window_end']} | "
          f"{meta['target_days']} target days | n={meta['n_scored_pairs']} | "
          f"{meta['countries_evaluable']} evaluable")
    print(f"panel_g {meta['panel_g_window_start']} -> {meta['panel_g_window_end']} | "
          f"{meta['panel_g_target_days']} target days | n={meta['panel_g_n_scored_pairs']}")
    print("\n== per country (WAPE %, panel) ==")
    print(fmt(table, ["country", "n", "ml_model", "wape_ml_band", "wape_d7_naive",
                      "wape_d1_naive", "wape_d2_naive", "wape_d14_naive",
                      "relbias_ml_band"]))
    print("\n== per country (WAPE %, panel_g) ==")
    print(fmt(table_g, ["country", "n", "wape_ml_g1", "wape_ml_g2",
                        "wape_d7_naive"]))
    print("\n== fleet medians (evaluable) ==")
    for k, v in record["fleet_medians"].items():
        print(f"  panel   {k:22} {v:7.2f}")
    for k, v in record["fleet_medians_panel_g"].items():
        print(f"  panel_g {k:22} {v:7.2f}")
    print(f"\n== A: reproduction of ABL-246 4.1 (panel_a, n={len(panel_a)} "
          f"vs pack 8436) ==")
    print(fmt(a_band_vs_d7.merge(table_a[["country", "wape_ml_band", "wape_d7_naive"]],
                                 on="country")
              [["country", "k_days", "wape_ml_band", "wape_d7_naive",
                "mean_daily_wape_diff", "ci_lo", "ci_hi", "readable"]]))
    print(f"  readable losers  {losers}")
    print(f"  ABL-246 losers   {ABL246_LOSERS}")
    print(f"  match: losers={record['section_a_reproduction']['losers_match']} "
          f"winners={record['section_a_reproduction']['winners_match']}")
    print("\n== B: one extra day of anchor staleness (ml_g2 - ml_g1) ==")
    print(fmt(b_g2_vs_g1))
    print("\n== C: ml_g2 vs D-7 / ml_g1 vs D-7 ==")
    print(fmt(c_g2_vs_d7.merge(c_g1_vs_d7, on="country", suffixes=("_g2", "_g1"))
              [["country", "mean_daily_wape_diff_g2", "readable_g2",
                "mean_daily_wape_diff_g1", "readable_g1"]]))
    print("\n== D: which day's actual does ml_g2 track? ==")
    print(fmt(d_anchor_g2, ["country", "argmin_k", "wape_at_k0", "wape_at_argmin",
                            "wape_vs_actual_lag1d", "wape_vs_actual_lag2d",
                            "wape_vs_actual_lag7d"]))
    print("\n== E: relative bias by target weekday (evaluable pooled) ==")
    ew = e_weekday[~e_weekday["country"].isin(NOT_EVALUABLE)]
    print(fmt(ew.groupby(["dow", "dow_name"], as_index=False)
              [["relbias_ml_g2", "relbias_ml_band", "relbias_d7_naive",
                "wape_ml_g2", "wape_d7_naive"]].mean()))
    print("\n== F: the two axes against the ml_g2 - D-7 gap ==")
    print(fmt(gap[["country", "weekly_amplitude_pct", "week_drift_pct",
                   "amplitude_over_drift", "mean_daily_wape_diff", "readable"]]))
    print(f"  Spearman(weekly amplitude, gap)     = {rho_amp:+.3f}")
    print(f"  Spearman(week drift, gap)           = {rho_drift:+.3f}")
    print(f"  Spearman(amplitude/drift, gap)      = {rho_ratio:+.3f}")
    print(f"  Spearman(history importance, gap)   = {rho_imp:+.3f}")
    print("\n== G: level share of MAE, and causal debias ==")
    print(fmt(g_split.merge(g_debias, on="country")
              [["country", "level_share_ml_g2", "level_share_d7_naive",
                "wape_ml_band", "wape_ml_band_debiased_causal",
                "wape_ml_band_debiased_insample"]]))
    print("\n== H: hour-of-day ==")
    print(fmt(h_hourly, ["hour", "wape_ml_band", "wape_ml_g2", "wape_d7_naive",
                         "relbias_ml_band", "relbias_d7_naive"], n=24))
    print("\n== I: artifacts ==")
    print(fmt(pd.DataFrame([a for a in i_artifacts if a.get("artifact_present")]),
              ["country", "algorithm", "saved_at", "n_features",
               "n_history_features", "history_importance_share",
               "imp_target_value_lag_7d", "imp_rolling"]))
    print(f"  artifact/served algorithm mismatch: {algo_mismatch}")
    print("\n== J: what the proxy row carried ==")
    if not j_summary.empty:
        print(fmt(j_summary))
    print(f"\n== K: day completeness (ABL-639, primary screen "
          f"{mdc:.2f}, sensitivity {sens:.2f}) ==")
    for name, blk in k_completeness.items():
        print(f"  {name} {blk['window_start']} -> {blk['window_end']} | "
              f"{blk['n_short_country_days']} of {blk['n_country_days']} "
              f"country-days short of the hours the window allows")
        short = pd.DataFrame(blk["short_country_days"])
        if not short.empty:
            print(fmt(short.sort_values("completeness"), n=25))
    cmp_a = k_compare["section_a_ml_band_vs_d7"]
    print("\n  section A, all days vs complete days only:")
    print(fmt(cmp_a[["country", "k_days_primary", "k_days_screened",
                     "k_days_short", "mean_daily_wape_diff_primary",
                     "ci_lo_primary", "mean_daily_wape_diff_screened",
                     "ci_lo_screened", "readable_primary", "readable_screened"]]))
    sens_a = record["section_k_day_completeness"]["sensitivity_section_a_readable"]
    print(f"  readable losers  primary   {losers}")
    print(f"  readable losers  screened  {sens_a['readable_losers']}")
    print(f"  readable winners primary   {winners}")
    print(f"  readable winners screened  {sens_a['readable_winners']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
