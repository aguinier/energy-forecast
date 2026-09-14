"""ABL-794 -- why ML load WAPE roughly doubled in the target week of 2026-09-07.

Diagnosis only. Fits no model and writes nothing but --json-out (and the
optional --rows-out). Every database read goes through
src.db.get_connection(readonly=True), which opens the replica as a mode=ro URI.

Questions, in the order the issue asks them
-------------------------------------------
Q1  Does the ABL-179 / ABL-607 proxy-row anchor reach the D+1 load path?
    Answered from the served rows, not from the code. For every stored
    forecast value this rebuilds each proxy row the serving path could have
    taken: the frame `Forecaster.predict_d2` builds (the same
    `load_training_data` + `create_all_features` calls over the same 21-day
    lookback), its row at the target hour on day d = run day - 0..6, under
    each weather run issued at or before the run instant (or none). Every
    candidate is predicted with the artifact whose model_version matches the
    served row, and the one that reproduces the stored value is kept. A match
    within MATCH_TOL MW *is* the served feature row, and its day is the anchor.
Q2  Is the doubling that defect meeting a bigger day-to-day swing? The same
    artifact is replayed on the target-aligned row under the same matched
    weather (an oracle: training-definition features, not servable at D+1),
    so served minus oracle isolates what the mis-anchored history block cost.
    That cost is set against how far the served history block sat from the
    aligned one, week 1 against week 2, with a week-1 through-origin slope
    read on week 2. Truth-only week shapes (weekday/weekend contrast, two-day
    swing) say whether the swing itself grew. A servable alternative proxy,
    the same-weekday row at T-7, is replayed alongside.
Q3  Serving-input health: which weather run matched and how old it was, the
    anchor gap by week, hours missing or zero in the serving frame, and how
    many served values no candidate reproduces.

Arms
----
api_d1    dashboard D+1: latest vintage with horizon_hours in [0, 30]
api_d2    dashboard D+2: latest vintage with horizon_hours in [24, 54]
d1_19     D+1 output of the T-1 19:00 UTC run (what api_d1 resolves to)
d2_19     D+2 output of the T-2 19:00 UTC run (the only genuine D+2)
oracle_d1 replay, target-aligned history, weather as matched for d1_19
t7_d1     replay, proxy row at T-7 from the d1_19 run's frame, same weather
t7_d2     replay, proxy row at T-7 from the d2_19 run's frame, its weather
d{k}      actual(T - k days)

Truth is energy_load hourly means (ABL-332) with 0.0 rows dropped
(ABL-111/109). NL is scored and held out of fleet summaries (gross vs net
basis, ABL-277/505/506). Out-of-sample throughout; the single fitted quantity
(Q2's slope) is fitted on week 1 and read on week 2.

Usage:
  .venv/Scripts/python.exe scripts/abl794_load_week_diagnosis.py
      --replica-db C:/Code/able/data/energy_dashboard.db
      --models-dir C:/Code/able/energy-forecast/models
      --json-out reports/abl_794_load_week_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db as src_db
from src.db import get_connection, load_training_data
from src.features import create_all_features

#: A replayed candidate within this many MW of the stored value is the served
#: row. CatBoost/XGBoost predictions are deterministic for a fixed artifact;
#: the slack absorbs float formatting and cross-platform libm, nothing more.
MATCH_TOL = 0.5

#: Looser identification for rows whose anchor-day actuals were revised after
#: the run (ENTSO-E re-fetch): the best candidate is within this share of the
#: served value AND every other anchor day is at least ID_SEPARATION times
#: further away. Reported separately from exact matches, never merged.
NEAR_TOL_SHARE = 0.005
ID_SEPARATION = 3.0

#: `Forecaster.predict_d2`'s own lookback.
LOOKBACK_DAYS = max(config.LAG_DAYS) + 7

#: Candidate anchor days: run day minus 0..6.
ANCHOR_OFFSETS = range(0, 7)

#: Candidate weather runs per target hour: the N latest issued at or before
#: the run instant, plus "none" (the proxy row keeps its own weather).
N_WEATHER_RUNS = 3

#: The last scheduled run of the day (docker/crontab: 07:00 14:00 15:30 19:00).
LAST_RUN_SLOT = "19:00"

#: History-block columns compared served vs aligned.
HISTORY = {"lag1": "target_value_lag_1d", "lag7": "target_value_lag_7d",
           "roll24": "target_value_roll_24h_mean", "roll168": "target_value_roll_168h_mean"}

#: ABL-277/505/506: NL realized load is net of behind-the-meter solar.
NOT_EVALUABLE = {"NL"}

#: The countries ABL-793/ABL-792 flagged for a load WAPE jump.
FLAGGED = ["IT", "HU", "FR", "FI", "HR", "EE", "SI", "SK", "PL", "LV"]

DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def load_served(cc: str, first: date, last: date) -> pd.DataFrame:
    sql = """
        SELECT target_timestamp_utc, generated_at, horizon_hours, forecast_value,
               model_name, model_version
        FROM forecasts
        WHERE country_code = ? AND forecast_type = 'load'
          AND target_timestamp_utc >= ? AND target_timestamp_utc < ?
    """
    with get_connection(readonly=True) as conn:
        df = pd.read_sql_query(sql, conn, params=(
            cc, first.isoformat(), (last + timedelta(days=1)).isoformat()))
    if df.empty:
        return df
    df["target"] = pd.to_datetime(df["target_timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df["gen"] = pd.to_datetime(df["generated_at"], format="mixed", utc=True).dt.tz_localize(None)
    df["g"] = (df["target"].dt.normalize() - df["gen"].dt.normalize()).dt.days
    df["slot"] = df["gen"].dt.floor("30min").dt.strftime("%H:%M")
    return df


def served_arms(df: pd.DataFrame) -> pd.DataFrame:
    """One value per target hour per served arm, keeping each arm's provenance."""
    def latest(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.sort_values("gen").groupby("target").last()

    b1 = latest(df[df["horizon_hours"].between(0, 30)])
    b2 = latest(df[df["horizon_hours"].between(24, 54)])
    r1 = latest(df[(df["g"] == 1) & (df["slot"] == LAST_RUN_SLOT)])
    r2 = latest(df[(df["g"] == 2) & (df["slot"] == LAST_RUN_SLOT)])
    out = pd.DataFrame(index=sorted(set(df["target"])))
    out["api_d1"] = b1["forecast_value"]
    out["api_d2"] = b2["forecast_value"]
    out["api_d2_g"] = b2["g"]
    out["api_d2_slot"] = b2["slot"]
    out["d1_19"] = r1["forecast_value"]
    out["d1_19_gen"] = r1["gen"]
    out["d2_19"] = r2["forecast_value"]
    out["d2_19_gen"] = r2["gen"]
    out.index.name = "target"
    return out


def load_truth(cc: str, start: date, end: date) -> pd.Series:
    sql = """
        SELECT timestamp_utc, load_mw FROM energy_load
        WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ?
    """
    with get_connection(readonly=True) as conn:
        df = pd.read_sql_query(sql, conn, params=(cc, start.isoformat(), end.isoformat()))
    df["ts"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df = df[(df["load_mw"] != 0)].dropna(subset=["load_mw"])
    return df.set_index("ts")["load_mw"].resample("h").mean().dropna()


def load_weather_runs(cc: str, first: date, last: date) -> Dict[str, List[dict]]:
    """Forecast weather rows keyed by the exact timestamp string serving queries
    (`load_weather_forecast_for_hour` matches `timestamp_utc = 'Y-m-d H:M:S'`),
    each list sorted by forecast_run_time descending as its ORDER BY does."""
    sql = """
        SELECT timestamp_utc, forecast_run_time, temperature_2m_k, relative_humidity_2m_frac
        FROM weather_data
        WHERE country_code = ? AND data_quality = 'forecast'
          AND timestamp_utc >= ? AND timestamp_utc < ?
    """
    with get_connection(readonly=True) as conn:
        df = pd.read_sql_query(sql, conn, params=(
            cc, first.isoformat(), (last + timedelta(days=1)).isoformat()))
    out: Dict[str, List[dict]] = {}
    for ts, grp in df.groupby("timestamp_utc"):
        out[ts] = grp.sort_values("forecast_run_time", ascending=False).to_dict("records")
    return out


def weather_candidates(runs: Dict[str, List[dict]], target: pd.Timestamp,
                       run_instant: pd.Timestamp) -> List[Optional[dict]]:
    rows = runs.get(target.strftime("%Y-%m-%d %H:%M:%S"), [])
    cutoff = run_instant.strftime("%Y-%m-%d %H:%M:%S")
    seen, picked = set(), []
    for r in rows:
        frt = r["forecast_run_time"]
        if frt is None or frt > cutoff or frt in seen:
            continue
        seen.add(frt)
        picked.append(r)
        if len(picked) == N_WEATHER_RUNS:
            break
    return picked + [None]


class Frames:
    """`predict_d2`'s feature frame per reference day, built by the real code."""

    def __init__(self, cc: str):
        self.cc = cc
        self.cache: Dict[date, pd.DataFrame] = {}
        self.health: Dict[str, dict] = {}

    def get(self, ref: date) -> pd.DataFrame:
        if ref in self.cache:
            return self.cache[ref]
        start = (ref - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = (ref + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = load_training_data(self.cc, "load", start, end)
        if raw.empty:
            feats = pd.DataFrame()
        else:
            feats = create_all_features(raw, "load")
            feats = feats.set_index("timestamp_utc", drop=False)
        dups = int(feats.index.duplicated().sum()) if not feats.empty else 0
        if dups:
            # `same_hour_data.iloc[-1:]` takes the last of a duplicated hour.
            feats = feats[~feats.index.duplicated(keep="last")]
        self.health[ref.isoformat()] = {
            "hours_expected": 24 * (LOOKBACK_DAYS + 1),
            "hours_in_frame": int(len(raw)),
            "zero_target_hours": int((raw["target_value"] == 0).sum()) if not raw.empty else 0,
            "rows_after_features": int(len(feats)),
            "duplicate_hours": dups,
        }
        self.cache[ref] = feats
        return feats


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def apply_overrides(X: pd.DataFrame, targets: pd.Series, temp: np.ndarray,
                    rh: np.ndarray, has_fc: np.ndarray, cols: List[str]) -> pd.DataFrame:
    """The calendar and weather overrides `predict_d2` applies to a proxy row."""
    h = targets.dt.hour.to_numpy()
    dow = targets.dt.weekday.to_numpy()
    m = targets.dt.month.to_numpy()
    cal = {
        "hour": h, "day_of_week": dow, "month": m, "is_weekend": (dow >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
        "day_sin": np.sin(2 * np.pi * dow / 7), "day_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12),
    }
    for k, v in cal.items():
        if k in cols:
            X[k] = v
    if has_fc.any():
        if "temperature_2m_k" in cols:
            X.loc[has_fc, "temperature_2m_k"] = temp[has_fc]
        if "relative_humidity_2m_frac" in cols:
            X.loc[has_fc, "relative_humidity_2m_frac"] = rh[has_fc]
        tc = temp - 273.15
        if "temperature_c" in cols:
            X.loc[has_fc, "temperature_c"] = tc[has_fc]
        # Python's max(0, x) with x = nan returns 0; np.where(x > 0, x, 0) agrees.
        hd, cd = 18 - tc, tc - 18
        if "heating_degree" in cols:
            X.loc[has_fc, "heating_degree"] = np.where(hd > 0, hd, 0.0)[has_fc]
        if "cooling_degree" in cols:
            X.loc[has_fc, "cooling_degree"] = np.where(cd > 0, cd, 0.0)[has_fc]
    return X


def _wval(w: Optional[dict], key: str) -> float:
    if w is None or w.get(key) is None:
        return np.nan
    return float(w[key])


def predict_rows(model, cols: List[str], bases: np.ndarray, targets: List[pd.Timestamp],
                 weathers: List[Optional[dict]]) -> np.ndarray:
    if len(bases) == 0:
        return np.array([])
    X = pd.DataFrame(bases, columns=cols)
    temp = np.array([_wval(w, "temperature_2m_k") for w in weathers], dtype=float)
    rh = np.array([_wval(w, "relative_humidity_2m_frac") for w in weathers], dtype=float)
    has_fc = np.array([w is not None for w in weathers])
    X = apply_overrides(X, pd.Series(pd.to_datetime(targets)), temp, rh, has_fc, cols)
    return np.asarray(model.predict(X[cols]), dtype=float).ravel()


def replay_country(arms: pd.DataFrame, artifact: dict,
                   runs: Dict[str, List[dict]], frames: Frames) -> pd.DataFrame:
    cols = list(artifact["feature_columns"])
    model = artifact["model"]
    hist_idx = {k: cols.index(c) for k, c in HISTORY.items() if c in cols}
    out = pd.DataFrame(index=arms.index)

    for arm in ("d1_19", "d2_19"):
        meta, bases, tgts, weathers = [], [], [], []
        for target, row in arms[arms[arm].notna()].iterrows():
            run_instant = row[f"{arm}_gen"]
            run_day = run_instant.date()
            F = frames.get(run_day)
            if F.empty:
                continue
            wc = weather_candidates(runs, target, run_instant)
            for off in ANCHOR_OFFSETS:
                ts = pd.Timestamp(run_day - timedelta(days=off)) + target.hour * HOUR
                if ts not in F.index:
                    continue
                base = F.loc[ts, cols].to_numpy(dtype=float)
                for rank, w in enumerate(wc):
                    meta.append({
                        "target": target, "anchor": ts,
                        "anchor_gap": int((target.normalize() - ts.normalize()) / DAY),
                        "w_rank": rank if w is not None else -1,
                        "w_run": None if w is None else w["forecast_run_time"],
                        "w_temp_k": _wval(w, "temperature_2m_k"),
                        "served": float(row[arm]), "run_instant": run_instant,
                        "i": len(bases),
                    })
                    bases.append(base)
                    tgts.append(target)
                    weathers.append(w)
        if not meta:
            continue
        B = np.vstack(bases)
        cand = pd.DataFrame(meta)
        cand["pred"] = predict_rows(model, cols, B, tgts, weathers)
        cand["resid"] = (cand["pred"] - cand["served"]).abs()
        best = cand.loc[cand.groupby("target")["resid"].idxmin()].set_index("target")
        other = cand.merge(best[["anchor_gap"]].rename(columns={"anchor_gap": "best_gap"}),
                           left_on="target", right_index=True)
        runner_up = other[other["anchor_gap"] != other["best_gap"]].groupby("target")["resid"].min()
        out[f"{arm}_anchor_gap"] = best["anchor_gap"]
        out[f"{arm}_resid"] = best["resid"]
        out[f"{arm}_runner_up_resid"] = runner_up
        out[f"{arm}_w_rank"] = best["w_rank"]
        out[f"{arm}_w_run"] = best["w_run"]
        out[f"{arm}_temp_c"] = best["w_temp_k"] - 273.15
        for k, j in hist_idx.items():
            out[f"{arm}_{k}"] = pd.Series(B[best["i"].to_numpy(), j], index=best.index)

        # Counterfactual replays under the matched weather.
        c_bases, c_tgts, c_w, c_kind = [], [], [], []
        for target, b in best.iterrows():
            w = weathers[int(b["i"])]
            run_day = b["run_instant"].date()
            if arm == "d1_19":
                F_t = frames.get(target.date())
                if not F_t.empty and target in F_t.index:
                    c_bases.append(F_t.loc[target, cols].to_numpy(dtype=float))
                    c_tgts.append(target); c_w.append(w); c_kind.append(("oracle_d1", target))
            F_r = frames.get(run_day)
            t7 = target - 7 * DAY
            if not F_r.empty and t7 in F_r.index:
                c_bases.append(F_r.loc[t7, cols].to_numpy(dtype=float))
                c_tgts.append(target); c_w.append(w)
                c_kind.append(("t7_d1" if arm == "d1_19" else "t7_d2", target))
        if c_bases:
            CB = np.vstack(c_bases)
            preds = predict_rows(model, cols, CB, c_tgts, c_w)
            s = pd.DataFrame({"kind": [k for k, _ in c_kind], "target": [t for _, t in c_kind],
                              "pred": preds, "i": range(len(c_kind))})
            for kind, grp in s.groupby("kind"):
                g = grp.set_index("target")
                out[kind] = g["pred"]
                if kind == "oracle_d1":
                    for k, j in hist_idx.items():
                        out[f"oracle_{k}"] = pd.Series(CB[g["i"].to_numpy(), j], index=g.index)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(a: pd.Series, f: pd.Series) -> dict:
    ok = a.notna() & f.notna()
    a, f = a[ok], f[ok]
    if len(a) == 0:
        return {"n": 0, "wape": None, "bias": None, "relbias": None}
    return {
        "n": int(len(a)),
        "wape": float((a - f).abs().sum() / a.abs().sum() * 100),
        # Dashboard convention: bias = actual - forecast (positive = under-forecast).
        "bias": float((a - f).mean()),
        "relbias": float((a - f).sum() / a.sum() * 100),
    }


def week_label(day: pd.Timestamp, first: date) -> str:
    return f"w{int((day.date() - first).days // 7)}"


def argmin_k(panel: pd.DataFrame, arm: str, ks=range(0, 8)) -> dict:
    res = {}
    for k in ks:
        res[k] = score(panel["actual" if k == 0 else f"d{k}"], panel[arm])["wape"]
    valid = {k: v for k, v in res.items() if v is not None}
    return {"wape_by_k": res, "argmin_k": min(valid, key=valid.get) if valid else None}


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def displacement_test(panel: pd.DataFrame, first: date) -> dict:
    """What the mis-anchored history block cost, against how far it sat from
    the aligned block, per target day.

    cost(T)   = (sum oracle - sum served) / sum actual * 100
                -- the served relbias minus the oracle relbias: same artifact,
                same weather, same calendar, only the history block differs.
    disp_x(T) = (sum served_x - sum aligned_x) / sum actual * 100
                for x in the history columns (lag1, lag7, roll24, roll168).

    If the week-2 jump is the same defect meeting a bigger swing, week 2's
    cost is predicted by week 2's displacement under week 1's slope.
    """
    need = ["actual", "d1_19", "oracle_d1", "d1_19_roll24", "oracle_roll24"]
    if not set(need) <= set(panel.columns):
        return {}
    p = panel.dropna(subset=need).copy()
    if p.empty:
        return {}
    p["day"] = p.index.normalize()
    agg = {"a": ("actual", "sum"), "s": ("d1_19", "sum"), "o": ("oracle_d1", "sum"), "n": ("actual", "size")}
    for k in HISTORY:
        if f"d1_19_{k}" in p and f"oracle_{k}" in p:
            agg[f"s_{k}"] = (f"d1_19_{k}", "sum")
            agg[f"o_{k}"] = (f"oracle_{k}", "sum")
    daily = p.groupby("day").agg(**agg)
    daily = daily[daily["n"] >= 20]
    daily["cost"] = (daily["o"] - daily["s"]) / daily["a"] * 100
    daily["served_relbias"] = (daily["a"] - daily["s"]) / daily["a"] * 100
    daily["oracle_relbias"] = (daily["a"] - daily["o"]) / daily["a"] * 100
    ks = [k for k in HISTORY if f"s_{k}" in daily]
    for k in ks:
        daily[f"disp_{k}"] = (daily[f"s_{k}"] - daily[f"o_{k}"]) / daily["a"] * 100
    daily["week"] = [week_label(d, first) for d in daily.index]
    out = {"by_week": {}, "days": []}
    for wk, grp in daily.groupby("week"):
        out["by_week"][wk] = {
            "n_days": int(len(grp)),
            "mean_abs_served_relbias": float(grp["served_relbias"].abs().mean()),
            "mean_abs_oracle_relbias": float(grp["oracle_relbias"].abs().mean()),
            "mean_abs_cost": float(grp["cost"].abs().mean()),
            **{f"mean_abs_disp_{k}": float(grp[f"disp_{k}"].abs().mean()) for k in ks},
            **{f"corr_cost_disp_{k}": pearson(grp[f"disp_{k}"], grp["cost"]) for k in ks},
        }
    w12 = daily[daily["week"].isin(["w1", "w2"])]
    out["corr_cost_disp_w1_w2"] = {k: pearson(w12[f"disp_{k}"], w12["cost"]) for k in ks}
    w1, w2 = daily[daily["week"] == "w1"], daily[daily["week"] == "w2"]
    out["w1_slope_read_on_w2"] = {}
    for k in ks:
        den = float((w1[f"disp_{k}"] ** 2).sum())
        if len(w1) >= 3 and len(w2) >= 3 and den > 0:
            slope = float((w1[f"disp_{k}"] * w1["cost"]).sum() / den)
            out["w1_slope_read_on_w2"][k] = {
                "slope": slope,
                "w2_mean_abs_cost_predicted": float((slope * w2[f"disp_{k}"]).abs().mean()),
                "w2_mean_abs_cost_observed": float(w2["cost"].abs().mean()),
            }
    out["days"] = [
        {"day": d.date().isoformat(), "week": r["week"], "dow": d.strftime("%a"),
         "served_relbias": round(float(r["served_relbias"]), 3),
         "oracle_relbias": round(float(r["oracle_relbias"]), 3),
         "cost": round(float(r["cost"]), 3),
         **{f"disp_{k}": round(float(r[f"disp_{k}"]), 3) for k in ks}}
        for d, r in daily.iterrows()
    ]
    return out


def weekday_relbias(panel: pd.DataFrame, arms: List[str]) -> dict:
    out = {}
    for wk, grp in panel.groupby("week"):
        out[wk] = {}
        for dow, g in grp.groupby(grp.index.dayofweek):
            out[wk][DOW[dow]] = {arm: score(g["actual"], g[arm])["relbias"] for arm in arms if arm in g}
    return out


def weekly_shape(truth: pd.Series, start: date, end: date) -> List[dict]:
    """Truth-only, per Monday-start week: weekday/weekend contrast, week-on-week
    drift and the two-day swing. No forecast enters, so none can be fitted."""
    s = truth[(truth.index >= pd.Timestamp(start)) & (truth.index < pd.Timestamp(end))]
    daily = s.resample("D").mean()
    rows = []
    for wk_start in pd.date_range(pd.Timestamp(start), pd.Timestamp(end) - DAY, freq="W-MON"):
        wk = s[(s.index >= wk_start) & (s.index < wk_start + 7 * DAY)]
        if len(wk) < 24 * 6:
            continue
        mean = wk.mean()
        wkday = wk[wk.index.dayofweek < 5].mean()
        wkend = wk[wk.index.dayofweek >= 5].mean()
        prev = truth.reindex(wk.index - 7 * DAY).to_numpy()
        diff = np.abs(wk.to_numpy() - prev)
        dd = daily[(daily.index >= wk_start) & (daily.index < wk_start + 7 * DAY)]
        sw2 = (dd - daily.reindex(dd.index - 2 * DAY).to_numpy()).abs() / dd * 100
        rows.append({
            "week_start": wk_start.date().isoformat(),
            "weekday_weekend_contrast_pct": float((wkday - wkend) / mean * 100),
            "week_on_week_drift_pct": (float(np.nanmean(diff) / mean * 100)
                                       if np.isfinite(diff).any() else None),
            "mean_abs_two_day_swing_pct": float(sw2.mean()),
            "mean_load_mw": float(mean),
        })
    return rows


def rounded(obj, nd: int = 4):
    """Round floats for the machine record; 4 decimals is far below any quoted precision."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, nd) for v in obj]
    return obj


def fmt_pct(v) -> str:
    return "  n/a" if v is None else f"{v:5.2f}"


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replica-db", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--rows-out", default=None, help="optional per-hour CSV (scratch)")
    ap.add_argument("--countries", default=",".join(config.SUPPORTED_COUNTRIES))
    ap.add_argument("--first-target", default="2026-08-24")
    ap.add_argument("--last-target", default="2026-09-13")
    ap.add_argument("--shape-start", default="2026-07-06")
    args = ap.parse_args()

    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        raise SystemExit(f"replica not found: {replica}")
    config.DATABASE_PATH = replica
    if src_db.config is not config:
        raise SystemExit("src.db resolved a different config module; reads would not be pinned")

    first = date.fromisoformat(args.first_target)
    last = date.fromisoformat(args.last_target)
    shape_start = date.fromisoformat(args.shape_start)
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    report = {
        "meta": {
            "issue": "ABL-794",
            "replica_db": str(replica),
            "opened_readonly_via": "src.db.get_connection(readonly=True)",
            "tables_read": ["forecasts", "energy_load", "weather_data"],
            "models_dir": str(Path(args.models_dir).resolve()),
            "python": platform.python_version(),
            "first_target": first.isoformat(), "last_target": last.isoformat(),
            "weeks": {f"w{i}": [(first + timedelta(days=7 * i)).isoformat(),
                                (first + timedelta(days=7 * i + 6)).isoformat()]
                      for i in range((last - first).days // 7 + 1)},
            "match_tol_mw": MATCH_TOL, "near_tol_share": NEAR_TOL_SHARE,
            "id_separation": ID_SEPARATION,
            "not_evaluable": sorted(NOT_EVALUABLE),
            "flagged": FLAGGED,
        },
        "countries": {},
    }
    all_rows = []
    for cc in countries:
        served = load_served(cc, first, last)
        if served.empty:
            report["countries"][cc] = {"skipped": "no served load rows"}
            continue
        arms = served_arms(served)
        arms = arms[(arms.index >= pd.Timestamp(first)) & (arms.index < pd.Timestamp(last) + DAY)]
        truth = load_truth(cc, min(shape_start, first - timedelta(days=15)), last + timedelta(days=1))

        versions = sorted(set(served["model_version"]))
        art_path = Path(args.models_dir) / cc / "load" / "model.joblib"
        artifact = joblib.load(art_path) if art_path.exists() else None
        local_version = artifact.get("model_version") if artifact else None
        replayable = artifact is not None and versions == [local_version]
        entry = {"served_versions": versions, "served_model_names": sorted(set(served["model_name"])),
                 "local_version": local_version, "replayable": replayable}

        panel = arms.copy()
        panel["actual"] = truth.reindex(panel.index)
        for k in list(range(1, 8)) + [14]:
            panel[f"d{k}"] = truth.reindex(panel.index - k * DAY).to_numpy()

        if replayable:
            frames = Frames(cc)
            runs = load_weather_runs(cc, first, last)
            panel = panel.join(replay_country(arms, artifact, runs, frames))
            fh = frames.health
            entry["frame_health"] = {
                "frames": len(fh),
                "min_hours_in_frame": min(v["hours_in_frame"] for v in fh.values()),
                "hours_expected": LOOKBACK_DAYS * 24 + 24,
                "zero_target_hours_max": max(v["zero_target_hours"] for v in fh.values()),
                "duplicate_hours_total": sum(v["duplicate_hours"] for v in fh.values()),
            }
        else:
            entry["replay_skipped"] = ("no local artifact" if artifact is None else
                                       f"served {versions} != local {local_version}")

        panel["day"] = panel.index.normalize()
        panel["week"] = [week_label(d, first) for d in panel["day"]]

        # api_d2 composition: which run each dashboard "D+2" hour really came from.
        comp = panel.dropna(subset=["api_d2"]).groupby(["api_d2_g", "api_d2_slot"]).size()
        entry["api_d2_source_runs"] = {f"g{int(g)}@{s}": int(n) for (g, s), n in comp.items()}
        both = panel.dropna(subset=["api_d2", "api_d1"])
        entry["api_d2_identical_to_api_d1_share"] = float((both["api_d2"] - both["api_d1"]).abs().lt(1e-6).mean())
        entry["api_d1_equals_d1_19_share"] = float(
            (panel["api_d1"] - panel["d1_19"]).abs().lt(1e-6)[panel["api_d1"].notna()].mean())

        arm_list = ["api_d1", "api_d2", "d1_19", "d2_19", "d7"]
        if replayable:
            arm_list += ["oracle_d1", "t7_d1", "t7_d2"]
        entry["weekly"] = {}
        for wk, grp in panel.groupby("week"):
            entry["weekly"][wk] = {arm: score(grp["actual"], grp[arm]) for arm in arm_list if arm in grp}
            if replayable and {"oracle_d1", "t7_d1"} <= set(grp.columns):
                common = grp.dropna(subset=["actual", "d1_19", "oracle_d1", "t7_d1", "d7"])
                entry["weekly"][wk]["paired_d1"] = {
                    arm: score(common["actual"], common[arm]) for arm in ("d1_19", "oracle_d1", "t7_d1", "d7")}
        entry["daily"] = {}
        for day, grp in panel.groupby("day"):
            d = {"dow": day.strftime("%a"),
                 **{arm: score(grp["actual"], grp[arm]) for arm in arm_list if arm in grp}}
            if replayable and "d1_19_anchor_gap" in grp:
                ex = grp[grp["d1_19_resid"] <= MATCH_TOL]
                d["d1_19_anchor_gap_counts_exact"] = {str(int(k)): int(v) for k, v in
                                                      ex["d1_19_anchor_gap"].value_counts().sort_index().items()}
                d["mean_temp_c_forecast"] = (float(grp["d1_19_temp_c"].mean())
                                             if grp["d1_19_temp_c"].notna().any() else None)
            entry["daily"][day.date().isoformat()] = d
        entry["argmin_k"] = {wk: {arm: argmin_k(grp, arm) for arm in ("d1_19", "d2_19")}
                             for wk, grp in panel.groupby("week")}
        entry["weekday_relbias"] = weekday_relbias(panel, [a for a in ("d1_19", "oracle_d1", "d7") if a in panel])

        if replayable:
            m = {}
            for arm in ("d1_19", "d2_19"):
                col = f"{arm}_resid"
                if col not in panel:
                    continue
                for wk, grp in panel.dropna(subset=[col]).groupby("week"):
                    exact = grp[col] <= MATCH_TOL
                    ru = grp[f"{arm}_runner_up_resid"].fillna(np.inf)
                    near = (~exact & (grp[col] <= NEAR_TOL_SHARE * grp[arm].abs())
                            & (ru >= ID_SEPARATION * grp[col]))
                    w_age = [(ri - pd.Timestamp(wr)) / HOUR
                             for ri, wr in zip(grp[f"{arm}_gen"], grp[f"{arm}_w_run"])
                             if isinstance(wr, str)]
                    gap_counts = lambda mask: {str(int(k)): int(v) for k, v in
                                               grp.loc[mask, f"{arm}_anchor_gap"].value_counts().sort_index().items()}
                    m.setdefault(arm, {})[wk] = {
                        "n": int(len(grp)),
                        "exact_share": float(exact.mean()),
                        "near_identified_share": float(near.mean()),
                        "unidentified_share": float((~exact & ~near).mean()),
                        "resid_p50": float(grp[col].median()),
                        "resid_p95": float(grp[col].quantile(0.95)),
                        "anchor_gap_counts_exact": gap_counts(exact),
                        "anchor_gap_counts_near": gap_counts(near),
                        "anchor_gap_zero_anywhere": int((grp[f"{arm}_anchor_gap"] == 0).sum()),
                        "weather_rank_counts_exact": {str(int(k)): int(v) for k, v in
                                                      grp.loc[exact, f"{arm}_w_rank"].value_counts().sort_index().items()},
                        "weather_age_hours_median": float(np.median(w_age)) if w_age else None,
                    }
                ex_all = panel[panel[col] <= MATCH_TOL]
                if len(ex_all):
                    ct = pd.crosstab(ex_all.index.hour, ex_all[f"{arm}_anchor_gap"])
                    m[arm]["anchor_gap_by_target_hour_exact"] = {
                        int(h): {str(int(g)): int(v) for g, v in r.items() if v} for h, r in ct.iterrows()}
            entry["match"] = m
            entry["displacement_test_d1"] = displacement_test(panel, first)

        entry["weekly_shape"] = weekly_shape(truth, shape_start, last + timedelta(days=1))
        report["countries"][cc] = entry

        if args.rows_out:
            keep = panel.drop(columns=["day"]).copy()
            keep.insert(0, "country", cc)
            all_rows.append(keep)
        wk = entry["weekly"]
        g = lambda arm, k: fmt_pct(wk.get(k, {}).get(arm, {}).get("wape"))
        msg = f"{cc}: d1_19 {g('d1_19','w1')}->{g('d1_19','w2')}  d7 {g('d7','w1')}->{g('d7','w2')}"
        if replayable:
            ex = entry["match"].get("d1_19", {}).get("w2", {})
            msg += (f"  oracle {g('oracle_d1','w1')}->{g('oracle_d1','w2')}  t7 {g('t7_d1','w1')}->{g('t7_d1','w2')}"
                    f"  exact/near(d1,w2) {ex.get('exact_share', 0):.2f}/{ex.get('near_identified_share', 0):.2f}")
        else:
            msg += f"  [{entry['replay_skipped']}]"
        print(msg, flush=True)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(rounded(report), default=str), encoding="utf-8")
    if args.rows_out and all_rows:
        pd.concat(all_rows).to_csv(args.rows_out)
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
