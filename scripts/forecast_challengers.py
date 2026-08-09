#!/usr/bin/env python3
"""Run the registered challengers in shadow beside the champion (ABL-68).

Same daily job, same serve-time inputs, distinct `model_name` rows in the
sidecar, never pushed to production. The champion runs first
(`forecast_chronos2.py --experiment V010`); this runs after it and, for
correction-layer challengers, reads the vintage the champion just wrote.

Nothing here touches the replica: writes go through `src.db.get_connection`,
which routes every write connection to `FORECAST_OUTPUT_DB`. Production safety
is enforced on the other side too — `push_net_position_forecast.py` names the
champion explicitly and filters on it, so a challenger row cannot be shipped
even if it is the newest vintage in the sidecar.

Usage:
    python scripts/forecast_challengers.py --experiments V012,V016 --save-to-db
    python scripts/forecast_challengers.py --experiments V012 --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.challengers.baseline import forecast_baseline_ensemble
from src.challengers.correction import (CountryCorrection, apply_correction,
                                        latest_residual)
from src.challengers.registry import CHAMPION_MODEL_NAME, spec_for
from src.challengers.v014 import load_model as load_v014_model
from src.challengers.v014_features import ServeWindow, build_cache, build_features
from src.db import get_connection
from src.evaluation.net_position import _parse_ts, _ro_connect, as_of_for_vintage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("challengers")

FORECAST_TYPE = "net_position"
EXCLUDED_COUNTRIES = ("LU", "GR")   # duplicate of DE; actuals outage (ABL-31)


# ---------------------------------------------------------------------------
# Reading serve-time inputs
# ---------------------------------------------------------------------------

def load_actuals(replica_db: str) -> dict[str, pd.Series]:
    con = _ro_connect(replica_db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, timestamp_utc, net_position_mw FROM net_position
               WHERE net_position_mw IS NOT NULL AND timestamp_utc >= date('now', '-120 days')""",
            con)
    finally:
        con.close()
    if df.empty:
        return {}
    df["ts"] = _parse_ts(df["timestamp_utc"]).dt.floor("h")
    df = df.sort_values("ts").groupby(["country_code", "ts"]).tail(1)
    return {c: g.set_index("ts")["net_position_mw"].sort_index()
            for c, g in df.groupby("country_code")}


def load_champion_vintage(sidecar_db: str, target_date: str
                          ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """The champion's newest forecast for `target_date`, and when it was made."""
    con = _ro_connect(sidecar_db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, forecast_value
               FROM forecasts WHERE forecast_type = ? AND model_name = ?""",
            con, params=(FORECAST_TYPE, CHAMPION_MODEL_NAME))
        q = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, quantile,
                      forecast_value FROM forecast_quantiles
               WHERE forecast_type = ? AND model_name = ?""",
            con, params=(FORECAST_TYPE, CHAMPION_MODEL_NAME))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(), None
    df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
    df["generated_at"] = _parse_ts(df["generated_at"])
    day = pd.Timestamp(target_date)
    day_rows = df[(df["target_ts"] >= day) & (df["target_ts"] < day + pd.Timedelta(days=1))]
    if day_rows.empty:
        return pd.DataFrame(), None
    vintage = day_rows["generated_at"].max()
    out = day_rows[day_rows["generated_at"] == vintage].copy()
    if not q.empty:
        q["target_ts"] = _parse_ts(q["target_timestamp_utc"])
        q["generated_at"] = _parse_ts(q["generated_at"])
        q = q[q["generated_at"] == vintage]
        wide = q.pivot_table(index=["country_code", "target_ts"],
                             columns="quantile", values="forecast_value")
        wide.columns = [f"q{int(round(c * 100))}" for c in wide.columns]
        out = out.merge(wide.reset_index(), on=["country_code", "target_ts"], how="left")
    return out, vintage


def champion_history(sidecar_db: str, actuals: dict[str, pd.Series]) -> pd.DataFrame:
    """Champion forecasts for hours that have since been measured — the source
    of the AR(1) term's most recent observable residual."""
    con = _ro_connect(sidecar_db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, forecast_value
               FROM forecasts WHERE forecast_type = ? AND model_name = ?""",
            con, params=(FORECAST_TYPE, CHAMPION_MODEL_NAME))
    finally:
        con.close()
    if df.empty:
        return df
    df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
    df["generated_at"] = _parse_ts(df["generated_at"])
    # Latest vintage per target hour: the freshest view of each past hour.
    df = (df.sort_values("generated_at")
            .drop_duplicates(["country_code", "target_ts"], keep="last"))
    df["actual"] = [
        actuals.get(cc, pd.Series(dtype=float)).get(ts, np.nan)
        for cc, ts in zip(df["country_code"], df["target_ts"])]
    return df


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save(rows: list[dict], quantile_rows: list[dict]) -> tuple[int, int]:
    with get_connection(readonly=False) as conn:
        cur = conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO forecasts (country_code, forecast_type,
               target_timestamp_utc, generated_at, horizon_hours, forecast_value,
               model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""",
            [(r["country_code"], FORECAST_TYPE, str(r["target_ts"]),
              str(r["generated_at"]), r["horizon_hours"], float(r["forecast_value"]),
              r["model_name"], r["model_version"]) for r in rows])
        cur.executemany(
            """INSERT OR REPLACE INTO forecast_quantiles (country_code, forecast_type,
               target_timestamp_utc, generated_at, quantile, forecast_value, model_name)
               VALUES (?,?,?,?,?,?,?)""",
            [(r["country_code"], FORECAST_TYPE, str(r["target_ts"]),
              str(r["generated_at"]), float(r["quantile"]), float(r["forecast_value"]),
              r["model_name"]) for r in quantile_rows])
    return len(rows), len(quantile_rows)


def _row(cc, ts, value, generated_at, model_name, version):
    return {"country_code": cc, "target_ts": ts, "forecast_value": value,
            "generated_at": generated_at, "model_name": model_name,
            "model_version": version,
            "horizon_hours": max(1, int((ts - generated_at).total_seconds() // 3600))}


# ---------------------------------------------------------------------------
# Challengers
# ---------------------------------------------------------------------------

def run_v012(spec, countries, target_date, generated_at, actuals, **_):
    """Persistence + hour-of-day climatology on serve-faithful actuals."""
    as_of = as_of_for_vintage(generated_at)
    targets = pd.date_range(pd.Timestamp(target_date), periods=24, freq="h")
    version = generated_at.strftime("%Y%m%d_%H%M%S")
    rows, skipped = [], []
    for cc in countries:
        series = actuals.get(cc, pd.Series(dtype=float))
        preds = forecast_baseline_ensemble(series, as_of, targets)
        usable = preds.dropna()
        if usable.empty:
            skipped.append(cc)
            continue
        # NaN hours are dropped, never written as 0.0: a 0 MW net position is a
        # real, balanced-border reading, not a stand-in for "unknown".
        rows += [_row(cc, ts, float(v), generated_at, spec.model_name, version)
                 for ts, v in usable.items()]
    if skipped:
        logger.warning("V012: no baseline for %s (no actuals before %s)",
                       ",".join(skipped), as_of)
    return rows, []


def run_v016(spec, countries, target_date, generated_at, actuals,
             sidecar_db=None, **_):
    """Affine recalibration + AR(1) applied to the champion's current vintage."""
    fit_path = config.EXPERIMENTS_DIR / spec.experiment_id / "correction.json"
    if not fit_path.exists():
        logger.error("V016: no fit at %s - refusing to serve. Run "
                     "scripts/fit_v016_correction.py first. Serving V016 without "
                     "a fit would just republish V010 under a second name.",
                     fit_path)
        return [], []
    doc = json.loads(fit_path.read_text())
    fits = {cc: CountryCorrection(**c) for cc, c in doc["corrections"].items()}

    champion, champ_vintage = load_champion_vintage(sidecar_db, target_date)
    if champion.empty:
        logger.error("V016: no %s vintage for target %s in the sidecar - the "
                     "champion must run first.", CHAMPION_MODEL_NAME, target_date)
        return [], []
    logger.info("V016: correcting champion vintage %s", champ_vintage)

    as_of = as_of_for_vintage(generated_at)
    history = champion_history(sidecar_db, actuals)
    version = generated_at.strftime("%Y%m%d_%H%M%S")
    q_levels = [c for c in champion.columns if c.startswith("q")]

    rows, qrows, identity = [], [], []
    for cc in countries:
        g = champion[champion["country_code"] == cc].sort_values("target_ts")
        if g.empty:
            continue
        fit = fits.get(cc)
        if fit is None:
            identity.append(f"{cc} (not in fit)")
            continue
        if fit.is_identity:
            identity.append(f"{cc} ({fit.reason})")
        resid, resid_ts = latest_residual(
            history[history["country_code"] == cc], as_of, fit)
        targets = pd.DatetimeIndex(g["target_ts"])
        corrected = apply_correction(g["forecast_value"].to_numpy(), targets,
                                     fit, resid, resid_ts)
        rows += [_row(cc, ts, float(v), generated_at, spec.model_name, version)
                 for ts, v in zip(targets, corrected)]
        # Quantiles get the same affine map plus the same AR shift. The map is
        # monotone (slope > 0 is a fit guard), so the band stays ordered.
        for qcol in q_levels:
            if g[qcol].isna().all():
                continue
            shifted = apply_correction(g[qcol].to_numpy(), targets, fit, resid, resid_ts)
            level = int(qcol[1:]) / 100.0
            qrows += [{"country_code": cc, "target_ts": ts, "quantile": level,
                       "forecast_value": float(v), "generated_at": generated_at,
                       "model_name": spec.model_name}
                      for ts, v in zip(targets, shifted) if np.isfinite(v)]
    if identity:
        logger.info("V016: passing through uncorrected (V016 == V010 here): %s",
                    "; ".join(identity))
    return rows, qrows


def run_v014(spec, countries, target_date, generated_at, actuals,
             replica_db=None, models_dir=None, **_):
    """Per-country XGBoost on serve-faithful features read from the replica.

    Unlike V012 and V016 this one does not consume `actuals` or the champion's
    vintage — it reads its own features straight from the replica, bounded by
    the serve window for this target day. That window is derived from the
    *target date*, not from `generated_at`, so a run that fires late still gets
    the cutoffs the schedule promises rather than the extra hours the clock
    happened to hand it. A late run must not be a better-informed run: the
    backtest and the training frame both assume the 06:00Z cutoffs, and a
    vintage built on more than that would be scored as if it had been built on
    the same information as the rest.
    """
    window = ServeWindow.for_target_day(target_date)
    if window.run_ts > pd.Timestamp(generated_at):
        logger.error("V014: target %s implies a run at %s, which is after this "
                     "run (%s). Refusing - the features would reach past what "
                     "exists.", target_date, window.run_ts, generated_at)
        return [], []

    version = pd.Timestamp(generated_at).strftime("%Y%m%d_%H%M%S")
    models_dir = Path(models_dir or config.MODELS_DIR)
    conn = _ro_connect(replica_db)
    rows, no_model, refused = [], [], []
    try:
        for cc in countries:
            try:
                model = load_v014_model(models_dir, cc)
            except FileNotFoundError:
                no_model.append(cc)
                continue
            cache = build_cache(conn, cc,
                                window.day_ahead_cutoff - pd.Timedelta(days=35),
                                window.target_index.max())
            preds = model.predict_frame(
                build_features(cache, window, neighbours=model.neighbours))
            usable = preds.dropna()
            if usable.empty:
                refused.append(cc)
                continue
            # A refused hour is dropped, never written as 0.0 — the same rule
            # V012 follows, for the same reason: 0 MW is a real balanced-border
            # reading, not a stand-in for "unknown".
            rows += [_row(cc, ts, float(v), generated_at, spec.model_name, version)
                     for ts, v in usable.items()]
    finally:
        conn.close()
    if no_model:
        logger.warning("V014: no trained model for %s - run scripts/train_v014.py",
                       ",".join(no_model))
    if refused:
        logger.warning("V014: refused %s - no anchor observation at the serve "
                       "cutoff", ",".join(refused))
    return rows, []


RUNNERS = {"V012": run_v012, "V014": run_v014, "V016": run_v016}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default="V012,V014,V016")
    p.add_argument("--countries", default="all")
    p.add_argument("--target-date", default=None, help="default: D+2 from today")
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--sidecar-db", default=config.FORECAST_OUTPUT_DB)
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--save-to-db", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.sidecar_db:
        logger.error("FORECAST_OUTPUT_DB (or --sidecar-db) is required: "
                     "challengers write to the sidecar, never the replica.")
        return 2

    target_date = args.target_date or (date.today() + timedelta(days=2)).isoformat()
    if args.countries == "all":
        countries = [c for c in config.SUPPORTED_COUNTRIES if c not in EXCLUDED_COUNTRIES]
    else:
        countries = [c for c in args.countries.split(",") if c not in EXCLUDED_COUNTRIES]

    # Naive UTC, matching the shape forecast_chronos2.py stores for the
    # champion. It is emphatically *not* the same instant: this is a separate
    # process and stamps its own clock, measured 3.8-12.3 s after the
    # champion's and truncated to the second where the champion carries
    # microseconds. Nothing may compare the two vintages by equality on this
    # column — the head-to-head pairs on the actuals cutoff a vintage could see
    # instead (`src/evaluation/head_to_head.py`, ABL-82). Stamped once here, so
    # every experiment in one invocation shares a vintage.
    generated_at = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    actuals = load_actuals(args.replica_db)
    logger.info("shadow run: target %s, %d countries, generated_at %s",
                target_date, len(countries), generated_at)

    save_rows = args.save_to_db and not args.dry_run
    exit_code = 0
    for exp in args.experiments.split(","):
        exp = exp.strip()
        if not exp:
            continue
        try:
            spec = spec_for(exp)
        except KeyError as exc:
            logger.error("%s", exc)
            exit_code = 1
            continue
        rows, qrows = RUNNERS[exp](
            spec, countries, target_date, pd.Timestamp(generated_at), actuals,
            sidecar_db=args.sidecar_db, replica_db=args.replica_db,
            models_dir=args.models_dir)
        if not rows:
            logger.error("%s (%s): produced nothing", exp, spec.model_name)
            exit_code = 1
            continue
        vals = np.array([r["forecast_value"] for r in rows])
        logger.info("%s (%s): %d points over %d countries, range [%.1f, %.1f] MW",
                    exp, spec.model_name, len(rows),
                    len({r["country_code"] for r in rows}), vals.min(), vals.max())
        if save_rows:
            n, nq = save(rows, qrows)
            logger.info("  saved %d point + %d quantile rows to the sidecar", n, nq)
        else:
            logger.info("  [dry run] not saved")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
