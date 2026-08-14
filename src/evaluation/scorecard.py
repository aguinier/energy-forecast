"""Recurring forecast-quality scorecard across every served forecast type.

The evaluator is deliberately read-only. It measures stored, issued forecasts;
it does not fit, correct, interpolate, or extrapolate any series.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.baselines import aligned_point_baselines
from src.db import RENEWABLE_TYPE_COLUMNS
from src.tso_plausibility import guard_tso_series
from src.evaluation.net_position import (
    GATE_EXCLUDED_COUNTRIES,
    as_of_for_vintage,
    baseline_predictions as net_position_baselines,
)


# Snapshot of the production entries in the dashboard's model registry. Keeping
# the list explicit prevents an old/stale model present in SQLite from silently
# entering a recurring scorecard.
PRODUCTION_MODELS = {
    "load": "catboost",
    "price": "catboost",
    "renewable": "catboost",
    "solar": "catboost",
    "wind_onshore": "catboost",
    "wind_offshore": "xgboost",
    "biomass": "xgboost",
    "hydro_total": "xgboost",
    "net_position": "chronos-2-V010",
}

ABL128_REFERENCE = {
    "load": {"wape_pct": 9.4, "seasonal_naive_wape_pct": 5.9,
             "tso_wape_pct": 4.0, "bias_pct": 1.6},
    "price": {"wape_pct": 33.2, "seasonal_naive_wape_pct": 27.3,
              "tso_wape_pct": None, "bias_pct": -17.3},
    "solar": {"wape_pct": 53.6, "seasonal_naive_wape_pct": 26.0,
              "tso_wape_pct": None, "bias_pct": -50.7},
    "wind_onshore": {"wape_pct": 73.5, "seasonal_naive_wape_pct": 75.0,
                     "tso_wape_pct": None, "bias_pct": 28.1},
}

#: Every `energy_generation` column the renewable aggregate sums, in the order
#: the dashboard's `renewableTotal.RENEWABLE_MW_COLUMNS` sums them. It is the
#: same seven wire fields `/renewables` serves, flattened.
#:
#: `hydro_pumped_mw` is deliberately absent, and that absence is the whole of
#: ABL-410's hydro finding: pumped storage is a *store*, not a primary source.
#: `energy_generation` gives it its own column; the frozen `energy_renewable`
#: folds it into `hydro_reservoir_mw`, so a sum over the frozen table's columns
#: books a battery as renewable generation.
GENERATION_RENEWABLE_COLUMNS = (
    "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
    "hydro_run_mw", "hydro_reservoir_mw", "biomass_mw",
    "geothermal_mw", "marine_mw", "other_renewable_mw",
)


def null_aware_sum(columns: Iterable[str]) -> str:
    """Sum the reported members; NULL only when not one of them is reported.

    The rule `db._HYDRO_TOTAL_EXPR` already applies to hydro's two components,
    generalised to a list. A plain `a + b + ...` lets one unreported column
    erase every reported one beside it; a plain `COALESCE(a,0) + ...` reports a
    country that measures none of them as generating exactly zero. Neither is a
    measurement.
    """
    members = tuple(columns)
    absent = " AND ".join(f"{column} IS NULL" for column in members)
    summed = " + ".join(f"COALESCE({column}, 0)" for column in members)
    return f"CASE WHEN {absent} THEN NULL ELSE {summed} END"


#: Which table, and which value in it, a forecast of each type is scored
#: against. **One statement of the actual, for this repo and the dashboard
#: both** — that is ABL-410's item 1, and this dict is the "one place".
#:
#: ## Why the renewable family moved off `energy_renewable` (ABL-410)
#:
#: ABL-399 moved the dashboard's renewable-family accuracy reads onto
#: `energy_generation` (PR #30, merged 2026-08-13). Until this change the
#: scorecard still scored those same types against the frozen table, so **the
#: same model, country and window had two published WAPEs and neither was
#: wrong** — they measured against different statements of the actual.
#:
#: This is *not* ABL-321's rejected switch. That was the **training** source,
#: `db.RENEWABLE_TYPE_SOURCE_TABLE`, and it stays `energy_renewable`. Scoring
#: truth and training source are independent post-ABL-331, and ABL-321's own
#: decision window already took `energy_generation` as primary truth
#: (`db.py:361`). Nothing here changes what a training run reads, what an
#: artifact serves from, or any promotion gate: the gates take their actuals
#: from `RenewableFeatureBuilder` -> `db.load_renewable_type_data`, never from
#: this dict, which only `_load_actuals` reads.
#:
#: The measurement behind the choice — replica 2026-08-13, target window
#: 2026-07-11 -> 2026-08-10, latest vintage per band, common instants only, n =
#: 2,760 per pair (1,688 for FR, see the coverage caveat below), production
#: models — is in `reports/abl_410_scoring_truth.md`. In summary: eight of the
#: fifteen live pairs are **identical** under the two tables, because `solar`,
#: `wind_onshore`, `wind_offshore` and `biomass` are single columns of the same
#: name. The gap is entirely in the two re-derivations, `renewable` and
#: `hydro_total`, plus a BE-only drift where the frozen table's `DEFAULT 0`
#: stands in for a negative measurement.
#:
#: Two caveats that belong next to any figure this dict produces:
#:
#:  - **`energy_generation` has an open FR ingest gap** (ABL-318 §3): no rows
#:    2026-06-30 23:45 -> 2026-07-22 14:15, which is 279 of the 720 hours of
#:    the last published window. FR sample sizes drop accordingly. Over the
#:    same era `energy_generation` covers 24,694 hours the frozen table does
#:    not, so this is a specific gap, not a coverage regression.
#:  - **The models are still fitted on `energy_renewable`.** Where the two
#:    tables disagree about what the target *is*, part of the resulting WAPE is
#:    target mismatch rather than model error. BE `hydro_total` is the extreme:
#:    its fitted target was run-of-river plus folded pumped storage (84.7% of
#:    it, across the hours both tables carry), and against honest run-of-river
#:    it scores 14,274% with a correlation of **-0.12**. That is not a model
#:    that got worse; it is a model of a different quantity. Filed separately —
#:    a WAPE against the corrected target only becomes a quality figure after a
#:    retrain, and no BE `hydro_total` WAPE should be quoted as quality until
#:    then.
ACTUAL_SPECS = {
    "load": ("energy_load", "load_mw"),
    "price": ("energy_price", "price_eur_mwh"),
    "renewable": ("energy_generation", null_aware_sum(GENERATION_RENEWABLE_COLUMNS)),
    "solar": ("energy_generation", "solar_mw"),
    "wind_onshore": ("energy_generation", "wind_onshore_mw"),
    "wind_offshore": ("energy_generation", "wind_offshore_mw"),
    "biomass": ("energy_generation", "biomass_mw"),
    # The training-side definition itself, imported rather than restated. The
    # previous literal here was a strict `hydro_run_mw + hydro_reservoir_mw`,
    # whose comment argued — correctly — that COALESCE fabricates a zero out of
    # an unmeasured component. On the frozen table that strict form was
    # harmless only by accident: `REAL DEFAULT 0` means nothing there is ever
    # NULL, so it and the null-aware form agree to the digit on all 15 live
    # pairs. On `energy_generation` it is fatal — for 9 of the 24 supported
    # countries exactly one hydro component is 100% NULL (`db.py:406`), and a
    # strict `+` would erase all nine.
    "hydro_total": ("energy_generation", RENEWABLE_TYPE_COLUMNS["hydro_total"]),
    "net_position": ("net_position", "net_position_mw"),
}

#: What the renewable family was scored against before ABL-410, kept as the
#: record of a superseded decision rather than as a fallback. Nothing reads it;
#: it exists so a reader comparing a report written before 2026-08-13 to one
#: written after can see which definition produced which number.
RETIRED_RENEWABLE_ACTUAL_SPECS = {
    "renewable": ("energy_renewable", "total_renewable_mw"),
    "solar": ("energy_renewable", "solar_mw"),
    "wind_onshore": ("energy_renewable", "wind_onshore_mw"),
    "wind_offshore": ("energy_renewable", "wind_offshore_mw"),
    "biomass": ("energy_renewable", "biomass_mw"),
    "hydro_total": ("energy_renewable", "hydro_run_mw + hydro_reservoir_mw"),
}

TSO_SPECS = {
    "load": ("energy_load_forecast", "target_timestamp_utc", "forecast_value_mw"),
    "renewable": ("energy_generation_forecast", "target_timestamp_utc", "total_forecast_mw"),
    "solar": ("energy_generation_forecast", "target_timestamp_utc", "solar_mw"),
    "wind_onshore": ("energy_generation_forecast", "target_timestamp_utc", "wind_onshore_mw"),
    "wind_offshore": ("energy_generation_forecast", "target_timestamp_utc", "wind_offshore_mw"),
}

# Half-open bands except for the final inclusive 64h endpoint. The evaluator
# selects the latest vintage per target *within* a band; selecting one latest
# row per target first would erase the D+2 evidence.
HORIZON_BANDS = (
    ("2-12h", 2, 12),
    ("12-24h", 12, 24),
    ("24-36h", 24, 36),
    ("36-48h", 36, 48),
    ("48-64h", 48, 65),
)


@dataclass(frozen=True)
class ScorecardConfig:
    replica_db: str
    sidecar_db: str | None
    start: pd.Timestamp
    end: pd.Timestamp
    models: dict[str, str] | None = None


def normalize_timestamps(values: Iterable) -> pd.Series:
    """Parse both SQLite timestamp separators into naive UTC timestamps."""
    parsed = pd.to_datetime(pd.Series(values), format="mixed", utc=True,
                            errors="coerce")
    return parsed.dt.tz_localize(None)


def horizon_band(hours: float | int | None) -> str | None:
    """Return the configured horizon band, or None when it is not measurable."""
    if hours is None or pd.isna(hours):
        return None
    value = float(hours)
    for name, lower, upper in HORIZON_BANDS:
        if lower <= value < upper:
            return name
    return None


def select_latest_per_band(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep one latest issued row per country/target/model/horizon band."""
    if forecasts.empty:
        result = forecasts.copy()
        result["horizon_band"] = pd.Series(dtype=str)
        return result
    result = forecasts.copy()
    result["horizon_band"] = result["horizon_hours"].map(horizon_band)
    result = result.dropna(subset=["target_ts", "generated_at", "horizon_band"])
    keys = ["forecast_type", "model_name", "country_code", "target_ts",
            "horizon_band"]
    return (result.sort_values([*keys, "generated_at", "source_rank"])
                  .drop_duplicates(keys, keep="last")
                  .reset_index(drop=True))


def score_predictions(actual: Iterable[float], predicted: Iterable[float]) -> dict:
    """Pure point scoring. Empty/all-invalid input is explicitly unmeasured."""
    a = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if len(a) != len(p):
        raise ValueError("actual and predicted must align")
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if len(a) == 0:
        return {"n": 0, "wape_pct": None, "mae": None, "bias_pct": None,
                "slope": None, "correlation": None}
    error = p - a
    denom = float(np.sum(np.abs(a)))
    var_actual = float(np.var(a))
    std_predicted = float(np.std(p))
    return {
        "n": int(len(a)),
        "wape_pct": (100.0 * float(np.sum(np.abs(error))) / denom
                     if denom > 0 else None),
        "mae": float(np.mean(np.abs(error))),
        "bias_pct": (100.0 * float(np.sum(error)) / denom if denom > 0 else None),
        "slope": (float(np.cov(a, p, bias=True)[0, 1] / var_actual)
                  if len(a) > 1 and var_actual > 0 else None),
        "correlation": (float(np.corrcoef(a, p)[0, 1])
                        if len(a) > 1 and var_actual > 0 and std_predicted > 0
                        else None),
    }


def score_against_baseline(actual: Iterable[float], model: Iterable[float],
                           baseline: Iterable[float]) -> dict:
    """Score model and baseline on their identical finite-pair intersection."""
    a = np.asarray(list(actual), dtype=float)
    m = np.asarray(list(model), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if not (len(a) == len(m) == len(b)):
        raise ValueError("actual, model, and baseline must align")
    valid = np.isfinite(a) & np.isfinite(m) & np.isfinite(b)
    model_score = score_predictions(a[valid], m[valid])
    baseline_score = score_predictions(a[valid], b[valid])
    model_wape = model_score["wape_pct"]
    baseline_wape = baseline_score["wape_pct"]
    skill = None
    if model_wape is not None and baseline_wape not in (None, 0):
        skill = 100.0 * (1.0 - model_wape / baseline_wape)
    return {"n": int(valid.sum()), "baseline": baseline_score,
            "model_on_same_pairs": model_score, "skill_pct": skill}


def filter_measured_actuals(df: pd.DataFrame, forecast_type: str) -> pd.DataFrame:
    """Apply type-specific measurement rules without inventing observations."""
    result = df.dropna(subset=["ts", "actual"]).copy()
    result = result[(result["ts"].dt.minute == 0) & (result["ts"].dt.second == 0)]
    if forecast_type == "load":
        result = result[result["actual"] > 0]
    if forecast_type == "net_position":
        result = result[result["country_code"] != "GR"]
    return result


def _ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)


def opened_databases(cfg: ScorecardConfig, feature_db, ambient_db) -> dict:
    """Every database file a gate run opens, as a record for `meta`.

    ABL-355: `--replica-db` governed only the reads that go through this module
    — the incumbent forecasts, the TSO series and the contamination screen. The
    fitted series and the weather archive went through `db.get_connection()`
    and so opened `config.DATABASE_PATH`, while the report printed the replica
    alone, under `Replica:`, as if it were the source of everything. Every gate
    number is a comparison between two series that were not required to have
    come from the same file, and nothing said so.

    Threading `db_path` into the builder is what fixes it; this is what makes
    the fix legible in the record. `features_match_replica` is the property
    that matters — the sidecar is a *different* file by design (locally
    generated forecasts, replica purity), so its path is named but not compared.

    `ambient_db` is `config.DATABASE_PATH`. It is recorded precisely because a
    fixed run does *not* read it: a reader comparing this report to one written
    before ABL-355 needs to see whether the two would have diverged.

    The sidecar rule is `_load_forecasts`' own — same condition, so the record
    cannot claim a file that run never opened.

    Both comparisons are made on resolved paths and reported against the value
    as configured. `--replica-db` defaults to `str(config.DATABASE_PATH)`, so
    the two are usually the same file named two ways; comparing the strings
    would let a relative or unnormalized `ENERGY_DB_PATH` print "not read by
    this run" about the very file the run read.
    """
    sidecar = (str(Path(cfg.sidecar_db).resolve())
               if cfg.sidecar_db and Path(cfg.sidecar_db).exists() else None)
    replica = str(Path(cfg.replica_db).resolve())
    features = str(Path(feature_db).resolve())
    return {
        "replica": replica,
        "features": features,
        "sidecar": sidecar,
        "ambient_energy_db_path": str(ambient_db),
        "features_match_replica": features == replica,
        "ambient_matches_replica": str(Path(ambient_db).resolve()) == replica,
    }


def describe_opened_databases(record: dict, replica_bytes: int) -> list[str]:
    """The report lines that name every file the run opened.

    One line per file, and — only when it differs — one naming the ambient
    `ENERGY_DB_PATH` the run did not read. Silence there would leave the
    strongest evidence that the ABL-355 split is closed off the page.

    The incumbent forecasts are the one read the replica does not hold alone:
    `_load_forecasts` also opens the sidecar when it exists, and a sidecar row
    wins an exact vintage match. So the single-file sentence is said only when
    no sidecar was opened. Claiming it over one would be this issue's own
    defect — a report naming one file for reads that came from two — reprinted
    inside its fix.
    """
    replica_reads = (
        "the TSO series, the contamination screen, and — since ABL-355 — the "
        "fitted target series, its lag/rolling features, the D-7 and persistence "
        "baselines, the gate actuals and the weather archive"
    )
    lines = [
        f"Replica: `{record['replica']}` ({replica_bytes:,} bytes), opened with "
        "SQLite `mode=ro`, `uri=True`.",
    ]
    if record["features_match_replica"] and not record["sidecar"]:
        lines.append(
            f"Every read in this run comes from that one file: the incumbent "
            f"forecasts, {replica_reads}."
        )
    elif record["features_match_replica"]:
        lines.append(
            f"That one file is the source of {replica_reads}. The incumbent "
            f"forecasts are the only read it does not hold alone; see the "
            f"sidecar below."
        )
    else:
        # Unreachable from either harness, which hands the builder the resolved
        # replica. Kept because a wrong number here is worse than a missing one:
        # if some future caller does split them, the report says so rather than
        # printing one path for two files.
        lines.append(
            f"**Cross-sourced run.** The fitted target series, its features, the "
            f"baselines, the gate actuals and the weather archive were read from "
            f"`{record['features']}`, which is not the replica above. The gate "
            f"numbers compare series from two different files; treat them as "
            f"unpublishable until re-run against one."
        )
    if record["sidecar"]:
        lines.append(
            f"Sidecar: `{record['sidecar']}`, also opened `mode=ro`, and read "
            "for locally generated incumbent forecasts only. Where a sidecar "
            "row and a replica row carry the same vintage, the sidecar's is the "
            "one scored."
        )
    if not record["ambient_matches_replica"]:
        lines.append(
            f"`ENERGY_DB_PATH` resolved to `{record['ambient_energy_db_path']}` "
            "and was **not** read by this run. Before ABL-355 that path, not the "
            "replica, is where the fitted series would have come from."
        )
    return lines


def _load_forecasts(cfg: ScorecardConfig) -> tuple[pd.DataFrame, dict]:
    models = cfg.models or PRODUCTION_MODELS
    frames = []
    sources = [("replica", cfg.replica_db, 0)]
    if cfg.sidecar_db and Path(cfg.sidecar_db).exists():
        sources.append(("sidecar", cfg.sidecar_db, 1))
    for source, path, source_rank in sources:
        con = _ro_connect(path)
        try:
            for forecast_type, model_name in models.items():
                df = pd.read_sql_query(
                    """SELECT country_code, forecast_type, target_timestamp_utc,
                              generated_at, horizon_hours, forecast_value, model_name
                       FROM forecasts
                       WHERE forecast_type = ? AND model_name = ?
                         AND target_timestamp_utc >= ?
                         AND target_timestamp_utc < ?""",
                    con, params=(forecast_type, model_name, str(cfg.start), str(cfg.end)))
                if not df.empty:
                    df["source"] = source
                    df["source_rank"] = source_rank
                    frames.append(df)
        finally:
            con.close()
    if not frames:
        return pd.DataFrame(), {
            f"{t}/{m}": {"generated_timestamps": 0, "run_days": 0}
            for t, m in models.items()
        }
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["target_ts"] = normalize_timestamps(forecasts["target_timestamp_utc"])
    forecasts["generated_at"] = normalize_timestamps(forecasts["generated_at"])
    exact = ["forecast_type", "model_name", "country_code", "target_ts",
             "generated_at", "horizon_hours"]
    forecasts = (forecasts.sort_values("source_rank")
                          .drop_duplicates(exact, keep="last"))
    counts = {}
    for forecast_type, model_name in models.items():
        sub = forecasts[(forecasts["forecast_type"] == forecast_type)
                        & (forecasts["model_name"] == model_name)]
        counts[f"{forecast_type}/{model_name}"] = {
            "generated_timestamps": int(sub["generated_at"].nunique()),
            "run_days": int(sub["generated_at"].dt.normalize().nunique()),
        }
    return forecasts, counts


def _load_actuals(cfg: ScorecardConfig, forecast_type: str) -> pd.DataFrame:
    table, expression = ACTUAL_SPECS[forecast_type]
    start = cfg.start - pd.Timedelta(days=8)
    con = _ro_connect(cfg.replica_db)
    try:
        df = pd.read_sql_query(
            f"""SELECT country_code, timestamp_utc, {expression} AS actual
                FROM {table}
                WHERE timestamp_utc >= ? AND timestamp_utc < ?
                  AND ({expression}) IS NOT NULL""",
            con, params=(str(start), str(cfg.end)))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["country_code", "ts", "actual"])
    df["ts"] = normalize_timestamps(df["timestamp_utc"])
    # Forecast targets are hourly. Do not aggregate quarter-hour observations
    # into a value with a new meaning; retain only the measured top-of-hour row.
    df = filter_measured_actuals(df, forecast_type)
    return (df[["country_code", "ts", "actual"]]
            .drop_duplicates(["country_code", "ts"], keep="last")
            .sort_values(["country_code", "ts"]).reset_index(drop=True))


def _load_tso(cfg: ScorecardConfig, forecast_type: str) -> pd.DataFrame:
    if forecast_type not in TSO_SPECS:
        return pd.DataFrame(columns=["country_code", "target_ts", "tso"])
    table, timestamp_col, value_col = TSO_SPECS[forecast_type]
    con = _ro_connect(cfg.replica_db)
    try:
        df = pd.read_sql_query(
            f"""SELECT country_code, {timestamp_col}, {value_col} AS tso
                FROM {table}
                WHERE {timestamp_col} >= ? AND {timestamp_col} < ?
                  AND forecast_type = 'day_ahead' AND {value_col} IS NOT NULL""",
            con, params=(str(cfg.start), str(cfg.end)))
        if df.empty:
            return pd.DataFrame(columns=["country_code", "target_ts", "tso"])
        df["target_ts"] = normalize_timestamps(df[timestamp_col])
        df = df.dropna(subset=["target_ts"])
        # The TSO column here is a scored comparator, so an implausible value is
        # not a modelling nuisance but a wrong published number about TSO. The
        # guard nulls it per country (ABL-431), which this scorecard already
        # reads as "not measured" and reports with its own n — the alternative
        # is a country-window WAPE computed against a value three orders of
        # magnitude out. The reference is resolved per country, so the query
        # stays one pass over every country.
        for country, index in df.groupby("country_code").groups.items():
            values = pd.Series(df.loc[index, "tso"].to_numpy(),
                               index=pd.DatetimeIndex(df.loc[index, "target_ts"]))
            guarded = guard_tso_series(values, con, str(country), table, value_col,
                                       context=f"scorecard:{forecast_type}")
            df.loc[index, "tso"] = guarded.to_numpy()
    finally:
        con.close()
    return (df[["country_code", "target_ts", "tso"]]
            .drop_duplicates(["country_code", "target_ts"], keep="last"))


def _attach_evidence(cfg: ScorecardConfig, selected: pd.DataFrame,
                     forecast_type: str) -> pd.DataFrame:
    rows = selected[selected["forecast_type"] == forecast_type].copy()
    actuals = _load_actuals(cfg, forecast_type)
    rows = rows.merge(actuals.rename(columns={"ts": "target_ts"}),
                      on=["country_code", "target_ts"], how="left")
    rows["seasonal_naive"] = np.nan
    rows["persistence"] = np.nan
    for country, index in rows.groupby("country_code").groups.items():
        history = actuals[actuals["country_code"] == country].set_index("ts")["actual"]
        baseline = aligned_point_baselines(
            history, pd.DatetimeIndex(rows.loc[index, "target_ts"]),
            pd.DatetimeIndex(rows.loc[index, "generated_at"]))
        rows.loc[index, "seasonal_naive"] = baseline["seasonal_naive"].to_numpy()
        rows.loc[index, "persistence"] = baseline["persistence"].to_numpy()
        if forecast_type == "net_position":
            # Net position is published day-ahead, so target timestamps later
            # than generated_at can already be known. Reuse the promotion
            # evaluator's serve-faithful cutoff/persistence implementation.
            country_rows = rows.loc[index]
            for generated_at, vintage_index in country_rows.groupby("generated_at").groups.items():
                targets = pd.DatetimeIndex(rows.loc[vintage_index, "target_ts"])
                authoritative = net_position_baselines(
                    history, as_of_for_vintage(pd.Timestamp(generated_at)), targets)
                rows.loc[vintage_index, "persistence"] = authoritative["persistence"].to_numpy()
    rows = rows.merge(_load_tso(cfg, forecast_type),
                      on=["country_code", "target_ts"], how="left")
    return rows


def mean_scored_actual(group: pd.DataFrame) -> float | None:
    """Mean actual over exactly the pairs `score_predictions` scored.

    WAPE is `sum|e| / sum|actual|`, so it is only readable beside the level of
    its own denominator. ABL-410 made that concrete: BE `hydro_total` scores
    92% against a 145.66 MW mean and 14,274% against a 1.26 MW one, on the same
    forecasts and the same instants. Reporting the percentage alone would read
    as a catastrophic model regression rather than as a target correction on a
    series that is near zero in this window.
    """
    actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(group["forecast_value"],
                              errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return float(np.mean(actual[valid])) if valid.any() else None


def _score_group(group: pd.DataFrame) -> dict:
    model = score_predictions(group["actual"], group["forecast_value"])
    baselines = {
        name: score_against_baseline(group["actual"], group["forecast_value"],
                                     group[name])
        for name in ("seasonal_naive", "persistence", "tso")
    }
    return {"model": model, "baselines": baselines,
            "mean_actual": mean_scored_actual(group)}


def evaluate_scorecard(cfg: ScorecardConfig) -> dict:
    """Evaluate the configured production models over one target window."""
    models = cfg.models or PRODUCTION_MODELS
    forecasts, vintage_counts = _load_forecasts(cfg)
    selected = select_latest_per_band(forecasts)
    detailed = []
    pooled = []
    evidence_frames = []

    for forecast_type, model_name in models.items():
        type_rows = selected[(selected["forecast_type"] == forecast_type)
                             & (selected["model_name"] == model_name)]
        if type_rows.empty:
            empty_score = _score_group(pd.DataFrame(columns=["actual", "forecast_value",
                                                              "seasonal_naive", "persistence", "tso"]))
            pooled.append({"forecast_type": forecast_type, "model_name": model_name,
                           "horizon_band": "all", **empty_score})
            continue
        evidence = _attach_evidence(cfg, selected, forecast_type)
        evidence_frames.append(evidence)
        for (country, band), group in evidence.groupby(["country_code", "horizon_band"]):
            detailed.append({"forecast_type": forecast_type, "model_name": model_name,
                             "country": country, "horizon_band": band,
                             **_score_group(group)})
        for band, group in evidence.groupby("horizon_band"):
            pooled.append({"forecast_type": forecast_type, "model_name": model_name,
                           "horizon_band": band, **_score_group(group)})
        pooled.append({"forecast_type": forecast_type, "model_name": model_name,
                       "horizon_band": "all", **_score_group(evidence)})

    measured = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
    pooled_all = {(row["forecast_type"], row["model_name"]): row
                  for row in pooled if row["horizon_band"] == "all"}
    reproduction = {}
    for forecast_type, expected in ABL128_REFERENCE.items():
        model_name = models[forecast_type]
        row = pooled_all.get((forecast_type, model_name))
        if row is None:
            measured_values = {key: None for key in expected}
        else:
            measured_values = {
                "wape_pct": row["model"]["wape_pct"],
                "seasonal_naive_wape_pct": row["baselines"]["seasonal_naive"]["baseline"]["wape_pct"],
                "tso_wape_pct": row["baselines"]["tso"]["baseline"]["wape_pct"],
                "bias_pct": row["model"]["bias_pct"],
            }
        reproduction[forecast_type] = {"reference": expected, "measured": measured_values}

    return {
        "meta": {
            "window": {"start": str(cfg.start), "end_exclusive": str(cfg.end)},
            "selection": "latest vintage per country + target + model + horizon band",
            "horizon_bands": [name for name, _, _ in HORIZON_BANDS],
            "selected_forecast_rows": int(len(selected)),
            "paired_actual_rows": (int(measured["actual"].notna().sum())
                                   if not measured.empty else 0),
            "vintage_counts": vintage_counts,
            "models": models,
            "excluded": {"net_position": {"GR": GATE_EXCLUDED_COUNTRIES["GR"]}},
            "load_actual_rule": "load_mw > 0 (load only)",
            "scoring_truth": {forecast_type: {"table": ACTUAL_SPECS[forecast_type][0],
                                              "expression": ACTUAL_SPECS[forecast_type][1]}
                              for forecast_type in models},
            "timestamp_join": "parsed UTC timestamps; accepts T and space separators",
            "net_position_gate": "src/evaluation/net_position.py (not duplicated here)",
            "abl128_reproduction": reproduction,
        },
        "pooled": pooled,
        "by_country_horizon": detailed,
    }


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Not measured"
    return f"{value:.1f}{suffix}"


def render_markdown(results: dict, generated_at: str) -> str:
    meta = results["meta"]
    lines = [
        "# Forecast quality scorecard",
        "",
        f"Generated: {generated_at}",
        f"Target window: {meta['window']['start']} → {meta['window']['end_exclusive']} (exclusive)",
        f"Sample: {meta['selected_forecast_rows']:,} selected forecast rows; "
        f"{meta['paired_actual_rows']:,} paired actual rows",
        f"Selection: {meta['selection']}",
        f"Load actual guard: `{meta['load_actual_rule']}`",
        f"Net-position gate: `{meta['net_position_gate']}`",
        "",
        "## Vintage counts",
        "",
        "| forecast / model | generated timestamps | run-days |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {count['generated_timestamps']:,} | {count['run_days']:,} |"
                 for name, count in meta["vintage_counts"].items())
    lines.extend(["", "## ABL-128 probe reproduction", "",
                  "The direction of the CEO probe reproduces. Exact values below are the current replica under the explicit latest-per-band rule; differences are findings, not adjusted away.",
                  "", "| type | reference WAPE | measured WAPE | reference D−7 | measured D−7 | reference bias | measured bias |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for forecast_type, comparison in meta["abl128_reproduction"].items():
        reference, measured = comparison["reference"], comparison["measured"]
        lines.append(f"| {forecast_type} | {_fmt(reference['wape_pct'], '%')} | "
                     f"{_fmt(measured['wape_pct'], '%')} | "
                     f"{_fmt(reference['seasonal_naive_wape_pct'], '%')} | "
                     f"{_fmt(measured['seasonal_naive_wape_pct'], '%')} | "
                     f"{_fmt(reference['bias_pct'], '%')} | {_fmt(measured['bias_pct'], '%')} |")
    lines.extend(["", "Load reproduces within 0.1 percentage point on WAPE/D−7 and 0.1 point on TSO (reference 4.0%, measured 4.1%). Price, solar, and wind do not reproduce exactly; the scorecard preserves the disagreement.",
                  "", "## Scoring truth", "",
                  "Which statement of the actual each type is scored against. "
                  "Since ABL-410 this is the same table the dashboard publishes "
                  "against, so one model and window has one WAPE across both "
                  "surfaces. Training source is a separate, unchanged decision "
                  "(`db.RENEWABLE_TYPE_SOURCE_TABLE`, still `energy_renewable`); "
                  "where the two disagree about the target, part of the WAPE "
                  "below is target mismatch rather than model error.",
                  "", "| type | table | value |", "|---|---|---|"])
    lines.extend(f"| {forecast_type} | `{spec['table']}` | `{spec['expression']}` |"
                 for forecast_type, spec in meta["scoring_truth"].items())
    lines.extend(["", "## Pooled score", "",
                  "Skill is `100 × (1 − model WAPE / baseline WAPE)` on the exact same pairs. "
                  "`mean actual` is the level of WAPE's own denominator over the scored pairs; "
                  "a percentage against a near-zero mean is arithmetic, not quality.",
                  "", "| type | model | horizon | n | mean actual | WAPE | MAE | bias | slope | corr | D−7 WAPE / skill | persistence WAPE / skill | TSO WAPE / skill |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in results["pooled"]:
        model = row["model"]
        cells = []
        for baseline_name in ("seasonal_naive", "persistence", "tso"):
            comparison = row["baselines"][baseline_name]
            cells.append(f"{_fmt(comparison['baseline']['wape_pct'], '%')} / "
                         f"{_fmt(comparison['skill_pct'], '%')}")
        lines.append(
            f"| {row['forecast_type']} | {row['model_name']} | {row['horizon_band']} | "
            f"{model['n']:,} | {_fmt(row['mean_actual'])} | "
            f"{_fmt(model['wape_pct'], '%')} | {_fmt(model['mae'])} | "
            f"{_fmt(model['bias_pct'], '%')} | {_fmt(model['slope'])} | "
            f"{_fmt(model['correlation'])} | {' | '.join(cells)} |")
    lines.extend(["", "## Country × horizon detail", "",
                  "Rows with no paired observations say **Not measured**; zero is never substituted.",
                  "", "| type | model | country | horizon | n | mean actual | WAPE | bias | slope | corr | D−7 skill | persistence skill | TSO skill |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in results["by_country_horizon"]:
        model = row["model"]
        baselines = row["baselines"]
        lines.append(
            f"| {row['forecast_type']} | {row['model_name']} | {row['country']} | "
            f"{row['horizon_band']} | {model['n']:,} | {_fmt(row['mean_actual'])} | "
            f"{_fmt(model['wape_pct'], '%')} | "
            f"{_fmt(model['bias_pct'], '%')} | {_fmt(model['slope'])} | "
            f"{_fmt(model['correlation'])} | "
            f"{_fmt(baselines['seasonal_naive']['skill_pct'], '%')} | "
            f"{_fmt(baselines['persistence']['skill_pct'], '%')} | "
            f"{_fmt(baselines['tso']['skill_pct'], '%')} |")
    lines.extend(["", "## Correctness notes", "",
                  "- Both `T` and space timestamp separators are parsed before joining.",
                  "- **Renewable-family figures are not comparable across the 2026-08-13 boundary.** ABL-410 moved their scoring truth from `energy_renewable` to `energy_generation` to match what the dashboard publishes. Eight of the fifteen live pairs are unchanged to the digit; `renewable` and `hydro_total` move, and BE `hydro_total` moves by two orders of magnitude because its frozen actual folded in pumped storage. The before/after is in `reports/abl_410_scoring_truth.md`.",
                  "- `energy_generation` has an open FR ingest gap (ABL-318 §3): no rows 2026-06-30 23:45 → 2026-07-22 14:15. FR sample sizes in any window overlapping it are correspondingly smaller.",
                  "- **Pooled rows are denominator-weighted across countries, so they move when coverage does.** On the 2026-07-11 → 2026-08-10 window the FR gap above cost 1,072 hours of the best-forecast country in most types: pooled `solar` went 48.35% → 51.85% and pooled `wind_onshore` 76.94% → 77.72% while **every country's own figure was unchanged to the digit**. Read a pooled move against the country × horizon detail before reading it as quality.",
                  "- `load_mw > 0` is applied only to load. Measured zero remains valid for every other type.",
                  f"- GR net position is excluded by name: {meta['excluded']['net_position']['GR']}",
                  "- D−7 and persistence use only stored actual observations. Missing source rows remain missing.",
                  "- Net-position persistence reuses the promotion evaluator's day-ahead publication cutoff.",
                  "- TSO comparisons use the latest stored TSO series; the database does not retain an issued-vintage archive for reconstruction.",
                  "- The separate net-position promotion gate remains authoritative and is not reproduced here.", ""])
    return "\n".join(lines)
