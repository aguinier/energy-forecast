#!/usr/bin/env python3
"""ABL-410: what the renewable-family scorecard measures under each candidate truth.

Read-only. Scores the *same* stored production forecasts, under the *same*
scorecard protocol (latest vintage per country/target/model/horizon band,
top-of-hour actuals only), against three different statements of the actual:

  ``frozen_strict``    `energy_renewable`, `hydro_run_mw + hydro_reservoir_mw`
                       -- what `scorecard.ACTUAL_SPECS` scored before this issue.
  ``frozen_nullaware`` `energy_renewable`, `db._HYDRO_TOTAL_EXPR`
                       -- the same table under the training-side hydro rule.
                       Included to separate "which table" from "which sum": on
                       this table every `*_mw` column is `REAL DEFAULT 0`, so
                       the two hydro forms should be *identical*, and the whole
                       frozen-vs-generation gap is the table.
  ``generation``       `energy_generation`, null-aware sums, hydro without the
                       pumped leg -- what the dashboard publishes since ABL-399
                       (PR #30, merged 2026-08-13T20:05Z).

It also decomposes the hydro actual into its two legs on each table, which is
where the largest divergence lives: for BE the frozen `hydro_reservoir_mw` is
pumped storage, not reservoir hydro.

Nothing here fits, corrects or promotes anything. Output is a markdown report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import _HYDRO_TOTAL_EXPR  # noqa: E402
from src.evaluation.scorecard import (  # noqa: E402
    PRODUCTION_MODELS, ScorecardConfig, _load_forecasts, _ro_connect,
    filter_measured_actuals, normalize_timestamps, score_predictions,
    select_latest_per_band,
)

#: The renewable family only. `load`, `price` and `net_position` are untouched
#: by ABL-399 and by this issue -- their actuals live in tables that have no
#: second candidate.
FAMILY = ("renewable", "solar", "wind_onshore", "wind_offshore", "biomass",
          "hydro_total")

#: `energy_generation`'s counterpart to the frozen `total_renewable_mw`, which
#: is a stored computed column with no equivalent there. These are the columns
#: the dashboard's `renewableTotal.RENEWABLE_MW_COLUMNS` sums, in its order --
#: pumped storage is deliberately absent from both.
GENERATION_RENEWABLE_COLUMNS = (
    "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
    "hydro_run_mw", "hydro_reservoir_mw", "biomass_mw",
    "geothermal_mw", "marine_mw", "other_renewable_mw",
)


def null_aware_sum(columns: tuple[str, ...]) -> str:
    """SQL that sums the reported members and is NULL only when none is."""
    reported = " AND ".join(f"{c} IS NULL" for c in columns)
    summed = " + ".join(f"COALESCE({c}, 0)" for c in columns)
    return f"CASE WHEN {reported} THEN NULL ELSE {summed} END"


TRUTH_VARIANTS: dict[str, dict[str, tuple[str, str]]] = {
    "frozen_strict": {
        "renewable": ("energy_renewable", "total_renewable_mw"),
        "solar": ("energy_renewable", "solar_mw"),
        "wind_onshore": ("energy_renewable", "wind_onshore_mw"),
        "wind_offshore": ("energy_renewable", "wind_offshore_mw"),
        "biomass": ("energy_renewable", "biomass_mw"),
        "hydro_total": ("energy_renewable", "hydro_run_mw + hydro_reservoir_mw"),
    },
    "frozen_nullaware": {
        "renewable": ("energy_renewable", "total_renewable_mw"),
        "solar": ("energy_renewable", "solar_mw"),
        "wind_onshore": ("energy_renewable", "wind_onshore_mw"),
        "wind_offshore": ("energy_renewable", "wind_offshore_mw"),
        "biomass": ("energy_renewable", "biomass_mw"),
        "hydro_total": ("energy_renewable", _HYDRO_TOTAL_EXPR),
    },
    "generation": {
        "renewable": ("energy_generation", null_aware_sum(GENERATION_RENEWABLE_COLUMNS)),
        "solar": ("energy_generation", "solar_mw"),
        "wind_onshore": ("energy_generation", "wind_onshore_mw"),
        "wind_offshore": ("energy_generation", "wind_offshore_mw"),
        "biomass": ("energy_generation", "biomass_mw"),
        "hydro_total": ("energy_generation", _HYDRO_TOTAL_EXPR),
    },
}

#: Per-leg series, so the hydro divergence can be attributed rather than
#: asserted. Name -> (table, expression).
HYDRO_LEGS = {
    "frozen run": ("energy_renewable", "hydro_run_mw"),
    "frozen reservoir": ("energy_renewable", "hydro_reservoir_mw"),
    "generation run": ("energy_generation", "hydro_run_mw"),
    "generation reservoir": ("energy_generation", "hydro_reservoir_mw"),
    "generation pumped": ("energy_generation", "hydro_pumped_mw"),
}


def load_series(replica_db: str, table: str, expression: str,
                start: pd.Timestamp, end: pd.Timestamp,
                forecast_type: str) -> pd.DataFrame:
    """One actual series, under the scorecard's own measurement rules."""
    con = _ro_connect(replica_db)
    try:
        df = pd.read_sql_query(
            f"""SELECT country_code, timestamp_utc, {expression} AS actual
                FROM {table}
                WHERE timestamp_utc >= ? AND timestamp_utc < ?
                  AND ({expression}) IS NOT NULL""",
            con, params=(str(start), str(end)))
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["country_code", "target_ts", "actual"])
    df["ts"] = normalize_timestamps(df["timestamp_utc"])
    df = filter_measured_actuals(df, forecast_type)
    return (df[["country_code", "ts", "actual"]]
            .drop_duplicates(["country_code", "ts"], keep="last")
            .rename(columns={"ts": "target_ts"})
            .reset_index(drop=True))


def scored_rows(selected: pd.DataFrame, forecast_type: str, model_name: str,
                actuals: pd.DataFrame,
                restrict: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pair the selected forecasts with one actual series.

    `restrict` is the common-instant index — the (country, target) pairs both
    candidate tables carry. Without it a frozen-vs-generation WAPE comparison
    silently compares two different samples: over this issue's window
    `energy_generation` is missing 2026-07-11 -> 2026-07-22 for FR entirely
    (the ABL-318 §3 ingest gap), which is 279 of the 720 hours.
    """
    rows = selected[(selected["forecast_type"] == forecast_type)
                    & (selected["model_name"] == model_name)].copy()
    if restrict is not None:
        rows = rows.merge(restrict, on=["country_code", "target_ts"], how="inner")
    return rows.merge(actuals, on=["country_code", "target_ts"], how="left")


def common_instants(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The (country, target) pairs every candidate truth actually carries."""
    keys = [frame[["country_code", "target_ts"]].drop_duplicates()
            for frame in frames if not frame.empty]
    if not keys:
        return pd.DataFrame(columns=["country_code", "target_ts"])
    shared = keys[0]
    for other in keys[1:]:
        shared = shared.merge(other, on=["country_code", "target_ts"], how="inner")
    return shared.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", required=True)
    parser.add_argument("--sidecar-db", default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    sidecar = (args.sidecar_db
               if args.sidecar_db and Path(args.sidecar_db).exists() else None)
    models = {t: m for t, m in PRODUCTION_MODELS.items() if t in FAMILY}
    cfg = ScorecardConfig(replica_db=args.replica_db, sidecar_db=sidecar,
                          start=start, end=end, models=models)
    forecasts, vintages = _load_forecasts(cfg)
    selected = select_latest_per_band(forecasts)

    variants = {name: {ft: load_series(args.replica_db, *spec[ft], start, end, ft)
                       for ft in models}
                for name, spec in TRUTH_VARIANTS.items()}

    lines = [
        "# ABL-410 — renewable-family scorecard under each candidate truth",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        f"Replica: `{Path(args.replica_db).resolve()}` "
        f"({Path(args.replica_db).stat().st_size:,} bytes), SQLite `mode=ro`.",
        f"Sidecar: `{sidecar}`" if sidecar else "Sidecar: not opened.",
        f"Target window: {start} -> {end} (exclusive).",
        "Selection: the scorecard's own — latest vintage per country + target + "
        "model + horizon band, top-of-hour actuals only, no aggregation.",
        "Models: the `PRODUCTION_MODELS` registry snapshot, renewable family only.",
        "All figures out-of-sample with respect to the stored forecasts (they "
        "were issued before their target); the *models* were fitted on windows "
        "this run does not know, so no in-sample claim is made either way.",
        "",
    ]

    def wape_cell(value):
        return f"{value:.2f}%" if value is not None else "not measured"

    def mean_cell(value):
        return "n/a" if value is None else f"{value:.2f}"

    def stat_cell(value):
        return "n/a" if value is None else f"{value:.3f}"

    for common_only in (True, False):
        lines.extend([
            "## Pooled WAPE by truth — "
            + ("common instants only" if common_only
               else "every instant each table carries"),
            "",
            ("Restricted to the (country, target) pairs both tables carry, so "
             "the three columns score the identical sample. This is the "
             "comparable table."
             if common_only else
             "Each truth on its own coverage — this is what each surface "
             "actually publishes, and the n columns are the reason the two are "
             "not the same measurement."),
            "",
            "| type | model | country | n frozen | WAPE frozen strict | "
            "WAPE frozen null-aware | n generation | WAPE generation | "
            "mean actual frozen | mean actual generation |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for forecast_type, model_name in models.items():
            restrict = (common_instants([variants[name][forecast_type]
                                         for name in TRUTH_VARIANTS])
                        if common_only else None)
            per_variant = {name: scored_rows(selected, forecast_type,
                                             model_name,
                                             variants[name][forecast_type],
                                             restrict)
                           for name in TRUTH_VARIANTS}
            countries = sorted(set().union(*(set(df["country_code"].unique())
                                             for df in per_variant.values())))
            # `(pooled)` is the row the scorecard's own `horizon_band == "all"`
            # publishes. It is denominator-weighted across countries, so when
            # the two truths cover different instants it moves on *composition*
            # even where every country is identical -- the FR ingest gap alone
            # drops 1,072 well-forecast hours out of the pool.
            for country in [*countries, "(pooled)"]:
                cells = {}
                for name, df in per_variant.items():
                    sub = df if country == "(pooled)" else df[df["country_code"] == country]
                    score = score_predictions(sub["actual"], sub["forecast_value"])
                    mean = (float(np.nanmean(sub["actual"]))
                            if sub["actual"].notna().any() else None)
                    cells[name] = (score, mean)
                strict, nullaware, gen = (cells["frozen_strict"],
                                          cells["frozen_nullaware"],
                                          cells["generation"])
                lines.append(
                    f"| {forecast_type} | {model_name} | {country} | "
                    f"{strict[0]['n']:,} | {wape_cell(strict[0]['wape_pct'])} | "
                    f"{wape_cell(nullaware[0]['wape_pct'])} | {gen[0]['n']:,} | "
                    f"{wape_cell(gen[0]['wape_pct'])} | "
                    f"{mean_cell(strict[1])} | {mean_cell(gen[1])} |")
        lines.append("")

    # ---- hydro leg decomposition -------------------------------------------
    lines.extend([
        "",
        "## Hydro, leg by leg",
        "",
        "The same stored `hydro_total` forecast scored against each component "
        "series on its own, on the common instants. `slope` and `corr` are the "
        "model against that leg: a model that tracks the store rather than the "
        "river shows it here.",
        "",
        "| country | leg | n | mean MW | WAPE | slope | corr |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    hydro_model = models["hydro_total"]
    hydro_rows = selected[(selected["forecast_type"] == "hydro_total")
                          & (selected["model_name"] == hydro_model)]
    leg_series = {name: load_series(args.replica_db, table, expr, start, end,
                                    "hydro_total")
                  for name, (table, expr) in HYDRO_LEGS.items()}
    hydro_common = common_instants([variants[name]["hydro_total"]
                                    for name in TRUTH_VARIANTS])
    for country in sorted(hydro_rows["country_code"].unique()):
        for name, series in leg_series.items():
            merged = scored_rows(selected, "hydro_total", hydro_model, series,
                                 hydro_common)
            sub = merged[merged["country_code"] == country]
            score = score_predictions(sub["actual"], sub["forecast_value"])
            mean = (float(np.nanmean(sub["actual"]))
                    if sub["actual"].notna().any() else None)
            lines.append(
                f"| {country} | {name} | {score['n']:,} | {mean_cell(mean)} | "
                f"{wape_cell(score['wape_pct'])} | {stat_cell(score['slope'])} | "
                f"{stat_cell(score['correlation'])} |")

    lines.extend(["", "## Vintage counts", "",
                  "| forecast / model | generated timestamps | run-days |",
                  "|---|---:|---:|"])
    lines.extend(f"| {name} | {c['generated_timestamps']:,} | {c['run_days']:,} |"
                 for name, c in vintages.items())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
