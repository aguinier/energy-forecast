#!/usr/bin/env python3
"""Fit and read the pre-registered ABL-253 serve-faithful solar gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.data_quality import find_suspect_constant_runs
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.scorecard import (
    ScorecardConfig, _load_forecasts, _load_tso, _ro_connect,
    describe_opened_databases, opened_databases,
    score_predictions, select_latest_per_band,
)
from src.evaluation.solar_retrain import (
    ALGORITHM, FEATURE_COLUMNS, INTENDED_N, PRIMARY_BANDS, SCHEDULE_N,
    attach_baselines, build_vintage_frame, common_scores, finite_training_rows,
    gate_cell, select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder


# ABL-378, porting the ABL-322 fix from the wind harness.  This harness used to
# read `COUNTRIES` directly and hardcode its pass bar to the 9 cells that set
# produces, so it could not be pointed at any of the 28 solar countries with no
# model without editing the constant -- at which point `== 9` silently became
# the wrong denominator.  Each scope now names its countries outright and the
# registered cell count is derived from *that table*, fixed in the file before
# the run, rather than from whatever the run turns out to score.  That keeps the
# property the literal existed to protect: a country that silently yields no
# gate rows still falls short of its scope's count instead of quietly leaving
# the denominator.  Adding a scope is a pre-registration and belongs in review.
#
# The country tuple is written out here rather than referencing
# `solar_retrain.COUNTRIES`, deliberately: a registration must not follow a
# shared mutable constant.  Adding AT to `COUNTRIES` -- it is the one other
# country with a solar incumbent -- would otherwise silently re-scope a
# dispositioned gate.  `test_gate_scope_registration` pins the two as equal
# today, so any divergence surfaces in review instead of applying itself.
SCOPES = {
    # ABL-253 as registered: BE/DE/FR solar.  3 countries x 3 primary bands = 9
    # cells.  Unchanged, and the default, so an unflagged run still reproduces
    # ABL-253 exactly.
    "abl253": ("BE", "DE", "FR"),
}

# The columns that must be *simultaneously finite* for a row to enter a gate
# cell.  This is a registered property of the scope, not a detail.
# `common_scores` keeps only rows where every named column is finite, and the
# incumbent is left-merged, so it is NaN on every row for a country with no rows
# in `forecasts`.  Naming `incumbent` in the basis therefore empties the
# intersection for such a country: n=0, all scores None, and the verdict below
# would render that as FAIL -- a model-quality disposition for a race that was
# never run.  That is how the ABL-322 pilot first reported, and it would land
# the same way on all 28 solar countries in ABL-316 that have no incumbent.
#
# ABL-253 keeps the four-way basis it was actually read under, so re-reading it
# does not silently move already-dispositioned numbers.  A new scope registers
# the basis its own pre-registration names.
GATE_BASIS = {
    "abl253": ("challenger", "incumbent", "seasonal_naive", "persistence"),
}
#: Always reported, each on its own intersection with the gate basis, so that a
#: comparator which never exists reads "Not measured" instead of voiding the gate.
REPORTED_COMPARATORS = ("challenger", "incumbent", "seasonal_naive", "persistence")


def _model():
    params = config.get_default_params(ALGORITHM)
    return CatBoostRegressor(**params), params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_runs(replica: str, country: str, start, end, source: str) -> list[dict]:
    """Screen the fitted series for ABL-188 zero-fill runs.

    ABL-345: `source` is the table the model was fitted on, not a constant. This
    read used to be hardcoded to `energy_renewable` while the builder's source
    became a per-run argument — a contamination audit of a series the model
    never saw. It reports the wrong way in both directions: an
    `energy_generation` fit inherits `energy_renewable`'s zero-fill runs (which
    are the reason to leave that table), and a genuine constant run in the
    fitted series goes unreported. `verdict` is derived from this list, so a
    mismatched screen moves the harness's disposition, not just its prose.
    """
    if source not in db._RENEWABLE_TYPE_SOURCES:
        raise ValueError(
            f"unknown renewable source table: {source!r}; "
            f"expected one of {db._RENEWABLE_TYPE_SOURCES}"
        )
    # Both tables name this column identically; `RENEWABLE_TYPE_COLUMNS` is the
    # one place that knows, and `load_renewable_type_data` already reads either
    # table through it.
    column = db.RENEWABLE_TYPE_COLUMNS["solar"]
    con = _ro_connect(replica)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {column} AS value FROM {source} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
            "ORDER BY timestamp_utc",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], format="mixed", utc=True
    ).dt.tz_localize(None)
    return [
        {"start": str(run.start), "end": str(run.end), "value": run.value,
         "n_rows": run.n_rows, "duration_hours": run.duration_hours}
        for run in find_suspect_constant_runs(frame, "value")
    ]


def _fmt(value, suffix=""):
    return "Not measured" if value is None else f"{value:.1f}{suffix}"


def disposition(gate_cells: list[dict], registered_cells: int,
                contaminated: bool) -> tuple[str, str]:
    """Map the scored cells onto a verdict and its recommendation.

    Extracted from `main` so the `UNREADABLE` branch is reachable in a test
    without training three models. The distinction it encodes is the whole point
    of ABL-378: a cell that scored no rows did not lose a race, it never ran one,
    and calling that FAIL reports a model-quality verdict on a comparison that
    never happened.
    """
    performance_pass = len(gate_cells) == registered_cells and gate_cells and all(
        row["gate"]["pass"] for row in gate_cells)
    if performance_pass and not contaminated:
        return "PASS", (
            "The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. "
            "Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.")
    if performance_pass:
        return "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION", (
            "The challenger clears the performance bar, but a suspect constant run touches the registered data window. "
            "Do not promote; send the run to the CEO/ingest owner for adjudication first.")
    unreadable = [row for row in gate_cells if row["gate"]["n"] == 0]
    if unreadable:
        return "UNREADABLE", (
            f"No disposition: {len(unreadable)}/{registered_cells} primary cells scored zero rows, so the challenger was "
            "never compared to the baseline in them. This is not a model-quality result and must not be reported as one. "
            "Fix the cause of the empty intersection and re-read the gate; the registered windows, bands, metric, baseline "
            "and minimum n are untouched by a run that produced no score.")
    passed = sum(row["gate"]["pass"] for row in gate_cells)
    return "FAIL", (
        f"Do not promote these artifacts: only {passed}/{registered_cells} primary cells clear the registered bar. Report the "
        "losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split.")


def render_markdown(result: dict) -> str:
    meta, cells = result["meta"], result["gate_cells"]
    passed = sum(cell["gate"]["pass"] for cell in cells)
    lines = [
        "# ABL-253 — Serve-faithful solar retrain gate", "",
        f"**Disposition: {result['verdict']}**", "",
        f"Generated: {meta['generated_at']}",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample gate targets: {meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive).",
        "Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.",
        # ABL-355: which *files* the run opened. `--replica-db` used to cover
        # the incumbent, TSO and screen only; the fitted series and weather came
        # from `ENERGY_DB_PATH`, and this heading named one path for two files.
        *describe_opened_databases(meta["databases"], meta["replica_bytes"]),
        # ABL-345: the two tables disagree — 9 months against 5.6 years of
        # history, and `energy_renewable` zero-fills what `energy_generation`
        # leaves NULL. Two runs of this report are not comparable unless both
        # say which table they read, so it is stated, never defaulted silently.
        f"Target series, features, baselines and contamination screen: `{meta['training_source']}`.",
        "", "## Gate read", "",
        f"Registered scope `{meta['scope']}`: {', '.join(meta['registered_countries'])}.",
        f"Gate basis — the columns that must be simultaneously finite for a row to be scored: {', '.join(f'`{c}`' for c in meta.get('gate_basis', []))}. "
        "Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that "
        "does not exist for a country reads Not measured instead of emptying the cell.",
        f"Strict full PASS requires challenger WAPE < D-7 in all {meta['registered_cells']} country × primary D+2-band cells and ≥95% of intended pairs. Result: **{passed}/{meta['registered_cells']} cells pass**.",
        "",
        "| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in cells:
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        # A cell that scored no rows has None on both sides. It renders as
        # "Not measured", never as a number and never as a crash.
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        lines.append(
            f"| {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill} | {_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} |"
        )
    # The protocol-count sentence below is a measured ABL-253 fact about that
    # scope's eight registered run instants. Rendering it for every scope would
    # state another scope's row counts as if they had been measured; the per-cell
    # `n` column above already carries that truth for whichever scope ran.
    if meta["scope"] == "abl253":
        lines.extend([
            "",
            "The exact eight registered run instants imply 210/570/720/720/510 selected rows by band. As in ABL-195, the frozen registered minimum for 48–64h remains 456 (95% of 480), while the schedule offers 510 rows.",
        ])
    basis_names = ", ".join(meta.get("gate_basis", []))
    lines.extend([
        "", "## Per-country all-D+2 summary", "",
        f"Gate-basis values (actual, {basis_names}) share one finite intersection; each comparator outside the basis is "
        "scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing "
        "`Not measured` had no finite rows at all.", "",
        "| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["country_d2"]:
        scores, tso = row["scores"], row["tso"]
        lines.append(
            f"| {row['country']} | {row['n']:,} | {_fmt(scores['challenger']['wape_pct'], '%')} | "
            f"{_fmt(scores['seasonal_naive']['wape_pct'], '%')} | {_fmt(scores['persistence']['wape_pct'], '%')} | "
            f"{_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(tso['wape_pct'], '%')} (n={tso['n']:,}) |"
        )
    lines.extend([
        "", "## Fit and missingness audit", "",
        "Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.", "",
        "| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in result["training"]:
        audit = row["audit"]
        lines.append(
            f"| {row['country']} | {row['algorithm']} | {audit['retained_rows']:,} / {audit['intended_rows']:,} | "
            f"{audit['unique_targets']:,} | {audit['excluded_missing_actual_or_feature']:,} | "
            f"{audit['degraded_lag_1d_rows']:,} | `{row['artifact_sha256']}` |"
        )
    lines.extend(["", "## Data quality and limits", ""])
    source = meta["training_source"]
    contaminated = [row for row in result["training"] if row["constant_runs"]]
    if contaminated:
        for row in contaminated:
            lines.append(f"- ABL-188 screening found suspect solar runs for {row['country']} in `{source}`: `{row['constant_runs']}`. The builder nulls these before fit; see the training audit and recommendation.")
    else:
        lines.append(f"- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `{source}` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.")
    # ABL-345: both notes below are findings about `energy_renewable` specifically
    # — the zero-fill run is ABL-188's `energy_renewable` mapper defect, and the
    # New Year's read was on that table. Printing them under an
    # `energy_generation` run would report a table this run never opened.
    if source == "energy_renewable" and not contaminated:
        lines.append("- The known DE zero-fill run (2025-09-08 22:00 → 2025-11-14 15:45 UTC; 6,408 quarter-hours) is outside this fit/lookback window.")
        lines.append("- The audit initially appeared to flag FR zero from 2025-12-31 17:00 to 2026-01-02 07:15 UTC, but the replica has no intervening New Year's Day rows and `energy_generation` independently agrees on zero for the available nighttime observations. `find_suspect_constant_runs` was incorrectly joining equal values across missing-time gaps despite its contiguous-run contract. The invariant now splits on cadence gaps; the original continuous DE defect remains covered by regression tests.")
    lines.extend([
        "- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.",
        "- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.",
        "- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.",
        "- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.",
        "", "## Recommendation to the CEO", "", result["recommendation"], "",
        "No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=str(config.FORECAST_OUTPUT_DB))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--artifact-dir", default="experiments/ABL253/artifacts")
    parser.add_argument("--json-out", default="experiments/ABL253/results.json")
    parser.add_argument("--report-out", default="reports/abl_253_solar_retrain.md")
    parser.add_argument("--scope", default="abl253", choices=sorted(SCOPES),
                        help="Registered country scope. Each scope fixes its countries "
                             "and its gate basis in the file; the cell count is derived "
                             "from them (default: abl253, which reproduces ABL-253).")
    # ABL-345: the 19 unmodelled solar pairs have ~9 months in `energy_renewable`
    # against ~5.6 years in `energy_generation`, so this harness cannot gate them
    # on one hardcoded table. Opt-in, so an unflagged run reproduces ABL-253.
    parser.add_argument("--renewable-source", default=None,
                        choices=list(db._RENEWABLE_TYPE_SOURCES),
                        help="Source table for the fitted series, its lag and rolling "
                             "features, the D-7/persistence baselines, the gate actuals "
                             "and the contamination screen (default: "
                             f"{db.RENEWABLE_TYPE_SOURCE_TABLE})")
    args = parser.parse_args()
    fit_start, gate_start, gate_end = map(pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    # Resolved once, here, and handed to every read site and to the record. The
    # builder would resolve a `None` identically (`Forecaster._resolved_training_source`,
    # `forecaster.py:132`), but then the run's source would be a default applied
    # in three places rather than one recorded fact — and the report could not
    # name the table it read.
    source = args.renewable_source or db.RENEWABLE_TYPE_SOURCE_TABLE

    cfg = ScorecardConfig(str(replica), args.sidecar_db, gate_start, gate_end,
                          models={"solar": "catboost"})
    incumbent_raw, vintage_counts = _load_forecasts(cfg)
    incumbent = select_latest_per_band(incumbent_raw)
    tso = _load_tso(cfg, "solar")
    artifact_dir = Path(args.artifact_dir)
    registered_countries = SCOPES[args.scope]
    registered_cells = len(registered_countries) * len(PRIMARY_BANDS)
    training, scored_frames = [], []
    for country in registered_countries:
        # ABL-342 records provenance from the builder rather than from a source
        # string, so passing the source here is also what makes the artifact's
        # `training_source` truthful.
        # ABL-355: `db_path` for the same reason `actuals_source` is here. The
        # builder resolved neither on its own — it read `config.DATABASE_PATH`,
        # so `--replica-db` bought the incumbent and the screen while the fitted
        # series came from wherever `ENERGY_DB_PATH` pointed. Passing the
        # resolved replica is what makes `--replica-db` mean the whole run.
        builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                          gate_end, actuals_source=source,
                                          db_path=str(replica))
        fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
        fit, audit = finite_training_rows(fit_raw, FEATURE_COLUMNS)
        model, params = _model()
        model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])

        # ABL-342: through `Forecaster.save`, so the artifact carries the table
        # it was fitted on and the ABL-183 intercept witness by construction.
        path = save_gate_artifact(
            artifact_dir / country / "solar" / "model.joblib",
            model=model, builder=builder, algorithm=ALGORITHM, params=params,
            feature_columns=FEATURE_COLUMNS, fit_window=(fit_start, gate_start),
        )

        gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)
        gate_finite, gate_audit = finite_training_rows(gate_raw, FEATURE_COLUMNS)
        gate_finite["challenger"] = model.predict(gate_finite[list(FEATURE_COLUMNS)])
        selected = attach_baselines(select_latest_challenger_per_band(gate_finite), builder._actuals)
        inc = incumbent[incumbent["country_code"] == country][
            ["target_ts", "horizon_band", "forecast_value"]
        ].rename(columns={"forecast_value": "incumbent"})
        selected = selected.merge(inc, on=["target_ts", "horizon_band"], how="left")
        selected = selected.merge(tso[tso["country_code"] == country][["target_ts", "tso"]],
                                  on="target_ts", how="left")
        selected["country"] = country
        scored_frames.append(selected)
        training.append({"country": country, "algorithm": ALGORITHM, "params": params,
                         "audit": audit, "gate_build_audit": gate_audit,
                         "constant_runs": _constant_runs(str(replica), country,
                                                          fit_start - pd.Timedelta(days=14), gate_end,
                                                          source),
                         "artifact_path": str(path.resolve()), "artifact_sha256": _sha256(path)})

    gate_basis = GATE_BASIS[args.scope]

    def scored(group):
        """Score on the scope's registered gate basis; report the rest beside it.

        Each comparator outside the basis is scored on its own intersection
        *with* the basis, so a comparator that is absent for this country costs
        its own row and nothing else. Returns the basis scores, the basis
        intersection, and each comparator's own n.
        """
        scores, common = common_scores(group, gate_basis)
        comparator_n = {name: len(common) for name in gate_basis}
        for name in REPORTED_COMPARATORS:
            if name in scores:
                continue
            sub_scores, sub_common = common_scores(group, (*gate_basis, name))
            scores[name], comparator_n[name] = sub_scores[name], len(sub_common)
        return scores, common, comparator_n

    all_scored = pd.concat(scored_frames, ignore_index=True)
    gate_cells, country_d2 = [], []
    for (country, band), group in all_scored.groupby(["country", "horizon_band"]):
        scores, common, comparator_n = scored(group)
        if band in PRIMARY_BANDS:
            gate_cells.append({"country": country, "horizon_band": band, "scores": scores,
                               "comparator_n": comparator_n,
                               "gate": gate_cell(scores["challenger"]["wape_pct"],
                                                 scores["seasonal_naive"]["wape_pct"],
                                                 len(common), INTENDED_N[band])})
    for country, group in all_scored[all_scored["horizon_band"].isin(PRIMARY_BANDS)].groupby("country"):
        scores, common, comparator_n = scored(group)
        tso_valid = np.isfinite(common[["actual", "tso"]].to_numpy(dtype=float)).all(axis=1)
        country_d2.append({"country": country, "n": len(common), "scores": scores,
                           "comparator_n": comparator_n,
                           "tso": score_predictions(common.loc[tso_valid, "actual"], common.loc[tso_valid, "tso"])})

    passed = sum(row["gate"]["pass"] for row in gate_cells)
    contaminated = any(row["constant_runs"] for row in training)
    verdict, recommendation = disposition(gate_cells, registered_cells, contaminated)

    result = {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
                       # ABL-355: the run's files, not just its tables. The
                       # builder is handed `replica`, so `features` equals
                       # `replica` by construction — recorded anyway, because
                       # what this issue cost was the *absence* of the record.
                       "databases": opened_databases(cfg, str(replica), config.DATABASE_PATH),
                       "training_source": source,
                       "scope": args.scope, "registered_countries": list(registered_countries),
                       "registered_cells": registered_cells, "gate_basis": list(gate_basis),
                       "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
                       "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
                       "registered_intended_n": INTENDED_N, "schedule_implied_n": SCHEDULE_N,
                       "vintage_counts": vintage_counts,
                       "selection": "latest vintage per country + target + model + horizon band"},
              "verdict": verdict, "recommendation": recommendation, "training": training,
              "gate_cells": sorted(gate_cells, key=lambda row: (row["country"], row["horizon_band"])),
              "country_d2": sorted(country_d2, key=lambda row: row["country"])}
    json_path, report_path = Path(args.json_out), Path(args.report_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"{verdict}: {passed}/{registered_cells} cells passed; wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
