#!/usr/bin/env python3
"""Fit and read a pre-registered serve-faithful solar gate (see SCOPES).

The default scope is `abl253` (solar BE/DE/FR), so an unflagged run reproduces
that gate exactly; `--scope` selects any other registered pair set.
"""

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
    score_predictions, select_latest_per_band,
)
from src.evaluation.solar_retrain import (
    ALGORITHM, FEATURE_COLUMNS, INTENDED_N, PRIMARY_BANDS, SCHEDULE_N,
    attach_baselines, build_vintage_frame, finite_training_rows, gate_cell,
    gate_verdict, scores_with_comparators, select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder


# ABL-379 ports ABL-322's scope model from the wind harness. A scope is a
# *pre-registration*, not a filter: each one names its pairs outright, in this
# file, before the run, and the registered cell count is derived from that table
# rather than from whatever a run turns out to score. That keeps the property
# the old hardcoded 9 existed to protect -- a pair that silently yields no gate
# rows falls short of its scope's count instead of quietly leaving the
# denominator -- while letting a two-pair tranche have a bar of 6.
#
# Adding a scope is a pre-registration and belongs in review, together with its
# `GATE_BASIS` and `SCOPE_OUTPUTS` entries (see `_check_scope_tables`).
SCOPES = {
    # ABL-253 as registered: solar BE/DE/FR. 3 pairs x 3 primary bands = 9
    # cells. Unchanged, and the default, so an unflagged run still reproduces
    # ABL-253 exactly.
    "abl253": (("solar", "BE"), ("solar", "DE"), ("solar", "FR")),
    # ABL-348's recommended first solar tranche (§5): the two pairs whose
    # `energy_generation` history starts on the same day as `energy_renewable`,
    # so the source change costs them no depth. 2 pairs x 3 bands = 6 cells.
    # Neither serves today, so this scope refits no live model. Both sit
    # outside ABL-348's `not_evaluable` list (EE and FI solar).
    "abl348-level": (("solar", "BG"), ("solar", "CH")),
}

# The columns that must be *simultaneously finite* for a row to enter a gate
# cell. This is a registered property of the scope, not a detail. Every one of
# ABL-316's 19 remaining solar pairs has zero rows in `forecasts`, so with
# `incumbent` in the basis `incumbent` is NaN on every row, the intersection is
# empty, and each cell scores n=0 with `scores["challenger"]` None -- which
# `gate_cell(...)` then subscripts. That is the ABL-322 defect verbatim, and it
# renders as a plausible FAIL rather than a crash.
#
# The registered bar names challenger and seasonal-naive D-7 only -- in
# ABL-253's registration and in ABL-348's -- so new-country scopes gate on
# exactly those. `abl253` keeps the four-way basis it was actually read under,
# for the same reason ABL-322 left `abl195` alone: its numbers are already
# dispositioned and re-basing them here would move them silently. Re-reading
# ABL-253 under a narrower basis is a separate decision for whoever owns that
# gate.
GATE_BASIS = {
    "abl253": ("challenger", "incumbent", "seasonal_naive", "persistence"),
    "abl348-level": ("challenger", "seasonal_naive"),
}

# Output paths are keyed off the scope so one scope cannot overwrite another's
# artifacts. The defaults used to be flat `experiments/ABL253/...` and
# `reports/abl_253_solar_retrain.md` for every invocation, so a tranche run that
# forgot three flags overwrote ABL-253's dispositioned gate read in place.
SCOPE_OUTPUTS = {
    "abl253": {"artifact_dir": "experiments/ABL253/artifacts",
               "json_out": "experiments/ABL253/results.json",
               "report_out": "reports/abl_253_solar_retrain.md"},
    # One directory level, like every other entry in `experiments/`: the
    # `.gitignore` rules that keep gate artifacts and results out of the repo
    # are `experiments/*/artifacts/` and `experiments/*/results.json`
    # (`.gitignore:53-56`), and a nested `experiments/ABL348/solar-level/...`
    # would slip past both globs and commit a 9 GB-derived artifact tree.
    "abl348-level": {"artifact_dir": "experiments/ABL348-solar-level/artifacts",
                     "json_out": "experiments/ABL348-solar-level/results.json",
                     "report_out": "reports/abl_348_solar_level_retrain.md"},
}

#: Always reported, each on its own intersection with the gate basis, so that a
#: comparator which never exists reads "Not measured" instead of voiding the gate.
REPORTED_COMPARATORS = ("challenger", "incumbent", "seasonal_naive", "persistence")


def _check_scope_tables() -> None:
    """Fail at import if a scope is registered in one table and not the others.

    `SCOPES[args.scope]` and `GATE_BASIS[args.scope]` would `KeyError` mid-run
    otherwise -- loud, but only after argparse has already accepted the scope
    and, for the basis, only after every pair has been fitted. Registering a
    scope means adding all three entries; this says so at the cheapest moment.
    """
    tables = {"SCOPES": SCOPES, "GATE_BASIS": GATE_BASIS,
              "SCOPE_OUTPUTS": SCOPE_OUTPUTS}
    for name, table in tables.items():
        missing = set(SCOPES) ^ set(table)
        if missing:
            raise RuntimeError(
                f"scope tables disagree: {name} differs from SCOPES on "
                f"{sorted(missing)}; a registered scope needs an entry in "
                f"{', '.join(sorted(tables))}")
    for scope, pairs in SCOPES.items():
        wrong = sorted({stream for stream, _ in pairs} - {"solar"})
        if wrong:
            raise RuntimeError(
                f"scope {scope!r} registers non-solar streams {wrong}; this "
                "harness fits solar only (see ALGORITHM)")
        if not {"challenger", "seasonal_naive"} <= set(GATE_BASIS[scope]):
            raise RuntimeError(
                f"scope {scope!r} gates on a basis missing a column its bar "
                "names: the bar is challenger WAPE < seasonal-naive D-7 WAPE")


_check_scope_tables()


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


def render_markdown(result: dict) -> str:
    meta, cells = result["meta"], result["gate_cells"]
    passed = sum(cell["gate"]["pass"] for cell in cells)
    registered_cells = meta["registered_cells"]
    lines = [
        f"# Serve-faithful solar retrain gate — registered scope `{meta['scope']}`", "",
        f"**Disposition: {result['verdict']}**", "",
        f"Generated: {meta['generated_at']}",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample gate targets: {meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive).",
        "Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.",
        f"Replica: `{meta['replica_db']}` ({meta['replica_bytes']:,} bytes), opened with SQLite `mode=ro`, `uri=True`.",
        # ABL-345: the two tables disagree — 9 months against 5.6 years of
        # history, and `energy_renewable` zero-fills what `energy_generation`
        # leaves NULL. Two runs of this report are not comparable unless both
        # say which table they read, so it is stated, never defaulted silently.
        f"Target series, features, baselines and contamination screen: `{meta['training_source']}`.",
        "", "## Gate read", "",
        f"Registered scope `{meta['scope']}`: {', '.join(f'{country} {stream}' for stream, country in meta['registered_pairs'])}.",
        f"Gate basis — the columns that must be simultaneously finite for a row to be scored: {', '.join(f'`{c}`' for c in meta['gate_basis'])}. "
        "Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that "
        "does not exist for a pair reads Not measured instead of emptying the cell.",
        f"Strict full PASS requires challenger WAPE < D-7 in all {registered_cells} country × primary D+2-band cells and ≥95% of intended pairs. Result: **{passed}/{registered_cells} cells pass**.",
    ]
    # ABL-379: a measured ABL-253 fact, not a property of every scope. It was
    # rendered unconditionally, so a tranche read would have carried ABL-253's
    # row counts as if they were its own.
    if meta["scope"] == "abl253":
        lines.append("The exact eight registered run instants imply 210/570/720/720/510 selected rows by band. As in ABL-195, the frozen registered minimum for 48–64h remains 456 (95% of 480), while the schedule offers 510 rows.")
    lines.extend([
        "",
        "| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in cells:
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        # ABL-379: a cell that scored no rows has None on both sides. This was a
        # bare division, so the no-incumbent case did not even reach its
        # plausible FAIL — it raised TypeError inside the report writer. It
        # renders as "Not measured", never as a number and never as a crash.
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        lines.append(
            f"| {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill} | {_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} |"
        )
    basis_names = ", ".join(meta["gate_basis"])
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
        # ABL-379: the window was written out as literal text, so a run with any
        # other `--fit-start`/`--gate-end` reported the interval it did not
        # screen. `screen_window` is the interval `_constant_runs` was actually
        # called with.
        screen = meta["screen_window"]
        lines.append(f"- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `{source}` over the registered fit/scoring interval plus 14-day feature lookback ({screen['start']} → {screen['end_exclusive']} UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.")
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
    # ABL-379: these default to the *scope's* registered paths, resolved after
    # parsing. They used to default to ABL-253's three paths for every
    # invocation, so a tranche run that forgot three flags overwrote a
    # dispositioned gate read in place.
    parser.add_argument("--artifact-dir", default=None,
                        help="Override the scope's registered artifact directory")
    parser.add_argument("--json-out", default=None,
                        help="Override the scope's registered results.json path")
    parser.add_argument("--report-out", default=None,
                        help="Override the scope's registered report path")
    # ABL-379: `--scope`, not `--countries`. A country filter cannot express
    # which pairs a run registered, and it leaves the cell bar and the gate
    # basis behind -- which is how the wind harness first reported its pilot as
    # a FAIL (ABL-322). Scoping a run is a new pre-registration; see SCOPES.
    parser.add_argument("--scope", default="abl253", choices=sorted(SCOPES),
                        help="Pre-registered pair set to fit and gate; the registered "
                             "cell count, gate basis and output paths follow from it "
                             "(default: abl253)")
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
    registered_pairs = SCOPES[args.scope]
    gate_basis = GATE_BASIS[args.scope]
    outputs = SCOPE_OUTPUTS[args.scope]
    # The bar is the scope's own size, fixed in the file before the run -- not
    # `len(gate_cells)`, and not 9. A two-pair tranche has a bar of 6, and a
    # pair that yields no rows still shortfalls the count.
    registered_cells = len(registered_pairs) * len(PRIMARY_BANDS)
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
    artifact_dir = Path(args.artifact_dir or outputs["artifact_dir"])
    screen_start = fit_start - pd.Timedelta(days=14)
    training, scored_frames = [], []
    for forecast_type, country in registered_pairs:
        # ABL-342 records provenance from the builder rather than from a source
        # string, so passing the source here is also what makes the artifact's
        # `training_source` truthful.
        builder = RenewableFeatureBuilder(country, forecast_type, screen_start,
                                          gate_end, actuals_source=source)
        fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
        fit, audit = finite_training_rows(fit_raw, FEATURE_COLUMNS)
        model, params = _model()
        model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])

        # ABL-342: through `Forecaster.save`, so the artifact carries the table
        # it was fitted on and the ABL-183 intercept witness by construction.
        path = save_gate_artifact(
            artifact_dir / country / forecast_type / "model.joblib",
            model=model, builder=builder, algorithm=ALGORITHM, params=params,
            feature_columns=FEATURE_COLUMNS, fit_window=(fit_start, gate_start),
        )

        gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)
        gate_finite, gate_audit = finite_training_rows(gate_raw, FEATURE_COLUMNS)
        gate_finite["challenger"] = model.predict(gate_finite[list(FEATURE_COLUMNS)])
        selected = attach_baselines(select_latest_challenger_per_band(gate_finite), builder._actuals)
        # ABL-379: `_load_forecasts` returns a *column-less* frame when no solar
        # row matches at all, so the subscript below raised KeyError rather than
        # leaving the incumbent unmeasured. A scope with no incumbent anywhere
        # is the normal case for the tranche, and it has to read, not crash.
        if incumbent.empty:
            selected["incumbent"] = np.nan
        else:
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
                                                          screen_start, gate_end,
                                                          source),
                         "artifact_path": str(path.resolve()), "artifact_sha256": _sha256(path)})

    def scored(group):
        return scores_with_comparators(group, gate_basis, REPORTED_COMPARATORS)

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

    contaminated = any(row["constant_runs"] for row in training)
    disposition = gate_verdict(gate_cells, registered_cells, contaminated)
    verdict, passed = disposition["verdict"], disposition["passed"]
    if verdict == "PASS":
        recommendation = "The challenger clears the pre-registered D-7 bar in every registered solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue."
    elif verdict.startswith("PERFORMANCE PASS"):
        recommendation = "The challenger clears the performance bar, but a suspect constant run touches the registered data window. Do not promote; send the run to the CEO/ingest owner for adjudication first."
    elif verdict == "UNREADABLE":
        # ABL-379 ports ABL-322's fourth outcome. A cell scoring no rows did not
        # lose a race -- it never ran one, and every one of the 19 new solar
        # pairs starts in exactly that position against the incumbent. Calling
        # it FAIL reads as a model-quality verdict, and the correct response to
        # the two is opposite.
        recommendation = (
            f"No disposition: {disposition['unreadable']}/{registered_cells} primary cells scored zero rows, so the challenger "
            "was never compared to the baseline in them. This is not a model-quality result and must not be reported as one. "
            "Fix the cause of the empty intersection and re-read the gate; the registered windows, bands, metric, baseline and "
            "minimum n are untouched by a run that produced no score."
        )
    else:
        recommendation = f"Do not promote these artifacts: only {passed}/{registered_cells} primary cells clear the registered bar. Report the losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split."

    result = {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
                       "training_source": source,
                       "scope": args.scope, "registered_pairs": list(registered_pairs),
                       "registered_cells": registered_cells, "gate_basis": list(gate_basis),
                       "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
                       "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
                       "screen_window": {"start": str(screen_start), "end_exclusive": str(gate_end)},
                       "registered_intended_n": INTENDED_N, "schedule_implied_n": SCHEDULE_N,
                       "vintage_counts": vintage_counts,
                       "selection": "latest vintage per country + target + model + horizon band"},
              "verdict": verdict, "disposition": disposition,
              "recommendation": recommendation, "training": training,
              "gate_cells": sorted(gate_cells, key=lambda row: (row["country"], row["horizon_band"])),
              "country_d2": sorted(country_d2, key=lambda row: row["country"])}
    json_path = Path(args.json_out or outputs["json_out"])
    report_path = Path(args.report_out or outputs["report_out"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"{verdict}: {passed}/{registered_cells} cells passed; wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
