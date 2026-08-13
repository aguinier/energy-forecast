#!/usr/bin/env python3
"""Fit and read a pre-registered serve-faithful wind gate (see SCOPES).

The default scope is `abl195`, so an unflagged run reproduces that gate exactly;
`--scope` selects any other registered pair set, and carries its own output paths
with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.data_quality import find_suspect_constant_runs
from src.evaluation.model_free_reference import (
    MODEL_FREE_COMPARATORS, attach_model_free_references, comparator_wape,
    levels_table, lost_to_a_model_free_reference, reference_prose,
)
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.gate_registration import (
    check_registration_tables, check_scope_outputs,
)
from src.evaluation.scorecard import (
    ScorecardConfig, _load_forecasts, _load_tso, _ro_connect,
    describe_opened_databases, opened_databases,
    select_latest_per_band,
)
from src.evaluation.wind_retrain import (
    FEATURE_COLUMNS, INTENDED_N, PRIMARY_BANDS, SCHEDULE_N, attach_baselines,
    build_vintage_frame, finite_training_rows, gate_cell,
    scored_with_comparators, select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder


ALGORITHMS = {"wind_offshore": "xgboost", "wind_onshore": "catboost"}

# A scope is a *pre-registration*, not a filter.  The first cut of ABL-322 added
# a `--countries` filter over one shared PAIRS, which is wrong in both
# directions: it left the cell bar at a hardcoded 15 while the filter changed
# how many cells a run produces, and it selected countries without selecting
# streams, so `--countries DE,NL` also refitted the serving DE wind_onshore pair
# and mixed it into the offshore gate.
#
# Each scope therefore names its pairs outright, and the registered cell count is
# derived from *that table* -- fixed in the file before the run -- rather than
# from whatever the run turns out to score.  This keeps the property the
# hardcoded 15 existed to protect: a pair that silently yields no gate rows still
# falls short of its scope's count and reads FAIL, instead of quietly leaving the
# denominator.  Adding a scope is a pre-registration and belongs in review.
SCOPES = {
    # ABL-195 as registered: offshore BE/FR, onshore BE/DE/FR.  5 pairs x 3
    # primary bands = 15 cells.  Unchanged, and the default, so an unflagged run
    # still reproduces ABL-195 exactly.
    "abl195": (("wind_offshore", "BE"), ("wind_offshore", "FR"),
               ("wind_onshore", "BE"), ("wind_onshore", "DE"), ("wind_onshore", "FR")),
    # ABL-322 offshore pilot: DE and NL wind_offshore only, fitted on
    # `energy_generation`.  2 pairs x 3 bands = 6 cells.  No onshore pair -- no
    # currently-serving model is refitted by this scope.
    "abl322-pilot": (("wind_offshore", "DE"), ("wind_offshore", "NL")),
    # ABL-380 -- ABL-316 tranche 1a: BG and CH wind_onshore, fitted on
    # `energy_generation` under the frozen registration at
    # `experiments/ABL348/config.json`.  2 pairs x 3 bands = 6 cells.
    #
    # This entry *is* the tranche's pre-registration, in exactly the sense the
    # comment above describes: the pair list is fixed here, in the file, and
    # committed before the first fit, so the cell bar cannot follow what the run
    # turned out to score.  Windows, metric, baseline, minimum n and source
    # table are ABL-348's and are deliberately not restated here -- thirty-seven
    # tranches must not become thirty-seven chances to shop a window.
    #
    # Onshore, but nothing here is a migration: BG and CH serve no wind model,
    # so this scope refits no live pair -- the same property `abl322-pilot`
    # holds, reached differently.
    #
    # CH is registered build-and-report by the CEO's decision on ABL-348: a
    # 12.9 MW gate-window mean cannot carry a promotion decision either way.
    # That is a *reading* rule and not a scoring one -- CH is fitted and scored
    # identically to BG, and the small-denominator caveat belongs in the
    # evidence pack rather than in this table.
    "abl380-tranche1a": (("wind_onshore", "BG"), ("wind_onshore", "CH")),
}

# ABL-387: where a scope writes is part of its registration, not a flag default.
# These three paths used to be fixed strings on the arguments themselves --
# ABL-195's -- resolved before `--scope` was ever consulted, so a scoped run that
# omitted three flags overwrote a *dispositioned* gate read in place.  That run
# succeeds and emits a full report; the damage is to evidence.
#
# Entries stay exactly one directory deep under `experiments/`, because
# `.gitignore:53-56` globs `experiments/*/results.json` and
# `experiments/*/artifacts/`.  A nested path slips past both globs and would
# commit a binary model artifact.
#
# Depth is necessary but not sufficient, and the two globs do not key on the
# same thing: `experiments/*/artifacts/` matches on the *directory name*, so any
# entry ending `artifacts` one level deep is ignored, while
# `experiments/*/results.json` matches on the *exact filename*.  A `json_out`
# named anything else is therefore tracked -- which is deliberate for
# `abl380-tranche1a` below, and is the one channel in which the overwrite this
# table prevents would have been visible to review at all.
SCOPE_OUTPUTS = {
    # Byte-for-byte the paths ABL-195 was read at, so an unflagged run still
    # reproduces the dispositioned read rather than relocating it.
    "abl195": {"artifact_dir": "experiments/ABL195/artifacts",
               "json_out": "experiments/ABL195/results.json",
               "report_out": "reports/abl_195_wind_retrain.md"},
    # The paths the pilot was actually read at: `experiments/ABL322/config.json`
    # registers `experiments/ABL322/artifacts` as its artifact location, and
    # `reports/abl_322_pilot_gate.md` is that run's committed report.
    "abl322-pilot": {"artifact_dir": "experiments/ABL322/artifacts",
                     "json_out": "experiments/ABL322/results.json",
                     "report_out": "reports/abl_322_pilot_gate.md"},
    # ABL-380, registered while this fix sat in review -- which is the check
    # above working rather than a merge accident: `abl380-tranche1a` landed in
    # `SCOPES` and `GATE_BASIS` at 69f8cd5, and merging it here raised
    # `SCOPE_OUTPUTS is missing 'abl380-tranche1a'` at import.  GitHub reported
    # the same merge MERGEABLE/CLEAN; the tables disagreeing is not a textual
    # conflict, so this is the only thing that could have caught it.
    #
    # These are the paths that run *actually wrote*, measured rather than
    # assigned: the two `model.joblib` files under `experiments/ABL348/artifacts`
    # hash to eb0f63d8...43ea (BG) and 5d2ec407...0840 (CH), the two SHA-256
    # values published in the gate report's fit-audit table.  The tranche writes
    # under ABL348 rather than a directory of its own because the registration it
    # is fitted under is `experiments/ABL348/config.json`, frozen at ABL-348.
    #
    # The `json_out` is deliberately not named `results.json`: at that name the
    # `.gitignore` glob would swallow it, and it is the machine record
    # `reports/abl_380_tranche1a_findings.md:9` cites for a PASS the Board has
    # been asked to review.  It is committed, and must stay reachable at the
    # path that report names.
    "abl380-tranche1a": {"artifact_dir": "experiments/ABL348/artifacts",
                         "json_out": "experiments/ABL348/results_abl380_tranche1a.json",
                         "report_out": "reports/abl_380_wind_onshore_tranche1a.md"},
}
COLUMNS = {"wind_offshore": "wind_offshore_mw", "wind_onshore": "wind_onshore_mw"}

# The columns that must be *simultaneously finite* for a row to enter a gate
# cell.  This is a registered property of the scope, not a detail: ABL-322 ran
# with the four-way basis below and every one of its 6 cells came back n=0, all
# scores None, verdict FAIL -- because DE and NL wind_offshore have zero rows in
# `forecasts`, so `incumbent` is NaN on every row and the intersection is empty.
# That FAIL reports a race that was never run, and it would land the same way on
# every new country in ABL-316's 37 remaining pairs.
#
# The registered bar names challenger and seasonal-naive D-7 only -- in ABL-195's
# registration and in ABL-322's -- so the pilot gates on exactly those.  ABL-195
# keeps the four-way basis it was actually read under: its published 48-64h cells
# scored 480 rows against the 510 the same report records as selected, so the
# incumbent conjunct dropped rows there and re-basing it would silently move
# already-dispositioned numbers.  Re-reading ABL-195 under a narrower basis is a
# separate decision for whoever owns that gate, not a side effect of this pilot.
GATE_BASIS = {
    "abl195": ("challenger", "incumbent", "seasonal_naive", "persistence"),
    "abl322-pilot": ("challenger", "seasonal_naive"),
    # ABL-380: BG and CH wind_onshore hold zero rows in `forecasts`, which is
    # the normal condition of all 37 remaining ABL-316 pairs rather than a fault
    # of these two.  Under the four-way basis every one of the 6 cells would
    # intersect to n=0 and the run would render FAIL -- the ABL-322 failure
    # verbatim, on the first tranche it was found in time to prevent.  Gates on
    # the two columns the registered bar names; the incumbent is still reported
    # on its own intersection, where it reads "Not measured".
    "abl380-tranche1a": ("challenger", "seasonal_naive"),
}
#: Always reported, each on its own intersection with the gate basis, so that a
#: comparator which never exists reads "Not measured" instead of voiding the gate.
#:
#: ABL-389 adds four model-free predictors -- a flat line and an hour-of-day
#: climatology, each in a causal and an oracle form.  They are *reported
#: references and not gate criteria* -- deliberately absent from every
#: `GATE_BASIS` entry above, pinned by `test_gate_model_free_reference.py`.
#: ABL-380 passed 6/6 and reported against its own pass that CH wind_onshore
#: scored 47.42% while a flat line at the gate-window median scored 40.29%, and
#: that BG's registered D-7 bar of 93.75% is cleared outright by a constant at
#: the fit-window mean (82.77%) with no model at all.  The registered D-7 bar was
#: set on pairs that had incumbents and real seasonal structure; on a
#: low-capacity-factor onshore pair it certifies close to nothing, and 33 more
#: such pairs are queued behind this.  So every read now prints what its PASS is
#: worth beside the PASS.
#:
#: The climatology is here because the constant alone was measured and found
#: insufficient: on solar it scores 63-95% on every cell, since a flat line
#: cannot represent a diurnal cycle at all.  On wind it is merely loose -- CH's
#: oracle climatology is 38.20% against the constant's 40.29%, which widens the
#: ABL-380 finding from 7.1pp to 9.2pp.  Both are kept because the gap between
#: them is the read: the constant asks whether the model predicts the level, the
#: climatology whether it predicts the level and the daily shape.
#:
#: A pair that clears D-7 but loses to one of these still reads PASS.  Moving a
#: bar after seeing a result is what the pre-registration apparatus exists to
#: prevent, and a conservative direction does not exempt it.
REPORTED_COMPARATORS = ("challenger", "incumbent", "seasonal_naive", "persistence",
                        *MODEL_FREE_COMPARATORS)

# ABL-387: the three tables above are one registration in three views.  Checked
# at import, so a scope registered in one and not the others fails before any fit
# -- and identically under `--help` and in the test suite -- rather than raising
# `KeyError` partway through a gate run, or writing over another scope's evidence.
check_registration_tables(SCOPES=SCOPES, GATE_BASIS=GATE_BASIS, SCOPE_OUTPUTS=SCOPE_OUTPUTS)
check_scope_outputs(SCOPE_OUTPUTS)


def _model(algorithm: str):
    params = config.get_default_params(algorithm)
    if algorithm == "xgboost":
        # The production defaults carry an early-stopping setting that requires
        # a validation set.  This pre-registered final fit uses every pre-gate
        # row and performs no gate-driven tuning, so early stopping is disabled.
        params.pop("early_stopping_rounds", None)
        return XGBRegressor(**params), params
    return CatBoostRegressor(**params), params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_runs(replica: str, country: str, forecast_type: str, start, end,
                   source: Optional[str] = None) -> list[dict]:
    # ABL-322: the contamination audit has to read the same table the model was
    # fitted on.  Hardcoding `energy_renewable` here while the builder trains
    # from `energy_generation` reports zero-fill runs for a series nothing used.
    table = source or db.RENEWABLE_TYPE_SOURCE_TABLE
    if table not in db._RENEWABLE_TYPE_SOURCES:
        raise ValueError(f"unknown renewable source table: {table!r}")
    con = _ro_connect(replica)
    try:
        df = pd.read_sql_query(
            f"SELECT timestamp_utc, {COLUMNS[forecast_type]} AS value FROM {table} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? ORDER BY timestamp_utc",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return [{"start": str(run.start), "end": str(run.end), "value": run.value,
             "n_rows": run.n_rows, "duration_hours": run.duration_hours}
            for run in find_suspect_constant_runs(df, "value")]


def _fmt(value, suffix=""):
    return "Not measured" if value is None else f"{value:.1f}{suffix}"


def render_markdown(result: dict) -> str:
    meta = result["meta"]
    cells = result["gate_cells"]
    lines = [
        f"# Serve-faithful wind retrain gate — registered scope `{meta['scope']}`",
        "",
        f"**Disposition: {result['verdict']}**",
        "",
        f"Generated: {meta['generated_at']}",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample gate targets: {meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive).",
        "Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.",
        # ABL-355: which *files* the run opened. `--replica-db` used to cover
        # the incumbent, TSO and screen only; the fitted series and weather came
        # from `ENERGY_DB_PATH`, and this heading named one path for two files.
        *describe_opened_databases(meta["databases"], meta["replica_bytes"]),
        "",
        "## Gate read",
        "",
        f"Registered scope `{meta['scope']}`: {', '.join(f'{c} {t}' for t, c in meta['registered_pairs'])}.",
        f"Target series, features, baselines and contamination screen: `{meta['training_source']}`.",
        f"Gate basis — the columns that must be simultaneously finite for a row to be scored: {', '.join(f'`{c}`' for c in meta.get('gate_basis', []))}. "
        "Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that "
        "does not exist for a pair reads Not measured instead of emptying the cell.",
        f"Strict full PASS requires challenger WAPE < D-7 in all {meta['registered_cells']} country × primary D+2-band cells and ≥95% of intended pairs. Result: **{sum(c['gate']['pass'] for c in cells)}/{meta['registered_cells']} cells pass**.",
        "",
        # ABL-389.  These four columns are references, not criteria: they are in
        # `REPORTED_COMPARATORS` and in no `GATE_BASIS` entry, so the PASS rule,
        # the bands, the bars and the windows are exactly what they were.  What
        # they add is the number that qualifies a PASS — ABL-380's CH cleared
        # all 3 cells at 47.42% while a flat line at the gate-window median
        # scored 40.29% and an hour-of-day median scored 38.20%, and BG's 93.75%
        # D-7 bar is cleared by a causal constant at 82.77% with no model at all.
        *reference_prose(),
        "",
        "| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in cells:
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        # A cell that scored no rows has None on both sides. It renders as
        # "Not measured", never as a number and never as a crash.
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        lines.append(
            f"| {row['forecast_type']} | {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill} | {_fmt(comparator_wape(scores, 'constant_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_oracle'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_oracle'), '%')} | "
            f"{_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} |"
        )
    lines.extend(levels_table(result["training"], key="forecast_type"))
    basis_names = ", ".join(meta.get("gate_basis", []))
    lines.extend(["", "## Per-country all-D+2 summary", "",
                  f"Gate-basis values (actual, {basis_names}) share one finite intersection; each comparator outside the basis is "
                  "scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing "
                  "`Not measured` had no finite rows at all.", "",
                  "| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result["country_d2"]:
        s = row["scores"]
        tso = row["tso"]
        lines.append(f"| {row['forecast_type']} | {row['country']} | {row['n']:,} | {_fmt(s['challenger']['wape_pct'], '%')} | "
                     f"{_fmt(s['seasonal_naive']['wape_pct'], '%')} | {_fmt(s['persistence']['wape_pct'], '%')} | "
                     f"{_fmt(comparator_wape(s, 'constant_causal'), '%')} | {_fmt(comparator_wape(s, 'constant_oracle'), '%')} | "
                     f"{_fmt(comparator_wape(s, 'climatology_causal'), '%')} | {_fmt(comparator_wape(s, 'climatology_oracle'), '%')} | "
                     f"{_fmt(s['incumbent']['wape_pct'], '%')} | {_fmt(tso['wape_pct'], '%')} (n={tso['n']:,}) |")
    # ABL-322 criterion 3.  The protocol-count sentence this replaces was a
    # measured ABL-195 fact (210/570/720/720/510 by band) rendered for every
    # scope; the per-cell `n` column above already carries that truth for
    # whichever scope actually ran.
    if meta["scope"] == "abl195":
        lines.extend([
            "Protocol count check (before fitting): the exact eight registered run instants produce 210/570/720/720/510 selected rows by band, not the registered 240/600/720/720/480. The primary 24–36h and 36–48h counts reproduce; 48–64h has 510 rows and is still judged against the frozen registered minimum of 456.",
            "",
        ])
    if any("timings_sec" in row for row in result["training"]):
        lines.extend(["## Training cost", "",
                      "Wall-clock on the rail interpreter, one pair at a time in a single process. "
                      "Feature build and fit are separated because they scale on different things. "
                      "Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.", "",
                      "| type | country | fit rows | feature build | fit | gate build + predict | pair total |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in result["training"]:
            t = row.get("timings_sec")
            if not t:
                continue
            lines.append(f"| {row['forecast_type']} | {row['country']} | {row.get('fit_rows', 0):,} | "
                         f"{t['fit_feature_build']:.1f} s | {t['fit']:.1f} s | {t['gate_build_and_predict']:.1f} s | "
                         f"**{t['pair_total']:.1f} s** |")
        total = sum(r["timings_sec"]["pair_total"] for r in result["training"] if r.get("timings_sec"))
        lines.append("")
        lines.append(f"Scope total across {len(result['training'])} pair(s): **{total:.1f} s** "
                     f"({total / max(len(result['training']), 1):.1f} s mean per pair).")
    lines.extend(["", "## Fit and missingness audit", "",
                  "Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.", "",
                  "| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for row in result["training"]:
        a = row["audit"]
        lines.append(f"| {row['forecast_type']} | {row['country']} | {row['algorithm']} | {a['retained_rows']:,} / {a['intended_rows']:,} | "
                     f"{a['unique_targets']:,} | {a['excluded_missing_actual_or_feature']:,} | {a['degraded_lag_1d_rows']:,} | `{row['artifact_sha256']}` |")
    lines.extend(["", "## Data quality and limits", ""])
    contaminated = [r for r in result["training"] if r["constant_runs"]]
    if contaminated:
        # ABL-322: this used to render one hardcoded sentence about a BE
        # offshore zero run, for any scope -- including scopes that never fit
        # BE.  The screen already returns the runs it found; name those.
        lines.append(f"- ABL-188 constant-run screening found suspect runs in {len(contaminated)} fitted pair(s), "
                     f"read against `{meta['training_source']}` — the table these pairs were fitted on. "
                     "Those labels and any dependent feature rows were treated as missing before fit. "
                     "Promotion remains on hold pending CEO/ingest adjudication.")
        lines.extend(["", "| type | country | run start | run end | value | rows | hours |",
                      "|---|---|---|---|---:|---:|---:|"])
        for row in contaminated:
            for run in row["constant_runs"]:
                lines.append(f"| {row['forecast_type']} | {row['country']} | {run['start']} | {run['end']} | "
                             f"{run['value']:.1f} MW | {run['n_rows']:,} | {run['duration_hours']:.0f} |")
        lines.append("")
    else:
        lines.append("- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.")
    lines.extend([
        "- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.",
        "- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.",
        "- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.",
    ])
    # The registered bar is "beats D-7", and on offshore wind D-7 is close to
    # uninformative -- so a cell can pass with almost no dynamic skill. Say that
    # in the report rather than leaving a PASS to imply the model is good.
    for row in result["country_d2"]:
        chal, tso = row["scores"]["challenger"]["wape_pct"], row["tso"]["wape_pct"]
        if chal is not None and tso is not None and tso < chal:
            lines.append(
                f"- **{row['country']} {row['forecast_type']}: the TSO forecast is better than the challenger** "
                f"({tso:.1f}% vs {chal:.1f}% WAPE over the same n={row['n']:,}). The gate is against D-7 and this pair clears it, "
                "but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as "
                "a feature to ingest, not merely as context.")
    # ABL-389: the same service for the model-free references. ABL-380 reported
    # 6/6 PASS on a pair whose challenger was 7.1pp worse than a flat line and
    # 9.2pp worse than an hour-of-day median; the first number was in the
    # evidence pack only because a human went looking, and the second was in no
    # evidence pack at all.
    lines.extend(lost_to_a_model_free_reference(
        cells, lambda row: f"{row['country']} {row['forecast_type']} {row['horizon_band']}"))
    lines.extend([
        "- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.",
        "",
        "## Recommendation to the CEO",
        "",
        result["recommendation"],
        "",
        "No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=str(config.FORECAST_OUTPUT_DB))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    # ABL-387: default to the *scope's* registered paths, resolved after parsing.
    # A fixed default here is resolved before `--scope` is read, which is how a
    # scoped run came to overwrite ABL-195's dispositioned artifacts in place.
    parser.add_argument("--artifact-dir", default=None,
                        help="Override the scope's registered artifact directory")
    parser.add_argument("--json-out", default=None,
                        help="Override the scope's registered results.json path")
    parser.add_argument("--report-out", default=None,
                        help="Override the scope's registered report path")
    # ABL-322: the pilot gates DE/NL wind_offshore off `energy_generation`.
    # Both stay opt-in so an unflagged run reproduces ABL-195 exactly.
    parser.add_argument("--scope", default="abl195", choices=sorted(SCOPES),
                        help="Pre-registered pair set to fit and gate; the registered "
                             "cell count follows from it (default: abl195)")
    parser.add_argument("--renewable-source", default=None,
                        choices=list(db._RENEWABLE_TYPE_SOURCES),
                        help="Source table for the fitted series, its features and the "
                             f"contamination audit (default: {db.RENEWABLE_TYPE_SOURCE_TABLE})")
    args = parser.parse_args()
    registered_pairs = SCOPES[args.scope]
    outputs = SCOPE_OUTPUTS[args.scope]
    registered_cells = len(registered_pairs) * len(PRIMARY_BANDS)
    fit_start, gate_start, gate_end = map(pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")

    cfg = ScorecardConfig(str(replica), args.sidecar_db, gate_start, gate_end,
                          models={"wind_offshore": "xgboost", "wind_onshore": "catboost"})
    incumbent_raw, vintage_counts = _load_forecasts(cfg)
    incumbent = select_latest_per_band(incumbent_raw)
    artifact_dir = Path(args.artifact_dir or outputs["artifact_dir"])
    training, scored_frames, tso_by_type = [], [], {}
    for forecast_type, country in registered_pairs:
        if forecast_type not in tso_by_type:
            tso_by_type[forecast_type] = _load_tso(cfg, forecast_type)
        tso = tso_by_type[forecast_type]
        algorithm = ALGORITHMS[forecast_type]
        # ABL-342 records provenance from the builder, not a source string,
        # so passing the source here is what makes the artifact truthful.
        # ABL-322 acceptance criterion 3: a per-pair cost figure, so the 37
        # pairs behind this pilot can be sized in sittings rather than guessed.
        # Feature build and fit are timed apart because they scale on different
        # things -- the build on the number of vintages and the country's
        # source resolution, the fit on retained rows x n_estimators -- so a
        # tranche estimate that lumps them together mis-sizes both.
        t0 = time.perf_counter()
        # ABL-355: hand the builder the resolved replica, so `--replica-db`
        # means the whole run. Without it the builder read
        # `config.DATABASE_PATH` and the fitted series could come from a
        # different file than the incumbent it is scored against.
        # `actuals_source` names the *table* and `db_path` the *file*; neither
        # implies the other, so both are passed.
        builder = RenewableFeatureBuilder(country, forecast_type,
                                           fit_start - pd.Timedelta(days=14), gate_end,
                                           actuals_source=args.renewable_source,
                                           db_path=str(replica))
        fit_raw = build_vintage_frame(builder, fit_start, gate_start)
        fit, audit = finite_training_rows(fit_raw)
        t_build = time.perf_counter() - t0
        model, params = _model(algorithm)
        t0 = time.perf_counter()
        model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])
        t_fit = time.perf_counter() - t0

        # ABL-342: through `Forecaster.save`, so the artifact carries the
        # table it was fitted on and the ABL-183 intercept witness by
        # construction.
        path = save_gate_artifact(
            artifact_dir / country / forecast_type / "model.joblib",
            model=model, builder=builder, algorithm=algorithm,
            params=params, feature_columns=FEATURE_COLUMNS,
            fit_window=(fit_start, gate_start),
        )

        t0 = time.perf_counter()
        gate_raw = build_vintage_frame(builder, gate_start, gate_end)
        gate_finite, gate_audit = finite_training_rows(gate_raw)
        gate_finite["challenger"] = model.predict(gate_finite[list(FEATURE_COLUMNS)])
        t_gate = time.perf_counter() - t0
        selected = select_latest_challenger_per_band(gate_finite)
        selected = attach_baselines(selected, builder._actuals)
        # ABL-389: from the same ABL-188-filtered series the baselines and the
        # gate actuals come from, so the reference is arithmetic on data already
        # loaded -- no refit, no second read, no extra fetch.
        selected, reference_levels = attach_model_free_references(
            selected, builder._actuals, fit_start, gate_start, gate_end)
        inc = incumbent[(incumbent["forecast_type"] == forecast_type) &
                        (incumbent["country_code"] == country)][
                            ["target_ts", "horizon_band", "forecast_value"]].rename(
                                columns={"forecast_value": "incumbent"})
        selected = selected.merge(inc, on=["target_ts", "horizon_band"], how="left")
        selected = selected.merge(tso[tso["country_code"] == country][["target_ts", "tso"]],
                                  on="target_ts", how="left")
        selected["country"] = country
        selected["forecast_type"] = forecast_type
        scored_frames.append(selected)
        training.append({"forecast_type": forecast_type, "country": country,
                         "algorithm": algorithm, "params": params,
                         "audit": audit, "gate_build_audit": gate_audit,
                         "model_free_reference_mw": reference_levels,
                         "constant_runs": _constant_runs(str(replica), country, forecast_type,
                                                           fit_start - pd.Timedelta(days=14), gate_end,
                                                           source=args.renewable_source),
                         "timings_sec": {"fit_feature_build": round(t_build, 1),
                                         "fit": round(t_fit, 1),
                                         "gate_build_and_predict": round(t_gate, 1),
                                         "pair_total": round(t_build + t_fit + t_gate, 1)},
                         "fit_rows": int(len(fit)),
                         "artifact_path": str(path.resolve()), "artifact_sha256": _sha256(path)})

    gate_basis = GATE_BASIS[args.scope]

    def scored(group):
        return scored_with_comparators(group, gate_basis, REPORTED_COMPARATORS)

    all_scored = pd.concat(scored_frames, ignore_index=True)
    gate_cells, country_d2 = [], []
    for (forecast_type, country, band), group in all_scored.groupby(["forecast_type", "country", "horizon_band"]):
        scores, common, comparator_n = scored(group)
        if band in PRIMARY_BANDS:
            gate_cells.append({"forecast_type": forecast_type, "country": country,
                               "horizon_band": band, "scores": scores,
                               "comparator_n": comparator_n,
                               "gate": gate_cell(scores["challenger"]["wape_pct"],
                                                 scores["seasonal_naive"]["wape_pct"],
                                                 len(common), INTENDED_N[band])})
    for (forecast_type, country), group in all_scored[all_scored["horizon_band"].isin(PRIMARY_BANDS)].groupby(["forecast_type", "country"]):
        scores, common, comparator_n = scored(group)
        tso_valid = np.isfinite(common[["actual", "tso"]].to_numpy(dtype=float)).all(axis=1)
        from src.evaluation.scorecard import score_predictions
        tso_score = score_predictions(common.loc[tso_valid, "actual"], common.loc[tso_valid, "tso"])
        country_d2.append({"forecast_type": forecast_type, "country": country,
                           "n": len(common), "scores": scores,
                           "comparator_n": comparator_n, "tso": tso_score})

    passed = sum(row["gate"]["pass"] for row in gate_cells)
    contaminated = any(row["constant_runs"] for row in training)
    performance_pass = len(gate_cells) == registered_cells and passed == registered_cells
    if performance_pass and contaminated:
        verdict = "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION"
        recommendation = (
            "The challenger clears the pre-registered D-7 performance bar in every served D+2 country-band cell. "
            "Do not promote yet: hand the constant runs tabulated below to the CEO/ingest owner for adjudication, "
            "then return these experiment artifacts and this evidence to the CEO for Board review. This issue does not promote them."
        )
    elif performance_pass:
        verdict = "PASS"
        recommendation = (
            "The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these "
            "experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue."
        )
    else:
        # A cell scoring no rows did not lose a race -- it never ran one.
        # Calling that FAIL reads as a model-quality verdict and is how this
        # harness first reported the ABL-322 pilot.
        unreadable = [row for row in gate_cells if row["gate"]["n"] == 0]
        if unreadable:
            verdict = "UNREADABLE"
            recommendation = (
                f"No disposition: {len(unreadable)}/{registered_cells} primary cells scored zero rows, so the challenger was "
                "never compared to the baseline in them. This is not a model-quality result and must not be reported as one. "
                "Fix the cause of the empty intersection and re-read the gate; the registered windows, bands, metric, baseline "
                "and minimum n are untouched by a run that produced no score."
            )
        else:
            verdict = "FAIL"
            recommendation = (
                f"Do not promote these artifacts: only {passed}/{registered_cells} primary cells clear the registered bar. Treat the losing "
                "country/bands as a model-quality finding and move next to stronger wind features/model selection on a fresh pre-registered split."
            )
    result = {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
                       # ABL-355: the run's files, not just its tables.
                       "databases": opened_databases(cfg, str(replica), config.DATABASE_PATH),
                       "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
                       "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
                       "scope": args.scope, "registered_pairs": list(registered_pairs),
                       "registered_cells": registered_cells, "gate_basis": list(gate_basis),
                       # ABL-389: the basis is what gates; this is what is
                       # merely reported beside it. Recorded so the two are
                       # distinguishable in the record and not only in the prose.
                       "reported_comparators": list(REPORTED_COMPARATORS),
                       "training_source": args.renewable_source or db.RENEWABLE_TYPE_SOURCE_TABLE,
                       "registered_intended_n": INTENDED_N, "schedule_implied_n": SCHEDULE_N,
                       "vintage_counts": vintage_counts,
                       "selection": "latest vintage per country + target + model + horizon band"},
              "verdict": verdict, "recommendation": recommendation,
              "training": training, "gate_cells": sorted(gate_cells, key=lambda r: (r["forecast_type"], r["country"], r["horizon_band"])),
              "country_d2": sorted(country_d2, key=lambda r: (r["forecast_type"], r["country"]))}
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
