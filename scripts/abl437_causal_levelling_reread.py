#!/usr/bin/env python3
"""ABL-437: re-read every graded ABL-316 pair under the amended causal levelling.

Arithmetic over the stored `results_*.json` files plus the two ABL-437 trailing
references, recomputed from the replica's target series **on the same rows the
stored cell was scored on**. No refit, no new model, no write to any
dispositioned path, and the replica is opened read-only.

The row set is not assumed, it is *proved*. Each cell's scored rows are rebuilt
from ABL-348's eight registered run instants -- the same `schedule_vintages` and
`horizon_band` the harness uses, latest vintage per (target, band) -- and the
reconstruction is then checked by recomputing that cell's published
`constant_causal` and `climatology_causal` WAPE and MAE from it. A cell whose
published numbers do not come back to 1e-9 is reported NOT RECONSTRUCTIBLE and
graded by nobody; it is not quietly dropped and it is not guessed at.

Order matters and is checkable in git: the amendment is registered in the commit
*before* this file exists (`experiments/ABL437/config.json`,
`reports/abl_437_causal_levelling_registration.md`).

Usage:

    .venv\\Scripts\\python.exe scripts/abl437_causal_levelling_reread.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.gate_grading import (  # noqa: E402
    SIGN_TEST, CellGrade, grade_cell, pair_grade,
)
from src.evaluation.model_free_reference import (  # noqa: E402
    FIT_WINDOW, TRAILING_28D, TRAILING_COMPARATORS, TRAILING_WINDOW_DAYS,
    level_inflation, trailing_reference_levels,
)
from src.evaluation.scorecard import horizon_band, score_predictions  # noqa: E402
from src.evaluation.wind_retrain import (  # noqa: E402
    FEATURE_COLUMNS as WIND_FEATURE_COLUMNS, PRIMARY_BANDS, build_vintage_frame,
    finite_training_rows, schedule_vintages, select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder, _load_actuals_series  # noqa: E402


#: The records this read covers, in tranche order. Every ABL-316 read that
#: carries the ABL-389 reference columns -- which is what makes an arithmetic
#: re-grade possible at all.
#:
#: `results_abl380_tranche1a.json` is deliberately absent and its absence is the
#: same fact ABL-435 was opened on: tranche 1a was fitted before ABL-389 existed,
#: so its record carries `challenger, seasonal_naive, incumbent, persistence` and
#: no reference at all. It cannot be re-graded by arithmetic under *either*
#: levelling, which is why ABL-435 re-read those two pairs as a new scope rather
#: than retro-grading them. `abl435-tranche2f` below is that re-read and is the
#: current record for BG and CH `wind_onshore`.
#:
#: `experiments/ABL376/results_abl376_*.json` are also absent: they belong to
#: ABL-253's registration and its BE/DE/FR solar pairs, not to ABL-316's ledger,
#: and neither is graded in the ABL-316 promotion set this read exists to
#: qualify. Re-reading them is a separate decision for whoever owns that gate.
RECORDS = (
    ("1b", "experiments/ABL348/results_abl381_tranche1b.json", "solar"),
    ("2a", "experiments/ABL348/results_abl405_tranche2a.json", "solar"),
    ("2b", "experiments/ABL348/results_abl406_tranche2b.json", "wind"),
    ("2c", "experiments/ABL348/results_abl419_tranche2c.json", "solar"),
    ("2d", "experiments/ABL348/results_abl421_tranche2d.json", "solar"),
    ("2e", "experiments/ABL348/results_abl417_tranche2e.json", "wind"),
    ("2f", "experiments/ABL348/results_abl435_tranche2f.json", "wind"),
)

#: How exactly a reconstruction has to reproduce the published reference before
#: this read will grade on it. Tight on purpose: the rows either are the rows or
#: they are not, and a tolerance loose enough to absorb a wrong row is loose
#: enough to absorb a wrong answer.
RECONSTRUCTION_TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_pair(cell: dict) -> tuple:
    return (cell.get("forecast_type", "solar"), cell["country"])


def _scored_rows(actuals: pd.Series, gate_start, gate_end) -> pd.DataFrame:
    """Every (target, band) row a gate read scores, latest vintage per band.

    `build_vintage_frame` + `select_latest_challenger_per_band`, with the feature
    columns left out because this read needs the row's identity and not its
    features. Whether that omission changed the row set is exactly what
    :func:`_validate` then measures.
    """
    rows = []
    for target in pd.date_range(gate_start, gate_end, freq="h", inclusive="left"):
        for generated_at in schedule_vintages(target):
            band = horizon_band((target - generated_at).total_seconds() / 3600.0)
            if band is not None:
                rows.append({"target_ts": target, "generated_at": generated_at,
                             "horizon_band": band})
    frame = (pd.DataFrame(rows)
             .sort_values(["target_ts", "horizon_band", "generated_at"])
             .drop_duplicates(["target_ts", "horizon_band"], keep="last")
             .reset_index(drop=True))
    frame["actual"] = [actuals.get(ts, np.nan) for ts in frame["target_ts"]]
    return frame.dropna(subset=["actual"]).reset_index(drop=True)


def _feature_columns(meta: dict, stream: str) -> tuple:
    """The list *that run* fitted on, read out of its own record where it says.

    ABL-395 added `meta.feature_columns` in the same change that moved the solar
    list from 25 to 27, so its absence dates the read (ABL-404's rule) -- and
    resolving it from today's harness instead would rebuild a 2026-08 frame with
    a 2026-09 list. Wind's 24 have not moved.
    """
    recorded = meta.get("feature_columns")
    if recorded:
        return tuple(recorded)
    if stream == "wind":
        return WIND_FEATURE_COLUMNS
    from scripts.evaluate_solar_retrain import LEGACY_FEATURE_COLUMNS
    return LEGACY_FEATURE_COLUMNS


def _rebuilt_rows(country: str, forecast_type: str, source: str, replica: str,
                  fit_start, gate_start, gate_end, feature_columns) -> pd.DataFrame:
    """The gate rows through the harness's own path, features and all.

    :func:`_scored_rows` rebuilds a cell's row set from the schedule alone, which
    is exact wherever no row was dropped -- and 93 of 113 cells are in that
    position. Where a row *was* dropped, only the feature vector knows which:
    `finite_training_rows` runs before `select_latest_challenger_per_band`, so a
    dropped vintage does not merely shrink n, it can promote the next-latest
    vintage into the band and move that row's issue instant. Guessing at that
    would put a wrong `generated_at` under a reference levelled on
    `generated_at`, so this rebuilds it instead. Slower by a feature build per
    pair; correct by construction.
    """
    builder = RenewableFeatureBuilder(country, forecast_type,
                                      fit_start - pd.Timedelta(days=14), gate_end,
                                      actuals_source=source, db_path=replica)
    raw = build_vintage_frame(builder, gate_start, gate_end, feature_columns)
    finite, _ = finite_training_rows(raw, feature_columns)
    selected = select_latest_challenger_per_band(finite)
    return selected[["target_ts", "generated_at", "horizon_band", "actual"]].reset_index(drop=True)


def _reference_columns(frame: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """The stored fit-window levels re-attached, for the reconstruction check."""
    result = frame.copy()
    constant = levels.get("constant_causal")
    result["constant_causal"] = np.nan if constant is None else float(constant)
    hourly = {int(hour): value
              for hour, value in (levels.get("climatology_causal") or {}).items()}
    result["climatology_causal"] = pd.DatetimeIndex(
        result["target_ts"]).hour.map(hourly).astype(float)
    return result


def _validate(frame: pd.DataFrame, cell: dict) -> dict:
    """Does this row set reproduce the cell's own published references?

    Both of them, on WAPE and on MAE. One reference agreeing could be a level
    that happens to fit; the constant and the 24-bucket climatology agreeing on
    two statistics each is the row set.
    """
    checks = {}
    for name in ("constant_causal", "climatology_causal"):
        published = cell["scores"].get(name) or {}
        sub = frame[np.isfinite(frame[name])]
        if published.get("wape_pct") is None or not len(sub):
            checks[name] = published.get("wape_pct") is None and not len(sub)
            continue
        mine = score_predictions(sub["actual"], sub[name])
        checks[name] = bool(
            len(sub) == (cell["comparator_n"].get(name) or 0)
            and abs(mine["wape_pct"] - published["wape_pct"]) < RECONSTRUCTION_TOLERANCE
            and abs(mine["mae"] - published["mae"]) < RECONSTRUCTION_TOLERANCE)
    return checks


def _trailing_scores(frame: pd.DataFrame, actuals: pd.Series) -> tuple[dict, dict]:
    """The two ABL-437 references, scored the way the harness scores comparators.

    Each on its **own** intersection with the row set, carrying its own n -- the
    ABL-322/ABL-378 property. A climatology is 24 levels, so an hour of day
    absent from a trailing window leaves those rows unscored for that column
    alone; it is never filled from a neighbour.
    """
    levels = trailing_reference_levels(actuals, pd.DatetimeIndex(frame["generated_at"]))
    as_of = pd.DatetimeIndex(frame["generated_at"])
    hours = pd.DatetimeIndex(frame["target_ts"]).hour
    constant_name, climatology_name = TRAILING_COMPARATORS
    columns = {
        constant_name: np.array([np.nan if levels[stamp]["constant"] is None
                                 else levels[stamp]["constant"] for stamp in as_of]),
        climatology_name: np.array([levels[stamp]["climatology"].get(int(hour), np.nan)
                                    for stamp, hour in zip(as_of, hours)]),
    }
    scores, counts = {}, {}
    for name, values in columns.items():
        finite = np.isfinite(values)
        if not finite.any():
            scores[name] = {"n": 0, "wape_pct": None, "mae": None, "bias_pct": None,
                            "slope": None, "correlation": None}
            counts[name] = 0
            continue
        scores[name] = score_predictions(frame["actual"][finite], pd.Series(values[finite]))
        counts[name] = int(finite.sum())
    constants = [entry["constant"] for entry in levels.values()
                 if entry["constant"] is not None]
    summary = {"window_days": TRAILING_WINDOW_DAYS, "as_of_count": len(levels),
               "constant_min_mw": min(constants) if constants else None,
               "constant_max_mw": max(constants) if constants else None,
               "constant_mean_mw": float(np.mean(constants)) if constants else None}
    return scores, {"comparator_n": counts, "levels": summary}


def _recorded_or_computed(cell: dict, stream: str) -> CellGrade:
    """The letter this cell carries today.

    A recorded ``grade`` block is read back rather than recomputed -- it is what
    that run decided, and re-deriving it here would be a second implementation of
    the ladder living in a reporting script. Tranches 1b, 2a and 2b predate
    ABL-418 and carry none, so their letters come from the same arithmetic
    ``scripts/abl418_retro_grade.py`` publishes, on the fit-window levelling they
    were read under.
    """
    recorded = cell.get("grade")
    if recorded:
        return CellGrade.from_dict(recorded)
    return grade_cell(cell["scores"], stream, levelling=FIT_WINDOW,
                      g23_readability=SIGN_TEST)


def read(root: Path, replica: str) -> dict:
    fit_start = gate_start = gate_end = None
    tranches, unreadable = [], []
    for label, relative, stream in RECORDS:
        path = root / relative
        record = json.loads(path.read_text(encoding="utf-8"))
        meta = record["meta"]
        fit_start = pd.Timestamp(meta["fit_window"]["start"])
        gate_start = pd.Timestamp(meta["gate_window"]["start"])
        gate_end = pd.Timestamp(meta["gate_window"]["end_exclusive"])
        source = meta["training_source"]
        levels_by_pair = {(row.get("forecast_type", "solar"), row["country"]):
                          row.get("model_free_reference_mw") or {}
                          for row in record["training"]}

        cells_by_pair = {}
        for cell in record["gate_cells"]:
            cells_by_pair.setdefault(_cell_pair(cell), []).append(cell)

        pairs = []
        for pair, cells in sorted(cells_by_pair.items()):
            forecast_type, country = pair
            actuals = _load_actuals_series(country, forecast_type,
                                           fit_start - pd.Timedelta(days=14), gate_end,
                                           source=source, db_path=replica)
            pair_levels = levels_by_pair.get(pair, {})
            rows = _reference_columns(_scored_rows(actuals, gate_start, gate_end), pair_levels)
            # Where the schedule alone does not reproduce a published reference,
            # rows were dropped on a feature this rebuild does not carry. Rebuild
            # the pair through the harness's own path and try again -- once, and
            # only for the pairs that need it.
            route = "schedule"
            if any(not all(_validate(rows[rows["horizon_band"] == cell["horizon_band"]]
                                     .reset_index(drop=True), cell).values())
                   for cell in cells):
                rows = _reference_columns(
                    _rebuilt_rows(country, forecast_type, source, replica,
                                  fit_start, gate_start, gate_end,
                                  _feature_columns(meta, stream)), pair_levels)
                route = "feature-rebuild"
            graded, before, after = [], [], []
            for cell in sorted(cells, key=lambda item: PRIMARY_BANDS.index(item["horizon_band"])):
                band_rows = rows[rows["horizon_band"] == cell["horizon_band"]].reset_index(drop=True)
                checks = _validate(band_rows, cell)
                published = _recorded_or_computed(cell, stream)
                if not all(checks.values()):
                    unreadable.append({"tranche": label, "pair": f"{country} {forecast_type}",
                                       "band": cell["horizon_band"], "checks": checks,
                                       "route": route, "rows_rebuilt": len(band_rows),
                                       "published_n": cell["comparator_n"].get("constant_causal")})
                    graded.append({"band": cell["horizon_band"], "reconstructed": False,
                                   "published_grade": published.label})
                    continue
                trailing, extra = _trailing_scores(band_rows, actuals)
                amended_scores = {**cell["scores"], **trailing}
                # ABL-444: still `sign_test`.  This document's amended column is
                # ABL-437's published result and must keep reproducing; the
                # floored re-read of the same cells is its own record,
                # `reports/abl_444_g23_floor_reread.md`.
                amended = grade_cell(amended_scores, stream, levelling=TRAILING_28D,
                                     g23_readability=SIGN_TEST)
                before.append(published)
                after.append(amended)
                graded.append({
                    "band": cell["horizon_band"], "reconstructed": True, "route": route,
                    "n": int(len(band_rows)),
                    "published_grade": published.label, "amended_grade": amended.label,
                    "published_failed": [name for name, _ in published.failed],
                    "amended_failed": [name for name, _ in amended.failed],
                    "wape": {
                        "challenger": cell["scores"]["challenger"]["wape_pct"],
                        "seasonal_naive": cell["scores"]["seasonal_naive"]["wape_pct"],
                        "constant_causal": (cell["scores"].get("constant_causal") or {}).get("wape_pct"),
                        "constant_causal_28d": trailing["constant_causal_28d"]["wape_pct"],
                        "constant_oracle": (cell["scores"].get("constant_oracle") or {}).get("wape_pct"),
                        "climatology_causal": (cell["scores"].get("climatology_causal") or {}).get("wape_pct"),
                        "climatology_causal_28d": trailing["climatology_causal_28d"]["wape_pct"],
                        "climatology_oracle": (cell["scores"].get("climatology_oracle") or {}).get("wape_pct"),
                    },
                    "level_inflation_pct": {
                        "constant_causal": level_inflation(amended_scores, "constant_causal"),
                        "constant_causal_28d": level_inflation(amended_scores, "constant_causal_28d"),
                    },
                    "skill_pct": {"published": dict(published.skill),
                                  "amended": dict(amended.skill)},
                    "floor_pct": amended.floor_pct,
                    "comparator_n": extra["comparator_n"],
                    "trailing_levels_mw": extra["levels"],
                })
            pairs.append({
                "pair": f"{country} {forecast_type}", "country": country,
                "forecast_type": forecast_type, "route": route,
                "published_pair_grade": pair_grade(before).label if before else "Not measured",
                "amended_pair_grade": pair_grade(after).label if after else "Not measured",
                "cells": graded,
            })
        tranches.append({"tranche": label, "scope": meta["scope"], "stream": stream,
                         "source": source, "record": relative,
                         "record_sha256": _sha256(path), "pairs": pairs})

    return {
        "issue": "ABL-437",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "experiments/ABL437/config.json",
        "levelling_before": FIT_WINDOW, "levelling_after": TRAILING_28D,
        "windows": {"fit_start": str(fit_start), "gate_start": str(gate_start),
                    "gate_end_exclusive": str(gate_end)},
        "replica": replica,
        "replica_bytes": Path(replica).stat().st_size,
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "not_reconstructible": unreadable,
        "tranches": tranches,
    }


def _fmt(value, suffix=""):
    return "Not measured" if value is None else f"{value:.1f}{suffix}"


def render(result: dict) -> str:
    moved, held = [], []
    for tranche in result["tranches"]:
        for pair in tranche["pairs"]:
            entry = (tranche, pair)
            (moved if pair["amended_pair_grade"] != pair["published_pair_grade"]
             else held).append(entry)

    lines = [
        "# ABL-437 — the amended ladder read, applied to every graded ABL-316 pair",
        "",
        f"Generated: {result['generated_at']}. Registration: `{result['registration']}`, "
        "committed before this read existed.",
        "",
        f"Levelling: **`{result['levelling_before']}` → `{result['levelling_after']}`**. "
        "Arithmetic over the stored records plus the two trailing references recomputed on the "
        "same rows — no refit, no new model, replica opened read-only.",
        f"Replica: `{result['replica']}` ({result['replica_bytes']:,} bytes).",
        "",
        "**No committed record is edited by this read.** It is a new document, on the ABL-418 "
        "retro-grade precedent.",
        "",
        "## 1. The row set is proved, not assumed",
        "",
        "Each cell's scored rows are rebuilt from ABL-348's eight registered run instants — the "
        "harness's own `schedule_vintages` and `horizon_band`, latest vintage per (target, band) — "
        "and then **checked by recomputing that cell's published `constant_causal` and "
        "`climatology_causal` WAPE *and* MAE from it**, to "
        f"{result['reconstruction_tolerance']:.0e}. A constant and a 24-bucket climatology agreeing "
        "on two statistics each is the row set; one agreeing alone would not be.",
        "",
    ]
    total = sum(len(pair["cells"]) for tranche in result["tranches"] for pair in tranche["pairs"])
    bad = len(result["not_reconstructible"])
    routes = {}
    for tranche in result["tranches"]:
        for pair in tranche["pairs"]:
            for cell in pair["cells"]:
                routes[cell.get("route")] = routes.get(cell.get("route"), 0) + 1
    lines.append(f"**{total - bad} of {total} cells reconstructed.** "
                 + ("Every cell." if not bad else
                    f"{bad} did not and are reported NOT RECONSTRUCTIBLE below, graded by nobody."))
    lines.extend([
        "",
        f"**{routes.get('schedule', 0)} of them came back on the schedule alone; "
        f"{routes.get('feature-rebuild', 0)} needed the harness's own feature build.** Where a gate "
        "row was dropped for a NaN feature, only the feature vector knows which — and because "
        "`finite_training_rows` runs *before* `select_latest_challenger_per_band`, a dropped vintage "
        "does not merely shrink n, it can promote the next-latest vintage into the band and move "
        "that row's issue instant. Under a reference levelled on `generated_at` that is not a "
        "detail, so those pairs are rebuilt through `RenewableFeatureBuilder` rather than "
        "estimated, on the feature list each record names for itself (`meta.feature_columns`, "
        "whose absence dates the read). Every one of them then reproduces its published references "
        "to the same tolerance.",
    ])
    if bad:
        lines.extend(["", "| tranche | pair | band | rebuilt rows | published n | failed check |",
                      "|---|---|---|---:|---:|---|"])
        for item in result["not_reconstructible"]:
            failed = ", ".join(name for name, ok in item["checks"].items() if not ok)
            lines.append(f"| {item['tranche']} | {item['pair']} | {item['band']} | "
                         f"{item['rows_rebuilt']} | {item['published_n']} | {failed} |")

    lines.extend(["", "## 2. Which pairs the amendment moves", "",
                  f"**{len(moved)} pairs move, {len(held)} hold.**", "",
                  "| tranche | pair | published | amended | what changed | flip margin, tightest-widest |",
                  "|---|---|:---:|:---:|---|---:|"])
    for tranche, pair in moved:
        reasons = sorted({name for cell in pair["cells"] if cell.get("reconstructed")
                          for name in set(cell["amended_failed"]) - set(cell["published_failed"])})
        recovered = sorted({name for cell in pair["cells"] if cell.get("reconstructed")
                            for name in set(cell["published_failed"]) - set(cell["amended_failed"])})
        what = ", ".join(filter(None, [
            f"now fails {', '.join(reasons)}" if reasons else "",
            f"no longer fails {', '.join(recovered)}" if recovered else ""])) or "band mix"
        margins = []
        for cell in pair["cells"]:
            if not cell.get("reconstructed"):
                continue
            for name in set(cell["amended_failed"]) - set(cell["published_failed"]):
                reference = {"G2": "constant_causal_28d", "G3": "climatology_causal_28d"}.get(name)
                value = (cell["skill_pct"]["amended"] or {}).get(reference)
                if value is not None:
                    margins.append(abs(value))
        # The range, not the minimum. A pair grades on its worst band, so one
        # decisive band is enough to move it -- and quoting only the tightest
        # margin would make a pair that fails by 12.9pp in one band and ties in
        # another read as a coin flip.
        span = ((f"{min(margins):.2f}pp" if len(margins) == 1 else
                 f"{min(margins):.2f}-{max(margins):.2f}pp") if margins else "-")
        lines.append(f"| {tranche['tranche']} | {pair['pair']} | {pair['published_pair_grade']} | "
                     f"**{pair['amended_pair_grade']}** | {what} | {span} |")
    if not moved:
        lines.append("| — | — | — | — | no pair changes grade | — |")
    lines.extend([
        "",
        "**Read the last column before reading the letter.** ABL-418 registers G2 and G3 as sign "
        "tests — `skill > 0` — where G1 carries a readability floor (7.51% wind, 10.65% solar at "
        "k=1). So a G2/G3 flip can sit far inside the margin at which one seed can resolve "
        "anything, and several of these do. A flip on a sub-1pp margin means **not demonstrated**, "
        "not *measured worse*. Widening G2/G3 to a floor test would be a second registration "
        "change on top of this one and is not made here; the margin is printed instead.",
    ])

    lines.extend(["", "## 3. Every cell, both levellings", "",
                  "`c` = constant, `clim` = climatology. `28d` is the ABL-437 trailing reference; "
                  "`causal` is the fit-window one, kept and reported. `inflation` is each causal "
                  "reference's WAPE over the oracle constant's — the residual mis-levelling, which "
                  "the amendment reduces rather than removes.", "",
                  "| tranche | pair | band | n | challenger | D-7 | c causal | c 28d | c oracle | "
                  "clim causal | clim 28d | clim oracle | inflation causal / 28d | published | amended |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|"])
    for tranche in result["tranches"]:
        for pair in tranche["pairs"]:
            for cell in pair["cells"]:
                if not cell.get("reconstructed"):
                    lines.append(f"| {tranche['tranche']} | {pair['pair']} | {cell['band']} | — | "
                                 "— | — | — | — | — | — | — | — | — | "
                                 f"{cell['published_grade']} | NOT RECONSTRUCTIBLE |")
                    continue
                wape, infl = cell["wape"], cell["level_inflation_pct"]
                lines.append(
                    f"| {tranche['tranche']} | {pair['pair']} | {cell['band']} | {cell['n']:,} | "
                    f"{_fmt(wape['challenger'], '%')} | {_fmt(wape['seasonal_naive'], '%')} | "
                    f"{_fmt(wape['constant_causal'], '%')} | {_fmt(wape['constant_causal_28d'], '%')} | "
                    f"{_fmt(wape['constant_oracle'], '%')} | {_fmt(wape['climatology_causal'], '%')} | "
                    f"{_fmt(wape['climatology_causal_28d'], '%')} | {_fmt(wape['climatology_oracle'], '%')} | "
                    f"{_fmt(infl['constant_causal'], '%')} / {_fmt(infl['constant_causal_28d'], '%')} | "
                    f"{cell['published_grade']} | {cell['amended_grade']} |")

    lines.extend(["", "## 4. Source records, unchanged", "",
                  "| tranche | scope | source table | record | SHA-256 |", "|---|---|---|---|---|"])
    for tranche in result["tranches"]:
        lines.append(f"| {tranche['tranche']} | `{tranche['scope']}` | `{tranche['source']}` | "
                     f"`{tranche['record']}` | `{tranche['record_sha256'][:16]}…` |")
    lines.extend(["", "Read-only. This script writes to no path any gate read owns.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ABL-437: the amended ladder read.")
    parser.add_argument("--replica-db")
    parser.add_argument("--render-only", action="store_true",
                        help="re-render the stored JSON without re-measuring")
    parser.add_argument("--repo-root", default=str(Path(__file__).parent.parent))
    parser.add_argument("--json-out", default="reports/abl_437_causal_levelling_reread.json")
    parser.add_argument("--report-out", default="reports/abl_437_causal_levelling_reread.md")
    args = parser.parse_args()
    if not args.render_only and not args.replica_db:
        parser.error("--replica-db is required unless --render-only is given")

    root = Path(args.repo_root)
    if args.render_only:
        # Re-render a completed read without re-measuring it. The measurement is
        # in the JSON; re-running the replica read to move a column would be
        # seven minutes spent proving nothing.
        result = json.loads((root / args.json_out).read_text(encoding="utf-8"))
    else:
        result = read(root, args.replica_db)
        (root / args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / args.report_out).write_text(render(result), encoding="utf-8")
    print(f"wrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
