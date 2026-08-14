#!/usr/bin/env python3
"""ABL-443: re-read DE/NL `wind_offshore` at ABL-437's trailing causal reference.

ABL-436 graded both pairs **A** under scope `abl322-pilot`, whose G2 and G3 read
`constant_causal` / `climatology_causal` -- levelled on ABL-348's fit window and
scored on its gate window, which ABL-437 identifies as different seasons. Both
pairs are inflated 18-27% on that reference. This restates both grades against
`constant_causal_28d` / `climatology_causal_28d` instead, under a **new scope id**,
`abl443-offshore-trailing`.

Registration: `experiments/ABL443/config.json` and
`reports/abl_443_trailing_reference_registration.md`, committed before this file
exists, so the order is checkable in git.

**No refit.** The challenger, D-7, persistence, both fit-window references and both
oracle references are read out of ABL-436's committed record as they stand; only the
two trailing references are computed, on the same rows. **No committed record is
edited**: this writes to `reports/abl_443_*` and refuses to write anywhere ABL-436,
ABL-437 or ABL-322 published.

The row-set reconstruction is ABL-437's, imported rather than reimplemented -- a
second copy of that logic living in a second reporting script is exactly how two
reads of one cell come to disagree about which rows it had.

Usage:

    .venv\\Scripts\\python.exe scripts/abl443_offshore_trailing_reread.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.abl437_causal_levelling_reread import (  # noqa: E402
    RECONSTRUCTION_TOLERANCE, _feature_columns, _rebuilt_rows, _recorded_or_computed,
    _reference_columns, _scored_rows, _sha256, _trailing_scores, _validate,
)
from src.evaluation.gate_grading import (  # noqa: E402
    LADDER_REFERENCES, grade_cell, pair_grade, readability_floor_pct,
)
from src.evaluation.model_free_reference import (  # noqa: E402
    FIT_WINDOW, TRAILING_28D, TRAILING_WINDOW_DAYS, comparator_wape, level_inflation,
)
from src.evaluation.wind_retrain import PRIMARY_BANDS  # noqa: E402
from src.wind_features import _load_actuals_series  # noqa: E402

#: This read's own scope id. It is not `abl322-pilot`: that scope is published and
#: pinned to `fit_window` in `CAUSAL_LEVELLING`, and re-reading it in place would
#: make a committed page of letters disagree with the run that regenerates it.
SCOPE = "abl443-offshore-trailing"

#: The record this read is arithmetic over, and the only one it opens.
RECORD = "experiments/ABL322/results_abl436_offshore_reread.json"

#: The stream, for the ABL-385 readability floor. Offshore and onshore share it:
#: ABL-385 registers one floor per stream, not per forecast type.
STREAM = "wind"

#: Paths this read must not write, from the registration's `paths_this_read_must_not_write`.
#: ABL-387's failure was a default output path quietly overwriting a dispositioned
#: read and exiting 0; ABL-438's rule is that a selection writes where its issue
#: writes. Refusal is cheaper than a post-hoc blob-hash argument.
PROTECTED = (
    "experiments/ABL322/results_abl436_offshore_reread.json",
    "experiments/ABL322/abl436_preregistration_recheck.json",
    "experiments/ABL322/results.json",
    "reports/abl_436_offshore_reference_grade.md",
    "reports/abl_436_offshore_evidence_pack.md",
    "reports/abl_322_pilot_gate.md",
    "reports/abl_437_causal_levelling_reread.json",
    "reports/abl_437_causal_levelling_reread.md",
    "reports/abl_418_retro_grade.json",
    "reports/abl_418_retro_grade.md",
)


def _refuse_protected(root: Path, *targets: str) -> None:
    """Refuse to write where another issue's read published. ABL-387/ABL-438."""
    protected = {(root / path).resolve() for path in PROTECTED}
    for target in targets:
        if (root / target).resolve() in protected:
            raise SystemExit(
                f"refusing to write {target}: it is a published path this read must "
                "not touch (experiments/ABL443/config.json)")


def _readability(margin, floor: float) -> str:
    """How a G2/G3 margin reads against the floor G1 gates on.

    ABL-437 declined to widen G2/G3 from a sign test to a floor test, and this read
    does not land that change either. So this is a **diagnostic on the margin**, not
    a ladder condition: it never moves a letter, and the letter is reported beside
    it either way.
    """
    if margin is None:
        return "not measured"
    if abs(margin) <= floor:
        return "not readable at one seed"
    return "readable" if margin > 0 else "readable loss"


def read(root: Path, replica: str) -> dict:
    path = root / RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    meta = record["meta"]
    fit_start = pd.Timestamp(meta["fit_window"]["start"])
    gate_start = pd.Timestamp(meta["gate_window"]["start"])
    gate_end = pd.Timestamp(meta["gate_window"]["end_exclusive"])
    source = meta["training_source"]
    floor = readability_floor_pct(STREAM, 1)

    levels_by_pair = {(row.get("forecast_type", "wind_offshore"), row["country"]):
                      row.get("model_free_reference_mw") or {}
                      for row in record["training"]}
    cells_by_pair = {}
    for cell in record["gate_cells"]:
        cells_by_pair.setdefault((cell["forecast_type"], cell["country"]), []).append(cell)

    pairs, unreadable = [], []
    for (forecast_type, country), cells in sorted(cells_by_pair.items()):
        actuals = _load_actuals_series(country, forecast_type,
                                       fit_start - pd.Timedelta(days=14), gate_end,
                                       source=source, db_path=replica)
        pair_levels = levels_by_pair.get((forecast_type, country), {})
        rows = _reference_columns(_scored_rows(actuals, gate_start, gate_end), pair_levels)
        route = "schedule"
        if any(not all(_validate(rows[rows["horizon_band"] == cell["horizon_band"]]
                                 .reset_index(drop=True), cell).values())
               for cell in cells):
            rows = _reference_columns(
                _rebuilt_rows(country, forecast_type, source, replica, fit_start,
                              gate_start, gate_end, _feature_columns(meta, STREAM)),
                pair_levels)
            route = "feature-rebuild"

        graded, before, after = [], [], []
        for cell in sorted(cells, key=lambda item: PRIMARY_BANDS.index(item["horizon_band"])):
            band_rows = rows[rows["horizon_band"] == cell["horizon_band"]].reset_index(drop=True)
            checks = _validate(band_rows, cell)
            published = _recorded_or_computed(cell, STREAM)
            if not all(checks.values()):
                unreadable.append({"pair": f"{country} {forecast_type}",
                                   "band": cell["horizon_band"], "checks": checks,
                                   "route": route, "rows_rebuilt": len(band_rows),
                                   "published_n": cell["comparator_n"].get("constant_causal")})
                graded.append({"band": cell["horizon_band"], "reconstructed": False,
                               "published_grade": published.label})
                continue
            trailing, extra = _trailing_scores(band_rows, actuals)
            amended_scores = {**cell["scores"], **trailing}
            amended = grade_cell(amended_scores, STREAM, levelling=TRAILING_28D)
            before.append(published)
            after.append(amended)

            challenger = cell["scores"]["challenger"]["wape_pct"]
            # Every margin, in every case -- including where no letter moves and
            # including where the verdict is "not readable". A margin that is not
            # printed cannot be checked.
            margins = {}
            for condition, name in sorted(LADDER_REFERENCES[TRAILING_28D].items()):
                margins[name] = {"skill_pct": amended.skill.get(name),
                                 "condition": condition,
                                 "readability": _readability(amended.skill.get(name), floor)}
            for condition, name in sorted(LADDER_REFERENCES[FIT_WINDOW].items()):
                margins[name] = {"skill_pct": published.skill.get(name),
                                 "condition": f"{condition} (as published)",
                                 "readability": _readability(published.skill.get(name), floor)}
            for name in ("seasonal_naive", "constant_oracle", "climatology_oracle"):
                reference = comparator_wape(amended_scores, name)
                margin = (None if reference in (None, 0) or challenger is None
                          else 100.0 * (reference - challenger) / reference)
                margins[name] = {"skill_pct": margin,
                                 "condition": "G1" if name == "seasonal_naive" else "reported only",
                                 "readability": _readability(margin, floor)}

            graded.append({
                "band": cell["horizon_band"], "reconstructed": True, "route": route,
                "n": int(len(band_rows)),
                "published_grade": published.label, "amended_grade": amended.label,
                "published_failed": [name for name, _ in published.failed],
                "amended_failed": [name for name, _ in amended.failed],
                "amended_conditions": dict(amended.conditions),
                "wape": {name: comparator_wape(amended_scores, name) for name in (
                    "challenger", "seasonal_naive", "persistence",
                    "constant_causal", "constant_causal_28d", "constant_oracle",
                    "climatology_causal", "climatology_causal_28d", "climatology_oracle")},
                "margins_pct": margins,
                "level_inflation_pct": {
                    "constant_causal": level_inflation(amended_scores, "constant_causal"),
                    "constant_causal_28d": level_inflation(amended_scores, "constant_causal_28d")},
                "floor_pct": amended.floor_pct,
                "bar_weaker_than_a_flat_line": amended.bar_weak,
                # ABL-438: the ladder cannot see coverage, so a coverage-short cell
                # that beat D-7 grades A exactly as a full one does. It nests under
                # `gate`, where a flat lookup passes vacuously -- read the value.
                "enough_pairs": (cell.get("gate") or {}).get("enough_pairs"),
                "comparator_n": {**cell["comparator_n"], **extra["comparator_n"]},
                "trailing_levels_mw": extra["levels"],
            })

        pairs.append({
            "pair": f"{country} {forecast_type}", "country": country,
            "forecast_type": forecast_type, "route": route,
            "published_pair_grade": pair_grade(before).label if before else "Not measured",
            "amended_pair_grade": pair_grade(after).label if after else "Not measured",
            "published_levels_mw": {name: pair_levels.get(name)
                                    for name in ("constant_causal", "constant_oracle")},
            "cells": graded,
        })

    return {
        "issue": "ABL-443",
        "scope": SCOPE,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "experiments/ABL443/config.json",
        "inherited_levelling_registration": "experiments/ABL437/config.json",
        "levelling_before": FIT_WINDOW, "levelling_after": TRAILING_28D,
        "trailing_window_days": TRAILING_WINDOW_DAYS,
        "source_record": RECORD,
        "source_record_sha256": _sha256(path),
        "source_record_scope": meta["scope"],
        "source_table": source,
        "stream": STREAM,
        "readability_floor_pct_k1": floor,
        "g2_g3_floor_is_a_ladder_condition": False,
        "windows": {"fit_start": str(fit_start), "gate_start": str(gate_start),
                    "gate_end_exclusive": str(gate_end)},
        "replica": replica,
        "replica_bytes": Path(replica).stat().st_size,
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "not_reconstructible": unreadable,
        "pairs": pairs,
    }


def _fmt(value, suffix="", places=2):
    return "Not measured" if value is None else f"{value:.{places}f}{suffix}"


def _signed(value, suffix="pp"):
    return "Not measured" if value is None else f"{value:+.2f}{suffix}"


def render(result: dict) -> str:
    floor = result["readability_floor_pct_k1"]
    moved = [pair for pair in result["pairs"]
             if pair["amended_pair_grade"] != pair["published_pair_grade"]]
    lines = [
        "# ABL-443 — DE/NL `wind_offshore` re-read at the trailing causal reference",
        "",
        f"Generated: {result['generated_at']}. Scope: **`{result['scope']}`** — a new scope, "
        f"not an edit to `{result['source_record_scope']}`.",
        f"Registration: `{result['registration']}`, committed before this read existed. "
        f"Levelling inherited from `{result['inherited_levelling_registration']}`.",
        "",
        f"Levelling: **`{result['levelling_before']}` → `{result['levelling_after']}`** "
        f"({result['trailing_window_days']}-day window ending at each row's own `generated_at`). "
        "Arithmetic over ABL-436's committed record plus the two trailing references recomputed "
        "on the same rows — **no refit, no new model**, replica opened read-only.",
        f"Source record: `{result['source_record']}` (SHA-256 `{result['source_record_sha256'][:16]}…`), "
        f"source table `{result['source_table']}`.",
        f"Replica: `{result['replica']}` ({result['replica_bytes']:,} bytes).",
        "",
        "**ABL-436's read is not edited, regenerated or withdrawn by this one.** It stands at its "
        "own path under its own scope, and its letters remain the letters decided on the "
        "fit-window references.",
        "",
        "## 1. The row set is proved, not assumed",
        "",
        "Each cell's rows are rebuilt from ABL-348's eight registered run instants and then "
        "checked by recomputing that cell's published `constant_causal` and `climatology_causal` "
        f"WAPE *and* MAE from it, to {result['reconstruction_tolerance']:.0e}. A constant and a "
        "24-bucket climatology agreeing on two statistics each is the row set; one agreeing alone "
        "would not be.",
        "",
    ]
    total = sum(len(pair["cells"]) for pair in result["pairs"])
    bad = len(result["not_reconstructible"])
    routes = {}
    for pair in result["pairs"]:
        for cell in pair["cells"]:
            routes[cell.get("route")] = routes.get(cell.get("route"), 0) + 1
    lines.append(
        f"**{total - bad} of {total} cells reconstructed.** "
        + ("Every cell." if not bad else
           f"{bad} did not and are reported NOT RECONSTRUCTIBLE below, graded by nobody.")
        + f" Route: {routes.get('schedule', 0)} on the schedule alone, "
        f"{routes.get('feature-rebuild', 0)} through the harness's own feature build.")
    if bad:
        lines.extend(["", "| pair | band | rebuilt rows | published n | failed check |",
                      "|---|---|---:|---:|---|"])
        for item in result["not_reconstructible"]:
            failed = ", ".join(name for name, ok in item["checks"].items() if not ok)
            lines.append(f"| {item['pair']} | {item['band']} | {item['rows_rebuilt']} | "
                         f"{item['published_n']} | {failed} |")

    lines.extend([
        "", "## 2. The two grades, restated", "",
        f"**{len(moved)} of {len(result['pairs'])} pairs move.** A pair grades on its worst band.",
        "",
        "| pair | published (fit-window) | amended (trailing 28d) | what changed |",
        "|---|:---:|:---:|---|",
    ])
    for pair in result["pairs"]:
        reasons = sorted({name for cell in pair["cells"] if cell.get("reconstructed")
                          for name in set(cell["amended_failed"]) - set(cell["published_failed"])})
        recovered = sorted({name for cell in pair["cells"] if cell.get("reconstructed")
                            for name in set(cell["published_failed"]) - set(cell["amended_failed"])})
        what = ", ".join(filter(None, [
            f"now fails {', '.join(reasons)}" if reasons else "",
            f"no longer fails {', '.join(recovered)}" if recovered else ""]))
        if not what:
            what = ("no condition changes outcome" if pair["amended_pair_grade"]
                    == pair["published_pair_grade"] else "band mix")
        lines.append(f"| {pair['pair']} | {pair['published_pair_grade']} | "
                     f"**{pair['amended_pair_grade']}** | {what} |")

    lines.extend([
        "", "## 3. Every margin, in every case", "",
        "The issue asks for the margin in every case, including where the verdict is *not "
        "readable*. `skill` is `(reference − challenger) / reference`, in percent — positive means "
        "the challenger is better. **G2 and G3 are sign tests**: ABL-437 declined to widen them to "
        f"a floor test and this read does not either, so the `{floor:.2f}%` k=1 wind floor below is "
        "a **diagnostic on the margin, never a ladder condition**. A letter that turns on a "
        "sub-floor margin is reported as the ladder computes it *and* flagged as not demonstrated "
        "at one seed.",
        "",
        "| pair | band | n | reference | condition | challenger WAPE | reference WAPE | skill | vs floor |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ])
    order = ("seasonal_naive", "constant_causal_28d", "climatology_causal_28d",
             "constant_causal", "climatology_causal", "constant_oracle", "climatology_oracle")
    for pair in result["pairs"]:
        for cell in pair["cells"]:
            if not cell.get("reconstructed"):
                lines.append(f"| {pair['pair']} | {cell['band']} | — | — | — | — | — | — | "
                             "NOT RECONSTRUCTIBLE |")
                continue
            for name in order:
                entry = cell["margins_pct"].get(name)
                if entry is None:
                    continue
                lines.append(
                    f"| {pair['pair']} | {cell['band']} | {cell['n']:,} | `{name}` | "
                    f"{entry['condition']} | {_fmt(cell['wape']['challenger'], '%')} | "
                    f"{_fmt(cell['wape'].get(name), '%')} | {_signed(entry['skill_pct'])} | "
                    f"{entry['readability']} |")

    lines.extend([
        "", "## 4. Every cell, both levellings", "",
        "`c` = constant, `clim` = climatology. `inflation` is each causal reference's WAPE over "
        "the oracle constant's — the residual mis-levelling, which the trailing window **reduces "
        "rather than removes**. Do not quote the corrected reference as exact.",
        "",
        "| pair | band | n | challenger | D-7 | c causal | c 28d | c oracle | clim causal | "
        "clim 28d | clim oracle | inflation causal / 28d | enough pairs | published | amended |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ])
    for pair in result["pairs"]:
        for cell in pair["cells"]:
            if not cell.get("reconstructed"):
                lines.append(f"| {pair['pair']} | {cell['band']} | — | — | — | — | — | — | — | — "
                             f"| — | — | — | {cell['published_grade']} | NOT RECONSTRUCTIBLE |")
                continue
            w, infl = cell["wape"], cell["level_inflation_pct"]
            enough = cell.get("enough_pairs")
            lines.append(
                f"| {pair['pair']} | {cell['band']} | {cell['n']:,} | {_fmt(w['challenger'], '%')} | "
                f"{_fmt(w['seasonal_naive'], '%')} | {_fmt(w['constant_causal'], '%')} | "
                f"{_fmt(w['constant_causal_28d'], '%')} | {_fmt(w['constant_oracle'], '%')} | "
                f"{_fmt(w['climatology_causal'], '%')} | {_fmt(w['climatology_causal_28d'], '%')} | "
                f"{_fmt(w['climatology_oracle'], '%')} | {_fmt(infl['constant_causal'], '%')} / "
                f"{_fmt(infl['constant_causal_28d'], '%')} | "
                f"{'yes' if enough else ('no' if enough is False else '—')} | "
                f"{cell['published_grade']} | {cell['amended_grade']} |")

    lines.extend([
        "", "## 5. The trailing reference's own levels", "",
        "The published fit-window constant beside the range the trailing window actually took "
        "across each cell's issue instants. A trailing window that never moves is a fit-window "
        "constant wearing a different name; one that moves a long way is carrying the level change "
        "the amendment exists to catch.",
        "",
        "| pair | band | as-of instants | c causal (fixed) | c 28d min | c 28d mean | c 28d max | c oracle |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pair in result["pairs"]:
        for cell in pair["cells"]:
            if not cell.get("reconstructed"):
                continue
            levels = cell["trailing_levels_mw"]
            published = pair["published_levels_mw"]
            lines.append(
                f"| {pair['pair']} | {cell['band']} | {levels['as_of_count']:,} | "
                f"{_fmt(published.get('constant_causal'), ' MW', 1)} | "
                f"{_fmt(levels['constant_min_mw'], ' MW', 1)} | "
                f"{_fmt(levels['constant_mean_mw'], ' MW', 1)} | "
                f"{_fmt(levels['constant_max_mw'], ' MW', 1)} | "
                f"{_fmt(published.get('constant_oracle'), ' MW', 1)} |")

    lines.extend([
        "", "## 6. What this read did not touch", "",
        "| path | why |", "|---|---|",
        f"| `{result['source_record']}` | ABL-436's committed record. Read, hashed, never written. |",
        "| `reports/abl_436_offshore_reference_grade.md` | ABL-436's published gate report. |",
        "| `reports/abl_436_offshore_evidence_pack.md` | ABL-436's published evidence pack. |",
        "| `reports/abl_437_causal_levelling_reread.{json,md}` | ABL-437's re-read, which does not cover these two pairs. |",
        "| `reports/abl_322_pilot_gate.md`, `experiments/ABL322/results.json` | the `abl322-pilot` scope's own outputs. |",
        "",
        "The refusal is in the script (`PROTECTED`), not in the operator's memory.",
        "",
        "Read-only on the replica. No model was loaded, fitted or scored again.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ABL-443: the offshore trailing-reference read.")
    parser.add_argument("--replica-db")
    parser.add_argument("--render-only", action="store_true",
                        help="re-render the stored JSON without re-measuring")
    parser.add_argument("--repo-root", default=str(Path(__file__).parent.parent))
    parser.add_argument("--json-out", default="reports/abl_443_offshore_trailing_reread.json")
    parser.add_argument("--report-out", default="reports/abl_443_offshore_trailing_reread.md")
    args = parser.parse_args()
    if not args.render_only and not args.replica_db:
        parser.error("--replica-db is required unless --render-only is given")

    root = Path(args.repo_root)
    _refuse_protected(root, args.json_out, args.report_out)
    if args.render_only:
        result = json.loads((root / args.json_out).read_text(encoding="utf-8"))
    else:
        result = read(root, args.replica_db)
        (root / args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / args.report_out).write_text(render(result), encoding="utf-8")
    print(f"wrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
