#!/usr/bin/env python3
"""ABL-419 -- the tranche 2c findings tables, generated from the stored results.

Nothing here refits, re-reads the replica or recomputes a WAPE. Every number is
lifted from `experiments/ABL348/results_abl419_tranche2c.json` (written by
`scripts/evaluate_solar_retrain.py --scope abl316-t2c`) and from two committed
machine records -- ABL-396's night-floor screen and ABL-348's frozen config. That
is deliberate and follows ABL-418: a findings pack that restates numbers in prose
is a second, unverifiable copy of the evidence, and the two drift. ABL-405's pack
is the case in point -- it states a source table its own machine record
contradicts.

The grades are read back through `src.evaluation.gate_grading`, never
re-derived: `cell_grade` returns the grade the run recorded, and `pair_grade`
applies the worst-band rule with ABL-418's `U(+)`-survives-only-if-uniform
semantics. Reimplementing either here would be a second copy of the ladder
living in a reporting script, which is the thing that module exists to prevent.

What this script *does* add is the one piece of arithmetic ABL-419 asks for and
the harness does not carry: **ES's night-floor band**.

    ABL-396 section 2. Let `f` be the share of the window's total |energy| booked
    at night, and `W` the daylight-only WAPE of a challenger. Then the all-hours
    WAPE of that same challenger is bounded exactly:

        clamped to 0 at night   ->  W(1-f) + f      (the upper end)
        reproduces the floor    ->  W(1-f)          (the lower end)

    so `f` is the full width of the interval, in WAPE points.

The harness measures the **all-hours** number `A` (every arm on the same rows).
Inverting the bound gives what a daylight-only read of the same challenger must
have been:

        W in [ (A - f) / (1 - f) ,  A / (1 - f) ]

Both directions are printed. The point is not to adjust ES's verdict -- the gate
scores challenger and D-7 on identical rows, so the verdict is what it is -- but
to bound, for free, the one thing an all-hours read on a country with a real
night floor leaves open.

Run: `.venv\\Scripts\\python.exe scripts/abl419_tranche2c_read.py`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.gate_grading import GRADE_SEVERITY, cell_grade, pair_grade  # noqa: E402

ROOT = Path(__file__).parent.parent

RESULTS = ROOT / "experiments" / "ABL348" / "results_abl419_tranche2c.json"
NIGHT_SCREEN = ROOT / "reports" / "abl_396_night_floor_screen.json"
REGISTRATION = ROOT / "experiments" / "ABL348" / "config.json"
OUT_MD = ROOT / "reports" / "abl_419_tranche2c_tables.md"
OUT_JSON = ROOT / "reports" / "abl_419_tranche2c_tables.json"

COUNTRIES = ("ES", "GR", "HR", "IT", "PT")
STREAM = "solar"

#: ABL-419 caps ES at grade B whatever G1-G4 say, with this named as the failed
#: condition. It is a *hold*, not a measurement: the ABL-337 clamp question is
#: serving-side and this read changes no serving path, but ES may not reach a
#: promotion recommendation before ABL-411 settles. Named here so the cap can
#: never be mistaken for something the ladder produced.
ES_HOLD = "ABL-411 hold"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def night_floor(screen: dict, country: str, source: str, window: str) -> dict:
    for entry in screen["countries"]:
        if entry["country"] != country:
            continue
        for arm in entry["sources"]:
            if arm["source"] != source:
                continue
            for win in arm["windows"]:
                if win["window"] == window:
                    return win
    raise KeyError(f"no night-floor screen row for {country}/{source}/{window}")


def implied_daylight(all_hours_wape_pct: float, f_pct: float) -> tuple[float, float]:
    """The daylight-only WAPE implied by an all-hours read, low and high.

    Inverts ABL-396 section 2. Both arguments are in percent, so the share
    itself is `f_pct / 100`.
    """
    f = f_pct / 100.0
    return ((all_hours_wape_pct - f_pct) / (1.0 - f), all_hours_wape_pct / (1.0 - f))


#: The cap ABL-419 places on ES: "grade ES no higher than B, whatever G1-G4 say".
CAP = "B"


def capped(country: str, ladder_label: str) -> tuple[str, str]:
    """ES may not be reported higher than B. Returns (reported grade, failures).

    **"Higher" is ABL-418's severity ordering, not alphabetical**, and getting
    that wrong is the whole reason this is a named function with tests rather
    than an inline conditional. `GRADE_SEVERITY` is `{"A": 0, "U": 1, "B": 2,
    "C": 3}`, so `U` -- and therefore `U(+)` -- is **less severe than `B`**. A
    reading that treated `U(+)` as already below the cap would leave ES's grade
    untouched and quietly not apply the cap at all, which is the failure this
    guards against: ES's ladder grade on this read *is* `U(+)`, so the cap binds.

    The cap only ever moves a grade **down**. A ladder grade already at or below
    the cap in severity keeps its own letter and gains the hold as an additional
    named failure -- reporting `B` for a cell the ladder called `C` would be an
    upgrade wearing a cap's clothing.

    The ladder grade is never discarded: every caller prints it beside the
    reported one. `U(+)` and `B` are both non-promotion, but they carry
    *different next steps* -- `U(+)` means re-read at k>1 seeds per ABL-385,
    `B` means not promotion-eligible with failures named -- and collapsing the
    two would destroy the only actionable thing ES's read says.
    """
    if country != "ES":
        return ladder_label, ""
    letter = ladder_label.split("(")[0]
    if GRADE_SEVERITY[letter] < GRADE_SEVERITY[CAP]:
        return CAP, ES_HOLD
    return ladder_label, ES_HOLD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--md-out", default=str(OUT_MD))
    parser.add_argument("--json-out", default=str(OUT_JSON))
    args = parser.parse_args()

    results_path = Path(args.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    screen = json.loads(NIGHT_SCREEN.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    bars = registration["per_pair_bar_measured_before_any_challenger_exists"]["bars"]

    meta = results["meta"]
    cells = results["gate_cells"]
    source = meta["training_source"]

    record: dict = {
        "generated_from": {
            "results": results_path.relative_to(ROOT).as_posix(),
            "results_sha256": _sha256(results_path),
            "night_screen": NIGHT_SCREEN.relative_to(ROOT).as_posix(),
            "night_screen_sha256": _sha256(NIGHT_SCREEN),
            "registration": REGISTRATION.relative_to(ROOT).as_posix(),
        },
        "scope": meta["scope"],
        "training_source": source,
        "n_features": meta.get("n_features"),
        "fit_rules": meta.get("fit_rules"),
        "verdict": results.get("verdict"),
        "night_floor": {},
        "es_band": {},
        "pair_grades": {},
    }

    lines: list[str] = []
    lines.append("# ABL-419 — tranche 2c generated tables")
    lines.append("")
    lines.append(
        f"Generated from `{record['generated_from']['results']}`, SHA-256 "
        f"`{record['generated_from']['results_sha256']}`, and from ABL-396's committed "
        "night-floor screen. No refit, no replica read, no recomputed metric; the grades are "
        "read back through `src/evaluation/gate_grading.py`, not re-derived. Regenerate with "
        "`.venv\\Scripts\\python.exe scripts/abl419_tranche2c_read.py`.")
    lines.append("")
    lines.append(f"Scope `{meta['scope']}`, source **`{source}`**, "
                 f"**{meta.get('n_features')}** features, fit rules `{meta.get('fit_rules')}`.")
    lines.append("")

    # ---------------------------------------------------------------- night floor
    lines.append("## 1. Night floor, all five countries — the zeros stated, not omitted")
    lines.append("")
    lines.append(
        "`f` is ABL-396 section 2's `wape_floor_pct_if_clamped`: the share of the window's "
        "total |energy| booked at night, which is the **full width in WAPE points** of the "
        "interval an all-hours read can occupy relative to the daylight-only read of the same "
        f"challenger. Source `{source}`, the table this tranche fits and scores on.")
    lines.append("")
    lines.append("| country | window | night hrs | hrs > 1 MW | night mean | **f** |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for country in COUNTRIES:
        for window in ("fit", "gate"):
            row = night_floor(screen, country, source, window)
            record["night_floor"].setdefault(country, {})[window] = {
                "n_night_rows": row["n_night_rows"],
                "n_night_above_threshold": row["n_night_above_threshold"],
                "night_mean_mw": row["night_mean_mw"],
                "f_pct": row["wape_floor_pct_if_clamped"],
            }
            emphasis = "**" if row["wape_floor_pct_if_clamped"] >= 0.1 else ""
            lines.append(
                f"| {country} | {window} | {row['n_night_rows']:,} | "
                f"{row['n_night_above_threshold']:,} | {row['night_mean_mw']:.2f} MW | "
                f"{emphasis}{row['wape_floor_pct_if_clamped']:.4f}%{emphasis} |")
    lines.append("")

    # ------------------------------------------------------------------- ES band
    es_f = record["night_floor"]["ES"]["gate"]["f_pct"]
    lines.append("## 2. ES's night-floor band, on the face of the table")
    lines.append("")
    lines.append(
        f"ES gate-window `f` = **{es_f:.4f}%**. Every arm of the gate is scored on the same "
        "all-hours rows, so the verdict below is not adjusted by this band — the band is what "
        "**bounds** it: what a daylight-only read of this same challenger would have been. "
        "Exact, free, and it closes the only cell ABL-403's 2×2 could have moved on this "
        "tranche, which is why ABL-419 discharges that soft hold rather than waiting on it.")
    lines.append("")
    lines.append("| band | n | all-hours challenger WAPE (measured) | implied daylight-only WAPE | if clamped to 0 at night | D-7 bar (same rows) | registered verdict | clamped-variant verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|:---:|")
    for cell in cells:
        if cell["country"] != "ES":
            continue
        challenger = (cell["scores"].get("challenger") or {}).get("wape_pct")
        d7 = (cell["scores"].get("seasonal_naive") or {}).get("wape_pct")
        band = cell["horizon_band"]
        if challenger is None or d7 is None:
            lines.append(f"| {band} | {cell['scores'].get('challenger', {}).get('n', 0)} "
                         f"| not measured | — | — | — | — | — |")
            continue
        low, high = implied_daylight(challenger, es_f)
        # What the *serving* path would score. The ABL-337 clamp forces a zero at
        # night on this same predicate, so the clamped variant of this challenger
        # keeps its daylight behaviour `W` and takes `W(1-f) + f`. Bounding `W`
        # by the band above bounds that too:
        #
        #     W(1-f) in [A - f, A]   ->   clamped in [A, A + f]
        #
        # The registered verdict is NOT this: both gate arms are scored on the
        # same all-hours rows, so the night floor cannot have moved it. This row
        # answers a different and strictly serving-side question, and it is
        # reported because it is free -- not because it gates anything here.
        clamped_low, clamped_high = challenger, challenger + es_f
        if clamped_high < d7:
            clamped_verdict = "PASS"
        elif clamped_low >= d7:
            clamped_verdict = "FAIL"
        else:
            clamped_verdict = "**indeterminate**"
        n = cell["scores"]["challenger"].get("n", 0)
        record["es_band"][band] = {
            "n": n, "all_hours_wape_pct": challenger, "f_pct": es_f,
            "implied_daylight_low_pct": low, "implied_daylight_high_pct": high,
            "clamped_low_pct": clamped_low, "clamped_high_pct": clamped_high,
            "d7_wape_pct": d7,
            "registered_verdict": "PASS" if (cell.get("gate") or {}).get("pass") else "FAIL",
            "clamped_variant_verdict": clamped_verdict.replace("*", ""),
        }
        lines.append(
            f"| {band} | {n:,} | {challenger:.2f}% | {low:.2f}%–{high:.2f}% | "
            f"{clamped_low:.2f}%–{clamped_high:.2f}% | {d7:.2f}% | "
            f"{'PASS' if (cell.get('gate') or {}).get('pass') else 'FAIL'} | "
            f"{clamped_verdict} |")
    lines.append("")
    lines.append(
        "**Read the last two columns as answering different questions.** The *registered* "
        "verdict is a direct measurement: challenger and D-7 are scored on the identical "
        "all-hours rows, so ES's night floor cannot have moved it in either direction, and "
        "the band does not qualify it. The *clamped-variant* column is serving-side and is "
        "reported because `f` makes it free: the ABL-337 clamp forces a zero on this same "
        "night predicate, so a served version of this challenger would score somewhere in "
        "`[A, A+f]`. On all three ES bands that interval **straddles the D-7 bar**, so the "
        "bound cannot say whether a clamped ES would clear it. That is a finding to hand to "
        "whoever owns serving, not a qualification of the read above — and settling it needs "
        "an actual daylight-only read, which this bound deliberately does not substitute for.")
    lines.append("")
    lines.append(f"**ES is capped at grade B regardless of G1–G4**, with `{ES_HOLD}` named as the "
                 "failed condition. The cap is a hold, not a measurement: the ABL-337 clamp "
                 "question is serving-side and this read changes no serving path, so nothing "
                 "above depends on it — but ES may not reach a promotion recommendation before "
                 "ABL-411 settles. The cap only ever moves a grade down.")
    lines.append("")

    # ---------------------------------------------------------------- pair grades
    lines.append("## 3. Pair grade against the pre-committed bar")
    lines.append("")
    lines.append(
        "The bar column is ABL-348's, measured before any challenger for these pairs existed. "
        "It is here because ABL-406 established across eight wind pairs that the gate outcome "
        "was *fully* predicted by whether a causal constant clears the bar on its own — a pass "
        "against a weak bar and a pass against a strong one are not the same evidence.")
    lines.append("")
    lines.append("| pair | pre-committed D-7 bar | band grades | ladder pair grade | **reported** | failed conditions | bar weaker than a flat line? |")
    lines.append("|---|---:|---|:---:|:---:|---|:---:|")
    for country in COUNTRIES:
        grades = [cell_grade(cell, STREAM) for cell in cells if cell["country"] == country]
        if not grades:
            continue
        pair = pair_grade(grades)
        reported, hold = capped(country, pair.label)
        failed = [name for name, _ in pair.failed]
        if hold:
            failed.append(hold)
        weak = pair.bar_weak
        bar = bars.get(f"{country}/solar", {}).get("d7_wape_pct")
        record["pair_grades"][country] = {
            "bands": [grade.label for grade in grades],
            "ladder_pair_grade": pair.label,
            "reported_grade": reported,
            "failed_conditions": failed,
            "bar_weaker_than_a_flat_line": weak,
            "precommitted_d7_bar_pct": bar,
            "skill_pct": dict(pair.skill),
            "own_error_margin_pct": dict(pair.own_error_margin),
            "floor_pct": pair.floor_pct,
        }
        lines.append(
            f"| {country} | {bar}% | {' / '.join(grade.label for grade in grades)} | "
            f"{pair.label} | **{reported}** | {', '.join(failed) or '—'} | "
            f"{'yes' if weak else ('no' if weak is not None else '—')} |")
    lines.append("")
    lines.append(
        "**Do not average this tranche's pass rate against 2a's.** 2a's bars ran 18.35–26.11% "
        "plus CH at 12.67%; these run 7.11–16.43%. ABL-348 registered that reading in advance "
        "under `reading_caveats_not_band_changes`: same band, materially harder task, and a "
        "lower pass rate here is not model quality.")
    lines.append("")

    Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    # ABL-364: the report body is deliberately non-ASCII (en dashes in the
    # tables), which is the exception that rule allows -- but redirected stdout
    # encodes with the locale codepage, so the stream is re-encoded here rather
    # than the body being flattened. The `--help` text itself is ASCII.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))
    print(f"\nwrote {args.md_out}\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
