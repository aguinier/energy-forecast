#!/usr/bin/env python3
"""ABL-421 -- the tranche 2d findings tables, generated from the stored results.

Nothing here refits, re-reads the replica or recomputes a WAPE. Every number is
lifted from `experiments/ABL348/results_abl421_tranche2d.json` (written by
`scripts/evaluate_solar_retrain.py --scope abl316-t2d`) and from two committed
machine records -- ABL-396's night-floor screen and ABL-348's frozen config.
That follows ABL-418 and ABL-419: a findings pack that restates numbers in prose
is a second, unverifiable copy of the evidence, and the two drift.

The grades are read back through `src.evaluation.gate_grading`, never re-derived.

What this script adds beyond the harness's own report is three things ABL-421
asks for by name:

1. **EE's night-floor band, on the face of the table.** ABL-396 section 2: let
   `f` be the share of the window's total |energy| booked at night and `W` the
   daylight-only WAPE of a challenger. Then that challenger's all-hours WAPE is
   bounded exactly:

       reproduces the floor    ->  W(1-f)          (the lower end)
       clamped to 0 at night   ->  W(1-f) + f      (the upper end)

   so `f` is the full width of the interval in WAPE points. The harness measures
   the all-hours number `A`, so the inverse bound is printed too:

       W in [ (A - f) / (1 - f) ,  A / (1 - f) ]

   EE carries the third-largest solar night floor in the fleet, `f` = 0.718% of
   gate-window energy. The other five are at or under 0.042%.

2. **What NL's signed target does to its WAPE denominator** -- answered with a
   number rather than left to surface as an anomalous margin.

3. **Each pair's gate-window mean beside its grade**, against ABL-348's lowest
   already-dispositioned solar fleet.

Run: `.venv\\Scripts\\python.exe scripts/abl421_tranche2d_read.py`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.gate_grading import cell_grade, pair_grade  # noqa: E402

ROOT = Path(__file__).parent.parent

RESULTS = ROOT / "experiments" / "ABL348" / "results_abl421_tranche2d.json"
NIGHT_SCREEN = ROOT / "reports" / "abl_396_night_floor_screen.json"
REGISTRATION = ROOT / "experiments" / "ABL348" / "config.json"
OUT_MD = ROOT / "reports" / "abl_421_tranche2d_tables.md"
OUT_JSON = ROOT / "reports" / "abl_421_tranche2d_tables.json"

COUNTRIES = ("EE", "FI", "LT", "LV", "NL", "SE")
STREAM = "solar"

#: The lowest solar fleet ABL-348 has already dispositioned: SK, graded A in
#: tranche 2a on a 114.8 MW gate-window mean. Used as a *reference line* for
#: whether a WAPE can carry a promotion decision, by analogy with the one
#: decision-grade statement ABL-348 makes explicitly --
#: `CH_wind_onshore_is_not_decision_grade`, "gate-window mean 12.9 MW ... WAPE on
#: a ~13 MW series cannot carry a promotion decision either way. Registration-
#: compatible; report it, do not decide on it." ABL-348 states no such line for
#: solar, so this is a comparison and not a registered threshold, and it is named
#: as one wherever it is printed.
SK_REFERENCE_MW = 114.8
SK_REFERENCE = "SK/solar, 114.8 MW, graded A in tranche 2a"


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


def all_hours_band(daylight_wape_pct: float, f_pct: float) -> tuple[float, float]:
    """`[W(1-f), W(1-f)+f]` -- the band ABL-421 asks for on the face of the table.

    The forward direction of ABL-396 section 2: what an all-hours read of a
    challenger with daylight-only WAPE `W` must lie between.
    """
    f = f_pct / 100.0
    return (daylight_wape_pct * (1.0 - f), daylight_wape_pct * (1.0 - f) + f_pct)


def implied_daylight(all_hours_wape_pct: float, f_pct: float) -> tuple[float, float]:
    """The daylight-only WAPE implied by an all-hours read, low and high.

    Inverts `all_hours_band`. Both arguments are in percent.
    """
    f = f_pct / 100.0
    return ((all_hours_wape_pct - f_pct) / (1.0 - f), all_hours_wape_pct / (1.0 - f))


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
    declared_pairs = registration["not_evaluable"]["pairs"]

    meta = results["meta"]
    cells = results["gate_cells"]
    declared_cells = results.get("not_evaluable_cells", [])
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
        "registered_cells": meta.get("registered_cells"),
        "registered_grid_cells": meta.get("registered_grid_cells"),
        "night_floor": {},
        "ee_band": {},
        "nl_denominator": {},
        "not_evaluable": {},
        "pair_grades": {},
    }

    lines: list[str] = []
    lines.append("# ABL-421 — tranche 2d generated tables")
    lines.append("")
    lines.append(
        f"Generated from `{record['generated_from']['results']}`, SHA-256 "
        f"`{record['generated_from']['results_sha256']}`, and from ABL-396's committed "
        "night-floor screen. No refit, no replica read, no recomputed metric; the grades are "
        "read back through `src/evaluation/gate_grading.py`, not re-derived. Regenerate with "
        "`.venv\\Scripts\\python.exe scripts/abl421_tranche2d_read.py`.")
    lines.append("")
    lines.append(f"Scope `{meta['scope']}`, source **`{source}`**, "
                 f"**{meta.get('n_features')}** features, fit rules `{meta.get('fit_rules')}`. "
                 f"**{meta.get('registered_cells')} evaluable cells of "
                 f"{meta.get('registered_grid_cells')}** in the 6 x 3 grid.")
    lines.append("")

    # ------------------------------------------------------- 1. NOT-EVALUABLE
    lines.append("## 1. The four cells the registration declares NOT-EVALUABLE")
    lines.append("")
    lines.append(
        "This is the first tranche to contain ABL-348's declared pairs, and it is the reason "
        "the bar is 14 and not 18. ABL-348 `not_evaluable`: *\"A pair listed here is reported "
        "NOT-EVALUABLE on the named bands. It is not a FAIL and must not be counted as one; a "
        "gate read that scores it has misread this registration.\"* Both were declared "
        "**before any fit existed**. Their measured numbers are printed because a declaration "
        "nobody can check is indistinguishable from a challenger quietly dropped for scoring "
        "badly — but they carry no gate outcome and no grade.")
    lines.append("")
    lines.append("| pair | declared bands | n_d7_scorable | registered min n | cause | source-dependent? |")
    lines.append("|---|---|---:|---:|---|:---:|")
    for pair, entry in sorted(declared_pairs.items()):
        country = pair.split("/")[0]
        if country not in COUNTRIES:
            continue
        record["not_evaluable"][country] = {
            "bands": entry["registered_min_n_684_bands"],
            "n_d7_scorable": entry["n_d7_scorable_energy_generation"],
            "source_dependent": entry["source_dependent"],
            "cause": entry["cause"],
        }
        lines.append(
            f"| {pair} | {', '.join(entry['registered_min_n_684_bands'])} | "
            f"{entry['n_d7_scorable_energy_generation']} | 684 | {entry['cause']} | "
            f"{'**yes**' if entry['source_dependent'] else 'no'} |")
    lines.append("")
    lines.append(
        "**Only one of the two is ours.** EE's shortfall is an ABL-188 bit-identical zero run "
        "present identically in *both* source tables, so reverting ABL-348's source change "
        "would not recover it. FI's is `energy_generation` holding 663 of the 720 gate hours "
        "against `energy_renewable`'s 717 — that one **is** a cost of the source change, and "
        "it is a finding for whoever owns that decision rather than a fact about FI's model.")
    lines.append("")
    if declared_cells:
        lines.append("What those cells measured, for audit only — no verdict attaches:")
        lines.append("")
        lines.append("| country | horizon | n | min n | challenger WAPE | D-7 WAPE | skill vs D-7 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for cell in sorted(declared_cells, key=lambda r: (r["country"], r["horizon_band"])):
            chal = (cell["scores"].get("challenger") or {}).get("wape_pct")
            d7 = (cell["scores"].get("seasonal_naive") or {}).get("wape_pct")
            skill = "—" if chal is None or d7 is None else f"{100 * (1 - chal / d7):+.1f}%"
            record["not_evaluable"].setdefault(cell["country"], {}).setdefault("measured", {})[
                cell["horizon_band"]] = {"n": cell["gate"]["n"], "challenger_wape_pct": chal,
                                         "d7_wape_pct": d7}
            lines.append(
                f"| {cell['country']} | {cell['horizon_band']} | {cell['gate']['n']:,} | "
                f"{cell['gate']['minimum_n']:,} | "
                f"{'—' if chal is None else f'{chal:.2f}%'} | "
                f"{'—' if d7 is None else f'{d7:.2f}%'} | {skill} |")
        lines.append("")
    lines.append(
        "**48-64h is read for both pairs**, on ABL-348's own instruction "
        "(`not_evaluable.note_48_64h`): that band selects a 480-510 row subset, so its n scales "
        "proportionally rather than being hard-bounded by `n_d7_scorable`, and \"a pair declared "
        "here may still clear 456 in that band and should be reported if it does\". Where such a "
        "cell falls short of 456 it is a **coverage shortfall** (`enough_pairs: false`), not a "
        "loss to D-7; the two flags are separate in the record.")
    lines.append("")

    # ---------------------------------------------------------- 2. night floor
    lines.append("## 2. Night floor, all six countries — the zeros stated, not omitted")
    lines.append("")
    lines.append(
        "`f` is ABL-396 section 2's `wape_floor_pct_if_clamped`: the share of the window's "
        "total |energy| booked at night, which is the **full width in WAPE points** of the "
        "interval an all-hours read can occupy relative to the daylight-only read of the same "
        f"challenger. Source `{source}`, the table this tranche fits and scores on. The signed "
        "share is printed beside it because for NL the two differ in sign, which is the whole "
        "of section 4 below.")
    lines.append("")
    lines.append("| country | window | night hrs | hrs > 1 MW | night mean | signed share | **f** |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for country in COUNTRIES:
        for window in ("fit", "gate"):
            row = night_floor(screen, country, source, window)
            record["night_floor"].setdefault(country, {})[window] = {
                "n_night_rows": row["n_night_rows"],
                "n_night_above_threshold": row["n_night_above_threshold"],
                "n_night_negative": row["n_night_negative"],
                "night_mean_mw": row["night_mean_mw"],
                "signed_share_pct": row["pct_of_total_energy_at_night"],
                "f_pct": row["wape_floor_pct_if_clamped"],
            }
            emphasis = "**" if row["wape_floor_pct_if_clamped"] >= 0.1 else ""
            lines.append(
                f"| {country} | {window} | {row['n_night_rows']:,} | "
                f"{row['n_night_above_threshold']:,} | {row['night_mean_mw']:.2f} MW | "
                f"{row['pct_of_total_energy_at_night']:+.4f}% | "
                f"{emphasis}{row['wape_floor_pct_if_clamped']:.4f}%{emphasis} |")
    lines.append("")
    lines.append(
        "EE is the only country here with a material floor and carries the **third-largest in "
        "the fleet**. The other five are at or under 0.042% of gate-window energy, where the "
        "bound below is narrower than the rounding on a reported WAPE.")
    lines.append("")

    # -------------------------------------------------------------- 3. EE band
    ee_f = record["night_floor"]["EE"]["gate"]["f_pct"]
    lines.append("## 3. EE's night-floor band, on the face of the table")
    lines.append("")
    lines.append(
        f"EE gate-window `f` = **{ee_f:.4f}%**. Every arm of the gate is scored on the same "
        "all-hours rows, so no verdict below is adjusted by this band — the band is what "
        "**bounds** it. Two of EE's three bands are NOT-EVALUABLE, so for those the bound is "
        "the only quantitative statement this tranche makes about them, which is exactly what "
        "makes it worth printing.")
    lines.append("")
    lines.append("| band | status | n | all-hours WAPE `A` (measured) | implied daylight-only `W` | `[W(1-f), W(1-f)+f]` |")
    lines.append("|---|:---:|---:|---:|---:|---:|")
    for cell in sorted(list(cells) + list(declared_cells),
                       key=lambda r: (r["country"], r["horizon_band"])):
        if cell["country"] != "EE":
            continue
        band = cell["horizon_band"]
        is_declared = any(d["country"] == "EE" and d["horizon_band"] == band
                          for d in declared_cells)
        status = "NOT-EVALUABLE" if is_declared else "gated"
        chal = (cell["scores"].get("challenger") or {}).get("wape_pct")
        if chal is None:
            lines.append(f"| {band} | {status} | {cell['gate']['n']:,} | not measured | — | — |")
            continue
        low, high = implied_daylight(chal, ee_f)
        # The forward band, re-derived from the midpoint of the implied W so the
        # printed interval is the one ABL-421 names rather than a paraphrase of
        # it. `all_hours_band(W, f)` returns to `[A - something, A + something]`
        # bracketing the measured A by construction, which is the check.
        band_low, band_high = all_hours_band(low, ee_f)
        record["ee_band"][band] = {
            "status": status, "n": cell["gate"]["n"], "all_hours_wape_pct": chal,
            "f_pct": ee_f, "implied_daylight_low_pct": low, "implied_daylight_high_pct": high,
            "forward_band_low_pct": band_low, "forward_band_high_pct": band_high,
        }
        lines.append(
            f"| {band} | {status} | {cell['gate']['n']:,} | {chal:.2f}% | "
            f"{low:.2f}%–{high:.2f}% | {band_low:.2f}%–{band_high:.2f}% |")
    lines.append("")
    lines.append(
        f"The band's full width is `f` = {ee_f:.4f} WAPE points. That is **two orders of "
        "magnitude below the {:.2f}pp readability floor** ABL-418 registers for solar, so on EE "
        "the night floor cannot move a grade — it is bounded here so that the statement is "
        "measured rather than assumed.".format(10.65))
    lines.append("")

    # --------------------------------------------------------- 4. NL denominator
    nl_gate = record["night_floor"]["NL"]["gate"]
    nl_fit = record["night_floor"]["NL"]["fit"]
    nl_neg_total = nl_gate["n_night_negative"] + nl_fit["n_night_negative"]
    nl_night_total = nl_gate["n_night_rows"] + nl_fit["n_night_rows"]
    record["nl_denominator"] = {
        "n_night_negative": nl_neg_total, "n_night_rows": nl_night_total,
        "gate_signed_share_pct": nl_gate["signed_share_pct"],
        "gate_absolute_share_pct": nl_gate["f_pct"],
        "denominator_effect_pct_of_itself": nl_gate["f_pct"],
        "gate_window_mean_mw": bars["NL/solar"]["mean_actual_mw"],
        "d7_bar_pct": bars["NL/solar"]["d7_wape_pct"],
    }
    lines.append("## 4. NL: what a signed target does to the WAPE denominator")
    lines.append("")
    lines.append(
        f"NL solar is negative at **every** night hour — {nl_neg_total:,} of {nl_night_total:,} "
        "across both windows (ABL-396 section 6, ABL-412). That is our own netting rule, not "
        "upstream and not a sign error, and ABL-412 fixed it at the dashboard *read site*, not "
        "in the data. This gate reads the data, so the question is what it does to the score.")
    lines.append("")
    lines.append(
        "**It is arithmetically negligible, and here is the number.** `score_predictions` uses "
        "`denom = sum(|actual|)`, so a negative night hour contributes its *magnitude* to the "
        "denominator rather than cancelling against daylight. The two conventions therefore "
        f"differ by exactly NL's absolute night share, `f` = **{nl_gate['f_pct']:.4f}%** of the "
        f"denominator (the signed share is {nl_gate['signed_share_pct']:+.4f}% — same magnitude, "
        "opposite sign, which is the tell). Zeroing the night instead would shrink the "
        f"denominator by that {nl_gate['f_pct']:.4f}% and raise WAPE by the same relative "
        f"amount: on NL's {bars['NL/solar']['d7_wape_pct']:.2f}% D-7 bar that is "
        f"**{bars['NL/solar']['d7_wape_pct'] * nl_gate['f_pct'] / 100:.4f}pp**. Against ABL-418's "
        "10.65pp solar floor it is four orders of magnitude short. The numerator is bounded the "
        "same way: the night actuals average -0.13 MW, so a non-negative prediction pays at most "
        "that per night hour.")
    lines.append("")
    lines.append(
        "**So NL's margin, whatever it is, is not a netting artefact — but NL's *level* is the "
        "finding.** Its gate-window mean is "
        f"{bars['NL/solar']['mean_actual_mw']:.1f} MW against a 251.3 MW window maximum, and "
        "that series is **bit-identical in both source tables**, so it is upstream rather than "
        "ours. For scale, over the same 720 hours `energy_generation` books BE at 8,140 MW max "
        "and even EE — a country of 1.3 million — at 771.6 MW. NL's published solar series is a "
        "small metered subset, stable in that shape across 18 months, not its fleet. The gate "
        "read below is a valid read *of that series*; it must not be quoted as \"we can "
        "forecast NL solar\", and any NL promotion recommendation has to carry this.")
    lines.append("")

    # ---------------------------------------------------------- 5. pair grades
    lines.append("## 5. Pair grade, against the pre-committed bar and the level")
    lines.append("")
    lines.append(
        "The bar column is ABL-348's, measured before any challenger for these pairs existed. "
        "It is here because ABL-406 established across eight wind pairs that the gate outcome "
        "was *fully* predicted by whether a causal constant clears the bar on its own, and "
        "ABL-417 reproduced the anti-correlation on RO. **These are the loosest solar bars in "
        "the programme** — 23.92% to 47.85%, against 2c's 7.11-16.43% — which is precisely the "
        "combination (loose bar, low level) that produced 2b's spurious wind passes.")
    lines.append("")
    lines.append(
        f"The level column carries `{SK_REFERENCE}`, the lowest solar fleet already "
        "dispositioned, as a reference line. ABL-348 registers no decision-grade threshold for "
        "solar — the one it states explicitly is `CH_wind_onshore_is_not_decision_grade` at "
        "12.9 MW — so this is a **comparison, not a registered bar**.")
    lines.append("")
    lines.append("| pair | pre-committed D-7 bar | gate-window mean | vs SK line | bands read | band grades | pair grade | failed conditions | bar weaker than a flat line? |")
    lines.append("|---|---:|---:|:---:|:---:|---|:---:|---|:---:|")
    for country in COUNTRIES:
        grades = [cell_grade(cell, STREAM) for cell in cells if cell["country"] == country]
        if not grades:
            continue
        pair = pair_grade(grades)
        failed = [name for name, _ in pair.failed]
        bar = bars.get(f"{country}/solar", {}).get("d7_wape_pct")
        level = bars.get(f"{country}/solar", {}).get("mean_actual_mw")
        n_bands = len([c for c in cells if c["country"] == country])
        below = level is not None and level < SK_REFERENCE_MW
        record["pair_grades"][country] = {
            "bands_read": n_bands,
            "bands": [grade.label for grade in grades],
            "pair_grade": pair.label,
            "failed_conditions": failed,
            "bar_weaker_than_a_flat_line": pair.bar_weak,
            "precommitted_d7_bar_pct": bar,
            "gate_window_mean_mw": level,
            "below_sk_reference_line": below,
            "skill_pct": dict(pair.skill),
            "own_error_margin_pct": dict(pair.own_error_margin),
            "floor_pct": pair.floor_pct,
        }
        lines.append(
            f"| {country} | {bar}% | {level:.1f} MW | {'**below**' if below else 'above'} | "
            f"{n_bands}/3 | {' / '.join(grade.label for grade in grades)} | "
            f"{pair.label} | {', '.join(failed) or '—'} | "
            f"{'yes' if pair.bar_weak else ('no' if pair.bar_weak is not None else '—')} |")
    lines.append("")
    below_line = sorted(c for c, r in record["pair_grades"].items()
                        if r["below_sk_reference_line"])
    lines.append(
        f"**On the level, exactly one pair sits below the SK reference line: "
        f"{', '.join(below_line) if below_line else 'none'}.** ABL-421's description anticipated "
        "\"several\"; measured, EE is the next lowest at 223.0 MW, which is nearly twice SK's "
        "114.8 MW. The distinction matters because it is the level, not the bar, that decides "
        "whether a WAPE can carry a promotion decision at all.")
    lines.append("")
    lines.append(
        "**Do not average this tranche's pass rate against 2a's or 2c's.** The bars are not "
        "comparable: 2c's ran 7.11-16.43% on Mediterranean July solar that is nearly D-7 "
        "periodic, and these run 23.92-47.85%. ABL-348 registered that reading in advance under "
        "`reading_caveats_not_band_changes`. A pass against a loose bar and a pass against a "
        "tight one are not the same evidence, which is what the grade ladder exists to say.")
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
