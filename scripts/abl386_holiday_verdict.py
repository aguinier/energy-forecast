"""ABL-386: apply the registered verdict rule to the holdout JSONs. No judgement here.

Everything this script does is fixed by `experiments/ABL386/config.json`, which
was committed before any arm was fitted. It exists so the verdict is a lookup
rather than something read off a table by eye - the thresholds, the sign
convention, the materiality rule and the recommendation mapping are all
pre-registered, and this file is just the arithmetic.

The registered rule, restated so it can be checked against the config:

- **Unit**: one (country, algorithm) cell. 4 countries x 2 algorithms = 8 cells.
- **Statistic**: seed-mean daylight MAE over seeds 42/1337/2718.
- **Primary contrast**: `geometry` (31 names, holidays) vs `geometry_noholiday`
  (27 names, no holidays). Both carry ABL-338 geometry, which is unconditional
  in `src/features.py`, so this is the decision as it would really be taken.
- **Sign**: `effect_pct = 100 * (holidays - no_holidays) / no_holidays`. Positive
  means the holiday features make it **worse**.
- **Materiality**: a cell counts only if the two arms' seed ranges are
  **disjoint** - `max(better) < min(worse)`. Not a fixed percentage: ABL-375
  registered 3.0% and then measured DE CatBoost moving 13.79% of its own mean
  across three seeds on this same window.
- **Verdict**: `d = +1` disjoint and holidays better, `-1` disjoint and
  no-holidays better, `0` overlapping. HELP if `sum(d) >= +4`, HARM if
  `sum(d) <= -4`, NO_EFFECT if at most 2 cells are disjoint at all, else MIXED.
- **Replicate**: the same rule on `control` (29) vs `control_noholiday` (25). A
  direction disagreement downgrades the verdict to MIXED.
- **Night guardrail**: excluding holidays must not raise `abs(night mean pred)`
  in any cell by more than that cell's own seed spread of the same quantity.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl386_holiday_verdict.py \\
        --inputs reports/abl_386_solar/holdout_abl386_catboost_cleaned.json \\
                 reports/abl_386_solar/holdout_abl386_xgboost_cleaned.json \\
        --out reports/abl_386_holiday_verdict_tables
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#: (holiday arm, no-holiday arm, label). Primary first.
CONTRASTS = [
    ("geometry", "geometry_noholiday", "primary", 31, 27),
    ("control", "control_noholiday", "replicate", 29, 25),
]

SEEDS = (42, 1337, 2718)


def _cells(payload: dict, algorithm: str, hol_arm: str, nohol_arm: str) -> list:
    """One record per country for a single contrast, with the seed spread of each arm."""
    out = []
    for country, block in payload["countries"].items():
        arms = block["arms"]
        rec = {"country": country, "algorithm": algorithm}
        ok = True
        for tag, arm in (("hol", hol_arm), ("nohol", nohol_arm)):
            mae, night, trees = [], [], []
            for s in SEEDS:
                key = f"{arm}@{s}"
                if key not in arms:
                    ok = False
                    break
                mae.append(arms[key]["daylight"]["mae_mw"])
                night.append(arms[key]["night"]["mean_pred_mw"])
                trees.append(arms[key].get("n_trees"))
            if not ok:
                break
            rec[f"{tag}_mae"] = mae
            rec[f"{tag}_mean"] = statistics.fmean(mae)
            rec[f"{tag}_min"] = min(mae)
            rec[f"{tag}_max"] = max(mae)
            # Spread as a share of the arm's own mean - the quantity ABL-375
            # found to be 13.79% on DE CatBoost, which is why no fixed
            # percentage threshold is registered here.
            rec[f"{tag}_spread_pct"] = 100.0 * (max(mae) - min(mae)) / statistics.fmean(mae)
            rec[f"{tag}_night_mean"] = statistics.fmean(night)
            rec[f"{tag}_night_abs_mean"] = statistics.fmean([abs(v) for v in night])
            rec[f"{tag}_night_abs_min"] = min(abs(v) for v in night)
            rec[f"{tag}_night_abs_max"] = max(abs(v) for v in night)
            rec[f"{tag}_trees"] = trees
        if not ok:
            continue

        rec["effect_pct"] = 100.0 * (rec["hol_mean"] - rec["nohol_mean"]) / rec["nohol_mean"]
        # Disjoint in the registered sense: the better arm's worst seed still
        # beats the worse arm's best seed.
        if rec["hol_max"] < rec["nohol_min"]:
            rec["d"], rec["disjoint"], rec["favours"] = 1, True, "holidays"
        elif rec["nohol_max"] < rec["hol_min"]:
            rec["d"], rec["disjoint"], rec["favours"] = -1, True, "no_holidays"
        else:
            rec["d"], rec["disjoint"], rec["favours"] = 0, False, "overlapping"

        # Night guardrail, on the absolute value per the registration.
        nohol_spread = rec["nohol_night_abs_max"] - rec["nohol_night_abs_min"]
        rec["night_abs_increase_from_excluding"] = (
            rec["nohol_night_abs_mean"] - rec["hol_night_abs_mean"]
        )
        rec["night_guardrail_pass"] = (
            rec["night_abs_increase_from_excluding"] <= max(nohol_spread, 0.0)
        )
        rec["night_guardrail_seed_spread_mw"] = nohol_spread
        out.append(rec)
    return out


def _verdict(cells: list) -> dict:
    total = sum(c["d"] for c in cells)
    n_disjoint = sum(1 for c in cells if c["disjoint"])
    if total >= 4:
        v = "HELP"
    elif total <= -4:
        v = "HARM"
    elif n_disjoint <= 2:
        v = "NO_EFFECT"
    else:
        v = "MIXED"
    return {
        "sum_d": total,
        "n_cells": len(cells),
        "n_disjoint": n_disjoint,
        "n_disjoint_favouring_holidays": sum(1 for c in cells if c["d"] == 1),
        "n_disjoint_favouring_exclusion": sum(1 for c in cells if c["d"] == -1),
        "verdict": v,
    }


RECOMMENDATION = {
    "NO_EFFECT": ("EXCLUDE", "Registered mapping. Four features that demonstrably do nothing on "
                             "this target are four extra split candidates and a live `holidays` "
                             "dependency in the solar path, and they entered the serving list by "
                             "accident rather than by decision. Excluding makes the list deliberate."),
    "HARM": ("EXCLUDE", "Registered mapping, and the harm is measured."),
    "HELP": ("KEEP", "Registered mapping. They must then also be evaluated on the other seven "
                     "types, since no serving artifact of any type carries them."),
    "MIXED": ("REPORT", "Registered mapping: recommend exclusion on parsimony only if no cell "
                        "shows disjoint HELP; otherwise name what a further read must measure."),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    # Deliberately NOT reports/abl_386_solar_holiday_verdict: that stem is the
    # hand-written evidence pack, and a re-run here must not overwrite it.
    parser.add_argument("--out", default="reports/abl_386_holiday_verdict_tables")
    args = parser.parse_args()

    payloads = {}
    for p in args.inputs:
        blob = json.loads(Path(p).read_text(encoding="utf-8"))
        payloads[blob["force_algorithm"]] = blob

    results = {}
    for hol_arm, nohol_arm, label, n_hol, n_nohol in CONTRASTS:
        cells = []
        for algorithm, payload in sorted(payloads.items()):
            cells.extend(_cells(payload, algorithm, hol_arm, nohol_arm))
        results[label] = {
            "holiday_arm": hol_arm, "no_holiday_arm": nohol_arm,
            "n_features_holiday": n_hol, "n_features_no_holiday": n_nohol,
            "cells": cells, **_verdict(cells),
        }

    primary, replicate = results["primary"], results["replicate"]
    final = primary["verdict"]
    downgraded = False
    if {primary["verdict"], replicate["verdict"]} == {"HELP", "HARM"}:
        final, downgraded = "MIXED", True

    action, why = RECOMMENDATION[final]
    payload = {
        "issue": "ABL-386",
        "registration": "experiments/ABL386/config.json",
        "seeds": list(SEEDS),
        "results": results,
        "primary_verdict": primary["verdict"],
        "replicate_verdict": replicate["verdict"],
        "downgraded_by_replicate_disagreement": downgraded,
        "final_verdict": final,
        "recommendation": action,
        "recommendation_rationale": why,
        "night_guardrail_failures": [
            f"{c['algorithm']}/{c['country']}"
            for c in primary["cells"] if not c["night_guardrail_pass"]
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"# ABL-386 verdict: **{final}** -> recommend **{action}**", ""]
    for label in ("primary", "replicate"):
        r = results[label]
        lines += [
            f"## {label}: `{r['holiday_arm']}` ({r['n_features_holiday']}) vs "
            f"`{r['no_holiday_arm']}` ({r['n_features_no_holiday']})",
            "",
            f"verdict **{r['verdict']}** - sum(d) {r['sum_d']:+d} over {r['n_cells']} cells, "
            f"{r['n_disjoint']} disjoint "
            f"({r['n_disjoint_favouring_holidays']} favour keeping, "
            f"{r['n_disjoint_favouring_exclusion']} favour excluding)",
            "",
            "| cell | daylight MAE, holidays (31/29) | no holidays (27/25) | effect | "
            "spread hol / nohol | ranges | d |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for c in sorted(r["cells"], key=lambda x: (x["algorithm"], x["country"])):
            lines.append(
                f"| {c['algorithm']}/{c['country']} "
                f"| {c['hol_mean']:,.1f} ({c['hol_min']:,.1f}-{c['hol_max']:,.1f}) "
                f"| {c['nohol_mean']:,.1f} ({c['nohol_min']:,.1f}-{c['nohol_max']:,.1f}) "
                f"| {c['effect_pct']:+.2f}% "
                f"| {c['hol_spread_pct']:.2f}% / {c['nohol_spread_pct']:.2f}% "
                f"| {'**disjoint**' if c['disjoint'] else 'overlapping'} "
                f"| {c['d']:+d} |"
            )
        lines.append("")

    lines += ["## Night guardrail (registered as |night mean|, MW)", "",
              "| cell | holidays | no holidays | change from excluding | "
              "nohol seed spread | pass |", "|---|---:|---:|---:|---:|---|"]
    for c in sorted(primary["cells"], key=lambda x: (x["algorithm"], x["country"])):
        lines.append(
            f"| {c['algorithm']}/{c['country']} | {c['hol_night_abs_mean']:,.1f} "
            f"| {c['nohol_night_abs_mean']:,.1f} "
            f"| {c['night_abs_increase_from_excluding']:+,.1f} "
            f"| {c['night_guardrail_seed_spread_mw']:,.1f} "
            f"| {'PASS' if c['night_guardrail_pass'] else '**FAIL**'} |"
        )
    lines += ["", f"_{why}_", ""]

    markdown = "\n".join(lines)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
