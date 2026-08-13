#!/usr/bin/env python3
"""Read ABL-406 tranche 2b's gate result against ABL-385's decision margin.

The gate harness answers "did the challenger beat the registered D-7 bar". This
answers the two questions that qualify that answer, and it is deliberately a
separate read: neither is a gate criterion, and computing them inside the
harness would put them one edit away from becoming one.

  1. **Is each comparison readable at all?**  Every reference in this report is
     deterministic -- seasonal-naive D-7, a flat line, an hour-of-day
     climatology -- so ABL-385's delta method has `c_B = 0` and the two-arm
     margin it publishes is a factor of sqrt(2) too wide here. The gate fits one
     model per pair at the pinned seed, so k = 1.

  2. **Was the registered bar what established the pass?**  ABL-380 measured
     that BG's registered 93.75% D-7 bar is cleared outright by a causal
     constant at 82.77% -- no model. Where D-7 is looser than the causal
     constant, a cell can clear the bar while losing to a flat line, and the
     bar is not what the PASS rests on. That is a statement about the bar and
     not about the model, and it is reported per pair rather than aggregated.

Reads the gate's own machine record. It refits nothing and opens no database, so
it cannot disagree with the report it qualifies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#: ABL-385 section 1, the **wind** stream: pooled per-fit CV over 12 (pair,
#: algorithm, arm) units, p90. Named here rather than borrowed from the solar
#: table, which is the mistake this constant exists to prevent -- the two
#: streams differ by a factor of 1.4 (solar p90 5.43%) and ABL-381 read its
#: margins against a percentile taken over a different stream's fits.
#:
#: It is a *fleet* percentile over the served pairs, and none of the eight pairs
#: in this tranche is among them. ABL-402 measured what that substitution costs,
#: on the only two pairs where both numbers exist: the fleet percentile ran
#: **1.8-2.4x wider** than the pairs' own CV. So this is conservative in the
#: direction that matters -- a margin clearing it is readable, a margin failing
#: it may still be readable and needs a pair-specific CV to say.
ABL385_WIND_FLEET_P90_CV = 0.038292934379344015

#: The same percentile restricted to the four units that match this challenger's
#: stream *and* algorithm (wind_onshore / catboost / control): AT 1.81%,
#: DE 2.03%, FR 2.50%, BE 3.96%. Reported beside the fleet value as a check that
#: the fleet p90 is not being inflated by the offshore or xgboost units -- it is
#: not; the matched maximum is 3.96% against the fleet p90's 3.83%. Four units
#: cannot support a percentile, so the maximum is used and is not called one.
ABL385_MATCHED_ONSHORE_CATBOOST_CV_MAX = 0.039642996663702794

Z95 = 1.96

#: The four ABL-389 model-free references, plus the registered bar. Ordered as
#: the report reads them: the bar first, then the level, then the level and the
#: daily shape.
REFERENCES = ("seasonal_naive", "constant_causal", "constant_oracle",
              "climatology_causal", "climatology_oracle")

LABELS = {"seasonal_naive": "seasonal-naive D-7 (the registered bar)",
          "constant_causal": "constant, causal (fit-window mean)",
          "constant_oracle": "constant, oracle (gate-window median)",
          "climatology_causal": "hour-of-day climatology, causal",
          "climatology_oracle": "hour-of-day climatology, oracle"}


def delta_min_pct(cv: float, k: int = 1, deterministic_reference: bool = True) -> float:
    """ABL-385's readable-gap floor, as a percentage of the challenger's error.

    Args:
        cv: The challenger's per-fit coefficient of variation.
        k: Seeds averaged per arm. The gate fits once, so k = 1.
        deterministic_reference: True when the other arm carries no fit noise --
            `c_B = 0`, which is the case for every reference in this report.
            The published table assumes two stochastic arms and is a factor of
            sqrt(2) larger; using it unchanged against a deterministic reference
            declares real margins unreadable.

    Returns:
        The smallest relative gap readable at two-sided 95%, in percent.
    """
    variance = cv ** 2 if deterministic_reference else 2 * cv ** 2
    return 100.0 * Z95 * (variance ** 0.5) / (k ** 0.5)


def margin_pct(challenger_wape: float, reference_wape: float) -> float:
    """The gap as a percentage of the challenger's *own* error (ABL-385's g).

    Positive means the challenger is ahead. Expressed against the challenger's
    error rather than the reference's because that is the denominator the CV is
    a CV *of*, so the margin and the threshold are in the same units.
    """
    return 100.0 * (reference_wape - challenger_wape) / challenger_wape


def _cell_rows(result: dict, cv: float, k: int) -> list[dict]:
    floor = delta_min_pct(cv, k)
    rows = []
    for cell in result["gate_cells"]:
        scores = cell["scores"]
        challenger = scores["challenger"]["wape_pct"]
        for name in REFERENCES:
            entry = scores.get(name) or {}
            reference = entry.get("wape_pct")
            if challenger is None or reference is None:
                rows.append({"country": cell["country"], "band": cell["horizon_band"],
                             "reference": name, "margin_pct": None, "readable": None,
                             "challenger_wape": challenger, "reference_wape": reference,
                             "n": cell["gate"]["n"], "comparator_n": cell["comparator_n"].get(name)})
                continue
            margin = margin_pct(challenger, reference)
            rows.append({
                "country": cell["country"], "band": cell["horizon_band"],
                "reference": name, "margin_pct": margin,
                "delta_min_pct": floor, "readable": abs(margin) >= floor,
                "challenger_wape": challenger, "reference_wape": reference,
                "n": cell["gate"]["n"], "comparator_n": cell["comparator_n"].get(name),
            })
    return rows


def _bar_weakness(result: dict) -> list[dict]:
    """Per pair: is the registered D-7 bar looser than a causal constant?

    Read on the 24-36h cell and on every cell, because the bar is per cell. A
    pair reads `bar_is_weaker` when D-7's WAPE exceeds the causal constant's --
    i.e. a flat line at the fit-window mean would itself clear the registered
    bar, so clearing it is not evidence of a model.
    """
    out = []
    for cell in result["gate_cells"]:
        scores = cell["scores"]
        d7 = scores["seasonal_naive"]["wape_pct"]
        const = (scores.get("constant_causal") or {}).get("wape_pct")
        clim = (scores.get("climatology_causal") or {}).get("wape_pct")
        out.append({
            "country": cell["country"], "band": cell["horizon_band"],
            "d7_wape": d7, "constant_causal_wape": const, "climatology_causal_wape": clim,
            # A bar looser than the no-model reference is a bar the reference
            # clears. Both comparisons are causal: an oracle reference knows the
            # gate window and could not have been used to set a bar in advance.
            "bar_weaker_than_constant": None if (d7 is None or const is None) else d7 > const,
            "bar_weaker_than_climatology": None if (d7 is None or clim is None) else d7 > clim,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/ABL348/results_abl406_tranche2b.json")
    parser.add_argument("--json-out", default="reports/abl_406_margins.json")
    parser.add_argument("--cv", type=float, default=ABL385_WIND_FLEET_P90_CV,
                        help="Per-fit CV of the challenger arm (default: ABL-385 wind fleet p90)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds averaged per arm; the gate fits once (default: 1)")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = _cell_rows(result, args.cv, args.seeds)
    weakness = _bar_weakness(result)
    payload = {
        "issue": "ABL-406",
        "reads": args.results,
        "scope": result["meta"]["scope"],
        "verdict": result["verdict"],
        "margin_protocol": {
            "definition": "(reference WAPE - challenger WAPE) / challenger WAPE, in percent",
            "cv_source": ("ABL-385 section 1, wind stream, pooled per-fit CV p90 over 12 "
                          "(pair, algorithm, arm) units"),
            "cv": args.cv,
            "matched_onshore_catboost_cv_max": ABL385_MATCHED_ONSHORE_CATBOOST_CV_MAX,
            "seeds_per_arm": args.seeds,
            "reference_arm_is_deterministic": True,
            "c_B": 0.0,
            "delta_min_pct": delta_min_pct(args.cv, args.seeds),
            "delta_min_pct_if_both_arms_were_stochastic": delta_min_pct(
                args.cv, args.seeds, deterministic_reference=False),
            "caveat": ("A fleet percentile over the served pairs, none of which is in this "
                       "tranche. ABL-402 measured it 1.8-2.4x wider than the pair-specific CV "
                       "on the two pairs where both exist, so it is conservative: a margin that "
                       "clears it is readable, a margin that fails it is unresolved rather than "
                       "absent."),
        },
        "cells": rows,
        "bar_weakness": weakness,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.stdout:
        floor = delta_min_pct(args.cv, args.seeds)
        print(f"delta_min(k={args.seeds}) vs a deterministic reference: {floor:.2f}% "
              f"of the challenger's own error (CV {100 * args.cv:.2f}%)")
        print(f"  the same CV read as two stochastic arms would give "
              f"{delta_min_pct(args.cv, args.seeds, False):.2f}% -- not used here")
        print()
        for name in REFERENCES:
            subset = [r for r in rows if r["reference"] == name and r["margin_pct"] is not None]
            if not subset:
                print(f"{LABELS[name]}: Not measured in any cell")
                continue
            ahead = sum(r["margin_pct"] > 0 for r in subset)
            readable = sum(bool(r["readable"]) for r in subset)
            print(f"{LABELS[name]}: challenger ahead in {ahead}/{len(subset)} cells, "
                  f"{readable}/{len(subset)} readable at k={args.seeds}")
        print()
        print("Pairs whose registered D-7 bar is looser than a causal constant:")
        weak = sorted({w["country"] for w in weakness if w["bar_weaker_than_constant"]})
        print("  " + (", ".join(weak) if weak else "none"))
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
