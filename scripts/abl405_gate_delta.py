#!/usr/bin/env python3
"""Read a solar gate result under ABL-385's decision margin, and diff two reads.

Two jobs, and they take different margins. That is the whole reason this exists
as one script rather than as arithmetic in a report.

1. **A read against its model-free references.** `seasonal_naive`, the two
   constants and the two climatologies are all *deterministic* functions of the
   data: refit the challenger at another seed and they do not move. So in
   ABL-385's

       delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)

   the reference's own CV `c_B` is **0**, and the k=1 margin is `1.96 * c_A`, not
   `1.96 * sqrt(2) * c_A`. At the registered fleet p90 CV of 5.43% that is
   **10.64%** of the challenger's own error, where a challenger-vs-challenger
   comparison would need 15.06%. Quoting the larger number against a constant is
   not conservatism, it is the wrong test.

2. **Two reads of the same cell against each other.** Here both arms are fitted,
   so `c_B` is not 0 and the margin *is* 15.06% at k=1 and the fleet p90. ABL-385
   section 4 measured the matched-seed correlation this assumption ignores at
   **0.113** across 48 cells, which makes the independent form mildly
   conservative rather than wrong; the correlation-adjusted margin is reported
   beside it and is never used to promote a verdict the independent one refuses.

Everything is quoted as a percentage of the **challenger's own error**, which is
ABL-385's registered form. A gap in WAPE points is not comparable across cells
whose challengers score 8% and 25%.

`--pair-cv` overrides the fleet percentile with a pair-specific CV where one has
been measured. The fleet p90 is a fleet percentile, and ABL-402 measured it about
2x too wide on BG and CH; a pair that has its own number should be read on it,
and the fleet value stays as the conservative default for a pair that does not.

Read-only. Consumes gate result JSONs and writes its own; fits nothing, opens no
database, and touches no registration table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#: ABL-385's registered fleet p90 per-fit CV, as a percentage of a fit's own
#: error. 1.96 * 5.43 = 10.64 and 1.96 * sqrt(2) * 5.43 = 15.06, which are the
#: two margins that report quotes at k = 1.
FLEET_P90_CV_PCT = 5.43
#: ABL-385 section 4: Fisher-z mean correlation between two same-algorithm arms
#: at matched seeds, over 48 cells.
MATCHED_SEED_CORRELATION = 0.113
#: The comparators that do not move when the challenger is refitted. Their CV is
#: 0 by construction, which is what halves the margin against them.
DETERMINISTIC_COMPARATORS = ("seasonal_naive", "persistence", "constant_causal",
                             "constant_oracle", "climatology_causal",
                             "climatology_oracle")


def margin_pct(c_a: float, c_b: float, k: int = 1,
               correlation: float = 0.0) -> float:
    """ABL-385's delta_min(k), as a percentage of a fit's own error.

    `correlation` is 0 for the independent form the registration states. A
    positive value is the measured matched-seed case and shrinks the margin,
    because part of the noise cancels inside the difference.
    """
    variance = c_a ** 2 + c_b ** 2 - 2 * correlation * c_a * c_b
    return 1.96 * (max(variance, 0.0) ** 0.5) / (k ** 0.5)


def feature_set_of(result: dict) -> tuple[str, int]:
    """The read's feature set, defaulted the way the harness renderer defaults it.

    A gate result written before ABL-395 carries no `feature_set`/`n_features` --
    ABL-381's does not. `render_markdown` reads those with
    `.get('feature_set', 'legacy25')` / `.get('n_features', 25)`, because every
    read up to that point was taken on the 25. Reporting `None` here instead
    would make the one comparison this script exists for -- 27 against 25 --
    unlabelled on the 25 side.
    """
    meta = result.get("meta", {})
    return (meta.get("feature_set") or "legacy25",
            meta.get("n_features") or 25)


def _cells(result: dict) -> dict:
    return {(row["country"], row["horizon_band"]): row
            for row in result.get("gate_cells", [])}


def _wape(row: dict, name: str):
    entry = row.get("scores", {}).get(name)
    return None if entry is None else entry.get("wape_pct")


def read_vs_references(result: dict, cv_for) -> list[dict]:
    """Every cell against every deterministic reference, at the k=1 margin."""
    out = []
    for (country, band), row in sorted(_cells(result).items()):
        challenger = _wape(row, "challenger")
        cv = cv_for(country)
        # c_B = 0: the reference does not move when the challenger is refitted.
        bar_pct = margin_pct(cv, 0.0)
        entry = {"country": country, "horizon_band": band,
                 "n": row["gate"]["n"], "gate_pass": row["gate"]["pass"],
                 "challenger_wape_pct": challenger,
                 "challenger_cv_pct": cv,
                 "margin_pct_of_own_error": bar_pct,
                 "comparisons": []}
        for name in DETERMINISTIC_COMPARATORS:
            reference = _wape(row, name)
            comparator_n = row.get("comparator_n", {}).get(name)
            if challenger is None or reference is None or not challenger:
                entry["comparisons"].append(
                    {"reference": name, "reference_wape_pct": reference,
                     "comparator_n": comparator_n, "readable": None,
                     "gap_pct_of_own_error": None,
                     "scored_on_same_rows": None})
                continue
            # Positive: the challenger is better than the reference.
            gap = 100.0 * (reference - challenger) / challenger
            entry["comparisons"].append({
                "reference": name, "reference_wape_pct": reference,
                "comparator_n": comparator_n,
                # A comparator scored on a different row count is not scored on
                # the same measurement. A climatology is 24 buckets and is the
                # first reference that can be partially measured.
                "scored_on_same_rows": comparator_n == row["gate"]["n"],
                "gap_pct_of_own_error": gap,
                "wins": gap > 0,
                "readable": abs(gap) > bar_pct,
            })
        out.append(entry)
    return out


def read_vs_read(result: dict, reference: dict, cv_for) -> list[dict]:
    """Cells present in both reads, at the challenger-vs-challenger margin."""
    read_cells, reference_cells = _cells(result), _cells(reference)
    out = []
    for key in sorted(set(read_cells) & set(reference_cells)):
        country, band = key
        new, old = _wape(read_cells[key], "challenger"), _wape(reference_cells[key], "challenger")
        cv = cv_for(country)
        independent = margin_pct(cv, cv)
        adjusted = margin_pct(cv, cv, correlation=MATCHED_SEED_CORRELATION)
        row = {"country": country, "horizon_band": band,
               "read_wape_pct": new, "reference_wape_pct": old,
               "read_n": read_cells[key]["gate"]["n"],
               "reference_n": reference_cells[key]["gate"]["n"],
               "challenger_cv_pct": cv,
               "margin_pct_independent": independent,
               "margin_pct_correlation_adjusted": adjusted}
        if new is not None and old is not None and old:
            # Signed so that positive is the *new* read being worse, which is
            # the direction a reader of a feature-list change cares about.
            delta_pp = new - old
            row.update({
                "delta_pp": delta_pp,
                "delta_pct_of_reference_error": 100.0 * delta_pp / old,
                "moves_more_than_margin": abs(100.0 * delta_pp / old) > independent,
                "moves_more_than_adjusted_margin": abs(100.0 * delta_pp / old) > adjusted,
                # A delta measured across a different row count is confounded by
                # the rows, not only by the change under test.
                "same_row_count": read_cells[key]["gate"]["n"] == reference_cells[key]["gate"]["n"],
            })
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--read", required=True,
                        help="Gate result JSON to read")
    parser.add_argument("--reference", default=None,
                        help="Second gate result JSON to diff the shared cells against")
    parser.add_argument("--pair-cv", default=None,
                        help="Per-country CV override, e.g. 'BG=2.52,CH=3.02'. "
                             "Countries not named take the registered fleet p90.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = json.loads(Path(args.read).read_text(encoding="utf-8"))
    overrides = {}
    if args.pair_cv:
        for item in args.pair_cv.split(","):
            country, _, value = item.strip().partition("=")
            overrides[country.strip().upper()] = float(value)

    def cv_for(country: str) -> float:
        return overrides.get(country, FLEET_P90_CV_PCT)

    payload = {
        "read": str(Path(args.read).resolve()),
        "read_scope": result["meta"]["scope"],
        "read_feature_set": feature_set_of(result)[0],
        "read_n_features": feature_set_of(result)[1],
        "fleet_p90_cv_pct": FLEET_P90_CV_PCT,
        "matched_seed_correlation": MATCHED_SEED_CORRELATION,
        "pair_cv_overrides": overrides,
        "margin_k1_vs_deterministic_reference_pct": margin_pct(FLEET_P90_CV_PCT, 0.0),
        "margin_k1_vs_another_fit_pct": margin_pct(FLEET_P90_CV_PCT, FLEET_P90_CV_PCT),
        "cells_vs_references": read_vs_references(result, cv_for),
    }
    if args.reference:
        other = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        payload["reference"] = str(Path(args.reference).resolve())
        payload["reference_scope"] = other["meta"]["scope"]
        payload["reference_feature_set"] = feature_set_of(other)[0]
        payload["reference_n_features"] = feature_set_of(other)[1]
        payload["reference_feature_set_was_defaulted"] = (
            other.get("meta", {}).get("feature_set") is None)
        payload["cells_vs_reference_read"] = read_vs_read(result, other, cv_for)

    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
