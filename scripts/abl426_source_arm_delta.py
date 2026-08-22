#!/usr/bin/env python3
"""ABL-426 -- what did reading tranche 2a on the wrong table actually cost?

Differences two committed machine records of the **same eight countries, windows,
bands, basis, fit rule, feature vector and grading registration**, which differ in
exactly one registered value: the source table.

    A  `abl316-t2a`             energy_renewable   ABL-405, published 2026-08-13
    B  `abl316-t2a-generation`  energy_generation  ABL-426, the registered read

Read-only over two JSON files. Opens no database, fits nothing, writes nothing to
either database, and does not re-grade: `cell_grade` returns the letter each
record *stored* where it has one (ABL-437's rule that a read is not re-decided
under a later registration) and computes it only where none exists.

**The D-7 column is the control.** ABL-348 pre-measured the seasonal-naive D-7 bar
on both tables before any challenger existed and recorded `bar_delta_pp = 0.00`
for all eight of these countries. D-7 is model-free, so if the two records' D-7
disagree the cause is not the table -- it is that arm B was read on a later
replica snapshot than arm A (10,175,365,120 bytes against 9,432,453,120). That
makes the D-7 delta a direct measurement of the replica-vintage confound, and it
is reported per cell beside the challenger delta rather than assumed away.

    .venv\\Scripts\\python.exe scripts/abl426_source_arm_delta.py --out reports/abl_426_source_arm_delta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    DELTA_MIN, SIGN_TEST, cell_grade, readability_floor_pct, skill_pct,
)
from src.evaluation.model_free_reference import FIT_WINDOW  # noqa: E402

#: The two arms, in the order the tables are argued about. `label` is what the
#: report prints; `scope` is asserted against each record's own `meta.scope`, so
#: pointing this script at the wrong file fails rather than mislabels.
ARM_A = {"key": "renewable", "scope": "abl316-t2a",
         "path": "experiments/ABL348/results_abl405_tranche2a.json",
         "expect_source": "energy_renewable"}
ARM_B = {"key": "generation", "scope": "abl316-t2a-generation",
         "path": "experiments/ABL348/results_abl426_tranche2a_generation.json",
         "expect_source": "energy_generation"}

#: The references the ABL-418 ladder scores, plus the two oracle forms ABL-316's
#: shipping decision turns on. Named here so a reference added later is an
#: explicit edit rather than a silently narrower comparison.
REFERENCES = ("seasonal_naive", "constant_causal", "climatology_causal",
              "constant_oracle", "climatology_oracle")

STREAM = "solar"
K = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(arm: dict) -> tuple[dict, str]:
    path = ROOT / arm["path"]
    if not path.exists():
        raise SystemExit(f"missing machine record: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    meta = record["meta"]
    # Both are asserted, not read: this script's whole claim is that the two
    # records differ in the source table and in nothing else registered, and a
    # mixed-up path would produce a plausible table of deltas for the wrong pair.
    if meta["scope"] != arm["scope"]:
        raise SystemExit(f"{path} records scope {meta['scope']!r}, expected {arm['scope']!r}")
    if meta["training_source"] != arm["expect_source"]:
        raise SystemExit(f"{path} records training_source {meta['training_source']!r}, "
                         f"expected {arm['expect_source']!r}")
    return record, _sha256(path)


def _wape(cell: dict, comparator: str):
    return cell["scores"].get(comparator, {}).get("wape_pct")


def _n(cell: dict, comparator: str):
    return cell["scores"].get(comparator, {}).get("n")


def _delta(b, a):
    """B - A in percentage points, or None where either side was not measured."""
    return None if a is None or b is None else b - a


def _verdict(cell: dict) -> str:
    """PASS / FAIL / no gate block, from ABL-434's `gate.pass`.

    Rendered as the string the reports print rather than the raw bool, so a
    verdict change reads as `PASS -> FAIL` in the summary instead of `True ->
    False` -- these two arms are read side by side with two harness reports that
    print the words.
    """
    block = cell.get("gate")
    if not isinstance(block, dict) or "pass" not in block:
        return "not recorded"
    return "PASS" if block["pass"] else "FAIL"


def _grade(cell: dict) -> str:
    # `abl316-t2a` and `abl316-t2a-generation` both pin FIT_WINDOW levelling and
    # SIGN_TEST readability, and both are k = 1, so the same call serves both
    # arms. Passing the pins explicitly rather than defaulting is deliberate:
    # `cell_grade`'s defaults are TRAILING_28D/FLOORED, which is neither arm's
    # registration, and a silently re-levelled letter is exactly what ABL-437
    # forbade.
    # `.label`, not `.grade`: the ladder distinguishes `U` from `U(+)` and the
    # bare field flattens the two. Three of ABL-405's published 2a cells are
    # `U(+)`, so reading `.grade` here would report a grade change on HU that is
    # only a rendering.
    return cell_grade(cell, STREAM, K, levelling=FIT_WINDOW,
                      g23_readability=SIGN_TEST, seed_readability=DELTA_MIN).label


def _grade_conditions(cell: dict) -> dict:
    """G1..G4 for the cell, so a letter change can be attributed to a condition."""
    return cell_grade(cell, STREAM, K, levelling=FIT_WINDOW,
                      g23_readability=SIGN_TEST,
                      seed_readability=DELTA_MIN).conditions


def compare(a_record: dict, b_record: dict) -> dict:
    floor = readability_floor_pct(STREAM, K)
    a_cells = {(c["country"], c["horizon_band"]): c for c in a_record["gate_cells"]}
    b_cells = {(c["country"], c["horizon_band"]): c for c in b_record["gate_cells"]}

    only_a = sorted(set(a_cells) - set(b_cells))
    only_b = sorted(set(b_cells) - set(a_cells))

    rows = []
    for key in sorted(set(a_cells) & set(b_cells)):
        a, b = a_cells[key], b_cells[key]
        # `gate` is ABL-434's coverage block, not a verdict string: it carries
        # `pass`, `beats_d7`, `enough_pairs` and the cell's own n. Comparing the
        # whole block would fire `gate_changed` on an n difference -- which CZ and
        # PL have between the two tables -- and report a coverage change as a
        # verdict change. The verdict is `pass`; the rest is reported beside it.
        row = {
            "country": key[0], "horizon_band": key[1],
            "n": {"renewable": _n(a, "challenger"), "generation": _n(b, "challenger")},
            "gate": {"renewable": _verdict(a), "generation": _verdict(b),
                     "block_renewable": a.get("gate"), "block_generation": b.get("gate")},
            "grade": {"renewable": _grade(a), "generation": _grade(b),
                      "conditions_renewable": _grade_conditions(a),
                      "conditions_generation": _grade_conditions(b)},
            "references": {},
        }
        row["gate_changed"] = row["gate"]["renewable"] != row["gate"]["generation"]
        # Reported separately, because the two fail for different reasons: a
        # `beats_d7` change is the model, an `enough_pairs` change is coverage,
        # and ABL-421's rule is that an unmeasured condition is not a satisfied one.
        row["gate_components_changed"] = sorted(
            component for component in ("beats_d7", "enough_pairs")
            if (a.get("gate") or {}).get(component) != (b.get("gate") or {}).get(component))
        row["grade_changed"] = row["grade"]["renewable"] != row["grade"]["generation"]
        ca, cb = _wape(a, "challenger"), _wape(b, "challenger")
        row["challenger_wape_pct"] = {"renewable": ca, "generation": cb,
                                      "delta_pp": _delta(cb, ca)}
        for ref in REFERENCES:
            ra, rb = _wape(a, ref), _wape(b, ref)
            row["references"][ref] = {
                "wape_pct": {"renewable": ra, "generation": rb, "delta_pp": _delta(rb, ra)},
                "skill_pct": {"renewable": skill_pct(ca, ra), "generation": skill_pct(cb, rb),
                              "delta_pp": _delta(skill_pct(cb, rb), skill_pct(ca, ra))},
                # The readability floor is what the ABL-418 ladder compares a
                # margin against, so "readable" is reported per arm rather than
                # left for a reader to eyeball against the printed skill.
                "readable": {
                    "renewable": _readable(skill_pct(ca, ra), floor),
                    "generation": _readable(skill_pct(cb, rb), floor),
                },
            }
        rows.append(row)

    return {
        "cells": rows,
        "cells_only_in_renewable_arm": only_a,
        "cells_only_in_generation_arm": only_b,
        "readability_floor_pct": floor,
    }


def _readable(skill, floor):
    return None if skill is None else bool(skill >= floor)


def summarise(comparison: dict) -> dict:
    rows = comparison["cells"]
    deltas = [r["challenger_wape_pct"]["delta_pp"] for r in rows
              if r["challenger_wape_pct"]["delta_pp"] is not None]
    d7 = [r["references"]["seasonal_naive"]["wape_pct"]["delta_pp"] for r in rows
          if r["references"]["seasonal_naive"]["wape_pct"]["delta_pp"] is not None]
    return {
        "n_cells_compared": len(rows),
        "n_gate_verdict_changed": sum(1 for r in rows if r["gate_changed"]),
        "n_grade_changed": sum(1 for r in rows if r["grade_changed"]),
        "gate_verdicts_changed": [f"{r['country']} {r['horizon_band']}: "
                                  f"{r['gate']['renewable']} -> {r['gate']['generation']}"
                                  for r in rows if r["gate_changed"]],
        "grades_changed": [f"{r['country']} {r['horizon_band']}: "
                           f"{r['grade']['renewable']} -> {r['grade']['generation']}"
                           for r in rows if r["grade_changed"]],
        "challenger_wape_delta_pp": _spread(deltas),
        # The control. A non-zero spread here is replica vintage, not the table:
        # ABL-348 measured the two tables' D-7 bar as identical on all eight.
        "seasonal_naive_delta_pp_is_the_vintage_control": _spread(d7),
        "oracle_reference_moves": [
            f"{r['country']} {r['horizon_band']} vs {ref}: "
            f"{r['references'][ref]['readable']['renewable']} -> "
            f"{r['references'][ref]['readable']['generation']}"
            for r in rows for ref in ("constant_oracle", "climatology_oracle")
            if (r["references"][ref]["readable"]["renewable"]
                != r["references"][ref]["readable"]["generation"])
        ],
    }


def _spread(values):
    if not values:
        return None
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": sum(values) / len(values),
            "max_abs": max(abs(v) for v in values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="Write the comparison JSON here as well as stdout")
    args = parser.parse_args()

    a_record, a_sha = _load(ARM_A)
    b_record, b_sha = _load(ARM_B)
    comparison = compare(a_record, b_record)

    out = {
        "issue": "ABL-426",
        "arms": {
            "renewable": {"scope": ARM_A["scope"], "path": ARM_A["path"], "sha256": a_sha,
                          "training_source": a_record["meta"]["training_source"],
                          "generated_at": a_record["meta"]["generated_at"],
                          "replica_bytes": a_record["meta"]["replica_bytes"],
                          "verdict": a_record["verdict"]},
            "generation": {"scope": ARM_B["scope"], "path": ARM_B["path"], "sha256": b_sha,
                           "training_source": b_record["meta"]["training_source"],
                           "generated_at": b_record["meta"]["generated_at"],
                           "replica_bytes": b_record["meta"]["replica_bytes"],
                           "verdict": b_record["verdict"]},
        },
        # Everything registered that must be equal for the difference to be
        # attributable to the source table, checked rather than asserted in prose.
        "controlled": _controls(a_record["meta"], b_record["meta"]),
        "summary": summarise(comparison),
        **comparison,
    }
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _controls(a_meta: dict, b_meta: dict) -> dict:
    """The registered values that must match, and whether they do.

    Reported as data rather than raised on, because a mismatch is a finding about
    the comparison and the reader needs to see which field moved. The one field
    expected to differ is named so its difference does not read as a failure.
    """
    fields = ("registered_countries", "registered_cells", "gate_basis", "fit_rules",
              "feature_columns", "n_features", "feature_set",
              "feature_set_is_registered_for_scope", "causal_levelling",
              "g23_readability", "seed_readability", "fit_window", "gate_window",
              "registered_intended_n", "reported_comparators")
    out = {"expected_to_differ": {"training_source": [a_meta.get("training_source"),
                                                      b_meta.get("training_source")]},
           "must_match": {}}
    for field in fields:
        a, b = a_meta.get(field), b_meta.get(field)
        out["must_match"][field] = {"equal": a == b, **({} if a == b else {"a": a, "b": b})}
    out["all_controls_hold"] = all(v["equal"] for v in out["must_match"].values())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
