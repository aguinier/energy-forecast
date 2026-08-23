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
replica snapshot than arm A. Both snapshots are read from the records themselves
(`meta.replica_bytes`) and printed under `arms`, rather than named here where they
would go stale. That makes the D-7 delta a direct measurement of the
replica-vintage confound, and it is reported per cell beside the challenger delta
rather than assumed away.

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
    DELTA_MIN, LADDER_REFERENCES, SIGN_TEST, cell_grade, readability_floor_pct,
    skill_pct,
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


#: Fields the two records must agree on for the difference to be attributable to
#: the source table.
CONTROL_FIELDS = ("registered_countries", "registered_cells", "gate_basis", "fit_rules",
                  "feature_columns", "n_features", "feature_set",
                  "feature_set_is_registered_for_scope", "fit_window", "gate_window",
                  "registered_intended_n")

#: The three grading registrations `main()` began recording *after* ABL-405 ran,
#: mapped to the pin `_grade` grades **both** arms under. They are absent from arm
#: A's meta -- not null, absent -- so comparing them field-to-field would report a
#: control failure for a record-schema difference and nothing else. What actually
#: has to hold is that the value arm B records is the value both arms are graded
#: under; that is checked here, by value, instead.
GRADING_REGISTRATIONS_ADDED_AFTER_ARM_A = {
    "causal_levelling": FIT_WINDOW,
    "g23_readability": SIGN_TEST,
    "seed_readability": DELTA_MIN,
}


def _controls(a_meta: dict, b_meta: dict) -> dict:
    """The registered values that must match, and whether they do.

    Reported as data rather than raised on, because a mismatch is a finding about
    the comparison and the reader needs to see which field moved. The one field
    expected to differ is named so its difference does not read as a failure.

    Three fields get a third treatment. `causal_levelling`, `g23_readability` and
    `seed_readability` are **absent from arm A's meta**: ABL-405 ran on
    2026-08-13, before the harness recorded them, and its cells carry `grade:
    null` for the same reason. Putting them in `must_match` would set
    `all_controls_hold` to False on every run of this tool -- a red control on a
    comparison that is in fact controlled, which is worse than no check, because
    the reader cannot tell it from a real one. They are reported separately, and
    the property that replaces equality is the one that makes the letters
    comparable: arm B's recorded value must equal the pin `_grade` passes for
    *both* arms. Where it does, the two arms' grades are taken under the same
    registration whether or not arm A wrote it down.
    """
    out = {"expected_to_differ": {"training_source": [a_meta.get("training_source"),
                                                      b_meta.get("training_source")]},
           "must_match": {}}
    for field in CONTROL_FIELDS:
        a, b = a_meta.get(field), b_meta.get(field)
        out["must_match"][field] = {"equal": a == b, **({} if a == b else {"a": a, "b": b})}
    out["all_controls_hold"] = all(v["equal"] for v in out["must_match"].values())

    reconciled = {}
    for field, pin in GRADING_REGISTRATIONS_ADDED_AFTER_ARM_A.items():
        a_present, b_value = field in a_meta, b_meta.get(field)
        entry = {"absent_in_arm_a": not a_present,
                 "arm_b": b_value,
                 "graded_under": pin,
                 "arm_b_matches_the_pin_both_arms_are_graded_under": b_value == pin}
        # If a later arm A ever does record them, fall back to plain equality --
        # the exemption is for the published record's age, not for the field.
        if a_present:
            entry["arm_a"] = a_meta.get(field)
            entry["equal"] = a_meta.get(field) == b_value
        reconciled[field] = entry
    out["grading_registrations_added_after_arm_a"] = reconciled
    out["grading_registration_reconciled"] = all(
        entry["arm_b_matches_the_pin_both_arms_are_graded_under"]
        and entry.get("equal", True) for entry in reconciled.values())
    out["reported_comparators"] = _reconcile_comparators(a_meta, b_meta)
    return out


def _reconcile_comparators(a_meta: dict, b_meta: dict) -> dict:
    """`reported_comparators` differs by *addition*, and addition is not a confound.

    ABL-437 gave the harness `constant_causal_28d` and `climatology_causal_28d`
    after ABL-405 ran, so arm B reports ten columns where arm A reports eight.
    Plain equality calls that a control failure, which it is not: a **reported**
    comparator is scored on its own intersection and cannot move a cell. Three
    things make that safe, and all three are checked rather than asserted:

      1. arm B is a **superset** -- nothing arm A scored is missing from arm B,
         which is the direction that would really invalidate the comparison;
      2. no added column is in the **gate basis**, so no added column can change
         which rows are scored or what `gate.pass` reads;
      3. no added column is one of the two the **grading levelling** reads.  Both
         arms are graded under `FIT_WINDOW`, whose G2/G3 are `constant_causal`
         and `climatology_causal`; the added pair is what `TRAILING_28D` would
         read, and neither arm registers it.  This is the check that makes the
         levelling pin load-bearing rather than decorative.
    """
    a = list(a_meta.get("reported_comparators") or [])
    b = list(b_meta.get("reported_comparators") or [])
    added = [c for c in b if c not in a]
    dropped = [c for c in a if c not in b]
    basis = set(b_meta.get("gate_basis") or []) | set(a_meta.get("gate_basis") or [])
    ladder = set(LADDER_REFERENCES[FIT_WINDOW].values())
    return {
        "arm_a": a, "arm_b": b,
        "added_after_arm_a": added,
        "dropped_from_arm_a": dropped,
        "arm_b_is_a_superset": not dropped,
        "no_added_column_is_in_the_gate_basis": not (set(added) & basis),
        "no_added_column_is_read_by_the_grading_levelling": not (set(added) & ladder),
        "graded_under": sorted(ladder),
        "reconciled": (not dropped and not (set(added) & basis)
                       and not (set(added) & ladder)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
