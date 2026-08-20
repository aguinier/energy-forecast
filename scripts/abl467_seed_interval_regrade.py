#!/usr/bin/env python3
"""ABL-467: re-grade ABL-427's six k=12 cells under the registered Student-t test.

Arithmetic over records already on disk. **No refit, no new model, no database
read at all** -- every number this needs was computed against the replica by
ABL-419 and ABL-427 and committed with the row set it was scored on, so
re-opening the replica here would recompute numbers that are pinned.

Two records are read and neither is written:

* ``reports/abl_427_tranche2c_seed_reread.json`` -- the per-seed challenger
  WAPEs (the draws), every deterministic comparator's WAPE, and the letters
  ABL-427 published. Its **blob hash is pinned** below and verified before
  anything is computed, so this read cannot silently take a different vintage of
  a file that merged mid-issue.
* ``experiments/ABL348/results_abl419_tranche2c.json`` -- ``slope`` and
  ``correlation`` for G4 only. See ``G4`` below.

What moves and what does not
-----------------------------

**Only G1 moves.** ABL-427's scope registers ``g23_readability: sign_test`` and
its G2/G3 margins are 77-93%, orders outside any width in play, so ABL-444's
floored form is untouched by this read even though the amendment covers it.

**G4 is carried, not re-measured.** It is a sign test on the challenger's own
slope and correlation, has no margin to read against any width, and is not what
this amendment touches. The values are ABL-419's committed seed-42 fit (slope
0.86-0.96, correlation 0.977-0.996 across the six), and ABL-427's own record has
``G4: true`` on all six cells under all three of its candidate floors -- so the
carried value and the re-read value agree, and neither is being invented here.

**The published letter is read, not recomputed.** ABL-427 decided its letters
against its own scope-level floor -- the upper 95% chi-square bound on the
measured CV -- which is *not* this module's ``delta_min`` and cannot be
reproduced by :func:`grade_cell`. Restating those letters from the record is the
only honest ``before`` column; re-deriving them under a rule ABL-427 did not use
would be a second implementation of somebody else's registration.

Order is checkable in git: the registration is committed *before* this file
exists (``experiments/ABL467/config.json``,
``reports/abl_467_seed_interval_readability_registration.md``).

Usage:

    .venv\\Scripts\\python.exe scripts/abl467_seed_interval_regrade.py
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
    DELTA_MIN, SIGN_TEST, STUDENT_T, cell_grade, pair_grade, readability_floor_pct,
)
from src.evaluation.model_free_reference import FIT_WINDOW  # noqa: E402

SOURCE = "reports/abl_427_tranche2c_seed_reread.json"
ABL419 = "experiments/ABL348/results_abl419_tranche2c.json"
JSON_OUT = "reports/abl_467_t2c_regrade.json"
REPORT_OUT = "reports/abl_467_t2c_regrade.md"

#: ABL-427's record as it merged to `main` (PR #80, `dbc37af`). Pinned because
#: that PR was **open** when ABL-467 was assigned and merged while it was in
#: flight: a re-grade that read whatever happened to be on disk could quote a
#: vintage nobody dispositioned. `git hash-object` of the file must equal this.
SOURCE_BLOB = "47e2d9a7fe1073bae84b695c4fbe206490fe6ef3"

STREAM = "solar"
K = 12
#: The scope this read publishes under. New, so it inherits the amendment;
#: `abl427-t2c-reread` stays pinned to `delta_min` and its pack stands.
SCOPE = "abl467-t2c-regrade"

def disposition(label: str) -> str:
    """Collapse ``U(+)`` to ``U`` on a read that **is** the re-read ``(+)`` asks for.

    ABL-418's ``(+)`` is an instruction -- *re-read at k>1 seeds per ABL-385* --
    and at k = 12 it has already been carried out, so emitting it again is
    vacuous. ABL-427 hit exactly this and collapsed it the same way, saying so in
    its pack: "the ladder is a pure function of one cell's scores and cannot know
    the re-read has happened". **The amendment does not fix that**, because
    ``plus`` is decided by G2/G3 clearing readably and has nothing to do with
    which width G1 was read against.

    That this collapse has now been written twice, outside a module whose doctrine
    is that nobody reimplements the ladder, is a defect in the ladder rather than
    a quirk of two scripts -- filed rather than fixed here, because suppressing
    ``plus`` at k > 1 is a change to what a letter means and this registration
    does not cover it.
    """
    return "U" if label == "U(+)" else label


def _normalised(path: Path) -> bytes:
    """The file's bytes with CRLF folded to LF.

    ``core.autocrlf`` is on for this repo, so the working tree is CRLF while the
    stored blob is LF and the two hash differently. A pin on *content* has to be
    invariant to that or it fires on a checkout policy rather than on a change --
    which is exactly what it did on the first run of this script.
    """
    return path.read_bytes().replace(b"\r\n", b"\n")


def blob_hash(path: Path) -> str:
    """`git hash-object` for a file, computed locally so this needs no git.

    Taken over the LF-normalised bytes, so it equals what `git rev-parse
    HEAD:<path>` reports regardless of how the tree was checked out.
    """
    data = _normalised(path)
    header = f"blob {len(data)}".encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def sha256(path: Path) -> str:
    """Over the same normalised bytes, for the same reason."""
    return hashlib.sha256(_normalised(path)).hexdigest()


def load_source() -> dict:
    """Read ABL-427's record, refusing a vintage other than the pinned one."""
    path = ROOT / SOURCE
    if not path.exists():
        raise SystemExit(
            f"{SOURCE} is not in this tree. It is ABL-427's machine record and this "
            "re-grade has no other source for the seed draws; there is nothing to "
            "fall back to and guessing would be worse than stopping.")
    found = blob_hash(path)
    if found != SOURCE_BLOB:
        raise SystemExit(
            f"{SOURCE} has blob hash {found}, not the pinned {SOURCE_BLOB}. That file "
            "merged mid-issue (PR #80). Re-grading a different vintage under this "
            "registration would publish letters nobody dispositioned -- re-pin "
            "deliberately if the change is intended.")
    return json.loads(path.read_text(encoding="utf-8"))


def comparator_wapes(record: dict) -> dict:
    """Every deterministic comparator's WAPE, per cell, from ABL-427's own re-run.

    `recomputed_wape_pct` rather than `abl419_wape_pct`: ABL-427 recomputed all
    42 against its own row set and recorded them identical, so the two agree, but
    the recomputed column is the one belonging to this record.
    """
    wapes: dict = {}
    for row in record["deterministic_reference_reproduction"]:
        key = (row["country"], row["horizon_band"])
        wapes.setdefault(key, {})[row["comparator"]] = row["recomputed_wape_pct"]
    return wapes


def directional(path: Path) -> dict:
    """G4's slope and correlation, from ABL-419's committed cells. Carried, not
    re-measured -- see the module docstring."""
    published = json.loads(path.read_text(encoding="utf-8"))
    return {(cell["country"], cell["horizon_band"]):
            (cell["scores"]["challenger"]["slope"],
             cell["scores"]["challenger"]["correlation"])
            for cell in published["gate_cells"]}


def scores_for(cell: dict, wapes: dict, slope: float, correlation: float) -> dict:
    """The `scores` mapping the ladder grades, assembled from the record.

    The challenger's WAPE is the **k-mean**, which is what ABL-427 printed its
    skill column from and what `grade_cell`'s draw-consistency guard checks the
    per-seed values against.
    """
    scores = {"challenger": {"wape_pct": cell["challenger_wape_pct_k_mean"],
                             "slope": slope, "correlation": correlation}}
    for name, wape in wapes.items():
        scores.setdefault(name, {})["wape_pct"] = wape
    return scores


def regrade(record: dict) -> dict:
    wapes = comparator_wapes(record)
    slopes = directional(ROOT / ABL419)
    cells, amended_grades = [], {}
    for cell in record["cells"]:
        key = (cell["country"], cell["horizon_band"])
        slope, correlation = slopes[key]
        # Graded through `cell_grade`, not `grade_cell`, so ABL-434's coverage
        # gate applies rather than being argued past in a registry exemption. All
        # six cells clear their registered minimum n, so no letter here is held --
        # but a coverage-short cell could not grade `A` through this path, which
        # is a structural guarantee where a written assurance is only a promise.
        # It is also what lets the draws be read off the cell rather than passed.
        graded = {"scores": scores_for(cell, wapes[key], slope, correlation),
                  "gate": {"n": cell["n"], "minimum_n": cell["minimum_n"],
                           "enough_pairs": cell["meets_minimum_n"]},
                  "challenger_wape_pct_per_seed": cell["challenger_wape_pct_per_seed"]}
        amended = cell_grade(graded, STREAM, k=K, levelling=FIT_WINDOW,
                             g23_readability=SIGN_TEST, seed_readability=STUDENT_T)
        # The `delta_min` arm is this module's own floor at k=12, which is NOT
        # what ABL-427 registered. Reported as the third column so a reader can
        # see that the amendment agrees with the unamended ladder on every cell
        # and disagrees only with ABL-427's stricter scope-level choice.
        unamended = cell_grade(graded, STREAM, k=K, levelling=FIT_WINDOW,
                               g23_readability=SIGN_TEST, seed_readability=DELTA_MIN)
        published = cell["grades"]["measured_upper95"]
        amended_grades.setdefault(cell["country"], []).append(amended)
        interval = amended.seed_interval["seasonal_naive"]
        cells.append({
            "country": cell["country"], "horizon_band": cell["horizon_band"],
            "k": K, "n": cell["n"], "minimum_n": cell["minimum_n"],
            "meets_minimum_n": cell["meets_minimum_n"],
            "skill_vs_d7_pct": cell["skill_vs_d7_pct"],
            "published": {
                "letter": published["label"],
                "disposition": cell["disposition"],
                "floor_pct": published["floor_pct"],
                "floor_basis": "ABL-427's upper 95% chi-square bound on the measured CV",
                "conditions": dict(published["conditions"]),
            },
            "amended": amended.as_dict(),
            # Both are recorded: the raw ladder label, and the disposition after
            # `(+)` is collapsed on a read that is itself the re-read it asks for.
            "amended_disposition": disposition(amended.label),
            "unamended_delta_min_at_k12": {
                "letter": disposition(unamended.label), "ladder_label": unamended.label,
                "floor_pct": unamended.floor_pct,
                "conditions": dict(unamended.conditions),
            },
            "seed_interval": interval,
            "moves": cell["disposition"] != disposition(amended.label),
            "g4_carried_from_abl419": {"slope": slopes[key][0],
                                       "correlation": slopes[key][1]},
        })

    pairs = {}
    for country, grades in amended_grades.items():
        overall = pair_grade(grades)
        published_pair = record["pair_grades"][country]["measured_upper95"]
        pairs[country] = {
            "published_letter": published_pair["label"],
            "published_disposition": published_pair["disposition"],
            "amended_letter": disposition(overall.label),
            "amended_ladder_label": overall.label,
            "amended_bands": [disposition(grade.label) for grade in grades],
            "moves": published_pair["disposition"] != disposition(overall.label),
        }

    return {
        "meta": {
            "issue": "ABL-467", "parent": "ABL-316", "scope": SCOPE,
            "regrade_of": record["meta"]["scope"],
            "registration": "experiments/ABL467/config.json",
            "evidence_pack": "reports/abl_467_seed_interval_readability_registration.md",
            "readability_test": STUDENT_T, "k": K, "stream": STREAM,
            "levelling": FIT_WINDOW, "g23_readability": SIGN_TEST,
            "seeds": record["meta"]["seeds"],
            "fit_window": record["meta"]["fit_window"],
            "gate_window": record["meta"]["gate_window"],
            "delta_min_floor_pct_at_k12": readability_floor_pct(STREAM, K),
            "delta_min_floor_pct_at_k1": readability_floor_pct(STREAM, 1),
            "refit": False, "replica_opened": False, "artifact_saved": False,
            "sources": {
                SOURCE: {"blob": blob_hash(ROOT / SOURCE),
                         "sha256": sha256(ROOT / SOURCE),
                         "pinned_blob_matches": blob_hash(ROOT / SOURCE) == SOURCE_BLOB},
                ABL419: {"sha256": sha256(ROOT / ABL419),
                         "used_for": "slope and correlation (G4) only"},
            },
            "what_moves": "G1 only. G2/G3 are sign tests on 77-93% margins under this "
                          "scope's registered form; G4 is a sign test on the "
                          "challenger's own slope and correlation and is carried from "
                          "ABL-419 unchanged.",
        },
        "cells": cells,
        "pair_grades": pairs,
        "prediction_registered_before_this_ran": {
            "source": "ABL-427 section 7.3, and experiments/ABL467/config.json",
            "expected": {"IT_24-36h": "A", "IT_36-48h": "U", "IT_48-64h": "A",
                         "HR_24-36h": "A", "HR_36-48h": "A", "HR_48-64h": "A",
                         "IT_pair": "U", "HR_pair": "A"},
        },
    }


def check_prediction(result: dict) -> list[str]:
    """The registered prediction against what actually came back.

    A re-grade that quietly returned something else would be the failure mode the
    registration exists to catch, so the comparison is in the record rather than
    left to a reader.
    """
    expected = result["prediction_registered_before_this_ran"]["expected"]
    lines = []
    for cell in result["cells"]:
        key = f"{cell['country']}_{cell['horizon_band']}"
        got = cell["amended_disposition"]
        lines.append(f"{key}: expected {expected[key]}, got {got}"
                     f"{'' if got == expected[key] else '   <-- MISMATCH'}")
    for country, pair in result["pair_grades"].items():
        key = f"{country}_pair"
        got = pair["amended_letter"]
        lines.append(f"{key}: expected {expected[key]}, got {got}"
                     f"{'' if got == expected[key] else '   <-- MISMATCH'}")
    return lines


def render(result: dict) -> str:
    meta = result["meta"]
    lines = [
        "# ABL-467 — tranche 2c's k=12 cells re-graded under the Student-t interval",
        "",
        f"Scope `{meta['scope']}`, a re-grade of `{meta['regrade_of']}`. Registration "
        f"`{meta['registration']}`, argued in `{meta['evidence_pack']}` and committed "
        "before this file existed.",
        "",
        "**No refit, no replica read, no artifact, no promotion.** Arithmetic over "
        f"`{SOURCE}` (blob pinned and verified) and ABL-419's committed slope and "
        "correlation. ABL-427's record is not edited or regenerated; this is a new "
        "scope and a new document.",
        "",
        f"**What moves:** {meta['what_moves']}",
        "",
        "## Verdict",
        "",
        "| pair | ABL-427 published (k=12) | **ABL-467 amended** | bands | moves |",
        "|---|:---:|:---:|---|:---:|",
    ]
    for country, pair in result["pair_grades"].items():
        lines.append(f"| **{country}** | `{pair['published_disposition']}` | "
                     f"**`{pair['amended_letter']}`** | "
                     f"{' / '.join(f'`{b}`' for b in pair['amended_bands'])} | "
                     f"{'**yes**' if pair['moves'] else 'no'} |")
    lines += [
        "",
        "## Per cell",
        "",
        "| pair | band | n / min | mean skill vs D-7 | 95% t-CI | t half-width | "
        "ABL-427 floor | published | **amended** | `delta_min` at k=12 |",
        "|---|---|---:|---:|---|---:|---:|:---:|:---:|:---:|",
    ]
    for cell in result["cells"]:
        interval = cell["seed_interval"]
        low, high = interval["ci95_pct"]
        lines.append(
            f"| {cell['country']} | {cell['horizon_band']} | "
            f"{cell['n']:,} / {cell['minimum_n']:,} | "
            f"{cell['skill_vs_d7_pct']:+.2f}% | "
            f"[{low:+.2f}, {high:+.2f}]% | {interval['half_width_pp']:.3f}pp | "
            f"{cell['published']['floor_pct']:.3f}pp | "
            f"`{cell['published']['disposition']}` | "
            f"**`{cell['amended_disposition']}`** | "
            f"`{cell['unamended_delta_min_at_k12']['letter']}` |")
    lines += [
        "",
        f"The `delta_min` column is **this module's own floor at k=12** "
        f"({meta['delta_min_floor_pct_at_k12']:.3f}pp), not the floor ABL-427 "
        "registered. It agrees with the amendment on every cell — the disagreement is "
        "with ABL-427's stricter scope-level choice alone, which is the double-count "
        "the registration argues against.",
        "",
        "## Seeds losing to the baseline outright",
        "",
        "Recorded on every amended cell, because an interval does not show it and it "
        "is the number that should govern any serving conversation (ABL-427 §5).",
        "",
        "| pair | band | fits losing to D-7 | sd of skill |",
        "|---|---|:---:|---:|",
    ]
    for cell in result["cells"]:
        interval = cell["seed_interval"]
        lines.append(f"| {cell['country']} | {cell['horizon_band']} | "
                     f"{interval['draws_losing']} / {interval['n_seeds']} | "
                     f"{interval['sd_skill_pp']:.2f}pp |")
    lines += [
        "",
        "## The prediction registered before this ran",
        "",
        "```",
        *check_prediction(result),
        "```",
        "",
        "## What this does not do",
        "",
        "It promotes nothing. `A` is ABL-418 promotion-**eligibility**, which is "
        "necessary and not sufficient; promotion is a CEO-to-Board decision on an "
        "evidence pack. IT remains `U` and is not close. Every caveat in "
        f"`{meta['evidence_pack']}` §7 travels with these letters — in particular that "
        "the three bands of one country share a fit and are not three independent "
        "estimates, so a pair letter is not a joint 95% statement.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--report-out", default=REPORT_OUT)
    args = parser.parse_args()

    result = regrade(load_source())
    (ROOT / args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.report_out).write_text(render(result), encoding="utf-8")

    print("\n".join(check_prediction(result)))
    moved = [c for c in result["cells"] if c["moves"]]
    print(f"\ncells moving: {len(moved)} "
          f"({', '.join(c['country'] + ' ' + c['horizon_band'] for c in moved) or 'none'})")
    for country, pair in result["pair_grades"].items():
        print(f"{country}: {pair['published_disposition']} -> {pair['amended_letter']}"
              f"{'  (moves)' if pair['moves'] else ''}")
    print(f"\nwrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
