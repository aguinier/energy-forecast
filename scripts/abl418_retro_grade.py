"""Retro-grade an already-dispositioned tranche from its published results file.

Pure arithmetic over stored `results_*.json` files. No refit, no new fit, no
replica read, no write to any dispositioned path. Every source file is opened
read-only and its SHA-256 is recorded in the output, so a later reader can tell
whether the grades below were computed from the bytes that were dispositioned.

The ladder itself lives in `src/evaluation/gate_grading.py` and is the same code
both gate harnesses call -- this script only reads, groups and renders. Adding a
second implementation here is the one thing it must not do. That rule is why
ABL-438 extended this script with a `--tranches` selector rather than writing a
second grader for tranche 1b: the registry below gained a row, the arithmetic
gained nothing.

**A run writes where its issue writes.** The defaults reproduce ABL-418's own
artifacts; any other selection must be given its own `--issue` and output paths,
because regenerating a predecessor's report under a new tranche list would
overwrite evidence that was already dispositioned (ABL-387, and the
`SCOPE_OUTPUTS` incident in CLAUDE.md).

Usage:

    .venv\\Scripts\\python.exe scripts/abl418_retro_grade.py --stdout
    .venv\\Scripts\\python.exe scripts/abl418_retro_grade.py --tranches 1b \\
        --issue ABL-438 --report-out reports/abl_438_retro_grade.md \\
        --json-out reports/abl_438_retro_grade.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.gate_grading import (  # noqa: E402
    CONDITIONS, PUBLISHED_FLOOR_PCT_K1, Z_95, grade_cell, margin_pct_of_own_error,
    pair_grade, readability_floor_pct, skill_pct,
)
from src.evaluation.model_free_reference import (  # noqa: E402
    FIT_WINDOW, comparator_wape,
)


#: A data hold that travels with a grade and is not derivable from the scores.
#: Grade `A` reads *promotion-eligible, subject to any named data hold*, so a
#: hold that lives only in a comment is a hold the next reader does not get.
#: Keyed by tranche, then by pair. Rendered under every table the pair appears
#: in and carried into the JSON, so an importer sees it too.
HOLDS = {
    "1b": {"BG": {"issue": "ABL-396", "kind": "night contamination, upstream",
                  "statement":
                      "76-85% of geometric-night hours carry above 1 MW, up to 1,097 MW, "
                      "~5-6% of all energy booked after dark. Identical on both actuals "
                      "tables, so it is upstream of this module. ABL-396's screen found BG "
                      "an outlier by 3.7x and the only country whose displacement band is "
                      "wide enough to threaten a verdict -- and that band is far wider than "
                      "BG's margin over the oracle climatology.",
                  "effect": "Grade A must not be reported for BG solar without this line."}},
}

#: The tranches this script can retro-grade, with the stream whose registered CV
#: sets their floor and the pair key their cells are grouped on. Every file is
#: named in the issue that asked for it; nothing is discovered. `2a`/`2b` are
#: ABL-418's; `1b` is ABL-438's, whose committed record already carried the full
#: eight-comparator reference suite -- ABL-418 simply never ran over it.
TRANCHES = {
    "1b": {"tranche": "1b", "stream": "solar", "scope": "abl316-t1b", "issue": "ABL-438",
           "results": "experiments/ABL348/results_abl381_tranche1b.json",
           "pack": "reports/abl_381_tranche1b_findings.md",
           "key": lambda cell: cell["country"]},
    "2a": {"tranche": "2a", "stream": "solar", "scope": "abl316-t2a", "issue": "ABL-418",
           "results": "experiments/ABL348/results_abl405_tranche2a.json",
           "pack": "reports/abl_405_tranche2a_findings.md",
           "key": lambda cell: cell["country"]},
    "2b": {"tranche": "2b", "stream": "wind", "scope": "abl406-tranche2b", "issue": "ABL-418",
           "results": "experiments/ABL348/results_abl406_tranche2b.json",
           "pack": "reports/abl_406_evidence_pack.md",
           "key": lambda cell: f"{cell['country']} {cell['forecast_type']}"},
}

#: The tranche list ABL-418 published, and the default. Changing it would change
#: `reports/abl_418_retro_grade.*` on the next regeneration, which is the thing
#: `--tranches` exists to avoid.
ABL418_TRANCHES = ("2a", "2b")

#: The CEO's own reading, from the issue description that asked for each tranche,
#: so the report can state agreement or name the cell that disagrees rather than
#: quietly adopting it. "check it rather than adopt it, and if your arithmetic
#: disagrees on a cell, your arithmetic wins and I want the cell named."
DESCRIPTION_READING = {
    "1b": {"BG": "A", "CH": "A"},
    "2a": {"BG": "A", "CH": "A", "CZ": "A", "HU": "U(+)", "PL": "A", "RO": "A",
           "SI": "A", "SK": "A"},
    "2b": {"ES wind_onshore": "C", "FI wind_onshore": "A", "GR wind_onshore": "A",
           "IT wind_onshore": "U", "NO wind_onshore": "B", "PL wind_onshore": "A",
           "PT wind_onshore": "C", "SE wind_onshore": "A"},
}

#: The references that stay reported and never gate, and what losing to one
#: means. Kept beside the grades because a U(+) that loses to both oracles is a
#: different object from one that beats them.
ORACLES = ("constant_oracle", "climatology_oracle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value, digits: int = 2) -> str:
    return "Not measured" if value is None else f"{value:+.{digits}f}%"


def read_tranche(root: Path, spec: dict) -> dict:
    """Grade one stored results file. Read-only; nothing is written back."""
    path = root / spec["results"]
    results = json.loads(path.read_text(encoding="utf-8"))
    stream, floor = spec["stream"], readability_floor_pct(spec["stream"])
    holds = HOLDS.get(spec["tranche"], {})
    order, pairs, graded = [], {}, []
    for cell in results["gate_cells"]:
        # ABL-437 pins this read to `fit_window`, the levelling ABL-418 was
        # published on. Tranches 2a and 2b carry no trailing reference column
        # at all -- they were fitted before ABL-437 existed -- so grading them
        # on the amended default would silently turn every G2 and G3 into "not
        # measured" and re-write a published report as a page of Bs. The
        # amended read of these same tranches is a separate record
        # (`reports/abl_437_causal_levelling_reread.md`), which is what a
        # re-grade is: a new document, not an edit to the old one.
        grade = grade_cell(cell["scores"], stream, levelling=FIT_WINDOW)
        label = spec["key"](cell)
        if label not in pairs:
            order.append(label)
            pairs[label] = []
        pairs[label].append(grade)
        challenger = cell["scores"]["challenger"]["wape_pct"]
        oracle_skill = {name: skill_pct(challenger, comparator_wape(cell["scores"], name))
                        for name in ORACLES}
        graded.append({"pair": label, "band": cell["horizon_band"],
                       "n": cell["gate"]["n"], "gate_pass": cell["gate"]["pass"],
                       # The ladder cannot see minimum n: it grades a margin, so a
                       # coverage-short cell that beats D-7 still grades A. Carried
                       # beside every grade rather than left nested in `gate`,
                       # where a flat lookup passes vacuously.
                       "minimum_n": cell["gate"].get("minimum_n"),
                       "enough_pairs": cell["gate"].get("enough_pairs"),
                       "challenger_wape_pct": challenger,
                       "oracle_skill_pct": oracle_skill,
                       # An oracle gates nothing, but a win *inside the floor* is
                       # not a readable win, and reporting it as one is the error
                       # the floor exists to prevent. Derived, never asserted.
                       "oracle_margin_readable": {
                           name: None if value is None else bool(abs(value) > floor)
                           for name, value in oracle_skill.items()},
                       "hold": holds.get(label),
                       **grade.as_dict()})
    return {"tranche": spec["tranche"], "scope": spec["scope"], "stream": stream,
            "issue": spec["issue"],
            "results_path": spec["results"], "results_sha256": _sha256(path),
            "evidence_pack": spec["pack"],
            "published_verdict": results["verdict"],
            "gate_window": results["meta"]["gate_window"],
            "training_source": results["meta"]["training_source"],
            "reported_comparators": results["meta"].get("reported_comparators"),
            "floor_pct": floor, "published_floor_pct": PUBLISHED_FLOOR_PCT_K1[stream],
            "holds": holds,
            "pair_order": order,
            "pair_grades": {label: pair_grade(pairs[label]).as_dict() for label in order},
            "cells": graded}


def sensitivity(tranche: dict, root: Path) -> list[dict]:
    """Cells whose grade would move if the floor were read on the other denominator.

    ABL-418 registers the test on the printed ``skill vs D-7`` column; ABL-406
    quoted its margins on the challenger's own error, which is the denominator
    ABL-385's CV is measured in. The two always agree in sign, so they can only
    disagree for a cell within the floor of it. This reports which -- if the list
    is empty, the choice of denominator did not decide anything.

    The 2-dp floors published in prose (10.64% / 7.51%) are checked the same way,
    against the exact 1.96 * c value the ladder uses.
    """
    results = json.loads((root / tranche["results_path"]).read_text(encoding="utf-8"))
    floor, published = tranche["floor_pct"], tranche["published_floor_pct"]
    moved = []
    for cell, graded in zip(results["gate_cells"], tranche["cells"]):
        challenger = cell["scores"]["challenger"]["wape_pct"]
        bar = cell["scores"]["seasonal_naive"]["wape_pct"]
        skill, own = skill_pct(challenger, bar), margin_pct_of_own_error(challenger, bar)
        if skill is None or own is None:
            continue
        under_own = ("U" if abs(own) <= floor else "A/B" if own > floor else "C")
        under_skill = ("U" if abs(skill) <= floor else "A/B" if skill > floor else "C")
        rounding = min(abs(skill), abs(own)) < max(floor, published) and \
            min(abs(skill), abs(own)) > min(floor, published)
        if under_own != under_skill or rounding:
            moved.append({"pair": graded["pair"], "band": graded["band"],
                          "skill_pct": skill, "own_error_margin_pct": own,
                          "under_skill": under_skill, "under_own_error": under_own,
                          "inside_floor_rounding_gap": rounding})
    return moved


def _boundary_tail(tranches: list[dict]) -> str:
    """The closing clause of the boundary section.

    ABL-418's own selection keeps ABL-418's published wording **verbatim**, so
    that regenerating `reports/abl_418_retro_grade.md` reproduces the document
    that was dispositioned rather than a re-worded one. Any other selection gets
    the general statement, which is the one that is true of it -- the 2a clause
    names a tranche that is not necessarily in the report.
    """
    if [tranche["tranche"] for tranche in tranches] == list(ABL418_TRANCHES):
        return ("for tranche 2a that hold is live and named in its own published disposition, "
                "which this document does not touch.")
    return ("holds registered against a pair below are printed under that pair's table and carried in "
            "the JSON, and none of them is derivable from the scores. A hold named only in a tranche's "
            "own published disposition stays there; this document does not touch it.")


def _subject(count: int) -> str:
    """"This tranche fits" / "Both tranches fit" / "All tranches fit"."""
    if count == 1:
        return "This tranche fits"
    return f"{'Both' if count == 2 else 'All'} tranches fit"


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` -- so a one-tranche run does not read
    like a two-tranche run with a word missing."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def render(tranches: list[dict], root: Path, issue: str = "ABL-418") -> str:
    plural = "s" if len(tranches) > 1 else ""
    named = _join([tranche["tranche"] for tranche in tranches])
    listed = _join([f"`{tranche['results_path']}`" for tranche in tranches])
    selection = "" if [tranche["tranche"] for tranche in tranches] == list(ABL418_TRANCHES) \
        else f" --tranches {','.join(tranche['tranche'] for tranche in tranches)} --issue {issue}"
    lines = [
        f"# {issue} — graded gate disposition (G1–G4), and the retro-grade of tranche{plural} {named}",
        "",
        "**Generated from the stored results files, not restated in prose.** Every grade below is produced by "
        "`src/evaluation/gate_grading.py` — the same code both gate harnesses now call — reading "
        f"{listed}. No refit, no new fit, no replica read, no write to any "
        f"dispositioned path. Regenerate with `.venv\\Scripts\\python.exe scripts/abl418_retro_grade.py{selection}`.",
        "",
        "## What is registered, and what is not",
        "",
        "**The bar is not re-opened.** Seasonal-naive D-7 stays the registered gate for every scope already "
        "dispositioned and every scope still to come. ABL-348's frozen windows, bands, metric, minimum n and source are "
        "unchanged. A cell that clears D-7 still reads PASS, and no verdict in any published report moves because of "
        "this document.",
        "",
        "**What changes is what a PASS entitles a cell to.** ABL-406 established across eight `wind_onshore` pairs that "
        "the gate outcome was *fully* predicted by whether a causal constant clears the registered bar on its own — five "
        "weak bars gave five passes, three strong bars gave three failures or ties, no exceptions — and that NO passed "
        "3/3 while anti-correlated with its own target (slope −0.08, corr −0.14). A PASS is necessary and not "
        "sufficient for a promotion recommendation. Tightening the bar after the fact would be shopping the "
        "registration; grading the pass is not, which is why the ladder was pre-registered on ABL-418 before any "
        "remaining tranche is fitted.",
        "",
        "| condition | test | source column |",
        "|---|---|---|",
    ]
    sources = {"G1": "`skill vs D-7`, against the floor below", "G2": "`constant causal WAPE`, already printed",
               "G3": "`climatology causal WAPE`, already printed", "G4": "`slope` and `corr`, already printed"}
    for name, role, question in CONDITIONS:
        lines.append(f"| **{name}** {role} | {question} | {sources[name]} |")
    lines.extend([
        "",
        "**A** — G1–G4 hold in every band. Promotion-eligible, subject to any named data hold. "
        "**B** — G1 holds, one or more of G2/G3/G4 fails; the failures are named. Not promotion-eligible. "
        "**C** — G1 fails readably. "
        "**U** — the G1 margin sits inside the readability floor, so the cell is unreadable at one seed; **U(+)** where "
        "G2–G4 clear readably, in which case the disposition is *re-read at k>1 seeds* per ABL-385, not *reject*.",
        "",
        "`U` takes precedence over `C`: both are \"G1 does not hold\", but a measured loss and an absence of measurement "
        "are different statements, and calling an unreadable cell a failure invites the wrong next move. A pair takes "
        "the worst grade of its bands (`C` > `B` > `U` > `A`), because grade A requires all four conditions in *every* "
        "band; `U(+)` survives to the pair only if every unreadable band in it is `U(+)`.",
        "",
        "**Causal references only.** The two oracle references stay reported and never gate — an oracle is not causally "
        "available, so losing to one bounds what a verdict means rather than voiding it. Both are reported below beside "
        "every grade, as is the bar-weakness flag (does `constant_causal` clear the registered D-7 bar on its own?).",
        "",
        "## The readability floor",
        "",
        "ABL-385 registers `delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)` as the minimum readable relative gap. "
        "Every reference on this ladder is **deterministic** — D-7, a flat line and an hour-of-day climatology do not "
        "move when the challenger is refitted — so `c_B = 0`, and the published two-arm margin is a factor of √2 too "
        f"wide. {_subject(len(tranches))} once per cell, so k = 1.",
        "",
        "| stream | fleet p90 per-fit CV (ABL-385 §1) | two-arm δ_min at k=1 | **floor used** = δ_min/√2 | published in prose |",
        "|---|---:|---:|---:|---:|",
    ]
    )
    for stream in dict.fromkeys(tranche["stream"] for tranche in tranches):
        tranche = next(item for item in tranches if item["stream"] == stream)
        floor = tranche["floor_pct"]
        # The CV is rendered back out of the floor with the ladder's own z, not a
        # retyped 1.96: two copies of a constant is how the floor drifts.
        lines.append(f"| {stream} | {floor / Z_95:.4f}% | {floor * 2 ** 0.5:.4f}% | "
                     f"**{floor:.4f}%** | {tranche['published_floor_pct']:.2f}% |")
    lines.extend([
        "",
        "The prose values are 2-dp renderings and are not what the ladder uses; the exact `1.96 · c` value is. The gap "
        f"between them is under 0.01pp and no cell of {'either' if len(tranches) == 2 else 'any'} tranche sits inside "
        f"it — checked per cell in §{len(tranches) + 1}.",
        "",
    ])

    for number, tranche in enumerate(tranches, start=1):
        lines.extend(_render_tranche(tranche, number))

    lines.extend([f"## {len(tranches) + 1}. Sensitivity: which denominator, and the 2-dp rounding", "",
                  "ABL-418 registers G1 on the printed `skill vs D-7` column, `100 · (1 − challenger/reference)`. "
                  "ABL-406 quoted its margins on the challenger's **own** error, `100 · (reference − challenger)/challenger`, "
                  "which is the denominator ABL-385's CV is measured in. The two always agree in sign and differ only in "
                  "magnitude, so they can disagree about a grade only for a cell sitting near the floor. Both are computed "
                  "for every cell and both are in the JSON.", ""])
    for tranche in tranches:
        moved = sensitivity(tranche, root)
        if not moved:
            lines.append(f"- **Tranche {tranche['tranche']} ({tranche['stream']}): no cell of "
                         f"{len(tranche['cells'])} changes grade** under either denominator, and none sits between the "
                         f"exact floor ({tranche['floor_pct']:.4f}%) and the 2-dp value published in prose "
                         f"({tranche['published_floor_pct']:.2f}%). The choice of denominator decided nothing here.")
            continue
        lines.append(f"- **Tranche {tranche['tranche']}: {len(moved)} cell(s) are denominator-sensitive** and are named "
                     "rather than resolved silently:")
        lines.extend(f"  - {row['pair']} {row['band']}: skill {row['skill_pct']:+.2f}% → `{row['under_skill']}`, "
                     f"own-error margin {row['own_error_margin_pct']:+.2f}% → `{row['under_own_error']}`"
                     for row in moved)
    lines.extend([
        "",
        f"## {len(tranches) + 2}. Boundary",
        "",
        "No promotion, no serving-registry change, no ingest change, no refit, no replica write, no sidecar write. "
        "The grades land here, under a new path; `abl253`, `abl376`, `abl316-t1a`, `abl316-t1b`, `abl316-t2a` and "
        "`abl406-tranche2b` results files and reports are byte-unchanged, verified by blob hash against the merge base "
        f"and recorded on {issue}.",
        "",
        "A grade is not a promotion recommendation and does not become one. Grade **A** means *promotion-eligible*, "
        f"subject to any named data hold — {_boundary_tail(tranches)}",
        "",
    ])
    return "\n".join(lines)


def _why_it_differs(label: str, expected: str, actual: str, cells: list[dict]) -> str:
    """Why the ladder lands somewhere the ABL-418 description did not.

    Derived from the cells, never asserted: the description invites exactly this
    ("if your arithmetic disagrees on a cell, your arithmetic wins and I want the
    cell named"), so a hand-written explanation for one case would be the wrong
    shape the first time a second case appeared.
    """
    floor = cells[0]["floor_pct"]
    bands = " / ".join(f"{cell['band']} {cell['label']}" for cell in cells)
    gaps = " / ".join(_pct(cell["skill_pct"]["seasonal_naive"], 2) for cell in cells)
    level = " / ".join(_pct(cell["skill_pct"]["constant_causal"], 1) for cell in cells)
    shape = " / ".join(_pct(cell["skill_pct"]["climatology_causal"], 1) for cell in cells)
    directional = all(cell["conditions"].get("G4") for cell in cells)
    detail = (f"Per band: {bands}. Skill vs D-7 {gaps} against a {floor:.2f}% floor; "
              f"vs `constant_causal` {level}; vs `climatology_causal` {shape}; "
              f"slope and correlation positive in {'all' if directional else 'not all'} bands.")
    if actual == "U(+)" and expected == "U":
        return (f"Its G1 margin is inside the floor in every band, so it is `U` either way — the `(+)` is what differs, "
                f"and it follows from the ladder's own text, *\"if G2–G4 clear readably\"*. {detail} All three "
                f"conditions clear, and G2/G3 clear by more than the floor, so the disposition is **re-read at k>1 "
                f"seeds**, not **report and do not decide**. It is the weaker of the two `U` readings to act on, and "
                f"the qualifier belongs with it: {label} loses readably to **both** oracle references, which gate "
                f"nothing but bound what a re-read could establish.")
    return detail


def _coverage_note(tranche: dict) -> str:
    """What the ladder cannot see: how much coverage the cell actually had.

    A grade is a reading of a *margin*, so a cell that scraped past its minimum n
    and beat D-7 grades exactly as well as one with full coverage. ABL-348's
    `enough_pairs` is the check that does see it, and it belongs beside the grade
    rather than nested in `gate` where a flat lookup passes vacuously.
    """
    cells = tranche["cells"]
    short = [cell for cell in cells if cell["enough_pairs"] is False]
    missing = [cell for cell in cells if cell["enough_pairs"] is None]
    if missing:
        return (f"`enough_pairs` is absent from {len(missing)} of {len(cells)} cells in this record, so ABL-348's "
                f"coverage check cannot be reported here — treat the grades as ungated on coverage.")
    covered = [cell for cell in cells if cell["minimum_n"]]
    tightest = min(covered, key=lambda cell: cell["n"] / cell["minimum_n"], default=None)
    lead = (f"All {len(cells)} cells clear ABL-348's minimum n" if not short
            else f"**{len(short)} of {len(cells)} cells are short of ABL-348's minimum n** "
                 f"({', '.join(f'{cell['pair']} {cell['band']}' for cell in short)})")
    if tightest is None:
        return f"{lead}."
    return (f"{lead}; the tightest is {tightest['pair']} {tightest['band']} at n = {tightest['n']:,} against a "
            f"minimum of {tightest['minimum_n']:,} (×{tightest['n'] / tightest['minimum_n']:.2f}). The ladder does "
            f"not read this column — it grades a margin — so it is reported beside the grades, not folded into them.")


def _render_tranche(tranche: dict, number: int) -> list[str]:
    reading = DESCRIPTION_READING[tranche["tranche"]]
    lines = [
        f"## {number}. Tranche {tranche['tranche']} — `{tranche['scope']}` ({tranche['stream']})",
        "",
        f"Source: `{tranche['results_path']}`, SHA-256 `{tranche['results_sha256']}`. "
        f"Evidence pack: `{tranche['evidence_pack']}`. "
        f"Published disposition, restated unchanged: **{tranche['published_verdict']}**. "
        f"Gate window {tranche['gate_window']['start']} → {tranche['gate_window']['end_exclusive']} (exclusive), "
        f"target series `{tranche['training_source']}`. Floor {tranche['floor_pct']:.4f}% at k=1.",
        "",
        "| pair | band | n | n ≥ min | gate | skill vs D-7 | vs constant causal | vs climatology causal | "
        "slope>0 & corr>0 | grade |",
        "|---|---|---:|:---:|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for cell in tranche["cells"]:
        conditions = cell["conditions"]
        failed = ", ".join(item["condition"] for item in cell["failed"])
        label = cell["label"] if not failed else f"{cell['label']} — fails {failed}"
        enough = ("yes" if cell["enough_pairs"] else "**no**"
                  if cell["enough_pairs"] is False else "—")
        lines.append(
            f"| {cell['pair']} | {cell['band']} | {cell['n']:,} | {enough} | "
            f"{'PASS' if cell['gate_pass'] else 'FAIL'} | "
            f"{_pct(cell['skill_pct']['seasonal_naive'])} | {_pct(cell['skill_pct']['constant_causal'])} | "
            f"{_pct(cell['skill_pct']['climatology_causal'])} | "
            f"{'yes' if conditions.get('G4') else 'no'} | **{label}** |")
    lines.extend(["", _coverage_note(tranche), ""])

    lines.extend([f"| pair | bands | grade | failed conditions | bar weaker than a flat line? | "
                  f"beats constant oracle? | beats climatology oracle? | {tranche['issue']} description | agrees? |",
                  "|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|"])
    disagreements, unreadable_wins = [], []
    for label in tranche["pair_order"]:
        grade = tranche["pair_grades"][label]
        cells = [cell for cell in tranche["cells"] if cell["pair"] == label]
        bands = " / ".join(cell["label"] for cell in cells)
        reasons = ", ".join(f"{item['condition']}" for item in grade["failed"]) or "—"
        weak = any(cell["bar_weaker_than_a_flat_line"] for cell in cells)
        oracles = []
        for name in ORACLES:
            values = [cell["oracle_skill_pct"][name] for cell in cells]
            verdict = ("Not measured" if all(v is None for v in values)
                       else "yes" if all(v is not None and v > 0 for v in values)
                       else "no" if all(v is not None and v <= 0 for v in values) else "mixed")
            # A win an oracle cannot be *read* to have lost is not a win the
            # floor lets anyone rank on. Same floor, same k, applied to the
            # column that gates nothing -- because that is the column a reader
            # will otherwise treat as the strong evidence.
            if verdict == "yes" and not all(cell["oracle_margin_readable"][name] for cell in cells):
                worst = min(values)
                verdict = f"yes, inside the floor ({worst:+.2f}%)"
                unreadable_wins.append((label, name, worst))
            oracles.append(verdict)
        expected = reading.get(label, "—")
        agrees = grade["label"] == expected
        if not agrees:
            disagreements.append((label, expected, grade["label"], cells))
        lines.append(f"| {label} | {bands} | **{grade['label']}** | {reasons} | {'yes' if weak else 'no'} | "
                     f"{oracles[0]} | {oracles[1]} | {expected} | {'yes' if agrees else '**no**'} |")

    counts = {}
    for label in tranche["pair_order"]:
        counts[tranche["pair_grades"][label]["label"]] = \
            counts.get(tranche["pair_grades"][label]["label"], 0) + 1
    lines.extend(["", "Pair grades: " + ", ".join(f"**{grade}** × {count}" for grade, count in sorted(counts.items())) + ".", ""])

    if unreadable_wins:
        lines.extend([
            f"**Beating an oracle inside the floor is not beating it readably.** The floor is "
            f"{tranche['floor_pct']:.4f}% at k=1 and it applies to any margin a reader ranks on, not only to the one "
            f"G1 gates on. These wins are positive in every band and none of them is readable at one seed:",
            "",
        ])
        lines.extend(f"- **{label}** vs `{name}`: {worst:+.2f}% at its worst band, against a "
                     f"{tranche['floor_pct']:.2f}% floor."
                     for label, name, worst in unreadable_wins)
        lines.append("")

    for label in tranche["pair_order"]:
        hold = tranche["holds"].get(label)
        if not hold:
            continue
        lines.extend([
            f"**Live data hold on {label} — {hold['issue']} ({hold['kind']}).** {hold['statement']} "
            f"{hold['effect']}",
            "",
        ])

    if not disagreements:
        lines.extend([f"Every pair reproduces the reading in the {tranche['issue']} description.", ""])
    else:
        lines.append(f"**Disagreements with the {tranche['issue']} description — the arithmetic wins, and the cells "
                     f"are named:**")
        lines.append("")
        lines.extend(f"- **{label}: the description reads `{expected}`, the ladder gives `{actual}`.** "
                     + _why_it_differs(label, expected, actual, cells)
                     for label, expected, actual, cells in disagreements)
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(Path(__file__).parent.parent),
                        help="Repository root the results files are read from.")
    parser.add_argument("--tranches", default=",".join(ABL418_TRANCHES),
                        help=f"Comma-separated tranche keys to grade, in order. "
                             f"One of {', '.join(sorted(TRANCHES))}. "
                             f"Default {','.join(ABL418_TRANCHES)} (ABL-418's own selection).")
    parser.add_argument("--issue", default="ABL-418",
                        help="Issue this run is published under; titles the report.")
    parser.add_argument("--report-out", default="reports/abl_418_retro_grade.md")
    parser.add_argument("--json-out", default="reports/abl_418_retro_grade.json")
    parser.add_argument("--stdout", action="store_true", help="Also print the report.")
    args = parser.parse_args()

    selected = [key.strip() for key in args.tranches.split(",") if key.strip()]
    unknown = [key for key in selected if key not in TRANCHES]
    if unknown:
        parser.error(f"unknown tranche(s) {', '.join(unknown)}; "
                     f"known: {', '.join(sorted(TRANCHES))}")
    # A non-default selection writing to the default paths would overwrite a
    # dispositioned report with a different tranche list under its own heading --
    # the SCOPE_OUTPUTS failure in CLAUDE.md, one directory over. Refused, not
    # warned about.
    if selected != list(ABL418_TRANCHES):
        for name, value in (("--report-out", args.report_out), ("--json-out", args.json_out)):
            if "abl_418_retro_grade" in value:
                parser.error(f"selection {','.join(selected)} may not write ABL-418's {name} ({value}); "
                             f"give this run its own output path")

    root = Path(args.repo_root)
    tranches = [read_tranche(root, TRANCHES[key]) for key in selected]
    report = render(tranches, root, args.issue)

    record = {"issue": args.issue,
              "tranche_selection": selected,
              "ladder": [{"condition": name, "role": role, "test": question}
                         for name, role, question in CONDITIONS],
              "tranches": [{**tranche,
                            "denominator_sensitive_cells": sensitivity(tranche, root)}
                           for tranche in tranches]}
    for path, text in ((root / args.report_out, report),
                       (root / args.json_out, json.dumps(record, indent=2) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    # The report body carries non-ASCII on purpose (the repo convention: help
    # text is ASCII, report bodies are not), so stdout is re-encoded here rather
    # than left to the console codepage -- ABL-364.
    if args.stdout:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)
    print(f"Wrote {args.report_out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
