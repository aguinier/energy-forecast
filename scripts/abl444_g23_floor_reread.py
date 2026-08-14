#!/usr/bin/env python3
"""ABL-444: re-read every graded ABL-316 cell with the G2/G3 readability floor on.

Arithmetic over records already on disk. **No refit, no new model, no database
read at all** -- the two trailing references this needs were already computed
against the replica by ABL-437 and committed with the row set they were scored
on, so re-opening the replica here would recompute numbers that are pinned.

Three records are read and none is written:

* each tranche's committed ``results_*.json``, for the challenger's own scores,
  slope, correlation, ``enough_pairs`` and ``minimum_n``;
* ``reports/abl_437_causal_levelling_reread.json``, for the trailing-28d
  reference WAPEs on the same cells;
* ``reports/abl_443_offshore_trailing_reread.json``, the same thing one scope
  over for DE/NL ``wind_offshore``, which merged to main while this issue was
  open.

All are SHA-256'd into the output, so a later reader can tell whether these
letters were derived from the bytes that were dispositioned.

Four arms are graded per cell, because **both registered axes stay live**:
``{fit_window, trailing_28d}`` x ``{sign_test, floored}``. The published letters
are the ``sign_test`` column of each levelling and are reproduced, not restated.

Order is checkable in git: the registration is committed *before* this file
exists (``experiments/ABL444/config.json``,
``reports/abl_444_g23_readability_floor_registration.md``).

Usage:

    .venv\\Scripts\\python.exe scripts/abl444_g23_floor_reread.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.gate_grading import (  # noqa: E402
    FLOORED, GRADE_SEVERITY, SIGN_TEST, grade_cell, margin_pct_of_own_error,
    pair_grade, readability_floor_pct,
)
from src.evaluation.model_free_reference import FIT_WINDOW, TRAILING_28D  # noqa: E402

ROOT = Path(__file__).parent.parent

#: ABL-437's re-read, which is this read's source for the trailing references.
SOURCE_REREAD = "reports/abl_437_causal_levelling_reread.json"

#: ABL-443's, the same shape one scope over. It landed on main while this issue
#: was open and it is *not* an ABL-316 tranche -- it is a re-read of the
#: `abl322-pilot` offshore pair set under a script-owned scope -- so it has its
#: own source record and its own reader here rather than a row in `TRANCHES`.
#:
#: It belongs in this document rather than being left out: its own margins table
#: already labels all six DE margins "not readable at one seed" and it records
#: `g2_g3_floor_is_a_ladder_condition: false`, which is the hook this issue
#: closes. Leaving it out would publish a floored read of the programme that
#: skipped the one pair its own author flagged.
OFFSHORE_REREAD = "reports/abl_443_offshore_trailing_reread.json"

#: Where this read writes. Hardcoded and refused if it points anywhere ABL-437
#: or ABL-418 owns -- the `SCOPE_OUTPUTS` failure one directory over, where a run
#: that kept a default path rewrote a dispositioned record and exited 0.
JSON_OUT = "reports/abl_444_g23_floor_reread.json"
REPORT_OUT = "reports/abl_444_g23_floor_reread.md"
PROTECTED = ("abl_437_", "abl_438_", "abl_418_")

#: Which causal pair each levelling's G2/G3 read. Imported shape, not a second
#: table: `gate_grading.LADDER_REFERENCES` is the registration and this only
#: names the two arms this document reports.
LEVELLINGS = (FIT_WINDOW, TRAILING_28D)
READABILITY = (SIGN_TEST, FLOORED)

#: Data holds that travel with a grade and that the ladder cannot see. A grade of
#: A -- or an N that would otherwise read as "just re-run it at k>1" -- must not
#: be reported for these pairs without the hold attached.
HOLDS = {
    ("solar", "BG"): "ABL-396 night contamination: BG books 152-246 MW in 76-85% of "
                     "its night hours, and 25.3% of its scored gate rows are night rows. "
                     "The displacement band is far wider than any margin in this document.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_key(cell: dict, default_type: str) -> tuple:
    """(forecast_type, country, band). Solar records omit `forecast_type`."""
    return (cell.get("forecast_type", default_type), cell["country"], cell["horizon_band"])


def _scores_for(committed: dict, trailing: dict) -> dict:
    """The committed cell's scores, plus ABL-437's trailing references.

    The challenger, D-7 and the four ABL-389 references come from the record the
    tranche was dispositioned on; only the two trailing columns are added, and
    they are copied from ABL-437's re-read rather than recomputed, so this
    document cannot disagree with that one about a number.
    """
    scores = dict(committed["scores"])
    for name in ("constant_causal_28d", "climatology_causal_28d"):
        value = trailing["wape"].get(name)
        scores[name] = {"wape_pct": value}
    return scores


def read(root: Path) -> dict:
    source = root / SOURCE_REREAD
    reread = json.loads(source.read_text(encoding="utf-8"))
    tranches = []
    for entry in reread["tranches"]:
        record_path = root / entry["record"]
        committed = json.loads(record_path.read_text(encoding="utf-8"))
        stream = entry["stream"]
        default_type = "solar" if stream == "solar" else None
        by_key = {_cell_key(cell, default_type): cell
                  for cell in committed["gate_cells"]}
        floor = readability_floor_pct(stream)
        pairs = []
        for pair in entry["pairs"]:
            country, forecast_type = pair["country"], pair["forecast_type"]
            cells = []
            for cell in pair["cells"]:
                key = (forecast_type, country, cell["band"])
                if key not in by_key:
                    raise KeyError(f"{entry['tranche']} {key} is in ABL-437's re-read "
                                   f"but not in {entry['record']}")
                committed_cell = by_key[key]
                scores = _scores_for(committed_cell, cell)
                grades = {}
                for levelling in LEVELLINGS:
                    for readability in READABILITY:
                        grades[f"{levelling}/{readability}"] = grade_cell(
                            scores, stream, levelling=levelling,
                            g23_readability=readability)
                challenger = scores["challenger"]["wape_pct"]
                cells.append({
                    "band": cell["band"],
                    "n": committed_cell["gate"]["n"],
                    "minimum_n": committed_cell["gate"]["minimum_n"],
                    # ABL-434's column. Reported beside every grade because the
                    # ladder cannot see it; deliberately NOT folded into it.
                    "enough_pairs": committed_cell["gate"]["enough_pairs"],
                    "gate_pass": committed_cell["gate"]["pass"],
                    "floor_pct": floor,
                    "wape": cell["wape"],
                    "grades": {arm: grade.as_dict() for arm, grade in grades.items()},
                    "labels": {arm: grade.label for arm, grade in grades.items()},
                    # Both denominators on every ladder reference, so §4 of the
                    # registration pack can be checked against this file.
                    "own_error_margin_pct": {
                        name: margin_pct_of_own_error(challenger, value)
                        for name, value in cell["wape"].items() if name != "challenger"},
                })
            rollup = {}
            for arm in cells[0]["grades"] if cells else {}:
                rollup[arm] = pair_grade(
                    [grade_cell(_scores_for(by_key[(forecast_type, country, cell["band"])],
                                            {"wape": cell["wape"]}),
                                stream,
                                levelling=arm.split("/")[0],
                                g23_readability=arm.split("/")[1])
                     for cell in pair["cells"]]).label
            pairs.append({
                "pair": pair["pair"], "country": country,
                "forecast_type": forecast_type,
                "pair_grades": rollup,
                "hold": HOLDS.get((forecast_type, country)),
                "cells": cells,
            })
        tranches.append({
            "tranche": entry["tranche"], "scope": entry["scope"], "stream": stream,
            "record": entry["record"],
            "record_sha256": _sha256(record_path),
            "pairs": pairs,
        })
    tranches.append(_offshore(root))
    return {
        "issue": "ABL-444",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "experiments/ABL444/config.json",
        "source_reread": SOURCE_REREAD,
        "source_reread_sha256": _sha256(source),
        "source_offshore_reread": OFFSHORE_REREAD,
        "source_offshore_reread_sha256": _sha256(root / OFFSHORE_REREAD),
        "readability_before": SIGN_TEST,
        "readability_after": FLOORED,
        "levellings_reported": list(LEVELLINGS),
        "floor_pct": {stream: readability_floor_pct(stream) for stream in ("solar", "wind")},
        "k": 1,
        "no_committed_record_is_edited": True,
        "tranches": tranches,
    }


def _offshore(root: Path) -> dict:
    """ABL-443's offshore re-read, graded the same four ways.

    Its record already carries every WAPE this needs, including both trailing
    references, so the only thing read from its *source* record is the
    challenger's slope and correlation -- G4 has no margin and cannot be
    recovered from a margins table.
    """
    offshore = json.loads((root / OFFSHORE_REREAD).read_text(encoding="utf-8"))
    source_path = root / offshore["source_record"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    by_key = {(cell.get("forecast_type"), cell["country"], cell["horizon_band"]): cell
              for cell in source["gate_cells"]}
    stream = offshore["stream"]
    floor = readability_floor_pct(stream)
    pairs = []
    for pair in offshore["pairs"]:
        cells = []
        for cell in pair["cells"]:
            key = (pair["forecast_type"], pair["country"], cell["band"])
            committed_cell = by_key[key]
            scores = _scores_for(committed_cell, cell)
            grades = {f"{levelling}/{readability}":
                      grade_cell(scores, stream, levelling=levelling,
                                 g23_readability=readability)
                      for levelling in LEVELLINGS for readability in READABILITY}
            challenger = scores["challenger"]["wape_pct"]
            cells.append({
                "band": cell["band"], "n": cell["n"],
                "minimum_n": committed_cell["gate"]["minimum_n"],
                "enough_pairs": cell["enough_pairs"],
                "gate_pass": committed_cell["gate"]["pass"],
                "floor_pct": floor,
                "wape": cell["wape"],
                "grades": {arm: grade.as_dict() for arm, grade in grades.items()},
                "labels": {arm: grade.label for arm, grade in grades.items()},
                "own_error_margin_pct": {
                    name: margin_pct_of_own_error(challenger, value)
                    for name, value in cell["wape"].items() if name != "challenger"},
            })
        rollup = {arm: pair_grade([grade_cell(
            _scores_for(by_key[(pair["forecast_type"], pair["country"], cell["band"])],
                        {"wape": cell["wape"]}),
            stream, levelling=arm.split("/")[0],
            g23_readability=arm.split("/")[1]) for cell in pair["cells"]]).label
            for arm in cells[0]["grades"]}
        pairs.append({"pair": pair["pair"], "country": pair["country"],
                      "forecast_type": pair["forecast_type"], "pair_grades": rollup,
                      "hold": HOLDS.get((pair["forecast_type"], pair["country"])),
                      "cells": cells})
    return {"tranche": "offshore", "scope": offshore["scope"], "stream": stream,
            "record": OFFSHORE_REREAD, "record_sha256": _sha256(root / OFFSHORE_REREAD),
            "source_record": offshore["source_record"],
            "source_record_sha256": _sha256(source_path),
            "pairs": pairs}


def _moves(record: dict, levelling: str) -> list:
    before, after = f"{levelling}/{SIGN_TEST}", f"{levelling}/{FLOORED}"
    moved = []
    for tranche in record["tranches"]:
        for pair in tranche["pairs"]:
            if pair["pair_grades"][before] == pair["pair_grades"][after]:
                continue
            reasons = {}
            for cell in pair["cells"]:
                for item in cell["grades"][after]["not_readable"]:
                    reasons.setdefault(item["condition"], []).append(
                        (cell["band"], cell["grades"][after]["skill_pct"]))
            moved.append({
                "tranche": tranche["tranche"], "pair": pair["pair"],
                "stream": tranche["stream"],
                "before": pair["pair_grades"][before],
                "after": pair["pair_grades"][after],
                "conditions": sorted(reasons),
                "hold": pair["hold"],
            })
    return moved


def _margins(cell: dict, arm: str, condition: str):
    """The abstaining condition's margin, in both denominators."""
    grade = cell["grades"][arm]
    reference = {"G2": {FIT_WINDOW: "constant_causal", TRAILING_28D: "constant_causal_28d"},
                 "G3": {FIT_WINDOW: "climatology_causal",
                        TRAILING_28D: "climatology_causal_28d"}}[condition][arm.split("/")[0]]
    return grade["skill_pct"].get(reference), cell["own_error_margin_pct"].get(reference)


def render(record: dict) -> str:
    lines = [
        "# ABL-444 — the G2/G3 readability floor, applied to every graded ABL-316 cell",
        "",
        f"Generated: {record['generated_at']}. Registration: "
        f"`{record['registration']}`, committed before this read existed.",
        "",
        f"Readability: **`{record['readability_before']}` → `{record['readability_after']}`**. "
        f"Floor: `readability_floor_pct` at k={record['k']} — "
        f"**{record['floor_pct']['solar']:.2f}% solar, {record['floor_pct']['wind']:.2f}% wind**.",
        "",
        "Arithmetic over records already on disk: each tranche's committed "
        "`results_*.json` for the challenger's own scores, and "
        f"`{record['source_reread']}` for ABL-437's trailing references on the same "
        "cells. **No refit, no new model, and no database read** — the trailing "
        "columns were computed against the replica by ABL-437 and are copied rather "
        "than recomputed, so this document cannot disagree with that one about a number.",
        "",
        "**No committed record is edited by this read.** It is a new document, on the "
        "ABL-418 / ABL-437 retro-grade precedent.",
        "",
        "Both registered axes stay live, so every cell is graded four ways: "
        "`{fit_window, trailing_28d} × {sign_test, floored}`. The `sign_test` column of "
        "each levelling is the published letter and is reproduced, not restated.",
        "",
        "## 1. What the floor moves",
        "",
        "`fit_window / sign_test` is what is **published today**. "
        "`trailing_28d / sign_test` is ABL-437's amended read.",
        "",
    ]
    for levelling in LEVELLINGS:
        moved = _moves(record, levelling)
        total = sum(len(tranche["pairs"]) for tranche in record["tranches"])
        from_a = [item for item in moved if item["before"] == "A"]
        lines += [
            f"### 1.{LEVELLINGS.index(levelling) + 1} Levelling `{levelling}` — "
            f"**{len(moved)} of {total} pair-records move**, "
            f"{len(from_a)} of them from `A`",
            "",
            "| tranche | pair | before | after | abstains on | hold |",
            "|---|---|:---:|:---:|---|---|",
        ]
        for item in moved:
            lines.append(f"| {item['tranche']} | {item['pair']} | {item['before']} | "
                         f"**{item['after']}** | {', '.join(item['conditions'])} | "
                         f"{'yes — see §4' if item['hold'] else '—'} |")
        if not moved:
            lines.append("| — | *no pair-record moves* | | | | |")
        lines.append("")
    lines += [
        "## 2. Every abstaining cell, with its margin in both denominators",
        "",
        "The CEO's constraint: the floor decides gradeability, it does not replace the "
        "number. `skill` is the registered column, `own` is ABL-385's own-error "
        "denominator, reported as the sensitivity. `n ≥ min` is ABL-434's column — "
        "reported beside every grade because the ladder cannot see it, and deliberately "
        "**not** folded into it.",
        "",
        "| levelling | tranche | pair | band | n | n ≥ min | condition | skill % | own % | floor % | letter |",
        "|---|---|---|---|---:|:---:|:---:|---:|---:|---:|:---:|",
    ]
    for levelling in LEVELLINGS:
        arm = f"{levelling}/{FLOORED}"
        for tranche in record["tranches"]:
            for pair in tranche["pairs"]:
                for cell in pair["cells"]:
                    for item in cell["grades"][arm]["not_readable"]:
                        skill, own = _margins(cell, arm, item["condition"])
                        lines.append(
                            f"| {levelling} | {tranche['tranche']} | {pair['pair']} | "
                            f"{cell['band']} | {cell['n']:,} | "
                            f"{'yes' if cell['enough_pairs'] else '**no**'} | "
                            f"{item['condition']} | {skill:+.2f} | {own:+.2f} | "
                            f"{cell['floor_pct']:.2f} | {cell['grades'][arm]['label']} |")
    lines += [
        "",
        "## 3. Every cell, all four arms",
        "",
        "| tranche | pair | band | n | fit/sign | fit/floor | 28d/sign | 28d/floor |",
        "|---|---|---|---:|:---:|:---:|:---:|:---:|",
    ]
    for tranche in record["tranches"]:
        for pair in tranche["pairs"]:
            for cell in pair["cells"]:
                labels = cell["labels"]
                lines.append(
                    f"| {tranche['tranche']} | {pair['pair']} | {cell['band']} | "
                    f"{cell['n']:,} | {labels[f'{FIT_WINDOW}/{SIGN_TEST}']} | "
                    f"{labels[f'{FIT_WINDOW}/{FLOORED}']} | "
                    f"{labels[f'{TRAILING_28D}/{SIGN_TEST}']} | "
                    f"**{labels[f'{TRAILING_28D}/{FLOORED}']}** |")
    lines += ["", "## 4. Holds that travel with these letters", ""]
    holds = {(tranche["tranche"], pair["pair"]): pair["hold"]
             for tranche in record["tranches"] for pair in tranche["pairs"] if pair["hold"]}
    for (tranche_id, pair_label), hold in sorted(holds.items()):
        lines.append(f"- **{tranche_id} {pair_label}** — {hold}")
    lines += [
        "",
        "## 5. What this does not say",
        "",
        "- **It changes gradeability, not skill.** No model is better or worse for this "
        "read; some verdicts become honest abstentions.",
        "- **It cannot raise a grade.** `N` is only ever reached from what would have "
        "been `A` or `B`. Note that `N` ranks *better* than `B` on the ladder — an "
        "abstention is a weaker negative than a named failure — so a `B → N` move lowers "
        "the severity while leaving the pair exactly as non-promotable.",
        "- **It touches no part of ABL-348's registration** — windows, bands, metric, "
        "baseline, minimum n, source, `not_evaluable` — so `voids_this_registration` is "
        "not triggered.",
        "- **Tranche 1a is absent**, for ABL-437's reason and not a new one: it was "
        "fitted before ABL-389 existed and carries no causal reference columns, so G2 "
        "and G3 read *not measured* there under every arm.",
        "- **This promotes nothing.** Promotion remains a pre-registered gate read plus "
        "a Board decision.",
        "",
        f"Source records, SHA-256: `{record['source_reread']}` "
        f"`{record['source_reread_sha256'][:16]}…`; per tranche —",
        "",
    ]
    for tranche in record["tranches"]:
        lines.append(f"- `{tranche['record']}` `{tranche['record_sha256'][:16]}…`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--report-out", default=REPORT_OUT)
    args = parser.parse_args()
    for path in (args.json_out, args.report_out):
        if any(marker in Path(path).name for marker in PROTECTED):
            raise SystemExit(f"refusing to write {path}: that path belongs to another read")
    record = read(ROOT)
    (ROOT / args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.report_out).write_text(render(record), encoding="utf-8")
    for levelling in LEVELLINGS:
        moved = _moves(record, levelling)
        from_a = sum(1 for item in moved if item["before"] == "A")
        print(f"{levelling}: {len(moved)} pair-records move, {from_a} from A")
    print(f"wrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
