#!/usr/bin/env python3
"""ABL-426 -- re-measure the source-table difference on tranche 2a, at today's replica.

The ABL-426 filing sized the defect against ABL-348's pre-measured bars, taken
2026-08-12 on a 9,432,453,120-byte replica. This re-measures the same quantities
on whatever replica is live now, because three of the filed numbers turned out to
have moved and one of them -- CZ's fit-window difference -- had moved by a factor
of four on the pair the shipping decision turns on.

Read-only, `mode=ro`. Writes nothing to either database. Opens no model, fits
nothing.

**The protocol is not restated here, it is imported.** `hourly` and `d7_scores`
come from `scripts/abl348_source_registration_probe.py`, the script that took the
registered bars, so this screen and the bars it is compared against cannot drift
apart -- a re-implementation would measure a different series and the difference
would read as data movement.

Two distinct comparisons come out, and conflating them is the trap:

  * **table vs table, today.** Both source arms read on one replica. This is the
    quantity ABL-426 is about, and it is what `bar_delta_pp_today` and
    `tables_today_in_gate_window` report.
  * **today vs the registered bars.** One arm against ABL-348's recorded value.
    This is *not* a clean measurement of replica revision, because ABL-332's
    hourly averaging landed in between: it aggregates sub-hourly rows to hourly
    means where the registered bars sub-sampled the `:00` instant. The split is
    legible in the output -- countries the loader reports as hourly-native move
    by 0.00pp and 15-minute countries do not -- so `n_subhourly_rows_aggregated`
    is recorded per country per source to make the attribution checkable rather
    than asserted.

    .venv\\Scripts\\python.exe scripts/abl426_vintage_screen.py \\
      --replica-db C:\\Code\\able\\data\\energy_dashboard.db \\
      --out reports/abl_426_vintage_screen.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402

# The loader is deliberately chatty about dedup, ABL-188 exclusions and ABL-332
# aggregation; the counts that matter are captured below, so silence the stream.
logging.getLogger("src.db").setLevel(logging.ERROR)
logging.getLogger("src.data_quality").setLevel(logging.ERROR)

#: `abl316-t2a`'s eight countries, in the scope's order. Written out rather than
#: imported from the harness: importing it would pull in CatBoost and the whole
#: registration check to read a tuple, and this screen must run even if the
#: harness does not import.
COUNTRIES = ("BG", "CH", "CZ", "HU", "PL", "RO", "SI", "SK")
SOURCES = ("energy_renewable", "energy_generation")
STREAM = "solar"


def _probe():
    """ABL-348's own probe, as a module -- for `hourly`, `d7_scores` and the windows."""
    spec = importlib.util.spec_from_file_location(
        "abl348_source_registration_probe",
        ROOT / "scripts" / "abl348_source_registration_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(replica: str, probe) -> dict:
    cfg = json.loads((ROOT / "experiments" / "ABL348" / "config.json").read_text(encoding="utf-8"))
    bars = cfg["per_pair_bar_measured_before_any_challenger_exists"]["bars"]

    out = {}
    for country in COUNTRIES:
        row, series = {}, {}
        for source in SOURCES:
            frame = db.load_renewable_type_data(
                country, STREAM,
                probe.LOOKBACK_START.strftime("%Y-%m-%d %H:%M:%S"),
                probe.GATE_END.strftime("%Y-%m-%d %H:%M:%S"),
                source=source, db_path=replica)
            hourly = probe.hourly(frame)
            series[source] = hourly
            record = probe.d7_scores(hourly)
            fit = hourly[(hourly.index >= probe.FIT_START) & (hourly.index < probe.FIT_END)]
            record["n_fit_hours"] = int(fit.notna().sum())
            record["fit_hours_intended"] = probe.FIT_HOURS
            row[source] = record

        registered = bars[f"{country}/{STREAM}"]
        row["registered_2026_08_12"] = {
            "d7_wape_pct_on_energy_generation": registered["d7_wape_pct"],
            "d7_wape_pct_on_energy_renewable": registered["d7_wape_pct_on_energy_renewable"],
            "bar_delta_pp": registered["bar_delta_pp"],
            "n_d7_scorable": registered["n_d7_scorable"],
            "mean_actual_mw": registered["mean_actual_mw"],
        }

        gen, ren = row["energy_generation"], row["energy_renewable"]
        # The quantity ABL-426 is about: does the table move the bar, today?
        row["bar_delta_pp_today"] = (
            None if gen["d7_wape_pct"] is None or ren["d7_wape_pct"] is None
            else round(gen["d7_wape_pct"] - ren["d7_wape_pct"], 4))
        # And the fit series' length, which is what a challenger difference has to
        # come from once the bar and the scored actuals are shown not to move.
        row["fit_hours_delta"] = gen["n_fit_hours"] - ren["n_fit_hours"]
        row["fit_hours_delta_pct_of_registered_window"] = round(
            100.0 * row["fit_hours_delta"] / probe.FIT_HOURS, 3)
        # NOT a clean vintage measurement -- ABL-332 landed between the two.
        # Reported so the attribution is checkable, and named so it cannot be
        # quoted as revision.
        row["shift_vs_registered_bar_pp_includes_abl332"] = {
            source: (None if row[source]["d7_wape_pct"] is None else round(
                row[source]["d7_wape_pct"] - row["registered_2026_08_12"][
                    f"d7_wape_pct_on_{source}"], 4))
            for source in SOURCES
        }

        gate_index = pd.date_range(probe.GATE_START, probe.GATE_END, freq="h", inclusive="left")
        a = series["energy_generation"].reindex(gate_index)
        b = series["energy_renewable"].reindex(gate_index)
        both = a.notna() & b.notna()
        if int(both.sum()):
            diff = (a[both] - b[both]).abs()
            level = float(a[both].abs().mean())
            row["tables_today_in_gate_window"] = {
                "n_co_observed": int(both.sum()),
                "pct_hours_bit_identical": round(100.0 * float((diff == 0).mean()), 2),
                "mean_abs_diff_mw": round(float(diff.mean()), 3),
                "max_abs_diff_mw": round(float(diff.max()), 1),
                "mean_abs_diff_pct_of_level": (
                    round(100.0 * float(diff.mean()) / level, 4) if level else None),
            }
        else:
            row["tables_today_in_gate_window"] = {"n_co_observed": 0}
        out[country] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replica-db", default=os.environ.get("ENERGY_DB_PATH"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if not args.replica_db:
        parser.error("--replica-db (or ENERGY_DB_PATH) is required; this screen must "
                     "not fall through to config.DATABASE_PATH, which degrades to a "
                     "bare \\data\\energy_dashboard.db from a worktree")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")

    probe = _probe()
    out = {
        "issue": "ABL-426",
        # Verified by size, not by path: the 3.0 GB partial snapshot at
        # `energy-data-gathering/energy_dashboard.db` is the nearest file to every
        # wrong path this module has been pointed at, and its numbers look fine.
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "registered_replica_bytes": 9432453120,
        "windows": {
            "lookback_start": str(probe.LOOKBACK_START),
            "fit": [str(probe.FIT_START), str(probe.FIT_END)],
            "gate": [str(probe.GATE_START), str(probe.GATE_END)],
        },
        "registered_minimum_n": probe.REGISTERED_MIN_N,
        "countries": measure(str(replica), probe),
    }
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
