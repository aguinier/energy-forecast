#!/usr/bin/env python3
"""Quantify the overnight floor in the solar *actuals* of an ABL-316 tranche.

The ABL-381 non-negativity probe reported, on the rows the gate actually scored,
a mean **actual** at night of **225.13 MW for BG** against **0.00 MW for CH**.
Night there is `solar_features.night_mask` -- the sun geometrically below
-8 degrees for the whole hour at the country's capacity-weighted point -- so a
non-zero night actual cannot be a mask artefact or a timezone offset. It is a
property of the series.

That is the shape of the defect ABL-337 filed against FR, which ABL-338 then
measured: FR's `energy_renewable.solar_mw` read above 1 MW at 488 of 11,614
night training rows, and excluding those rows moved FR's mean night *prediction*
from 22.46 to 0.05 MW **and improved its daylight MAE by 1.5%**. So a night floor
in the target is not cosmetic -- the model learns it, and it costs daylight
accuracy to carry.

This script sizes the same question for a tranche's countries, on both renewable
source tables, over the registered fit and gate windows. It answers:

  * how many night hours read above a threshold, and how high they go;
  * what share of the series' total energy is booked when the sun is down;
  * whether the floor is source-specific (`energy_generation` vs
    `energy_renewable`) or present in both, which separates an ingest defect
    from a source-mapping defect.

Read-only against the replica (`mode=ro`). Writes nothing but its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import db
from src.solar_features import night_mask

#: A night hour above this reads as "not plausibly solar". ABL-338 used the same
#: 1 MW threshold on FR, so the counts here are directly comparable to that pack.
NIGHT_MW_THRESHOLD = 1.0


def _hourly(country: str, start, end, source: str, replica: str) -> pd.Series:
    frame = db.load_renewable_type_data(country, "solar", str(start), str(end),
                                        source=source, db_path=replica)
    if frame.empty:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed",
                            utc=True).dt.tz_localize(None)
    return pd.Series(frame["target_value"].to_numpy(dtype=float),
                     index=stamps).sort_index()


def _window(country: str, series: pd.Series, label: str) -> dict:
    if series.empty:
        return {"window": label, "n_rows": 0, "note": "no rows"}
    night = night_mask(country, list(series.index))
    values = series.to_numpy(dtype=float)
    night_values = values[night]
    above = night_values > NIGHT_MW_THRESHOLD
    total_energy = float(np.abs(values).sum())
    return {
        "window": label,
        "n_rows": int(len(values)),
        "n_night_rows": int(night.sum()),
        "n_night_above_threshold": int(above.sum()),
        "pct_of_night_rows_above_threshold": (
            round(float(above.mean() * 100.0), 2) if night.sum() else None),
        "night_mean_mw": round(float(night_values.mean()), 2) if night.sum() else None,
        "night_median_mw": round(float(np.median(night_values)), 2) if night.sum() else None,
        "night_max_mw": round(float(night_values.max()), 2) if night.sum() else None,
        "night_min_mw": round(float(night_values.min()), 2) if night.sum() else None,
        "series_mean_mw": round(float(values.mean()), 2),
        # What share of everything the series books is booked in the dark. This
        # is the number that says whether the floor is a rounding artefact or a
        # material part of the target the model is being fitted to.
        "pct_of_total_energy_at_night": (
            round(float(night_values.sum() / total_energy * 100.0), 2)
            if total_energy else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default="BG,CH")
    parser.add_argument("--replica-db", required=True)
    parser.add_argument("--sources", default="energy_generation,energy_renewable")
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))

    result = {
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "night_mw_threshold": NIGHT_MW_THRESHOLD,
        "night_definition": "solar_features.night_mask -- sun below "
                            "NIGHT_ELEVATION_THRESHOLD_DEG for the whole hour",
        "windows": {"fit": [str(fit_start), str(gate_start)],
                    "gate": [str(gate_start), str(gate_end)]},
        "countries": [],
    }
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for country in [c.strip().upper() for c in args.countries.split(",")]:
        entry = {"country": country, "sources": []}
        for source in sources:
            windows = [
                _window(country, _hourly(country, fit_start, gate_start, source,
                                         str(replica)), "fit"),
                _window(country, _hourly(country, gate_start, gate_end, source,
                                         str(replica)), "gate"),
            ]
            entry["sources"].append({"source": source, "windows": windows})
        result["countries"].append(entry)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
