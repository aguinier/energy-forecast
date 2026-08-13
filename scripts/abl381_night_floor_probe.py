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

What `pct_of_total_energy_at_night` buys you (ABL-396)
-----------------------------------------------------
It is not a descriptive statistic. Call it `f`, and let `W` be the daylight-only
WAPE of any challenger over the same window. Night actuals are non-negative, so
the two extreme cases bound the all-hours WAPE exactly:

  * a challenger that predicts **0** at night -- which is what the ABL-337
    serving clamp forces, since it zeroes on this same predicate -- scores
    `W*(1-f) + f`;
  * a challenger that reproduces the floor **perfectly** scores `W*(1-f)`.

So `f` is simultaneously (a) the full width, in WAPE percentage points, of the
interval an all-hours read can sit in relative to the daylight-only read of the
same challenger, and (b) a hard lower bound on the WAPE of any *served* solar
forecast, because the clamp cannot do better than zero against a floor. Both are
model-free. That makes `f` the rankable quantity for "how much would this move a
gate read", which is why every window reports it.

Checked against the one country where a real gate read exists: BG's gate `f` is
4.98% at a daylight-only WAPE of 18.90%, so the band is [17.96%, 22.94%], and
ABL-381 measured the all-hours read at 18.89% -- inside it, near the
floor-reproducing end, which is the same conclusion that pack reached from the
model's own night predictions (224.78 MW against 225.13 MW actual).

Attribution, and why daylight is measured too
---------------------------------------------
"Is the floor upstream or introduced by the source mapping" is only answerable on
rows both tables actually carry, so the comparison runs on the **intersection**
of the two sources' hourly indices rather than on their separate summaries. The
verdict uses no threshold that is not already registered -- the classifying
constant is `solar_features.IMPOSSIBLE_NIGHT_THRESHOLD_MW`, and a between-table
difference smaller than it cannot move a single row across it:

  * `no_floor`        -- neither table has a night hour above the threshold.
  * `upstream`        -- night rows agree to within the threshold. The mapping
                         is not the author; switching source will not fix it.
  * `source_mapping`  -- daylight agrees to within the threshold and night does
                         not. The mapping is the author.
  * `series_differ`   -- both bands differ. Not attributable from here, and the
                         usual cause is benign: the two tables published at
                         different resolutions (ABL-332 -- 22 of 24 countries
                         carry sub-hourly rows in at least one table) so their
                         hourly means legitimately disagree. `native_resolution`
                         is reported per source so this is visible rather than
                         inferred.

Daylight statistics are what stop an ABL-188 zero-fill reading as a mapping fix:
a table that erased the whole series also has no night floor, and only the
daylight comparison tells the two apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.solar_features import IMPOSSIBLE_NIGHT_THRESHOLD_MW, night_mask

#: A night hour above this reads as "not plausibly solar". ABL-338 used the same
#: 1 MW threshold on FR, so the counts here are directly comparable to that pack.
#: It is the registered fit-exclusion constant rather than a second copy of it --
#: ABL-396 made the attribution verdict depend on this number, and two thresholds
#: that drift apart would let a row be "impossible" for the fit and "identical"
#: for the source comparison at the same time.
NIGHT_MW_THRESHOLD = IMPOSSIBLE_NIGHT_THRESHOLD_MW

#: Two source tables are treated as carrying the same series in a band when
#: their energies over that band differ by less than this, relatively.
#:
#: Unlike `NIGHT_MW_THRESHOLD` this is a **reporting choice, not a registered
#: constant**, and it is deliberately relative: 1 MW is the right absolute test
#: for a night hour, whose honest value is exactly 0, and a category error for a
#: daylight hour running to 5,000 MW. The raw per-band differences are reported
#: beside every verdict so a reader who wants a different cut can re-verdict
#: without re-running. Measured separation on the two known cases (2026-08-13,
#: ABL-348 fit window): BG 0.031% night / 0.178% daylight -- one table, one
#: defect; CH 100% night / 4.018% daylight -- two genuinely different series.
SOURCE_AGREEMENT_TOLERANCE_PCT = 1.0


def _hourly(country: str, start, end, source: str, replica: str) -> pd.Series:
    frame = db.load_renewable_type_data(country, "solar", str(start), str(end),
                                        source=source, db_path=replica)
    if frame.empty:
        return pd.Series(dtype=float)
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed",
                            utc=True).dt.tz_localize(None)
    return pd.Series(frame["target_value"].to_numpy(dtype=float),
                     index=stamps).sort_index()


def _native_resolution(country: str, start, end, source: str, replica: str) -> dict:
    """Cadence of the raw rows, before `load_renewable_type_data` averages them.

    Reported because it is the benign explanation for a `series_differ` verdict:
    two tables that published the same country at different resolutions produce
    genuinely different hourly means, and that is not a mapping defect.
    """
    sql = (f"SELECT COUNT(*) AS n, "
           f"COUNT(DISTINCT CAST(strftime('%M', timestamp_utc) AS INTEGER)) AS minutes "
           f"FROM {source} WHERE country_code = ? "
           f"AND timestamp_utc >= ? AND timestamp_utc < ? AND solar_mw IS NOT NULL")
    with db.get_connection(readonly=True, db_path=replica) as conn:
        row = conn.execute(sql, (country, str(start), str(end))).fetchone()
    n, minutes = int(row[0]), int(row[1] or 0)
    return {"raw_rows": n,
            "distinct_minute_marks": minutes,
            "sub_hourly": bool(minutes > 1)}


def _window(country: str, series: pd.Series, label: str) -> dict:
    if series.empty:
        return {"window": label, "n_rows": 0, "note": "no rows"}
    night = night_mask(country, list(series.index))
    values = series.to_numpy(dtype=float)
    # ABL-396: an hour can survive the loader as NaN -- measured on the replica
    # 2026-08-13, CZ `energy_renewable` carries 93 of them in the fit window.
    # Every statistic below is taken over finite rows only and the missing count
    # is reported rather than absorbed: reading a hole as 0.0 would invent a
    # night zero, which is exactly the direction this screen is trying to
    # measure, and would flatter a table into looking clean.
    finite = np.isfinite(values)
    night, day = night & finite, (~night) & finite
    night_values, day_values = values[night], values[day]
    above = night_values > NIGHT_MW_THRESHOLD
    total_energy = float(np.abs(values[finite]).sum())
    return {
        "window": label,
        "n_rows": int(len(values)),
        "n_missing_rows": int((~finite).sum()),
        "n_night_rows": int(night.sum()),
        "n_night_above_threshold": int(above.sum()),
        "pct_of_night_rows_above_threshold": (
            round(float(above.mean() * 100.0), 2) if night.sum() else None),
        "night_mean_mw": round(float(night_values.mean()), 2) if night.sum() else None,
        "night_median_mw": round(float(np.median(night_values)), 2) if night.sum() else None,
        "night_max_mw": round(float(night_values.max()), 2) if night.sum() else None,
        "night_min_mw": round(float(night_values.min()), 2) if night.sum() else None,
        "series_mean_mw": round(float(values[finite].mean()), 2) if finite.any() else None,
        # What share of everything the series books is booked in the dark. This
        # is the number that says whether the floor is a rounding artefact or a
        # material part of the target the model is being fitted to.
        "pct_of_total_energy_at_night": (
            round(float(night_values.sum() / total_energy * 100.0), 2)
            if total_energy else None),
        # ABL-396: the same share taken on |MW|, and the one to rank on. The
        # field above is signed, so a series that reads *negative* at night --
        # NL is every night hour of both windows, -1.47 to -0.12 MW -- reports a
        # negative share and sorts as though it were the cleanest country in the
        # fleet. The WAPE bound in the module docstring is an absolute-value
        # argument and needs this form: a clamped forecast pays the magnitude of
        # a night error regardless of its sign.
        "wape_floor_pct_if_clamped": (
            round(float(np.abs(night_values).sum() / total_energy * 100.0), 3)
            if total_energy else None),
        # A night hour that is negative is a different defect from a night hour
        # that is too high, and only this count tells them apart.
        "n_night_negative": int((night_values < 0).sum()),
        # ABL-396: the daylight band, so a table that zero-filled the whole
        # series (ABL-188) does not read as a table that fixed the night floor.
        "n_daylight_rows": int(day.sum()),
        "daylight_mean_mw": (round(float(day_values.mean()), 2)
                             if day_values.size else None),
        "daylight_max_mw": (round(float(day_values.max()), 2)
                            if day_values.size else None),
    }


def _compare_sources(country: str, label: str,
                     series_by_source: dict) -> dict:
    """Attribute a night floor to the feed or to the source mapping (ABL-396).

    Runs on the intersection of the two sources' hourly indices, so the two
    tables are compared on rows they both carry rather than through summaries
    over different row sets. Verdict rules are in the module docstring; the only
    constant they use is `NIGHT_MW_THRESHOLD`.
    """
    named = [(name, s) for name, s in series_by_source.items() if not s.empty]
    if len(named) < 2:
        return {"window": label,
                "verdict": "single_source",
                "note": f"{len(named)} of {len(series_by_source)} sources have rows"}

    (name_a, a), (name_b, b) = named[0], named[1]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"window": label, "verdict": "no_common_rows", "n_common_rows": 0}

    va, vb = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    night = night_mask(country, list(common))
    # An attribution has to compare rows both tables actually report. An hour
    # finite in one and NaN in the other says nothing about the mapping's
    # treatment of night, so it is excluded here and counted instead.
    both = np.isfinite(va) & np.isfinite(vb)
    night, day = night & both, (~night) & both
    diff = np.abs(va - vb)

    def band(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"n": 0, "rel_energy_diff_pct": None,
                    "agree_within_threshold_pct": None, "max_abs_diff_mw": None,
                    "sum_mwh": {name_a: None, name_b: None}}
        sa, sb = float(va[mask].sum()), float(vb[mask].sum())
        return {
            "n": int(mask.sum()),
            # The verdict's discriminator. Relative *energy* rather than a row
            # count, because energy is what drives the gate impact `f` above,
            # and because a handful of revised hours should not decide an
            # attribution -- BG's fit window carries exactly one such day
            # (2026-02-14), 27 daylight hours out of 2,722.
            "rel_energy_diff_pct": (round(abs(sa - sb) / sa * 100.0, 4)
                                    if sa else (None if sb == 0 else 100.0)),
            "agree_within_threshold_pct": round(
                float((diff[mask] <= NIGHT_MW_THRESHOLD).mean() * 100.0), 2),
            "max_abs_diff_mw": round(float(diff[mask].max()), 3),
            "sum_mwh": {name_a: round(sa, 1), name_b: round(sb, 1)},
        }

    night_band, day_band = band(night), band(day)
    max_night = max(float(va[night].max()), float(vb[night].max())) if night.any() else 0.0

    def differs(b: dict) -> bool:
        return b["rel_energy_diff_pct"] is not None and \
            b["rel_energy_diff_pct"] > SOURCE_AGREEMENT_TOLERANCE_PCT

    if max_night <= NIGHT_MW_THRESHOLD:
        verdict = "no_floor"
    elif not differs(night_band):
        verdict = "upstream"
    elif not differs(day_band):
        verdict = "source_mapping"
    else:
        verdict = "series_differ"

    return {
        "window": label,
        "sources": [name_a, name_b],
        "n_common_rows": int(len(common)),
        "n_common_rows_finite_in_both": int(both.sum()),
        "night": night_band,
        "daylight": day_band,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", default="BG,CH",
                        help="comma-separated codes, or 'all' for every "
                             "config.SUPPORTED_COUNTRIES entry (ABL-396)")
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
    if args.countries.strip().lower() == "all":
        countries = list(config.SUPPORTED_COUNTRIES)
    else:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    result["countries_screened"] = countries

    spans = [("fit", fit_start, gate_start), ("gate", gate_start, gate_end)]
    for country in countries:
        entry = {"country": country, "sources": []}
        # Series are loaded once and kept, because the ABL-396 attribution
        # compares the two tables row by row on their common index rather than
        # comparing the per-source summaries below.
        loaded = {label: {} for label, _, _ in spans}
        for source in sources:
            windows = []
            for label, start, end in spans:
                series = _hourly(country, start, end, source, str(replica))
                loaded[label][source] = series
                windows.append(_window(country, series, label))
            entry["sources"].append({
                "source": source,
                "windows": windows,
                "native_resolution": {
                    label: _native_resolution(country, start, end, source, str(replica))
                    for label, start, end in spans
                },
            })
        entry["source_comparison"] = [
            _compare_sources(country, label, loaded[label]) for label, _, _ in spans
        ]
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
