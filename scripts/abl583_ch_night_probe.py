#!/usr/bin/env python
"""What the ABL-583 CH solar artifact predicts after dark, before and after the clamp.

ABL-583's "night question", in scope for CH and only for CH. ABL-580 section 6
found CZ and RO carrying a **positive** night floor that the clamp zeroes 16/16
hours, and argued it reads as a global 2-3% level floor rather than as a night
defect. CH is the other case: ABL-581 reports its night negative-prediction rate
at 64.06% at the gate seed, down from 80.47% at 25 features.

**Report it; do not fix it.** The fit path, the builder and the class are the
ones that were graded.

WHY THE CLAMP LOG CANNOT ANSWER THIS ON ITS OWN
-----------------------------------------------
`forecast_clamp_log` is the right instrument for CZ and RO and the wrong one for
CH, and the reason is one line of `src/solar_clamp.py`:

    raised = (~zeroing) & (original < 0.0)

For a country where the night mask applies -- CH is `night_generation_possible`
False, so it does -- a **negative night prediction is not counted in
`hours_raised_floor` at all**. It lands in `hours_zeroed_night`, because that
counter is on `|prediction| > threshold` and takes either sign. So for CH:

    hours_raised_floor  counts DAYLIGHT negatives only
    hours_zeroed_night  counts night rows of either sign, indistinguishably
    mw_removed_night    sums the night predictions signed, so a night floor that
                        is mostly negative shows up here as a NEGATIVE number

Reading CH's `hours_raised_floor == 0` as "no negative predictions" would
therefore be wrong in exactly the way that matters, and it is the reading the
ABL-580 table invites. This probe reconstructs the same served frame and reports
the pre-clamp distribution directly, then applies the clamp and checks the two
post-clamp invariants by measurement rather than by construction.

WHAT IT DOES NOT CLAIM
----------------------
- **The negative rate is not a defect measurement.** CLAUDE.md is explicit and
  ABL-395 measured it: CH's night-negative rate over eight *control* fits is
  77.05% +/- 10.11 with a 27.34pp single-seed null, and both 80.47% (f25) and
  64.06% (f27) at seed 42 sit inside it. One fit's night-hour fraction is one
  draw. This probe reports the number for the served artifact and takes no
  position on whether 27 features moved it.
- **The night actual is not re-measured here.** It is read from the batch's
  committed contamination-screen record so the two cannot disagree, and the
  forecast targets are two future days with no actuals of their own.

Read-only against the replica. Writes only its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.forecaster import Forecaster  # noqa: E402
from src.solar_clamp import (  # noqa: E402
    ZEROED_NIGHT_MW_THRESHOLD,
    clamp_solar_forecasts,
)
from src.solar_geometry import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    NIGHT_GENERATION_POSSIBLE,
    SOLAR_REPRESENTATIVE_POINTS,
    is_night_hour,
)

COUNTRY = "CH"
FORECAST_TYPE = "solar"

#: The horizons ABL-583 item 6 serving-verifies, so this probe measures the same
#: rows `forecast_daily.py --horizons 1,2` produces.
HORIZON_DAYS = (1, 2)


def served_frame(model, reference_date):
    """The rows `forecast_daily.py` would hand to `save_forecasts`.

    Built through `Forecaster.predict_d2`, which is the entry point the runner
    uses, so this is the served vector and not a re-implementation of it. The
    columns are the ones `clamp_solar_forecasts` reads.
    """
    frames = []
    for horizon in HORIZON_DAYS:
        frame = model.predict_d2(reference_date=reference_date, horizon_days=horizon)
        frame = frame.copy()
        frame["horizon_days"] = horizon
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["country_code"] = COUNTRY
    out["forecast_type"] = FORECAST_TYPE
    out["renewable_type"] = FORECAST_TYPE
    out["model_name"] = model.model_name if hasattr(model, "model_name") else "abl583"
    return out


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "n": int(len(values)),
        "mean_mw": round(float(values.mean()), 4),
        "min_mw": round(float(values.min()), 4),
        "max_mw": round(float(values.max()), 4),
        "n_negative": int((values < 0).sum()),
        "pct_negative": round(float((values < 0).mean() * 100.0), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description=("Measure the ABL-583 CH solar artifact's night predictions "
                     "either side of the ABL-337 serving clamp."))
    parser.add_argument("--replica-db", default=config.DATABASE_PATH,
                        help="Read-only replica (default: ENERGY_DB_PATH).")
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument(
        "--screens",
        default="reports/abl_583_contamination_screens.json",
        help=("The batch's committed screen record, read for the night ACTUAL "
              "level so this probe cannot disagree with it."))
    parser.add_argument("--reference-date", default=None,
                        help="Forecast reference date (default: today).")
    parser.add_argument("--json-out", default="reports/abl_583_ch_night_probe.json")
    args = parser.parse_args()

    replica = Path(args.replica_db)
    if not replica.is_file():
        raise SystemExit(f"replica not found: {replica}")
    config.DATABASE_PATH = str(replica)

    reference_date = (date.fromisoformat(args.reference_date)
                      if args.reference_date else date.today())

    path = Path(args.models_dir) / COUNTRY / FORECAST_TYPE / "model.joblib"
    model = Forecaster.load(COUNTRY, FORECAST_TYPE, path=str(path))
    frame = served_frame(model, reference_date)

    hour_starts = pd.to_datetime(frame["target_timestamp_utc"]).dt.floor("h")
    night = np.asarray(is_night_hour(COUNTRY, list(hour_starts),
                                     NIGHT_ELEVATION_THRESHOLD_DEG))
    pre = frame["forecast_value"].to_numpy(dtype=float)

    clamped, stats = clamp_solar_forecasts(frame)
    post = clamped["forecast_value"].to_numpy(dtype=float)
    stat = next(s for s in stats if s.country_code == COUNTRY)

    night_actual = None
    screens = Path(args.screens)
    if screens.is_file():
        record = json.loads(screens.read_text(encoding="utf-8"))
        pair = next((p for p in record["pairs"]
                     if p["country"] == COUNTRY and p["forecast_type"] == FORECAST_TYPE),
                    None)
        if pair and "night_floor" in pair:
            whole = next(w for w in pair["night_floor"]["windows"]
                         if w["window"] == "whole_fit_window")
            night_actual = {
                "source": str(screens.as_posix()),
                "window": [whole["start"], whole["end_exclusive"]],
                "n_night_rows": whole["n_night_rows"],
                "night_mean_mw": whole["night_mean_mw"],
                "night_max_mw": whole["night_max_mw"],
                "n_night_above_1mw": whole["n_night_above_threshold"],
                "n_night_negative": whole["n_night_negative"],
            }

    payload = {
        "issue": "ABL-583",
        "country": COUNTRY,
        "forecast_type": FORECAST_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_date": str(reference_date),
        "horizon_days": list(HORIZON_DAYS),
        "target_window": [str(hour_starts.min()), str(hour_starts.max())],
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "artifact": {
            "path": str(path),
            "n_features": len(model.feature_columns),
            "training_source": model.training_source,
            "algorithm": getattr(model, "algorithm", None),
        },
        "clamp_configuration": {
            "night_threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
            "zeroed_night_mw_threshold": ZEROED_NIGHT_MW_THRESHOLD,
            "night_generation_possible": NIGHT_GENERATION_POSSIBLE.get(COUNTRY),
            "night_mask_applied": stat.night_mask_applied,
            "representative_point": SOLAR_REPRESENTATIVE_POINTS.get(COUNTRY),
        },
        "pre_clamp": {
            "all_hours": describe(pre),
            "night_hours": describe(pre[night]),
            "daylight_hours": describe(pre[~night]),
        },
        "post_clamp": {
            "all_hours": describe(post),
            "night_hours": describe(post[night]),
            # The two invariants, measured rather than asserted from the source.
            "any_served_row_negative": bool((post < 0).any()),
            "every_night_row_exactly_zero": bool(np.all(post[night] == 0.0)),
        },
        # The clamp-log fields the deployed run will write, computed here from
        # the same frame so the two records are checkable against each other.
        "clamp_log_fields": {
            "rows_total": stat.rows_total,
            "night_hours": stat.night_hours,
            "hours_zeroed_night": stat.hours_zeroed_night,
            "hours_raised_floor": stat.hours_raised_floor,
            "mw_removed_night": round(stat.mw_removed_night, 4),
            "mw_added_floor": round(stat.mw_added_floor, 4),
            "mw_removed_total": round(stat.mw_removed_total, 4),
            "min_forecast_mw": round(stat.min_forecast_mw, 4),
            "max_night_forecast_mw": round(stat.max_night_forecast_mw, 4),
        },
        "how_to_read_hours_raised_floor": (
            "For CH the night mask applies, and `raised` is `(~zeroing) & "
            "(original < 0.0)`, so hours_raised_floor counts DAYLIGHT negatives "
            "only. A negative night prediction is counted by hours_zeroed_night, "
            "which is on |prediction| and takes either sign. Read "
            "pre_clamp.night_hours.pct_negative for the night question."),
        "night_actual": night_actual,
        "not_a_defect_measurement": (
            "ABL-395 measured CH's night-negative rate over eight control fits at "
            "77.05% +/- 10.11 with a 27.34pp single-seed null; 80.47% (f25) and "
            "64.06% (f27) at seed 42 both sit inside it. This is one fit's draw, "
            "reported for the artifact being shipped. Nothing here grades, and "
            "nothing here is fixed."),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pre_night = payload["pre_clamp"]["night_hours"]
    print(f"[{COUNTRY}/{FORECAST_TYPE}] {stat.rows_total} rows, "
          f"{stat.night_hours} night hours")
    print(f"  pre-clamp night: mean {pre_night['mean_mw']} MW, "
          f"min {pre_night['min_mw']}, max {pre_night['max_mw']}, "
          f"negative {pre_night['n_negative']}/{pre_night['n']} "
          f"({pre_night['pct_negative']}%)")
    print(f"  clamp: zeroed {stat.hours_zeroed_night}/{stat.night_hours}, "
          f"raised {stat.hours_raised_floor} (daylight only for CH), "
          f"mw_removed_night {stat.mw_removed_night:.2f}")
    print(f"  post-clamp: any negative = "
          f"{payload['post_clamp']['any_served_row_negative']}, "
          f"night all zero = "
          f"{payload['post_clamp']['every_night_row_exactly_zero']}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
