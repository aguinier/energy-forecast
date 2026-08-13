"""ABL-386: how much holiday signal is in the window at all, before anything is fitted.

An A/B on `is_holiday` is only informative if the window *contains* holidays. If
the holdout has two holiday days out of 44, `is_holiday` is 0 on ~95% of rows and
a null result means "this window could not have shown an effect" rather than
"these features do nothing". That distinction has to be settled before the fit,
not argued afterwards, so this probe runs first and its numbers go into the
registration.

It measures, per country, for the fit window and the holdout window separately:

- holiday days and holiday hours, and the share of rows with `is_holiday == 1`
- bridge days, and the share of rows with `is_bridge_day == 1`
- the same restricted to **daylight** hours, which is where the primary metric
  lives - a holiday that falls entirely in the night band cannot move daylight MAE
- the spread of `days_to_holiday` / `days_from_holiday`, which are the two
  features that are non-constant even on non-holiday rows

Late April to mid June is unusually holiday-dense in Western Europe (Labour Day,
Ascension, Whit Monday, and Corpus Christi in parts of DE), so the expectation is
that this window has *more* power than a random one, not less. Measured either
way.

Reads the replica read-only. No fit, no write to the replica.

ABL-393 reuses this probe for load and price, and added three things to it:

``--type``
    Any type `scripts/abl338_solar_holdout.py` can fit, so the window that goes
    into a registration is measured by the same code that then fits it. Bands
    are solar-only here exactly as they are there; every other type reports the
    unbanded numbers alone.

``holiday_affected``
    The share of rows that are a holiday, a bridge day, or within one day of a
    holiday - the widest set these four features can tell apart from an ordinary
    day, and the subset `abl338_solar_holdout.py --holiday-subsets` scores. A
    count of red days understates a December window, whose signal is a
    contiguous low-demand fortnight rather than the four dates inside it.

zero and null target screen
    ABL-109 / ABL-111 are zero-as-missing rows in `energy_load`. They did not
    touch a solar target and are directly in the way of a load one, so the count
    is measured per window rather than assumed: `db.load_energy_data` applies no
    `> 0` guard on the training path, unlike the scorecard's ABL-35 read.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl386_holiday_density_probe.py \\
        --countries AT,BE,DE,FR --holdout 2026-04-30:2026-06-12 \\
        --out reports/abl_386_holiday_density.json

    .venv\\Scripts\\python.exe scripts/abl386_holiday_density_probe.py \\
        --countries AT,BE,DE,FR --type load --holdout 2025-12-06:2026-01-18 \\
        --out reports/abl_393_holiday_density_load_winter.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.db import load_training_data  # noqa: E402
from src.features import (  # noqa: E402
    HOLIDAY_FEATURES,
    create_all_features,
    holiday_subset_masks,
)
from src.solar_features import night_mask, solar_geometry_frame  # noqa: E402

logger = logging.getLogger("energy_forecast")

#: Same set as `abl338_solar_holdout.FITTABLE_TYPES`, derived from `config` here
#: rather than imported from a sibling script: a window has to be measurable by
#: whatever the A/B can then fit, and importing across `scripts/` is the flat
#: sibling import this repo does not do.
FITTABLE_TYPES = tuple(sorted(set(config.RENEWABLE_TYPES) | {"load", "price"}))


def _bands(country_code: str, timestamps: pd.Series) -> np.ndarray:
    """Identical banding to `abl338_solar_holdout._bands` - the serving clamp's predicate."""
    elevation = solar_geometry_frame(country_code, timestamps)["sun_elevation_deg"].to_numpy()
    night = night_mask(country_code, timestamps)
    return np.where(night, "night", np.where(elevation <= 0.0, "shoulder", "daylight"))


def _summarise(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"label": label, "n_rows": 0}
    # Solar is the only type with bands; for every other one `band` is the single
    # constant label the holdout script also uses, so `daylight` is empty and the
    # two daylight fields below read 0 rather than a number nobody measured.
    daylight = frame.loc[frame["band"] == "daylight"]
    # The subset the arms are scored on, from the holdout script's own predicate
    # rather than a second copy of it.
    affected = holiday_subset_masks(frame).get("holiday_affected")
    out = {
        "label": label,
        "n_rows": int(len(frame)),
        "n_days": int(pd.to_datetime(frame["timestamp_utc"]).dt.date.nunique()),
        "n_daylight_rows": int(len(daylight)),
        "holiday_hours": int(frame["is_holiday"].sum()),
        "holiday_share_pct": round(100.0 * float(frame["is_holiday"].mean()), 3),
        "holiday_days": int(
            pd.to_datetime(frame.loc[frame["is_holiday"] == 1, "timestamp_utc"]).dt.date.nunique()
        ),
        "bridge_hours": int(frame["is_bridge_day"].sum()),
        "bridge_share_pct": round(100.0 * float(frame["is_bridge_day"].mean()), 3),
        # ABL-393: the widest subset these four features can distinguish. A count
        # of red days understates December, where the low-demand stretch between
        # Christmas and Epiphany is what `days_to_holiday` / `days_from_holiday`
        # actually mark.
        "holiday_affected_hours": int(affected.sum()) if affected is not None else 0,
        "holiday_affected_share_pct":
            round(100.0 * float(affected.mean()), 3) if affected is not None else 0.0,
        # ABL-109 / ABL-111. Measured, not assumed: nothing on the training path
        # screens a zero load out, so a zero-as-missing row is a real training
        # target unless this count is 0.
        "target_zero_rows": int((frame["target_value"] == 0).sum()),
        "target_null_rows": int(frame["target_value"].isna().sum()),
        "target_mean": round(float(frame["target_value"].mean()), 3),
        # The primary metric is daylight MAE, so daylight holiday rows are the
        # ones that can actually move it.
        "daylight_holiday_hours": int(daylight["is_holiday"].sum()) if len(daylight) else 0,
        "daylight_holiday_share_pct":
            round(100.0 * float(daylight["is_holiday"].mean()), 3) if len(daylight) else 0.0,
        # These two are non-constant on ordinary rows, so they carry signal even
        # in a window with few holidays. Their spread is what a tree can split on.
        "days_to_holiday_unique": int(frame["days_to_holiday"].nunique()),
        "days_from_holiday_unique": int(frame["days_from_holiday"].nunique()),
        "days_to_holiday_mean": round(float(frame["days_to_holiday"].mean()), 3),
        "days_from_holiday_mean": round(float(frame["days_from_holiday"].mean()), 3),
    }
    holiday_dates = sorted(
        str(d) for d in
        pd.to_datetime(frame.loc[frame["is_holiday"] == 1, "timestamp_utc"]).dt.date.unique()
    )
    out["holiday_dates"] = holiday_dates
    # A feature that is constant across the whole window cannot be split on at
    # all - that is a stronger statement than "few holidays" and worth flagging.
    out["constant_features"] = [
        c for c in HOLIDAY_FEATURES if c in frame.columns and frame[c].nunique() <= 1
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default="AT,BE,DE,FR")
    parser.add_argument("--type", dest="forecast_type", default="solar",
                        choices=list(FITTABLE_TYPES),
                        help="Forecast type to measure (default: solar). Must be a type "
                             "scripts/abl338_solar_holdout.py can fit, so the window a "
                             "registration records is the window the arms are fitted on.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--holdout", required=True, help="START:END, both YYYY-MM-DD, inclusive")
    parser.add_argument("--out", default="reports/abl_386_holiday_density.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    holdout_start, holdout_end = args.holdout.split(":")
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    forecast_type = args.forecast_type
    # ABL-331: only an individual renewable type has a choice of source table.
    # `load` and `price` read one fixed table each and take `None`.
    source = "energy_renewable" if forecast_type in config.RENEWABLE_TYPES else None

    payload = {
        "issue": "ABL-386" if forecast_type == "solar" else "ABL-393",
        "purpose": "holiday density in the registered fit and holdout windows, measured before any fit",
        "replica_db": str(config.DATABASE_PATH),
        "forecast_type": forecast_type,
        "source_table": source,
        "start_date": args.start,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "holiday_features": list(HOLIDAY_FEATURES),
        "countries": {},
    }

    for country in countries:
        raw = load_training_data(
            country, forecast_type, args.start,
            (pd.Timestamp(holdout_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            source=source,
        )
        featured = create_all_features(
            raw, forecast_type, country_code=country).reset_index(drop=True)
        timestamps = pd.to_datetime(featured["timestamp_utc"])
        featured["band"] = (_bands(country, timestamps) if forecast_type == "solar" else "all")

        is_holdout = (timestamps >= pd.Timestamp(holdout_start)) & (
            timestamps <= pd.Timestamp(holdout_end) + pd.Timedelta(hours=23)
        )
        fit_frame = featured.loc[~is_holdout.to_numpy()].reset_index(drop=True)
        holdout_frame = featured.loc[is_holdout.to_numpy()].reset_index(drop=True)

        payload["countries"][country] = {
            "fit": _summarise(fit_frame, "fit"),
            "holdout": _summarise(holdout_frame, "holdout"),
        }
        h = payload["countries"][country]["holdout"]
        logger.info(
            f"{country}/{forecast_type}: holdout {h['n_rows']} rows, {h['holiday_days']} "
            f"holiday days ({h['holiday_share_pct']}% of rows, "
            f"{h['holiday_affected_share_pct']}% holiday-affected), "
            f"{h['target_zero_rows']} zero targets, dates {h['holiday_dates']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
