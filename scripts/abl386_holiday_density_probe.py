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

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl386_holiday_density_probe.py \\
        --countries AT,BE,DE,FR --holdout 2026-04-30:2026-06-12 \\
        --out reports/abl_386_holiday_density.json
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
from src.features import HOLIDAY_FEATURES, create_all_features  # noqa: E402
from src.solar_features import night_mask, solar_geometry_frame  # noqa: E402

logger = logging.getLogger("energy_forecast")


def _bands(country_code: str, timestamps: pd.Series) -> np.ndarray:
    """Identical banding to `abl338_solar_holdout._bands` - the serving clamp's predicate."""
    elevation = solar_geometry_frame(country_code, timestamps)["sun_elevation_deg"].to_numpy()
    night = night_mask(country_code, timestamps)
    return np.where(night, "night", np.where(elevation <= 0.0, "shoulder", "daylight"))


def _summarise(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"label": label, "n_rows": 0}
    daylight = frame.loc[frame["band"] == "daylight"]
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
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--holdout", required=True, help="START:END, both YYYY-MM-DD, inclusive")
    parser.add_argument("--out", default="reports/abl_386_holiday_density.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    holdout_start, holdout_end = args.holdout.split(":")
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    payload = {
        "issue": "ABL-386",
        "purpose": "holiday density in the registered fit and holdout windows, measured before any fit",
        "replica_db": str(config.DATABASE_PATH),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "holiday_features": list(HOLIDAY_FEATURES),
        "countries": {},
    }

    for country in countries:
        raw = load_training_data(
            country, "solar", args.start,
            (pd.Timestamp(holdout_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            source="energy_renewable",
        )
        featured = create_all_features(raw, "solar", country_code=country).reset_index(drop=True)
        timestamps = pd.to_datetime(featured["timestamp_utc"])
        featured["band"] = _bands(country, timestamps)

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
            f"{country}: holdout {h['n_rows']} rows, {h['holiday_days']} holiday days "
            f"({h['holiday_share_pct']}% of rows, {h['daylight_holiday_share_pct']}% of daylight), "
            f"dates {h['holiday_dates']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
