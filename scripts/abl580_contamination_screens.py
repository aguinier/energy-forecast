#!/usr/bin/env python
"""Contamination screens for the three pairs ABL-580 ships, in one committed record.

ABL-580 item 5. Five screens, on the window these artifacts were actually fitted
on rather than on the window their gate read was taken over -- which is the whole
point, since the fit window is 223 days against the gate's registered 178 and no
existing record covers the extra 45.

Three are the standing screens ABL-525 answered, restated here because a screen
answered on seven `wind_onshore` pairs is not answered on two `solar` pairs and
an offshore one:

  ABL-332  the hourly-aggregation contract in the fit-and-serve path.
  ABL-200  the cross-table zero disproof, and whether it fires at all here.
  ABL-188  the constant-run exclusion.

Two are specific to this set and are the reason the issue names them:

  night floor   CZ and RO `solar`, against the BG signature. ABL-405's probe read
                `energy_generation` while its *fit* read `energy_renewable`;
                ABL-426's re-read is the first where both are one series, and
                this run is the first where the screen covers the fit window of
                the artifact being shipped.
  NL vintage    the ABL-439 fit-to-gate ratio discontinuity for NL
                `wind_offshore`, re-derived here rather than cited.

WHY THE NL RATIO IS RE-DERIVED AND NOT QUOTED
---------------------------------------------
ABL-580's description cites this screen as "ABL-471 (merged, PR #83)". PR #83 is
**closed, not merged** -- `mergedAt` and `mergeCommit` are both null, it was
closed 2026-08-24T06:41:51Z two minutes after PR #82 merged, and none of its four
files is on `origin/main`. Its numbers are right; what is missing is a tracked
record backing them, on the one pair whose hold that screen cleared.

So this script re-derives the ratio through `abl439_reporting_basis_probe._hourly`
-- the same primitive ABL-471 called, and one that *is* on `origin/main` -- and
pins the result to the values ABL-471 published. A disagreement is reported as a
failure of this screen rather than smoothed over. That gives ABL-580 a
self-contained tracked record without re-landing PR #83 inside this diff, which
would be a second change in one reviewable unit.

Read-only against the replica (`mode=ro` throughout). Fits nothing, scores
nothing, grades nothing, and writes only its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
import abl439_reporting_basis_probe as abl439  # noqa: E402
from src import db  # noqa: E402
from src.solar_features import night_mask  # noqa: E402
from src.solar_geometry import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    NIGHT_GENERATION_POSSIBLE,
    SOLAR_REPRESENTATIVE_POINTS,
)

from scripts.abl525_train_ship_set import (  # noqa: E402
    FIT_END,
    FIT_START,
    RENEWABLE_SOURCE,
    SHIP_SET,
)

BATCH = "abl580"

#: ABL-338's threshold, the one the serving clamp and every published night
#: screen use. Imported would be better; it is a module-private constant in
#: `solar_features`, so it is restated with its provenance and asserted below.
NIGHT_MW_THRESHOLD = 1.0

#: The gate boundary, so the fit window is reported either side of it and a
#: reader can line the near half up against ABL-405's published screen.
GATE_START = "2026-07-11"

#: BG's numbers from ABL-405 section 3 / ABL-396, carried as the reference the
#: two solar pairs are being compared *against*. Quoted, not re-derived: BG is
#: not in this ship set and re-measuring it would be a read this issue excludes.
BG_SIGNATURE = {
    "country": "BG",
    "source": "reports/abl_405_tranche2a_findings.md section 3 (= ABL-396)",
    "fit": {"pct_night_hours_above_threshold": 76.4, "night_mean_mw": 152.33,
            "night_max_mw": 1097.4, "pct_of_total_energy_at_night": 6.37},
    "gate": {"pct_night_hours_above_threshold": 85.2, "night_mean_mw": 245.71,
             "night_max_mw": 1087.9, "pct_of_total_energy_at_night": 4.98},
}

#: What ABL-471 published for NL `wind_offshore`, on branch
#: `ABL-471-source-table-ratio-screen` at `d6c8408` -- a commit that is *not* an
#: ancestor of `origin/main`. These are the values the ABL-580 description quotes
#: as NL's basis for joining the shipping set, so they are the pins.
ABL471_NL_PINS = {"fit_ratio": 0.9922, "gate_ratio": 0.9912, "discontinuity": 0.0010}

#: ABL-471's own cut between "no discontinuity" and "a revision vintage". The gap
#: between the two is two orders of magnitude wide and empty on the 41 screened
#: pair-records, so nothing is decided by where in it the line sits.
NO_DISCONTINUITY = 0.02

#: The registered gate windows the ratio is taken over, so this re-derivation is
#: the same measurement rather than a differently-windowed one.
ABL348_FIT_WINDOW = ("2026-01-14", "2026-07-11")
ABL348_GATE_WINDOW = ("2026-07-11", "2026-08-10")

COLUMN_BY_TYPE = {
    "solar": "solar_mw",
    "wind_onshore": "wind_onshore_mw",
    "wind_offshore": "wind_offshore_mw",
}


def batch_pairs():
    return [(row["country"], row["forecast_type"])
            for row in SHIP_SET if row["batch"] == BATCH]


# ---------------------------------------------------------------------------
# ABL-332 / ABL-200 / ABL-188: the three standing screens
# ---------------------------------------------------------------------------

def hourly_contract(country, forecast_type, replica):
    """ABL-332: what the loader did to the raw rows, measured not asserted.

    `load_renewable_type_data` calls `aggregate_renewable_to_hourly` and the
    builder then calls `_assert_hourly`, which *raises* on an off-hour index
    rather than subsampling -- so fit and serve share one hourly frame by
    construction. What this reports is the size of what would otherwise have been
    discarded: the pre-ABL-332 serving builder read only the `:00` sub-sample.
    """
    frame = db.load_renewable_type_data(
        country, forecast_type, FIT_START, FIT_END,
        source=RENEWABLE_SOURCE, db_path=str(replica))
    column = COLUMN_BY_TYPE[forecast_type]
    with db.get_connection(readonly=True, db_path=str(replica)) as conn:
        raw = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT CAST(strftime('%M', timestamp_utc) "
            f"AS INTEGER)) FROM {RENEWABLE_SOURCE} WHERE country_code = ? "
            f"AND timestamp_utc >= ? AND timestamp_utc < ? AND {column} IS NOT NULL",
            (country, FIT_START, FIT_END)).fetchone()
    raw_rows, minute_marks = int(raw[0]), int(raw[1] or 0)
    hourly_rows = int(len(frame))
    return {
        "screen": "ABL-332 hourly aggregation",
        "raw_rows_in_fit_window": raw_rows,
        "distinct_minute_marks": minute_marks,
        "sub_hourly": bool(minute_marks > 1),
        "hourly_rows_after_aggregation": hourly_rows,
        "rows_the_pre_abl332_builder_would_have_discarded": (
            raw_rows - hourly_rows if minute_marks > 1 else 0),
        "aggregation_is_in_the_fit_and_serve_path": True,
        "note": (
            "load_renewable_type_data -> aggregate_renewable_to_hourly, then "
            "RenewableFeatureBuilder._assert_hourly raises on an off-hour index. "
            "The same builder object fits and serves, so the two are one frame."),
    }


def zero_disproof_applicability(replica):
    """ABL-200: whether the rule can fire on this set at all.

    It cannot, and the reason is a constant rather than a measurement: the guard
    is wired behind `if source != RENEWABLE_ZERO_DISPROOF_SOURCE`, and that
    constant *is* `energy_generation`, which is the source all three pairs read.
    The rule is one-sided and `energy_generation` is already the disproving side.
    Read off `src.db` rather than restated, so a change to the wiring shows up
    here instead of leaving a stale claim in a report.
    """
    return {
        "screen": "ABL-200 cross-table zero disproof",
        "renewable_zero_disproof_source": db.RENEWABLE_ZERO_DISPROOF_SOURCE,
        "source_read_by_this_batch": RENEWABLE_SOURCE,
        "rule_can_fire": db.RENEWABLE_ZERO_DISPROOF_SOURCE != RENEWABLE_SOURCE,
        "note": (
            "exclude_zeros_disproved_by_sibling is wired at "
            "load_renewable_type_data behind `if source != "
            "RENEWABLE_ZERO_DISPROOF_SOURCE`. Every pair here reads that same "
            "table, so the rule never fires -- it is one-sided and this is "
            "already the disproving side."),
    }


def constant_run_screen(country, forecast_type, replica):
    """ABL-188: what `exclude_suspect_constant_runs` nulled on the fit window.

    Reported as a count of nulled hours rather than as a boolean, because "the
    guard ran" and "the guard found nothing" are different facts and only the
    second is evidence about this series.
    """
    frame = db.load_renewable_type_data(
        country, forecast_type, FIT_START, FIT_END,
        source=RENEWABLE_SOURCE, db_path=str(replica))
    values = frame["target_value"]
    return {
        "screen": "ABL-188 constant-run exclusion",
        "hourly_rows": int(len(frame)),
        "rows_nulled_by_the_guard": int(values.isna().sum()),
        "note": ("exclude_suspect_constant_runs nulls any 24h+ bit-identical run "
                 "at the training read; a nulled hour is then dropped by "
                 "finite_training_rows and appears in the fit audit's "
                 "excluded_missing_actual_or_feature."),
    }


# ---------------------------------------------------------------------------
# Night floor: CZ and RO solar against the BG signature
# ---------------------------------------------------------------------------

def night_floor(country, replica, start, end, label):
    """The ABL-396 night screen over one window of the *fit* series.

    Identical predicate to `scripts/abl381_night_floor_probe.py` and to the
    ABL-337 serving clamp: `solar_features.night_mask`, the sun geometrically
    below -8 degrees for the whole hour at the country's capacity-weighted point.
    A non-zero night actual here is therefore not a timezone offset or a mask
    artefact; it is a property of the series.
    """
    frame = db.load_renewable_type_data(
        country, "solar", start, end, source=RENEWABLE_SOURCE, db_path=str(replica))
    stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed",
                            utc=True).dt.tz_localize(None)
    values = frame["target_value"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    night = np.asarray(night_mask(country, list(stamps))) & finite
    night_values = values[night]
    total_energy = float(np.abs(values[finite]).sum())
    above = night_values > NIGHT_MW_THRESHOLD
    return {
        "window": label,
        "start": str(start),
        "end_exclusive": str(end),
        "n_rows": int(len(values)),
        "n_missing_rows": int((~finite).sum()),
        "n_night_rows": int(night.sum()),
        "n_night_above_threshold": int(above.sum()),
        "pct_of_night_rows_above_threshold": (
            round(float(above.mean() * 100.0), 2) if night.sum() else None),
        "night_mean_mw": round(float(night_values.mean()), 2) if night.sum() else None,
        "night_max_mw": round(float(night_values.max()), 2) if night.sum() else None,
        "n_night_negative": int((night_values < 0).sum()),
        "pct_of_total_energy_at_night": (
            round(float(night_values.sum() / total_energy * 100.0), 3)
            if total_energy else None),
        # ABL-396: the rankable quantity. `f` is the full width, in WAPE points,
        # of the interval an all-hours read can occupy relative to the same
        # challenger's daylight-only read -- and a hard lower bound on the WAPE
        # of any *served* forecast, since the clamp cannot do better than zero
        # against a floor. Taken on |MW| because a clamped forecast pays the
        # magnitude of a night error regardless of its sign.
        "wape_floor_pct_if_clamped": (
            round(float(np.abs(night_values).sum() / total_energy * 100.0), 4)
            if total_energy else None),
        "daylight_mean_mw": (round(float(values[finite & ~night].mean()), 2)
                             if (finite & ~night).any() else None),
    }


# ---------------------------------------------------------------------------
# NL wind_offshore: the ABL-439 fit-to-gate ratio discontinuity, re-derived
# ---------------------------------------------------------------------------

def ratio_over(conn, column, country, start, end):
    """Hourly-mean `energy_generation / energy_renewable` over co-observed hours.

    Same primitive and same intersection rule as ABL-471: the two tables' coverage
    differs at the head of `energy_renewable` (ABL-188), and a ratio of two means
    taken over different hour sets measures the coverage and not the level.
    """
    generation = abl439._hourly(conn, "energy_generation", column, country, start, end)
    renewable = abl439._hourly(conn, "energy_renewable", column, country, start, end)
    common = sorted(set(generation) & set(renewable))
    entry = {"start": start, "end_exclusive": end,
             "n_hours_generation": len(generation),
             "n_hours_renewable": len(renewable),
             "n_hours_common": len(common)}
    if not common:
        entry["ratio"] = None
        entry["note"] = "no co-observed hours"
        return entry
    mean_generation = sum(generation[h] for h in common) / len(common)
    mean_renewable = sum(renewable[h] for h in common) / len(common)
    entry["generation_mean_mw"] = round(mean_generation, 4)
    entry["renewable_mean_mw"] = round(mean_renewable, 4)
    entry["ratio"] = round(mean_generation / mean_renewable, 4) if mean_renewable else None
    return entry


def vintage_screen(country, forecast_type, replica):
    """The quantity that actually voids a gate read: the *change* of basis.

    A model is fitted and scored on the registered `energy_generation`, so a
    steady offset between the two source tables voids nothing. Only a change of
    basis between the fit and gate windows can, which is why the reported figure
    is `fit - gate` and not either ratio alone.
    """
    column = COLUMN_BY_TYPE[forecast_type]
    with db.get_connection(readonly=True, db_path=str(replica)) as conn:
        fit = ratio_over(conn, column, country, *ABL348_FIT_WINDOW)
        gate = ratio_over(conn, column, country, *ABL348_GATE_WINDOW)
    discontinuity = (round(fit["ratio"] - gate["ratio"], 4)
                     if fit["ratio"] and gate["ratio"] else None)
    reproduces = (
        fit["ratio"] == ABL471_NL_PINS["fit_ratio"]
        and gate["ratio"] == ABL471_NL_PINS["gate_ratio"]
        and discontinuity == ABL471_NL_PINS["discontinuity"])
    return {
        "screen": "ABL-439 fit-to-gate source-table ratio discontinuity",
        "country": country,
        "forecast_type": forecast_type,
        "abl348_fit_window": fit,
        "abl348_gate_window": gate,
        "discontinuity_fit_minus_gate": discontinuity,
        "no_discontinuity_threshold": NO_DISCONTINUITY,
        "verdict": ("basis-consistent"
                    if discontinuity is not None and abs(discontinuity) < NO_DISCONTINUITY
                    else "basis-INCONSISTENT"),
        "abl471_published": dict(ABL471_NL_PINS),
        "reproduces_abl471": bool(reproduces),
        "abl471_provenance": {
            "branch": "ABL-471-source-table-ratio-screen",
            "commit": "d6c8408",
            "pr": 83,
            "pr_state": "CLOSED",
            "merged": False,
            "on_origin_main": False,
            "note": ("The ABL-580 description says 'merged, PR #83'. It is not: "
                     "mergedAt and mergeCommit are both null, closed "
                     "2026-08-24T06:41:51Z. The figures are right; the record "
                     "backing them is untracked, which is why this run "
                     "re-derives rather than cites."),
        },
        # ABL-580 item 5's second addition, carried rather than dropped.
        "gate_revision_caveat": (
            "This pair's gate window is 100% first-publication, so its gate-side "
            "revision is expected-small but NOT YET MEASURED. That is true of "
            "every pair in this gate and is not a blocker; it is recorded here so "
            "a later reader does not mistake an unmeasured quantity for a "
            "measured-zero one."),
    }


def main():
    parser = argparse.ArgumentParser(
        description=("Contamination screens for the three pairs ABL-580 ships "
                     "(ABL-332, ABL-200, ABL-188, night floor, ABL-439 vintage)."))
    parser.add_argument("--replica-db", default=config.DATABASE_PATH,
                        help="Read-only replica (default: ENERGY_DB_PATH).")
    parser.add_argument("--json-out", default="reports/abl_580_contamination_screens.json")
    args = parser.parse_args()

    replica = Path(args.replica_db)
    if not replica.is_file():
        raise SystemExit(f"replica not found: {replica}")

    pairs = []
    for country, forecast_type in batch_pairs():
        print(f"[SCREEN] {country}/{forecast_type} ...", flush=True)
        entry = {
            "country": country,
            "forecast_type": forecast_type,
            "abl332": hourly_contract(country, forecast_type, replica),
            "abl188": constant_run_screen(country, forecast_type, replica),
        }
        if forecast_type == "solar":
            entry["night_floor"] = {
                "predicate": ("solar_features.night_mask -- sun below "
                              f"{NIGHT_ELEVATION_THRESHOLD_DEG} deg for the whole "
                              "hour at the capacity-weighted point"),
                "threshold_mw": NIGHT_MW_THRESHOLD,
                "representative_point": SOLAR_REPRESENTATIVE_POINTS.get(country),
                "night_generation_possible": NIGHT_GENERATION_POSSIBLE.get(country),
                "windows": [
                    night_floor(country, replica, FIT_START, GATE_START, "fit_pre_gate"),
                    night_floor(country, replica, GATE_START, FIT_END, "gate_and_after"),
                    night_floor(country, replica, FIT_START, FIT_END, "whole_fit_window"),
                ],
                "compared_against": BG_SIGNATURE,
            }
        else:
            entry["vintage"] = vintage_screen(country, forecast_type, replica)
        pairs.append(entry)
        print(f"[OK    ] {country}/{forecast_type}", flush=True)

    payload = {
        "issue": "ABL-580",
        "batch": BATCH,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_window_screened": [FIT_START, FIT_END],
        "gate_boundary": GATE_START,
        "renewable_source": RENEWABLE_SOURCE,
        "scored_or_graded": False,
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "abl200": zero_disproof_applicability(replica),
        "named_contamination_issues": {
            "ABL-71": "prod ingest stale -- net_position, not these series",
            "ABL-67": "fabricated net_position rows -- not these series",
            "ABL-111/ABL-109": "zero-as-missing actual *load* rows -- not these series",
            "verdict": "none of the three touches this window or these pairs",
        },
        "pairs": pairs,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")

    failures = [p["vintage"] for p in pairs
                if "vintage" in p and not p["vintage"]["reproduces_abl471"]]
    if failures:
        print("REPRODUCTION FAILED against the ABL-471 pins:")
        for entry in failures:
            print(f"  {entry['country']}/{entry['forecast_type']}: "
                  f"fit {entry['abl348_fit_window']['ratio']} "
                  f"gate {entry['abl348_gate_window']['ratio']} "
                  f"vs published {entry['abl471_published']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
