"""ABL-385: the pre-registration probe. Reads only - it never fits a model.

ABL-375 registered a solar A/B and had to return AMBIGUOUS, because DE CatBoost's
daylight MAE moved 13.79% of its own mean across three seeds while the
cross-algorithm gap it was reading was 4.5%. The noise floor everything upstream
of it quoted (~1.5%, ABL-338) was a perturbation estimate on a different
question. This issue replaces the remembered floor with a measured one and turns
it into a **registered minimum decision margin**.

That registration has to be frozen before the first fit, per the ABL-322 /
ABL-375 standard, and a registration is only honest if its scope is feasible. So
this probe establishes, from reads alone:

1. **Which pairs are actually served**, and with which algorithm, source table
   and artifact version. The sweep scope is derived from the artifacts on disk,
   not from a remembered list.
2. **Whether `--force-algorithm` reconstructs the serving configuration.**
   `abl338_solar_holdout.py` drops the incumbent's hyperparameters when the
   algorithm is forced, falling back to `config.get_default_params()`. ABL-375
   measured that all four *solar* artifacts are field-identical to those
   defaults, which is what let it call the CatBoost arm "the serving
   configuration refitted". That claim has never been checked for wind, and this
   issue extends the scope to wind - so it is checked here, per pair, before any
   arm is registered as standing in for a serving model.
3. **Whether every registered (pair, window) cell can actually be scored.** DE
   solar's `energy_renewable` history begins 2025-09-08, which is why its fit
   window is the shortest of the four solar countries and - on ABL-375's reading
   - why it may be underdetermined. A rolling-origin design that silently
   produces an empty or tiny fit frame for the earliest windows would answer the
   variance question with the sample-size question mixed in.
4. **Which contamination touches each fit window**, counted rather than recalled.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl385_variance_probe.py \\
        --out reports/abl_385_probe.json

`ENERGY_DB_PATH` must be passed explicitly from a worktree - `.env` is gitignored
and `config.DATABASE_PATH` otherwise degrades to a bare `\\data\\energy_dashboard.db`.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.db import load_training_data  # noqa: E402
from src.features import create_all_features, get_feature_columns  # noqa: E402
from src.forecaster import Forecaster  # noqa: E402
from src.solar_features import SOLAR_GEOMETRY_FEATURES, night_mask  # noqa: E402

logger = logging.getLogger("energy_forecast")

#: The live artifacts, in the primary checkout the scheduled job serves from.
#: `models/` is gitignored, so a worktree has none of its own.
LIVE_MODELS_DIR = Path(r"C:\Code\able\energy-forecast\models")

#: The candidate scope: every served pair whose target is an individual
#: renewable type. Aggregates (`load`, `price`, `renewable`, `price_cascade`) are
#: out of scope - this issue is about the renewable gate reads, and the solar
#: harness's band logic only means anything for a generation target.
#:
#: Enumerated here rather than globbed so the registration names a fixed list
#: that review can check, and so a pair appearing on disk tomorrow cannot widen a
#: frozen scope. `--discover` re-derives it from the filesystem to prove the list
#: is complete at probe time.
CANDIDATE_PAIRS = [
    ("AT", "solar"), ("BE", "solar"), ("DE", "solar"), ("FR", "solar"),
    ("AT", "wind_onshore"), ("BE", "wind_onshore"),
    ("DE", "wind_onshore"), ("FR", "wind_onshore"),
    ("BE", "wind_offshore"), ("FR", "wind_offshore"),
    ("BE", "biomass"), ("FR", "biomass"),
    ("BE", "hydro_total"), ("FR", "hydro_total"),
]

#: The rolling-origin holdout blocks. Six consecutive 30-day windows ending at
#: the replica's last full day, so window variance is measured across a real
#: seasonal sweep (mid-February to mid-August) rather than at one point in it.
#:
#: They are contiguous and non-overlapping, so no holdout row is scored twice and
#: each window's fit frame is every featured row strictly before it - an
#: *expanding* fit, which is what a retrain actually gets. That deliberately
#: confounds fit length with season across windows; the `FIT_LENGTH_ABLATION`
#: below is what separates them, by moving the fit start with the window held
#: fixed.
WINDOWS = [
    ("W1", "2026-02-13", "2026-03-14"),
    ("W2", "2026-03-15", "2026-04-13"),
    ("W3", "2026-04-14", "2026-05-13"),
    ("W4", "2026-05-14", "2026-06-12"),
    ("W5", "2026-06-13", "2026-07-12"),
    ("W6", "2026-07-13", "2026-08-11"),
]

#: Fit-start dates for the season-controlled fit-length ablation, all scored on
#: W6. ABL-375's two competing explanations for DE - "DE prefers XGBoost" versus
#: "DE CatBoost is unstable on a 156-day fit" - differ in what happens to the
#: spread when the fit gets longer, and nothing else here can tell them apart.
FIT_LENGTH_ABLATION = ["2023-01-01", "2025-11-01", "2026-01-01", "2026-03-01"]

#: Disclosed before the registration is frozen: one timing calibration was run
#: before this probe, to size the sweep. It fitted DE solar / CatBoost / geometry
#: at seeds 42 and 1337 on 2026-06-13..2026-08-11 - a cell ABL-375 has already
#: published at those seeds (report section 3, `holdout_noisefloor_summer_catboost_
#: cleaned.json`), so it revealed no number that was not already committed. Seed
#: 42 reproduced ABL-375's published maximum, 3,694.8 MW daylight MAE, exactly.
TIMING_CALIBRATION = {
    "why": "size the registered sweep; a design has to be affordable to be frozen",
    "cell": "DE/solar/catboost/geometry, holdout 2026-06-13..2026-08-11, seeds 42,1337",
    "already_published_by": "ABL-375 report section 3 (summer, post-hoc seeded, 3 seeds)",
    "new_information": False,
    "reproduced": "seed 42 daylight MAE 3694.8 MW == ABL-375's published summer maximum",
    "measured_wall_clock_s": 6.5,
}


def _discover_pairs() -> list:
    """Every (country, individual renewable type) with an artifact on disk."""
    found = []
    if not LIVE_MODELS_DIR.exists():
        return found
    for country_dir in sorted(LIVE_MODELS_DIR.iterdir()):
        if not country_dir.is_dir() or len(country_dir.name) != 2:
            continue
        for type_dir in sorted(country_dir.iterdir()):
            if type_dir.is_dir() and (type_dir / "model.joblib").exists():
                if type_dir.name in config.RENEWABLE_TYPES:
                    found.append((country_dir.name, type_dir.name))
    return found


def _hyperparams_vs_defaults(incumbent: Forecaster) -> dict:
    """Does this artifact carry its algorithm's `config` defaults, field for field?

    `--force-algorithm` on the holdout harness passes `hyperparams=None`, so a
    forced arm is fitted at `config.get_default_params(algorithm)`. That arm can
    only be described as "the serving configuration refitted" where the answer
    here is `identical`. Anywhere it is not, the forced arm is a *stand-in*, and
    the report has to say so rather than inherit ABL-375's solar-only finding.
    """
    defaults = config.get_default_params(incumbent.algorithm)
    live = dict(incumbent.hyperparams or {})
    differing = {
        key: {"artifact": live.get(key, "<absent>"), "config_default": value}
        for key, value in defaults.items()
        if key not in live or live[key] != value
    }
    extra = {k: v for k, v in live.items() if k not in defaults}
    return {
        "identical": not differing and not extra,
        "differing_fields": differing,
        "fields_only_on_artifact": extra,
    }


def probe_pair(country: str, forecast_type: str) -> dict:
    """Everything the registration needs about one pair, without fitting."""
    live_path = LIVE_MODELS_DIR / country / forecast_type / "model.joblib"
    out = {"country_code": country, "forecast_type": forecast_type,
           "artifact_path": str(live_path)}
    if not live_path.exists():
        out["served"] = False
        return out
    out["served"] = True

    incumbent = Forecaster.load(country, forecast_type, path=str(live_path))
    source = incumbent.training_source
    out.update({
        "algorithm": incumbent.algorithm,
        "training_source": source,
        "incumbent_version": incumbent.model_version,
        "n_feature_columns_on_artifact": len(incumbent.feature_columns or []),
        "hyperparams": _hyperparams_vs_defaults(incumbent),
    })

    # The whole span the sweep can draw on, read once and reused for every
    # window. `load_training_data` applies the ABL-188 constant-run screen and
    # the ABL-332 hourly aggregation, so this is the frame a fit actually sees.
    raw = load_training_data(country, forecast_type, "2023-01-01",
                             "2026-08-12", source=source)
    if raw.empty:
        out["data"] = {"n_rows": 0, "note": f"no rows in {source}"}
        return out

    timestamps = pd.to_datetime(raw["timestamp_utc"])
    out["data"] = {
        "n_rows": int(len(raw)),
        "first": str(timestamps.min()),
        "last": str(timestamps.max()),
        "span_days": int((timestamps.max() - timestamps.min()).days),
    }

    featured = create_all_features(raw, forecast_type, country_code=country)
    featured = featured.reset_index(drop=True)
    ftimes = pd.to_datetime(featured["timestamp_utc"])
    expected = get_feature_columns(forecast_type)
    if forecast_type != "solar":
        expected = [c for c in expected if c not in SOLAR_GEOMETRY_FEATURES]
    out["features"] = {
        "n_rows_after_featuring": int(len(featured)),
        "n_expected_columns": len(expected),
        "n_present": int(sum(c in featured.columns for c in expected)),
        "missing": [c for c in expected if c not in featured.columns],
    }

    # Per-window feasibility. A cell is only registrable if both frames exist:
    # an expanding fit that starts empty is a sample-size finding, not a
    # variance one.
    windows = []
    for name, start, end in WINDOWS:
        in_holdout = (ftimes >= pd.Timestamp(start)) & (
            ftimes <= pd.Timestamp(end) + pd.Timedelta(hours=23)
        )
        # The harness bounds its own read at `holdout_end + 1 day` and then
        # trains on the complement, so its fit frame is exactly the rows
        # strictly before the holdout start - ABL-375 records DE's as ending
        # 2026-04-29 for a 04-30 holdout, which is that property observed. This
        # probe reads the whole span at once, so it has to apply the bound
        # itself or it would count future rows the sweep will never see.
        is_fit = (ftimes < pd.Timestamp(start)).to_numpy()
        fit_times = ftimes.loc[is_fit]
        entry = {
            "window": name, "start": start, "end": end,
            "n_holdout": int(in_holdout.sum()),
            "n_train_strictly_before": int(len(fit_times)),
            "fit_span_days": int((pd.Timestamp(start) - fit_times.min()).days)
            if len(fit_times) else 0,
        }
        if forecast_type == "solar" and int(in_holdout.sum()) > 0:
            htimes = ftimes.loc[in_holdout.to_numpy()]
            entry["n_holdout_daylight_or_shoulder"] = int((~night_mask(country, htimes)).sum())
            # ABL-337: physically impossible night actuals, counted in the fit
            # frame the sweep would drop them from.
            if len(fit_times):
                fnight = night_mask(country, fit_times)
                entry["abl337_impossible_night_rows_in_fit"] = int(
                    (fnight & (featured.loc[is_fit, "target_value"].to_numpy() > 1.0)).sum()
                )
        windows.append(entry)
    out["windows"] = windows

    # The season-controlled ablation, W6 only.
    ablation = []
    w6_start = pd.Timestamp(WINDOWS[-1][1])
    for fit_start in FIT_LENGTH_ABLATION:
        n = int(((ftimes >= pd.Timestamp(fit_start)) & (ftimes < w6_start)).sum())
        ablation.append({
            "fit_start": fit_start,
            "n_train": n,
            "fit_span_days": int((w6_start - max(pd.Timestamp(fit_start),
                                                 ftimes.min())).days),
        })
    out["fit_length_ablation_W6"] = ablation
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/abl_385_probe.json")
    parser.add_argument("--discover", action="store_true",
                        help="Also re-derive the served pair list from the filesystem "
                             "and record any disagreement with CANDIDATE_PAIRS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)

    db_path = Path(config.DATABASE_PATH)
    payload = {
        "issue": "ABL-385",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "replica_db": str(db_path),
        "replica_db_bytes": db_path.stat().st_size if db_path.exists() else None,
        "live_models_dir": str(LIVE_MODELS_DIR),
        "windows": [{"window": n, "start": s, "end": e} for n, s, e in WINDOWS],
        "fit_length_ablation_starts": FIT_LENGTH_ABLATION,
        "timing_calibration_before_registration": TIMING_CALIBRATION,
        "pairs": {},
    }
    if args.discover:
        discovered = _discover_pairs()
        payload["discovered_pairs"] = [f"{c}/{t}" for c, t in discovered]
        payload["candidate_not_on_disk"] = [
            f"{c}/{t}" for c, t in CANDIDATE_PAIRS if (c, t) not in discovered
        ]
        payload["on_disk_not_candidate"] = [
            f"{c}/{t}" for c, t in discovered if (c, t) not in CANDIDATE_PAIRS
        ]

    for country, forecast_type in CANDIDATE_PAIRS:
        key = f"{country}/{forecast_type}"
        print(f"probing {key} ...", flush=True)
        try:
            payload["pairs"][key] = probe_pair(country, forecast_type)
        except Exception as exc:  # a pair that cannot be probed must not be registered
            payload["pairs"][key] = {"country_code": country,
                                     "forecast_type": forecast_type,
                                     "error": f"{type(exc).__name__}: {exc}"}
            print(f"  !! {type(exc).__name__}: {exc}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\npair                 served algo      source            rows  span_d  "
          "hp==defaults  W1_train  W6_train")
    for key, entry in payload["pairs"].items():
        if entry.get("error"):
            print(f"{key:20s} ERROR {entry['error'][:60]}")
            continue
        data = entry.get("data", {})
        wins = {w["window"]: w for w in entry.get("windows", [])}
        print(
            f"{key:20s} {str(entry.get('served')):6s} {entry.get('algorithm', '-'):9s} "
            f"{entry.get('training_source', '-'):17s} {data.get('n_rows', 0):6,d} "
            f"{data.get('span_days', 0):6d}  "
            f"{str(entry.get('hyperparams', {}).get('identical')):12s}  "
            f"{wins.get('W1', {}).get('n_train_strictly_before', 0):8,d}  "
            f"{wins.get('W6', {}).get('n_train_strictly_before', 0):8,d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
