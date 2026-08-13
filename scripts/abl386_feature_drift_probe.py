"""ABL-386: which serving artifacts disagree with `get_feature_columns()`, and how.

ABL-375 found, as a side effect, that `get_feature_columns('solar')` returns 31
names while every serving solar artifact carries 25. Two of the six extras are
ABL-338's geometry pair, which was added deliberately and is documented. The
other four - `is_holiday`, `days_to_holiday`, `days_from_holiday`,
`is_bridge_day` - are undocumented drift, and nobody has read them on a solar
target.

Existing artifacts are safe: `Forecaster.load` reads `feature_columns` off the
artifact, so a serving model keeps its own list forever. The exposure is the
*next fit*. Whatever `get_feature_columns()` returns on the day someone retrains
a country is what that country starts serving, evaluated or not.

This probe measures the gap rather than assuming it, for **every** (country,
forecast type) artifact under the live models directory - not just solar. Scope
item 4 of ABL-386 asks whether the same drift exists on the other renewable
types or only on solar, and that is a question about all eight lists at once.

It reads artifacts and never writes one. No fit happens here.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl386_feature_drift_probe.py \\
        --out reports/abl_386_feature_drift.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.features import HOLIDAY_FEATURES, get_feature_columns  # noqa: E402
from src.solar_features import SOLAR_GEOMETRY_FEATURES  # noqa: E402

#: Same constant the ABL-338/375 holdout script uses: `models/` is gitignored, so
#: a worktree has none and the live artifacts live in the primary checkout.
LIVE_MODELS_DIR = Path(r"C:\Code\able\energy-forecast\models")


def _artifact_paths():
    """Every serving `model.joblib` under the live models directory.

    Only the top-level `model.joblib` per (country, type) is a serving artifact.
    The `candidate/`, `previous/`, `centroid/` and `multipoint/` subdirectories
    are experiment output and shadow rails, and are counted but not compared -
    reading them as serving would overstate the blast radius.
    """
    for country_dir in sorted(LIVE_MODELS_DIR.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        # `chronos2`, `net_position` and `tso_correction` are runner directories,
        # not country codes, and carry no per-type joblib in this layout.
        if len(country) != 2 or not country.isalpha():
            continue
        for type_dir in sorted(country_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            path = type_dir / "model.joblib"
            if path.exists():
                yield country, type_dir.name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/abl_386_feature_drift.json")
    args = parser.parse_args()

    # The expected list per forecast type, as this repo would build it *today*.
    # `include_holidays=False` is what the exclusion recommendation would use, so
    # it is measured here rather than assumed to equal the serving list.
    expected = {}
    for ftype in sorted({t for _, t, _ in _artifact_paths()}):
        with_holidays = get_feature_columns(ftype)
        without = get_feature_columns(ftype, include_holidays=False)
        expected[ftype] = {
            "get_feature_columns": with_holidays,
            "n": len(with_holidays),
            "get_feature_columns_no_holidays": without,
            "n_no_holidays": len(without),
            "holiday_names_present": [c for c in HOLIDAY_FEATURES if c in with_holidays],
            "geometry_names_present": [c for c in SOLAR_GEOMETRY_FEATURES if c in with_holidays],
        }

    artifacts = []
    for country, ftype, path in _artifact_paths():
        try:
            blob = joblib.load(path)
        except Exception as exc:  # a corrupt or unreadable artifact is a finding
            artifacts.append({"country": country, "forecast_type": ftype,
                              "path": str(path), "error": repr(exc)})
            continue
        served = list(blob.get("feature_columns") or [])
        exp = expected[ftype]["get_feature_columns"]
        missing = [c for c in exp if c not in served]      # next fit would ADD these
        extra = [c for c in served if c not in exp]        # next fit would DROP these
        artifacts.append({
            # An artifact with no feature list at all is not feature drift, and
            # folding it into the drift groups would report a 26-name "gap" that
            # is really "this model class does not use this list". BE
            # price_cascade is the one such artifact; it is reported separately.
            "has_feature_list": bool(served),
            "country": country,
            "forecast_type": ftype,
            "algorithm": blob.get("algorithm"),
            "model_version": blob.get("model_version"),
            "saved_at": blob.get("saved_at"),
            "n_served": len(served),
            "n_expected": len(exp),
            "served_feature_columns": served,
            "missing_from_artifact": missing,
            "extra_in_artifact": extra,
            "drifted": bool(missing or extra),
            "holiday_in_artifact": [c for c in HOLIDAY_FEATURES if c in served],
            "geometry_in_artifact": [c for c in SOLAR_GEOMETRY_FEATURES if c in served],
            # Order matters to nothing at fit time (the frame is selected by name)
            # but an order difference alongside a set match is worth seeing.
            "order_matches_expected_on_intersection":
                [c for c in served if c in exp] == [c for c in exp if c in served],
        })

    # Group the drift by (type, what is missing, what is extra) so the report says
    # "these 22 artifacts share one gap" instead of listing 190 rows.
    groups = defaultdict(list)
    for a in artifacts:
        if a.get("error") or not a.get("has_feature_list"):
            continue
        key = (a["forecast_type"], tuple(a["missing_from_artifact"]), tuple(a["extra_in_artifact"]))
        groups[key].append(f"{a['country']}:{a['algorithm']}:{a['model_version']}")

    drift_groups = [
        {
            "forecast_type": ftype,
            "missing_from_artifact": list(missing),
            "extra_in_artifact": list(extra),
            "n_artifacts": len(members),
            "artifacts": sorted(members),
            "is_holiday_only_gap": sorted(missing) == sorted(HOLIDAY_FEATURES) and not extra,
            "is_holiday_plus_geometry_gap":
                sorted(missing) == sorted(tuple(HOLIDAY_FEATURES) + tuple(SOLAR_GEOMETRY_FEATURES))
                and not extra,
        }
        for (ftype, missing, extra), members in sorted(groups.items())
    ]

    solar = [a for a in artifacts if a["forecast_type"] == "solar" and not a.get("error")]
    exp_solar = expected.get("solar", {})
    # The claim this issue rests on, stated as a checkable predicate rather than
    # as prose: is the serving solar list exactly today's list minus holidays
    # minus geometry? If yes, an `include_holidays=False` solar list plus the
    # geometry pair reproduces the serving feature set exactly.
    reconstructed = [c for c in exp_solar.get("get_feature_columns_no_holidays", [])
                     if c not in SOLAR_GEOMETRY_FEATURES]
    solar_check = {
        "expected_n": exp_solar.get("n"),
        "serving_n": sorted({a["n_served"] for a in solar}),
        "all_serving_solar_agree_on_the_same_25":
            len({tuple(a["served_feature_columns"]) for a in solar}) == 1 if solar else None,
        "reconstructed_from_no_holidays_minus_geometry": reconstructed,
        "reconstruction_n": len(reconstructed),
        "reconstruction_equals_every_serving_artifact":
            all(a["served_feature_columns"] == reconstructed for a in solar) if solar else None,
        "artifacts": [f"{a['country']}:{a['algorithm']}:{a['model_version']}:{a['n_served']}"
                      for a in solar],
    }

    payload = {
        "issue": "ABL-386",
        "purpose": "measure get_feature_columns() vs every serving artifact's feature_columns",
        "live_models_dir": str(LIVE_MODELS_DIR),
        "holiday_features": list(HOLIDAY_FEATURES),
        "geometry_features": list(SOLAR_GEOMETRY_FEATURES),
        "expected_by_type": expected,
        "n_artifacts": len(artifacts),
        "n_with_feature_list": sum(1 for a in artifacts if a.get("has_feature_list")),
        "n_drifted": sum(1 for a in artifacts if a.get("drifted") and a.get("has_feature_list")),
        "n_missing_all_four_holidays": sum(
            1 for a in artifacts
            if a.get("has_feature_list")
            and all(c in a["missing_from_artifact"] for c in HOLIDAY_FEATURES)
        ),
        "artifacts_without_a_feature_list": [
            f"{a['country']}:{a['forecast_type']}:{a['algorithm']}:{a['model_version']}"
            for a in artifacts if not a.get("error") and not a.get("has_feature_list")
        ],
        "drift_groups": drift_groups,
        "solar_reconstruction_check": solar_check,
        "artifacts": artifacts,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(artifacts)} artifacts, {payload['n_with_feature_list']} with a feature list, "
          f"{payload['n_drifted']} drifted from get_feature_columns(), "
          f"{payload['n_missing_all_four_holidays']} missing all four holiday features")
    if payload["artifacts_without_a_feature_list"]:
        print(f"  no feature list (not drift): {payload['artifacts_without_a_feature_list']}")
    for g in drift_groups:
        tag = ""
        if g["is_holiday_only_gap"]:
            tag = "  <- HOLIDAY-ONLY GAP"
        elif g["is_holiday_plus_geometry_gap"]:
            tag = "  <- HOLIDAY + GEOMETRY GAP"
        print(f"  {g['forecast_type']:<14} n={g['n_artifacts']:<3} "
              f"missing={g['missing_from_artifact']} extra={g['extra_in_artifact']}{tag}")
    print(f"\nsolar reconstruction exact: {solar_check['reconstruction_equals_every_serving_artifact']}"
          f" (n={solar_check['reconstruction_n']})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
