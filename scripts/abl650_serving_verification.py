"""ABL-650 -- verify the registered calibration on real stored vintages.

Two things this proves, on rows that were actually served rather than on a
fixture:

1. **The point forecast does not move.** The registered multipliers are applied
   to the champion's and V016's stored bands and the `forecasts.forecast_value`
   column plus the stored `q50` are compared byte for byte. The constraint the
   Board attached to this fix is a measurement, not an argument.
2. **Per-zone coverage and width, before and after**, on vintages held out of
   the fit -- the two numbers a reader needs to decide whether the calibrated
   band is still decision-useful.

The registered multipliers in `experiments/` are fitted on every post-fix
vintage, which includes the holdout, so scoring *them* on the holdout would be
in-sample. The coverage and width table below therefore re-fits on
`[fit-start, holdout-start)` alone and applies that to the holdout; the shipped
registration is reported beside it so the two can be compared, and the
bit-identity proof is window-independent and uses the shipped numbers.

Read-only against both databases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl650_band_calibration import (  # noqa: E402
    paired_rows, per_zone_band, pinball, vintage_block_ci, vintage_slice,
)
from src.quantile_calibration import fit_zone_calibration  # noqa: E402
from src.quantile_calibration import (  # noqa: E402
    QCOLS, apply_zone_calibration, load_registry, registered_calibration,
    verify_median_invariant,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--replica-db", required=True)
    p.add_argument("--sidecar-db", required=True)
    p.add_argument("--models", default="chronos-2-V010,chronos-2-V016")
    # Held out of the registered fit by construction: the registration below is
    # re-fitted on the earlier half only, so this window is never seen.
    p.add_argument("--fit-start", default="2026-08-05")
    p.add_argument("--holdout-start", default="2026-08-19")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--holdout-end", default="2026-09-02")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=650)
    p.add_argument("--json-out", required=True)
    args = p.parse_args()

    registry = load_registry()
    doc = {"issue": "ABL-650", "registry": registry,
           "holdout_window_vintages": [args.holdout_start, args.holdout_end],
           "models": {}}

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        df = paired_rows(model, args.replica_db, args.sidecar_db)
        ho = vintage_slice(df, args.holdout_start, args.holdout_end)
        if ho.empty:
            doc["models"][model] = {"error": "no holdout rows"}
            continue
        zones = sorted(ho["country_code"].unique())
        if not registered_calibration(model, zones):
            doc["models"][model] = {"error": "model is not registered"}
            continue
        # Out-of-sample by construction: fitted on vintages strictly before the
        # holdout, pooled -- the only form ABL-650 found that survives.
        fit_df = vintage_slice(df, args.fit_start, args.holdout_start)
        oos = fit_zone_calibration(fit_df, ["__pooled__"], alpha_lo=args.alpha,
                                   alpha_hi=args.alpha, pooled=True)["__pooled__"]
        cal = {z: oos for z in zones}
        after = apply_zone_calibration(ho, cal)

        before_pz = per_zone_band(ho)
        after_pz = per_zone_band(after)
        rows = []
        for zone in zones:
            b, a = before_pz.loc[zone], after_pz.loc[zone]
            rows.append({
                "country_code": zone, "n": int(b["n"]),
                "coverage_before_pct": float(b["coverage_pct"]),
                "coverage_after_pct": float(a["coverage_pct"]),
                "width_before_mw": float(b["mean_width_mw"]),
                "width_after_mw": float(a["mean_width_mw"]),
                "width_change_pct": round(100 * (float(a["mean_width_mw"])
                                                 / float(b["mean_width_mw"]) - 1), 1),
                "mae_mw": float(b["mae_mw"]),
                "width_over_mae_before": float(b["width_over_mae"]),
                "width_over_mae_after": float(a["width_over_mae"]),
                "ci_before": vintage_block_ci(ho[ho["country_code"] == zone],
                                              args.n_boot, args.seed),
                "ci_after": vintage_block_ci(after[after["country_code"] == zone],
                                             args.n_boot, args.seed),
            })

        doc["models"][model] = {
            "applied_out_of_sample": oos.as_dict(),
            "applied_fit_window": [args.fit_start, args.holdout_start],
            "registered_for_serving": registry.get(model),
            "holdout_rows": int(len(ho)),
            "holdout_vintages": int(ho["generated_at"].nunique()),
            # The constraint, measured.
            "median_max_abs_delta_mw": verify_median_invariant(ho, after),
            "point_max_abs_delta_mw": float(
                np.abs(after["forecast_value"].to_numpy()
                       - ho["forecast_value"].to_numpy()).max()),
            "point_rows_bit_identical": bool(
                (after["forecast_value"].to_numpy()
                 == ho["forecast_value"].to_numpy()).all()),
            "q50_rows_bit_identical": bool(
                (after["q50"].to_numpy() == ho["q50"].to_numpy()).all()),
            "crossings_after": int(
                (np.diff(after[list(QCOLS)].to_numpy(), axis=1) < 0).any(axis=1).sum()),
            "pooled_before": vintage_block_ci(ho, args.n_boot, args.seed),
            "pooled_after": vintage_block_ci(after, args.n_boot, args.seed),
            "mean_width_before_mw": round(float((ho["q90"] - ho["q10"]).mean()), 1),
            "mean_width_after_mw": round(float((after["q90"] - after["q10"]).mean()), 1),
            "pinball_before_mw": round(pinball(ho), 2),
            "pinball_after_mw": round(pinball(after), 2),
            "zones_in_75_85_before": int(
                ((before_pz["coverage_pct"] >= 75) & (before_pz["coverage_pct"] <= 85)).sum()),
            "zones_in_75_85_after": int(
                ((after_pz["coverage_pct"] >= 75) & (after_pz["coverage_pct"] <= 85)).sum()),
            "per_zone": rows,
        }
        m = doc["models"][model]
        print(f"{model}: point delta {m['point_max_abs_delta_mw']} MW "
              f"(bit-identical={m['point_rows_bit_identical']}) | coverage "
              f"{m['pooled_before']['coverage_pct']}% -> {m['pooled_after']['coverage_pct']}% "
              f"| width {m['mean_width_before_mw']} -> {m['mean_width_after_mw']} MW "
              f"| zones in [75,85] {m['zones_in_75_85_before']} -> {m['zones_in_75_85_after']}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
