#!/usr/bin/env python3
"""ABL-651 -- measure and fit the intercept-only static bias correction.

Three questions, kept apart because they have different answers:

1. **Currency.** Does the DE / FI / PL bias the ABL-595 read measured on
   2026-08-28 still hold, and does any zone that was not on that list now
   qualify? Measured over the whole post-fix window, split into halves.
2. **Generalisation.** Fit the intercepts on the first half only, freeze them,
   and score the held-out second half. The fit window ends two target days
   before the evaluation window starts -- that gap is the D+2 horizon, and it is
   what makes the split serve-faithful rather than merely disjoint.
3. **Invariance.** Verify numerically that per-zone slope, corr and sd_ratio are
   unchanged by the correction. They are, by construction; measuring it is the
   point, because it is the property the Board relied on.

Read-only against both databases unless --write-sidecar is passed, which writes
the corrected series to the sidecar under its own model_name and never touches
the champion's rows or the replica.

Usage:
    python scripts/abl651_static_bias.py --replica-db ... --sidecar-db ... \
        --json-out reports/abl_651_static_bias.json
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

from src.evaluation.net_position import (EvalConfig, FIX_DEPLOYED_UTC, GATE_COUNTRIES,
                                         GATE_EXCLUDED_COUNTRIES, as_of_for_vintage,
                                         load_actuals, load_forecasts, point_metrics)
from src.evaluation.static_bias import (BREAK_EVEN_ABS_T, MIN_TARGET_DAYS,
                                        Thresholds, ZoneDecision, apply_to_frame,
                                        fit_static_bias, level_drift_diagnostic,
                                        measure, split_halves)

CHAMPION = "chronos-2-V010"

# A target day is scored only when every gated zone published every hour of it.
# A partly-published trailing day drops zones unevenly, which moves a per-zone
# bias by changing *which* zones the day contributes -- the ABL-639 shape.
EXPECTED_ZONE_HOURS = 24


def load_pairs(replica_db: str, sidecar_db: str, model: str) -> pd.DataFrame:
    cfg = EvalConfig(replica_db=replica_db, sidecar_db=sidecar_db, model_name=model)
    fc = load_forecasts(cfg)
    if fc.empty:
        raise SystemExit(f"no '{model}' net_position forecasts found")
    act = load_actuals(cfg)
    paired = fc.merge(act.rename(columns={"ts": "target_ts"}),
                      on=["country_code", "target_ts"], how="left")
    paired.attrs["overlap_max_abs_diff_mw"] = fc.attrs.get("overlap_max_abs_diff_mw")
    paired.attrs["source_counts"] = fc.attrs.get("source_counts")
    paired.attrs["actuals_max_ts"] = act["ts"].max() if len(act) else None
    return paired


def complete_target_days(scored: pd.DataFrame, countries) -> tuple[pd.DataFrame, dict]:
    """Keep only target days where every gated zone has all 24 hours scored."""
    d = scored["target_ts"].dt.normalize()
    per_day = scored.assign(day=d).groupby(["day", "country_code"])["target_ts"] \
                    .nunique().unstack(fill_value=0)
    full = (per_day.reindex(columns=sorted(countries), fill_value=0)
            == EXPECTED_ZONE_HOURS).all(axis=1)
    keep = set(full[full].index)
    dropped = {str(day.date()): int(per_day.loc[day].sum())
               for day in full[~full].index}
    return scored[d.isin(keep)].copy(), {
        "days_kept": len(keep),
        "days_dropped_incomplete": dropped,
        "expected_zone_hours": EXPECTED_ZONE_HOURS,
    }


def contamination_screen(scored: pd.DataFrame) -> dict:
    """The ABL-595 section 8 screen, re-run on this window's actual hours."""
    act = (scored[["country_code", "target_ts", "actual"]]
           .drop_duplicates(["country_code", "target_ts"]))
    runs = {}
    zeros = {}
    for cc, g in act.sort_values("target_ts").groupby("country_code"):
        v = g["actual"].to_numpy(dtype=float)
        best, cur = 1, 1
        for i in range(1, len(v)):
            cur = cur + 1 if v[i] == v[i - 1] else 1
            best = max(best, cur)
        runs[cc] = int(best)
        n0 = int(np.sum(v == 0.0))
        if n0:
            zeros[cc] = n0
    dup = int(act.duplicated(["country_code", "target_ts"]).sum())
    return {
        "actual_hours_screened": int(len(act)),
        "nulls": int(act["actual"].isna().sum()),
        "duplicated_hours": dup,
        "exact_zero_hours": zeros,
        "longest_bit_identical_run_hours": max(runs.values()) if runs else 0,
        "longest_run_zone": max(runs, key=runs.get) if runs else None,
        "degenerate_vintage_zone_days": int(
            (scored.groupby(["country_code", "generated_at"])["forecast_value"]
                   .apply(lambda s: float(np.nanmax(np.abs(s.to_numpy(dtype=float)))))
             <= 1.0).sum()),
    }


def score_before_after(pairs: pd.DataFrame, decisions) -> dict:
    """Per-zone and pooled point metrics for the champion and the corrected series."""
    applied = apply_to_frame(pairs, decisions)
    per_zone = {}
    for cc, g in applied.groupby("country_code"):
        a = g["actual"].to_numpy(dtype=float)
        before = point_metrics(a, g["forecast_value"].to_numpy(dtype=float))
        after = point_metrics(a, g["corrected"].to_numpy(dtype=float))
        dec = decisions.get(cc)
        per_zone[cc] = {
            "n": before["n"],
            "corrected": bool(dec is not None and dec.applied),
            "intercept_mw": float(dec.intercept_mw) if dec is not None else 0.0,
            "before": before,
            "after": after,
            "mae_delta_mw": after["mae_mw"] - before["mae_mw"],
            "mae_delta_pct": (100.0 * (after["mae_mw"] - before["mae_mw"])
                              / before["mae_mw"]) if before["mae_mw"] else None,
            "slope_abs_delta": (abs(after["slope"] - before["slope"])
                                if before.get("slope") is not None
                                and after.get("slope") is not None else None),
            "corr_abs_delta": (abs(after["corr"] - before["corr"])
                               if before.get("corr") is not None
                               and after.get("corr") is not None else None),
            "sd_ratio_abs_delta": (abs(after["sd_ratio"] - before["sd_ratio"])
                                   if before.get("sd_ratio") is not None
                                   and after.get("sd_ratio") is not None else None),
            "bias_frac_before_pct": (100.0 * abs(before["bias_mw"])
                                     / before["mean_abs_actual_mw"]
                                     if before.get("mean_abs_actual_mw") else None),
            "bias_frac_after_pct": (100.0 * abs(after["bias_mw"])
                                    / after["mean_abs_actual_mw"]
                                    if after.get("mean_abs_actual_mw") else None),
        }
    a = applied["actual"].to_numpy(dtype=float)
    pooled = {
        "before": point_metrics(a, applied["forecast_value"].to_numpy(dtype=float)),
        "after": point_metrics(a, applied["corrected"].to_numpy(dtype=float)),
    }
    # A pooled MAE over zones of wildly different size is dominated by the big
    # ones; the mean of per-zone MAEs is the view the gate's per-country
    # criteria actually reflect. Both are reported, neither is a verdict.
    pooled["mean_of_zone_mae_before"] = float(
        np.mean([m["before"]["mae_mw"] for m in per_zone.values()]))
    pooled["mean_of_zone_mae_after"] = float(
        np.mean([m["after"]["mae_mw"] for m in per_zone.values()]))
    pooled["zones_improved"] = int(sum(1 for m in per_zone.values()
                                       if m["mae_delta_mw"] < 0))
    pooled["zones_worsened"] = int(sum(1 for m in per_zone.values()
                                       if m["mae_delta_mw"] > 0))
    pooled["zones_corrected"] = int(sum(1 for m in per_zone.values() if m["corrected"]))
    pooled["max_slope_abs_delta"] = max(
        [m["slope_abs_delta"] for m in per_zone.values()
         if m["slope_abs_delta"] is not None] or [None])
    pooled["max_corr_abs_delta"] = max(
        [m["corr_abs_delta"] for m in per_zone.values()
         if m["corr_abs_delta"] is not None] or [None])
    pooled["max_sd_ratio_abs_delta"] = max(
        [m["sd_ratio_abs_delta"] for m in per_zone.values()
         if m["sd_ratio_abs_delta"] is not None] or [None])
    return {"per_zone": per_zone, "pooled": pooled}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replica-db", required=True)
    p.add_argument("--sidecar-db", required=True)
    p.add_argument("--model", default=CHAMPION)
    p.add_argument("--fit-end", default="2026-08-21",
                   help="exclusive target-day bound of the fitting window (UTC)")
    p.add_argument("--eval-vintage-start", default="2026-08-21",
                   help="first generated_at day of the held-out window (UTC)")
    p.add_argument("--min-target-days", type=int, default=MIN_TARGET_DAYS,
                   help="coverage floor for the delivered coefficients")
    p.add_argument("--holdout-min-target-days", type=int, default=14,
                   help="coverage floor used for the held-out fit. Below the "
                        "delivery floor on purpose: halving the cohort is the "
                        "only out-of-sample test this much data supports, and "
                        "what it costs is itself a result")
    p.add_argument("--json-out", required=True)
    p.add_argument("--coefficients-out",
                   help="frozen per-zone intercepts, for a serve decision to "
                        "act on without re-deriving them")
    args = p.parse_args()

    paired = load_pairs(args.replica_db, args.sidecar_db, args.model)
    scored = paired.dropna(subset=["actual"]).copy()
    scored = scored[(scored["generated_at"] >= FIX_DEPLOYED_UTC)
                    & (scored["country_code"].isin(GATE_COUNTRIES))]
    scored, completeness = complete_target_days(scored, GATE_COUNTRIES)

    fit_end = pd.Timestamp(args.fit_end)
    ev_start = pd.Timestamp(args.eval_vintage_start)
    day = scored["target_ts"].dt.normalize()
    fit = scored[day < fit_end]
    ev = scored[scored["generated_at"] >= ev_start]
    # The gap between them is the whole serve-faithfulness argument: the last
    # actual the fit reads is published before the first evaluated vintage runs.
    buffer = scored[(day >= fit_end) & (scored["generated_at"] < ev_start)]

    th = Thresholds(min_target_days=args.min_target_days)
    th_holdout = Thresholds(min_target_days=args.holdout_min_target_days,
                            min_pairs=args.holdout_min_target_days * 24)
    full_decisions = fit_static_bias(scored, GATE_COUNTRIES, th)
    fit_decisions = fit_static_bias(fit, GATE_COUNTRIES, th_holdout)
    # What the qualification test is worth: the same frozen intercepts, applied
    # to every zone with no test at all. A test that does not beat this is
    # bureaucracy.
    untested = {cc: ZoneDecision(country=cc, applied=True,
                                 intercept_mw=float(measure(
                                     fit[fit["country_code"] == cc]).bias_mw or 0.0),
                                 reason="no qualification test (counterfactual arm)")
                for cc in GATE_COUNTRIES}

    currency = {}
    for cc in GATE_COUNTRIES:
        g = scored[scored["country_code"] == cc]
        h1_df, h2_df = split_halves(g)
        w, h1, h2 = measure(g), measure(h1_df), measure(h2_df)
        currency[cc] = {
            "window": w.to_dict(), "half1": h1.to_dict(), "half2": h2.to_dict(),
            "decision": full_decisions[cc].to_dict(),
            "level_drift": level_drift_diagnostic(h1, h2, w),
        }

    # The arm that validates what is actually delivered: the zones that qualify
    # under the delivery thresholds, carrying the intercept fitted on the fit
    # window alone. Selection is intersected with the fit-window verdict so the
    # arm cannot be selecting on the window it is scored on.
    delivered = {cc: fit_decisions[cc] for cc in GATE_COUNTRIES
                 if full_decisions[cc].applied and fit_decisions[cc].applied}

    holdout = score_before_after(ev, fit_decisions)
    holdout_untested = score_before_after(ev, untested)
    holdout_delivered = score_before_after(ev, delivered)
    in_sample = score_before_after(fit, fit_decisions)

    first_ev_vintage = ev["generated_at"].min()
    doc = {
        "issue": "ABL-651",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "databases": {"replica": args.replica_db, "sidecar": args.sidecar_db,
                      "access": "read-only URI (file:...?mode=ro) on both"},
        "correction": {
            "form": "corrected = forecast - intercept_mw, per zone",
            "free_parameters_per_zone": 1,
            "slope_term": None,
            "thresholds": th.to_dict(),
            "holdout_thresholds": th_holdout.to_dict(),
            "break_even_abs_t": BREAK_EVEN_ABS_T,
        },
        "windows": {
            "cohort": f"post-fix vintages (generated_at >= {FIX_DEPLOYED_UTC})",
            "scored_target_days": [str(day.min()), str(day.max())],
            "completeness": completeness,
            "full": {"pairs": int(len(scored)),
                     "target_days": int(day.nunique()),
                     "vintages": int(scored["generated_at"].nunique())},
            "fit": {"pairs": int(len(fit)),
                    "target_days": int(fit["target_ts"].dt.normalize().nunique()),
                    "vintages": int(fit["generated_at"].nunique()),
                    "target_span": [str(fit["target_ts"].min()),
                                    str(fit["target_ts"].max())],
                    "vintage_span": [str(fit["generated_at"].min()),
                                     str(fit["generated_at"].max())]},
            "buffer": {"pairs": int(len(buffer)),
                       "target_days": int(buffer["target_ts"].dt.normalize().nunique()),
                       "why": "the D+2 gap -- target days the fit may not read and "
                              "the held-out vintages did not target"},
            "eval": {"pairs": int(len(ev)),
                     "target_days": int(ev["target_ts"].dt.normalize().nunique()),
                     "vintages": int(ev["generated_at"].nunique()),
                     "target_span": [str(ev["target_ts"].min()),
                                     str(ev["target_ts"].max())],
                     "vintage_span": [str(ev["generated_at"].min()),
                                      str(ev["generated_at"].max())]},
            "serve_faithful": {
                "first_eval_vintage": str(first_ev_vintage),
                "its_actuals_cutoff": str(as_of_for_vintage(first_ev_vintage)),
                "last_fit_target_hour": str(fit["target_ts"].max()),
                "fit_is_observable_at_first_eval_vintage": bool(
                    fit["target_ts"].max() < as_of_for_vintage(first_ev_vintage)),
            },
        },
        "excluded_by_name": dict(GATE_EXCLUDED_COUNTRIES),
        "contamination_screen": contamination_screen(scored),
        "provenance": {
            "sidecar_vs_pushed_max_abs_diff_mw":
                paired.attrs.get("overlap_max_abs_diff_mw"),
            "source_counts": paired.attrs.get("source_counts"),
            "actuals_max_ts": str(paired.attrs.get("actuals_max_ts")),
        },
        "currency_measurement": currency,
        "holdout_delivered_set": {
            "zones": sorted(delivered),
            "selection": "qualified under the delivery thresholds on the full "
                         "window AND under the holdout thresholds on the fit "
                         "window; intercept from the fit window only",
            **holdout_delivered},
        "holdout_at_holdout_floor": holdout,
        "holdout_no_test_counterfactual": holdout_untested,
        "fit_window_in_sample": in_sample,
        "fit_decisions": {cc: d.to_dict() for cc, d in fit_decisions.items()},
        "recommended_coefficients": {
            cc: {"intercept_mw": d.intercept_mw, "applied": d.applied,
                 "reason": d.reason}
            for cc, d in full_decisions.items()},
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")

    if args.coefficients_out:
        coeff = Path(args.coefficients_out)
        coeff.parent.mkdir(parents=True, exist_ok=True)
        coeff.write_text(json.dumps({
            "issue": "ABL-651",
            "status": "MEASURED -- not served. The serve decision is the CEO's.",
            "model_corrected": args.model,
            "form": "corrected = forecast - intercept_mw, per zone; no slope term",
            "fitted_on": {
                "cohort": f"post-fix vintages (generated_at >= {FIX_DEPLOYED_UTC})",
                "target_days": [str(day.min().date()), str(day.max().date())],
                "n_target_days": int(day.nunique()),
                "pairs_per_zone": int(len(scored) // len(GATE_COUNTRIES)),
            },
            "thresholds": th.to_dict(),
            "intercept_mw": {cc: (round(d.intercept_mw, 4) if d.applied else 0.0)
                             for cc, d in full_decisions.items()},
            "applied": sorted(cc for cc, d in full_decisions.items() if d.applied),
            "left_alone": {cc: d.reason for cc, d in full_decisions.items()
                           if not d.applied},
            "revalidate": "Refit and re-qualify before serving if more than 14 "
                          "target days have accrued since the fitted_on window: "
                          "the qualification is a property of the window, and "
                          "three of the four zones on the authorising list "
                          "changed verdict in one week.",
        }, indent=2, default=str), encoding="utf-8")
        print(f"wrote {coeff}")

    corrected = [cc for cc, d in fit_decisions.items() if d.applied]
    print(f"fit-window qualifying zones: {corrected or 'none'}")
    print(f"full-window qualifying zones: "
          f"{[cc for cc, d in full_decisions.items() if d.applied] or 'none'}")
    print(f"holdout: pooled MAE {holdout['pooled']['before']['mae_mw']:.1f} -> "
          f"{holdout['pooled']['after']['mae_mw']:.1f} MW, "
          f"max |slope delta| {holdout['pooled']['max_slope_abs_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
