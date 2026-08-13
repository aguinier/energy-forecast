"""ABL-375: read the pre-registered bar off the fitted arms. No fitting here.

Kept separate from `scripts/abl338_solar_holdout.py` on purpose. That script
fits and scores; this one only *reads*, against the conditions frozen in
`experiments/ABL375/config.json` before any arm was fitted. Splitting them is
what makes the verdict auditable: the bar cannot be quietly adjusted to the
numbers, because the code that knows the bar never sees a fit.

The comparison
--------------
Both algorithms are refitted on the identically truncated window, with ABL-338's
solar geometry on both, at three registered seeds. The decision baseline is the
other algorithm's geometry arm -- an algorithm choice is a within-challenger
question -- with literal seasonal-naive D-7 as the sanity floor.

The registered bar (DE, daylight), all four required
----------------------------------------------------
1. seed-mean daylight MAE, xgboost+geometry < catboost+geometry
2. strict seed non-overlap: max(xgboost) < min(catboost)
3. relative gap >= MATERIAL_GAP_PCT
4. night guardrail: DE night mean prediction under xgboost+geometry <= catboost's

Condition 2 is why the seeds are here at all. A cross-library MAE gap read from
one fit each is not distinguishable from the spread one arm shows against its own
seed, and ABL-338's ~1.5% noise floor was a perturbation estimate on a different
window. Measuring the spread on *this* window replaces that figure; the 3.0%
threshold stays fixed because it was committed before the spread was seen.

The two already-observed ABL-338 windows are re-read here as EXPLORATORY and are
labelled so in every table. They generated the hypothesis, so they cannot
confirm it, and the verdict never reads them.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl375_read_gate.py \\
        --out reports/abl_375_de_solar_algorithm
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402,F401  (imported for LOG_FORMAT and the import contract)

logger = logging.getLogger("energy_forecast")

#: Registered in experiments/ABL375/config.json before any fit: twice ABL-338's
#: ~1.5% perturbation noise floor. Named here so a reader can see the one number
#: the verdict is sensitive to without opening the JSON.
MATERIAL_GAP_PCT = 3.0

#: The country the primary bar is about. BE/FR/AT are reported, never gating.
PRIMARY_COUNTRY = "DE"

#: The registered confirmatory read: one file per algorithm, both forced, so
#: neither inherits the other's hyperparameter dict.
CONFIRMATORY = {
    "catboost": "reports/abl_375_solar/holdout_registered_catboost_cleaned.json",
    "xgboost": "reports/abl_375_solar/holdout_registered_xgboost_cleaned.json",
}

#: Already fitted and scored under ABL-338 before this registration existed.
#: `incumbent` means "the artifact's own algorithm", which is xgboost for AT and
#: catboost for BE/DE/FR — so the per-country algorithm is read off each payload
#: rather than inferred from the filename.
#: `seeded` is a POST-HOC noise characterisation of the same two windows, DE
#: only, run *after* the primary verdict above was read and committed (d990fcf).
#: It exists because the registered read found DE CatBoost moving 13.79% of its
#: mean across seeds, which makes every single-seed number in ABL-338 - including
#: the 10.7% this issue was filed on - unreadable until its own spread is known.
#: It cannot and does not change the registered verdict; the git order is the
#: evidence that it was not used to.
EXPLORATORY = {
    "summer": {
        "window": ["2026-06-13", "2026-08-11"],
        "files": ["reports/abl_338_solar/holdout_summer_incumbent_cleaned.json",
                  "reports/abl_338_solar/holdout_summer_xgboost_cleaned.json"],
        "seeded": ["reports/abl_375_solar/holdout_noisefloor_summer_catboost_cleaned.json",
                   "reports/abl_375_solar/holdout_noisefloor_summer_xgboost_cleaned.json"],
    },
    "spring": {
        "window": ["2026-03-01", "2026-04-29"],
        "files": ["reports/abl_338_solar/holdout_spring_incumbent_cleaned.json",
                  "reports/abl_338_solar/holdout_spring_xgboost_cleaned.json"],
        "seeded": ["reports/abl_375_solar/holdout_noisefloor_spring_catboost_cleaned.json",
                   "reports/abl_375_solar/holdout_noisefloor_spring_xgboost_cleaned.json"],
    },
}

BANDS = ("daylight", "shoulder", "night")


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} is missing. The confirmatory arms must be fitted first: "
            f"scripts/abl338_solar_holdout.py --holdout 2026-04-30:2026-06-12 "
            f"--drop-impossible-night --arms control,geometry --seeds 42,1337,2718 "
            f"--force-algorithm <algo> --out reports/abl_375_solar --tag registered"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _spread(values: list) -> dict:
    """Mean and range of one arm's metric across its seeds."""
    if not values:
        return {}
    mean = sum(values) / len(values)
    out = {
        "n_seeds": len(values),
        "mean": round(mean, 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }
    # The seed spread *is* the noise floor for this window. Quoting a floor
    # measured elsewhere would be quoting a different experiment.
    out["spread_pct_of_mean"] = round(100.0 * (max(values) - min(values)) / mean, 2) if mean else None
    return out


def summarise_window(label: str, per_algorithm: dict, status: str) -> dict:
    """Collect band metrics per country x algorithm x arm for one holdout window."""
    countries = sorted(set().union(*(p["countries"].keys() for p in per_algorithm.values())))
    out = {"label": label, "status": status, "countries": {}}
    for country in countries:
        entry = {"arms": {}}
        for algorithm, payload in per_algorithm.items():
            result = payload["countries"].get(country)
            if result is None:
                continue
            entry.setdefault("n_holdout", result["n_holdout"])
            entry.setdefault("bands", result["bands"])
            entry.setdefault("baseline_seasonal_naive_d7", {
                b: {"mae_mw": round(result["baseline_seasonal_naive_d7"][b]["mae_mw"], 1)}
                for b in BANDS
            })
            entry.setdefault("contamination", result["contamination"])
            entry.setdefault("n_train", result["n_train"])
            # The algorithm this payload actually fitted, not the filename's claim.
            fitted = result["algorithm"]
            for arm_key, cell in result["arms"].items():
                arm = cell.get("arm", arm_key)
                slot = entry["arms"].setdefault(f"{fitted}+{arm}", {b: [] for b in BANDS})
                slot["daylight"].append(cell["daylight"]["mae_mw"])
                slot["shoulder"].append(cell["shoulder"]["mae_mw"])
                slot["night"].append(cell["night"]["mean_pred_mw"])
                slot.setdefault("night_max_pred_mw", []).append(cell["night"]["max_pred_mw"])
                slot.setdefault("n_negative_pred", []).append(cell["all"]["n_negative_pred"])
                slot.setdefault("daylight_wape_pct", []).append(
                    cell["daylight"].get("wape_pct"))
                slot.setdefault("n_features", cell["n_features"])
        entry["arms"] = {
            name: {
                "daylight_mae_mw": _spread(v["daylight"]),
                "shoulder_mae_mw": _spread(v["shoulder"]),
                "night_mean_pred_mw": _spread(v["night"]),
                "night_max_pred_mw": _spread(v["night_max_pred_mw"]),
                "n_negative_pred_max": max(v["n_negative_pred"]),
                "daylight_wape_pct": _spread([x for x in v["daylight_wape_pct"] if x is not None]),
                "n_features": v["n_features"],
            }
            for name, v in entry["arms"].items()
        }
        out["countries"][country] = entry
    return out


def read_bar(confirmatory: dict) -> dict:
    """Evaluate the four registered conditions. Reads only the confirmatory window."""
    per_country = {}
    for country, entry in confirmatory["countries"].items():
        cat = entry["arms"].get("catboost+geometry")
        xgb = entry["arms"].get("xgboost+geometry")
        if not cat or not xgb:
            per_country[country] = {"verdict": "NOT-EVALUABLE",
                                    "reason": "one algorithm's geometry arm is missing"}
            continue
        c_day, x_day = cat["daylight_mae_mw"], xgb["daylight_mae_mw"]
        gap_pct = 100.0 * (c_day["mean"] - x_day["mean"]) / c_day["mean"]
        conditions = {
            "mean_favours_xgboost": x_day["mean"] < c_day["mean"],
            "seed_ranges_disjoint": (x_day["max"] < c_day["min"]) or (c_day["max"] < x_day["min"]),
            "gap_at_least_material": abs(gap_pct) >= MATERIAL_GAP_PCT,
            "night_guardrail_xgboost_not_worse":
                xgb["night_mean_pred_mw"]["mean"] <= cat["night_mean_pred_mw"]["mean"],
        }
        # The bar is directional and was written for DE, where the registered
        # expectation is that xgboost wins. For a country the mean sends the
        # other way, the same three magnitude conditions are read in favour of
        # catboost — a reversal has to be reported with the same rigour as a win,
        # or the counter-cases are decoration.
        winner = "xgboost" if conditions["mean_favours_xgboost"] else "catboost"
        material = conditions["gap_at_least_material"] and conditions["seed_ranges_disjoint"]
        if winner == "xgboost":
            if material and conditions["night_guardrail_xgboost_not_worse"]:
                verdict = "PASS"
            elif material:
                verdict = "AMBIGUOUS"
            else:
                verdict = "AMBIGUOUS"
        else:
            verdict = "FAIL" if material else "AMBIGUOUS"
        per_country[country] = {
            "verdict": verdict,
            "favours": winner,
            "daylight_gap_pct_catboost_minus_xgboost_over_catboost": round(gap_pct, 2),
            "material_gap_threshold_pct": MATERIAL_GAP_PCT,
            "measured_noise_floor_pct": {
                "catboost+geometry": c_day["spread_pct_of_mean"],
                "xgboost+geometry": x_day["spread_pct_of_mean"],
            },
            "conditions": conditions,
            "catboost_geometry_daylight_mae_mw": c_day,
            "xgboost_geometry_daylight_mae_mw": x_day,
            "night_mean_pred_mw": {
                "catboost+geometry": cat["night_mean_pred_mw"]["mean"],
                "xgboost+geometry": xgb["night_mean_pred_mw"]["mean"],
            },
            "is_the_registered_primary": country == PRIMARY_COUNTRY,
        }
    return per_country


def _fmt(spread: dict, decimals: int = 1) -> str:
    if not spread:
        return "n/a"
    return (f"{spread['mean']:,.{decimals}f} "
            f"({spread['min']:,.{decimals}f}–{spread['max']:,.{decimals}f})")


def _render(payload: dict) -> str:
    conf = payload["confirmatory"]
    bar = payload["bar"]
    lines = [
        "# ABL-375 — DE solar: XGBoost vs the serving CatBoost configuration",
        "",
        f"Generated {payload['generated_at']}. Registration: "
        "`experiments/ABL375/config.json`, committed before the first fit.",
        "",
        "Both arms are **refits on the identically truncated window** — never the live",
        "artifacts, which are fitted through roughly today and would score in-sample.",
        "ABL-338's solar geometry is on **both** arms: `src/features.py` appends it to",
        "every solar fit unconditionally on `origin/main`, so the geometry arm is what a",
        "routine retrain would actually produce.",
        "",
        f"Cells are **seed-mean (min–max)** over seeds {payload['seeds']}. MAE in MW.",
        "Night is MW only: its denominator is ~0.",
        "",
        "## The registered read",
        "",
        f"Holdout **{conf['window'][0]} .. {conf['window'][1]}** "
        f"({conf['label']}), n = {conf['hours']:,} hours. Never fitted or scored for",
        "this comparison before the registration — the gap between ABL-338's two",
        "committed holdouts.",
        "",
        "| country | n_train | daylight n | CatBoost+geom daylight | XGBoost+geom daylight | gap | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for country, entry in conf["countries"].items():
        v = bar[country]
        lines.append(
            f"| {country}{' **(primary)**' if country == PRIMARY_COUNTRY else ''} "
            f"| {entry['n_train']:,} | {entry['bands']['daylight']:,} "
            f"| {_fmt(entry['arms'].get('catboost+geometry', {}).get('daylight_mae_mw', {}))} "
            f"| {_fmt(entry['arms'].get('xgboost+geometry', {}).get('daylight_mae_mw', {}))} "
            f"| {v.get('daylight_gap_pct_catboost_minus_xgboost_over_catboost', float('nan')):+.1f}% "
            f"| {v['verdict']} ({v.get('favours', '-')}) |"
        )
    lines += ["", "### All bands, all arms, registered window", ""]
    for country, entry in conf["countries"].items():
        base = entry["baseline_seasonal_naive_d7"]
        lines += [
            f"#### {country} — n_holdout {entry['n_holdout']:,} "
            f"(daylight {entry['bands']['daylight']:,} / shoulder {entry['bands']['shoulder']:,} "
            f"/ night {entry['bands']['night']:,})",
            "",
            "| arm | features | daylight MAE | daylight WAPE | shoulder MAE | night mean pred | night max pred | negative preds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| _seasonal-naive D-7_ | – | {base['daylight']['mae_mw']:,.1f} | – "
            f"| {base['shoulder']['mae_mw']:,.1f} | – | – | – |",
        ]
        for name in sorted(entry["arms"]):
            a = entry["arms"][name]
            lines.append(
                f"| {name} | {a['n_features']} | {_fmt(a['daylight_mae_mw'])} "
                f"| {_fmt(a['daylight_wape_pct'], 2)}% | {_fmt(a['shoulder_mae_mw'])} "
                f"| {_fmt(a['night_mean_pred_mw'], 2)} | {_fmt(a['night_max_pred_mw'])} "
                f"| {a['n_negative_pred_max']} |"
            )
        c = entry["contamination"]
        lines += [
            "",
            f"ABL-337 contamination: {c['train_night_rows_above_1mw']:,} of "
            f"{c['train_night_rows']:,} fit-window night rows read above 1 MW "
            f"(max {c['train_night_max_mw']:,.1f} MW), dropped from the fit and never "
            f"from the score.",
            "",
        ]
    lines += [
        "## Already-observed windows (EXPLORATORY — not a second gate)",
        "",
        "Fitted and scored under ABL-338 before this registration existed. Seeing them",
        "is what created the hypothesis, so they cannot confirm it. Single seed each.",
        "",
    ]
    for label, window in payload["exploratory"].items():
        lines += [
            f"### {label} — {window['window'][0]} .. {window['window'][1]} (EXPLORATORY)",
            "",
            "| country | CatBoost+geom daylight | XGBoost+geom daylight | gap | CatBoost+geom night | XGBoost+geom night |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for country, entry in window["countries"].items():
            cat = entry["arms"].get("catboost+geometry", {})
            xgb = entry["arms"].get("xgboost+geometry", {})
            if not cat or not xgb:
                lines.append(f"| {country} | – | – | one algorithm only | – | – |")
                continue
            cm = cat["daylight_mae_mw"]["mean"]
            xm = xgb["daylight_mae_mw"]["mean"]
            lines.append(
                f"| {country} | {cm:,.1f} | {xm:,.1f} | {100.0 * (cm - xm) / cm:+.1f}% "
                f"| {cat['night_mean_pred_mw']['mean']:,.2f} "
                f"| {xgb['night_mean_pred_mw']['mean']:,.2f} |"
            )
        seeded = window["seeded_de"]
        de = seeded["countries"].get("DE")
        if de:
            v = seeded["bar_read_for_reference_only"]["DE"]
            lines += [
                "",
                f"**DE re-fitted at the registered seeds on this already-observed window "
                f"(POST-HOC, {seeded['status'].split(' - ')[0]}).** Read for reference only: "
                f"the registered verdict is the confirmatory window's and is not revised here.",
                "",
                "| arm | daylight MAE (mean, min–max) | seed spread | shoulder MAE | night mean pred |",
                "|---|---:|---:|---:|---:|",
            ]
            for name in sorted(de["arms"]):
                a = de["arms"][name]
                lines.append(
                    f"| {name} | {_fmt(a['daylight_mae_mw'])} "
                    f"| {a['daylight_mae_mw']['spread_pct_of_mean']:.2f}% "
                    f"| {_fmt(a['shoulder_mae_mw'])} | {_fmt(a['night_mean_pred_mw'], 2)} |"
                )
            lines += [
                "",
                f"Geometry-arm gap {v['daylight_gap_pct_catboost_minus_xgboost_over_catboost']:+.1f}% "
                f"favouring {v['favours']}; seed ranges "
                f"{'disjoint' if v['conditions']['seed_ranges_disjoint'] else 'OVERLAPPING'}. "
                f"Would read {v['verdict']} had this been the registered window - it was not.",
                "",
            ]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/abl_375_de_solar_algorithm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    conf_payloads = {algo: _load(path) for algo, path in CONFIRMATORY.items()}
    one = next(iter(conf_payloads.values()))
    confirmatory = summarise_window(
        "registered confirmatory", conf_payloads, "CONFIRMATORY - pre-registered, unread before the fit")
    confirmatory["window"] = [one["holdout_start"], one["holdout_end"]]
    confirmatory["hours"] = next(iter(one["countries"].values()))["n_holdout"]

    exploratory = {}
    for label, spec in EXPLORATORY.items():
        payloads = {}
        for path in spec["files"]:
            p = _load(path)
            payloads[p.get("force_algorithm") or "incumbent"] = p
        window = summarise_window(label, payloads,
                                  "EXPLORATORY - already observed under ABL-338, single seed")
        window["window"] = spec["window"]
        seeded = {algo: _load(path) for algo, path in
                  zip(("catboost", "xgboost"), spec["seeded"])}
        window["seeded_de"] = summarise_window(
            f"{label} seeded", seeded,
            "POST-HOC - DE seed characterisation, run after the primary verdict was committed")
        window["seeded_de"]["bar_read_for_reference_only"] = read_bar(window["seeded_de"])
        exploratory[label] = window

    payload = {
        "issue": "ABL-375",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "registration": "experiments/ABL375/config.json",
        "seeds": one.get("seeds"),
        "material_gap_threshold_pct": MATERIAL_GAP_PCT,
        "primary_country": PRIMARY_COUNTRY,
        "confirmatory": confirmatory,
        "exploratory": exploratory,
    }
    payload["bar"] = read_bar(confirmatory)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_render(payload), encoding="utf-8")
    # The markdown is written, not printed: it carries en dashes, and this box's
    # console codepage mangles them (the ABL-364 failure mode). Stdout stays ASCII.
    print(f"Wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    for country, v in payload["bar"].items():
        primary = " [PRIMARY]" if v.get("is_the_registered_primary") else ""
        logger.info(
            f"{country}{primary}: {v['verdict']} favours={v.get('favours')} "
            f"gap={v.get('daylight_gap_pct_catboost_minus_xgboost_over_catboost')}% "
            f"disjoint={v.get('conditions', {}).get('seed_ranges_disjoint')} "
            f"seed_spread_pct={v.get('measured_noise_floor_pct')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
