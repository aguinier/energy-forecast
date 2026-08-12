"""Per-country net-position re-read: baseline table, level-vs-shape split (ABL-280).

ABL-280 filed a claim about one country (RO loses to hour-of-day climatology by
23.3%, and the failure is day-level bias rather than intraday shape) on **six
vintage days**. The CEO's triage kept exactly one thing on that issue — the
confirmatory re-read at the pre-registered `GATE_MIN_LIVE_VINTAGES` — and this
module is what makes that re-read a re-run instead of a re-derivation.

It adds three things `src/evaluation/net_position.py` does not have, and
deliberately reuses that module for everything it does have (`point_metrics`,
`baseline_predictions`, `as_of_for_vintage`, the cohort split) so the two can
never disagree about a country's MAE.

**1. The zero forecast as a named baseline.** `mean|actual|` is the MAE of
predicting 0 MW everywhere, so `skill_vs_zero < 0` is exactly `WAPE > 100%` —
the same fact twice. Naming it stops "WAPE > 100%" from reading as an emergency
by itself: zero is not a baseline anyone would serve for net position, and a
zone can lose to it while still beating persistence by 20%. The identity is
pinned in the tests so the zero baseline cannot be re-derived wrongly.

**2. The level-vs-shape split.** Removing each vintage day's mean from *both*
series separates "the model has the wrong profile" from "the model has the right
profile at the wrong level". For RO on the filing cohort the pooled correlation
is 0.501 and the within-day correlation is 0.830 — the shape is fine and the
level moves. That distinction is what refuted a static per-country offset for RO
(a constant cannot track a bias that swings +259 to -1095 MW across six days).

**3. Evidence vintages counted honestly.** `net_position.build_gate_scope`
counts vintages off the *left-merged* frame, so a vintage whose targets have no
published actuals yet still counts. That is never zero vintages: the rail
generates at D for D+2, so the two newest vintages are structurally unscorable.
Measured 2026-08-12, 9 vintages exist and 7 carry any scored pair.
`evidence_vintages` reports both numbers side by side and
`meets_min_vintages` reads the **scored** one, so this module's own precondition
cannot be satisfied by vintages that contributed no evidence. This module does
not change the gate — that is pre-registered and not ours to move; it declines
to inherit the ambiguity.

Conventions are `net_position.py`'s, unchanged: bias = mean(forecast - actual),
slope = OLS of forecast on actual, every vintage-target pair counts. The
per-vintage-day bias spread is reported with **ddof=1** (a sample sd over the
vintage days observed); ddof=0 over the same six days reads 658.6 against 721.5,
so the choice is stated rather than left to a default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .net_position import (
    FIX_DEPLOYED_UTC,
    GATE_EXCLUDED_COUNTRIES,
    GATE_MIN_LIVE_VINTAGES,
    point_metrics,
)

# Baselines reported beside the model, in the order the report prints them.
# "zero" is computed here (it needs no history); the other three are the
# serve-faithful columns `net_position.evaluate` already attaches to `scored`.
BASELINE_COLUMNS = ("persistence", "climatology", "baseline_ensemble")


def vintage_day(generated_at: pd.Series) -> pd.Series:
    """The UTC run-day behind a vintage. Grouping key for the level analysis.

    A same-day re-run (measured: 2026-08-06 has both a 06:00 and a 10:52 run)
    is two vintages but one day of evidence, and the level error is a property
    of the day's inputs, not of the run. So the level split groups by day while
    the vintage counts below stay in vintages, matching the gate's own units.
    """
    return pd.to_datetime(generated_at).dt.normalize()


def evidence_vintages(paired: pd.DataFrame,
                      cohort_split: pd.Timestamp = FIX_DEPLOYED_UTC) -> dict:
    """Vintages present vs vintages that contribute at least one scored pair.

    `paired` is the left-merged frame (`actual` NaN where no actual is
    published yet). Both counts are returned because they routinely differ and
    only one of them is evidence.
    """
    if paired.empty:
        return {"counted": 0, "scored": 0, "counted_days": 0, "scored_days": 0,
                "unscored_vintages": []}
    window = paired[pd.to_datetime(paired["generated_at"]) >= cohort_split]
    scored = window.dropna(subset=["actual"])
    counted_v = set(window["generated_at"].unique())
    scored_v = set(scored["generated_at"].unique())
    return {
        "counted": len(counted_v),
        "scored": len(scored_v),
        "counted_days": int(vintage_day(window["generated_at"]).nunique()),
        "scored_days": int(vintage_day(scored["generated_at"]).nunique()) if len(scored) else 0,
        "unscored_vintages": [str(pd.Timestamp(v)) for v in sorted(counted_v - scored_v)],
    }


def zero_baseline_mae(actual: np.ndarray) -> float:
    """MAE of forecasting 0 MW. Identically `mean|actual|`, hence WAPE's
    denominator over n — which is why skill against it is a restatement of
    `WAPE > 100%` and not an independent test."""
    return float(np.mean(np.abs(np.asarray(actual, dtype=float)))) if len(actual) else float("nan")


def _skill(model_mae: float, baseline_mae: float) -> float | None:
    """100 * (1 - model/baseline). Positive = model wins. None when the
    baseline is itself perfect (division by zero), never 0."""
    if baseline_mae is None or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return None
    return 100.0 * (1.0 - model_mae / baseline_mae)


def baseline_table(scored: pd.DataFrame) -> list[dict]:
    """Model MAE plus every baseline's MAE and the model's skill against it.

    Each baseline is scored on **its own** available pairs and the model's MAE
    is recomputed on that same subset, so a baseline that is NaN for part of
    the window (no history yet) cannot flatter or penalise the comparison by
    changing the denominator under it.
    """
    rows: list[dict] = []
    a = scored["actual"].to_numpy()
    f = scored["forecast_value"].to_numpy()
    model_mae = float(np.mean(np.abs(f - a))) if len(a) else float("nan")
    rows.append({"baseline": "model", "n": int(len(a)), "mae_mw": model_mae,
                 "skill_pct": None})
    zero_mae = zero_baseline_mae(a)
    rows.append({"baseline": "zero", "n": int(len(a)), "mae_mw": zero_mae,
                 "skill_pct": _skill(model_mae, zero_mae)})
    for name in BASELINE_COLUMNS:
        if name not in scored.columns:
            continue
        sub = scored.dropna(subset=[name])
        if sub.empty:
            rows.append({"baseline": name, "n": 0, "mae_mw": None, "skill_pct": None})
            continue
        b_mae = float(np.mean(np.abs(sub[name].to_numpy() - sub["actual"].to_numpy())))
        m_mae = float(np.mean(np.abs(sub["forecast_value"].to_numpy()
                                     - sub["actual"].to_numpy())))
        rows.append({"baseline": name, "n": int(len(sub)), "mae_mw": b_mae,
                     "skill_pct": _skill(m_mae, b_mae)})
    return rows


def level_vs_shape(scored: pd.DataFrame) -> dict:
    """Split the error into a day-level component and an intraday-shape one.

    Demeaning both series **within each vintage day** removes exactly the
    per-day level and leaves the profile. A model that knows the profile and
    misses the level reads as `within_day.corr >> pooled.corr`; a model with a
    genuinely wrong shape does not improve on demeaning.

    `bias_sd_mw` is the sd (ddof=1) of the per-day mean error — how far the
    level wanders between days — reported against `mean_abs_actual_mw` so the
    swing has a scale. When that ratio is near or above 1, the level error
    alone is the size of the signal, and no *static* offset can absorb it.
    """
    if scored.empty:
        return {"n": 0}
    g = scored.copy()
    g["vintage_day"] = vintage_day(g["generated_at"])
    a = g["actual"].to_numpy()
    f = g["forecast_value"].to_numpy()
    pooled = point_metrics(a, f)
    a_d = (g["actual"] - g.groupby("vintage_day")["actual"].transform("mean")).to_numpy()
    f_d = (g["forecast_value"]
           - g.groupby("vintage_day")["forecast_value"].transform("mean")).to_numpy()
    within = point_metrics(a_d, f_d)
    day_bias = g.groupby("vintage_day").apply(
        lambda d: float(np.mean(d["forecast_value"].to_numpy() - d["actual"].to_numpy())),
        include_groups=False)
    mean_abs_actual = pooled.get("mean_abs_actual_mw")
    bias_sd = float(np.std(day_bias.to_numpy(), ddof=1)) if len(day_bias) > 1 else None
    return {
        "n": int(len(g)),
        "vintage_days": int(len(day_bias)),
        "pooled": {"corr": pooled.get("corr"), "slope": pooled.get("slope")},
        "within_day": {"corr": within.get("corr"), "slope": within.get("slope")},
        "bias_sd_mw": bias_sd,
        "mean_abs_actual_mw": mean_abs_actual,
        "bias_sd_frac_of_mean_abs_actual": (
            bias_sd / mean_abs_actual
            if bias_sd is not None and mean_abs_actual else None),
        "day_bias_mw": {str(k.date()): float(v) for k, v in day_bias.items()},
    }


def per_vintage_day(scored: pd.DataFrame) -> list[dict]:
    """One row per vintage day: n, WAPE, corr, bias. This is the cut that shows
    whether a >100% WAPE is a chronic level error or two bad days carrying it."""
    if scored.empty:
        return []
    g = scored.copy()
    g["vintage_day"] = vintage_day(g["generated_at"])
    rows = []
    for day, d in g.groupby("vintage_day"):
        m = point_metrics(d["actual"].to_numpy(), d["forecast_value"].to_numpy())
        rows.append({"vintage_day": str(day.date()), "n": m["n"],
                     "wape_pct": m.get("wape_pct"), "corr": m.get("corr"),
                     "bias_mw": m.get("bias_mw"), "mae_mw": m.get("mae_mw")})
    return rows


def country_reread(paired: pd.DataFrame, country: str,
                   cohort_split: pd.Timestamp = FIX_DEPLOYED_UTC,
                   min_scored_vintages: int = GATE_MIN_LIVE_VINTAGES) -> dict:
    """The whole ABL-280 re-read for one country, from the left-merged frame.

    `meets_min_vintages` reads **scored** vintages. Below the threshold this is
    an interim read and says so in its own payload — the caller does not have to
    remember, and a report that quotes it cannot silently present six vintage
    days as the confirmatory answer.
    """
    ev_all = evidence_vintages(paired, cohort_split)
    sub = paired[paired["country_code"] == country]
    ev = evidence_vintages(sub, cohort_split)
    window = sub[pd.to_datetime(sub["generated_at"]) >= cohort_split]
    scored = window.dropna(subset=["actual"]).copy()

    out = {
        "country": country,
        "excluded_from_gate": GATE_EXCLUDED_COUNTRIES.get(country),
        "cohort_split_utc": str(cohort_split),
        "n_pairs": int(len(scored)),
        "vintages": ev,
        "vintages_all_countries": ev_all,
        "min_scored_vintages": int(min_scored_vintages),
        "meets_min_vintages": ev["scored"] >= min_scored_vintages,
        "read_kind": ("confirmatory" if ev["scored"] >= min_scored_vintages
                      else "interim"),
    }
    if scored.empty:
        out["coverage"] = "no_paired_actuals"
        return out
    out["target_window"] = [str(scored["target_ts"].min()),
                            str(scored["target_ts"].max())]
    out["metrics"] = point_metrics(scored["actual"].to_numpy(),
                                   scored["forecast_value"].to_numpy())
    out["baselines"] = baseline_table(scored)
    out["level_vs_shape"] = level_vs_shape(scored)
    out["per_vintage_day"] = per_vintage_day(scored)
    out["loses_to"] = [r["baseline"] for r in out["baselines"]
                       if r["baseline"] != "model" and r["skill_pct"] is not None
                       and r["skill_pct"] < 0]
    return out


def fleet_summary(paired: pd.DataFrame, countries,
                  cohort_split: pd.Timestamp = FIX_DEPLOYED_UTC,
                  min_scored_vintages: int = GATE_MIN_LIVE_VINTAGES) -> list[dict]:
    """One row per country: skill against each baseline, plus the level/shape gap.

    A single country's "loses to climatology" is not interpretable alone. If
    four of nineteen zones lose to it, the finding is about the model class, not
    about that zone — and a per-country served fallback would be the wrong
    remedy. Measured 2026-08-12 on the interim cohort, RO (-23.3%) and NL
    (-18.3%) lose materially while BE (-2.5%) and HR (-0.2%) lose inside noise,
    so the sweep is what keeps an RO-shaped conclusion from being drawn on an
    NL-shaped fact.

    `level_gap` = within-day corr minus pooled corr. A large positive value is
    the day-level-bias signature: right profile, wrong level.
    """
    rows = []
    for cc in countries:
        read = country_reread(paired, cc, cohort_split, min_scored_vintages)
        if read.get("coverage") == "no_paired_actuals" or not read.get("metrics"):
            rows.append({"country": cc, "n": 0, "coverage": "no_paired_actuals"})
            continue
        b = {r["baseline"]: r for r in read["baselines"]}
        lvs = read["level_vs_shape"]
        pooled_corr, within_corr = lvs["pooled"]["corr"], lvs["within_day"]["corr"]
        rows.append({
            "country": cc,
            "n": read["n_pairs"],
            "scored_vintages": read["vintages"]["scored"],
            "wape_pct": read["metrics"].get("wape_pct"),
            "mae_mw": read["metrics"].get("mae_mw"),
            "skill_vs_zero_pct": b["zero"]["skill_pct"],
            "skill_vs_climatology_pct": b.get("climatology", {}).get("skill_pct"),
            "skill_vs_persistence_pct": b.get("persistence", {}).get("skill_pct"),
            "skill_vs_ensemble_pct": b.get("baseline_ensemble", {}).get("skill_pct"),
            "corr_pooled": pooled_corr,
            "corr_within_day": within_corr,
            "level_gap": (within_corr - pooled_corr
                          if pooled_corr is not None and within_corr is not None
                          else None),
            "bias_sd_frac": lvs.get("bias_sd_frac_of_mean_abs_actual"),
            "loses_to": read["loses_to"],
        })
    return rows


def render_fleet_markdown(rows: list[dict], generated_at: str,
                          min_scored_vintages: int = GATE_MIN_LIVE_VINTAGES) -> str:
    scored = [r for r in rows if r.get("n")]
    v = max((r["scored_vintages"] for r in scored), default=0)
    kind = "CONFIRMATORY" if v >= min_scored_vintages else "INTERIM"
    L = [f"# Net-position baselines across the gate countries (ABL-280 context)", "",
         f"Generated {generated_at}. **Read kind: {kind}** — {v} scored vintages "
         f"against a pre-registered minimum of {min_scored_vintages}.", ""]
    if kind == "INTERIM":
        L += ["> Below the minimum. Flags, not findings.", ""]
    L += ["| country | n | WAPE | vs zero | vs climatology | vs persistence | "
          "vs ensemble | corr pooled | corr within-day | bias sd / mean\\|actual\\| |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(scored, key=lambda x: (x["skill_vs_climatology_pct"] is None,
                                           x["skill_vs_climatology_pct"])):
        def s(k):
            return "—" if r.get(k) is None else f"{r[k]:+.1f}%"
        L.append(f"| {r['country']} | {r['n']} | {_f(r['wape_pct'])}% | "
                 f"{s('skill_vs_zero_pct')} | {s('skill_vs_climatology_pct')} | "
                 f"{s('skill_vs_persistence_pct')} | {s('skill_vs_ensemble_pct')} | "
                 f"{_f(r['corr_pooled'], 2)} | {_f(r['corr_within_day'], 2)} | "
                 f"{_f(r['bias_sd_frac'], 2)} |")
    for name, key in (("zero", "skill_vs_zero_pct"),
                      ("climatology", "skill_vs_climatology_pct"),
                      ("persistence", "skill_vs_persistence_pct"),
                      ("ensemble", "skill_vs_ensemble_pct")):
        lose = sorted(r["country"] for r in scored
                      if r.get(key) is not None and r[key] < 0)
        L += [f"", f"- Loses to **{name}**: {', '.join(lose) if lose else 'none'} "
                   f"({len(lose)}/{len(scored)})"]
    return "\n".join(L) + "\n"


def _f(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{nd}f}"


def render_markdown(read: dict, generated_at: str) -> str:
    """Report text. The read kind and the vintage counts lead, because a number
    quoted without them is the thing this issue exists to stop."""
    ev, ev_all = read["vintages"], read["vintages_all_countries"]
    L = [f"# Net-position re-read — {read['country']} (ABL-280)", "",
         f"Generated {generated_at}. Cohort: vintages at or after "
         f"`{read['cohort_split_utc']}` (the context-cutoff fix).", ""]
    L += [f"**Read kind: {read['read_kind'].upper()}** — "
          f"{ev['scored']} scored vintages against a pre-registered minimum of "
          f"{read['min_scored_vintages']}."]
    if not read["meets_min_vintages"]:
        L += ["", "> This is **not** the confirmatory read. Below the minimum, "
              "treat every number here as a flag, not a finding."]
    L += ["",
          f"- Vintages present in the window: **{ev['counted']}** "
          f"({ev['counted_days']} run-days)",
          f"- Vintages contributing >=1 scored pair: **{ev['scored']}** "
          f"({ev['scored_days']} run-days)"]
    if ev["unscored_vintages"]:
        L += [f"- Present but unscored (targets not yet published): "
              f"{', '.join(ev['unscored_vintages'])}",
              "",
              "  The rail generates at D for D+2, so the newest vintages are "
              "structurally unscorable. `net_position.build_gate_scope` counts "
              "the first number; this read uses the second."]
    if read.get("excluded_from_gate"):
        L += ["", f"> Excluded from the promotion gate by name: "
                  f"{read['excluded_from_gate']}"]
    if read.get("coverage") == "no_paired_actuals":
        L += ["", "No paired actuals in this window — nothing scored.", ""]
        return "\n".join(L)

    m = read["metrics"]
    L += ["", f"Pairs scored: **{m['n']}**; targets "
              f"{read['target_window'][0]} .. {read['target_window'][1]} UTC.", "",
          "## Baselines", "",
          "| baseline | n | MAE (MW) | model skill |", "|---|---:|---:|---:|"]
    for r in read["baselines"]:
        skill = "—" if r["skill_pct"] is None else (
            f"{r['skill_pct']:+.1f}%" + (" (loses)" if r["skill_pct"] < 0 else ""))
        name = "**the model**" if r["baseline"] == "model" else r["baseline"]
        L.append(f"| {name} | {r['n']} | {_f(r['mae_mw'])} | {skill} |")
    L += ["",
          f"WAPE {_f(m.get('wape_pct'))}%, bias {_f(m.get('bias_mw'))} MW "
          f"against mean |actual| {_f(m.get('mean_abs_actual_mw'))} MW.",
          "",
          "`zero` is the 0 MW forecast: its MAE is mean |actual|, so losing to "
          "it is the same statement as WAPE > 100%. It is not a baseline anyone "
          "would serve — read the climatology and ensemble rows for the "
          "decision-relevant comparison."]

    lvs = read["level_vs_shape"]
    L += ["", "## Level vs shape", "",
          "| | corr | slope |", "|---|---:|---:|",
          f"| pooled | {_f(lvs['pooled']['corr'], 3)} | {_f(lvs['pooled']['slope'], 3)} |",
          f"| within vintage-day | {_f(lvs['within_day']['corr'], 3)} | "
          f"{_f(lvs['within_day']['slope'], 3)} |",
          "",
          f"Per-vintage-day bias sd (ddof=1): **{_f(lvs['bias_sd_mw'])} MW** against "
          f"mean |actual| {_f(lvs['mean_abs_actual_mw'])} MW "
          f"({_f((lvs['bias_sd_frac_of_mean_abs_actual'] or 0) * 100)}%).",
          "", "| vintage day | n | WAPE | corr | bias (MW) |",
          "|---|---:|---:|---:|---:|"]
    for r in read["per_vintage_day"]:
        L.append(f"| {r['vintage_day']} | {r['n']} | {_f(r['wape_pct'])}% | "
                 f"{_f(r['corr'], 3)} | {_f(r['bias_mw'])} |")
    L += ["", "---", "",
          "Conventions: bias = mean(forecast - actual); slope = OLS of forecast "
          "on actual; every vintage-target pair counts. Baselines are "
          "serve-faithful (`net_position.as_of_for_vintage`). Both databases "
          "opened read-only.", ""]
    return "\n".join(L)
