"""ABL-393: apply the registered verdict rule to the load/price holdout JSONs.

No judgement here. Everything this script does is fixed by
`experiments/ABL393/config.json`, which was committed before any registered arm
was fitted (12ccbe5). The thresholds, the sign convention, the materiality test
and the recommendation mapping are all pre-registered; this file is the
arithmetic.

It is a separate script from `abl386_holiday_verdict.py` on purpose, and the
distinction matters: the *harness* is shared - both issues fit their arms with
`scripts/abl338_solar_holdout.py`, which is why there are not two instruments
measuring the same four features - but the *rule* is not. ABL-386 registered
unpaired seed-range disjointness on daylight MAE over three seeds. ABL-393
registers the paired-by-seed difference over eight, which ABL-386's own
corrections block names as the first thing a follow-up should fix. Making one
script parametric over both would put ABL-386's published verdict one refactor
away from moving.

The registered rule, restated so it can be checked against the config:

- **Unit**: one (type, window, country, algorithm) cell. Three registered
  (type, window) groups x 4 countries x 2 algorithms = 24 cells.
- **Statistic**: the paired-by-seed relative difference in all-hours MAE,
  `delta_s = 100 * (hol_s - nohol_s) / nohol_s`, at each of the 8 registered
  seeds. **NEGATIVE means the holiday features are BETTER** - the opposite of
  ABL-386's convention, so it is printed on every table rather than inherited.
- **Cell verdict**: `k` = how many of the 8 seeds favour holidays. `d = +1` if
  `k >= 7` (two-sided sign test p <= 0.0703; p = 0.0078 at 8/8), `d = -1` if
  `k <= 1`, else 0. No fixed percentage threshold, for ABL-386's reason.
- **Group verdict** over 8 cells: HELP if `sum(d) >= +4`, HARM if `<= -4`,
  NO_EFFECT if at most 2 cells are material, else MIXED.
- **Type verdict**: price is its single spring group. Load is spring, downgraded
  to MIXED if the winter replication disagrees in direction. Winter can weaken a
  load verdict and can never create one.
- **Secondary, reported and never gating**: the identical statistic over
  `holiday` and `holiday_affected` rows, and the margin of the holiday arm
  against D-7 and the four model-free references.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl393_holiday_verdict.py \\
        --inputs reports/abl_393_load_price/*.json \\
        --out reports/abl_393_holiday_verdict_tables
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#: The one contrast. There is no geometry on load or price, so ABL-386's 2x2
#: collapses to this and `control_noholiday` is exactly the serving list.
HOLIDAY_ARM, NO_HOLIDAY_ARM = "control", "control_noholiday"

#: The repo's standing eight-seed set (ABL-376, ABL-395), inherited rather than
#: chosen. Eight is the smallest set at which a two-sided sign test reaches
#: p = 0.0078, which is what the cell rule below needs.
SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)

#: Registered windows, keyed by the bounds the harness records, so a run cannot
#: be filed under a window it was not fitted on.
WINDOWS = {("2026-04-30", "2026-06-12"): "spring", ("2025-12-06", "2026-01-18"): "winter"}

#: The registered scope. Read from here and never from what the inputs turned
#: out to contain: a group that produced no file has to shortfall the count and
#: say so, rather than silently narrowing the read (ABL-322/ABL-378's rule, in a
#: harness that has no gate).
#:
#: `("price", "winter")` is deliberately absent. AT and DE price are 67.3%
#: covered in that window behind ingest holes of 1,651 h and 1,309 h, and
#: `create_lag_features` shifts by rows, so the fortnight after a hole carries
#: lags reaching across it - inside the holdout on AT. Registered as rejected.
REGISTERED_GROUPS = (("load", "spring"), ("price", "spring"), ("load", "winter"))

#: The primary metric block, and the two registered secondary ones. `all` decides
#: the verdict; the subsets are reported beside it and decide nothing.
PRIMARY_SUBSET = "all"
SECONDARY_SUBSETS = ("holiday", "holiday_affected")

#: Reported beside every cell, never a bar (ABL-389). The literal D-7 first, then
#: the four model-free predictors in the canonical module's order.
REFERENCES = ("baseline_seasonal_naive_d7", "constant_causal", "constant_oracle",
              "climatology_causal", "climatology_oracle")

#: How many of the 8 seeds must agree in sign for a cell to be material.
MATERIAL_K = 7


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p for `k` successes in `n` at p0 = 0.5.

    Written out rather than pulled from scipy because it is four lines and this
    repo's `.venv` should not gain a dependency for them. n = 8: 8/8 -> 0.0078,
    7/8 -> 0.0703, 6/8 -> 0.2891.
    """
    tail = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n)


def _cell(block: dict, country: str, algorithm: str, subset: str) -> dict:
    """One (country, algorithm) cell for one metric subset, or None if incomplete.

    A missing seed makes the cell `None` rather than a mean over whatever seeds
    happened to run: a paired statistic over an unequal set of pairs is not the
    registered statistic.
    """
    arms = block["arms"]
    paired = []
    for seed in SEEDS:
        hol, nohol = arms.get(f"{HOLIDAY_ARM}@{seed}"), arms.get(f"{NO_HOLIDAY_ARM}@{seed}")
        if hol is None or nohol is None:
            return None
        a, b = hol.get(subset, {}), nohol.get(subset, {})
        if not a.get("n") or not b.get("n"):
            return None
        paired.append((seed, a["mae_mw"], b["mae_mw"]))

    hol_mae = [h for _, h, _ in paired]
    nohol_mae = [b for _, _, b in paired]
    deltas = [100.0 * (h - b) / b for _, h, b in paired]
    k = sum(1 for d in deltas if d < 0)

    if k >= MATERIAL_K:
        d, favours = 1, "holidays"
    elif k <= len(SEEDS) - MATERIAL_K:
        d, favours = -1, "no_holidays"
    else:
        d, favours = 0, "inconclusive"

    return {
        "country": country, "algorithm": algorithm, "subset": subset,
        "n_scored": arms[f"{HOLIDAY_ARM}@{SEEDS[0]}"][subset]["n"],
        "hol_mean": statistics.fmean(hol_mae), "nohol_mean": statistics.fmean(nohol_mae),
        "deltas": deltas,
        "delta_mean": statistics.fmean(deltas), "delta_sd": statistics.stdev(deltas),
        "delta_min": min(deltas), "delta_max": max(deltas),
        "k_favouring_holidays": k, "sign_test_p": sign_test_p(k, len(SEEDS)),
        # The unpaired quantity a one-seed read would have been stuck with. It is
        # here so a sign-consistent 0.2% effect cannot be read as if it were the
        # several percent a single fit can move on its own (ABL-376).
        "hol_seed_spread_pct": 100.0 * (max(hol_mae) - min(hol_mae)) / statistics.fmean(hol_mae),
        "nohol_seed_spread_pct":
            100.0 * (max(nohol_mae) - min(nohol_mae)) / statistics.fmean(nohol_mae),
        "d": d, "material": d != 0, "favours": favours,
    }


def _references(block: dict, subset: str) -> dict:
    """What the holiday arm's seed-mean MAE is worth against the free predictors.

    ABL-381's standing ask: a margin quoted only against D-7, or only against a
    flat line, flatters a model on any diurnal series, and load and price are
    both strongly diurnal. Each reference carries its own `n`, because a
    climatology is 24 levels and can be partially measurable - two MAEs scored on
    different rows are not the same measurement.
    """
    arms = block["arms"]
    challenger = statistics.fmean(
        arms[f"{HOLIDAY_ARM}@{s}"][subset]["mae_mw"] for s in SEEDS
        if f"{HOLIDAY_ARM}@{s}" in arms
    )
    out = {"challenger_mae": challenger, "references": {}}
    for name in REFERENCES:
        # D-7 is the harness's own baseline block; the other four come from the
        # `model_free_reference` block `attach_model_free_references` filled.
        holder = (block if name == "baseline_seasonal_naive_d7"
                  else block.get("model_free_reference", {}))
        scores = holder.get(name, {}).get(subset)
        if not scores or not scores.get("n"):
            out["references"][name] = None
            continue
        out["references"][name] = {
            "mae": scores["mae_mw"], "n": scores["n"],
            # Skill in the ordinary sense: positive means the fitted model beats
            # the free predictor. Quoted as a share of the REFERENCE's own error,
            # which is what "beats it by X%" means.
            "skill_pct": 100.0 * (scores["mae_mw"] - challenger) / scores["mae_mw"],
            "n_matches_challenger": scores["n"] == arms[f"{HOLIDAY_ARM}@{SEEDS[0]}"][subset]["n"],
        }
    return out


def _gain_decomposition(primary: list, affected: list) -> list:
    """Where the all-hours gain comes from, as arithmetic on the registered subsets.

    Not a new statistic and not a test — `holiday_affected` and `ordinary`
    partition the holdout, and MAE times n is a sum of absolute errors, so
    `total_gain = affected_gain + ordinary_gain` exactly. This just states the
    split, because it is the internal check that decides whether a headline is
    credible: if the four holiday features are doing what the mechanism says,
    nearly all of the gain has to land on the rows they can distinguish. A gain
    spread evenly over ordinary rows would be an effect looking for an
    explanation.
    """
    by_cell = {(c["algorithm"], c["country"]): c for c in affected}
    out = []
    for c in primary:
        sub = by_cell.get((c["algorithm"], c["country"]))
        if sub is None:
            continue
        total = (c["nohol_mean"] - c["hol_mean"]) * c["n_scored"]
        part = (sub["nohol_mean"] - sub["hol_mean"]) * sub["n_scored"]
        out.append({
            "country": c["country"], "algorithm": c["algorithm"],
            "total_gain_abs": total, "holiday_affected_gain_abs": part,
            "holiday_affected_share_of_rows_pct": 100.0 * sub["n_scored"] / c["n_scored"],
            # A share of a NEGATIVE total is not a share of anything. Where the
            # holiday arm is worse overall there is no gain to apportion, and
            # printing "9.5%" there would read as "most of the gain went
            # elsewhere" rather than "there was no gain". None, and the table
            # says net loss.
            "holiday_affected_share_of_gain_pct":
                (100.0 * part / total) if total > 0 else None,
            "net_loss": total <= 0,
        })
    return out


def _group_verdict(cells: list) -> dict:
    total = sum(c["d"] for c in cells)
    n_material = sum(1 for c in cells if c["material"])
    if total >= 4:
        verdict = "HELP"
    elif total <= -4:
        verdict = "HARM"
    elif n_material <= 2:
        verdict = "NO_EFFECT"
    else:
        verdict = "MIXED"
    return {
        "sum_d": total, "n_cells": len(cells), "n_material": n_material,
        "n_favouring_holidays": sum(1 for c in cells if c["d"] == 1),
        "n_favouring_exclusion": sum(1 for c in cells if c["d"] == -1),
        "verdict": verdict,
    }


RECOMMENDATION = {
    "HELP": ("KEEP", "Registered mapping. get_feature_columns() should keep the four names for "
                     "this type. THIS IS A FINDING AND NOT A PROMOTION: no serving-registry "
                     "change and no retrain follows from this issue - that is the CEO's "
                     "decision. Report the per-country size, and that 24 countries of this "
                     "type serve without them today."),
    "HARM": ("EXCLUDE", "Registered mapping, and the harm is measured."),
    "NO_EFFECT": ("EXCLUDE", "Registered mapping: exclude on parsimony, AND record that "
                             "src/features.py's 'high impact for load forecasting' comment is "
                             "contradicted by measurement and should be corrected."),
    # The second clause above is load-specific and the registration wrote it
    # assuming NO_EFFECT would be the LOAD outcome. It fired on price instead,
    # where the comment says nothing and load read HELP - so on price that clause
    # is inapplicable, not merely unhelpful. Recorded as a visible correction in
    # experiments/ABL393/config.json (corrections_after_the_fit) rather than
    # edited out of the mapping, and the recommendation itself is untouched: the
    # registered action for NO_EFFECT is EXCLUDE either way.
    "NO_EFFECT@price": ("EXCLUDE", "Registered mapping's action, unchanged: exclude the four "
                                   "names from the PRICE list on parsimony. The mapping's "
                                   "second clause is about src/features.py's 'high impact for "
                                   "LOAD forecasting' comment and does not apply here - load "
                                   "read HELP, so that comment is vindicated, not refuted."),
    "MIXED": ("REPORT", "Registered mapping: report the disagreement and name what a further "
                        "read must measure. Exclude on parsimony only if no cell shows a "
                        "material HELP."),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="reports/abl_393_holiday_verdict_tables")
    args = parser.parse_args()

    payloads = {}
    for path in args.inputs:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        window = WINDOWS.get((blob["holdout_start"], blob["holdout_end"]))
        if window is None:
            parser.error(
                f"{path} was fitted on {blob['holdout_start']}..{blob['holdout_end']}, which is "
                f"not a registered window. Registered: {sorted(WINDOWS.values())}"
            )
        payloads[(blob["forecast_type"], window, blob["force_algorithm"])] = blob

    groups, missing = {}, []
    for forecast_type, window in REGISTERED_GROUPS:
        cells, secondary, references = [], {s: [] for s in SECONDARY_SUBSETS}, []
        for (t, w, algorithm), blob in sorted(payloads.items()):
            if (t, w) != (forecast_type, window):
                continue
            for country, block in blob["countries"].items():
                cell = _cell(block, country, algorithm, PRIMARY_SUBSET)
                if cell is None:
                    missing.append(f"{forecast_type}/{window}/{algorithm}/{country}")
                    continue
                cells.append(cell)
                for subset in SECONDARY_SUBSETS:
                    sub = _cell(block, country, algorithm, subset)
                    if sub is not None:
                        secondary[subset].append(sub)
                references.append({
                    "country": country, "algorithm": algorithm,
                    **_references(block, PRIMARY_SUBSET),
                })
        groups[f"{forecast_type}/{window}"] = {
            "forecast_type": forecast_type, "window": window,
            "cells": cells, "references": references,
            "secondary": {s: {"cells": c, **_group_verdict(c)} for s, c in secondary.items()},
            "gain_decomposition": _gain_decomposition(cells, secondary["holiday_affected"]),
            **_group_verdict(cells),
        }

    # Type verdicts. Load's replication can weaken and never create.
    type_verdicts = {"price": groups["price/spring"]["verdict"]}
    spring, winter = groups["load/spring"]["verdict"], groups["load/winter"]["verdict"]
    downgraded = {spring, winter} == {"HELP", "HARM"}
    type_verdicts["load"] = "MIXED" if downgraded else spring

    def rationale(forecast_type: str) -> tuple:
        """The registered mapping, with the one clause that is type-specific.

        `RECOMMENDATION[v]` is the registration verbatim. The `@type` override
        exists only where a registered rationale sentence names a forecast type
        and fires on a different one; it never changes the recommended action.
        """
        verdict = type_verdicts[forecast_type]
        return RECOMMENDATION.get(f"{verdict}@{forecast_type}", RECOMMENDATION[verdict])

    payload = {
        "issue": "ABL-393",
        "registration": "experiments/ABL393/config.json",
        "seeds": list(SEEDS),
        "contrast": {"holiday_arm": HOLIDAY_ARM, "no_holiday_arm": NO_HOLIDAY_ARM,
                     "sign": "delta < 0 means the holiday features are BETTER"},
        "registered_groups": [f"{t}/{w}" for t, w in REGISTERED_GROUPS],
        "cells_missing_a_seed": missing,
        "groups": groups,
        "load_spring_verdict": spring,
        "load_winter_replication_verdict": winter,
        "load_downgraded_by_replication_disagreement": downgraded,
        "type_verdicts": type_verdicts,
        "recommendations": {t: rationale(t)[0] for t in type_verdicts},
        "recommendation_rationale": {t: rationale(t)[1] for t in type_verdicts},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    unit = {"load": "MW", "price": "EUR/MWh"}
    lines = [
        "# ABL-393 verdict — do the four holiday features help load and price?",
        "",
        f"**load: {type_verdicts['load']} -> {rationale('load')[0]}** · "
        f"**price: {type_verdicts['price']} -> {rationale('price')[0]}**",
        "",
        f"Registered rule: `experiments/ABL393/config.json`. Paired by seed over "
        f"{len(SEEDS)} seeds {list(SEEDS)}; `delta = 100 * (holidays - no_holidays) / "
        f"no_holidays` on all-hours MAE, so **negative means the holiday features are "
        f"better**. A cell is material at {MATERIAL_K}/{len(SEEDS)} seeds agreeing in sign "
        f"(two-sided sign test p <= {sign_test_p(MATERIAL_K, len(SEEDS)):.4f}).",
        "",
    ]
    if missing:
        lines += [f"**Cells short of a registered seed and dropped:** {missing}", ""]

    for key, group in groups.items():
        u = unit[group["forecast_type"]]
        lines += [
            f"## {key} — **{group['verdict']}**",
            "",
            f"sum(d) {group['sum_d']:+d} over {group['n_cells']} cells, {group['n_material']} "
            f"material ({group['n_favouring_holidays']} favour keeping, "
            f"{group['n_favouring_exclusion']} favour excluding)"
            + (" · *registered replication: can weaken the load verdict, never create it*"
               if key == "load/winter" else ""),
            "",
            f"| cell | n | MAE holidays ({u}) | no holidays | paired delta (mean +- sd) | range | "
            "k/8 | sign p | own seed spread hol/nohol | d |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in sorted(group["cells"], key=lambda x: (x["algorithm"], x["country"])):
            lines.append(
                f"| {c['algorithm']}/{c['country']} | {c['n_scored']:,} "
                f"| {c['hol_mean']:,.2f} | {c['nohol_mean']:,.2f} "
                f"| {c['delta_mean']:+.2f}% +- {c['delta_sd']:.2f} "
                f"| {c['delta_min']:+.2f}% .. {c['delta_max']:+.2f}% "
                f"| {c['k_favouring_holidays']}/8 | {c['sign_test_p']:.4f} "
                f"| {c['hol_seed_spread_pct']:.2f}% / {c['nohol_seed_spread_pct']:.2f}% "
                f"| {c['d']:+d} |"
            )
        lines.append("")

        for subset in SECONDARY_SUBSETS:
            sub = group["secondary"][subset]
            if not sub["cells"]:
                continue
            lines += [
                f"### secondary (reported, never gating): `{subset}` rows — would read "
                f"**{sub['verdict']}**",
                "",
                f"sum(d) {sub['sum_d']:+d}, {sub['n_material']} material. A holiday is a few "
                "days in a 44-day window, so this is where the mechanism lives and the "
                "all-hours table above is where it is diluted.",
                "",
                f"| cell | n | MAE holidays ({u}) | no holidays | paired delta | k/8 | sign p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for c in sorted(sub["cells"], key=lambda x: (x["algorithm"], x["country"])):
                lines.append(
                    f"| {c['algorithm']}/{c['country']} | {c['n_scored']:,} "
                    f"| {c['hol_mean']:,.2f} | {c['nohol_mean']:,.2f} "
                    f"| {c['delta_mean']:+.2f}% | {c['k_favouring_holidays']}/8 "
                    f"| {c['sign_test_p']:.4f} |"
                )
            lines.append("")

        if group["gain_decomposition"]:
            lines += [
                "### Where the all-hours gain lands",
                "",
                "Arithmetic on the two registered subsets, not a further test: `holiday_affected` "
                "and `ordinary` partition the holdout and MAE x n is a sum of absolute errors, so "
                "the two gains add to the total exactly. This is the internal check on the "
                "headline — if these four features are doing what the mechanism says, the gain "
                "has to land on the rows they can distinguish, and a gain spread evenly over "
                "ordinary rows would be an effect in search of an explanation.",
                "",
                f"| cell | holiday-affected share of rows | ...of the gain | total error saved "
                f"(MAE x n, {u} x h) |",
                "|---|---:|---:|---:|",
            ]
            for row in sorted(group["gain_decomposition"],
                              key=lambda x: (x["algorithm"], x["country"])):
                share = row["holiday_affected_share_of_gain_pct"]
                lines.append(
                    f"| {row['algorithm']}/{row['country']} "
                    f"| {row['holiday_affected_share_of_rows_pct']:.1f}% "
                    f"| {'**net loss**' if share is None else f'{share:.1f}%'} "
                    f"| {row['total_gain_abs']:,.0f} |")
            lines.append("")

        lines += [
            "### What the fitted model is worth against no model at all",
            "",
            "ABL-381/ABL-389. `constant_*` is a flat line (fit-window mean / gate-window "
            "median), `climatology_*` the same per hour of day. Positive skill means the "
            "holiday arm beats the free predictor, as a share of that predictor's own error. "
            "**Check each `n`** — a climatology is 24 levels and can be partially measurable, "
            "and two MAEs scored on different rows are not the same measurement.",
            "",
            "| cell | holiday arm MAE | " + " | ".join(
                f"vs {r.replace('baseline_seasonal_naive_d7', 'D-7')}" for r in REFERENCES)
            + " |",
            "|---|---:|" + "---:|" * len(REFERENCES),
        ]
        for row in sorted(group["references"], key=lambda x: (x["algorithm"], x["country"])):
            cells = []
            for name in REFERENCES:
                ref = row["references"].get(name)
                if ref is None:
                    cells.append("Not measured")
                else:
                    flag = "" if ref["n_matches_challenger"] else f" (n={ref['n']:,})"
                    cells.append(f"{ref['skill_pct']:+.1f}%{flag}")
            lines.append(
                f"| {row['algorithm']}/{row['country']} | {row['challenger_mae']:,.2f} | "
                + " | ".join(cells) + " |")
        lines.append("")

        # ABL-380's lesson, restated in prose because nobody compares two numbers
        # in a seven-column table: a cell can carry a materially significant
        # feature effect and still be beaten by a table of hourly averages. Both
        # statements are true at once and the second qualifies the first.
        beaten = [
            (f"{r['algorithm']}/{r['country']}", name, r["references"][name],
             r["challenger_mae"])
            for r in sorted(group["references"], key=lambda x: (x["algorithm"], x["country"]))
            for name in ("constant_oracle", "climatology_oracle")
            if r["references"].get(name) and r["references"][name]["skill_pct"] < 0
        ]
        if beaten:
            lines += [
                "**A model-free predictor chosen with hindsight beats the fitted model in "
                f"{len(beaten)} cell(s) here.** That changes nothing above — this issue has no "
                "gate and the holiday effect is measured within the model, not against these — "
                "but it bounds what the cell is worth:",
                "",
            ]
            lines += [f"- {cell}: holiday arm {challenger:,.2f} vs `{name}` {ref['mae']:,.2f} "
                      f"{u} ({ref['skill_pct']:+.1f}%)"
                      for cell, name, ref, challenger in beaten]
            lines.append("")

    lines += [
        "## Type verdicts",
        "",
        f"- **load**: spring **{spring}**, winter replication **{winter}**"
        + (" — direction disagreement, downgraded to **MIXED**" if downgraded
           else " — no direction disagreement, spring stands")
        + f". Verdict **{type_verdicts['load']}** -> "
          f"**{rationale('load')[0]}**.",
        f"- **price**: spring **{groups['price/spring']['verdict']}** -> "
        f"**{rationale('price')[0]}**. Winter was not registered for "
        "price: AT and DE are 67.3% covered there behind 1,651 h and 1,309 h ingest holes.",
        "",
    ]
    for t in ("load", "price"):
        lines += [f"_{t}: {rationale(t)[1]}_", ""]

    markdown = "\n".join(lines)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
