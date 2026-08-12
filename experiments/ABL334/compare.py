#!/usr/bin/env python3
"""ABL-334 — read the three runs against each other and fire (or not) the trigger.

Pure arithmetic over three result JSONs. Fits no model, reads no database,
decides nothing on its own authority.

    recorded  experiments/ABL321/results_w1.json          pre-fix builder, commit de369a6
    control   experiments/ABL334/results_prefix_control.json  pre-fix builder, commit 11a5c44
    rerun     experiments/ABL334/results_rerun.json       post-fix builder, commit 11a5c44

Three differences, each answering a different question:

    recorded -> rerun     the CEO's registered revert trigger (Amendment 3).
                          Two variables: ABL-332 and nine other commits.
    recorded -> control   are those nine other commits inert on this path?
    control  -> rerun     the effect attributable to ABL-332 alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MATERIAL_PCT = 2.0
PRIMARY_TRUTH = "truth_gen"

SERVED_PAIRS = [
    ("AT", "solar"), ("BE", "solar"), ("DE", "solar"), ("FR", "solar"),
    ("AT", "wind_onshore"), ("BE", "wind_onshore"),
    ("DE", "wind_onshore"), ("FR", "wind_onshore"),
    ("BE", "wind_offshore"), ("FR", "wind_offshore"),
]
#: The three pairs whose regression decided criterion 2 in ABL-321 section 6.
ORIGINAL_REGRESSORS = [("AT", "solar"), ("DE", "wind_onshore"), ("BE", "wind_onshore")]

#: Amendment 3, registered addition 3. BE carries zero sub-hourly observations in
#: either source table over the builder's span, so `aggregate_renewable_to_hourly`
#: returns early and ABL-332 is a literal no-op for these three pairs. They are an
#: untreated control group: they must not move, and if they do it is not the fix.
UNTREATED = {("BE", "solar"), ("BE", "wind_onshore"), ("BE", "wind_offshore")}


def treated(pair: tuple[str, str]) -> bool:
    return pair not in UNTREATED


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_per_pair(result: dict, truth: str = PRIMARY_TRUTH) -> dict:
    """(country, stream) -> the aggregated D+2 row for `truth`."""
    return {(r["country"], r["stream"]): r
            for r in result["per_pair"] if r["truth"] == truth}


def rel(a: float, b: float) -> float | None:
    """Relative change from a to b, in percent."""
    if a in (None, 0) or b is None:
        return None
    return 100.0 * (b - a) / a


def verdict(r: float | None) -> str:
    if r is None:
        return "not measured"
    if r > MATERIAL_PCT:
        return "WORSE"
    if r < -MATERIAL_PCT:
        return "better"
    return "no material change"


def arm_table(title: str, left: dict, right: dict, arm: str,
              left_label: str, right_label: str) -> tuple[list[str], list[dict]]:
    """One arm, two runs, per serving pair."""
    lines = [f"### {title}", "",
             f"| pair | ABL-332 | n {left_label} | n {right_label} | {left_label} WAPE | "
             f"{right_label} WAPE | Δ pp | relative | verdict |",
             "|---|:--|---:|---:|---:|---:|---:|---:|:---|"]
    rows = []
    for country, stream in SERVED_PAIRS:
        lrow, rrow = left.get((country, stream)), right.get((country, stream))
        mark = "treated" if treated((country, stream)) else "**no-op**"
        if not lrow or not rrow:
            lines.append(f"| {country} {stream} | {mark} | — | — | — | — | — | — | missing |")
            continue
        lw = lrow[arm]["wape_pct"]
        rw = rrow[arm]["wape_pct"]
        r = rel(lw, rw)
        same_n = lrow["n"] == rrow["n"]
        flag = "" if same_n else " ⚠"
        lines.append(
            f"| {country} {stream} | {mark} | {lrow['n']:,}{flag} | {rrow['n']:,}{flag} | "
            f"{lw:.4f}% | {rw:.4f}% | {rw - lw:+.4f} | {r:+.2f}% | {verdict(r)} |")
        rows.append({"country": country, "stream": stream,
                     "treated": treated((country, stream)),
                     "left_wape": lw, "right_wape": rw,
                     "left_n": lrow["n"], "right_n": rrow["n"],
                     "same_n": same_n, "relative_pct": r, "verdict": verdict(r)})
    lines.append("")
    return lines, rows


def check_be_prediction(rows: list[dict], what: str) -> list[str]:
    """Amendment 3's registered prediction: the untreated pairs must not move."""
    untreated = [r for r in rows if not r["treated"]]
    if not untreated:
        return []
    moved = [r for r in untreated if r["relative_pct"] is None
             or abs(r["relative_pct"]) > 0.005]
    out = [f"**Registered prediction — the untreated BE pairs must not move ({what}).**", ""]
    for r in untreated:
        state = ("moved" if r in moved else "identical")
        out.append(f"- BE {r['stream']}: {r['left_wape']:.4f}% → {r['right_wape']:.4f}% "
                   f"({r['relative_pct']:+.4f}%) — **{state}**")
    out += ["", ("**Prediction holds.** ABL-332 is a no-op for BE and BE does not move, so "
                 "the harness is deterministic across these runs and any movement elsewhere "
                 "is attributable to the treated pairs' feature change."
                 if not moved else
                 f"**Prediction fails on {len(moved)} of {len(untreated)} untreated pairs.** "
                 "ABL-332 cannot have caused this — it is nondeterminism or another commit. "
                 "Movement on treated pairs cannot be cleanly attributed to the fix until "
                 "this is explained."), ""]
    return out


def criterion2_table(result: dict, truth: str = PRIMARY_TRUTH) -> list[str]:
    """arm A vs arm B within one run -- the source-switch question itself."""
    per = index_per_pair(result, truth)
    lines = ["| pair | n | before (ren) | after (gen) | Δ pp | relative | "
             "before skill vs D-7 | after skill vs D-7 | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---:|:---|"]
    for country, stream in SERVED_PAIRS:
        row = per.get((country, stream))
        if not row:
            lines.append(f"| {country} {stream} | — | — | — | — | — | — | — | missing |")
            continue
        a, b = row["before"]["wape_pct"], row["after"]["wape_pct"]
        r = row["relative_change_pct"]
        v = ("**after WORSE**" if r is not None and r > MATERIAL_PCT
             else "after better" if r is not None and r < -MATERIAL_PCT
             else "no material change")
        lines.append(
            f"| {country} {stream} | {row['n']:,} | {a:.4f}% | {b:.4f}% | "
            f"{b - a:+.4f} | {r:+.2f}% | {row['before_skill_vs_d7_pct']:+.1f}% | "
            f"{row['after_skill_vs_d7_pct']:+.1f}% | {v} |")
    lines.append("")
    return lines


def main() -> int:
    recorded = load(REPO / "experiments/ABL321/results_w1.json")
    rerun = load(REPO / "experiments/ABL334/results_rerun.json")
    control = load(REPO / "experiments/ABL334/results_prefix_control.json")

    if recorded is None or rerun is None:
        print("need at least results_w1.json and results_rerun.json", file=sys.stderr)
        return 1

    rec_i, run_i = index_per_pair(recorded), index_per_pair(rerun)
    ctl_i = index_per_pair(control) if control else None

    out = ["# ABL-334 — re-run of the ABL-321 source-switch A/B on the corrected builder", "",
           "Protocol: `experiments/ABL321/protocol.md` §1–§6 unchanged, Amendment 3 registered "
           "before the run. Window 1: fit 2026-01-14 → 2026-07-11, score 2026-07-11 → 2026-08-10 "
           "(exclusive), D+2 bands 24-36h / 36-48h / 48-64h, catboost `random_seed=42`, eight "
           "pre-registered vintages per target hour, common rows only.",
           f"Primary truth: `energy_generation`. Material threshold: {MATERIAL_PCT}% relative.", ""]

    # -- Deliverable 4 and the revert trigger -------------------------------
    out += ["## Deliverable 4 — the ABL-332 landing, measured on served accuracy", "",
            "Arm A is `energy_renewable`, the source that serves today. This is the "
            "comparison the CEO registered as the revert trigger.", ""]
    lines, d4 = arm_table(
        "Arm A: ABL-321 as recorded (pre-fix) → this re-run (post-fix)",
        rec_i, run_i, "before", "recorded", "re-run")
    out += lines

    worse = [r for r in d4 if r["relative_pct"] is not None and r["relative_pct"] > MATERIAL_PCT]
    better = [r for r in d4 if r["relative_pct"] is not None and r["relative_pct"] < -MATERIAL_PCT]
    flat = [r for r in d4 if r["relative_pct"] is not None and abs(r["relative_pct"]) <= MATERIAL_PCT]
    n_mismatch = [r for r in d4 if not r["same_n"]]

    out += ["**Trigger reading.** "
            f"{len(worse)} of 10 serving pairs materially worse, {len(better)} materially "
            f"better, {len(flat)} unmoved.", ""]
    if worse:
        out.append("Materially worse: " + ", ".join(
            f"{r['country']} {r['stream']} ({r['relative_pct']:+.2f}%)" for r in worse) + ".")
    if better:
        out.append("Materially better: " + ", ".join(
            f"{r['country']} {r['stream']} ({r['relative_pct']:+.2f}%)" for r in better) + ".")
    out.append("")
    fired = len(worse) >= 2
    out += [f"**TRIGGER {'FIRES' if fired else 'DOES NOT FIRE'}** — "
            + ("≥2 pairs materially worse. Per Amendment 3 this is a revert recommendation "
               "on the ABL-332 merge, it goes to the CEO, and criterion 2 is NOT interpreted "
               "in this run."
               if fired else
               "0 or 1 pair regressed. The ABL-332 landing is verified on served accuracy "
               "and the criterion-2 re-read proceeds."), ""]
    if n_mismatch:
        out += ["⚠ Sample size changed between runs for: " + ", ".join(
            f"{r['country']} {r['stream']} ({r['left_n']:,}→{r['right_n']:,})"
            for r in n_mismatch) + ". Those cells are not scored on identical rows.", ""]

    out += check_be_prediction(d4, "recorded → re-run")

    treated_worse = [r for r in worse if r["treated"]]
    untreated_worse = [r for r in worse if not r["treated"]]
    if fired:
        out += ["**Composition of the trigger.** "
                f"{len(treated_worse)} of the regressing pairs are ABL-332-treated and "
                f"{len(untreated_worse)} are pairs ABL-332 cannot touch. Only 7 of the 10 "
                "serving pairs are capable of moving at all.", ""]

    gap = load(REPO / "experiments/ABL334/truth_convention_gap.json")
    if gap:
        out += ["### The convention floor, for scale (measured before any result was read)", "",
                "The hourly mean scored as a forecast of the `:00` sample — what this harness "
                "charges a *perfect* hourly-mean predictor, with no model in the loop:", "",
                "| pair | convention floor WAPE | arm A recorded WAPE |", "|---|---:|---:|"]
        for p in gap["pairs"]:
            key = (p["country"], p["stream"])
            src = p["sources"].get("energy_renewable", {})
            floor = src.get("wape_pct" if "wape_pct" in src else "wape_mean_vs_00_pct")
            rec = rec_i.get(key)
            if floor is None:
                continue
            out.append(f"| {p['country']} {p['stream']} | {floor:.4f}% | "
                       + (f"{rec['before']['wape_pct']:.4f}% |" if rec else "— |"))
        out += ["", "Post-fix the fitted target is the hourly mean while truth stays the `:00` "
                "sample, so this floor is charged to the post-fix arm and not to the pre-fix "
                "one. It is 4–9 % absolute on every treated pair and exactly 0 % on BE — "
                "multiples of the 2.0 % relative margin the trigger reads.", ""]

    # -- Attribution --------------------------------------------------------
    out += ["## Attribution — is ABL-332 actually the cause?", ""]
    if ctl_i is None:
        out += ["Isolation control not present. Attribution unresolved.", ""]
    else:
        chk = control["meta"].get("abl334_control", {}).get("neutralisation_check")
        out += [f"Control neutralisation self-check: `{chk}`.", ""]
        lines, drift = arm_table(
            "Are the other nine commits inert? ABL-321 recorded → control (both pre-fix builder)",
            rec_i, ctl_i, "before", "recorded", "control")
        out += lines
        moved = [r for r in drift if r["relative_pct"] is not None
                 and abs(r["relative_pct"]) > 0.005]
        out += [("**The other nine commits are inert on this path** — the control reproduces "
                 "the recorded arm A on every pair, so the trigger comparison is one-variable "
                 "after all and needs no attribution caveat."
                 if not moved else
                 f"**The control does not reproduce the recorded arm A on {len(moved)} pair(s)**, "
                 "so part of the recorded→re-run movement is not ABL-332's. "
                 + ", ".join(f"{r['country']} {r['stream']} ({r['relative_pct']:+.3f}%)"
                             for r in moved) + "."), ""]

        out += check_be_prediction(drift, "recorded → control")

        lines, clean = arm_table(
            "ABL-332 alone: control (pre-fix) → re-run (post-fix), same commit",
            ctl_i, run_i, "before", "control", "re-run")
        out += lines
        cw = [r for r in clean if r["relative_pct"] is not None and r["relative_pct"] > MATERIAL_PCT]
        cb = [r for r in clean if r["relative_pct"] is not None and r["relative_pct"] < -MATERIAL_PCT]
        out += [f"Attributable to ABL-332 alone: {len(cw)} pair(s) materially worse, "
                f"{len(cb)} materially better.", ""]
        out += check_be_prediction(clean, "control → re-run")

    # -- Deliverables 1-3 ---------------------------------------------------
    out += ["## Deliverables 1–3 — criterion 2 re-read on the corrected builder", ""]
    if fired:
        out += ["**Not interpreted.** Amendment 3 stops here when the revert trigger fires. "
                "The table is printed for completeness only and carries no verdict.", ""]
    out += ["### Post-fix re-run, arm A (`energy_renewable`) vs arm B (`energy_generation`)", ""]
    out += criterion2_table(rerun)

    out += ["### The three pairs that decided criterion 2 in ABL-321", "",
            "| pair | ABL-321 relative | re-run relative | ABL-321 verdict | re-run verdict |",
            "|---|---:|---:|:---|:---|"]
    flips = []
    for country, stream in ORIGINAL_REGRESSORS:
        old, new = rec_i.get((country, stream)), run_i.get((country, stream))
        if not old or not new:
            continue
        o, n = old["relative_change_pct"], new["relative_change_pct"]
        ov, nv = verdict(o), verdict(n)
        out.append(f"| {country} {stream} | {o:+.2f}% | {n:+.2f}% | {ov} | {nv} |")
        if ov != nv:
            flips.append((country, stream, o, n))
    out.append("")

    still_worse = [p for p in SERVED_PAIRS
                   if (r := run_i.get(p)) and r["relative_change_pct"] is not None
                   and r["relative_change_pct"] > MATERIAL_PCT]
    out += [f"**Criterion 2 on the re-run:** {len(still_worse)} of 10 serving pairs materially "
            f"worse under `energy_generation`"
            + (" — " + ", ".join(f"{c} {s}" for c, s in still_worse) if still_worse else "")
            + ".", "",
            ("**Verdict HOLDS** — the switch still regresses a serving pair on the corrected "
             "instrument, so ABL-321 §6's withholding stands unchanged."
             if still_worse else
             "**Verdict FLIPS** — no serving pair is materially worse on the corrected "
             "instrument. Per deliverable 3 this is a CEO decision to reopen: "
             "`RENEWABLE_TYPE_SOURCE_TABLE` is not flipped and the withholding test is not "
             "edited in this run."), ""]
    if flips:
        out += ["Per-pair verdict changes among the original three: " + "; ".join(
            f"{c} {s} {ov:+.2f}% → {nv:+.2f}%" for c, s, ov, nv in flips) + ".", ""]

    # -- secondary truth ----------------------------------------------------
    out += ["### Secondary truth (`energy_renewable`) — robustness", "",
            "A conclusion is reported as robust only where the two truths agree.", ""]
    out += criterion2_table(rerun, "truth_ren")

    report = "\n".join(out)
    dest = REPO / "reports/abl_334_rerun_findings.md"
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
