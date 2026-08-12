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
             f"| pair | n {left_label} | n {right_label} | {left_label} WAPE | "
             f"{right_label} WAPE | Δ pp | relative | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|:---|"]
    rows = []
    for country, stream in SERVED_PAIRS:
        lrow, rrow = left.get((country, stream)), right.get((country, stream))
        if not lrow or not rrow:
            lines.append(f"| {country} {stream} | — | — | — | — | — | — | missing |")
            continue
        lw = lrow[arm]["wape_pct"]
        rw = rrow[arm]["wape_pct"]
        r = rel(lw, rw)
        same_n = lrow["n"] == rrow["n"]
        flag = "" if same_n else " ⚠"
        lines.append(
            f"| {country} {stream} | {lrow['n']:,}{flag} | {rrow['n']:,}{flag} | "
            f"{lw:.4f}% | {rw:.4f}% | {rw - lw:+.4f} | {r:+.2f}% | {verdict(r)} |")
        rows.append({"country": country, "stream": stream,
                     "left_wape": lw, "right_wape": rw,
                     "left_n": lrow["n"], "right_n": rrow["n"],
                     "same_n": same_n, "relative_pct": r, "verdict": verdict(r)})
    lines.append("")
    return lines, rows


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

        lines, clean = arm_table(
            "ABL-332 alone: control (pre-fix) → re-run (post-fix), same commit",
            ctl_i, run_i, "before", "control", "re-run")
        out += lines
        cw = [r for r in clean if r["relative_pct"] is not None and r["relative_pct"] > MATERIAL_PCT]
        cb = [r for r in clean if r["relative_pct"] is not None and r["relative_pct"] < -MATERIAL_PCT]
        out += [f"Attributable to ABL-332 alone: {len(cw)} pair(s) materially worse, "
                f"{len(cb)} materially better.", ""]

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
