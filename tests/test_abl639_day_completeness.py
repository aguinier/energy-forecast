"""ABL-639: a partial target day must not be weighted like a whole one.

`paired_daily` gives every country-day one `a - b`, whatever the number of
hours that day was scored on, and `k = len(d)` -- which keys `T_CRIT` -- counts
it as a full observation. So a country-day holding 2 surviving hours entered
ABL-607's paired interval with the same weight as a 24-hour day, and because
`panel_a` is an inner merge the truncation is per-country: the countries were
not scored over the same window.

Two properties are pinned here, and they pull in opposite directions, which is
why neither can be left to inspection:

1. **The default changes nothing.** The screen is off at 0.0 and the pack's
   published protocol is untouched. Pinned against a verbatim copy of the
   pre-ABL-639 function rather than against a stored number, so the claim is
   "identical to the old code" and not "identical to a vintage of the replica"
   -- the replica moves under a re-read (ABL-619) and a pinned number would go
   red for a reason that has nothing to do with this change.

2. **The screen is a ratio, never a constant 24.** This is the trap. ABL-607's
   window runs `2026-08-13 08:00` -> `2026-08-28 00:00`, so its first day
   legitimately expects 16 hours and its last legitimately expects **1**. A
   "require 24 hours" screen would drop both ends for all 24 countries at once
   and say nothing; and that terminal hour is the single largest difference the
   pack recorded between its two reads. The regression would be severe and
   near-invisible, so it gets a test rather than a comment.

The synthetic panels below are shaped on ABL-607's real window and reproduce
the LV truncation the issue measured on the live replica (23, 22 and 20 hours
on 08-16/18/19, `k` 16 -> 13). Synthetic on purpose: a rule pinned against live
data stops being a test the day the data moves. The live numbers are published
in `section_k_day_completeness` of the run's own record.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl607_d2_load_diagnosis import (  # noqa: E402
    NOT_EVALUABLE,
    SENSITIVITY_MIN_DAY_COMPLETENESS,
    T_CRIT,
    day_completeness,
    hours_expected_per_day,
    paired_daily,
    readable_cells,
    wape,
)

SCRIPT = Path(__file__).parent.parent / "scripts" / "abl607_d2_load_diagnosis.py"

#: ABL-607's own scored window, and the shape of the trap: the first day can
#: only ever carry 16 hours and the last only 1.
WINDOW_START = "2026-08-13 08:00"
WINDOW_END = "2026-08-28 00:00"
WINDOW_DAYS = 16


# --------------------------------------------------------------------------
# the pre-ABL-639 function, copied verbatim from `origin/main`
# --------------------------------------------------------------------------
#
# The control arm of property 1. Copied rather than imported because the point
# is to compare against code that no longer exists: importing the current
# `paired_daily` and calling it twice would compare it to itself.


def _paired_daily_pre_abl639(panel, arm_a, arm_b):
    rows = []
    panel = panel.copy()
    panel["day"] = panel["target"].dt.normalize()
    for country, grp in panel.groupby("country_code", sort=True):
        diffs = []
        for _, day in grp.groupby("day"):
            a = wape((day[arm_a] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            b = wape((day[arm_b] - day["actual"]).to_numpy(), day["actual"].to_numpy())
            diffs.append(a - b)
        d = np.array(diffs, dtype=float)
        k = len(d)
        mean = float(d.mean())
        if k > 1:
            se = float(d.std(ddof=1) / np.sqrt(k))
            tcrit = T_CRIT.get(k, 2.086)
            lo, hi = mean - tcrit * se, mean + tcrit * se
        else:
            lo = hi = float("nan")
        rows.append({"country": country, "k_days": k, "mean_daily_wape_diff": mean,
                     "ci_lo": lo, "ci_hi": hi,
                     "readable": bool(k > 1 and (lo > 0 or hi < 0)),
                     "days_arm_a_better": int((d < 0).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def _panel(countries=("AT", "LV"), start=WINDOW_START, end=WINDOW_END, seed=7):
    """A fully covered scored panel over ABL-607's window shape.

    A diurnal actual with noise, an ML arm carrying a positive bias and a D-7
    arm that is unbiased but noisier -- enough structure that the daily WAPE
    differences vary day to day, so a t-interval on them is not degenerate.
    """
    span = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(seed)
    frames = []
    for i, cc in enumerate(countries):
        base = (1000.0 + 250.0 * i
                + 180.0 * np.sin(np.arange(len(span)) * 2 * np.pi / 24)
                + rng.normal(0, 25, len(span)))
        frames.append(pd.DataFrame({
            "country_code": cc, "target": span, "actual": base,
            "ml_band": base + rng.normal(45, 55, len(span)),
            "d7_naive": base + rng.normal(0, 70, len(span)),
        }))
    return pd.concat(frames, ignore_index=True)


def _truncate(panel, drops):
    """Keep only the first `n` hours of the named country-days.

    `drops` is `{(country, "YYYY-MM-DD"): hours_to_keep}`. Mirrors what an
    outage leaves behind: the day is present but short, which is the dangerous
    case -- a country retaining *zero* hours contributes no daily diff at all
    and is already safe.
    """
    day = panel["target"].dt.normalize()
    keep = pd.Series(True, index=panel.index)
    for (country, when), n_keep in drops.items():
        rows = panel.index[(panel["country_code"] == country)
                           & (day == pd.Timestamp(when))]
        assert len(rows) > n_keep, (
            f"{country} {when} has {len(rows)} hours; truncating to {n_keep} "
            f"would not shorten it and the test would prove nothing")
        keep.loc[rows[n_keep:]] = False
    return panel[keep].reset_index(drop=True)


#: The truncation the issue measured on the live replica, for LV, inside
#: ABL-607's own published window.
LV_TRUNCATION = {("LV", "2026-08-16"): 23,
                 ("LV", "2026-08-18"): 22,
                 ("LV", "2026-08-19"): 20}


# --------------------------------------------------------------------------
# 1. the default screens nothing
# --------------------------------------------------------------------------

def test_the_default_threshold_reproduces_the_pre_abl639_function_exactly():
    """The protocol-neutrality claim, checked against the old code.

    `check_exact=True` on purpose. The daily differences are summed in
    iteration order, so a reordering of the day loop would move the mean in the
    last bits -- invisible to a tolerance comparison, and enough to flip a
    marginal `ci_lo` like LV's +0.64 in a later re-read.
    """
    panel = _truncate(_panel(), LV_TRUNCATION)

    reference = _paired_daily_pre_abl639(panel, "ml_band", "d7_naive")
    current = paired_daily(panel, "ml_band", "d7_naive")

    pd.testing.assert_frame_equal(
        current[reference.columns], reference, check_exact=True)
    assert (current["k_days_screened_out"] == 0).all()


def test_that_comparison_is_not_vacuous():
    """The panel above must actually contain the defect, or the equality is
    just two functions agreeing on clean data."""
    panel = _truncate(_panel(), LV_TRUNCATION)
    table = day_completeness(panel)

    short = table[table["is_short"]]
    assert set(short["country"]) == {"LV"}
    assert sorted(short["hours_present"]) == [20, 22, 23]
    # ...and the unscreened interval really does absorb them at full weight.
    assert paired_daily(panel, "ml_band", "d7_naive").set_index("country").loc[
        "LV", "k_days"] == WINDOW_DAYS


def test_the_flag_defaults_to_off():
    """Read out of the source, because "the default is 0.0" is the protocol
    claim itself -- a run at any other default is a different published
    protocol wearing the same command line."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    defaults = [
        {kw.arg: kw.value for kw in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args and getattr(node.args[0], "value", None)
        == "--min-day-completeness"
    ]
    assert len(defaults) == 1, "no --min-day-completeness argument in the script"
    assert defaults[0]["default"].value == 0.0
    assert defaults[0]["type"].id == "float"


# --------------------------------------------------------------------------
# 2. the trap: the screen is a ratio against the window, never a constant 24
# --------------------------------------------------------------------------

def test_the_windows_partial_end_days_expect_fewer_than_24_hours():
    """The denominator, stated. 16 on the first day, 1 on the last."""
    expected = hours_expected_per_day(_panel())

    assert expected.loc[pd.Timestamp("2026-08-13")] == 16
    assert expected.loc[pd.Timestamp("2026-08-28")] == 1
    assert expected.loc[pd.Timestamp("2026-08-20")] == 24
    assert len(expected) == WINDOW_DAYS
    assert expected.sum() == 16 + 14 * 24 + 1


def test_a_complete_days_only_screen_keeps_both_partial_end_days():
    """**The trap.** At the strictest threshold there is on a fully covered
    panel, nothing may be dropped -- including the terminal target hour, which
    is the single largest difference ABL-607 recorded between its two reads.

    The last two assertions are the counterfactual: a constant-24 screen would
    have taken both end days from every country, and `k` would have fallen
    without one number in the record moving to say so.
    """
    panel = _panel()

    screened = paired_daily(panel, "ml_band", "d7_naive",
                            SENSITIVITY_MIN_DAY_COMPLETENESS)

    assert (screened["k_days"] == WINDOW_DAYS).all()
    assert (screened["k_days_screened_out"] == 0).all()
    assert (screened["k_days_short"] == 0).all()
    # every country-day is complete, terminal hour included
    table = day_completeness(panel).set_index(["country", "day"])
    assert not table["is_short"].any()
    assert table.loc[("LV", "2026-08-28"), "hours_present"] == 1
    assert table.loc[("LV", "2026-08-28"), "completeness"] == 1.0

    expected = hours_expected_per_day(panel)
    would_be_dropped = sorted(str(d.date()) for d, n in expected.items() if n < 24)
    assert would_be_dropped == ["2026-08-13", "2026-08-28"], (
        "a constant-24 screen would drop exactly these two legitimate days "
        "for every country; the ratio screen above drops neither")


def test_the_terminal_day_survives_even_when_an_earlier_day_is_screened_out():
    """The two behaviours together, since it is their combination that is the
    requirement: the short days go and the legitimately-partial ends stay."""
    panel = _truncate(_panel(), LV_TRUNCATION)

    screened = paired_daily(panel, "ml_band", "d7_naive",
                            SENSITIVITY_MIN_DAY_COMPLETENESS)
    kept = day_completeness(panel).set_index(["country", "day"])

    lv = screened.set_index("country").loc["LV"]
    assert lv["k_days"] == WINDOW_DAYS - 3          # the issue's 16 -> 13
    assert lv["k_days_screened_out"] == 3
    assert not kept.loc[("LV", "2026-08-13"), "is_short"]
    assert not kept.loc[("LV", "2026-08-28"), "is_short"]


# --------------------------------------------------------------------------
# 3. the screen is per-country, and measured against one shared window
# --------------------------------------------------------------------------

def test_only_the_short_country_loses_days():
    """`panel_a` is an inner merge, so the truncation is heterogeneous. A
    screen that dropped a day fleet-wide because one country was short would
    throw away 23 good country-days to fix one."""
    panel = _truncate(_panel(), LV_TRUNCATION)

    primary = paired_daily(panel, "ml_band", "d7_naive").set_index("country")
    screened = paired_daily(panel, "ml_band", "d7_naive", 1.0).set_index("country")

    assert primary.loc["LV", "k_days"] == WINDOW_DAYS
    assert primary.loc["LV", "k_days_short"] == 3
    assert screened.loc["LV", "k_days"] == WINDOW_DAYS - 3
    assert screened.loc["AT", "k_days"] == WINDOW_DAYS
    assert screened.loc["AT", "k_days_screened_out"] == 0
    # AT's interval is untouched by LV's truncation, at full precision.
    assert screened.loc["AT", "ci_lo"] == primary.loc["AT", "ci_lo"]


def test_the_window_is_fleet_wide_so_a_truncated_country_stays_visible():
    """The vacuity failure this design avoids.

    Derived per country, the window would be defined by the very truncation it
    is meant to detect: a country missing the window's first 8 hours would have
    a "first day" of 8 hours, expect 8, and score 100% complete. One shared
    span is what makes the ratio able to fire at all.
    """
    panel = _panel()
    late = panel[~((panel["country_code"] == "LV")
                   & (panel["target"] < pd.Timestamp("2026-08-13 16:00")))]

    table = day_completeness(late.reset_index(drop=True)).set_index(["country", "day"])

    assert table.loc[("LV", "2026-08-13"), "hours_expected"] == 16
    assert table.loc[("LV", "2026-08-13"), "hours_present"] == 8
    assert table.loc[("LV", "2026-08-13"), "is_short"]
    assert table.loc[("AT", "2026-08-13"), "hours_present"] == 16
    assert not table.loc[("AT", "2026-08-13"), "is_short"]


def test_every_country_day_is_recorded_not_only_the_short_ones():
    """The diagnostic is emitted unconditionally: a reader must be able to see
    that a day was checked and found complete, not only that no alarm fired."""
    panel = _truncate(_panel(("AT", "LV", "PL")), LV_TRUNCATION)

    table = day_completeness(panel)

    assert len(table) == 3 * WINDOW_DAYS
    assert set(table.columns) == {"country", "day", "hours_expected",
                                  "hours_present", "completeness", "is_short"}
    assert table["is_short"].sum() == 3
    assert table["completeness"].between(0, 1).all()


# --------------------------------------------------------------------------
# 4. the trade the screen makes, and the accounting that shows it
# --------------------------------------------------------------------------

def test_the_days_add_up_whatever_the_threshold():
    """`k_days + k_days_screened_out` is the day count either way, so a reader
    can tell a dropped day from a day that never existed."""
    panel = _truncate(_panel(), LV_TRUNCATION)

    for threshold in (0.0, 0.5, 0.9, 1.0):
        out = paired_daily(panel, "ml_band", "d7_naive", threshold)
        assert (out["k_days"] + out["k_days_screened_out"] == WINDOW_DAYS).all(), (
            f"days unaccounted for at threshold {threshold}")

    # 20/24 survives 0.5 and 0.9 but not 1.0; 23/24 survives 0.9.
    at_half = paired_daily(panel, "ml_band", "d7_naive", 0.5).set_index("country")
    assert at_half.loc["LV", "k_days"] == WINDOW_DAYS
    at_full = paired_daily(panel, "ml_band", "d7_naive", 1.0).set_index("country")
    assert at_full.loc["LV", "k_days"] == WINDOW_DAYS - 3


def test_screening_costs_precision_which_is_why_it_is_not_the_default():
    """The caveat, as arithmetic rather than as prose. `T_CRIT` is keyed on
    `k`, so every day the screen removes widens the interval before any change
    in the data is considered. That is a real cost, and it is the reason the
    screen is reported beside the primary instead of replacing it."""
    assert T_CRIT[13] > T_CRIT[16]
    assert SENSITIVITY_MIN_DAY_COMPLETENESS == 1.0


# --------------------------------------------------------------------------
# 5. the note agrees with the record it was written from
# --------------------------------------------------------------------------
#
# ABL-619 is the failure this guards: three merged texts went on describing a
# measurement the committed artifact did not carry, because the code moved and
# nothing regenerated the report. `reports/abl_639_day_completeness.md` states
# a conditional finding -- LV's readable loss does not survive the screen -- and
# the whole force of it is in numbers a later re-read will move. So the prose is
# derived from the record here rather than trusted, in both directions.

RECORD = (Path(__file__).parent.parent / "reports"
          / "abl_607_d2_load_diagnosis_completeness.json")
NOTE = Path(__file__).parent.parent / "reports" / "abl_639_day_completeness.md"


def _flat(text):
    """Prose with its line wrapping collapsed, so reflowing a paragraph does
    not turn a pin red -- otherwise the next person to rewrap this file learns
    to delete the test rather than to trust it."""
    return " ".join(text.split())


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sf(value, dp=2):
    """Signed, fixed precision, ASCII minus -- the note's own convention."""
    return f"{value:+.{dp}f}"


def test_the_note_quotes_the_record_it_was_written_from():
    """Every figure in the note, recomputed. A record that moves under a note
    left as it was is red, and so is a note that drifts from the record."""
    record = _record()
    prose = _flat(NOTE.read_text(encoding="utf-8"))
    meta = record["meta"]
    section = record["section_k_day_completeness"]

    # protocol
    assert f"**n = {meta['panel_a_n_scored_pairs']}** scored pairs" in prose
    assert (f"`{meta['window_start'][:16]}` → `{meta['window_end'][:16]}` "
            f"inclusive, {meta['target_days']} target days") in prose
    assert f"`zero_rows_dropped = {meta['zero_rows_dropped']}`" in prose
    assert f"defaults to `{meta['min_day_completeness']}`" in prose
    assert meta["min_day_completeness"] == 0.0, (
        "the committed record was produced with a screen applied; the note "
        "describes it as the unscreened default read")

    # the short-day counts, both panels
    for name, phrase in (("panel_a", "**{short} of {total} country-days"),
                         ("panel_g", "({short} of {total} on")):
        blk = section["panels"][name]
        assert phrase.format(short=blk["n_short_country_days"],
                             total=blk["n_country_days"]) in prose, (
            f"{name} carries {blk['n_short_country_days']} of "
            f"{blk['n_country_days']} short country-days; the note disagrees")

    # the two table rows
    rows = {c["country"]: c
            for c in section["primary_vs_sensitivity"]["section_a_ml_band_vs_d7"]}
    yes_no = {True: "yes", False: "no"}
    for cc in ("LV", "EE"):
        c = rows[cc]
        row = (f"| {cc} | {c['k_days_primary']} | {c['k_days_screened']} | "
               f"{_sf(c['mean_daily_wape_diff_primary'])} → "
               f"{_sf(c['mean_daily_wape_diff_screened'])} | "
               f"{_sf(c['ci_lo_primary'])} → {_sf(c['ci_lo_screened'])} | "
               f"{yes_no[c['readable_primary']]} → "
               f"{yes_no[c['readable_screened']]} |")
        assert row in prose, f"the note's {cc} row is not the record's:\n{row}"

    # the loser sets, as membership and as counts
    primary = section["primary_section_a_readable"]
    screened = section["sensitivity_section_a_readable"]
    assert (f"`{' '.join(primary['readable_losers'])}` "
            f"({len(primary['readable_losers'])})") in prose
    assert (f"`{' '.join(screened['readable_losers'])}` "
            f"({len(screened['readable_losers'])})") in prose
    assert primary["readable_winners"] == screened["readable_winners"] == ["GR"], (
        "the note says GR is the one readable winner under both arms")


def test_the_note_names_every_cell_whose_readability_moved():
    """The claim that makes this a finding rather than a caveat: LV and only
    LV. A count could not notice a second cell the note should name and does
    not, so the set is recomputed and compared as a set."""
    record = _record()
    prose = _flat(NOTE.read_text(encoding="utf-8"))
    rows = record["section_k_day_completeness"]["primary_vs_sensitivity"][
        "section_a_ml_band_vs_d7"]

    moved = {c["country"] for c in rows
             if c["readable_primary"] != c["readable_screened"]}
    assert moved == {"LV"}, (
        f"readability now moves on {sorted(moved)}; the note claims LV alone "
        f"and says no other cell's readability moves")
    assert "no other cell's readability moves" in prose

    short = {c["country"] for c in rows if c["k_days_short"]}
    assert short == {"EE", "LV"}, (
        f"country-days are now short on {sorted(short)}; the note says the "
        f"short days fall on two countries, EE and LV")
    assert "they fall on two countries" in prose


def test_the_note_states_the_terminal_days_expected_hours():
    """The trap, quoted. Both end-day figures come from the record's own
    expected-hours map, not from the window arithmetic repeated by hand."""
    record = _record()
    prose = _flat(NOTE.read_text(encoding="utf-8"))
    expected = record["section_k_day_completeness"]["panels"]["panel_a"][
        "hours_expected_per_day"]

    first, last = min(expected), max(expected)
    assert f"`{first}` can only carry {expected[first]} target hours" in prose
    assert f"`{last}` exactly **{expected[last]}**" in prose
    assert expected[last] == 1 and expected[first] == 16


def test_readable_cells_holds_nl_out():
    """The losers/winners split, reused by both arms so the primary and the
    sensitivity read cannot drift apart. NL is a basis mismatch, not skill."""
    paired = pd.DataFrame([
        {"country": "AT", "mean_daily_wape_diff": 2.0, "readable": True},
        {"country": "GR", "mean_daily_wape_diff": -1.5, "readable": True},
        {"country": "PL", "mean_daily_wape_diff": 3.0, "readable": False},
        {"country": "NL", "mean_daily_wape_diff": 9.0, "readable": True},
    ])

    assert "NL" in NOT_EVALUABLE
    assert readable_cells(paired) == {"readable_losers": ["AT"],
                                      "readable_winners": ["GR"]}


def _drop_day(panel, country, when):
    day = panel["target"].dt.normalize()
    return panel[~((panel["country_code"] == country)
                   & (day == pd.Timestamp(when)))].reset_index(drop=True)


def test_a_country_screened_below_two_days_reports_no_interval():
    """`k = 1` has no t-interval and `k = 0` has no mean either. Neither may
    raise, and neither may come back readable -- a screen that turned a
    marginal cell into a silently NaN-readable one would be worse than the
    defect it was added to fix.

    LV's terminal day is removed rather than truncated, because a day that
    expects one hour and has one hour is complete by construction and cannot
    be screened out. That is the trap holding, viewed from the other side.
    """
    base = _drop_day(_panel(("AT", "LV")), "LV", "2026-08-28")
    thin = _truncate(base, {("LV", f"2026-08-{d:02d}"): 1 for d in range(14, 28)})

    lv = paired_daily(thin, "ml_band", "d7_naive", 1.0).set_index("country").loc["LV"]
    assert lv["k_days"] == 1                 # 2026-08-13 alone, complete at 16/16
    assert lv["k_days_screened_out"] == 14
    assert not lv["readable"]
    assert np.isnan(lv["ci_lo"]) and np.isnan(lv["ci_hi"])
    assert not np.isnan(lv["mean_daily_wape_diff"])

    stripped = _truncate(thin, {("LV", "2026-08-13"): 10})
    lv0 = paired_daily(stripped, "ml_band", "d7_naive",
                       1.0).set_index("country").loc["LV"]
    assert lv0["k_days"] == 0
    assert lv0["k_days_screened_out"] == 15
    assert not lv0["readable"]
    assert np.isnan(lv0["mean_daily_wape_diff"])
    # ...and AT, which was never short, still has its full interval.
    at = paired_daily(stripped, "ml_band", "d7_naive",
                      1.0).set_index("country").loc["AT"]
    assert at["k_days"] == WINDOW_DAYS and at["readable"] in (True, False)
    assert not np.isnan(at["ci_lo"])
