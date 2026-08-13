"""ABL-403: the published numbers are the machine record's numbers.

`test_abl403_interaction_stats.py` checks the contrast machinery on synthetic
frames. This checks the other half -- that what the *report* and `CLAUDE.md`
say about the run matches `reports/abl_403_night_rule_interaction.json`, which
is the only artifact the 64 fits actually produced.

That gap is not hypothetical. The merged text quoted `exclusion_at_f27`'s
statistics (8/8 seeds, p = 0.0078) against `exclusion_at_f25`'s endpoints
(20.09% -> 12.97%, which is 7/8, p = 0.070), and separately double-rounded two
table cells by hand. Neither moved a verdict, and neither was reachable by any
test: the report is prose, and prose about a committed JSON file is exactly the
kind of claim that drifts from it without a conflict, a failing run, or anything
in `git status`.

The night-negative axis is pinned hardest because it is the axis the doctrine
now warns about. Its whole point is that an 8/8 sign test on that metric is
still inside its seed null, so a future edit that quotes it as a readable result
should fail here rather than in the next tranche's disposition.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent
RECORD = REPO / "reports" / "abl_403_night_rule_interaction.json"
REPORT = REPO / "reports" / "abl_403_night_rule_interaction.md"
DOCTRINE = REPO / "CLAUDE.md"

ARM_ORDER = ("f25_off", "f27_off", "f25_on", "f27_on")

# `## 4a. BG` table row label -> effects key. The daylight rows share a label
# stem with the gate-band rows, so both are keyed explicitly rather than
# derived, which is what makes a mislabelled row fail instead of matching the
# wrong metric.
BG_TABLE_ROWS = {
    "night MAE (MW)": "night_mae_mw",
    "night bias, pred - actual (MW)": "night_bias_mw",
    "night WAPE (%)": "night_wape_pct",
    "night rows predicted negative (%)": "pct_of_night_rows_negative",
    "WAPE 24-36h (%)": "24-36h|challenger_wape_pct",
    "WAPE 36-48h (%)": "36-48h|challenger_wape_pct",
    "WAPE 48-64h (%)": "48-64h|challenger_wape_pct",
    "daylight WAPE 24-36h (%)": "24-36h|daylight_challenger_wape_pct",
    "daylight WAPE 36-48h (%)": "36-48h|daylight_challenger_wape_pct",
    "daylight WAPE 48-64h (%)": "48-64h|daylight_challenger_wape_pct",
}


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bg(record):
    return next(c for c in record["countries"] if c["country"] == "BG")


@pytest.fixture(scope="module")
def report_text():
    return REPORT.read_text(encoding="utf-8")


def _number(cell: str) -> float:
    """One markdown table cell -> float, tolerating bold and a Unicode minus."""
    cleaned = cell.strip().strip("*").replace("−", "-").replace("+", "")
    return float(cleaned)


def _table_rows(text: str, heading: str) -> dict:
    """`label -> [cells]` for the first markdown table under `heading`."""
    section = text.split(heading, 1)[1]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        label = cells[0].strip("*").replace("−", "-")
        rows[label] = cells[1:]
    return rows


# --------------------------------------------------------------------------
# The 4a table is a hand transcription of the machine record. Pin every cell.
# --------------------------------------------------------------------------

def test_bg_arm_means_in_the_report_match_the_machine_record(bg, report_text):
    rows = _table_rows(report_text, "### 4a. BG")
    missing = [label for label in BG_TABLE_ROWS if label not in rows]
    assert not missing, f"4a table is missing rows the record has: {missing}"

    for label, key in BG_TABLE_ROWS.items():
        arm_means = bg["effects"][key]["arm_means"]
        for column, arm in enumerate(ARM_ORDER):
            published = _number(rows[label][column])
            # Half-up at 2 dp, quoted the way a human writes a table -- the two
            # corrected cells came from rounding 9.854875 twice, to 9.855 and
            # then 9.86.
            expected = round(arm_means[arm] + 1e-12, 2)
            assert published == pytest.approx(expected, abs=0.005), (
                f"4a row {label!r}, arm {arm}: report says {published}, "
                f"record says {arm_means[arm]!r}")


def test_the_four_columns_are_the_registered_arms_in_order(report_text):
    rows = _table_rows(report_text, "### 4a. BG")
    header = next(iter(rows))
    assert header == "quantity"
    assert [c.strip() for c in rows["quantity"][:4]] == list(ARM_ORDER)


# --------------------------------------------------------------------------
# The night-negative axis: unreadable in every contrast, and the doctrine says
# so. This is the claim the merged text got wrong.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "contrast", ["exclusion_at_f25", "exclusion_at_f27", "both_vs_neither"])
def test_no_night_negative_contrast_on_bg_clears_its_null(bg, contrast):
    block = bg["effects"]["pct_of_night_rows_negative"][contrast]
    assert block["outside_the_null"] is False, (
        f"{contrast} on the night-negative rate now clears its null; the "
        "doctrine in CLAUDE.md and report section 5c says none of them do")
    assert abs(block["paired_mean"]) < block["null_max"]


def test_the_eight_of_eight_negative_contrast_is_still_inside_the_null(bg):
    """The exact trap: a clean sign test on a metric too noisy to read."""
    block = bg["effects"]["pct_of_night_rows_negative"]["exclusion_at_f27"]
    assert block["seeds_down"] == 8 and block["n_seeds"] == 8
    assert block["sign_test_p"] == pytest.approx(0.0078125, abs=1e-6)
    assert block["outside_the_null"] is False


@pytest.mark.parametrize("contrast", ["exclusion_at_f25", "exclusion_at_f27"])
def test_night_mae_on_bg_does_clear_its_null(bg, contrast):
    """The asymmetry the finding rests on: the level metric reads, the sign one does not."""
    block = bg["effects"]["night_mae_mw"][contrast]
    assert block["outside_the_null"] is True
    assert block["seeds_up"] == 8
    assert block["paired_mean"] > block["null_max"]


def test_report_quotes_each_negative_contrast_against_its_own_control(report_text):
    """7/8 belongs to the f25 endpoints, 8/8 to the f27 ones -- never crossed."""
    section = report_text.split("**5c.", 1)[1].split("**5d.", 1)[0]
    assert "20.09% → 12.97%" in section
    assert "21.63% → 9.85%" in section
    # The diagonal's endpoints must not appear as a contrast in 5c.
    assert "20.09% → 9.85%" not in section
    assert "20.09% to 9.85%" not in section


def test_report_states_the_negative_axis_is_unreadable(report_text):
    section = report_text.split("**5c.", 1)[1].split("**5d.", 1)[0]
    assert "14.06pp" in section
    assert "outside_the_null" in section


# --------------------------------------------------------------------------
# CLAUDE.md carries the same claim, and is the copy people actually read.
# --------------------------------------------------------------------------

def test_doctrine_does_not_attach_statistics_to_the_diagonal():
    """The exact merged defect: diagonal endpoints carrying f27's sign test.

    Naming the diagonal in order to forbid it is the point of that paragraph, so
    presence alone is not the defect -- an earlier draft of this guard failed on
    the prohibition itself. What must never appear is the diagonal *quoted as a
    result*, which in practice means a seed count or a p-value trailing it.
    """
    text = DOCTRINE.read_text(encoding="utf-8")
    section = text.split("Never disposition a night-floor change", 1)[1][:2000]
    for diagonal in ("20.09% -> 9.86%", "20.09% -> 9.85%",
                     "20.09% → 9.86%", "20.09% → 9.85%"):
        start = section.find(diagonal)
        if start == -1:
            continue
        trailing = section[start + len(diagonal):start + len(diagonal) + 90]
        assert not re.search(r"\d\s*/\s*8|p\s*=|seeds", trailing), (
            f"CLAUDE.md quotes {diagonal!r} with statistics attached "
            f"({trailing.strip()!r}); those endpoints cross both factors of the "
            "2x2, so no contrast's sign test belongs to them")


def test_doctrine_carries_the_null_that_makes_the_metric_unreadable(bg):
    text = DOCTRINE.read_text(encoding="utf-8")
    section = text.split("Never disposition a night-floor change", 1)[1][:2000]
    null_max = bg["effects"]["pct_of_night_rows_negative"]["exclusion_at_f25"]["null_max"]
    assert f"{null_max:.2f}pp" in section, (
        "CLAUDE.md must quote the null that makes this metric unreadable, not "
        "just the effect and its p-value")
    assert "outside_the_null" in section
