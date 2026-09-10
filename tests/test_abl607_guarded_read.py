"""ABL-607: the archive read is exempt from the guard, and exempt *correctly*.

ABL-462 widened the plausibility sweep past `src/` and it named this script.
ABL-611 settled the disposition: exempt, not guarded. ABL-617 then split the
exemption into a checked category -- `ML_SLICE_ONLY_EXEMPT` -- so the reasoning
lives in `test_tso_plausibility.py` and in the docstring of
`plausibility_census`; what is pinned here is that the code does what the
exemption claims, in the two respects that are particular to this file.

Two claims, and neither is checkable by the static sweep -- which stops looking
at a file the moment it goes on any exempt list:

1. **The read filters nothing.** The guard is one-sided, so on the arm under
   test the only rows it could remove are our own largest over-forecasts --
   the errors this pack measures. A regression that "helpfully" wired
   `guard_tso_frame` in would bias the D+2-vs-D-7 ranking in our own model's
   favour and no other test in the repo would notice.
2. **The exemption still carries a measurement.** The pack publishes a count
   of what the guard *would* have refused. An exemption whose census could not
   have detected anything is the vacuous kind.

A third claim -- **the file still reads only our own `source = 'ml'` rows** --
was pinned here too, and is not any more. It is not particular to this file:
it is what `ML_SLICE_ONLY_EXEMPT` *means*, so ABL-617 made it the condition of
joining that list (`ml_slice_violations`, with negative controls per way the
claim can rot). Pinning it per file left the next entry to remember to bring
its own test, which is the same intent-claim problem one level up.

Not run against the live replica on purpose: a rule pinned against live data
stops being a test the day the data moves. The live count is measured by the
script and published in section 0 of
`reports/abl_607_d2_load_diagnosis_reread.json` (ABL-619): **0 of 67,008 rows
would have been refused**, 24/24 countries evaluable. The first run of the pack
predates the census, which is why the record that carries it is the re-read.
Section 3 below holds every text that quotes it against that artifact -- the
numbers as well as the tense, and the list of texts itself -- so neither can
drift from the other.
"""

import ast
import json
import math
import re
import sqlite3
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl607_d2_load_diagnosis import plausibility_census  # noqa: E402
from src.tso_plausibility import clear_reference_cache  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "abl607_d2_load_diagnosis.py"
#: The record that carries the census. It is the *re-read*, not the pack's
#: first run: the census landed in the script after that run, and the replica
#: was under an exclusive writer for the rest of the day (ABL-619). The two
#: records are kept apart rather than the first overwritten, because the
#: re-read is a later replica vintage and every metric in it moved slightly --
#: merging them would leave the prose of the first quoting numbers its own
#: machine record no longer contained, which is the defect this file exists to
#: prevent, at a hundred sites instead of three.
REPORT = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis_reread.json"
#: The pack's first run, kept for the vintage comparison in section 3.1.
PUBLISHED = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis.json"
PROSE = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis.md"


@pytest.fixture(autouse=True)
def _no_cross_test_cache():
    """The reference cache is process-wide and keyed on the database file --
    every in-memory fixture reports the same key, so they must not share one."""
    clear_reference_cache()
    yield
    clear_reference_cache()


def _hours(n, start="2026-08-13"):
    return pd.date_range(start, periods=n, freq="h")


def _replica(load_actuals, countries=("HU",)):
    """A replica-shaped database holding only what the reference is built from."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE energy_load (
            country_code TEXT, timestamp_utc TIMESTAMP, load_mw REAL);
        CREATE TABLE forecast_vintage_archive (
            source TEXT NOT NULL, forecast_type TEXT NOT NULL,
            country_code TEXT NOT NULL, target_timestamp_utc TEXT NOT NULL,
            model_name TEXT NOT NULL, run_timestamp_utc TEXT NOT NULL,
            horizon_hours INTEGER, forecast_value REAL NOT NULL,
            first_seen_at TEXT NOT NULL);
    """)
    for cc in countries:
        conn.executemany(
            "INSERT INTO energy_load (country_code, timestamp_utc, load_mw) "
            "VALUES (?, ?, ?)",
            [(cc, str(t), float(v)) for t, v in
             zip(_hours(len(load_actuals), "2026-01-01"), load_actuals)])
    conn.commit()
    return conn


def _ml_frame(values, country="HU"):
    """The frame `load_archive` hands the census, with the columns it uses."""
    targets = _hours(len(values))
    return pd.DataFrame({
        "country_code": country,
        "target_timestamp_utc": [str(t) for t in targets],
        "model_name": "load-catboost",
        "horizon_hours": 30,
        "forecast_value": [float(v) for v in values],
        "first_seen_at": "2026-08-12T06:00:00Z",
        "target": targets,
    })


# --------------------------------------------------------------------------
# 1. the read filters nothing
# --------------------------------------------------------------------------

def test_an_implausible_row_is_reported_and_kept():
    """The load-bearing test. A row at the ABL-431 incident's own scale is
    detected and **stays in the panel**, because it is our own model's error
    and this pack exists to score it."""
    conn = _replica(load_actuals=[280.0] * 500)
    values = [275.0] * 47 + [140996.245]
    df = _ml_frame(values)

    out = plausibility_census(df, conn)

    assert out.attrs["guard_would_refuse"] == 1
    assert out.attrs["guard_rows_dropped"] == 0
    assert len(out) == len(df)
    assert list(out["forecast_value"]) == values
    census = out.attrs["guard_census"][0]
    assert census["n_would_be_refused"] == 1
    assert census["max_over_threshold"] > 100


def test_the_frame_is_returned_row_for_row_identical():
    """Nothing about the panel may move -- not a value, not a row, not the
    order. `build_ml_arms` breaks ties with
    `sort_values("first_seen").groupby(...).last()` and pandas' default sort is
    quicksort, which is not stable, so even a reordering could change which
    vintage wins a tie."""
    conn = _replica(load_actuals=[280.0] * 500, countries=("HU", "SK"))
    df = _ml_frame([200.0, 210.0, 220.0, 230.0])
    df.loc[[0, 2], "country_code"] = "SK"

    out = plausibility_census(df, conn)

    pd.testing.assert_frame_equal(out, df)


def test_the_source_file_never_calls_a_filtering_entry_point():
    """The exemption's claim, checked against the file rather than trusted.

    `guard_tso_series` / `guard_tso_frame` / `guard_series` all null values.
    None may appear here: a future edit that adds one would silently start
    deleting our own worst forecasts, and every other test would still pass.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    for filtering_call in ("guard_tso_series(", "guard_tso_frame(",
                           "guard_series("):
        assert filtering_call not in text, (
            f"{SCRIPT.name} calls {filtering_call} -- this read is "
            f"ML_SLICE_ONLY_EXEMPT and must not filter; see "
            f"plausibility_census's docstring")


# The third claim -- "the file still reads only our own `source = 'ml'` rows"
# -- used to be pinned here, per file. ABL-617 made it the membership condition
# of `ML_SLICE_ONLY_EXEMPT` instead (`ml_slice_violations` in
# `tests/test_tso_plausibility.py`, with its own negative controls), so it now
# holds for every file that ever joins that category rather than for this one.
# Deleted here rather than kept alongside: two copies of the same rule is how
# the weaker one ends up being the one that runs.


# --------------------------------------------------------------------------
# 2. the exemption carries a measurement that could have fired
# --------------------------------------------------------------------------

def test_a_plausible_arm_reports_the_headroom_it_cleared_by():
    """A bare `0 would be refused` is not evidence; the ratio is what makes it
    one."""
    conn = _replica(load_actuals=[280.0] * 500)
    values = [200.0, 275.0, 310.0, 250.0]

    out = plausibility_census(_ml_frame(values), conn)

    assert out.attrs["guard_would_refuse"] == 0
    census = out.attrs["guard_census"][0]
    assert census["threshold_mw"] == pytest.approx(3.0 * 280.0)
    assert census["max_over_threshold"] == pytest.approx(310.0 / 840.0)


def test_the_reference_is_not_set_by_our_own_arm():
    """The registration's load-bearing exclusion, exercised end to end.

    Our `source = 'ml'` rows are excluded from the reference by
    `forecast_read`, so an ML arm that overshot cannot lift the bar it is
    measured against. If it could, the census would certify whatever we
    happened to publish.
    """
    conn = _replica(load_actuals=[280.0] * 500)

    out = plausibility_census(_ml_frame([140996.245] * 400), conn)

    assert out.attrs["guard_census"][0]["threshold_mw"] == pytest.approx(840.0)
    assert out.attrs["guard_would_refuse"] == 400
    assert len(out) == 400


def test_the_test_is_one_sided_so_an_under_forecast_is_never_flagged():
    """Which is why filtering would be one-directional: the census can only
    ever name our over-forecasts, never the under-forecasts that hurt our WAPE
    just as much."""
    conn = _replica(load_actuals=[280.0] * 500)

    out = plausibility_census(_ml_frame([0.0, 1.0, 275.0]), conn)

    assert out.attrs["guard_would_refuse"] == 0


def test_a_non_evaluable_reference_is_reported_as_such():
    """A country with no fleet history has no scale to be held to. The census
    says so rather than reporting a clean zero: a cell that was never
    evaluated is not a cell that passed."""
    conn = _replica(load_actuals=[0.0] * 500)

    out = plausibility_census(_ml_frame([200.0, 5000.0]), conn)

    census = out.attrs["guard_census"][0]
    assert census["evaluable"] is False
    assert census["max_over_threshold"] is None
    assert census["n_would_be_refused"] == 0


# --------------------------------------------------------------------------
# 3. the measurement is on disk, not merely computable (ABL-619)
# --------------------------------------------------------------------------
#
# Everything above pins the *code*. None of it could catch ABL-619, which is
# the failure that actually happened: the census landed, the report was never
# regenerated, and three merged texts went on saying its output was published.
# The exemption's warrant is "the exemption carries a measurement and not
# just a claim" -- a sentence about an artifact. So it is checked against the
# artifact.


def _script_dict_keys(name: str) -> tuple:
    """The literal keys of a top-level `name = {...}` in the script.

    Read out of the source rather than copied: a copy is the failure this test
    exists to prevent. If the section is renamed, this follows it and still
    demands the report carry it -- whereas a hardcoded key would quietly start
    asserting nothing.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if [t.id for t in node.targets if isinstance(t, ast.Name)] != [name]:
            continue
        return tuple(k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str))
    raise AssertionError(f"no top-level `{name} = {{...}}` in {SCRIPT.name}")


def _script_dict_literal(name: str, *path: str) -> str:
    """The literal string at `name[path...]` in the script, read by AST.

    Out of the source for the same reason `_script_dict_keys` is, and it is the
    load-bearing half of the test below: a copy of the string kept here would
    make that test assert a constant equals itself the moment the script was
    edited -- the vacuity `_census_text` already had to dodge once in this
    module.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if [t.id for t in node.targets if isinstance(t, ast.Name)] != [name]:
            continue
        current = node.value
        for key in path:
            assert isinstance(current, ast.Dict), (
                f"{name}{list(path)}: {key!r} is not under a dict literal")
            values = [v for k, v in zip(current.keys, current.values)
                      if isinstance(k, ast.Constant) and k.value == key]
            assert values, f"{name}{list(path)}: no {key!r} key in {SCRIPT.name}"
            current, = values
        assert isinstance(current, ast.Constant) and isinstance(current.value, str), (
            f"{name}{list(path)} is no longer a plain string literal, so this "
            f"test can no longer read what the script writes")
        return current.value
    raise AssertionError(f"no `{name} = {{...}}` in {SCRIPT.name}")


#: Where the archive read's category is recorded, in the script and in the
#: report that script writes. Two sites, because `meta.guard` mirrors the
#: section's `disposition` and a rewrite of one alone is the same defect.
DISPOSITION_SITES = (
    (("meta", "guard"), ("meta", "guard")),
    (("record", "section_0_plausibility_census", "disposition"),
     ("section_0_plausibility_census", "disposition")),
)

#: Every committed report this script has written. Both carry both strings, so
#: a re-tensing that regenerated only one would be caught here too.
DISPOSITION_REPORTS = (
    REPORT,
    REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis_completeness.json",
)


def test_the_scripts_disposition_still_matches_every_report_it_wrote():
    """ABL-669: a comment was the only thing holding this.

    `main()` carries a reviewed decision *not* to re-tense the archive read's
    category from `EXEMPT_READS (ABL-611)` to ABL-617's `ML_SLICE_ONLY_EXEMPT`:
    a report is the record of a run, and the committed reports were produced
    under the old category with nothing since to reconcile them. The decision
    is right -- the strings do agree today -- and it was written down as an
    instruction to a future editor ("change this only in a commit that also
    regenerates the report") rather than as a check. An instruction in a
    comment is precisely what ABL-619 was: code claiming one thing, the
    published artifact carrying another, and nothing red.

    So the **agreement** is pinned, both directions and both artifacts:
    rewrite the string without regenerating and this fails; regenerate under a
    new category without re-tensing the script and it fails the same way. The
    static sweep cannot cover either -- it stops looking at this file the
    moment the file goes on an exempt list, which is the whole reason ABL-617
    made that list a checked category.
    """
    for path in DISPOSITION_REPORTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        for script_path, report_path in DISPOSITION_SITES:
            in_script = _script_dict_literal(*script_path)
            in_report = report
            for key in report_path:
                assert isinstance(in_report, dict) and key in in_report, (
                    f"{path.name} has no {'.'.join(report_path)} -- it was "
                    f"written by a script older than that key, or the key was "
                    f"renamed without regenerating")
                in_report = in_report[key]
            assert in_script == in_report, (
                f"{SCRIPT.name} writes {'.'.join(script_path)} = "
                f"{in_script!r}, but the committed {path.name} carries "
                f"{in_report!r}. One was rewritten without the other: either "
                f"regenerate the report in this commit, or put the string "
                f"back. Do not 'fix' this by editing the JSON -- a report is "
                f"the record of a run (ABL-619).")


#: The qualifier the three texts carry while the census is *not* on disk. It is
#: the whole mechanism: prose cannot read a JSON file, so instead the prose
#: declares which of the two states it is describing, and the test holds that
#: declaration against the artifact.
PENDING_MARKER = "(pending: ABL-619)"

#: Every merged text that describes the census. Each asserted the count was
#: published while the committed report had no such key -- that is ABL-619.
#: Membership is swept, not asserted: this read "the three merged texts" and
#: the pack's own prose report was a fourth, quoting the count in three places
#: and listed nowhere, so both pins below simply skipped it (ABL-669).
CENSUS_TEXTS = (
    ("scripts/abl607_d2_load_diagnosis.py", "the protocol block"),
    ("tests/test_tso_plausibility.py", "the ML_SLICE_ONLY_EXEMPT entry's reason"),
    ("tests/test_abl607_guarded_read.py", "this module's docstring"),
    ("reports/abl_607_d2_load_diagnosis.md", "the pack's prose report"),
)

#: Where a census text can live. Bounded rather than a whole-repo walk, and
#: deliberately no JSON: a record that carries the number is not a text
#: claiming it.
CENSUS_TEXT_SEARCH = ("*.md", "docs/**/*.md", "reports/*.md",
                      "scripts/*.py", "src/**/*.py", "tests/*.py")

#: Comment and emphasis markers, which sit inside these claims rather than
#: around them: `# ` wraps two of the texts and `**` bolds the numbers in a
#: third. Removed before matching so one derived fragment can be held against
#: all four, and so re-wrapping or re-bolding a paragraph cannot turn a pin
#: red. Kept out of `_flat`, which section 4 uses for pins that match the
#: asterisks deliberately.
_CENSUS_NOISE = re.compile(r"[#*`]")

#: The census claim in the two shapes the texts write it in. Matched over
#: *every* occurrence rather than as a substring: the prose report states the
#: count in three places, so "the number appears somewhere in this file" passes
#: with two sites corrected and one left stale, which is the drift this section
#: exists to catch.
_CENSUS_ROWS = re.compile(r"(\d[\d,]*) of ([\d,]{3,}) (?:archive )?rows")
#: Scoped to the evaluability claim, because the pack says "N of M countries"
#: about other things -- 12 of 24 gained a target day, 11 of 23 find one arm
#: readable, 23 of 24 share production's algorithm. Those are not this claim
#: and must not be held to its numbers.
_CENSUS_COUNTRIES = re.compile(
    r"(\d+)\s*(?:/|of)\s*(\d+) countries[^.]{0,45}?evaluable")


def _census_text(relpath: str) -> str:
    """The prose of one of the three texts, isolated from the machinery.

    For this module that is the module docstring alone: `PENDING_MARKER` and
    `CENSUS_TEXTS` are defined here too, and a naive substring search over the
    file would find the constant and report the marker as present no matter
    what the docstring said -- a test that always passes.
    """
    path = REPO_ROOT / relpath
    if path.name == Path(__file__).name:
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    return path.read_text(encoding="utf-8")


def _census_prose(relpath: str) -> str:
    """One census text, flattened to the words it actually claims.

    The four texts carry the same claim in four typographies: two `#` comment
    blocks, a docstring, and a Markdown report that bolds the numbers. Dropping
    the markers is what lets one fragment derived from the record be held
    against all four, and it is what keeps a re-wrap -- or a re-bolding -- from
    turning these pins red, which is the trap `_flat` exists for one section
    down.
    """
    return " ".join(_CENSUS_NOISE.sub(" ", _census_text(relpath)).split())


def test_no_text_claims_a_census_the_committed_report_does_not_carry():
    """ABL-619's finding, pinned as an invariant rather than a re-wording.

    At `ffb097f` all three texts said the count the guard would have refused
    was published. All three were true of the code and false of the artifact:
    `section_0_plausibility_census` and the four `meta.guard_*` mirrors were
    absent, because the script grew a census and the replica went under an
    exclusive writer before anything re-ran.

    Re-tensing the sentences fixes that instance and nothing else. So what is
    pinned here is the **agreement**, in both directions:

      * report carries the census  -> no text may still say it is pending
      * report does not            -> every text must say so

    which makes the failure red whichever way it happens: publishing the
    report without dropping the qualifier, or claiming publication without
    regenerating the report. The static sweep cannot cover either -- it stops
    looking at a file the moment that file goes on an exempt list.
    """
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    section, = [k for k in _script_dict_keys("record") if "census" in k]
    published = section in report

    for relpath, what in CENSUS_TEXTS:
        text = _census_text(relpath)
        assert text, f"{relpath}: {what} is empty; this test has gone blind"
        marked = PENDING_MARKER in text
        assert marked != published, (
            f"{relpath} ({what}) and {REPORT.name} disagree: the report "
            f"{'carries' if published else 'does not carry'} {section!r}, but "
            f"the text {'still carries' if marked else 'omits'} "
            f"{PENDING_MARKER!r}. "
            + (f"Drop the qualifier -- the census is published now."
               if published else
               f"Either regenerate the report or mark the text pending; a flat "
               f"claim that the count is published is false today."))

    if not published:
        return

    guard_keys = [k for k in _script_dict_keys("meta") if k.startswith("guard")]
    assert guard_keys, "meta no longer mirrors the census; this test is stale"
    missing = [k for k in guard_keys if k not in report["meta"]]
    assert not missing, (
        f"{REPORT.name} carries {section!r} but its meta is missing "
        f"{missing} -- the report was written by a script older than the "
        f"mirrors, or the mirrors were dropped")


def test_every_text_quotes_the_census_numbers_the_report_carries():
    """The other half of ABL-619, and the half the qualifier cannot reach.

    The test above pins *whether* the census is published. It says nothing
    about what it found, and every text quotes that: the count refused, the
    rows read, how many countries carried an evaluable reference. Measured at
    `7fc1fde` -- falsify the three numbers in the three texts that were listed
    then, and the suite is still `1786 passed, 1 skipped`. So the
    `ML_SLICE_ONLY_EXEMPT` entry saying
    it is "pinned by `tests/test_abl607_guarded_read.py`, which fails if this
    comment and that artifact ever disagree, in either direction" was wider
    than what was checked: a text claiming more reach than the check has, which
    is ABL-669, at a site where the numbers are a published measurement.

    Derived from the record in section 4's idiom, so a re-read that moves the
    window reds every text still quoting the old one -- rather than leaving
    three merged texts describing a run nobody can find, which is what ABL-619
    was.
    """
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    section, = [k for k in _script_dict_keys("record") if "census" in k]
    census = report.get(section)
    if census is None:
        pytest.skip("no census on disk -- "
                    "test_no_text_claims_a_census_the_committed_report_does_"
                    "not_carry owns that failure")

    per_country = census["per_country"]
    assert per_country, "census ran over no countries"
    evaluable = [c for c in per_country if c["evaluable"]]
    expected = (
        (_CENSUS_ROWS,
         (str(census["rows_would_be_refused"]), f"{census['rows_read']:,}"),
         "the count refused out of the rows read"),
        (_CENSUS_COUNTRIES,
         (str(len(evaluable)), str(len(per_country))),
         "the countries carrying an evaluable reference"),
    )

    assert CENSUS_TEXTS, "no census texts left -- this test certifies nothing"
    for relpath, what in CENSUS_TEXTS:
        prose = _census_prose(relpath)
        assert prose, f"{relpath}: {what} is empty; this test has gone blind"
        for pattern, numbers, described in expected:
            sites = pattern.findall(prose)
            assert sites, (
                f"{relpath} ({what}) no longer states {described}. Every text "
                f"on CENSUS_TEXTS quotes the census; one that has stopped "
                f"belongs off the list, not silently unchecked.")
            wrong = sorted(set(sites) - {numbers})
            assert not wrong, (
                f"{relpath} ({what}) states {described} as {wrong} at one or "
                f"more of its {len(sites)} sites; {REPORT.name} records "
                f"{numbers}. Either the text is stale or the report moved "
                f"under it -- ABL-619 was the first of those, and the "
                f"exemption entry claims it cannot happen unnoticed.")

    # The headroom bound, in whichever texts carry it. Rounded *up*: it is
    # stated as a bound ("never above"), and a bound quoted at the value's own
    # rounding is false whenever the value rounds down -- the largest ratio on
    # record is 32.338% of a country's threshold, which is not <= 32.3%. That
    # was the merged wording; the number was right at 1 dp and the sentence
    # was not (ABL-669).
    #
    # The limit, stated rather than left to be discovered: this holds the bound
    # wherever a text writes it, and requires at least one text to. Dropping
    # the clause from one text while another keeps it is not red. "Every text
    # states it" would be the wrong rule -- this module's own docstring gives
    # the count without the headroom, correctly.
    ratios = [c["max_over_threshold"] for c in evaluable
              if c["max_over_threshold"] is not None]
    assert ratios, "no evaluable country recorded headroom"
    bound = math.ceil(max(ratios) * 1000) / 10
    headroom = f"never above {bound:.1f}% of any country's threshold"

    stating = [(relpath, prose)
               for relpath, prose in ((r, _census_prose(r))
                                      for r, _ in CENSUS_TEXTS)
               if "never above" in prose]
    assert stating, (
        "no census text states the headroom bound any more. It is what makes "
        "the zero a tested zero rather than an untested one, so dropping it "
        "removes the evidence the exemption rests on")
    for relpath, prose in stating:
        assert headroom in prose, (
            f"{relpath} states a headroom bound {REPORT.name} does not "
            f"support: the largest published value is "
            f"{max(ratios) * 100:.3f}% of a country's threshold, so the "
            f"bound at the precision the text uses is {bound:.1f}% -- "
            f"expected {headroom!r}")


def test_census_texts_names_every_text_that_quotes_the_census():
    """`CENSUS_TEXTS` is a membership claim, and nothing held it.

    Both pins above iterate it, so removing an entry stops checking that text
    with nothing going red -- and the list was in fact short. It described
    itself as "the three merged texts" while `reports/abl_607_d2_load_
    diagnosis.md` quoted the count in three places and named itself nowhere,
    so ABL-619's own pin had never looked at the pack's prose report (ABL-669).

    A count of texts cannot see a text it is missing; only the set recomputed
    from the repo can. That is section 4's lesson about the margin table --
    "both columns are membership assertions below, not counts" -- applied to
    the list that decides which prose section 3 reads at all.
    """
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    section, = [k for k in _script_dict_keys("record") if "census" in k]
    census = report.get(section)
    if census is None:
        pytest.skip("no census on disk -- nothing for a text to quote")

    signature = f"{census['rows_would_be_refused']} of {census['rows_read']:,}"
    found = set()
    for pattern in CENSUS_TEXT_SEARCH:
        for path in REPO_ROOT.glob(pattern):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if signature in " ".join(_CENSUS_NOISE.sub(" ", text).split()):
                found.add(path.relative_to(REPO_ROOT).as_posix())

    assert found, (
        f"nothing in the repo quotes {signature!r}. Either the census moved "
        f"and every text is stale, or this sweep has stopped looking where "
        f"the texts live -- see CENSUS_TEXT_SEARCH")

    listed = {relpath for relpath, _ in CENSUS_TEXTS}
    assert found == listed, (
        f"CENSUS_TEXTS and the repo disagree on which texts quote the census. "
        f"Quoting it and unlisted: {sorted(found - listed)}; listed and no "
        f"longer quoting it: {sorted(listed - found)}. A text that states the "
        f"count has to be held to the record -- being absent from this tuple "
        f"is how a text goes on describing a run nobody can find (ABL-619).")


def test_the_published_census_could_have_detected_something():
    """A census that evaluated nothing would satisfy the test above while
    certifying whatever we published -- the vacuous exemption in artifact form.

    So the on-disk record has to show a reference was actually built: at least
    one country with `evaluable` true, and headroom recorded for it. `0 rows
    would be refused` is evidence only once it is `0 out of N, against a
    threshold, with a ratio we cleared it by`.
    """
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    section, = [k for k in _script_dict_keys("record") if "census" in k]
    census = report.get(section)
    if census is None:
        pytest.skip("no census on disk -- test_the_published_report_carries_"
                    "the_census_the_texts_promise owns that failure")

    per_country = census["per_country"]
    assert per_country, "census ran over no countries"

    evaluable = [c for c in per_country if c["evaluable"]]
    assert evaluable, (
        "no country had an evaluable plausibility reference, so the census "
        "refused nothing because it tested nothing")
    assert all(c["max_over_threshold"] is not None for c in evaluable)

    # The report must not contradict itself: the totals the texts quote are
    # the per-country rows added up, and the read filters nothing.
    assert census["rows_dropped"] == 0, (
        "the report records dropped rows -- this read is ML_SLICE_ONLY_EXEMPT "
        "and must filter nothing; see plausibility_census's docstring")
    assert census["rows_read"] == sum(c["n_rows"] for c in per_country)
    assert census["rows_would_be_refused"] == sum(
        c["n_would_be_refused"] for c in per_country)
    assert census["rows_read"] == report["meta"]["guard_rows_read"]
    assert census["rows_would_be_refused"] == report["meta"]["guard_would_refuse"]
    assert census["rows_dropped"] == report["meta"]["guard_rows_dropped"]

# --------------------------------------------------------------------------
# 4. the corrected count carries its own fragility
# --------------------------------------------------------------------------
#
# Section 3.1 corrects a published count -- 10 readable losers to 9 -- and then
# argues the 9 is no firmer than the 10 was. Both halves are quantitative, and
# both are recomputable: the pack's first run and the re-read sit on disk side
# by side, which is what makes the vintage comparison checkable rather than
# asserted. So the prose is held to the two records here.
#
# This is not belt-and-braces. Review on PR #101 caught one drifted figure in
# this section by eye (SI quoted as 15.41 against a record holding 15.4046),
# and the first run of these tests caught a second the review had not: BE
# belongs in the revised set and the list named three countries. A section
# whose whole argument is that numbers move underneath you is the last place
# in the repo to quote them from memory.
#
# Review on PR #102 then caught a third, and it is the reason every list in
# this section is now derived rather than counted: BE also qualified for the
# margin table's right-hand column -- a loss of +1.78 pp whose `ci_lo` of
# -1.15 is inside the 1.34 pp step the section measures against -- and the
# table named three cells. A row-count assertion cannot see that. It pins the
# rows the table *shows*; only a set recomputed from the record can notice a
# row the table should show and does not. So both columns are membership
# assertions below, not counts.

#: U+2212. The reports use a real minus sign, not a hyphen.
MINUS = "−"

#: Spelled out in the prose, so a pin on the count has to spell it too.
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
                12: "twelve"}


def _flat(text):
    """Prose with its line wrapping collapsed.

    Every claim below spans a line break somewhere, and re-wrapping a paragraph
    must not turn a pin red -- otherwise the next person to reflow this file
    learns to delete the test rather than to trust it.
    """
    return " ".join(text.split())


def _paired_by_country(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {row["country"]: row
            for row in report["section_a_reproduction"]["paired_ml_band_vs_d7"]}


def _per_country(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {row["country"]: row
            for row in report["section_a_reproduction"]["per_country"]}


def _margin_table(raw_prose):
    """The section 3.1 margin table, as {country: shown ci_lo string}.

    Scoped to the table rather than swept over the whole file: several other
    tables in this report carry a country beside a signed number, and a regex
    loose enough to catch those would pin the wrong rows.
    """
    start = raw_prose.index("| readable losers, by margin |")
    block = raw_prose[start:raw_prose.index("\n\n", start)]
    return dict(re.findall(rf"\| ([A-Z]{{2}}) \| ([+{MINUS}][\d.]+) \|", block))


def test_the_margin_table_agrees_with_the_records():
    """Each surviving cell's distance from the readability threshold, measured
    against the size of one observed vintage step.

    Fails in both directions: a table row that drifts from the record is red,
    and so is a record that moves under a table left as it was.
    """
    raw = PROSE.read_text(encoding="utf-8")
    prose = _flat(raw)
    published, reread = _paired_by_country(PUBLISHED), _paired_by_country(REPORT)

    # The size of one vintage step, at the precision the prose states it to.
    steps = {c: abs(reread[c]["ci_lo"] - published[c]["ci_lo"])
             for c in reread.keys() & published.keys()}
    worst = max(steps, key=steps.__getitem__)
    assert f"at most **{steps[worst]:.2f} pp**" in prose, (
        f"3.1 states a maximum vintage step the two records do not give; "
        f"recomputed {steps[worst]:.2f} pp on {worst}")
    assert f"({worst})" in prose, f"the largest step is {worst}'s"
    assert f"across the {len(steps)} countries" in prose

    # The two summary figures quoted beside the maximum. The median is
    # estimator-independent at 2dp; p75 is not -- 0.19 under the linear
    # interpolation numpy and pandas default to, 0.20 under the stdlib's -- so
    # this pin also records which estimator the sentence means.
    ordered = sorted(steps.values())
    assert f"median {statistics.median(ordered):.2f}" in prose
    assert f"p75 {statistics.quantiles(ordered, n=4)[2]:.2f}" in prose

    # Every cell the table names, at the sign and precision shown.
    shown = _margin_table(raw)
    for country, quoted in shown.items():
        actual = f"{reread[country]['ci_lo']:+.2f}".replace("-", MINUS)
        assert actual == quoted, (
            f"3.1 shows {country} at ci_lo {quoted}, the re-read has {actual}")

    # Both columns as membership, not as a row count. A count pins the rows the
    # table shows and cannot notice a row it should show and does not, which is
    # how BE was missing from the right-hand column on first writing.
    survivors = {c for c, r in reread.items()
                 if r["readable"] and r["mean_daily_wape_diff"] > 0}
    inside = {c for c in survivors if reread[c]["ci_lo"] < steps[worst]}
    assert {c for c, v in shown.items() if v.startswith("+")} == inside, (
        f"the survivors inside one vintage step are {sorted(inside)}; 3.1's "
        f"left column shows {sorted(c for c, v in shown.items() if v[0] == '+')}")
    assert (f"{NUMBER_WORDS[len(inside)]} of the {NUMBER_WORDS[len(survivors)]} "
            f"survivors") in prose

    below = {c for c, r in reread.items()
             if r["mean_daily_wape_diff"] > 0 and r["ci_lo"] < 0
             and -r["ci_lo"] < steps[worst]}
    assert {c for c, v in shown.items() if v.startswith(MINUS)} == below, (
        f"the cells with a positive central estimate sitting inside one vintage "
        f"step below the line are {sorted(below)}; 3.1's right column shows "
        f"{sorted(c for c, v in shown.items() if v.startswith(MINUS))}")
    assert (f"{NUMBER_WORDS[len(below)]} cells with a positive central "
            f"estimate") in prose

    # The claim that separates a finding from a provisional count: the cells
    # called out as robust are exactly the losers outside one observed step.
    robust = survivors - inside
    assert robust == {"AT", "CZ", "SI", "SK"}, (
        f"the losers outside one vintage step are now {sorted(robust)}; "
        f"3.1 names SK, AT, CZ and SI")
    assert "**SK, AT, CZ and SI**" in prose
    margins = sorted(reread[c]["ci_lo"] for c in robust)
    assert f"+{margins[0]:.2f} … +{margins[-1]:.2f}" in prose, (
        f"the range row for the four robust cells should read "
        f"+{margins[0]:.2f} ... +{margins[-1]:.2f}")

    # ...and the sentence naming them is scoped to the readable cells, because
    # plenty of unreadable ones sit outside the step too. GR is quoted by its
    # binding bound, which for a win is ci_hi.
    readable = {c for c, r in reread.items() if r["readable"]}
    assert f"Of the {NUMBER_WORDS[len(readable)]} readable cells" in prose
    winners = readable - survivors
    assert winners == {"GR"}, f"the readable winners are now {sorted(winners)}"
    ci_hi = f"{reread['GR']['ci_hi']:.2f}".replace("-", MINUS)
    assert f"`ci_hi` = {ci_hi}" in prose, (
        f"3.1 quotes GR's binding bound; the re-read has {ci_hi}")


def test_the_margin_table_is_not_vacuous():
    """A table the parse silently found no rows in would satisfy the test above
    just as well, and would certify whatever the section said."""
    reread = _paired_by_country(REPORT)
    shown = _margin_table(PROSE.read_text(encoding="utf-8"))

    assert shown, "the margin table parsed to nothing -- the pin above is vacuous"
    for country, quoted in shown.items():
        assert country in reread, f"{country} is not a scored country"
        assert quoted.startswith(MINUS) == (reread[country]["ci_lo"] < 0), (
            f"{country} is quoted as {quoted}, against an interval of the "
            f"other sign -- the parse is matching decoration, not the table")


def test_the_vintage_paragraph_agrees_with_the_records():
    """Section 3.1's attribution paragraph: why the move from 10 to 9 is a data
    vintage and not a correction. These numbers *are* the argument."""
    prose = _flat(PROSE.read_text(encoding="utf-8"))
    published, reread = _per_country(PUBLISHED), _per_country(REPORT)
    n_countries = len(reread)

    # Arm one: the countries that gained rows.
    gained = {c: reread[c]["n"] - published[c]["n"]
              for c in reread if reread[c]["n"] != published[c]["n"]}
    assert f"**{len(gained)} of {n_countries} countries gained a target day**" in prose, (
        f"{len(gained)} of {n_countries} countries gained rows, "
        f"which is not what 3.1 states")
    by_one = [c for c, d in gained.items() if d == 1]
    biggest = max(gained, key=gained.__getitem__)
    assert f"{len(by_one)} of them by exactly one row, {biggest} by " in prose

    a, b = (json.loads(p.read_text(encoding="utf-8"))["meta"]["panel_a_n_scored_pairs"]
            for p in (PUBLISHED, REPORT))
    assert f"for +{b - a} rows in total, `{a:,} → {b:,}`" in prose, (
        f"3.1 states a row delta the records do not give; "
        f"recomputed +{b - a}, {a:,} -> {b:,}")

    # Arm two: the countries whose actuals were revised under a fixed panel.
    revised = sorted(c for c in reread
                     if reread[c]["n"] == published[c]["n"]
                     and reread[c]["days"] == published[c]["days"]
                     and reread[c]["mean_load_mw"] != published[c]["mean_load_mw"])
    assert len(gained) + len(revised) == n_countries, (
        f"the two arms no longer partition the countries, so 3.1's claim that "
        f"not one of the {n_countries} is unchanged does not follow: "
        f"{len(gained)} gained + {len(revised)} revised")
    assert f"**the other {len(revised)} had already-scored actuals revised" in prose
    assert f"**not one of the {n_countries} countries is unchanged**" in prose

    # Every country whose revision is visible at the precision 3.1 quotes.
    visible = [c for c in revised
               if round(reread[c]["wape_ml_band"], 2)
               != round(published[c]["wape_ml_band"], 2)]
    assert f"in {NUMBER_WORDS[len(visible)]} of them the revision reaches" in prose, (
        f"{len(visible)} countries move WAPE at two decimals; "
        f"3.1 states a different count")
    for country in visible:
        pair = (f"{country} {published[country]['wape_ml_band']:.2f}"
                f"→{reread[country]['wape_ml_band']:.2f}")
        assert pair in prose, f"3.1's visible-revision list is missing {pair}"
