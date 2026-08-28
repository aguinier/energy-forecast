"""ABL-607: the archive read is exempt from the guard, and exempt *correctly*.

ABL-462 widened the plausibility sweep past `src/` and it named this script.
ABL-611 settled the disposition: `EXEMPT_READS`, not a guard. The reasoning is
in `EXEMPT_READS` in `test_tso_plausibility.py` and in the docstring of
`plausibility_census`; what is pinned here is that the code does what the
exemption claims.

Three claims, and none is checkable by the static sweep -- which stops looking
at a file the moment it goes on `EXEMPT_READS`:

1. **The read filters nothing.** The guard is one-sided, so on the arm under
   test the only rows it could remove are our own largest over-forecasts --
   the errors this pack measures. A regression that "helpfully" wired
   `guard_tso_frame` in would bias the D+2-vs-D-7 ranking in our own model's
   favour and no other test in the repo would notice.
2. **The exemption still carries a measurement.** The pack publishes a count
   of what the guard *would* have refused. An exemption whose census could not
   have detected anything is the vacuous kind.
3. **The file still reads only our own `source = 'ml'` rows.** That is the
   premise the whole exemption rests on, and it is the one an ordinary future
   edit can quietly break -- add a TSO arm for comparison and the read becomes
   exactly what ABL-431 was filed about, still exempt.

Not run against the live replica on purpose: a rule pinned against live data
stops being a test the day the data moves. The live count is measured by the
script and belongs in section 0 of `reports/abl_607_d2_load_diagnosis.json` --
which does not yet carry it **(pending: ABL-619)**: the script grew the census,
the replica went under an exclusive writer, and the report was never
regenerated. Section 3 below is what makes that state impossible to state
wrongly, in either direction.
"""

import ast
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl607_d2_load_diagnosis import plausibility_census  # noqa: E402
from src.tso_plausibility import (  # noqa: E402
    VINTAGE_ARCHIVE_TABLE,
    clear_reference_cache,
)

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "abl607_d2_load_diagnosis.py"
REPORT = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis.json"


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
            f"{SCRIPT.name} calls {filtering_call} -- this read is EXEMPT_READS "
            f"and must not filter; see plausibility_census's docstring")


def _sweep_tso_tables() -> tuple:
    """`TSO_TABLES` from the sweep, read rather than copied.

    Parsed out of the source instead of imported: `tests/` is not a package and
    nothing else in the repo imports across test modules, so an import idiom
    would be new here. Single-sourced because a copy is the failure this test
    exists to prevent -- an exempt file is skipped by the sweep, so if the
    sweep widened its table list and this list did not, the widening would
    reach every file in the repo except the one holding an intent claim.
    """
    sweep = ast.parse((REPO_ROOT / "tests" / "test_tso_plausibility.py")
                      .read_text(encoding="utf-8"))
    names = {"VINTAGE_ARCHIVE_TABLE": VINTAGE_ARCHIVE_TABLE}
    for node in sweep.body:
        if not isinstance(node, ast.Assign):
            continue
        if [t.id for t in node.targets
                if isinstance(t, ast.Name)] != ["TSO_TABLES"]:
            continue
        return tuple(
            e.value if isinstance(e, ast.Constant) else names[e.id]
            for e in node.value.elts)
    raise AssertionError("TSO_TABLES not found in tests/test_tso_plausibility.py")


def test_no_query_in_the_file_ever_selects_a_tso_slice():
    """The other half of the exemption, and the half that can rot.

    `EXEMPT_READS` carries an *intent* claim -- "this file reads only our own
    `source = 'ml'` rows" -- and the sweep stops looking once a file is on the
    list. So the day someone adds a TSO arm here for comparison, the file would
    be reading a genuinely unguarded TSO forecast with a stale exemption
    covering it, and the sweep would be silent by construction.

    Checked over the parsed source so that the four `tso` mentions in comments
    and docstrings, the census's `n_tso_day_ahead_rows` column and the
    `src.tso_plausibility` import cannot satisfy or trip it: only string
    constants that are actually SQL against one of the sweep's tables count,
    and each must pin `source` to `'ml'`.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    docstrings = {id(ast.get_docstring(n, clean=False))
                  for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef,
                                    ast.AsyncFunctionDef, ast.ClassDef))}
    tso_tables = _sweep_tso_tables()

    queries = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and "FROM " in n.value and id(n.value) not in docstrings
        and any(t in n.value for t in tso_tables)
    ]
    assert queries, (
        "no SQL against a TSO-bearing table found in "
        f"{SCRIPT.name} -- this test has stopped checking anything")

    for sql in queries:
        assert "source = 'ml'" in sql, (
            f"a query in {SCRIPT.name} reads {tso_tables} without pinning "
            f"source = 'ml'; the EXEMPT_READS entry claims this file reads "
            f"only our own forecasts:\n{sql}")
        assert "'tso'" not in sql, (
            f"a query in {SCRIPT.name} selects TSO rows; the EXEMPT_READS "
            f"entry no longer holds and the read needs a guard:\n{sql}")


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
# The `EXEMPT_READS` warrant is "the exemption carries a measurement and not
# just a claim" -- a sentence about an artifact. So it is checked against the
# artifact.


def _script_dict_keys(name: str) -> tuple:
    """The literal keys of a top-level `name = {...}` in the script.

    Read out of the source rather than copied, for the same reason
    `_sweep_tso_tables` is: a copy is the failure this test exists to prevent.
    If the section is renamed, this follows it and still demands the report
    carry it -- whereas a hardcoded key would quietly start asserting nothing.
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


#: The qualifier the three texts carry while the census is *not* on disk. It is
#: the whole mechanism: prose cannot read a JSON file, so instead the prose
#: declares which of the two states it is describing, and the test holds that
#: declaration against the artifact.
PENDING_MARKER = "(pending: ABL-619)"

#: The three merged texts that describe the census. Each one asserted the count
#: was published while the committed report had no such key -- that is ABL-619.
CENSUS_TEXTS = (
    ("scripts/abl607_d2_load_diagnosis.py", "the protocol block"),
    ("tests/test_tso_plausibility.py", "the EXEMPT_READS entry's reason"),
    ("tests/test_abl607_guarded_read.py", "this module's docstring"),
)


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
    looking at a file the moment that file goes on `EXEMPT_READS`.
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
        "the report records dropped rows -- this read is EXEMPT_READS and "
        "must filter nothing; see plausibility_census's docstring")
    assert census["rows_read"] == sum(c["n_rows"] for c in per_country)
    assert census["rows_would_be_refused"] == sum(
        c["n_would_be_refused"] for c in per_country)
    assert census["rows_read"] == report["meta"]["guard_rows_read"]
    assert census["rows_would_be_refused"] == report["meta"]["guard_would_refuse"]
    assert census["rows_dropped"] == report["meta"]["guard_rows_dropped"]
