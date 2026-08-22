"""ABL-426: which table a scope is read on is part of its registration.

ABL-387 established that *where* a scope writes cannot be a flag default, because
a scoped run that omits the flag then overwrites another scope's evidence. This
file is the same argument about what a scope **reads**, established the same way
-- by it happening.

`--renewable-source` (ABL-345) was opt-in, and the fall-through was the global
`db.RENEWABLE_TYPE_SOURCE_TABLE`, consulted without reference to the scope.
ABL-405 ran `--scope abl316-t2a` without the flag. ABL-348 registers
`energy_generation` for all 37 tranche pairs and names the source table in
`voids_this_registration`; the run read `energy_renewable` for the fitted series,
its lag and rolling features, the D-7 and persistence baselines, the gate actuals
and the ABL-188 screen, then fitted, scored, graded and emitted a 24-cell evidence
pack, and exited 0. Its own machine record was truthful throughout
(`meta.training_source: energy_renewable`) while the report H1 and the findings
pack both said `energy_generation`. Three published artefacts of one read
disagreed and nothing said which was wrong.

So the properties here are the two halves of that:

  * **Resolution.** A scope's source is elected in the file, in review, before the
    fit -- and an unflagged run of any scope reads *that scope's* table. This is
    the behaviour change; had it existed, ABL-405's unflagged run would have read
    the registered table and there would be no defect.
  * **Correspondence.** The registration table, the committed machine record and
    the report heading of every published scope agree about which table was read.
    Each of the three was individually correct at some point in ABL-405's history;
    what was missing was anything requiring them to be correct *together*.

`test_a_published_scope_heading_cannot_name_a_table_the_run_did_not_read` is the
one that catches this defect from the direction it was actually noticed from --
a reader opening two files and finding two different tables named.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src import db  # noqa: E402

SOLAR_HARNESS = REPO / "scripts" / "evaluate_solar_retrain.py"

#: The table names a report heading could plausibly name, so a title claiming one
#: can be checked against the record. Derived from the loader's own registry
#: rather than listed, so a third source table added later is covered without an
#: edit here.
KNOWN_SOURCES = tuple(db._RENEWABLE_TYPE_SOURCES)


def _module_const(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _checked_tables(source: str) -> set:
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "check_registration_tables"):
            return {kw.arg for kw in node.keywords}
    raise AssertionError("check_registration_tables call not found")


@pytest.fixture(scope="module")
def harness_source():
    return SOLAR_HARNESS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def harness():
    """The harness as a module object.

    Several of the registration tables hold *names* rather than literals --
    `FIT_WINDOW`, `SIGN_TEST`, `DELTA_MIN`, `LEGACY_FEATURE_COLUMNS` -- which
    `ast.literal_eval` cannot read. Importing is also the stronger check for the
    A/B assertions below: it compares the values the run would resolve, not two
    spellings that happen to match in the source.

    Loaded under a name of its own so its `if __name__ == '__main__'` guard stays
    shut, per `tests/test_solar_gate_source.py`.
    """
    spec = importlib.util.spec_from_file_location(
        "scripts_evaluate_solar_retrain_abl426", SOLAR_HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scope_sources(harness_source):
    return _module_const(harness_source, "SCOPE_SOURCES")


@pytest.fixture(scope="module")
def scopes(harness_source):
    return _module_const(harness_source, "SCOPES")


@pytest.fixture(scope="module")
def scope_outputs(harness_source):
    return _module_const(harness_source, "SCOPE_OUTPUTS")


@pytest.fixture(scope="module")
def scope_titles(harness_source):
    return _module_const(harness_source, "SCOPE_TITLES")


def _published(scope_outputs):
    """Scopes whose machine record is committed, keyed to the loaded record.

    Derived from `SCOPE_OUTPUTS` and the working tree rather than listed, for the
    reason the sibling tests give: a hand-maintained set of published scopes is
    one more thing that can disagree with the evidence.
    """
    out = {}
    for scope, entry in scope_outputs.items():
        path = REPO / entry["json_out"]
        if path.exists():
            out[scope] = (path, json.loads(path.read_text(encoding="utf-8")))
    return out


# ---------------------------------------------------------------- resolution


def test_every_scope_registers_a_source(scopes, scope_sources):
    """No scope may resolve its source through a global default.

    The complement of this -- a scope with no row -- is what ABL-405 was. It is
    unreachable from a run, because `check_registration_tables` fires at import;
    this asserts the same property against the table directly, so the reason it
    holds is legible without tracing that call.
    """
    missing = sorted(set(scopes) - set(scope_sources))
    assert not missing, (
        f"scopes with no registered source table: {missing}. A scope that omits a "
        "`SCOPE_SOURCES` row falls through to a global constant chosen without "
        "reference to the registration, which is ABL-426.")


def test_no_scope_registers_a_source_that_does_not_exist(scope_sources):
    """A typo must fail at import, not at the first SQL read or -- worse -- on
    another real table that happens to carry a `solar_mw` column."""
    for scope, source in scope_sources.items():
        assert source in KNOWN_SOURCES, (
            f"SCOPE_SOURCES[{scope!r}] is {source!r}, which "
            f"`db._RENEWABLE_TYPE_SOURCES` does not know: {KNOWN_SOURCES}")


def test_the_source_table_is_checked_at_import_not_declared_unchecked(harness_source):
    """`SCOPE_SOURCES` must be in `check_registration_tables`, not exempted.

    The exemption list's contract is "an omitted row for this table defaults
    silently", and a silent default is the entire mechanism of ABL-426. The
    ABL-387/ABL-404 objection to joining the call -- that it aborts on scopes
    whose absence is deliberate -- does not reach this table: every scope is read
    on exactly one table, always, so no absence here is deliberate.
    """
    assert "SCOPE_SOURCES" in _checked_tables(harness_source)
    declared = _module_const(harness_source, "UNCHECKED_REGISTRATION_TABLES")
    assert "SCOPE_SOURCES" not in declared


def test_an_unflagged_run_reads_the_scope_s_registered_table(scopes, scope_sources, harness):
    """The behaviour change, asserted through the resolver the run uses.

    Not through `main()`: the builder-spy route needs a replica fixture per scope
    and would re-prove `tests/test_solar_gate_source.py`'s passthrough property
    rather than this one. What is new here is that the *default* is per-scope, and
    `source_for` is the single place `main` takes it from.
    """
    module = harness
    for scope in scopes:
        assert module.source_for(scope) == scope_sources[scope]
    # And the global constant is no longer the fall-through for a scope that
    # registers something else -- the specific substitution ABL-426 makes.
    off_default = [s for s, t in scope_sources.items() if t != db.RENEWABLE_TYPE_SOURCE_TABLE]
    assert off_default, (
        "no scope registers a table other than the global default, so this test "
        "would pass even if the substitution had not been made")
    for scope in off_default:
        assert module.source_for(scope) != db.RENEWABLE_TYPE_SOURCE_TABLE


# ------------------------------------------------------------ correspondence


def test_every_published_scope_registers_the_table_its_record_says_it_read(
        scope_outputs, scope_sources):
    """The registration must record the read, not an intention about it.

    This is the assertion that keeps `SCOPE_SOURCES['abl316-t2a']` honest. Setting
    that row to `energy_generation` -- the table ABL-348 *registers* -- would read
    as a fix and would be the ABL-404 failure mode: an unflagged `--scope
    abl316-t2a` would then refit eight countries on a different table and
    overwrite a dispositioned 24-cell pack in place, under a heading naming
    ABL-405. Correcting the read is a separate scope; this row's job is that a
    re-run reproduces what was published.
    """
    for scope, (path, record) in _published(scope_outputs).items():
        recorded = record["meta"]["training_source"]
        assert scope_sources[scope] == recorded, (
            f"SCOPE_SOURCES[{scope!r}] is {scope_sources[scope]!r} but "
            f"{path.relative_to(REPO)} records `meta.training_source` "
            f"{recorded!r}. A re-run of this scope would not reproduce its "
            "published read.")


def test_a_published_scope_heading_cannot_name_a_table_the_run_did_not_read(
        scope_outputs, scope_titles):
    """The direction ABL-426 was actually noticed from.

    A heading is the first thing quoted out of an evidence pack and
    `meta.training_source` is not, so a heading naming the wrong table survives
    being read, cited and retro-graded -- which is what happened: ABL-418 graded
    all 24 of ABL-405's cells off a record whose report was headed
    `energy_generation` over a run on `energy_renewable`.

    Only headings that name a table at all are constrained. `abl253`'s and
    `abl376`'s name none and are left alone, which is why this is not an
    instruction to put the table in every heading.
    """
    for scope, (path, record) in _published(scope_outputs).items():
        title = scope_titles.get(scope)
        if title is None:
            continue
        named = [s for s in KNOWN_SOURCES if s in title]
        if not named:
            continue
        assert named == [record["meta"]["training_source"]], (
            f"SCOPE_TITLES[{scope!r}] names {named} but "
            f"{path.relative_to(REPO)} records `meta.training_source` "
            f"{record['meta']['training_source']!r}. The report heading and the "
            "machine record are two artefacts of one read and cannot say "
            "different things about which table it was.")


def test_a_published_report_heading_matches_the_registered_title(
        scope_outputs, scope_titles):
    """And the committed report file carries the heading the harness would emit.

    Without this, correcting `SCOPE_TITLES` fixes only what a *future* run would
    print, and the published file keeps the wrong heading -- so the test above
    would pass while the artefact a reader opens still contradicts the record.
    """
    for scope, entry in scope_outputs.items():
        report = REPO / entry["report_out"]
        title = scope_titles.get(scope)
        if not report.exists() or title is None:
            continue
        first = report.read_text(encoding="utf-8").splitlines()[0]
        assert first == f"# {title}", (
            f"{report.relative_to(REPO)} is headed {first!r} but "
            f"SCOPE_TITLES[{scope!r}] would emit '# {title}'.")


# ------------------------------------------------------- the corrected re-read


def test_tranche2a_generation_is_tranche2a_on_the_registered_table(
        scopes, scope_sources):
    """The A/B is on the source table and on nothing else.

    Same countries in the same order; the two source rows differ, and they are the
    only registration values that do. Asserted rather than commented because the
    entire value of this scope is that one difference: a second difference makes
    the diff between the two machine records unattributable, and would not
    otherwise announce itself.
    """
    assert scopes["abl316-t2a-generation"] == scopes["abl316-t2a"]
    assert scope_sources["abl316-t2a"] == "energy_renewable"
    assert scope_sources["abl316-t2a-generation"] == "energy_generation"


@pytest.mark.parametrize("table", ["GATE_BASIS", "FIT_RULES", "CAUSAL_LEVELLING",
                                   "G23_READABILITY", "SEED_READABILITY"])
def test_tranche2a_generation_holds_every_other_registered_value(harness, table):
    """Every value-carrying registration table agrees between the two arms.

    Parametrised over the tables by name so a table added by a later tranche is a
    one-line addition here rather than a silently uncovered confound.
    `SCOPE_FEATURES` is deliberately not in the list: **both** scopes are absent
    from it, so they resolve through the same `DEFAULT_SCOPE_FEATURES` in the same
    process, which is a stronger identity than two pins. `SCOPE_NOT_EVALUABLE` is
    absent from both for the same reason -- neither declares a cell unscorable.
    """
    values = getattr(harness, table)
    assert values["abl316-t2a-generation"] == values["abl316-t2a"], (
        f"{table} differs between the two arms of the ABL-426 A/B. Only the "
        "source table may.")


@pytest.mark.parametrize("scope", ["SCOPE_FEATURES", "SCOPE_NOT_EVALUABLE"])
def test_neither_arm_pins_a_table_the_other_inherits(harness, scope):
    """The absences are symmetric too.

    A row on one arm only would flip `meta.feature_set_is_registered_for_scope` in
    one machine record and not the other -- a difference between the two records
    that is not a difference between the two reads.
    """
    values = getattr(harness, scope)
    assert ("abl316-t2a" in values) == ("abl316-t2a-generation" in values)


def test_the_corrected_read_writes_nowhere_the_published_read_writes(scope_outputs):
    """ABL-387's property, on the one pair of scopes where it bites hardest.

    These two name the same eight countries, so a shared `artifact_dir` would
    overwrite `experiments/ABL405/artifacts/<CC>/solar/model.joblib` for all eight
    -- the artifacts ABL-405's machine record cites by SHA-256 -- and `git status`
    would show nothing, because `.gitignore:56` ignores both directories.
    `check_scope_outputs` enforces distinctness across the whole table; this names
    the pair the argument is about.
    """
    published, corrected = scope_outputs["abl316-t2a"], scope_outputs["abl316-t2a-generation"]
    for key in ("artifact_dir", "json_out", "report_out"):
        assert published[key] != corrected[key], f"both scopes register {key} {published[key]!r}"


# ------------------------------------------------------ the off-registration line


def _minimal_result(harness, scope, source, registered_source):
    """The smallest `result` dict `render_markdown` will accept.

    Built from the harness's own registration tables rather than literals, so this
    fixture cannot drift from them -- the idiom `tests/test_solar_gate_source.py`
    uses for the same reason.
    """
    from src.evaluation.scorecard import ScorecardConfig, opened_databases
    databases = opened_databases(ScorecardConfig("r.db", None, "2026-07-11", "2026-08-10"),
                                 "r.db", "r.db")
    return {
        "meta": {"generated_at": "2026-08-22 00:00 UTC", "replica_db": "r.db",
                 "replica_bytes": 1, "databases": databases,
                 "training_source": source,
                 "registered_source": registered_source,
                 "source_is_scope_registered": source == registered_source,
                 "scope": scope,
                 "registered_countries": list(harness.SCOPES[scope]),
                 "registered_cells": len(harness.SCOPES[scope]) * len(harness.PRIMARY_BANDS),
                 "gate_basis": list(harness.GATE_BASIS[scope]),
                 "fit_window": {"start": "2026-01-14", "end_exclusive": "2026-07-11"},
                 "gate_window": {"start": "2026-07-11", "end_exclusive": "2026-08-10"}},
        "verdict": "PASS", "recommendation": "-", "gate_cells": [], "country_d2": [],
        "training": [{"country": "BG", "algorithm": "catboost",
                      "audit": {"retained_rows": 1, "intended_rows": 1, "unique_targets": 1,
                                "excluded_missing_actual_or_feature": 0,
                                "degraded_lag_1d_rows": 0},
                      "constant_runs": [], "artifact_sha256": "abc"}],
    }


def test_a_compliant_run_says_nothing_new(harness):
    """The line must be silent when the two agree, or it moves published text.

    Every compliant run is this case, so a version of this that always printed
    would change the report of every scope in the table for a condition none of
    them is in.
    """
    markdown = harness.render_markdown(
        _minimal_result(harness, "abl316-t2a-generation",
                        "energy_generation", "energy_generation"))
    assert "OFF-REGISTRATION" not in markdown
    assert "contamination screen: `energy_generation`." in markdown


def test_an_off_registration_run_says_so_on_the_page(harness):
    """Both tables named, in the report, not only in the machine record.

    This is the half of the guard that addresses how ABL-426 was actually noticed:
    a reader opened the findings pack and the machine record and found two
    different tables, with nothing on either page saying which was the
    registration. A run in that state now cannot produce a page that looks
    compliant.
    """
    markdown = harness.render_markdown(
        _minimal_result(harness, "abl316-t2a-generation",
                        "energy_renewable", "energy_generation"))
    assert "OFF-REGISTRATION" in markdown
    assert "`abl316-t2a-generation` registers `energy_generation`" in markdown
    assert "this run read `energy_renewable`" in markdown


def test_a_record_written_before_this_field_existed_renders_unchanged(harness):
    """ABL-405's committed record has no `source_is_scope_registered` key.

    It must render as it always did. Retro-flagging it from `SCOPE_SOURCES` was
    considered and rejected: that table records what each scope *was read on*, so
    `abl316-t2a` looks compliant against its own row by construction. The record
    of the violation belongs in ABL-405's pack, which states it in prose, and in
    this table's comment -- not in a boolean that would read as though the run had
    measured something it did not.
    """
    result = _minimal_result(harness, "abl316-t2a", "energy_renewable", "energy_renewable")
    del result["meta"]["source_is_scope_registered"]
    del result["meta"]["registered_source"]
    markdown = harness.render_markdown(result)
    assert "OFF-REGISTRATION" not in markdown
    assert "contamination screen: `energy_renewable`." in markdown
