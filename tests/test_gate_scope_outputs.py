"""ABL-387: a gate scope's output paths are part of its registration.

Both harnesses gained `--scope` (ABL-322 for wind, ABL-378 for solar), but
neither keyed its output paths off it. The three defaults were fixed strings on
the arguments themselves, resolved by argparse *before* `--scope` was consulted:

    scripts/evaluate_wind_retrain.py   experiments/ABL195/{artifacts,results.json}
                                       reports/abl_195_wind_retrain.md
    scripts/evaluate_solar_retrain.py  experiments/ABL253/{artifacts,results.json}
                                       reports/abl_253_solar_retrain.md

So a scoped run that omitted three flags overwrote a *dispositioned* gate read in
place. Not hypothetical: `experiments/ABL195/results.json` (35,099 bytes,
2026-08-11) and `experiments/ABL253/results.json` (20,436 bytes, 2026-08-12) both
back dispositions that had already been reported, and wind already had a second
registered scope with more tranches queued behind it.

The failure mode is the dangerous one — the run **succeeds**, emits a full
report, and the damage is to evidence rather than to anything its exit status
shows. Nothing in a passing suite or a green run would have surfaced it, so the
properties are pinned here.

Read by AST rather than by import, following `test_gate_scope_registration.py`:
both harnesses import catboost and xgboost at module scope, and the registration
tables are dict literals, so `ast.literal_eval` reads them without a fit.
"""
import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.gate_registration import (
    REQUIRED_OUTPUT_KEYS, check_registration_tables, check_scope_outputs,
)

WIND_HARNESS = REPO / "scripts" / "evaluate_wind_retrain.py"
SOLAR_HARNESS = REPO / "scripts" / "evaluate_solar_retrain.py"

#: The paths each gate was actually read at, written out here rather than
#: referenced, so a table edit that relocates a dispositioned read has to
#: disagree with a literal in a test instead of moving quietly with the code.
ABL195_OUTPUTS = {"artifact_dir": "experiments/ABL195/artifacts",
                  "json_out": "experiments/ABL195/results.json",
                  "report_out": "reports/abl_195_wind_retrain.md"}
ABL253_OUTPUTS = {"artifact_dir": "experiments/ABL253/artifacts",
                  "json_out": "experiments/ABL253/results.json",
                  "report_out": "reports/abl_253_solar_retrain.md"}

#: The three flags whose defaults carried the defect.
OUTPUT_FLAGS = ("--artifact-dir", "--json-out", "--report-out")


def _module_const(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _add_argument_calls(source: str):
    return {node.args[0].value: node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args and isinstance(node.args[0], ast.Constant)}


def _module_level_calls(source: str):
    """Names of functions called at module scope — i.e. run on import."""
    return {getattr(node.value.func, "id", getattr(node.value.func, "attr", ""))
            for node in ast.parse(source).body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)}


HARNESSES = pytest.mark.parametrize("harness", [WIND_HARNESS, SOLAR_HARNESS],
                                    ids=["wind", "solar"])


@pytest.fixture(scope="module")
def sources():
    return {path: path.read_text(encoding="utf-8") for path in (WIND_HARNESS, SOLAR_HARNESS)}


# --------------------------------------------------------------------------
# The registered paths themselves
# --------------------------------------------------------------------------

def test_wind_default_scope_keeps_the_abl195_paths(sources):
    """An unflagged wind run still writes exactly where ABL-195 was read.

    This is the compatibility half of the fix: keying outputs off the scope must
    not relocate a dispositioned read, or the fix for overwriting the evidence
    would itself orphan it.
    """
    outputs = _module_const(sources[WIND_HARNESS], "SCOPE_OUTPUTS")
    assert outputs["abl195"] == ABL195_OUTPUTS


def test_solar_default_scope_keeps_the_abl253_paths(sources):
    """An unflagged solar run still writes exactly where ABL-253 was read."""
    outputs = _module_const(sources[SOLAR_HARNESS], "SCOPE_OUTPUTS")
    assert outputs["abl253"] == ABL253_OUTPUTS


def test_the_default_scope_is_the_one_holding_the_historical_paths(sources):
    """`--scope`'s default must be the scope registered at the historical paths.

    Pinned separately from the two above: if the argparse default moved to
    another scope while `SCOPE_OUTPUTS` kept its entries, an unflagged run would
    still not reproduce the dispositioned read, and both tests above would pass.
    """
    for harness, scope in ((WIND_HARNESS, "abl195"), (SOLAR_HARNESS, "abl253")):
        scope_arg = _add_argument_calls(sources[harness])["--scope"]
        default = next(kw.value.value for kw in scope_arg.keywords if kw.arg == "default")
        assert default == scope, f"{harness.name} defaults to scope {default!r}"


def test_tranche1a_registers_the_paths_its_gate_read_was_published_at(sources):
    """`abl380-tranche1a` writes where ABL-380's PASS was actually read.

    Pinned as literals for the same reason ABL-195's are: this scope's read is
    dispositioned — 6/6 PASS, with the Board asked to review it — so an edit here
    relocates evidence someone has already been pointed at. The two artifact
    SHA-256 values published in that report's fit-audit table,
    `eb0f63d8...43ea` (BG) and `5d2ec407...0840` (CH), were reproduced from the
    files under `experiments/ABL348/artifacts`, which is what makes this triple
    measured rather than assigned.

    It writes under `ABL348` and not an `ABL380` directory of its own because
    the registration it is fitted under is `experiments/ABL348/config.json`,
    frozen at ABL-348 — the scope name keys the table, the issue number does not.
    """
    outputs = _module_const(sources[WIND_HARNESS], "SCOPE_OUTPUTS")
    assert outputs["abl380-tranche1a"] == {
        "artifact_dir": "experiments/ABL348/artifacts",
        "json_out": "experiments/ABL348/results_abl380_tranche1a.json",
        "report_out": "reports/abl_380_wind_onshore_tranche1a.md"}


def test_tranche1a_machine_record_stays_out_of_the_results_json_glob(sources):
    """Its `json_out` must not be renamed into `.gitignore`'s blind spot.

    `experiments/*/results.json` matches on the exact filename, so renaming this
    entry to `results.json` for consistency would silently untrack the machine
    record `reports/abl_380_tranche1a_findings.md:9` cites, and would restore
    exactly the review-invisibility that made this issue's failure mode
    unobservable: an overwritten gate read shows nothing in `git status`.
    """
    outputs = _module_const(sources[WIND_HARNESS], "SCOPE_OUTPUTS")
    json_out = Path(outputs["abl380-tranche1a"]["json_out"])
    assert json_out.name != "results.json", (
        "renaming this to results.json puts a committed, cited gate record back "
        "under .gitignore:53, where an overwrite is invisible to review")
    assert (REPO / json_out).exists(), f"{json_out} is cited by a report but absent"


@HARNESSES
def test_every_scope_resolves_a_distinct_output_triple(sources, harness):
    """Two scopes sharing an output path is the defect, between named scopes.

    `check_scope_outputs` enforces this at import; asserting it here as well
    means the property is stated where a reviewer reads the registration, and a
    harness that dropped the import-time call still fails on the content.
    """
    outputs = _module_const(sources[harness], "SCOPE_OUTPUTS")
    check_scope_outputs(outputs)
    for key in REQUIRED_OUTPUT_KEYS:
        paths = [entry[key] for entry in outputs.values()]
        assert len(set(paths)) == len(paths), f"{harness.name}: duplicate {key}"


@HARNESSES
def test_registration_tables_are_keyed_by_the_same_scopes(sources, harness):
    """SCOPES / GATE_BASIS / SCOPE_OUTPUTS are one registration in three views."""
    source = sources[harness]
    check_registration_tables(
        SCOPES=_module_const(source, "SCOPES"),
        GATE_BASIS=_module_const(source, "GATE_BASIS"),
        SCOPE_OUTPUTS=_module_const(source, "SCOPE_OUTPUTS"))


@HARNESSES
def test_experiment_outputs_stay_one_directory_deep(sources, harness):
    """`.gitignore:53` and `:56` glob one level, so a nested path is committable.

    `experiments/*/results.json` and `experiments/*/artifacts/` do not match
    `experiments/ABL322/pilot/artifacts`. A scope registered at a nested path
    would commit a binary model artifact on the next `git add`, which is the
    opposite of what those two globs are for.

    Depth alone does not decide tracking, and this test does not claim it does.
    The artifacts glob keys on the directory *name* and the results glob on the
    exact *filename*, so a one-level `json_out` not named `results.json` is
    tracked — `abl380-tranche1a`'s is, deliberately, and is pinned below. What
    this test governs is only the depth, which is the half that decides whether
    a model binary can be committed by accident.
    """
    for scope, entry in _module_const(sources[harness], "SCOPE_OUTPUTS").items():
        for key in ("artifact_dir", "json_out"):
            parts = Path(entry[key]).as_posix().split("/")
            assert parts[0] == "experiments" and len(parts) == 3, (
                f"{scope}.{key} = {entry[key]!r} is not experiments/<dir>/<name>")


# --------------------------------------------------------------------------
# The resolution order — the actual defect
# --------------------------------------------------------------------------

@HARNESSES
def test_output_flags_carry_no_hardcoded_default(sources, harness):
    """The defect itself: a literal default here is resolved before `--scope`.

    `argparse` fills these in at `parse_args`, so any non-None default is the
    path the run writes to whatever scope was asked for. They must default to
    `None` and be resolved against `SCOPE_OUTPUTS` after parsing.
    """
    calls = _add_argument_calls(sources[harness])
    for flag in OUTPUT_FLAGS:
        assert flag in calls, f"{harness.name} no longer defines {flag}"
        defaults = [kw.value for kw in calls[flag].keywords if kw.arg == "default"]
        assert defaults, f"{harness.name} {flag} declares no default"
        assert all(isinstance(node, ast.Constant) and node.value is None
                   for node in defaults), (
            f"{harness.name} {flag} has a scope-independent default; a scoped run "
            "that omits this flag would write over another scope's evidence")


@HARNESSES
def test_an_explicit_flag_still_overrides_the_registration(sources, harness):
    """Each output is resolved as `args.<flag> or outputs[<key>]`.

    Both halves matter: the `or` is what keeps an explicit path working for
    ad-hoc reads, and the right operand is what makes the unflagged run land on
    the scope's own paths.
    """
    resolutions = {}
    for node in ast.walk(ast.parse(sources[harness])):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and len(node.values) == 2):
            continue
        left, right = node.values
        if isinstance(left, ast.Attribute) and isinstance(right, ast.Subscript):
            resolutions[left.attr] = right
    for flag in OUTPUT_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        assert attr in resolutions, (
            f"{harness.name}: {flag} is not resolved as `args.{attr} or outputs[...]`")
        assert resolutions[attr].slice.value == attr, (
            f"{harness.name}: {flag} falls back to the wrong SCOPE_OUTPUTS key")


@HARNESSES
def test_harness_checks_its_registration_tables_on_import(sources, harness):
    """The check runs at module scope, so it fires before any fit.

    A scope registered in `SCOPES` but not `SCOPE_OUTPUTS` would otherwise raise
    `KeyError` inside `main()` — after argument parsing, and in the wind harness
    after the replica has been opened. At import it also fires under `--help`
    and in this suite.
    """
    called = _module_level_calls(sources[harness])
    assert "check_registration_tables" in called
    assert "check_scope_outputs" in called


# --------------------------------------------------------------------------
# The shared checker
# --------------------------------------------------------------------------

def test_check_registration_tables_accepts_identical_keys():
    check_registration_tables(SCOPES={"a": 1, "b": 2}, GATE_BASIS={"b": 3, "a": 4})


def test_check_registration_tables_names_every_omission():
    """One scope added to one of three tables reports both gaps at once.

    Reporting one per run turns a three-table registration into three edit-run
    cycles, which is how a table gets half-filled in the first place.
    """
    with pytest.raises(KeyError) as excinfo:
        check_registration_tables(SCOPES={"a": 1, "new": 2}, GATE_BASIS={"a": 3},
                                  SCOPE_OUTPUTS={"a": 4})
    message = str(excinfo.value)
    assert "GATE_BASIS is missing 'new'" in message
    assert "SCOPE_OUTPUTS is missing 'new'" in message


def test_check_registration_tables_rejects_a_single_table():
    """Nothing is cross-checked, so silently passing would be a false green."""
    with pytest.raises(ValueError):
        check_registration_tables(SCOPES={"a": 1})


def test_check_scope_outputs_rejects_a_shared_path():
    """Two scopes writing the same results.json is this defect between scopes."""
    with pytest.raises(ValueError, match="overwrite that scope's evidence"):
        check_scope_outputs({
            "a": {"artifact_dir": "experiments/A/artifacts",
                  "json_out": "experiments/SHARED/results.json",
                  "report_out": "reports/a.md"},
            "b": {"artifact_dir": "experiments/B/artifacts",
                  "json_out": "experiments/SHARED/results.json",
                  "report_out": "reports/b.md"}})


def test_check_scope_outputs_rejects_an_incomplete_entry():
    """A missing or typo'd key leaves that output falling through to a default."""
    with pytest.raises(KeyError):
        check_scope_outputs({"a": {"artifact_dir": "experiments/A/artifacts",
                                   "json_out": "experiments/A/results.json"}})
    with pytest.raises(KeyError):
        check_scope_outputs({"a": {"artifact_dir": "experiments/A/artifacts",
                                   "json_out": "experiments/A/results.json",
                                   "report_out": "reports/a.md",
                                   "reprot_out": "reports/typo.md"}})
