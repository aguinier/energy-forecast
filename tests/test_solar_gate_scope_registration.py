"""ABL-379: the solar gate harness can read a gate for a country with no incumbent.

ABL-322 rebuilt the wind harness around a scope/pre-registration model and fixed
two defects that would otherwise have made every new-country gate read unusable.
`scripts/evaluate_solar_retrain.py` received none of it, and it is the path
behind 19 of ABL-316's 37 remaining pairs. On `origin/main` at `97a1b49` it:

  * had no `--scope` and no `--countries` -- `COUNTRIES = ("BE", "DE", "FR")` was
    a module constant in `src/evaluation/solar_retrain.py`, so reading a BG gate
    meant editing a source file, and an unedited run refitted the three serving
    incumbents instead;
  * gated on `("challenger", "incumbent", "seasonal_naive", "persistence")`,
    hardcoded at two sites, so for any pair with zero rows in `forecasts` the
    intersection was empty and every cell scored n=0 with all scores None;
  * compared `len(gate_cells) == 9`, so a two-pair tranche could not return PASS
    whatever its numbers said;
  * had no verdict for "the comparison never happened".

The failure mode is the dangerous one: not a crash, a plausible FAIL. A tranche
whose gate reads FAIL because the intersection was empty looks exactly like a
tranche whose model is bad, and the correct response to those two is opposite.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.solar_retrain import PRIMARY_BANDS

HARNESS = REPO / "scripts" / "evaluate_solar_retrain.py"

#: The three pairs serving today. ABL-253 registered exactly these; a tranche
#: scope must refit none of them, because a refit on a different source silently
#: changes a live model's gate evidence.
SERVING_PAIRS = {("solar", "BE"), ("solar", "DE"), ("solar", "FR")}


def _module_const(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


@pytest.fixture(scope="module")
def harness_source():
    return HARNESS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def scopes(harness_source):
    return {name: {tuple(pair) for pair in pairs}
            for name, pairs in _module_const(harness_source, "SCOPES").items()}


@pytest.fixture(scope="module")
def gate_basis(harness_source):
    return {name: tuple(cols)
            for name, cols in _module_const(harness_source, "GATE_BASIS").items()}


@pytest.fixture(scope="module")
def scope_outputs(harness_source):
    return _module_const(harness_source, "SCOPE_OUTPUTS")


# --------------------------------------------------------------------------
# 1. The denominator is derived from the scope, not hardcoded to 9.
# --------------------------------------------------------------------------

def test_every_scope_is_reachable(scopes):
    """A full pass must be able to return PASS under every registered scope.

    Under the old `len(gate_cells) == 9 and passed == 9` this was false for
    every scope but ABL-253's: a two-pair tranche produces 6 cells, so
    `performance_pass` was False whatever the numbers said, and the FAIL text
    rendered a flawless run as "only 6/9 primary cells clear the registered bar".
    """
    for name, pairs in scopes.items():
        registered_cells = len(pairs) * len(PRIMARY_BANDS)
        assert registered_cells > 0, f"scope {name!r} registers no cells"
        assert registered_cells == len(pairs) * len(PRIMARY_BANDS)


def test_the_bar_is_not_a_literal_nine(harness_source):
    """`== 9` and `/9` must be gone from the disposition and its prose."""
    assert "len(gate_cells) == 9" not in harness_source
    assert "passed}/9" not in harness_source
    assert "{registered_cells}" in harness_source


def test_registered_cells_is_derived_from_the_scopes_pair_count(harness_source):
    """The bar is `len(registered_pairs) * len(PRIMARY_BANDS)`, as in wind."""
    assert "registered_cells = len(registered_pairs) * len(PRIMARY_BANDS)" in harness_source


# --------------------------------------------------------------------------
# 2. The default/unflagged invocation still reproduces ABL-253.
# --------------------------------------------------------------------------

def test_default_scope_reproduces_abl253(scopes):
    """The unflagged run is still the ABL-253 gate: same pairs, same 9 cells.

    Pinned against `experiments/ABL253/config.json` -- the frozen
    pre-registration -- rather than against `git show origin/main:...`. The wind
    twin of this test reads main's `PAIRS` constant, so it went red the moment
    ABL-322 merged and replaced that constant with `SCOPES`: a test whose
    reference is a moving branch stops testing the thing it names as soon as it
    lands. The registration is the authority for what ABL-253 registered, and it
    does not move.
    """
    registered = json.loads(
        (REPO / "experiments" / "ABL253" / "config.json").read_text(encoding="utf-8"))
    registered_pairs = {(stream, country)
                        for stream, spec in registered["pairs"].items()
                        for country in spec["countries"]}

    assert scopes["abl253"] == registered_pairs == SERVING_PAIRS, (
        "the default scope no longer reproduces ABL-253's registered pair set")
    assert len(scopes["abl253"]) * len(PRIMARY_BANDS) == 9


def test_default_scope_flag_is_abl253(harness_source):
    """An unflagged run must select `abl253`, not merely be able to."""
    scope_arg = _add_argument(harness_source, "--scope")
    default = next(kw.value.value for kw in scope_arg.keywords if kw.arg == "default")
    assert default == "abl253"


def test_default_scope_keeps_abl253s_registered_output_paths(scope_outputs):
    """An unflagged run must still write exactly where ABL-253 was read from."""
    assert scope_outputs["abl253"] == {
        "artifact_dir": "experiments/ABL253/artifacts",
        "json_out": "experiments/ABL253/results.json",
        "report_out": "reports/abl_253_solar_retrain.md",
    }


# --------------------------------------------------------------------------
# 3. Scope selection is a pre-registration, not a country filter.
# --------------------------------------------------------------------------

def _add_argument(source: str, flag: str):
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            return node
    raise AssertionError(f"{flag} not found")


def test_scope_is_a_choice_not_a_country_filter(harness_source):
    """A country filter cannot express which pairs a run registered.

    In wind it also left the cell bar and the gate basis behind, which is how
    that harness first reported its pilot as a FAIL.
    """
    flags = {node.args[0].value for node in ast.walk(ast.parse(harness_source))
             if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
             and node.args and isinstance(node.args[0], ast.Constant)}
    assert "--scope" in flags
    assert "--countries" not in flags, (
        "--countries is a filter over the registered scope; scoping a run is a "
        "new pre-registration (see SCOPES)")
    assert "choices" in {kw.arg for kw in _add_argument(harness_source, "--scope").keywords}


def test_reading_a_new_country_needs_no_source_edit():
    """`COUNTRIES` is gone: a second source of truth for the registered set."""
    from src.evaluation import solar_retrain
    assert not hasattr(solar_retrain, "COUNTRIES")
    assert "COUNTRIES" not in solar_retrain.__all__


def test_tranche_scope_refits_nothing_that_serves(scopes):
    for name, pairs in scopes.items():
        if name == "abl253":
            continue
        assert not pairs & SERVING_PAIRS, (
            f"scope {name!r} refits serving pairs: {sorted(pairs & SERVING_PAIRS)}")


def test_the_level_scope_is_abl348s_recommended_first_tranche(scopes):
    """BG and CH solar: the two pairs whose `energy_generation` history starts
    on the same day as `energy_renewable`, so the source change costs them no
    depth (ABL-348 §5). Both sit outside ABL-348's `not_evaluable` list."""
    assert scopes["abl348-level"] == {("solar", "BG"), ("solar", "CH")}
    assert len(scopes["abl348-level"]) * len(PRIMARY_BANDS) == 6


# --------------------------------------------------------------------------
# 4. The gate basis is registered per scope, and no new scope gates on the
#    incumbent it does not have.
# --------------------------------------------------------------------------

def test_every_scope_registers_a_gate_basis(scopes, gate_basis):
    """The basis is a registered property of the scope, like the pair list."""
    assert set(gate_basis) == set(scopes), (
        "every registered scope needs a registered gate basis")


def test_gate_basis_always_contains_the_two_columns_the_bar_names(gate_basis):
    """The bar is `challenger WAPE < seasonal-naive D-7 WAPE`. Both must be in it."""
    for name, basis in gate_basis.items():
        assert {"challenger", "seasonal_naive"} <= set(basis), (
            f"scope {name!r} gates on a basis missing the columns its bar names")


def test_tranche_scopes_do_not_gate_on_the_incumbent(scopes, gate_basis):
    """The defect this pins: all 19 new solar pairs have 0 rows in `forecasts`.

    With `incumbent` in the basis the intersection is empty for those pairs, so
    every cell scores n=0 with all scores None and the harness renders FAIL -- a
    model-quality verdict on a comparison that never happened.
    """
    for name in scopes:
        if name == "abl253":
            continue
        assert "incumbent" not in gate_basis[name], (
            f"scope {name!r} gates on an incumbent no registered pair has")
    assert gate_basis["abl348-level"] == ("challenger", "seasonal_naive")


def test_abl253_keeps_the_basis_it_was_published_under(gate_basis):
    """ABL-253's read is dispositioned; this change must not restate it.

    Same reason ABL-322 left `abl195` alone: re-basing an already-read gate
    would move numbers the Board has already seen, silently and as a side
    effect of a harness change.
    """
    assert gate_basis["abl253"] == (
        "challenger", "incumbent", "seasonal_naive", "persistence")


def test_the_incumbent_conjunct_is_not_hardcoded_at_the_scoring_sites(harness_source):
    """Both scoring sites read `GATE_BASIS[args.scope]`, not a literal tuple."""
    literal = '("challenger", "incumbent", "seasonal_naive", "persistence")'
    body = harness_source.split("def main(")[1]
    assert literal not in body, (
        "a scoring site still hardcodes the four-way basis; it must come from "
        "GATE_BASIS[args.scope]")
    assert "gate_basis = GATE_BASIS[args.scope]" in harness_source


# --------------------------------------------------------------------------
# 5. Output paths are keyed off the scope.
# --------------------------------------------------------------------------

def test_output_paths_are_registered_per_scope(scopes, scope_outputs):
    assert set(scope_outputs) == set(scopes)
    for name, paths in scope_outputs.items():
        assert set(paths) == {"artifact_dir", "json_out", "report_out"}, name


def test_no_two_scopes_share_an_output_path(scope_outputs):
    """A tranche run that forgets three flags must not overwrite ABL-253's
    dispositioned gate read in place."""
    for key in ("artifact_dir", "json_out", "report_out"):
        values = [paths[key] for paths in scope_outputs.values()]
        assert len(values) == len(set(values)), f"scopes share a {key}"


def test_output_paths_stay_one_level_under_experiments(scope_outputs):
    """`.gitignore` globs one level: `experiments/*/artifacts/` and
    `experiments/*/results.json` (`.gitignore:53-56`). A nested
    `experiments/ABL348/solar-level/...` slips past both and would commit a
    gate artifact tree derived from the 9 GB replica."""
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "experiments/*/artifacts/" in ignored
    assert "experiments/*/results.json" in ignored
    for name, paths in scope_outputs.items():
        assert paths["artifact_dir"].count("/") == 2, (
            f"scope {name!r} artifact_dir escapes the .gitignore glob")
        assert paths["json_out"].count("/") == 2, (
            f"scope {name!r} json_out escapes the .gitignore glob")


def test_path_flags_default_to_none_so_the_scope_decides(harness_source):
    for flag in ("--artifact-dir", "--json-out", "--report-out"):
        node = _add_argument(harness_source, flag)
        default = next(kw.value for kw in node.keywords if kw.arg == "default")
        assert isinstance(default, ast.Constant) and default.value is None, (
            f"{flag} still carries a hardcoded default; one scope can overwrite "
            "another's artifacts")


# --------------------------------------------------------------------------
# 6. The three scope tables cannot drift apart.
# --------------------------------------------------------------------------

def _harness_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_solar_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registering_a_scope_in_one_table_only_fails_at_import():
    """`SCOPES[args.scope]` would KeyError mid-run, and `GATE_BASIS[args.scope]`
    only after every pair had already been fitted. Say so at import."""
    module = _harness_module()
    module.SCOPES = {**module.SCOPES, "abl999-typo": (("solar", "AT"),)}
    with pytest.raises(RuntimeError, match="scope tables disagree"):
        module._check_scope_tables()


def test_a_scope_cannot_register_a_non_solar_stream():
    """This harness fits solar only. A scope naming another stream would be
    fitted with solar's algorithm and features."""
    module = _harness_module()
    module.SCOPES = {"abl253": (("wind_onshore", "BE"),)}
    module.GATE_BASIS = {"abl253": ("challenger", "seasonal_naive")}
    module.SCOPE_OUTPUTS = {"abl253": module.SCOPE_OUTPUTS["abl253"]}
    with pytest.raises(RuntimeError, match="non-solar streams"):
        module._check_scope_tables()


def test_a_scope_cannot_drop_a_column_its_bar_names():
    module = _harness_module()
    module.GATE_BASIS = {name: ("challenger", "persistence") for name in module.SCOPES}
    with pytest.raises(RuntimeError, match="missing a column its bar names"):
        module._check_scope_tables()
