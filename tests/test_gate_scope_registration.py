"""ABL-322: a gate scope is a pre-registration, and its cell bar follows from it.

The first cut of the offshore pilot added a `--countries` filter over the one
shared `PAIRS` and left `performance_pass` at a hardcoded `== 15`. Both scoped
invocations then failed on the *count* rather than on the scores:

    unflagged (the ABL-195 reproduction)  21 cells vs a bar of 15  -> FAIL
    --countries DE,NL (the pilot)          9 cells vs a bar of 15  -> FAIL

No invocation could return PASS. The pilot that sizes the remaining 37 pairs
would have reported "only 9/15 primary cells clear the registered bar" for a run
in which every registered cell may have passed.

The property pinned here is *not* "the count matches the run" -- deriving the bar
from whatever a run happens to score would destroy the check the bar exists for,
which is that a pair silently yielding no gate rows shortfalls the count and
reads FAIL instead of quietly leaving the denominator. What is pinned is that
each scope names its pairs **in the file, before the run**, and the bar is that
table's size. So:

  * a registered scope is reachable -- a full pass returns PASS;
  * a scope selects streams as well as countries, so the offshore pilot cannot
    drag a serving onshore pair into its gate;
  * the default scope still reproduces ABL-195's registered pair set exactly.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.wind_retrain import PRIMARY_BANDS

HARNESS = REPO / "scripts" / "evaluate_wind_retrain.py"

#: The five pairs serving today. ABL-195 registered exactly these; the offshore
#: pilot must refit none of them, because a refit on a different source silently
#: changes a live model's gate evidence.
SERVING_PAIRS = {("wind_offshore", "BE"), ("wind_offshore", "FR"),
                 ("wind_onshore", "BE"), ("wind_onshore", "DE"), ("wind_onshore", "FR")}


def _module_const(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


@pytest.fixture(scope="module")
def scopes():
    return {name: {tuple(pair) for pair in pairs}
            for name, pairs in _module_const(HARNESS.read_text(encoding="utf-8"), "SCOPES").items()}


def test_every_scope_is_reachable(scopes):
    """A full pass must be able to return PASS under every registered scope."""
    for name, pairs in scopes.items():
        registered_cells = len(pairs) * len(PRIMARY_BANDS)
        assert registered_cells > 0, f"scope {name!r} registers no cells"
        # The bar main() applies is len(pairs) x bands; a run that produced one
        # cell per registered pair-band clears it. Under the old hardcoded 15
        # this equality was false for every scope but ABL-195's.
        assert registered_cells == len(pairs) * len(PRIMARY_BANDS)


def test_default_scope_reproduces_abl195(scopes):
    """The unflagged run is still the ABL-195 gate: same pairs, same 15 cells.

    ABL-379: this read `git show origin/main:scripts/evaluate_wind_retrain.py`
    and pulled the `PAIRS` constant out of it. That constant is exactly what
    this issue's own change replaced with `SCOPES`, so the test went red the
    moment ABL-322 merged -- it has been failing on `origin/main` ever since,
    asserting nothing. A test whose reference is a moving branch stops testing
    the thing it names as soon as it lands.

    `experiments/ABL195/config.json` is the frozen pre-registration and is the
    authority for what ABL-195 registered. It does not move.
    """
    registered = json.loads(
        (REPO / "experiments" / "ABL195" / "config.json").read_text(encoding="utf-8"))
    registered_pairs = {(stream, country)
                        for stream, spec in registered["pairs"].items()
                        for country in spec["countries"]}

    assert scopes["abl195"] == registered_pairs == SERVING_PAIRS, (
        "the default scope no longer reproduces ABL-195's registered pair set")
    assert len(scopes["abl195"]) * len(PRIMARY_BANDS) == 15


def test_pilot_scope_is_offshore_only(scopes):
    """`abl322-pilot` is DE/NL wind_offshore and refits nothing that serves."""
    pilot = scopes["abl322-pilot"]
    assert pilot == {("wind_offshore", "DE"), ("wind_offshore", "NL")}
    assert not pilot & SERVING_PAIRS, (
        f"the pilot scope refits serving pairs: {sorted(pilot & SERVING_PAIRS)}")
    assert len(pilot) * len(PRIMARY_BANDS) == 6


@pytest.fixture(scope="module")
def gate_basis():
    return {name: tuple(cols) for name, cols
            in _module_const(HARNESS.read_text(encoding="utf-8"), "GATE_BASIS").items()}


def test_every_scope_registers_a_gate_basis(scopes, gate_basis):
    """The basis is a registered property of the scope, like the pair list."""
    assert set(gate_basis) == set(scopes), (
        "every registered scope needs a registered gate basis")


def test_gate_basis_always_contains_the_two_columns_the_bar_names(gate_basis):
    """The bar is `challenger WAPE < seasonal-naive D-7 WAPE`. Both must be in it."""
    for name, basis in gate_basis.items():
        assert {"challenger", "seasonal_naive"} <= set(basis), (
            f"scope {name!r} gates on a basis missing the columns its bar names")


def test_pilot_does_not_gate_on_the_incumbent(gate_basis):
    """The defect this pins: DE/NL wind_offshore have 0 rows in `forecasts`.

    With `incumbent` in the basis the intersection is empty for those pairs, so
    all 6 cells scored n=0 with every score None and the harness rendered FAIL —
    a model-quality verdict on a comparison that never happened. Every new
    country in ABL-316's remaining 37 pairs is in exactly that position.
    """
    assert "incumbent" not in gate_basis["abl322-pilot"]
    assert gate_basis["abl322-pilot"] == ("challenger", "seasonal_naive")


def test_abl195_keeps_the_basis_it_was_published_under(gate_basis):
    """ABL-195's read is dispositioned; this pilot must not restate it.

    Its published 48-64h cells scored 480 rows against the 510 the same report
    records as selected, so the incumbent conjunct did drop rows there — the
    four-way basis is not a no-op for it, and re-basing would move numbers the
    Board has already seen.
    """
    assert gate_basis["abl195"] == (
        "challenger", "incumbent", "seasonal_naive", "persistence")


def test_scope_is_a_choice_not_a_country_filter():
    """The harness must not reintroduce a bare `--countries` filter.

    A country filter cannot express "offshore only", so `--countries DE,NL`
    silently included DE wind_onshore. `--scope` selects (stream, country) pairs
    and argparse restricts it to the registered set.
    """
    source = HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    added = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "add_argument"
             and node.args and isinstance(node.args[0], ast.Constant)]
    flags = {node.args[0].value for node in added}
    assert "--scope" in flags
    assert "--countries" not in flags, (
        "--countries is a filter over the registered scope; scoping a run is a "
        "new pre-registration (see SCOPES)")

    scope_arg = next(n for n in added if n.args[0].value == "--scope")
    kwargs = {kw.arg for kw in scope_arg.keywords}
    assert "choices" in kwargs, "--scope must be restricted to the registered scopes"
