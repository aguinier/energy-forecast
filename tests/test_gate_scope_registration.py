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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.evaluation.wind_retrain import PRIMARY_BANDS

HARNESS = REPO / "scripts" / "evaluate_wind_retrain.py"
SOLAR_HARNESS = REPO / "scripts" / "evaluate_solar_retrain.py"

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

    ABL-378: this used to read `PAIRS` out of `origin/main` and compare against
    it. That made the test self-invalidating — the moment the ABL-322 PR merged,
    `PAIRS` no longer existed on main and the assertion became `PAIRS not found`
    rather than a statement about the scope. It was red on `origin/main` from the
    merge until this fix.

    The reference is now the registered pair set written out here. That is the
    property actually worth pinning: ABL-195 registered these five pairs, and the
    default scope must still be exactly them regardless of what main looks like.
    """
    assert scopes["abl195"] == SERVING_PAIRS, (
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


def test_tranche1a_scope_is_bg_ch_onshore(scopes):
    """ABL-380 registers ABL-316's first tranche: BG and CH wind_onshore, 6 cells.

    Pinned for the same reason `abl322-pilot` is. The pair list is the thing the
    cell bar is derived from, so an edit to it silently moves the denominator a
    gate read is dispositioned against — and this scope is the template the
    remaining 33 pairs will be tranched under.
    """
    tranche = scopes["abl380-tranche1a"]
    assert tranche == {("wind_onshore", "BG"), ("wind_onshore", "CH")}
    assert len(tranche) * len(PRIMARY_BANDS) == 6
    assert not tranche & SERVING_PAIRS, (
        f"tranche 1a refits serving pairs: {sorted(tranche & SERVING_PAIRS)}")


def test_tranche1a_does_not_gate_on_the_incumbent(gate_basis):
    """All 37 remaining ABL-316 pairs have zero rows in `forecasts`.

    BG and CH are the first two to be gated, so this is where the ABL-322 defect
    would have recurred: with `incumbent` in the basis all 6 cells intersect to
    n=0 and the harness renders FAIL on a comparison that never ran.
    """
    assert "incumbent" not in gate_basis["abl380-tranche1a"]
    assert gate_basis["abl380-tranche1a"] == ("challenger", "seasonal_naive")


# --------------------------------------------------------------------------
# ABL-378: the same two properties for the solar harness.
#
# The wind harness was fixed by ABL-322; `evaluate_solar_retrain.py` was not,
# and it is the harness the solar half of ABL-316 must be gated with. On
# `origin/main` it hardcoded `len(gate_cells) == 9 and passed == 9`, had no
# scoping flag at all, and named `incumbent` in both scoring calls. Measured
# against the live replica on 2026-08-13, 28 of the 32 solar countries with
# generation data have zero rows in `forecasts`, so the incumbent is NaN on
# every row and `common_scores` empties the intersection -- 0 cells scored,
# rendered as `FAIL`, and then a crash formatting `None / None` as a skill
# percentage.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solar_scopes():
    return {name: tuple(countries) for name, countries
            in _module_const(SOLAR_HARNESS.read_text(encoding="utf-8"), "SCOPES").items()}


@pytest.fixture(scope="module")
def solar_scope_outputs():
    return {name: dict(outputs) for name, outputs
            in _module_const(SOLAR_HARNESS.read_text(encoding="utf-8"),
                             "SCOPE_OUTPUTS").items()}


@pytest.fixture(scope="module")
def solar_gate_basis():
    return {name: tuple(cols) for name, cols
            in _module_const(SOLAR_HARNESS.read_text(encoding="utf-8"), "GATE_BASIS").items()}


def test_solar_default_scope_reproduces_abl253(solar_scopes):
    """The unflagged run is still the ABL-253 gate: BE/DE/FR, 9 cells."""
    assert solar_scopes["abl253"] == ("BE", "DE", "FR")
    assert len(solar_scopes["abl253"]) * len(PRIMARY_BANDS) == 9


def test_solar_registered_scope_does_not_follow_the_shared_constant(solar_scopes):
    """`abl253` is written out in the harness, and must still equal `COUNTRIES`.

    Pinned as an equality rather than a reference: AT is the one other country
    with a solar incumbent, and adding it to the shared `COUNTRIES` constant
    must not silently re-scope a gate that has already been dispositioned.
    Divergence is a review conversation, not a side effect.
    """
    from src.evaluation.solar_retrain import COUNTRIES
    assert solar_scopes["abl253"] == tuple(COUNTRIES)


def test_solar_every_scope_registers_a_gate_basis(solar_scopes, solar_gate_basis):
    assert set(solar_gate_basis) == set(solar_scopes), (
        "every registered solar scope needs a registered gate basis")


def test_solar_gate_basis_contains_the_two_columns_the_bar_names(solar_gate_basis):
    for name, basis in solar_gate_basis.items():
        assert {"challenger", "seasonal_naive"} <= set(basis), (
            f"solar scope {name!r} gates on a basis missing the columns its bar names")


def test_solar_abl253_keeps_the_basis_it_was_published_under(solar_gate_basis):
    """ABL-253 is dispositioned; porting the scope registry must not restate it."""
    assert solar_gate_basis["abl253"] == (
        "challenger", "incumbent", "seasonal_naive", "persistence")


def test_solar_bar_is_derived_from_the_scope_not_a_literal():
    """`performance_pass` must compare against `registered_cells`, never `9`.

    This is the hardcoded-15 defect in its solar form. It is latent on
    `origin/main` only because no scoping flag exists there yet; adding one
    without this change reproduces the wind failure exactly -- a bar no scoped
    invocation can clear.
    """
    source = SOLAR_HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assign = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "performance_pass")
    literals = [n.value for n in ast.walk(assign.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, int)]
    assert not literals, (
        f"the solar pass bar still compares against literal(s) {literals}; it must "
        "derive from the registered scope table")
    names = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)}
    assert "registered_cells" in names


def test_solar_harness_scope_flag_is_a_choice_not_a_country_filter():
    source = SOLAR_HARNESS.read_text(encoding="utf-8")
    added = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "add_argument"
             and node.args and isinstance(node.args[0], ast.Constant)]
    flags = {node.args[0].value for node in added}
    assert "--scope" in flags
    assert "--countries" not in flags, (
        "--countries is a filter over the registered scope; scoping a run is a "
        "new pre-registration (see SCOPES)")
    scope_arg = next(n for n in added if n.args[0].value == "--scope")
    assert "choices" in {kw.arg for kw in scope_arg.keywords}


#: The scoring entry points a harness may call. ABL-389 moved the duplicated
#: `scored` closure out of both harnesses into `scored_with_comparators`, so the
#: basis now arrives one call further out; without naming it here this test would
#: have found no `common_scores` call in the harness and passed vacuously,
#: which is the failure mode `test_default_scope_reproduces_abl195` was written
#: about. Any new scoring entry point belongs in this set.
SCORING_CALLS = ("common_scores", "scored_with_comparators")


@pytest.mark.parametrize("harness", [HARNESS, SOLAR_HARNESS], ids=["wind", "solar"])
def test_scoring_calls_use_the_registered_basis_not_a_hardcoded_tuple(harness):
    """Every scoring call site must pass the scope's basis, never a literal.

    On `origin/main` the solar harness inlined `("challenger", "incumbent",
    "seasonal_naive", "persistence")` at both call sites, which is what made an
    absent incumbent empty the gate.
    """
    source = harness.read_text(encoding="utf-8")
    calls = [call for call in ast.walk(ast.parse(source))
             if isinstance(call, ast.Call)
             and getattr(call.func, "id", "") in SCORING_CALLS]
    assert calls, (
        f"{harness.name} calls none of {SCORING_CALLS}; this test has stopped "
        "pinning anything -- name the new scoring entry point in SCORING_CALLS")
    for call in calls:
        basis_arg = call.args[1]
        inlined = (isinstance(basis_arg, ast.Tuple)
                   and all(isinstance(e, ast.Constant) for e in basis_arg.elts))
        assert not inlined, (
            f"{getattr(call.func, 'id', '?')} is called with a hardcoded comparator "
            "tuple; it must use the scope's registered GATE_BASIS")


#: The solar countries carrying a model today. ABL-253 registered BE/DE/FR; AT is
#: the fourth country with rows in `forecasts` and is deliberately included here,
#: because the guard a solar tranche needs is "refits nothing that serves", which
#: is a larger set than "reproduces ABL-253". Measured on the live replica
#: (9,432,453,120 bytes) on 2026-08-13: `forecasts` holds solar rows for exactly
#: BE (34,036), FR (32,664), AT (32,592) and DE (32,064), and none for any other
#: country. The wind harness has carried `SERVING_PAIRS` since ABL-322; solar had
#: no equivalent, so nothing stopped a solar scope from silently refitting a live
#: pair on a different source table.
SERVING_SOLAR_COUNTRIES = {"BE", "DE", "FR", "AT"}


def test_solar_tranche1b_scope_is_bg_ch(solar_scopes):
    """ABL-381 registers ABL-316's solar tranche 1b: BG and CH, 6 cells.

    Pinned for the reason `abl380-tranche1a` is on the wind side: the country
    list is what the cell bar is derived from, so an edit to it silently moves
    the denominator a dispositioned gate read was measured against.
    """
    tranche = set(solar_scopes["abl316-t1b"])
    assert tranche == {"BG", "CH"}
    assert len(solar_scopes["abl316-t1b"]) * len(PRIMARY_BANDS) == 6


def test_no_abl316_tranche_refits_a_serving_country(solar_scopes):
    """No scope in the ABL-316 rollout may touch a country with a live solar model.

    A tranche is meant to extend coverage to countries that have no model. If one
    silently includes a serving pair it refits it on a different source table,
    under a different registration, and the resulting artifact is the one a later
    reader finds -- replacing the gate evidence a live model was promoted on with
    a read nobody asked for.

    Scoped to the ABL-316 tranches rather than to every scope. The first version
    of this test asserted the latter and exempted `abl253` by name, which was the
    wrong shape twice over: it read a deliberate re-read of the serving countries
    as a fault, and it could only be kept correct by extending a hardcoded list
    every time someone registered one. ABL-376 then did exactly that -- BE/DE/FR
    with `exclude_impossible_night` on, a controlled A/B against `abl253` -- and
    this test failed on a merge for a scope that is doing nothing wrong. What
    protects `abl253`'s evidence from such a scope is that the two write to
    different registered paths, which is
    `test_no_two_solar_scopes_share_an_output_path` below, not a country list.
    """
    for name, countries in solar_scopes.items():
        if not name.startswith("abl316-"):
            continue
        overlap = set(countries) & SERVING_SOLAR_COUNTRIES
        assert not overlap, (
            f"ABL-316 tranche {name!r} refits serving countries: {sorted(overlap)}")


def test_no_two_solar_scopes_share_an_output_path(solar_scope_outputs):
    """Two scopes may share countries, but never a place to write.

    This is the property the serving-country guard above was really reaching for.
    `abl253` and `abl376` deliberately fit the same three countries; what keeps
    the second from destroying the first's dispositioned evidence is that every
    registered path differs. ABL-387 made these paths part of the registration
    precisely so this is checkable, and a scope added by copy-paste is exactly
    how it would stop being true -- silently, since the run would succeed and
    emit a full report over the top of another scope's.
    """
    for key in ("artifact_dir", "json_out", "report_out"):
        seen = {}
        for scope, outputs in solar_scope_outputs.items():
            path = outputs[key]
            assert path not in seen, (
                f"solar scopes {seen[path]!r} and {scope!r} both write {key} to "
                f"{path!r}; one would overwrite the other's evidence")
            seen[path] = scope


def test_solar_tranche1b_does_not_gate_on_the_incumbent(solar_gate_basis):
    """BG and CH hold zero solar rows in `forecasts`, as do all 37 ABL-316 pairs.

    With `incumbent` in the basis all 6 cells intersect to n=0. Since ABL-378
    that renders UNREADABLE rather than FAIL, so it no longer *misreports* the
    model — but it still yields no gate read at all, which is the same wasted
    tranche. The two-way basis is what makes these cells scorable.
    """
    assert "incumbent" not in solar_gate_basis["abl316-t1b"]
    assert solar_gate_basis["abl316-t1b"] == ("challenger", "seasonal_naive")
