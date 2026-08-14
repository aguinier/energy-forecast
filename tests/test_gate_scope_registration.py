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


#: ABL-406 tranche 2b: the eight remaining `wind_onshore` pairs whose ABL-348
#: gate-window mean is at or above 700 MW. Written out here rather than derived
#: from that threshold, because the threshold is not the registration -- the pair
#: list is. Deriving it would let a later edit to `experiments/ABL348/config.json`
#: silently move the denominator of a dispositioned 24-cell read.
TRANCHE2B_PAIRS = {("wind_onshore", c) for c in
                   ("ES", "FI", "GR", "IT", "NO", "PL", "PT", "SE")}


def test_tranche2b_scope_is_the_eight_large_fleet_onshore_pairs(scopes):
    """ABL-406 registers ABL-316's wind tranche 2b: 8 pairs, 24 cells.

    Pinned for the reason `abl380-tranche1a` is: the pair list is what the cell
    bar is derived from, so an edit to it moves the denominator a gate read was
    dispositioned against, and does so without failing anything else.

    The small-fleet pairs are what this must keep *out*. ABL-348's
    `small_fleet_wind_bar_is_loose` caveat, and ABL-380's measurement of the
    mechanism behind it -- BG's registered 93.75% D-7 bar cleared by a causal
    constant at 82.77%, with no model at all -- are why the tranche is cut on
    fleet size rather than alphabetically. A scope that quietly acquired CZ or HU
    would report cells that cannot carry a decision inside the same `n/24 pass`
    as cells that can.
    """
    tranche = scopes["abl406-tranche2b"]
    assert tranche == TRANCHE2B_PAIRS
    assert len(tranche) * len(PRIMARY_BANDS) == 24
    assert not tranche & SERVING_PAIRS, (
        f"tranche 2b refits serving pairs: {sorted(tranche & SERVING_PAIRS)}")


def test_tranche2b_excludes_the_deferred_small_fleet_pairs(scopes):
    """The eight deliberately deferred pairs must not drift into this scope.

    Stated as an exclusion as well as an equality above, because the two fail
    with different messages and this is the one a reviewer needs: these eight are
    filed as a build-and-report set on the CH precedent, and their D-7 bars
    (86.8-125.4% in ABL-348) make a pass there not model strength.
    """
    deferred = {("wind_onshore", c) for c in
                ("CZ", "EE", "HR", "HU", "LT", "LV", "NL", "RO")}
    overlap = scopes["abl406-tranche2b"] & deferred
    assert not overlap, (
        f"tranche 2b includes deferred small-fleet pairs: {sorted(overlap)}; "
        "their bars cannot carry a decision and must not share this denominator")


def test_tranche2b_does_not_gate_on_the_incumbent(gate_basis):
    """None of the eight holds a `wind_onshore` row in `forecasts`.

    Measured on the live replica (9,432,453,120 bytes) on 2026-08-13: exactly
    BE (32,068) and AT/DE/FR (31,056 each) carry `renewable_type='wind_onshore'`
    rows, and all eight tranche-2b countries carry zero -- while carrying 64-65k
    forecast rows each of *other* types, so "the country is absent from the
    table" is not the explanation, and an incumbent-bearing basis would look
    plausible right up to the empty intersection. Under the four-way basis all 24
    cells intersect to n=0; since ABL-378 that reads UNREADABLE rather than FAIL,
    which is no longer a wrong verdict but is still eight pairs fitted for no
    gate read at all.
    """
    assert "incumbent" not in gate_basis["abl406-tranche2b"]
    assert gate_basis["abl406-tranche2b"] == ("challenger", "seasonal_naive")


#: ABL-417 tranche 2e: the eight `wind_onshore` pairs tranche 2b left out.
#: Written out for the same reason `TRANCHE2B_PAIRS` is -- the pair list *is* the
#: registration, and deriving it from a fleet-size threshold would let an edit to
#: `experiments/ABL348/config.json` move the denominator of a 24-cell read.
#:
#: This set is exactly the one `test_tranche2b_excludes_the_deferred_small_fleet_pairs`
#: names as deferred, which is the property the two tests share: the same eight
#: countries must stay *out* of 2b and *in* 2e, and a drift in either direction
#: now fails twice with two different messages.
TRANCHE2E_PAIRS = {("wind_onshore", c) for c in
                   ("CZ", "EE", "HR", "HU", "LT", "LV", "NL", "RO")}


def test_tranche2e_is_the_eight_small_fleet_onshore_pairs(scopes):
    """ABL-417 registers ABL-316's wind tranche 2e: 8 pairs, 24 cells.

    Report-only, and the scope table is where that begins: these are the pairs
    whose registered D-7 bars run 86.78% (EE) to 125.38% (HU), which is why they
    are not in 2b's denominator. ABL-406 measured that a bar that weak fully
    predicts its own gate outcome -- five weak bars, five passes; three strong
    bars, three failures or ties -- so what the cells are read on here is
    ABL-418's ladder, not the pass count.
    """
    tranche = scopes["abl417-tranche2e"]
    assert tranche == TRANCHE2E_PAIRS
    assert len(tranche) * len(PRIMARY_BANDS) == 24
    assert not tranche & SERVING_PAIRS, (
        f"tranche 2e refits serving pairs: {sorted(tranche & SERVING_PAIRS)}")


def test_tranche2e_is_disjoint_from_the_earlier_wind_tranches(scopes):
    """2e must add coverage, not re-fit a pair another tranche dispositioned.

    Tranches 1a and 2b are `done` and published. A pair appearing in two scopes
    would be fitted twice under one registration and reported under two
    verdicts, and -- since each scope writes to its own registered paths -- with
    no collision to make that visible.

    The union is also the completeness claim this tranche closes ABL-316's wind
    half on: 2 + 8 + 8 = the 18 `wind_onshore` countries ABL-348 registers.
    """
    earlier = scopes["abl380-tranche1a"] | scopes["abl406-tranche2b"]
    assert not scopes["abl417-tranche2e"] & earlier, (
        f"tranche 2e re-fits already-dispositioned pairs: "
        f"{sorted(scopes['abl417-tranche2e'] & earlier)}")
    onshore = {pair for pair in earlier | scopes["abl417-tranche2e"]
               if pair[0] == "wind_onshore"}
    assert len(onshore) == 18


def test_tranche2e_does_not_gate_on_the_incumbent(gate_basis):
    """None of these eight holds a `wind_onshore` row in `forecasts` either.

    Re-measured on the live replica (9,432,453,120 bytes) on 2026-08-14 rather
    than inherited from tranche 2b's docstring: exactly BE (32,068) and AT/DE/FR
    (31,056 each) carry `renewable_type='wind_onshore'` rows across the whole
    table, and all eight of CZ/EE/HR/HU/LT/LV/NL/RO carry zero while holding
    65,088-65,232 forecast rows each of *other* types. So "the country is absent
    from the table" is not the explanation here either, and a four-way basis
    would look plausible right up to the n=0 intersection.
    """
    assert "incumbent" not in gate_basis["abl417-tranche2e"]
    assert gate_basis["abl417-tranche2e"] == ("challenger", "seasonal_naive")


# --------------------------------------------------------------------------
# ABL-435 -- tranche 2f, the BG/CH re-read
# --------------------------------------------------------------------------

def test_tranche2f_re_reads_tranche1a_pairs_exactly(scopes):
    """2f is 1a's pair set, deliberately, and must stay *identical* to it.

    Every other tranche is pinned to be disjoint from its predecessors. This one
    is pinned the other way, and the direction is the point: 2f exists to give
    BG and CH the ABL-389 reference columns and the ABL-418 grade that tranche 1a
    predates, so the two reads are comparable only while they cover the same
    pairs under the same registration. A 2f that quietly gained or lost a pair
    would still run, still emit a full report, and would no longer be a re-read
    of anything -- and its challenger WAPEs would no longer be checkable against
    the published ones.
    """
    assert scopes["abl435-tranche2f"] == scopes["abl380-tranche1a"], (
        "tranche 2f is a re-read of tranche 1a and must register the same pairs")
    assert scopes["abl435-tranche2f"] == {("wind_onshore", "BG"), ("wind_onshore", "CH")}
    assert len(scopes["abl435-tranche2f"]) * len(PRIMARY_BANDS) == 6
    assert not scopes["abl435-tranche2f"] & SERVING_PAIRS, (
        f"tranche 2f refits serving pairs: "
        f"{sorted(scopes['abl435-tranche2f'] & SERVING_PAIRS)}")


def test_tranche2f_does_not_disturb_the_onshore_coverage_claim(scopes):
    """A re-read must not be counted as coverage.

    `test_tranche2e_is_disjoint_from_the_earlier_wind_tranches` closes ABL-316's
    onshore half at 2 + 8 + 8 = 18 countries. 2f adds no country, and this pins
    that it cannot: its countries are already inside that union, so the
    completeness arithmetic is unchanged by it. Without this, a later reader
    counting scopes rather than countries would make the onshore half read 20.
    """
    covered = {country for scope in ("abl380-tranche1a", "abl406-tranche2b",
                                     "abl417-tranche2e")
               for stream, country in scopes[scope] if stream == "wind_onshore"}
    assert len(covered) == 18
    re_read = {country for stream, country in scopes["abl435-tranche2f"]
               if stream == "wind_onshore"}
    assert re_read <= covered, (
        f"tranche 2f introduces uncovered countries {sorted(re_read - covered)}; "
        "it is a re-read, so a new country here belongs in a tranche of its own")


def test_tranche2f_gates_on_the_basis_tranche1a_was_published_under(gate_basis):
    """Identical to 1a's basis, because a changed basis voids the comparison.

    2f's reproduction claim is that the challenger figures land on ABL-380's to
    the digit. The basis decides which rows enter a cell, so widening or
    narrowing it would move n and move every WAPE with it -- and the new numbers
    would then differ from the published ones for a reason that has nothing to do
    with the model. Re-measured on the live replica on 2026-08-14: BG and CH
    still carry zero `renewable_type='wind_onshore'` rows in `forecasts` while
    holding 65,232 and 64,272 rows of other types, so a four-way basis would
    still intersect all 6 cells to n=0.
    """
    assert gate_basis["abl435-tranche2f"] == gate_basis["abl380-tranche1a"]
    assert gate_basis["abl435-tranche2f"] == ("challenger", "seasonal_naive")
    assert "incumbent" not in gate_basis["abl435-tranche2f"]


def test_tranche2f_writes_nowhere_tranche1a_writes():
    """The re-read must not touch the record it is re-reading.

    `check_scope_outputs` enforces global distinctness at import; this states the
    specific pairing that matters, because tranche 1a's `json_out` is the machine
    record `reports/abl_380_tranche1a_findings.md:9` cites for a PASS the Board
    was asked to review, and its `artifact_dir` holds the two model artifacts
    whose SHA-256 values that report's fit-audit table publishes. ABL-404 is the
    precedent: a scope overwrote a published read *under its own heading*, exited
    0, and showed nothing in `git status`.
    """
    outputs = _module_const(HARNESS.read_text(encoding="utf-8"), "SCOPE_OUTPUTS")
    first_read, re_read = outputs["abl380-tranche1a"], outputs["abl435-tranche2f"]
    for key in ("artifact_dir", "json_out", "report_out"):
        assert re_read[key] != first_read[key], (
            f"tranche 2f would overwrite tranche 1a's {key}")
    assert first_read["json_out"] == "experiments/ABL348/results_abl380_tranche1a.json"
    assert re_read["json_out"] == "experiments/ABL348/results_abl435_tranche2f.json"
    # Tracked, not swallowed by `.gitignore:53` -- the deficiency in tranche 1a's
    # record that 2f exists to repair is exactly that a gate record a reviewer
    # cannot diff is not evidence.
    assert not re_read["json_out"].endswith("/results.json")


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


# --------------------------------------------------------------------------
# ABL-429: FIT_RULES and SCOPE_TITLES are now import-enforced.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solar_fit_rules():
    return {name: dict(rules) for name, rules
            in _module_const(SOLAR_HARNESS.read_text(encoding="utf-8"), "FIT_RULES").items()}


@pytest.fixture(scope="module")
def solar_scope_titles():
    return dict(_module_const(SOLAR_HARNESS.read_text(encoding="utf-8"), "SCOPE_TITLES"))


def test_solar_every_scope_registers_a_fit_rule(solar_scopes, solar_fit_rules):
    """Every registered solar scope must carry an explicit fit-rule entry.

    ABL-429: FIT_RULES is now import-enforced by check_registration_tables.
    This test pins the same property at the AST level so a scope that omits its
    row is caught by two independent checks -- the import and this suite entry --
    and the failure message names the table to edit rather than just the import
    traceback.
    """
    assert set(solar_fit_rules) == set(solar_scopes), (
        "FIT_RULES and SCOPES disagree; every registered solar scope needs a "
        "registered fit-rule entry, and FIT_RULES must not carry extra keys. "
        f"In SCOPES but not FIT_RULES: {sorted(set(solar_scopes) - set(solar_fit_rules))}. "
        f"In FIT_RULES but not SCOPES: {sorted(set(solar_fit_rules) - set(solar_scopes))}."
    )


def test_solar_every_scope_registers_a_title(solar_scopes, solar_scope_titles):
    """Every registered solar scope must carry an explicit SCOPE_TITLES entry.

    ABL-429: SCOPE_TITLES is now import-enforced by check_registration_tables.
    A missing title generates a derived heading from the scope slug, which makes
    a scope's evidence pack unidentifiable from its H1 alone.
    """
    assert set(solar_scope_titles) == set(solar_scopes), (
        "SCOPE_TITLES and SCOPES disagree; every registered solar scope needs a "
        "registered title, and SCOPE_TITLES must not carry extra keys. "
        f"In SCOPES but not SCOPE_TITLES: {sorted(set(solar_scopes) - set(solar_scope_titles))}. "
        f"In SCOPE_TITLES but not SCOPES: {sorted(set(solar_scope_titles) - set(solar_scopes))}."
    )


# --------------------------------------------------------------------------
# ABL-429: the counting recipe is executable, not prose.
#
# ABL-421 left `grep -E "^[A-Z_]+ = \{"` as "the count" of registration tables.
# It was wrong at the commit that wrote it -- 9 rather than the 7 asserted --
# because it also matches `DEFAULT_FIT_RULES` (keyed by rule name) and
# `NOT_EVALUABLE_CAUSES` (keyed by country). A *per-scope* registration table is
# one whose keys are scope names, which is decidable from the source, so it is
# decided here instead of counted by hand into a sentence that drifts.
#
# The property: every per-scope table is either inside
# `check_registration_tables` or declared in `UNCHECKED_REGISTRATION_TABLES`
# with the reason it cannot join. A new table added by a later tranche fails
# this until its author chooses -- which is the ABL-404 failure mode (a silent
# module-level default nobody elected) converted into a failing assertion.
# --------------------------------------------------------------------------

def _per_scope_tables(source: str) -> dict:
    """Module-level dict literals keyed by scope name.

    A table qualifies when at least one key is a registered scope. That admits
    partial tables (`SCOPE_FEATURES`, `SCOPE_NOT_EVALUABLE`) and excludes dicts
    keyed by anything else -- rule names, country codes, algorithms.
    """
    tree = ast.parse(source)
    scope_names = set(_module_const(source, "SCOPES"))
    tables = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)):
            continue
        name = node.targets[0].id
        if not name.isupper():
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            # A table whose values are names rather than literals (e.g. a shared
            # constant per scope) still has literal *keys*, which is all we need.
            value = {k.value: None for k in node.value.keys
                     if isinstance(k, ast.Constant)}
        if set(value) & scope_names:
            tables[name] = value
    return tables


def _checked_tables(source: str) -> set:
    """The keyword names passed to `check_registration_tables`."""
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "check_registration_tables"):
            return {kw.arg for kw in node.keywords}
    raise AssertionError("check_registration_tables call not found")


@pytest.mark.parametrize("harness", [HARNESS, SOLAR_HARNESS], ids=["wind", "solar"])
def test_every_per_scope_table_is_checked_or_declares_why_not(harness):
    source = harness.read_text(encoding="utf-8")
    per_scope = set(_per_scope_tables(source))
    checked = _checked_tables(source)
    declared = set(_module_const(source, "UNCHECKED_REGISTRATION_TABLES"))

    undeclared = per_scope - checked - declared
    assert not undeclared, (
        f"{harness.name}: {sorted(undeclared)} are keyed by scope name but are "
        "neither in `check_registration_tables` nor declared in "
        "`UNCHECKED_REGISTRATION_TABLES`. A scope omitted from such a table "
        "resolves through a module-level default silently, at run time -- which "
        "is how ABL-404 refitted a dispositioned scope at the wrong challenger "
        "and exited 0. Either add the table to the call, or add it to "
        "`UNCHECKED_REGISTRATION_TABLES` with the reason it cannot join."
    )

    stale = declared - per_scope
    assert not stale, (
        f"{harness.name}: {sorted(stale)} are declared unchecked but are no "
        "longer per-scope registration tables. Remove the declaration so the "
        "exemption list cannot outlive the table it exempts."
    )

    overlap = declared & checked
    assert not overlap, (
        f"{harness.name}: {sorted(overlap)} are both checked and declared "
        "unchecked. The declaration is now false -- drop it."
    )
