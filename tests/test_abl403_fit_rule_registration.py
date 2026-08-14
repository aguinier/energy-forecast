"""ABL-403: the standing `exclude_impossible_night` registration survives a merge.

The CEO adopted PR #58 with a registration that binds scopes which did not exist
when it was made: **`exclude_impossible_night: False` for every remaining
ABL-316 solar tranche, ES and EE included** -- 2c (ABL-419) and 2d (ABL-421).

ABL-429: `FIT_RULES` is now one of the five tables `check_registration_tables`
enforces (import-time), so an omitted row raises before any fit. The standing
DEFAULT_FIT_RULES fallback still exists for the resolution path described in the
ABL-403 registration comment, but a scope that omits its row now fails at import
rather than silently inheriting the default.

That is not hypothetical. The registration first landed as a trailing block at
the tail of `FIT_RULES`, which is exactly where new tranche rows are appended:
it conflicted with ABL-419 on contact, and either side of that resolution drops
it silently. Nothing failed, because nothing looked. Hence this file.

These tests read the harness *source*, not the imported module -- a comment is
not in the AST and not in any runtime object, so the text is the artifact.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent
HARNESS = REPO / "scripts" / "evaluate_solar_retrain.py"

#: The scopes the registration binds, by prefix. `abl316-t2a` predates it and is
#: covered by its own row; anything added later under this prefix is in scope.
BOUND_SCOPE_PREFIX = "abl316-t2"

#: Load-bearing phrases. Kept short and semantic rather than quoting the whole
#: paragraph, so ordinary rewording does not fail while a deletion does.
REGISTRATION_MARKERS = (
    "ABL-403 registered the value",
    "exclude_impossible_night: False",
    "every\n# remaining ABL-316 solar tranche, ES and EE included",
    "ABL-419",
    "ABL-421",
)


@pytest.fixture(scope="module")
def source() -> str:
    return HARNESS.read_text(encoding="utf-8")


def test_the_registration_is_still_in_the_harness(source: str) -> None:
    """A merge resolution that drops the standing registration fails here."""
    missing = [m for m in REGISTRATION_MARKERS if m not in source]
    assert not missing, (
        "the ABL-403 standing fit-rule registration has lost "
        f"{missing!r} from {HARNESS.name}. `FIT_RULES` is not import-checked, "
        "so this comment is the only record that the value was chosen rather "
        "than inherited from DEFAULT_FIT_RULES. Restore it or re-register."
    )


def test_the_registration_sits_above_the_table_not_inside_it(source: str) -> None:
    """It binds the table, so an appended row must not be able to absorb it.

    Rows land at the tail of `FIT_RULES`. A standing rule parked there reads as
    the docstring of whatever row lands under it, and collides textually with
    every future tranche registration -- which is how it met ABL-419.
    """
    anchor = source.index("ABL-403 registered the value")
    # At line start: `DEFAULT_FIT_RULES = {` ends with this same substring, and
    # it is defined above the registration, so a bare `.index` matches it and
    # the assertion passes for the wrong reason.
    table = source.index("\nFIT_RULES = {")
    assert anchor < table, (
        "the ABL-403 standing registration has moved inside `FIT_RULES`. It "
        "binds the table rather than any one row, and rows are appended at the "
        "tail -- keep it above `FIT_RULES = {` so a new tranche row cannot "
        "collide with it or be read as its subject."
    )


def test_the_registration_is_not_duplicated(source: str) -> None:
    """Taking both sides of a conflict is as wrong as taking neither."""
    n = source.count("ABL-403 registered the value")
    assert n == 1, (
        f"found {n} copies of the ABL-403 standing registration; expected 1. "
        "Two copies can drift apart, and a reader cannot tell which one binds."
    )


def test_every_bound_tranche_registers_the_value_that_was_registered(
    source: str,
) -> None:
    """The rows the standing registration binds must actually carry False.

    Inert for the tranches that exist today and forward-binding for the ones
    that do not: a later scope registered `True` under this prefix is a change
    to an adopted registration, and has to move the registration rather than
    slip past it in a row.
    """
    from scripts.evaluate_solar_retrain import FIT_RULES  # noqa: PLC0415

    bound = {
        scope: rules
        for scope, rules in FIT_RULES.items()
        if scope.startswith(BOUND_SCOPE_PREFIX)
    }
    assert bound, (
        "no scope matches "
        f"{BOUND_SCOPE_PREFIX!r} -- the prefix this guard keys on has been "
        "renamed, so it is silently guarding nothing."
    )

    offenders = {
        scope: rules
        for scope, rules in bound.items()
        if rules.get("exclude_impossible_night") is not False
    }
    assert not offenders, (
        f"{offenders!r} contradict the standing ABL-403 registration "
        "(`exclude_impossible_night: False` for every remaining ABL-316 solar "
        "tranche). On BG the rule alone raises night MAE +61.05 MW (8/8 seeds, "
        "outside a 6.96 MW null) and eats 47% of ABL-405's D-7 margin; on ES it "
        "would delete real CSP generation (ABL-411). Turning it on for a "
        "tranche is a new registration, not a row edit."
    )


def test_the_default_the_registration_relies_on_has_not_moved(source: str) -> None:
    """The comment's argument depends on the default also being False."""
    from scripts.evaluate_solar_retrain import DEFAULT_FIT_RULES  # noqa: PLC0415

    assert DEFAULT_FIT_RULES["exclude_impossible_night"] is False, (
        "DEFAULT_FIT_RULES has flipped. The ABL-403 registration is written "
        "on the premise that an omitted row yields the registered behaviour "
        "silently; if the default is now True, an omitted row is an active "
        "defect and this comment understates it."
    )


def test_the_registration_names_where_its_evidence_lives(source: str) -> None:
    """A registered value with no path to its measurement is an assertion."""
    assert "reports/abl_403_night_rule_interaction.md" in source, (
        "the standing registration no longer cites its evidence pack."
    )


def test_fit_rules_is_inside_the_import_check(source: str) -> None:
    """ABL-429: FIT_RULES must be in check_registration_tables.

    A future cleanup that removes it would silently un-enforce the table --
    scopes could drift out of sync without failing at import.
    """
    call = re.search(
        r"check_registration_tables\((.*?)\)", source, re.DOTALL
    )
    assert call is not None, "check_registration_tables call not found"
    assert "FIT_RULES" in call.group(1), (
        "`FIT_RULES` has been removed from `check_registration_tables`. "
        "ABL-429 added it so a scope missing a fit-rule row fails at import. "
        "Restore it -- and if `DEFAULT_FIT_RULES` makes the abort undesirable, "
        "add a note explaining the new policy rather than silently dropping the guard."
    )


def test_scope_titles_is_inside_the_import_check(source: str) -> None:
    """ABL-429: SCOPE_TITLES must be in check_registration_tables.

    A missing title row silently generates a derived heading, making a scope's
    evidence pack unidentifiable from its H1 alone.
    """
    call = re.search(
        r"check_registration_tables\((.*?)\)", source, re.DOTALL
    )
    assert call is not None, "check_registration_tables call not found"
    assert "SCOPE_TITLES" in call.group(1), (
        "`SCOPE_TITLES` has been removed from `check_registration_tables`. "
        "ABL-429 added it so a scope missing a title row fails at import. Restore it."
    )
