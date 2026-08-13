"""The per-country night-generation registration (ABL-425).

`solar_geometry.NIGHT_GENERATION_POSSIBLE` states one physical fact per country:
can this solar fleet produce while the sun is below
`NIGHT_ELEVATION_THRESHOLD_DEG`. Two mechanisms read it — the ABL-337 serving
clamp and ABL-376's `exclude_impossible_night` fit rule — and each applies its
own policy on top.

The property under test here is that the table cannot degrade quietly. It has no
default, an absence aborts, and a country added to the fleet without a
declaration fails this file rather than inheriting "cannot generate at night" —
which for an ES-like country deletes real generation and logs it as a
correction.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.solar_geometry import (
    NIGHT_GENERATION_POSSIBLE,
    SOLAR_REPRESENTATIVE_POINTS,
    UndeclaredNightGenerationError,
    night_generation_possible,
)


def test_es_is_registered_as_able_to_generate_after_dark():
    # ABL-411: Red Eléctrica's own PV + CSP split accounts for 98.55% of the MW
    # the replica books for ES at night, 80.1% of it `solar térmica`, against a
    # 263.5 MW mean night level. This is the entry the whole issue exists for.
    assert night_generation_possible("ES") is True


def test_es_is_the_only_country_declared_night_capable():
    # A change-detector on purpose. Flipping a country to True exempts it from
    # the serving clamp's hard zero, so it takes evidence of ABL-411's kind — an
    # external series that accounts for the overnight MW — not a plausible
    # argument. Size of the night floor is explicitly not the predicate: BG's is
    # 3.7x ES's and BG is False.
    capable = {c for c, v in NIGHT_GENERATION_POSSIBLE.items() if v}
    assert capable == {"ES"}


def test_the_largest_night_floors_are_still_declared_impossible():
    # ABL-396's top of the ranking after ES. Each fails ES's charge/discharge
    # discriminator (+0.084 vs ES's +0.515), so the floor is contamination and
    # the clamp must keep firing for them.
    for country in ("BG", "EE", "SI", "SK", "FR"):
        assert night_generation_possible(country) is False, country


def test_an_undeclared_country_aborts_rather_than_defaulting():
    with pytest.raises(UndeclaredNightGenerationError):
        night_generation_possible("XX")


def test_the_abort_is_not_a_lookup_error():
    # `solar_clamp` catches KeyError around the geometry lookup and degrades to
    # the floor alone. If this abort were a KeyError (or any LookupError), that
    # handler would swallow it and hand back the silent default the registration
    # exists to refuse.
    with pytest.raises(UndeclaredNightGenerationError) as excinfo:
        night_generation_possible("XX")
    assert not isinstance(excinfo.value, LookupError)


def test_the_abort_message_names_the_country_and_the_table():
    with pytest.raises(UndeclaredNightGenerationError) as excinfo:
        night_generation_possible("XX")
    message = str(excinfo.value)
    assert "XX" in message
    assert "NIGHT_GENERATION_POSSIBLE" in message


def test_every_country_with_a_representative_point_is_declared():
    # The clamp reaches the declaration before the geometry, so a country that
    # can be night-masked must be declared. Enforced here rather than at import:
    # `check_registration_tables`' comment in `scripts/evaluate_solar_retrain.py`
    # records what an import-time abort costs every branch already in flight.
    missing = sorted(set(SOLAR_REPRESENTATIVE_POINTS) - set(NIGHT_GENERATION_POSSIBLE))
    assert missing == []


def test_every_supported_country_is_declared():
    missing = [c for c in config.SUPPORTED_COUNTRIES if c not in NIGHT_GENERATION_POSSIBLE]
    assert missing == []


def test_the_table_declares_nothing_it_cannot_place():
    # The reverse direction. An entry with no representative point cannot be
    # acted on by either mechanism, so it is a typo rather than a registration.
    unplaceable = sorted(set(NIGHT_GENERATION_POSSIBLE) - set(SOLAR_REPRESENTATIVE_POINTS))
    assert unplaceable == []


def test_every_declaration_is_a_bool():
    # `None` here would read as "undeclared" at a glance while resolving as
    # falsy — the exact silent default the table refuses to have.
    non_bool = {c: v for c, v in NIGHT_GENERATION_POSSIBLE.items() if not isinstance(v, bool)}
    assert non_bool == {}


def test_no_registered_gate_scope_pairs_the_fit_rule_with_a_night_capable_country():
    # The consistency assertion of issue item 4, checked over the registration
    # rather than only at call time: no scope may turn `exclude_impossible_night`
    # on for a country whose registered fact says its night output is real.
    # Nothing here changes a registered value — ABL-403's stay as they are.
    from scripts.evaluate_solar_retrain import SCOPES, fit_rules_for

    incoherent = [
        (scope, country)
        for scope, countries in SCOPES.items()
        if fit_rules_for(scope)["exclude_impossible_night"]
        for country in countries
        if night_generation_possible(country)
    ]
    assert incoherent == []
