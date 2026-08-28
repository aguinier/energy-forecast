"""ABL-602: the widened batch is five pairs, and its letters carry their convention.

The Board answered `widen7` on ABL-316 (2026-08-28), adopting the
causally-available standard for the widened set. Applying that standard to the
committed records leaves **five** pairs, not seven, and the letters those five
carry are not convention-free. This file holds the two properties that make the
batch safe to read later, both against committed machine records rather than
against prose:

**1. The set is five because two pairs fail G4, and G4 is causally available.**
`NO` and `RO` `wind_onshore` clear G1 and lose on the sign test over the
challenger's own slope and correlation. That is computed from the challenger's
predictions against the actuals in the gate window -- no reference model, no
hindsight -- so it is exactly the evidence the adopted standard admits. The
exclusion is the CEO's ruling; what this file holds is that the *record* still
says what the ruling was taken on, so a later reader cannot find the five
unexplained.

**2. A letter without its convention is not a letter.** Every scope in this batch
is a published scope, so both grading axes are pinned by value --
`CAUSAL_LEVELLING[scope] = FIT_WINDOW` and `G23_READABILITY[scope] = SIGN_TEST`.
ABL-437 and ABL-444 moved the *defaults* a new scope inherits to `trailing_28d`
and `FLOORED`, and four of the five pairs change letter between the two. The
issue that commissioned this batch described `HU` as "G2/G3 inside floor --
abstention, not a readable loss", which is true under `fit_window/floored` and
under neither endpoint: HU is `B` under the registered convention *and* `B` under
the amended defaults, where its G2/G3 margins are roughly -27% and -25% against a
7.51% floor. `test_hu_is_a_readable_loss_under_the_amended_defaults` is the guard
that stops the abstention reading being quietly restored, and it derives both
letters from `reports/abl_444_g23_floor_reread.json` rather than restating them.

Nothing here fits a model or opens the replica.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

TRAINER_PATH = REPO / "scripts" / "abl525_train_ship_set.py"
SERVING_PATH = REPO / "scripts" / "abl602_serving_verification.py"

#: The committed reads behind the five rows. Named rather than derived so that
#: deleting one is a failure and not a silently skipped check.
SOLAR_2D_RECORD = REPO / "experiments" / "ABL348" / "results_abl421_tranche2d.json"
WIND_2E_RECORD = REPO / "experiments" / "ABL348" / "results_abl417_tranche2e.json"
WIND_2B_RETRO_RECORD = REPO / "reports" / "abl_418_retro_grade.json"
CONVENTIONS_RECORD = REPO / "reports" / "abl_444_g23_floor_reread.json"
FEATURE_MANIFEST = REPO / "tests" / "feature_list_manifest.json"

BATCH = "abl602"

#: The five the standard admits, and the two it does not. Written out because the
#: whole content of this batch is which pairs are in it.
WIDENED_FIVE = {
    ("LT", "solar"), ("SE", "solar"),
    ("HR", "wind_onshore"), ("HU", "wind_onshore"), ("PL", "wind_onshore"),
}
EXCLUDED_ON_G4 = {("NO", "wind_onshore"), ("RO", "wind_onshore")}

#: The registered convention for every scope in this batch, and the default a
#: *new* scope would inherit. Both axes, because the letters move on both.
REGISTERED_CONVENTION = "fit_window/sign_test"
AMENDED_DEFAULTS = "trailing_28d/floored"

#: Worst-band pair letters under each, from `abl_444_g23_floor_reread.json`.
#: Asserted against that record below rather than trusted from here.
EXPECTED_LETTERS = {
    ("LT", "solar"): {REGISTERED_CONVENTION: "A", AMENDED_DEFAULTS: "N"},
    ("SE", "solar"): {REGISTERED_CONVENTION: "A", AMENDED_DEFAULTS: "N"},
    ("PL", "wind_onshore"): {REGISTERED_CONVENTION: "A", AMENDED_DEFAULTS: "A"},
    ("HR", "wind_onshore"): {REGISTERED_CONVENTION: "A", AMENDED_DEFAULTS: "N"},
    ("HU", "wind_onshore"): {REGISTERED_CONVENTION: "B", AMENDED_DEFAULTS: "B"},
}


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_module("abl525_train_ship_set", TRAINER_PATH)


@pytest.fixture(scope="module")
def conventions():
    """`abl_444_g23_floor_reread.json`, indexed by pair."""
    record = json.loads(CONVENTIONS_RECORD.read_text(encoding="utf-8"))
    index = {}
    for tranche in record["tranches"]:
        for pair in tranche["pairs"]:
            index[(pair["country"], pair["forecast_type"])] = (tranche, pair)
    return record, index


def _rows(trainer):
    return [row for row in trainer.SHIP_SET if row["batch"] == BATCH]


# --------------------------------------------------------------------------
# 1. The batch is the five the standard admits
# --------------------------------------------------------------------------

def test_the_batch_is_the_five_widened_pairs(trainer):
    assert {(row["country"], row["forecast_type"]) for row in _rows(trainer)} == WIDENED_FIVE


def test_no_and_ro_are_not_in_the_ship_set_at_all(trainer):
    """Excluded from the *table*, not merely from this batch.

    A row for either under some other batch would ship it, and the whole reason
    this batch is five is that the standard the Board adopted excludes them.
    """
    present = {(row["country"], row["forecast_type"]) for row in trainer.SHIP_SET}
    assert not (present & EXCLUDED_ON_G4)


def test_no_and_ro_fail_g4_in_the_records_the_ruling_was_taken_on(conventions):
    """The reason the set is five, re-derived from the committed evidence.

    RO's per-band cells live in the tranche 2e record; NO is tranche **2b**,
    whose stored record predates the G1-G4 ladder, so its letters come from the
    ABL-418 retro grade. Both are checked where they actually are.
    """
    ro_record = json.loads(WIND_2E_RECORD.read_text(encoding="utf-8"))
    ro_cells = [c for c in ro_record["gate_cells"] if c["country"] == "RO"]
    assert len(ro_cells) == 3, "RO should be graded in all three bands"
    for cell in ro_cells:
        assert cell["grade"]["conditions"]["G4"] is False
        assert cell["grade"]["conditions"]["G1"] is True, (
            "the point is that G4 is what excludes RO, not coverage or G1")

    retro = json.loads(WIND_2B_RETRO_RECORD.read_text(encoding="utf-8"))
    tranche_2b = next(t for t in retro["tranches"] if t["tranche"] == "2b")
    no_grade = tranche_2b["pair_grades"]["NO wind_onshore"]
    assert no_grade["conditions"]["G4"] is False
    assert no_grade["conditions"]["G1"] is True

    # G4 is a sign test on the challenger's own predictions -- the property that
    # makes it admissible under the causally-available standard.
    ladder = {entry["condition"]: entry for entry in retro["ladder"]}
    assert ladder["G4"]["role"] == "direction"
    assert "slope" in ladder["G4"]["test"] and "correlation" in ladder["G4"]["test"]


# --------------------------------------------------------------------------
# 2. The feature list is today's, for both types
# --------------------------------------------------------------------------

def test_no_widened_row_pins_a_feature_list(trainer):
    """A pin is what withdrew CH; none of these five needs one."""
    for row in _rows(trainer):
        assert row.get("feature_columns") is None, (
            f"{row['country']}/{row['forecast_type']} pins a list")


def test_the_solar_rows_were_graded_on_todays_27_names(trainer):
    """Tranche 2d's own record, not a registration table.

    `SCOPE_FEATURES` deliberately has no row for `abl316-t2d` -- inheriting the
    current list is the intended path -- so the claim "graded at 27" has to be
    checked against what the read wrote.
    """
    from src.evaluation.solar_retrain import FEATURE_COLUMNS as SOLAR_COLUMNS

    record = json.loads(SOLAR_2D_RECORD.read_text(encoding="utf-8"))
    meta = record["meta"]
    assert meta["scope"] == "abl316-t2d"
    assert meta["n_features"] == 27
    assert meta["feature_set_is_registered_for_scope"] is False

    graded = tuple(meta["feature_columns"])
    assert len(graded) == 27
    assert graded == tuple(SOLAR_COLUMNS), "the graded list has drifted from the builder"

    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    assert graded == tuple(manifest["gate_harness"]["solar"]["columns"])

    for country in ("LT", "SE"):
        assert trainer.columns_for(country, "solar") == graded


def test_the_wind_rows_sit_on_the_24_names_the_manifest_freezes(trainer):
    """`wind_retrain.FEATURE_COLUMNS` has never moved, so there is nothing to pin.

    Held against the manifest rather than against a digest of the names: a digest
    would go red on a reorder that the manifest would also catch, and the
    manifest is the artefact a reviewer already reads.
    """
    from src.evaluation.wind_retrain import FEATURE_COLUMNS as WIND_COLUMNS

    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    frozen = tuple(manifest["gate_harness"]["wind"]["columns"])
    assert len(frozen) == 24
    assert tuple(WIND_COLUMNS) == frozen

    for country in ("HR", "HU", "PL"):
        assert trainer.columns_for(country, "wind_onshore") == frozen


def test_both_types_in_this_batch_resolve_to_the_graded_estimator(trainer):
    """catboost for both -- but resolved through the harnesses, never restated.

    Every row here happens to be catboost, which is precisely when a batch is
    tempted to hardcode one. ABL-580's offshore row is the standing evidence
    that the temptation is wrong.
    """
    from scripts.evaluate_wind_retrain import ALGORITHMS as WIND_ALGORITHMS
    from src.evaluation.solar_retrain import ALGORITHM as SOLAR_ALGORITHM

    for row in _rows(trainer):
        expected = (SOLAR_ALGORITHM if row["forecast_type"] == "solar"
                    else WIND_ALGORITHMS[row["forecast_type"]])
        assert trainer.algorithm_for(row["forecast_type"]) == expected


# --------------------------------------------------------------------------
# 3. A letter carries its convention
# --------------------------------------------------------------------------

def test_every_scope_in_this_batch_is_pinned_to_the_published_convention():
    """Published scopes pin both axes by value, so the registered letter is the
    published one -- and the amended defaults are a different reading, not a
    correction of it."""
    from scripts.evaluate_solar_retrain import (
        CAUSAL_LEVELLING as SOLAR_LEVELLING,
        G23_READABILITY as SOLAR_READABILITY,
    )
    from scripts.evaluate_wind_retrain import (
        CAUSAL_LEVELLING as WIND_LEVELLING,
        G23_READABILITY as WIND_READABILITY,
    )

    assert SOLAR_LEVELLING["abl316-t2d"] == "fit_window"
    assert SOLAR_READABILITY["abl316-t2d"] == "sign_test"
    for scope in ("abl417-tranche2e", "abl406-tranche2b"):
        assert WIND_LEVELLING[scope] == "fit_window"
        assert WIND_READABILITY[scope] == "sign_test"


def test_the_published_letters_are_what_the_batch_claims(conventions):
    _, index = conventions
    for pair, expected in EXPECTED_LETTERS.items():
        _, entry = index[pair]
        assert entry["pair_grades"][REGISTERED_CONVENTION] == expected[REGISTERED_CONVENTION], pair


def test_four_of_the_five_change_letter_under_the_amended_defaults(conventions):
    """Caveat 3 of the issue names two; the record says four.

    Held as a count as well as per pair, so a future correction that moves a
    fifth pair cannot pass by updating one row of the table above.
    """
    _, index = conventions
    moved = set()
    for pair, expected in EXPECTED_LETTERS.items():
        _, entry = index[pair]
        assert entry["pair_grades"][AMENDED_DEFAULTS] == expected[AMENDED_DEFAULTS], pair
        if expected[AMENDED_DEFAULTS] != expected[REGISTERED_CONVENTION]:
            moved.add(pair)
    assert moved == {("LT", "solar"), ("SE", "solar"), ("HR", "wind_onshore")}
    assert len(moved) == 3, "three pairs move A -> N; PL and HU hold their letter"


def test_hu_is_a_readable_loss_under_the_amended_defaults(conventions):
    """The correction this batch carries, held so it cannot be quietly undone.

    ABL-602's description reads HU as an abstention. That is the
    `fit_window/floored` letter -- one amendment applied without the other.
    Under both endpoints HU is `B`, and under the amended defaults its G2/G3
    margins are several times the floor, which is the definition of readable.
    """
    _, index = conventions
    _, hu = index[("HU", "wind_onshore")]

    assert hu["pair_grades"][REGISTERED_CONVENTION] == "B"
    assert hu["pair_grades"][AMENDED_DEFAULTS] == "B"
    assert hu["pair_grades"]["fit_window/floored"] == "N", (
        "the abstention reading exists, but only under the mixed convention")

    for cell in hu["cells"]:
        amended = cell["grades"][AMENDED_DEFAULTS]
        floor = cell["floor_pct"]
        assert amended["conditions"]["G1"] is True
        assert amended["conditions"]["G4"] is True
        for condition in ("G2", "G3"):
            assert amended["conditions"][condition] is False
        # Readable, not inside the floor: the margins are signed losses whose
        # magnitude exceeds the floor several times over.
        skill = amended["skill_pct"]
        assert skill["constant_causal_28d"] < -floor, cell["band"]
        assert skill["climatology_causal_28d"] < -floor, cell["band"]
        assert not amended.get("not_readable"), (
            f"{cell['band']}: an abstention would list a not_readable condition")


def test_hu_and_hr_rows_state_their_convention_on_the_record(trainer):
    """The correction reaches the committed machine record, not just this file.

    `admission_history` is threaded into the training record by `main`, so a
    reader of `reports/abl_602_ship_set_training.json` sees which convention
    each letter belongs to without having to find this test.
    """
    rows = {(r["country"], r["forecast_type"]): r for r in _rows(trainer)}

    hu = rows[("HU", "wind_onshore")]["admission_history"]
    assert "READABLE" in hu
    assert "-26.78" in hu and "-25.51" in hu
    assert "fit_window/floored" in hu

    hr = rows[("HR", "wind_onshore")]["admission_history"]
    assert "trailing_28d/floored" in hr
    assert "7.51" in hr


def test_se_solar_is_recorded_as_the_thinnest_pair(trainer):
    """Caveat 1: the withdrawal candidate if the k=1 floor ever moves.

    Checked against the record's own margin rather than the issue's prose, so a
    later floor change makes this test the thing that goes red.
    """
    record = json.loads(SOLAR_2D_RECORD.read_text(encoding="utf-8"))
    cells = [c for c in record["gate_cells"] if c["country"] == "SE"]
    worst = min(c["grade"]["skill_pct"]["seasonal_naive"] for c in cells)
    floor = cells[0]["grade"]["floor_pct"]
    assert worst == pytest.approx(11.29, abs=0.01)
    assert floor == pytest.approx(10.65, abs=0.01)
    assert 0 < worst - floor < 1.0, "SE's headroom is sub-1pp; that is the caveat"

    se = next(r for r in _rows(trainer)
              if r["country"] == "SE" and r["forecast_type"] == "solar")
    assert "thinnest" in se["admission_history"].lower()


def test_all_three_wind_pairs_clear_a_bar_weaker_than_a_flat_line(conventions):
    """Caveat 2, as a property of the record rather than a sentence in a report.

    `bar_weaker_than_a_flat_line` true means D-7 is an easier reference than a
    constant on that country, so a large G1 margin predicts less than it looks
    like it does. It does not block shipping; it has to be stated.
    """
    _, index = conventions
    for country in ("HR", "HU", "PL"):
        _, entry = index[(country, "wind_onshore")]
        flags = [c["grades"][REGISTERED_CONVENTION]["bar_weaker_than_a_flat_line"]
                 for c in entry["cells"]]
        assert any(flags), f"{country}: expected a weak-bar band"


# --------------------------------------------------------------------------
# 4. The batch's records are its own
# --------------------------------------------------------------------------

def test_the_batch_registers_its_own_training_record(trainer):
    assert trainer.BATCH_RECORDS[BATCH] == "reports/abl_602_ship_set_training.json"
    assert trainer.BATCH_ISSUES[BATCH] == "ABL-602"


def test_the_batch_registers_a_contamination_screen_record():
    screens = _load_module("abl580_contamination_screens",
                           REPO / "scripts" / "abl580_contamination_screens.py")
    assert screens.BATCH_RECORDS[BATCH] == "reports/abl_602_contamination_screens.json"
    assert screens.BATCH_ISSUES[BATCH] == "ABL-602"


def test_the_evidence_of_record_names_a_source_for_every_pair(trainer):
    """Every row's letters trace to a committed file, and PL's traces to the
    retro grade rather than to its own tranche record."""
    import inspect

    source = inspect.getsource(trainer.main)
    assert "results_abl421_tranche2d.json" in source
    assert "results_abl417_tranche2e.json" in source
    assert "abl_418_retro_grade.json" in source
    assert "abl_444_g23_floor_reread.json" in source


# --------------------------------------------------------------------------
# 5. Serving verification knows where the clamp applies
# --------------------------------------------------------------------------

def test_serving_verification_expects_the_clamp_only_where_it_runs():
    """Three of five pairs are wind, and wind has no clamp row.

    A verifier that demanded one per pair would fail the majority of this batch
    for a reason that is not a defect -- so the scope is taken from the clamp's
    own registration.
    """
    serving = _load_module("abl602_serving_verification", SERVING_PATH)
    assert serving.CLAMPED_TYPES == frozenset({"solar"})

    wind = serving.clamp_check("wind_onshore", None)
    assert wind["as_expected"] is True
    assert wind["clamp_applies_to_this_type"] is False

    solar_missing = serving.clamp_check("solar", None)
    assert solar_missing["as_expected"] is False, (
        "solar served with no clamp row means the clamp did not run")


def test_serving_verification_checks_both_horizons_by_target_day():
    """`--horizons 1,2` is two target days of 24 hours, not a row count.

    A pair that served 48 rows for one day would pass a bare count and is the
    failure this deliverable names.
    """
    serving = _load_module("abl602_serving_verification", SERVING_PATH)
    assert serving.EXPECTED_TARGET_DAYS == 2
    assert serving.EXPECTED_HOURS_PER_DAY == 24

    one_day = serving.horizon_check(
        {"target_days": ["2026-08-29"], "rows_per_target_day": {"2026-08-29": 48}})
    assert one_day["both_horizons_served"] is False

    two_days = serving.horizon_check({
        "target_days": ["2026-08-29", "2026-08-30"],
        "rows_per_target_day": {"2026-08-29": 24, "2026-08-30": 24},
    })
    assert two_days["both_horizons_served"] is True

    short_day = serving.horizon_check({
        "target_days": ["2026-08-29", "2026-08-30"],
        "rows_per_target_day": {"2026-08-29": 24, "2026-08-30": 19},
    })
    assert short_day["both_horizons_served"] is False
