"""ABL-580: the ship set grew, so its trainer has to say which read authorises which fit.

The Board approved a *rule* alongside the `ship8` roster (ABL-316 ledger 14.6,
restated in 15.1): a held pair that later satisfies the same rule joins the
shipping set without a new Board card, and a shipping pair a later correction
moves outside it is withdrawn. So `scripts/abl525_train_ship_set.py` is not a
one-shot script for eight rows; it is the trainer for a table that grows and
shrinks by CEO disposition. This file holds the three properties that made
growing it safe.

**1. The algorithm is a property of the forecast type, and the type's own gate
harness is the authority.** ABL-525's eight rows were seven `wind_onshore` plus
one `solar`, both catboost, so the script carried a single
`ALGORITHM = "catboost"` module constant and nothing distinguished "the estimator
this pair was graded with" from "the estimator this file happens to name".
ABL-580 adds NL `wind_offshore`, and `evaluate_wind_retrain.ALGORITHMS` is
`{"wind_offshore": "xgboost", "wind_onshore": "catboost"}` -- the pilot fitted
offshore with xgboost, and its committed record says so. Reusing the constant
would have shipped an offshore model no gate read, which is the failure ABL-525
item 2 exists to prevent, arriving through an estimator rather than through a
feature list. `test_the_offshore_row_resolves_to_the_estimator_its_read_recorded`
is that check, taken from the evidence rather than from the table it is checking.

**2. The change is a no-op for the seven already fitted.** A trainer edit that
silently re-specifies what an artifact on disk was fitted with is worse than the
defect it fixes, because `abl525_repro_check.py` compares *predictions* across a
refit and would report a drift that is really an edit. Held directly rather than
argued in a comment.

**3. A batch cannot land on another batch's record.** `SCOPE_OUTPUTS` exists
because ABL-387 watched a scoped run overwrite another scope's published
evidence when the output path was a flag default. `BATCH_RECORDS` is the same
shape of protection one layer down, and `--json-out` defaulting to `None` is the
part that makes an unbatched run refuse rather than guess.

**4. A pair appears once, and its row states the read that authorises it as it
stands now.** ABL-583 readmitted CH `solar` after the CEO withdrew it two hours
earlier, which is the first time a row changed batch rather than being added.
The temptation is a second row -- keep the withdrawn one for the history, add a
shipping one -- and that is exactly the defect this file exists to prevent:
`columns_for` matches on `(country, forecast_type)` and takes the first hit, so
two rows for one pair would silently decide the artifact's feature list by
source order. The history goes in `admission_history` on the single row, and
`test_a_pair_appears_once` is the guard.

Nothing here fits a model or opens the replica. Every assertion is against the
registration tables in the script and against committed machine records, so this
file runs in milliseconds and stays green on a box with no database.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

TRAINER_PATH = REPO / "scripts" / "abl525_train_ship_set.py"

#: The committed read behind each ABL-580 row. Named here rather than derived so
#: that deleting a record is a test failure and not a silently skipped check.
SOLAR_2A_RECORD = REPO / "experiments" / "ABL348" / "results_abl426_tranche2a_generation.json"
OFFSHORE_PILOT_RECORD = REPO / "experiments" / "ABL322" / "results_abl436_offshore_reread.json"

#: ABL-581's read: the one that readmitted CH `solar` to the ship set.
CH_SOLAR_F27_RECORD = REPO / "experiments" / "ABL348" / "results_abl581_ch_solar_f27.json"

#: What the ABL-525 batch was fitted with, before this file's subject existed.
#: Every row of that batch must still resolve to it.
ABL525_ALGORITHM = "catboost"


def _load_module(name, path):
    """Import a `scripts/` module by path.

    `scripts/` is not a package on `sys.path` by name here, and these modules
    insert the repo root themselves at import time, so a spec load is the shape
    that works from `tests/` without a conftest change.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_trainer():
    return _load_module("abl525_train_ship_set", TRAINER_PATH)


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


def _rows(trainer, batch):
    return [row for row in trainer.SHIP_SET if row["batch"] == batch]


# --------------------------------------------------------------------------
# 1. The estimator comes from the harness that graded the pair
# --------------------------------------------------------------------------

def test_the_offshore_row_resolves_to_the_estimator_its_read_recorded(trainer):
    """NL `wind_offshore` is fitted with what the pilot fitted it with.

    Taken from `experiments/ABL322/results_abl436_offshore_reread.json` -- the
    record the fit itself wrote -- rather than from `ALGORITHMS`, so this cannot
    pass by both sides drifting together.
    """
    record = json.loads(OFFSHORE_PILOT_RECORD.read_text(encoding="utf-8"))
    graded = {
        entry["country"]: entry["algorithm"]
        for entry in record["training"]
        if entry["forecast_type"] == "wind_offshore"
    }
    assert graded["NL"] == "xgboost", "the pilot record no longer says what it said"
    assert trainer.algorithm_for("wind_offshore") == graded["NL"]


def test_the_offshore_row_is_fitted_with_the_pilots_hyperparameters(trainer):
    """And with the same params, which is the half a type lookup cannot cover.

    `build_model` pops `early_stopping_rounds` because this fit has no validation
    set; the pilot's `_model` does the same. If either stops doing it the two
    dicts diverge and this fails.
    """
    record = json.loads(OFFSHORE_PILOT_RECORD.read_text(encoding="utf-8"))
    graded = next(
        entry["params"] for entry in record["training"]
        if entry["forecast_type"] == "wind_offshore" and entry["country"] == "NL"
    )
    _, params = trainer.build_model(trainer.algorithm_for("wind_offshore"))
    assert params == graded


def test_the_algorithm_table_is_the_harnesses_own(trainer):
    """`ALGORITHM_BY_TYPE` is imported, not restated.

    The property that matters is that a later move in either harness arrives here
    without an edit to the trainer, so it is asserted as identity of values rather
    than as a literal.
    """
    from scripts.evaluate_wind_retrain import ALGORITHMS as WIND_ALGORITHMS
    from src.evaluation.solar_retrain import ALGORITHM as SOLAR_ALGORITHM

    for forecast_type, algorithm in WIND_ALGORITHMS.items():
        assert trainer.ALGORITHM_BY_TYPE[forecast_type] == algorithm
    assert trainer.ALGORITHM_BY_TYPE["solar"] == SOLAR_ALGORITHM


def test_every_ship_set_row_has_an_algorithm(trainer):
    for row in trainer.SHIP_SET:
        assert trainer.algorithm_for(row["forecast_type"])


# --------------------------------------------------------------------------
# 2. The change is a no-op for the batch already on disk
# --------------------------------------------------------------------------

def test_the_abl525_batch_still_resolves_to_the_estimator_it_was_fitted_with(trainer):
    """Replacing the module constant must not re-specify seven live artifacts.

    `abl525_repro_check.py` refits through `fit_one` and compares predictions at
    1e-12. If this ever failed, that check would report a difference which is an
    edit to this file and not a drift in the artifacts -- the most expensive kind
    of false alarm the ship set can raise.
    """
    for row in _rows(trainer, "abl525"):
        assert trainer.algorithm_for(row["forecast_type"]) == ABL525_ALGORITHM


def test_fit_one_defaults_to_the_types_estimator_not_to_a_constant(trainer):
    """`abl525_repro_check.py` calls `fit_one` positionally with no algorithm.

    So the default has to resolve per type. Asserted on the signature rather than
    by fitting, which would need the replica.
    """
    import inspect

    default = inspect.signature(trainer.fit_one).parameters["algorithm"].default
    assert default is None, "a non-None default here refits every pair as one type"


# --------------------------------------------------------------------------
# 3. Feature lists: a row pins one only when its read was taken on a list the
#    builder has since moved off
# --------------------------------------------------------------------------

@pytest.mark.parametrize("country", ["CZ", "RO"])
def test_the_solar_ship_rows_take_the_list_their_read_recorded(trainer, country):
    """CZ and RO solar are the inverse of CH: the graded list *is* the current one.

    `abl316-t2a-generation` is deliberately absent from
    `evaluate_solar_retrain.SCOPE_FEATURES`, so there is no pin to check against
    and the authority is `meta.feature_columns` in the record the fit wrote. This
    asserts the equality the ABL-580 description asks to be verified rather than
    assumed -- and it fails, rather than silently re-basing the fit, if
    `solar_retrain.FEATURE_COLUMNS` moves to 28.
    """
    record = json.loads(SOLAR_2A_RECORD.read_text(encoding="utf-8"))
    graded = tuple(record["meta"]["feature_columns"])
    assert len(graded) == record["meta"]["n_features"] == 27
    assert trainer.columns_for(country, "solar") == graded


def test_the_offshore_row_takes_the_onshore_list(trainer):
    """One wind list, no per-type branch -- the ABL-580 item 2 wind check."""
    from src.evaluation.wind_retrain import FEATURE_COLUMNS as WIND_FEATURE_COLUMNS

    assert trainer.columns_for("NL", "wind_offshore") == WIND_FEATURE_COLUMNS
    assert trainer.columns_for("CZ", "wind_onshore") == WIND_FEATURE_COLUMNS


def test_only_a_withdrawn_row_pins_its_own_feature_list(trainer):
    """A pin is evidence of a hold, so the two must not come apart.

    A shipping row that pins a list is a per-country serving fork, which ABL-525
    item 2 forbids in terms; a held row that does not pin one would fit at the
    current list if `--include-held` were ever exercised, which is the CH failure
    the pin exists to make impossible.

    Since ABL-583 readmitted CH, no row pins a list, so this passes vacuously
    today. It is kept, and the vacuity is named here rather than left to be
    discovered, because the ship set shrinks as well as grows and the next
    withdrawal has to land on a rule that is already in place.
    `test_the_only_hold_is_a_disposition_and_no_row_pins_a_list` is the other
    half -- and ABL-602's HU is that next withdrawal, which is why it now names
    a subject instead of asserting an empty set.
    """
    for row in trainer.SHIP_SET:
        if row.get("feature_columns"):
            assert row["hold"], f"{row['country']}/{row['forecast_type']} pins a list but ships"


def test_the_only_hold_is_a_disposition_and_no_row_pins_a_list(trainer):
    """The converse, restated once a row was actually held.

    This was `test_no_row_pins_a_list_while_none_is_held` and asserted the
    vacuity of the test above: no row held, no row pinning. ABL-602 withdrew
    `HU` `wind_onshore` on 2026-08-28, so half of that is no longer true, and
    the honest invariant is the one that survives a withdrawal:

    * **no row pins a feature list** -- unchanged, and the ABL-525 item 2 guard
      against a per-country serving fork;
    * **every hold is a disposition hold, not a feature-list hold.** The two
      are different hazards. CH's hold was a *pin* against a builder that had
      moved, so `--include-held` would have refitted it at a list nobody
      graded. HU's hold is a decision about a pair whose feature list is
      exactly the one it was graded on, so `--include-held` refits it
      faithfully; what must not happen to HU is a *deploy*, not a refit.
    """
    held = [row for row in trainer.SHIP_SET if row["hold"]]
    assert [(row["country"], row["forecast_type"]) for row in held] == [
        ("HU", "wind_onshore")
    ]
    assert not [row for row in trainer.SHIP_SET if row.get("feature_columns")]
    assert not [row for row in held if row.get("feature_columns")], (
        "HU's hold is a disposition, not a pin -- a pin here would mean the "
        "list it was graded on had moved, which is a different finding")


def test_ch_solar_rejoined_at_the_current_27_name_list(trainer):
    """ABL-583: the readmission, held as a property rather than as a comment.

    CH was withdrawn because its read was taken at the legacy 25 while the
    builder had moved to 27. ABL-581 re-read it at 27 under a new scope, so what
    has to be true now is the *inverse* of the withdrawal condition: the list
    this artifact is fitted at is element-for-element the list ABL-581's read
    recorded, and that list is not the legacy 25.

    Taken from `meta.feature_columns` in the record the read wrote, not from
    `SCOPE_FEATURES` -- `abl581-ch-solar-f27` is deliberately absent from that
    table and resolves through `DEFAULT_SCOPE_FEATURES`, which is why
    `feature_set_is_registered_for_scope` is False and correct.
    """
    from scripts.evaluate_solar_retrain import LEGACY_FEATURE_COLUMNS

    record = json.loads(CH_SOLAR_F27_RECORD.read_text(encoding="utf-8"))
    meta = record["meta"]
    assert meta["scope"] == "abl581-ch-solar-f27"
    assert meta["registered_source"] == "energy_generation"
    assert meta["feature_set_is_registered_for_scope"] is False

    graded = tuple(meta["feature_columns"])
    assert len(graded) == meta["n_features"] == 27
    assert trainer.columns_for("CH", "solar") == graded
    assert trainer.columns_for("CH", "solar") != tuple(LEGACY_FEATURE_COLUMNS)
    assert len(LEGACY_FEATURE_COLUMNS) == 25


def test_ch_solar_ships_and_carries_both_of_its_dispositions(trainer):
    """One row, no hold, and the withdrawal still on the record.

    A readmitted pair that reads as one which never left would hide the reason
    the pin machinery exists. The history is a field on the row so it reaches
    the committed machine record, not a comment that stops at this file.
    """
    ch = next(row for row in trainer.SHIP_SET
              if row["country"] == "CH" and row["forecast_type"] == "solar")
    assert ch["hold"] is None, "CH solar was readmitted on ABL-581's read"
    assert ch.get("feature_columns") is None, "a shipping row must not pin a list"
    assert ch["batch"] == "abl583"
    assert "WITHDRAWN" in ch["admission_history"]
    assert "READMITTED" in ch["admission_history"]
    assert "abl581-ch-solar-f27" in ch["admission_history"]


# --------------------------------------------------------------------------
# 4. A batch cannot land on another batch's record
# --------------------------------------------------------------------------

def test_every_batch_has_a_registered_record(trainer):
    assert set(trainer.BATCHES) == set(trainer.BATCH_RECORDS)


def test_no_two_batches_share_a_record(trainer):
    paths = list(trainer.BATCH_RECORDS.values())
    assert len(paths) == len(set(paths))


def test_no_batch_record_is_swallowed_by_gitignore(trainer):
    """ABL-440: `experiments/*/results.json` is ignored and is still open.

    A machine record that cannot be diffed cannot be evidence, and the trap is
    specifically that the path *looks* like every other experiment output.
    """
    for batch, path in trainer.BATCH_RECORDS.items():
        assert path.startswith("reports/"), f"{batch} writes outside reports/"
        assert not path.endswith("/results.json"), f"{batch} hits the ABL-440 glob"


def test_the_batches_partition_the_ship_set(trainer):
    """Every row belongs to exactly one batch, so `--batch` cannot drop a pair."""
    covered = sum(len(_rows(trainer, batch)) for batch in trainer.BATCHES)
    assert covered == len(trainer.SHIP_SET)


def test_a_pair_appears_once(trainer):
    keys = [(row["country"], row["forecast_type"]) for row in trainer.SHIP_SET]
    assert len(keys) == len(set(keys))


def test_the_abl580_batch_is_the_three_pairs_the_rule_admitted(trainer):
    assert {(row["country"], row["forecast_type"]) for row in _rows(trainer, "abl580")} == {
        ("CZ", "solar"), ("RO", "solar"), ("NL", "wind_offshore"),
    }


def test_the_abl583_batch_is_the_one_pair_the_rule_readmitted(trainer):
    assert {(row["country"], row["forecast_type"]) for row in _rows(trainer, "abl583")} == {
        ("CH", "solar"),
    }


def test_the_abl525_batch_is_the_seven_still_on_it(trainer):
    """CH left this batch when it was readmitted, so the seven are wind_onshore.

    Stated as a property because `test_the_abl525_batch_still_resolves_to_the
    _estimator_it_was_fitted_with` would keep passing if a row went missing --
    an empty loop asserts nothing.
    """
    rows = _rows(trainer, "abl525")
    assert len(rows) == 7
    assert {row["forecast_type"] for row in rows} == {"wind_onshore"}


def test_the_screens_script_registers_a_record_for_every_batch_it_screens(trainer):
    """The trainer and the screens agree on where a batch's evidence lands.

    The screens script covers the renewable-specific batches only (ABL-525's
    seven answered their screens on ABL-525 itself), so this asserts inclusion
    rather than equality -- and that no screen record collides with another.
    """
    screens = _load_module("abl580_contamination_screens",
                           REPO / "scripts" / "abl580_contamination_screens.py")
    assert set(screens.BATCH_RECORDS) <= set(trainer.BATCHES)
    assert set(screens.BATCH_RECORDS) == set(screens.BATCH_ISSUES)
    paths = list(screens.BATCH_RECORDS.values())
    assert len(paths) == len(set(paths))
    for batch, path in screens.BATCH_RECORDS.items():
        assert path.startswith("reports/"), f"{batch} writes outside reports/"
        assert path != trainer.BATCH_RECORDS[batch], (
            f"{batch}: the screens would overwrite the training record")


def test_every_batch_issue_is_registered(trainer):
    assert set(trainer.BATCH_ISSUES) == set(trainer.BATCHES)


def test_the_ship_set_reads_one_source_table(trainer):
    """ABL-321/ABL-348 register `energy_generation` for every pair in this set."""
    assert trainer.RENEWABLE_SOURCE == "energy_generation"


# --------------------------------------------------------------------------
# 5. The instrument the readmission premise is checked with
# --------------------------------------------------------------------------
#
# ABL-583 section 1 first checked the feature-list constant chain by hashing
# `ast.dump` of each constant's value node. Two of the four names are derived
# expressions -- `DEFAULT_SCOPE_FEATURES = FEATURE_COLUMNS` is a bare `Name`, and
# `FEATURE_COLUMNS` ends `*SOLAR_GEOMETRY_FEATURES` -- so that hash does not move
# when the list they resolve to moves. That is not hypothetical: it is ABL-395's
# 25 -> 27, the move that withdrew CH `solar` in the first place. These hold the
# replacement instrument, and the demonstration that the replacement was needed.

SCOPE_CHECK_PATH = REPO / "scripts" / "abl583_scope_value_check.py"


@pytest.fixture(scope="module")
def scope_check():
    return _load_module("abl583_scope_value_check", SCOPE_CHECK_PATH)


def test_the_constant_chain_is_resolved_in_dependency_order(scope_check):
    """A derived expression must be evaluated after what it derives from.

    `LEGACY_FEATURE_COLUMNS` and `DEFAULT_SCOPE_FEATURES` both reference
    `FEATURE_COLUMNS`, which references `SOLAR_GEOMETRY_FEATURES`. Resolve them
    out of order and the `eval` raises `NameError`. The order is the contract, so
    it is asserted rather than assumed.
    """
    names = [name for _, name in scope_check.CONSTANT_CHAIN]
    assert names.index("SOLAR_GEOMETRY_FEATURES") < names.index("FEATURE_COLUMNS")
    for derived in ("LEGACY_FEATURE_COLUMNS", "DEFAULT_SCOPE_FEATURES",
                    "SCOPE_FEATURES"):
        assert names.index("FEATURE_COLUMNS") < names.index(derived)


def test_the_chain_spans_the_three_modules_it_actually_lives_in(scope_check):
    """The four names are not in one file, and a check that assumes they are is wrong.

    This is the error the first draft of the pack made: it named the constants as
    though `SCOPE_FEATURES` sat beside `FEATURE_COLUMNS` in
    `src/evaluation/solar_retrain.py`. It does not.
    """
    by_name = dict((name, path) for path, name in scope_check.CONSTANT_CHAIN)
    assert by_name["SOLAR_GEOMETRY_FEATURES"] == "src/solar_features.py"
    assert by_name["FEATURE_COLUMNS"] == "src/evaluation/solar_retrain.py"
    for name in ("LEGACY_FEATURE_COLUMNS", "DEFAULT_SCOPE_FEATURES",
                 "SCOPE_FEATURES"):
        assert by_name[name] == "scripts/evaluate_solar_retrain.py"


def test_an_annotated_assignment_is_found(scope_check):
    """`SOLAR_GEOMETRY_FEATURES` carries a `Tuple[str, ...]` annotation.

    A walker that only knows `ast.Assign` misses it and dies with a `KeyError`
    that reads like a deleted constant. Held because the first working version of
    this resolver had exactly that bug.
    """
    source = "X: Tuple[str, ...] = ('a', 'b')\nY = ('c',)\n"
    found = scope_check.assigned_expressions(source, {"X", "Y"})
    assert found["X"] == "('a', 'b')"
    assert found["Y"] == "('c',)"


def test_the_ast_instrument_is_blind_to_an_upstream_feature_list_move(scope_check):
    """The demonstration, run rather than described.

    Appending one name to `SOLAR_GEOMETRY_FEATURES` takes `FEATURE_COLUMNS` from
    27 to 28 -- precisely the move
    `test_ch_solar_rejoined_at_the_current_27_name_list` exists to catch -- and
    leaves the `ast.dump` hash of both `FEATURE_COLUMNS` and
    `DEFAULT_SCOPE_FEATURES` untouched. If this ever fails because the AST hash
    *did* move, the constants have been respelled as literals and the
    resolved-value check is no longer load-bearing; that is a good failure and
    the report's section 1 should be re-read, not the test deleted.
    """
    demonstration = scope_check.blind_spot_demonstration("HEAD", REPO)
    rows = {row["constant"]: row for row in demonstration["constants"]}

    assert rows["FEATURE_COLUMNS"]["n_actual"] == 27
    assert rows["FEATURE_COLUMNS"]["n_probed"] == 28

    # The value instrument sees it; the AST instrument does not.
    for name in ("FEATURE_COLUMNS", "DEFAULT_SCOPE_FEATURES"):
        assert rows[name]["value_detects_the_change"], name
        assert not rows[name]["ast_detects_the_change"], name
        assert rows[name]["ast_is_blind_to_this_change"], name

    assert set(demonstration["constants_the_ast_check_would_miss"]) == {
        "FEATURE_COLUMNS", "DEFAULT_SCOPE_FEATURES"}
    # It is a demonstration, not a migration: it must not touch the tree.
    assert demonstration["writes_to_the_tree"] is False


def test_the_readmission_premise_holds_on_the_current_tree(scope_check):
    """CH solar's readmission rests on these, so they are asserted, not reported.

    `abl581-ch-solar-f27` absent from `SCOPE_FEATURES` is the *correct*
    configuration -- it inherits the current 27 through `DEFAULT_SCOPE_FEATURES`.
    Registering it would pin this scope to a list that could later drift from the
    builder, which is the CH failure mode in reverse.
    """
    resolved = scope_check.resolve("HEAD", REPO)
    current = tuple(resolved["FEATURE_COLUMNS"])
    legacy = tuple(resolved["LEGACY_FEATURE_COLUMNS"])
    scope_features = resolved["SCOPE_FEATURES"]

    assert len(current) == 27 and len(legacy) == 25
    assert tuple(resolved["DEFAULT_SCOPE_FEATURES"]) == current
    assert scope_check.NEW_SCOPE not in scope_features
    assert tuple(scope_features[scope_check.LEGACY_PINNED_SCOPE]) == legacy
    assert all(tuple(v) == legacy for v in scope_features.values())
