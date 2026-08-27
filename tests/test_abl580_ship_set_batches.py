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

#: What the ABL-525 batch was fitted with, before this file's subject existed.
#: Every row of that batch must still resolve to it.
ABL525_ALGORITHM = "catboost"


def _load_trainer():
    """Import the trainer by path.

    `scripts/` is not a package on `sys.path` by name here, and the trainer
    inserts the repo root itself at import time, so a spec load is the shape that
    works from `tests/` without a conftest change.
    """
    spec = importlib.util.spec_from_file_location("abl525_train_ship_set", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    """
    for row in trainer.SHIP_SET:
        if row.get("feature_columns"):
            assert row["hold"], f"{row['country']}/{row['forecast_type']} pins a list but ships"


def test_ch_solar_is_still_held(trainer):
    """The CEO's 2026-08-27 ruling, held as a property rather than as a comment."""
    ch = next(row for row in trainer.SHIP_SET
              if row["country"] == "CH" and row["forecast_type"] == "solar")
    assert ch["hold"], "CH solar was withdrawn; a run must not fit it by default"
    assert "WITHDRAWN" in ch["hold"]


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


def test_the_ship_set_reads_one_source_table(trainer):
    """ABL-321/ABL-348 register `energy_generation` for every pair in this set."""
    assert trainer.RENEWABLE_SOURCE == "energy_generation"
