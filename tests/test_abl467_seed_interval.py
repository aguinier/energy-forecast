"""ABL-467: the readability test at k > 1 is the Student-t interval on the draws.

Three things are pinned here, in descending order of what they would cost if they
broke:

1. **No published letter moves.** Every committed graded cell is ``k = 1``, and a
   ``k = 1`` grade must serialise byte-identically to what the module wrote before
   this amendment -- no new keys, no changed values, whatever the scope's
   ``SEED_READABILITY`` row says. That is what makes this an addition.
2. **The arithmetic is ABL-427's.** The six cells re-read at 12 seeds are pinned
   by their raw per-seed WAPEs, not by their conclusions, so the interval is
   recomputed here rather than copied.
3. **It moves letters in both directions.** A sharper test is not a laxer one,
   and the case where it grades *down* is constructed and pinned so nobody has to
   take that on trust.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import pathlib
import statistics
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate_grading import (  # noqa: E402
    DELTA_MIN, SEED_READABILITY_FORMS, STUDENT_T, T_CRIT_95, T_CRIT_95_MAX_DF,
    Z_95, CellGrade, cell_grade, grade_cell, readability_floor_pct,
    seed_skill_draws, seed_wapes_from, skill_interval, t_crit_95,
)

HARNESS_PATHS = {"solar": ROOT / "scripts" / "evaluate_solar_retrain.py",
                 "wind": ROOT / "scripts" / "evaluate_wind_retrain.py"}


def _harness(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Imported rather than parsed: the tables hold the module's own constants, so
#: `ast.literal_eval` cannot read them and a test that compared *strings* would
#: pass against a table pinning the wrong symbol.
HARNESSES = {"solar": _harness("evaluate_solar_retrain"),
             "wind": _harness("evaluate_wind_retrain")}

#: ABL-427's six cells, pinned by the numbers a reader can check against
#: `reports/abl_427_tranche2c_seed_reread.json` rather than by its verdicts. The
#: per-seed WAPEs are the *input*; `published_*` are what that read recorded, and
#: this file recomputes them.
ABL427_CELLS = (
    {
        "country": 'IT', "band": '24-36h',
        "seasonal_naive_wape_pct": 6.955756827018057,
        "challenger_wape_pct_k_mean": 6.678530603026604,
        "per_seed": (
            6.6251086331297335, 6.596047655239623,
            6.625372469584932, 6.2917624164080275,
            6.624863861078841, 6.807204747913263,
            6.329344619276436, 7.401121687357291,
            7.146076055410404, 6.798610703140579,
            6.309321847169561, 6.587532540610555,
        ),
        "published_half_width_pp": 3.0210178659493234,
        "published_ci95_pct": (0.9645473411986689, 7.006583073097316),
        "published_seeds_losing": 2,
        "registered_floor_pct": 4.771756396713317,
    },
    {
        "country": 'IT', "band": '36-48h',
        "seasonal_naive_wape_pct": 6.955756827018057,
        "challenger_wape_pct_k_mean": 6.7689343576479635,
        "per_seed": (
            6.7072856541720505, 6.672471993199962,
            6.674603258608406, 6.400162165986374,
            6.762056874784613, 6.918002046660992,
            6.363403974186221, 7.4787287169454215,
            7.220303936205106, 6.9498269047441905,
            6.402387574859685, 6.677979191422536,
        ),
        "published_half_width_pp": 3.0515234764587267,
        "published_ci95_pct": (-0.3656551516141304, 5.737391801303323),
        "published_seeds_losing": 2,
        "registered_floor_pct": 4.755566992956483,
    },
    {
        "country": 'IT', "band": '48-64h',
        "seasonal_naive_wape_pct": 6.584479313077244,
        "challenger_wape_pct_k_mean": 6.170296827507673,
        "per_seed": (
            6.0492168235602115, 6.284448197056603,
            5.997329441397598, 5.802819845935647,
            6.084079651477858, 6.338743831158179,
            5.812484142980215, 6.772660715755421,
            6.49522028144358, 6.4864526663747935,
            5.782664969555503, 6.137441363396463,
        ),
        "published_half_width_pp": 3.018243829961235,
        "published_ci95_pct": (3.2720407297757124, 9.308528389698182),
        "published_seeds_losing": 1,
        "registered_floor_pct": 4.884624717976266,
    },
    {
        "country": 'HR', "band": '24-36h',
        "seasonal_naive_wape_pct": 16.22328463627126,
        "challenger_wape_pct_k_mean": 15.3484062047093,
        "per_seed": (
            14.95212041295267, 14.96330834040932,
            14.841808441095864, 15.942993485493611,
            14.42118650350938, 15.011728896068723,
            15.112331601970471, 16.852967671935673,
            14.282003900484275, 15.481139862153032,
            15.659737205721852, 16.659548134716758,
        ),
        "published_half_width_pp": 3.1638811060992444,
        "published_ci95_pct": (2.2288519388844104, 8.556614151082899),
        "published_seeds_losing": 2,
        "registered_floor_pct": 5.071742069560667,
    },
    {
        "country": 'HR', "band": '36-48h',
        "seasonal_naive_wape_pct": 16.22328463627126,
        "challenger_wape_pct_k_mean": 15.352036960279975,
        "per_seed": (
            15.060056281584137, 14.942974510638399,
            14.74024739242564, 16.026060797478586,
            14.431001306249524, 14.98751463316439,
            15.01451490427881, 16.896516371355286,
            14.420038850779598, 15.534658772823922,
            15.629038228197935, 16.541821474383443,
        ),
        "published_half_width_pp": 3.1143741683202704,
        "published_ci95_pct": (2.2559789723957095, 8.48472730903625),
        "published_seeds_losing": 2,
        "registered_floor_pct": 4.991201114209303,
    },
    {
        "country": 'HR', "band": '48-64h',
        "seasonal_naive_wape_pct": 16.188733724194265,
        "challenger_wape_pct_k_mean": 15.440535539791767,
        "per_seed": (
            14.793581270537342, 15.261471622789266,
            14.650000068822706, 16.208765964256674,
            14.675572082421096, 15.018844697435332,
            15.358004303831974, 16.90331496213176,
            14.52372877651387, 15.410973229939872,
            15.795065152002477, 16.68710434681882,
        ),
        "published_half_width_pp": 3.1523412148223198,
        "published_ci95_pct": (1.469380268460264, 7.7740626981049035),
        "published_seeds_losing": 3,
        "registered_floor_pct": 5.012394456213951,
    },
)


def _scores(cell, *, level=1.0, shape=1.0, slope=1.0, correlation=0.9):
    """A cell's `scores` mapping. G2/G3 references are set far outside any floor
    unless a test is about them, so a letter here turns on G1 alone."""
    d7 = cell["seasonal_naive_wape_pct"]
    return {"challenger": {"wape_pct": cell["challenger_wape_pct_k_mean"],
                           "slope": slope, "correlation": correlation},
            "seasonal_naive": {"wape_pct": d7},
            "constant_causal": {"wape_pct": d7 * 14 * level},
            "climatology_causal": {"wape_pct": d7 * 4 * shape}}


def _committed_score_blocks():
    """Every `scores` mapping committed to the repo, from any record shape."""
    paths = sorted((ROOT / "reports").glob("*.json")) + sorted((ROOT / "experiments").rglob("*.json"))
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        stack = [document]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                scores = node.get("scores")
                if isinstance(scores, dict) and "challenger" in scores:
                    yield path, scores
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)


def _module_const(stream: str, name: str):
    """A registration table, off the imported harness module."""
    table = getattr(HARNESSES[stream], name, None)
    assert table is not None, f"{name} not found in the {stream} harness"
    return table


# --------------------------------------------------------------------------
# 1. The critical values, pinned rather than imported
# --------------------------------------------------------------------------

def test_the_pinned_t_table_matches_scipy_on_every_row():
    """The reason to pin is that a dependency upgrade must not move a verdict.
    The reason to check is that a pinned number can be a typo."""
    stats = pytest.importorskip("scipy.stats")
    for df, value in T_CRIT_95.items():
        assert value == pytest.approx(stats.t.ppf(0.975, df), rel=1e-12), f"df={df}"


def test_the_table_covers_every_degree_of_freedom_up_to_its_stated_maximum():
    assert set(T_CRIT_95) == set(range(1, T_CRIT_95_MAX_DF + 1))


def test_the_critical_value_falls_to_z_above_the_table_and_is_bounded():
    """The fallback is documented as anti-conservative and bounded by 3.9% of the
    correct half-width. Both halves of that claim are asserted, because a bound
    stated only in a comment is how the wrong figure first written there got past
    its own author."""
    stats = pytest.importorskip("scipy.stats")
    assert t_crit_95(T_CRIT_95_MAX_DF + 1) == Z_95
    exact = stats.t.ppf(0.975, T_CRIT_95_MAX_DF + 1)
    assert (exact - Z_95) / exact < 0.039
    # ...and it only improves from there, so the boundary is the binding case.
    far = stats.t.ppf(0.975, 120)
    assert (far - Z_95) / far < 0.011


def test_a_single_draw_has_no_degrees_of_freedom_and_raises():
    """`delta_min` exists precisely because this case has no internal spread."""
    with pytest.raises(ValueError, match="no internal estimate"):
        t_crit_95(0)


def test_t_is_always_wider_than_z_which_is_what_pays_for_the_estimated_sd():
    assert all(value > Z_95 for value in T_CRIT_95.values())


# --------------------------------------------------------------------------
# 2. The draws, and the property that licenses one set of them
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cell", ABL427_CELLS, ids=lambda c: f"{c['country']}-{c['band']}")
def test_the_mean_of_the_skill_draws_is_the_printed_skill_column(cell):
    """The load-bearing identity: skill is affine in WAPE against a deterministic
    reference, so re-estimating the *width* cannot move the *point estimate*."""
    draws = seed_skill_draws(cell["per_seed"], cell["seasonal_naive_wape_pct"])
    printed = 100.0 * (1.0 - cell["challenger_wape_pct_k_mean"] / cell["seasonal_naive_wape_pct"])
    assert statistics.fmean(draws) == pytest.approx(printed, abs=1e-12)


def test_fewer_than_two_draws_yields_no_interval_rather_than_a_zero_width_one():
    """A zero-width interval would read as infinitely decisive."""
    assert skill_interval(None) is None
    assert skill_interval([]) is None
    assert skill_interval([4.0]) is None
    assert skill_interval([4.0, 5.0]) is not None


def test_draws_against_an_unmeasured_reference_are_none_not_zero():
    assert seed_skill_draws([1.0, 2.0], None) is None
    assert seed_skill_draws([1.0, 2.0], 0.0) is None
    assert seed_skill_draws(None, 5.0) is None


# --------------------------------------------------------------------------
# 3. ABL-427's six cells reproduce
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cell", ABL427_CELLS, ids=lambda c: f"{c['country']}-{c['band']}")
def test_the_interval_reproduces_what_abl427_published(cell):
    interval = skill_interval(seed_skill_draws(cell["per_seed"], cell["seasonal_naive_wape_pct"]))
    assert interval["n_seeds"] == 12
    assert interval["t_crit_95"] == T_CRIT_95[11]
    assert interval["half_width_pp"] == pytest.approx(cell["published_half_width_pp"], abs=1e-9)
    assert interval["ci95_pct"][0] == pytest.approx(cell["published_ci95_pct"][0], abs=1e-9)
    assert interval["ci95_pct"][1] == pytest.approx(cell["published_ci95_pct"][1], abs=1e-9)
    assert interval["draws_losing"] == cell["published_seeds_losing"]


#: What the amendment decides on each of ABL-427's cells, registered here before
#: the re-grade runs. IT stays `U` on 36-48h and so stays `U` overall; HR resolves
#: `A`. `reports/abl_467_seed_interval_readability_registration.md` §3.
EXPECTED_UNDER_THE_AMENDMENT = {
    ("IT", "24-36h"): True, ("IT", "36-48h"): False, ("IT", "48-64h"): True,
    ("HR", "24-36h"): True, ("HR", "36-48h"): True, ("HR", "48-64h"): True,
}


@pytest.mark.parametrize("cell", ABL427_CELLS, ids=lambda c: f"{c['country']}-{c['band']}")
def test_the_amended_g1_verdict_is_the_one_registered_before_the_regrade(cell):
    grade = grade_cell(_scores(cell), "solar", k=12, levelling="fit_window",
                       g23_readability="sign_test", seed_wapes=cell["per_seed"],
                       seed_readability=STUDENT_T)
    expected = EXPECTED_UNDER_THE_AMENDMENT[(cell["country"], cell["band"])]
    assert grade.conditions["G1"] is expected
    assert grade.readability_test == STUDENT_T
    # `A` iff G1 holds, since this fixture puts G2-G4 far clear.
    assert grade.grade == ("A" if expected else "U")


@pytest.mark.parametrize("cell", ABL427_CELLS, ids=lambda c: f"{c['country']}-{c['band']}")
def test_abl427s_own_registered_floor_is_the_one_that_disagrees(cell):
    """The amendment agrees with this module's *unamended* floor at k=12 on all
    six cells. Only ABL-427's scope-level chi-square upper bound differs, and on
    exactly the two cells its pack named."""
    skill = 100.0 * (1.0 - cell["challenger_wape_pct_k_mean"] / cell["seasonal_naive_wape_pct"])
    unamended = skill > readability_floor_pct("solar", 12)
    amended = EXPECTED_UNDER_THE_AMENDMENT[(cell["country"], cell["band"])]
    assert unamended is amended
    registered = skill > cell["registered_floor_pct"]
    disagrees = (cell["country"], cell["band"]) in {("IT", "24-36h"), ("HR", "48-64h")}
    assert (registered != amended) is disagrees


# --------------------------------------------------------------------------
# 4. Nothing already published can move
# --------------------------------------------------------------------------

def test_every_committed_graded_cell_is_k1_which_is_the_whole_blast_radius():
    """The issue asserts this; it is checked rather than trusted. A cell graded at
    k > 1 would carry a floor other than the two k = 1 values."""
    seen, floors = 0, set()
    paths = sorted((ROOT / "reports").glob("*.json")) + sorted((ROOT / "experiments").rglob("*.json"))
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        stack = [document]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "grade" in node and isinstance(node.get("floor_pct"), (int, float)):
                    seen += 1
                    floors.add(round(node["floor_pct"], 4))
                    assert "readability_test" not in node, f"{path.name} carries a k>1 grade"
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    assert seen >= 600, f"expected the committed corpus, found {seen} graded cells"
    assert floors == {round(readability_floor_pct("solar"), 4),
                      round(readability_floor_pct("wind"), 4)}, floors


def test_a_k1_grade_carries_no_abl467_key_whatever_the_registered_form_says():
    """The amendment is structurally unreachable at k = 1: there is nothing to
    take a t of, so the table cannot make a one-fit read move."""
    cell = ABL427_CELLS[0]
    for form in SEED_READABILITY_FORMS:
        record = grade_cell(_scores(cell), "solar", k=1, seed_readability=form).as_dict()
        assert "readability_test" not in record
        assert "half_width_pct" not in record
        assert "seed_interval" not in record


def test_no_committed_score_block_grades_differently_under_the_amendment():
    """The regression that matters: replay every committed `scores` mapping the way
    a re-render does -- no draws -- and require the record to be byte-identical."""
    checked = 0
    for path, scores in _committed_score_blocks():
        for stream in ("solar", "wind"):
            for levelling in ("fit_window", "trailing_28d"):
                for readability in ("sign_test", "floored"):
                    plain = grade_cell(scores, stream, 1, levelling, readability).as_dict()
                    passed = grade_cell(scores, stream, 1, levelling, readability,
                                        seed_wapes=None, seed_readability=STUDENT_T).as_dict()
                    assert plain == passed, f"{path.name} {stream} {levelling} {readability}"
                    assert plain["floor_pct"] == readability_floor_pct(stream, 1)
                    checked += 1
    assert checked >= 1500, f"expected the committed corpus, replayed {checked}"


def test_a_scope_pinned_to_delta_min_ignores_the_draws_it_was_handed():
    """The pin is what leaves ABL-427's published pack standing. Handing the same
    cell the same draws under the pinned form must change nothing."""
    cell = ABL427_CELLS[5]          # HR 48-64h -- the cell the two tests disagree on
    pinned = grade_cell(_scores(cell), "solar", k=12, levelling="fit_window",
                        g23_readability="sign_test", seed_wapes=cell["per_seed"],
                        seed_readability=DELTA_MIN)
    blind = grade_cell(_scores(cell), "solar", k=12, levelling="fit_window",
                       g23_readability="sign_test")
    assert pinned.as_dict() == blind.as_dict()
    assert pinned.readability_test == DELTA_MIN
    assert pinned.seed_interval == {}


# --------------------------------------------------------------------------
# 5. It is a sharper test, not a laxer one
# --------------------------------------------------------------------------

def test_the_amendment_grades_down_as_well_as_up():
    """A cell whose own seeds scatter more than the fleet p90 gets a *wider*
    half-width than `delta_min` and loses the letter `delta_min` would award.
    Constructed rather than found, because no such cell has been fitted yet."""
    d7 = 10.0
    # Mean WAPE 9.0 -> +10% skill, comfortably clear of the k=4 delta_min floor
    # (5.32%), but with a seed spread far wider than the fleet's.
    per_seed = (5.4, 12.6, 6.3, 11.7)
    scores = {"challenger": {"wape_pct": statistics.fmean(per_seed), "slope": 1.0,
                             "correlation": 0.9},
              "seasonal_naive": {"wape_pct": d7},
              "constant_causal": {"wape_pct": d7 * 14},
              "climatology_causal": {"wape_pct": d7 * 4}}
    lax = grade_cell(scores, "solar", k=4, levelling="fit_window",
                     g23_readability="sign_test", seed_readability=DELTA_MIN)
    sharp = grade_cell(scores, "solar", k=4, levelling="fit_window",
                       g23_readability="sign_test", seed_wapes=per_seed,
                       seed_readability=STUDENT_T)
    assert lax.grade == "A" and lax.conditions["G1"] is True
    assert sharp.grade == "U" and sharp.conditions["G1"] is False
    assert sharp.half_width_for("seasonal_naive") > lax.floor_pct
    # The point estimate is identical in both; only the width moved.
    assert sharp.skill["seasonal_naive"] == pytest.approx(lax.skill["seasonal_naive"])


def test_the_floor_stays_on_the_record_so_the_two_widths_can_be_compared():
    cell = ABL427_CELLS[3]
    grade = grade_cell(_scores(cell), "solar", k=12, levelling="fit_window",
                       g23_readability="sign_test", seed_wapes=cell["per_seed"],
                       seed_readability=STUDENT_T)
    assert grade.floor_pct == readability_floor_pct("solar", 12)
    assert grade.half_width_for("seasonal_naive") != grade.floor_pct
    # HR is graded against a half-width *wider* than the unamended floor and still
    # clears it, which is the fact that stops this reading as a laxer test.
    assert grade.half_width_for("seasonal_naive") > grade.floor_pct
    assert grade.conditions["G1"] is True


# --------------------------------------------------------------------------
# 6. The draws have to belong to the cell they grade
# --------------------------------------------------------------------------

def test_a_seed_count_disagreeing_with_k_raises_rather_than_guessing():
    cell = ABL427_CELLS[0]
    with pytest.raises(ValueError, match="per-seed WAPEs were passed"):
        grade_cell(_scores(cell), "solar", k=11, seed_wapes=cell["per_seed"],
                   seed_readability=STUDENT_T)


def test_draws_from_another_cell_raise_rather_than_centring_the_interval_wrongly():
    """The failure this catches is a paste, and it is silent otherwise: the
    interval would be a spread around one cell's mean and the graded margin
    another's."""
    scores = _scores(ABL427_CELLS[0])
    with pytest.raises(ValueError, match="must record the mean over its own seeds"):
        grade_cell(scores, "solar", k=12, seed_wapes=ABL427_CELLS[3]["per_seed"],
                   seed_readability=STUDENT_T)


def test_an_unknown_seed_readability_form_raises():
    with pytest.raises(ValueError, match="unknown seed readability form"):
        grade_cell(_scores(ABL427_CELLS[0]), "solar", seed_readability="whatever")


@pytest.mark.parametrize("recorded,expected", [
    ({"42": 1.0, "7": 2.0}, [1.0, 2.0]),
    ([1.0, 2.0], [1.0, 2.0]),
    (None, None), ({}, None), ([], None),
])
def test_a_cell_records_its_draws_as_a_dict_or_a_list_or_not_at_all(recorded, expected):
    assert seed_wapes_from({"challenger_wape_pct_per_seed": recorded}) == expected
    assert seed_wapes_from({}) is None


def test_cell_grade_takes_the_draws_off_the_cell_the_way_abl434_takes_coverage():
    """`grade_cell` stays a function of `scores` plus what it is explicitly given;
    `cell_grade` holds the whole cell, so it is the layer that reads the draws."""
    cell = ABL427_CELLS[5]
    record = {"scores": _scores(cell),
              "challenger_wape_pct_per_seed": dict(zip("abcdefghijkl", cell["per_seed"]))}
    graded = cell_grade(record, "solar", k=12, levelling="fit_window",
                        g23_readability="sign_test", seed_readability=STUDENT_T)
    assert graded.readability_test == STUDENT_T
    assert graded.seed_interval["seasonal_naive"]["n_seeds"] == 12


def test_a_recorded_grade_is_still_read_back_and_not_re_decided():
    """ABL-437's rule, unchanged: a stored letter is the record of what that read
    decided, and handing the cell draws afterwards must not re-open it."""
    cell = ABL427_CELLS[5]
    stored = grade_cell(_scores(cell), "solar", k=12, seed_readability=DELTA_MIN).as_dict()
    record = {"scores": _scores(cell), "grade": stored,
              "challenger_wape_pct_per_seed": list(cell["per_seed"])}
    assert cell_grade(record, "solar", k=12, seed_readability=STUDENT_T).as_dict() == stored


# --------------------------------------------------------------------------
# 7. The record round-trips
# --------------------------------------------------------------------------

def test_an_amended_grade_survives_a_round_trip_through_its_record():
    cell = ABL427_CELLS[3]
    grade = grade_cell(_scores(cell), "solar", k=12, levelling="fit_window",
                       g23_readability="floored", seed_wapes=cell["per_seed"],
                       seed_readability=STUDENT_T)
    rebuilt = CellGrade.from_dict(grade.as_dict())
    assert rebuilt.as_dict() == grade.as_dict()
    assert rebuilt.readability_test == STUDENT_T
    assert rebuilt.half_width_for("seasonal_naive") == grade.half_width_for("seasonal_naive")


def test_a_record_without_the_key_rebuilds_as_delta_min_because_absence_dates_it():
    rebuilt = CellGrade.from_dict({"grade": "A", "floor_pct": 10.6482})
    assert rebuilt.readability_test == DELTA_MIN
    assert rebuilt.half_width_for("seasonal_naive") == 10.6482


def test_g2_and_g3_are_floored_against_their_own_reference_not_g1s():
    """Each condition's margin has its own denominator, so each gets its own
    half-width. Sharing G1's would floor a 93% margin against a 3% width."""
    cell = ABL427_CELLS[0]
    d7 = cell["seasonal_naive_wape_pct"]
    mean = cell["challenger_wape_pct_k_mean"]
    scores = {"challenger": {"wape_pct": mean, "slope": 1.0, "correlation": 0.9},
              "seasonal_naive": {"wape_pct": d7},
              # A constant the challenger barely beats: the G2 margin lands inside
              # its own Student-t half-width and must abstain, not pass.
              "constant_causal": {"wape_pct": mean * 1.004},
              "climatology_causal": {"wape_pct": d7 * 4}}
    grade = grade_cell(scores, "solar", k=12, levelling="fit_window",
                       g23_readability="floored", seed_wapes=cell["per_seed"],
                       seed_readability=STUDENT_T)
    widths = grade.half_width_pct
    assert set(widths) == {"seasonal_naive", "constant_causal", "climatology_causal"}
    assert widths["constant_causal"] != widths["seasonal_naive"]
    assert grade.grade == "N"
    assert [name for name, _ in grade.not_readable] == ["G2"]
    assert "Student-t half-width on 12 seeds" in dict(grade.not_readable)["G2"]


# --------------------------------------------------------------------------
# 8. The registration tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_every_published_scope_pins_delta_min(stream):
    """Stronger than presence in `check_registration_tables`: the *value* is
    asserted, which is what actually leaves a published letter standing."""
    table = _module_const(stream, "SEED_READABILITY")
    assert table, f"{stream} declares no SEED_READABILITY table"
    assert set(table.values()) == {DELTA_MIN}, table


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_seed_readability_covers_every_scope_the_g23_table_does(stream):
    """The two tables are structural twins keyed on the same scope set, so a scope
    added to one and forgotten in the other is a drift this catches."""
    assert (set(_module_const(stream, "SEED_READABILITY"))
            >= set(_module_const(stream, "G23_READABILITY")))


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_seed_readability_is_declared_unchecked_with_its_reason(stream):
    """ABL-429's rule: a per-scope table is either in `check_registration_tables`
    or declared here with why it cannot join."""
    declared = _module_const(stream, "UNCHECKED_REGISTRATION_TABLES")
    assert "SEED_READABILITY" in declared
    assert "ABL-467" in declared["SEED_READABILITY"]


@pytest.mark.parametrize("stream", sorted(HARNESSES))
def test_the_harness_does_not_reimplement_the_interval(stream):
    """One implementation of the test, in the ladder. A harness computing its own
    `t * sd / sqrt(k)` is the second implementation this module exists to prevent."""
    source = HARNESS_PATHS[stream].read_text(encoding="utf-8")
    assert "t_crit" not in source
    assert "stdev(" not in source


def test_the_amended_default_is_the_amendment_and_the_pinned_scopes_opt_out():
    """Same direction as ABL-437 and ABL-444: a new scope inherits the amendment.
    Safe here for a reason those two did not have -- a new scope at k = 1, which
    is every scope to date, cannot be affected by the default at all."""
    solar = _module_const("solar", "SEED_READABILITY")
    assert "abl427-t2c-reread" in solar, "ABL-427's own scope must be pinned"
    assert solar["abl427-t2c-reread"] == DELTA_MIN
    assert "abl467-t2c-regrade" not in solar, (
        "the re-grade is a new scope and must inherit the amendment, not pin it")


def test_the_module_takes_no_new_dependency_to_decide_a_verdict():
    """`gate_grading` imports `math` and `statistics` and nothing else. Pinning the
    t table rather than importing `scipy` is what keeps a registered verdict
    immune to a dependency upgrade."""
    source = (ROOT / "src" / "evaluation" / "gate_grading.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    assert imported == {"__future__", "math", "statistics", "dataclasses", "src"}, imported


def test_the_half_width_is_exactly_t_times_the_standard_error():
    """Spelled out once so the arithmetic is checkable without reading the source:
    a reviewer can recompute this line from any t-table."""
    draws = [4.0, 6.0, 5.0, 9.0, 1.0]
    interval = skill_interval(draws)
    expected = T_CRIT_95[4] * statistics.stdev(draws) / math.sqrt(5)
    assert interval["half_width_pp"] == pytest.approx(expected, rel=1e-15)
    assert interval["mean_skill_pct"] == pytest.approx(5.0)
    assert interval["draws_losing"] == 0
