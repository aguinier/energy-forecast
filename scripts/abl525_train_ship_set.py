#!/usr/bin/env python
"""Fit production artifacts for the ABL-316 ship set approved by the Board (ABL-525/580/583).

The Board answered `abl316:ship-decision:v4` with `ship8` on 2026-08-22. This
script fits the artifacts that answer authorises, through the *graded* code path
and nothing else.

BATCHES, AND WHY THIS STAYS ONE SCRIPT
--------------------------------------
The Board approved a *rule* alongside the `ship8` roster (ABL-316 ledger 14.6,
restated in 15.1): a held pair that later satisfies the same rule joins the
shipping set without a new Board card, and a shipping pair a later correction
moves outside it is withdrawn. The ship set is therefore a growing table, not a
fixed one, and `SHIP_SET` carries a `batch` per row rather than growing a second
script per admission. `--batch` restricts a run; each batch writes its own
machine record, and no run refits another batch's artifacts.

    abl525   the seven `wind_onshore` pairs of the original `ship8` roster.
    abl580   CZ `solar`, RO `solar`, NL `wind_offshore`, admitted under the
             standing rule once ABL-426 (tranche 2a re-read on the registered
             `energy_generation`) and ABL-471 (the last four vintage screens)
             cleared their holds.
    abl583   CH `solar`, readmitted under the same standing rule on 2026-08-27
             once ABL-581 read it at the current 27-name list under its own
             pre-registered scope. It was pair 8 of the original `ship8` roster
             and was withdrawn on 2026-08-27; the row records both moves.
    abl602   the five pairs the Board's `widen7` answer admits once the
             causally-available standard is applied to the machine record, of
             which **four ship**: `HU` `wind_onshore` was fitted and then
             withdrawn by CEO ruling on 2026-08-28, and its row carries a
             `hold`. This batch is a *widening* of the roster rather than an
             admission under the standing rule, so it carries a Board answer of
             its own; see "THE ABL-602 WIDENED SET" below for why it is five
             and not seven, and "HU IS WITHDRAWN" for why four of the five ship.

A pair appears **once**. A batch is the admission that authorises the row as it
stands, not a log of every disposition the row has had -- that is what
`admission_history` is for. Two rows for one pair would give `columns_for` two
answers and no way to choose, which is precisely the class of defect this script
exists to avoid.

THE ALGORITHM IS A PROPERTY OF THE FORECAST TYPE, NOT OF THIS FILE
------------------------------------------------------------------
ABL-525's eight were seven `wind_onshore` plus one `solar`, and both types are
catboost, so this script carried a single `ALGORITHM = "catboost"`. ABL-580 adds
NL `wind_offshore`, and **the wind gate fits offshore with xgboost**:
`evaluate_wind_retrain.ALGORITHMS` is `{"wind_offshore": "xgboost",
"wind_onshore": "catboost"}`, and the pilot's own committed record
(`experiments/ABL322/results_abl436_offshore_reread.json`) shows DE and NL both
fitted `"algorithm": "xgboost"` with exactly `config.get_default_params(
"xgboost")` less `early_stopping_rounds`. Fitting offshore with catboost here
would ship a model no gate read -- the same class of error ABL-525 item 2 exists
to prevent, arriving through a constant rather than through a feature list.

`ALGORITHM_BY_TYPE` therefore *imports* the harnesses' tables instead of
restating them, so this script cannot come to disagree with the code that graded
the pair. It resolves to catboost for every ABL-525 row, so the change is a no-op
for the seven already fitted and `abl525_repro_check.py` still reproduces them.

WHY NOT `scripts/train.py`
--------------------------
ABL-525 item 1 names `scripts/train.py`; item 2 requires the ABL-183 / ABL-191
serve-faithful builders and the feature set the tranches graded. Those are two
different pipelines and only one of them can be served:

    train.py     -> features.create_all_features + select_feature_columns
                    = 28 names (wind) / 31 (solar), including 4 holiday columns
    gate harness -> wind_features.RenewableFeatureBuilder + to_vector
                    = 24 names (wind) / 27 (solar), no holiday columns

`Forecaster.predict_d2` routes wind_onshore/wind_offshore/solar to
`_predict_d2_serve_faithful`, which calls `to_vector(row, artifact.feature_columns)`,
and `to_vector` raises `KeyError` on a column the builder does not produce. The
builder produces no holiday column (`wind_features.py:179`). So a `train.py`
artifact for any of these pairs loads clean and then raises on its first serving
row -- `forecast_daily.py` books a failed result and the pair serves zero rows.
This script therefore fits the way the tranches did: same builder, same
FEATURE_COLUMNS, same algorithm, written through `save_gate_artifact` so
`training_source` (ABL-331/ABL-342) and the ABL-183 intercept witness are
derived from the fit rather than claimed by the caller.

THE FIT WINDOW IS BOUNDED BY WEATHER, NOT BY ACTUALS
----------------------------------------------------
Item 1 asks for full available history. `energy_generation` reaches back to
2021-01-01 for every pair here, but a serve-faithful row also needs the weather
*forecast* archive, and `weather_data` with `data_quality='forecast'` begins
2026-01-11 for every one of these countries -- re-measured on the 2026-08-27
replica for CZ, RO and NL, all three first run at 2026-01-11 18:00. An earlier
target gets NaN weather and `finite_training_rows` drops it. So the widest honest
window is 2026-01-11 -> 2026-08-22 (223 days) against the gate's registered 178,
and the run records what was actually retained rather than what was requested.

`FIT_END` stays at the Board's decision date for both batches even though the
2026-08-27 replica carries actuals to 2026-08-26. Two reasons, and neither is
inertia: every artifact in the ship set is then on one window, so the deploy is a
homogeneous batch; and `abl525_repro_check.py` refits through `fit_one` on these
module constants, so moving them would make the seven ABL-525 artifacts report a
prediction difference that is a window change and not a drift. Five days is 2.2%
of the window; a false drift signal on the committed record costs more.

Because that window covers ABL-348's gate window, these artifacts have been
fitted on the rows the tranches scored. That is what item 1 asks for and is
correct for production, but it means the tranche gate figures are NOT
out-of-sample for these artifacts. This script scores nothing and grades nothing.

CH SOLAR: WITHDRAWN AT 25 NAMES, READMITTED AT 27
--------------------------------------------------
ABL-395 moved the solar gate list from 25 names to 27 (adding ABL-338's
`sun_elevation_deg` and `is_night`). CH solar was graded under tranche 1b, and
`evaluate_solar_retrain.SCOPE_FEATURES['abl316-t1b']` is pinned to the legacy 25
for exactly that reason -- so on 2026-08-27 the CEO withdrew it: fitting at 27
would have shipped a model nobody graded, and fitting at 25 would have been a
per-country serving fork on a list the current builder no longer produces.

ABL-581 closed that gap the only way the withdrawal ruling left open -- a fresh
pre-registered gate read at 27 under a **new** scope id, `abl581-ch-solar-f27`,
registered at `82e3108` and read at `49ab9e9`, PASS 3/3 grade A/A/A. So the
condition that held CH is gone, and the standing rule (ledger 14.6 / 15.1)
readmits it with no new Board card. The row below therefore ships at the
**default** list rather than at a pin, which is the same shape as CZ and RO --
and the equality that makes that safe is checked against the record the read
wrote, not against a registration table:

    experiments/ABL348/results_abl581_ch_solar_f27.json
      meta.scope                                abl581-ch-solar-f27
      meta.n_features                           27
      meta.feature_set                          legacy25+geometry
      meta.feature_set_is_registered_for_scope  False    (inherited the default)
      meta.registered_source                    energy_generation
      meta.feature_columns == src.evaluation.solar_retrain.FEATURE_COLUMNS -> True

`feature_set_is_registered_for_scope` being **False** is the correct
configuration and not a gap: `SCOPE_FEATURES` is one of the tables whose absence
encodes a choice (CLAUDE.md, gate-harness section), and inheriting the current
list through `DEFAULT_SCOPE_FEATURES` is the intended path for a new scope. A
pin would have frozen this scope to a list that could later drift from the
builder -- the CH failure mode in reverse.

`tests/test_abl580_ship_set_batches.py` holds the equality element-for-element,
so a later move of `FEATURE_COLUMNS` to 28 fails the suite rather than silently
re-basing this artifact.

THE SAME CHECK, RUN AGAINST ABL-580'S TWO SOLAR PAIRS, PASSES
-------------------------------------------------------------
CZ and RO solar are tranche 2a, dispositioned on the ABL-426 re-read
(`abl316-t2a-generation`). That scope is **deliberately absent** from
`SCOPE_FEATURES` and resolves through `DEFAULT_SCOPE_FEATURES`, so the claim
"read at 27" has to be checked against the record the fit wrote rather than
against a pin. It was: `experiments/ABL348/results_abl426_tranche2a_generation.json`
carries `meta.feature_columns` as 27 literal names, and that tuple is
element-for-element identical to today's `solar_retrain.FEATURE_COLUMNS`. So
these two are the opposite of CH -- the graded list and the current builder's
list are the same object, and shipping them at the default forks nothing.

THE ABL-602 WIDENED SET: FIVE, NOT SEVEN
-----------------------------------------
The Board answered `widen7` on ABL-316 on 2026-08-28, adopting the
causally-available standard for the widened set. Applying that standard to the
committed gate records leaves five pairs, not seven: `NO` and `RO`
`wind_onshore` also fail **G4** -- slope > 0 and correlation > 0, computed from
the challenger's own predictions against the actuals in the gate window, which
is causally available and not a reference comparison. Re-verified here against
the records rather than taken from the card:

    NO wind_onshore   reports/abl_418_retro_grade.json, tranche 2b -> grade B,
                      G4 false, and B under all four levelling/readability
                      conventions ABL-444 reports.
    RO wind_onshore   experiments/ABL348/results_abl417_tranche2e.json -> grade
                      B with G4 false in all three bands (24-36h, 36-48h,
                      48-64h).

Neither is in `SHIP_SET`. That exclusion is the CEO's ruling reported on
ABL-316, not a judgement this script makes.

THE LETTER DEPENDS ON THE CONVENTION, AND FOUR OF THE FIVE MOVE
---------------------------------------------------------------
Every scope in this batch is a *published* scope and is therefore pinned by
value on both grading axes -- `CAUSAL_LEVELLING[scope] = FIT_WINDOW` and
`G23_READABILITY[scope] = SIGN_TEST` for `abl316-t2d`, `abl417-tranche2e` and
`abl406-tranche2b`. So the registered letter is the published one. ABL-437 and
ABL-444 moved the *defaults* a new scope would inherit to `trailing_28d` and
`FLOORED`, and `reports/abl_444_g23_floor_reread.json` reports every pair under
all four combinations. The worst-band letters:

    pair              registered              amended defaults
                      fit_window/sign_test    trailing_28d/floored
    LT solar          A                       N   (G3 -6.87%, inside 10.65)
    SE solar          A                       N   (G3 -8.65%, inside 10.65)
    PL wind_onshore   A                       A
    HR wind_onshore   A                       N   (G2 +2.80 / G3 +2.02, inside 7.51)
    HU wind_onshore   B                       B   (G2 -26.78 / G3 -25.51)

Only PL holds grade A under all four conventions. That is a property of the
evidence, not a defect in it -- an abstention (`N`) is "the margin is inside the
floor", not a loss -- but a reader given only the published letter would not
know which four move.

HU IS WITHDRAWN: FIVE FITTED, FOUR SHIP
----------------------------------------
`HU` is the row to read twice. ABL-602's own description characterises it as
"G2/G3 inside floor -- abstention, not a readable loss". That is true under
`fit_window/floored` and under neither endpoint: HU is `B` under the registered
convention (G2 -1.75%, G3 -2.13% at the worst band -- a loss, unreadable but
signed) and `B` under the fully amended defaults, where the same margins become
**-26.78% and -25.51%**, three and a half times the 7.51% wind floor and
unambiguously readable. `N` appears only when ABL-444's floor is applied
*without* ABL-437's levelling.

On that reading the CEO **withdrew HU on 2026-08-28**, on a bar stated once and
reusable: *a pair ships unless a causally-available screen shows a readable
loss; an abstention -- a margin inside the readability floor -- does not block.*
That is why LT, SE and HR ship despite moving A -> N under the amended defaults
(`N` there is "cannot be read", not "lost") and why HU does not: its sign is
negative in 6 of 6 band x reference cells under **both** levellings, and only
the magnitude is convention-dependent. It is the same argument that took NO and
RO out on G4 -- G2 and G3 are as causally available as G4 is -- so the standard
in force stays single. The bar reads both conventions and can therefore only
ever *remove* a pair, never add one; it cannot launder a pair in through the
amended defaults.

This is a **disposition, not a re-grade**: no registered letter moves and
`abl417-tranche2e` stays pinned at `fit_window`/`sign_test`.

HU was already fitted when the ruling landed, so its artifact exists on disk in
a gitignored `models/` tree. Its row therefore keeps a `hold` rather than
leaving the table: a deleted row would fit nothing *and say nothing*, and the
hazard here is an artifact that exists and must not be served. The hold is a
disposition hold, not the CH-shaped feature-list hold -- HU pins no list, and
under `--include-held` it would refit at the same 24 names it was graded on.
The correction is carried in `reports/abl_602_widen5_ship.md` sections 3.3 and
1.1, the shipping four are named in `reports/abl_602_ship_disposition.json`,
and both are reported on ABL-316.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import config
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.solar_retrain import (
    ALGORITHM as SOLAR_ALGORITHM,
    FEATURE_COLUMNS as SOLAR_FEATURE_COLUMNS,
)
from src.evaluation.wind_retrain import (
    FEATURE_COLUMNS as WIND_FEATURE_COLUMNS,
    build_vintage_frame,
    finite_training_rows,
)
from src.wind_features import RenewableFeatureBuilder

# The estimator: the wind gate's per-type algorithm table is
# imported, not restated, so `wind_offshore` cannot silently be fitted with the
# `wind_onshore` algorithm here while the gate read it with another.
from scripts.evaluate_wind_retrain import ALGORITHMS as WIND_ALGORITHMS  # noqa: E402

#: The ship set, with the batch that admitted each row and the tranche it was
#: graded under. The ABL-525 rows' figures live in
#: `reports/abl_444_g23_floor_reread.json` (blob 1e8f37f6, sha256 45fa753f...);
#: the ABL-580 rows' are ABL-426's registered-source read for CZ/RO solar and
#: ledger 14.3 for NL `wind_offshore`; the ABL-583 row's is ABL-581's read,
#: `experiments/ABL348/results_abl581_ch_solar_f27.json`. Nothing here re-derives
#: any of them -- this script fits and records; it scores nothing. `hold` marks a
#: pair this run refuses to fit, with the reason, so the committed record carries
#: the absence. Exactly one row is held today: `HU` `wind_onshore`, withdrawn by
#: CEO ruling 2026-08-28 *after* it had been fitted, which is why the hold and
#: not a deletion -- the artifact exists on disk and the table is what says it
#: must not be served. CH is the worked example of the same machinery in both
#: directions.
SHIP_SET = (
    {"country": "EE", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2e", "hold": None},
    {"country": "GR", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2b", "hold": None},
    {"country": "SE", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2b", "hold": None},
    {"country": "BG", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2f", "hold": None},
    {"country": "CZ", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2e", "hold": None},
    {"country": "FI", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2b", "hold": None},
    {"country": "LT", "forecast_type": "wind_onshore", "batch": "abl525",
     "tranche": "2e", "hold": None},
    # ABL-580: the three pairs the standing rule admitted on 2026-08-27. None of
    # them pins a feature list -- CZ/RO solar because the ABL-426 record's 27
    # names are the current list (module docstring), NL wind_offshore because
    # `wind_retrain.FEATURE_COLUMNS` has moved in exactly one commit, `601f10f`
    # (2026-08-11), its introduction, which predates the 2026-08-14 offshore
    # re-read; and the wind harness applies that one module constant at all
    # three of its fit/predict sites with no per-type branch, so `wind_offshore`
    # takes the same 24 names as `wind_onshore` by construction.
    {"country": "CZ", "forecast_type": "solar", "batch": "abl580",
     "tranche": "2a (ABL-426 re-read)", "hold": None},
    {"country": "RO", "forecast_type": "solar", "batch": "abl580",
     "tranche": "2a (ABL-426 re-read)", "hold": None},
    {"country": "NL", "forecast_type": "wind_offshore", "batch": "abl580",
     "tranche": "pilot (abl322-pilot, ABL-436 re-read)", "hold": None},
    # ABL-583: the pair the standing rule readmitted on 2026-08-27. It pins no
    # feature list, and that is the whole content of the readmission -- the
    # condition that withdrew it was a pin (tranche 1b at the legacy 25) against
    # a builder that had moved to 27, and ABL-581 re-read it at the 27 the
    # builder produces today. See the module docstring.
    {
        "country": "CH",
        "forecast_type": "solar",
        "batch": "abl583",
        "tranche": "abl581-ch-solar-f27 (fresh pre-registered read at 27)",
        "hold": None,
        # The row's own disposition history, so the record carries both moves
        # rather than presenting a readmitted pair as one that never left.
        "admission_history": (
            "Pair 8 of the Board's ship8 roster (2026-08-22), graded under "
            "abl316-t1b at the legacy 25-name solar list. WITHDRAWN by CEO "
            "ruling 2026-08-27: ABL-395 had moved solar.FEATURE_COLUMNS to 27 "
            "(adds sun_elevation_deg, is_night) while SCOPE_FEATURES "
            "['abl316-t1b'] pins the tranche to the legacy 25, so fitting at 27 "
            "would ship a model nobody graded and fitting at 25 would be a "
            "per-country serving fork on a list the builder no longer produces. "
            "READMITTED 2026-08-27 under the standing rule (ABL-316 ledger 14.6 "
            "/ 15.1, no new Board card) on ABL-581's fresh read under the new "
            "scope abl581-ch-solar-f27: registered at 82e3108, read at 49ab9e9, "
            "PASS 3/3 grade A/A/A on the registered energy_generation. The "
            "readmission is the CEO's disposition; nothing here re-derives it."
        ),
    },
    # ABL-602: the five pairs the Board's `widen7` answer (2026-08-28) admits
    # once the causally-available standard is applied to the committed records.
    # Four of them ship -- HU was fitted and then withdrawn the same day; see
    # its `hold` and the "HU IS WITHDRAWN" section of the module docstring.
    # None pins a feature list, and for each type the reason is checked rather
    # than assumed:
    #
    #   solar (LT, SE)  tranche 2d records `meta.n_features: 27` and a
    #                   `meta.feature_columns` element-for-element identical to
    #                   `tests/feature_list_manifest.json -> gate_harness.solar
    #                   .columns` and to today's `solar_retrain.FEATURE_COLUMNS`.
    #                   The manifest's frozen note names only `abl253` and
    #                   `abl316-t1b` as fitted at the legacy 25, so 2d is not a
    #                   CH-shaped case and needs no ABL-581-style re-read.
    #   wind (HR,HU,PL) `wind_retrain.FEATURE_COLUMNS` is 24 names whose
    #                   comma-joined sha256 is `2a034c79a1cd` at every commit
    #                   that ever touched the file (601f10f 2026-08-11,
    #                   eaab3e3, 75adff8, and origin/main today), so the graded
    #                   list and the builder's list cannot have diverged.
    #                   ABL-395's geometry features are solar-only by design.
    #
    # Both types here are catboost, but `algorithm_for` still resolves per type
    # from the harnesses -- the ABL-580 lesson, which is that a batch whose rows
    # happen to agree with a constant is not a batch that may restate one.
    {"country": "LT", "forecast_type": "solar", "batch": "abl602",
     "tranche": "2d (abl316-t2d)", "hold": None},
    {"country": "SE", "forecast_type": "solar", "batch": "abl602",
     "tranche": "2d (abl316-t2d)", "hold": None,
     "admission_history": (
         "Thinnest pair in the widened set: +11.29% worst-band D-7 skill "
         "against a 10.65% k=1 readability floor, 0.64pp of headroom. It plays "
         "the role CZ wind_onshore plays for the approved set (ABL-316 ledger "
         "15.2) -- if any future correction moves the k=1 floor, SE solar is "
         "the first pair withdrawn."
     )},
    {"country": "HR", "forecast_type": "wind_onshore", "batch": "abl602",
     "tranche": "2e (abl417-tranche2e)", "hold": None,
     "admission_history": (
         "Published letter A under the registered fit_window/sign_test "
         "convention. Under the ABL-437/444 amended defaults "
         "(trailing_28d/floored) the 24-36h band abstains: G2 +2.80% and G3 "
         "+2.02% both sit inside the 7.51% wind floor, so the pair letter is N. "
         "Quote the convention with the letter; see the module docstring."
     )},
    {"country": "HU", "forecast_type": "wind_onshore", "batch": "abl602",
     "tranche": "2e (abl417-tranche2e)",
     "hold": (
         "WITHDRAWN by CEO ruling 2026-08-28, after the artifact had been "
         "fitted. The bar: a pair ships unless a causally-available screen "
         "shows a READABLE loss; an abstention (margin inside the readability "
         "floor) does not block. HU is the only pair in the widened batch on "
         "the wrong side of it -- G2/G3 are negative in 6 of 6 band x "
         "reference cells under BOTH levellings, and under trailing_28d they "
         "are -22.83% to -28.68% against a 7.51% floor, recorded as `failed` "
         "rather than `not_readable` in reports/abl_444_g23_floor_reread.json. "
         "There is no reading of the record under which HU beats a flat line. "
         "This is a disposition, not a re-grade: no registered letter moves "
         "and abl417-tranche2e stays pinned at fit_window/sign_test. The "
         "artifact from the 2026-08-28T12:05:25Z run exists on disk and MUST "
         "NOT be deployed; reports/abl_602_ship_disposition.json is the "
         "machine-readable list of what does."
     ),
     "admission_history": (
         "Grade B under the registered convention AND under the amended "
         "defaults -- not an abstention under either endpoint. G1 holds "
         "(+15.13% vs D-7 at the worst band) and G4 holds, but G2/G3 are a "
         "signed loss to constant_causal and climatology_causal at -1.75% / "
         "-2.13% under fit-window levelling, and -26.78% / -25.51% under "
         "trailing-28d levelling, where they are far outside the 7.51% floor "
         "and therefore READABLE. The N reading in ABL-602's description comes "
         "from fit_window/floored, which applies ABL-444 without ABL-437. "
         "ADMITTED to the batch on the Board's widen7 answer and fitted "
         "2026-08-28T12:05:25Z; WITHDRAWN by CEO ruling the same day on that "
         "readable G2/G3 loss -- the same causally-available argument that "
         "excluded NO and RO on G4. See `hold`; the row records both moves."
     )},
    {"country": "PL", "forecast_type": "wind_onshore", "batch": "abl602",
     "tranche": "2b (abl406-tranche2b, graded by the ABL-418 retro ladder)",
     "hold": None,
     "admission_history": (
         "The only pair in this batch that holds grade A under all four "
         "levelling/readability conventions ABL-444 reports. Its stored tranche "
         "record predates the G1-G4 ladder, so the letters come from "
         "reports/abl_418_retro_grade.json."
     )},
)

#: The source table every one of these pairs was graded on (ABL-321/ABL-348).
RENEWABLE_SOURCE = "energy_generation"

#: Widest window the serve-faithful builder can actually populate; see the module
#: docstring. End is exclusive, as `build_vintage_frame` uses a left-closed range.
FIT_START = "2026-01-11"
FIT_END = "2026-08-22"

#: The builder needs actuals before `FIT_START` for the 14-day point lag and the
#: 168-hour rolling anchors. Same value the gate harnesses use.
LOOKBACK_DAYS = 14

FEATURE_COLUMNS_BY_TYPE = {
    "wind_onshore": WIND_FEATURE_COLUMNS,
    "wind_offshore": WIND_FEATURE_COLUMNS,
    "solar": SOLAR_FEATURE_COLUMNS,
}

#: The estimator each type's gate harness fitted, taken from the harnesses
#: themselves. `wind_offshore` is xgboost and `wind_onshore` is catboost -- one
#: table, two values, and this script gets both from the code that graded them
#: rather than from a constant of its own. See the module docstring; the wind
#: pair comes in as a dict, so `**` rather than two rows, and any later move in
#: `evaluate_wind_retrain.ALGORITHMS` arrives here without an edit.
ALGORITHM_BY_TYPE = {**WIND_ALGORITHMS, "solar": SOLAR_ALGORITHM}


def algorithm_for(forecast_type):
    """The estimator this type's gate harness fitted.

    ABL-525's rows all resolve to catboost, so this replaces the old module
    constant without changing what the seven were fitted with -- asserted rather
    than asserted-by-comment: `tests/test_abl580_ship_set_batches.py` holds it.
    """
    return ALGORITHM_BY_TYPE[forecast_type]


def columns_for(country, forecast_type):
    """The feature list this pair was graded on.

    Defaults to the type's current gate list, but a SHIP_SET row may pin its
    own -- which is not a per-country fork of the *builder*, the thing ABL-525
    item 2 forbids. It is the opposite: a pin to the list that pair's tranche
    was read on, so the artifact matches the approval instead of drifting with
    a constant that moved afterwards.
    """
    for entry in SHIP_SET:
        if entry["country"] == country and entry["forecast_type"] == forecast_type:
            pinned = entry.get("feature_columns")
            if pinned:
                return tuple(pinned)
            break
    return FEATURE_COLUMNS_BY_TYPE[forecast_type]


def build_model(algorithm):
    """The estimator and the exact params it is fitted with.

    Mirrors `evaluate_wind_retrain._model`: the production xgboost defaults carry
    an early-stopping setting that needs a validation set, and this fit uses every
    row with no tuning, so it is removed rather than left to fail.
    """
    params = config.get_default_params(algorithm)
    if algorithm == "xgboost":
        params.pop("early_stopping_rounds", None)
        return XGBRegressor(**params), params
    return CatBoostRegressor(**params), params


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_one(country, forecast_type, replica_db, models_dir, algorithm=None):
    """Fit one pair and write its artifact. Returns the provenance record.

    `algorithm` defaults to the type's graded estimator rather than to a module
    constant, so `abl525_repro_check.py` -- which calls this with four positional
    arguments and no algorithm -- refits every pair with what that pair was
    fitted with, including an offshore pair the original constant would have
    silently refitted as catboost and then reported as a prediction difference.
    """
    algorithm = algorithm or algorithm_for(forecast_type)
    columns = columns_for(country, forecast_type)
    fit_start = pd.Timestamp(FIT_START)
    fit_end = pd.Timestamp(FIT_END)

    t0 = time.perf_counter()
    builder = RenewableFeatureBuilder(
        country,
        forecast_type,
        fit_start - pd.Timedelta(days=LOOKBACK_DAYS),
        fit_end,
        actuals_source=RENEWABLE_SOURCE,
        db_path=str(replica_db),
    )
    frame = build_vintage_frame(builder, fit_start, fit_end, feature_columns=columns)
    fit, audit = finite_training_rows(frame, feature_columns=columns)
    t_build = time.perf_counter() - t0

    if fit.empty:
        raise RuntimeError(
            f"{country}/{forecast_type}: no finite training rows in "
            f"{FIT_START}..{FIT_END} -- refusing to write an artifact"
        )

    model, params = build_model(algorithm)
    t0 = time.perf_counter()
    model.fit(fit[list(columns)], fit["actual"])
    t_fit = time.perf_counter() - t0

    path = save_gate_artifact(
        Path(models_dir) / country / forecast_type / "model.joblib",
        model=model,
        builder=builder,
        algorithm=algorithm,
        params=params,
        feature_columns=columns,
        fit_window=(fit_start, fit_end),
    )

    retained_targets = pd.DatetimeIndex(fit["target_ts"])
    predictions = np.asarray(model.predict(fit[list(columns)]), dtype=float)

    return {
        "country": country,
        "forecast_type": forecast_type,
        "algorithm": algorithm,
        "training_source": builder.actuals_source,
        "fit_window_requested": [FIT_START, FIT_END],
        "fit_window_retained": [
            str(retained_targets.min()),
            str(retained_targets.max()),
        ],
        "n_features": len(columns),
        "feature_columns": list(columns),
        "intended_rows": audit["intended_rows"],
        "retained_rows": audit["retained_rows"],
        "excluded_missing_actual_or_feature": audit["excluded_missing_actual_or_feature"],
        "unique_fit_targets": audit["unique_targets"],
        "degraded_lag_1d_rows": audit["degraded_lag_1d_rows"],
        "hyperparams": params,
        "artifact_path": str(path),
        "artifact_sha256": sha256_of(path),
        "seconds_feature_build": round(t_build, 2),
        "seconds_fit": round(t_fit, 2),
        # ABL-525 item 7: an artifact sha256 cannot witness a refit, because
        # `Forecaster.save` stamps `saved_at`. Predictions can.
        "in_sample_prediction_digest": hashlib.sha256(
            predictions.tobytes()
        ).hexdigest(),
        "in_sample_prediction_mean": float(predictions.mean()),
    }


#: Where each batch's committed machine record lands. A batch that names no
#: record here has to be given `--json-out` explicitly rather than silently
#: overwriting another batch's; `main` refuses otherwise. None of these is an
#: `experiments/*/results.json` path -- that glob is gitignored (the ABL-440
#: trap, still open), and a record that cannot be diffed cannot be evidence.
BATCH_RECORDS = {
    "abl525": "reports/abl_525_ship_set_training.json",
    "abl580": "reports/abl_580_ship_set_training.json",
    "abl583": "reports/abl_583_ship_set_training.json",
    "abl602": "reports/abl_602_ship_set_training.json",
}

#: The issue each batch's record is filed under. Kept beside `BATCH_RECORDS` so
#: adding a batch is one place, not two -- the old inline `.get` on a literal
#: dict was a second table waiting to disagree with the first.
BATCH_ISSUES = {"abl525": "ABL-525", "abl580": "ABL-580", "abl583": "ABL-583",
                "abl602": "ABL-602"}

BATCHES = tuple(dict.fromkeys(entry["batch"] for entry in SHIP_SET))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit production artifacts for the Board-approved ABL-316 ship set "
            "through the graded gate-harness path (ABL-525, ABL-580)."
        )
    )
    parser.add_argument(
        "--replica-db",
        default=config.DATABASE_PATH,
        help="Read-only replica to fit from (default: ENERGY_DB_PATH).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(config.MODELS_DIR),
        help="Artifact root; a pair lands at <root>/<country>/<type>/model.joblib.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Committed machine record. Defaults to the record registered for "
            "--batch. Not an experiments/*/results.json path -- that glob is "
            "gitignored (the ABL-440 trap)."
        ),
    )
    parser.add_argument(
        "--batch",
        default=None,
        choices=BATCHES,
        help=(
            "Restrict the run to one admission batch. Omitted, the run covers "
            "every batch and needs an explicit --json-out, so a full-set refit "
            "cannot land on one batch's record."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated COUNTRY/TYPE to restrict the run, e.g. CZ/wind_onshore.",
    )
    parser.add_argument(
        "--include-held",
        action="store_true",
        help="Fit pairs marked hold in SHIP_SET. Off by default and deliberately.",
    )
    args = parser.parse_args()

    json_out = args.json_out or BATCH_RECORDS.get(args.batch)
    if json_out is None:
        parser.error(
            "--json-out is required without a --batch whose record is "
            f"registered in BATCH_RECORDS ({', '.join(sorted(BATCH_RECORDS))})"
        )

    replica = Path(args.replica_db)
    if not replica.is_file():
        raise SystemExit(f"replica not found: {replica}")

    only = None
    if args.only:
        only = {item.strip() for item in args.only.split(",") if item.strip()}

    records, held = [], []
    for entry in SHIP_SET:
        key = f"{entry['country']}/{entry['forecast_type']}"
        if args.batch is not None and entry["batch"] != args.batch:
            continue
        if only is not None and key not in only:
            continue
        if entry["hold"] and not args.include_held:
            held.append({**entry, "status": "held"})
            print(f"[HOLD] {key}: {entry['hold']}")
            continue
        print(f"[FIT ] {key} ({entry['tranche']}) ...", flush=True)
        record = fit_one(
            entry["country"], entry["forecast_type"], replica, args.models_dir
        )
        record["batch"] = entry["batch"]
        record["tranche"] = entry["tranche"]
        if entry.get("admission_history"):
            record["admission_history"] = entry["admission_history"]
        records.append(record)
        print(
            f"[OK  ] {key}: {record['retained_rows']}/{record['intended_rows']} rows, "
            f"{record['unique_fit_targets']} targets, {record['n_features']} features, "
            f"{record['algorithm']}, "
            f"build {record['seconds_feature_build']}s fit {record['seconds_fit']}s",
            flush=True,
        )

    payload = {
        "issue": BATCH_ISSUES.get(args.batch, "ABL-316"),
        "batch": args.batch,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "board_decision": "abl316:ship-decision:v4 = ship8, 2026-08-22T08:25Z",
        "admission_rule": (
            "ABL-316 ledger 14.6 / 15.1, approved by the Board alongside ship8: "
            "a held pair that later satisfies the same rule joins the shipping "
            "set without a new Board card; a shipping pair a later correction "
            "moves outside it is withdrawn. Both are reported by comment on "
            "ABL-316. Membership is the CEO's; this script fits what membership "
            "names and scores nothing."
        ),
        "evidence_of_record": {
            "abl525": {
                "path": "reports/abl_444_g23_floor_reread.json",
                "blob": "1e8f37f6b1c4befd2c938363306e664ea58a21e7",
                "sha256_prefix": "45fa753fc356b123",
                "note": "Verified byte-unchanged since the a0b9ffd pin. "
                        "Not re-derived here.",
            },
            "abl580": {
                "cz_solar": "experiments/ABL348/results_abl426_tranche2a_generation.json",
                "ro_solar": "experiments/ABL348/results_abl426_tranche2a_generation.json",
                "nl_wind_offshore": "ABL-316 ledger 14.3; the pilot's own read is "
                                    "experiments/ABL322/results_abl436_offshore_reread.json",
                "note": "Grades and margins are the CEO's disposition on ABL-316. "
                        "Nothing here re-derives or re-grades them.",
            },
            "abl583": {
                "ch_solar": "experiments/ABL348/results_abl581_ch_solar_f27.json",
                "scope": "abl581-ch-solar-f27, registered 82e3108, read 49ab9e9, "
                         "merged as PR #89 (dda64fc3)",
                "note": "PASS 3/3, grade A/A/A. The read is ABL-581's and the "
                        "readmission is the CEO's; nothing here re-derives "
                        "either. Note that this fit window COVERS that gate "
                        "window, so those figures are not out-of-sample for "
                        "this artifact -- see fitted_on_the_gate_window.",
            },
            "abl602": {
                "lt_solar": "experiments/ABL348/results_abl421_tranche2d.json",
                "se_solar": "experiments/ABL348/results_abl421_tranche2d.json",
                "hr_wind_onshore": "experiments/ABL348/results_abl417_tranche2e.json",
                "hu_wind_onshore": "experiments/ABL348/results_abl417_tranche2e.json",
                "pl_wind_onshore": "reports/abl_418_retro_grade.json "
                                   "(tranche 2b; the stored record "
                                   "experiments/ABL348/results_abl406_tranche2b.json "
                                   "predates the G1-G4 ladder)",
                "levelling_conventions": "reports/abl_444_g23_floor_reread.json",
                "board_answer": "abl316 widen7, 2026-08-28, adopting the "
                                "causally-available standard for the widened set",
                "disposition": "FIVE FITTED, FOUR SHIP. HU wind_onshore was "
                               "withdrawn by CEO ruling 2026-08-28 after the "
                               "12:05:25Z fit, on a readable G2/G3 loss under "
                               "the amended levelling -- the same "
                               "causally-available argument that excluded NO "
                               "and RO on G4. Its SHIP_SET row now carries a "
                               "hold, so a re-run of this script fits four. "
                               "The shipping four and their digests are in "
                               "reports/abl_602_ship_disposition.json.",
                "note": "Read at origin/main 79433d0b. Registered convention for "
                        "all three scopes is fit_window levelling with a "
                        "sign_test G2/G3 (CAUSAL_LEVELLING / G23_READABILITY "
                        "pin every published scope by value). Under the "
                        "ABL-437/444 amended defaults three of the five "
                        "worst-band letters move between the endpoints -- LT "
                        "A->N, SE A->N, HR A->N -- while HU is B->B and PL is "
                        "A->A; a fourth move, HU B->N, appears only under the "
                        "mixed fit_window/floored convention, which is why "
                        "only PL holds grade A under all four. NO and RO "
                        "wind_onshore are NOT in this batch -- both fail G4, "
                        "which is causally available, so the standard the "
                        "Board adopted excludes them. Grades and membership "
                        "are the CEO's disposition; nothing here re-derives or "
                        "re-grades them.",
            },
        },
        "protocol": {
            "fit_path": "src.wind_features.RenewableFeatureBuilder + "
                        "src.evaluation.wind_retrain.build_vintage_frame",
            "artifact_writer": "src.evaluation.gate_artifacts.save_gate_artifact "
                               "(-> Forecaster.save)",
            "renewable_source": RENEWABLE_SOURCE,
            "vintages_per_target": 8,
            "lookback_days": LOOKBACK_DAYS,
            "algorithm_by_type": dict(ALGORITHM_BY_TYPE),
            "algorithm_source": (
                "evaluate_wind_retrain.ALGORITHMS and solar_retrain.ALGORITHM, "
                "imported not restated"
            ),
            "scored_or_graded": False,
            "fitted_on_the_gate_window": True,
            "fit_window_bounded_by": (
                "weather_data data_quality='forecast' begins 2026-01-11 for every "
                "country in this set; energy_generation reaches 2021-01-01 but a "
                "serve-faithful row cannot be built without the weather archive"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "replica_db": str(replica),
            "replica_bytes": replica.stat().st_size,
            "models_dir": str(args.models_dir),
        },
        "pairs": records,
        "held": held,
    }

    out = Path(json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out} ({len(records)} fitted, {len(held)} held)")


if __name__ == "__main__":
    main()
