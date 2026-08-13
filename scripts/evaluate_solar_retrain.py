#!/usr/bin/env python3
"""Fit and read a pre-registered serve-faithful solar gate (see SCOPES).

The default scope is `abl253`, so an unflagged run reproduces that gate exactly;
`--scope` selects any other registered country set, and carries its own output
paths with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.data_quality import find_suspect_constant_runs
from src.evaluation.gate_grading import (
    attach_grades, cell_grade, grade_summary_table, grading_prose,
)
from src.evaluation.model_free_reference import (
    MODEL_FREE_COMPARATORS, attach_model_free_references, comparator_wape,
    levels_table, lost_to_a_model_free_reference, reference_prose,
)
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.gate_registration import (
    check_registration_tables, check_scope_outputs,
)
from src.evaluation.scorecard import (
    ScorecardConfig, _load_forecasts, _load_tso, _ro_connect,
    describe_opened_databases, opened_databases,
    score_predictions, select_latest_per_band,
)
from src.evaluation.solar_retrain import (
    ALGORITHM, FEATURE_COLUMNS, INTENDED_N, PRIMARY_BANDS, SCHEDULE_N,
    attach_baselines, build_vintage_frame, finite_training_rows, gate_cell,
    scored_with_comparators, select_latest_challenger_per_band,
)
from src.solar_features import (
    IMPOSSIBLE_NIGHT_THRESHOLD_MW, NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_GEOMETRY_FEATURES, exclude_impossible_night_rows,
)
from src.wind_features import RenewableFeatureBuilder


# ABL-378, porting the ABL-322 fix from the wind harness.  This harness used to
# read `COUNTRIES` directly and hardcode its pass bar to the 9 cells that set
# produces, so it could not be pointed at any of the 28 solar countries with no
# model without editing the constant -- at which point `== 9` silently became
# the wrong denominator.  Each scope now names its countries outright and the
# registered cell count is derived from *that table*, fixed in the file before
# the run, rather than from whatever the run turns out to score.  That keeps the
# property the literal existed to protect: a country that silently yields no
# gate rows still falls short of its scope's count instead of quietly leaving
# the denominator.  Adding a scope is a pre-registration and belongs in review.
#
# The country tuple is written out here rather than referencing
# `solar_retrain.COUNTRIES`, deliberately: a registration must not follow a
# shared mutable constant.  Adding AT to `COUNTRIES` -- it is the one other
# country with a solar incumbent -- would otherwise silently re-scope a
# dispositioned gate.  `test_gate_scope_registration` pins the two as equal
# today, so any divergence surfaces in review instead of applying itself.
SCOPES = {
    # ABL-253 as registered: BE/DE/FR solar.  3 countries x 3 primary bands = 9
    # cells.  Unchanged, and the default, so an unflagged run still reproduces
    # ABL-253 exactly.
    "abl253": ("BE", "DE", "FR"),
    # ABL-381 -- ABL-316 tranche 1b: BG and CH solar, fitted on
    # `energy_generation` under the frozen registration at
    # `experiments/ABL348/config.json`.  2 countries x 3 bands = 6 cells.
    #
    # This entry *is* the tranche's pre-registration, in the sense the comment
    # above describes: the country list is fixed here, in the file, and committed
    # before the first fit, so the cell bar cannot follow what the run turned out
    # to score.  Windows, metric, baseline, minimum n and source table are
    # ABL-348's and are deliberately not restated here -- thirty-seven tranches
    # must not become thirty-seven chances to shop a window.  The solar
    # counterpart of `abl380-tranche1a` on the wind harness.
    #
    # Neither country serves a solar model: measured on the live replica
    # (9,432,453,120 bytes, mode=ro) on 2026-08-13, `forecasts` holds solar rows
    # for BE/DE/FR/AT only, and zero for both BG and CH.  So this scope refits no
    # live pair -- the property `abl253` protects, reached by the same route as
    # `abl322-pilot` on the wind side.
    "abl316-t1b": ("BG", "CH"),
    # ABL-405 -- ABL-316 tranche 2a: eight continental solar countries on
    # `energy_generation`, under the same frozen registration at
    # `experiments/ABL348/config.json`.  8 countries x 3 bands = 24 cells.
    #
    # Six are new coverage (CZ, HU, PL, RO, SI, SK).  BG and CH are a deliberate
    # **re-read at 27 features**, on the CEO decision recorded on ABL-401: the
    # published `abl316-t1b` cells were fitted at 25, and folding the two pairs in
    # here costs two extra fits and no extra run while leaving the solar coverage
    # table homogeneous at 27 rather than 17-at-27 plus 2-at-25.  It is a *new
    # scope* precisely so that re-read cannot land on ABL-381's evidence: this
    # entry's `SCOPE_OUTPUTS` row writes nowhere ABL-381 wrote, and
    # `experiments/ABL348/results_abl381_tranche1b.json` and `reports/abl_381_*`
    # are byte-unchanged by any run of this scope.  Re-basing `abl316-t1b` in
    # place would have been the ABL-387 failure mode with a feature list in place
    # of a path -- and is separately live as ABL-404, since that scope still holds
    # no `SCOPE_FEATURES` row and so resolves to 27 against its own 25-feature
    # published read.  Nothing here fixes that; this scope simply does not rely on
    # it.
    #
    # The eight are grouped by the pre-committed D-7 bar (18.35-26.11%, plus CH at
    # 12.67%) so the tranche's pass rate reads as one number.  The Mediterranean
    # tight-bar set (ES/GR/HR/IT/PT) and the northern low-level set
    # (EE/FI/LT/LV/NL/SE) are deliberately *not* here; they are separate tranches
    # with separate readings, and pulling them in would average two different
    # tasks into one rate.
    #
    # Windows, bands, metric, baseline, minimum n and source table are ABL-348's
    # and are deliberately not restated here -- thirty-three remaining tranches
    # must not become thirty-three chances to shop a window.
    #
    # No country here serves a solar model: measured on the live replica
    # (9,432,453,120 bytes, mode=ro) on 2026-08-13, `forecasts` holds solar rows
    # for BE/DE/FR/AT only and zero for all eight.  So this scope refits no live
    # pair, the property `abl253` protects.
    "abl316-t2a": ("BG", "CH", "CZ", "HU", "PL", "RO", "SI", "SK"),
    # ABL-376: ABL-253's countries, window and basis with exactly one thing
    # changed -- the fit drops night rows the sun says cannot exist (`FIT_RULES`
    # below).  Registered as its own scope rather than as a flag on `abl253`
    # because it is not the same protocol: re-reading `abl253` under a different
    # fit rule would move numbers already dispositioned and leave no record of
    # which rule produced which read.  Same countries, same basis, same windows,
    # so the pair is a controlled A/B on the rule alone.
    "abl376": ("BE", "DE", "FR"),
}

# ABL-387: where a scope writes is part of its registration, not a flag default.
# These three paths used to be fixed strings on the arguments themselves --
# ABL-253's -- resolved before `--scope` was ever consulted, so a scoped run that
# omitted three flags overwrote a *dispositioned* gate read in place.  That run
# succeeds and emits a full report; the damage is to evidence.
#
# Entries stay exactly one directory deep under `experiments/`, because
# `.gitignore:53-56` globs `experiments/*/results.json` and
# `experiments/*/artifacts/`.  A nested path slips past both globs and would
# commit a binary model artifact.
#
# Depth is necessary but not sufficient, and the two globs do not key on the
# same thing: `experiments/*/artifacts/` matches on the *directory name*, so any
# entry ending `artifacts` one level deep is ignored, while
# `experiments/*/results.json` matches on the *exact filename*.  A `json_out`
# named anything else is therefore tracked.  ABL-380 took that second option on
# the wind twin deliberately, and a solar tranche registering here should decide
# the same question rather than inherit `results.json` by imitation: an ignored
# machine record is one no reviewer can diff.
SCOPE_OUTPUTS = {
    # Byte-for-byte the paths ABL-253 was read at, so an unflagged run still
    # reproduces the dispositioned read rather than relocating it.
    "abl253": {"artifact_dir": "experiments/ABL253/artifacts",
               "json_out": "experiments/ABL253/results.json",
               "report_out": "reports/abl_253_solar_retrain.md"},
    # ABL-381, registered on merging ABL-387 in -- the same sequence the wind
    # twin records, and the same check working: `abl316-t1b` landed in `SCOPES`
    # and `GATE_BASIS` at 776bfe7, before `SCOPE_OUTPUTS` existed, and merging
    # ABL-387 raised `SCOPE_OUTPUTS is missing 'abl316-t1b'` at import.  The
    # merge itself was textually clean, so the tables disagreeing is the only
    # thing that could have caught it.
    #
    # These are the paths the run *actually wrote*, measured rather than
    # assigned.  The `json_out` sits under ABL348 because the registration these
    # fits are read under is `experiments/ABL348/config.json`, frozen at ABL-348
    # and shared with the wind tranche; the artifacts sit under ABL316 because
    # they belong to that rollout, not to the registration.
    #
    # The first read wrote one level deeper, at `experiments/ABL316/artifacts/t1b`,
    # to keep the 33 remaining tranches from sharing a directory;
    # `test_experiment_outputs_stay_one_directory_deep` rejects that, so those
    # files were moved here rather than loosening another issue's guard.  The
    # ABL-389 re-read refits into this registered path directly, so the `t1b`
    # layout and the write-time-path caveat that went with it are both gone.
    #
    # A refit is safe to repeat here and is *not* a second look at the result:
    # `random_seed` is fixed at 42 in `config.py` and the fit is deterministic,
    # so the re-read reproduced challenger and D-7 WAPE to 1e-12 in all six
    # cells.  That equality -- not the artifact SHA-256 -- is what witnesses it.
    # `Forecaster.save` stamps `"saved_at": datetime.now().isoformat()` into
    # every bundle, so two bit-identical models are *guaranteed* different
    # hashes; the SHA in `training[].artifact_sha256` identifies a file, and
    # reading a changed one as a changed model is the wrong inference.  ABL-375's
    # 4.6-13.8% cross-seed spread is the reason this only holds while the seed
    # is pinned.
    #
    # Depth here is a proxy for the property that actually matters, and on this
    # path the proxy and the property disagreed: `.gitignore:56`
    # `experiments/*/artifacts/` matches on the *directory name*, so
    # `experiments/ABL316/artifacts/` is ignored and everything beneath it is
    # too -- `git check-ignore -v` confirmed the `t1b/` layout was ignored by
    # that same line.  Conforming anyway rather than loosening another issue's
    # freshly-landed guard; ABL-381's evidence pack reports the distinction, as
    # it decides whether the remaining 33 tranches can group per tranche.
    #
    # The `json_out` is deliberately not named `results.json`: at that name the
    # `.gitignore` glob would swallow it, and it is the machine record this
    # tranche's evidence cites for a PASS the Board has been asked to review.
    "abl316-t1b": {"artifact_dir": "experiments/ABL316/artifacts",
                   "json_out": "experiments/ABL348/results_abl381_tranche1b.json",
                   "report_out": "reports/abl_381_solar_tranche1b.md"},
    # ABL-405.  Every path is disjoint from `abl316-t1b`'s above, and that is the
    # entire mechanism protecting ABL-381's dispositioned evidence from this
    # tranche's BG/CH re-read: same two countries, same windows, same basis, a
    # different feature vector, and therefore a different challenger whose numbers
    # must not land on the published ones.  `abl316-t1b`'s three paths are
    # byte-unchanged by any run of this scope, which is the ABL-387 property
    # stated over a feature list instead of a flag default.
    #
    # `artifact_dir` is `experiments/ABL405/artifacts` and not
    # `experiments/ABL316/artifacts`: the latter is `abl316-t1b`'s, and BG and CH
    # appear in both scopes, so sharing it would have this run overwrite
    # `experiments/ABL316/artifacts/BG/solar/model.joblib` -- the 25-feature
    # artifact whose SHA-256 ABL-381's machine record cites -- with a 27-feature
    # one.  Nothing in `git status` would show it: `.gitignore:56`
    # (`experiments/*/artifacts/`) matches on the *directory name*, so both are
    # ignored.  One directory deep, so that glob still takes it and no binary
    # artifact becomes committable.
    #
    # `json_out` takes the tracked form and sits under ABL348 for `abl316-t1b`'s
    # reason: the registration these fits are read under is
    # `experiments/ABL348/config.json`.  One level deep and *not* named
    # `results.json`, so `.gitignore:53` -- which matches on the exact filename --
    # does not swallow it.  That matters here more than usual: this is the machine
    # record behind a 24-cell read and behind the 27-vs-25 delta that decides
    # whether `abl253` is ever re-read, and an ignored `results.json` is the one
    # gate record a reviewer cannot diff.
    "abl316-t2a": {"artifact_dir": "experiments/ABL405/artifacts",
                   "json_out": "experiments/ABL348/results_abl405_tranche2a.json",
                   "report_out": "reports/abl_405_solar_tranche2a.md"},
    # ABL-376 takes the tracked form the section above recommends, and for the
    # reason given there: this read is meant to be dispositioned against
    # `abl253`, and a `results.json` is the one gate record `git checkout --`
    # cannot recover and a reviewer cannot diff.  `results_abl376_night_fit.json`
    # is one level deep but is not named `results.json`, so `.gitignore:53` --
    # which matches on the exact filename -- does not take it; `artifacts` one
    # level deep still is taken, by `.gitignore:56`, which matches on the
    # directory name.
    "abl376": {"artifact_dir": "experiments/ABL376/artifacts",
               "json_out": "experiments/ABL376/results_abl376_night_fit.json",
               "report_out": "reports/abl_376_solar_night_fit.md"},
}

# The columns that must be *simultaneously finite* for a row to enter a gate
# cell.  This is a registered property of the scope, not a detail.
# `common_scores` keeps only rows where every named column is finite, and the
# incumbent is left-merged, so it is NaN on every row for a country with no rows
# in `forecasts`.  Naming `incumbent` in the basis therefore empties the
# intersection for such a country: n=0, all scores None, and the verdict below
# would render that as FAIL -- a model-quality disposition for a race that was
# never run.  That is how the ABL-322 pilot first reported, and it would land
# the same way on all 28 solar countries in ABL-316 that have no incumbent.
#
# ABL-253 keeps the four-way basis it was actually read under, so re-reading it
# does not silently move already-dispositioned numbers.  A new scope registers
# the basis its own pre-registration names.
GATE_BASIS = {
    "abl253": ("challenger", "incumbent", "seasonal_naive", "persistence"),
    # ABL-381: BG and CH hold zero solar rows in `forecasts` -- verified against
    # the live replica, not assumed -- which is the normal condition of all 37
    # remaining ABL-316 pairs rather than a fault of these two.  Under the
    # four-way basis every one of the 6 cells would intersect to n=0 and the run
    # would render UNREADABLE, having compared nothing.  Gates on the two columns
    # the registered bar names; the incumbent is still reported, on its own
    # intersection, where it reads "Not measured" by construction rather than by
    # omission -- which is what ABL-348's `incumbent` field already anticipated.
    "abl316-t1b": ("challenger", "seasonal_naive"),
    # ABL-405: identical to `abl316-t1b`'s above, and for the identical reason --
    # all eight countries hold zero solar rows in `forecasts`, so under the
    # four-way basis every one of the 24 cells would intersect to n=0 and the run
    # would render UNREADABLE, having compared nothing.  Keeping it identical is
    # also what makes the BG/CH re-read a controlled comparison against ABL-381 on
    # the feature vector alone: moving the basis at the same time would confound
    # the two, which is the argument `abl376`'s own entry makes about the fit rule.
    # The incumbent is still reported on its own intersection, where it reads
    # "Not measured" by construction rather than by omission.
    "abl316-t2a": ("challenger", "seasonal_naive"),
    # Deliberately identical to `abl253`'s.  The A/B is on the fit rule; moving
    # the basis at the same time would confound the two.
    "abl376": ("challenger", "incumbent", "seasonal_naive", "persistence"),
}
#: Always reported, each on its own intersection with the gate basis, so that a
#: comparator which never exists reads "Not measured" instead of voiding the gate.
#:
#: ABL-389 adds four model-free predictors, identically to the wind harness and
#: from the same module, so the two gates cannot drift into computing the same
#: named reference differently.  They are *reported references and not gate
#: criteria*: deliberately absent from every `GATE_BASIS` entry above, pinned by
#: `test_gate_model_free_reference.py`.  A pair that clears its D-7 bar while
#: losing to one still reads PASS — beside the number that qualifies it.
#:
#: The climatology columns matter most on this harness.  Measured on ABL-381's
#: six solar cells, the flat line scores 63-95% WAPE on every one of them: a
#: constant cannot represent a diurnal cycle, and on solar the diurnal cycle is
#: the signal, so the constant alone would certify a PASS against a comparator
#: it cannot lose to.  The hour-of-day form is what carries information here --
#: CH's challenger at 8.16% beats a hindsight climatology at 9.02% by 0.86pp,
#: which is the actual worth of that PASS.
#:
#: See `src/evaluation/model_free_reference.py` for both measurements, and for
#: why moving the bar instead would have been wrong.
REPORTED_COMPARATORS = ("challenger", "incumbent", "seasonal_naive", "persistence",
                        *MODEL_FREE_COMPARATORS)

# ABL-418: which of ABL-385's registered per-stream CVs the readability floor is
# taken from.  A property of the harness, not of the scope -- every scope here
# fits solar.  It is deliberately not the wind value: solar's fleet p90 per-fit
# CV is the larger of the two, so its floor is the wider one, and reading a solar
# cell against wind's narrower floor would call an unreadable margin readable.
# ABL-381 quoted its margins against another stream's fits; that is the mistake.
# The numbers themselves live in `src/evaluation/gate_grading.py` and nowhere
# else, so this file cannot come to disagree with the registration it cites.
GRADE_STREAM = "solar"

# ABL-376: what the fit is allowed to *see* is a registered property of the scope
# too, and for the same reason as the basis -- two gate reads are not comparable
# unless both say what they trained on.  `energy_renewable` carries solar for FR
# at sun elevations down to -65 deg; a model fitted through that learns a night
# floor faithfully, which is why the defect is in the training target rather than
# in the model.
#
# The rule is stated over countries, not for FR: the predicate is the sun's, so a
# country whose data is clean loses nothing.  Measured on the replica 2026-08-13
# over the registered fit window, it removes nothing at all for AT and BE.
#
# `exclude_impossible_night` is **fit-side only** wherever it is True.  The gate
# frame is never filtered, and that asymmetry is the point: we refuse to train on
# values the sun says are impossible, and we still score against whatever the
# source reports.  A scope that filtered its own gate would be marking its own
# homework.
#: What a scope gets if it registers no fit rule at all.  Off: the pre-ABL-376
#: behaviour, which is what every already-dispositioned read was taken under.
DEFAULT_FIT_RULES = {"exclude_impossible_night": False}

FIT_RULES = {
    # ABL-253 was read without this rule.  Stated rather than left to the
    # default, because "this read predates the rule" is a fact about the
    # registration and not an absence.
    "abl253": {"exclude_impossible_night": False},
    "abl376": {"exclude_impossible_night": True},
    # ABL-381's tranche 1b was fitted and dispositioned before ABL-376 landed,
    # so it is registered False for the same reason `abl253` is: the read exists
    # and the rule was not in it.  Left explicit rather than resting on
    # `DEFAULT_FIT_RULES`, so nobody has to infer from an absence whether the
    # rule was declined or forgotten.
    #
    # This is the scope ABL-376's rule is most likely to move, and that is a
    # finding rather than a caveat: BG's actuals carry a large overnight floor
    # (`reports/abl_381_tranche1b_findings.md` §5), which is exactly the "values
    # the sun says are impossible" the rule refuses to train on.  Re-fitting
    # BG/CH under the rule is a real experiment and belongs in its own issue with
    # its own scope, not in an edit to this row -- flipping it here would move
    # six dispositioned cells and leave no record of which rule produced which
    # read, which is the confound ABL-376 registered a separate scope to avoid.
    "abl316-t1b": {"exclude_impossible_night": False},
    # ABL-405 registers the rule **off**, which is also what `DEFAULT_FIT_RULES`
    # would have given it -- stated rather than inherited, because this table is
    # one of the three `check_registration_tables` does *not* check, so an absence
    # here is indistinguishable from an oversight and defaults silently.
    #
    # Off is the right value for two independent reasons, not merely the cheap one:
    #
    # - The BG/CH cells in this scope are a controlled A/B against ABL-381 on the
    #   **feature vector alone**.  ABL-381 was read with the rule off; turning it
    #   on here would move the fit frame at the same time as the column list and
    #   make the 27-vs-25 delta unattributable -- the confound ABL-376 registered
    #   a separate scope to avoid.
    # - ABL-348's registration does not contain the rule, and this tranche is read
    #   under it unchanged.
    #
    # BG is the country this rule would most move -- ABL-381 §5 measured 76-85% of
    # its night hours carrying 152-246 MW, which is exactly the "values the sun
    # says are impossible" the rule refuses to train on -- and the six new
    # countries are unscreened for a night floor, since ABL-396 has not landed.
    # Both are reported as findings in this tranche's evidence pack; neither is a
    # reason to edit this row.  Re-fitting any of these eight under the rule is a
    # real experiment and belongs in its own scope.
    "abl316-t2a": {"exclude_impossible_night": False},
}

# ABL-395: and so is the feature *vector*, for exactly the reason stated above
# `FIT_RULES` -- two gate reads are not comparable unless both say what they
# trained on, and a column list is a stronger statement about that than a row
# filter is.
#
# `solar_retrain.FEATURE_COLUMNS` is now 27: the 25 every read from ABL-253 to
# ABL-381 was taken on, plus ABL-338's two adopted daylight-safe geometry
# features, which the harness had never asked the builder for.  That is the right
# default for a *new* tranche and it is what an unregistered scope gets.
#
# It is not right for a scope already dispositioned.  Measured on the two ABL-381
# pairs (`scripts/abl395_geometry_feature_probe.py`, one vintage frame, both arms
# from the same retained rows, seed 42): CH's 24-36h cell moves 8.16% -> 7.78%
# WAPE and BG's 18.89% -> 19.95%.  Those are real movements in cells that have
# been read, so silently re-basing `abl253` or `abl376` would move published
# numbers with nothing in `git status` to show it -- the ABL-387 failure mode
# with a feature list in place of a path.  Both therefore pin the 25 they were
# read on, and whether either is re-read at 27 is ABL-401's decision, not a
# side-effect of this table.
#
# `abl376` pins it for a second reason: its whole registration is a controlled
# A/B against `abl253` on the fit rule alone.  Moving its feature vector at the
# same time would confound the two, which is the argument its own `GATE_BASIS`
# entry already makes.
LEGACY_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS
                               if c not in SOLAR_GEOMETRY_FEATURES)

#: What a scope gets if it registers no feature set: the current list.  A new
#: ABL-316 tranche is fitted at 27 without touching this table.
DEFAULT_SCOPE_FEATURES = FEATURE_COLUMNS

SCOPE_FEATURES = {
    "abl253": LEGACY_FEATURE_COLUMNS,
    "abl376": LEGACY_FEATURE_COLUMNS,
    # ABL-405 (`abl316-t2a`) is deliberately **absent** and takes
    # `DEFAULT_SCOPE_FEATURES` -- the current 27.  Fitting the tranche at 27 was
    # the sole gate on re-tranching the remaining solar pairs, so inheriting the
    # default here is the intended path and not an omission, and this comment is
    # what makes the two distinguishable: this table is not one of the three
    # `check_registration_tables` checks, so an absence defaults silently.  The
    # run records the resolved value either way -- `meta.feature_set`,
    # `meta.n_features` and `meta.feature_set_is_registered_for_scope`, which
    # reads False for this scope and prints as such in the report.
    #
    # Worth stating, because it is the same shape as the live defect next door:
    # once this tranche's read is dispositioned it is in `abl316-t1b`'s position,
    # a published read with no row here, and a later move of `FEATURE_COLUMNS`
    # would re-base it silently.  That is **ABL-404**, which is about the
    # mechanism rather than about any one scope, and pinning a row to
    # `FEATURE_COLUMNS` here would not fix it anyway -- that binds to the same
    # mutable constant `LEGACY_FEATURE_COLUMNS` is derived from.  A real pin is a
    # literal column tuple, and choosing that for every dispositioned scope is
    # ABL-404's call, not a side-effect of this tranche.
}

# The report's H1.  This was the string literal "ABL-253 -- Serve-faithful solar
# retrain gate", which put ABL-253's name on the top line of every other scope's
# report -- a mislabel that survives being copied into an evidence pack, since
# the heading is the first thing quoted and the `scope` field is not.  `abl253`
# keeps its heading character-for-character so the dispositioned read still
# renders identically.
SCOPE_TITLES = {
    "abl253": "ABL-253 — Serve-faithful solar retrain gate",
    "abl376": "ABL-376 — Serve-faithful solar retrain gate, impossible night rows excluded from the fit",
    # Registered rather than left to `title_for`'s derived fallback, which would
    # head this tranche's evidence pack "abl316-t1b".  The scope slug is a key,
    # not a title, and this report is cited in a PASS the Board has been asked to
    # review.
    "abl316-t1b": "ABL-381 — Serve-faithful solar retrain gate, ABL-316 tranche 1b: BG and CH on energy_generation",
    # Registered for `abl316-t1b`'s reason -- `title_for`'s fallback would head a
    # 24-cell evidence pack "abl316-t2a", and a scope slug is a key, not a title.
    # The heading names the feature set because that is what distinguishes this
    # read from ABL-381's on the two countries they share; a reader quoting the H1
    # of either report should not have to reach the `feature_set` field to know
    # which challenger it describes.
    "abl316-t2a": "ABL-405 — Serve-faithful solar retrain gate, ABL-316 tranche 2a: 8 continental countries on energy_generation at 27 features",
}


def fit_rules_for(scope: str) -> dict:
    """The scope's registered fit rules, over the defaults."""
    return {**DEFAULT_FIT_RULES, **FIT_RULES.get(scope, {})}


def features_for(scope: str) -> tuple:
    """The scope's registered feature vector, or the current default (ABL-395)."""
    return tuple(SCOPE_FEATURES.get(scope, DEFAULT_SCOPE_FEATURES))


def title_for(scope: str) -> str:
    """The scope's report heading, or a derived one."""
    return SCOPE_TITLES.get(scope, f"{scope} — Serve-faithful solar retrain gate")


# ABL-387: the three tables above are one registration in three views.  Checked
# at import, so a scope registered in one and not the others fails before any fit
# -- and identically under `--help` and in the test suite -- rather than raising
# `KeyError` partway through a gate run, or writing over another scope's evidence.
#
# ABL-376's two tables are deliberately **not** in this check, and the asymmetry
# is the point rather than an oversight.  What makes the three strict is that
# omitting an entry fails *destructively and silently*: a missing `SCOPE_OUTPUTS`
# row sends a run's results over another scope's dispositioned evidence, and no
# exit status shows it.  A missing `FIT_RULES` or `SCOPE_TITLES` row does not --
# it resolves through `fit_rules_for`/`title_for` to the pre-ABL-376 behaviour,
# and the report then says in as many words that the rule is not registered for
# that scope.  Self-documenting degradation does not need an import-time abort.
#
# The cost of getting this wrong is concrete and was measured, not imagined:
# `ABL-381-tranche-1b` and `fix/abl-379-solar-gate-scope` are both live and both
# add a solar scope to the three tables.  Had the new tables joined the strict
# check, either merge order would produce a **textually CLEAN** merge that raises
# on `import` -- taking `--help` and the whole suite with it -- with nothing on
# GitHub to warn either author.  Adding a required table is not free; it is a
# tax on every branch already in flight.
check_registration_tables(SCOPES=SCOPES, GATE_BASIS=GATE_BASIS, SCOPE_OUTPUTS=SCOPE_OUTPUTS)
check_scope_outputs(SCOPE_OUTPUTS)


def _model():
    params = config.get_default_params(ALGORITHM)
    return CatBoostRegressor(**params), params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_runs(replica: str, country: str, start, end, source: str) -> list[dict]:
    """Screen the fitted series for ABL-188 zero-fill runs.

    ABL-345: `source` is the table the model was fitted on, not a constant. This
    read used to be hardcoded to `energy_renewable` while the builder's source
    became a per-run argument — a contamination audit of a series the model
    never saw. It reports the wrong way in both directions: an
    `energy_generation` fit inherits `energy_renewable`'s zero-fill runs (which
    are the reason to leave that table), and a genuine constant run in the
    fitted series goes unreported. `verdict` is derived from this list, so a
    mismatched screen moves the harness's disposition, not just its prose.
    """
    if source not in db._RENEWABLE_TYPE_SOURCES:
        raise ValueError(
            f"unknown renewable source table: {source!r}; "
            f"expected one of {db._RENEWABLE_TYPE_SOURCES}"
        )
    # Both tables name this column identically; `RENEWABLE_TYPE_COLUMNS` is the
    # one place that knows, and `load_renewable_type_data` already reads either
    # table through it.
    column = db.RENEWABLE_TYPE_COLUMNS["solar"]
    con = _ro_connect(replica)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {column} AS value FROM {source} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
            "ORDER BY timestamp_utc",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], format="mixed", utc=True
    ).dt.tz_localize(None)
    return [
        {"start": str(run.start), "end": str(run.end), "value": run.value,
         "n_rows": run.n_rows, "duration_hours": run.duration_hours}
        for run in find_suspect_constant_runs(frame, "value")
    ]


def _fmt(value, suffix=""):
    return "Not measured" if value is None else f"{value:.1f}{suffix}"


def _mw(value):
    """Megawatts, or `n/a` for a quantity that has no value rather than no measurement.

    `_fmt` renders `None` as "Not measured", which is right for a comparator that
    does not exist and wrong for ABL-376's per-country night table: a country
    whose data is clean has no largest excluded actual because there was nothing
    to exclude, not because nobody looked.
    """
    return "n/a" if value is None else f"{value:,.1f} MW"


def disposition(gate_cells: list[dict], registered_cells: int,
                contaminated: bool) -> tuple[str, str]:
    """Map the scored cells onto a verdict and its recommendation.

    Extracted from `main` so the `UNREADABLE` branch is reachable in a test
    without training three models. The distinction it encodes is the whole point
    of ABL-378: a cell that scored no rows did not lose a race, it never ran one,
    and calling that FAIL reports a model-quality verdict on a comparison that
    never happened.
    """
    performance_pass = len(gate_cells) == registered_cells and gate_cells and all(
        row["gate"]["pass"] for row in gate_cells)
    if performance_pass and not contaminated:
        return "PASS", (
            "The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. "
            "Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.")
    if performance_pass:
        return "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION", (
            "The challenger clears the performance bar, but a suspect constant run touches the registered data window. "
            "Do not promote; send the run to the CEO/ingest owner for adjudication first.")
    unreadable = [row for row in gate_cells if row["gate"]["n"] == 0]
    if unreadable:
        return "UNREADABLE", (
            f"No disposition: {len(unreadable)}/{registered_cells} primary cells scored zero rows, so the challenger was "
            "never compared to the baseline in them. This is not a model-quality result and must not be reported as one. "
            "Fix the cause of the empty intersection and re-read the gate; the registered windows, bands, metric, baseline "
            "and minimum n are untouched by a run that produced no score.")
    passed = sum(row["gate"]["pass"] for row in gate_cells)
    return "FAIL", (
        f"Do not promote these artifacts: only {passed}/{registered_cells} primary cells clear the registered bar. Report the "
        "losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split.")


def render_markdown(result: dict) -> str:
    meta, cells = result["meta"], result["gate_cells"]
    passed = sum(cell["gate"]["pass"] for cell in cells)
    # ABL-376: the title used to be the ABL-253 literal, which put that issue's
    # name on every other scope's report. `abl253` keeps its exact heading, so
    # the dispositioned read still renders byte-for-byte.
    lines = [
        f"# {title_for(meta['scope'])}", "",
        f"**Disposition: {result['verdict']}**", "",
        f"Generated: {meta['generated_at']}",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample gate targets: {meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive).",
        "Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.",
        # ABL-355: which *files* the run opened. `--replica-db` used to cover
        # the incumbent, TSO and screen only; the fitted series and weather came
        # from `ENERGY_DB_PATH`, and this heading named one path for two files.
        *describe_opened_databases(meta["databases"], meta["replica_bytes"]),
        # ABL-345: the two tables disagree — 9 months against 5.6 years of
        # history, and `energy_renewable` zero-fills what `energy_generation`
        # leaves NULL. Two runs of this report are not comparable unless both
        # say which table they read, so it is stated, never defaulted silently.
        f"Target series, features, baselines and contamination screen: `{meta['training_source']}`.",
        # ABL-395. Stated for the same reason the source table is: the feature
        # vector moved when the ABL-338 geometry pair was added to it, so two
        # reads of this gate are not comparable unless both name the set they
        # fitted. `legacy25` is what every read up to ABL-381 was taken on.
        f"Feature set: **{meta.get('feature_set', 'legacy25')}** "
        f"({meta.get('n_features', 25)} columns), "
        + ("registered for this scope."
           if meta.get("feature_set_is_registered_for_scope")
           else "the module default -- this scope registers no feature set of its own."),
        "", "## Gate read", "",
        f"Registered scope `{meta['scope']}`: {', '.join(meta['registered_countries'])}.",
        f"Gate basis — the columns that must be simultaneously finite for a row to be scored: {', '.join(f'`{c}`' for c in meta.get('gate_basis', []))}. "
        "Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that "
        "does not exist for a country reads Not measured instead of emptying the cell.",
        f"Strict full PASS requires challenger WAPE < D-7 in all {meta['registered_cells']} country × primary D+2-band cells and ≥95% of intended pairs. Result: **{passed}/{meta['registered_cells']} cells pass**.",
        "",
        # ABL-389.  References, not criteria: in `REPORTED_COMPARATORS` and in
        # no `GATE_BASIS` entry, so the PASS rule, the bands, the bars and the
        # windows are exactly what they were.
        *reference_prose(),
        "",
        # ABL-418: what the PASS in the last column entitles the cell to. The
        # gate column is unchanged and still says whether the cell cleared D-7.
        *grading_prose(GRADE_STREAM),
        "",
        "| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate | grade |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in cells:
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        # A cell that scored no rows has None on both sides. It renders as
        # "Not measured", never as a number and never as a crash.
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        grade = cell_grade(row, GRADE_STREAM).detail
        lines.append(
            f"| {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill} | {_fmt(comparator_wape(scores, 'constant_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_oracle'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_oracle'), '%')} | "
            f"{_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} | {grade} |"
        )
    lines.extend(grade_summary_table(cells, GRADE_STREAM, lambda row: row["country"]))
    lines.extend(levels_table(result["training"]))
    # The protocol-count sentence below is a measured ABL-253 fact about that
    # scope's eight registered run instants. Rendering it for every scope would
    # state another scope's row counts as if they had been measured; the per-cell
    # `n` column above already carries that truth for whichever scope ran.
    if meta["scope"] == "abl253":
        lines.extend([
            "",
            "The exact eight registered run instants imply 210/570/720/720/510 selected rows by band. As in ABL-195, the frozen registered minimum for 48–64h remains 456 (95% of 480), while the schedule offers 510 rows.",
        ])
    basis_names = ", ".join(meta.get("gate_basis", []))
    lines.extend([
        "", "## Per-country all-D+2 summary", "",
        f"Gate-basis values (actual, {basis_names}) share one finite intersection; each comparator outside the basis is "
        "scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing "
        "`Not measured` had no finite rows at all.", "",
        "| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["country_d2"]:
        scores, tso = row["scores"], row["tso"]
        lines.append(
            f"| {row['country']} | {row['n']:,} | {_fmt(scores['challenger']['wape_pct'], '%')} | "
            f"{_fmt(scores['seasonal_naive']['wape_pct'], '%')} | {_fmt(scores['persistence']['wape_pct'], '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_causal'), '%')} | {_fmt(comparator_wape(scores, 'constant_oracle'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal'), '%')} | {_fmt(comparator_wape(scores, 'climatology_oracle'), '%')} | "
            f"{_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(tso['wape_pct'], '%')} (n={tso['n']:,}) |"
        )
    lines.extend([
        "", "## Fit and missingness audit", "",
        "Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.", "",
        "| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in result["training"]:
        audit = row["audit"]
        lines.append(
            f"| {row['country']} | {row['algorithm']} | {audit['retained_rows']:,} / {audit['intended_rows']:,} | "
            f"{audit['unique_targets']:,} | {audit['excluded_missing_actual_or_feature']:,} | "
            f"{audit['degraded_lag_1d_rows']:,} | `{row['artifact_sha256']}` |"
        )
    # ABL-376. Rendered whether the rule is on or off, and saying which: a
    # section that appears only when rows were removed cannot distinguish "the
    # rule is not registered here" from "the rule ran and the data was clean",
    # and telling a data fix from a rule change is the whole reason this is
    # reported per country rather than summarised.
    rules = meta.get("fit_rules", {})
    lines.extend(["", "### Physically impossible night rows (ABL-376)", ""])
    if not rules.get("exclude_impossible_night"):
        lines.append(
            f"Not registered for scope `{meta['scope']}`. The fit saw every night row, "
            "including any whose actual the sun says is impossible."
        )
    else:
        lines.extend([
            f"Night is `solar_geometry.is_night_hour` — the serving clamp's own predicate, sun below "
            f"{meta['night_threshold_deg']:g} deg geometric for the whole hour. A night row whose actual "
            f"exceeds **{meta['impossible_night_threshold_mw']:g} MW** is physically impossible and was dropped "
            "**from the fit only**. The gate frame below was not filtered: a contaminated actual still scores "
            "against the challenger, which is why the daylight numbers above are not marking their own homework.",
            "",
            "Rows are per (target, vintage); `hours` is the distinct contaminated target hours behind them — "
            "the row count is what the fit lost, the hour count is what the source got wrong.",
            "",
            "| country | night fit rows | excluded rows | excluded hours | max excluded actual | mean night actual (before) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in result["training"]:
            night = row.get("night_fit_audit")
            if not night:
                continue
            lines.append(
                f"| {row['country']} | {night['night_rows']:,} | {night['excluded_rows']:,} | "
                f"{night['excluded_targets']:,} | {_mw(night['max_excluded_mw'])} | "
                f"{_mw(night['mean_night_actual_mw'])} |"
            )
        lines.extend([
            "",
            "A country reading 0 excluded is the rule finding clean data, not the rule being off — "
            "the predicate is the sun's, so it is stated over countries rather than for the one that "
            "prompted it.",
        ])
    lines.extend(["", "## Data quality and limits", ""])
    source = meta["training_source"]
    contaminated = [row for row in result["training"] if row["constant_runs"]]
    if contaminated:
        for row in contaminated:
            lines.append(f"- ABL-188 screening found suspect solar runs for {row['country']} in `{source}`: `{row['constant_runs']}`. The builder nulls these before fit; see the training audit and recommendation.")
    else:
        lines.append(f"- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `{source}` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.")
    # ABL-345: both notes below are findings about `energy_renewable` specifically
    # — the zero-fill run is ABL-188's `energy_renewable` mapper defect, and the
    # New Year's read was on that table. Printing them under an
    # `energy_generation` run would report a table this run never opened.
    if source == "energy_renewable" and not contaminated:
        lines.append("- The known DE zero-fill run (2025-09-08 22:00 → 2025-11-14 15:45 UTC; 6,408 quarter-hours) is outside this fit/lookback window.")
        lines.append("- The audit initially appeared to flag FR zero from 2025-12-31 17:00 to 2026-01-02 07:15 UTC, but the replica has no intervening New Year's Day rows and `energy_generation` independently agrees on zero for the available nighttime observations. `find_suspect_constant_runs` was incorrectly joining equal values across missing-time gaps despite its contiguous-run contract. The invariant now splits on cadence gaps; the original continuous DE defect remains covered by regression tests.")
    lines.extend([
        "- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.",
        "- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.",
        "- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.",
        "- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.",
    ])
    # ABL-389: a reader who sees PASS will not otherwise compare two numbers in
    # a fourteen-column table. Reporting only; changes no verdict above.
    lines.extend(lost_to_a_model_free_reference(
        cells, lambda row: f"{row['country']} solar {row['horizon_band']}"))
    lines.extend([
        "", "## Recommendation to the CEO", "", result["recommendation"], "",
        "No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=str(config.FORECAST_OUTPUT_DB))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    # ABL-387: default to the *scope's* registered paths, resolved after parsing.
    # A fixed default here is resolved before `--scope` is read, which is how a
    # scoped run came to overwrite ABL-253's dispositioned artifacts in place.
    parser.add_argument("--artifact-dir", default=None,
                        help="Override the scope's registered artifact directory")
    parser.add_argument("--json-out", default=None,
                        help="Override the scope's registered results.json path")
    parser.add_argument("--report-out", default=None,
                        help="Override the scope's registered report path")
    parser.add_argument("--scope", default="abl253", choices=sorted(SCOPES),
                        help="Registered country scope. Each scope fixes its countries "
                             "and its gate basis in the file; the cell count is derived "
                             "from them (default: abl253, which reproduces ABL-253).")
    # ABL-345: the 19 unmodelled solar pairs have ~9 months in `energy_renewable`
    # against ~5.6 years in `energy_generation`, so this harness cannot gate them
    # on one hardcoded table. Opt-in, so an unflagged run reproduces ABL-253.
    parser.add_argument("--renewable-source", default=None,
                        choices=list(db._RENEWABLE_TYPE_SOURCES),
                        help="Source table for the fitted series, its lag and rolling "
                             "features, the D-7/persistence baselines, the gate actuals "
                             "and the contamination screen (default: "
                             f"{db.RENEWABLE_TYPE_SOURCE_TABLE})")
    args = parser.parse_args()
    fit_start, gate_start, gate_end = map(pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    # Resolved once, here, and handed to every read site and to the record. The
    # builder would resolve a `None` identically (`Forecaster._resolved_training_source`,
    # `forecaster.py:132`), but then the run's source would be a default applied
    # in three places rather than one recorded fact — and the report could not
    # name the table it read.
    source = args.renewable_source or db.RENEWABLE_TYPE_SOURCE_TABLE

    cfg = ScorecardConfig(str(replica), args.sidecar_db, gate_start, gate_end,
                          models={"solar": "catboost"})
    incumbent_raw, vintage_counts = _load_forecasts(cfg)
    incumbent = select_latest_per_band(incumbent_raw)
    tso = _load_tso(cfg, "solar")
    outputs = SCOPE_OUTPUTS[args.scope]
    artifact_dir = Path(args.artifact_dir or outputs["artifact_dir"])
    registered_countries = SCOPES[args.scope]
    fit_rules = fit_rules_for(args.scope)
    # ABL-395: the scope's registered feature vector, not the module constant.
    # `abl253`/`abl376` pin the 25 they were read on; anything new gets 27.
    features = features_for(args.scope)
    registered_cells = len(registered_countries) * len(PRIMARY_BANDS)
    training, scored_frames = [], []
    for country in registered_countries:
        # ABL-342 records provenance from the builder rather than from a source
        # string, so passing the source here is also what makes the artifact's
        # `training_source` truthful.
        # ABL-355: `db_path` for the same reason `actuals_source` is here. The
        # builder resolved neither on its own — it read `config.DATABASE_PATH`,
        # so `--replica-db` bought the incumbent and the screen while the fitted
        # series came from wherever `ENERGY_DB_PATH` pointed. Passing the
        # resolved replica is what makes `--replica-db` mean the whole run.
        builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                          gate_end, actuals_source=source,
                                          db_path=str(replica))
        fit_raw = build_vintage_frame(builder, fit_start, gate_start, features)
        fit, audit = finite_training_rows(fit_raw, features)
        # ABL-376, and only where the scope registered it.  Fit frame only: the
        # gate frame below is built from the same builder and is deliberately
        # *not* filtered, so a contaminated actual still scores against us.
        # Applied after `finite_training_rows` so the two audits partition the
        # dropped rows instead of double-counting a missing actual as impossible.
        if fit_rules["exclude_impossible_night"]:
            fit, night_audit = exclude_impossible_night_rows(fit, country)
        else:
            night_audit = None
        model, params = _model()
        model.fit(fit[list(features)], fit["actual"])

        # ABL-342: through `Forecaster.save`, so the artifact carries the table
        # it was fitted on and the ABL-183 intercept witness by construction.
        path = save_gate_artifact(
            artifact_dir / country / "solar" / "model.joblib",
            model=model, builder=builder, algorithm=ALGORITHM, params=params,
            feature_columns=features, fit_window=(fit_start, gate_start),
        )

        gate_raw = build_vintage_frame(builder, gate_start, gate_end, features)
        gate_finite, gate_audit = finite_training_rows(gate_raw, features)
        gate_finite["challenger"] = model.predict(gate_finite[list(features)])
        selected = attach_baselines(select_latest_challenger_per_band(gate_finite), builder._actuals)
        # ABL-389: from the same ABL-188-filtered series the baselines and the
        # gate actuals come from, so the reference is arithmetic on data already
        # loaded -- no refit, no second read, no additional upstream fetch.
        selected, reference_levels = attach_model_free_references(
            selected, builder._actuals, fit_start, gate_start, gate_end)
        inc = incumbent[incumbent["country_code"] == country][
            ["target_ts", "horizon_band", "forecast_value"]
        ].rename(columns={"forecast_value": "incumbent"})
        selected = selected.merge(inc, on=["target_ts", "horizon_band"], how="left")
        selected = selected.merge(tso[tso["country_code"] == country][["target_ts", "tso"]],
                                  on="target_ts", how="left")
        selected["country"] = country
        scored_frames.append(selected)
        training.append({"country": country, "algorithm": ALGORITHM, "params": params,
                         "audit": audit, "gate_build_audit": gate_audit,
                         # None when the scope did not register the rule, so the
                         # record distinguishes "rule off" from "rule on, removed
                         # nothing" -- which is exactly the distinction a later
                         # run needs to tell a data fix from a rule change.
                         "night_fit_audit": night_audit,
                         "model_free_reference_mw": reference_levels,
                         "constant_runs": _constant_runs(str(replica), country,
                                                          fit_start - pd.Timedelta(days=14), gate_end,
                                                          source),
                         "artifact_path": str(path.resolve()), "artifact_sha256": _sha256(path)})

    gate_basis = GATE_BASIS[args.scope]

    def scored(group):
        return scored_with_comparators(group, gate_basis, REPORTED_COMPARATORS)

    all_scored = pd.concat(scored_frames, ignore_index=True)
    gate_cells, country_d2 = [], []
    for (country, band), group in all_scored.groupby(["country", "horizon_band"]):
        scores, common, comparator_n = scored(group)
        if band in PRIMARY_BANDS:
            gate_cells.append({"country": country, "horizon_band": band, "scores": scores,
                               "comparator_n": comparator_n,
                               "gate": gate_cell(scores["challenger"]["wape_pct"],
                                                 scores["seasonal_naive"]["wape_pct"],
                                                 len(common), INTENDED_N[band])})
    for country, group in all_scored[all_scored["horizon_band"].isin(PRIMARY_BANDS)].groupby("country"):
        scores, common, comparator_n = scored(group)
        tso_valid = np.isfinite(common[["actual", "tso"]].to_numpy(dtype=float)).all(axis=1)
        country_d2.append({"country": country, "n": len(common), "scores": scores,
                           "comparator_n": comparator_n,
                           "tso": score_predictions(common.loc[tso_valid, "actual"], common.loc[tso_valid, "tso"])})

    # ABL-418.  One call, here, so the markdown table and `results.json` cannot
    # disagree about a grade.  It reads only columns already in `scores` and
    # changes no verdict: `passed` and `disposition` below are computed exactly
    # as they were.  The stream selects ABL-385's registered CV, and passing the
    # wrong one would silently apply wind's narrower floor --
    # `tests/test_gate_grading.py` reads this literal out of the AST.
    attach_grades(gate_cells, GRADE_STREAM)
    passed = sum(row["gate"]["pass"] for row in gate_cells)
    contaminated = any(row["constant_runs"] for row in training)
    verdict, recommendation = disposition(gate_cells, registered_cells, contaminated)

    result = {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
                       # ABL-355: the run's files, not just its tables. The
                       # builder is handed `replica`, so `features` equals
                       # `replica` by construction — recorded anyway, because
                       # what this issue cost was the *absence* of the record.
                       "databases": opened_databases(cfg, str(replica), config.DATABASE_PATH),
                       "training_source": source,
                       "scope": args.scope, "registered_countries": list(registered_countries),
                       "registered_cells": registered_cells, "gate_basis": list(gate_basis),
                       # ABL-376: what the fit was allowed to see, recorded beside
                       # what it was scored on. Two reads of this gate are not
                       # comparable unless both state it.
                       "fit_rules": dict(fit_rules),
                       # ABL-395: and what it was allowed to see column-wise. A
                       # scope pinned to the legacy 25 and a scope on the current
                       # 27 produce artifacts that are indistinguishable after
                       # the fact unless the read says which it was.
                       "feature_columns": list(features),
                       "n_features": len(features),
                       "feature_set": ("legacy25" if tuple(features) == LEGACY_FEATURE_COLUMNS
                                       else "legacy25+geometry"
                                       if tuple(features) == tuple(FEATURE_COLUMNS)
                                       else "custom"),
                       "feature_set_is_registered_for_scope": args.scope in SCOPE_FEATURES,
                       "night_threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
                       "impossible_night_threshold_mw": IMPOSSIBLE_NIGHT_THRESHOLD_MW,
                       # ABL-389: the basis is what gates; this is what is
                       # merely reported beside it. Recorded so the two are
                       # distinguishable in the record and not only in the prose.
                       "reported_comparators": list(REPORTED_COMPARATORS),
                       "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
                       "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
                       "registered_intended_n": INTENDED_N, "schedule_implied_n": SCHEDULE_N,
                       "vintage_counts": vintage_counts,
                       "selection": "latest vintage per country + target + model + horizon band"},
              "verdict": verdict, "recommendation": recommendation, "training": training,
              "gate_cells": sorted(gate_cells, key=lambda row: (row["country"], row["horizon_band"])),
              "country_d2": sorted(country_d2, key=lambda row: row["country"])}
    json_path = Path(args.json_out or outputs["json_out"])
    report_path = Path(args.report_out or outputs["report_out"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"{verdict}: {passed}/{registered_cells} cells passed; wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
