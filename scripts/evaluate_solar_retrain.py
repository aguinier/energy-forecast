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
    FLOORED, SIGN_TEST, attach_grades, cell_grade, grade_summary_table,
    grading_prose,
)
from src.evaluation.model_free_reference import (
    FIT_WINDOW, MODEL_FREE_COMPARATORS, TRAILING_28D,
    attach_model_free_references, comparator_wape, level_inflation,
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
    # of a path -- and was separately live as ABL-404, since that scope held no
    # `SCOPE_FEATURES` row and so resolved to 27 against its own 25-feature
    # published read.  Nothing here fixed that; this scope simply did not rely on
    # it.  ABL-404 has since pinned `abl316-t1b` to `LEGACY_FEATURE_COLUMNS`, so
    # that route is closed as well as unused.
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
    # ABL-419 -- ABL-316 tranche 2c: the five Mediterranean solar countries, under
    # the same frozen registration at `experiments/ABL348/config.json`.
    # 5 countries x 3 bands = 15 cells.
    #
    # All five are new coverage.  Unlike `abl316-t2a` this scope shares no country
    # with any dispositioned solar read, so it re-reads nothing and its outputs
    # collide with no published evidence by construction as well as by
    # `check_scope_outputs`.
    #
    # Grouped by the pre-committed D-7 bar and not by the alphabet, which is the
    # rule `abl316-t2a` states above: IT 7.11%, GR 10.37%, ES 11.78%, PT 13.09%,
    # HR 16.43% (`experiments/ABL348/config.json`,
    # `per_pair_bar_measured_before_any_challenger_exists.bars`, measured before a
    # challenger existed for any of these pairs).  These are the tightest solar
    # bars in the programme -- 2a's ran 18.35-26.11% plus CH at 12.67% -- because
    # Mediterranean July/August solar is nearly D-7 periodic.  ABL-348 registered
    # that reading in advance under `reading_caveats_not_band_changes`: same band,
    # materially harder task, and a lower pass rate here is not model quality.
    # **This tranche's pass rate must not be averaged against 2a's.**
    #
    # Neither `EE/solar` nor `FI/solar` -- ABL-348's two declared NOT-EVALUABLE
    # pairs -- is in this scope, so no cell here risks being scored against a
    # registration that forbids scoring it.  All five carry `n_d7_scorable` 720 in
    # that same table.
    #
    # Windows, bands, metric, baseline, minimum n and source table are ABL-348's
    # and are deliberately not restated here -- twenty-eight remaining tranches
    # must not become twenty-eight chances to shop a window.
    #
    # No country here serves a solar model: measured on the live replica
    # (9,432,453,120 bytes, mode=ro) on 2026-08-14, `forecasts` holds solar rows
    # for AT/BE/DE/FR only and **zero** for each of ES, GR, HR, IT and PT.  So this
    # scope refits no live pair, the property `abl253` protects.
    "abl316-t2c": ("ES", "GR", "HR", "IT", "PT"),
    # ABL-421 -- ABL-316 tranche 2d, the **final solar tranche**: the six northern
    # solar countries, under the same frozen registration at
    # `experiments/ABL348/config.json`.  6 countries x 3 bands = 18 country-band
    # cells, of which **14 are evaluable** -- see `SCOPE_NOT_EVALUABLE` below,
    # which is the whole reason this scope needed a new table.
    #
    # All six are new coverage and none shares a country with any dispositioned
    # solar read, so this scope re-reads nothing and its outputs collide with no
    # published evidence by construction as well as by `check_scope_outputs`.
    #
    # Grouped by the pre-committed D-7 bar and not by the alphabet, which is the
    # rule `abl316-t2a` states above: SE 23.92%, LT 30.84%, EE 36.67%, FI 37.88%,
    # NL 46.53%, LV 47.85% (`experiments/ABL348/config.json`,
    # `per_pair_bar_measured_before_any_challenger_exists.bars`, measured before a
    # challenger existed for any of these pairs).  These are the **loosest** solar
    # bars in the programme -- 2c's ran 7.11-16.43% and 2a's 12.67-26.11% -- on the
    # lowest levels: NL's gate-window mean is 66.7 MW.  ABL-418 registered the
    # ladder precisely because a loose bar on a low level is what produced 2b's
    # spurious wind passes, so a high pass rate here carries less than 2c's low one.
    # **This tranche's pass rate must not be averaged against 2a's or 2c's.**
    #
    # Unlike every earlier tranche this scope **does** contain ABL-348's two
    # declared NOT-EVALUABLE pairs, EE/solar and FI/solar.  That is deliberate and
    # is what the tranche is for: they are two of the six northern countries, the
    # rollout has to give them an auditable answer, and ABL-348's own
    # `not_evaluable.note_48_64h` says their 48-64h band may still be readable and
    # "should be reported if it does".  Scoring their two 684-bands anyway is what
    # the registration forbids; omitting the countries entirely would leave the
    # rollout with no record of why.  `SCOPE_NOT_EVALUABLE` is how both are
    # satisfied at once.
    #
    # No country here serves a solar model: measured on the live replica
    # (9,432,453,120 bytes, mode=ro) on 2026-08-14, `forecasts` holds solar rows
    # for AT/BE/DE/FR only (34,228 BE / 32,856 FR / 32,784 AT / 32,256 DE) and
    # **zero** for each of EE, FI, LT, LV, NL and SE.  So this scope refits no live
    # pair, the property `abl253` protects.
    "abl316-t2d": ("EE", "FI", "LT", "LV", "NL", "SE"),
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
    # ABL-419.  Same three-way shape as `abl316-t2a` above and for the same
    # reasons, so only what differs is argued here.
    #
    # `artifact_dir` is `experiments/ABL419/artifacts` -- its own directory, one
    # level deep so `.gitignore:56` (`experiments/*/artifacts/`, which matches on
    # the *directory name*) still takes it and no binary CatBoost bundle becomes
    # committable.  A per-tranche directory is not merely tidy here: it is what
    # `check_scope_outputs` can then enforce.  This scope names no country any
    # other scope names, so unlike the 2a/1b pair there is no
    # `experiments/.../BG/solar/model.joblib` this run could overwrite even if the
    # directory were shared -- but sharing would still put fifteen artifacts in a
    # directory whose name says ABL405, and an artifact whose SHA-256 a published
    # machine record cites must stay findable from that record.
    #
    # `json_out` takes the tracked form -- one level deep and deliberately **not**
    # named `results.json`, which `.gitignore:53` matches by exact filename -- and
    # sits under ABL348 for the reason `abl316-t1b` gives: the registration these
    # fits are read under is `experiments/ABL348/config.json`.  This is the machine
    # record behind a 15-cell read whose margins are the tightest in the
    # programme, and an ignored `results.json` is the one gate record a reviewer
    # cannot diff.
    "abl316-t2c": {"artifact_dir": "experiments/ABL419/artifacts",
                   "json_out": "experiments/ABL348/results_abl419_tranche2c.json",
                   "report_out": "reports/abl_419_solar_tranche2c.md"},
    # ABL-421.  Same three-way shape as `abl316-t2c` above and for the same
    # reasons, so only what differs is argued here.
    #
    # `artifact_dir` is `experiments/ABL421/artifacts` -- its own directory, one
    # level deep so `.gitignore:56` (`experiments/*/artifacts/`, which matches on
    # the *directory name*) still takes it and no binary CatBoost bundle becomes
    # committable.  This scope names no country any other scope names, so there is
    # no `experiments/.../<CC>/solar/model.joblib` this run could overwrite even if
    # the directory were shared; the per-tranche directory is what
    # `check_scope_outputs` can then enforce, and it keeps an artifact whose
    # SHA-256 a published machine record cites findable from that record.
    #
    # `json_out` takes the tracked form -- one level deep and deliberately **not**
    # named `results.json`, which `.gitignore:53` matches by exact filename -- and
    # sits under ABL348 for the reason `abl316-t1b` gives: the registration these
    # fits are read under is `experiments/ABL348/config.json`.  It matters more than
    # usual here: this is the only machine record that will carry the four
    # NOT-EVALUABLE cells with their measured n beside them, and "we declined to
    # score EE and FI on two bands" is a claim a reviewer must be able to diff
    # rather than take from prose.  An ignored `results.json` is the one gate record
    # they could not.
    "abl316-t2d": {"artifact_dir": "experiments/ABL421/artifacts",
                   "json_out": "experiments/ABL348/results_abl421_tranche2d.json",
                   "report_out": "reports/abl_421_solar_tranche2d.md"},
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
    # ABL-419: identical to the two above, and for the identical reason -- all five
    # countries hold **zero** solar rows in `forecasts` (verified on the live
    # replica, not assumed; see this scope's `SCOPES` entry), so under the four-way
    # basis every one of the 15 cells would intersect to n=0 and the run would
    # render UNREADABLE, having compared nothing.  Keeping it identical to 2a's is
    # also what lets the two tranches be read side by side on the bar alone, which
    # is the comparison ABL-348's `southern_solar_bar_is_tight` caveat asks for.
    # The incumbent is still reported on its own intersection, where it reads
    # "Not measured" by construction rather than by omission.
    "abl316-t2c": ("challenger", "seasonal_naive"),
    # ABL-421: identical to the three above, and for the identical reason -- all six
    # countries hold **zero** solar rows in `forecasts` (verified on the live
    # replica, not assumed; see this scope's `SCOPES` entry), so under the four-way
    # basis every one of the 18 cells would intersect to n=0 and the run would
    # render UNREADABLE, having compared nothing.  Keeping it identical to 2a's and
    # 2c's is also what lets the three solar tranches be read side by side on the
    # bar alone, which is the comparison ABL-348's bar caveats ask for -- and this
    # tranche sits at the loose end of that spread, where the comparison matters
    # most.  The incumbent is still reported on its own intersection, where it reads
    # "Not measured" by construction rather than by omission.
    "abl316-t2d": ("challenger", "seasonal_naive"),
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

# ABL-437: which pair of causal references the ABL-418 ladder's G2 and G3 read.
# The wind harness carries the twin of this table and the same rule; the reasons
# are stated once, over `evaluate_wind_retrain.py`'s copy, and the short form is:
# every scope below is published, its letters were decided against the fit-window
# references, and a re-run that silently disagreed with its own record is the
# ABL-404 failure mode.  So published scopes are pinned and a new scope defaults
# to `trailing_28d`.
#
# Solar is the stream where the amendment moves *least* on the constant and
# potentially most on the climatology.  The mis-levelling ABL-437 measured is
# 0-8% on every solar pair, because a flat line's WAPE there is dominated by the
# diurnal cycle rather than by the level -- but `climatology_causal` is a
# Jan-to-Jul hour-of-day mean scored on high summer, and re-levelling it to a
# trailing 28 days makes G3 a materially harder question on this stream.  That is
# the intended direction and it is why this is a registration rather than a fix.
#
# This is the *eighth* scope-keyed table in this file, and only three are in the
# `check_registration_tables` call at the bottom.  Count them with
# `grep -E "^[A-Z_]+ = \{"` rather than trusting this sentence, which is exactly
# the number that drifted twice before.
CAUSAL_LEVELLING = {
    "abl253": FIT_WINDOW,
    "abl316-t1b": FIT_WINDOW,
    "abl316-t2a": FIT_WINDOW,
    "abl316-t2c": FIT_WINDOW,
    "abl316-t2d": FIT_WINDOW,
    "abl376": FIT_WINDOW,
}


def causal_levelling_for(scope: str) -> str:
    """The scope's registered causal levelling, or ABL-437's amended default."""
    return CAUSAL_LEVELLING.get(scope, TRAILING_28D)


# ABL-444: whether the ABL-418 ladder's G2 and G3 are decided by a bare sign
# test or against the readability floor G1 already carries.  The wind harness
# carries the twin of this table and the argument is stated once, over its copy
# -- including why this table defaults *toward* the amendment where
# `SCOPE_FEATURES` and `SCOPE_NOT_EVALUABLE` do not, and why it is deliberately
# absent from `check_registration_tables` while three PRs are in flight.
#
# The short form: every scope below is published and its ABL-418 letters were
# decided by a sign test, so each is pinned to `sign_test` and nothing already
# published moves; a scope registering nothing gets `floored`.
#
# Solar is where this matters most under ABL-437's amended levelling and least
# under the published one.  Against the fit-window climatology a solar
# challenger's G3 margin is tens of percent -- 38.30% on PL, far outside this
# stream's floor -- so the sign test and the floored test agree.  Against the
# trailing-28d climatology the same cell's margin is -1.13%, and the letter that
# turned on it was ABL-437's PL solar A -> B.  Registering the floor is what
# stops the two amendments compounding into a graded verdict neither of them
# measured.
G23_READABILITY = {
    "abl253": SIGN_TEST,
    "abl316-t1b": SIGN_TEST,
    "abl316-t2a": SIGN_TEST,
    "abl316-t2c": SIGN_TEST,
    "abl316-t2d": SIGN_TEST,
    "abl376": SIGN_TEST,
}


def g23_readability_for(scope: str) -> str:
    """The scope's registered G2/G3 readability form, or ABL-444's amended default."""
    return G23_READABILITY.get(scope, FLOORED)


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

# ABL-403 registered the value for the tranches that did not exist yet, on the
# CEO's adoption of PR #58: **`exclude_impossible_night: False` for every
# remaining ABL-316 solar tranche, ES and EE included** -- 2c (ABL-419) and 2d
# (ABL-421).  Registered on the evidence and before those fits, not against a
# result, which is the only way a pre-registration means anything.  It binds the
# table, not any one row, which is why it sits above the table: rows are appended
# at the tail, so a standing rule parked there would be read by the next editor as
# the docstring of whatever row lands under it.
#
# ABL-429 put this table inside `check_registration_tables` (see the call at the
# bottom of this file), so a new scope that omits its row now fails at *import*
# rather than resolving through `DEFAULT_FIT_RULES` -- which is also False, so
# the omission used to produce the registered *behaviour* while leaving no record
# that anyone chose it: right answer, absent registration, and the next reader
# could not tell the two apart.
#
# That closes half of what this comment was written for, and not the half that
# matters.  **The check compares keys, never values.**  A tranche registered
# `True` here is a change to an adopted registration, and it imports, runs and
# exits 0 exactly as a compliant one does.  So the record of what was chosen, and
# why, is still text -- pinned by `tests/test_abl403_fit_rule_registration.py`,
# which holds every `abl316-t2*` row to the registered False, rather than left to
# survive the next merge on goodwill.
#
# The measurement behind the value: on BG the rule alone raises night MAE
# +61.05 MW (8/8 seeds, p = 0.0078, outside a 6.96 MW null), drives night bias
# -2.1 -> +88.5 MW, costs 1.4-1.9pp of gate-band WAPE and eats 47% of the D-7
# margin ABL-405's PASS was carrying.  It refuses 76.4% of BG's night fit rows
# while 25.3% of the *scored* gate rows are night rows at a 225 MW mean -- you
# cannot forbid a model to learn what you still grade it on.  ES is the stronger
# case still: its overnight MW is real CSP dispatch (ABL-411), so the rule would
# delete generation rather than noise.
# `reports/abl_403_night_rule_interaction.md`.
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
    # would have given it -- stated rather than inherited.  Since ABL-429 the row
    # itself is required at import, but `check_registration_tables` compares keys
    # and not values, so its *presence* is enforced and its *content* is not: this
    # text is still the only record that False was chosen rather than defaulted
    # into.
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
    # ABL-419 registers the rule **off**, which is also what `DEFAULT_FIT_RULES`
    # would have given it -- stated rather than inherited, for `abl316-t2a`'s
    # reason: ABL-429 requires the row but not its value, so an unstated False is
    # indistinguishable from a False nobody chose.
    #
    # Off is right here for three reasons, and the third is specific to this
    # tranche:
    #
    # - ABL-348's registration does not contain the rule, and this tranche is read
    #   under it unchanged.  Turning it on would be a source-and-fit-frame change
    #   after the bars were published, which `voids_this_registration` names.
    # - ABL-403 measured the geometry x night-exclusion 2x2 and left the rule off;
    #   ABL-419 discharges that issue's soft hold on this read rather than waiting
    #   on it, because the only cell the 2x2 could have moved here is ES, and ES's
    #   night floor is bounded exactly and for free by ABL-396's `f` (see below).
    # - **On ES the rule would be mostly wrong, and measurably so.**  ES is the one
    #   country in this scope with a material night floor -- ABL-396 section 2 puts
    #   `f` at 1.352% of gate-window energy -- and ABL-411 verified it against Red
    #   Electrica's own PV/CSP split rather than inferring it: over 3,196 night
    #   hours REE's `solFot + solTer` accounts for **98.55%** of the MW the replica
    #   books for ES with the sun down, MAE 5.55 MW against a 263.5 MW mean night
    #   level.  **80.1%** of that annual night energy is CSP, rising to **91.4% in
    #   July** -- which is this gate window.  `exclude_impossible_night` refuses to
    #   train on values *the sun* says are impossible; CSP discharges stored heat
    #   after sunset, so on ES the predicate is true of real generation, and turning
    #   the rule on would drop real megawatt-hours from the fit and teach the model
    #   a night floor of zero it would then be scored against.
    #
    #   ABL-411's refinement is carried, not dropped: the floor is **not all** CSP.
    #   REE's own *solar fotovoltaica* series reports 44-59 MW at sun elevations of
    #   -40 to -49 deg, where PV cannot generate, and that is **18.5%** of ES's
    #   annual night floor -- a TSO-side estimation artifact mirrored faithfully by
    #   ENTSO-E and by our ingest.  So the honest statement is that most of the
    #   floor is real and some of it is not, which is a reason to bound ES's read
    #   (this issue does, with `f`) rather than to filter its fit.  This is the one
    #   place in the programme where the rule and the physics disagree, and it is a
    #   finding for ABL-403's design question rather than an edit to this row.
    #
    # GR, HR, IT and PT screen at <= 0.009% night floor (ABL-396 section 3), so the
    # rule would remove essentially nothing for them either way.
    "abl316-t2c": {"exclude_impossible_night": False},
    # ABL-421 registers the rule **off**, which is also what `DEFAULT_FIT_RULES`
    # would have given it -- stated rather than inherited, for `abl316-t2a`'s
    # reason: this table is one of the four `check_registration_tables` does *not*
    # check, so an absence here is indistinguishable from an oversight and defaults
    # silently.
    #
    # Off is right here for three reasons, and the third is specific to this
    # tranche:
    #
    # - ABL-348's registration does not contain the rule, and this tranche is read
    #   under it unchanged.  Turning it on would be a fit-frame change after the
    #   bars were published, which `voids_this_registration` names.
    # - ABL-403 measured the geometry x night-exclusion 2x2 and left the rule off,
    #   on the general ground that a fit-side exclusion is only defensible when the
    #   excluded rows are both genuinely contaminated *and* a small enough minority
    #   that the score is not dominated by them.
    # - **On this tranche the rule would remove almost nothing, and on EE it would
    #   remove the wrong thing.**  Five of the six screen at or under 0.041% of
    #   gate-window energy at night (ABL-396 section 3: FI 0.002%, LT 0.018%, SE
    #   0.033%, LV 0.042%, NL 0.040% in absolute value), so the rule is a no-op for
    #   them either way.  **EE is the exception** and carries the third-largest
    #   solar night floor in the fleet at `f` = 0.718% of gate-window energy, 68 of
    #   its 86 gate night hours above the 1 MW threshold at a 12.64 MW mean.  But EE
    #   is exactly the pair whose two 684-bands ABL-348 declares NOT-EVALUABLE for
    #   an *unrelated* reason (an ABL-188 zero run), so turning the rule on would
    #   change the fit for a pair whose gate we are largely declining to read --
    #   moving a challenger nobody scores, and confounding the one band we do read.
    #   EE's floor is bounded exactly and for free by `f` instead, which is what
    #   this tranche's evidence pack prints on the face of its table.
    "abl316-t2d": {"exclude_impossible_night": False},
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
    # ABL-404.  `abl316-t1b` belongs here for exactly the reason the two above do,
    # and was missing only because of merge order: PR #40 registered the scope at
    # 18:34Z and PR #46 added this table afterwards, off a branch cut before the
    # scope existed.  Neither merge conflicted and nothing on GitHub flagged it,
    # which is the same shape as the ABL-387/ABL-380 near-miss recorded in the
    # `check_registration_tables` comment: a new scope landing beside a new table.
    #
    # Unpinned, this row's absence was not inert.  `SCOPE_OUTPUTS['abl316-t1b']`
    # writes `experiments/ABL348/results_abl381_tranche1b.json` and
    # `reports/abl_381_solar_tranche1b.md` -- ABL-381's published PASS 6/6 -- so
    # `--scope abl316-t1b` refitted BG and CH at 27, overwrote both in place, and
    # exited 0 under ABL-381's registered `SCOPE_TITLES` heading.  Using ABL-395's
    # own measurement on these two pairs, that moves BG's 24-36h cell 18.89% ->
    # 19.95% WAPE, from beating its 19.15% hour-of-day climatology to losing to
    # it, inside a file still titled ABL-381 and still reporting PASS.  The gate
    # verdict survives (D-7 is 24.40%) but ABL-381 section 3's reference
    # comparison inverts.
    "abl316-t1b": LEGACY_FEATURE_COLUMNS,
    # ABL-405 (`abl316-t2a`) is deliberately **absent** and takes
    # `DEFAULT_SCOPE_FEATURES` -- the current 27.  Fitting the tranche at 27 was
    # the sole gate on re-tranching the remaining solar pairs, so inheriting the
    # default here is the intended path and not an omission, and this comment is
    # what makes the two distinguishable: this table is one of the two
    # `check_registration_tables` does *not* check -- and cannot, since this very
    # absence is correct -- so an absence defaults silently.  The
    # run records the resolved value either way -- `meta.feature_set`,
    # `meta.n_features` and `meta.feature_set_is_registered_for_scope`, which
    # reads False for this scope and prints as such in the report.
    #
    # This tranche's read is now dispositioned, so the position this comment
    # anticipated -- a published read with no row here, silently re-based by a
    # later move of `FEATURE_COLUMNS` -- has arrived.  **ABL-404 covers it, and
    # still without a row here**, which is why the absence above survives.
    #
    # The objection ABL-404 had to answer was recorded here: pinning a row to
    # `FEATURE_COLUMNS` would not fix anything, because that binds to the same
    # mutable constant `LEGACY_FEATURE_COLUMNS` is derived from, and a real pin is
    # a literal column tuple.  It is -- and this scope already has one.  Its
    # committed machine record carries `meta.feature_columns`, the 27 literal
    # names this tranche was actually fitted on, written by the run itself.
    # `test_a_dispositioned_scope_still_resolves_to_the_list_it_was_read_on` reads
    # that list back out of the evidence and asserts `features_for` still resolves
    # to it, so moving `FEATURE_COLUMNS` to 28 fails the suite here rather than
    # re-basing this read.  A registration table cannot be a better witness of
    # what a fit consumed than the record the fit wrote.
    #
    # So the rule the guard enforces is not "every scope must be pinned" -- that
    # would fail on this row, which is the whole point of the tranche.  It is
    # "every scope whose evidence is committed must still resolve to the list that
    # evidence was taken on", which this row satisfies by inheriting the default,
    # and which `abl316-t1b` above did not.
    #
    # ABL-419 (`abl316-t2c`) is **absent for the same reason and deliberately**,
    # and this comment is the registration of that absence -- the issue asked for
    # all three silent tables to be registered explicitly, and on this one the
    # explicit registration is a stated absence rather than a row.  Writing
    # `"abl316-t2c": FEATURE_COLUMNS` would look like a pin and would not be one:
    # it binds to the same mutable constant `DEFAULT_SCOPE_FEATURES` already binds
    # to, so it protects nothing that the default does not, while flipping
    # `meta.feature_set_is_registered_for_scope` to True -- printing a claim about
    # this registration that is not true of it.  ABL-404 made exactly that argument
    # about this table and it is the reason the 2a row above does not exist either.
    #
    # What actually pins this read is the record the fit writes:
    # `meta.feature_columns` in `experiments/ABL348/results_abl419_tranche2c.json`
    # carries the 27 literal names, and
    # `test_a_dispositioned_scope_still_resolves_to_the_list_it_was_read_on` derives
    # its scope list from `SCOPE_OUTPUTS` and holds each published read to its own
    # recorded names.  So this scope is covered on the commit that publishes it,
    # with nothing hand-maintained here, and moving `FEATURE_COLUMNS` to 28 fails
    # the suite rather than silently re-basing fifteen dispositioned cells.
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
    # ABL-419, registered for the same reason the two above are: `title_for`'s
    # fallback would head a 15-cell evidence pack "abl316-t2c", and a scope slug is
    # a key, not a title.  The heading names the source table and the feature set
    # because those are the two things that distinguish one solar gate read from
    # another once the windows are frozen, and a reader quoting the H1 should not
    # have to reach `meta.training_source` to know which table the challenger was
    # fitted and scored on.
    "abl316-t2c": "ABL-419 — Serve-faithful solar retrain gate, ABL-316 tranche 2c: 5 Mediterranean countries on energy_generation at 27 features",
    # ABL-421, registered for the same reason the three above are: `title_for`'s
    # fallback would head this tranche's evidence pack "abl316-t2d", and a scope
    # slug is a key, not a title.  The heading names the evaluable cell count
    # because that is the one thing about this read a reader must not have to
    # discover: 6 countries x 3 bands is 18, and this scope reads 14 of them.
    "abl316-t2d": "ABL-421 — Serve-faithful solar retrain gate, ABL-316 tranche 2d: 6 northern countries on energy_generation at 27 features, 14 evaluable cells of 18",
}

# ABL-421.  Which country-band cells the frozen registration declares
# **NOT-EVALUABLE**, and therefore refuses to have scored.
#
# This is a registration table like `FIT_RULES` and `SCOPE_TITLES` -- a scope that
# registers nothing gets `{}` through `not_evaluable_for` and behaves exactly as
# every scope did before this table existed.  It is deliberately **not** in
# `check_registration_tables` for `SCOPE_FEATURES`' reason: an empty entry is the
# correct and overwhelmingly common case, so requiring one would raise `KeyError`
# at import for every scope whose absence is right, taking `--help` and the whole
# suite with it.
#
# It exists because ABL-348 `not_evaluable` states a rule this harness had no way
# to obey:
#
#     "A pair listed here is reported NOT-EVALUABLE on the named bands. It is not
#      a FAIL and must not be counted as one; a gate read that scores it has
#      misread this registration."
#
# `gate_cell` builds a cell for every country-band the run produces rows for and
# marks it `pass: False` when `n` falls under the registered minimum, so on the
# pre-ABL-421 harness EE's and FI's four declared cells would each have arrived as
# an ordinary failed cell and been counted into `passed/18` -- which is precisely
# the misreading quoted above, rendered as a model-quality verdict on a comparison
# the registration forbids. Nothing in the run's exit status would show it. Every
# earlier tranche dodged this by excluding the two pairs (`abl316-t2c` says so in
# as many words); 2d is the tranche they belong to and cannot.
#
# The two pairs are declared for different causes and only one is our doing --
# recorded here because the disposition differs:
#
# - **EE/solar**: 630 of 720 D-7-scorable gate hours. ABL-188 excludes a 44.8h
#   bit-identical zero run, 2026-07-21 00:00 -> 2026-07-22 20:45, present
#   identically in **both** source tables, so it is not caused by the source
#   change and switching back would not recover it. `source_dependent: false`.
# - **FI/solar**: 650 of 720. `energy_generation` holds 663 of the 720 gate hours
#   against `energy_renewable`'s 717 -- the ABL-322 section 3.3 phenomenon on a
#   second pair. `source_dependent: **true**`: this one *is* a cost of the
#   ABL-348 source change, and is a finding for whoever owns that decision rather
#   than a fact about FI's solar model.
#
# Both are declared only on the two bands whose registered minimum n is 684
# (`registered_minimum_n`), because `n_d7_scorable` bounds those two directly.
# **48-64h is deliberately absent for both**, and that is ABL-348's instruction
# rather than an omission -- `not_evaluable.note_48_64h`: that band selects a
# 480-510 row subset, so its n scales proportionally rather than being hard
# bounded, and "a pair declared here may still clear 456 in that band and should
# be reported if it does". So those two cells are read, on the bar, like any
# other. Whether they clear 456 is a measurement this run makes, not an
# assumption it encodes: proportionally EE projects to ~420 and FI to ~433,
# both under 456, and if a cell lands there it is a **coverage shortfall**
# (`enough_pairs: False`) and not a loss to D-7 -- the cell dict carries the two
# flags separately and the report prints both, so the distinction survives into
# the evidence pack.
SCOPE_NOT_EVALUABLE = {
    "abl316-t2d": {
        ("EE", "24-36h"), ("EE", "36-48h"),
        ("FI", "24-36h"), ("FI", "36-48h"),
    },
}

#: One short line per declared country, for the report's NOT-EVALUABLE table.
#: These are a **restatement** of `not_evaluable.pairs[*].cause` in
#: `experiments/ABL348/config.json`, and a restatement is exactly the thing that
#: drifts from its source -- so
#: `tests/test_abl421_not_evaluable.py::test_the_causes_match_the_frozen_registration`
#: reads the frozen config and holds these to it, including the
#: `source_dependent` flag, which is the half that decides whose problem each one
#: is. Do not edit one without the other; the config is the original.
NOT_EVALUABLE_CAUSES = {
    "EE": ("ABL-188 excludes a 44.8h bit-identical zero run (2026-07-21 -> 2026-07-22), "
           "present identically in **both** source tables; not source-dependent"),
    "FI": ("`energy_generation` holds 663 of 720 gate hours against `energy_renewable`'s 717 "
           "(the ABL-322 s3.3 phenomenon); **source-dependent**"),
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


def not_evaluable_for(scope: str) -> frozenset:
    """The (country, band) cells the registration declares NOT-EVALUABLE (ABL-421).

    Empty for every scope that registers nothing, which is the pre-ABL-421
    behaviour and the common case. Returned as a frozenset so a caller cannot
    mutate the registration table it was read from.
    """
    return frozenset(SCOPE_NOT_EVALUABLE.get(scope, ()))


# ABL-387: the registration tables above are one registration in five views.
# Checked at import, so a scope registered in one and not the others fails before
# any fit -- and identically under `--help` and in the test suite -- rather than
# raising `KeyError` partway through a gate run, or writing over another scope's
# evidence.
#
# ABL-429: `FIT_RULES` and `SCOPE_TITLES` are now in this check.  They were
# deliberately excluded until ABL-419 merged: adding a required table raises on
# import for every branch already in flight.  That window is closed.  The repo
# queues are at zero, and an absent `FIT_RULES` or `SCOPE_TITLES` row is an
# undocumented choice with no self-documenting degradation -- so they are enforced.
#
# **The call below names five tables, and this file carries seven.**  The two not
# in the call are excluded for stated structural reasons, not oversight:
#
# - `SCOPE_FEATURES` **cannot** join this call: `abl316-t2a` is deliberately absent
#   from it (inheriting the current `FEATURE_COLUMNS` is the intended path for a
#   new tranche, ABL-404), so adding it here would raise `KeyError` at import for
#   a scope whose absence is correct and published.
#   `test_a_published_read_that_recorded_its_own_list_needs_no_scope_features_row`
#   pins that absence and the two would fail against each other.
#
# - `SCOPE_NOT_EVALUABLE` defaults *toward scoring*: a scope that forgets it scores
#   every cell it can build, which for a pair ABL-348 declares NOT-EVALUABLE is a
#   wrong verdict, not self-documenting degradation.  `tests/test_abl421_not_evaluable.py`
#   holds the line for the one scope that registers it, cross-derived from the
#   pre-registration rather than restated, so the test cannot drift from the declaration.
check_registration_tables(SCOPES=SCOPES, GATE_BASIS=GATE_BASIS, SCOPE_OUTPUTS=SCOPE_OUTPUTS,
                          FIT_RULES=FIT_RULES, SCOPE_TITLES=SCOPE_TITLES)
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


def not_evaluable_table(result: dict) -> list[str]:
    """The cells the registration declared unscorable, with what they measured.

    Renders nothing at all for a scope that declares none, so every report
    already published is byte-unchanged by this function existing (ABL-421).

    The numbers are printed deliberately. A declaration the reader cannot check
    is indistinguishable from a challenger quietly dropped for scoring badly, and
    the whole value of declaring before the fit is that the check is available
    afterwards. What the numbers may **not** do is carry a verdict: there is no
    gate column and no grade here, because ABL-348 forbids counting these cells
    either way, and a PASS/FAIL in this table would be that count in all but
    name.
    """
    cells = result.get("not_evaluable_cells") or []
    if not cells:
        return []
    declared = result["meta"].get("not_evaluable_declared_by", "the registration")
    lines = [
        "", "## Cells the registration declares NOT-EVALUABLE", "",
        f"Declared by `{declared}` **before any fit existed**, and excluded from the "
        f"{result['meta']['registered_cells']}-cell bar above. ABL-348's rule: *\"A pair listed here is "
        "reported NOT-EVALUABLE on the named bands. It is not a FAIL and must not be counted as one; a "
        "gate read that scores it has misread this registration.\"* These rows are therefore measured and "
        "shown, but carry no gate outcome and no grade, and are counted neither as passes nor as failures.",
        "",
        "The cause is per pair and only one of the two is ours: EE's shortfall is an ABL-188 "
        "bit-identical zero run present in **both** source tables (`source_dependent: false`), so it "
        "would not be recovered by reverting the source; FI's is `energy_generation` holding fewer gate "
        "hours than `energy_renewable` (`source_dependent: **true**`), which is a cost of ABL-348's "
        "source change and a finding for whoever owns that decision rather than a fact about FI's model.",
        "",
        "| country | horizon | n | registered min n | challenger WAPE | D-7 WAPE | skill vs D-7 | declared cause |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    causes = result["meta"].get("not_evaluable_causes", {})
    for row in sorted(cells, key=lambda r: (r["country"], r["horizon_band"])):
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        lines.append(
            f"| {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{row['gate']['minimum_n']:,} | {_fmt(chal, '%')} | {_fmt(naive, '%')} | {skill} | "
            f"{causes.get(row['country'], 'see the registration')} |")
    return lines


def render_markdown(result: dict) -> str:
    meta, cells = result["meta"], result["gate_cells"]
    passed = sum(cell["gate"]["pass"] for cell in cells)
    # ABL-437: read the levelling from the *record*, not from `CAUSAL_LEVELLING`,
    # so re-rendering a stored read cannot re-decide it under a later
    # registration. A record written before ABL-437 has no such key and is
    # `fit_window` -- absence dates the read, as ABL-404 reads a missing list.
    levelling = meta.get("causal_levelling", FIT_WINDOW)
    # ABL-444, same rule and same reason: a record with no `g23_readability` key
    # was decided by a sign test, so re-rendering it must not floor it.
    readability = meta.get("g23_readability", SIGN_TEST)
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
        *grading_prose(GRADE_STREAM, levelling=levelling, g23_readability=readability),
        "",
        "| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | level inflation (causal / 28d) | incumbent WAPE | MAE | bias | slope | corr | gate | grade |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in cells:
        scores = row["scores"]
        chal, naive = scores["challenger"]["wape_pct"], scores["seasonal_naive"]["wape_pct"]
        # A cell that scored no rows has None on both sides. It renders as
        # "Not measured", never as a number and never as a crash.
        skill = "Not measured" if chal is None or naive is None else f"{100 * (1 - chal / naive):+.1f}%"
        grade = cell_grade(row, GRADE_STREAM, levelling=levelling,
                           g23_readability=readability).detail
        lines.append(
            f"| {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill} | {_fmt(comparator_wape(scores, 'constant_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_causal_28d'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_oracle'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal_28d'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_oracle'), '%')} | "
            f"{_fmt(level_inflation(scores), '%')} / {_fmt(level_inflation(scores, 'constant_causal_28d'), '%')} | "
            f"{_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} | {grade} |"
        )
    lines.extend(grade_summary_table(cells, GRADE_STREAM, lambda row: row["country"],
                                     levelling=levelling, g23_readability=readability))
    lines.extend(not_evaluable_table(result))
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
        # ABL-421. This table pools every primary band, so for a country with a
        # declared cell it pools rows the gate above deliberately did not score.
        # That is tolerable for a reported aggregate and intolerable unsaid: a
        # reader comparing this row to the gate table would otherwise find an n
        # that reconciles with neither the bar nor the declaration.
        *([f"**Pooling caveat.** This is a reported aggregate over *all* primary bands and is not a gate read. "
           f"For {', '.join(sorted({d['country'] for d in meta['not_evaluable_cells_declared']}))} it therefore "
           "pools the band(s) the registration declares NOT-EVALUABLE, so the row is not the pooled form of that "
           "country's gate cells and must not be quoted as one.", ""]
          if meta.get("not_evaluable_cells_declared") else []),
        "| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["country_d2"]:
        scores, tso = row["scores"], row["tso"]
        lines.append(
            f"| {row['country']} | {row['n']:,} | {_fmt(scores['challenger']['wape_pct'], '%')} | "
            f"{_fmt(scores['seasonal_naive']['wape_pct'], '%')} | {_fmt(scores['persistence']['wape_pct'], '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_causal'), '%')} | {_fmt(comparator_wape(scores, 'constant_causal_28d'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'constant_oracle'), '%')} | {_fmt(comparator_wape(scores, 'climatology_causal'), '%')} | "
            f"{_fmt(comparator_wape(scores, 'climatology_causal_28d'), '%')} | {_fmt(comparator_wape(scores, 'climatology_oracle'), '%')} | "
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
    # ABL-421: the bar is the scope's *evaluable* cells.  Still derived from the
    # registration tables and never a literal (`test_solar_bar_is_derived_...`);
    # what changed is that the grid is no longer necessarily complete, because
    # ABL-348 can declare a country-band unscorable before any fit exists.
    # Empty for every scope that registers nothing, so this is an identity for all
    # six scopes that predate it.
    not_evaluable = not_evaluable_for(args.scope)
    registered_cells = (len(registered_countries) * len(PRIMARY_BANDS)
                        - len(not_evaluable))
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
    gate_cells, not_evaluable_cells, country_d2 = [], [], []
    for (country, band), group in all_scored.groupby(["country", "horizon_band"]):
        scores, common, comparator_n = scored(group)
        if band in PRIMARY_BANDS:
            cell = {"country": country, "horizon_band": band, "scores": scores,
                    "comparator_n": comparator_n,
                    "gate": gate_cell(scores["challenger"]["wape_pct"],
                                      scores["seasonal_naive"]["wape_pct"],
                                      len(common), INTENDED_N[band])}
            # ABL-421.  A declared cell is *measured and reported* -- its WAPEs,
            # its n and every comparator are computed exactly as any other cell's
            # -- but it is kept out of `gate_cells`, which is the list `passed`,
            # `disposition` and `attach_grades` all read.  So it cannot be counted
            # as a FAIL (ABL-348's rule), cannot be counted as a PASS, and cannot
            # be graded: a grade on a cell the registration refuses to score would
            # be a disposition by the back door.  Reporting the numbers anyway is
            # what makes the declaration auditable rather than a hole in the pack
            # -- a reader can see what EE and FI *would* have scored and check the
            # n against the cause ABL-348 gives.
            if (country, band) in not_evaluable:
                not_evaluable_cells.append(cell)
            else:
                gate_cells.append(cell)
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
    attach_grades(gate_cells, GRADE_STREAM, levelling=causal_levelling_for(args.scope),
                  g23_readability=g23_readability_for(args.scope))
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
                       "causal_levelling": causal_levelling_for(args.scope),
                       "g23_readability": g23_readability_for(args.scope),
                       # ABL-421: the *evaluable* cell count, which is the grid
                       # minus whatever the registration declared unscorable. The
                       # grid size is recorded beside it rather than left to be
                       # re-derived, because "14 of 18" is the sentence this read
                       # has to survive being quoted as, and a reader who
                       # multiplies 6 x 3 and finds 18 must be able to see where
                       # the other four went without opening the harness.
                       "registered_cells": registered_cells,
                       "registered_grid_cells": len(registered_countries) * len(PRIMARY_BANDS),
                       "not_evaluable_cells_declared": [
                           {"country": c, "horizon_band": b}
                           for c, b in sorted(not_evaluable)],
                       "not_evaluable_declared_by":
                           "experiments/ABL348/config.json -> not_evaluable.pairs",
                       # Scoped to the countries this scope actually declares, so
                       # a record cannot carry a cause for a country it never
                       # declined to score.
                       "not_evaluable_causes": {
                           country: NOT_EVALUABLE_CAUSES[country]
                           for country in sorted({c for c, _ in not_evaluable})
                           if country in NOT_EVALUABLE_CAUSES},
                       "gate_basis": list(gate_basis),
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
              # ABL-421: measured, recorded, and kept out of every count. An empty
              # list for the six scopes that predate this table, so their records
              # gain one empty key and nothing else.
              "not_evaluable_cells": sorted(not_evaluable_cells,
                                            key=lambda row: (row["country"], row["horizon_band"])),
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
