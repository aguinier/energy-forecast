"""ABL-418: the graded gate disposition (G1-G4), shared by both gate harnesses.

The registered bar is **not** re-opened here. Seasonal-naive D-7 stays the gate
for every scope already dispositioned and every scope still to come; ABL-348's
frozen windows, bands, metric, minimum n and source are untouched, and a cell
that clears D-7 still reads ``PASS`` in the gate column. What this module adds is
**what a PASS entitles a cell to**.

Why it exists. ABL-406 measured, across eight ``wind_onshore`` pairs, that the
gate outcome was *fully* predicted by whether a causal constant clears the
registered bar on its own -- five weak bars gave five passes, three strong bars
gave three failures or ties, no exceptions -- and that NO passed 3/3 while
**anti-correlated with its own target** (slope -0.08, correlation -0.14). A pair
can therefore clear the registered bar on level alone, carrying no directional
information at all. So a PASS is *necessary and not sufficient* for a promotion
recommendation. Tightening the bar after seeing that result would be shopping the
registration; grading the pass is not, which is why this is a registration change
pre-registered on ABL-418 before any remaining tranche is fitted.

The ladder, per cell (country x band), computed from columns the gate table
already prints -- no new baseline, no new fit, no second implementation:

``G1`` *gate*
    the challenger beats ``seasonal_naive`` by more than the deterministic
    readability floor.
``G2`` *level*
    the challenger beats ``constant_causal`` -- a flat line at the fit-window
    mean. Does it predict the level?
``G3`` *shape*
    the challenger beats ``climatology_causal`` -- an hour-of-day mean over the
    fit window. Does it predict the level *and* the daily shape?
``G4`` *direction*
    ``slope > 0`` **and** ``correlation > 0``. Does it carry any directional
    information about its own target?

and the grades built from it:

==========  ==========================================================
``A``       G1-G4 hold in every band. Promotion-eligible, subject to any
            named data hold.
``B``       G1 holds; one or more of G2/G3/G4 fails. The failures are
            named. Not promotion-eligible.
``C``       G1 fails readably -- the challenger loses to the registered
            bar by more than the floor.
``U``       the G1 margin sits inside the readability floor: unreadable
            at one seed. ``U(+)`` where G2-G4 clear readably, in which
            case the disposition is *re-read at k>1 seeds* per ABL-385,
            not *reject*.
``N``       (ABL-444, ``floored`` scopes only) G1 holds readably, nothing
            below it fails readably, and at least one of G2/G3 sits
            inside the floor. Not demonstrated **in either direction**;
            not promotion-eligible.
==========  ==========================================================

**``U`` takes precedence over ``C``.** Both are "G1 does not hold", but they are
different statements: ``C`` is a measured loss, ``U`` is an absence of
measurement. Calling an unreadable cell a failure invites the same wrong next
move ABL-378's ``UNREADABLE`` verdict exists to prevent.

**The floor.** ABL-385 registers ``delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) /
sqrt(k)`` as the minimum readable relative gap, where ``c`` is a fit's per-seed
CV. Every reference on this ladder is **deterministic** -- D-7, a flat line and
an hour-of-day climatology do not move when the challenger is refitted -- so
``c_B = 0`` and the published two-arm margin is a factor of ``sqrt(2)`` too wide.
The floor is therefore ``1.96 * c_A / sqrt(k)``, which at the fleet p90 CV and
one fit per cell is **10.65% on solar** and **7.51% on wind**. Quoting the
two-arm 15.06% against a constant is not conservatism, it is the wrong test.

**Which denominator.** ``G1`` is tested on the ``skill vs D-7`` column the gate
table already prints, ``100 * (1 - challenger / reference)``, because that is
what ABL-418 registered and what a reader can check against the report by eye.
ABL-406 quoted its margins on the challenger's *own* error instead,
``100 * (reference - challenger) / challenger``, which is the denominator
ABL-385's CV is measured in. The two always agree in sign and differ only in
magnitude, so they can disagree only for a cell sitting near the floor;
:func:`margin_pct_of_own_error` computes the second form so a read can report the
sensitivity rather than assume it away. Measured over tranches 2a and 2b on
2026-08-13, **no cell of the 48 changes grade** between them.

**The oracle references stay reported and never gating**, exactly as ABL-389
registered them. An oracle is not causally available, so losing to one bounds
what a verdict means rather than voiding it -- and the bar-weakness flag
(:func:`bar_weaker_than_a_flat_line`) is reported for the same reason. Neither
appears in the ladder.

**A condition that cannot be evaluated is not satisfied.** A comparator that
scored no rows reads ``Not measured``, and the cell it belongs to cannot grade
``A`` on a criterion nobody measured -- the net-position gate's ``INCOMPLETE``
rule, one level down. Such a condition is named in ``failed`` like any other.

ABL-437: which causal references G2 and G3 read
-----------------------------------------------

G2 and G3 were registered on ``constant_causal`` and ``climatology_causal``,
which are levelled on the **fit** window and scored on the **gate** window.
ABL-437 measured what that costs on a seasonal series: across every committed
tranche record the causal constant runs up to **205% worse** than the
correctly-levelled oracle constant (NL ``wind_onshore``, 225.54% against
73.85%), with nine more wind pairs between 21% and 102% and every solar pair
between 0% and 8%. A reference inflated that far is a strawman, and G2/G3 clear
it for free -- the third instance of the ABL-406 / ABL-417 / ABL-435 pattern.

The amendment keeps both conditions on the ladder and **re-levels the reference
they read**, to ``constant_causal_28d`` / ``climatology_causal_28d``: the same
two predictors over the 28 days ending at each row's own ``generated_at``. The
alternative considered and rejected was flagging G2/G3 not evaluable outside a
registered level band, which discards a measurable number to buy an abstention
and abstains on a pair that beats the corrected references as readily as on one
that loses to them (``reports/abl_437_causal_levelling_registration.md`` has the
comparison).

Two properties are load-bearing and are what make this an amendment rather than
a re-registration:

* **Which pair a scope reads is registered, not global.** ``CAUSAL_LEVELLING``
  in each harness pins every already-published scope to
  ``model_free_reference.FIT_WINDOW``, so a dispositioned read still reproduces
  and every letter already published still means what it meant. New scopes
  default to ``TRAILING_28D``.
* **The registered bar is still not re-opened.** G1 is seasonal-naive D-7,
  unchanged; ABL-348's windows, bands, metric, baseline, minimum n and source
  are untouched, so this is not a change ``voids_this_registration`` names. Both
  oracle references stay reported and on no ladder.

ABL-444: a readability floor on G2 and G3
------------------------------------------

G1 carries a floor; G2 and G3 were registered as bare sign tests, ``skill > 0``.
So a G2/G3 verdict could turn on a margin far inside the spread one seed
resolves, and the ladder reported it with the same letter it uses for a decisive
result. ABL-437's re-read is where it became unignorable: PL solar's G3 flips A
to B on **0.36%** of skill, 3.4% of the solar floor.

**This is not a widening of the ladder's existing floor -- it is the removal of
an inconsistency inside it.** ABL-418 *already* applies
:func:`readability_floor_pct` to G2 and G3, on this same ``skill vs X`` column,
when it decides ``U`` against ``U(+)``: the plus requires G2 and G3 to clear
*readably*. The defect is that the identical margin decides ``A`` against ``B``
on a bare sign, one branch of the same function away.

The third outcome is an **abstention**, not a failure. ``N`` is neither ``U``
(undecided on the *gate*, a different statement) nor ``B`` (a measured failure,
which is precisely what is not being claimed). ``A`` still requires every
condition to hold, so an unreadable condition cannot promote: *a condition that
could not be measured is not satisfied*, the same rule the net-position gate's
``INCOMPLETE`` and ABL-421's ``SCOPE_NOT_EVALUABLE`` state one level down.

Three properties, in the order they were argued:

* **A definite failure outranks an abstention.** A cell whose G4 fails and whose
  G3 is unreadable is ``B``, not ``N``: there is something to report, and
  ``N`` would bury it. Hence ``GRADE_SEVERITY`` places ``N`` between ``A`` and
  ``U`` -- both ``N`` and ``U`` say *re-read at k>1*, and ``U`` is the deeper of
  the two because it cannot resolve the registered gate itself, where ``N`` has
  cleared it readably.
* **The margin prints either way.** The floor decides *gradeability*; it does
  not replace the number, and both the skill form and the challenger's-own-error
  form (ABL-385's denominator) stay in the record. Measured over all 113
  committed cells, the two denominators disagree on the readability of **2 of
  452** condition-observations and on **no cell letter and no pair letter**.
* **It is registered per scope and it cannot raise a grade.** ``N`` is only ever
  reached from what would have been ``A`` or ``B``, so the promotable set shrinks
  or holds. ABL-348's windows, bands, metric, baseline, minimum n and source are
  untouched -- ``voids_this_registration`` is not triggered -- and ABL-434 (the
  ladder cannot see minimum n) is a different guard, deliberately not folded in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.evaluation.model_free_reference import (
    FIT_WINDOW, TRAILING_28D, comparator_wape, level_inflation,
)


#: Two-sided 95% call, matching ABL-385's registration.
Z_95 = 1.96

#: The fleet 90th-percentile per-fit CV registered by ABL-385 (§1), per stream,
#: as a fraction. Read from `reports/abl_385_decision_margin.json`
#: (`fleet_margin.<stream>.cv_rms_p90`) and pinned here so the ladder does not
#: import a report at run time. `tests/test_gate_grading.py` checks both values
#: against that file.
#:
#: `solar` is 16 (pair, algorithm, arm) units, `wind` is 12. The wind value is
#: deliberately *not* the solar one: ABL-381 read its margins against a different
#: stream's fits, and that is the mistake being avoided here.
STREAM_FLEET_CV_P90 = {"solar": 0.05432773918515768, "wind": 0.038292934379344015}

#: The same floors as ABL-405 and ABL-406 published them in prose, to 2 dp. Kept
#: only as a cross-check on the derivation -- `readability_floor_pct` is the
#: number used. They differ from the exact values by under 0.01pp (solar 10.6482,
#: wind 7.5054), and no cell of tranche 2a or 2b sits in that gap.
PUBLISHED_FLOOR_PCT_K1 = {"solar": 10.64, "wind": 7.51}

#: The grade letters, worst first. `pair_grade` takes the worst cell grade under
#: this order, so a pair is never graded better than any band in it.
#:
#: `C` outranks `B` because a readable loss to the registered bar is a stronger
#: negative than losing to a reference the bar never named. `B` outranks `U`
#: because B is a *measured* failure and U is a deferral: a pair with one B band
#: and one U band has something definite to say, and saying "re-read it" would
#: bury it. By the same argument `B` outranks `N` (ABL-444).
#:
#: `N` sits between `A` and `U` because both are deferrals and `U`'s is the
#: deeper one: an `N` cell has cleared the registered gate readably and cannot
#: resolve a condition below it, where a `U` cell cannot resolve the gate. The
#: consequence to know is that a pair with one `A` band and one `N` band grades
#: `N` -- a pair is never graded better than any band in it, and an abstention
#: in any band is enough to stop it being promotion-eligible.
GRADE_SEVERITY = {"A": 0, "N": 1, "U": 2, "B": 3, "C": 4}

#: ABL-444: whether G2 and G3 are decided by a bare sign test -- `skill > 0`, as
#: ABL-418 registered them -- or by the same readability floor G1 carries.
#:
#: `sign_test` is what every published letter was decided under, and pinning it
#: per scope is what leaves those reads standing. `floored` is the amendment.
SIGN_TEST = "sign_test"
FLOORED = "floored"

#: Both forms, for a caller validating a registration table. `floored` is the
#: default for a scope that registers nothing; see the harness tables for why
#: that direction, and `reports/abl_444_g23_readability_floor_registration.md`
#: for the argument against the opposite one.
G23_READABILITY_FORMS = (SIGN_TEST, FLOORED)

#: The two causal references G2 and G3 are scored against, per registered
#: levelling form (ABL-437). The oracle references appear in neither entry and
#: are on no ladder, which is the property ABL-389 registered and this amendment
#: does not touch.
LADDER_REFERENCES = {
    FIT_WINDOW: {"G2": "constant_causal", "G3": "climatology_causal"},
    TRAILING_28D: {"G2": "constant_causal_28d", "G3": "climatology_causal_28d"},
}

#: How each levelling form describes its two references, for the report. Kept
#: beside `LADDER_REFERENCES` so a column name and the sentence explaining it
#: cannot drift apart.
_REFERENCE_PROSE = {
    FIT_WINDOW: {"G2": "beats constant_causal -- a flat line at the fit-window mean",
                 "G3": "beats climatology_causal -- an hour-of-day mean over the fit window"},
    TRAILING_28D: {"G2": "beats constant_causal_28d -- a flat line at the mean of the 28 days "
                         "ending at the forecast issue instant",
                   "G3": "beats climatology_causal_28d -- an hour-of-day mean over the same "
                         "trailing 28 days"},
}


def conditions_for(levelling: str = TRAILING_28D) -> tuple:
    """What each ladder condition asks, for the report, under one levelling.

    The letter is the identity used in `failed`, so a named failure and a table
    column cannot drift apart -- and since ABL-437 the *reference* a letter
    names depends on the scope's registered levelling, so the question text has
    to be derived rather than written down once.
    """
    prose = _REFERENCE_PROSE[levelling]
    return (
        ("G1", "gate", "beats seasonal_naive D-7 by more than the readability floor"),
        ("G2", "level", prose["G2"]),
        ("G3", "shape", prose["G3"]),
        ("G4", "direction", "slope > 0 and correlation > 0"),
    )


#: The pre-ABL-437 constant, kept as the fit-window rendering so a caller that
#: never had a levelling to pass still reads the ladder it was written against.
CONDITIONS = conditions_for(FIT_WINDOW)


def readability_floor_pct(stream: str, k: int = 1) -> float:
    """The deterministic readability floor, in percent, for one stream at k seeds.

    ``1.96 * c_A / sqrt(k)`` -- ABL-385's ``delta_min`` with ``c_B = 0``, which
    is the correct form against a reference that does not move when the
    challenger is refitted. That is every reference on this ladder.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return Z_95 * STREAM_FLEET_CV_P90[stream] * 100.0 / math.sqrt(k)


def skill_pct(challenger, reference):
    """``100 * (1 - challenger / reference)`` -- the printed ``skill vs X`` column.

    ``None`` where either side was not measured, or where the reference scored
    zero error and the ratio is undefined. Positive means the challenger wins.
    """
    if challenger is None or reference is None or not reference:
        return None
    return 100.0 * (1.0 - challenger / reference)


def margin_pct_of_own_error(challenger, reference):
    """The same gap as a percentage of the **challenger's own** error.

    The denominator ABL-385's CV is measured in, and the one ABL-406 quoted its
    margins on. Reported beside :func:`skill_pct` as a sensitivity, never
    substituted for it: the ladder registered the printed column.
    """
    if challenger is None or reference is None or not challenger:
        return None
    return 100.0 * (reference - challenger) / challenger


def bar_weaker_than_a_flat_line(scores: dict):
    """Does ``constant_causal`` clear the registered D-7 bar on its own?

    ABL-380 measured it first (BG's 93.75% bar cleared by a constant at 82.77%,
    no model), and ABL-406 found the gate outcome fully predicted by it across
    eight pairs. Reported beside every grade and gating nothing -- it is a
    property of the *bar*, not of the challenger.

    ``None`` where either was not measured.
    """
    # `or {}` rather than a default, matching `comparator_wape`: a record that
    # carries the key with a null score and one that omits it entirely both mean
    # "not measured", and neither may raise.
    bar = (scores.get("seasonal_naive") or {}).get("wape_pct")
    constant = comparator_wape(scores, "constant_causal")
    if bar is None or constant is None:
        return None
    return bool(constant < bar)


@dataclass(frozen=True)
class CellGrade:
    """One cell's grade, with everything needed to check it by hand."""

    grade: str | None
    #: ``True`` only on a ``U`` cell whose G2-G4 clear readably -- ``U(+)``.
    plus: bool = False
    #: Every condition's outcome, keyed ``G1``..``G4``. ``None`` = not measured.
    conditions: dict = field(default_factory=dict)
    #: The conditions that did not hold, in ladder order, each with its reason.
    failed: tuple = ()
    #: ``skill vs X`` for each of the three scored references, in percent.
    skill: dict = field(default_factory=dict)
    #: The same gaps on the challenger's own error, as a sensitivity.
    own_error_margin: dict = field(default_factory=dict)
    floor_pct: float = 0.0
    bar_weak: bool | None = None
    #: Which pair of causal references G2/G3 were read against (ABL-437).
    levelling: str = FIT_WINDOW
    #: The causal constant's WAPE over the oracle constant's, in percent, per
    #: causal reference. Reported, never gating -- a property of the reference.
    level_inflation_pct: dict = field(default_factory=dict)
    #: Whether G2/G3 were decided by a sign test or against the floor (ABL-444).
    g23_readability: str = SIGN_TEST
    #: The conditions whose margin sits inside the floor, in ladder order, each
    #: with its reason. Always empty under ``sign_test``. Distinct from
    #: ``failed``: an abstention is not a failure, and neither is it a pass.
    not_readable: tuple = ()

    @property
    def label(self) -> str:
        """``A`` / ``B`` / ``C`` / ``N`` / ``U`` / ``U(+)``, or ``Not measured``."""
        if self.grade is None:
            return "Not measured"
        return "U(+)" if self.plus else self.grade

    @property
    def detail(self) -> str:
        """The label with its failed and unreadable conditions named."""
        parts = []
        if self.failed:
            parts.append(f"fails {', '.join(name for name, _ in self.failed)}")
        if self.not_readable:
            parts.append(f"not readable on {', '.join(name for name, _ in self.not_readable)}")
        return f"{self.label} — {'; '.join(parts)}" if parts else self.label

    def as_dict(self) -> dict:
        return {"grade": self.grade, "plus": self.plus, "label": self.label,
                "conditions": dict(self.conditions),
                "failed": [{"condition": name, "reason": reason} for name, reason in self.failed],
                "not_readable": [{"condition": name, "reason": reason}
                                 for name, reason in self.not_readable],
                "skill_pct": dict(self.skill),
                "own_error_margin_pct": dict(self.own_error_margin),
                "floor_pct": self.floor_pct, "bar_weaker_than_a_flat_line": self.bar_weak,
                "causal_levelling": self.levelling,
                "level_inflation_pct": dict(self.level_inflation_pct),
                "g23_readability": self.g23_readability}

    @classmethod
    def from_dict(cls, record: dict) -> "CellGrade":
        """Rebuild a grade a run already wrote, rather than recomputing it.

        A stored grade is the record of what that read decided. Re-deriving it
        from the scores would be a second implementation of the ladder living in
        the renderer, which is the thing this module exists to prevent.
        """
        return cls(grade=record["grade"], plus=record.get("plus", False),
                   conditions=dict(record.get("conditions") or {}),
                   failed=tuple((item["condition"], item["reason"])
                                for item in record.get("failed") or ()),
                   not_readable=tuple((item["condition"], item["reason"])
                                      for item in record.get("not_readable") or ()),
                   skill=dict(record.get("skill_pct") or {}),
                   own_error_margin=dict(record.get("own_error_margin_pct") or {}),
                   floor_pct=record.get("floor_pct", 0.0),
                   bar_weak=record.get("bar_weaker_than_a_flat_line"),
                   levelling=record.get("causal_levelling", FIT_WINDOW),
                   level_inflation_pct=dict(record.get("level_inflation_pct") or {}),
                   g23_readability=record.get("g23_readability", SIGN_TEST))


def cell_grade(cell: dict, stream: str, k: int = 1,
               levelling: str = TRAILING_28D,
               g23_readability: str = FLOORED) -> CellGrade:
    """One cell's grade: the one the run recorded, or computed if it recorded none.

    The fallback is what lets a ``results.json`` written before ABL-418 be
    graded — which is the whole of the tranche 2a/2b retro-grade — and what lets
    a stored read be re-rendered without re-deciding it.

    A record written before ABL-437 carries no ``causal_levelling`` key and
    rebuilds as :data:`FIT_WINDOW`, and one written before ABL-444 carries no
    ``g23_readability`` key and rebuilds as :data:`SIGN_TEST`, because those are
    what they were decided under. Absence dates the read; it is not a default
    anyone chose after the fact.
    """
    recorded = cell.get("grade")
    if recorded:
        return CellGrade.from_dict(recorded)
    return grade_cell(cell["scores"], stream, k, levelling, g23_readability)


def scored_conditions(levelling: str = TRAILING_28D) -> tuple:
    """The reference each condition is scored against, and the letter it answers.

    ``G4`` is a sign test on the challenger alone and is not here. ``G1`` is
    seasonal-naive D-7 under every levelling: ABL-437 re-levels the two causal
    *references*, and re-levelling the registered bar would be re-opening it.
    """
    references = LADDER_REFERENCES[levelling]
    return (("G1", "seasonal_naive"), ("G2", references["G2"]), ("G3", references["G3"]))


def grade_cell(scores: dict, stream: str, k: int = 1,
               levelling: str = TRAILING_28D,
               g23_readability: str = FLOORED) -> CellGrade:
    """Grade one gate cell from the scores the harness already computed.

    ``scores`` is a cell's ``scores`` mapping as written to ``results.json``:
    ``challenger``, ``seasonal_naive`` and the ABL-389/ABL-437 model-free
    references, each a dict with ``wape_pct``, ``slope`` and ``correlation``.

    ``levelling`` selects which causal pair G2 and G3 read (ABL-437), and
    ``g23_readability`` whether they are decided by a sign test or against the
    readability floor (ABL-444). Both are per-scope registrations in the harness,
    never run-time choices: a read that could pick its own references or its own
    decision rule after seeing them is the thing this ladder was pre-registered
    to prevent.
    """
    if g23_readability not in G23_READABILITY_FORMS:
        raise ValueError(f"unknown G2/G3 readability form: {g23_readability!r}")
    floor = readability_floor_pct(stream, k)
    conditions_asked = conditions_for(levelling)
    scored = scored_conditions(levelling)
    challenger = (scores.get("challenger") or {}).get("wape_pct")
    skill = {name: skill_pct(challenger, comparator_wape(scores, name))
             for _, name in scored}
    own = {name: margin_pct_of_own_error(challenger, comparator_wape(scores, name))
           for _, name in scored}
    inflation = {name: level_inflation(scores, name)
                 for name in ("constant_causal", "constant_causal_28d")}

    if skill["seasonal_naive"] is None:
        # No gate margin at all: the cell scored nothing, or D-7 did. It is not
        # a C -- nothing lost a race here.
        return CellGrade(grade=None, skill=skill, own_error_margin=own,
                         floor_pct=floor, bar_weak=bar_weaker_than_a_flat_line(scores),
                         levelling=levelling, level_inflation_pct=inflation,
                         g23_readability=g23_readability)

    slope = (scores.get("challenger") or {}).get("slope")
    correlation = (scores.get("challenger") or {}).get("correlation")
    directional = (None if slope is None or correlation is None
                   else bool(slope > 0 and correlation > 0))
    level, shape = (name for _, name in scored if name != "seasonal_naive")
    conditions = {
        "G1": bool(skill["seasonal_naive"] > floor),
        "G2": None if skill[level] is None else bool(skill[level] > 0),
        "G3": None if skill[shape] is None else bool(skill[shape] > 0),
        "G4": directional,
    }
    unreadable = abs(skill["seasonal_naive"]) <= floor

    # ABL-444: under `floored`, a G2/G3 margin inside the floor is neither a pass
    # nor a failure. `G4` is a sign test on the challenger's own slope and
    # correlation -- there is no margin to read against a floor -- so it is never
    # routed here, under either form.
    reference_of = {"G2": level, "G3": shape}
    failed, not_readable = [], []
    for name, _, question in conditions_asked:
        if name == "G1":
            continue
        if conditions[name] is None:
            failed.append((name, f"not measured ({question})"))
            continue
        margin = skill.get(reference_of.get(name))
        if (g23_readability == FLOORED and margin is not None
                and abs(margin) <= floor):
            not_readable.append((name, f"margin {margin:+.2f}% sits inside the "
                                       f"{floor:.2f}% readability floor ({question})"))
            continue
        if not conditions[name]:
            failed.append((name, question))

    common = {"skill": skill, "own_error_margin": own, "floor_pct": floor,
              "bar_weak": bar_weaker_than_a_flat_line(scores),
              "levelling": levelling, "level_inflation_pct": inflation,
              "g23_readability": g23_readability, "not_readable": tuple(not_readable)}
    if unreadable:
        # G2/G3 must clear *readably* for the plus, on the same floor G1 uses.
        # This test predates ABL-444 and is where the floor was already applied
        # to these two conditions -- the amendment carries it to the other branch
        # rather than introducing it. G4 enters as-is, being a sign test.
        readable = all(skill[name] is not None and skill[name] > floor
                       for name in (level, shape))
        return CellGrade(grade="U", plus=bool(readable and conditions["G4"]),
                         conditions=conditions, failed=tuple(failed), **common)
    if not conditions["G1"]:
        # A readable loss to the registered bar. Nothing below G1 can rescue it,
        # but the other conditions are still recorded rather than skipped.
        return CellGrade(grade="C", conditions=conditions,
                         failed=(("G1", "readable loss to seasonal_naive D-7"), *failed),
                         **common)
    if failed:
        # A definite failure outranks an abstention: there is something to
        # report, and grading it `N` would bury it.
        return CellGrade(grade="B", conditions=conditions, failed=tuple(failed), **common)
    if not_readable:
        return CellGrade(grade="N", conditions=conditions, failed=(), **common)
    return CellGrade(grade="A", conditions=conditions, failed=(), **common)


def attach_grades(cells: list[dict], stream: str, k: int = 1,
                  levelling: str = TRAILING_28D,
                  g23_readability: str = FLOORED) -> list[dict]:
    """Add a ``grade`` block to every gate cell, in place, and return them.

    One call site per harness, right where ``gate_cells`` is assembled, so the
    markdown table and ``results.json`` cannot disagree about a grade. Cells
    written before ABL-418 carry no ``grade`` key; the retro-grade script grades
    them through this same function rather than reimplementing the ladder.
    """
    for cell in cells:
        cell["grade"] = grade_cell(cell["scores"], stream, k, levelling,
                                   g23_readability).as_dict()
    return cells


def pair_grade(cell_grades) -> CellGrade:
    """The grade for a country x band *set* -- the worst band in it.

    ``A`` requires all four conditions in **every** band, which is what makes the
    worst-cell rule the right one rather than a convenience: one unreadable band
    is enough to stop a pair being promotion-eligible. ``U(+)`` survives to the
    pair only if every ``U`` band in it is ``U(+)``; a single plain ``U`` band
    means the pair is not uniformly re-readable.

    An ``N`` band (ABL-444) takes the pair to ``N`` by the same rule, unless some
    other band is worse. **Read that the way it is meant**: an ``A / A / N`` pair
    is not "mostly promotable", it is a pair with a band nobody has resolved.
    """
    grades = [grade for grade in cell_grades if grade.grade is not None]
    if not grades:
        return CellGrade(grade=None)
    worst = max(grades, key=lambda grade: GRADE_SEVERITY[grade.grade])
    if worst.grade != "U":
        return worst
    plus = all(grade.plus for grade in grades if grade.grade == "U")
    return CellGrade(grade="U", plus=plus, conditions=worst.conditions,
                     failed=worst.failed, skill=worst.skill,
                     own_error_margin=worst.own_error_margin,
                     floor_pct=worst.floor_pct, bar_weak=worst.bar_weak,
                     levelling=worst.levelling,
                     level_inflation_pct=worst.level_inflation_pct,
                     g23_readability=worst.g23_readability,
                     not_readable=worst.not_readable)


def grading_prose(stream: str, k: int = 1, levelling: str = TRAILING_28D,
                  g23_readability: str = FLOORED) -> list[str]:
    """The paragraph that says what the grade column is and what it is not."""
    floor = readability_floor_pct(stream, k)
    references = LADDER_REFERENCES[levelling]
    readability = (
        [f"**G2/G3 readability (ABL-444): `{g23_readability}`.** G2 and G3 are decided against the same "
         f"**{floor:.2f}%** floor G1 carries, not by a bare sign test. A margin inside it is **`N` — not readable**: "
         f"neither demonstrated nor refuted at k={k}, an abstention rather than a failure, and not promotion-eligible, "
         f"because a condition that could not be measured is not satisfied. `B` still outranks `N`, so a cell that also "
         f"fails something readably reads `B`. **The margin is printed either way** — the floor decides gradeability, it "
         f"does not replace the number. ABL-418 already applied this floor to G2/G3 when deciding `U` against `U(+)`; the "
         f"amendment carries it to the `A`/`B` branch, where a letter could turn on 0.36% (PL solar's G3)."]
        if g23_readability == FLOORED else
        [f"**G2/G3 readability (ABL-444): `{g23_readability}`.** G2 and G3 are bare sign tests, `skill > 0`, which is what "
         f"this scope was registered and published under. ABL-444 registers a floored form for new scopes — a margin "
         f"inside the **{floor:.2f}%** readability floor grades `N`, not readable — and re-reads the published scopes "
         f"under it separately (`reports/abl_444_g23_floor_reread.md`) rather than restating their letters here."])
    amendment = (
        [f"**Causal levelling (ABL-437): `{levelling}`.** G2 and G3 read `{references['G2']}` and "
         f"`{references['G3']}` — the flat line and the hour-of-day mean over the **28 days ending at each row's own "
         f"forecast issue instant**, not over the whole fit window. The fit-window forms are levelled on "
         f"2026-01-14 → 2026-07-11 and scored on high summer, which on a seasonal series makes them a strawman: measured "
         f"across every committed tranche record, a causal constant runs up to **205% worse** than the correctly-levelled "
         f"oracle constant (NL `wind_onshore`), which passes G2/G3 for free. The trailing form is strictly causal — same "
         f"anchor and same filtered series as the challenger's own `target_value_roll_168h_mean` feature. Both fit-window "
         f"references stay **reported** beside it, and the `level inflation` column prints the residual, so nothing is "
         f"discarded. **G1 is unchanged**: the registered D-7 bar is not re-opened, and no oracle is on the ladder."]
        if levelling == TRAILING_28D else
        [f"**Causal levelling (ABL-437): `{levelling}`.** G2 and G3 read `{references['G2']}` and "
         f"`{references['G3']}` — levelled on the fit window, which is what this scope was registered and published "
         f"under. ABL-437 measured that form to be inflated by up to 205% on a seasonal pair and re-levels it for new "
         f"scopes; this scope keeps the reference its published letters were decided against, and the trailing columns "
         f"are reported beside it so the difference is readable rather than asserted."])
    return [
        f"Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, "
        f"ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. "
        f"What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the "
        f"gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five "
        f"passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own "
        f"target. A PASS is necessary and not sufficient.",
        "",
        f"**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every "
        f"reference here is deterministic, which is **{floor:.2f}%** for this stream at k={k}. **G2** level: beats "
        f"`{references['G2']}`. **G3** shape: beats `{references['G3']}`. **G4** direction: slope > 0 and corr > 0. "
        f"**A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or "
        f"more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the "
        f"cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.",
        "",
        "Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally "
        "available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does "
        "`constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the "
        "ladder. A condition that could not be measured is not satisfied, and is named like any other failure.",
        "",
        *amendment,
        "",
        *readability,
    ]


def grade_summary_table(cells: list[dict], stream: str, key, k: int = 1,
                        levelling: str = TRAILING_28D,
                        g23_readability: str = FLOORED) -> list[str]:
    """The per-pair grade roll-up, under the per-cell table.

    ``key`` maps a cell to its pair label, so the wind harness can key on
    ``(forecast_type, country)`` and the solar harness on ``country`` alone.
    """
    if not cells:
        return []
    order, pairs = [], {}
    for cell in cells:
        label = key(cell)
        if label not in pairs:
            order.append(label)
            pairs[label] = []
        pairs[label].append(cell_grade(cell, stream, k, levelling, g23_readability))
    lines = ["", "### Graded disposition, per pair", "",
             "| pair | bands | grade | failed conditions | not readable | "
             "bar weaker than a flat line? |",
             "|---|---|:---:|---|---|:---:|"]
    for label in order:
        grades = pairs[label]
        overall = pair_grade(grades)
        bands = " / ".join(grade.label for grade in grades)
        reasons = ", ".join(f"{name} ({reason})" for name, reason in overall.failed) or "—"
        # Unioned across bands, unlike `failed`: an abstention in any band is why
        # the pair is not promotion-eligible, and naming only the worst band's
        # would hide the others behind a letter that already says "re-read this".
        abstained = {name: reason for grade in grades for name, reason in grade.not_readable}
        unreadable = ", ".join(f"{name} ({reason})" for name, reason in abstained.items()) or "—"
        weak = [grade.bar_weak for grade in grades]
        flag = ("Not measured" if all(value is None for value in weak)
                else "yes" if any(value for value in weak) else "no")
        lines.append(f"| {label} | {bands} | **{overall.label}** | {reasons} | {unreadable} | {flag} |")
    return lines
