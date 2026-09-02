"""ABL-623: ABL-246's pack and the ABL-607 re-read may not drift apart.

ABL-246 published `reports/abl_246_tso_d1_load_pack.md` with a count in its
headline -- our D+2 load model readably loses to a D-7 seasonal naive in **10**
of 23 countries. ABL-607/ABL-619 then re-ran that exact comparison on a later
replica vintage and got **9**, and landed that on the same `main`. For a while
both numbers were asserted as current, in two files, a directory apart: the
ABL-607 report pointed *up* at the pack (`meta.parent_pack`) and the pack had no
pointer *down* at the correction. A reader arriving at the pack -- which is the
CEO/Board-facing deliverable -- read a superseded number as current.

Re-wording fixed that once. What is pinned here is that it stays fixed, and it
is pinned **in both directions**, because only one of them is the interesting
failure:

1. If the records disagree, the texts must say so and must quote the count the
   record actually holds. A future re-read that moves 9 to 8 turns this red
   until the prose follows, rather than leaving a third stale number behind.
2. If the records ever agree again -- a re-read reproducing the published set --
   a text still announcing a correction is equally wrong. Test 6 asserts the
   disagreement is real, so tests 1-5 can never certify prose by vacuity.

Every number below is **derived from the two machine records**, never written
into this file. `abl246_losers` is the pack's own published set, carried inside
both records, which is what makes the comparison possible without hardcoding
either side of it. Counting is avoided where membership is available: a count
assertion cannot see a country that swapped for another, and the whole point of
this pin is which cell moved.

Not run against the replica. These are text-vs-artifact invariants; a rule
pinned against live data stops being a test the day the data moves -- the same
reasoning as the header of `test_abl607_guarded_read.py`.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

#: The pack under correction, and the two records of the same comparison.
PACK = REPO_ROOT / "reports" / "abl_246_tso_d1_load_pack.md"
PUBLISHED = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis.json"
REREAD = REPO_ROOT / "reports" / "abl_607_d2_load_diagnosis_reread.json"

#: The ABL-607 script's module docstring states the pack's count as its premise,
#: so it is a third text carrying the same claim and is held to the same record.
DIAGNOSIS_SCRIPT = REPO_ROOT / "scripts" / "abl607_d2_load_diagnosis.py"

#: U+2212. The reports use a real minus sign, not a hyphen.
MINUS = "−"

#: How far from a superseded count a correction marker may sit and still be
#: found by the same reader. Generous on purpose: the failure this guards is a
#: *new* uncorrected site, not a paragraph that grew by a sentence.
MARKER_WINDOW = 500

#: Any one of these, near a superseded count, marks it as superseded.
CORRECTION_MARKERS = ("ABL-607", "ABL-623", "re-read", "corrected", "Corrected")


def _flat(text):
    """Prose with line wrapping collapsed and markdown emphasis stripped.

    Both matter. Every claim here spans a line break somewhere, so a re-flow of
    the paragraph must not turn a pin red. And each count is bolded at some
    sites and not others -- pinning `10 of 23` while the file says `**10** of
    23` would make the guard a decoration check, and it would have missed the
    section 8 site, where the phrase wraps *between* the number and `of 23`.
    """
    return " ".join(text.replace("*", "").replace("`", "").split())


def _section_a(path):
    return json.loads(path.read_text(encoding="utf-8"))["section_a_reproduction"]


def _paired(path):
    return {row["country"]: row for row in _section_a(path)["paired_ml_band_vs_d7"]}


def _counts():
    """(published count, current count, evaluable countries), from the records."""
    section = _section_a(REREAD)
    n_evaluable = json.loads(
        REREAD.read_text(encoding="utf-8"))["meta"]["countries_evaluable"]
    return (len(section["abl246_losers"]),
            len(section["readable_losers"]),
            n_evaluable)


def _signed(value):
    """A number formatted the way the reports write one: two decimals, explicit
    sign, and a real minus."""
    return f"{value:+.2f}".replace("-", MINUS)


# --------------------------------------------------------------------------
# 1-2. the texts name the record, and quote the count the record holds
# --------------------------------------------------------------------------

TEXTS = pytest.mark.parametrize(
    "path", [PACK, DIAGNOSIS_SCRIPT], ids=["pack", "diagnosis_script"])


@TEXTS
def test_every_text_carrying_the_count_names_the_superseding_record(path):
    """A correction a reader cannot follow to its working is an assertion."""
    if _section_a(REREAD)["losers_match"]:
        pytest.skip("records agree; nothing to point at")

    prose = _flat(path.read_text(encoding="utf-8"))
    assert REREAD.name in prose, (
        f"{path.name} states a count the re-read supersedes but never names "
        f"{REREAD.name}, so the correction cannot be checked from the text")


@TEXTS
def test_every_text_states_the_count_the_record_actually_holds(path):
    """The pin that moves when the data does: whatever the current record says,
    the prose has to say the same number."""
    _, current, n_evaluable = _counts()
    prose = _flat(path.read_text(encoding="utf-8"))

    assert f"{current} of {n_evaluable}" in prose, (
        f"the re-read record holds {current} readable losers of "
        f"{n_evaluable} evaluable countries; {path.name} never states that")


def test_the_packs_two_claim_sentences_resolve_to_the_current_count():
    """Section 1 and section 8 item 4 are where the count is *used* -- the
    headline a reader quotes and the recommendation it feeds. A pointer in the
    header does not help if these still read as current on their own."""
    published, current, n_evaluable = _counts()
    prose = _flat(PACK.read_text(encoding="utf-8"))

    for claim in (f"loses to a D-7 seasonal naive in {published} of {n_evaluable}",
                  f"loses to a lag in {current} of {n_evaluable}"):
        assert claim in prose, f"section text changed shape; {claim!r} not found"

    headline = prose[prose.index("The finding that should worry us more"):]
    headline = headline[:headline.index("And the premise this issue was gated")]
    assert f"{current} of {n_evaluable}" in headline, (
        f"the headline states {published} and never resolves it to the current "
        f"count of {current}")


# --------------------------------------------------------------------------
# 3-4. which cell moved, and by how much
# --------------------------------------------------------------------------

def test_the_pack_names_exactly_the_cells_that_moved():
    """Membership, not a count, and asserted both ways.

    A count assertion ("one cell moved") passes just as happily when the prose
    names the wrong cell, or names a cell that did not move. Both are live
    risks here: nine of the ten countries in section 4.1's list are still
    readable losers, so naming any of them reads perfectly plausibly.
    """
    section = _section_a(REREAD)
    moved = set(section["abl246_losers"]) - set(section["readable_losers"])
    stayed = set(section["readable_losers"])
    prose = _flat(PACK.read_text(encoding="utf-8"))

    start = prose.index("Corrected count (ABL-623)")
    block = prose[start:start + 1400]

    for country in moved:
        assert re.search(rf"\b{country}\b", block), (
            f"{country} left the readable-loser set and section 4.1's "
            f"correction never names it")
    named_but_stayed = {c for c in stayed
                        if re.search(rf"\b{c}\b.{{0,40}}only cell", block)}
    assert not named_but_stayed, (
        f"section 4.1 calls {sorted(named_but_stayed)} the cell that moved, "
        f"but the record still has it among the readable losers")
    assert len(moved) == 1 and "only cell that moves" in block, (
        f"section 4.1 claims a single moved cell; the records give "
        f"{sorted(moved)}")


def test_the_moved_cells_interval_is_quoted_at_the_packs_own_precision():
    """Both endpoints of both intervals, formatted to the two decimals the pack
    prints and string-compared.

    A neighbouring figure being right does not vouch for the third one, and a
    rounding that differs in the last place is exactly the drift that a reader
    checking the record would catch and a reader trusting the prose would not.
    """
    section = _section_a(REREAD)
    moved = set(section["abl246_losers"]) - set(section["readable_losers"])
    prose = _flat(PACK.read_text(encoding="utf-8"))

    for country in moved:
        for path in (PUBLISHED, REREAD):
            row = _paired(path)[country]
            quoted = (f"{_signed(row['mean_daily_wape_diff'])} "
                      f"[{_signed(row['ci_lo'])}, {_signed(row['ci_hi'])}]")
            assert quoted in prose, (
                f"{country}'s interval from {path.name} is "
                f"{quoted}, which section 4.1 does not quote")
        assert not _paired(REREAD)[country]["readable"], (
            f"{country} is still readable in the re-read; the pack's claim "
            f"that it moved out of readability is wrong")
        assert _paired(REREAD)[country]["mean_daily_wape_diff"] > 0, (
            f"{country}'s central estimate is no longer a loss; the pack says "
            f"it is")


# --------------------------------------------------------------------------
# 5-6. every site, and the guard against certifying by vacuity
# --------------------------------------------------------------------------

def test_no_site_states_the_superseded_count_without_a_marker():
    """The sweep the first grep of this defect failed.

    Looking for the superseded count by eye found two sites in the pack and
    missed a third, because section 8 wraps the phrase between the number and
    `of 23`. Deriving the test set from the record and sweeping the flattened
    text finds every site, including any added later.
    """
    published, current, n_evaluable = _counts()
    if published == current:
        pytest.skip("records agree; there is no superseded count to mark")

    for path in (PACK, DIAGNOSIS_SCRIPT):
        prose = _flat(path.read_text(encoding="utf-8"))
        sites = [m.start()
                 for m in re.finditer(rf"\b{published} of {n_evaluable}\b", prose)]
        assert sites, f"{path.name} no longer states the published count at all"
        for start in sites:
            window = prose[max(0, start - MARKER_WINDOW):start + MARKER_WINDOW]
            assert any(marker in window for marker in CORRECTION_MARKERS), (
                f"{path.name} states the superseded count "
                f"'{published} of {n_evaluable}' at offset {start} with no "
                f"correction marker within {MARKER_WINDOW} characters: "
                f"...{prose[max(0, start - 120):start + 120]}...")


def test_the_pin_is_not_vacuous():
    """Everything above is conditioned on the two records disagreeing. If they
    agreed, tests 1, 2 and 5 would pass or skip on prose saying anything at all
    -- so the disagreement itself is asserted, from the records.
    """
    section = _section_a(REREAD)
    published, current, _ = _counts()

    assert not section["losers_match"], (
        "the re-read now reproduces the published loser set; the correction "
        "in the pack and this whole module are describing a disagreement that "
        "no longer exists and must be revisited, not deleted")
    assert published != current, (
        f"both records give {current} readable losers, so there is no "
        f"correction to pin")
    assert section["winners_match"], (
        "the readable *winner* set has moved too; the pack's correction says "
        "only the loser set did")
    assert set(section["readable_losers"]) < set(section["abl246_losers"]), (
        "the current loser set is not a subset of the published one, so the "
        "pack's 'DE is the only cell that moves' framing no longer holds: "
        f"published {sorted(section['abl246_losers'])}, "
        f"current {sorted(section['readable_losers'])}")
