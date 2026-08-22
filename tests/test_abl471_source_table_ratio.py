"""ABL-471: guards on the completed source-table ratio screen.

This is a read-only screen, so there is no model to regress. What can rot is the
*record* and the *reasoning around it*, and these are the four ways it would:

1. **The verdicts stay derivable from the record's own numbers.** Every
   `mechanism` and band verdict in `reports/abl_471_source_table_ratio_screen.json`
   is re-derived here from that pair's stored ratios by calling the screen's own
   classifier. A report that states a verdict its numbers do not support is how
   a held pair ships, and it is not visible by reading the report.

2. **The reference row keeps working.** NL `wind_onshore` is in the record to make
   "LV is the same defect, +0.1706 where NL is +1.2659" a comparison. If the
   classifier ever stops calling that pair a `revision_vintage`, the classifier
   disagrees with ABL-439 and the LV verdict loses its yardstick -- this is the
   case that caught a first draft keyed on bit-identical convergence, which NL
   `wind_onshore` does not satisfy (it converges to 0.9933).

3. **The two windows stay distinguishable.** The screen reports the ABL-439
   comparator window and ABL-348's registered fit window separately *because they
   disagree on NL `wind_offshore`* -- 0.9648 against 0.9922, one outside ledger
   5.6's descriptive band and one inside. Collapsing them to a single "the ratio"
   is the specific mistake this record exists to prevent, so the test asserts they
   are still both present and still differ on that pair.

4. **The descriptive band never becomes the rule.** Ledger 5.6's 0.99-1.07 is
   where the clean pairs landed; ABL-439 decides on `abs(ratio - 1) > 0.15`. Both
   verdicts are reported per pair and the test pins that they can disagree, so a
   later reader cannot quietly promote the description into a criterion.

The measured values themselves are pinned in `EXPECTED` and are only re-derivable
against the replica, which CI does not have -- so the numeric pins are asserted
against the committed record and the live re-measurement is left to the script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RECORD = json.loads((ROOT / "reports" / "abl_471_source_table_ratio_screen.json")
                    .read_text(encoding="utf-8"))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = _module("abl471_source_table_ratio_screen")

#: `(comparator, fit, gate, mechanism)` as measured on the 10,220,126,208-byte
#: replica. Reproduced in the report's section 1.
EXPECTED = {
    ("DE", "wind_offshore"): (1.0000, 1.0017, 1.0000, "no_fit_gate_discontinuity"),
    ("NL", "wind_offshore"): (0.9648, 0.9922, 0.9912, "no_fit_gate_discontinuity"),
    ("EE", "solar"): (1.1695, 1.1764, 1.0000, "revision_vintage"),
    ("FI", "solar"): (1.0000, 1.0000, 1.0000, "no_fit_gate_discontinuity"),
    ("LV", "solar"): (1.1143, 1.1706, 1.0000, "revision_vintage"),
    ("NL", "wind_onshore"): (2.4647, 2.2592, 0.9933, "revision_vintage"),
}

#: The four the ABL-439 sweep could not reach, plus the outlier it could not
#: explain. The scope ABL-471 was opened on; a record that screens fewer pairs
#: than this has not answered the issue.
REQUIRED_PAIRS = {("DE", "wind_offshore"), ("NL", "wind_offshore"),
                  ("EE", "solar"), ("FI", "solar"), ("LV", "solar")}


def _pairs():
    return {(p["country"], p["forecast_type"]): p for p in RECORD["pairs"]}


def test_every_unscreened_pair_is_screened():
    """The scope is the issue's, not whatever the script happened to run."""
    assert REQUIRED_PAIRS <= set(_pairs())


def test_recorded_ratios_match_the_published_values():
    for key, (comparator, fit, gate, _) in EXPECTED.items():
        verdict = _pairs()[key]["verdict"]
        assert verdict["ratio_abl439_comparator_window"] == comparator, key
        assert verdict["ratio_abl348_fit_window"] == fit, key
        assert verdict["ratio_abl348_gate_window"] == gate, key


def test_mechanism_is_rederivable_from_the_records_own_numbers():
    """Re-run the classifier over each pair's stored ratios.

    Guard 1 and guard 2: the verdict has to follow from the numbers beside it, and
    the ABL-439 reference pair has to keep classifying as the vintage it was
    diagnosed as.
    """
    for key, entry in _pairs().items():
        rederived = SCREEN._verdict(entry)
        assert rederived["mechanism"] == entry["verdict"]["mechanism"], key
        if key in EXPECTED:
            assert rederived["mechanism"] == EXPECTED[key][3], key


def test_nl_wind_onshore_reference_is_classified_as_the_vintage():
    """ABL-439's diagnosis is the yardstick the LV verdict is stated against."""
    reference = _pairs()[("NL", "wind_onshore")]
    assert reference["verdict"]["mechanism"] == "revision_vintage"
    # And specifically not via bit-identical convergence, which it fails.
    assert reference["convergence"]["bit_identical_after"] is False
    assert reference["verdict"]["fit_gate_discontinuity"] > 1.0


def test_the_two_windows_are_reported_separately_and_disagree_on_nl_offshore():
    """Guard 3. The whole reason the record carries three windows."""
    assert "abl439_comparator" in RECORD["windows_registered"]
    assert "abl348_fit_window" in RECORD["windows_registered"]
    verdict = _pairs()[("NL", "wind_offshore")]["verdict"]
    assert verdict["ratio_abl439_comparator_window"] != verdict["ratio_abl348_fit_window"]
    # One outside ledger 5.6's descriptive band, one inside -- which is the point.
    assert verdict["in_descriptive_band_comparator"] is False
    assert verdict["in_descriptive_band_fit_window"] is True


def test_descriptive_band_is_not_the_screens_rule():
    """Guard 4. NL `wind_offshore` fails the description and passes the rule."""
    verdict = _pairs()[("NL", "wind_offshore")]["verdict"]
    assert verdict["in_descriptive_band_comparator"] is False
    assert verdict["basis_affected_by_abl439_rule_comparator"] is False
    # The rule is ABL-439's own constant, read rather than retyped.
    import abl439_reporting_basis_probe as abl439
    assert abl439.SWEEP_MATERIAL_RATIO == 0.15


def test_the_affected_and_clean_pairs_are_not_a_close_call():
    """No pair is decided by where in the empty gap the threshold sits."""
    affected, clean = [], []
    for entry in _pairs().values():
        bucket = (affected if entry["verdict"]["mechanism"] == "revision_vintage"
                  else clean)
        bucket.append(abs(entry["verdict"]["fit_gate_discontinuity"]))
    assert min(affected) > 10 * max(clean)
    assert max(clean) <= SCREEN.NO_DISCONTINUITY
    assert min(affected) > SCREEN.VINTAGE_DISCONTINUITY


def test_the_reproduction_pins_reproduce():
    """A run that does not reproduce ABL-439's record is reading something else."""
    pins = RECORD["reproduction_pins"]
    assert len(pins) == len(SCREEN.REPRODUCTION_PINS)
    assert all(pin["reproduces"] for pin in pins), [p for p in pins
                                                    if not p["reproduces"]]
    # Section 5.6's three tabulated ratios specifically.
    quoted = {(p["country"], p["forecast_type"]): p["measured"] for p in pins}
    assert quoted[("NL", "wind_onshore")] == 2.4647
    assert quoted[("NL", "solar")] == 1.6269
    assert quoted[("GR", "solar")] == 0.7945
    assert quoted[("LV", "solar")] == 1.1143


def test_the_offshore_record_the_screen_missed_is_tracked_and_not_ignored():
    """The screen's gap was a glob directory, not ABL-440's gitignore.

    Ledger 5.6 attributes DE/NL `wind_offshore` being unscreened to the record
    being gitignored. This asserts the file is there to be read, which is what
    makes that attribution wrong and the screen reachable all along.
    """
    record = ROOT / "experiments" / "ABL322" / "results_abl436_offshore_reread.json"
    assert record.exists()
    cells = json.loads(record.read_text(encoding="utf-8"))["gate_cells"]
    assert {(c["country"], c["forecast_type"]) for c in cells} == {
        ("DE", "wind_offshore"), ("NL", "wind_offshore")}
