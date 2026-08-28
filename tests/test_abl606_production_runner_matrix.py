"""A workstation-only runner must not be in the scheduled matrix (ABL-606).

`MODEL_RUNNERS` carried a `"production"` key on every entry, with a per-entry
comment explaining the choice, and **nothing read it**. The external-runner
selection in `forecast_daily` was `type == 'external' and enabled`, so the two
entries declaring `"production": False` were launched on every scheduled run --
including inside the production container, where their configured interpreters
(`C:/Users/guill/.openclaw/.../chronos-venv/Scripts/python.exe` and
`C:/Users/guill/miniconda3/python.exe`) cannot resolve under any dependency pin.

ABL-601 measured it inside the rebuilt container:

    Total: 440, Success: 154, Empty: 0, Unreported: 0, Skipped: 278, Failed: 8

All 8 were `Executable not found`. `test_matrix_arithmetic_reproduces_abl601`
reproduces both numbers from `config` alone, with no container: the 8 are
exactly `chronos-bolt-small` at BE x price x {D+1, D+2} and `tso-correction` at
BE x {solar, wind_onshore, wind_offshore} x {D+1, D+2}.

The two dispositions ABL-606 had to choose between were "these are meant to
serve" and "these are workstation-only". The second, on three independent
pieces of evidence:

  * neither model name has written a forecast row since **2026-03-03** -- two
    days before this repo's initial commit;
  * the dashboard deliberately does not register either
    (`server/src/config/forecastModels.ts`, held by `forecastModels.test.ts`);
  * `tests/test_script_imports.py::test_model_runner_launches` already refuses
    to use a runner's `python_executable`, on the grounds that it "is an
    absolute path to one box's venv".

So the flag was right and the selection was wrong. These tests hold the flag
load-bearing, and hold the invariant that produced the defect in the first
place: an absolute non-`.venv` interpreter path means workstation-only.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import config  # noqa: E402
import forecast_daily  # noqa: E402

#: Verbatim from ABL-601's in-container dry-run.
ABL601_TOTAL = 440
ABL601_FAILED = 8

#: The two runners and their cell counts, from the same run's failure list.
ABL601_FAILING_RUNNERS = {"chronos-bolt-small": 2, "tso-correction": 6}


def _cells(runner, countries, forecast_types):
    """Matrix cells `forecast_daily`'s external loop would visit for a runner."""
    runner_countries = runner.get("countries", [])
    runner_types = runner.get("forecast_types", [])
    return sum(
        len(config.get_horizons_for_type(forecast_type))
        for country in countries
        if runner_countries == "all" or country in runner_countries
        for forecast_type in forecast_types
        if runner_types == "all" or forecast_type in runner_types
    )


@pytest.fixture(scope="module")
def matrix():
    countries = forecast_daily.get_countries("all")
    forecast_types = forecast_daily.get_forecast_types("all")
    return countries, forecast_types


# --- the flag is load-bearing ------------------------------------------------

def test_scheduled_run_excludes_non_production_runners():
    selected = forecast_daily.select_external_runners(config.MODEL_RUNNERS)
    assert all(r.get("production") for r in selected), (
        "a `production: False` runner is in the default (scheduled) selection: "
        f"{[r['name'] for r in selected if not r.get('production')]}"
    )


def test_opt_in_flag_brings_them_back():
    default = forecast_daily.select_external_runners(config.MODEL_RUNNERS)
    opted_in = forecast_daily.select_external_runners(
        config.MODEL_RUNNERS, include_non_production=True
    )
    assert {r["name"] for r in opted_in} > {r["name"] for r in default}, (
        "--include-non-production must be able to reach runners the default "
        "selection holds back, or the workstation loses them entirely"
    )
    assert {r["name"] for r in opted_in} == {
        r["name"] for r in config.MODEL_RUNNERS
        if r.get("type") == "external" and r.get("enabled")
    }


def test_parked_runner_stays_parked_under_the_opt_in():
    """`enabled` and `production` are different questions.

    `chronos-2` is `enabled: False` *and* `production: False` and has
    `python_executable: None`; --include-non-production must not start it.
    """
    opted_in = forecast_daily.select_external_runners(
        config.MODEL_RUNNERS, include_non_production=True
    )
    assert "chronos-2" not in {r["name"] for r in opted_in}


def test_missing_production_key_defaults_to_not_production():
    """Absent is not "yes". A new runner has to opt in explicitly."""
    runner = {"name": "new", "type": "external", "enabled": True, "script": "x.py"}
    assert forecast_daily.select_external_runners([runner]) == []
    assert forecast_daily.select_external_runners(
        [runner], include_non_production=True
    ) == [runner]


def test_selection_never_returns_a_builtin_runner():
    for include in (False, True):
        selected = forecast_daily.select_external_runners(
            config.MODEL_RUNNERS, include_non_production=include
        )
        assert all(r["type"] == "external" for r in selected)


# --- the invariant that caused the defect ------------------------------------

@pytest.mark.parametrize(
    "runner",
    [r for r in config.MODEL_RUNNERS if r.get("python_executable")],
    ids=[r["name"] for r in config.MODEL_RUNNERS if r.get("python_executable")],
)
def test_box_specific_interpreter_implies_non_production(runner):
    """The rule, not the two instances of it.

    A configured `python_executable` is an absolute path on one workstation.
    Unless it is the repo `.venv` -- which is the one interpreter the container
    also has -- the runner cannot run anywhere else, so registering it in the
    scheduled matrix guarantees a standing `Failed` count.
    """
    exe = Path(runner["python_executable"])
    in_repo_venv = REPO_ROOT / ".venv" in exe.parents
    assert in_repo_venv or not runner.get("production"), (
        f"MODEL_RUNNERS['{runner['name']}'] is marked production but runs under "
        f"{exe}, which exists on one box only. In the container this fails "
        "`Executable not found` on every scheduled run (ABL-606)."
    )


# --- the measurement this reproduces -----------------------------------------

def test_matrix_arithmetic_reproduces_abl601(matrix):
    """440 total and 8 failing, from config alone -- no container needed."""
    countries, forecast_types = matrix

    builtin = len(countries) * sum(
        len(config.get_horizons_for_type(t)) for t in forecast_types
    )
    pre_fix = [
        r for r in config.MODEL_RUNNERS
        if r.get("type") == "external" and r.get("enabled", False)
    ]
    external = {r["name"]: _cells(r, countries, forecast_types) for r in pre_fix}

    assert builtin + sum(external.values()) == ABL601_TOTAL
    assert external == ABL601_FAILING_RUNNERS
    assert sum(external.values()) == ABL601_FAILED


def test_fix_removes_exactly_those_eight_cells(matrix):
    """And nothing else: the 432 builtin cells are untouched.

    This is the number the next deploy diffs against (ABL-603 hazard 1), so it
    is asserted rather than described: after the fix the scheduled matrix is
    432 cells and its expected `Failed` floor is 0.
    """
    countries, forecast_types = matrix

    builtin = len(countries) * sum(
        len(config.get_horizons_for_type(t)) for t in forecast_types
    )
    after = sum(
        _cells(r, countries, forecast_types)
        for r in forecast_daily.select_external_runners(config.MODEL_RUNNERS)
    )

    assert builtin == ABL601_TOTAL - ABL601_FAILED == 432
    assert after == 0, (
        "no external runner is production today; if one is added, update this "
        "and re-state the expected Total for the deploy diff"
    )
    assert builtin + after == ABL601_TOTAL - ABL601_FAILED


# --- the CLI surface ---------------------------------------------------------

def test_include_non_production_flag_exists_and_defaults_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["forecast_daily.py"])
    assert forecast_daily.parse_args().include_non_production is False

    monkeypatch.setattr(
        sys, "argv", ["forecast_daily.py", "--include-non-production"]
    )
    assert forecast_daily.parse_args().include_non_production is True
