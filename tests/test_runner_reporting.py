"""A subprocess runner that produced nothing must not report OK (ABL-370).

`forecast_daily` decided the outcome of an external `MODEL_RUNNERS` subprocess
from its exit code alone, and read a row count out of stdout only if the runner
happened to print one of two ad-hoc phrasings. `tso_correction_forecaster`
prints neither when it skips every type, so a run that generated zero forecasts
logged:

    [tso-correction] OK: BE solar D+2
    Total: 10, Success: 2, Skipped: 8, Failed: 0

`records` stayed 0, and 0 is invisible inside a `Total forecasts:` sum the
in-process xgboost models push into the thousands. That is the reporting shape
that let ABL-354's dead runner — every BE solar/wind forecast failing at its
import line — sit undetected.

These tests hold the three-way distinction that replaced it: produced rows,
ran and produced none, failed. `test_zero_output_runner_is_not_success` is the
regression: it feeds `run_external_model` the exact transcript above.
"""

import ast
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import config  # noqa: E402
import forecast_daily  # noqa: E402
from src.runner_report import (  # noqa: E402
    RECORD_COUNT_PREFIX,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNREPORTED,
    emit_record_count,
    format_runner_summary,
    parse_record_count,
    status_for_count,
    summarize_by_runner,
)

LOGGER = logging.getLogger("test_runner_reporting")

# Verbatim from the ABL-370 report: a clean, correct run that produced nothing.
ZERO_OUTPUT_TRANSCRIPT = """Processing BE/solar D+2 -> 2026-08-15
No TSO forecast for solar on 2026-08-15, skipping
Processing BE/wind_onshore D+2 -> 2026-08-15
No TSO forecast for wind_onshore on 2026-08-15, skipping
Processing BE/wind_offshore D+2 -> 2026-08-15
No TSO forecast for wind_offshore on 2026-08-15, skipping
No forecasts generated"""


# --- the contract line -------------------------------------------------------

def test_emit_and_parse_round_trip(capsys):
    emit_record_count(96)
    assert parse_record_count(capsys.readouterr().out) == 96


def test_emit_zero_is_a_report_not_a_silence(capsys):
    emit_record_count(0)
    assert parse_record_count(capsys.readouterr().out) == 0


def test_missing_line_parses_as_unknown_not_zero():
    # The whole point: absent != 0. 0 is a measurement; absent is not.
    assert parse_record_count(ZERO_OUTPUT_TRANSCRIPT) is None
    assert parse_record_count("") is None


def test_prose_row_counts_are_not_the_contract():
    """The ad-hoc shapes this replaced must not be read as a count.

    `Forecast (0 rows)` and `Saved 0 forecast records` were parsed before
    ABL-370; leaving them parseable would keep two contracts alive.
    """
    assert parse_record_count("\nForecast (48 rows):\n") is None
    assert parse_record_count("Saved 48 forecast records\n") is None


def test_prefix_must_start_the_line():
    assert parse_record_count(f"see {RECORD_COUNT_PREFIX}12 in the log") is None


def test_malformed_count_is_unknown():
    assert parse_record_count(f"{RECORD_COUNT_PREFIX}many") is None
    assert parse_record_count(f"{RECORD_COUNT_PREFIX}") is None


def test_last_report_wins():
    stdout = f"{RECORD_COUNT_PREFIX}12\nnoise\n{RECORD_COUNT_PREFIX}48\n"
    assert parse_record_count(stdout) == 48


@pytest.mark.parametrize(
    "records,expected",
    [(None, STATUS_UNREPORTED), (0, STATUS_EMPTY), (1, STATUS_SUCCESS), (96, STATUS_SUCCESS)],
)
def test_status_for_count(records, expected):
    assert status_for_count(records) == expected


# --- run_external_model, against real subprocesses ---------------------------

def _run_fake(tmp_path, body: str) -> dict:
    """Run `body` as a runner through the real `run_external_model`.

    `python_executable` is this interpreter and the script is an absolute path,
    so `build_runner_command` launches it by path — the mode a runner outside
    `src/` gets, and the one that needs no package.
    """
    script = tmp_path / "fake_runner.py"
    script.write_text(body, encoding="utf-8")
    runner = {
        "name": "fake-runner",
        "script": str(script),
        "python_executable": sys.executable,
    }
    return forecast_daily.run_external_model(
        runner, "BE", "solar", 2, date(2026, 8, 13), True, LOGGER
    )


def test_zero_output_runner_is_not_success(tmp_path):
    """The ABL-370 regression, on the exact transcript from the report."""
    result = _run_fake(tmp_path, f"print({ZERO_OUTPUT_TRANSCRIPT!r})")

    assert result["status"] != STATUS_SUCCESS, (
        "a runner that exited 0 having generated no forecasts was reported as "
        "success — the summary cannot then tell it from one that produced rows "
        "(ABL-370)."
    )
    assert result["status"] == STATUS_UNREPORTED
    assert result["records"] is None, "unknown must not be recorded as 0"
    assert RECORD_COUNT_PREFIX in result["error"]


def test_reported_zero_is_empty_not_success(tmp_path):
    body = (
        f"print({ZERO_OUTPUT_TRANSCRIPT!r})\n"
        f"print('{RECORD_COUNT_PREFIX}0')\n"
    )
    result = _run_fake(tmp_path, body)

    assert result["status"] == STATUS_EMPTY
    assert result["records"] == 0
    assert result["error"] is None


def test_reported_rows_are_success(tmp_path):
    result = _run_fake(tmp_path, f"print('{RECORD_COUNT_PREFIX}96')")

    assert result["status"] == STATUS_SUCCESS
    assert result["records"] == 96


def test_nonzero_exit_still_fails(tmp_path):
    body = f"import sys\nprint('{RECORD_COUNT_PREFIX}96')\nsys.exit(3)"
    result = _run_fake(tmp_path, body)

    assert result["status"] == STATUS_FAILED, (
        "the count line reports what a run produced; it does not overrule a "
        "crash."
    )
    assert result["records"] == 0


def test_runner_name_is_carried_into_the_result(tmp_path):
    result = _run_fake(tmp_path, f"print('{RECORD_COUNT_PREFIX}96')")
    assert result["runner"] == "fake-runner"


# --- every configured runner honours the contract ----------------------------

RUNNERS = [r for r in config.MODEL_RUNNERS if r.get("script")]
RUNNER_IDS = [f"{r['name']}:{r['script']}" for r in RUNNERS]


def test_model_runners_are_covered():
    # Guards the parametrisation below against silently covering nothing.
    assert len(RUNNERS) >= 3, f"only found {len(RUNNERS)} runners with a script"


@pytest.mark.parametrize("runner", RUNNERS, ids=RUNNER_IDS)
def test_runner_emits_the_record_count(runner):
    """Each subprocess runner calls `emit_record_count` on its exit-0 path.

    Static, because running one for real needs a database, a trained model and
    (for chronos) a second venv. What it catches is the case that matters: a
    runner added or edited without the count line, which `forecast_daily` can
    then only report as `unreported` — never as OK, but never as a row count
    either.
    """
    script = REPO_ROOT / runner["script"]
    tree = ast.parse(script.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "emit_record_count" in called, (
        f"{runner['script']} never calls emit_record_count, so forecast_daily "
        f"cannot tell whether a run of the {runner['name']} runner produced "
        "anything.\nAdd `from src.runner_report import emit_record_count` (or "
        "the relative form inside src/) and call it with the row count on "
        "every path that exits 0, zero included (ABL-370)."
    )


def _statement_lists(tree):
    """Every block of statements in the module, as lists."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and all(isinstance(s, ast.stmt) for s in block):
                yield block


def _is_emit(stmt):
    return (
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "emit_record_count"
    )


@pytest.mark.parametrize(
    "path", ["src/tso_correction_forecaster.py", "scripts/forecast_chronos2.py"]
)
def test_emit_is_outside_the_non_empty_guard(path):
    """The empty path is the one this issue is about — prove it still emits.

    Both runners print their human-readable block only when the frame has rows.
    `emit_record_count` has to sit *before* that guard, or an empty run goes
    back to saying nothing and reads as `unreported` instead of as zero.
    """
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))

    for block in _statement_lists(tree):
        emits = [i for i, stmt in enumerate(block) if _is_emit(stmt)]
        guards = [
            i for i, stmt in enumerate(block)
            if isinstance(stmt, ast.If) and "empty" in ast.dump(stmt.test)
        ]
        if emits and guards:
            assert min(emits) < min(guards), (
                f"{path} emits the record count after the `.empty` guard, so a "
                "run that produced nothing reports nothing (ABL-370)."
            )
            return

    pytest.fail(
        f"{path}: no block holds both emit_record_count and the `.empty` guard "
        "— the emit is nested somewhere it may not run on the empty path."
    )


def test_configured_runner_scripts_import_the_contract():
    """`--help` proves the added import resolves under the launch mode used."""
    for runner in RUNNERS:
        cmd = forecast_daily.build_runner_command(runner, ["--help"], repo_root=REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, *cmd[1:]],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
        )
        assert proc.returncode == 0, (
            f"`{' '.join(cmd[1:])}` exits {proc.returncode} after the "
            f"runner_report import was added.\n\nstderr:\n{proc.stderr[-2000:]}"
        )


# --- the per-runner summary --------------------------------------------------

def _result(runner, status, records, error=None):
    return {
        "country_code": "BE", "forecast_type": "solar", "horizon_days": 2,
        "runner": runner, "status": status, "records": records, "error": error,
    }


def test_summary_splits_rows_by_runner():
    """The zero has to survive being added to the builtin models' thousands."""
    results = [
        _result("builtin", STATUS_SUCCESS, 2400),
        _result("builtin", STATUS_FAILED, 0, "Model not found: price"),
        _result("tso-correction", STATUS_EMPTY, 0),
        _result("tso-correction", STATUS_EMPTY, 0),
    ]
    by_runner = {r["runner"]: r for r in summarize_by_runner(results)}

    assert by_runner["builtin"]["rows"] == 2400
    assert by_runner["builtin"]["success"] == 1
    assert by_runner["builtin"]["skipped"] == 1
    assert by_runner["builtin"]["failed"] == 0, "a missing model is a skip, not a failure"

    assert by_runner["tso-correction"]["rows"] == 0
    assert by_runner["tso-correction"]["empty"] == 2
    assert by_runner["tso-correction"]["runs"] == 2


def test_unreported_runs_contribute_no_rows():
    rows = summarize_by_runner([_result("tso-correction", STATUS_UNREPORTED, None)])
    assert rows[0]["rows"] == 0
    assert rows[0]["unreported"] == 1


def test_an_all_unreported_runner_is_not_printed_as_zero_rows():
    """Unknown must not become a printed 0 in the one place a human reads it."""
    text = "\n".join(format_runner_summary([
        _result("builtin", STATUS_SUCCESS, 2400),
        _result("tso-correction", STATUS_UNREPORTED, None),
    ]))
    tso_lines = [line for line in text.splitlines() if "tso-correction" in line]

    assert tso_lines
    for line in tso_lines:
        assert "no reported rows" in line, line
        assert "0 rows" not in line, line


def test_summary_names_a_runner_that_produced_nothing():
    results = [
        _result("builtin", STATUS_SUCCESS, 2400),
        _result("tso-correction", STATUS_EMPTY, 0),
    ]
    text = "\n".join(format_runner_summary(results))

    assert "Runners that produced no forecasts:" in text
    assert "tso-correction" in text.split("Runners that produced no forecasts:")[1]
    assert "builtin" not in text.split("Runners that produced no forecasts:")[1]


def test_a_runner_with_only_skips_is_not_called_out():
    """Nothing to run is not the same as ran and produced nothing."""
    results = [_result("builtin", STATUS_FAILED, 0, "Model not trained yet")]
    assert "Runners that produced no forecasts:" not in "\n".join(
        format_runner_summary(results)
    )


def test_results_without_a_runner_key_are_attributed_to_builtin():
    rows = summarize_by_runner([{"status": STATUS_SUCCESS, "records": 24}])
    assert rows[0]["runner"] == "builtin"
