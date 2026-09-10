"""The net-position cron must run a pinned checkout, and say which one (ABL-692).

The defect these tests exist to prevent had no failing symptom. The daily
net-position forecast ran `scripts/forecast_chronos2.py` out of
`C:\\Code\\able\\energy-forecast`, the tree agents use as a scratch working
copy. It had been parked on `ABL-584-reland-abl471` since 2026-08-28, so every
serving change merged to `origin/main` after 2026-08-27 16:10 was inert for
twelve days -- including the ABL-650 band recalibration (PR #107, merge
`3419031`), which is an ancestor of `origin/main` and not of that tree's HEAD.

Nothing alerted, because there is no deploy step: the cron executes whatever is
on disk. The log could not have shown it either -- every run from 2026-09-04 to
2026-09-10 logged the plain `Saved 216 quantile forecasts to DB`, with no way
to tell a stale tree from a current one.

So two properties are held here, and the second matters more than the first:

  1. the job runs a checkout that is hard-reset to `origin/main`, and
  2. every run writes the resolved commit into the transcript.

A stale tree is recoverable the moment it is visible. An unattributable vintage
is not: you cannot say afterwards which code produced it.

The third group of tests holds the *seams* the split created. The serving
checkout carries only git-tracked code, so three gitignored inputs had to move
to parameters -- the ABL-69 rail interpreter, the 4.2 GB `models/` tree, and
the eval report root that ABL-30/ABL-34 read at the dev-checkout path. Each one
fails silently if it regresses: a wrong `--models-dir` makes V014 log "no
trained model" and exit 0, and an eval root under the disposable checkout would
leave every existing reader on a directory that quietly stopped updating.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = REPO_ROOT / "scripts" / "workstation"
LAUNCHER = WORKSTATION / "run-net-position-serving.ps1"
JOB = WORKSTATION / "run-net-position.ps1"
INSTALLER = WORKSTATION / "install-serving-checkout.ps1"

#: The shared dev checkout. Serving must never resolve to this path.
DEV_CHECKOUT = r"C:\Code\able\energy-forecast"

#: Grepped by operators and by the ABL-692 runbook to attribute a vintage.
WITNESS_PREFIX = "net-position serving commit:"


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Vacuity guard: every assertion below is a search over this text, and a
    # search over an empty or truncated file passes by not matching.
    assert len(text) > 500, f"{path.name} is too short to be the real script"
    return text


@pytest.fixture(scope="module")
def launcher() -> str:
    return _read(LAUNCHER)


@pytest.fixture(scope="module")
def job() -> str:
    return _read(JOB)


def _code(text: str) -> str:
    """`text` with whole-line PowerShell comments removed.

    These scripts carry long rationale comments that name the very constructs
    the "must not appear" assertions forbid -- the `git clean` guard below
    failed on the comment explaining why there is no `git clean`. A negative
    assertion has to read code.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _param_default(text: str, name: str) -> str:
    """The default of a PowerShell `[string] $Name = "value"` param."""
    m = re.search(
        r"\[string\]\s*\$" + re.escape(name) + r"\s*=\s*\"([^\"]+)\"", text
    )
    assert m, f"no [string] ${name} param with a literal default"
    return m.group(1)


# ---------------------------------------------------------------------------
# 1. The cron runs a pinned checkout.
# ---------------------------------------------------------------------------

def test_launcher_exists():
    assert LAUNCHER.is_file(), (
        "the scheduled task's entry point is missing; without it the task falls "
        "back to running the dev checkout, which is the ABL-692 defect"
    )


def test_launcher_fetches_and_hard_resets_to_the_pinned_branch(launcher):
    assert re.search(r"git -C \$Serving fetch", launcher), "no fetch before serving"
    assert re.search(
        r"git -C \$Serving reset --hard \"origin/\$Branch\"", launcher
    ), "no hard reset to origin/$Branch: a merge would still not reach production"


def test_pinned_branch_defaults_to_main(launcher):
    assert _param_default(launcher, "Branch") == "main"


def test_launcher_refuses_to_serve_from_the_shared_dev_checkout(launcher):
    """The naive fix -- `git checkout main` in the dev tree -- is its own hazard.

    It would move HEAD under a live agent run and discard in-progress work, so
    the launcher refuses that target rather than automating it.
    """
    assert DEV_CHECKOUT in launcher, "the dev checkout path is not named"
    assert re.search(r"\$servingFull\s*-ieq\s*\$devFull", launcher), (
        "no guard comparing the serving target against the dev checkout"
    )
    guard = launcher[launcher.index("$servingFull -ieq $devFull"):]
    assert "throw" in guard[:400], "the dev-checkout guard does not refuse"


def test_launcher_does_not_clean_the_serving_tree(launcher):
    """`reset --hard` is the whole requirement; `clean -xdf` is not.

    The job writes gitignored eval reports into the tree, and this box keeps a
    4.2 GB gitignored `models/` directory in the sibling checkout. A `clean`
    here would be a silent, expensive data loss with no serving benefit.
    """
    code = _code(launcher)
    assert "git clean" not in code
    assert "clean -xdf" not in code
    # ...and the comment saying why must survive, or the next editor adds one.
    assert "git clean" in launcher


# ---------------------------------------------------------------------------
# 2. Every run is attributable to a commit.
# ---------------------------------------------------------------------------

def test_launcher_declares_the_witness_prefix(launcher):
    assert f'$WitnessPrefix = "{WITNESS_PREFIX}"' in launcher, (
        f"the witness prefix must stay exactly {WITNESS_PREFIX!r}: the runbook "
        "and the ABL-677 re-base both grep for it"
    )


def test_launcher_logs_the_resolved_sha(launcher):
    assert re.search(r"git -C \$Serving rev-parse HEAD", launcher), "no SHA resolved"
    emits = re.findall(r"Write-Host \"\$WitnessPrefix \$sha[^\"]*\"", launcher)
    assert emits, "the resolved SHA is never written to the transcript"


def test_a_failed_sync_still_logs_a_sha_and_still_forecasts():
    """A missing vintage is worse than a one-day-stale one.

    The fix's real property is attributability, not freshness: on a sync
    failure the run continues, but the witness line is still emitted and is
    marked STALE, so the staleness is loud instead of silent.
    """
    text = _read(LAUNCHER)
    emits = re.findall(r"Write-Host \"\$WitnessPrefix \$sha[^\"]*\"", text)
    assert len(emits) == 2, (
        "expected the witness line on both the synced and the failed-sync path, "
        f"found {len(emits)}"
    )
    assert any("STALE" in e for e in emits), "a failed sync is not marked STALE"
    # The forecast is not gated on $synced: the job runs after the branch.
    body = text[text.index("$job = Join-Path"):]
    assert "if ($synced)" not in body, "the job was made conditional on the sync"


def test_only_the_launcher_starts_a_transcript(launcher, job):
    """The sync lines must land inside the log, and PS 5.1 throws on a nested
    Start-Transcript -- so the launcher owns the transcript and the job has none.
    """
    assert "Start-Transcript" in _code(launcher)
    assert "Start-Transcript" not in _code(job), (
        "the job starting its own transcript would both nest (and throw) and "
        "push the commit witness line outside the log it belongs in"
    )


# ---------------------------------------------------------------------------
# 3. The seams the split created.
# ---------------------------------------------------------------------------

def test_job_does_not_default_to_the_shared_dev_checkout(job):
    """This single default *is* the defect. Nothing else here is load-bearing
    if $Repo points back at the tree agents work in.
    """
    assert _param_default(job, "Repo") != DEV_CHECKOUT


def test_job_passes_models_dir_explicitly(job):
    """`config.MODELS_DIR` resolves against the repo root, and `models/` is
    gitignored, so the serving clone has none. V014 would log "no trained
    model" for all 24 countries, write nothing, and exit 0.
    """
    assert "--models-dir $ModelsDir" in job
    models_dir = _param_default(job, "ModelsDir")
    assert models_dir.startswith(DEV_CHECKOUT), (
        "the artifacts live in the dev checkout; CLAUDE.md documents that "
        "resolution and train.py still writes there"
    )


def test_models_dir_is_a_real_flag_on_the_challenger_script():
    """A flag the parser rejects would abort the challengers with a usage error
    that reads as a plain non-zero exit. Checked structurally, not by grep.
    """
    src = (REPO_ROOT / "scripts" / "forecast_challengers.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    flags = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--models-dir" in flags, f"forecast_challengers.py rejects --models-dir; has {sorted(flags)}"


def test_champion_does_not_need_the_models_tree():
    """Why the champion survives a clone with no `models/` at all.

    `forecast_chronos2.py` only resolves `config.MODELS_DIR` when the
    experiment sets `training.fine_tune`; V010 does not, so it serves the
    pretrained Chronos-2. If V010 is ever fine-tuned, the job needs a
    models path of its own and this test is the reminder.
    """
    cfg = json.loads((REPO_ROOT / "experiments" / "V010" / "config.json").read_text())
    assert cfg.get("training", {}).get("fine_tune") is False


def test_eval_reports_stay_where_their_readers_look(job):
    """ABL-30/ABL-34 and docs/claude/04-database.md read the eval reports at the
    dev-checkout path. Writing them under the disposable serving checkout would
    strand every reader on a directory that silently stopped updating.
    """
    eval_root = _param_default(job, "EvalRoot")
    assert eval_root == DEV_CHECKOUT + r"\reports\net_position_eval"
    assert "$Repo\\reports\\net_position_eval" not in _code(job)


def test_backtest_references_move_with_the_code(job):
    """The opposite call from the eval root, and deliberately so: the
    --candidate-backtest files are tracked, so they come from the pinned tree
    and a vintage is scored against the backtest its own code shipped with.
    """
    assert job.count("$Repo\\experiments\\V0") >= 3
    assert "$Repo\\comparison_net_position_servefaithful.json" in job


def test_job_pins_the_rail_interpreter(job):
    """ABL-69: an xgboost-3.3.0 artifact loaded under the conda 2.1.4 keeps its
    trees and silently resets the fitted intercept, which reads as a bad model
    rather than a bad load. The interpreter is part of the configuration.
    """
    assert _param_default(job, "Venv") == DEV_CHECKOUT + r"\.venv"
    assert "$Venv\\Scripts\\python.exe" in _code(job)
    # No bare `python` invocation may have survived the rewrite: bare `python`
    # on this box is conda 3.11 / xgboost 2.1.4, the wrong side of ABL-69.
    assert not re.search(r"&\s*\"?python(\.exe)?\"?\s", _code(job))


def test_installer_does_not_repoint_the_task_by_default():
    """Creating the clone is inert; re-pointing the task changes what
    production forecasts tomorrow. They must not be one unconditional step.
    """
    text = _read(INSTALLER)
    assert "[switch] $UpdateScheduledTask" in text
    assert re.search(r"if \(\$UpdateScheduledTask\)", text), (
        "Set-ScheduledTask must be gated on the switch"
    )
    gated = text[text.index("if ($UpdateScheduledTask)"):]
    assert "Set-ScheduledTask" in gated
    assert "Set-ScheduledTask" not in text[: text.index("if ($UpdateScheduledTask)")]
