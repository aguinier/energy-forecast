"""The forecast container's entrypoint runs the command it was given (ABL-596).

`docker/entrypoint.sh` ended in an unconditional `exec cron -f`. Docker's
`ENTRYPOINT` is not replaced by a `CMD` override -- the override is *appended*
as arguments -- so a script that ignores `"$@"` runs cron no matter what the
caller typed. A read-only version probe during the ABL-585 rebuild:

    docker run --rm docker-forecast:latest python3 -c "import xgboost; print(xgboost.__version__)"

never ran that one-liner. It started a second cron scheduler off the new image,
which then held the crontab's `0 7,14,19 * * *` plus `30 15 * * *` and would
have run `scripts/forecast_daily.py` unscoped over the whole country/type
matrix, into the same forecast database as the live container. It was caught by
`docker ps` inside a minute; nothing about the invocation announced it.

The blast radius is what makes this worth a test rather than a one-line fix and
a memory: every `docker run` against this image is affected the same way -- a
debug shell, a CI smoke check, an interactive `bash` -- and each one becomes a
duplicate unscoped production writer. The cost of the mistake is not paid by the
person who makes it.

**Getting it wrong in the other direction is worse**, which is the other half of
what this file pins. If the guard is written so that the *no-command* path stops
reaching cron, the scheduler silently does not start: `docker compose up -d
forecast` reports a healthy running container, nothing errors, and forecasts
just stop being produced until somebody notices missing rows days later. So both
branches are asserted, not just the one the incident was about.

The override branch is checked by running the real script -- it is safe to
execute with arguments precisely because the fix `exec`s them before the script
touches `/etc/environment` or the log volume, and that ordering is itself part
of what is being asserted. The cron branch cannot be executed here (it needs
root, a real cron and `/app/logs`), so it is checked statically, against the
source, for the fallthrough that must survive.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh"

BASH = shutil.which("bash")


def _source() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def test_entrypoint_exists_and_is_a_bash_script():
    assert ENTRYPOINT.is_file(), f"{ENTRYPOINT} is missing"
    assert _source().startswith("#!/bin/bash"), "entrypoint must keep its bash shebang"


@pytest.mark.skipif(BASH is None, reason="bash not available on this box")
def test_a_command_override_runs_instead_of_cron():
    """`docker run <img> <cmd>` must run <cmd>. This is the ABL-596 bug."""
    result = subprocess.run(
        [BASH, str(ENTRYPOINT), "printf", "ran-the-override"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"entrypoint failed on the override path: {result.stderr!r}"
    )
    assert "ran-the-override" in result.stdout, (
        "the entrypoint did not run the command it was given -- it discarded "
        f"the override. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "Starting energy forecast scheduler" not in result.stdout, (
        "a command override announced the scheduler; the `exec \"$@\"` branch "
        "must sit above the cron setup so a probe leaves no scheduler trace"
    )


@pytest.mark.skipif(BASH is None, reason="bash not available on this box")
def test_the_override_exec_happens_before_any_side_effect():
    """An override must not write /etc/environment or the log volume.

    Both writes target absolute paths that exist in the container, so on a
    developer box they either fail (permission) or, worse, land somewhere real.
    Reaching them at all means the guard is in the wrong place.
    """
    result = subprocess.run(
        [BASH, str(ENTRYPOINT), "true"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        "the entrypoint reached a side effect before exec'ing the override: "
        f"{result.stderr!r}"
    )
    assert result.stderr == "", (
        f"unexpected stderr on the override path: {result.stderr!r}"
    )


def test_the_no_command_path_still_reaches_cron():
    """The compose path passes no CMD and must still start the scheduler.

    Asserted on the source: the last `exec` in the file is cron, and it is not
    nested inside the argument guard.
    """
    source = _source()

    assert "exec cron -f" in source, (
        "the no-command path no longer starts cron -- a container that starts "
        "cleanly and never forecasts is the worse failure of the two"
    )

    lines = [line.rstrip() for line in source.splitlines()]
    exec_args = [i for i, line in enumerate(lines) if line.strip() == 'exec "$@"']
    exec_cron = [i for i, line in enumerate(lines) if line.strip() == "exec cron -f"]

    assert exec_args, 'the entrypoint must honour an override with `exec "$@"`'
    assert exec_cron, "the entrypoint must still fall through to `exec cron -f`"
    assert max(exec_args) < min(exec_cron), (
        "`exec cron -f` must be the fallthrough, reached only when no command "
        "was passed"
    )

    # The cron fallthrough is at top level, not inside the guard: an `exec`
    # indented under the `if` would never run for the compose path.
    cron_line = lines[min(exec_cron)]
    assert cron_line == "exec cron -f", (
        f"`exec cron -f` is indented ({cron_line!r}) -- it must be top level so "
        "the no-command path reaches it"
    )


def test_cron_env_passthrough_survives_on_the_scheduler_path():
    """cron runs in a clean env; ENERGY_DB_PATH has to be handed to it.

    Losing this does not stop the scheduler -- it silently repoints the write.
    """
    source = _source()
    assert "/etc/environment" in source and "ENERGY_DB_PATH" in source, (
        "the cron environment passthrough was dropped; cron would run without "
        "ENERGY_DB_PATH and write to the default database path"
    )
