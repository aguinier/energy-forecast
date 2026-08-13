"""How a subprocess model runner reports what it produced (ABL-370).

`scripts/forecast_daily.py` launches the external entries in
`config.MODEL_RUNNERS` as subprocesses and used to decide the outcome from the
exit code alone: exit 0 meant `OK`. A runner that ran cleanly and generated
**nothing** — every `tso-correction` invocation on a day the upstream Elia
forecast has not landed yet — printed the same `OK` as one that saved 96 rows,
and contributed a 0 to a `Total forecasts:` sum the in-process xgboost models
push into the thousands. Nothing distinguished "ran and produced N" from "ran
and produced none", which is the exact condition that kept ABL-354's dead
runner invisible for as long as it was.

So the exit code carries "did it crash", and this line carries "what did it
produce":

    FORECAST_RECORDS=<n>

Every external runner emits it on stdout exactly once per run, including — and
especially — when `n` is 0. `forecast_daily` parses it back and gets three
outcomes instead of two: rows produced, ran-but-produced-none, failed.

`None` is the fourth thing, and it is not 0. A runner that exits 0 without
emitting the line has told us nothing about what it did; recording that as 0
would be a number we did not measure, and recording it as success is the bug
this module exists to remove. It is reported as `unreported` and shown in the
summary as its own outcome.

Nothing here imports pandas or touches the DB, deliberately: a runner may be
launched under a foreign interpreter (`chronos-bolt-small` uses its own venv),
and importing this must never be the thing that breaks it.
"""

from typing import List, Optional

# The whole contract. Written only by emit_record_count, read only by
# parse_record_count, so the two sides cannot drift apart.
RECORD_COUNT_PREFIX = "FORECAST_RECORDS="

# Outcomes forecast_daily distinguishes. 'failed' is decided by the exit code
# and lives in forecast_daily; the three below are decided by the count.
STATUS_SUCCESS = "success"        # ran, produced >= 1 row
STATUS_EMPTY = "empty"            # ran, reported exactly 0 rows
STATUS_UNREPORTED = "unreported"  # ran, reported nothing — count unknown
STATUS_FAILED = "failed"


def emit_record_count(count: int) -> None:
    """Report how many forecast rows this run produced, on stdout.

    Call it on every path that exits 0, zero rows included. Flushed, because
    the parent captures stdout through a pipe and a runner that is killed by
    the 300s timeout would otherwise lose a buffered line.
    """
    print(f"{RECORD_COUNT_PREFIX}{int(count)}", flush=True)


def parse_record_count(stdout: str) -> Optional[int]:
    """Read the count back out of a runner's stdout, or None if it never said.

    Matches only a whole line, so prose that happens to quote the prefix does
    not count as a report. If a runner emits more than once the last line wins
    — a later total supersedes an earlier partial — but the contract is one
    line per run.
    """
    count = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith(RECORD_COUNT_PREFIX):
            continue
        try:
            count = int(line[len(RECORD_COUNT_PREFIX):].strip())
        except ValueError:
            # A malformed count is not a count. Leave whatever we had; if it is
            # still None the run reads as `unreported`, which is the truth.
            continue
    return count


def status_for_count(records: Optional[int]) -> str:
    """Outcome for a runner that exited 0, from what it reported."""
    if records is None:
        return STATUS_UNREPORTED
    return STATUS_SUCCESS if records > 0 else STATUS_EMPTY


def summarize_by_runner(results: List[dict]) -> List[dict]:
    """Per-runner outcome counts, in first-seen order.

    The aggregate `Total forecasts:` line cannot show a zero: the in-process
    models contribute thousands of rows and a runner producing nothing is
    arithmetically invisible inside that sum. This splits the sum by runner so
    the zero has somewhere to be seen.

    `rows` sums only the counts a runner actually reported; `unreported` runs
    contribute no rows because their count is unknown, not because it is 0.
    """
    order: List[str] = []
    by_name = {}

    for result in results:
        name = result.get("runner") or "builtin"
        if name not in by_name:
            order.append(name)
            by_name[name] = {
                "runner": name,
                "runs": 0,
                "success": 0,
                "empty": 0,
                "unreported": 0,
                "skipped": 0,
                "failed": 0,
                "rows": 0,
            }
        entry = by_name[name]
        entry["runs"] += 1
        entry["rows"] += result.get("records") or 0

        status = result.get("status")
        if status == STATUS_FAILED and is_skip(result):
            entry["skipped"] += 1
        elif status in entry:
            entry[status] += 1
        else:
            entry["failed"] += 1

    return [by_name[name] for name in order]


def is_skip(result: dict) -> bool:
    """A 'failed' result that is really "there was no model to run".

    `forecast_daily` has counted these separately from real failures for a long
    time, but it decided by looking for `not found` in the error text, and that
    text is not only produced by a missing model. A runner whose configured
    `python_executable` does not exist fails with `Executable not found:
    [WinError 2]` — a runner that cannot run at all, counted as benign. Same
    species of defect as ABL-370, one layer down.

    So the skip is a flag the one place that knows sets, not a phrase anyone can
    trip over.
    """
    return bool(result.get("skipped"))


def format_runner_summary(results: List[dict]) -> List[str]:
    """The per-runner block of the run summary, as log lines."""
    rows = summarize_by_runner(results)
    if not rows:
        return []

    # A runner every one of whose runs went unreported has no row count at all;
    # printing 0 for it would state a number we never measured.
    def rows_text(r):
        return "no reported" if r["unreported"] == r["runs"] else f"{r['rows']:,}"

    width = max(len(r["runner"]) for r in rows)
    lines = ["Per runner:"]
    for r in rows:
        lines.append(
            f"  {r['runner']:<{width}}  {r['success']:>3} ok, {r['empty']:>3} empty, "
            f"{r['unreported']:>3} unreported, {r['skipped']:>3} skipped, "
            f"{r['failed']:>3} failed -> {rows_text(r)} rows"
        )

    # The line ABL-354 needed: a runner that ran and produced nothing at all.
    barren = [r for r in rows if r["rows"] == 0 and r["runs"] > r["skipped"]]
    if barren:
        lines.append("")
        lines.append("Runners that produced no forecasts:")
        for r in barren:
            lines.append(
                f"  - {r['runner']}: {r['runs']} runs, {rows_text(r)} rows "
                f"({r['empty']} empty, {r['unreported']} unreported, "
                f"{r['skipped']} skipped, {r['failed']} failed)"
            )
    return lines
