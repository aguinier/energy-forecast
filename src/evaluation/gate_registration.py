"""Registration tables for the pre-registered gate harnesses, checked together.

ABL-387. `scripts/evaluate_wind_retrain.py` and `scripts/evaluate_solar_retrain.py`
each carry a set of module-level tables keyed by scope name -- `SCOPES` (the
pre-registered pairs or countries), `GATE_BASIS` (the columns that must be
simultaneously finite for a row to be scored) and now `SCOPE_OUTPUTS` (where the
run writes). They are three views of one registration, and a scope that appears
in one but not the others is not a partial registration: it is a run that either
raises `KeyError` at an arbitrary point, or -- the case this module exists for --
writes its results over a different scope's evidence.

The output tables are the reason this is checked rather than commented. Before
ABL-387 the three `--artifact-dir` / `--json-out` / `--report-out` defaults were
fixed strings resolved before `--scope` was consulted, so a scoped run that
omitted three flags overwrote a *dispositioned* gate read in place --
`experiments/ABL195/results.json` and `experiments/ABL253/results.json`, both
backing dispositions already reported. That run succeeds. It emits a full report.
The damage is to evidence rather than to anything the run's exit status shows,
which is why the check has to fire at import and not at the write.

`check_registration_tables` is called at module scope in both harnesses, so a
scope added to `SCOPES` without its basis and its outputs fails on import --
before a fit, and identically in `--help`, in the test suite and in a gate run.
It lives here rather than in either harness because landing a fix in one twin and
not the other has been this pair's recurring failure mode (ABL-322/ABL-379,
ABL-345/ABL-347): one import, one message, both harnesses.
"""

from __future__ import annotations

from typing import Iterable, Mapping

#: The keys a `SCOPE_OUTPUTS` entry must carry, one per output path a harness
#: writes. Named here so a new scope cannot register two of the three and send
#: its third output to the previous occupant of that path.
REQUIRED_OUTPUT_KEYS = ("artifact_dir", "json_out", "report_out")


def check_registration_tables(**tables: Mapping[str, object]) -> None:
    """Require every named registration table to be keyed by the same scopes.

    Args:
        **tables: The registration tables, passed by their module-level name --
            ``check_registration_tables(SCOPES=SCOPES, GATE_BASIS=GATE_BASIS,
            SCOPE_OUTPUTS=SCOPE_OUTPUTS)``. The names are used verbatim in the
            error, so the caller reads which table to edit rather than which
            argument position disagreed.

    Raises:
        ValueError: If fewer than two tables are given (nothing is being
            cross-checked, which is a caller bug rather than a registration
            one), or if any scope is absent from any table. The message names
            every missing (table, scope) pair, so a scope added to one of three
            tables reports both omissions at once instead of one per re-run.
    """
    if len(tables) < 2:
        raise ValueError("check_registration_tables needs at least two tables to compare")
    every_scope = sorted({scope for table in tables.values() for scope in table})
    missing = [f"{name} is missing {scope!r}"
               for name, table in tables.items()
               for scope in every_scope if scope not in table]
    if missing:
        raise KeyError(
            "gate scope registration tables disagree; every scope must appear in all of "
            # ASCII only: this is raised at *import*, so it can surface in a
            # traceback printed to a cp1252 pipe, where a non-ASCII character
            # raises UnicodeEncodeError and masks the message (cf. ABL-364).
            f"{', '.join(tables)}: " + "; ".join(missing))


def check_scope_outputs(scope_outputs: Mapping[str, Mapping[str, str]],
                        required_keys: Iterable[str] = REQUIRED_OUTPUT_KEYS) -> None:
    """Require every scope's output entry to be complete and to collide with no other.

    Args:
        scope_outputs: The harness's `SCOPE_OUTPUTS`, scope name -> output path
            mapping.
        required_keys: The output names each entry must carry.

    Raises:
        KeyError: If an entry omits a required output, or carries one that is
            not required -- a typo'd key would otherwise leave the real key
            unset and fall through to whatever the caller defaults to.
        ValueError: If two scopes register the same path for the same output.
            Distinctness is the whole point of the table: two scopes sharing a
            `json_out` reproduces the defect keyed outputs were introduced to
            remove, only between two named scopes rather than between a scope
            and a hardcoded default.
    """
    required = tuple(required_keys)
    for scope, entry in scope_outputs.items():
        if set(entry) != set(required):
            raise KeyError(
                f"SCOPE_OUTPUTS[{scope!r}] must register exactly {sorted(required)}, "
                f"got {sorted(entry)}")
    for key in required:
        seen: dict[str, str] = {}
        for scope, entry in scope_outputs.items():
            path = entry[key]
            if path in seen:
                raise ValueError(
                    f"SCOPE_OUTPUTS[{scope!r}][{key!r}] is {path!r}, already registered by "
                    f"scope {seen[path]!r}; a gate run would overwrite that scope's evidence")
            seen[path] = scope
