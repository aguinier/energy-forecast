"""Every `scripts/*.py` entry point must actually import (ABL-340).

`scripts/train.py` was import-dead from ABL-188 (`574eb80`) until ABL-340 —
seven months of a documented CLI that raised `ImportError` on `--help`. Nothing
caught it because no test ever imported a script.

The bug class: a script puts `src/` on `sys.path` and imports its modules flat
(`import db`), so `src/db.py` loads with no parent package and its
`from .data_quality import ...` raises. The invariant these tests hold is that
`src` is imported *as a package* — repo root on `sys.path`, `from src.db import
...` — everywhere, so no module is ever reachable under two names.

`test_script_import_preamble` executes each script's module-level import
statements (and only those: no `def`, no `class`, no `if __name__`), which is
the cheapest thing that proves the import graph resolves. It runs the whole set
in one subprocess and purges repo-local modules between scripts, so each script
re-imports `src.*` from scratch while third-party wheels stay cached.

`config.MODEL_RUNNERS` is the second set of entry points, and it was not
covered here until ABL-354: `forecast_daily.py` launches those as subprocesses,
and one of them (`src/tso_correction_forecaster.py`) lives *inside* the package.
Running a package module by path gives it no parent package, so the relative
import that `test_no_flat_intra_src_imports` requires is the one by-path
execution forbids — the two guards pointed opposite ways, and every BE solar /
wind forecast from the `tso-correction` runner failed for it while the job still
logged `[DONE]`. The runner tests below close that: they assert each runner is
launched in a mode its own imports can survive, and then prove it by launching
it.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Every entry point: scripts/ plus the repo-root runners (run_forecast.py, config.py).
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py")) + sorted(REPO_ROOT.glob("*.py"))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import config  # noqa: E402
import forecast_daily  # noqa: E402

# Every configured subprocess entry point, enabled or not: a disabled runner is
# one config flip away from being launched, so it is held to the same rule.
RUNNERS = [r for r in config.MODEL_RUNNERS if r.get("script")]

# Top-level module names inside src/. Imported flat, each of these either
# resolves to a second, independent copy of a src module or raises ImportError.
SRC_MODULES = sorted(
    p.stem for p in (REPO_ROOT / "src").iterdir()
    if (p.suffix == ".py" and p.stem != "__init__") or (p.is_dir() and (p / "__init__.py").exists())
)

_CHILD = r'''
import ast, json, sys, traceback
from pathlib import Path

root = Path(sys.argv[1])
scripts = sorted((root / "scripts").glob("*.py")) + sorted(root.glob("*.py"))


def is_path_call(node, text):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and "sys.path" in (ast.get_source_segment(text, node) or ""))


def preamble(path):
    """Module-level imports + sys.path setup, and nothing that does real work.

    Bounded by the last import / sys.path statement, so plain `NAME = ...`
    bindings the path expressions rely on (`ROOT = Path(__file__)...`) are kept
    while constants and side effects further down the module are dropped.
    """
    text = path.read_text(encoding="utf-8")
    top = ast.parse(text).body
    last = max((i for i, n in enumerate(top)
                if isinstance(n, (ast.Import, ast.ImportFrom)) or is_path_call(n, text)),
               default=-1)

    future, body = [], []
    for node in top[:last + 1]:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) or is_path_call(node, text):
            body.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) for t in node.targets
        ):
            body.append(node)

    head = ast.unparse(ast.Module(body=future, type_ignores=[]))
    rest = ast.unparse(ast.Module(body=body, type_ignores=[]))
    # __future__ must stay first; __file__ drives the sys.path expressions.
    return "%s\n__file__ = %r\n%s" % (head, str(path), rest)


def purge():
    """Drop repo-local modules so the next script re-imports them for real."""
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or "site-packages" in f:
            continue
        try:
            Path(f).resolve().relative_to(root)
        except ValueError:
            continue
        del sys.modules[name]


results = {}
for script in scripts:
    saved = list(sys.path)
    try:
        exec(compile(preamble(script), "<%s preamble>" % script.name, "exec"),
             {"__name__": "__preamble__"})
        results[script.name] = None
    except BaseException:
        results[script.name] = traceback.format_exc().strip()
    finally:
        sys.path[:] = saved
        purge()

print("---RESULTS---")
print(json.dumps(results))
'''


@pytest.fixture(scope="module")
def preamble_errors():
    """{script name: traceback or None}, from one subprocess for the whole set."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(REPO_ROOT)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
    )
    marker = "---RESULTS---"
    if marker not in proc.stdout:
        pytest.fail(
            "import probe did not report:\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout.split(marker, 1)[1])


def test_scripts_directory_is_not_empty():
    # Guards the parametrisation below against silently covering nothing.
    assert len(SCRIPTS) >= 30, f"only found {len(SCRIPTS)} scripts"


@pytest.mark.parametrize("script", [p.name for p in SCRIPTS])
def test_script_import_preamble(script, preamble_errors):
    error = preamble_errors[script]
    assert error is None, (
        f"scripts/{script} does not import.\n\n{error}\n\n"
        "Import src as a package: put the repo root (not src/) on sys.path and "
        "use `from src.db import ...`, not `import db` (ABL-340)."
    )


@pytest.mark.parametrize(
    "py_file",
    [p.relative_to(REPO_ROOT).as_posix() for p in sorted((REPO_ROOT / "src").rglob("*.py"))],
)
def test_no_flat_intra_src_imports(py_file):
    """Inside src/, sibling modules are imported relatively — never flat.

    A flat `from db import ...` inside src/ resolves only while src/ is on
    sys.path, and then loads a *second* copy of the module: `db` and `src.db`
    become separate objects with separate module state.
    """
    tree = ast.parse((REPO_ROOT / py_file).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in SRC_MODULES:
                offenders.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in SRC_MODULES:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, (
        f"{py_file} imports a sibling src module flat:\n  " + "\n  ".join(offenders)
        + "\nUse a relative import (`from .db import ...`) instead (ABL-340)."
    )


# --- config.MODEL_RUNNERS entry points (ABL-354) -----------------------------

def _runner_id(runner):
    return f"{runner['name']}:{runner['script']}"


RUNNER_IDS = [_runner_id(r) for r in RUNNERS]


def test_model_runners_are_covered():
    # Guards the parametrisation below against silently covering nothing.
    assert len(RUNNERS) >= 3, f"only found {len(RUNNERS)} runners with a script"


@pytest.mark.parametrize("runner", RUNNERS, ids=RUNNER_IDS)
def test_model_runner_script_exists(runner):
    script = REPO_ROOT / runner["script"]
    assert script.is_file(), (
        f"MODEL_RUNNERS['{runner['name']}'] points at {runner['script']}, "
        "which does not exist."
    )


@pytest.mark.parametrize("runner", RUNNERS, ids=RUNNER_IDS)
def test_model_runner_launch_mode_matches_its_imports(runner):
    """Each runner is launched in a mode its own import style can survive.

    Two modes, and each rules something out:

    - ``-m src.foo`` needs a real package — every directory from the repo root
      down must have an ``__init__.py``, or the module has no parent package and
      the relative import raises anyway.
    - by path needs no relative imports at all, since a file executed by path is
      ``__main__`` with no package.

    This is the static half; ``test_model_runner_launches`` runs the command.
    """
    cmd = forecast_daily.build_runner_command(runner, [], repo_root=REPO_ROOT)
    script = REPO_ROOT / runner["script"]
    tree = ast.parse(script.read_text(encoding="utf-8"))
    relative = [
        f"line {n.lineno}: from {'.' * n.level}{n.module or ''} import ..."
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.level > 0
    ]

    if cmd[1:2] == ["-m"]:
        module = cmd[2]
        assert module == ".".join(Path(runner["script"]).with_suffix("").parts)
        package = script.parent
        while package != REPO_ROOT:
            assert (package / "__init__.py").is_file(), (
                f"{runner['script']} is launched as `-m {module}`, but "
                f"{package.relative_to(REPO_ROOT).as_posix()}/ has no "
                "__init__.py, so it is not a package."
            )
            package = package.parent
    else:
        assert not relative, (
            f"{runner['script']} is launched by path:\n  {cmd[1]}\n"
            "but imports relatively:\n  " + "\n  ".join(relative)
            + "\nA file run by path has no parent package, so these raise "
            "ImportError. Move the runner under src/ and launch it as "
            "`-m src.<module>` (ABL-354)."
        )


@pytest.mark.parametrize("runner", RUNNERS, ids=RUNNER_IDS)
def test_model_runner_launches(runner):
    """The command `forecast_daily` builds actually starts (ABL-354).

    `--help` is the cheapest argv that runs every module-level import and then
    exits 0 without touching the database or a model. The runner's configured
    `python_executable` is deliberately *not* used: it is an absolute path to
    one box's venv, and this test is about the import graph, not that box.
    Launched with `sys.executable` — the same interpreter the preamble probe
    above uses — and from the repo root, which is where `forecast_daily` runs it.
    """
    cmd = forecast_daily.build_runner_command(runner, ["--help"], repo_root=REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, *cmd[1:]],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
    )
    assert proc.returncode == 0, (
        f"`{' '.join(cmd[1:])}` exits {proc.returncode} — the {runner['name']} "
        f"runner cannot start.\n\nstderr:\n{proc.stderr[-2000:]}\n\n"
        "forecast_daily runs this as a subprocess and only logs the failure, so "
        "the daily job still reports [DONE] with these forecasts missing."
    )
