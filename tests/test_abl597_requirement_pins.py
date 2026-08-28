"""ABL-597: hold `requirements.txt` at exact pins for the artifact load path.

Every served forecast comes out of a `models/<CC>/<type>/model.joblib` that was
pickled by one specific library build. `models/` is gitignored, so no commit
protects those bytes -- the only thing standing between a serving artifact and
the library that loads it is this requirements file. A `>=` floor means the next
*unrelated* rebuild is free to float that library, and the two failure modes are
not equally loud:

- the load raises, and the runner logs "no trained model", writes nothing for
  that pair, and still exits 0; or
- the load succeeds and the predictions change.

The second is the one this test exists for. CLAUDE.md records the worked
example: an xgboost-3.3.0 pickle read under 2.1.4 keeps its trees and silently
resets the fitted intercept, producing a series with shape and no level -- which
reads as a bad model rather than a bad load.

**Which packages** was determined, not guessed: a `pickletools` opcode replay
over all 67 top-level artifacts, resolving every `GLOBAL` / `STACK_GLOBAL` to a
`(module, qualname)` pair -- those names are literally in the artifact bytes.
67/67 parsed, 0 errors. The counts below are a snapshot and `models/` is
gitignored, so re-derive rather than trust them:

    .venv\\Scripts\\python.exe scripts/abl597_artifact_load_path.py

Full pack: `reports/abl_597_artifact_load_path.md`.

    catboost.core.CatBoostRegressor                48 artifacts
    numpy.core.multiarray.scalar + numpy.dtype     49
    xgboost.core.Booster + .sklearn.XGBRegressor   18
    lightgbm.basic.Booster + .sklearn.LGBMRegressor 1

`scikit-learn`, `scipy` and `joblib` name no symbol in any artifact but are on
the load path all the same -- `xgboost.sklearn` cannot import without sklearn,
sklearn cannot import without scipy, and joblib is the reader itself.

**Values** come from the live container, not from a fresh resolve
(`docker exec energy-forecast python3 -m pip freeze`, 2026-08-28, ABL-598).
`_PRODUCTION_AT_FREEZE` below is that capture. Freezing it here is the point:
the file and the recorded production state have to be edited together, so a
version move is a reviewed decision instead of a resolver outcome.

**This is not a lockfile.** It pins the direct requirements only; the transitive
closure (narwhals, six, threadpoolctl, python-dateutil, ...) still floats. That
gap is real and named in the evidence pack, not silently papered over here.

Raising a pin means re-reading the artifacts under the new version first.
`base_score` is the witness that catches a silent level reset -- and note that
none of the 67 artifacts on disk carry the ABL-183 `xgboost_version` /
`base_score` metadata keys, so the in-artifact guard cannot fire on them. This
file is the only guard those 67 have.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# The artifact load path: unpickle-critical modules plus what they import.
# Every one of these must be `==`, never a floor.
LOAD_PATH_PACKAGES = frozenset(
    {
        "xgboost",
        "lightgbm",
        "catboost",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "joblib",
    }
)

# `docker exec energy-forecast python3 -m pip freeze`, 2026-08-28 (ABL-598),
# restricted to what this file declares. Not a fresh resolve.
_PRODUCTION_AT_FREEZE = {
    "xgboost": "3.2.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "scikit-learn": "1.9.0",
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "scipy": "1.17.1",
    "optuna": "4.9.0",
    "holidays": "0.103",
    "python-dotenv": "1.2.3",
    "pytz": "2026.3.post1",
    "tqdm": "4.70.0",
    "joblib": "1.5.3",
}

_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?P<op>==|>=|<=|~=|>|<|!=)\s*(?P<version>[^\s;#]+)"
)


def _parse(path: Path) -> dict:
    """name -> (operator, version) for every requirement line in `path`."""
    parsed = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT.match(line)
        assert match is not None, f"unparsed requirement line: {raw!r}"
        parsed[match.group("name")] = (match.group("op"), match.group("version"))
    return parsed


@pytest.fixture(scope="module")
def requirements() -> dict:
    return _parse(REQUIREMENTS)


def test_every_load_path_package_is_pinned_exactly(requirements):
    """A floor on any of these lets a rebuild move the library under 67 pickles."""
    floors = {
        name: op
        for name, (op, _) in requirements.items()
        if name in LOAD_PATH_PACKAGES and op != "=="
    }
    assert not floors, (
        f"artifact load-path packages must be pinned with '==', found floors: {floors}. "
        "See this module's docstring: `models/` is gitignored, so requirements.txt is "
        "the only thing pinning the library that loads the serving artifacts."
    )


def test_no_requirement_uses_a_floor(requirements):
    """The whole file is exact, so a rebuild reproduces the image it replaces.

    Scoped to `requirements.txt` -- `requirements-chronos.txt` is the workstation
    GPU venv and is explicitly not installed in the prod container image.
    """
    floors = {name: op for name, (op, _) in requirements.items() if op != "=="}
    assert not floors, f"requirements.txt must use exact pins, found: {floors}"


def test_load_path_packages_are_all_declared(requirements):
    """Determined set, not a guess -- an undeclared member floats transitively."""
    missing = sorted(LOAD_PATH_PACKAGES - set(requirements))
    assert not missing, (
        f"these packages are on the artifact load path but are not declared in "
        f"requirements.txt, so nothing pins them: {missing}"
    )


def test_pins_match_the_recorded_production_versions(requirements):
    """ABL-597 is a freeze at today's known-good, not a dependency bump.

    Editing a pin without editing `_PRODUCTION_AT_FREEZE` goes red. That is
    deliberate: it forces a version move to be a decision with a recorded
    provenance, rather than whatever the resolver returned on rebuild day.
    """
    drift = {
        name: {"requirements.txt": version, "production": _PRODUCTION_AT_FREEZE[name]}
        for name, (_, version) in requirements.items()
        if name in _PRODUCTION_AT_FREEZE and version != _PRODUCTION_AT_FREEZE[name]
    }
    assert not drift, (
        f"pin(s) differ from the recorded production set: {drift}. "
        "If this is a deliberate upgrade, re-read the serving artifacts under the "
        "new version and update _PRODUCTION_AT_FREEZE with a fresh container "
        "`pip freeze` -- do not resolve a conflict by moving a version."
    )


def test_recorded_production_set_covers_every_requirement(requirements):
    """No requirement may exist without a recorded production version behind it."""
    unrecorded = sorted(set(requirements) - set(_PRODUCTION_AT_FREEZE))
    assert not unrecorded, (
        f"requirement(s) with no recorded production version: {unrecorded}. "
        "Add them from a live container `pip freeze`, not from a fresh resolve."
    )
