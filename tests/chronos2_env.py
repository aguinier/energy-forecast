"""Whether the chronos-2 dependency set can be imported here (ABL-715).

`src/chronos2/engine.py:14` and `src/chronos2/finetuner.py:19` do a module-level
`import torch`, and **torch is installed out of band**: `requirements-chronos.txt`
documents a CUDA-index command in a comment, and no requirements file in this
repo declares a version pip resolves on its own. So the chronos-2 entry points
cannot be imported from a clean checkout on any machine -- CI, a second
workstation, or the serving container. That is a real gap, not a CI artifact,
and it is filed as one; what is decided here is only whether the suite should
answer it by installing torch or by gating.

It gates, for two reasons that are about cost rather than tidiness:

- Installing torch on every pull request is multi-GB of download and build for
  four tests, paid on every run of every PR forever.
- The torch a Linux runner would install is not the one this box serves from.
  `requirements-chronos.txt` pins the workstation's wheel by CUDA index and
  Python 3.14; CI is Python 3.11 on ubuntu. Installing *a* torch to make four
  tests green would be testing a dependency nothing serves, which is the exact
  failure ABL-597 froze `requirements.txt` to prevent.

`chronos-2` is `enabled: False, production: False` in `config.MODEL_RUNNERS`
(config.py:545-553), so nothing these tests cover is being served. If that
changes -- if chronos-2 is ever enabled -- the dependency has to become
installable first, and this gate should go with it.

**The gate is the dependency's absence, never the runner's name.** Where torch
is present, which is the workstation venv these scripts are actually run from,
every one of these tests runs for real and must pass. A gate keyed on the name
would have made them dead everywhere, which is worse than the red build it
replaced.
"""

import importlib.util

__all__ = ["TORCH_AVAILABLE", "SCRIPTS", "RUNNERS", "REASON", "skip_reason"]


def _torch_available() -> bool:
    """True if `import torch` would resolve for this interpreter.

    `find_spec` raises rather than returning None when a *parent* package is
    broken, and a half-installed torch is exactly the state this predicate has
    to survive: the answer we want in that case is still "no".
    """
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


TORCH_AVAILABLE = _torch_available()

#: The `scripts/*.py` entry points that reach `src/chronos2/`, by filename as
#: `test_script_imports` parametrises them. Listed rather than derived: a new
#: chronos-2 script should fail loudly and be added deliberately, not be gated
#: by a pattern match the day someone names a file well.
SCRIPTS = frozenset({"forecast_chronos2.py", "train_chronos2.py"})

#: `config.MODEL_RUNNERS` names whose script is one of the above.
RUNNERS = frozenset({"chronos-2"})

REASON = (
    "torch is not installed here, and no requirements file in this repo "
    "declares it (requirements-chronos.txt documents a CUDA-index install in a "
    "comment). The chronos-2 entry points import it at module level, so they "
    "cannot be imported from a clean checkout -- ABL-715. Gated, not fixed: "
    "chronos-2 is enabled=False, production=False, and installing a multi-GB "
    "GPU dependency on every PR to cover four tests would also be testing a "
    "torch build nothing serves."
)


def skip_reason(name: str) -> str:
    """The gate reason with the thing being skipped named in it.

    CI runs pytest with `-rs`, so this string is what a reader gets when they
    ask why the count moved -- it should say which entry point went and why,
    not just that something did.
    """
    return f"{name}: {REASON}"
