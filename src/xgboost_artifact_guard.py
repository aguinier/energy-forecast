"""Xgboost intercept witness — a reusable guard against the cross-interpreter
corruption diagnosed for V014 under ABL-69 and required by ABL-183 for the
legacy `Forecaster.save`/`load` renewable artifacts.

An xgboost booster pickled by one xgboost version and unpickled by an
incompatible one does not raise: it silently resets the fitted intercept
(`base_score`) to the library default and predicts a near-flat series,
emitting nothing louder than a `UserWarning`. `src/challengers/v014.py`
carries its own copy of this check, written first; this module exists so a
second caller (`Forecaster`) does not have to re-derive or drift from that
logic. `v014.py` is intentionally left on its own copy rather than refactored
to import this — its guard's error message and tests are pinned to V014's own
wording, and this issue's job is to extend the protection, not to touch code
neither ABL-179 nor ABL-183 asked for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class ModelArtifactError(RuntimeError):
    """Raised when a stored model did not survive deserialisation intact."""


#: The corruption this catches is not a rounding difference — it resets the
#: intercept to the 0.5 default from a value in the hundreds or thousands of
#: MW — so the tolerance only has to absorb float32 round-tripping.
BASE_SCORE_TOLERANCE_MW = 1.0


def base_score(booster) -> Optional[float]:
    """The booster's fitted intercept, read back out of its own saved config.

    xgboost 3.x serialises `base_score` as a JSON array string
    (`'[4.8775327E3]'`, one entry per target); 2.x as a bare scalar string
    (`'5E-1'`). Parsing only the scalar form returns None on every 3.x
    artifact, which would disable the guard silently rather than loudly.
    """
    try:
        cfg = json.loads(booster.get_booster().save_config())
        raw = cfg["learner"]["learner_model_param"]["base_score"]
        if isinstance(raw, str) and raw.lstrip().startswith("["):
            values = json.loads(raw)
            return float(values[0]) if values else None
        return float(raw)
    except Exception:  # noqa: BLE001 - an unreadable config must not break saving
        return None


def assert_survived_load(
    path: Path,
    stored_base_score: Optional[float],
    stored_xgboost_version: Optional[str],
    booster,
    artifact_label: str,
) -> None:
    """Refuse a model whose intercept did not survive deserialisation.

    `stored_base_score` is None for every artifact written before this guard
    existed (all wind artifacts as of ABL-183, some without even an
    `xgboost_version` key) — that means "cannot check", not "corrupt", so
    those load unchecked, same as `load_model` treats an absent witness in
    `src/challengers/v014.py`. The guard only bites on artifacts saved after
    this change, once a real witness exists to compare against.
    """
    if stored_base_score is None:
        return
    actual = base_score(booster)
    if actual is not None and abs(actual - stored_base_score) <= BASE_SCORE_TOLERANCE_MW:
        return

    import sys

    import xgboost

    read = "unreadable" if actual is None else f"{actual:,.4f}"
    raise ModelArtifactError(
        f"{artifact_label} model {path.name} did not survive loading: its "
        f"intercept read back as {read} but was saved as {stored_base_score:,.4f}. "
        f"This model was written by xgboost {stored_xgboost_version or '?'} and is "
        f"being loaded by xgboost {xgboost.__version__} on {sys.executable}. Loading "
        f"it anyway would predict a near-zero-mean series and report the model as "
        f"bad rather than raising. Run this under the interpreter that trained it, "
        f"or retrain."
    )
