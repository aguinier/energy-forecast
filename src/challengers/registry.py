"""Which challengers exist, and what each is called in the database (ABL-68).

`model_name` is the identity that matters downstream: the eval filters on it
(`evaluate_net_position.py --model`), and the prod push filters on it to keep
shadow models out of production. `model_version` is the vintage timestamp, not
a model identity — two challengers sharing a `generated_at` are told apart by
`model_name` alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class ChallengerSpec:
    experiment_id: str
    model_name: str
    kind: str            # "baseline_ensemble" | "correction_layer"
    description: str
    promotion_candidate: bool
    parent: str | None = None   # experiment whose forecast this one corrects


CHALLENGERS: dict[str, ChallengerSpec] = {
    "V012": ChallengerSpec(
        experiment_id="V012",
        model_name="baseline-V012",
        kind="baseline_ensemble",
        description="Serve-faithful persistence + hour-of-day climatology. The "
                    "floor every challenger must beat; not a promotion candidate.",
        promotion_candidate=False,
    ),
    "V016": ChallengerSpec(
        experiment_id="V016",
        model_name="chronos-2-V016",
        kind="correction_layer",
        description="Per-country affine recalibration of V010 plus AR(1) error "
                    "correction at the true serve lead.",
        promotion_candidate=True,
        parent="V010",
    ),
}

CHAMPION_EXPERIMENT = "V010"
CHAMPION_MODEL_NAME = "chronos-2-V010"


def spec_for(experiment_id: str) -> ChallengerSpec:
    try:
        return CHALLENGERS[experiment_id]
    except KeyError:
        raise KeyError(
            f"{experiment_id} is not a registered challenger. Serving an "
            f"unregistered model would put rows in the sidecar that nothing "
            f"knows how to score. Registered: {', '.join(sorted(CHALLENGERS))}"
        ) from None


def model_name_for(experiment_id: str) -> str:
    """Stored `model_name` for any experiment, challenger or champion."""
    if experiment_id in CHALLENGERS:
        return CHALLENGERS[experiment_id].model_name
    cfg_path = config.EXPERIMENTS_DIR / experiment_id / "config.json"
    if cfg_path.exists():
        cfg = json.loads(Path(cfg_path).read_text())
        declared = cfg.get("model", {}).get("model_name")
        if declared:
            return declared
        model_type = cfg.get("model", {}).get("type")
        if model_type:
            return f"{model_type}-{experiment_id}"
    raise KeyError(f"cannot resolve a model_name for {experiment_id}")
