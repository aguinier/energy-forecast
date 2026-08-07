"""Challenger models served in shadow beside the champion (ABL-68).

Challengers run in the same daily job on the same serve-time inputs and write
distinct `model_name` rows to the sidecar. They are never pushed to production —
`push_net_position_forecast.py` names the champion explicitly and filters on it.

Registered here rather than discovered, so "what runs in shadow tomorrow" is a
reviewable list rather than a directory listing.
"""

from .registry import CHALLENGERS, ChallengerSpec, model_name_for, spec_for

__all__ = ["CHALLENGERS", "ChallengerSpec", "model_name_for", "spec_for"]
