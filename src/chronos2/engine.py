"""Chronos-2 forecast engine for the energy dashboard.

Ported from netpredict2's chronos_engine.py, adapted from zone-based to
country-based architecture. Uses the Chronos2Pipeline API with dict-based
inputs supporting covariates.

Key parameters (from netpredict2 research):
- Context length: 672 hours (4 weeks) — proven optimal
- Prediction length: 24 hours (1 day)
- 9 quantiles: [0.1, 0.2, ..., 0.9]
"""

import logging
import torch
import numpy as np
from pathlib import Path
from chronos import BaseChronosPipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger("energy_forecast.chronos2")


class ChronosEngine:
    """Wrapper around Chronos-2 pipeline for energy forecasting.

    Uses Chronos2Pipeline which supports covariates via dict-based inputs:
    - target: 1D array of shape (history_length,)
    - past_covariates: dict of {name: array(history_length)}
    - future_covariates: dict of {name: array(prediction_length)}
      Keys in future_covariates must be a subset of past_covariates keys.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        context_length: int | None = None,
        prediction_length: int | None = None,
        point_forecast_quantile: float | None = None,
    ):
        """Initialize the Chronos-2 engine.

        Args:
            model_path: Path to fine-tuned model checkpoint, or None for pretrained.
            device: 'cuda' or 'cpu'
            context_length: Number of past hours to use (default: 672 from config)
            prediction_length: Number of hours to forecast (default: 24 from config)
            point_forecast_quantile: If set, use this quantile as point forecast
                                     instead of mean.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.context_length = context_length or config.CHRONOS2_CONTEXT_LENGTH
        self.prediction_length = prediction_length or config.CHRONOS2_PREDICTION_LENGTH
        self.quantile_levels = config.CHRONOS2_QUANTILE_LEVELS

        # Configurable point forecast (mean vs specific quantile)
        self.point_forecast_quantile = point_forecast_quantile
        self._point_quantile_idx = None
        if point_forecast_quantile is not None:
            if point_forecast_quantile not in self.quantile_levels:
                raise ValueError(
                    f"point_forecast_quantile={point_forecast_quantile} "
                    f"not in quantile_levels={self.quantile_levels}"
                )
            self._point_quantile_idx = self.quantile_levels.index(point_forecast_quantile)

        # Load model
        if model_path and Path(model_path).exists():
            from chronos import Chronos2Pipeline
            logger.info(f"Loading fine-tuned Chronos-2 from '{model_path}' on {self.device}")
            self.pipeline = Chronos2Pipeline.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        else:
            model_name = config.CHRONOS2_MODEL_NAME
            if model_path:
                logger.warning(f"Fine-tuned model not found at '{model_path}', using pretrained")
            logger.info(f"Loading pretrained Chronos-2 '{model_name}' on {self.device}")
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        logger.info("Chronos-2 model loaded")

    def _clean_target(self, target: np.ndarray) -> np.ndarray:
        """Truncate target to context_length and replace NaNs."""
        if len(target) > self.context_length:
            target = target[-self.context_length:]
        return np.nan_to_num(target, nan=0.0).astype(np.float32)

    def _clean_past_covariates(
        self, past_covariates: dict[str, np.ndarray], target_len: int
    ) -> dict[str, np.ndarray]:
        """Clean and align past covariates to target length."""
        cleaned = {}
        for name, arr in past_covariates.items():
            arr_clean = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            if len(arr_clean) > target_len:
                arr_clean = arr_clean[-target_len:]
            elif len(arr_clean) < target_len:
                pad = np.zeros(target_len - len(arr_clean), dtype=np.float32)
                arr_clean = np.concatenate([pad, arr_clean])
            cleaned[name] = arr_clean
        return cleaned

    def _clean_future_covariates(
        self, future_covariates: dict[str, np.ndarray],
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Clean and align future covariates to the horizon."""
        horizon = prediction_length or self.prediction_length
        cleaned = {}
        for name, arr in future_covariates.items():
            arr_clean = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            if len(arr_clean) > horizon:
                arr_clean = arr_clean[-horizon:]
            elif len(arr_clean) < horizon:
                pad_val = arr_clean[-1] if len(arr_clean) > 0 else 0.0
                pad = np.full(
                    horizon - len(arr_clean), pad_val, dtype=np.float32
                )
                arr_clean = np.concatenate([arr_clean, pad])
            cleaned[name] = arr_clean
        return cleaned

    def _extract_result(self, quantiles: np.ndarray, mean: np.ndarray) -> dict:
        """Build result dict from raw model output."""
        if self._point_quantile_idx is not None:
            point_forecast = quantiles[:, self._point_quantile_idx]
        else:
            point_forecast = mean

        return {
            "median": point_forecast,
            "mean": mean,
            "quantiles": quantiles.T,  # (n_quantiles, prediction_length)
        }

    def forecast(
        self,
        target: np.ndarray,
        past_covariates: dict[str, np.ndarray] | None = None,
        future_covariates: dict[str, np.ndarray] | None = None,
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Forecast a single target series.

        Args:
            target: 1D array of past target values, shape (history_length,)
            past_covariates: dict of {name: array(history_length)}
            future_covariates: dict of {name: array(prediction_length)}
                Keys must be subset of past_covariates keys.
            prediction_length: Horizon for this call, overriding the instance
                default. The D+2 path varies it per run, because the context
                ends wherever the observations actually stop.

        Returns:
            dict with:
                "median": shape (prediction_length,) — point forecast
                "mean": shape (prediction_length,) — raw mean
                "quantiles": shape (n_quantiles, prediction_length)
        """
        horizon = prediction_length or self.prediction_length
        target_clean = self._clean_target(target)
        input_dict = {"target": target_clean}

        if past_covariates:
            input_dict["past_covariates"] = self._clean_past_covariates(
                past_covariates, len(target_clean)
            )

        if future_covariates:
            input_dict["future_covariates"] = self._clean_future_covariates(
                future_covariates, horizon
            )

        quantiles_list, mean_list = self.pipeline.predict_quantiles(
            [input_dict],
            prediction_length=horizon,
            quantile_levels=self.quantile_levels,
        )

        quantiles = quantiles_list[0].squeeze(0).numpy()  # (prediction_length, n_quantiles)
        mean = mean_list[0].squeeze(0).numpy()  # (prediction_length,)

        return self._extract_result(quantiles, mean)

    def forecast_batch(
        self,
        inputs: list[dict],
        batch_size: int = 64,
    ) -> list[dict[str, np.ndarray]]:
        """Forecast multiple targets in a batch.

        Args:
            inputs: list of dicts, each with 'target' and optional covariates.
            batch_size: batch size for inference.

        Returns:
            List of result dicts with 'median', 'mean', and 'quantiles'.
        """
        prepared = []
        for inp in inputs:
            target_clean = self._clean_target(inp["target"])
            d = {"target": target_clean}

            if inp.get("past_covariates"):
                d["past_covariates"] = self._clean_past_covariates(
                    inp["past_covariates"], len(target_clean)
                )

            if inp.get("future_covariates"):
                d["future_covariates"] = self._clean_future_covariates(
                    inp["future_covariates"]
                )

            prepared.append(d)

        quantiles_list, mean_list = self.pipeline.predict_quantiles(
            prepared,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            batch_size=batch_size,
        )

        results = []
        for i in range(len(prepared)):
            quantiles = quantiles_list[i].squeeze(0).numpy()
            mean = mean_list[i].squeeze(0).numpy()
            results.append(self._extract_result(quantiles, mean))

        return results


# Country-based cross-learning groups (adapted from netpredict2's zone-based groups)
CROSS_LEARNING_GROUPS = {
    "central_europe": ["DE", "AT", "CZ", "PL", "SK", "HU", "CH"],
    "western_europe": ["FR", "BE", "NL"],
    "nordic": ["NO", "SE", "FI", "EE", "LT", "LV"],
    "iberian": ["ES", "PT"],
    "southeastern": ["BG", "RO", "GR", "HR", "SI"],
    "italian": ["IT"],
}


def group_inputs_for_cross_learning(
    inputs: list[dict],
    country_codes: list[str],
) -> list[tuple[list[dict], list[str]]]:
    """Group inputs by geographic region for joint batch processing.

    Series within the same region are processed together in a single batch,
    allowing the model to share patterns across correlated series.

    Args:
        inputs: list of input dicts
        country_codes: corresponding country codes

    Returns:
        list of (grouped_inputs, grouped_codes) tuples
    """
    code_to_idx = {code: i for i, code in enumerate(country_codes)}

    code_to_group = {}
    for group_name, members in CROSS_LEARNING_GROUPS.items():
        for code in members:
            if code in code_to_idx:
                code_to_group[code] = group_name

    groups: dict[str, tuple[list[dict], list[str]]] = {}
    ungrouped_inputs = []
    ungrouped_codes = []

    for code, idx in code_to_idx.items():
        group = code_to_group.get(code)
        if group:
            if group not in groups:
                groups[group] = ([], [])
            groups[group][0].append(inputs[idx])
            groups[group][1].append(code)
        else:
            ungrouped_inputs.append(inputs[idx])
            ungrouped_codes.append(code)

    result = list(groups.values())
    if ungrouped_inputs:
        result.append((ungrouped_inputs, ungrouped_codes))

    return result
