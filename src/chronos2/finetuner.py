"""Fine-tune Chronos-2 on energy dashboard data.

Ported from netpredict2's finetuner.py, adapted to use the dashboard's
InputBuilder for data loading from SQLite instead of CSV files.

The fine-tuning process:
1. InputBuilder loads target + covariates from DB
2. Training inputs are built for all (country, forecast_type) pairs
3. Chronos2Pipeline.fit() handles random-window cropping internally
4. Model checkpoint saved for inference
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.chronos2.input_builder import InputBuilder

logger = logging.getLogger("energy_forecast.chronos2")


class ChronosFinetuner:
    """Fine-tune Chronos-2 on energy time series from the dashboard database."""

    def __init__(
        self,
        context_length: int | None = None,
        prediction_length: int | None = None,
    ):
        self.context_length = context_length or config.CHRONOS2_CONTEXT_LENGTH
        self.prediction_length = prediction_length or config.CHRONOS2_PREDICTION_LENGTH
        self.input_builder = InputBuilder(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )

    def prepare_training_data(
        self,
        countries: list[str],
        forecast_types: list[str],
        start_date: str,
        end_date: str,
        exclude_backtest: bool = True,
        include_neighbors: bool = False,
        val_fraction: float = 0.0,
    ) -> tuple[list[dict], list[dict] | None, list[tuple[str, str]]]:
        """Prepare training data from the database.

        Args:
            countries: List of country codes, or ["all"]
            forecast_types: List of forecast types, or ["all"]
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            exclude_backtest: Whether to NaN-mask backtest weeks
            include_neighbors: Whether to include neighbor features
            val_fraction: Fraction of series for validation (0 = no validation)

        Returns:
            (train_inputs, val_inputs, series_labels)
        """
        exclude_dates = config.get_backtest_exclude_dates() if exclude_backtest else None

        logger.info(f"Preparing training data: {countries} x {forecast_types}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        if exclude_dates:
            logger.info(f"  Excluding {len(exclude_dates)} backtest periods")

        train_inputs, val_inputs, labels = self.input_builder.build_batch_training_inputs(
            countries=countries,
            forecast_types=forecast_types,
            start_date=start_date,
            end_date=end_date,
            exclude_dates=exclude_dates,
            include_neighbors=include_neighbors,
            val_fraction=val_fraction,
        )

        return train_inputs, val_inputs, labels

    def train(
        self,
        train_inputs: list[dict],
        val_inputs: list[dict] | None = None,
        output_dir: str = "models/chronos2/finetuned",
        model_name: str | None = None,
        num_steps: int = 5000,
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 4,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.1,
        device: str = "cuda",
    ) -> object:
        """Fine-tune Chronos-2 on the prepared training inputs.

        Uses Chronos2Pipeline.fit() which accepts dict-based inputs with
        target + covariates and does internal random-window cropping.

        Args:
            train_inputs: List of training input dicts from prepare_training_data()
            val_inputs: Optional validation inputs
            output_dir: Directory to save the fine-tuned model checkpoint
            model_name: Base model to fine-tune from (default: config value)
            num_steps: Number of training steps (5000 = netpredict2 sweet spot)
            learning_rate: Learning rate (1e-5 = netpredict2 default)
            batch_size: Batch size per step
            gradient_accumulation_steps: Gradient accumulation (effective batch = batch_size * this)
            lr_scheduler_type: LR scheduler ('cosine' recommended from netpredict2)
            warmup_ratio: Warmup fraction of total steps
            device: 'cuda' or 'cpu'

        Returns:
            Fine-tuned Chronos2Pipeline
        """
        from chronos import Chronos2Pipeline

        if model_name is None:
            model_name = config.CHRONOS2_MODEL_NAME

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        device_actual = device if torch.cuda.is_available() else "cpu"
        if device_actual != device:
            logger.warning(f"CUDA not available, falling back to CPU")

        logger.info(f"Loading base model '{model_name}' for fine-tuning on {device_actual}...")
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device_actual,
            torch_dtype=torch.float32,
        )

        logger.info(
            f"Fine-tuning: {len(train_inputs)} series, {num_steps} steps, "
            f"batch_size={batch_size}, grad_accum={gradient_accumulation_steps}, "
            f"lr={learning_rate}, scheduler={lr_scheduler_type}"
        )

        extra_kwargs = {
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataloader_num_workers": 0,  # Windows compatibility
            "report_to": "tensorboard",
            "logging_dir": str(output_path / "runs"),
            "logging_steps": 10,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
        }

        if val_inputs:
            extra_kwargs.update({
                "load_best_model_at_end": True,
                "eval_strategy": "steps",
                "eval_steps": 100,
                "metric_for_best_model": "eval_loss",
                "save_strategy": "steps",
                "save_steps": 100,
                "save_total_limit": 3,
            })
            logger.info(f"Validation: {len(val_inputs)} series, eval every 100 steps")

        finetuned = pipeline.fit(
            inputs=train_inputs,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=str(output_path),
            validation_inputs=val_inputs,
            **extra_kwargs,
        )

        logger.info(f"Fine-tuning complete. Model saved to {output_path}")
        return finetuned

    def run_experiment(
        self,
        experiment_id: str,
        device: str = "cuda",
    ) -> object:
        """Run a complete fine-tuning experiment from its config.

        Loads experiment config from experiments/{id}/config.json,
        prepares data, trains, and saves results.

        Args:
            experiment_id: Experiment ID (e.g., "V003")
            device: 'cuda' or 'cpu'

        Returns:
            Fine-tuned Chronos2Pipeline
        """
        # Load experiment config
        exp_dir = config.EXPERIMENTS_DIR / experiment_id
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")

        with open(config_path) as f:
            exp_config = json.load(f)

        logger.info(f"Running experiment {experiment_id}: {exp_config.get('model', {}).get('type', 'unknown')}")

        model_config = exp_config.get("model", {})
        training_config = exp_config.get("training", {})
        data_config = exp_config.get("training_data", {})

        # Check if this is a fine-tuning experiment
        if not training_config.get("fine_tune", False):
            logger.info(f"Experiment {experiment_id} is zero-shot (no fine-tuning)")
            return None

        # Prepare training data
        countries = exp_config.get("countries", ["all"])
        forecast_types = exp_config.get("forecast_types", ["all"])
        include_neighbors = "neighbor" in exp_config.get("covariates", {}).get("suffix_1", [])

        train_inputs, val_inputs, labels = self.prepare_training_data(
            countries=countries,
            forecast_types=forecast_types,
            start_date=data_config.get("start", "2023-01-01"),
            end_date=data_config.get("end", "2026-03-01"),
            exclude_backtest=data_config.get("backtest_exclusion", True),
            include_neighbors=include_neighbors,
            val_fraction=training_config.get("val_fraction", 0.0),
        )

        if not train_inputs:
            raise ValueError(f"No valid training data for experiment {experiment_id}")

        # Train
        output_dir = config.MODELS_DIR / "chronos2" / experiment_id / "finetuned-ckpt"

        pipeline = self.train(
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            output_dir=str(output_dir),
            model_name=model_config.get("base", config.CHRONOS2_MODEL_NAME),
            num_steps=training_config.get("fine_tune_steps", 5000),
            learning_rate=training_config.get("learning_rate", 1e-5),
            batch_size=training_config.get("batch_size", 32),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            device=device,
        )

        # Save training metadata
        metadata = {
            "experiment_id": experiment_id,
            "trained_at": datetime.now().isoformat(),
            "num_series": len(train_inputs),
            "num_val_series": len(val_inputs) if val_inputs else 0,
            "series_labels": [f"{cc}/{ft}" for cc, ft in labels],
            "training_config": training_config,
            "model_path": str(output_dir),
        }
        metadata_path = exp_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved to {metadata_path}")

        # Update registry status
        _update_registry_status(experiment_id, "completed")

        return pipeline


def _update_registry_status(experiment_id: str, status: str):
    """Update experiment status in registry.json."""
    registry_path = config.EXPERIMENTS_DIR / "registry.json"
    if not registry_path.exists():
        return

    with open(registry_path) as f:
        registry = json.load(f)

    for exp in registry.get("experiments", []):
        if exp["id"] == experiment_id:
            exp["status"] = status
            break

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
