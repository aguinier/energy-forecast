#!/usr/bin/env python3
"""Train (fine-tune) Chronos-2 models for energy forecasting.

Usage:
    # Run experiment V003 (fine-tuned on all countries/types)
    python scripts/train_chronos2.py --experiment V003 --device cuda

    # Quick test: small subset, few steps
    python scripts/train_chronos2.py --experiment V003 --countries DE --types load --steps 100 --device cuda

    # Zero-shot (no training needed, just validate config)
    python scripts/train_chronos2.py --experiment V002 --device cuda
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.chronos2.finetuner import ChronosFinetuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("energy_forecast.chronos2")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 for energy forecasting")
    parser.add_argument("--experiment", required=True, help="Experiment ID (e.g., V003)")
    parser.add_argument("--countries", default=None, help="Comma-separated country codes, or 'all'")
    parser.add_argument("--types", default=None, help="Comma-separated forecast types, or 'all'")
    parser.add_argument("--steps", type=int, default=None, help="Override num fine-tune steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--start-date", default=None, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--context-length", type=int, default=None, help="Override context length")
    parser.add_argument("--no-backtest-exclusion", action="store_true", help="Don't exclude backtest weeks")
    args = parser.parse_args()

    # Load experiment config
    exp_dir = config.EXPERIMENTS_DIR / args.experiment
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        logger.error(f"Experiment config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        exp_config = json.load(f)

    logger.info(f"=== Experiment {args.experiment}: {exp_config.get('model', {}).get('type', 'unknown')} ===")

    model_config = exp_config.get("model", {})
    training_config = exp_config.get("training", {})
    data_config = exp_config.get("training_data", {})

    # Check if this is a fine-tuning experiment
    if not training_config.get("fine_tune", False):
        logger.info(f"Experiment {args.experiment} is zero-shot — no training needed.")
        logger.info("Model will use pretrained weights directly at inference time.")
        # Update registry
        from src.chronos2.finetuner import _update_registry_status
        _update_registry_status(args.experiment, "ready")
        sys.exit(0)

    # Apply CLI overrides
    countries = args.countries.split(",") if args.countries else exp_config.get("countries", ["all"])
    forecast_types = args.types.split(",") if args.types else exp_config.get("forecast_types", ["all"])
    num_steps = args.steps if args.steps is not None else training_config.get("fine_tune_steps", 5000)
    learning_rate = args.lr if args.lr is not None else training_config.get("learning_rate", 1e-5)
    batch_size = args.batch_size if args.batch_size is not None else training_config.get("batch_size", 32)
    start_date = args.start_date or data_config.get("start", "2023-01-01")
    end_date = args.end_date or data_config.get("end", "2026-03-01")
    context_length = args.context_length or model_config.get("context_length")
    exclude_backtest = not args.no_backtest_exclusion and data_config.get("backtest_exclusion", True)

    logger.info(f"Countries: {countries}")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Steps: {num_steps}, LR: {learning_rate}, Batch: {batch_size}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Backtest exclusion: {exclude_backtest}")

    # Initialize finetuner
    finetuner = ChronosFinetuner(context_length=context_length)

    # Prepare data
    train_inputs, val_inputs, labels = finetuner.prepare_training_data(
        countries=countries,
        forecast_types=forecast_types,
        start_date=start_date,
        end_date=end_date,
        exclude_backtest=exclude_backtest,
        include_neighbors="neighbor" in json.dumps(exp_config.get("covariates", {})),
        val_fraction=training_config.get("val_fraction", 0.0),
    )

    if not train_inputs:
        logger.error("No valid training data found!")
        sys.exit(1)

    logger.info(f"Training data ready: {len(train_inputs)} series")
    if val_inputs:
        logger.info(f"Validation data: {len(val_inputs)} series")

    # Train
    output_dir = config.MODELS_DIR / "chronos2" / args.experiment / "finetuned-ckpt"

    pipeline = finetuner.train(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        output_dir=str(output_dir),
        model_name=model_config.get("base", config.CHRONOS2_MODEL_NAME),
        num_steps=num_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        device=args.device,
    )

    # Save experiment metadata
    metadata = {
        "experiment_id": args.experiment,
        "trained_at": datetime.now().isoformat(),
        "num_series": len(train_inputs),
        "num_val_series": len(val_inputs) if val_inputs else 0,
        "series_labels": [f"{cc}/{ft}" for cc, ft in labels],
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "model_path": str(output_dir),
    }

    metadata_path = exp_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    from src.chronos2.finetuner import _update_registry_status
    _update_registry_status(args.experiment, "completed")

    logger.info(f"=== Training complete for {args.experiment} ===")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
