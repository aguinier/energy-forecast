#!/usr/bin/env python3
"""
Export model metadata to JSON for the web dashboard
"""

import os
import sys
import json
import joblib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import MAPE_THRESHOLDS, DEFAULT_MAPE_THRESHOLD, mape


def recalculate_mape_if_needed(metrics: dict, forecast_type: str) -> dict:
    """
    Check if MAPE value is astronomical and needs note.
    We can't recalculate without the original data, but we can flag it.
    """
    if not metrics:
        return metrics

    current_mape = metrics.get("mape", 0)

    # If MAPE is absurdly high (> 1000%), it's likely due to near-zero values
    # Mark it as unreliable and suggest using SMAPE instead
    if current_mape > 1000:
        metrics = metrics.copy()
        metrics["mape_note"] = "Unreliable due to near-zero values. Use SMAPE instead."
        metrics["mape_reliable"] = False
    else:
        metrics = metrics.copy()
        metrics["mape_reliable"] = True

    return metrics


def export_model_metadata():
    """Extract metadata from all trained models and export to JSON"""
    models_dir = Path(__file__).parent.parent / "models"
    output_file = models_dir / "metadata.json"

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    all_metadata = []

    # Iterate through country directories
    for country_dir in models_dir.iterdir():
        if not country_dir.is_dir():
            continue

        country_code = country_dir.name

        # Iterate through forecast type directories
        for type_dir in country_dir.iterdir():
            if not type_dir.is_dir():
                continue

            model_file = type_dir / "model.joblib"
            if not model_file.exists():
                continue

            try:
                # Load the model metadata
                model_data = joblib.load(model_file)

                # Extract metadata
                forecast_type = model_data.get("forecast_type", type_dir.name)
                raw_metrics = model_data.get("training_metrics", {})

                # Flag unreliable MAPE values from old models
                training_metrics = recalculate_mape_if_needed(
                    raw_metrics, forecast_type
                )

                metadata = {
                    "country_code": model_data.get("country_code", country_code),
                    "forecast_type": forecast_type,
                    "algorithm": model_data.get("algorithm", "xgboost"),
                    "model_version": model_data.get("model_version", "unknown"),
                    "training_metrics": training_metrics,
                    "hyperparams": model_data.get("hyperparams", {}),
                    "grid_search_results": model_data.get("grid_search_results", None),
                    "saved_at": model_data.get("saved_at", ""),
                    "feature_count": len(model_data.get("feature_columns", [])),
                    "feature_columns": model_data.get("feature_columns", []),
                }

                all_metadata.append(metadata)
                print(f"[OK] Exported metadata for {country_code}/{type_dir.name}")

            except Exception as e:
                print(f"[ERROR] Failed to read {country_code}/{type_dir.name}: {e}")

    # Write metadata to JSON file
    with open(output_file, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(
        f"\n[SUCCESS] Exported metadata for {len(all_metadata)} models to {output_file}"
    )
    return all_metadata


if __name__ == "__main__":
    export_model_metadata()
