"""
Model Registry for Energy Forecasting

Manages model versioning with production/candidate/history structure.
Handles model storage, retrieval, and version cleanup.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


logger = logging.getLogger('energy_forecast')

# Maximum number of historical versions to keep per country/type
MAX_HISTORY_VERSIONS = 5


class ModelRegistry:
    """
    Model Registry for managing model versions.

    Directory structure:
    models/
    ├── {country}/
    │   └── {type}/
    │       ├── production/model.joblib    # Currently deployed
    │       ├── candidate/model.joblib     # Latest trained
    │       └── history/
    │           ├── v20250115_180000.joblib
    │           └── v20250108_180000.joblib
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Args:
            models_dir: Root directory for models (default: config.MODELS_DIR)
        """
        self.models_dir = models_dir or config.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(
        self,
        country_code: str,
        forecast_type: str,
        location: str = "production",
        version: Optional[str] = None,
    ) -> Path:
        """
        Get path to a model file.

        Args:
            country_code: Country code
            forecast_type: Type of forecast
            location: 'production', 'candidate', or 'history'
            version: For history, the version string (e.g., 'v20250115_180000')

        Returns:
            Path to model file
        """
        base = self.models_dir / country_code / forecast_type

        if location == "history" and version:
            return base / "history" / f"{version}.joblib"
        elif location in ("production", "candidate"):
            return base / location / "model.joblib"
        else:
            # Legacy path (direct model.joblib)
            return base / "model.joblib"

    def model_exists(
        self,
        country_code: str,
        forecast_type: str,
        location: str = "production",
    ) -> bool:
        """Check if a model exists at the given location."""
        path = self.get_model_path(country_code, forecast_type, location)
        return path.exists()

    def load_model(
        self,
        country_code: str,
        forecast_type: str,
        location: str = "production",
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a model from the registry.

        Args:
            country_code: Country code
            forecast_type: Type of forecast
            location: 'production', 'candidate', 'history', or 'legacy'
            version: For history, the version string

        Returns:
            Dictionary with model and metadata, or None if not found
        """
        # Try new structure first
        path = self.get_model_path(country_code, forecast_type, location, version)

        if not path.exists():
            # Fall back to legacy path
            legacy_path = self.models_dir / country_code / forecast_type / "model.joblib"
            if legacy_path.exists():
                path = legacy_path
            else:
                return None

        try:
            data = joblib.load(path)
            logger.debug(f"Loaded model from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return None

    def save_model(
        self,
        model_data: Dict[str, Any],
        country_code: str,
        forecast_type: str,
        location: str = "candidate",
    ) -> Path:
        """
        Save a model to the registry.

        Args:
            model_data: Dictionary with 'model' and metadata
            country_code: Country code
            forecast_type: Type of forecast
            location: 'candidate' or 'production' (not 'history')

        Returns:
            Path where model was saved
        """
        if location not in ("candidate", "production"):
            raise ValueError(f"Invalid location for save: {location}")

        path = self.get_model_path(country_code, forecast_type, location)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure model_version is set
        if "model_version" not in model_data:
            model_data["model_version"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

        return path

    def promote_to_production(
        self,
        country_code: str,
        forecast_type: str,
    ) -> bool:
        """
        Promote candidate model to production.

        1. Archive current production to history
        2. Copy candidate to production
        3. Cleanup old history versions

        Returns:
            True if promotion successful
        """
        candidate_path = self.get_model_path(country_code, forecast_type, "candidate")
        production_path = self.get_model_path(country_code, forecast_type, "production")
        history_dir = self.models_dir / country_code / forecast_type / "history"

        if not candidate_path.exists():
            logger.error(f"No candidate model to promote for {country_code}/{forecast_type}")
            return False

        # Archive current production if it exists
        if production_path.exists():
            prod_data = joblib.load(production_path)
            version = prod_data.get("model_version", datetime.now().strftime("%Y%m%d_%H%M%S"))
            archive_path = history_dir / f"v{version}.joblib"
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(production_path, archive_path)
            logger.info(f"Archived production model to {archive_path}")

        # Copy candidate to production
        production_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, production_path)
        logger.info(f"Promoted candidate to production for {country_code}/{forecast_type}")

        # Cleanup old history
        self._cleanup_history(country_code, forecast_type)

        return True

    def rollback_to_version(
        self,
        country_code: str,
        forecast_type: str,
        version: str,
    ) -> bool:
        """
        Rollback production to a specific historical version.

        Args:
            version: Version string (e.g., 'v20250115_180000')

        Returns:
            True if rollback successful
        """
        history_path = self.get_model_path(country_code, forecast_type, "history", version)
        production_path = self.get_model_path(country_code, forecast_type, "production")

        if not history_path.exists():
            logger.error(f"History version {version} not found for {country_code}/{forecast_type}")
            return False

        # Archive current production first
        if production_path.exists():
            prod_data = joblib.load(production_path)
            curr_version = prod_data.get("model_version", "unknown")
            archive_path = self.models_dir / country_code / forecast_type / "history" / f"v{curr_version}.joblib"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(production_path, archive_path)

        # Restore from history
        shutil.copy2(history_path, production_path)
        logger.info(f"Rolled back {country_code}/{forecast_type} to version {version}")

        return True

    def _cleanup_history(
        self,
        country_code: str,
        forecast_type: str,
    ):
        """
        Remove old history versions, keeping only MAX_HISTORY_VERSIONS.
        """
        history_dir = self.models_dir / country_code / forecast_type / "history"

        if not history_dir.exists():
            return

        # List all history files sorted by name (newest first due to timestamp format)
        history_files = sorted(
            history_dir.glob("v*.joblib"),
            key=lambda p: p.name,
            reverse=True,
        )

        # Remove excess versions
        for old_file in history_files[MAX_HISTORY_VERSIONS:]:
            old_file.unlink()
            logger.debug(f"Removed old history: {old_file}")

    def get_history_versions(
        self,
        country_code: str,
        forecast_type: str,
    ) -> List[str]:
        """
        Get list of available history versions.

        Returns:
            List of version strings (e.g., ['v20250115_180000', 'v20250108_180000'])
        """
        history_dir = self.models_dir / country_code / forecast_type / "history"

        if not history_dir.exists():
            return []

        versions = sorted(
            [f.stem for f in history_dir.glob("v*.joblib")],
            reverse=True,
        )

        return versions

    def get_model_info(
        self,
        country_code: str,
        forecast_type: str,
        location: str = "production",
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model without loading the full model.
        """
        data = self.load_model(country_code, forecast_type, location)

        if data is None:
            return None

        # Return metadata only (exclude the actual model object)
        return {
            k: v for k, v in data.items()
            if k != "model"
        }

    def migrate_legacy_model(
        self,
        country_code: str,
        forecast_type: str,
    ) -> bool:
        """
        Migrate a legacy model (direct model.joblib) to the new structure.

        Legacy models are moved to production/ location.

        Returns:
            True if migration successful or already migrated
        """
        legacy_path = self.models_dir / country_code / forecast_type / "model.joblib"
        production_path = self.get_model_path(country_code, forecast_type, "production")

        # Already migrated
        if production_path.exists():
            return True

        # No legacy model
        if not legacy_path.exists():
            return False

        # Migrate
        production_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(production_path))
        logger.info(f"Migrated legacy model to {production_path}")

        return True

    def list_all_models(self) -> List[Dict[str, str]]:
        """
        List all models in the registry.

        Returns:
            List of dicts with country_code, forecast_type, locations
        """
        models = []

        for country_dir in self.models_dir.iterdir():
            if not country_dir.is_dir():
                continue

            country_code = country_dir.name

            for type_dir in country_dir.iterdir():
                if not type_dir.is_dir():
                    continue

                forecast_type = type_dir.name
                locations = []

                # Check each location
                for loc in ("production", "candidate"):
                    if self.model_exists(country_code, forecast_type, loc):
                        locations.append(loc)

                # Check legacy
                if (type_dir / "model.joblib").exists() and "production" not in locations:
                    locations.append("legacy")

                # Check history
                history_versions = self.get_history_versions(country_code, forecast_type)

                if locations or history_versions:
                    models.append({
                        "country_code": country_code,
                        "forecast_type": forecast_type,
                        "locations": locations,
                        "history_count": len(history_versions),
                    })

        return models

    def migrate_all_legacy_models(self) -> int:
        """
        Migrate all legacy models to the new structure.

        Returns:
            Number of models migrated
        """
        migrated = 0

        for model_info in self.list_all_models():
            if "legacy" in model_info["locations"]:
                if self.migrate_legacy_model(
                    model_info["country_code"],
                    model_info["forecast_type"],
                ):
                    migrated += 1

        logger.info(f"Migrated {migrated} legacy models")
        return migrated


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing Model Registry...")

    registry = ModelRegistry()

    # List existing models
    print("\n1. Listing all models...")
    models = registry.list_all_models()
    print(f"   Found {len(models)} model types")
    for m in models[:5]:
        print(f"   - {m['country_code']}/{m['forecast_type']}: {m['locations']}")

    # Test migration
    print("\n2. Testing legacy migration...")
    migrated = registry.migrate_all_legacy_models()
    print(f"   Migrated {migrated} models")

    # Test model info
    print("\n3. Testing model info...")
    if models:
        info = registry.get_model_info(
            models[0]['country_code'],
            models[0]['forecast_type'],
            'production',
        )
        if info:
            print(f"   Model version: {info.get('model_version', 'unknown')}")
            print(f"   Features: {len(info.get('feature_columns', []))}")

    # Test history
    print("\n4. Testing history versions...")
    if models:
        versions = registry.get_history_versions(
            models[0]['country_code'],
            models[0]['forecast_type'],
        )
        print(f"   History versions: {versions}")

    print("\n[OK] Model Registry tests complete!")
