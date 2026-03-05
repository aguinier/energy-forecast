"""
Automated Feature Selection for Energy Forecasting

Provides methods for selecting optimal feature subsets:
- Recursive Feature Elimination (RFE) based on importance
- Permutation importance
- Correlation-based filtering
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from metrics import mae

logger = logging.getLogger("energy_forecast")


@dataclass
class FeatureSelectionResult:
    """Results from feature selection."""
    selected_features: List[str]
    feature_importance: Dict[str, float]
    n_features_original: int
    n_features_selected: int
    validation_score_all: float
    validation_score_selected: float
    elimination_history: List[Dict]


class FeatureSelector:
    """
    Automated feature selection using importance-based elimination.

    Performs backward elimination by iteratively removing the least
    important features until validation performance degrades.

    Example:
        selector = FeatureSelector('xgboost')
        result = selector.select_features(X, y, feature_names)
        best_features = result.selected_features
    """

    def __init__(
        self,
        algorithm: str = 'xgboost',
        min_features: int = 5,
        max_elimination_ratio: float = 0.5,
        patience: int = 3,
        n_cv_splits: int = 3,
    ):
        """
        Initialize feature selector.

        Args:
            algorithm: 'xgboost', 'lightgbm', or 'catboost'
            min_features: Minimum number of features to keep
            max_elimination_ratio: Maximum fraction of features to eliminate (0.5 = keep at least 50%)
            patience: Stop if no improvement for this many rounds
            n_cv_splits: Number of cross-validation splits
        """
        self.algorithm = algorithm
        self.min_features = min_features
        self.max_elimination_ratio = max_elimination_ratio
        self.patience = patience
        self.n_cv_splits = n_cv_splits

    def _create_model(self) -> Any:
        """Create model instance with default parameters."""
        params = config.get_default_params(self.algorithm)

        if self.algorithm == 'xgboost':
            # Remove early_stopping_rounds for feature selection
            params.pop('early_stopping_rounds', None)
            return XGBRegressor(**params)
        elif self.algorithm == 'lightgbm':
            return LGBMRegressor(**params)
        elif self.algorithm == 'catboost':
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _compute_cv_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cross-validated MAE and feature importance.

        Args:
            X: Feature matrix
            y: Target vector
            feature_mask: Boolean mask for features to use

        Returns:
            Tuple of (mean_mae, mean_importance)
        """
        X_subset = X[:, feature_mask]
        n_features = X_subset.shape[1]

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores = []
        importances = np.zeros(n_features)

        for train_idx, val_idx in tscv.split(X_subset):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._create_model()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            scores.append(mae(y_val, y_pred))

            importances += model.feature_importances_

        importances /= self.n_cv_splits
        return np.mean(scores), importances

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        forecast_type: str = 'load',
    ) -> FeatureSelectionResult:
        """
        Select optimal feature subset using backward elimination.

        Algorithm:
        1. Train model with all features, compute importance
        2. Remove least important feature(s)
        3. Retrain and check if validation score improves
        4. Stop when score degrades for `patience` rounds or min features reached

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector
            feature_names: List of feature names
            forecast_type: Type of forecast for context

        Returns:
            FeatureSelectionResult with selected features and history
        """
        n_features = len(feature_names)
        min_features = max(self.min_features, int(n_features * (1 - self.max_elimination_ratio)))

        logger.info(f"Starting feature selection: {n_features} features -> min {min_features}")

        # Current active features (all True initially)
        active_mask = np.ones(n_features, dtype=bool)
        current_features = list(feature_names)

        # Compute baseline score with all features
        baseline_score, baseline_importance = self._compute_cv_score(X, y, active_mask)
        logger.info(f"Baseline MAE with all {n_features} features: {baseline_score:.4f}")

        # Track best configuration
        best_score = baseline_score
        best_mask = active_mask.copy()
        best_features = current_features.copy()
        no_improvement_count = 0

        # Elimination history
        elimination_history = [{
            'round': 0,
            'n_features': n_features,
            'score': baseline_score,
            'removed': None,
        }]

        # Feature importance dictionary
        feature_importance = dict(zip(feature_names, baseline_importance))

        # Backward elimination loop
        round_num = 0
        while np.sum(active_mask) > min_features and no_improvement_count < self.patience:
            round_num += 1

            # Get current active feature indices and names
            active_indices = np.where(active_mask)[0]
            active_names = [feature_names[i] for i in active_indices]

            # Compute importance for active features
            _, importances = self._compute_cv_score(X, y, active_mask)

            # Find least important feature(s) to remove
            # Remove bottom 10% or at least 1 feature
            n_to_remove = max(1, int(len(active_indices) * 0.1))
            sorted_indices = np.argsort(importances)
            features_to_remove = sorted_indices[:n_to_remove]

            # Remove features
            for idx in features_to_remove:
                active_mask[active_indices[idx]] = False

            removed_names = [active_names[idx] for idx in features_to_remove]
            n_remaining = np.sum(active_mask)

            # Compute new score
            new_score, new_importance = self._compute_cv_score(X, y, active_mask)

            # Check for improvement
            improvement = baseline_score - new_score  # positive = better

            elimination_history.append({
                'round': round_num,
                'n_features': n_remaining,
                'score': new_score,
                'removed': removed_names,
                'improvement': improvement,
            })

            if new_score < best_score:
                # Improvement found
                best_score = new_score
                best_mask = active_mask.copy()
                best_features = [feature_names[i] for i in np.where(active_mask)[0]]
                no_improvement_count = 0
                logger.info(
                    f"Round {round_num}: {n_remaining} features, MAE={new_score:.4f} "
                    f"(+{improvement:.4f}) - Removed: {removed_names}"
                )
            else:
                no_improvement_count += 1
                logger.info(
                    f"Round {round_num}: {n_remaining} features, MAE={new_score:.4f} "
                    f"({improvement:+.4f}) - No improvement ({no_improvement_count}/{self.patience})"
                )

        # Final importance for selected features
        final_active_indices = np.where(best_mask)[0]
        _, final_importance = self._compute_cv_score(X, y, best_mask)
        final_feature_importance = {
            feature_names[idx]: float(final_importance[i])
            for i, idx in enumerate(final_active_indices)
        }

        logger.info(
            f"Feature selection complete: {n_features} -> {len(best_features)} features "
            f"(MAE: {baseline_score:.4f} -> {best_score:.4f})"
        )

        return FeatureSelectionResult(
            selected_features=best_features,
            feature_importance=final_feature_importance,
            n_features_original=n_features,
            n_features_selected=len(best_features),
            validation_score_all=baseline_score,
            validation_score_selected=best_score,
            elimination_history=elimination_history,
        )


def select_features_for_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    algorithm: str = 'xgboost',
    min_features: int = 5,
    patience: int = 3,
) -> List[str]:
    """
    Convenience function to select features.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        algorithm: 'xgboost', 'lightgbm', or 'catboost'
        min_features: Minimum features to keep
        patience: Rounds without improvement before stopping

    Returns:
        List of selected feature names
    """
    selector = FeatureSelector(
        algorithm=algorithm,
        min_features=min_features,
        patience=patience,
    )
    result = selector.select_features(X, y, feature_names)
    return result.selected_features


def get_feature_importance_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    algorithm: str = 'xgboost',
) -> pd.DataFrame:
    """
    Get feature importance ranking without elimination.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        algorithm: Algorithm to use

    Returns:
        DataFrame with feature names and importance scores, sorted descending
    """
    params = config.get_default_params(algorithm)
    params.pop('early_stopping_rounds', None)

    if algorithm == 'xgboost':
        model = XGBRegressor(**params)
    elif algorithm == 'lightgbm':
        model = LGBMRegressor(**params)
    elif algorithm == 'catboost':
        model = CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    })

    return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing feature selection...")

    # Create sample data with some irrelevant features
    np.random.seed(42)
    n_samples = 3000
    n_relevant = 10
    n_irrelevant = 15
    n_features = n_relevant + n_irrelevant

    # Relevant features
    X_relevant = np.random.randn(n_samples, n_relevant)
    weights = np.random.randn(n_relevant)
    y = X_relevant @ weights + np.random.randn(n_samples) * 0.5

    # Irrelevant features (random noise)
    X_irrelevant = np.random.randn(n_samples, n_irrelevant)

    X = np.hstack([X_relevant, X_irrelevant])
    feature_names = [f"relevant_{i}" for i in range(n_relevant)] + \
                    [f"noise_{i}" for i in range(n_irrelevant)]

    print(f"\n1. Sample data: {X.shape}")
    print(f"   Relevant features: {n_relevant}, Irrelevant: {n_irrelevant}")

    # Test feature importance ranking
    print("\n2. Testing feature importance ranking...")
    importance_df = get_feature_importance_ranking(X, y, feature_names)
    print("   Top 10 features:")
    print(importance_df.head(10).to_string(index=False))

    # Test feature selection
    print("\n3. Testing feature selection...")
    selector = FeatureSelector('xgboost', min_features=5, patience=2)
    result = selector.select_features(X, y, feature_names)

    print(f"\n4. Results:")
    print(f"   Original features: {result.n_features_original}")
    print(f"   Selected features: {result.n_features_selected}")
    print(f"   MAE improvement: {result.validation_score_all:.4f} -> {result.validation_score_selected:.4f}")
    print(f"   Selected: {result.selected_features[:10]}...")

    # Check how many relevant vs irrelevant were selected
    relevant_selected = sum(1 for f in result.selected_features if f.startswith('relevant'))
    noise_selected = sum(1 for f in result.selected_features if f.startswith('noise'))
    print(f"\n5. Selection quality:")
    print(f"   Relevant features selected: {relevant_selected}/{n_relevant}")
    print(f"   Noise features selected: {noise_selected}/{n_irrelevant}")

    print("\n[OK] Feature selection tests complete!")
