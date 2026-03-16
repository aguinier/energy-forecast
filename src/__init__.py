"""
Energy Forecasting Module

D+2 daily forecasting for load, price, and renewable generation.
"""

__version__ = "0.2.0"

# Core metrics
from .metrics import (
    mae,
    rmse,
    mape,
    smape,
    mase,
    directional_accuracy,
    peak_hour_accuracy,
    skill_score,
    calculate_all_metrics,
)

# Baseline models
from .baselines import (
    PersistenceBaseline,
    SeasonalNaiveBaseline,
    WeeklyAverageBaseline,
    TSOBaseline,
    compute_baseline_metrics,
    get_all_baseline_predictions,
)

# Validation
from .validation import (
    WalkForwardValidator,
    TimeSeriesValidator,
    ValidationSplit,
    ValidationResult,
    WalkForwardResults,
    create_validation_summary,
    format_validation_report,
)

# Evaluation
# from .evaluation import (
#     EvaluationReport,
#     generate_evaluation_report,
#     format_evaluation_report,
#     compare_models,
# )

# Model Registry
from .model_registry import (
    ModelRegistry,
    get_registry,
)

# Deployment
from .deployment import (
    PromotionResult,
    get_deployment_status,
    compare_candidate_vs_production,
    auto_promote_if_better,
    promote_to_production,
    rollback_deployment,
    batch_auto_promote,
    get_promotion_candidates,
)
