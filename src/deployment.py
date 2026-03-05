"""
Deployment Management for Energy Forecasting

Handles model promotion, rollback, and auto-deployment based on evaluation metrics.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from model_registry import get_registry, ModelRegistry
from db import (
    get_latest_evaluation,
    get_deployed_model,
    save_deployment,
    rollback_deployment as db_rollback,
    get_all_deployed_models,
    get_deployment_history,
)


logger = logging.getLogger('energy_forecast')


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""
    promoted: bool
    reason: str
    candidate_version: Optional[str] = None
    production_version: Optional[str] = None
    candidate_skill: Optional[float] = None
    production_skill: Optional[float] = None


def get_deployment_status(
    country_code: Optional[str] = None,
    forecast_type: Optional[str] = None,
) -> List[Dict]:
    """
    Get current deployment status.

    Args:
        country_code: Filter by country (optional)
        forecast_type: Filter by type (optional)

    Returns:
        List of deployment status dictionaries
    """
    df = get_all_deployed_models()

    if country_code:
        df = df[df['country_code'] == country_code]

    if forecast_type:
        df = df[df['forecast_type'] == forecast_type]

    return df.to_dict('records')


def compare_candidate_vs_production(
    country_code: str,
    forecast_type: str,
) -> Dict:
    """
    Compare candidate model evaluation against production.

    Returns:
        Dictionary with comparison metrics
    """
    candidate_eval = get_latest_evaluation(country_code, forecast_type, "candidate")
    production_eval = get_latest_evaluation(country_code, forecast_type, "production")

    result = {
        "country_code": country_code,
        "forecast_type": forecast_type,
        "candidate": None,
        "production": None,
        "comparison": {},
    }

    if candidate_eval:
        result["candidate"] = {
            "model_version": candidate_eval.get("model_version"),
            "mae": candidate_eval.get("mae"),
            "mape": candidate_eval.get("mape"),
            "skill_vs_persistence": candidate_eval.get("skill_vs_persistence"),
            "skill_vs_seasonal": candidate_eval.get("skill_vs_seasonal"),
        }

    if production_eval:
        result["production"] = {
            "model_version": production_eval.get("model_version"),
            "mae": production_eval.get("mae"),
            "mape": production_eval.get("mape"),
            "skill_vs_persistence": production_eval.get("skill_vs_persistence"),
            "skill_vs_seasonal": production_eval.get("skill_vs_seasonal"),
        }

    # Compute comparison if both exist
    if candidate_eval and production_eval:
        cand_skill = candidate_eval.get("skill_vs_persistence") or 0
        prod_skill = production_eval.get("skill_vs_persistence") or 0

        cand_mae = candidate_eval.get("mae") or float('inf')
        prod_mae = production_eval.get("mae") or float('inf')

        result["comparison"] = {
            "skill_improvement": cand_skill - prod_skill,
            "mae_improvement": prod_mae - cand_mae,  # Positive = candidate is better
            "candidate_better": cand_skill > prod_skill,
        }

    return result


def auto_promote_if_better(
    country_code: str,
    forecast_type: str,
    min_skill_threshold: float = 0.0,
    min_skill_improvement: float = 0.0,
) -> PromotionResult:
    """
    Automatically promote candidate model if it beats production.

    Promotion criteria:
    1. Candidate must beat baselines (skill_vs_persistence > min_skill_threshold)
    2. Candidate must improve over production by min_skill_improvement

    Args:
        country_code: Country code
        forecast_type: Type of forecast
        min_skill_threshold: Minimum skill score vs persistence to be deployment-worthy
        min_skill_improvement: Minimum improvement over production required

    Returns:
        PromotionResult with details
    """
    registry = get_registry()

    # Check candidate exists
    if not registry.model_exists(country_code, forecast_type, "candidate"):
        return PromotionResult(
            promoted=False,
            reason="No candidate model exists",
        )

    # Get candidate evaluation
    candidate_eval = get_latest_evaluation(country_code, forecast_type, "candidate")

    if candidate_eval is None:
        return PromotionResult(
            promoted=False,
            reason="No evaluation found for candidate model",
        )

    candidate_skill = candidate_eval.get("skill_vs_persistence") or 0
    candidate_version = candidate_eval.get("model_version")

    # Check if candidate beats baselines
    if candidate_skill <= min_skill_threshold:
        return PromotionResult(
            promoted=False,
            reason=f"Candidate does not beat persistence baseline (skill={candidate_skill:.4f} <= {min_skill_threshold})",
            candidate_version=candidate_version,
            candidate_skill=candidate_skill,
        )

    # Get production evaluation (if exists)
    production_eval = get_latest_evaluation(country_code, forecast_type, "production")
    deployed = get_deployed_model(country_code, forecast_type)

    # If no production model, promote candidate
    if deployed is None or production_eval is None:
        # Perform promotion
        success = _execute_promotion(
            country_code, forecast_type, candidate_eval,
            deployment_reason="First deployment (no previous production model)",
        )

        return PromotionResult(
            promoted=success,
            reason="First deployment (no previous production model)" if success else "Promotion failed",
            candidate_version=candidate_version,
            candidate_skill=candidate_skill,
        )

    production_skill = production_eval.get("skill_vs_persistence") or 0
    production_version = production_eval.get("model_version")

    # Compare skills
    skill_improvement = candidate_skill - production_skill

    if skill_improvement > min_skill_improvement:
        # Perform promotion
        success = _execute_promotion(
            country_code, forecast_type, candidate_eval,
            deployment_reason=f"Improved skill: {candidate_skill:.4f} > {production_skill:.4f}",
        )

        return PromotionResult(
            promoted=success,
            reason=f"Improved skill: {candidate_skill:.4f} > {production_skill:.4f}" if success else "Promotion failed",
            candidate_version=candidate_version,
            production_version=production_version,
            candidate_skill=candidate_skill,
            production_skill=production_skill,
        )

    return PromotionResult(
        promoted=False,
        reason=f"Candidate does not improve over production (improvement={skill_improvement:.4f} <= {min_skill_improvement})",
        candidate_version=candidate_version,
        production_version=production_version,
        candidate_skill=candidate_skill,
        production_skill=production_skill,
    )


def _execute_promotion(
    country_code: str,
    forecast_type: str,
    candidate_eval: Dict,
    deployment_reason: str,
) -> bool:
    """
    Execute the actual promotion (registry + database).

    Returns:
        True if successful
    """
    registry = get_registry()

    try:
        # Update registry (move candidate to production, archive old production)
        if not registry.promote_to_production(country_code, forecast_type):
            logger.error(f"Registry promotion failed for {country_code}/{forecast_type}")
            return False

        # Record deployment in database
        save_deployment(
            country_code=country_code,
            forecast_type=forecast_type,
            model_version=candidate_eval.get("model_version", "unknown"),
            deployed_by="auto_promote",
            deployment_reason=deployment_reason,
            mae_at_deployment=candidate_eval.get("mae"),
            mape_at_deployment=candidate_eval.get("mape"),
            skill_score_at_deployment=candidate_eval.get("skill_vs_persistence"),
        )

        logger.info(f"Promoted {country_code}/{forecast_type} to production")
        return True

    except Exception as e:
        logger.error(f"Promotion failed for {country_code}/{forecast_type}: {e}")
        return False


def promote_to_production(
    country_code: str,
    forecast_type: str,
    deployed_by: str = "manual",
) -> PromotionResult:
    """
    Manually promote candidate to production (skip quality checks).

    Use this for manual promotions that bypass auto-promotion criteria.
    """
    registry = get_registry()

    if not registry.model_exists(country_code, forecast_type, "candidate"):
        return PromotionResult(
            promoted=False,
            reason="No candidate model exists",
        )

    candidate_eval = get_latest_evaluation(country_code, forecast_type, "candidate")
    candidate_version = candidate_eval.get("model_version") if candidate_eval else "unknown"

    success = _execute_promotion(
        country_code, forecast_type,
        candidate_eval or {"model_version": candidate_version},
        deployment_reason=f"Manual promotion by {deployed_by}",
    )

    return PromotionResult(
        promoted=success,
        reason="Manual promotion" if success else "Promotion failed",
        candidate_version=candidate_version,
        candidate_skill=candidate_eval.get("skill_vs_persistence") if candidate_eval else None,
    )


def rollback_deployment(
    country_code: str,
    forecast_type: str,
    target_version: Optional[str] = None,
    rollback_by: str = "manual",
) -> Tuple[bool, str]:
    """
    Rollback deployment to a previous version.

    Args:
        country_code: Country code
        forecast_type: Type of forecast
        target_version: Specific version to rollback to (if None, uses previous)
        rollback_by: Who initiated the rollback

    Returns:
        Tuple of (success, message)
    """
    registry = get_registry()

    if target_version:
        # Rollback to specific version
        if not registry.rollback_to_version(country_code, forecast_type, target_version):
            return False, f"Version {target_version} not found in history"

        # Update database
        save_deployment(
            country_code=country_code,
            forecast_type=forecast_type,
            model_version=target_version,
            deployed_by=rollback_by,
            deployment_reason=f"Rollback to {target_version}",
        )

        return True, f"Rolled back to {target_version}"

    else:
        # Rollback to previous version (using database tracking)
        success = db_rollback(country_code, forecast_type, rollback_by)

        if success:
            return True, "Rolled back to previous version"
        else:
            return False, "No previous version to rollback to"


def batch_auto_promote(
    countries: Optional[List[str]] = None,
    forecast_types: Optional[List[str]] = None,
    min_skill_threshold: float = 0.0,
) -> List[PromotionResult]:
    """
    Auto-promote all eligible candidate models.

    Args:
        countries: List of countries to check (default: all)
        forecast_types: List of types to check (default: all)
        min_skill_threshold: Minimum skill score required

    Returns:
        List of PromotionResults
    """
    if countries is None:
        countries = config.SUPPORTED_COUNTRIES

    if forecast_types is None:
        forecast_types = ["load", "price", "solar", "wind_onshore", "wind_offshore"]

    results = []

    for country in countries:
        for ftype in forecast_types:
            result = auto_promote_if_better(
                country, ftype,
                min_skill_threshold=min_skill_threshold,
            )
            results.append(result)

            if result.promoted:
                logger.info(
                    f"[PROMOTED] {country}/{ftype}: {result.reason}"
                )
            else:
                logger.debug(
                    f"[SKIP] {country}/{ftype}: {result.reason}"
                )

    promoted_count = sum(1 for r in results if r.promoted)
    logger.info(f"Batch promotion complete: {promoted_count}/{len(results)} promoted")

    return results


def get_promotion_candidates() -> List[Dict]:
    """
    Get list of candidate models that are eligible for promotion.

    Returns candidates that:
    1. Have a candidate model
    2. Have evaluation metrics
    3. Beat baselines (skill > 0)
    """
    registry = get_registry()
    candidates = []

    for model in registry.list_all_models():
        if "candidate" not in model["locations"]:
            continue

        country = model["country_code"]
        ftype = model["forecast_type"]

        candidate_eval = get_latest_evaluation(country, ftype, "candidate")
        if not candidate_eval:
            continue

        skill = candidate_eval.get("skill_vs_persistence") or 0
        if skill <= 0:
            continue

        # Get production info for comparison
        production_eval = get_latest_evaluation(country, ftype, "production")
        prod_skill = production_eval.get("skill_vs_persistence") if production_eval else None

        candidates.append({
            "country_code": country,
            "forecast_type": ftype,
            "candidate_version": candidate_eval.get("model_version"),
            "candidate_skill": skill,
            "candidate_mae": candidate_eval.get("mae"),
            "production_skill": prod_skill,
            "improvement": skill - prod_skill if prod_skill else None,
        })

    # Sort by improvement (best candidates first)
    candidates.sort(
        key=lambda x: x.get("improvement") or x.get("candidate_skill") or 0,
        reverse=True,
    )

    return candidates


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("Testing Deployment Management...")

    # Get current deployment status
    print("\n1. Current deployment status...")
    status = get_deployment_status()
    print(f"   {len(status)} models deployed")
    for s in status[:5]:
        print(f"   - {s['country_code']}/{s['forecast_type']}: {s['model_version']}")

    # Get promotion candidates
    print("\n2. Promotion candidates...")
    candidates = get_promotion_candidates()
    print(f"   {len(candidates)} candidates eligible")
    for c in candidates[:5]:
        print(f"   - {c['country_code']}/{c['forecast_type']}: "
              f"skill={c['candidate_skill']:.4f}, improvement={c.get('improvement', 'N/A')}")

    # Test comparison
    print("\n3. Testing candidate vs production comparison...")
    if candidates:
        comparison = compare_candidate_vs_production(
            candidates[0]['country_code'],
            candidates[0]['forecast_type'],
        )
        print(f"   Candidate: {comparison.get('candidate')}")
        print(f"   Production: {comparison.get('production')}")
        print(f"   Comparison: {comparison.get('comparison')}")

    print("\n[OK] Deployment tests complete!")
