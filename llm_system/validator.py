"""Two-tier validation: schema checking and simulation testing."""

import pandas as pd
from pydantic import ValidationError

from llm_system.schemas import Recommendation
from llm_system.metrics import calculate_all
from simulator import AutoScalingSimulator


def validate_schema(recommendation: dict) -> tuple[bool, str]:
    """Validate recommendation against Pydantic schema (Tier 1).

    Args:
        recommendation: Raw recommendation dict from LLM

    Returns:
        Tuple of (valid, message):
        - (True, "valid") if validation passes
        - (False, error_description) if validation fails
    """
    try:
        Recommendation(**recommendation)
        return (True, "valid")
    except ValidationError as e:
        return (False, str(e))


def test_recommendation(
    recommendation: dict,
    baseline_config: dict,
    baseline_metrics: dict,
    simulation_csv_path: str
) -> dict:
    """Test recommendation by re-running simulator (Tier 2).

    Args:
        recommendation: Validated recommendation dict
        baseline_config: Original simulator config
        baseline_metrics: Baseline metrics from calculate_all()
        simulation_csv_path: Path to baseline simulation CSV

    Returns:
        Dict with new_metrics, improvement percentages, accepted flag, and reason
    """
    # 1. Extract workload arrays from baseline CSV
    baseline_df = pd.read_csv(simulation_csv_path)
    actual_load = baseline_df['actual_load'].values
    predicted_load = baseline_df['predicted_load'].values

    # 2. Build modified config
    new_config = baseline_config.copy()
    param = recommendation['parameter']
    new_value = recommendation['proposed_value']

    # Handle arima_order special case (stays as string)
    new_config[param] = new_value

    # 3. Re-run simulator with modified config
    sim = AutoScalingSimulator(actual_load, predicted_load, new_config)
    new_df = sim.run()

    # 4. Calculate new metrics
    new_metrics = calculate_all(new_df, new_config)

    # 5. Calculate improvement percentages
    # SLA improvement: positive means fewer violations
    if baseline_metrics['sla_violations'] > 0:
        sla_improvement_pct = (
            (baseline_metrics['sla_violations'] - new_metrics['sla_violations'])
            / baseline_metrics['sla_violations']
        ) * 100
    else:
        sla_improvement_pct = 0.0

    # Cost change: negative means cost decreased (good)
    if baseline_metrics['total_cost'] > 0:
        cost_change_pct = (
            (new_metrics['total_cost'] - baseline_metrics['total_cost'])
            / baseline_metrics['total_cost']
        ) * 100
    else:
        cost_change_pct = 0.0

    # Scaling events change: negative means fewer events (good)
    if baseline_metrics['scaling_events'] > 0:
        scaling_events_change_pct = (
            (new_metrics['scaling_events'] - baseline_metrics['scaling_events'])
            / baseline_metrics['scaling_events']
        ) * 100
    else:
        scaling_events_change_pct = 0.0

    # 6. Apply acceptance criteria (ANY of these → accepted):
    #    a) new_sla_violations < baseline_sla_violations
    #    b) new_cost < baseline_cost AND new_sla_violations <= baseline_sla_violations
    #    c) new_scaling_events < baseline AND new_sla_violations <= baseline

    accepted = False
    reason = ""

    # Condition a: SLA violations reduced
    if new_metrics['sla_violations'] < baseline_metrics['sla_violations']:
        accepted = True
        reason = f"SLA violations reduced from {baseline_metrics['sla_violations']} to {new_metrics['sla_violations']}"

    # Condition b: Cost reduced without increasing SLA violations
    elif (new_metrics['total_cost'] < baseline_metrics['total_cost'] and
          new_metrics['sla_violations'] <= baseline_metrics['sla_violations']):
        accepted = True
        reason = f"Cost reduced from ${baseline_metrics['total_cost']:.2f} to ${new_metrics['total_cost']:.2f} (SLA violations unchanged at {baseline_metrics['sla_violations']})"

    # Condition c: Scaling events reduced without increasing SLA violations
    elif (new_metrics['scaling_events'] < baseline_metrics['scaling_events'] and
          new_metrics['sla_violations'] <= baseline_metrics['sla_violations']):
        accepted = True
        reason = f"Scaling events reduced from {baseline_metrics['scaling_events']} to {new_metrics['scaling_events']} (SLA violations unchanged at {baseline_metrics['sla_violations']})"

    # Rejected
    else:
        if new_metrics['sla_violations'] > baseline_metrics['sla_violations']:
            reason = f"SLA violations increased from {baseline_metrics['sla_violations']} to {new_metrics['sla_violations']}"
        else:
            reason = f"No improvement in SLA, cost, or scaling events"

    return {
        "new_metrics": new_metrics,
        "sla_improvement_pct": sla_improvement_pct,
        "cost_change_pct": cost_change_pct,
        "scaling_events_change_pct": scaling_events_change_pct,
        "accepted": accepted,
        "reason": reason,
    }
