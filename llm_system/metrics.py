"""Extract performance metrics from autoscaling simulation results."""

import pandas as pd
import numpy as np


def calculate_all(df: pd.DataFrame, config: dict) -> dict:
    """Calculate all summary metrics from simulation DataFrame.

    Args:
        df: Simulator output DataFrame with columns:
            instances, actual_load, predicted_load, sla_violation,
            cost, scaled_up, scaled_down
        config: Simulator config dict with capacity_per_instance

    Returns:
        Dict with 10 metrics: sla_violations, total_cost, scaling_events,
        scale_up_events, scale_down_events, avg_utilization, max_instances,
        min_instances, mape, mae
    """
    # Basic counts
    sla_violations = int(df['sla_violation'].sum())
    total_cost = float(df['cost'].sum())
    scale_up_events = int(df['scaled_up'].sum())
    scale_down_events = int(df['scaled_down'].sum())
    scaling_events = scale_up_events + scale_down_events

    # Instance stats
    max_instances = int(df['instances'].max())
    min_instances = int(df['instances'].min())

    # Average utilization (percentage)
    capacity_per_instance = config['capacity_per_instance']
    total_capacity = df['instances'] * capacity_per_instance

    # Filter out rows where total_capacity is 0 to avoid division by zero
    valid_utilization_mask = total_capacity > 0
    if valid_utilization_mask.sum() > 0:
        utilization = df[valid_utilization_mask]['actual_load'] / total_capacity[valid_utilization_mask]
        avg_utilization = float(utilization.mean() * 100)
    else:
        avg_utilization = 0.0

    # MAPE (Mean Absolute Percentage Error)
    # Filter to rows where actual_load > 0 to avoid division by zero
    valid_mape_mask = df['actual_load'] > 0
    if valid_mape_mask.sum() > 0:
        percentage_errors = (
            np.abs(df[valid_mape_mask]['actual_load'] - df[valid_mape_mask]['predicted_load'])
            / df[valid_mape_mask]['actual_load']
        )
        mape = float(percentage_errors.mean() * 100)
    else:
        mape = 0.0

    # MAE (Mean Absolute Error)
    mae = float(np.abs(df['actual_load'] - df['predicted_load']).mean())

    return {
        "sla_violations": sla_violations,
        "total_cost": total_cost,
        "scaling_events": scaling_events,
        "scale_up_events": scale_up_events,
        "scale_down_events": scale_down_events,
        "avg_utilization": avg_utilization,
        "max_instances": max_instances,
        "min_instances": min_instances,
        "mape": mape,
        "mae": mae,
    }


def detect_flapping_windows(df: pd.DataFrame, window_size: int = 30, threshold: int = 6) -> list[dict]:
    """Detect time windows with excessive scaling events (flapping).

    Args:
        df: Simulator output DataFrame
        window_size: Size of sliding window in minutes
        threshold: Minimum scaling events to flag as flapping

    Returns:
        List of dicts with start_minute, end_minute, events.
        Returns empty list if no flapping detected.
    """
    # Calculate total scaling events at each timestep
    df = df.copy()
    df['scaling_event'] = df['scaled_up'] + df['scaled_down']

    # Find windows with excessive scaling
    flapping_windows = []

    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        events = int(window['scaling_event'].sum())

        if events >= threshold:
            flapping_windows.append({
                'start_minute': i,
                'end_minute': i + window_size - 1,
                'events': events
            })

    # Merge overlapping windows
    if not flapping_windows:
        return []

    merged = []
    current = flapping_windows[0]

    for next_window in flapping_windows[1:]:
        # If overlapping or adjacent, merge
        if next_window['start_minute'] <= current['end_minute'] + 1:
            current['end_minute'] = max(current['end_minute'], next_window['end_minute'])
            current['events'] = max(current['events'], next_window['events'])
        else:
            merged.append(current)
            current = next_window

    merged.append(current)
    return merged
