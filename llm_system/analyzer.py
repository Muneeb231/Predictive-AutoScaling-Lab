"""LLM-powered analysis of autoscaling simulation results using Claude API."""

import json
import os
import pandas as pd
import anthropic

from llm_system.metrics import calculate_all, detect_flapping_windows
from llm_system.schemas import RecommendationResponse


SYSTEM_MESSAGE = """You are an expert AIOps engineer specializing in autoscaling optimization.
You analyze simulation logs to identify performance problems and recommend
specific configuration changes. You always cite specific metrics and time
windows as evidence for your recommendations."""


def analyze(simulation_csv_path: str, config: dict) -> list[dict]:
    """Analyze simulation results and generate recommendations using Claude API.

    Args:
        simulation_csv_path: Path to simulator_output.csv
        config: Simulator configuration dict

    Returns:
        List of recommendation dicts (raw dicts, not Pydantic models)
    """
    # Load and analyze simulation data
    df = pd.read_csv(simulation_csv_path)
    summary_metrics = calculate_all(df, config)
    flapping_windows = detect_flapping_windows(df)

    # Build prompt
    system_msg, user_msg = _build_prompt(df, config, summary_metrics, flapping_windows)

    # Call Claude API with retry logic
    response_text = _call_claude(system_msg, user_msg)

    # Parse and return recommendations
    recommendations = _parse_response(response_text, system_msg, user_msg)
    return recommendations


def _build_prompt(df: pd.DataFrame, config: dict, summary_metrics: dict, flapping_windows: list[dict]) -> tuple[str, str]:
    """Build system and user messages for Claude API.

    Args:
        df: Simulation DataFrame
        config: Simulator config
        summary_metrics: Pre-computed metrics from calculate_all()
        flapping_windows: Pre-computed flapping windows

    Returns:
        Tuple of (system_message, user_message)
    """
    # Format config as pretty JSON
    config_json = json.dumps(config, indent=2)

    # Format flapping windows
    if flapping_windows:
        flapping_text = "\n".join([
            f"- Minutes {w['start_minute']}-{w['end_minute']}: {w['events']} scaling events"
            for w in flapping_windows
        ])
    else:
        flapping_text = "None detected"

    # Format CSV data as JSON array
    csv_data_json = df.to_json(orient='records', indent=2)

    # Get JSON schema from Pydantic model
    schema_json = json.dumps(RecommendationResponse.model_json_schema(), indent=2)

    # Build user message
    user_message = f"""TASK: Analyze the autoscaling simulation data below and recommend 1-3
configuration changes. Prioritize reducing SLA violations over cost savings
when trade-offs are necessary.

CURRENT CONFIGURATION:
{config_json}

SUMMARY METRICS:
- SLA Violations: {summary_metrics['sla_violations']} out of 1440 minutes
- Total Cost: ${summary_metrics['total_cost']:.2f}
- Scaling Events: {summary_metrics['scaling_events']} ({summary_metrics['scale_up_events']} up, {summary_metrics['scale_down_events']} down)
- Average Utilization: {summary_metrics['avg_utilization']:.1f}%
- Forecast Error (MAPE): {summary_metrics['mape']:.1f}%
- Forecast Error (MAE): {summary_metrics['mae']:.1f}
- Max Instances: {summary_metrics['max_instances']}
- Min Instances: {summary_metrics['min_instances']}

FLAPPING WINDOWS DETECTED:
{flapping_text}

SIMULATION DATA (1440 minutes):
{csv_data_json}

INSTRUCTIONS:
1. Identify patterns: flapping, forecast errors, over/under-provisioning,
   SLA violation clusters
2. For each recommendation, cite SPECIFIC numbers from the data
   (e.g., "23 scaling events between minute 480-510")
3. Explain WHY the problem occurs and WHY your fix will help
4. Provide quantitative expected impact (e.g., "reduce violations by ~40%")

ALLOWED PARAMETERS:
- cooldown: 1-60 (minutes)
- capacity_per_instance: 10-1000
- initial_instances: 1-10
- arima_order: tuple of (p, d, q) where 0 <= p,d,q <= 5

Output ONLY valid JSON matching this schema:
{schema_json}"""

    return SYSTEM_MESSAGE, user_message


def _call_claude(system_msg: str, user_msg: str) -> str:
    """Call Claude API and return response text.

    Args:
        system_msg: System message defining role
        user_msg: User message with task and data

    Returns:
        Response text from Claude

    Raises:
        Exception: If API call fails
    """
    # Initialize client (reads ANTHROPIC_API_KEY from environment)
    client = anthropic.Anthropic()

    # Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-20241022",
        max_tokens=4096,
        temperature=0,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}]
    )

    # Extract text from response
    return response.content[0].text


def _parse_response(response_text: str, system_msg: str, user_msg: str) -> list[dict]:
    """Parse JSON response from Claude, with retry logic.

    Args:
        response_text: Raw response text from Claude
        system_msg: Original system message (for retry)
        user_msg: Original user message (for retry)

    Returns:
        List of recommendation dicts

    Raises:
        Exception: If parsing fails after retry
    """
    try:
        # Try to parse JSON
        parsed = json.loads(response_text)
        return parsed["recommendations"]
    except json.JSONDecodeError as e:
        # Retry once with clarification
        print(f"Warning: Initial JSON parse failed: {e}")
        print("Retrying with clarification...")

        retry_msg = user_msg + "\n\nYour previous response was not valid JSON. Output ONLY the JSON object, no other text."

        # Call Claude again
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20241022",
            max_tokens=4096,
            temperature=0,
            system=system_msg,
            messages=[{"role": "user", "content": retry_msg}]
        )
        retry_text = response.content[0].text

        try:
            parsed = json.loads(retry_text)
            return parsed["recommendations"]
        except json.JSONDecodeError as retry_error:
            # Both attempts failed
            raise Exception(
                f"Failed to parse JSON after retry.\n"
                f"Original error: {e}\n"
                f"Retry error: {retry_error}\n"
                f"Original response:\n{response_text}\n\n"
                f"Retry response:\n{retry_text}"
            )
