import json
import os
import re
import pandas as pd
import google.generativeai as genai

from llm_system.metrics import calculate_all, detect_flapping_windows
from llm_system.schemas import RecommendationResponse


SYSTEM_MESSAGE = """You are an expert AIOps engineer specializing in autoscaling optimization.
You analyze simulation logs to identify performance problems and recommend
specific configuration changes. You always cite specific metrics and time
windows as evidence for your recommendations."""


def analyze(simulation_csv_path: str, config: dict) -> list[dict]:
    df = pd.read_csv(simulation_csv_path)
    summary_metrics = calculate_all(df, config)
    flapping_windows = detect_flapping_windows(df)

    prompt = _build_prompt(df, config, summary_metrics, flapping_windows)
    response_text = _call_gemini(prompt)

    return _parse_response(response_text)


def _build_prompt(df: pd.DataFrame, config: dict, summary_metrics: dict, flapping_windows: list[dict]) -> str:
    config_json = json.dumps(config, indent=2)

    if flapping_windows:
        flapping_text = "\n".join([
            f"- Minutes {w['start_minute']}-{w['end_minute']}: {w['events']} scaling events"
            for w in flapping_windows
        ])
    else:
        flapping_text = "None detected"

    csv_data_json = df.to_json(orient='records', indent=2)
    schema_json = json.dumps(RecommendationResponse.model_json_schema(), indent=2)

    return f"""{SYSTEM_MESSAGE}

TASK: Analyze the autoscaling simulation data below and recommend 1-3
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
1. Identify patterns: flapping, forecast errors, over/under-provisioning, SLA violation clusters
2. Cite SPECIFIC numbers from the data
3. Explain WHY the problem occurs and WHY your fix will help
4. Provide quantitative expected impact

ALLOWED PARAMETERS:
- cooldown: 1-60 (minutes)
- capacity_per_instance: 10-1000
- initial_instances: 1-10
- arima_order: tuple of (p, d, q) where 0 <= p,d,q <= 5

OUTPUT FORMAT:
- Output ONLY raw JSON, no markdown code blocks
- Do NOT wrap the JSON in ```json``` or any other formatting
- Start your response with {{ and end with }}
- The JSON must match this exact schema:
{schema_json}"""


def _call_gemini(prompt: str) -> str:
    genai.configure(api_key=os.environ["GEMINI_3_FLASH_PREVIEW"])
    model = genai.GenerativeModel('gemini-3-flash-preview')

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=16384,
        )
    )

    return response.text


def _parse_response(response_text: str) -> list[dict]:
    try:
        parsed = json.loads(response_text)
        return parsed["recommendations"]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group(1))
        return parsed["recommendations"]

    raise Exception(f"Could not parse JSON from response:\n{response_text}")
