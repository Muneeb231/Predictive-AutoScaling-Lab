#!/usr/bin/env python
"""CLI orchestrator for LLM-powered autoscaling recommendation analysis."""

import argparse
import json
import pandas as pd
from datetime import datetime
import os

from llm_system.analyzer import analyze
from llm_system.validator import validate_schema, test_recommendation
from llm_system.metrics import calculate_all


def main():

    
    """Main entry point for recommendation analysis."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="LLM Recommendation Analysis")
    
    
    parser.add_argument("--data", default="simulator_output.csv",
                       help="Path to simulator output CSV")
    parser.add_argument("--config", default="experiments/baseline_config.json",
                       help="Path to baseline config JSON")
    parser.add_argument("--schema-only", action="store_true",
                       help="Skip simulation testing, only validate schema")
    args = parser.parse_args()

    # Load data
    
    df = pd.read_csv(args.data)
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Calculate baseline metrics
    baseline_metrics = calculate_all(df, config)

    # Print header and baseline summary
    print("=== LLM Recommendation Analysis ===")
    print(f"Baseline: {baseline_metrics['sla_violations']} SLA violations | "
          f"${baseline_metrics['total_cost']:.2f} cost | "
          f"{baseline_metrics['scaling_events']} scaling events | "
          f"MAPE {baseline_metrics['mape']:.1f}%")
    print()

    # Get recommendations from LLM
    recommendations = analyze(args.data, config)

    # Process each recommendation
    results = []
    for idx, rec in enumerate(recommendations, 1):
        print(f"Recommendation {idx}: {rec.get('type', 'unknown')} — "
              f"{rec.get('parameter', 'unknown')} "
              f"{rec.get('current_value', '?')} → {rec.get('proposed_value', '?')}")

        # Schema validation (Tier 1)
        schema_ok, schema_msg = validate_schema(rec)

        if schema_ok:
            print(f"  Schema:     ✓ PASS")
        else:
            print(f"  Schema:     ✗ FAIL — {schema_msg}")
            print(f"  Decision:   REJECTED")
            print()
            results.append({
                "recommendation": rec,
                "schema_valid": False,
                "schema_error": schema_msg,
                "simulation_result": None
            })
            continue

        # Simulation testing (Tier 2)
        sim_result = None

        if args.schema_only:
            print(f"  Simulation: ⊘ SKIPPED (--schema-only flag)")
            print(f"  Decision:   SCHEMA VALID")

        elif rec['type'] == 'parameter_tuning':
            # Run simulation test
            sim_result = test_recommendation(rec, config, baseline_metrics, args.data)

            # Format output
            sla_old = baseline_metrics['sla_violations']
            sla_new = sim_result['new_metrics']['sla_violations']
            sla_pct = sim_result['sla_improvement_pct']

            cost_old = baseline_metrics['total_cost']
            cost_new = sim_result['new_metrics']['total_cost']
            cost_pct = sim_result['cost_change_pct']

            if sim_result['accepted']:
                print(f"  Simulation: ✓ PASS — {sim_result['reason']}")
                print(f"              SLA violations {sla_old} → {sla_new} ({sla_pct:+.1f}%), "
                      f"cost ${cost_old:.2f} → ${cost_new:.2f} ({cost_pct:+.1f}%)")
                print(f"  Decision:   ACCEPTED")
            else:
                print(f"  Simulation: ✗ FAIL — {sim_result['reason']}")
                print(f"              SLA violations {sla_old} → {sla_new} ({sla_pct:+.1f}%), "
                      f"cost ${cost_old:.2f} → ${cost_new:.2f} ({cost_pct:+.1f}%)")
                print(f"  Decision:   REJECTED")

        elif rec['type'] in ('model_tuning', 'policy_suggestion'):
            # Skip simulation for these types
            skip_reason = {
                'model_tuning': 'requires ARIMA retraining',
                'policy_suggestion': 'requires code changes'
            }[rec['type']]
            print(f"  Simulation: ⊘ SKIPPED ({skip_reason})")
            print(f"  Decision:   NOTED (manual validation required)")

        else:
            print(f"  Simulation: ⊘ SKIPPED (unknown type)")
            print(f"  Decision:   NOTED")

        print()

        results.append({
            "recommendation": rec,
            "schema_valid": True,
            "simulation_result": sim_result
        })

    # Calculate summary statistics
    total = len(recommendations)
    schema_valid = sum(1 for r in results if r['schema_valid'])
    simulation_eligible = sum(1 for r in results
                             if r['schema_valid'] and
                             r['recommendation'].get('type') == 'parameter_tuning')
    simulation_tested = sum(1 for r in results
                           if r['simulation_result'] is not None)
    simulation_skipped = schema_valid - simulation_tested
    accepted = sum(1 for r in results
                  if r['simulation_result'] and r['simulation_result']['accepted'])

    acceptance_rate = accepted / simulation_tested if simulation_tested > 0 else 0.0

    summary = {
        "total_recommendations": total,
        "schema_valid": schema_valid,
        "simulation_eligible": simulation_eligible,
        "simulation_tested": simulation_tested,
        "simulation_skipped": simulation_skipped,
        "accepted": accepted,
        "acceptance_rate": acceptance_rate
    }

    # Build experiment result
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    experiment_result = {
        "timestamp": timestamp,
        "baseline_config": config,
        "baseline_metrics": baseline_metrics,
        "recommendations": results,
        "summary": summary
    }

    # Save to file
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"experiments/results/experiment_{timestamp_file}.json"

    # Ensure output directory exists
    os.makedirs("experiments/results", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(experiment_result, f, indent=2)

    # Print summary
    print(f"Summary: {schema_valid}/{total} schema valid, "
          f"{simulation_tested}/{simulation_eligible} simulation tested, "
          f"{accepted}/{simulation_tested if simulation_tested > 0 else 0} accepted")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":


    main()
