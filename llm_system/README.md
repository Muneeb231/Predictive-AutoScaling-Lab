# Quick LLM Autoscaling Recommendation System ReadMe

Uses LLMs to analyze autoscaling simulation data and recommend configuration improvements to reduce SLA violations.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install pandas pydantic google-generativeai
```

### 2. Set API Key

Add your Gemini API key to `.env`:

```bash
GEMINI_3_FLASH_PREVIEW=your-api-key-here
```

### 3. Run Analysis

```bash
python run_analysis.py
```

## Example Output

```
=== LLM Recommendation Analysis ===
Baseline: 87 SLA violations | $1936.00 cost | 68 scaling events | MAPE 33.1%

Recommendation 1: parameter_tuning — cooldown 5 → 3
  Schema:     ✓ PASS
  Simulation: ✓ PASS — SLA violations reduced from 87 to 82
  Decision:   ACCEPTED

Summary: 2/2 schema valid, 2/2 simulation tested, 1/2 accepted
Results saved to: experiments/results/experiment_20260228_160052.json
```
