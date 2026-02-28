"""Pydantic schemas and validation constants for LLM recommendations."""

import re
from typing import Dict, Literal, Union
from pydantic import BaseModel, field_validator, model_validator

# Constants
ALLOWED_PARAMETERS = ["cooldown", "capacity_per_instance", "initial_instances", "arima_order"]

ALLOWED_TYPES = ["parameter_tuning", "model_tuning", "policy_suggestion"]

# Parameter ranges for numeric params
PARAM_RANGES = {
    "cooldown": (1, 60),
    "capacity_per_instance": (10, 1000),
    "initial_instances": (1, 10),
}

# arima_order: string "(p,d,q)" where 0 <= p,d,q <= 5
ARIMA_ORDER_PATTERN = r"^\(\s*([0-5])\s*,\s*([0-5])\s*,\s*([0-5])\s*\)$"


class SimulatorConfig(BaseModel):
    """Configuration for autoscaling simulator."""
    capacity_per_instance: int
    cost_per_instance: float
    cooldown: int
    initial_instances: int

    @field_validator('capacity_per_instance')
    @classmethod
    def validate_capacity(cls, v):
        if not (10 <= v <= 1000):
            raise ValueError('capacity_per_instance must be between 10 and 1000')
        return v

    @field_validator('cost_per_instance')
    @classmethod
    def validate_cost(cls, v):
        if v <= 0:
            raise ValueError('cost_per_instance must be greater than 0')
        return v

    @field_validator('cooldown')
    @classmethod
    def validate_cooldown(cls, v):
        if not (1 <= v <= 60):
            raise ValueError('cooldown must be between 1 and 60')
        return v

    @field_validator('initial_instances')
    @classmethod
    def validate_initial_instances(cls, v):
        if not (1 <= v <= 10):
            raise ValueError('initial_instances must be between 1 and 10')
        return v


class Recommendation(BaseModel):
    """LLM recommendation for autoscaling configuration improvement."""
    type: Literal["parameter_tuning", "model_tuning", "policy_suggestion"]
    parameter: Literal["cooldown", "capacity_per_instance", "initial_instances", "arima_order"]
    current_value: Union[int, float, str]
    proposed_value: Union[int, float, str]
    rationale: str
    expected_impact: str
    confidence: float
    evidence: Dict[str, Union[int, float, str]]

    @field_validator('rationale')
    @classmethod
    def validate_rationale_length(cls, v):
        if len(v) < 20:
            raise ValueError('rationale must be at least 20 characters')
        return v

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v

    @field_validator('evidence')
    @classmethod
    def validate_evidence_count(cls, v):
        if len(v) < 2:
            raise ValueError('evidence must contain at least 2 entries')
        return v

    @model_validator(mode='after')
    def validate_parameter_value(self):
        """Validate proposed_value based on parameter type."""
        param = self.parameter
        proposed = self.proposed_value

        # Validate numeric parameters
        if param in PARAM_RANGES:
            min_val, max_val = PARAM_RANGES[param]
            if not isinstance(proposed, (int, float)):
                raise ValueError(f'{param} proposed_value must be numeric')
            if not (min_val <= proposed <= max_val):
                raise ValueError(f'{param} proposed_value must be between {min_val} and {max_val}')

        # Validate arima_order
        elif param == "arima_order":
            if not isinstance(proposed, str):
                raise ValueError('arima_order proposed_value must be a string')
            if not re.match(ARIMA_ORDER_PATTERN, proposed):
                raise ValueError('arima_order must match pattern "(p,d,q)" where 0<=p,d,q<=5')

        return self


class RecommendationResponse(BaseModel):
    """Response containing multiple recommendations."""
    recommendations: list[Recommendation]
