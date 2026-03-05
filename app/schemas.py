from typing import Literal
from pydantic import BaseModel, model_validator


class DesignInput(BaseModel):
    thickness: float
    length: float
    width: float


class OptimizationBounds(BaseModel):
    thickness_min: float
    thickness_max: float
    length_min: float
    length_max: float
    width_min: float
    width_max: float


class GAOptimizationRequest(OptimizationBounds):
    pop_size: int = 100
    n_generations: int = 50


class SensitivityRequest(BaseModel):
    variable: Literal["thickness", "length", "width"]
    fixed_thickness: float = 5.0
    fixed_length: float = 12.0
    fixed_width: float = 6.0
    sweep_min: float
    sweep_max: float
    n_points: int = 60

    @model_validator(mode="after")
    def check_sweep_range(self) -> "SensitivityRequest":
        if self.sweep_min >= self.sweep_max:
            raise ValueError("sweep_min must be less than sweep_max")
        return self
