from typing import Literal
from pydantic import BaseModel, Field, model_validator


class DesignInput(BaseModel):
    h: float = Field(gt=0, le=2.0,   description="Weld size [0.1, 2.0] in")
    l: float = Field(gt=0, le=10.0,  description="Weld length [0.1, 10.0] in")
    t: float = Field(gt=0, le=10.0,  description="Bar thickness [0.1, 10.0] in")
    b: float = Field(gt=0, le=2.0,   description="Bar height [0.1, 2.0] in")


class OptimizationBounds(BaseModel):
    h_min: float
    h_max: float
    l_min: float
    l_max: float
    t_min: float
    t_max: float
    b_min: float
    b_max: float

    @model_validator(mode="after")
    def check_bounds_order(self) -> "OptimizationBounds":
        for var in ("h", "l", "t", "b"):
            lo, hi = getattr(self, f"{var}_min"), getattr(self, f"{var}_max")
            if lo >= hi:
                raise ValueError(f"{var}_min ({lo}) must be less than {var}_max ({hi})")
        return self


class GAOptimizationRequest(OptimizationBounds):
    pop_size: int = 100
    n_generations: int = 50


class SensitivityRequest(BaseModel):
    variable: Literal["h", "l", "t", "b"]
    fixed_h: float = 0.5
    fixed_l: float = 5.0
    fixed_t: float = 5.0
    fixed_b: float = 0.5
    sweep_min: float
    sweep_max: float
    n_points: int = 60

    @model_validator(mode="after")
    def check_sweep_range(self) -> "SensitivityRequest":
        if self.sweep_min >= self.sweep_max:
            raise ValueError("sweep_min must be less than sweep_max")
        return self
