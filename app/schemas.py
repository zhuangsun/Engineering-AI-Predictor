from pydantic import BaseModel


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
