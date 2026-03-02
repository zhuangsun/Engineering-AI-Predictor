from pydantic import BaseModel


class DesignInput(BaseModel):
    thickness: float
    length: float
    width: float
