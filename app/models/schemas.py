from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    request_id: str