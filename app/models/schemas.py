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


class PredictionBatchInput(BaseModel):

    inputs: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=100
    )


class PredictionBatchOutput(BaseModel):

    predictions: list[PredictionOutput]