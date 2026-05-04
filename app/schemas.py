from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Union[list[float], list[list[float]]] = Field(
        ..., description="Single feature row or batch of feature rows"
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str
    n_features: int
