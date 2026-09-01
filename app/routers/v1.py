import time
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Request, HTTPException

from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput,
)


router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@router.get("/health")
def health(request: Request):

    model = request.app.state.model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ---------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------

@router.post("/predict", response_model=PredictionOutput)
def predict(
    data: PredictionInput,
    request: Request
):

    model = request.app.state.model
    logger = request.app.state.logger

    request_id = request.state.request_id

    input_data = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width,
    }])

    start_time = time.perf_counter()

    try:

        prediction = model.predict(input_data)

        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            confidence = float(probabilities.max())

        duration = time.perf_counter() - start_time

        logger.info(
            f"Prediction successful | "
            f"request_id={request_id} | "
            f"prediction={int(prediction[0])} | "
            f"duration={duration:.4f}s"
        )

        return {
            "prediction": int(prediction[0]),
            "confidence": confidence,
            "request_id": request_id
        }

    except Exception as exc:

        logger.error(
            f"Prediction error | "
            f"request_id={request_id} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------

@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):

    model = request.app.state.model
    logger = request.app.state.logger

    request_id = request.state.request_id

    batch_size = len(data.inputs)

    # -----------------------------------------------------
    # Task 12 - Maximum Batch Size
    # -----------------------------------------------------

    if batch_size > settings.MAX_BATCH_SIZE:

        logger.warning(
            f"Batch size exceeded | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"max_batch_size={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size cannot exceed "
                f"{settings.MAX_BATCH_SIZE}"
            )
        )

    if batch_size == 0:

        raise HTTPException(
            status_code=400,
            detail="Batch cannot be empty"
        )

    start_time = time.perf_counter()

    try:

        # Convert complete batch into DataFrame
        input_data = pd.DataFrame([
            {
                "sepal length (cm)": item.sepal_length,
                "sepal width (cm)": item.sepal_width,
                "petal length (cm)": item.petal_length,
                "petal width (cm)": item.petal_width,
            }
            for item in data.inputs
        ])

        # -------------------------------------------------
        # Efficient batch prediction
        # -------------------------------------------------

        predictions = model.predict(input_data)

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)

        duration = time.perf_counter() - start_time

        results = []

        for index, prediction in enumerate(predictions):

            confidence = 0.0

            if probabilities is not None:
                confidence = float(probabilities[index].max())

            results.append(
                PredictionOutput(
                    prediction=int(prediction),
                    confidence=confidence,
                    request_id=request_id
                )
            )

        logger.info(
            f"Batch prediction successful | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s"
        )

        return {
            "predictions": results
        }

    except Exception as exc:

        logger.error(
            f"Batch prediction error | "
            f"request_id={request_id} | "
            f"batch_size={batch_size} | "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )


# ---------------------------------------------------------
# Model Info
# ---------------------------------------------------------

@router.get("/model-info")
def model_info(request: Request):

    model = request.app.state.model

    return {
        "model_type": type(model).__name__,
        "version": "1.0.0",
        "training_date": "2026-09-01",
        "expected_features": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    }