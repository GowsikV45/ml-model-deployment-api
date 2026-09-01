from contextlib import asynccontextmanager
from uuid import uuid4
import time

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import setup_logging
from app.routers.v1 import router as v1_router


model = None

logger = setup_logging()


class PredictionError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load(settings.MODEL_PATH)

    app.state.model = model
    app.state.logger = logger

    logger.info("RandomForest model loaded successfully!")

    yield


app = FastAPI(
    title=settings.API_TITLE,
    lifespan=lifespan
)


# ---------------------------------------------------------
# Task 9 - Request Logging Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = str(uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    logger.info(
        f"Request | "
        f"request_id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path} | "
        f"status_code={response.status_code} | "
        f"duration={duration:.4f}s"
    )

    return response


# ---------------------------------------------------------
# Task 8 - Custom Exception Handler
# ---------------------------------------------------------

@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request,
    exc: PredictionError
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.error(
        f"Prediction failed | "
        f"request_id={request_id}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction failed"
        }
    )


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }


# ---------------------------------------------------------
# Task 10+ - Versioned API Router
# ---------------------------------------------------------

app.include_router(v1_router)