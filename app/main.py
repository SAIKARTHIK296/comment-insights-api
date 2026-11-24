import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.anomalies import AnomalyEngine
from app.auth import validate_auth_config, verify_token
from app.language import LanguageDetector
from app.quality import QualityEngine
from app.redact import RedactionEngine
from app.schemas import (
    AnomalyPointRequest,
    AnomalyResponse,
    QualityRequestItem,
    QualityResponseItem,
    RedactionRequestItem,
    RedactionResponseItem,
    ThemeRequestItem,
    ThemeResponseItem,
)
from app.themes import ThemeEngine


def get_allowed_origins() -> list[str]:
    return [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_config()
    lang_detector = LanguageDetector()
    app.state.language_detector = lang_detector
    app.state.theme_engine = ThemeEngine(language_detector=lang_detector.detect)
    app.state.redaction_engine = RedactionEngine()
    app.state.quality_engine = QualityEngine(language_detector=lang_detector.detect)
    app.state.anomaly_engine = AnomalyEngine()
    yield


app = FastAPI(
    title="Comment Insights API",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "defaultModelsExpandDepth": -1,
    },
)

allowed_origins = get_allowed_origins()
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.middleware("http")
async def add_observability_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "models": {
            "themes": {
                "version": request.app.state.theme_engine.model_version,
                "cvMacroF1": request.app.state.theme_engine.metadata.get("cvMacroF1"),
            },
            "language": {
                "version": request.app.state.language_detector.model_version,
                "cvAccuracy": request.app.state.language_detector.metadata.get("cvAccuracy"),
            },
            "redactionNer": request.app.state.redaction_engine.spacy_model,
        },
    }


@app.post("/v1/themes", response_model=list[ThemeResponseItem])
async def tag_themes(
    request: Request,
    items: list[ThemeRequestItem] = Body(..., min_length=1, max_length=500),
    _: str = Depends(verify_token),
):
    return await run_in_threadpool(request.app.state.theme_engine.predict_batch, items)


@app.post("/v1/redact", response_model=list[RedactionResponseItem])
async def redact_comments(
    request: Request,
    items: list[RedactionRequestItem] = Body(..., min_length=1, max_length=500),
    _: str = Depends(verify_token),
):
    return await run_in_threadpool(request.app.state.redaction_engine.redact_batch, items)


@app.post("/v1/comment-quality", response_model=list[QualityResponseItem])
async def check_comment_quality(
    request: Request,
    items: list[QualityRequestItem] = Body(..., min_length=1, max_length=500),
    _: str = Depends(verify_token),
):
    return await run_in_threadpool(request.app.state.quality_engine.evaluate_batch, items)


@app.post("/v1/anomalies", response_model=AnomalyResponse)
async def detect_anomalies(
    request: Request,
    points: list[AnomalyPointRequest] = Body(..., min_length=1, max_length=5000),
    _: str = Depends(verify_token),
):
    return await run_in_threadpool(request.app.state.anomaly_engine.detect, points)