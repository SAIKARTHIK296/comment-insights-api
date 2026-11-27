from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.anomalies import AnomalyEngine
from app.auth import validate_auth_config, verify_token
from app.language import LanguageDetector
from app.quality import QualityEngine
from app.redact import RedactionEngine
from app.redact_views import apply_redaction_view
from app.schemas import (
    AnomalyPointRequest,
    AnomalyResponse,
    QualityRequestItem,
    QualityResponseItem,
    RedactionEntity,
    RedactionRequestItem,
    RedactionResponseItem,
    ThemeRequestItem,
    ThemeResponseItem,
)
from app.themes import ThemeEngine

MAX_BATCH_SIZE = 500


def _detect_language(detector: Any, text: Optional[str], override: Optional[str] = None) -> str:
    if override and str(override).strip():
        return str(override).strip().lower()
    if text is None or not str(text).strip():
        return "unknown"

    raw = str(text)
    for method_name in ("detect", "predict", "_detect_language"):
        fn = getattr(detector, method_name, None)
        if callable(fn):
            res = fn(raw)
            if isinstance(res, tuple):
                return str(res[0])
            if isinstance(res, dict):
                return str(res.get("language", "unknown"))
            return str(res)
    return "unknown"


def _extract_spans_from_redactor(redactor: Any, item: RedactionRequestItem) -> list[Any]:
    raw_text = item.text or ""
    for method_name in ("redact_text", "redact"):
        fn = getattr(redactor, method_name, None)
        if callable(fn):
            res = fn(raw_text)
            if isinstance(res, tuple):
                return list(res[1])
            if hasattr(res, "entities") and res.entities is not None:
                return list(res.entities)
            if hasattr(res, "redactions") and res.redactions is not None:
                return list(res.redactions)
            if isinstance(res, dict):
                return list(res.get("entities") or res.get("redactions") or [])

    batch_fn = getattr(redactor, "redact_batch", None)
    if callable(batch_fn):
        batch_res = batch_fn([item])
        if batch_res:
            first = batch_res[0]
            if hasattr(first, "entities") and first.entities is not None:
                return list(first.entities)
            if hasattr(first, "redactions") and first.redactions is not None:
                return list(first.redactions)
            if isinstance(first, dict):
                return list(first.get("entities") or first.get("redactions") or [])

    return []


def _evaluate_quality_item(
    quality_engine: Any,
    lang_detector: Any,
    item: QualityRequestItem,
) -> QualityResponseItem:
    raw_text = item.text or ""
    detected_lang = _detect_language(lang_detector, raw_text, item.language)

    fn = getattr(quality_engine, "evaluate", None) or getattr(quality_engine, "score", None)
    if callable(fn):
        try:
            res = fn(raw_text, language=detected_lang)
        except TypeError:
            try:
                res = fn(raw_text)
            except TypeError:
                res = fn(item)

        if isinstance(res, QualityResponseItem):
            return res
        if isinstance(res, dict):
            res.setdefault("id", item.id)
            res.setdefault("language", detected_lang)
            return QualityResponseItem(**res)
        if isinstance(res, tuple):
            is_m = bool(res[0])
            sc = float(res[1]) if len(res) > 1 else (1.0 if is_m else 0.0)
            rsn = res[2] if len(res) > 2 else None
            return QualityResponseItem(
                id=item.id,
                isMeaningful=is_m,
                qualityScore=round(sc, 4),
                reason=rsn,
                language=detected_lang,
            )
        if hasattr(res, "model_dump"):
            data = res.model_dump()
            data.setdefault("id", item.id)
            data.setdefault("language", detected_lang)
            return QualityResponseItem(**data)

    return QualityResponseItem(
        id=item.id,
        isMeaningful=bool(raw_text.strip()),
        qualityScore=1.0 if raw_text.strip() else 0.0,
        reason=None if raw_text.strip() else "empty_text",
        language=detected_lang,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_config()
    app.state.lang_detector = LanguageDetector()
    app.state.theme_engine = ThemeEngine()
    app.state.redaction_engine = RedactionEngine()
    app.state.quality_engine = QualityEngine()
    app.state.anomaly_engine = AnomalyEngine()
    yield


app = FastAPI(
    title="Comment Insights API",
    version="1.1.0",
    description=(
        "Stateless hospital patient feedback microservice for multilingual complaint theme tagging, "
        "personal-data redaction (strict, staff, and reveal_all views), comment quality scoring, "
        "and daily department anomaly alerts."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request):
    theme_engine = getattr(request.app.state, "theme_engine", None)
    lang_detector = getattr(request.app.state, "lang_detector", None)
    return {
        "status": "ok",
        "themeModelVersion": getattr(theme_engine, "model_version", "themes_v1.0.0"),
        "languageModelVersion": getattr(lang_detector, "model_version", "lang_v1.0.0"),
    }


@app.post(
    "/v1/themes",
    response_model=list[ThemeResponseItem],
)
def predict_themes(
    items: list[ThemeRequestItem],
    request: Request,
    _: None = Depends(verify_token),
):
    if len(items) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}.")

    theme_engine: ThemeEngine = request.app.state.theme_engine
    lang_detector = request.app.state.lang_detector

    results: list[ThemeResponseItem] = []
    for item in items:
        lang = _detect_language(lang_detector, item.text, item.language)
        preds, omitted = theme_engine.predict_text(item.text)
        results.append(
            ThemeResponseItem(
                id=item.id,
                themes=preds,
                language=lang,
                modelVersion=theme_engine.model_version,
                omittedReason=omitted,
            )
        )
    return results


@app.post(
    "/v1/redact",
    response_model=list[RedactionResponseItem],
    response_model_exclude_none=True,
)
def redact_comments(
    items: list[RedactionRequestItem],
    request: Request,
    view: Literal["strict", "staff", "reveal_all"] = Query(
        default="strict",
        description=(
            "strict = mask all PII (default, benchmark/test safe); "
            "staff = mask redactedText while returning matchedText + label in redactions; "
            "reveal_all = keep Name, MRN/UHID, Phone, Email visible inline with action='kept' "
            "(government IDs Aadhaar/Emirates ID/Iqama stay force-redacted)"
        ),
    ),
    keep_types: Optional[list[str]] = Query(
        default=None,
        description="Optional entity types to keep visible (e.g., NAME, PHONE, MRN_UHID, EMAIL)",
    ),
    include_themes: bool = Query(
        default=False,
        description="Set true to also return complaint themes for each comment",
    ),
    _: None = Depends(verify_token),
):
    if len(items) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}.")

    redactor = request.app.state.redaction_engine
    theme_engine: ThemeEngine = request.app.state.theme_engine

    results: list[RedactionResponseItem] = []
    for item in items:
        raw_text = item.text or ""
        raw_spans = _extract_spans_from_redactor(redactor, item)
        formatted_text, formatted_dicts = apply_redaction_view(
            original_text=raw_text,
            spans=raw_spans,
            view=view,
            keep_types=keep_types,
        )
        formatted_entities = [RedactionEntity(**d) for d in formatted_dicts]

        preds = None
        if include_themes:
            preds, _ = theme_engine.predict_text(raw_text)

        if view == "strict" and not keep_types:
            results.append(
                RedactionResponseItem(
                    id=item.id,
                    redactedText=formatted_text,
                    entities=formatted_entities,
                    themes=preds,
                )
            )
        else:
            results.append(
                RedactionResponseItem(
                    id=item.id,
                    redactedText=formatted_text,
                    redactions=formatted_entities,
                    themes=preds,
                )
            )
    return results


@app.post(
    "/v1/comment-quality",
    response_model=list[QualityResponseItem],
)
def comment_quality(
    items: list[QualityRequestItem],
    request: Request,
    _: None = Depends(verify_token),
):
    if len(items) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}.")

    quality_engine = request.app.state.quality_engine
    lang_detector = request.app.state.lang_detector
    return [_evaluate_quality_item(quality_engine, lang_detector, item) for item in items]


@app.post(
    "/v1/anomalies",
    response_model=AnomalyResponse,
)
def detect_anomalies(
    points: list[AnomalyPointRequest],
    request: Request,
    _: None = Depends(verify_token),
):
    anomaly_engine = request.app.state.anomaly_engine
    res = anomaly_engine.detect(points)
    if isinstance(res, AnomalyResponse):
        return res
    if hasattr(res, "model_dump"):
        return AnomalyResponse(**res.model_dump())
    if isinstance(res, dict):
        return AnomalyResponse(**res)
    return res