from datetime import date as DateType
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------
# Health Check
# -----------------------------
class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "ok"
    themeModelVersion: Optional[str] = None
    languageModelVersion: Optional[str] = None


# -----------------------------
# Task 1: Complaint Themes
# -----------------------------
class ThemeRequestItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    text: Optional[str] = None
    language: Optional[str] = None


class ThemePrediction(BaseModel):
    name: str
    confidence: float


class ThemeResponseItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    themes: list[ThemePrediction] = Field(default_factory=list)
    language: str = "unknown"
    modelVersion: str = "themes_v1.0.0"
    omittedReason: Optional[str] = None


# -----------------------------
# Task 2: Personal-Data Redaction
# -----------------------------
class RedactionRequestItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    text: Optional[str] = None


class RedactionEntity(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    label: Optional[str] = None
    start: int
    end: int
    action: Optional[Literal["redacted", "kept"]] = None
    matchedText: Optional[str] = None
    replacement: Optional[str] = None


# Backwards-compatible alias
RedactionSpan = RedactionEntity


class RedactionResponseItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    redactedText: str
    entities: Optional[list[RedactionEntity]] = None
    redactions: Optional[list[RedactionEntity]] = None
    themes: Optional[list[ThemePrediction]] = None


# -----------------------------
# Task 3: Comment Quality
# -----------------------------
class QualityRequestItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    text: Optional[str] = None
    language: Optional[str] = None


class QualityResponseItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    isMeaningful: bool
    qualityScore: float
    reason: Optional[str] = None
    language: str = "unknown"


# -----------------------------
# Task 4: Daily Anomaly Alerts
# -----------------------------
class AnomalyPointRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: DateType
    departmentId: str
    metric: str
    count: float


class ExpectedRange(BaseModel):
    model_config = ConfigDict(extra="allow")


class FlaggedDay(BaseModel):
    model_config = ConfigDict(extra="allow")


class SkippedSeries(BaseModel):
    model_config = ConfigDict(extra="allow")


class AnomalyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    flaggedDays: list[Any] = Field(default_factory=list)
    skippedSeries: list[Any] = Field(default_factory=list)