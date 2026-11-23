from datetime import date
from pydantic import BaseModel, Field


class ThemeRequestItem(BaseModel):
    id: str = Field(..., min_length=1)
    text: str | None = None
    language: str | None = None


class ThemePrediction(BaseModel):
    name: str
    confidence: float


class ThemeResponseItem(BaseModel):
    id: str
    themes: list[ThemePrediction]
    language: str
    modelVersion: str
    omittedReason: str | None = None


class RedactionRequestItem(BaseModel):
    id: str = Field(..., min_length=1)
    text: str | None = None


class RedactionSpan(BaseModel):
    type: str
    start: int
    end: int
    replacement: str


class RedactionResponseItem(BaseModel):
    id: str
    redactedText: str
    redactions: list[RedactionSpan]
    omittedReason: str | None = None


class QualityRequestItem(BaseModel):
    id: str = Field(..., min_length=1)
    text: str | None = None


class QualityResponseItem(BaseModel):
    id: str
    isMeaningful: bool
    qualityScore: float
    reason: str
    language: str


class AnomalyPointRequest(BaseModel):
    date: date
    departmentId: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class ExpectedRange(BaseModel):
    low: float
    high: float


class FlaggedDay(BaseModel):
    date: date
    departmentId: str
    metric: str
    actualCount: int
    expectedRange: ExpectedRange
    reason: str


class SkippedSeries(BaseModel):
    departmentId: str
    metric: str
    reason: str


class AnomalyResponse(BaseModel):
    flaggedDays: list[FlaggedDay]
    skippedSeries: list[SkippedSeries]
