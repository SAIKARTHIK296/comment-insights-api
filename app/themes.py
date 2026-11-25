import json
import os
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from app.schemas import ThemePrediction, ThemeRequestItem, ThemeResponseItem
from app.text_utils import MODELS_DIR, clean_and_expand_hospital_text, split_mixed_clauses

DEFAULT_THEMES_MODEL = MODELS_DIR / "themes_v1.0.0.pkl"


def resolve_themes_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    env_path = os.getenv("THEMES_MODEL_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_THEMES_MODEL


class ThemeEngine:
    def __init__(
        self,
        model_path: Path | str | None = None,
        language_detector: Callable[[str | None], str] | None = None,
    ) -> None:
        self.model_path = resolve_themes_model_path(model_path)
        self.language_detector = language_detector
        if not self.model_path.exists():
            raise RuntimeError(
                f"Required model file {self.model_path} not found. "
                "Run training/train_themes.py before starting the service."
            )
        artifact = joblib.load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.themes: list[str] = list(artifact["themes"])
        self.thresholds: dict[str, float] = dict(artifact["thresholds"])
        self.model_version: str = str(artifact["model_version"])
        meta_path = self.model_path.with_suffix(".json")
        self.metadata = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {"modelVersion": self.model_version}
        )

    def _detect_language(self, item: ThemeRequestItem) -> str:
        if item.text is None or not str(item.text).strip():
            return "unknown"
        if item.language and item.language.strip():
            return item.language.strip().lower()
        if self.language_detector is not None:
            return self.language_detector(item.text)
        return "unknown"

    def predict_text(self, text: str | None) -> tuple[list[ThemePrediction], str | None]:
        if text is None or not str(text).strip():
            return [], "missing_or_empty_text"

        raw_text = str(text).strip()
        clauses = split_mixed_clauses(raw_text)
        cleaned = [clean_and_expand_hospital_text(c) for c in clauses]
        cleaned = [c for c in cleaned if c]

        if not cleaned:
            # Keep valid non-Latin scripts (Malayalam, Hindi, Arabic) from being
            # mislabeled as missing/empty after ASCII text normalization.
            if any(ch.isalpha() for ch in raw_text):
                return [], None
            return [], "missing_or_empty_text"

        probs = self.pipeline.predict_proba(cleaned)
        max_probs = np.max(probs, axis=0)
        results: list[ThemePrediction] = []
        for idx, theme_name in enumerate(self.themes):
            score = float(max_probs[idx])
            if score >= float(self.thresholds.get(theme_name, 0.50)):
                results.append(ThemePrediction(name=theme_name, confidence=round(score, 4)))
        results.sort(key=lambda item: item.confidence, reverse=True)
        return results, None

    def predict_batch(self, items: list[ThemeRequestItem]) -> list[ThemeResponseItem]:
        responses: list[ThemeResponseItem] = []
        for item in items:
            themes, omitted_reason = self.predict_text(item.text)
            lang = self._detect_language(item)
            responses.append(
                ThemeResponseItem(
                    id=item.id,
                    themes=themes,
                    language=lang,
                    modelVersion=self.model_version,
                    omittedReason=omitted_reason,
                )
            )
        return responses