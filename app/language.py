import json
import os
from pathlib import Path
import joblib
from app.text_utils import MODELS_DIR

DEFAULT_LANG_MODEL = MODELS_DIR / "lang_v1.0.0.pkl"


def resolve_lang_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    env_path = os.getenv("LANG_MODEL_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_LANG_MODEL


def _detect_non_latin_script(text: str) -> str | None:
    ml = hi = ar = total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        code = ord(ch)
        if 0x0D00 <= code <= 0x0D7F:
            ml += 1
        elif 0x0900 <= code <= 0x097F:
            hi += 1
        elif 0x0600 <= code <= 0x06FF:
            ar += 1
    if total == 0:
        return "unknown"
    top_lang, top_count = max([("ml", ml), ("hi", hi), ("ar", ar)], key=lambda item: item[1])
    if top_count / total >= 0.30:
        return top_lang
    return None


class LanguageDetector:
    def __init__(self, model_path: Path | str | None = None, min_confidence: float = 0.55) -> None:
        self.model_path = resolve_lang_model_path(model_path)
        self.min_confidence = min_confidence
        if not self.model_path.exists():
            raise RuntimeError(
                f"Required language model file {self.model_path} not found. "
                "Run training/train_lang.py before starting the service."
            )
        artifact = joblib.load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.model_version = str(artifact["model_version"])
        meta_path = self.model_path.with_suffix(".json")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"modelVersion": self.model_version}

    def detect(self, text: str | None) -> str:
        if text is None or not str(text).strip():
            return "unknown"
        raw = str(text).strip()
        script_lang = _detect_non_latin_script(raw)
        if script_lang is not None:
            return script_lang
        probs = self.pipeline.predict_proba([raw.lower()])[0]
        best_idx = int(probs.argmax())
        if float(probs[best_idx]) < self.min_confidence:
            return "unknown"
        return str(self.pipeline.classes_[best_idx])
