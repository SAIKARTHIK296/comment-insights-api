from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import joblib

from app.schemas import ThemePrediction
from app.text_utils import MODELS_DIR, clean_and_expand_hospital_text

DEFAULT_MODEL_PATH = MODELS_DIR / "themes_v1.0.0.pkl"

# Detects Malayalam, Devanagari (Hindi), or Arabic scripts so native text is preserved
_NON_LATIN_RE = re.compile(r"[\u0900-\u097F\u0D00-\u0D7F\u0600-\u06FF]")

# Multilingual sentence & contrastive splitter (English, Malayalam, Manglish, Hindi, Arabic)
CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.!?;।؟\n]+|\s*,\s*|\b(?:but|however|although|though|yet|whereas|while|pakshe|pakshae)\b|പക്ഷേ|എന്നാൽ|लेकिन|परंतु|मगर|لكن|ولكن)",
    flags=re.IGNORECASE | re.UNICODE,
)

CONJ_SPLIT_RE = re.compile(
    r"(?:\b(?:and)\b|और|तथा|കൂടാതെ|\s+و\s+)",
    flags=re.IGNORECASE | re.UNICODE,
)

# Clauses that explicitly negate a complaint ("no waiting time", "not rude")
NEGATED_COMPLAINT_RE = re.compile(
    r"\b(?:no\s+(?:waiting|wait|delay|delays|issue|issues|problem|problems|complaint|complaints|trouble)|"
    r"zero\s+(?:waiting|wait|delay|issues|problems)|"
    r"without\s+(?:any\s+)?(?:waiting|wait|delay|delays|issue|issues|problem|problems|hassle)|"
    r"never\s+(?:had\s+to\s+wait|waited|rude|delayed)|"
    r"not\s+(?:rude|dirty|slow|bad|cold|delayed|crowded|unclean)|"
    r"nothing\s+(?:wrong|bad))\b",
    flags=re.IGNORECASE | re.UNICODE,
)

# Pure praise clauses (English + Manglish)
PURE_PRAISE_RE = re.compile(
    r"^\s*(?:doctor|doctors|staff|nurse|nurses|hospital|food|room|rooms|billing|pharmacy|service|waiting\s+time)?\s*"
    r"(?:was|were|is|are)?\s*"
    r"(?:very\s+|really\s+|extremely\s+|so\s+)?"
    r"(?:good|great|excellent|awesome|nice|polite|helpful|kind|friendly|clean|spotless|quick|fast|smooth|clear|clearly|"
    r"explained\s+(?:everything\s+)?(?:very\s+)?(?:clearly|well|nicely|patiently)|"
    r"nallathaanu|nallathu|nannayi)\b[^!\n]*$",
    flags=re.IGNORECASE | re.UNICODE,
)

NEGATIVE_CUE_RE = re.compile(
    r"\b(?:not|no|never|didn't|didnt|wasn't|wasnt|couldn't|couldnt|rude|slow|long|late|delay|delayed|"
    r"wait|waited|waiting|dirty|smell|smelly|cold|bad|poor|worst|horrible|terrible|broken|full|"
    r"cancel|cancelled|missing|out\s+of\s+stock|overcharged|confusing|unhelpful|ignored|rushed|"
    r"kooduthal|mosham|vrithiked|thamasam|illa|illayirunnu|paranjilla|kittilla|vannilla)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _split_clauses(text: str) -> list[str]:
    primary = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c and c.strip()]
    clauses: list[str] = []
    for part in primary:
        cleaned_part = re.sub(r"^(?:and|but)\s+", "", part, flags=re.IGNORECASE).strip()
        if cleaned_part:
            clauses.append(cleaned_part)
        sub = [s.strip() for s in CONJ_SPLIT_RE.split(cleaned_part) if s and len(s.strip()) >= 8]
        if len(sub) > 1:
            clauses.extend(sub)

    seen: set[str] = set()
    out: list[str] = []
    for c in clauses:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _is_non_complaint_clause(clause: str) -> bool:
    if NEGATED_COMPLAINT_RE.search(clause):
        return True
    if PURE_PRAISE_RE.search(clause) and not NEGATIVE_CUE_RE.search(clause):
        return True
    return False


class ThemeEngine:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        env_path = os.getenv("THEMES_MODEL_PATH")
        if model_path is not None:
            path = Path(model_path)
        elif env_path and env_path.strip():
            path = Path(env_path.strip())
        else:
            path = DEFAULT_MODEL_PATH

        if not path.exists():
            raise RuntimeError(f"Theme model artifact not found: {path}")

        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.themes: list[str] = list(artifact["themes"])
        self.thresholds: dict[str, float] = dict(artifact["thresholds"])
        self.model_version: str = str(artifact.get("model_version", "themes_v1.0.0"))

    def _score_texts(self, texts: list[str]) -> list[dict[str, float]]:
        safe_inputs: list[str] = []
        for raw_t in texts:
            if _NON_LATIN_RE.search(raw_t):
                safe_inputs.append(raw_t.strip())
            else:
                cleaned_t = clean_and_expand_hospital_text(raw_t)
                safe_inputs.append(cleaned_t if cleaned_t.strip() else raw_t.strip())

        probs_matrix = self.pipeline.predict_proba(safe_inputs)
        results: list[dict[str, float]] = []
        for row in probs_matrix:
            results.append({theme: float(row[idx]) for idx, theme in enumerate(self.themes)})
        return results

    def predict_text(self, text: Optional[str]) -> tuple[list[ThemePrediction], Optional[str]]:
        if text is None or not str(text).strip():
            return [], "missing_or_empty_text"

        raw = str(text).strip()
        if not re.search(r"[A-Za-z\u0900-\u097F\u0D00-\u0D7F\u0600-\u06FF]", raw):
            return [], None

        clauses = _split_clauses(raw)
        active_clauses = [c for c in clauses if not _is_non_complaint_clause(c)]
        if not active_clauses:
            return [], None

        score_rows = self._score_texts(active_clauses)

        best_per_theme: dict[str, float] = {}
        for scores in score_rows:
            for theme, prob in scores.items():
                th = float(self.thresholds.get(theme, 0.50))
                if prob >= th and prob > best_per_theme.get(theme, 0.0):
                    best_per_theme[theme] = prob

        preds = [
            ThemePrediction(name=theme, confidence=round(conf, 4))
            for theme, conf in sorted(best_per_theme.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return preds, None