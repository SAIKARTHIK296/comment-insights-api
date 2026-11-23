import math
import re
from collections import Counter
from typing import Callable
from app.schemas import QualityRequestItem, QualityResponseItem

FILLER_EXACT_RE = re.compile(
    r"^(?:ok|okay|k|kk|nil|na|n/a|none|nothing|no|yes|nope|yep|fine|good|test|testing|tested|"
    r"dummy|sample|abc|xyz|asdf|qwerty|lorem ipsum|no comments?|nothing much|same|same as above|"
    r"nil comment|na\.?|ok\.?)$",
    re.IGNORECASE,
)
REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
QWERTY_RUN_RE = re.compile(r"(?:asdf|sdfg|dfgh|fghj|ghjk|hjkl|qwer|wert|erty|rtyu|tyui|yuio|uiop|zxcv|xcvb|cvbn|vbnm)", re.IGNORECASE)
CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}", re.IGNORECASE)
LATIN_WORD_RE = re.compile(r"\b[a-zA-Z]{6,}\b")
LATIN_VOWELS = frozenset("aeiouyAEIOUY")


def _shannon_entropy(s: str) -> float:
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else 0.0


def _is_keyboard_mash(text: str) -> bool:
    if QWERTY_RUN_RE.search(text):
        return True
    latin_words = LATIN_WORD_RE.findall(text)
    if not latin_words:
        return False
    hits = sum(
        1 for w in latin_words
        if (sum(1 for ch in w if ch in LATIN_VOWELS) / len(w) < 0.12) or CONSONANT_RUN_RE.search(w)
    )
    if hits / len(latin_words) >= 0.5:
        return True
    stripped = re.sub(r"\s+", "", text.lower())
    if stripped.isascii() and stripped.isalpha() and len(stripped) >= 10:
        uniq_bi = len({stripped[i : i + 2] for i in range(len(stripped) - 1)})
        if uniq_bi / (len(stripped) - 1) > 0.95 and _shannon_entropy(stripped) > 3.8 and len(text.split()) <= 2:
            return True
    return False


class QualityEngine:
    def __init__(self, language_detector: Callable[[str | None], str] | None = None) -> None:
        self.language_detector = language_detector

    def score_text(self, text: str | None) -> tuple[bool, float, str]:
        if text is None or not str(text).strip():
            return False, 0.0, "missing_or_empty_text"
        raw = str(text).strip()
        compact = re.sub(r"\s+", "", raw)
        penalties: list[tuple[float, str]] = []
        if len(compact) < 3:
            penalties.append((0.7, "too_short"))
        if not any(ch.isalpha() for ch in raw):
            penalties.append((1.0, "numbers_only" if any(ch.isdigit() for ch in raw) else "lone_emoji_or_symbols"))
        normalized = re.sub(r"\s+", " ", raw).strip(" .!?,-_")
        if normalized and FILLER_EXACT_RE.fullmatch(normalized):
            penalties.append((0.9, "filler_or_test_text"))
        if REPEATED_CHAR_RE.search(compact) or (len(compact) >= 8 and len(set(compact.lower())) / len(compact) < 0.20):
            penalties.append((0.8, "repeated_characters"))
        if _is_keyboard_mash(raw):
            penalties.append((0.8, "keyboard_mashing"))
        score = max(0.0, round(1.0 - sum(p for p, _ in penalties), 2))
        if not penalties:
            return True, score, "meaningful_comment"
        top_reason = max(penalties, key=lambda item: item[0])[1]
        is_meaningful = score >= 0.50
        return is_meaningful, score, "meaningful_comment" if is_meaningful else top_reason

    def evaluate_batch(self, items: list[QualityRequestItem]) -> list[QualityResponseItem]:
        out: list[QualityResponseItem] = []
        for item in items:
            meaningful, score, reason = self.score_text(item.text)
            if reason in ("missing_or_empty_text", "lone_emoji_or_symbols", "numbers_only"):
                lang = "unknown"
            elif self.language_detector is not None:
                lang = self.language_detector(item.text)
            else:
                lang = "unknown"
            out.append(QualityResponseItem(id=item.id, isMeaningful=meaningful, qualityScore=score, reason=reason, language=lang))
        return out
