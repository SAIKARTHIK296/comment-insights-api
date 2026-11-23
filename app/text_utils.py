import re
from pathlib import Path
import nltk

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EMOJI_MAP = {
    "😍": " emoji_love_excellent ",
    "😊": " emoji_happy_satisfied ",
    "🙂": " emoji_pleasant_good ",
    "👍": " emoji_thumbs_up_good ",
    "🙏": " emoji_grateful_thank_you ",
    "😐": " emoji_neutral_average ",
    "😕": " emoji_confused_unsure ",
    "😢": " emoji_crying_sad_pain ",
    "😭": " emoji_sobbing_very_sad ",
    "😡": " emoji_angry_furious_worst ",
    "🤬": " emoji_rude_unacceptable ",
    "👎": " emoji_thumbs_down_poor ",
    ":)": " emoji_happy_good ",
    ":(": " emoji_sad_dissatisfied ",
}

SLANG_ABBR_MAP = {
    r"\bgr8\b": "great",
    r"\bgud\b": "good",
    r"\bvry\b": "very",
    r"\bpls\b": "please",
    r"\bthx\b": "thanks",
    r"\bhrs\b": "hours",
    r"\bhr\b": "hour",
    r"\bmins\b": "minutes",
    r"\bopd\b": "outpatient department",
    r"\bappt\b": "appointment",
    r"\bdoc\b": "doctor",
    r"\bdr\b": "doctor",
    r"\brx\b": "prescription",
    r"\blab\b": "laboratory",
    r"\bdissatisifed\b": "dissatisfied",
    r"\brecption\b": "reception",
}

CONTRAST_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|\s+\b(?=(?:but|however|although|though|except|yet|whereas)\b)",
    flags=re.IGNORECASE,
)


def clean_and_expand_hospital_text(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    for emo, rep in EMOJI_MAP.items():
        s = s.replace(emo, rep)
    s = s.lower()
    for pat, rep in SLANG_ABBR_MAP.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_mixed_clauses(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    raw_s = str(text).strip()
    try:
        sentences = nltk.tokenize.sent_tokenize(raw_s)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", raw_s)
    clauses: list[str] = []
    for sent in sentences:
        parts = [p.strip() for p in CONTRAST_SPLIT_RE.split(sent) if p and len(p.strip()) >= 3]
        clauses.extend(parts)
    return clauses if clauses else [raw_s]
