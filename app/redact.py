import re
from dataclasses import dataclass
import spacy
from app.schemas import RedactionRequestItem, RedactionResponseItem, RedactionSpan

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
EMIRATES_ID_RE = re.compile(r"\b784[\s-]?\d{4}[\s-]?\d{7}[\s-]?\d\b")
AADHAAR_CANDIDATE_RE = re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")
IQAMA_CANDIDATE_RE = re.compile(r"\b[12]\d{9}\b")
IQAMA_CONTEXT_RE = re.compile(r"\b(iqama|residency|resident|saudi\s*id|id\s*no|id\s*number)\b", re.IGNORECASE)
MRN_UHID_RE = re.compile(r"(?i)\b(?:mrn|uhid|pid|patient[\s_-]?id|ip[\s_-]?no|op[\s_-]?no)[\s:#-]*([a-z0-9-]{4,16})\b")

PHONE_PATTERNS = (
    re.compile(r"(?<!\d)(?:\+971|00971)[\s-]?5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+966|00966)[\s-]?5\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+91|0091)[\s-]?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    re.compile(r"(?<!\d)05\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
)

NAME_CUE_PATTERNS = (
    re.compile(
        r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Prof\.?|Nurse|Sister|Patient|Attender)\s+"
        r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b"
    ),
    re.compile(
        r"\b(?i:my name is|patient name is|patient name:|name:|attended by|spoke to|helped by|treated by|met with)\s+"
        r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b"
    ),
)

TYPE_PRIORITY = {
    "EMIRATES_ID": 7,
    "AADHAAR": 7,
    "IQAMA": 7,
    "MRN_UHID": 6,
    "EMAIL": 5,
    "PHONE": 4,
    "NAME": 3,
}

ALLOWLIST = frozenset({
    "opd", "ipd", "icu", "nicu", "picu", "mri", "ct", "ecg", "eeg", "cbc", "tpa", "ent", "er",
    "cardiology", "neurology", "orthopedics", "pediatrics", "gynecology", "oncology", "radiology",
    "urology", "dermatology", "pharmacy", "pathology", "physiotherapy", "diagnostics", "emergency",
    "reception", "billing", "insurance", "housekeeping", "security", "cafeteria", "canteen",
    "parking", "valet", "cashier", "hospital", "clinic", "ward", "room", "bed", "floor", "block",
    "doctor", "nurse", "sister", "staff", "patient", "attender", "specialist", "consultant",
    "surgeon", "pharmacist", "technician", "receptionist", "manager", "lab", "laboratory",
    "paracetamol", "insulin", "antibiotics", "tablets", "syrup", "capsule", "injection",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "aed", "sar", "inr", "rs", "usd", "mins", "hrs",
    "waited", "waiting", "blood", "scan", "test", "report", "reports", "service", "clean",
    "good", "great", "poor", "slow", "fast", "rude", "polite", "very", "smooth", "thanks",
})


@dataclass(frozen=True)
class _CandidateSpan:
    entity_type: str
    start: int
    end: int


def is_valid_verhoeff(digits: str) -> bool:
    if not digits.isdigit() or len(digits) != 12 or digits[0] in ("0", "1"):
        return False
    c = 0
    for idx, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[idx % 8][int(ch)]]
    return c == 0


def compute_verhoeff_check_digit(prefix_11: str) -> str:
    c = 0
    for idx, ch in enumerate(reversed(prefix_11)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(idx + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def _trim_name(text: str, start: int, end: int) -> tuple[int, int] | None:
    snippet = text[start:end].strip()
    if not snippet or any(ch.isdigit() for ch in snippet):
        return None
    words = snippet.split()
    while words and words[-1].strip(".,:;!?").lower() in ALLOWLIST:
        words.pop()
    while words and words[0].strip(".,:;!?").lower() in ALLOWLIST:
        words.pop(0)
    if not words or all(w.strip(".,:;!?").lower() in ALLOWLIST for w in words):
        return None
    trimmed = " ".join(words)
    rel_idx = text.find(trimmed, start, end)
    return (rel_idx, rel_idx + len(trimmed)) if rel_idx != -1 else None


class RedactionEngine:
    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        try:
            self.nlp = spacy.load(spacy_model, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
        except Exception as exc:
            raise RuntimeError(f"Required spaCy model '{spacy_model}' is not installed.") from exc

    def _collect(self, text: str) -> list[_CandidateSpan]:
        cands: list[_CandidateSpan] = []
        for m in EMAIL_RE.finditer(text):
            cands.append(_CandidateSpan("EMAIL", m.start(), m.end()))
        for m in EMIRATES_ID_RE.finditer(text):
            cands.append(_CandidateSpan("EMIRATES_ID", m.start(), m.end()))
        for m in AADHAAR_CANDIDATE_RE.finditer(text):
            if is_valid_verhoeff(re.sub(r"\D", "", m.group(0))):
                cands.append(_CandidateSpan("AADHAAR", m.start(), m.end()))
        for m in IQAMA_CANDIDATE_RE.finditer(text):
            if IQAMA_CONTEXT_RE.search(text[max(0, m.start() - 35) : min(len(text), m.end() + 20)]):
                cands.append(_CandidateSpan("IQAMA", m.start(), m.end()))
        for m in MRN_UHID_RE.finditer(text):
            if sum(ch.isdigit() for ch in m.group(1)) >= 3:
                cands.append(_CandidateSpan("MRN_UHID", m.start(), m.end()))
        for pat in PHONE_PATTERNS:
            for m in pat.finditer(text):
                cands.append(_CandidateSpan("PHONE", m.start(), m.end()))
        for pat in NAME_CUE_PATTERNS:
            for m in pat.finditer(text):
                trimmed = _trim_name(text, m.start(1), m.end(1))
                if trimmed:
                    cands.append(_CandidateSpan("NAME", trimmed[0], trimmed[1]))
        for ent in self.nlp(text).ents:
            if ent.label_ == "PERSON":
                trimmed = _trim_name(text, ent.start_char, ent.end_char)
                if trimmed:
                    cands.append(_CandidateSpan("NAME", trimmed[0], trimmed[1]))
        return cands

    def _resolve(self, cands: list[_CandidateSpan]) -> list[_CandidateSpan]:
        ordered = sorted(cands, key=lambda c: (-TYPE_PRIORITY[c.entity_type], -(c.end - c.start), c.start))
        accepted: list[_CandidateSpan] = []
        for cand in ordered:
            if not any(not (cand.end <= ex.start or cand.start >= ex.end) for ex in accepted):
                accepted.append(cand)
        return sorted(accepted, key=lambda c: c.start)

    def redact_text(self, text: str | None) -> tuple[str, list[RedactionSpan], str | None]:
        if text is None or not str(text).strip():
            return "", [], "missing_or_empty_text"
        raw = str(text)
        resolved = self._resolve(self._collect(raw))
        redacted = raw
        for span in sorted(resolved, key=lambda s: s.start, reverse=True):
            redacted = redacted[: span.start] + f"[{span.entity_type}]" + redacted[span.end :]
        spans = [
            RedactionSpan(type=s.entity_type, start=s.start, end=s.end, replacement=f"[{s.entity_type}]")
            for s in resolved
        ]
        return redacted, spans, None

    def redact_batch(self, items: list[RedactionRequestItem]) -> list[RedactionResponseItem]:
        out: list[RedactionResponseItem] = []
        for item in items:
            redacted_text, spans, omitted = self.redact_text(item.text)
            out.append(RedactionResponseItem(id=item.id, redactedText=redacted_text, redactions=spans, omittedReason=omitted))
        return out
