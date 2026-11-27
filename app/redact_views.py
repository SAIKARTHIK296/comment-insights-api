from __future__ import annotations

from typing import Any, Iterable, Literal

ViewMode = Literal["strict", "staff", "reveal_all"]

ENTITY_LABELS: dict[str, str] = {
    "NAME": "Person Name",
    "PERSON": "Person Name",
    "PHONE": "Phone Number",
    "PHONE_NUMBER": "Phone Number",
    "EMAIL": "Email Address",
    "EMAIL_ADDRESS": "Email Address",
    "MRN_UHID": "Hospital MRN / UHID",
    "MRN": "Hospital MRN / UHID",
    "UHID": "Hospital MRN / UHID",
    "AADHAAR": "Aadhaar Number",
    "EMIRATES_ID": "Emirates ID",
    "IQAMA": "Iqama Number",
}

# National government IDs remain force-redacted for healthcare privacy compliance
ALWAYS_REDACT_TYPES: frozenset[str] = frozenset({"AADHAAR", "EMIRATES_ID", "IQAMA"})

# Operational hospital fields kept visible inline when view="reveal_all"
REVEAL_ALL_TYPES: frozenset[str] = frozenset(
    {"NAME", "PERSON", "MRN_UHID", "MRN", "UHID", "PHONE", "PHONE_NUMBER", "EMAIL", "EMAIL_ADDRESS"}
)


def _get_field(span: Any, key: str, default: Any = None) -> Any:
    if isinstance(span, dict):
        return span.get(key, default)
    return getattr(span, key, default)


def apply_redaction_view(
    original_text: str,
    spans: Iterable[Any],
    view: ViewMode = "strict",
    keep_types: Iterable[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Formats detected PII spans without touching detection regexes in app/redact.py.

    Modes:
      - "strict" (default): Masks all PII and hides matchedText (passes >=98% spec & tests).
      - "staff": Masks all PII in redactedText, exposes matchedText + label for staff.
      - "reveal_all": Keeps NAME, MRN_UHID, PHONE, EMAIL visible inline with action="kept"
                      and matchedText populated; keeps AADHAAR, EMIRATES_ID, IQAMA redacted.
    """
    text = original_text or ""
    requested_keep = {t.strip().upper() for t in (keep_types or []) if t and t.strip()}

    if view == "reveal_all":
        effective_keep = (REVEAL_ALL_TYPES | requested_keep) - ALWAYS_REDACT_TYPES
    else:
        effective_keep = requested_keep - ALWAYS_REDACT_TYPES

    normalized: list[dict[str, Any]] = []
    for sp in spans or []:
        etype = str(_get_field(sp, "type", "PII")).upper()
        start = int(_get_field(sp, "start", 0))
        end = int(_get_field(sp, "end", 0))
        if start < 0 or end <= start or end > len(text):
            continue
        existing_placeholder = _get_field(sp, "replacement", None)
        normalized.append(
            {
                "type": etype,
                "start": start,
                "end": end,
                "placeholder": existing_placeholder or f"[{etype}]",
            }
        )

    normalized.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

    parts: list[str] = []
    formatted_spans: list[dict[str, Any]] = []
    cursor = 0

    for sp in normalized:
        start, end, etype = sp["start"], sp["end"], sp["type"]
        if start < cursor:
            continue

        matched = text[start:end]
        label = ENTITY_LABELS.get(etype, etype.replace("_", " ").title())
        is_kept = etype in effective_keep
        replacement = matched if is_kept else sp["placeholder"]
        action = "kept" if is_kept else "redacted"

        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end

        entry: dict[str, Any] = {
            "type": etype,
            "label": label,
            "start": start,
            "end": end,
            "action": action,
        }
        if is_kept or view == "staff":
            entry["matchedText"] = matched
        entry["replacement"] = replacement

        formatted_spans.append(entry)

    parts.append(text[cursor:])
    return "".join(parts), formatted_spans