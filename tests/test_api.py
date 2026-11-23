import random
from datetime import date, timedelta
import pytest
from fastapi.testclient import TestClient
from app.anomalies import AnomalyEngine
from app.language import LanguageDetector
from app.main import app
from app.quality import QualityEngine
from app.redact import RedactionEngine
from app.schemas import AnomalyPointRequest
from app.themes import ThemeEngine
from generators.gen_redact import DEV_NAMES, HELDOUT_NAMES, make_emirates_id, make_iqama, make_verhoeff_id


def test_startup_refuses_without_token(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="API_TOKEN"):
        with TestClient(app):
            pass


def test_startup_refuses_without_models(monkeypatch, tmp_path):
    monkeypatch.setenv("API_TOKEN", "secret-123")
    monkeypatch.setenv("THEMES_MODEL_PATH", str(tmp_path / "missing.pkl"))
    with pytest.raises(RuntimeError, match="not found"):
        with TestClient(app):
            pass


def test_health_and_auth_guards(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-123")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/themes", json=[{"id": "1", "text": "Long wait"}]).status_code == 401
        assert client.post("/v1/redact", json=[{"id": "1", "text": "hi"}]).status_code == 401
        assert client.post("/v1/comment-quality", json=[{"id": "1", "text": "ok"}]).status_code == 401
        assert client.post("/v1/anomalies", json=[]).status_code == 401


def test_themes_clause_splitting_negation_and_empty(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-123")
    engine = ThemeEngine()
    preds, _ = engine.predict_text("Doctor explained clearly, but waiting time at OPD was two hours.")
    names = {p.name for p in preds}
    assert "waiting time" in names and "doctor communication" not in names
    neg_preds, _ = engine.predict_text("Reception staff were not rude at all and helped politely.")
    assert neg_preds == []
    with TestClient(app) as client:
        res = client.post(
            "/v1/themes",
            headers={"Authorization": "Bearer secret-123"},
            json=[{"id": "1", "text": ""}, {"id": "2", "text": "OPD waiting time valare kooduthal aayirunnu."}],
        )
    assert res.status_code == 200
    assert res.json()[0]["omittedReason"] == "missing_or_empty_text"
    assert res.json()[1]["language"] == "manglish"


def test_redaction_entities_hard_negatives_and_disjoint_names():
    assert set(DEV_NAMES).isdisjoint(set(HELDOUT_NAMES))
    rng = random.Random(77)
    engine = RedactionEngine()
    syn_v = make_verhoeff_id(rng, True)
    syn_e = make_emirates_id(rng)
    syn_iq = make_iqama(rng)
    txt = f"Dr. Fatima called +91 98765 43210, +971 50 123 4567, test@example.com, UHID-55412, ID {syn_v}, Emirates ID {syn_e}, Iqama {syn_iq}."
    red, spans, _ = engine.redact_text(txt)
    assert "Dr. [NAME]" in red
    assert {"NAME", "PHONE", "EMAIL", "MRN_UHID", "AADHAAR", "EMIRATES_ID", "IQAMA"}.issubset({s.type for s in spans})
    clean_txt = "Waited 45 mins in Cardiology, paid AED 850 on 14/11/2026 in Room 402 for 500mg tablets."
    assert engine.redact_text(clean_txt)[0] == clean_txt


@pytest.mark.parametrize(
    ("text", "expected_ok"),
    [
        ("", False), ("👍", False), ("12345", False), ("ok", False), ("nil", False),
        ("aaaaaaaa", False), ("asdfghjkl", False), ("hi", False),
        ("Long wait", True), ("Ok service, but billing was slow.", True), ("ഡോക്ടർ നന്നായി സംസാരിച്ചു.", True),
    ],
)
def test_quality_rules(text, expected_ok):
    assert QualityEngine().score_text(text)[0] is expected_ok


def test_anomalies_spike_drop_and_short_series():
    start = date(2026, 9, 1)
    pts = [AnomalyPointRequest(date=start + timedelta(days=i), departmentId="billing", metric="billing", count=5 + (i % 2)) for i in range(30)]
    pts.append(AnomalyPointRequest(date=start + timedelta(days=30), departmentId="billing", metric="billing", count=22))
    pts.append(AnomalyPointRequest(date=start + timedelta(days=31), departmentId="billing", metric="billing", count=0))
    pts.extend([AnomalyPointRequest(date=start + timedelta(days=i), departmentId="tiny", metric="food", count=2) for i in range(5)])
    res = AnomalyEngine().detect(pts)
    flagged_dates = [f.date for f in res.flaggedDays]
    assert start + timedelta(days=30) in flagged_dates
    assert start + timedelta(days=31) not in flagged_dates
    assert len(res.skippedSeries) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Waiting time outside OPD was nearly two hours.", "en"),
        ("ഡോക്ടർ എല്ലാം വളരെ വ്യക്തമായി പറഞ്ഞു തന്നു.", "ml"),
        ("OPD waiting time valare kooduthal aayirunnu pakshe doctor nallathaanu.", "manglish"),
        ("बिलिंग काउंटर पर बहुत देर लगी।", "hi"),
        ("وقت الانتظار في العيادة كان طويلا جدا.", "ar"),
        ("👍", "unknown"),
    ],
)
def test_language_detection(text, expected):
    assert LanguageDetector().detect(text) == expected
