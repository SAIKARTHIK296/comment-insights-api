# Comment Insights API

Stateless FastAPI microservice for analyzing patient feedback comments. Zero persistence; all ML models are loaded at startup from versioned `.pkl` artifacts in `models/`.

## Endpoints

| Method | Path | Auth | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Public | Service status and loaded model versions |
| `POST` | `/v1/themes` | Bearer | Multi-label complaint theme tagging (11 themes) + language detection |
| `POST` | `/v1/redact` | Bearer | PII redaction (names, India/UAE/Saudi phones, emails, Aadhaar, Emirates ID, Iqama, MRN/UHID) |
| `POST` | `/v1/comment-quality` | Bearer | Junk, filler, and keyboard-mash comment filter + language detection |
| `POST` | `/v1/anomalies` | Bearer | Daily department complaint spike detection |

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader punkt_tab
python -m spacy download en_core_web_sm