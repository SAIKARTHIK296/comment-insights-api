import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.constants import THEMES, THEME_SET
from app.embedder import MultilingualEmbedder
from app.text_utils import MODELS_DIR, clean_and_expand_hospital_text

DEFAULT_TRAIN_CSV = BASE_DIR / "data" / "generated_train" / "themes_train.csv"
TFIDF_THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MULTILINGUAL_THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75]
UNICODE_TOKEN_PATTERN = r"[^\s.,!?;:\"'()\[\]{}।]+"

# Contrastive hard-negative praise examples across all 11 domains (zero test leakage)
HARD_NEGATIVE_PRAISE: list[tuple[str, str]] = [
    ("The doctor explained everything clearly and patiently.", ""),
    ("Doctor explained the diagnosis and treatment clearly.", ""),
    ("Doctor was very polite and explained the medicines clearly.", ""),
    ("Our doctor explained clearly and answered every question.", ""),
    ("Great doctor, listened carefully and explained every step well.", ""),
    ("Consultation was helpful, doctor explained the reports well.", ""),
    ("No waiting time at all, very quick consultation.", ""),
    ("Waiting time was minimal and smooth.", ""),
    ("Staff behaviour was polite, courteous, and helpful.", ""),
    ("Receptionist was very friendly and helpful.", ""),
    ("Nursing care was excellent, attentive, and compassionate.", ""),
    ("Nurses checked regularly and took great care.", ""),
    ("Billing and insurance clearance was fast and hassle-free.", ""),
    ("Room and bathroom were spotless, clean, and hygienic.", ""),
    ("Hospital food was warm, fresh, and tasty.", ""),
    ("Parking and hospital facilities were clean and convenient.", ""),
    ("Appointment scheduling was smooth and right on time.", ""),
    ("Lab reports were accurate and delivered promptly.", ""),
    ("Pharmacy counter was fast and all medicines were in stock.", ""),
]


def build_classifier(model_type: str) -> OneVsRestClassifier:
    if model_type == "multilingual":
        # No artificial class_weight inflation so unseen held-out comments stay high-precision
        return OneVsRestClassifier(
            LogisticRegression(
                C=3.0,
                class_weight=None,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )
        )
    return OneVsRestClassifier(
        LogisticRegression(
            C=5.0 if model_type == "upgrade" else 2.0,
            class_weight={0: 1.0, 1: 3.0},
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
    )


def build_tfidf_pipeline(model_type: str) -> Pipeline:
    if model_type == "baseline":
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            token_pattern=UNICODE_TOKEN_PATTERN,
        )
    else:
        vec = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        token_pattern=UNICODE_TOKEN_PATTERN,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    return Pipeline([("vec", vec), ("clf", build_classifier(model_type))])


def parse_labels(raw: str) -> list[str]:
    if pd.isna(raw) or not str(raw).strip():
        return []
    labels = [t.strip() for t in str(raw).split("|") if t.strip()]
    if any(t not in THEME_SET for t in labels):
        raise ValueError(f"Invalid theme label in {labels}")
    return sorted(set(labels))


def strat_keys(label_lists: list[list[str]], n_splits: int) -> list[str]:
    raw = ["__none__" if not lbls else "|".join(lbls) for lbls in label_lists]
    counts = Counter(raw)
    keys = [k if counts[k] >= n_splits else "__rare__" for k in raw]
    key_counts = Counter(keys)
    if key_counts.get("__rare__", 0) < n_splits and "__rare__" in key_counts:
        largest = max(
            (k for k in key_counts if k != "__rare__"),
            key=lambda k: key_counts[k],
            default="__rare__",
        )
        keys = [largest if k == "__rare__" else k for k in keys]
    return keys


def select_threshold(y_true: np.ndarray, probs: np.ndarray, model_type: str) -> float:
    grid = MULTILINGUAL_THRESHOLD_GRID if model_type == "multilingual" else TFIDF_THRESHOLD_GRID
    scored = []
    for th in grid:
        preds = (probs >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        scored.append((th, f1, prec))

    best_f1 = max(s[1] for s in scored)
    candidates = [s for s in scored if s[1] >= best_f1 - 0.01 and s[2] >= 0.95]
    if candidates:
        if model_type == "multilingual":
            return float(max(candidates, key=lambda s: s[0])[0])
        return float(min(candidates, key=lambda s: s[0])[0])
    return float(max(scored, key=lambda s: s[1])[0])


def train_and_export(train_csv: Path, model_type: str, version: str) -> None:
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {train_csv}\n"
            f"Run 'python generators/gen_themes.py' first, or pass --train-csv <path>."
        )

    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    if "text" not in df.columns or "themes" not in df.columns:
        raise ValueError(f"Expected columns ['text', 'themes'] in {train_csv}, got {list(df.columns)}")

    mlb = MultiLabelBinarizer(classes=list(THEMES))
    x_arr = np.array([clean_and_expand_hospital_text(t) for t in df["text"].fillna("")])
    label_lists = [parse_labels(v) for v in df["themes"]]
    y = mlb.fit_transform(label_lists)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(x_arr), len(THEMES)), dtype=float)
    tr_f1s, va_f1s = [], []

    embedder = MultilingualEmbedder() if model_type == "multilingual" else None
    if embedder is not None:
        print(f"Encoding {len(x_arr)} training rows once with {embedder.model_name} ...")
        x_emb = embedder.transform(x_arr)
        for tr_idx, va_idx in skf.split(x_arr, strat_keys(label_lists, 5)):
            clf = build_classifier("multilingual")
            clf.fit(x_emb[tr_idx], y[tr_idx])
            tr_p = clf.predict_proba(x_emb[tr_idx])
            va_p = clf.predict_proba(x_emb[va_idx])
            oof_probs[va_idx] = va_p
            tr_f1s.append(f1_score(y[tr_idx], (tr_p >= 0.5).astype(int), average="macro", zero_division=0))
            va_f1s.append(f1_score(y[va_idx], (va_p >= 0.5).astype(int), average="macro", zero_division=0))
    else:
        for tr_idx, va_idx in skf.split(x_arr, strat_keys(label_lists, 5)):
            pipe = build_tfidf_pipeline(model_type)
            pipe.fit(x_arr[tr_idx], y[tr_idx])
            tr_p = pipe.predict_proba(x_arr[tr_idx])
            va_p = pipe.predict_proba(x_arr[va_idx])
            oof_probs[va_idx] = va_p
            tr_f1s.append(f1_score(y[tr_idx], (tr_p >= 0.5).astype(int), average="macro", zero_division=0))
            va_f1s.append(f1_score(y[va_idx], (va_p >= 0.5).astype(int), average="macro", zero_division=0))

    thresholds: dict[str, float] = {}
    oof_preds = np.zeros_like(oof_probs, dtype=int)
    for idx, theme in enumerate(THEMES):
        best_th = select_threshold(y[:, idx], oof_probs[:, idx], model_type)
        thresholds[theme] = best_th
        oof_preds[:, idx] = (oof_probs[:, idx] >= best_th).astype(int)

    cv_macro = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    gap = round(float(np.mean(tr_f1s) - np.mean(va_f1s)), 4)
    print(f"=== TASK 1 CV ({model_type}) | Macro-F1={cv_macro:.4f} | OverfitGap={gap:.4f} ===")

    for idx, theme in enumerate(THEMES):
        p = precision_score(y[:, idx], oof_preds[:, idx], zero_division=0)
        r = recall_score(y[:, idx], oof_preds[:, idx], zero_division=0)
        f = f1_score(y[:, idx], oof_preds[:, idx], zero_division=0)
        cm = confusion_matrix(y[:, idx], oof_preds[:, idx], labels=[0, 1]).ravel().tolist()
        print(f"  {theme:<24} th={thresholds[theme]:.2f} P={p:.3f} R={r:.3f} F1={f:.3f} CM={cm}")

    praise_rows = HARD_NEGATIVE_PRAISE * 4
    praise_texts = np.array([clean_and_expand_hospital_text(t) for t, _ in praise_rows])
    praise_y = mlb.transform([parse_labels(lbl) for _, lbl in praise_rows])

    if embedder is not None:
        praise_emb = embedder.transform(praise_texts)
        final_clf = build_classifier("multilingual")
        final_clf.fit(np.vstack([x_emb, praise_emb]), np.vstack([y, praise_y]))
        final_pipe = Pipeline([("embed", embedder), ("clf", final_clf)])
    else:
        final_pipe = build_tfidf_pipeline(model_type)
        final_pipe.fit(np.concatenate([x_arr, praise_texts]), np.vstack([y, praise_y]))

    model_version = f"themes_{version}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": final_pipe,
            "themes": list(THEMES),
            "thresholds": thresholds,
            "model_version": model_version,
        },
        MODELS_DIR / f"{model_version}.pkl",
    )
    (MODELS_DIR / f"{model_version}.json").write_text(
        json.dumps(
            {
                "modelVersion": model_version,
                "modelType": model_type,
                "encoder": embedder.model_name if embedder is not None else "tfidf",
                "trainedAtUtc": datetime.now(timezone.utc).isoformat(),
                "cvMacroF1": round(cv_macro, 4),
                "overfitGap": gap,
                "thresholds": thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved -> {MODELS_DIR / f'{model_version}.pkl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--model-type", choices=["baseline", "upgrade", "multilingual"], default="multilingual")
    p.add_argument("--version", default="v1.0.0")
    args = p.parse_args()
    train_and_export(args.train_csv, args.model_type, args.version)