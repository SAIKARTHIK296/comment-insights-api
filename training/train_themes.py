import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.constants import THEMES, THEME_SET
from app.text_utils import MODELS_DIR, clean_and_expand_hospital_text

DEFAULT_TRAIN_CSV = BASE_DIR / "data" / "generated_train" / "themes_train.csv"
THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def build_pipeline(model_type: str) -> Pipeline:
    if model_type == "baseline":
        return Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", OneVsRestClassifier(LogisticRegression(C=2.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42))),
        ])
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42), method="sigmoid", cv=3)
    return Pipeline([("vec", features), ("clf", OneVsRestClassifier(clf))])


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
    return [k if k != "__rare__" else "__none__" for k in keys] if Counter(keys).get("__rare__", 0) < n_splits else keys


def train_and_export(train_csv: Path, model_type: str, version: str) -> None:
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    x_arr = np.array([clean_and_expand_hospital_text(t) for t in df["text"].fillna("")])
    label_lists = [parse_labels(v) for v in df["themes"]]
    y = MultiLabelBinarizer(classes=list(THEMES)).fit_transform(label_lists)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(x_arr), len(THEMES)), dtype=float)
    tr_f1s, va_f1s = [], []
    for tr_idx, va_idx in skf.split(x_arr, strat_keys(label_lists, 5)):
        pipe = build_pipeline(model_type)
        pipe.fit(x_arr[tr_idx], y[tr_idx])
        tr_p = pipe.predict_proba(x_arr[tr_idx])
        va_p = pipe.predict_proba(x_arr[va_idx])
        oof_probs[va_idx] = va_p
        tr_f1s.append(f1_score(y[tr_idx], (tr_p >= 0.5).astype(int), average="macro", zero_division=0))
        va_f1s.append(f1_score(y[va_idx], (va_p >= 0.5).astype(int), average="macro", zero_division=0))
    thresholds: dict[str, float] = {}
    oof_preds = np.zeros_like(oof_probs, dtype=int)
    for idx, theme in enumerate(THEMES):
        best_th = max(THRESHOLD_GRID, key=lambda th: f1_score(y[:, idx], (oof_probs[:, idx] >= th).astype(int), zero_division=0))
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
    final_pipe = build_pipeline(model_type)
    final_pipe.fit(x_arr, y)
    model_version = f"themes_{version}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipe, "themes": list(THEMES), "thresholds": thresholds, "model_version": model_version}, MODELS_DIR / f"{model_version}.pkl")
    (MODELS_DIR / f"{model_version}.json").write_text(
        json.dumps({"modelVersion": model_version, "modelType": model_type, "trainedAtUtc": datetime.now(timezone.utc).isoformat(), "cvMacroF1": round(cv_macro, 4), "overfitGap": gap, "thresholds": thresholds}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {MODELS_DIR / f'{model_version}.pkl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--model-type", choices=["baseline", "upgrade"], default="upgrade")
    p.add_argument("--version", default="v1.0.0")
    args = p.parse_args()
    train_and_export(args.train_csv, args.model_type, args.version)
