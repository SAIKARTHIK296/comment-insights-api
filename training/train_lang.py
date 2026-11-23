import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.text_utils import MODELS_DIR


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
    ])
    clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42), method="sigmoid", cv=3)
    return Pipeline([("vec", features), ("clf", clf)])


def train(train_csv: Path, version: str) -> None:
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    latin = df[df["language"].isin(["en", "manglish"])].reset_index(drop=True)
    x = latin["text"].fillna("").astype(str).str.lower().to_numpy()
    y = latin["language"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.empty_like(y)
    tr_accs, va_accs = [], []
    for tr_i, va_i in skf.split(x, y):
        pipe = build_pipeline()
        pipe.fit(x[tr_i], y[tr_i])
        oof[va_i] = pipe.predict(x[va_i])
        tr_accs.append(accuracy_score(y[tr_i], pipe.predict(x[tr_i])))
        va_accs.append(accuracy_score(y[va_i], oof[va_i]))
    cv_acc = float(accuracy_score(y, oof))
    gap = round(float(np.mean(tr_accs) - np.mean(va_accs)), 4)
    print(f"=== TASK 5 CV (en vs manglish) | Acc={cv_acc:.4f} | Gap={gap:.4f} ===")
    print(confusion_matrix(y, oof, labels=["en", "manglish"]))
    final_pipe = build_pipeline()
    final_pipe.fit(x, y)
    model_version = f"lang_{version}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipe, "model_version": model_version, "classes": ["en", "manglish"]}, MODELS_DIR / f"{model_version}.pkl")
    (MODELS_DIR / f"{model_version}.json").write_text(
        json.dumps({"modelVersion": model_version, "trainedAtUtc": datetime.now(timezone.utc).isoformat(), "cvAccuracy": round(cv_acc, 4), "overfitGap": gap}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {MODELS_DIR / f'{model_version}.pkl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=BASE_DIR / "data" / "generated_train" / "language_train.csv")
    p.add_argument("--version", default="v1.0.0")
    args = p.parse_args()
    train(args.train_csv, args.version)
