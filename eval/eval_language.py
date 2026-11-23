import argparse
import csv
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.language import LanguageDetector

LABELS = ["en", "ml", "manglish", "hi", "ar"]


def evaluate(csv_path: Path) -> int:
    det = LanguageDetector()
    yt, yp = [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            yt.append(row["language"].strip().lower())
            yp.append(det.detect(row.get("text")))
    acc = accuracy_score(yt, yp) * 100.0
    print(f"=== TASK 5 EVAL ({det.model_version} | Rows={len(yt)}) ===")
    print(classification_report(yt, yp, labels=LABELS, zero_division=0))
    status = "PASS" if acc >= 95.0 else "FAIL"
    print(f"Accuracy={acc:.2f}% (Target >= 95.00%) -> {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "heldout" / "language_test_250.csv")
    raise SystemExit(evaluate(p.parse_args().csv))
