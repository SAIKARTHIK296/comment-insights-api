import argparse
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.quality import QualityEngine


def evaluate(csv_path: Path) -> int:
    engine = QualityEngine()
    tp = fp = fn = tn = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            truth = row["is_meaningful"].strip().lower() == "true"
            pred, _, _ = engine.score_text(row.get("text"))
            if not truth and not pred:
                tp += 1
            elif not truth and pred:
                fn += 1
            elif truth and not pred:
                fp += 1
            else:
                tn += 1
    junk_rec = tp / (tp + fn) * 100.0 if (tp + fn) else 100.0
    fa_rate = fp / (fp + tn) * 100.0 if (fp + tn) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) * 100.0
    ok = junk_rec >= 95.0 and fa_rate < 3.0
    print(f"=== TASK 3 EVAL | Acc={acc:.2f}% | JunkRecall={junk_rec:.2f}% | FalseAlarm={fa_rate:.2f}% -> {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "heldout" / "quality_test_200.csv")
    raise SystemExit(evaluate(p.parse_args().csv))
