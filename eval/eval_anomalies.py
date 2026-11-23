import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.anomalies import AnomalyEngine
from app.schemas import AnomalyPointRequest


def evaluate(csv_path: Path) -> int:
    points, truth, non_spike_days = [], {}, defaultdict(int)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            dt = date.fromisoformat(row["date"].strip())
            dept, metric, cnt, sp = row["departmentId"].strip(), row["metric"].strip(), int(row["count"]), int(row["is_planted_spike"])
            points.append(AnomalyPointRequest(date=dt, departmentId=dept, metric=metric, count=cnt))
            truth[(dept, metric, dt)] = sp
            if sp == 0:
                non_spike_days[dept] += 1
    flagged = {(f.departmentId, f.metric, f.date) for f in AnomalyEngine().detect(points).flaggedDays}
    tp = sum(1 for k, v in truth.items() if v == 1 and k in flagged)
    fn = sum(1 for k, v in truth.items() if v == 1 and k not in flagged)
    fp = sum(1 for k, v in truth.items() if v == 0 and k in flagged)
    recall = tp / (tp + fn) * 100.0 if (tp + fn) else 100.0
    fa_per_90 = fp / (sum(non_spike_days.values()) / 90.0)
    ok = recall >= 90.0 and fa_per_90 <= 1.0
    print(f"=== TASK 4 EVAL | TP={tp} FN={fn} FP={fp} | Recall={recall:.2f}% (Target >= 90%) | FA/90d={fa_per_90:.2f} (Target <= 1.0) -> {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "heldout" / "anomalies_test_1080.csv")
    raise SystemExit(evaluate(p.parse_args().csv))
