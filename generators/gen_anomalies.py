import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def generate(out_path: Path, days: int = 180) -> None:
    rng = np.random.default_rng(606)
    start_date = date(2026, 5, 1)
    profiles = [
        ("dept-opd", "waiting time", 18.0),
        ("dept-billing", "billing and insurance", 12.0),
        ("dept-pharmacy", "pharmacy", 14.0),
        ("dept-lab", "reports and lab", 8.0),
        ("dept-ward", "nursing care", 3.0),
        ("dept-diet", "food", 1.8),
    ]
    spike_plan = {0: [42, 78, 119, 156], 1: [36, 91, 140, 170], 2: [48, 84, 132], 3: [55, 104, 162], 4: [60, 115, 150], 5: [50, 99, 145]}
    drop_plan = {0: [65], 1: [110], 2: [100], 3: [125], 4: [88]}
    rows: list[list] = []
    for idx, (dept, metric, mean) in enumerate(profiles):
        spikes = set(spike_plan.get(idx, []))
        drops = set(drop_plan.get(idx, []))
        for d in range(days):
            curr = start_date + timedelta(days=d)
            lam = mean * (1.0 + 0.08 * (d / days)) * (0.55 if curr.weekday() == 6 else 1.0)
            cnt = int(rng.poisson(lam))
            is_spike = 0
            if d in spikes:
                cnt = int(round(max(lam * float(rng.uniform(3.0, 4.5)), lam + 8)))
                is_spike = 1
            elif d in drops:
                cnt = max(0, int(round(lam * 0.15)))
            rows.append([curr.isoformat(), dept, metric, cnt, is_spike])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "departmentId", "metric", "count", "is_planted_spike"])
        w.writerows(rows)
    print(f"Generated {len(rows)} anomaly rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=BASE_DIR / "data" / "heldout" / "anomalies_test_1080.csv")
    args = p.parse_args()
    generate(args.out)
