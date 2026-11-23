import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.redact import RedactionEngine


def _overlaps(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return not (a_e <= b_s or a_s >= b_e)


def _covered(gt_s: int, gt_e: int, preds: list[tuple[int, int]]) -> bool:
    cov: set[int] = set()
    for ps, pe in preds:
        if max(gt_s, ps) < min(gt_e, pe):
            cov.update(range(max(gt_s, ps), min(gt_e, pe)))
    return len(cov) == (gt_e - gt_s)


def evaluate(csv_path: Path) -> int:
    engine = RedactionEngine()
    tot_by, rem_by = defaultdict(int), defaultdict(int)
    ord_words = fa_words = rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            text = row.get("text") or ""
            gts = json.loads(row.get("spans_json") or "[]")
            _, pred_objs, _ = engine.redact_text(text)
            pranges = [(p.start, p.end) for p in pred_objs]
            for gt in gts:
                tot_by[gt["type"]] += 1
                if _covered(gt["start"], gt["end"], pranges):
                    rem_by[gt["type"]] += 1
            for tok in re.finditer(r"\w+", text):
                if not any(_overlaps(tok.start(), tok.end(), gt["start"], gt["end"]) for gt in gts):
                    ord_words += 1
                    if any(_overlaps(tok.start(), tok.end(), ps, pe) for ps, pe in pranges):
                        fa_words += 1
    tot = sum(tot_by.values())
    rem = sum(rem_by.values())
    recall = rem / tot * 100.0 if tot else 100.0
    fa_pct = fa_words / ord_words * 100.0 if ord_words else 0.0
    print(f"=== TASK 2 EVAL (Rows={rows} | Planted={tot} | OrdinaryWords={ord_words}) ===")
    for k in sorted(tot_by):
        print(f"  {k:<14} {rem_by[k]:>4}/{tot_by[k]:<4} ({rem_by[k] / tot_by[k] * 100.0:6.2f}%)")
    ok = recall >= 98.0 and fa_pct < 2.0
    print(f"Recall={recall:.2f}% (Target >= 98%) | FalseAlarm={fa_pct:.2f}% (Target < 2%) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "heldout" / "redact_test_500.csv")
    raise SystemExit(evaluate(p.parse_args().csv))
