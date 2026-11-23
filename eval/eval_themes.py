import argparse
import csv
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.constants import THEMES, THEME_SET
from app.themes import ThemeEngine


def evaluate(csv_path: Path) -> int:
    engine = ThemeEngine()
    idx_map = {t: i for i, t in enumerate(THEMES)}
    yt_rows, yp_rows = [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            t_set = {t.strip() for t in (row.get("themes") or "").split("|") if t.strip()}
            if t_set - THEME_SET:
                raise ValueError(f"Unknown themes: {t_set - THEME_SET}")
            preds, _ = engine.predict_text(row.get("text"))
            p_set = {p.name for p in preds}
            tv, pv = [0] * len(THEMES), [0] * len(THEMES)
            for t in t_set:
                tv[idx_map[t]] = 1
            for t in p_set:
                pv[idx_map[t]] = 1
            yt_rows.append(tv)
            yp_rows.append(pv)
    yt, yp = np.array(yt_rows), np.array(yp_rows)
    print(f"=== TASK 1 EVAL ({engine.model_version} | Rows={len(yt)}) ===")
    for i, t in enumerate(THEMES):
        p = precision_score(yt[:, i], yp[:, i], zero_division=0)
        r = recall_score(yt[:, i], yp[:, i], zero_division=0)
        f = f1_score(yt[:, i], yp[:, i], zero_division=0)
        cm = confusion_matrix(yt[:, i], yp[:, i], labels=[0, 1]).ravel().tolist()
        print(f"  {t:<24} P={p:.3f} R={r:.3f} F1={f:.3f} CM={cm}")
    macro = float(f1_score(yt, yp, average="macro", zero_division=0))
    micro = float(f1_score(yt, yp, average="micro", zero_division=0))
    status = "PASS" if macro >= 0.75 else "FAIL"
    print(f"Micro-F1={micro:.3f} | Macro-F1={macro:.3f} (Target >= 0.750) -> {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BASE_DIR / "data" / "heldout" / "themes_test_300.csv")
    raise SystemExit(evaluate(p.parse_args().csv))
