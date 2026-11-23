import argparse
from pathlib import Path
import pandas as pd

HEADER_KEYWORDS = {"id", "comment", "comments", "feedback", "remarks", "text", "department", "dept", "date", "themes", "ward"}
FFILL_HINTS = ("department", "dept", "unit", "ward", "location", "category")


def detect_header_row(raw_df: pd.DataFrame, max_scan: int = 25) -> int:
    best_idx, best_score = 0, -1
    for idx in range(min(len(raw_df), max_scan)):
        vals = [str(v).strip().lower() for v in raw_df.iloc[idx].tolist() if pd.notna(v) and str(v).strip()]
        if not vals:
            continue
        score = sum(1 for v in vals if any(kw in v for kw in HEADER_KEYWORDS)) * 10 + len(vals)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def convert_excel_to_csv(input_path: Path, output_path: Path, sheet: str | int = 0) -> int:
    raw = pd.read_excel(input_path, sheet_name=sheet, header=None, engine="openpyxl")
    h_idx = detect_header_row(raw)
    headers = [str(v).strip() if pd.notna(v) and str(v).strip() else f"unnamed_{i}" for i, v in enumerate(raw.iloc[h_idx].tolist())]
    df = raw.iloc[h_idx + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all").reset_index(drop=True)
    for col in df.columns:
        if any(h in col.lower() for h in FFILL_HINTS):
            df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True).ffill()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return len(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sheet", default=0)
    args = p.parse_args()
    sh = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    print(f"Converted {convert_excel_to_csv(args.input, args.output, sh)} rows -> {args.output}")
