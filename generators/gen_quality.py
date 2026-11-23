import argparse
import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

JUNK = [
    ("", "missing_or_empty_text"), ("   ", "missing_or_empty_text"),
    ("👍", "lone_emoji_or_symbols"), ("👎👎", "lone_emoji_or_symbols"), ("!!!...", "lone_emoji_or_symbols"),
    ("987654", "numbers_only"), ("5/5", "numbers_only"), ("12345", "numbers_only"),
    ("ok", "filler_or_test_text"), ("okay", "filler_or_test_text"), ("nil", "filler_or_test_text"),
    ("testing", "filler_or_test_text"), ("dummy", "filler_or_test_text"), ("none", "filler_or_test_text"),
    ("bbbbbbb", "repeated_characters"), ("aaaaaaaa", "repeated_characters"), ("-------", "repeated_characters"),
    ("zxcvbnm", "keyboard_mashing"), ("asdfghjkl", "keyboard_mashing"), ("qpxztrm fghjkl", "keyboard_mashing"),
    ("hi", "too_short"), ("..", "too_short"),
]

MEANINGFUL = [
    ("Slow lab", "short_valid"), ("Great doctor", "short_valid"), ("Dirty toilet", "short_valid"),
    ("Ok consultation, however pharmacy queue was huge.", "ok_inside_sentence"),
    ("Nurses responded quickly to the bedside call bell.", "english_valid"),
    ("Basement parking was completely full at 9 AM.", "english_valid"),
    ("Doctor nalla care aanu, pakshe billing slow aayirunnu.", "manglish_valid"),
    ("ആശുപത്രി മുറി വളരെ വൃത്തിയുള്ളതായിരുന്നു.", "malayalam_valid"),
    ("नर्सिंग स्टाफ बहुत विनम्र और मददगार था।", "hindi_valid"),
    ("وقت الانتظار في العيادة كان طويلا جدا.", "arabic_valid"),
]


def generate(out_path: Path, count: int) -> None:
    rng = random.Random(404)
    half = count // 2
    rows = [(f"QJ{i + 1:03d}", JUNK[i % len(JUNK)][0], "false", JUNK[i % len(JUNK)][1]) for i in range(half)]
    rows += [(f"QM{i + 1:03d}", MEANINGFUL[i % len(MEANINGFUL)][0], "true", MEANINGFUL[i % len(MEANINGFUL)][1]) for i in range(count - half)]
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "is_meaningful", "category"])
        w.writerows(rows)
    print(f"Generated {len(rows)} quality rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=BASE_DIR / "data" / "heldout" / "quality_test_200.csv")
    p.add_argument("--count", type=int, default=200)
    args = p.parse_args()
    generate(args.out, args.count)
