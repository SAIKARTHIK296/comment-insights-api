import argparse
import csv
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.redact import compute_verhoeff_check_digit

DEV_NAMES = (
    "Arjun Nair", "Meera Menon", "Rahul Verma", "Sneha Pillai", "Vikram Rao",
    "Priya Sharma", "Karthik Iyer", "Ananya Das", "Rohan Kulkarni", "Divya Krishnan",
    "Tariq Mansoor", "Layla Hassan", "Omar Alharbi", "Nadia Qureshi", "Zaid Ibrahim",
)
HELDOUT_NAMES = (
    "Ramesh Varma", "Lakshmi Kurup", "Nikhil Hegde", "Aishwarya Shenoy", "Pranav Deshmukh",
    "Haritha Mathew", "Siddharth Bose", "Gayathri Subramanian", "Manoj Tiwari", "Pooja Reddy",
    "Farhan Alotaibi", "Amina Alzahrani", "Yusuf Darwish", "Huda Alghamdi", "Bilal Siddiqui",
)
HARD_NEGATIVES = {
    "dev": (
        "Waited 45 mins in Cardiology and paid AED 850 on 14/11/2026 in Room 402.",
        "Prescribed 500mg Paracetamol twice daily after the MRI scan in Block B.",
        "Discharge took 3 hrs and final bill was Rs 14200 at the cashier counter.",
        "Paid SAR 1250 for blood work and X-ray on 08/10/2026 at diagnostics desk.",
    ),
    "heldout": (
        "Spent 55 mins outside Neurology before being sent to Room 312 on 21/09/2026.",
        "Pharmacy charged AED 640 for 250mg antibiotics and insulin vials on Wednesday.",
        "Lab CBC and ECG reports were uploaded in 4 hrs with zero billing issues.",
        "Paid SAR 980 at the insurance TPA desk after a 90 mins wait on Tuesday.",
    ),
}


def make_verhoeff_id(rng: random.Random, spaced: bool) -> str:
    prefix = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    full = prefix + compute_verhoeff_check_digit(prefix)
    if spaced:
        sep = rng.choice([" ", "-"])
        return f"{full[:4]}{sep}{full[4:8]}{sep}{full[8:]}"
    return full


def make_emirates_id(rng: random.Random) -> str:
    sep = rng.choice(["-", " ", ""])
    return f"784{sep}{rng.randint(1960, 2005)}{sep}{''.join(str(rng.randint(0, 9)) for _ in range(7))}{sep}{rng.randint(0, 9)}"


def make_iqama(rng: random.Random) -> str:
    return str(rng.choice([1, 2])) + "".join(str(rng.randint(0, 9)) for _ in range(9))


def make_mrn(rng: random.Random) -> str:
    prefix = rng.choice(["MRN-", "UHID-", "PID:", "MRN "])
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(6))


def make_phone(rng: random.Random, region: str) -> str:
    if region == "IN":
        a = str(rng.randint(6, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(4))
        b = "".join(str(rng.randint(0, 9)) for _ in range(5))
        return rng.choice([f"+91 {a} {b}", f"+91-{a}-{b}", f"{a}{b}"])
    p1 = str(rng.randint(0, 9))
    p2 = "".join(str(rng.randint(0, 9)) for _ in range(3))
    p3 = "".join(str(rng.randint(0, 9)) for _ in range(4))
    code = "+971" if region == "UAE" else "+966"
    return rng.choice([f"{code} 5{p1} {p2} {p3}", f"05{p1}-{p2}-{p3}", f"{code}5{p1}{p2}{p3}"])


def make_email(name: str, domain: str, rng: random.Random) -> str:
    return f"{name.lower().replace(' ', '.')}{rng.randint(10, 99)}@{domain}"


def build_from_parts(parts: list[tuple[str, str | None]]) -> tuple[str, list[dict]]:
    chunks, spans, cursor = [], [], 0
    for chunk, etype in parts:
        start, end = cursor, cursor + len(chunk)
        chunks.append(chunk)
        if etype:
            spans.append({"start": start, "end": end, "type": etype})
        cursor = end
    return "".join(chunks), spans


def generate_split(split: str, out_path: Path, count: int) -> None:
    rng = random.Random(101 if split == "dev" else 2026)
    names = DEV_NAMES if split == "dev" else HELDOUT_NAMES
    domain = "example-dev.test" if split == "dev" else "example-eval.test"
    negs = HARD_NEGATIVES[split]
    planted_n = int(count * 400 / 550)
    cycle = ["NAME_CUED", "NAME_UNCUED", "PHONE_IN", "PHONE_UAE", "PHONE_SA", "EMAIL", "AADHAAR", "EMIRATES_ID", "IQAMA", "MRN_UHID"]
    rows: list[tuple[str, str, list[dict]]] = []
    for i in range(planted_n):
        kind = cycle[i % len(cycle)]
        name = rng.choice(names)
        if kind == "NAME_CUED":
            parts = [(f"{rng.choice(['Dr.', 'Patient', 'Nurse', 'Mr.', 'Ms.'])} ", None), (name, "NAME"), (" waited at OPD for 40 mins.", None)]
        elif kind == "NAME_UNCUED":
            parts = [(rng.choice(["Attended by ", "Spoke to ", "My name is ", "Patient name: "]), None), (name, "NAME"), (" during Cardiology visit.", None)]
        elif kind == "PHONE_IN":
            parts = [("Please call ", None), (make_phone(rng, "IN"), "PHONE"), (" for my lab summary.", None)]
        elif kind == "PHONE_UAE":
            parts = [("Reception took mobile ", None), (make_phone(rng, "UAE"), "PHONE"), (" without sending SMS.", None)]
        elif kind == "PHONE_SA":
            parts = [("Contact number ", None), (make_phone(rng, "SA"), "PHONE"), (" was written on the slip.", None)]
        elif kind == "EMAIL":
            parts = [("Send final invoice to ", None), (make_email(name, domain, rng), "EMAIL"), (" after approval.", None)]
        elif kind == "AADHAAR":
            parts = [("Submitted ID ", None), (make_verhoeff_id(rng, rng.choice([True, False])), "AADHAAR"), (" at admission desk.", None)]
        elif kind == "EMIRATES_ID":
            parts = [("Verified Emirates ID ", None), (make_emirates_id(rng), "EMIRATES_ID"), (" before scan.", None)]
        elif kind == "IQAMA":
            parts = [(rng.choice(["Checked Iqama ", "Residency ID ", "Iqama number "]), None), (make_iqama(rng), "IQAMA"), (" at front desk.", None)]
        else:
            parts = [("File ", None), (make_mrn(rng), "MRN_UHID"), (" was misplaced during handover.", None)]
        if rng.random() < 0.45:
            parts.extend([(" Ref ", None), (make_mrn(rng), "MRN_UHID"), (".", None)])
        text, spans = build_from_parts(parts)
        rows.append((f"R{i + 1:04d}", text, spans))
    for j in range(count - planted_n):
        rows.append((f"RN{j + 1:04d}", negs[j % len(negs)], []))
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "spans_json"])
        for rid, txt, sp in rows:
            w.writerow([rid, txt, json.dumps(sp)])
    print(f"Generated {len(rows)} {split} redact rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    p.add_argument("--out", type=Path, default=BASE_DIR / "data" / "heldout" / "redact_test_500.csv")
    p.add_argument("--count", type=int, default=550)
    args = p.parse_args()
    generate_split(args.split, args.out, args.count)
