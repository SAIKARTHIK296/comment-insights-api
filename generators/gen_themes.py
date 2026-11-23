import argparse
import csv
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.constants import THEMES

TRAIN_BANK: dict[str, tuple[list[str], list[str], list[str]]] = {
    "waiting time": (
        ["Had to wait", "Kept waiting", "Stuck in queue", "Delayed for", "Waiting time was"],
        ["over 90 minutes", "almost two hours", "way too long", "nearly three hours", "over an hour"],
        ["outside OPD room", "at registration desk", "in the waiting lounge", "before token call", "at triage counter"],
    ),
    "staff behaviour": (
        ["Front desk employee", "Reception clerk", "Security guard", "Counter assistant", "Helpdesk staff"],
        ["spoke very rudely", "had an arrogant attitude", "shouted at patients", "ignored my questions", "acted hostile and impolite"],
        ["during check-in", "at the inquiry window", "while scrolling on phone", "in front of visitors", "with zero courtesy"],
    ),
    "doctor communication": (
        ["Consulting doctor", "Attending physician", "The specialist", "Senior consultant", "OPD doctor"],
        ["barely listened to symptoms", "rushed the checkup in two minutes", "never explained the diagnosis", "did not explain side effects", "dismissed my concerns"],
        ["during consultation", "without answering questions", "before giving prescription", "leaving me confused", "and hurried me out"],
    ),
    "nursing care": (
        ["Night-shift nurses", "Ward nurse", "Duty nursing staff", "Bedside nurse", "Inpatient nursing team"],
        ["ignored the call bell repeatedly", "handled the IV cannula roughly", "forgot scheduled painkiller", "took ages to check empty drip", "did not respond to pain"],
        ["throughout the overnight stay", "in the surgical ward", "while chatting at station", "during shift handover", "causing discomfort"],
    ),
    "billing and insurance": (
        ["Discharge billing counter", "Insurance TPA desk", "Accounts department", "Final hospital invoice", "Cashier counter"],
        ["added unexplained hidden charges", "delayed cashless approval for hours", "overcharged for consumables", "billed twice for same visit", "refused itemized breakdown"],
        ["at discharge time", "after submitting all documents", "during claim settlement", "with zero transparency", "causing financial stress"],
    ),
    "cleanliness": (
        ["Inpatient washroom", "Private ward room", "OPD corridor", "Bed linen provided", "Waiting hall restroom"],
        ["was filthy and smelled terrible", "had visible blood stains and dust", "was never mopped all day", "had overflowing waste bins", "had cockroaches near sink"],
        ["even after informing housekeeping", "next to patient bed", "during visiting hours", "raising infection worries", "with foul odor"],
    ),
    "food": (
        ["Patient lunch tray", "Hospital cafeteria meal", "Inpatient breakfast", "Evening soup and tea", "Prescribed diet meal"],
        ["was served ice cold and bland", "tasted stale and oily", "arrived two hours late", "ignored diabetic diet rules", "was watery and inedible"],
        ["for recovering inpatient", "in the dining canteen", "despite nutritionist notes", "with poorly washed utensils", "so patient ate nothing"],
    ),
    "facilities and parking": (
        ["Basement parking area", "Waiting room AC", "Patient elevator bank", "Entrance wheelchair bay", "Seating lounge upstairs"],
        ["had zero parking spots and chaotic valet", "was broken leaving hall hot", "had only one lift working", "had no wheelchairs available", "had broken chairs and no water"],
        ["during morning rush", "near emergency entrance", "for elderly patients", "on the third floor", "needing urgent repair"],
    ),
    "appointments": (
        ["Online appointment portal", "Booking helpline", "Scheduled follow-up slot", "Hospital app reservation", "Tele-booking desk"],
        ["cancelled my visit without SMS", "charged fee without giving slot", "double-booked the exact same time", "never answered phone calls", "rescheduled to wrong specialist"],
        ["after traveling across town", "despite confirmed booking", "creating check-in confusion", "forcing next-week rebooking", "wasting my morning"],
    ),
    "reports and lab": (
        ["Blood test report", "Diagnostic laboratory", "MRI scan summary", "Sample collection desk", "Pathology department"],
        ["was delayed by two full days", "was not uploaded to portal", "lost my urine sample", "mixed up my scan results", "pricked arm multiple times clumsily"],
        ["past promised delivery time", "holding up doctor review", "after paying urgent charges", "with careless tracking", "requiring repeat visit"],
    ),
    "pharmacy": (
        ["In-house pharmacy", "Ground-floor medicine counter", "Dispensing pharmacist", "Hospital medical store", "Prescription pickup window"],
        ["was out of stock for tablets", "handed over wrong dosage strip", "had one slow dispensing window", "did not explain insulin usage", "gave near-expiry syrup"],
        ["right after OPD visit", "for cardiac medicines", "leaving crowded line", "at discharge pickup", "forcing outside chemist search"],
    ),
}

HELDOUT_BANK: dict[str, list[str]] = {
    "waiting time": [
        "Sat outside the clinic for 110 minutes past my scheduled token.",
        "Queue did not budge for almost two hours during morning check-in.",
        "Waited endlessly in the outpatient hall with zero queue updates.",
        "Turnaround delay at registration took more than 75 mins.",
    ],
    "staff behaviour": [
        "Registration receptionist rolled her eyes and spoke harshly to us.",
        "Gate security was rude and argued when we asked for directions.",
        "Front desk clerk snapped impatiently at an elderly visitor.",
        "Helpdesk representative showed a dismissive attitude all morning.",
    ],
    "doctor communication": [
        "Physician rushed my consultation and never explained the diagnosis.",
        "Specialist barely listened and brushed aside my medication questions.",
        "Doctor did not explain possible side effects of the new tablets.",
        "Consultant gave a confusing treatment summary and hurried me out.",
    ],
    "nursing care": [
        "Pressed the bedside nurse call bell three times with no response.",
        "Ward nurse inserted the IV cannula roughly and left it bleeding.",
        "Night nursing team missed my scheduled midnight painkiller dose.",
        "Duty nurses ignored the empty saline drip until we called them.",
    ],
    "billing and insurance": [
        "Cashless insurance approval stalled our discharge for five hours.",
        "Final bill included unexplained charges and duplicate fees.",
        "Accounts clerk refused to explain the overpriced consumables invoice.",
        "TPA desk mishandled our policy paperwork and delayed settlement.",
    ],
    "cleanliness": [
        "Washroom near OPD was wet, smelly, and clearly uncleaned.",
        "Found stained bedsheets and dusty floors inside the private ward.",
        "Overflowing trash bins sat in the corridor all afternoon.",
        "Room floor stayed sticky and housekeeping never came to mop.",
    ],
    "food": [
        "Inpatient dinner tray arrived lukewarm and the soup tasted sour.",
        "Cafeteria served stale snacks and cold tea at high prices.",
        "Hospital kitchen sent sugary dessert to a diabetic patient.",
        "Breakfast delivery came ninety minutes late and was inedible.",
    ],
    "facilities and parking": [
        "Basement parking was packed and both lifts were out of service.",
        "Broken air conditioning made the waiting hall unbearable.",
        "No wheelchair was ready at the emergency gate for my father.",
        "Cramped waiting lounge had broken seating and empty water cooler.",
    ],
    "appointments": [
        "Booked a cardiology slot on the app yet reception lost my record.",
        "My follow-up appointment was cancelled without any SMS alert.",
        "Booking helpline rang nonstop and nobody answered to reschedule.",
        "Clinic overbooked the 10 AM appointment slot causing total chaos.",
    ],
    "reports and lab": [
        "Fasting blood report was still missing on the portal two days late.",
        "Pathology lab misplaced my sample and demanded a repeat draw.",
        "MRI report delivery was delayed and held up my review visit.",
        "Lab technician poked my vein three times during blood collection.",
    ],
    "pharmacy": [
        "Hospital pharmacy did not stock half of my prescribed tablets.",
        "Pharmacist handed over the wrong dosage strip at pickup.",
        "Single open pharmacy window created a massive medicine queue.",
        "Dispensing counter gave zero instructions on storing the insulin.",
    ],
}

ZERO_TRAIN = [
    "Doctor explained every test result patiently and clearly",
    "Waiting time in OPD was under ten minutes",
    "Reception and security staff were polite and helpful",
    "Ward nurses answered the call bell promptly and gently",
    "Billing and insurance discharge finished quickly with transparency",
    "Rooms and washrooms were spotless and fresh",
    "Inpatient food arrived hot and followed the diet plan",
    "Parking was organized and lifts and AC worked well",
    "Online appointment booking worked smoothly on time",
    "Blood test reports arrived on the portal the same evening",
    "Pharmacy had all medicines in stock with quick service",
    "Reception staff were not rude at all and helped politely",
    "Billing was not slow and had no hidden charges",
    "Washroom was not dirty and housekeeping cleaned regularly",
    "Routine annual checkup completed smoothly",
]

ZERO_HELDOUT = [
    "Front desk staff were not rude at all and got my file ready fast.",
    "Doctor was calm and clear, rooms were spotless, and pharmacy was quick.",
    "No waiting delay today and billing settlement was transparent.",
    "Nurses were not careless and monitored the drip gently.",
    "Completed my routine vaccination visit without any problem.",
]

NOISE = [("doctor", "dr"), ("hours", "hrs"), ("minutes", "mins"), ("reception", "recption"), ("pharmacy", "parmacist"), ("outpatient", "opd")]


def _add_noise(text: str, complaint: bool, rng: random.Random) -> str:
    if rng.random() > 0.18:
        return text
    out = text
    for old, new in NOISE:
        if old in out.lower() and rng.random() < 0.4:
            out = out.replace(old, new)
            break
    if rng.random() < 0.3:
        out += " " + (rng.choice(["😡", "👎"]) if complaint else rng.choice(["👍", "😊"]))
    return out


def generate_train(out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for theme in THEMES:
        openers, cores, places = TRAIN_BANK[theme]
        added = 0
        for _ in range(4000):
            if added >= 220:
                break
            tail = rng.choice(["", " yesterday", " this morning", " during our visit", " at the hospital"])
            text = _add_noise(f"{rng.choice(openers)} {rng.choice(cores)} {rng.choice(places)}{tail}.", True, rng)
            if text.lower() not in seen:
                seen.add(text.lower())
                rows.append((text, theme))
                added += 1
    pairs = [
        ("waiting time", "pharmacy"), ("waiting time", "billing and insurance"),
        ("staff behaviour", "billing and insurance"), ("waiting time", "appointments"),
        ("waiting time", "reports and lab"), ("cleanliness", "food"),
        ("cleanliness", "facilities and parking"), ("nursing care", "staff behaviour"),
        ("doctor communication", "waiting time"), ("pharmacy", "billing and insurance"),
        ("facilities and parking", "waiting time"),
    ]
    for t1, t2 in pairs:
        o1, c1, _ = TRAIN_BANK[t1]
        o2, c2, _ = TRAIN_BANK[t2]
        added = 0
        for _ in range(1000):
            if added >= 55:
                break
            text = _add_noise(f"{rng.choice(o1)} {rng.choice(c1)} and {rng.choice(o2).lower()} {rng.choice(c2)}.", True, rng)
            if text.lower() not in seen:
                seen.add(text.lower())
                rows.append((text, "|".join(sorted((t1, t2)))))
                added += 1
    prefixes = ["", "Overall, ", "Today, ", "Honestly, ", "Glad to say ", "During checkup, ", "Thankfully, ", "From start to finish, "]
    suffixes = [".", " throughout the visit.", " with no issues at all.", " which we appreciated.", " during the stay."]
    added_z = 0
    for _ in range(3000):
        if added_z >= 400:
            break
        base = rng.choice(ZERO_TRAIN)
        pre = rng.choice(prefixes)
        body = base[0].lower() + base[1:] if pre else base
        text = _add_noise(f"{pre}{body}{rng.choice(suffixes)}", False, rng)
        if text.lower() not in seen:
            seen.add(text.lower())
            rows.append((text, ""))
            added_z += 1
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "themes"])
        for i, (txt, lbl) in enumerate(rows, 1):
            w.writerow([f"TR{i:04d}", txt, lbl])
    print(f"Generated {len(rows)} train rows -> {out_path}")


def generate_heldout(out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    suffixes = ["", " Yesterday.", " This morning.", " On Monday.", " Very frustrating.", " Needs fixing."]
    for theme in THEMES:
        for i in range(20):
            base = HELDOUT_BANK[theme][i % len(HELDOUT_BANK[theme])]
            suf = suffixes[(i // len(HELDOUT_BANK[theme])) % len(suffixes)]
            rows.append((f"{base}{suf}", theme))
    for i in range(50):
        t1 = THEMES[i % len(THEMES)]
        t2 = THEMES[(i + 3) % len(THEMES)]
        if i % 2 == 0:
            praise = ZERO_HELDOUT[i % len(ZERO_HELDOUT)].rstrip(".")
            comp = HELDOUT_BANK[t1][i % len(HELDOUT_BANK[t1])]
            rows.append((f"{praise}, but {comp[0].lower()}{comp[1:]}", t1))
        else:
            c1 = HELDOUT_BANK[t1][i % len(HELDOUT_BANK[t1])].rstrip(".")
            c2 = HELDOUT_BANK[t2][i % len(HELDOUT_BANK[t2])]
            rows.append((f"{c1}, however {c2[0].lower()}{c2[1:]}", "|".join(sorted((t1, t2)))))
    for i in range(30):
        base = ZERO_HELDOUT[i % len(ZERO_HELDOUT)]
        suf = suffixes[(i // len(ZERO_HELDOUT)) % len(suffixes)]
        rows.append((f"{base}{suf}", ""))
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "themes"])
        for i, (txt, lbl) in enumerate(rows, 1):
            w.writerow([f"T{i:03d}", txt, lbl])
    print(f"Generated {len(rows)} heldout rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.split == "train":
        generate_train(args.out or BASE_DIR / "data" / "generated_train" / "themes_train.csv", args.seed)
    else:
        generate_heldout(args.out or BASE_DIR / "data" / "heldout" / "themes_test_300.csv", args.seed + 100)
