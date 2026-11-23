import argparse
import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

POOLS = {
    "train": {
        "en": [
            "Waiting time outside the consultation room was over ninety minutes.",
            "The attending physician explained the treatment plan patiently.",
            "Discharge billing desk added hidden consumable charges to our receipt.",
            "Ward nurses responded promptly whenever we pressed the call bell.",
            "Hospital pharmacy was out of stock for standard cardiac tablets.",
            "Basement parking was crowded and only one elevator worked.",
            "Blood test reports arrived on the online portal the same evening.",
            "Washrooms near the outpatient corridor were clean and fresh.",
            "Reception clerk spoke politely during morning check-in.",
            "Inpatient breakfast arrived lukewarm and lacked flavor.",
            "Online booking portal failed to send an appointment SMS.",
            "The specialist rushed through the checkup without answering questions.",
            "Cashless insurance clearance took four hours at the accounts counter.",
            "Housekeeping sanitized the ward room quickly after we called.",
            "No wheelchairs were available near the main entrance gate.",
            "The pharmacist explained every insulin dosage step clearly.",
        ],
        "manglish": [
            "OPD waiting time valare kooduthal aayirunnu innu.",
            "Doctor nalla care aayirunnu pakshe billing slow aanu.",
            "Pharmacy counteril medicine stock illa ennu paranju.",
            "Nurse staff valare friendly aanu nalla service kitti.",
            "Lab report varan randu divasam delay aayi.",
            "Parking space kittan kure samayam wait cheyyendi vannu.",
            "Room clean aayirunnu pakshe washroomil nalla smell illa.",
            "Appointment book cheythittum token vilikkan valare late aayi.",
            "Receptionil ulla staff mosham aayi samsarichu.",
            "Food thanuthu poyirunnu taste onnum illa.",
            "Cashless insurance approve cheyyan orupadu samayam eduthu.",
            "Doctor ellam vyakthamayi paranju thannu valare santhosham.",
            "Lift onnum work cheyyunnilla valare budhimuttayi.",
            "Wardil ulla nurses call bell kettappol udane vannu.",
            "Discharge billil extra charge kooduthal aanu ennu thonni.",
            "Blood test cheyyan lab counteril valiya queue aayirunnu.",
        ],
        "ml": [
            "ഒപിഡിയിൽ രണ്ടു മണിക്കൂർ കാത്തിരിക്കേണ്ടി വന്നു.",
            "ഡോക്ടർ എല്ലാം വളരെ വ്യക്തമായി പറഞ്ഞു തന്നു.",
            "ബില്ലിംഗ് കൗണ്ടറിൽ വലിയ തിരക്കും താമസവും ഉണ്ടായിരുന്നു.",
            "നഴ്സുമാർ വളരെ സ്നേഹത്തോടെയും ശ്രദ്ധയോടെയും പരിചരിച്ചു.",
            "ഫാർമസിയിൽ മരുന്ന് സ്റ്റോക്ക് ഇല്ലായിരുന്നു.",
            "ലാബ് റിപ്പോർട്ട് കിട്ടാൻ രണ്ടു ദിവസം വൈകി.",
        ],
        "hi": [
            "ओपीडी में डॉक्टर को दिखाने के लिए दो घंटे इंतजार करना पड़ा।",
            "डॉक्टर ने बीमारी और दवा के बारे में अच्छे से समझाया।",
            "बिलिंग काउंटर पर डिस्चार्ज में बहुत समय लगा।",
            "नर्सिंग स्टाफ बहुत विनम्र और मददगार था।",
            "अस्पताल की फार्मेसी में जरूरी दवा उपलब्ध नहीं थी।",
            "ब्लड टेस्ट की रिपोर्ट दो दिन बाद मिली।",
        ],
        "ar": [
            "وقت الانتظار في العيادة الخارجية كان طويلا جدا.",
            "الطبيب شرح التشخيص وخطة العلاج بوضوح تام.",
            "إجراءات الفواتير والتأمين استغرقت ساعات طويلة.",
            "طاقم التمريض كان متعاونا ولطيفا للغاية.",
            "الدواء الموصوف لم يكن متوفرا في صيدلية المستشفى.",
            "تأخرت نتائج التحاليل المخبرية لمدة يومين.",
        ],
    },
    "heldout": {
        "en": [
            "Sat in the clinic hall for nearly two hours past my slot.",
            "My surgeon answered every concern calmly before surgery.",
            "Insurance pre-authorization stalled our evening checkout.",
            "Night nursing crew forgot my scheduled painkiller dose.",
            "Ground-floor chemist handed over the wrong dosage strip.",
            "Diagnostic X-ray summary was missing from the mobile app.",
            "Toilet near triage smelled awful and needed mopping.",
            "Cafeteria soup tasted stale and arrived cold.",
            "Front desk coordinator shouted at senior visitors.",
            "Wheelchairs were unavailable near the emergency gate.",
        ],
        "manglish": [
            "Token vilikkan oru manikkoor kooduthal edukkum ennu aarum paranjilla.",
            "Ward nurse call bell adichittum vannilla raathri.",
            "Billing deskil extra charge add cheythu ennu thonni.",
            "OPD doctor nannayi kelkkunnund pakshe queue bhayankara long aanu.",
            "Scan report upload cheythilla portalil ithuvare.",
            "Canteenil ninnum kittiya lunch theere nallathalla.",
            "Lift work cheyyunnilla elder patientsinu budhimuttayi.",
            "Hospital pharmacyil syrup illa purathu ninnum vangendi vannu.",
            "Security staff valare rude aayi behave cheythu gateil.",
            "Follow up slot appil book cheythu pakshe avide record illa.",
        ],
        "ml": [
            "അപ്പോയിന്റ്മെന്റ് എടുത്തിട്ടും ഒരുപാട് നേരം കാത്തുനിൽക്കേണ്ടി വന്നു.",
            "വാർഡിലെ നഴ്സിംഗ് സേവനം മികച്ചതായിരുന്നു.",
            "ഇൻഷുറൻസ് ക്ലിയറൻസിന് അഞ്ചു മണിക്കൂർ എടുത്തു.",
            "കാന്റീനിലെ ഭക്ഷണം തണുത്തതും രുചിയില്ലാത്തതുമായിരുന്നു.",
            "എക്സ്-റേ റിപ്പോർട്ട് പോർട്ടലിൽ അപ്‌ലോഡ് ചെയ്തില്ല.",
        ],
        "hi": [
            "अपॉइंटमेंट लेने के बाद भी बहुत देर लाइन में खड़ा रहना पड़ा।",
            "रिसेप्शन के कर्मचारी का व्यवहार अच्छा नहीं था।",
            "मरीज का खाना ठंडा और देर से आया।",
            "लैब रिपोर्ट समय पर मिल गई और स्टाफ अच्छा था।",
            "इंश्योरेंस अप्रूवल में पूरा दोपहर निकल गया।",
        ],
        "ar": [
            "موعد العيادة تأخر أكثر من ساعة دون إشعار.",
            "موظف الاستقبال كان غير متعاون أثناء التسجيل.",
            "طعام المرضى وصل باردا ومتأخرا عن موعده.",
            "المختبر سلم نتائج فحص الدم بسرعة ودقة.",
            "المكيف في صالة الانتظار لم يكن يعمل.",
        ],
    },
}

MODS = {
    "train": {
        "en": ["", " Yesterday morning.", " At the main branch.", " During follow-up.", " Please fix this."],
        "manglish": ["", " Innu ravile.", " Main blockil.", " Valare kashtam aayi.", " Nalla experience."],
        "ml": ["", " ഇന്ന് രാവിലെ.", " വളരെ ബുദ്ധിമുട്ടായി.", " നന്ദി."],
        "hi": ["", " आज सुबह।", " कृपया सुधार करें।", " धन्यवाद।"],
        "ar": ["", " صباح اليوم.", " يرجى تحسين الخدمة.", " شكرا لكم."],
    },
    "heldout": {
        "en": ["", " On Tuesday.", " Needs urgent attention.", " Really appreciated it."],
        "manglish": ["", " Innale evening.", " Oru update um illa.", " Thank you."],
        "ml": ["", " ഇന്നലെ വൈകുന്നേരം.", " പെട്ടെന്ന് പരിഹരിക്കണം."],
        "hi": ["", " कल शाम।", " अनुभव अच्छा रहा।"],
        "ar": ["", " مساء أمس.", " تجربة ممتازة."],
    },
}


def generate(split: str, out_path: Path, per_lang: int) -> None:
    rng = random.Random(707 if split == "train" else 808)
    rows: list[tuple[str, str, str]] = []
    idx = 1
    for lang in ("en", "ml", "manglish", "hi", "ar"):
        bases, mods = POOLS[split][lang], MODS[split][lang]
        for i in range(per_lang):
            rows.append((f"L{idx:04d}", f"{bases[i % len(bases)]}{mods[(i // len(bases)) % len(mods)]}".strip(), lang))
            idx += 1
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text", "language"])
        w.writerows(rows)
    print(f"Generated {len(rows)} {split} language rows -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    p.add_argument("--out", type=Path, default=BASE_DIR / "data" / "generated_train" / "language_train.csv")
    p.add_argument("--per-lang", type=int, default=500)
    args = p.parse_args()
    generate(args.split, args.out, args.per_lang)
