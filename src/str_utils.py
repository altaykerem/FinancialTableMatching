import re

_tr_not_regex = r"[^a-zığüşöç 0-9]"
_tr_lower_table = str.maketrans("İI", "iı")
_tr_months = [
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
]
_tr_date_regex = r"(\d{1,2}) (" + "|".join(_tr_months) + r")(?: (\d{4}))?"


def to_lower(s: str, lang="tr") -> str:
    if lang == "tr":
        return s.translate(_tr_lower_table).lower()
    return s.lower()


def remove_repeating_spaces(s):
    return re.sub(r" +", " ", s).strip()


def preprocess(s: str):
    s = to_lower(s)
    s = re.sub(_tr_not_regex, " ", s)
    s = remove_repeating_spaces(s)
    return s


def ocr_number_correction(x):
    if isinstance(x, str):
        x = x.replace("I", "1").replace("l", "1")
        x = x.replace("o", "0").replace("O", "0")
    return x


def parse_period(s):
    if s is None:
        return None
    s = preprocess(str(s))
    dates = re.findall(_tr_date_regex, s)
    years = re.findall(r"\b(?:19|20)\d{2}\b", s)
    if not years:
        return None

    year = int(years[-1])
    if not dates:
        return {"type": "year", "year": year}

    def to_iso(day, month, y):
        return f"{int(y or year):04d}-{_tr_months.index(month) + 1:02d}-{int(day):02d}"

    end = to_iso(*dates[-1])
    if len(dates) > 1:
        return {"type": "duration", "start": to_iso(*dates[0]), "end": end}
    return {"type": "date", "end": end}


def preprocess_cased_ner(s: str):
    """ÖZAK GAYRİMENKUL A.Ş. -> Özak Gayrimenkul A.Ş."""
    words = []
    for word in s.split():
        if "." in word or len(word) <= 1:
            words.append(word)
        else:
            words.append(word[0] + to_lower(word[1:]))
    return " ".join(words)
