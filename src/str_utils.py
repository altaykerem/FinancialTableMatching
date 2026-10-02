import re

_tr_not_regex = r"[^a-zığüşöç 0-9]"
_tr_lower_table = str.maketrans("İI", "iı")


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
