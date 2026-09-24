"""Repair one JSON serialization error without changing the judge's numeric score."""
import re
from sft import style_grade as sg, language_grade as lg

# JSON permits escaped double quotes, but not escaped apostrophes. Preserve any
# paired backslashes and remove only the invalid odd escape before an apostrophe.
_APOSTROPHE = re.compile(r"(?<!\\)((?:\\\\)*)\\'")

def repair_apostrophes(raw):
    return _APOSTROPHE.sub(lambda m:m.group(1)+"'",raw)

def install():
    for module in (sg,lg):
        original=module._parse_score
        if getattr(original,'_icl_repaired',False):continue
        def parse(raw,_original=original):
            score,reason=_original(raw)
            if score is None and raw.strip().endswith('}'):
                score,reason=_original(repair_apostrophes(raw))
            return score,reason
        parse._icl_repaired=True
        module._parse_score=parse
