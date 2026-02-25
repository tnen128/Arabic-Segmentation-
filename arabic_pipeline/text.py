from __future__ import annotations

import unicodedata
from typing import List, Sequence, Tuple

from .config import ALEF_RE, ARABIC_BLOCK_RE, DIACRITICS_RE, WHITESPACE_RE


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def normalize_arabic_text(text: str, keep_plus: bool = False) -> str:
    text = ALEF_RE.sub("\u0627", text)
    text = text.replace("\u0649", "\u064A")
    text = text.replace("\u0640", "")
    text = DIACRITICS_RE.sub("", text)

    out: List[str] = []
    for ch in text:
        if ch == "+" and keep_plus:
            out.append(ch)
            continue
        if ch.isspace():
            out.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("N"):
            out.append(ch)
            continue
        if cat.startswith("L") and is_arabic_char(ch):
            out.append(ch)
            continue
        out.append(" ")
    return WHITESPACE_RE.sub(" ", "".join(out)).strip()


def normalize_word(token: str) -> str:
    return normalize_arabic_text(token, keep_plus=False).replace(" ", "")


def parse_segmented_tokens(raw_line: str) -> List[Tuple[str, List[str]]]:
    pairs: List[Tuple[str, List[str]]] = []
    for token in raw_line.split():
        if "+" not in token:
            continue
        token_plus = normalize_arabic_text(token, keep_plus=True)
        if "+" not in token_plus:
            continue
        segs = [normalize_word(x) for x in token_plus.split("+") if x.strip()]
        segs = [s for s in segs if s]
        if len(segs) < 2:
            continue
        word = "".join(segs)
        if len(word) < 2 or not ARABIC_BLOCK_RE.search(word):
            continue
        pairs.append((word, segs))
    return pairs


def split_word_to_boundaries(segments: Sequence[str]) -> set:
    boundaries = set()
    pos = 0
    for seg in segments[:-1]:
        pos += len(seg)
        boundaries.add(pos)
    return boundaries

