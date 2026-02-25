from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CORPUS = Path("/Users/I772971/Documents/Shared Task /babylm_corpus.txt")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_DIR = ROOT / "arabic_seg"

VARIANTS = ("type", "token", "frequency")
SPLITS = ("train", "val", "test")
EVAL_SPLITS = ("val", "test")
SUBWORD_BASELINES = ("bpe", "wordpiece")
SUBWORD_VOCAB_SIZES = (8000, 16000, 32000)

DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ALEF_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")
WHITESPACE_RE = re.compile(r"\s+")
ARABIC_BLOCK_RE = re.compile(r"[\u0600-\u06FF]")

SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]", "[TAB]"]
PUNCT_TOKENS = [".", ",", "!", "?", "،", "؛", "؟", ":", ";", "-", "_", "(", ")"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    lines_limit: Optional[int]
    min_freq: int
    keep_freq: int
    val_eval_words: int
    test_eval_words: int
    seed: int

