from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import ARABIC_BLOCK_RE, SPLITS
from .text import normalize_arabic_text, parse_segmented_tokens
from .utils import ensure_dir


def ensure_eval_guideline(exp_dir: Path) -> None:
    guideline = exp_dir / "eval_guideline.md"
    if guideline.exists():
        return
    text = (
        "# Evaluation Guideline\n\n"
        "- Scheme: clitic-style with `+` separators.\n"
        "- Normalization: remove diacritics/tatweel, normalize Alef forms, normalize `ى` to `ي`.\n"
        "- Compare systems on exactly the same normalized words.\n"
        "- Metrics: exact match, boundary precision/recall/F1, avg segments/word, OOV vs train vocab.\n"
    )
    guideline.write_text(text, encoding="utf-8")


def read_corpus_lines(corpus_path: Path, lines_limit: Optional[int]) -> List[str]:
    lines: List[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            if lines_limit is not None and len(lines) >= lines_limit:
                break
    return lines


def split_lines(lines: List[str], seed: int) -> Dict[str, List[str]]:
    import random

    idx = list(range(len(lines)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    shuffled = [lines[i] for i in idx]
    n = len(shuffled)
    if n < 3:
        return {"train": shuffled, "val": [], "test": []}

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    if n_val == 0:
        n_val = 1
    if n_test == 0:
        n_test = 1
    if n_train + n_val + n_test > n:
        n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n - n_val - n_test)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def prepare_split_data(
    corpus_path: Path,
    lines_limit: Optional[int],
    data_root: Path,
    seed: int,
) -> Tuple[Counter, Dict[str, Dict[str, List[str]]], Dict[str, int], Dict[str, Path]]:
    ensure_dir(data_root)
    split_dir = data_root / "split"
    processed_dir = data_root / "processed"
    ensure_dir(split_dir)
    ensure_dir(processed_dir)

    lines = read_corpus_lines(corpus_path, lines_limit=lines_limit)
    splits = split_lines(lines, seed=seed)

    refs_by_split: Dict[str, Dict[str, List[str]]] = {}
    train_counts: Counter = Counter()
    stats = {
        "raw_lines": len(lines),
        "train_lines": len(splits["train"]),
        "val_lines": len(splits["val"]),
        "test_lines": len(splits["test"]),
        "raw_plus_tokens": 0,
        "reference_pairs": 0,
    }
    split_files: Dict[str, Path] = {}

    for split_name in SPLITS:
        raw_out = split_dir / f"{split_name}.txt"
        proc_out = processed_dir / f"{split_name}.txt"
        split_files[f"raw_{split_name}"] = raw_out
        split_files[f"processed_{split_name}"] = proc_out

        df = pd.DataFrame({"raw": splits[split_name]})
        df["raw"] = df["raw"].astype(str)

        raw_lines = df["raw"].tolist()
        raw_text = "\n".join(raw_lines)
        raw_out.write_text((raw_text + "\n") if raw_text else "", encoding="utf-8")

        votes: Dict[str, Counter] = defaultdict(Counter)
        pair_lists = df["raw"].map(parse_segmented_tokens)
        stats["raw_plus_tokens"] += int(
            df["raw"].map(lambda x: sum(1 for t in x.split() if "+" in t)).sum()
        )
        stats["reference_pairs"] += int(pair_lists.map(len).sum())

        for pairs in pair_lists.tolist():
            for word, segs in pairs:
                votes[word]["+".join(segs)] += 1

        processed = df["raw"].str.replace("+", "", regex=False).map(normalize_arabic_text)
        processed = processed[processed != ""]
        proc_lines = processed.tolist()
        proc_text = "\n".join(proc_lines)
        proc_out.write_text((proc_text + "\n") if proc_text else "", encoding="utf-8")

        if split_name == "train" and not processed.empty:
            train_tokens = processed.str.split().explode()
            train_tokens = train_tokens[train_tokens.notna()]
            if not train_tokens.empty:
                token_counts = train_tokens.value_counts()
                train_counts.update({str(w): int(c) for w, c in token_counts.items()})

        refs_by_split[split_name] = {
            word: seg_vote.most_common(1)[0][0].split("+")
            for word, seg_vote in votes.items()
        }

    return train_counts, refs_by_split, stats, split_files


def write_word_frequency(path: Path, word_counts: Dict[str, int], min_freq: int) -> Dict[str, int]:
    df = pd.DataFrame(
        [{"word": str(w), "freq": int(c)} for w, c in word_counts.items()]
    )
    if df.empty:
        pd.DataFrame(columns=["word", "freq"]).to_csv(path, sep="\t", index=False)
        return {}

    df = df[
        (df["freq"] >= min_freq)
        & (df["word"].str.contains(ARABIC_BLOCK_RE, regex=True, na=False))
    ]
    df = df.sort_values(["freq", "word"], ascending=[False, True], kind="mergesort")
    df.to_csv(path, sep="\t", index=False)
    return {str(w): int(c) for w, c in zip(df["word"], df["freq"])}


def read_word_frequency(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype={"word": "string"}, keep_default_na=False)
    if "word" not in df.columns or "freq" not in df.columns:
        return {}
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df = df.dropna(subset=["word", "freq"])
    return {str(w): int(c) for w, c in zip(df["word"], df["freq"])}


def write_word_list(path: Path, words: Sequence[str]) -> None:
    pd.DataFrame({"word": list(words)}).to_csv(path, index=False, header=False)


def read_word_list(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, header=None, names=["word"], dtype="string", keep_default_na=False)
    return {str(w).strip() for w in df["word"].tolist() if str(w).strip()}


def save_eval_file(path: Path, words: Sequence[str], refs: Dict[str, List[str]]) -> None:
    df = pd.DataFrame({"word": list(words), "gold": ["+".join(refs[w]) for w in words]})
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False, encoding="utf-8")
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        return
    df.to_csv(path, sep=",", index=False, encoding="utf-8-sig")


def load_eval_file(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    if "word" not in df.columns or "gold" not in df.columns:
        return {}
    out: Dict[str, List[str]] = {}
    for row in df.itertuples(index=False):
        word = str(row.word).strip()
        if not word:
            continue
        seg = str(row.gold)
        out[word] = [s for s in seg.split("+") if s]
    return out

