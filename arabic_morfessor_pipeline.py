#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import morfessor
import pandas as pd


DEFAULT_CORPUS = Path("/Users/I772971/Documents/Shared Task /babylm_corpus.txt")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_DIR = ROOT / "arabic_seg"
VARIANTS = ("type", "token", "frequency")

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
    dev_eval_words: int
    test_eval_words: int
    seed: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def metrics_from_sequences(preds: Sequence[List[str]], golds: Sequence[List[str]]) -> Dict[str, float]:
    n = len(golds)
    if n == 0:
        return {
            "num_words": 0,
            "exact_match": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "avg_segments_per_word": 0.0,
        }

    exact = 0
    tp = 0
    fp = 0
    fn = 0
    seg_sum = 0

    for p, g in zip(preds, golds):
        seg_sum += len(p)
        if p == g:
            exact += 1
        pb = split_word_to_boundaries(p)
        gb = split_word_to_boundaries(g)
        tp += len(pb & gb)
        fp += len(pb - gb)
        fn += len(gb - pb)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "num_words": n,
        "exact_match": exact / n,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "avg_segments_per_word": seg_sum / n,
    }


def metric_sanity_check() -> Dict[str, object]:
    gold = [
        ["ab", "cd", "ef"],
        ["mn", "op"],
        ["k", "lmn"],
        ["qrst"],
        ["aa", "bb", "cc"],
        ["xx", "yy"],
        ["abc", "de"],
        ["gh", "ij", "kl"],
        ["mnop"],
        ["uv", "wx", "yz"],
    ]
    pred = [
        ["ab", "c", "def"],
        ["mnop"],
        ["k", "lmn"],
        ["qrst"],
        ["aabb", "cc"],
        ["x", "xyy"],
        ["abcde"],
        ["gh", "ijkl"],
        ["mnop"],
        ["uv", "w", "xyz"],
    ]
    metrics = metrics_from_sequences(pred, gold)
    expected_precision = 5 / 8
    expected_recall = 5 / 12
    expected_f1 = 0.5
    ok = (
        math.isclose(metrics["boundary_precision"], expected_precision, rel_tol=1e-9)
        and math.isclose(metrics["boundary_recall"], expected_recall, rel_tol=1e-9)
        and math.isclose(metrics["boundary_f1"], expected_f1, rel_tol=1e-9)
    )
    return {
        "pass": ok,
        "expected_precision": expected_precision,
        "expected_recall": expected_recall,
        "expected_f1": expected_f1,
        "computed": metrics,
    }


def select_words(words: Sequence[str], n: int, seed: int) -> List[str]:
    if n <= 0:
        return []
    arr = list(words)
    rng = random.Random(seed)
    rng.shuffle(arr)
    return arr[: min(n, len(arr))]


def stable_eval_selection_check(dev_words: Sequence[str], test_words: Sequence[str], seed: int) -> Dict[str, object]:
    dev_a = select_words(dev_words, len(dev_words), seed)
    dev_b = select_words(dev_words, len(dev_words), seed)
    test_a = select_words(test_words, len(test_words), seed)
    test_b = select_words(test_words, len(test_words), seed)
    return {
        "pass": dev_a == dev_b and test_a == test_b,
        "dev_size": len(dev_a),
        "test_size": len(test_a),
    }


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bootstrap_ci_diff(
    preds_a: Dict[str, List[str]],
    preds_b: Dict[str, List[str]],
    refs: Dict[str, List[str]],
    metric_name: str,
    seed: int,
    samples: int = 400,
) -> Dict[str, object]:
    words = list(refs.keys())
    if not words:
        return {"observed": 0.0, "ci95": [0.0, 0.0], "significant": False}

    gold = [refs[w] for w in words]
    a = [preds_a[w] for w in words]
    b = [preds_b[w] for w in words]
    observed = metrics_from_sequences(a, gold)[metric_name] - metrics_from_sequences(b, gold)[metric_name]

    rng = random.Random(seed)
    n = len(words)
    diffs: List[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        a_s = [a[i] for i in idx]
        b_s = [b[i] for i in idx]
        g_s = [gold[i] for i in idx]
        diff = metrics_from_sequences(a_s, g_s)[metric_name] - metrics_from_sequences(b_s, g_s)[metric_name]
        diffs.append(diff)

    diffs.sort()
    low = percentile(diffs, 0.025)
    high = percentile(diffs, 0.975)
    return {
        "observed": observed,
        "ci95": [low, high],
        "significant": not (low <= 0.0 <= high),
    }


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
    idx = list(range(len(lines)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    shuffled = [lines[i] for i in idx]
    n = len(shuffled)
    if n < 3:
        return {"train": shuffled, "dev": [], "test": []}

    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    n_test = n - n_train - n_dev

    if n_dev == 0:
        n_dev = 1
    if n_test == 0:
        n_test = 1
    if n_train + n_dev + n_test > n:
        n_train = n - n_dev - n_test
    if n_train <= 0:
        n_train = max(1, n - n_dev - n_test)

    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev : n_train + n_dev + n_test]
    return {"train": train, "dev": dev, "test": test}


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
        "dev_lines": len(splits["dev"]),
        "test_lines": len(splits["test"]),
        "raw_plus_tokens": 0,
        "reference_pairs": 0,
    }
    split_files: Dict[str, Path] = {}

    for split_name in ("train", "dev", "test"):
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
        # Internal evaluation uses TSV; keep it UTF-8.
        df.to_csv(path, sep="\t", index=False, encoding="utf-8")
        # Also emit Excel-friendly CSV with BOM.
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        return
    # CSV output: use BOM so Excel detects UTF-8 Arabic correctly.
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


def write_training_file(path: Path, train_counts: Dict[str, int], variant: str, keep_words: set) -> Dict[str, int]:
    if variant == "frequency":
        source = {w: c for w, c in train_counts.items() if w not in keep_words}
    else:
        source = dict(train_counts)

    with path.open("w", encoding="utf-8") as f:
        for word, count in sorted(source.items(), key=lambda x: (-x[1], x[0])):
            if variant == "type" or variant == "frequency":
                f.write(f"1 {word}\n")
            else:
                f.write(f"{count} {word}\n")
    return source


def token_count_modifier(x: int) -> int:
    return max(1, int(round(math.log(x + 1, 30))))


def train_variant_model(train_file: Path, model_file: Path, variant: str) -> morfessor.BaselineModel:
    io = morfessor.MorfessorIO()
    data = io.read_corpus_list_file(str(train_file))
    model = morfessor.BaselineModel()
    if variant == "token":
        model.load_data(data, count_modifier=token_count_modifier)
    else:
        model.load_data(data, count_modifier=lambda _: 1)
    model.train_batch()
    io.write_binary_model_file(str(model_file), model)
    return model


def load_model(model_file: Path) -> morfessor.BaselineModel:
    io = morfessor.MorfessorIO()
    return io.read_binary_model_file(str(model_file))


def predict_word(variant: str, model: morfessor.BaselineModel, word: str, keep_words: set) -> List[str]:
    if variant == "frequency" and word in keep_words:
        return [word]
    segs, _ = model.viterbi_segment(word)
    if not segs:
        return [word]
    return list(segs)


def predict_words(
    variant: str,
    model: morfessor.BaselineModel,
    words: Sequence[str],
    keep_words: set,
) -> Dict[str, List[str]]:
    return {w: predict_word(variant, model, w, keep_words) for w in words}


def evaluate_predictions(
    preds: Dict[str, List[str]],
    refs: Dict[str, List[str]],
    train_vocab: set,
) -> Dict[str, float]:
    words = list(refs.keys())
    pred_list = [preds[w] for w in words]
    gold_list = [refs[w] for w in words]
    metrics = metrics_from_sequences(pred_list, gold_list)
    oov = sum(1 for w in words if w not in train_vocab)
    metrics["oov_rate"] = oov / metrics["num_words"] if metrics["num_words"] else 0.0
    return metrics


def build_vocab_file(
    vocab_file: Path,
    variant: str,
    model: morfessor.BaselineModel,
    words: Sequence[str],
    keep_words: set,
) -> None:
    vocab = list(SPECIAL_TOKENS)
    for token in PUNCT_TOKENS:
        if token not in vocab:
            vocab.append(token)
    for w in words:
        for seg in predict_word(variant, model, w, keep_words):
            if seg not in vocab:
                vocab.append(seg)
    with vocab_file.open("w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{idx} {token}\n")


def token_stats(
    variant: str,
    model: morfessor.BaselineModel,
    train_counts: Dict[str, int],
    keep_words: set,
) -> Dict[str, float]:
    counter: Counter = Counter()
    weighted_seg_sum = 0
    word_sum = 0
    for w, c in train_counts.items():
        segs = predict_word(variant, model, w, keep_words)
        for s in segs:
            counter[s] += c
        weighted_seg_sum += len(segs) * c
        word_sum += c
    total = sum(counter.values())
    if total == 0:
        return {"vocab_size": 0, "entropy": 0.0, "top10_share": 0.0, "avg_segments_per_word": 0.0}
    probs = [v / total for v in counter.values() if v > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    top10 = sum(v for _, v in counter.most_common(10))
    return {
        "vocab_size": float(len(counter)),
        "entropy": entropy,
        "top10_share": top10 / total,
        "avg_segments_per_word": (weighted_seg_sum / word_sum) if word_sum else 0.0,
    }


def run_variants(
    cfg: RunConfig,
    model_root: Path,
    candidates: Dict[str, int],
    train_vocab: set,
    dev_refs: Dict[str, List[str]],
    test_refs: Dict[str, List[str]],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, Dict[str, List[str]]]]]:
    ensure_dir(model_root)
    keep_words = {w for w, c in candidates.items() if c >= cfg.keep_freq}

    results: Dict[str, object] = {}
    preds_all: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for variant in VARIANTS:
        v_dir = model_root / variant
        ensure_dir(v_dir)
        train_file = v_dir / "train.txt"
        model_file = v_dir / "morfessor.bin"
        vocab_file = v_dir / "vocab.txt"
        keep_file = v_dir / "keep_words.txt"

        used_counts = write_training_file(train_file, candidates, variant, keep_words)
        model = train_variant_model(train_file, model_file, variant)
        build_vocab_file(vocab_file, variant, model, candidates.keys(), keep_words)

        if variant == "frequency":
            with keep_file.open("w", encoding="utf-8") as f:
                for w in sorted(keep_words):
                    f.write(w + "\n")
        else:
            keep_file.write_text("", encoding="utf-8")

        dev_preds = predict_words(variant, model, dev_refs.keys(), keep_words)
        test_preds = predict_words(variant, model, test_refs.keys(), keep_words)

        results[variant] = {
            "dev": evaluate_predictions(dev_preds, dev_refs, train_vocab),
            "test": evaluate_predictions(test_preds, test_refs, train_vocab),
            "token_stats": token_stats(variant, model, candidates, keep_words),
            "paths": {
                "train_file": str(train_file),
                "model_file": str(model_file),
                "vocab_file": str(vocab_file),
                "keep_words_file": str(keep_file),
            },
            "train_words": len(used_counts),
            "keep_words": len(keep_words) if variant == "frequency" else 0,
        }
        preds_all[variant] = {"dev": dev_preds, "test": test_preds}

    return results, preds_all


def write_variant_table(report_root: Path, variant_results: Dict[str, object]) -> None:
    table = report_root / "metrics_table.md"
    with table.open("w", encoding="utf-8") as f:
        f.write("| variant | split | exact | boundary_f1 | avg_seg | oov |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for variant in VARIANTS:
            data = variant_results[variant]
            for split in ("dev", "test"):
                m = data[split]
                f.write(
                    f"| {variant} | {split} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} | {m['oov_rate']:.4f} |\n"
                )


def write_manifest(
    manifest_path: Path,
    cfg: RunConfig,
    stats: Dict[str, int],
    split_files: Dict[str, Path],
    eval_dev: Path,
    eval_test: Path,
    freq_file: Path,
    train_vocab_file: Path,
) -> None:
    manifest = {
        "config": {
            "name": cfg.name,
            "lines_limit": cfg.lines_limit,
            "min_freq": cfg.min_freq,
            "keep_freq": cfg.keep_freq,
            "dev_eval_words": cfg.dev_eval_words,
            "test_eval_words": cfg.test_eval_words,
            "seed": cfg.seed,
        },
        "stats": stats,
        "files": {
            "split_train": str(split_files["processed_train"]),
            "split_dev": str(split_files["processed_dev"]),
            "split_test": str(split_files["processed_test"]),
            "eval_dev": str(eval_dev),
            "eval_test": str(eval_test),
            "word_frequency_train": str(freq_file),
            "train_vocab_all": str(train_vocab_file),
        },
        "sha256": {
            "split_train": sha256_file(split_files["processed_train"]),
            "split_dev": sha256_file(split_files["processed_dev"]),
            "split_test": sha256_file(split_files["processed_test"]),
            "eval_dev": sha256_file(eval_dev),
            "eval_test": sha256_file(eval_test),
            "word_frequency_train": sha256_file(freq_file),
            "train_vocab_all": sha256_file(train_vocab_file),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(cfg: RunConfig, corpus_path: Path, exp_dir: Path) -> Dict[str, object]:
    ensure_eval_guideline(exp_dir)
    data_root = exp_dir / "data" / cfg.name
    model_root = exp_dir / "models" / cfg.name
    report_root = exp_dir / "reports" / cfg.name
    ensure_dir(report_root)

    train_counts_all, refs_split, stats, split_files = prepare_split_data(
        corpus_path=corpus_path,
        lines_limit=cfg.lines_limit,
        data_root=data_root,
        seed=cfg.seed,
    )

    freq_file = data_root / "word_frequency_train.tsv"
    candidates = write_word_frequency(freq_file, train_counts_all, cfg.min_freq)
    if not candidates:
        raise RuntimeError("No train candidates after frequency filtering.")
    train_vocab_file = data_root / "train_vocab_all.txt"
    write_word_list(train_vocab_file, sorted(train_counts_all.keys()))

    dev_candidates = list(refs_split["dev"].keys())
    test_candidates = list(refs_split["test"].keys())
    dev_words = select_words(dev_candidates, cfg.dev_eval_words, seed=cfg.seed + 11)
    test_words = select_words(test_candidates, cfg.test_eval_words, seed=cfg.seed + 17)
    if not dev_words or not test_words:
        raise RuntimeError("Evaluation splits are empty after selection.")

    dev_refs = {w: refs_split["dev"][w] for w in dev_words}
    test_refs = {w: refs_split["test"][w] for w in test_words}

    eval_root = data_root / "eval"
    ensure_dir(eval_root)
    dev_file = eval_root / "dev_gold.tsv"
    test_file = eval_root / "test_gold.tsv"
    save_eval_file(dev_file, dev_words, refs_split["dev"])
    save_eval_file(test_file, test_words, refs_split["test"])

    variant_results, variant_preds = run_variants(
        cfg=cfg,
        model_root=model_root,
        candidates=candidates,
        train_vocab=set(train_counts_all.keys()),
        dev_refs=dev_refs,
        test_refs=test_refs,
    )
    write_variant_table(report_root, variant_results)

    metrics_ok = metric_sanity_check()
    stable_ok = stable_eval_selection_check(dev_words, test_words, seed=cfg.seed)
    health = {
        "word_frequency_exists": freq_file.exists(),
        "dev_gold_exists": dev_file.exists(),
        "test_gold_exists": test_file.exists(),
        "all_variant_models_exist": all((model_root / v / "morfessor.bin").exists() for v in VARIANTS),
    }

    write_manifest(
        manifest_path=data_root / "manifest.json",
        cfg=cfg,
        stats=stats,
        split_files=split_files,
        eval_dev=dev_file,
        eval_test=test_file,
        freq_file=freq_file,
        train_vocab_file=train_vocab_file,
    )

    report = {
        "run_name": cfg.name,
        "config": {
            "lines_limit": cfg.lines_limit,
            "min_freq": cfg.min_freq,
            "keep_freq": cfg.keep_freq,
            "dev_eval_words": cfg.dev_eval_words,
            "test_eval_words": cfg.test_eval_words,
            "seed": cfg.seed,
        },
        "paths": {
            "data_root": str(data_root),
            "model_root": str(model_root),
            "report_root": str(report_root),
            "word_frequency_train": str(freq_file),
            "train_vocab_all": str(train_vocab_file),
            "dev_gold": str(dev_file),
            "test_gold": str(test_file),
            "manifest": str(data_root / "manifest.json"),
            "eval_guideline": str(exp_dir / "eval_guideline.md"),
        },
        "stats": {
            **stats,
            "train_vocab_raw_size": len(train_counts_all),
            "train_vocab_size_after_filter": len(candidates),
            "dev_eval_size": len(dev_words),
            "test_eval_size": len(test_words),
        },
        "variants": variant_results,
        "tests": {
            "pipeline_health": health,
            "metric_sanity": metrics_ok,
            "selection_stability": stable_ok,
            "run_pass": all(health.values()) and metrics_ok["pass"] and stable_ok["pass"],
        },
    }
    out = report_root / "metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{cfg.name}] wrote {out}")
    return report


def parse_farasa_segments(text: str, original_word: str) -> List[str]:
    candidate = normalize_arabic_text(text, keep_plus=True).replace(" ", "")
    if not candidate:
        fallback = normalize_word(original_word)
        return [fallback] if fallback else [original_word]
    if "+" in candidate:
        parts = [normalize_word(x) for x in candidate.split("+") if x]
        parts = [p for p in parts if p]
        if parts:
            return parts
    single = normalize_word(candidate)
    return [single] if single else [normalize_word(original_word)]


def load_keep_words(path: Path) -> set:
    if not path.exists():
        return set()
    words = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words


def run_farasa_eval(exp_dir: Path, seed: int) -> Dict[str, object]:
    full_data = exp_dir / "data" / "full"
    full_models = exp_dir / "models" / "full"
    full_reports = exp_dir / "reports" / "full"
    dev_file = full_data / "eval" / "dev_gold.tsv"
    test_file = full_data / "eval" / "test_gold.tsv"

    if not dev_file.exists() or not test_file.exists():
        result = {"status": "skipped", "reason": "Run full first to create dev/test gold files."}
        out = full_reports / "farasa_metrics.json"
        ensure_dir(full_reports)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[farasa] wrote {out}")
        return result

    try:
        from farasa.segmenter import FarasaSegmenter  # type: ignore
    except Exception as exc:
        result = {"status": "skipped", "reason": f"farasapy unavailable: {exc}"}
        out = full_reports / "farasa_metrics.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[farasa] wrote {out}")
        return result

    runtime_root = exp_dir / "tools" / "farasa_runtime"
    ensure_dir(runtime_root)
    ensure_dir(runtime_root / "tmp")
    ensure_dir(runtime_root / "farasa_bin")
    FarasaSegmenter.base_dir = runtime_root
    FarasaSegmenter.bin_dir = runtime_root / "farasa_bin"

    jar = runtime_root / "farasa_bin" / "lib" / "FarasaSegmenterJar.jar"
    if not jar.exists():
        result = {"status": "skipped", "reason": f"Farasa binaries unavailable: {jar} not found"}
        out = full_reports / "farasa_metrics.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[farasa] wrote {out}")
        return result

    try:
        segmenter = FarasaSegmenter(interactive=True, cache=False)
    except Exception as exc:
        result = {"status": "skipped", "reason": f"Farasa initialization failed: {exc}"}
        out = full_reports / "farasa_metrics.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[farasa] wrote {out}")
        return result

    refs = {"dev": load_eval_file(dev_file), "test": load_eval_file(test_file)}
    farasa_preds: Dict[str, Dict[str, List[str]]] = {}
    for split in ("dev", "test"):
        preds: Dict[str, List[str]] = {}
        for w in refs[split].keys():
            raw = segmenter.segment(w)
            preds[w] = parse_farasa_segments(raw, w)
        farasa_preds[split] = preds

    farasa_metrics = {
        "dev": evaluate_predictions(farasa_preds["dev"], refs["dev"], train_vocab=set()),
        "test": evaluate_predictions(farasa_preds["test"], refs["test"], train_vocab=set()),
    }
    for split in ("dev", "test"):
        farasa_metrics[split].pop("oov_rate", None)

    train_vocab_all_file = full_data / "train_vocab_all.txt"
    train_vocab = read_word_list(train_vocab_all_file)
    if not train_vocab:
        train_counts = read_word_frequency(full_data / "word_frequency_train.tsv")
        train_vocab = set(train_counts.keys())

    morf_preds: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    morf_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in VARIANTS:
        model_file = full_models / variant / "morfessor.bin"
        keep_file = full_models / variant / "keep_words.txt"
        if not model_file.exists():
            continue
        model = load_model(model_file)
        keep_words = load_keep_words(keep_file) if variant == "frequency" else set()
        morf_preds[variant] = {}
        morf_metrics[variant] = {}
        for split in ("dev", "test"):
            preds = predict_words(variant, model, refs[split].keys(), keep_words)
            morf_preds[variant][split] = preds
            morf_metrics[variant][split] = evaluate_predictions(preds, refs[split], train_vocab)

    compare: Dict[str, object] = {}
    for split in ("dev", "test"):
        compare[split] = {}
        for variant in VARIANTS:
            if variant not in morf_preds:
                continue
            ci_exact = bootstrap_ci_diff(
                farasa_preds[split], morf_preds[variant][split], refs[split], "exact_match", seed=seed + 3
            )
            ci_f1 = bootstrap_ci_diff(
                farasa_preds[split], morf_preds[variant][split], refs[split], "boundary_f1", seed=seed + 7
            )
            compare[split][variant] = {
                "delta_farasa_minus_morfessor": {
                    "exact_match": farasa_metrics[split]["exact_match"] - morf_metrics[variant][split]["exact_match"],
                    "boundary_f1": farasa_metrics[split]["boundary_f1"] - morf_metrics[variant][split]["boundary_f1"],
                },
                "bootstrap_farasa_minus_morfessor": {
                    "exact_match": ci_exact,
                    "boundary_f1": ci_f1,
                },
            }

    result = {
        "status": "ok",
        "splits": farasa_metrics,
        "morfessor_metrics": morf_metrics,
        "comparison_mode": "farasa_minus_morfessor",
        "compare": compare,
    }

    out = full_reports / "farasa_metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    table = full_reports / "compare_table.md"
    with table.open("w", encoding="utf-8") as f:
        f.write("| split | model | exact | boundary_f1 | avg_seg |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for split in ("dev", "test"):
            for variant in VARIANTS:
                if variant not in morf_metrics:
                    continue
                m = morf_metrics[variant][split]
                f.write(
                    f"| {split} | morfessor_{variant} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} |\n"
                )
            fm = farasa_metrics[split]
            f.write(
                f"| {split} | farasa | {fm['exact_match']:.4f} | {fm['boundary_f1']:.4f} | {fm['avg_segments_per_word']:.4f} |\n"
            )

    print(f"[farasa] wrote {out}")
    return result


def run_mode(mode: str, corpus_path: Path, seed: int, exp_dir: Path) -> None:
    ensure_dir(exp_dir)
    smoke = RunConfig(
        name="smoke",
        lines_limit=500,
        min_freq=2,
        keep_freq=200,
        dev_eval_words=100,
        test_eval_words=100,
        seed=seed,
    )
    full = RunConfig(
        name="full",
        lines_limit=None,
        min_freq=5,
        keep_freq=1700,
        dev_eval_words=1000,
        test_eval_words=1000,
        seed=seed,
    )

    if mode in {"smoke", "all"}:
        run_experiment(smoke, corpus_path=corpus_path, exp_dir=exp_dir)
    if mode in {"full", "all"}:
        run_experiment(full, corpus_path=corpus_path, exp_dir=exp_dir)
    if mode in {"farasa", "all"}:
        run_farasa_eval(exp_dir=exp_dir, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arabic Morfessor segmentation pipeline.")
    parser.add_argument("mode", choices=["smoke", "full", "farasa", "all"])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")
    run_mode(mode=args.mode, corpus_path=args.corpus, seed=args.seed, exp_dir=args.exp_dir)


if __name__ == "__main__":
    main()
