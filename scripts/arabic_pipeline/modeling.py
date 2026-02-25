from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import morfessor

from .config import EVAL_SPLITS, PUNCT_TOKENS, SPECIAL_TOKENS, SUBWORD_BASELINES, SUBWORD_VOCAB_SIZES, VARIANTS
from .metrics import evaluate_predictions
from .text import normalize_arabic_text, normalize_word
from .utils import ensure_dir


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


def build_subword_tokenizer(train_file: Path, baseline: str, vocab_size: int):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer

    if baseline == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )
    else:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=[str(train_file)], trainer=trainer)
    return tokenizer


def load_subword_tokenizer(tokenizer_file: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file))


def subword_tokens_to_segments(tokens: Sequence[str], original_word: str) -> List[str]:
    if not tokens:
        return [original_word]
    if any(t == "[UNK]" for t in tokens):
        return [original_word]

    segs: List[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t or t in SPECIAL_TOKENS:
            continue
        if t.startswith("##"):
            piece = t[2:]
            if not piece:
                continue
            segs.append(piece)
            continue
        if t.endswith("</w>"):
            t = t[: -len("</w>")]
        t = t.replace("▁", "").replace("Ġ", "")
        if t:
            segs.append(t)

    if not segs:
        return [original_word]

    pred_join = normalize_word("".join(segs))
    gold_join = normalize_word(original_word)
    if pred_join != gold_join:
        return [original_word]
    return segs


def predict_subword_words(tokenizer, words: Sequence[str]) -> Dict[str, List[str]]:
    preds: Dict[str, List[str]] = {}
    for w in words:
        encoded = tokenizer.encode(w)
        preds[w] = subword_tokens_to_segments(encoded.tokens, w)
    return preds


def subword_token_stats(tokenizer, train_counts: Dict[str, int]) -> Dict[str, float]:
    counter: Counter = Counter()
    weighted_seg_sum = 0
    word_sum = 0
    for w, c in train_counts.items():
        segs = subword_tokens_to_segments(tokenizer.encode(w).tokens, w)
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


def run_subword_baselines(
    model_root: Path,
    train_text_file: Path,
    train_counts: Dict[str, int],
    train_vocab: set,
    val_refs: Dict[str, List[str]],
    test_refs: Dict[str, List[str]],
) -> Dict[str, object]:
    out: Dict[str, object] = {"status": "ok", "baselines": {}}
    subword_root = model_root / "subword"
    ensure_dir(subword_root)

    for baseline in SUBWORD_BASELINES:
        baseline_dir = subword_root / baseline
        ensure_dir(baseline_dir)
        runs: Dict[str, object] = {}
        best_vocab: Optional[int] = None
        best_key = (-1.0, -1.0)

        for vocab_size in SUBWORD_VOCAB_SIZES:
            run_dir = baseline_dir / str(vocab_size)
            ensure_dir(run_dir)

            tokenizer = build_subword_tokenizer(train_text_file, baseline, vocab_size=vocab_size)
            tok_file = run_dir / "tokenizer.json"
            tokenizer.save(str(tok_file))

            val_preds = predict_subword_words(tokenizer, val_refs.keys())
            test_preds = predict_subword_words(tokenizer, test_refs.keys())
            val_metrics = evaluate_predictions(val_preds, val_refs, train_vocab)
            test_metrics = evaluate_predictions(test_preds, test_refs, train_vocab)
            stats = subword_token_stats(tokenizer, train_counts)

            runs[str(vocab_size)] = {
                "val": val_metrics,
                "test": test_metrics,
                "token_stats": stats,
                "paths": {"tokenizer_file": str(tok_file)},
            }
            key = (val_metrics["boundary_f1"], val_metrics["exact_match"])
            if key > best_key:
                best_key = key
                best_vocab = vocab_size

        if best_vocab is None:
            out["baselines"][baseline] = {"status": "failed", "reason": "No vocab size completed."}
            continue

        out["baselines"][baseline] = {
            "status": "ok",
            "selected_by": "best val boundary_f1, tie-break by val exact_match",
            "best_vocab_size": best_vocab,
            "runs": runs,
            "val": runs[str(best_vocab)]["val"],
            "test": runs[str(best_vocab)]["test"],
            "token_stats": runs[str(best_vocab)]["token_stats"],
            "paths": runs[str(best_vocab)]["paths"],
        }

    return out


def run_variants(
    keep_freq: int,
    model_root: Path,
    candidates: Dict[str, int],
    train_vocab: set,
    val_refs: Dict[str, List[str]],
    test_refs: Dict[str, List[str]],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, Dict[str, List[str]]]]]:
    ensure_dir(model_root)
    keep_words = {w for w, c in candidates.items() if c >= keep_freq}

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

        val_preds = predict_words(variant, model, val_refs.keys(), keep_words)
        test_preds = predict_words(variant, model, test_refs.keys(), keep_words)

        results[variant] = {
            "val": evaluate_predictions(val_preds, val_refs, train_vocab),
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
        preds_all[variant] = {"val": val_preds, "test": test_preds}

    return results, preds_all


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
