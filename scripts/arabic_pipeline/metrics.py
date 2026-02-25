from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from .text import split_word_to_boundaries
from .utils import percentile, select_words


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


def stable_eval_selection_check(val_words: Sequence[str], test_words: Sequence[str], seed: int) -> Dict[str, object]:
    val_a = select_words(val_words, len(val_words), seed)
    val_b = select_words(val_words, len(val_words), seed)
    test_a = select_words(test_words, len(test_words), seed)
    test_b = select_words(test_words, len(test_words), seed)
    return {
        "pass": val_a == val_b and test_a == test_b,
        "val_size": len(val_a),
        "test_size": len(test_a),
    }


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

