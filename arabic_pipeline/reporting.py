from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import EVAL_SPLITS, SUBWORD_BASELINES, VARIANTS
from .utils import sha256_file


def write_variant_table(report_root: Path, variant_results: Dict[str, object]) -> None:
    table = report_root / "metrics_table.md"
    with table.open("w", encoding="utf-8") as f:
        f.write("| variant | split | exact | boundary_f1 | avg_seg | oov |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for variant in VARIANTS:
            data = variant_results[variant]
            for split in EVAL_SPLITS:
                m = data[split]
                f.write(
                    f"| {variant} | {split} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} | {m['oov_rate']:.4f} |\n"
                )


def write_subword_table(report_root: Path, subword_results: Dict[str, object]) -> None:
    table = report_root / "subword_metrics_table.md"
    with table.open("w", encoding="utf-8") as f:
        f.write("| model | split | vocab | exact | boundary_f1 | avg_seg | oov |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        baselines = subword_results.get("baselines", {})
        for baseline in SUBWORD_BASELINES:
            b = baselines.get(baseline, {})
            if b.get("status") != "ok":
                continue
            vocab = b["best_vocab_size"]
            for split in EVAL_SPLITS:
                m = b[split]
                f.write(
                    f"| {baseline} | {split} | {vocab} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} | {m['oov_rate']:.4f} |\n"
                )


def write_manifest(
    manifest_path: Path,
    cfg,
    stats: Dict[str, int],
    split_files: Dict[str, Path],
    eval_val: Path,
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
            "val_eval_words": cfg.val_eval_words,
            "test_eval_words": cfg.test_eval_words,
            "seed": cfg.seed,
        },
        "stats": stats,
        "files": {
            "split_train": str(split_files["processed_train"]),
            "split_val": str(split_files["processed_val"]),
            "split_test": str(split_files["processed_test"]),
            "eval_val": str(eval_val),
            "eval_test": str(eval_test),
            "word_frequency_train": str(freq_file),
            "train_vocab_all": str(train_vocab_file),
        },
        "sha256": {
            "split_train": sha256_file(split_files["processed_train"]),
            "split_val": sha256_file(split_files["processed_val"]),
            "split_test": sha256_file(split_files["processed_test"]),
            "eval_val": sha256_file(eval_val),
            "eval_test": sha256_file(eval_test),
            "word_frequency_train": sha256_file(freq_file),
            "train_vocab_all": sha256_file(train_vocab_file),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_compact_results_file(
    report_root: Path,
    variant_results: Dict[str, object],
    subword_results: Dict[str, object],
    farasa_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    rows: List[Dict[str, object]] = []

    for variant in VARIANTS:
        if variant not in variant_results:
            continue
        for split in EVAL_SPLITS:
            m = variant_results[variant][split]
            rows.append(
                {
                    "split": split,
                    "model": f"morfessor_{variant}",
                    "exact_match": m["exact_match"],
                    "boundary_f1": m["boundary_f1"],
                    "avg_segments_per_word": m["avg_segments_per_word"],
                    "oov_rate": m.get("oov_rate"),
                    "selected_vocab_size": "",
                }
            )

    baselines = subword_results.get("baselines", {}) if isinstance(subword_results, dict) else {}
    for baseline in SUBWORD_BASELINES:
        b = baselines.get(baseline, {})
        if b.get("status") != "ok":
            continue
        vocab = b.get("best_vocab_size", "")
        for split in EVAL_SPLITS:
            m = b[split]
            rows.append(
                {
                    "split": split,
                    "model": baseline,
                    "exact_match": m["exact_match"],
                    "boundary_f1": m["boundary_f1"],
                    "avg_segments_per_word": m["avg_segments_per_word"],
                    "oov_rate": m.get("oov_rate"),
                    "selected_vocab_size": vocab,
                }
            )

    if farasa_metrics:
        for split in EVAL_SPLITS:
            if split not in farasa_metrics:
                continue
            m = farasa_metrics[split]
            rows.append(
                {
                    "split": split,
                    "model": "farasa",
                    "exact_match": m["exact_match"],
                    "boundary_f1": m["boundary_f1"],
                    "avg_segments_per_word": m["avg_segments_per_word"],
                    "oov_rate": "",
                    "selected_vocab_size": "",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        model_order = [f"morfessor_{v}" for v in VARIANTS] + list(SUBWORD_BASELINES) + ["farasa"]
        split_order = {s: i for i, s in enumerate(EVAL_SPLITS)}
        model_order_map = {m: i for i, m in enumerate(model_order)}
        df["_split_order"] = df["split"].map(split_order).fillna(9999)
        df["_model_order"] = df["model"].map(model_order_map).fillna(9999)
        df = df.sort_values(["_split_order", "_model_order", "boundary_f1"], ascending=[True, True, False])
        df = df.drop(columns=["_split_order", "_model_order"])

    out = report_root / "results_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_segmentation_compare_files(
    report_root: Path,
    refs: Dict[str, Dict[str, List[str]]],
    model_preds: Dict[str, Dict[str, Dict[str, List[str]]]],
) -> None:
    model_names = sorted(model_preds.keys())

    for split in EVAL_SPLITS:
        gold_map = refs.get(split, {})
        words = list(gold_map.keys())
        rows: List[Dict[str, object]] = []
        for word in words:
            row: Dict[str, object] = {
                "word": word,
                "gold": "+".join(gold_map[word]),
            }
            for model_name in model_names:
                pred_by_split = model_preds.get(model_name, {})
                pred_by_word = pred_by_split.get(split, {})
                pred = pred_by_word.get(word, [word])
                row[model_name] = "+".join(pred)
            rows.append(row)

        df = pd.DataFrame(rows)
        tsv_path = report_root / f"{split}_segmentation_compare.tsv"
        csv_path = report_root / f"{split}_segmentation_compare.csv"
        df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
