from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_CORPUS, DEFAULT_EXP_DIR, EVAL_SPLITS, RunConfig, SUBWORD_BASELINES, VARIANTS
from .data import (
    ensure_eval_guideline,
    load_eval_file,
    prepare_split_data,
    read_word_frequency,
    read_word_list,
    save_eval_file,
    write_word_frequency,
    write_word_list,
)
from .metrics import bootstrap_ci_diff, metric_sanity_check, stable_eval_selection_check
from .modeling import (
    load_keep_words,
    load_model,
    load_subword_tokenizer,
    parse_farasa_segments,
    predict_subword_words,
    predict_words,
    run_subword_baselines,
    run_variants,
)
from .reporting import (
    write_compact_results_file,
    write_manifest,
    write_segmentation_compare_files,
    write_subword_table,
    write_variant_table,
)
from .utils import ensure_dir, select_words


def _best_subword_predictions(
    refs: Dict[str, Dict[str, List[str]]],
    baselines: Dict[str, object],
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for baseline in SUBWORD_BASELINES:
        b = baselines.get(baseline, {})
        if b.get("status") != "ok":
            continue
        tok_file = Path(b["paths"]["tokenizer_file"])
        if not tok_file.exists():
            continue
        tokenizer = load_subword_tokenizer(tok_file)
        out[baseline] = {}
        for split in EVAL_SPLITS:
            out[baseline][split] = predict_subword_words(tokenizer, refs[split].keys())
    return out


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

    val_candidates = list(refs_split["val"].keys())
    test_candidates = list(refs_split["test"].keys())
    val_words = select_words(val_candidates, cfg.val_eval_words, seed=cfg.seed + 11)
    test_words = select_words(test_candidates, cfg.test_eval_words, seed=cfg.seed + 17)
    if not val_words or not test_words:
        raise RuntimeError("Evaluation splits are empty after selection.")

    val_refs = {w: refs_split["val"][w] for w in val_words}
    test_refs = {w: refs_split["test"][w] for w in test_words}

    eval_root = data_root / "eval"
    ensure_dir(eval_root)
    val_file = eval_root / "val_gold.tsv"
    test_file = eval_root / "test_gold.tsv"
    save_eval_file(val_file, val_words, refs_split["val"])
    save_eval_file(test_file, test_words, refs_split["test"])

    variant_results, variant_preds = run_variants(
        keep_freq=cfg.keep_freq,
        model_root=model_root,
        candidates=candidates,
        train_vocab=set(train_counts_all.keys()),
        val_refs=val_refs,
        test_refs=test_refs,
    )
    write_variant_table(report_root, variant_results)

    subword_results = run_subword_baselines(
        model_root=model_root,
        train_text_file=split_files["processed_train"],
        train_counts=train_counts_all,
        train_vocab=set(train_counts_all.keys()),
        val_refs=val_refs,
        test_refs=test_refs,
    )
    write_subword_table(report_root, subword_results)
    refs_by_split = {"val": val_refs, "test": test_refs}
    model_preds: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        f"morfessor_{variant}": variant_preds[variant] for variant in VARIANTS
    }
    subword_best = _best_subword_predictions(
        refs=refs_by_split,
        baselines=subword_results.get("baselines", {}),
    )
    model_preds.update(subword_best)
    write_segmentation_compare_files(
        report_root=report_root,
        refs=refs_by_split,
        model_preds=model_preds,
    )
    write_compact_results_file(
        report_root=report_root,
        variant_results=variant_results,
        subword_results=subword_results,
        farasa_metrics=None,
    )

    metrics_ok = metric_sanity_check()
    stable_ok = stable_eval_selection_check(val_words, test_words, seed=cfg.seed)
    health = {
        "word_frequency_exists": freq_file.exists(),
        "val_gold_exists": val_file.exists(),
        "test_gold_exists": test_file.exists(),
        "all_variant_models_exist": all((model_root / v / "morfessor.bin").exists() for v in VARIANTS),
    }

    write_manifest(
        manifest_path=data_root / "manifest.json",
        cfg=cfg,
        stats=stats,
        split_files=split_files,
        eval_val=val_file,
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
            "val_eval_words": cfg.val_eval_words,
            "test_eval_words": cfg.test_eval_words,
            "seed": cfg.seed,
        },
        "paths": {
            "data_root": str(data_root),
            "model_root": str(model_root),
            "report_root": str(report_root),
            "word_frequency_train": str(freq_file),
            "train_vocab_all": str(train_vocab_file),
            "val_gold": str(val_file),
            "test_gold": str(test_file),
            "manifest": str(data_root / "manifest.json"),
            "eval_guideline": str(exp_dir / "eval_guideline.md"),
        },
        "stats": {
            **stats,
            "train_vocab_raw_size": len(train_counts_all),
            "train_vocab_size_after_filter": len(candidates),
            "val_eval_size": len(val_words),
            "test_eval_size": len(test_words),
        },
        "variants": variant_results,
        "subword_baselines": subword_results,
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


def run_farasa_eval(exp_dir: Path, seed: int) -> Dict[str, object]:
    full_data = exp_dir / "data" / "full"
    full_models = exp_dir / "models" / "full"
    full_reports = exp_dir / "reports" / "full"
    val_file = full_data / "eval" / "val_gold.tsv"
    test_file = full_data / "eval" / "test_gold.tsv"

    if not val_file.exists() or not test_file.exists():
        result = {"status": "skipped", "reason": "Run full first to create val/test gold files."}
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

    refs = {"val": load_eval_file(val_file), "test": load_eval_file(test_file)}
    farasa_preds: Dict[str, Dict[str, List[str]]] = {}
    for split in EVAL_SPLITS:
        preds: Dict[str, List[str]] = {}
        for w in refs[split].keys():
            raw = segmenter.segment(w)
            preds[w] = parse_farasa_segments(raw, w)
        farasa_preds[split] = preds

    from .metrics import evaluate_predictions

    farasa_metrics = {
        "val": evaluate_predictions(farasa_preds["val"], refs["val"], train_vocab=set()),
        "test": evaluate_predictions(farasa_preds["test"], refs["test"], train_vocab=set()),
    }
    for split in EVAL_SPLITS:
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
        for split in EVAL_SPLITS:
            preds = predict_words(variant, model, refs[split].keys(), keep_words)
            morf_preds[variant][split] = preds
            morf_metrics[variant][split] = evaluate_predictions(preds, refs[split], train_vocab)

    compare: Dict[str, object] = {}
    for split in EVAL_SPLITS:
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
    subword_best: Dict[str, object] = {}
    full_metrics_file = full_reports / "metrics.json"
    if full_metrics_file.exists():
        try:
            full_report = json.loads(full_metrics_file.read_text(encoding="utf-8"))
            baselines = full_report.get("subword_baselines", {}).get("baselines", {})
            for baseline in SUBWORD_BASELINES:
                b = baselines.get(baseline, {})
                if b.get("status") == "ok":
                    subword_best[baseline] = b
        except Exception:
            subword_best = {}

    model_preds: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        f"morfessor_{variant}": morf_preds.get(variant, {}) for variant in VARIANTS if variant in morf_preds
    }
    model_preds.update(_best_subword_predictions(refs=refs, baselines=subword_best))
    model_preds["farasa"] = farasa_preds

    with table.open("w", encoding="utf-8") as f:
        f.write("| split | model | exact | boundary_f1 | avg_seg |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for split in EVAL_SPLITS:
            for variant in VARIANTS:
                if variant not in morf_metrics:
                    continue
                m = morf_metrics[variant][split]
                f.write(
                    f"| {split} | morfessor_{variant} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} |\n"
                )
            for baseline in SUBWORD_BASELINES:
                if baseline not in subword_best:
                    continue
                m = subword_best[baseline][split]
                f.write(
                    f"| {split} | {baseline} | {m['exact_match']:.4f} | {m['boundary_f1']:.4f} | {m['avg_segments_per_word']:.4f} |\n"
                )
            fm = farasa_metrics[split]
            f.write(
                f"| {split} | farasa | {fm['exact_match']:.4f} | {fm['boundary_f1']:.4f} | {fm['avg_segments_per_word']:.4f} |\n"
            )

    write_segmentation_compare_files(
        report_root=full_reports,
        refs=refs,
        model_preds=model_preds,
    )

    write_compact_results_file(
        report_root=full_reports,
        variant_results=morf_metrics,
        subword_results={"baselines": subword_best},
        farasa_metrics=farasa_metrics,
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
        val_eval_words=100,
        test_eval_words=100,
        seed=seed,
    )
    full = RunConfig(
        name="full",
        lines_limit=None,
        min_freq=5,
        keep_freq=1700,
        val_eval_words=1000,
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
