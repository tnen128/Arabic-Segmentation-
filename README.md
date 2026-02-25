# Arabic Segmentation

A reproducible Arabic segmentation project for comparing morphology-aware and subword tokenizers under a low-resource setup.  
The pipeline trains and evaluates Morfessor variants, BPE, and WordPiece, and can optionally compare against Farasa.

## Project Overview

This project focuses on intrinsic segmentation quality using fixed `train/val/test` splits from an Arabic corpus.

### Implemented models
- `morfessor_type`
- `morfessor_token`
- `morfessor_frequency`
- `bpe`
- `wordpiece`
- `farasa` (optional comparison)

### Core outputs
- Unified metrics summary (`results_summary.csv`)
- Per-word segmentation comparison files (`val_segmentation_compare.*`, `test_segmentation_compare.*`)
- Detailed metrics and manifests for reproducibility

## Project Structure

```text
scripts/
├─ arabic_morfessor_pipeline.py          # Thin entrypoint script for the Arabic pipeline
└─ arabic_pipeline/
   ├─ config.py                          # Constants, regex rules, run config dataclass
   ├─ data.py                            # Corpus reading, split prep, eval file IO
   ├─ text.py                            # Arabic normalization and token parsing helpers
   ├─ metrics.py                         # Segmentation metrics and statistical checks
   ├─ modeling.py                        # Morfessor/BPE/WordPiece/Farasa modeling logic
   ├─ reporting.py                       # Metrics tables, summary export, compare files
   ├─ runner.py                          # Pipeline orchestration (smoke/full/farasa/all)
   └─ utils.py                           # Shared utilities (dirs, hashing, sampling)

arabic_seg/
├─ data/                                 # Generated train/val/test and eval artifacts
├─ models/                               # Trained model artifacts
└─ reports/                              # Metrics, summaries, per-word comparisons

requirements.txt                         # Pinned Python dependencies
```

## Setup and Run

### 1) Create virtual environment
```bash
cd "tokenizer project-2"
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run full pipeline (Morfessor + BPE + WordPiece)
```bash
python3 scripts/arabic_morfessor_pipeline.py full --exp-dir arabic_seg
```

### 4) Run Farasa comparison (optional)
```bash
python3 scripts/arabic_morfessor_pipeline.py farasa --exp-dir arabic_seg
```

### 5) Run everything in one command
```bash
python3 scripts/arabic_morfessor_pipeline.py all --exp-dir arabic_seg
```

## Results Location

After `full`:
- `arabic_seg/reports/full/results_summary.csv`
- `arabic_seg/reports/full/metrics.json`
- `arabic_seg/reports/full/val_segmentation_compare.csv`
- `arabic_seg/reports/full/test_segmentation_compare.csv`

After `farasa`:
- `arabic_seg/reports/full/farasa_metrics.json`
- `arabic_seg/reports/full/compare_table.md`

## Notes

- Farasa mode requires Farasa runtime binaries/JAR to be available in the expected runtime path.
- `results_summary.csv` is the main single-file view for quick comparison.
- Detailed files are kept for auditability and reproducibility.
