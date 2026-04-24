<div align="center">

# Utility-Aware Data Pricing

**Token-Level Quality and Empirical Training Gain for LLMs**

<p>
  <img src="assets/readme_fig.png" width="100%" alt="Framework Overview">
</p>

<p>
  <img src="https://img.shields.io/badge/python-≥3.9-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://github.com/BDS-SDU/utility-aware-data-pricing/graphs/commit-activity" target="_blank">
    <img alt="Commits last month" src="https://img.shields.io/github/commit-activity/m/BDS-SDU/utility-aware-data-pricing?labelColor=%20%2332b583&color=%20%2312b76a"></a>
  <a href="https://github.com/BDS-SDU/utility-aware-data-pricing/issues?q=is%3Aissue%20is%3Aclosed" target="_blank">
    <img alt="Issues closed" src="https://img.shields.io/github/issues-search?query=repo%3ABDS-SDU%2Futility-aware-data-pricing%20is%3Aissue%20is%3Aclosed&label=issues%20closed&labelColor=%20%237d89b0&color=%20%235d6b98"></a>
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">
</p>

<p>
  <a href="./README_CN.md">简体中文</a> | English
</p>

</div>

> **One-line summary:** A dynamic data valuation framework that moves beyond static "row-count × quality" pricing to utility-based pricing — combining token-level Shannon entropy metrics, empirical training gain (influence functions, proxy models, Data Shapley), and cryptographic verifiability (Merkle trees, hash-based commitments) — enabling data to be priced by its actual contribution to LLM intelligence.

Official implementation of the paper on utility-aware data pricing, which proposes a principled framework for valuing training data sources by combining token-level document quality scoring with empirical training gain estimation.

## Overview

This framework prices training data sources through four complementary valuation signals:

- **Document Quality Score (DQS)** — measures information density, syntactic coherence, and semantic richness at the token level using an n-gram reference model.
- **Proxy Gain** — estimates the utility of each data source by measuring improvement in a fast proxy model (hashed logistic regression).
- **Influence Score** — quantifies model sensitivity to the inclusion of each source.
- **Shapley Value** — approximates the marginal contribution of each source via Monte Carlo sampling.

These signals are normalized and ensembled into a unified score with configurable weights. A **Proof-of-Training Ledger** records immutable parameter commitments and Merkle roots for verification and auditability.

## Repository Structure

```
utility-aware-data-pricing/
├── data_pricing/              # Core library
│   ├── __init__.py            # Package exports
│   ├── data_types.py          # DocumentRecord, DocumentScore, SourceScore, ValuationReport
│   ├── pipeline.py            # DynamicDataValuationPipeline (main orchestrator)
│   ├── quality.py             # DQS: n-gram model, info density, syntactic/semantic scoring
│   ├── valuation.py           # ProxyEvaluator: hashed-text logistic regression
│   ├── shapley.py             # Monte Carlo Shapley value approximation
│   ├── verification.py        # Proof-of-Training Ledger (Merkle roots, hash chains)
│   ├── real_datasets.py       # Real dataset specs (Alpaca, GSM8K, CodeXGLue)
│   ├── demo_data.py           # Synthetic demo data generator
│   └── io_utils.py            # JSONL/CSV/JSON I/O helpers
├── scripts/                   # All runnable scripts
│   ├── run_valuation.py                           # Run valuation pipeline
│   ├── run_experiments.py                         # Ranking, robustness & ablation experiments
│   ├── run_real_multidomain_experiments.py        # Real multi-domain experiments
│   ├── plot_results.py                            # SVG visualization
│   ├── plot_real_multidomain_matplotlib.py        # Matplotlib paper-quality plots
│   ├── export_latex_figures.py                    # Export PDF figures for LaTeX
│   ├── generate_demo_data.py                      # Generate demo JSONL files
│   └── download_dataset.py                        # Download real datasets from Hugging Face
├── assets/                    # Static assets
│   └── readme_fig.png         # README cover image
├── data/                      # Training and validation data (JSONL, git-ignored)
└── outputs/                   # Generated reports, figures, and tables (git-ignored)
    ├── figures/               # SVG, PDF, PNG figures
    └── tables/                # CSV result tables
```

## Installation

The implementation is pure Python with minimal dependencies:

```bash
pip install datasets numpy matplotlib
```

> No compiled extensions are required. The proxy model and n-gram scorer are implemented from scratch for full reproducibility.

## Quick Start

### 1. Run Valuation on Demo Data

```bash
python scripts/run_valuation.py
```

This runs the full valuation pipeline on synthetic demo data. Outputs are saved to `outputs/`:
- `valuation_report.json` — complete valuation report
- `tables/valuation_source_scores.csv` — per-source scores (DQS, proxy gain, influence, Shapley, unified)
- `tables/valuation_document_scores.csv` — per-document quality metrics

Custom data paths:

```bash
python scripts/run_valuation.py --train data/my_train.jsonl --val data/my_val.jsonl --output outputs/my_report.json
```

### 2. Run Experiments (Ranking, Robustness, Ablation)

```bash
python scripts/run_experiments.py
```

Compares all valuation methods against actual training gains, learns optimal ensemble weights via correlation analysis, and runs robustness (noise duplication attack) and ablation (leave-one-signal-out) studies. Outputs:
- `outputs/experiment_report.json`
- `outputs/tables/ranking_metrics.csv`
- `outputs/tables/source_estimators.csv`
- `outputs/tables/learned_weights.csv`
- `outputs/tables/robustness.csv`
- `outputs/tables/ablations.csv`

### 3. Real Multi-Domain Experiments

First, download the real datasets:

```bash
python scripts/download_dataset.py
```

Then run the multi-domain experiment:

```bash
python scripts/run_real_multidomain_experiments.py
```

This evaluates valuation across three domains — general instruction (Alpaca), math reasoning (GSM8K), and code summarization (CodeXGLue) — producing per-domain and cross-domain results.

Options:

```bash
python scripts/run_real_multidomain_experiments.py \
  --datasets alpaca_instruction gsm8k_math codexglue_python \
  --train-limit-per-domain 120 \
  --shapley-iterations 32 \
  --seed 17
```

### 4. Generate Figures

```bash
# SVG figures
python scripts/plot_results.py

# Paper-quality PDF/PNG figures (requires matplotlib)
python scripts/plot_real_multidomain_matplotlib.py
```

All figures are saved to `outputs/figures/`.

## Input Data Format

Training and validation data use JSONL format. Each line is a JSON object:

```json
{"doc_id": "doc_001", "source_id": "source_a", "text": "document content...", "label": 0}
```

| Field       | Description                                    |
|-------------|------------------------------------------------|
| `doc_id`    | Unique document identifier                     |
| `source_id` | Data source the document belongs to            |
| `text`      | Document text content                          |
| `label`     | Integer label (used by proxy model)            |

## Pipeline Configuration

The `DynamicDataValuationPipeline` accepts two key weight tuples:

| Parameter            | Default              | Description                                     |
|----------------------|----------------------|-------------------------------------------------|
| `dqs_weights`        | `(0.4, 0.3, 0.3)`   | Weights for (info density, syntactic, semantic) |
| `ensemble_weights`   | `(0.25, 0.35, 0.2, 0.2)` | Weights for (DQS, proxy, influence, Shapley) |
| `ngram_order`        | `3`                  | N-gram order for the reference language model   |

## Datasets

| Dataset | Domain | Source |
|---------|--------|--------|
| Alpaca | General instruction following | `tatsu-lab/alpaca` |
| GSM8K | Math reasoning | `openai/gsm8k` |
| CodeXGLue Python | Code summarization | `google/code_x_glue_ct_code_to_text` |


## Disclaimer

This repository is a **research prototype** provided for academic and educational purposes only. It is **not intended for production or industrial use**.

## License

This project is licensed under the [MIT License](LICENSE).
