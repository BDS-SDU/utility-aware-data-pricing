# Supplementary Experiments — Cross-Scale Validation of Proxy-Based Data Valuation

This directory contains the supplementary experiment for the paper
*Utility-Aware Data Pricing: Token-Level Quality and Empirical Training Gain
for LLMs*. It tests the paper's proxy-to-target assumption (Section 2.2,
Eq. 10–11): **does data valuation computed with the paper's small proxy model
predict the effect of the same data sources on substantially larger language
models?**

Read [`SUPPLEMENTARY.md`](./SUPPLEMENTARY.md) for motivation, protocol,
results tables, and interpretation.

## What it does

**Phase 1 — smoke-scale screen** (120 docs/domain, 6 shards, 2 seeds):

| Stage | Script | Output |
|---|---|---|
| 1. Build data (3 domains × 6 shards × 20 docs + 32 val docs) | `build_data.py` | `data/*.jsonl` |
| 2. L0 proxy baseline — the paper's own pipeline, unchanged | `run_proxy_baseline.py` | `results/proxy_baseline.csv` |
| 3. LM ladder LOSO — Pythia 70m/160m/410m fine-tuning | `run_lm_ladder.py` | `results/lm_gains_*.csv`, `results/lm_detail_*.json` |
| 3b. Seed replication | same + `--seed 43 --out-dir results_seed43` | `results_seed43/…` |
| 4. Analysis + figures | `analyze_cross_scale.py`, `analyze_seed_stability.py` | `results/*.csv`, `results/summary*.json`, `figures/fig1–4` |

**Phase 2 — power-boosted confirmatory study** (480 docs/domain, 8 shards, 5 seeds, pre-registered criteria; S3.5 of `SUPPLEMENTARY.md`):

| Stage | Script | Output |
|---|---|---|
| 5. Build large corpora | `build_data.py --out-dir supplementary/data_large --train-limit 480 --val-limit 96 --shards 8` | `data_large/*.jsonl` |
| 6. L0 baseline (large) | `run_proxy_baseline.py --data-dir supplementary/data_large --out-dir supplementary/results_large` | `results_large/proxy_baseline.csv` |
| 7. LM ladder × 5 seeds (405 runs) | `run_lm_ladder.py --data-dir supplementary/data_large --out-dir supplementary/results_large_seed$S --seed $S`, S ∈ {17,43,7,101,2025} | `results_large_seed*/…` |
| 8. Condition verdicts C1–C5 | `analyze_power_boosted.py` | `results_large/power_conditions.csv`, `results_large/summary_power.json`, `figures/fig5` |

**Headline finding (pre-registered, permutation p < 0.05):** cross-scale
transfer **holds** when the proxy is itself a small language model
(70m/160m) and the decision is coarse-grained — top-25% high-value source
screening (4/6 hits vs 1.5 expected by chance, direction AUC 0.64–0.74). It
is **not established** for the paper's logistic-regression proxy (ρ ≈ 0
full ranking; below-chance screening) and for fine-grained full-ranking
transfer (70m→410m ρ = 0.30, p = 0.085, bounded by the reference ranking's
own seed-reproducibility).

## Quick start

```bash
pip install torch transformers datasets pandas scipy matplotlib huggingface_hub

# Phase 1 (~15 min on one 24 GB GPU)
python supplementary/build_data.py
python supplementary/run_proxy_baseline.py
python supplementary/run_lm_ladder.py                       # seed 17
python supplementary/run_lm_ladder.py --seed 43 --out-dir supplementary/results_seed43
python supplementary/analyze_cross_scale.py
python supplementary/analyze_seed_stability.py

# Phase 2 (~2.5 h; 405 fine-tuning runs)
python supplementary/build_data.py --out-dir supplementary/data_large --train-limit 480 --val-limit 96 --shards 8
python supplementary/run_proxy_baseline.py --data-dir supplementary/data_large --out-dir supplementary/results_large
for S in 17 43 7 101 2025; do
  python supplementary/run_lm_ladder.py --data-dir supplementary/data_large --out-dir supplementary/results_large_seed$S --seed $S
done
python supplementary/analyze_power_boosted.py
```

If huggingface.co is unreachable: `export HF_ENDPOINT=https://hf-mirror.com`.

## Committed artifacts

`results/`, `results_large/`, `results_large_seed*/` (CSV tables +
summaries), `figures/` (PDF + PNG), `data/` and `data_large/` (the exact
sampled corpora, for bit-level reproducibility), and all scripts.
`data*/_codexglue_cache/` is a download cache and is ignored.

## Hardware used for the released numbers

1× NVIDIA RTX 4090 (24 GB), PyTorch 2.8.0, transformers 4.57.0, CUDA 12.8.
531 fine-tuning runs in total: 126 in the smoke phase (2 seeds × 3 scales ×
3 domains × 7 runs) + 405 in the confirmatory phase (5 seeds × 3 scales × 3
domains × 9 runs).
