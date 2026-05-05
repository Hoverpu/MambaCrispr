# MambaCrispr

CRISPR-Cas9 indel prediction using Mamba state space models with multi-scale TCN and BiLSTM feature extraction.

## Project Structure

```
MambaCrispr/
├── mambacrispr/              # Model package
│   ├── models/
│   │   ├── mambacrispr.py    # MambaCrispr
│   │   └── blocks.py         # MultiScaleTCN, CSIM
│   └── utils/
│       ├── encoding.py       # DNA one-hot and token encoding
│       ├── data.py           # Data loading utilities
│       ├── metrics.py        # Evaluation metrics
│       └── seed.py           # Random seed setup
├── train/                    # Training scripts
│   ├── config.py             # Hyperparameters and constants
│   ├── data.py               # Data loading and 76.5/8.5/15 split
│   ├── evaluate.py           # Model evaluation
│   ├── trainer.py            # Standard and DNLR training
│   └── main.py               # Entry point with CLI
└── data/                     # CRISPR datasets (CSV)
```

## Quick Start

### Standard Training

```bash
python -m train.main
```

### DNLR Training

```bash
python -m train.main --dnlr
```

## Data Split

Data is split into train/val/test with ratio 76.5/8.5/15:
- **Train** (76.5%): model training
- **Val** (8.5%): early stopping / warmup monitoring
- **Test** (15%): final metric reporting

## DNLR Framework

Deep Noise Label Refinement (DNLR) is an optional training strategy that:

1. **Warmup phase**: standard training until validation Spearman plateaus
2. **DNLR phase**: detects noisy samples by loss percentile, applies EMA soft-label refinement

Enable with `--dnlr` flag. Parameters are fixed:
- Alpha (EMA coefficient): 0.9
- Warmup patience: 10 epochs
- Noise percentile: 75th

## Datasets

12 CRISPR datasets in `data/`:
WT, ESP, HF, Sniper-Cas9, xCas, SpCas9-NG, HypaCas9, CRISPRON, HT_Cas9, HELA, HCT116, HL60

## Requirements

- Python 3.8+
- PyTorch
- mamba-ssm
- scikit-learn
- scipy
- numpy
- pandas
