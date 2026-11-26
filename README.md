# Protein Secondary Structure Prediction

Sequence-to-sequence deep learning models that read an amino-acid sequence and predict the secondary structure of **every residue**, in both the 8-state (Q8) and 3-state (Q3) annotation schemes, built in PyTorch.

Built for a Kaggle competition in the *Deep Learning & Generative AI* course of the IIT Madras BS in Data Science programme (Dec 2025).

## Problem

```
input   seq : M K T A Y I A K Q R Q I S F V K ...
output  Q8  : C C C H H H H H H H H T T C E E ...   (B, C, E, G, H, I, S, T)
output  Q3  : C C C H H H H H H H H C C C E E ...   (C = coil, E = strand, H = helix)
```

| | |
|---|---|
| Task | Per-token classification, output length = input length |
| Train / test | 7,262 / 1,816 sequences |
| Vocabulary | 21 amino-acid tokens + padding |
| Metric | Harmonic mean of masked macro F1 on Q8 and Q3 |
| Main difficulty | Severe Q8 class imbalance: helix `H` ≈ 448k residues vs. `I` ≈ 253 |

## Approach

I started from the required baselines and built up step by step.

| # | Model | Idea |
|---|---|---|
| 1 | **Bidirectional RNN** | Baseline: embedding → BiRNN → two linear heads (Q8, Q3) |
| 2 | **BiLSTM** (2 layers) | Gates handle long-range dependencies; structure depends on both sides of a residue |
| 3 | **BiGRU** + class weights | Fewer parameters than LSTM; inverse-frequency loss weights for rare classes |
| 4 | **CNN + BiGRU** | 1-D convolutions capture local motifs before the recurrent layer |
| 5 | **Multi-kernel CNN + BiLSTM + CRF** (final) | Parallel conv kernels (3/5/7) in two stacked blocks with BatchNorm, a residual concat of the raw embeddings, a 2-layer BiLSTM, and a **CRF layer on Q8** to enforce valid label transitions |

```
tokens ─► embedding ─┬─► [conv3 | conv5 | conv7] ─► [conv3 | conv5 | conv7] ─┐
                     └──────────────── residual concat ──────────────────────┴─► BiLSTM ─┬─► CRF  ─► Q8
                                                                                         └─► Linear ─► Q3
```

**Engineering details**
- Custom `Dataset` and a `collate_fn` that pads variable-length sequences and builds attention masks
- Masked loss and masked macro F1, so padding never counts toward the metrics
- Multi-task loss: CRF negative log-likelihood (Q8) + class-weighted cross-entropy (Q3)
- Experiment tracking with **Trackio** on Hugging Face Spaces; final weights published to **KaggleHub**, and inference loads them from there

## Key takeaways

- **Local context and global context complement each other.** CNNs pick up short motifs, BiLSTMs carry information along the whole chain, and combining them beat either alone.
- **A CRF helps structured outputs.** Secondary-structure labels come in runs (helices are long, contiguous segments). A CRF models those transitions instead of predicting each residue independently.
- **Imbalance needs explicit handling.** Without class weights, rare Q8 states like `I` and `B` were almost never predicted.

## Repository structure

```
.
├── notebooks/
│   └── protein_secondary_structure.ipynb   # vocab → dataset → 5 architectures → inference
└── README.md
```

## Running it

The competition data can't be redistributed. The notebook was built to run on Kaggle (GPU).

```bash
pip install torch pandas scikit-learn pytorch-crf trackio kagglehub
```
