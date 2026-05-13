# Link Prediction on Bipartite Graphs Using Heterogeneous GNNs

An empirical study of link prediction on user–movie bipartite graphs using heterogeneous Graph Neural Networks with custom message passing, multiple loss functions, and progressive architectural improvements.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-green.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project investigates how architectural design decisions in heterogeneous GNNs affect link prediction performance on bipartite graphs. Starting from a simple unidirectional baseline, we incrementally introduce bidirectional message passing, rating-aware edge weighting, skip connections, L2 normalization, dropout, and MLP-based scoring — measuring the impact of each change.

**Key Results:**
| Model | Architecture | Best AUC |
|-------|-------------|----------|
| Model A | Unidirectional, dot-product | 0.9086 |
| Model B | Bidirectional, shared weights, cosine similarity | ~0.91 |
| Model C | Bidirectional, separate weights, sigmoid gating, MLP | **0.9271** |
| PPI Transfer | Adapted architecture on protein interactions | 0.9029 |

## Project Structure

```
├── notebooks/
│   ├── Copy_of_9_Link_Prediction_on_MovieLens (2).ipynb   # Model A — Baseline
│   ├── link_prediction_with_edges (2).ipynb                # Model B — Bidirectional
│   ├── link_prediction_with_edges_enhanced.ipynb           # Model C — Enhanced
│   ├── PPI.ipynb                                           # PPI generalizability experiment
│   └── temporal_data.ipynb                                 # Temporal extension (failed)
│
├── report/
│   └── link_prediction_report.tex                          # IEEE-format LaTeX report
│
├── figures/                                                # Training curves & visualizations
│   ├── BCE loss.png
│   ├── BCE auc.png
│   ├── prediction spread.png
│   └── ...
│
└── README.md
```

## Models

### Model A — Unidirectional Baseline
- **Message passing**: User → Movie only (movie embeddings refined, user embeddings static)
- **Edge weighting**: Additive (`1 + rating`)
- **Update rule**: Linear transform, no skip connection
- **Scorer**: Dot product + learned edge bias
- **Loss**: BCE | **AUC: 0.9086** (30 epochs, CPU)

### Model B — Bidirectional with Shared Weights
- **Message passing**: Bidirectional (user ↔ movie) using reversed edge indices
- **Edge weighting**: Additive (`1 + rating`)
- **Update rule**: Linear transform with skip connection (`aggr + x_dst`)
- **Scorer**: L2-normalized dot product (cosine similarity) + edge bias
- **Conv weights**: Shared across both directions (regularization via parameter tying)
- **Loss**: BCE

### Model C — Enhanced Bidirectional
- **Message passing**: Bidirectional with **separate** directional weights
- **Edge weighting**: Sigmoid gating (`σ(rating)`) — bounds influence in (0, 1)
- **Update rule**: Skip connection + **L2 normalization** inside convolution
- **Scorer**: 2-layer MLP (`concat → Linear → ReLU → Linear`)
- **Dropout**: `p=0.3` after ReLU in GNN and before scorer
- **Loss**: BCE | **AUC: 0.9271** (20 epochs, GPU)

### Loss Functions Compared
All three loss functions are implemented in Model C with a switchable `LOSS_TYPE` variable:
- **BCE**: `F.binary_cross_entropy_with_logits` — pointwise classification
- **BPR**: Bayesian Personalized Ranking — pairwise learning-to-rank
- **MSE**: Mean Squared Error on sigmoid outputs — regression framing

## Additional Experiments

### PPI (Protein-Protein Interaction)
Tested the core message passing architecture on the PPI biological network dataset to verify generalizability. Required removing heterogeneous node types and edge attributes. Achieved **validation AUC of 0.9029** over 30 epochs.

### Temporal Graph Extension (GDELT)
Attempted temporal link prediction using a custom `RelationalTGN` with GRU-based memory, sinusoidal time encoding, and relation embeddings on the GDELT geopolitical events dataset. The experiment **failed** due to catastrophic temporal data splits (241 train / 10,548 test events), non-differentiable memory updates, and time decay instability. Detailed failure analysis is provided in the report.

## Dataset

**MovieLens Latest Small** ([link](https://grouplens.org/datasets/movielens/)):
- 610 users, 9,742 movies, 100,836 ratings
- Movie features: 20-dim multi-hot genre vectors
- Split: 80% train / 10% val / 10% test (edge-level)
- Mini-batching: `LinkNeighborLoader` with 2-hop sampling (20, 10 neighbors)

## Tech Stack

- **Framework**: PyTorch + PyTorch Geometric
- **Key modules**: `MessagePassing`, `HeteroData`, `LinkNeighborLoader`, `RandomLinkSplit`
- **Training**: Adam optimizer, lr=0.001, hidden_dim=64
- **Evaluation**: scikit-learn `roc_auc_score`
- **Visualization**: matplotlib

## Getting Started

### Prerequisites
```bash
pip install torch torch-geometric scikit-learn matplotlib tqdm
```

### Run Model A (Baseline)
```bash
# Open in Jupyter/Colab and run all cells
jupyter notebook "Copy_of_9_Link_Prediction_on_MovieLens (2).ipynb"
```

### Run Model C (Enhanced) with different loss functions
```python
# In the training cell, change:
LOSS_TYPE = "bce"   # options: "bce", "mse", "bpr"
```

### Run PPI Experiment
```bash
jupyter notebook PPI.ipynb
```

## Report

The full IEEE-format research report is available in `report/`. It includes:
- Formal problem formulation with equations for all message passing variants
- Architectural comparison table across all three models
- Qualitative ablation analysis
- PPI generalizability results
- Detailed temporal extension failure analysis
- 26 academic references

## Key Takeaways

1. **Bidirectional message passing** is the single most impactful improvement — it transforms user embeddings from static lookups into structure-aware representations
2. **Skip connections** preserve node identity through GNN layers
3. **Sigmoid gating** provides smoother edge weighting than additive scaling
4. **L2 normalization** inside convolution layers prevents embedding magnitude drift
5. **Loss function choice** (BCE vs BPR) is secondary to architectural decisions at this dataset scale

## Citation

If you use this work, please cite:
```bibtex
@misc{linkpred2025,
  title={An Empirical Study of Link Prediction on Bipartite Graphs Using Heterogeneous GNNs},
  author={Student Name},
  year={2025},
  note={Minor Project, Department of CSE}
}
```

## License

This project is for academic/educational purposes.

