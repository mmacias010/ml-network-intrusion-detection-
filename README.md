# ml-network-intrusion-detection-
CS 4361 final project — ML for network intrusion detection on CIC-IDS2017
# Machine Learning for Network Intrusion Detection

**CS 4361 — Machine Learning, Spring 2026**
**Team:** Miranda Macias, Jocelyn Zamora, Deniss Garcia

This project builds and evaluates machine learning models for binary classification of network traffic (normal vs. cyber attack) using the CIC-IDS2017 dataset. We trained four model types and compared them under three experimental setups of increasing realism.

## Overview

We structured our evaluation across three stages:

1. **Stage 1 — Random-split baseline.** Standard 80/20 stratified split on the full dataset to establish per-model performance.
2. **Stage 2 — Temporal split (Tuesday → Wednesday).** Train on Tuesday traffic, test on Wednesday traffic. A realistic deployment setup that exposes how poorly models generalize to attack types they haven't seen.
3. **Stage 3 — Active learning.** Use entropy-based uncertainty sampling to selectively label a small number of Wednesday samples and retrain. Includes a random-sampling control.

## Models Compared

- Decision Tree (with and without `class_weight="balanced"`)
- Logistic Regression (with and without `class_weight="balanced"`)
- k-Nearest Neighbors (k=5)
- Multilayer Perceptron (3 hidden layers, 128 → 64 → 32, ReLU, Adam)

## Dataset

We used the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains ~2.1M labeled network flows across 5 days (Monday–Friday) with 78 features per flow and a binary attack/benign label. Each day contains different attack types:
- Monday: benign only
- Tuesday: brute-force attacks (FTP-Patator, SSH-Patator)
- Wednesday: DoS attacks (Hulk, GoldenEye, Slowloris, Slowhttptest) and Heartbleed
- Thursday: web attacks, infiltration
- Friday: botnet, port scan, DDoS

The dataset files are not included in this repository due to size (~2 GB total). To reproduce, download the raw files from the link above.

## Repository Structure

```
.
├── README.md
├── REPORT_ML.pdf                       # Final team report
├── Miranda_Macias_Writeup.docx         # Member 2 writeup (classical models + AL)
├── scripts/
│   ├── member2_classical_models.py     # Stage 1: DT, LR, kNN on random split
│   ├── member2_temporal_split.py       # Stage 2: Tuesday → Wednesday
│   ├── member2_active_learning.py      # Stage 3: entropy-based AL + random control
│   └── member3_nn.py                   # Neural network (Member 3)
└── results/
    ├── member2_results.csv             # Stage 1 classical model results
    ├── member2_temporal_results.csv    # Stage 2 results
    ├── member2_active_learning_results.csv  # Stage 3 results
    └── member3_results.csv             # NN results
```

## Methodology

- **Preprocessing:** Removed rows with `inf` or `NaN` values. Converted string `Label` column to binary `Attack_Flag` (0 = BENIGN, 1 = any attack type).
- **Feature scaling:** `StandardScaler` for Logistic Regression, kNN, and the NN. Decision Tree was trained on unscaled features.
- **Class imbalance:** Addressed via `class_weight="balanced"` for the classical models (and `sample_weight` for the NN), which weights minority-class errors more heavily during training.
- **Train/test splits:** Saved as `.npy` files so all team members evaluate on identical data.

## Reproducing Results

1. Download the CIC-IDS2017 per-day CSV files from the Canadian Institute for Cybersecurity.
2. Place them in the project root with their original filenames.
3. Run scripts in order:

```bash
python scripts/member2_classical_models.py     # Stage 1
python scripts/member2_temporal_split.py       # Stage 2 — also saves .npy split files
python scripts/member2_active_learning.py      # Stage 3 — uses .npy files from Stage 2
python scripts/member3_nn.py                   # Neural network — uses Stage 2 .npy files
```

## Key Findings

1. **Random-split evaluation overstates real-world performance.** The Decision Tree achieved 0.9985 F1 on a random split but dropped to 0.0015 F1 on the temporal split (Tuesday → Wednesday) — the same model went from catching 99.87% of attacks to catching 0.08%.
2. **Distribution shift between consecutive days is severe** when attack types differ. Tuesday's brute-force training data does not transfer to Wednesday's DoS attacks.
3. **Pure entropy-based active learning misses out-of-distribution attacks.** Our entropy-based AL picked zero attack samples out of 500 because the model was *confidently wrong* about Wednesday's DoS attacks rather than uncertain. Random sampling outperformed entropy-based AL by 16× on attack recall.

## Limitations

- CIC-IDS2017 has known duplicate-row and label-quality issues that inflate Stage 1 numbers.
- Active learning was implemented as a single-round experiment with one batch of 500 samples. Real systems iterate over many rounds.
- No hyperparameter tuning was performed. Reasonable defaults were used throughout.

## License

Coursework submission for academic use only.
