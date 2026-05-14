# Machine Learning for Network Intrusion Detection

**CS 4361 — Machine Learning, Spring 2026**
**Team:** Miranda Macias, Jocelyn Zamora, Deniss Garcia

This project builds and evaluates machine learning models for binary classification of network traffic (normal vs. cyber attack) using the CIC-IDS2017 dataset. We compared four model architectures across three experimental setups of increasing realism and discovered that standard ML evaluation dramatically overstates real-world performance.

## Overview

We structured our evaluation across three stages:

1. **Stage 1 — Random-split baseline.** Standard 80/20 stratified split on the full dataset. Used to establish per-model performance under favorable conditions.
2. **Stage 2 — Temporal split (Tuesday → Wednesday).** Train on Tuesday traffic, test on Wednesday traffic. A realistic deployment setup that tests generalization to attack types not seen during training.
3. **Stage 3 — Active learning.** Use entropy-based uncertainty sampling to selectively label a small number of Wednesday samples and retrain. Includes a random-sampling control that revealed an unexpected failure mode.

## Headline Results

| Model | Stage 1 F1 (random split) | Stage 2 F1 (temporal split) |
|---|---|---|
| Decision Tree | 0.9985 | 0.0015 |
| kNN (k=5) | 0.9913 | 0.0041 |
| Neural Network (MLP) | 0.9817 | 0.0074 |
| Logistic Regression | 0.9430 | 0.0752 |

**Stage 3 Active Learning:** Random sampling outperformed entropy-based active learning by 16× on attack recall (64.5% vs 4.1%), because the model was *confidently wrong* about Wednesday's new attack types rather than uncertain.

## Key Findings

1. **Random-split evaluation overstates real-world performance.** The Decision Tree achieved 0.9985 F1 on the random split but dropped to 0.0015 F1 on the temporal split — the same model went from catching 99.87% of attacks to catching 0.08%.
2. **Distribution shift defeats all four model architectures.** Decision Tree, Logistic Regression, kNN, and the Neural Network all collapsed to under 5% attack recall on Wednesday's DoS attacks. Deep architecture wasn't the fix — this is a data problem, not a model problem.
3. **Pure entropy-based active learning misses out-of-distribution attacks.** Entropy AL picked zero attack samples out of 500 because the model was *confidently wrong* about Wednesday's DoS attacks rather than uncertain. Random sampling caught 184 attacks by chance and boosted recall from 4% to 65%.

## Models Compared

- **Decision Tree** with and without `class_weight="balanced"` (max_depth=20)
- **Logistic Regression** with and without `class_weight="balanced"` (max_iter=1000)
- **k-Nearest Neighbors** (k=5, training capped at 100,000 rows for tractability)
- **Multilayer Perceptron** (3 hidden layers: 128 → 64 → 32, ReLU, Adam, sample_weight for imbalance)

## Dataset

We used the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains ~2.1M labeled network flows across 5 days (Monday–Friday) with 78 features per flow and a binary attack/benign label. Each day contains different attack types:
- **Monday:** benign only (no attacks)
- **Tuesday:** brute-force attacks (FTP-Patator, SSH-Patator) — ~14k attacks
- **Wednesday:** DoS attacks (Hulk, GoldenEye, Slowloris, Slowhttptest) and Heartbleed — ~252k attacks
- **Thursday:** web attacks, infiltration
- **Friday:** botnet, port scan, DDoS

The dataset files are not included in this repository due to size (~2 GB total). To reproduce, download the raw per-day CSVs from the link above.

## Repository Structure

```
.
├── README.md
├── REPORT_ML.pdf                              # Final team report
├── scripts/
│   ├── member2_classical_models.py            # Stage 1: DT, LR, kNN on random split
│   ├── member2_temporal_split.py              # Stage 2: Tuesday → Wednesday
│   ├── member2_active_learning.py             # Stage 3: entropy AL + random control
│   ├── member3_stage1.py                      # Neural network Stage 1
│   └── member3_temporal_split.py              # Neural network Stage 2
├── results/
│   ├── member2_results.csv                    # Classical model Stage 1 results
│   ├── member2_temporal_results.csv           # Classical model Stage 2 results
│   ├── member2_active_learning_results.csv    # Stage 3 active learning results
│   ├── member3_stage1_results.csv             # NN Stage 1 results
│   └── member3_temporal_results.csv           # NN Stage 2 results
└── writeups/
    └── Miranda_Macias_Writeup.docx            # Member 2 individual writeup
```

## Methodology

- **Preprocessing:** Removed rows containing `inf` or `NaN` values. Converted the string `Label` column to binary `Attack_Flag` (0 = BENIGN, 1 = any attack type).
- **Feature scaling:** `StandardScaler` for Logistic Regression, kNN, and the Neural Network. Decision Tree was trained on unscaled features since splits don't depend on feature magnitude.
- **Class imbalance:** Addressed via `class_weight="balanced"` for the classical models and `sample_weight` (equivalent formula) for the neural network. Weights are inversely proportional to class frequency, penalizing minority-class errors more heavily during training.
- **Train/test splits:** Saved as `.npy` files so all team members evaluate on identical data — critical for the four-way model comparison to be valid.
- **Reproducibility:** All experiments use `random_state=42`.

## Reproducing Results

1. Download the CIC-IDS2017 per-day CSV files from the Canadian Institute for Cybersecurity.
2. Place them in the project root with their original filenames.
3. Run scripts in order:

```bash
python scripts/member2_classical_models.py     # Stage 1 (classical models, full dataset)
python scripts/member2_temporal_split.py       # Stage 2 (classical, Tue → Wed) — saves .npy files
python scripts/member2_active_learning.py      # Stage 3 (active learning) — uses Stage 2 .npy files
python scripts/member3_stage1.py               # Stage 1 (neural network, full dataset)
python scripts/member3_temporal_split.py       # Stage 2 (neural network, Tue → Wed)
```

Runtime: classical models ~10 min total, neural network ~30 min on CPU, active learning ~3 min.

## Limitations

- **CIC-IDS2017** is known in the literature for producing high scores across most classifiers in random-split settings, with critique of duplicates and label quality issues that inflate Stage 1 numbers.
- **Active learning** was implemented as a single-round experiment with one batch of 500 samples. Real systems iterate over many rounds with smaller batches.
- **No hyperparameter tuning** was performed. Reasonable defaults were used throughout.
- **Limited temporal scope.** Stage 2 and Stage 3 use only Tuesday and Wednesday. Other day pairs would test whether the failure mode generalizes.

## Future Work

- **Smarter active learning:** margin sampling, query-by-committee, or diversity-based selection to avoid the "confidently wrong" blind spot.
- **Out-of-distribution detection:** flag inputs unlike the training distribution (e.g., Mahalanobis distance, autoencoder reconstruction error) before classifying.
- **Ensemble methods:** Random Forest or gradient boosting for stronger and more robust baselines.
- **Multi-day training:** expose the model to broader attack-type diversity than a single day provides.

## License

Coursework submission for academic use only.
