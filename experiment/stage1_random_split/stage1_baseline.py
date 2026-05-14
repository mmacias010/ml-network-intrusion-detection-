"""
Member 3 -- Stage 1: Neural Network with random split baseline.
Uses the full cleaned CIC-IDS2017 dataset and creates a stratified 80/20
random split so results are directly comparable to the classical models'
Stage 1 random-split baseline.

Input:
  data/cleaned/CIC_IDS2017_Clean.csv

Outputs:
  experiments/stage1_random_split/stage1_results.csv
  data/splits/x_train_stage1_random.npy
  data/splits/y_train_stage1_random.npy
  data/splits/x_test_stage1_random.npy
  data/splits/y_test_stage1_random.npy
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEANED_FULL_DATASET_PATH = CLEANED_DATA_DIR / "CIC_IDS2017_Clean.csv"
RESULTS_PATH = Path(__file__).resolve().with_name("stage1_results.csv")

RANDOM_STATE = 42
CLASS_NAMES  = ["normal", "attack"]
TARGET_COL = "Attack_Flag"
TEST_SIZE = 0.20

X_TRAIN_PATH = SPLITS_DIR / "x_train_stage1_random.npy"
Y_TRAIN_PATH = SPLITS_DIR / "y_train_stage1_random.npy"
X_TEST_PATH  = SPLITS_DIR / "x_test_stage1_random.npy"
Y_TEST_PATH  = SPLITS_DIR / "y_test_stage1_random.npy"

# =========================================================================
# HELPERS
# =========================================================================
def evaluate(name, model, X_te, y_te):
    t0 = time.time()
    y_pred = model.predict(X_te)
    pred_t = time.time() - t0

    print(f"\n{name}")
    print(f"  predict time: {pred_t:.1f}s")
    print(f"  accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print("  classification report:")
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print(f"  confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_te, y_pred)}")

    rep = classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    return {
        "model":            name,
        "accuracy":         rep["accuracy"],
        "normal_precision": rep["normal"]["precision"],
        "normal_recall":    rep["normal"]["recall"],
        "normal_f1":        rep["normal"]["f1-score"],
        "attack_precision": rep["attack"]["precision"],
        "attack_recall":    rep["attack"]["recall"],
        "attack_f1":        rep["attack"]["f1-score"],
        "macro_f1":         rep["macro avg"]["f1-score"],
        "weighted_f1":      rep["weighted avg"]["f1-score"],
        "predict_sec":      pred_t,
    }

# =========================================================================
# LOAD full cleaned dataset + create random split
# =========================================================================
print("Stage 1: Random-split baseline (FULL DATASET, ~2.1M rows)\n")
print(f"Loading cleaned full dataset from {CLEANED_FULL_DATASET_PATH}...")
if not CLEANED_FULL_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Missing cleaned full dataset: {CLEANED_FULL_DATASET_PATH}"
    )

df = pd.read_csv(CLEANED_FULL_DATASET_PATH)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

if TARGET_COL not in df.columns:
    raise ValueError(f"Expected target column '{TARGET_COL}' in cleaned dataset.")

non_num = df.drop(columns=[TARGET_COL]).select_dtypes(exclude=np.number).columns
if len(non_num) > 0:
    print(f"  dropping non-numeric columns: {list(non_num)}")
    df = df.drop(columns=list(non_num))

X = df.drop(columns=[TARGET_COL]).values
y = df[TARGET_COL].values.astype(int)

print("Creating stratified 80/20 random split...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
np.save(X_TRAIN_PATH, X_train)
np.save(Y_TRAIN_PATH, y_train)
np.save(X_TEST_PATH, X_test)
np.save(Y_TEST_PATH, y_test)
print(f"  saved -> {X_TRAIN_PATH}, {Y_TRAIN_PATH}, {X_TEST_PATH}, {Y_TEST_PATH}")

print(f"  Train -> {X_train.shape[0]:,} rows   "
      f"normal: {(y_train==0).sum():,}   attack: {(y_train==1).sum():,}")
print(f"  Test  -> {X_test.shape[0]:,} rows   "
      f"normal: {(y_test==0).sum():,}   attack: {(y_test==1).sum():,}")

# =========================================================================
# SCALE
# =========================================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =========================================================================
# CLASS WEIGHTS (replaces SMOTE)
# =========================================================================
classes, counts = np.unique(y_train, return_counts=True)
weight_map     = {c: len(y_train) / (len(classes) * cnt)
                  for c, cnt in zip(classes, counts)}
sample_weights = np.array([weight_map[y] for y in y_train])

# =========================================================================
# TRAIN
# =========================================================================
print("\nTraining Neural Network (128-64-32, relu, adam)...")
t0 = time.time()
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=RANDOM_STATE,
    verbose=False,
)
nn.fit(X_train_s, y_train, sample_weight=sample_weights)
print(f"  fit: {time.time()-t0:.1f}s   iterations: {nn.n_iter_}")

# =========================================================================
# EVALUATE + SAVE
# =========================================================================
results = []
results.append(evaluate("Neural Network (128-64-32)", nn, X_test_s, y_test))

print("\n" + "="*90)
print("STAGE 1 SUMMARY -- random split 80/20 on FULL dataset")
print("="*90)
summary = pd.DataFrame(results).set_index("model")
print(summary.round(4))
summary.to_csv(RESULTS_PATH)
print(f"\nSaved -> {RESULTS_PATH}")
