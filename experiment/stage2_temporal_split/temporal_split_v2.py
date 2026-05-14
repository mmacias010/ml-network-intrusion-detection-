"""
Member 3 -- Stage 2: Neural Network on temporal split (Tuesday -> Wednesday).
FIXED: now includes sanity-check assertions that confirm the loaded files
actually contain Tuesday/Wednesday data, not Stage 1 random-split data.

Files expected in the same folder:
  X_train_tue.npy   (Tuesday training data, ~445k rows)
  y_train_tue.npy
  X_test_wed.npy    (Wednesday test data, ~691k rows)
  y_test_wed.npy

Outputs:
  member3_temporal_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_PATH = Path(__file__).resolve().with_name("temporal_results_v2.csv")

RANDOM_STATE = 42
CLASS_NAMES  = ["normal", "attack"]

X_TRAIN_PATH = SPLITS_DIR / "x_train_tue.npy"
Y_TRAIN_PATH = SPLITS_DIR / "y_train_tue.npy"
X_TEST_PATH  = SPLITS_DIR / "x_test_wed.npy"
Y_TEST_PATH  = SPLITS_DIR / "y_test_wed.npy"

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
# LOAD + SANITY CHECK
# =========================================================================
print("Setup: tuesday_wednesday\n")
print("Loading .npy files...")

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH).astype(int)
X_test  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH).astype(int)

print(f"  Train (Tuesday)  -> {X_train.shape[0]:,} rows   "
      f"normal: {(y_train==0).sum():,}   attack: {(y_train==1).sum():,}")
print(f"  Test  (Wednesday) -> {X_test.shape[0]:,} rows   "
      f"normal: {(y_test==0).sum():,}   attack: {(y_test==1).sum():,}")

# FIX: assertions that will fail loudly if the wrong files are loaded.
# Tuesday training should be ~445k rows; Wednesday test should be ~691k rows.
# If you see an AssertionError below, the .npy files in your folder are wrong.

expected_train_rows = 445_645
expected_test_rows  = 691_406
tolerance           = 0.05  # 5% tolerance for any small variance

def _ok(actual, expected):
    return abs(actual - expected) / expected < tolerance

assert _ok(X_train.shape[0], expected_train_rows), (
    f"\nTRAINING FILE LOOKS WRONG.\n"
    f"  Got {X_train.shape[0]:,} rows, expected ~{expected_train_rows:,}.\n"
    f"  Tuesday should have ~445k rows. You may be loading the wrong file.\n"
    f"  Ask Miranda to re-share X_train_tue.npy / y_train_tue.npy."
)

assert _ok(X_test.shape[0], expected_test_rows), (
    f"\nTEST FILE LOOKS WRONG.\n"
    f"  Got {X_test.shape[0]:,} rows, expected ~{expected_test_rows:,}.\n"
    f"  Wednesday should have ~691k rows with ~252k attacks.\n"
    f"  You may have loaded the Stage 1 random-split test file instead.\n"
    f"  Ask Miranda to re-share X_test_wed.npy / y_test_wed.npy."
)

print("\n  Sanity check passed: row counts match expected Tuesday/Wednesday sizes.\n")

# =========================================================================
# SCALE (fit on train)
# =========================================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =========================================================================
# CLASS WEIGHTS
# =========================================================================
classes, counts = np.unique(y_train, return_counts=True)
weight_map     = {c: len(y_train) / (len(classes) * cnt)
                  for c, cnt in zip(classes, counts)}
sample_weights = np.array([weight_map[y] for y in y_train])

# =========================================================================
# TRAIN
# =========================================================================
print("Training Neural Network (128-64-32, relu, adam)...")
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
print("TEMPORAL SPLIT SUMMARY -- setup: tuesday_wednesday")
print("="*90)
summary = pd.DataFrame(results).set_index("model")
print(summary.round(4))
summary.to_csv(RESULTS_PATH)
print(f"\nSaved -> {RESULTS_PATH}")
print(f"Share {RESULTS_PATH.name} with the team for the 4-way comparison!")
