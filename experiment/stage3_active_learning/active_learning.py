"""
Member 2 -- Active learning with entropy-based uncertainty sampling.

The setup mirrors a realistic scenario: a model trained on yesterday's data
(Monday) needs to handle today's traffic (Tuesday), which has new attack types
it hasn't seen. Instead of labeling all of Tuesday, we use active learning to
pick the most informative samples to label and retrain on.

Procedure:
  1. Train Logistic Regression on Monday (the "old" data).
  2. Predict probabilities on Tuesday (the "new" data).
  3. Compute entropy of each Tuesday prediction. High entropy = model is
     uncertain about that sample. Entropy for a binary classifier is:
        H = -p*log(p) - (1-p)*log(1-p)
     and peaks at p=0.5 (maximum uncertainty).
  4. Pick the top-N highest-entropy Tuesday samples. These are the ones the
     model would benefit most from seeing the true labels of.
  5. Add those N samples to the training set, retrain.
  6. Evaluate the new model on the remaining (held-out) Tuesday samples.

We compare three conditions:
  (a) Baseline: Monday-only model evaluated on Tuesday (no AL)
  (b) Active learning: Monday + N entropy-picked Tuesday samples, evaluated
      on the remaining Tuesday samples
  (c) Random sampling: Monday + N randomly-picked Tuesday samples, evaluated
      on the remaining Tuesday samples (this is the control -- shows that
      AL's gain isn't just from "more data")

LR is the natural choice here because it gives calibrated probability outputs
and retrains in seconds, making the AL loop fast.

Reference: https://arindam-dey.medium.com/a-gentle-introduction-to-active-learning-e983b9d175cb
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------- config ----------------
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
SPLITS_DIR    = PROJECT_ROOT / "data" / "splits"
RESULTS_PATH  = Path(__file__).resolve().with_name("active_learning_results.csv")
RANDOM_STATE  = 42
N_TO_LABEL    = 500   # how many Tuesday samples we "label" and add to training
CLASS_NAMES   = ["normal", "attack"]
EPS           = 1e-12 # small constant to avoid log(0)

# ---------------- 1. load the Tuesday/Wednesday split saved by member2_temporal_split.py ----------------
print("Loading Tuesday/Wednesday split from saved .npy files...")
X_train = np.load(SPLITS_DIR / "x_train_tue.npy")
y_train = np.load(SPLITS_DIR / "y_train_tue.npy")
X_test  = np.load(SPLITS_DIR / "x_test_wed.npy")
y_test  = np.load(SPLITS_DIR / "y_test_wed.npy")

print(f"  Tuesday (train): {X_train.shape[0]:,} rows   "
      f"normal: {(y_train==0).sum():,}   attack: {(y_train==1).sum():,}")
print(f"  Wednesday (pool): {X_test.shape[0]:,} rows   "
      f"normal: {(y_test==0).sum():,}   attack: {(y_test==1).sum():,}")

# ---------------- 2. scale ----------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------- evaluation helper ----------------
def evaluate(name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    print(f"\n{name}")
    print(f"  accuracy: {accuracy_score(y_eval, y_pred):.4f}")
    print("  classification report:")
    print(classification_report(y_eval, y_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print(f"  confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_eval, y_pred)}")
    rep = classification_report(y_eval, y_pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    return {
        "scenario":         name,
        "accuracy":         rep["accuracy"],
        "attack_precision": rep["attack"]["precision"],
        "attack_recall":    rep["attack"]["recall"],
        "attack_f1":        rep["attack"]["f1-score"],
        "macro_f1":         rep["macro avg"]["f1-score"],
    }

results = []

# =========================================================================
# (a) BASELINE: train on Monday, evaluate on ALL of Tuesday
# =========================================================================
print("\n" + "="*70)
print(f"(a) BASELINE -- Monday-only model evaluated on all of Tuesday")
print("="*70)
t0 = time.time()
lr_base = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=RANDOM_STATE)
lr_base.fit(X_train_s, y_train)
print(f"  fit: {time.time()-t0:.1f}s")
results.append(evaluate("(a) baseline: Monday-only -> Tuesday", lr_base,
                        X_test_s, y_test))

# =========================================================================
# (b) ACTIVE LEARNING: pick high-entropy Tuesday samples, add to training
# =========================================================================
print("\n" + "="*70)
print(f"(b) ACTIVE LEARNING -- add top-{N_TO_LABEL} most-uncertain Tuesday samples")
print("="*70)

# Get probability predictions on Tuesday
proba = lr_base.predict_proba(X_test_s)[:, 1]  # P(attack)

# Binary entropy: H = -p*log(p) - (1-p)*log(1-p), peaks at p=0.5
entropy = -(proba * np.log(proba + EPS) + (1 - proba) * np.log(1 - proba + EPS))
print(f"  entropy range: min={entropy.min():.4f}  max={entropy.max():.4f}  "
      f"mean={entropy.mean():.4f}")

# Pick the N most-uncertain Tuesday samples
al_idx       = np.argsort(entropy)[-N_TO_LABEL:]   # highest-entropy indices
remaining_idx = np.setdiff1d(np.arange(len(X_test_s)), al_idx)

print(f"  labeling {N_TO_LABEL} Tuesday samples -> retraining")
print(f"  of those {N_TO_LABEL}, {(y_test[al_idx]==1).sum()} are attacks, "
      f"{(y_test[al_idx]==0).sum()} are normal")

# Augment training set with the newly labeled Tuesday samples
X_train_al = np.vstack([X_train_s, X_test_s[al_idx]])
y_train_al = np.concatenate([y_train, y_test[al_idx]])

# Retrain
t0 = time.time()
lr_al = LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=RANDOM_STATE)
lr_al.fit(X_train_al, y_train_al)
print(f"  retrain fit: {time.time()-t0:.1f}s")

# Evaluate on the REMAINING Tuesday samples (the ones we didn't label)
results.append(evaluate(f"(b) AL: Monday + {N_TO_LABEL} uncertain Tuesday",
                        lr_al, X_test_s[remaining_idx], y_test[remaining_idx]))

# =========================================================================
# (c) CONTROL: same N samples but picked RANDOMLY (not by entropy)
# This isolates the effect of "smart selection" vs just "more data".
# =========================================================================
print("\n" + "="*70)
print(f"(c) RANDOM CONTROL -- add {N_TO_LABEL} randomly-picked Tuesday samples")
print("="*70)

rng = np.random.RandomState(RANDOM_STATE)
rand_idx          = rng.choice(len(X_test_s), size=N_TO_LABEL, replace=False)
remaining_idx_ctl = np.setdiff1d(np.arange(len(X_test_s)), rand_idx)

print(f"  randomly labeling {N_TO_LABEL} Tuesday samples -> retraining")
print(f"  of those {N_TO_LABEL}, {(y_test[rand_idx]==1).sum()} are attacks, "
      f"{(y_test[rand_idx]==0).sum()} are normal")

X_train_rand = np.vstack([X_train_s, X_test_s[rand_idx]])
y_train_rand = np.concatenate([y_train, y_test[rand_idx]])

t0 = time.time()
lr_rand = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=RANDOM_STATE)
lr_rand.fit(X_train_rand, y_train_rand)
print(f"  retrain fit: {time.time()-t0:.1f}s")

results.append(evaluate(f"(c) RANDOM: Monday + {N_TO_LABEL} random Tuesday",
                        lr_rand, X_test_s[remaining_idx_ctl],
                        y_test[remaining_idx_ctl]))

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("ACTIVE LEARNING SUMMARY")
print("="*70)
summary = pd.DataFrame(results).set_index("scenario")
print(summary.round(4))
summary.to_csv(RESULTS_PATH)
print(f"\nSaved -> {RESULTS_PATH}")

print("""
Key comparison:
  - (a) shows how badly the Monday-only model fails on Tuesday's new attacks
  - (b) shows the improvement from entropy-based sample selection
  - (c) controls for "did AL help, or was random sampling enough?"
  If (b) > (c), entropy-based AL is doing real work, not just adding data.
""")
