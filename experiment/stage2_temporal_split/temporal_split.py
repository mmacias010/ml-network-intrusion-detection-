"""
Member 2 -- Temporal split analysis for CIC-IDS2017.
Train on Monday data, test on Tuesday data (or whatever split the professor
confirms).

Handles the per-day raw files Jocelyn sent:
  - clean_Monday-WorkingHours_pcap_ISCX.csv  (529,481 rows, all BENIGN)
  - clean_Tuesday-WorkingHours_pcap_ISCX.csv (445,645 rows; 13,832 attacks)

Both files have a 'Label' column with string values (BENIGN, FTP-Patator,
SSH-Patator). We convert that to a binary Attack_Flag (0=BENIGN, 1=anything else)
to match Member 2's existing scripts.

NOTE: Monday is 100% benign. A Monday-only training set cannot teach the
classifier the attack class. Set SETUP below based on professor's response:
  - "monday_plus_tuesday_morning" -> train on Monday + first chunk of Tuesday
  - "tuesday_wednesday"            -> use Tuesday->Wednesday instead (will need
                                       Wednesday file too)
  - "mixed_pool"                   -> combine Monday+Tuesday, shuffle, split
                                       (this is NOT temporal but is a fallback)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================================
# CONFIG -- edit SETUP based on what the professor says
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_PATH = Path(__file__).resolve().with_name("temporal_results.csv")

SETUP = "tuesday_wednesday"   # train on Tuesday, test on Wednesday

MONDAY_PATH    = CLEANED_DATA_DIR / "clean_Monday-WorkingHours.pcap_ISCX.csv"
TUESDAY_PATH   = CLEANED_DATA_DIR / "clean_Tuesday-WorkingHours.pcap_ISCX.csv"
WEDNESDAY_PATH = CLEANED_DATA_DIR / "clean_Wednesday-workingHours.pcap_ISCX.csv"

TUESDAY_MORNING_FRAC = 0.20   # what fraction of Tuesday goes into training
                              # (used only for the "monday_plus_tuesday_morning" setup)

LABEL_COL    = "Label"
TARGET_COL   = "Attack_Flag"
RANDOM_STATE = 42
KNN_TRAIN_CAP = 100_000
CLASS_NAMES  = ["normal", "attack"]

# =========================================================================
# HELPERS
# =========================================================================
def load_and_label(path):
    """Load a per-day CSV and convert Label (string) -> Attack_Flag (0/1)."""
    print(f"  Loading {path.name}...")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset file: {path}\n"
            "Place the full cleaned CSV dataset files in data/cleaned/."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # CIC files often have leading spaces

    # Clean inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Convert string Label to binary Attack_Flag
    df[TARGET_COL] = (df[LABEL_COL].str.strip().str.upper() != "BENIGN").astype(int)
    df = df.drop(columns=[LABEL_COL])

    # Drop any non-numeric remaining columns
    non_num = df.drop(columns=[TARGET_COL]).select_dtypes(exclude=np.number).columns
    if len(non_num) > 0:
        print(f"    dropping non-numeric: {list(non_num)}")
        df = df.drop(columns=list(non_num))

    n_a = (df[TARGET_COL] == 1).sum()
    n_b = (df[TARGET_COL] == 0).sum()
    print(f"    rows: {len(df):,}   benign: {n_b:,}   attack: {n_a:,}")
    return df


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
# BUILD TRAIN / TEST BASED ON SETUP
# =========================================================================
print(f"Setup: {SETUP}\n")
print("Loading day files...")

if SETUP == "monday_plus_tuesday_morning":
    # Train: all of Monday (benign-only) + first X% of Tuesday
    # Test:  the rest of Tuesday
    df_mon = load_and_label(MONDAY_PATH)
    df_tue = load_and_label(TUESDAY_PATH)

    # Sort Tuesday so "morning" really means earliest rows (CIC files are
    # already in temporal order)
    split_idx = int(len(df_tue) * TUESDAY_MORNING_FRAC)
    df_tue_train = df_tue.iloc[:split_idx]
    df_tue_test  = df_tue.iloc[split_idx:]

    train_df = pd.concat([df_mon, df_tue_train], ignore_index=True)
    test_df  = df_tue_test.reset_index(drop=True)

    print(f"\n  Train -> Monday ({len(df_mon):,}) + Tuesday first {TUESDAY_MORNING_FRAC*100:.0f}% ({len(df_tue_train):,}) = {len(train_df):,}")
    print(f"  Test  -> Tuesday remaining {(1-TUESDAY_MORNING_FRAC)*100:.0f}% ({len(test_df):,})")

elif SETUP == "tuesday_wednesday":
    # Requires Wednesday file. Uncomment WEDNESDAY_PATH above.
    df_tue = load_and_label(TUESDAY_PATH)
    df_wed = load_and_label(WEDNESDAY_PATH)
    train_df, test_df = df_tue, df_wed

elif SETUP == "mixed_pool":
    # Fallback: not really temporal, just combines both and does a stratified
    # random split. Same as Stage 1 but on Mon+Tue data only.
    from sklearn.model_selection import train_test_split
    df_all = pd.concat([load_and_label(MONDAY_PATH),
                        load_and_label(TUESDAY_PATH)], ignore_index=True)
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, stratify=df_all[TARGET_COL],
        random_state=RANDOM_STATE
    )

else:
    raise ValueError(f"Unknown SETUP: {SETUP}")

# Align columns (Monday and Tuesday should match, but just in case)
common_cols = [c for c in train_df.columns if c in test_df.columns]
train_df = train_df[common_cols]
test_df  = test_df[common_cols]

X_train = train_df.drop(columns=[TARGET_COL]).values
y_train = train_df[TARGET_COL].values.astype(int)
X_test  = test_df.drop(columns=[TARGET_COL]).values
y_test  = test_df[TARGET_COL].values.astype(int)

print(f"\n  Train final -> {X_train.shape[0]:,} rows   "
      f"normal: {(y_train==0).sum():,}   attack: {(y_train==1).sum():,}")
print(f"  Test  final -> {X_test.shape[0]:,} rows   "
      f"normal: {(y_test==0).sum():,}   attack: {(y_test==1).sum():,}")

# Save for Stage 3
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
np.save(SPLITS_DIR / "x_train_tue.npy", X_train)
np.save(SPLITS_DIR / "y_train_tue.npy", y_train)
np.save(SPLITS_DIR / "x_test_wed.npy",  X_test)
np.save(SPLITS_DIR / "y_test_wed.npy",  y_test)
print(f"  saved -> {SPLITS_DIR / 'x_train_tue.npy'}, {SPLITS_DIR / 'y_train_tue.npy'}, {SPLITS_DIR / 'x_test_wed.npy'}, {SPLITS_DIR / 'y_test_wed.npy'}")

# =========================================================================
# SCALE + TRAIN + EVAL
# =========================================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = []

print("\nTraining Decision Tree (balanced)...")
t0 = time.time()
dt = DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
print(f"  fit: {time.time()-t0:.1f}s")
results.append(evaluate("Decision Tree (balanced)", dt, X_test, y_test))

print("\nTraining Logistic Regression (balanced)...")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
print(f"  fit: {time.time()-t0:.1f}s")
results.append(evaluate("Logistic Regression (balanced)", lr, X_test_s, y_test))

n_knn = min(KNN_TRAIN_CAP, len(X_train_s))
print(f"\nTraining kNN on {n_knn:,}-row subsample...")
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(X_train_s), size=n_knn, replace=False)
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_s[idx], y_train[idx])
print(f"  fit: {time.time()-t0:.1f}s")
results.append(evaluate("kNN (k=5)", knn, X_test_s, y_test))

print("\n" + "="*90)
print(f"TEMPORAL SPLIT SUMMARY -- setup: {SETUP}")
print("="*90)
summary = pd.DataFrame(results).set_index("model")
print(summary.round(4))
summary.to_csv(RESULTS_PATH)
print(f"\nSaved -> {RESULTS_PATH}")
