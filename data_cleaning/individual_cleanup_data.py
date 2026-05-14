from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "original"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEAN_PREFIX = "clean_"
MISSING_COLUMN_THRESHOLD = 0.50


def clean_csv(input_path: Path) -> Path:
    """Clean one CIC IDS CSV file and save it with clean_ in front."""
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEANED_DATA_DIR / f"{CLEAN_PREFIX}{input_path.name}"

    df = pd.read_csv(input_path, low_memory=False)
    rows_before, cols_before = df.shape

    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        df[column] = df[column].str.strip()

    df.replace(
        [np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"],
        np.nan,
        inplace=True,
    )

    minimum_non_missing = int(len(df) * MISSING_COLUMN_THRESHOLD)
    df.dropna(axis=1, thresh=minimum_non_missing, inplace=True)
    df.dropna(axis=0, inplace=True)

    df.to_csv(output_path, index=False)

    print(
        f"{input_path.name}: "
        f"{rows_before:,} rows x {cols_before:,} cols -> "
        f"{len(df):,} rows x {len(df.columns):,} cols"
    )
    print(f"  saved {output_path}")

    return output_path


def main() -> None:
    csv_files = sorted(
        path
        for path in ORIGINAL_DATA_DIR.glob("*.csv")
        if not path.name.startswith(CLEAN_PREFIX)
    )

    if not csv_files:
        print(f"No original CSV files found in {ORIGINAL_DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files to clean.")
    for csv_file in csv_files:
        clean_csv(csv_file)


if __name__ == "__main__":
    main()
