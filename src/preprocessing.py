"""
preprocessing.py
----------------
Handles stratified splitting of the isiZulu ( isolezwe) authorship dataset.

Usage (from project root):
    python src/preprocessing.py

Outputs:
    data/splits/train.csv ( 70 % of data)
    data/splits/val.csv ( 15 % of data)
    data/splits/test.csv ( 15 % of data)
"""

#Importing necessary libraries and modules for data manipulation, splitting, and file handling.
import os
import json
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?utm_source=chatgpt.com

# ── Paths Configs ────────────────────────────────────────────────────────────────────
RAW_CSV    = os.path.join("data", "raw", "isizulu_authors_dataset_cleaned.csv")
SPLITS_DIR = os.path.join("data", "splits")
METRICS_DIR = os.path.join("outputs", "metrics")

#This insures reproducibility of the splits by setting a fixed random state. Adjusting this value will yield different splits, which can be useful for experimentation, but for the final version we want a consistent split to report results on.
RANDOM_STATE = 42
TEST_SIZE    = 0.15   # 15 % test
VAL_SIZE     = 0.15   # 15 % val  (taken from the 30 % remainder)


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and validate required columns exist."""
    df = pd.read_csv(path)
    required = {"author", "text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    print(f"[load]  Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def stratified_split(df: pd.DataFrame,
                     test_size: float  = TEST_SIZE,
                     val_size: float   = VAL_SIZE,
                     random_state: int = RANDOM_STATE):
    """
    Two-stage stratified split:
        1. Split df → train (70 %) + temp (30 %)
        2. Split temp → val (15 %) + test (15 %)

    Stratification is on the 'author' column so every author
    appears in every split at the same relative proportion.
    """

    #df = df.groupby("author").filter(lambda x: len(x) >= 5)
    #print(df["author"].value_counts())

    # Stage 1
    temp_frac = test_size + val_size        # 0.30
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_frac,
        random_state=random_state,
        #Stratification is a technique used to ensure that the distribution of classes (in this case, authors) is preserved in both the training and validation/test sets. By stratifying on the 'author' column, we ensure that each split contains a representative proportion of articles from each author, which is crucial for training a model that can generalize well across all authors.
        stratify=df["author"]
    )

    # Stage 2 – val is exactly half of temp (15 / 30 = 0.50)
    val_frac = val_size / temp_frac         # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["author"]
    )

    return train_df, val_df, test_df


def verify_split(train_df, val_df, test_df, original_df):
    """
    Print a verification table and check that:
      - Every author appears in every split.
      - Relative proportions are consistent.
      - No article appears in more than one split.
    """
    print("\n" + "=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)
    print(f"{'Split':<12} {'Articles':>8}  {'% of total':>10}")
    total = len(original_df)

    #It prints dataset sizes and their percentages of the total dataset, providing a quick check that the splits are approximately 70/15/15 as intended.
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"{name:<12} {len(df):>8}  {len(df)/total*100:>9.1f}%")
    print(f"{'TOTAL':<12} {total:>8}  {'100.0%':>10}")

    print("\n── Author distribution per split ──")
    #Author distribution check
    dist = pd.DataFrame({
        "Train": train_df["author"].value_counts(),
        "Val":   val_df["author"].value_counts(),
        "Test":  test_df["author"].value_counts(),
    }).fillna(0).astype(int)
    print(dist.to_string())

    # Check every author in every split
    authors_all   = set(original_df["author"].unique())
    authors_train = set(train_df["author"].unique())
    authors_val   = set(val_df["author"].unique())
    authors_test  = set(test_df["author"].unique())

    missing_val  = authors_all - authors_val
    missing_test = authors_all - authors_test
    if missing_val or missing_test:
        print(f"\n⚠  WARNING: authors missing from val={missing_val}, test={missing_test}")
        print("   With only 11 articles for Simangaliso Ntshangase this can happen.")
        print("   If it occurs, try RANDOM_STATE=0 or 1 until all authors appear.")
    else:
        print("\n✓  All 7 authors present in every split.")

    # Check no overlap (using URL as a unique key)
    ids_train = set(train_df["url"])
    ids_val   = set(val_df["url"])
    ids_test  = set(test_df["url"])
    overlap = (ids_train & ids_val) | (ids_train & ids_test) | (ids_val & ids_test)
    if overlap:
        print(f"\n⚠  OVERLAP DETECTED: {len(overlap)} articles appear in multiple splits!")
    else:
        print("✓  Zero overlap between splits (data leakage check passed).")

    print("=" * 60)


def save_splits(train_df, val_df, test_df, splits_dir: str):
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(splits_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(splits_dir, "test.csv"),  index=False)
    print(f"\n✓  Splits saved to {splits_dir}/")


def save_split_metadata(train_df, val_df, test_df, metrics_dir: str):
    """Save split sizes and author counts to JSON for reproducibility."""
    os.makedirs(metrics_dir, exist_ok=True)
    meta = {
        "random_state": RANDOM_STATE,
        "train_size": len(train_df),
        "val_size":   len(val_df),
        "test_size":  len(test_df),
        "author_counts": {
            "train": train_df["author"].value_counts().to_dict(),
            "val":   val_df["author"].value_counts().to_dict(),
            "test":  test_df["author"].value_counts().to_dict(),
        }
    }
    path = os.path.join(metrics_dir, "split_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓  Split metadata saved to {path}")


def run():
    df = load_data(RAW_CSV)

    #Filtering out authors min samples to be 20 
    #This is a common preprocessing step to ensure that each class (author) has enough samples for the model to learn from. With only 11 articles for Simangaliso Ntshangase, we might want to set a higher threshold to ensure better model performance, but this will reduce the number of authors in the dataset. Adjusting this threshold will be a trade-off between having more classes and having enough data per class.
    #But should I remove this filter ? Do we have enough data for all authors ? If we set min_samples to 20, we will lose Simangaliso Ntshangase who has only 11 articles. This will reduce our dataset to 6 authors instead of 7. We need to decide if we want to keep all 7 authors with some having very few samples, or if we want to ensure a minimum number of samples per author at the cost of losing one author. Given that we have a small dataset, it might be better to keep all authors and deal with the imbalance in other ways (e.g., data augmentation, class weighting) rather than removing an entire author from the dataset.
    min_samples = 20 

    df = df.groupby("author").filter(lambda x: len(x) >= min_samples)
    print(f"\nAfter filtering authors with < {min_samples} samples:")   


    print(f"[filter] Dataset reduced to {len(df)} rows after removing small authors")

    train_df, val_df, test_df = stratified_split(df)
    verify_split(train_df, val_df, test_df, df)
    save_splits(train_df, val_df, test_df, SPLITS_DIR)
    save_split_metadata(train_df, val_df, test_df, METRICS_DIR)

    return train_df, val_df, test_df


if __name__ == "__main__":
    run()