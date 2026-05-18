"""
src / preprocessing.py

Handles stratified splitting of the isiZulu (isolezwe newsletter) authorship dataset.

Outputs:
    data/splits/train.csv ( 70 % of data)
    data/splits/val.csv ( 15 % of data)
    data/splits/test.csv ( 15 % of data)
"""

#Importing necessary libraries and modules for data manipulation and splitting.
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#Paths Configs 
RAW_CSV    = os.path.join("data", "raw", "isizulu_authors_dataset_cleaned.csv")
SPLITS_DIR = os.path.join("data", "splits")
METRICS_DIR = os.path.join("outputs", "metrics")

#This insures reproducibility of the splits by setting a fixed random state. Adjusting this value will yield different splits, 
#which can be useful for experimentation, but for the final version we want a consistent split to report results on.
RANDOM_STATE = 42
TEST_SIZE    = 0.15   # 15 % test
VAL_SIZE     = 0.15   # 15 % val  (taken from the 30 % remainder)


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data and validate required columns exist"""
    df = pd.read_csv(path)
    required = {"author", "text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    print(f"[load]  Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df

#Ensures every author appears in train, val, and test 
# so that the model sees all writing styles during training.
def stratified_split(df: pd.DataFrame,
                     test_size: float  = TEST_SIZE,
                     val_size: float   = VAL_SIZE,
                     random_state: int = RANDOM_STATE):
    """
    Two-stage stratified split to preserve author distribution across train, val, and test sets:
        1. Split df → train (70 %) + temp (30 %)
        2. Split temp → val (15 %) + test (15 %)

        Inputs:
            df: DataFrame with 'author' column for stratification
            test_size: fraction of total data for test set (default 0.15)
            val_size: fraction of total data for val set (default 0.15)
            random_state: seed for reproducible splits (default 42)
        
        Outputs:
            train_df: 70 % of data, stratified by author
            val_df:   15 % of data, stratified by author
            test_df:  15 % of data, stratified by author

    Stratification is on the 'author' column so every author
    appears in EVERY split at the same relative proportion.
    """

    # Stage 1
    temp_frac = test_size + val_size        # 0.30
    train_df, holdout_df = train_test_split(
        df,
        test_size=temp_frac,
        random_state=random_state,
        stratify=df["author"]
    )

    # Stage 2 – val is exactly half of temp (15 / 30 = 0.50)
    val_frac = val_size / temp_frac         # 0.50
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=val_frac,
        random_state=random_state,
        stratify=holdout_df["author"]
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df), \
    "Rows were lost during splitting"


    assert set(train_df["author"]) == set(df["author"]), \
    "Some authors are missing from training set"

    return train_df, val_df, test_df

#Validates the the train/validation/test splits are safe for
#authorship attribution experiments by checking class coverage and data leakage between splits.
def verify_split(train_df, val_df, test_df, original_df):
    """
    Validates that the stratified train/validation/test split is safe
    for authorship attribution experiments.

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
    author_distribution_table = pd.DataFrame({
        "Train": train_df["author"].value_counts(),
        "Val":   val_df["author"].value_counts(),
        "Test":  test_df["author"].value_counts(),
    }).fillna(0).astype(int)
    print(author_distribution_table.to_string())

    # Check every author in every split
    expected_authors   = set(original_df["author"].unique())
    authors_train = set(train_df["author"].unique())
    authors_val   = set(val_df["author"].unique())
    authors_test  = set(test_df["author"].unique())

    missing_val  = expected_authors - authors_val
    missing_test = expected_authors - authors_test
    if missing_val or missing_test:
        print(f"\n WARNING: authors missing from val={missing_val}, test={missing_test}")
        print("   With only 11 articles for Simangaliso Ntshangase this can happen.")
        print("   If it occurs, try RANDOM_STATE=0 or 1 until all authors appear.")
    else:
        print("\n  All 7 authors present in every split.")
    
    
    # Check no overlap (using URL as a unique key)
    train_urls = set(train_df["url"])
    val_urls   = set(val_df["url"])
    test_urls  = set(test_df["url"])
    overlap = (train_urls & val_urls) | (train_urls & test_urls) | (val_urls & test_urls)


    if overlap:
        print(f"\nOVERLAP DETECTED: {len(overlap)} articles appear in multiple splits!")
    else:
        print("Zero overlap between splits (data leakage check passed).")

    print("=" * 60)
    #No rows lost
    assert len(train_df) + len(val_df) + len(test_df) == len(original_df), \
    "Some rows were lost during splitting"
    #No overlap
    assert len(overlap) == 0, \
    "Data leakage detected between splits"
    #Training must contain all authors
    assert authors_train == expected_authors, \
    "Some authors are missing from training set"

## Save fixed dataset partitions to guarantee reproducible
# experiments across tokenization and model training stages.

def save_splits(train_df, val_df, test_df, splits_dir: str):
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(splits_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(splits_dir, "test.csv"),  index=False)
    print(f"\n--  Splits saved to {splits_dir}/")




def save_split_metadata(train_df, val_df, test_df, metrics_dir: str):
    """Save split sizes and author counts to JSON for reproducibility """
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

    train_df, val_df, test_df = stratified_split(df)
    verify_split(train_df, val_df, test_df, df)
    save_splits(train_df, val_df, test_df, SPLITS_DIR)
    save_split_metadata(train_df, val_df, test_df, METRICS_DIR)

    return train_df, val_df, test_df


if __name__ == "__main__":
    run()