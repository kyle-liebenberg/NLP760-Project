"""
src / preprocessing.py

Handles text chunking data augmentation and stratified splitting of the 
expanded isiZulu authorship dataset.
"""

import os
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Paths Configs 
RAW_CSV    = os.path.join("data", "raw", "isizulu_authors_dataset.csv")
SPLITS_DIR = os.path.join("data", "splits")
METRICS_DIR = os.path.join("outputs", "metrics")

RANDOM_STATE = 42
TEST_SIZE    = 0.15   # 15% test
VAL_SIZE     = 0.15   # 15% val  (taken from the 30% remainder)

# Chunking Configuration for Data Augmentation
CHUNK_SIZE   = 150    # Words per chunk
CHUNK_OVERLAP = 30    # Words overlapping between adjacent chunks

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data and validate required columns exist"""
    df = pd.read_csv(path)
    required = {"author", "text", "url"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    print(f"[load]  Loaded {len(df)} raw rows.")
    return df

def chunk_text_data(df: pd.DataFrame, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> pd.DataFrame:
    """
    Applies data augmentation by slicing long articles into smaller overlapping chunks.
    This expands the training footprint for the deep learning architecture.
    """
    chunked_records = []
    print(f"[augment] Slicing text into chunks of {chunk_size} words (overlap={overlap})...")
    
    for _, row in df.iterrows():
        author = row['author']
        url = row['url']
        words = str(row['text']).split()
        
        # If an article is too short to even fill one chunk, keep it as is
        if len(words) <= chunk_size:
            chunked_records.append({'author': author, 'url': url, 'text': " ".join(words)})
            continue
            
        # Sliding window implementation
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            
            # Drop very tiny tail end fragments (less than 40 words) to minimize noise
            if len(chunk_words) < 40:
                break
                
            chunk_text = " ".join(chunk_words)
            chunked_records.append({
                'author': author, 
                'url': f"{url}#chunk{chunk_idx}", 
                'text': chunk_text
            })
            
            start += (chunk_size - overlap)
            chunk_idx += 1
            
    augmented_df = pd.DataFrame(chunked_records)
    print(f"[augment] Expanded dataset from {len(df)} articles to {len(augmented_df)} chunks!")
    return augmented_df

def stratified_split(df: pd.DataFrame, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    """Two-stage stratified split to preserve author distribution across train, val, and test sets"""
    temp_frac = test_size + val_size        # 0.30
    train_df, holdout_df = train_test_split(
        df,
        test_size=temp_frac,
        random_state=random_state,
        stratify=df["author"]
    )

    val_frac = val_size / temp_frac         # 0.50
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=val_frac,
        random_state=random_state,
        stratify=holdout_df["author"]
    )

    return train_df, val_df, test_df

def verify_split(train_df, val_df, test_df, original_df):
    print("\n" + "=" * 60)
    print("SPLIT VERIFICATION (AUGMENTED CORPUS)")
    print("=" * 60)
    print(f"{'Split':<12} {'Chunks':>8}  {'% of total':>10}")
    total = len(original_df)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"{name:<12} {len(df):>8}  {len(df)/total*100:>9.1f}%")
    print(f"{'TOTAL':<12} {total:>8}  {'100.0%':>10}")

    print("\n── Author chunk distribution per split ──")
    author_distribution_table = pd.DataFrame({
        "Train": train_df["author"].value_counts(),
        "Val":   val_df["author"].value_counts(),
        "Test":  test_df["author"].value_counts(),
    }).fillna(0).astype(int)
    print(author_distribution_table.to_string())
    print("=" * 60)

def save_splits(train_df, val_df, test_df, splits_dir: str):
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(splits_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(splits_dir, "test.csv"),  index=False)
    print(f"\n-- Splits saved to {splits_dir}/")

def save_split_metadata(train_df, val_df, test_df, metrics_dir: str):
    os.makedirs(metrics_dir, exist_ok=True)
    meta = {
        "random_state": RANDOM_STATE,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "train_chunks": len(train_df),
        "val_chunks":   len(val_df),
        "test_chunks":  len(test_df),
    }
    path = os.path.join(metrics_dir, "split_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Split metadata saved to {path}")

def run():
    df = load_data(RAW_CSV)
    # Perform chunking augmentation first
    augmented_df = chunk_text_data(df)
    
    train_df, val_df, test_df = stratified_split(augmented_df)
    verify_split(train_df, val_df, test_df, augmented_df)
    save_splits(train_df, val_df, test_df, SPLITS_DIR)
    save_split_metadata(train_df, val_df, test_df, METRICS_DIR)

if __name__ == "__main__":
    run()