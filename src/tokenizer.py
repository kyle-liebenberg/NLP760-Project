"""
tokenizer.py
------------
Trains a Byte-Pair Encoding (BPE) tokenizer on the TRAINING split only
(preventing data leakage), then encodes all three splits and pads them
to a fixed sequence length determined from the 90th percentile of
training lengths.

Usage (from project root):
    python src/tokenizer.py

Outputs:
    outputs/bpe_tokenizer/tokenizer.json
    outputs/figures/sequence_length_histogram.png
    data/splits/X_train.npy  X_val.npy  X_test.npy
    data/splits/y_train.npy  y_val.npy  y_test.npy
    data/splits/label_classes.json
    outputs/metrics/tokenizer_metadata.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ── Paths ─────────────────────────────────────────────────────────────────
SPLITS_DIR    = os.path.join("data", "splits")
TOKENIZER_DIR = os.path.join("outputs", "bpe_tokenizer")
FIGURES_DIR   = os.path.join("outputs", "figures")
METRICS_DIR   = os.path.join("outputs", "metrics")

TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")

# ── Hyper-parameters ───────────────────────────────────────────────────────
VOCAB_SIZE     = 2000   # small vocab → forces subword learning on small corpus
MIN_FREQUENCY  = 2      # ignore tokens appearing only once in training set
PERCENTILE     = 90     # use 90th percentile of lengths as MAX_SEQ_LEN
RANDOM_STATE   = 42


def load_splits():
    train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    print(f"[load]  train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def train_bpe_tokenizer(train_texts: list) -> Tokenizer:
    """
    Train BPE on training texts only.

    Why train-only?
    ---------------
    If we train the tokenizer on all 157 articles and then split, the
    tokenizer has 'seen' validation and test vocabulary, which is a form
    of data leakage. The tokenizer must learn from training data only,
    exactly as the model will.
    """
    print(f"\n[BPE]   Training tokenizer on {len(train_texts)} articles …")
    print(f"        vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQUENCY}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"✓  Tokenizer saved → {TOKENIZER_PATH}")


def encode_split(texts: list, tokenizer: Tokenizer) -> list:
    """Return list of integer-id sequences (variable length)."""
    return [tokenizer.encode(text).ids for text in texts]


def analyse_lengths(train_seqs: list, percentile: int = PERCENTILE) -> int:
    """
    Plot histogram of tokenized training-sequence lengths and return
    the chosen MAX_SEQ_LEN (= percentile-th percentile).
    """
    lengths = [len(s) for s in train_seqs]
    max_len = int(np.percentile(lengths, percentile))

    print(f"\n[lengths]  min={min(lengths)}, mean={np.mean(lengths):.0f}, "
          f"max={max(lengths)}, {percentile}th pct={max_len}")
    print(f"           → MAX_SEQ_LEN set to {max_len}")

    # Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=30, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(max_len, color="#E8593C", linewidth=2,
               linestyle="--", label=f"{percentile}th pct = {max_len} tokens")
    ax.axvline(np.mean(lengths), color="#2ECC71", linewidth=1.5,
               linestyle=":", label=f"mean = {np.mean(lengths):.0f} tokens")
    ax.set_xlabel("Sequence length (BPE tokens)", fontsize=12)
    ax.set_ylabel("Number of articles", fontsize=12)
    ax.set_title("Training-set tokenized sequence lengths (BPE, vocab=2000)", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "sequence_length_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓  Histogram saved → {out}")
    return max_len


def pad_and_truncate(sequences: list, max_len: int) -> np.ndarray:
    """
    Manual post-padding to avoid TensorFlow import just for padding.
    Equivalent to tf.keras.preprocessing.sequence.pad_sequences
    with padding='post', truncating='post', value=0.
    """
    result = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]
    return result


def encode_labels(train_df, val_df, test_df):
    """Fit LabelEncoder on training authors and transform all splits."""
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["author"])
    y_val   = le.transform(val_df["author"])
    y_test  = le.transform(test_df["author"])
    print(f"\n[labels]  Classes: {list(le.classes_)}")
    return le, y_train, y_val, y_test


def show_tokenization_examples(train_df, tokenizer, n=3):
    """
    Print example tokenizations so you can verify BPE is splitting
    isiZulu morphemes (prefixes, suffixes) correctly.
    """
    print("\n" + "=" * 60)
    print("BPE TOKENIZATION EXAMPLES")
    print("=" * 60)
    for author in train_df["author"].unique()[:n]:
        text = train_df[train_df["author"] == author]["text"].iloc[0]
        snippet = " ".join(text.split()[:15])  # first 15 words
        encoded  = tokenizer.encode(snippet)
        tokens   = encoded.tokens
        ids      = encoded.ids
        print(f"\nAuthor : {author}")
        print(f"Input  : {snippet}")
        print(f"Tokens : {tokens}")
        print(f"IDs    : {ids}")
    print("=" * 60)


def save_arrays(X_train, X_val, X_test, y_train, y_val, y_test,
                le, max_len, tokenizer):
    """Save all numpy arrays and metadata."""
    np.save(os.path.join(SPLITS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SPLITS_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(SPLITS_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SPLITS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SPLITS_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(SPLITS_DIR, "y_test.npy"),  y_test)
    print(f"\n✓  Arrays saved to {SPLITS_DIR}/")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val   shape: {X_val.shape}")
    print(f"   X_test  shape: {X_test.shape}")

    # Label classes JSON (needed by Diya for the CNN-LSTM)
    label_path = os.path.join(SPLITS_DIR, "label_classes.json")
    with open(label_path, "w") as f:
        json.dump(list(le.classes_), f, indent=2)
    print(f"✓  Label classes saved → {label_path}")

    # Tokenizer metadata
    os.makedirs(METRICS_DIR, exist_ok=True)
    meta = {
        "vocab_size":    VOCAB_SIZE,
        "min_frequency": MIN_FREQUENCY,
        "max_seq_len":   max_len,
        "percentile":    PERCENTILE,
        "actual_vocab":  tokenizer.get_vocab_size(),
        "num_classes":   len(le.classes_),
        "classes":       list(le.classes_),
        "X_train_shape": list(X_train.shape),
        "X_val_shape":   list(X_val.shape),
        "X_test_shape":  list(X_test.shape),
    }
    meta_path = os.path.join(METRICS_DIR, "tokenizer_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓  Tokenizer metadata saved → {meta_path}")


def run():
    train_df, val_df, test_df = load_splits()

    # --- BPE training (train split only) ---
    tokenizer = train_bpe_tokenizer(train_df["text"].tolist())
    save_tokenizer(tokenizer)

    # --- Encode ---
    train_seqs = encode_split(train_df["text"].tolist(), tokenizer)
    val_seqs   = encode_split(val_df["text"].tolist(),   tokenizer)
    test_seqs  = encode_split(test_df["text"].tolist(),  tokenizer)

    # --- Length analysis and MAX_SEQ_LEN decision ---
    max_len = analyse_lengths(train_seqs)

    # --- Padding ---
    X_train = pad_and_truncate(train_seqs, max_len)
    X_val   = pad_and_truncate(val_seqs,   max_len)
    X_test  = pad_and_truncate(test_seqs,  max_len)

    # --- Labels ---
    le, y_train, y_val, y_test = encode_labels(train_df, val_df, test_df)

    # --- Show examples ---
    show_tokenization_examples(train_df, tokenizer)

    # --- Save everything ---
    save_arrays(X_train, X_val, X_test, y_train, y_val, y_test,
                le, max_len, tokenizer)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, max_len, tokenizer


if __name__ == "__main__":
    run()