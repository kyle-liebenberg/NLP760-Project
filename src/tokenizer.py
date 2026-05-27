"""
tokenizer.py
------------
Trains a Byte-Pair Encoding (BPE) tokenizer on the TRAINING split only, 
then encodes all three splits and pads them to a fixed sequence length 
determined from the 90th percentile of training lengths.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

SPLITS_DIR    = os.path.join("data", "splits")
TOKENIZER_DIR = os.path.join("outputs", "bpe_tokenizer")
FIGURES_DIR   = os.path.join("outputs", "figures")
METRICS_DIR   = os.path.join("outputs", "metrics")

TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")

# HYPERPARAMETERS
# small vocabulary prevents memorizing entire isiZulu words and encourages meaningful subword learning on a tiny corpus.
VOCAB_SIZE     = 3000   
MIN_FREQUENCY  = 2 # ignore tokens appearing only once in training set
# 95th percentile balances:
# - reducing truncation of long articles
# - avoiding excessive padding overhead
PERCENTILE     = 95    
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
    """
    print(f"\n[BPE]   Training tokenizer on {len(train_texts)} articles …")
    print(f"        vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQUENCY}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # preserve word boundaries before BPE learns subword merges
    tokenizer.pre_tokenizer = Whitespace()

    bpe_trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        # [UNK] handles unseen subwords during inference
        # [PAD] enables fixed-length batching for neural networks
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=True,
    )

    # asserts before training
    assert len(train_texts) > 0, \
    "Training corpus is empty"
    assert all(isinstance(text, str) for text in train_texts), \
    "All training samples must be strings"

    tokenizer.train_from_iterator(train_texts, trainer=bpe_trainer)

    # asserts after training
    assert tokenizer.get_vocab_size() > 2, \
    "Tokenizer vocabulary was not learned correctly"

    return tokenizer

# save tokenizer state
def save_tokenizer(tokenizer: Tokenizer):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"  Tokenizer saved → {TOKENIZER_PATH}")

def encode_split(texts: list, tokenizer: Tokenizer) -> list:
    """Return list of integer-id sequences (variable length)."""
    return [tokenizer.encode(text).ids for text in texts]

def analyse_lengths(train_seqs: list, percentile: int = PERCENTILE) -> int:
    """
    Analyze tokenized sequence lengths and choose an efficient
    fixed maximum sequence length for neural-network batching.

    A percentile-based threshold is used instead of the absolute
    maximum to reduce excessive padding caused by outlier articles.

    Inputs:
        train_seqs : list of token-ID sequences
        percentile : percentile threshold for max sequence length

    Outputs:
        max_sequence_length : integer fixed sequence length

    Assumptions:
        - sequences are tokenized training samples
        - sequences contain integer token IDs
    """
    sequence_lengths = [len(s) for s in train_seqs]
    max_sequence_length = int(np.percentile(sequence_lengths, percentile))

    print(f"\n[lengths]  min={min(sequence_lengths)}, mean={np.mean(sequence_lengths):.0f}, "
          f"max={max(sequence_lengths)}, {percentile}th pct={max_sequence_length}")
    print(f"           → MAX_SEQ_LEN set to {max_sequence_length}")

    # visualize the distribution of sequence lengths to confirm the percentile choice
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(sequence_lengths, bins=30, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(max_sequence_length, color="#E8593C", linewidth=2,
               linestyle="--", label=f"{percentile}th pct = {max_sequence_length} tokens")
    ax.axvline(np.mean(sequence_lengths), color="#2ECC71", linewidth=1.5,
               linestyle=":", label=f"mean = {np.mean(sequence_lengths):.0f} tokens")
    ax.set_xlabel("Sequence length (BPE tokens)", fontsize=12)
    ax.set_ylabel("Number of articles", fontsize=12)
    ax.set_title("Training-set tokenized sequence lengths (BPE, vocab=3000)", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "sequence_length_histogram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Histogram saved → {out}")

    
    assert len(train_seqs) > 0, \
    "No training sequences provided"

    assert 0 < percentile <= 100, \
    "Percentile must be between 1 and 100"

    assert all(isinstance(token_id, int)
           for seq in train_seqs
           for token_id in seq), \
    "Sequences must contain integer token IDs"

    return max_sequence_length

def pad_and_truncate(sequences: list, max_sequence_length: int, pad_id: int = 1) -> np.ndarray:
    """
    Convert variable-length token-ID sequences into fixed-size
    padded tensors for neural-network batching.

    Sequences longer than max_sequence_length are truncated.
    Sequences shorter than max_sequence_length are padded using the [PAD] token.

    Padding uses pad_id instead of zero because zero may represent
    a meaningful token such as [UNK]. The model's masking layer
    must be able to distinguish padding from real content.

    Inputs:
        sequences: list of token-ID sequences
        max_sequence_length: fixed output sequence length
        pad_id: integer ID representing [PAD]

    Outputs:
        NumPy array of shape (num_sequences, max_sequence_length)

    Assumptions:
        - sequences contain integer token IDs
        - tokenizer reserves pad_id for [PAD] training.
    """


    padded_sequences = np.full((len(sequences), max_sequence_length), fill_value=pad_id, dtype=np.int32)

    assert padded_sequences.shape == (len(sequences), max_sequence_length)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_sequence_length)

        padded_sequences[i, :length] = seq[:length]
    return padded_sequences

def encode_labels(train_df, val_df, test_df):
    """
    Convert author names into integer class labels for supervised learning.

    The encoder is fit on training authors only, then applied to validation
    and test sets to maintain a consistent class-ID mapping across splits.

    Inputs:
        train_df : training dataframe
        val_df   : validation dataframe
        test_df  : test dataframe

    Outputs:
        label_encoder : fitted sklearn LabelEncoder
        y_train       : encoded training labels
        y_val         : encoded validation labels
        y_test        : encoded test labels

    Assumptions:
        - all validation/test authors appear in training data
        - author column exists in each dataframe
        """
    le = LabelEncoder()
    assert set(val_df["author"]).issubset(set(train_df["author"])), \
    "Validation set contains unseen authors"

    assert set(test_df["author"]).issubset(set(train_df["author"])), \
        "Test set contains unseen authors"

    y_train = le.fit_transform(train_df["author"])
    y_val   = le.transform(val_df["author"])
    y_test  = le.transform(test_df["author"])
    print(f"\n[labels]  Classes: {list(le.classes_)}")
    return le, y_train, y_val, y_test

def inspect_bpe_segmentations(train_df, tokenizer, n=3):
    """
    Print example tokenizations to verify BPE is splitting
    isiZulu morphemes (prefixes, suffixes) correctly.

    Inputs:
    - train_df: dataframe containing training articles
    - tokenizer: trained BPE tokenizer
    - n: number of authors/examples to display

    Outputs:
        Printed tokenization examples

    Assumptions:
        - tokenizer has already been trained
        - training dataframe contains author/text columns
    """
    print("\n" + "=" * 60)
    print("BPE TOKENIZATION EXAMPLES")
    print("=" * 60)
    for author in train_df["author"].unique()[:n]:
        text = train_df[train_df["author"] == author]["text"].iloc[0]

        # Use a short snippet so tokenization patterns remain readable.
        snippet = " ".join(text.split()[:15])  # first 15 words

        encoded_snippet = tokenizer.encode(snippet) 
        tokens   = encoded_snippet.tokens
        ids      = encoded_snippet.ids
        print(f"\nAuthor : {author}")
        print(f"Input  : {snippet}")
        print(f"Tokens : {tokens}")
        print(f"IDs    : {ids}")
    print("=" * 60)

def evaluate_tokenizer(tokenizer, train_seqs, val_seqs, test_seqs,
                       train_texts, val_texts, test_texts,
                       max_sequence_length):
    """
    Evaluate tokenizer quality for authorship attribution.

    Metrics:
    --------
    1. UNK token rate - How often the tokenizer fails to represent text the lower the better.
    2. Fertility (subwords per word) - Average number of subword tokens produced per word.
    3. Truncation percentage - How many articles lose information because they exceed max sequence length.
    4. Vocabulary coverage - How much of the learned vocabulary is actually used.
    """

    print("\n" + "=" * 60)
    print("TOKENIZER EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. UNK RATE
    # ------------------------------------------------------------
    # Needed to measure how often the tokenizer fails to represent unseen text fragments.
    unk_id = tokenizer.token_to_id("[UNK]")

    assert unk_id is not None, \
    "[UNK] token missing from tokenizer"

    all_seqs = train_seqs + val_seqs + test_seqs
    total_tokens = sum(len(seq) for seq in all_seqs)

    unk_tokens = sum(
        token == unk_id
        for seq in all_seqs
        for token in seq
    )

    unk_rate = (unk_tokens / total_tokens) * 100

    print(f"[UNK rate]          {unk_rate:.2f}%")

    # ------------------------------------------------------------
    # 2. FERTILITY
    # ------------------------------------------------------------
    all_texts = train_texts + val_texts + test_texts

    total_words = sum(len(text.split()) for text in all_texts)

    # Higher fertility means words are split into more subwords.
    fertility = total_tokens / total_words

    print(f"[fertility]         {fertility:.2f} subwords/word")

    # ------------------------------------------------------------
    # 3. TRUNCATION RATE
    # ------------------------------------------------------------
    # Estimate how many articles lose information due to truncation.
    truncated = sum(len(seq) > max_sequence_length for seq in all_seqs)

    trunc_rate = (truncated / len(all_seqs)) * 100

    print(f"[truncation rate]   {trunc_rate:.2f}%")

    # ------------------------------------------------------------
    # 4. VOCAB COVERAGE
    # ------------------------------------------------------------
    vocab_size = tokenizer.get_vocab_size()

    used_tokens = set(
        token
        for seq in all_seqs
        for token in seq
    )

    vocab_coverage_percent = (len(used_tokens) / vocab_size) * 100

    print(f"[vocab coverage]    {vocab_coverage_percent:.2f}%")

    print("=" * 60)

    return {
        "unk_rate_percent": round(unk_rate, 4),
        "fertility": round(fertility, 4),
        "truncation_rate_percent": round(trunc_rate, 4),
        "vocab_coverage_percent": round(vocab_coverage_percent, 4),
    }

def save_arrays(X_train, X_val, X_test, y_train, y_val, y_test,
                le, max_sequence_length, tokenizer, tokenizer_metrics):
    """
    Persist processed datasets and experiment metadata.

    Saves:
    - padded token-ID tensors
    - encoded labels
    - class mappings
    - tokenizer configuration metadata

    These artifacts ensure downstream CNN-LSTM experiments
    use identical preprocessing and label mappings.
    """
    np.save(os.path.join(SPLITS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SPLITS_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(SPLITS_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(SPLITS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SPLITS_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(SPLITS_DIR, "y_test.npy"),  y_test)
    print(f"\n[+]  Arrays saved to {SPLITS_DIR}/")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val   shape: {X_val.shape}")
    print(f"   X_test  shape: {X_test.shape}")

    label_path = os.path.join(SPLITS_DIR, "label_classes.json")
    with open(label_path, "w") as f:
        json.dump(list(le.classes_), f, indent=2)
    print(f"[+]  Label classes saved → {label_path}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    meta = {
        "vocab_size":    VOCAB_SIZE,
        "min_frequency": MIN_FREQUENCY,
        "max_sequence_length":   max_sequence_length,
        "percentile":    PERCENTILE,
        "actual_vocab":  tokenizer.get_vocab_size(),
        "num_classes":   len(le.classes_),
        "classes":       list(le.classes_),
        "X_train_shape": list(X_train.shape),
        "X_val_shape":   list(X_val.shape),
        "X_test_shape":  list(X_test.shape),
        "tokenizer_evaluation": tokenizer_metrics,
    }
    meta_path = os.path.join(METRICS_DIR, "tokenizer_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+]  Tokenizer metadata saved → {meta_path}")


def run():
    train_df, val_df, test_df = load_splits()

    # bpe training (train split only)
    tokenizer = train_bpe_tokenizer(train_df["text"].tolist())
    save_tokenizer(tokenizer)

    train_seqs = encode_split(train_df["text"].tolist(), tokenizer)
    val_seqs   = encode_split(val_df["text"].tolist(),   tokenizer)
    test_seqs  = encode_split(test_df["text"].tolist(),  tokenizer)

    max_sequence_length = analyse_lengths(train_seqs)

    # padding
    pad_id  = tokenizer.token_to_id("[PAD]")
    X_train = pad_and_truncate(train_seqs, max_sequence_length, pad_id)
    X_val   = pad_and_truncate(val_seqs,   max_sequence_length, pad_id)
    X_test  = pad_and_truncate(test_seqs,  max_sequence_length, pad_id)

    # labels
    le, y_train, y_val, y_test = encode_labels(train_df, val_df, test_df)

    # show examples of BPE segmentations
    inspect_bpe_segmentations(train_df, tokenizer)
    
    # evaluate tokenizer
    tokenizer_metrics = evaluate_tokenizer(
        tokenizer,
        train_seqs,
        val_seqs,
        test_seqs,
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
        max_sequence_length
    )

    # save everything
    save_arrays(X_train, X_val, X_test, y_train, y_val, y_test,
                le, max_sequence_length, tokenizer , tokenizer_metrics)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, max_sequence_length, tokenizer


if __name__ == "__main__":
    run()