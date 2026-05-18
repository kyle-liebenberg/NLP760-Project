## **Sprint 1: Data Preprocessing & Baselines (Days 1–5)**
**Primary Lead:** 
**Support:** 

### **What should be done:**
1. **Subword Tokenisation:** You need to convert the raw isiZulu text into subword units. Train a Byte-Pair Encoding (BPE) or SentencePiece tokenizer on your 157-article dataset.
2. **Sequence Padding:** Since neural networks require fixed-length inputs, you must constrain each article to a specific sequence length (e.g., 500 tokens), padding shorter texts with zeros and truncating longer ones.
3. **Traditional Baselines:** Before the deep learning starts, extract traditional Bag-of-Words (BoW) and TF-IDF features from the text. Train a simple baseline classifier (like a Support Vector Machine or Logistic Regression) using these features.
### **Outcome:**
- A clean, tokenised dataset formatted as tensors/arrays ready to be fed into Keras or PyTorch.
- Baseline accuracy and F1-scores established (e.g., "Our BoW + SVM model achieved 60% accuracy"). This is what the CNN-LSTM will try to beat.


# Sprint 1 — Data Preparation, Tokenization, and Traditional NLP Baselines

## Objective

The goal of Sprint 1 was to build a clean, reproducible preprocessing and evaluation pipeline for isiZulu authorship attribution before neural-network experimentation.

This sprint establishes:

* reliable dataset splitting
* leakage-safe preprocessing
* subword tokenization
* fixed-length neural inputs
* strong traditional NLP baselines

These outputs form the foundation for Sprint 2 CNN-LSTM experimentation.

---

# Dataset Overview

## Task

Multi-class authorship attribution for isiZulu news articles.

## Dataset Characteristics

* 157 total articles
* 7 unique authors
* mildly imbalanced class distribution
* highly agglutinative language structure

Important observations discovered during EDA:

* Nonkululeko Nhlapo writes significantly shorter articles
* Simangaliso Ntshangase has the fewest samples and is expected to be the hardest author to classify

---

# Sprint 1 Pipeline

## 1. Exploratory Data Analysis (EDA)

The EDA stage verified:

* missing values
* duplicate articles
* author distribution
* article length distributions
* vocabulary overlap between authors

Visualizations generated:

* author distribution charts
* sequence length histograms
* per-author boxplots

---

## 2. Stratified Dataset Splitting

The dataset was split into:

* 70% training
* 15% validation
* 15% test

Splitting used:

```python
stratify=df["author"]
```

This preserves author proportions across all splits and prevents minority authors from disappearing from validation or test sets.

---

# Important Design Decision

The tokenizer is trained ONLY on the training split.

Reason:
Training the tokenizer before splitting would leak vocabulary information from validation/test data into training.

This preserves:

* fair evaluation
* experimental integrity
* proper ML methodology

---

## 3. Byte-Pair Encoding (BPE) Tokenizer

A custom BPE tokenizer was trained for isiZulu subword segmentation.

### Configuration

```python
VOCAB_SIZE = 3000
MIN_FREQUENCY = 2
PERCENTILE = 95
```

### Why BPE?

isiZulu is highly agglutinative.

BPE helps the model learn:

* prefixes
* suffixes
* stems
* reusable morphemes

instead of memorizing entire words.

---

## 4. Sequence Encoding and Padding

Articles were converted into integer token-ID sequences.

Sequences were:

* truncated to a fixed maximum length
* padded using the `[PAD]` token

Padding uses the tokenizer’s actual PAD token ID rather than manually inserting zeros.

This allows future neural models to correctly mask padding positions during training.

---

## 5. Label Encoding

Author names were converted into integer class IDs using `LabelEncoder`.

The encoder is fit only on the training set and reused consistently across validation and test splits.

---

## 6. Tokenizer Evaluation

Tokenizer quality was evaluated using:

* UNK token rate
* fertility (subwords per word)
* truncation rate
* vocabulary coverage

### Results

| Metric              | Result             |
| ------------------- | ------------------ |
| UNK Rate            | ~0%                |
| Fertility           | ~2.1 subwords/word |
| Truncation Rate     | ~6%                |
| Vocabulary Coverage | ~95%               |

These results indicate that the tokenizer captures isiZulu morphology effectively while minimizing information loss.

---

# Traditional NLP Baselines

The following baselines were implemented:

1. BoW + word n-grams + SVM
2. BoW + character n-grams + SVM
3. TF-IDF + word n-grams + SVM
4. TF-IDF + character n-grams + SVM
5. TF-IDF + Logistic Regression

---

# Evaluation Strategy

## Primary Metric

```text
Macro F1-score
```

Macro F1 was selected instead of accuracy because:

* the dataset is mildly imbalanced
* every author should contribute equally to evaluation
* accuracy can hide poor minority-class performance

---

# Expected Strongest Baseline

The strongest traditional model is expected to be:

```text
TFIDF_char_SVM
```

Character n-grams are especially effective for:

* stylometry
* morphology
* spelling habits
* agglutinative languages

This baseline becomes the benchmark that the CNN-LSTM must surpass in Sprint 2.

---

# Generated Outputs

## Saved Splits

```text
data/splits/
```

Files:

* train.csv
* val.csv
* test.csv

---

## Neural Input Arrays

```text
X_train.npy
X_val.npy
X_test.npy

y_train.npy
y_val.npy
y_test.npy
```

---

## Tokenizer Artifacts

```text
outputs/bpe_tokenizer/tokenizer.json
```

---

## Metadata

```text
outputs/metrics/tokenizer_metadata.json
outputs/metrics/baseline_report.json
```

---

# Reproducibility Features

The pipeline includes:

* deterministic random states
* defensive assertions
* split verification
* overlap detection
* metadata tracking
* reusable preprocessing artifacts

This ensures experiments can be reproduced consistently across Sprint 2 development.

---

# Sprint 2 Goal

Sprint 2 focuses on training a CNN-LSTM architecture using the preprocessing outputs from Sprint 1.

The core research question is:

> Can subword-aware neural architectures outperform strong traditional stylometric baselines for low-resource isiZulu authorship attribution?


=================================================================
SUMMARY TABLE FOR BASELINES
=================================================================
Model                   F1 Macro  Precision   Recall   Accuracy
-----------------------------------------------------------------
BoW_word_SVM              0.8593     0.9111   0.8611     0.8636
BoW_char_SVM              0.7077     0.7861   0.7361     0.7273
TFIDF_word_SVM            0.9095     0.9444   0.9028     0.9091 ← BEST
TFIDF_char_SVM            0.8123     0.8611   0.8194     0.8182
TFIDF_word_LogReg         0.8540     0.9028   0.8611     0.8636
=================================================================
---





