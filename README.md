# NLP760-Project

This project applies subword embeddings to authorship attribution, aiming to identify an author’s distinctive writing style by analysing subword patterns rather than whole words.

## Requirements

`Python 3.12.x` is required.

**Mac:**

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:** to be added.

To re-run the data scraper, also install: `pip install requests beautifulsoup4`.

---

## Pipeline overview

To run the entire pipeline of the program:
```bash
python3 run_pipeline.py
```

Run scripts from the project root **in this order**:

| Step | Script | Output |
|------|--------|--------|
| 1. Data | `src/build_author_dataset.py` | `data/raw/isizulu_authors_dataset.csv` |
| 2. Split | `src/preprocessing.py` | `data/splits/{train,val,test}.csv` |
| 3. Tokeniser | `src/tokenizer.py` | BPE model + `X_*.npy` / `y_*.npy` |
| 4. Baselines | `src/baselines.py` | Metrics and comparison figures |

Detailed documentation: [`docs/1-data.md`](docs/1-data.md) · [`docs/2.2-preprocessing.md`](docs/2.2-preprocessing.md) · [`docs/2.1-tokeniser.md`](docs/2.1-tokeniser.md) · [`docs/2.3-baselines.md`](docs/2.3-baselines.md)

---

## 1. Data collection

isiZulu articles were scraped from [Isolezwe](https://isolezwe.co.za) for 11 journalists. We use Isolezwe because existing corpora did not provide enough labelled text per author.

| Author | Profile |
|--------|---------|
| Mhlengi Shangase | `https://isolezwe.co.za/authors/journalist/` |
| Zimbili Vilakazi | `https://isolezwe.co.za/authors/zee/` |
| Sabelo Nsele | `https://isolezwe.co.za/authors/mangethe/` |
| Simangaliso Ntshangase | `https://isolezwe.co.za/authors/health-and-lifestyle-journalist/` |
| Fanelesibonge Bengu | `https://isolezwe.co.za/authors/fanele-bengu/` |
| Nonkululeko Nhlapo | `https://isolezwe.co.za/authors/online/` |
| Nokubongwa Phenyane | `https://isolezwe.co.za/authors/maphenyane/` |
| Zakhele Xaba | `https://isolezwe.co.za/authors/zakhele-xaba/` |
| Sibusiso Mdlalose | `https://isolezwe.co.za/authors/mdlalose-wezemidlalo/` |
| Mthokozisi Mncuseni | `https://isolezwe.co.za/authors/your-football-guy/` |
| Charles Khuzwayo | `https://isolezwe.co.za/authors/entertainment` |

```bash
python3 src/build_author_dataset.py
```

**Current dataset:** 263 articles, ~69,700 words, 11 authors (19–25 articles each). Although small by NLP standards, it fits a low-resource African language authorship study.

→ [Full documentation](docs/1-data.md)

---

## 2. Preprocessing and baselines

### 2.1. Dataset splitting (`src/preprocessing.py`)

Loads the raw CSV and performs a **stratified 70% / 15% / 15%** train / validation / test split on the `author` column. Every author appears in every split; article URLs are checked so no row leaks across splits.

```bash
python3 src/preprocessing.py
```

**Outputs:** `data/splits/train.csv` (184), `val.csv` (39), `test.csv` (40), plus `outputs/metrics/split_metadata.json`.

→ [Full documentation](docs/2.2-preprocessing.md)

### 2.2. BPE tokeniser (`src/tokenizer.py`)

Trains **Byte-Pair Encoding** on the **training split only** (no vocabulary leakage). BPE learns frequent character merges so the model can represent isiZulu morphology (prefixes and suffixes such as `uku-`, `ama-`, `nga-`) without a hand-built analyser.

- Vocabulary size: **3000**
- Max sequence length: **95th percentile** of training lengths (**1012** tokens in the latest run)
- Encodes labels with `LabelEncoder`, pads sequences, saves NumPy arrays for the CNN-LSTM

```bash
python3 src/tokenizer.py
```

**Outputs:** `outputs/bpe_tokenizer/tokenizer.json`, `data/splits/X_{train,val,test}.npy`, `y_*.npy`, `label_classes.json`, `outputs/metrics/tokenizer_metadata.json`.

→ [Full documentation](docs/2.1-tokeniser.md)

### 2.3. Classical baselines (`src/baselines.py`)

Five sklearn pipelines establish benchmarks on the **same CSV splits** (raw text, not BPE):

| Model | Features | Classifier |
|-------|----------|------------|
| BoW_word_SVM | Word n-grams (1–2) | LinearSVC |
| BoW_char_SVM | Character n-grams (2–4) | LinearSVC |
| TFIDF_word_SVM | Word TF-IDF (1–2) | LinearSVC |
| TFIDF_char_SVM | Character TF-IDF (2–4) | LinearSVC |
| TFIDF_word_LogReg | Word TF-IDF (1–2) | Logistic Regression |

isiZulu is agglutinative: character n-grams capture shared roots (e.g. *hamba* in *ngihamba* and *uyahamba*) that word-level bag-of-words can miss. Word-level TF-IDF remains a strong competitor on this corpus.

```bash
python3 src/baselines.py
```

**Latest test-set results (macro F1):**

| Model | F1 macro | Accuracy |
|-------|----------|----------|
| BoW_word_SVM | 0.6697 | 0.7250 |
| BoW_char_SVM | 0.8473 | 0.8750 |
| TFIDF_word_SVM | 0.8582 | 0.8750 |
| TFIDF_char_SVM | 0.8379 | 0.8500 |
| **TFIDF_word_LogReg** | **0.8602** | **0.8750** |

**Best baseline:** `TFIDF_word_LogReg` — **F1 macro = 0.8602**. The CNN-LSTM must exceed this on the same test split to justify the subword approach.

**Outputs:** `outputs/metrics/baseline_report.json`, `outputs/figures/baseline_comparison.png`, confusion matrix for the best model.

→ [Full documentation](docs/2.3-baselines.md)

---

## 3. Neural model (Sprint 2)

The CNN-LSTM uses the padded BPE arrays from step 2.2 (`X_train.npy`, `y_train.npy`, etc.) and `label_classes.json` for class names. Target: beat **F1 macro = 0.8602** on the held-out test set.
