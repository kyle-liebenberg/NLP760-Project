# isiZulu authorship attribution (NLP760)

This project applies **subword (BPE) embeddings** and a **hybrid CNN-LSTM** to authorship attribution in isiZulu. The goal is to identify an author’s distinctive writing style from news text, with classical TF-IDF baselines and SHAP-based explanations for comparison and interpretability.

The corpus comprises articles by **11 Isolezwe journalists**, scraped and labelled by author. The pipeline supports data augmentation via overlapping word chunks, train-only BPE tokenisation, sklearn baselines, neural training, and post-hoc SHAP attribution maps.

---

## Requirements

- **Python 3.12.x**
- Dependencies in [`requirements.txt`](requirements.txt)
- **Data collection only:** Google Chrome + ChromeDriver (for Selenium scraping)

### macOS

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```cmd
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install [Python 3.12](https://www.python.org/downloads/) and ensure `py` or `python` is on your PATH. For scraping, install [Google Chrome](https://www.google.com/chrome/) and a matching [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/).

---

## Quick start: run the full pipeline

After the raw dataset exists at `data/raw/isizulu_authors_dataset.csv`, you can run **preprocessing → tokenisation → baselines → CNN-LSTM → SHAP** in one command:

**macOS / Linux:**

```bash
python3 run_pipeline.py
```

**Windows:**

```cmd
python run_pipeline.py
```

Run from the **project root**. The orchestrator does **not** re-scrape data; use `src/build_author_dataset.py` separately when you need a fresh corpus.

Outputs are written under `outputs/metrics/`, `outputs/figures/`, `outputs/bpe_tokenizer/`, and `models/`.

---

## Pipeline overview

| Step | Script | Output (summary) |
|------|--------|------------------|
| 0. Data (optional) | `src/build_author_dataset.py` | `data/raw/isizulu_authors_dataset.csv` |
| 1. Preprocessing | `src/preprocessing.py` | `data/splits/{train,val,test}.csv` |
| 2. Tokeniser | `src/tokenizer.py` | BPE model + `X_*.npy` / `y_*.npy` |
| 3. Baselines | `src/baselines.py` | Metrics and comparison figures |
| 4. Deep learning | `src/train_cnn_lstm.py` | `models/cnn_lstm_best.keras` |
| 5. Interpretability | `src/interpret_model.py` | SHAP attribution plots |

### Documentation

| Topic | Document |
|-------|----------|
| Data collection | [`docs/1-data.md`](docs/1-data.md) |
| Preprocessing & splits | [`docs/2.1-preprocessing.md`](docs/2.1-preprocessing.md) |
| BPE tokeniser | [`docs/2.2-tokeniser.md`](docs/2.2-tokeniser.md) |
| Classical baselines | [`docs/2.3-baselines.md`](docs/2.3-baselines.md) |
| CNN-LSTM architecture | [`docs/3.1-deep_learning_model_architecture.md`](docs/3.1-deep_learning_model_architecture.md) |
| Training loop | [`docs/3.2-deep_learning_training_loop.md`](docs/3.2-deep_learning_training_loop.md) |
| SHAP interpretability | [`docs/4-shap_interpretability.md`](docs/4-shap_interpretability.md) |

---

## 1. Data collection

Articles are scraped from [Isolezwe](https://isolezwe.co.za) using Selenium (to load additional pages) and `requests` (to extract body text). Eleven author profiles are configured in `src/build_author_dataset.py`.

**Current raw corpus:** 877 articles (77–80 per author).

**macOS / Linux:**

```bash
python3 src/build_author_dataset.py
```

**Windows:**

```cmd
python src\build_author_dataset.py
```

→ [Full documentation](docs/1-data.md)

---

## 2. Preprocessing, tokenisation, and baselines

### 2.1 Preprocessing (`src/preprocessing.py`)

Loads the raw CSV, slices long articles into **150-word chunks** (30-word overlap), then applies a **stratified 70% / 15% / 15%** train / validation / test split on `author`.

**Latest run:** 2,040 chunks → train 1,428 · val 306 · test 306.

```bash
python3 src/preprocessing.py          # macOS / Linux
python src\preprocessing.py           # Windows
```

→ [Full documentation](docs/2.1-preprocessing.md)

### 2.2 BPE tokeniser (`src/tokenizer.py`)

Trains **Byte-Pair Encoding** on the **training split only** (vocabulary size **3000**; max length from the **95th percentile** of train lengths, currently **404** tokens). Saves padded NumPy arrays for the neural model.

```bash
python3 src/tokenizer.py
python src\tokenizer.py
```

→ [Full documentation](docs/2.2-tokeniser.md)

### 2.3 Classical baselines (`src/baselines.py`)

Five sklearn pipelines (BoW and TF-IDF, word and character n-grams, SVM and logistic regression) on the **same CSV chunks**.

**Latest test-set results (macro F1):**

| Model | F1 macro | Accuracy |
|-------|----------|----------|
| BoW_word_SVM | 0.9034 | 0.9020 |
| BoW_char_SVM | 0.8776 | 0.8758 |
| **TFIDF_word_SVM** | **0.9360** | **0.9359** |
| TFIDF_char_SVM | 0.9149 | 0.9150 |
| TFIDF_word_LogReg | 0.8971 | 0.8954 |

**Best baseline:** `TFIDF_word_SVM` - **F1 macro = 0.9360**.

```bash
python3 src/baselines.py
python src\baselines.py
```

→ [Full documentation](docs/2.3-baselines.md)

---

## 3. Neural model and interpretability

The CNN-LSTM reads padded BPE arrays (`X_train.npy`, etc.) and `label_classes.json`. Architecture and training are documented in depth in `docs/3.1` and `docs/3.2`.

**Latest CNN-LSTM test metrics:** F1 macro **0.8740**, accuracy **0.8725** (see `outputs/metrics/dl_report.json`). This is below the best TF-IDF baseline on the same chunked test split.

```bash
python3 src/train_cnn_lstm.py
python src\train_cnn_lstm.py

python3 src/interpret_model.py
python src\interpret_model.py
```

→ [Architecture](docs/3.1-deep_learning_model_architecture.md) · [Training](docs/3.2-deep_learning_training_loop.md) · [SHAP](docs/4-shap_interpretability.md)

---

## Project layout

```
NLP760-Project/
├── run_pipeline.py          # Orchestrates steps 1–5 (excludes scraping)
├── requirements.txt
├── data/
│   ├── raw/                 # Scraped CSV
│   └── splits/              # CSV chunks + NumPy tensors
├── src/                     # Pipeline scripts
├── models/                  # Saved Keras weights
├── outputs/
│   ├── metrics/             # JSON reports
│   ├── figures/             # Plots (baselines, SHAP, histograms)
│   └── bpe_tokenizer/       # tokenizer.json
├── docs/                    # Detailed documentation
└── notebooks/               # Exploratory notebooks
```

---

## Reproducibility

- Random seed **42** for splitting and sklearn models; Keras seed **19** in training.
- Re-run `preprocessing.py` → `tokenizer.py` after changing the raw dataset or chunking settings.
- Baselines and the neural model both consume the same `data/splits/` artefacts for fair comparison.
