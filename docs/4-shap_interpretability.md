# 4. SHAP interpretability (`src/interpret_model.py`)

This script applies **SHAP** (SHapley Additive exPlanations) to the trained CNN-LSTM, producing bar charts that show which BPE subwords most influenced each prediction for the true author class.

## Purpose

High accuracy alone does not show *what* the model learned. For a low-resource isiZulu study, attributions help verify whether the network relies on plausible morphological and lexical cues (prefixes, verbal roots, topic words) rather than spurious patterns.

SHAP assigns each input token a Shapley value: positive values push the prediction toward the target class; negative values suppress it.

## Prerequisites

Complete training first:

1. [`src/tokenizer.py`](2.2-tokeniser.md): `tokenizer.json`, `X_*.npy`, `label_classes.json`
2. [`src/train_cnn_lstm.py`](3.2-deep_learning_training_loop.md): `models/cnn_lstm_best.keras`

| Required artifact | Path |
|-------------------|------|
| Trained model | `models/cnn_lstm_best.keras` |
| BPE tokenizer | `outputs/bpe_tokenizer/tokenizer.json` |
| Test tensors | `data/splits/X_test.npy`, `y_test.npy` |
| Test text | `data/splits/test.csv` |
| Background data | `data/splits/X_train.npy` (first 30 rows) |
| Class names | `data/splits/label_classes.json` |

## Method

| Step | Detail |
|------|--------|
| Explainer | `shap.Explainer` with a **model-agnostic** wrapper (not gradient-based DeepSHAP) |
| Model wrapper | `model_predict_wrapper` converts NumPy inputs to `tf.int32` tensors and returns softmax probabilities |
| Background | First **30** training sequences - baseline expectation for perturbations |
| Perturbation budget | `max_evals=1000` per explanation |
| Test samples | Indices **0**, **10**, and **20** from the test set |
| Display | Up to **25** non-padding BPE tokens per plot (readability) |

For each sample, attributions are taken for the **true author’s** output dimension (`true_label_idx`), aligned with BPE tokens decoded from the raw chunk text in `test.csv`.

### Visual encoding

| Colour | Meaning |
|--------|---------|
| Red (`#E8593C`) | Positive SHAP: pushes prediction toward the true author |
| Blue (`#4A90D9`) | Negative SHAP: suppresses the true author class |

Plots include a zero baseline and rotated subword labels on the x-axis.

## Outputs

| File | Description |
|------|-------------|
| `outputs/figures/shap_attribution_sample_0.png` | Attribution map for test index 0 |
| `outputs/figures/shap_attribution_sample_10.png` | Test index 10 |
| `outputs/figures/shap_attribution_sample_20.png` | Test index 20 |

Filenames use the test-set row index, not the author name.

## Usage

From the project root:

**macOS / Linux:**

```bash
python3 src/interpret_model.py
```

**Windows:**

```cmd
python src\interpret_model.py
```

This step is **phase 5** (final) of [`run_pipeline.py`](../run_pipeline.py).

Requires a trained `cnn_lstm_best.keras`; SHAP computation can take several minutes per sample depending on hardware.

## Main flow (`main`)

1. Load tokenizer, label classes, train/test arrays, and `test.csv`.
2. Load `cnn_lstm_best.keras`.
3. Build `shap.Explainer` with background `X_train[:30]`.
4. For each sample index, compute SHAP values for the true class and save a bar chart.

## Dependencies

- `shap`
- `tensorflow`
- `tokenizers`
- `numpy`, `pandas`, `matplotlib`, `seaborn`

Listed in `requirements.txt`.

## Design choices

1. **Model-agnostic SHAP**: Avoids version-specific gradient explainers for Keras 3 / TensorFlow 2.16+; trades some speed for compatibility.
2. **Limited token display**: Capping at 25 tokens keeps figures readable; full sequences can be hundreds of BPE ids after chunking.
3. **True-class attributions**: Explains why the model assigned probability to the *correct* author, which supports error analysis when predictions are wrong (extend indices in code as needed).

## Limitations

- SHAP on long sequences is expensive; `max_evals=1000` is a practical compromise, not an exhaustive search.
- Attributions are local to each explained instance; they do not define global feature importance across the corpus.
- Padding tokens are excluded from plots via the pad token id from the tokenizer.

## Related documentation

- Model architecture: [`3.1-deep_learning_model_architecture.md`](3.1-deep_learning_model_architecture.md)
- Training and test metrics: [`3.2-deep_learning_training_loop.md`](3.2-deep_learning_training_loop.md)
